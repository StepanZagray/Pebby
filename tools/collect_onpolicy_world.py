"""Bounded generated-training-only DAgger pilot; policy choices see public H8 only."""
from pebby.ls20.provenance import generated_context, validate_difficulty

import argparse
from collections import Counter
import json
import multiprocessing
import tempfile
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent import world_data as wd
from pebby.agent.world_model import load_world_checkpoint
from pebby.agent.world_train import require_verified_data, require_winning_coverage, load_dataset
from pebby.ls20.generate import FORMAT as LEVEL_FORMAT, GENERATOR_VERSION
from tools.goal_attribute_probes import digest, stratified_seeds
from tools.validate_extended_collector import checked_expansion

expert_budget = wd.expert_budget


def collect_level(spec, policy, max_actions=48):
    if not 0 <= int(spec['seed']) < 1_000_000:
        raise ValueError('only generated training seeds allowed')
    if policy.config()['history'] != 8 or type(max_actions) is not int or not 1 <= max_actions <= 150:
        raise ValueError('collector requires H8 and 1..150 actions')
    initial, oracle, proof = wd.verified_context(spec, search_limit=600000)
    if initial is None:
        raise ValueError(f'context verification failed: {proof}')
    env = wd.clone_env(initial)
    frames, actions = [env.render()], [-1]
    seen, rows, stats, checks = set(), [], Counter(), Counter()
    refusal_run = 0
    original = wd._expand
    def expand(*args, **kwargs):
        return checked_expansion(original, checks, *args, **kwargs)
    stop = 'action_limit'
    with patch.object(wd, '_expand', side_effect=expand), torch.inference_mode():
        for step in range(max_actions):
            observed, valid, previous = wd.history_arrays(frames, actions, 8)
            key = wd.state_history_key(env, oracle, observed, valid, previous)
            if key in seen:
                stop = 'exact_attractor_repeat'
                break
            # The only action selection: no labels, state, coordinates, or Oracle arguments.
            logits = policy(torch.from_numpy(observed[None]).long(),
                            history_valid=torch.from_numpy(valid[None]),
                            previous_actions=torch.from_numpy(previous[None]))
            if logits.shape != (1, 4) or not torch.isfinite(logits).all():
                raise ValueError('invalid public policy logits')
            choice = int(logits[0].argmax())
            before = oracle.distance_for(oracle.state_of(env))
            targets, branches, results, mask = expand(env, oracle, before, spec['seed'], step)
            rows.append(wd._row(targets, observed, valid, previous, spec['seed'], proof['context_index']))
            seen.add(key)
            stats['rows_after_refusals_' + str(min(refusal_run, 8))] += 1
            stats['optimal_choices'] += bool(mask & (1 << choice))
            stats['policy_actions'] += 1
            old_lives = env.lives()
            env, result = branches[choice], results[choice]
            lost_life = env.lives() < old_lives
            unchanged = not lost_life and np.array_equal(frames[-1], result.frame)
            refusal_run = refusal_run + 1 if unchanged else 0
            stats['unchanged_frame_actions'] += unchanged
            stats['max_consecutive_unchanged_frames'] = max(stats['max_consecutive_unchanged_frames'], refusal_run)
            stats['life_losses'] += lost_life
            if result.finished:
                stop = 'won' if result.won else 'game_over'
                break
            if lost_life:
                frames, actions = [result.frame], [-1]
            else:
                frames, actions = (frames + [result.frame])[-8:], (actions + [choice])[-8:]
        on_policy_count = len(rows)
        solution = oracle.solution(seed=spec['seed'])
        expert, indices = wd._expert(wd.clone_env(initial), oracle, solution, spec,
                                     proof['context_index'], 8, expert_budget(len(solution)), seen, 0)
        rows.extend(expert)
        failures, failure_proof = wd._failures(initial, oracle, spec, proof['context_index'], 8)
        rows.extend(failures)
    if not any(row['won'].any() for row in rows):
        raise ValueError('expert anchors failed winning coverage')
    return rows, {**proof, 'difficulty': spec['difficulty'], 'fog': bool(spec.get('fog')),
                  'search_truncated': False, 'on_policy_samples': on_policy_count,
                  'expert_samples': len(expert), 'expert_indices': indices, 'samples': len(rows),
                  'expert_budget': expert_budget(len(solution)), **failure_proof,
                  'life_loss_branches': sum(int(row['lost_life'].sum()) for row in rows),
                  'terminal_death_branches': sum(int((row['terminal'] & ~row['won']).sum()) for row in rows),
                  'policy_unlabelled_rows': sum(int(row['optimal']) == 0 for row in rows),
                  'win_rows': sum(bool(row['won'].any()) for row in rows), 'win_covered': True,
                  'stop': stop, 'stats': dict(stats), 'branch_checks': dict(checks)}, on_policy_count


def select(path, count, rng_seed, allowed=None):
    specs = [json.loads(line) for line in path.read_text().splitlines()]
    if any(s.get('format') != LEVEL_FORMAT or s.get('generator_version') != GENERATOR_VERSION
           or not 0 <= int(s['seed']) < 1_000_000 for s in specs):
        raise ValueError('not a current generated training bank')
    by_seed = {int(s['seed']): s for s in specs}
    if len(by_seed) != len(specs):
        raise ValueError('duplicate source seeds')
    eligible = {seed: s for seed, s in by_seed.items() if (allowed is None or seed in allowed) and not (generated_context(s) == 0 and s.get('launchers'))}
    return [eligible[seed] for seed in stratified_seeds(eligible, count, rng_seed)]


_WORKER_POLICY = None


def init_worker(checkpoint, expected_hash):
    global _WORKER_POLICY
    torch.set_num_threads(1)
    if digest(checkpoint) != expected_hash:
        raise ValueError('worker checkpoint hash mismatch')
    _WORKER_POLICY, _ = load_world_checkpoint(checkpoint, 'cpu')
    _WORKER_POLICY.requires_grad_(False)


def worker_collect(spec):
    rows, proof, count = collect_level(spec, _WORKER_POLICY)
    proof['worker_pid'] = os.getpid()
    proof['worker_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return rows, proof, count


class RowStore:
    """Disk-backed fixed-capacity columns; only one level's rows are resident."""
    def __init__(self, path, capacity):
        self.path, self.capacity, self.count, self.columns = Path(path), capacity, 0, {}

    def append(self, rows):
        if self.count + len(rows) > self.capacity:
            # Anchor counts now scale with the verified route length. Grow the
            # disk-backed arrays instead of guessing a fixed 50-row upper bound.
            capacity = max(self.capacity * 2, self.count + len(rows))
            for key, column in list(self.columns.items()):
                path = self.path / (key + '.npy')
                replacement = path.with_suffix('.grow.npy')
                grown = np.lib.format.open_memmap(replacement, mode='w+', dtype=column.dtype,
                                                   shape=(capacity, *column.shape[1:]))
                for start in range(0, self.count, 128):
                    grown[start:min(start + 128, self.count)] = column[start:min(start + 128, self.count)]
                grown.flush()
                del grown
                column._mmap.close()
                replacement.replace(path)
                self.columns[key] = np.lib.format.open_memmap(path, mode='r+')
            self.capacity = capacity
        if not self.columns:
            self.columns = {key: np.lib.format.open_memmap(self.path / (key + '.npy'), mode='w+',
                            dtype=np.asarray(value).dtype, shape=(self.capacity, *np.shape(value)))
                            for key, value in rows[0].items()}
        for row in rows:
            for key, column in self.columns.items():
                column[self.count] = row[key]
            self.count += 1

    def arrays(self):
        for column in self.columns.values():
            column.flush()
        return {key: column[:self.count] for key, column in self.columns.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/ls20-world-query-glyph-b1024.epoch8.pt'))
    p.add_argument('--old-bank', type=Path, default=Path('data/ls20-verified-train.jsonl'))
    p.add_argument('--extended-bank', type=Path, default=Path('data/extended-bank-v1/train.jsonl'))
    p.add_argument('--per-source', type=int, default=32)
    p.add_argument('--workers', type=int, choices=(1, 2), default=1)
    p.add_argument('--combined-train', type=Path, default=Path('data/ls20-world-combined-train.npz'))
    p.add_argument('--out', type=Path, default=Path('data/ls20-world-onpolicy-pilot-train.npz'))
    p.add_argument('--report', type=Path, default=Path('artifacts/world-onpolicy-pilot.json'))
    args = p.parse_args(argv)
    if not 1 <= args.per_source <= 512:
        p.error('bounded pilot allows at most 512 levels per source')
    if args.out.exists() or args.out.with_suffix('.jsonl').exists():
        raise ValueError('refusing output overwrite')
    started = time.monotonic()
    torch.set_num_threads(1)
    report = {'status': 'running', 'pid': os.getpid(), 'workers': args.workers, 'torch_threads': 1,
              'device': 'cpu', 'policy_checkpoint': str(args.checkpoint), 'policy_sha256': digest(args.checkpoint),
              'oracle_actions_in_policy_rollout': 0, 'official_inputs_used': False,
              'history': 8, 'max_policy_actions_per_level': 48, 'levels': [],
              'refusal_definition': 'Consecutive byte-identical public frames without life loss; not a privileged refusal classifier.',
              'limits': 'Checks all four branches of retained states including unreachable failure states, '
                        'not all reachable states. One-step dynamics only. '
                        'Expert anchors: at least three or one per four route actions; '
                        'up to three extra actual pre-death states per level.'}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temp = args.report.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.report)
    print('PID', os.getpid(), flush=True)
    persist()
    scratch = tempfile.TemporaryDirectory(prefix='onpolicy-rows-', dir=args.out.parent)
    try:
        with np.load(args.combined_train, allow_pickle=False) as base:
            allowed = set(map(int, np.unique(base['seeds'])))
        if not allowed or any(seed < 0 or seed >= 1_000_000 for seed in allowed):
            raise ValueError('combined source contains non-training seeds')
        report['combined_train'] = {'path': str(args.combined_train), 'sha256': digest(args.combined_train), 'levels': len(allowed)}
        selections = [select(args.old_bank, args.per_source, 2026, allowed), select(args.extended_bank, args.per_source, 2027, allowed)]
        specs = sum(selections, [])
        if len({s['seed'] for s in specs}) != len(specs):
            raise ValueError('source seed overlap')
        report['sources'] = [{'path': str(path), 'sha256': digest(path), 'selection_rng_seed': rng,
                              'selected_seeds': [s['seed'] for s in group]}
                             for path, group, rng in zip((args.old_bank, args.extended_bank), selections, (2026, 2027))]
        policy, checkpoint = load_world_checkpoint(args.checkpoint, 'cpu')
        policy.requires_grad_(False)
        report['policy_parameters'] = policy.parameter_count()
        store, on_policy_rows, auxiliary_rows = RowStore(scratch.name, len(specs) * 50), [], []
        with multiprocessing.get_context('spawn').Pool(args.workers, initializer=init_worker,
                 initargs=(args.checkpoint, report['policy_sha256'])) as pool:
            report['worker_pids'] = [process.pid for process in pool._pool]
            persist()
            # Fixed small batches prevent Pool's result queue retaining all rendered levels.
            for offset in range(0, len(specs), args.workers):
                remaining = 900 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('collection exceeded fifteen-minute deadline')
                results = pool.map_async(worker_collect, specs[offset:offset + args.workers]).get(timeout=remaining)
                for collected, proof, count in results:
                    if count < 1:
                        raise ValueError('selected level produced no policy rows')
                    on_policy_rows.extend(range(store.count, store.count + count))
                    auxiliary_rows.extend(range(store.count + count, store.count + len(collected)))
                    store.append(collected)
                    report['levels'].append(proof)
                    report['rows'] = store.count
                    print(f"{len(report['levels'])}/{len(specs)} seed={proof['seed']} rows={count} stop={proof['stop']}", flush=True)
                    persist()
        report['workers_exited'] = all(not Path(f'/proc/{pid}').exists() for pid in report['worker_pids'])
        if not report['workers_exited']:
            raise RuntimeError('worker cleanup incomplete')
        arrays = store.arrays()
        row_count = store.count
        difficulty_counts = Counter(level['difficulty'] for level in report['levels'] if level['on_policy_samples'])
        report['on_policy_levels_by_difficulty'] = dict(difficulty_counts)
        if args.per_source == 512 and any(difficulty_counts[d] < 103 for d in range(1, 6)):
            raise ValueError('insufficient distinct policy levels per difficulty')
        for source in [*report['sources'], report['combined_train']]:
            if digest(source['path']) != source['sha256']:
                raise ValueError('source changed during collection')
        report['sources_unchanged'] = True
        arrays['meta'] = {'format': wd.FORMAT, 'source': 'generated_only', 'oracle_search': 'complete_only',
                          'state_supervision_version': wd.STATE_SUPERVISION_VERSION,
                          'successor_policy_supervision_version': wd.SUCCESSOR_POLICY_SUPERVISION_VERSION,
                          'history': 8, 'alternatives_per_state': 4, 'samples': row_count,
                          'seeds': sorted(s['seed'] for s in specs), 'accepted_levels': len(specs),
                          'win_covered_levels': len(specs), 'coverage': 'on_policy_with_expert_anchors',
                          'on_policy_rows': on_policy_rows, 'auxiliary_rows': auxiliary_rows,
                          'auxiliary_collection': 'route_proportional_expert_and_real_exhaustion',
                          'on_policy_provenance': report.copy(),
                          'collection_policy': 'model_greedy',
                          'behavior_checkpoint': {'path': str(args.checkpoint), 'sha256': report['policy_sha256'],
                                                  'parameters': policy.parameter_count(), 'config': policy.config()},
                          'levels': report['levels']}
        require_verified_data(arrays)
        require_winning_coverage(arrays)
        report['checkpoint_unchanged'] = digest(args.checkpoint) == report['policy_sha256']
        if not report['checkpoint_unchanged']:
            raise ValueError('policy checkpoint changed during collection')
        temp = args.out.with_suffix('.tmp.npz')
        wd.save(temp, arrays)
        loaded = load_dataset(temp)
        require_verified_data(loaded)
        require_winning_coverage(loaded)
        if set(map(int, loaded['seeds'])) != {s['seed'] for s in specs}:
            raise ValueError('requested seed retention failed')
        temp.replace(args.out)
        args.out.with_suffix('.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in specs))
        report.update(status='complete', output=str(args.out), output_sha256=digest(args.out),
                      on_policy_rows=len(on_policy_rows),
                      expert_rows=sum(level['expert_samples'] for level in report['levels']),
                      failure_rows=sum(level['failure_samples'] for level in report['levels']))
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        scratch.cleanup()
        persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
