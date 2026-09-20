"""Collect current-policy TRAIN mistakes and short, exact recovery demonstrations.

The policy always chooses from public H8. Teachers act only in separately marked
auxiliary trajectories; every retained row has four real-engine checked branches.
"""
import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent import world_data as wd
from pebby.agent.spatial_outcome_policy import load_checkpoint
from pebby.ls20 import names
from pebby.ls20.bank import load
from pebby.ls20.plan import Oracle
from pebby.ls20.reference_profiles import SEARCH_LIMITS
from tools.collect_reference_onpolicy import BANK, BANK_SHA, select_levels, start_ticks, write
from tools.diagnose_reference_policy import digest, memory_available, public_choice
from tools.validate_extended_collector import checked_expansion

# Exact completed spatial checkpoint; no encoder-only policy substitution.
SOURCE_SHA = '5ea4cccdd01d7b7d43e75565a72b0109031811f4fe156b356e6b3da232201d57'
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'artifacts/reference-spatial-outcome-v1/fit/model.pt'
QUOTAS = (128, 96, 80, 72, 64, 48, 24)
PUBLIC = ['frames', 'history_valid', 'previous_actions']
GIB = 2**30


def spec_sha(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def marked(row, kind, trajectory, step, choice):
    return {**row, 'row_kind': np.uint8(kind), 'trajectory_id': np.int64(trajectory),
            'step': np.int64(step), 'chosen_action': np.int8(choice)}


def native_context(spec):
    def native(*args, **kwargs):
        return Oracle(*args, **kwargs, engine='fast')
    with patch.object(wd, 'Oracle', native):
        initial, oracle, proof = wd.verified_context(spec, search_limit=SEARCH_LIMITS[spec['difficulty'] - 1])
    if (initial is None or oracle is None or oracle.truncated or oracle.engine != 'fast'
            or proof.get('oracle_backend') != 'fast'):
        raise ValueError(f'complete native teacher required: {proof}')
    return initial, oracle, proof


def collect_level(spec, policy, max_actions=150, recovery_horizon=16, max_contexts=3):
    if not 0 <= int(spec['seed']) < 1_000_000:
        raise ValueError('only generated TRAIN seeds allowed')
    if policy.config().get('history') != 8 or not 1 <= max_actions <= 150:
        raise ValueError('H8 and 1..150 policy actions required')
    if not 1 <= recovery_horizon <= 64 or not 1 <= max_contexts <= 3:
        raise ValueError('bounded recovery horizon and context count required')
    initial, oracle, proof = native_context(spec)
    context, seed = proof['context_index'], spec['seed']
    env = wd.clone_env(initial)
    frames, actions = [env.render()], [-1]
    seen, rows, stats, checks, candidates = set(), [], Counter(), Counter(), {}
    original = wd._expand
    def expand(*args, **kwargs):
        return checked_expansion(original, checks, *args, **kwargs)
    def remember(reason, policy_step):
        if reason in candidates or oracle.distance_for(oracle.state_of(env)) is None:
            return
        observed, valid, previous = wd.history_arrays(frames, actions, 8)
        candidates[reason] = (wd.clone_env(env), list(frames), list(actions), policy_step,
                             wd.state_history_key(env, oracle, observed, valid, previous))
    refusal_run, stop, trajectories = 0, 'action_limit', []
    with patch.object(wd, '_expand', side_effect=expand), torch.inference_mode():
        for step in range(max_actions):
            if memory_available() < 6 * GIB:
                raise MemoryError('six GiB host reserve exhausted')
            observed, valid, previous = wd.history_arrays(frames, actions, 8)
            key = wd.state_history_key(env, oracle, observed, valid, previous)
            if key in seen:
                remember('repeated_public_history', step)
                stop = 'exact_attractor_repeat'
                break
            choice, _ = public_choice(policy, observed, valid, previous)
            before = oracle.distance_for(oracle.state_of(env))
            targets, branches, results, mask = expand(env, oracle, before, seed, step)
            rows.append(marked(wd._row(targets, observed, valid, previous, seed, context), 0, 0, step, choice))
            seen.add(key)
            stats['policy_actions'] += 1
            stats['optimal_choices'] += bool(mask & (1 << choice))
            stats['rows_after_refusals_' + str(min(refusal_run, 8))] += 1
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
            # Recovery begins AFTER the mistake with the actual resulting H8.
            if not lost_life and not (mask & (1 << choice)):
                remember('first_suboptimal_successor', step + 1)
            if unchanged:
                remember('first_refusal_successor', step + 1)
            if refusal_run >= 8:
                remember('repeated_public_history', step + 1)
        policy_rows = len(rows)
        trajectories.append(dict(trajectory_id=0, kind='policy', rows=policy_rows, stop=stop))
        selected, candidate_keys = [], set()
        for reason in ('repeated_public_history', 'first_suboptimal_successor', 'first_refusal_successor'):
            if reason in candidates and candidates[reason][-1] not in candidate_keys:
                selected.append((reason, candidates[reason]))
                candidate_keys.add(candidates[reason][-1])
        recovery_rows = 0
        for trajectory, (reason, candidate) in enumerate(selected[:max_contexts], 1):
            env, frames, actions, policy_step, _ = candidate
            start_distance = oracle.distance_for(oracle.state_of(env))
            recovery_stop, taken = 'horizon', []
            for step in range(recovery_horizon):
                before = oracle.distance_for(oracle.state_of(env))
                action = oracle.action_at(env, seed=seed)
                if action is None or before is None:
                    raise ValueError('recovery teacher became unreachable')
                choice = names.ACTION_IDS.index(action)
                observed, valid, previous = wd.history_arrays(frames, actions, 8)
                targets, branches, results, mask = expand(env, oracle, before, seed, step)
                if not mask & (1 << choice) or bool(targets['lost_life'][choice]):
                    raise ValueError('recovery action fails optimal/life-preservation check')
                rows.append(marked(wd._row(targets, observed, valid, previous, seed, context), 1, trajectory, step, choice))
                recovery_rows += 1
                taken.append(choice)
                env, result = branches[choice], results[choice]
                if result.finished:
                    if not result.won:
                        raise ValueError('teacher recovery died')
                    recovery_stop = 'won'
                    break
                if oracle.distance_for(oracle.state_of(env)) != before - 1:
                    raise ValueError('recovery did not reduce exact distance by one')
                frames, actions = (frames + [result.frame])[-8:], (actions + [choice])[-8:]
            trajectories.append(dict(trajectory_id=trajectory, kind='recovery', reason=reason,
                policy_step=policy_step, start_distance=start_distance, actions=taken, rows=len(taken), stop=recovery_stop))
        solution = oracle.solution(seed=seed)
        expert, indices = wd._expert(wd.clone_env(initial), oracle, solution, spec, context,
                                     8, wd.expert_budget(len(solution)), seen, 0)
        expert_trajectory = len(selected[:max_contexts]) + 1
        rows.extend(marked(row, 1, expert_trajectory, index, names.ACTION_IDS.index(solution[index]))
                    for row, index in zip(expert, indices))
        trajectories.append(dict(trajectory_id=expert_trajectory, kind='expert_anchors', rows=len(expert),
                                 steps=indices, verified_full_route_won=True))
        failures, failure_proof = wd._failures(initial, oracle, spec, context, 8)
        rows.extend(marked(row, 2, expert_trajectory + 1, index, failure_proof['failure_actions'][index])
                    for row, index in zip(failures, failure_proof['failure_indices']))
        trajectories.append(dict(trajectory_id=expert_trajectory + 1, kind='exhaustion', rows=len(failures),
                                 steps=failure_proof['failure_indices'], stop=failure_proof['failure_stop']))
    if not any(row['won'].any() for row in rows) or checks['branches'] != 4 * len(rows):
        raise ValueError('winning coverage or checked branch count failed')
    return rows, {**proof, 'difficulty': spec['difficulty'], 'rows': len(rows), 'policy_rows': policy_rows,
        'recovery_rows': recovery_rows, 'expert_rows': len(expert), 'exhaustion_rows': len(failures),
        'stop': stop, 'stats': dict(stats), 'trajectories': trajectories, 'branch_checks': dict(checks),
        'win_covered': True, 'unlabelled_rows': sum(int(row['optimal']) == 0 for row in rows)}


def verify_bindings(bindings):
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise ValueError('source changed during collection')


def worker(args):
    torch.set_num_threads(1)
    report = json.loads((args.out_dir / 'report.json').read_text())
    verify_bindings(report['source_bindings'])
    policy, _ = load_checkpoint(args.checkpoint, 'cpu')
    policy.eval().requires_grad_(False)
    specs = load(args.worker_bank)
    for spec in specs:
        if memory_available() < (10 if spec['difficulty'] >= 6 else 9) * GIB:
            raise MemoryError('insufficient host reserve before native teacher search')
        started = time.monotonic()
        rows, proof = collect_level(spec, policy, args.max_actions, args.recovery_horizon)
        array = args.out_dir / 'levels' / f"{spec['seed']}.npz"
        temporary = array.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
        temporary.replace(array)
        proof.update(status='complete', source_checkpoint_sha256=SOURCE_SHA, spec_sha256=spec_sha(spec),
                     array_sha256=digest(array), source_bindings=report['source_bindings'], worker_pid=os.getpid(),
                     elapsed_seconds=time.monotonic() - started,
                     peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        write(array.with_suffix('.json'), proof)
        print(json.dumps(dict(seed=spec['seed'], difficulty=spec['difficulty'], rows=len(rows),
                             recovery_rows=proof['recovery_rows'], seconds=proof['elapsed_seconds'])), flush=True)
        del rows, proof
        gc.collect()
    verify_bindings(report['source_bindings'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--quotas', type=int, nargs=7, default=QUOTAS)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--max-actions', type=int, default=150)
    parser.add_argument('--recovery-horizon', type=int, default=16)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--worker-bank', type=Path)
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    if args.worker_bank:
        worker(args)
        return
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if digest(args.checkpoint) != SOURCE_SHA or digest(BANK) != BANK_SHA:
        raise ValueError('exact current checkpoint and TRAIN bank required')
    selected = select_levels(load(BANK), args.count, args.seed, args.quotas)
    metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cache = ROOT / 'data/reference-outcome-inputs-v1'
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != metadata['cache_manifest_sha256']:
        raise ValueError('original cache manifest differs from checkpoint')
    split_paths = [cache / split / 'seeds.npy' for split in ('train', 'validation')]
    seed_sets = []
    for split, path in zip(('train', 'validation'), split_paths):
        if digest(path) != manifest['arrays'][split]['seeds']['sha256']:
            raise ValueError('cache seed digest mismatch')
        seed_sets.append(set(map(int, np.load(path, allow_pickle=False))))
    if seed_sets[0] & seed_sets[1] or not {s['seed'] for s in selected} <= seed_sets[0]:
        raise ValueError('selected seeds outside checkpoint TRAIN or overlapping VAL')
    del metadata
    args.out_dir.mkdir(parents=True)
    (args.out_dir / 'levels').mkdir()
    paths = [args.checkpoint, BANK, manifest_path, *split_paths, Path(__file__), ROOT / 'tools/validate_extended_collector.py',
             ROOT / 'tools/collect_reference_onpolicy.py', ROOT / 'tools/diagnose_reference_policy.py',
             ROOT / 'third_party/ls20/ls20.py',
             *sorted((ROOT / 'pebby/agent').glob('*.py')), *sorted((ROOT / 'pebby/ls20').glob('*.py')),
             *sorted((ROOT / 'pebby/ls20').glob('*.c')), *sorted((ROOT / 'pebby/ls20').glob('*.so'))]
    bindings = {str(path.resolve()): digest(path) for path in paths}
    report = dict(status='running', pid=os.getpid(), start_ticks=start_ticks(os.getpid()),
        started_local=datetime.now().astimezone().isoformat(), source_checkpoint_sha256=SOURCE_SHA,
        official_inputs_used=False, public_inputs=PUBLIC, selected_bank='train.jsonl', source_bindings=bindings,
        count=args.count, quotas=args.quotas, seed=args.seed, max_actions=args.max_actions,
        recovery_horizon=args.recovery_horizon, max_recovery_contexts=3, levels=[], workers=[],
        policy_oracle_actions=0, planner_horizon=1,
        limitations='One collection round from fixed source policy. Recovery is within-life and capped; no learned reset or persistent memory.')
    bank_text = ''.join(json.dumps(s) + '\n' for s in selected)
    (args.out_dir / 'train.jsonl').write_text(bank_text)
    shards = [[s for s in selected if s['difficulty'] >= 6], [], []]
    for index, spec in enumerate(s for s in selected if s['difficulty'] < 6):
        shards[1 + index % 2].append(spec)
    processes, streams, started = [], [], time.monotonic()
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        write(args.out_dir / 'report.json', report)
    def interrupted(*_):
        raise KeyboardInterrupt('collector terminated')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    persist()
    try:
        if memory_available() < 18 * GIB:
            raise MemoryError('three-worker collection requires 6 GiB reserve plus 12 GiB aggregate headroom')
        for index, specs in enumerate(shards):
            if not specs:
                continue
            bank = args.out_dir / f'worker-{index}.jsonl'
            bank.write_text(''.join(json.dumps(s) + '\n' for s in specs))
            stream = (args.out_dir / f'worker-{index}.log').open('w')
            streams.append(stream)
            cmd = [sys.executable, '-m', 'tools.collect_spatial_recovery', '--out-dir', str(args.out_dir),
                   '--checkpoint', str(args.checkpoint), '--worker-bank', str(bank),
                   '--max-actions', str(args.max_actions), '--recovery-horizon', str(args.recovery_horizon)]
            process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            processes.append(process)
            report['workers'].append(dict(pid=process.pid, start_ticks=start_ticks(process.pid), shard=index))
            persist()
        last_count = -1
        while True:
            codes = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'collection worker failed: {codes}; see worker logs')
            if time.monotonic() - started > args.seconds:
                raise TimeoutError('collection deadline expired')
            if memory_available() < 6 * GIB:
                raise MemoryError('six GiB host reserve exhausted')
            count = len(list((args.out_dir / 'levels').glob('*.json')))
            if count != last_count:
                report['completed_levels'] = count
                persist()
                print(json.dumps(dict(completed=count, count=args.count, elapsed_seconds=report['elapsed_seconds'])), flush=True)
                last_count = count
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
        verify_bindings(bindings)
        for spec in selected:
            relative = Path('levels') / f"{spec['seed']}.npz"
            proof_path = relative.with_suffix('.json')
            proof = json.loads((args.out_dir / proof_path).read_text())
            sha = digest(args.out_dir / relative)
            if (proof['status'] != 'complete' or proof['array_sha256'] != sha
                    or proof['spec_sha256'] != spec_sha(spec) or not proof['policy_rows']):
                raise ValueError('incomplete selected level')
            report['levels'].append(dict(seed=spec['seed'], difficulty=spec['difficulty'], array_path=str(relative),
                sha256=sha, proof_path=str(proof_path), proof_sha256=digest(args.out_dir / proof_path),
                rows=proof['rows'], policy_rows=proof['policy_rows'], recovery_rows=proof['recovery_rows'],
                stop=proof['stop'], elapsed_seconds=proof['elapsed_seconds']))
        report.update(status='complete', sources_unchanged=True,
                      rows=sum(level['rows'] for level in report['levels']))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process, record in zip(processes, report['workers']):
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            record.update(exitcode=process.returncode, reaped=start_ticks(process.pid) != record['start_ticks'])
        for stream in streams:
            stream.close()
        report['workers_reaped'] = all(r['reaped'] for r in report['workers'])
        persist()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
