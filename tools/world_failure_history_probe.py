"""Fixed generated validation histories from an Oracle-free behavior controller.

This is a diagnostic artifact, deliberately incompatible with training world-data
arrays: it contains no actual successor images and never enters the collector.
Later policies can be compared at these SAME old-behavior histories only.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from pebby.agent import world_data as wd
from pebby.agent.world_model import load_world_checkpoint
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level, FORMAT, GENERATOR_VERSION
from pebby.ls20 import names
from pebby.ls20.plan import simulate
from tools.goal_attribute_probes import digest

CATEGORIES = ('initial', 'first_wrong_action', 'after_1_unchanged', 'after_2_unchanged',
              'after_4_unchanged', 'after_8_unchanged', 'first_unreachable', 'first_life_reset')


def probe_level(spec, policy, max_actions=120):
    if not 1_000_000 <= int(spec['seed']) < 2_000_000:
        raise ValueError('probe accepts generated validation seeds only')
    if policy.config()['history'] != 8 or not 1 <= max_actions <= 120:
        raise ValueError('probe requires H8 and at most 120 actions')
    context = int(spec.get('training_context_index', spec['seed'] % 7))
    env, oracle, proof = wd.verified_context(spec, context, search_limit=600000)
    if env is None:
        env = Ls20Scenario(build_level(spec), context)
    frames, actions, rows, row_keys = [env.render()], [-1], [], {}
    events, logits_cache, trace = {}, {}, []
    unchanged_run, was_reset = 0, False
    stats, ending = Counter(), 'capped'
    with torch.inference_mode():
        for step in range(max_actions):
            observed, valid, previous = wd.history_arrays(frames, actions, 8)
            public_key = wd.history_key(observed, valid, previous)
            # Pure public-input argmax, before any per-step teacher computation.
            if public_key not in logits_cache:
                logits = policy(torch.from_numpy(observed[None]).long(),
                                history_valid=torch.from_numpy(valid[None]),
                                previous_actions=torch.from_numpy(previous[None]))[0].float()
                if logits.shape != (4,) or not torch.isfinite(logits).all():
                    raise ValueError('invalid behavior logits')
                logits_cache[public_key] = logits.numpy().copy()
                stats['policy_forwards'] += 1
            scores = torch.from_numpy(logits_cache[public_key])
            choice = int(scores.argmax())
            probabilities = scores.softmax(-1).numpy()
            state = oracle.state_of(env) if oracle is not None else None
            distance = oracle.distance_for(state) if oracle is not None else None
            mask = wd.successor_optimal_mask(oracle, state) if oracle is not None else 0
            status = 'unsupported' if oracle is None else 'unreachable' if distance is None else 'valid'
            if status == 'valid' and not mask:
                raise ValueError('live reachable teacher state has no optimal actions')
            correct = bool(mask & (1 << choice)) if mask else None
            stats['steps_' + status] += 1
            stats['optimal_choices'] += bool(correct)
            categories = []
            if step == 0:
                categories.append('initial')
            if correct is False:
                categories.append('first_wrong_action')
            if unchanged_run in (1, 2, 4, 8):
                categories.append(f'after_{unchanged_run}_unchanged')
            if status == 'unreachable':
                categories.append('first_unreachable')
            if was_reset:
                categories.append('first_life_reset')
            categories = [name for name in categories if name not in events]
            if categories:
                key = (public_key, state, env.lives())
                if key not in row_keys:
                    row_keys[key] = len(rows)
                    rows.append({'frames': observed, 'history_valid': valid, 'previous_actions': previous,
                                 'optimal': np.uint8(mask), 'seed': np.int32(spec['seed']),
                                 'step': np.int16(step), 'distance': np.int32(-1 if distance is None else distance),
                                 'actor_action': np.int8(choice), 'actor_logits': scores.numpy().copy(),
                                 'category_bits': np.uint8(0)})
                row_index = row_keys[key]
                for name in categories:
                    rows[row_index]['category_bits'] |= np.uint8(1 << CATEGORIES.index(name))
                    events[name] = {'row_index': row_index, 'status': status, 'step': step,
                                    'optimal_mask': mask, 'actions_needed': distance,
                                    'actor_action': choice, 'actor_confidence': float(probabilities[choice]),
                                    'actor_optimal': correct, 'unchanged_run': unchanged_run}
            trace.append(choice)
            old_lives = env.lives()
            predicted, outcome = simulate(oracle.layout, state, choice, oracle.refills) if oracle is not None else (None, None)
            result = env.perform(names.ACTION_IDS[choice])
            if oracle is not None:
                if outcome == 'won':
                    agrees = result.won and env.lives() == old_lives
                elif outcome == 'died':
                    agrees = not result.won and env.lives() == old_lives - 1 and (result.finished or oracle.state_of(env) == oracle.start)
                else:
                    agrees = not result.finished and env.lives() == old_lives and oracle.state_of(env) == predicted
                if not agrees:
                    raise ValueError(f'actual behavior branch disagrees with teacher at seed {spec["seed"]}, step {step}')
                stats['checked_behavior_transitions'] += 1
            was_reset = env.lives() < old_lives
            unchanged = not was_reset and np.array_equal(frames[-1], result.frame)
            unchanged_run = unchanged_run + 1 if unchanged else 0
            stats['unchanged_frame_actions'] += unchanged
            stats['life_losses'] += was_reset
            if result.finished:
                ending = 'won' if result.won else 'game_over'
                break
            if was_reset:
                frames, actions = [result.frame], [-1]
            else:
                frames, actions = (frames + [result.frame])[-8:], (actions + [choice])[-8:]
    if len(rows) > 8:
        raise AssertionError('probe exceeded eight histories per level')
    for category in CATEGORIES:
        events.setdefault(category, {'status': 'not_observed', 'row_index': None})
    return rows, {**proof, 'seed': spec['seed'], 'difficulty': spec['difficulty'],
                  'fog': bool(spec.get('fog')), 'context_index': context, 'events': events,
                  'actions': len(trace), 'action_indices': trace, 'ending': ending,
                  'stats': dict(stats), 'probe_histories': len(rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/ls20-world-query-glyph-b1024.epoch8.pt'))
    p.add_argument('--bank', type=Path, default=Path('data/ls20-verified-validation-monitor.jsonl'))
    p.add_argument('--out', type=Path, default=Path('data/ls20-world-query8-validation-history-probe.npz'))
    p.add_argument('--report', type=Path, default=Path('artifacts/world-query8-validation-history-probe.json'))
    args = p.parse_args(argv)
    if args.out.exists() or args.report.exists():
        raise ValueError('refusing to overwrite a fixed validation probe')
    specs = [json.loads(line) for line in args.bank.read_text().splitlines()]
    if len(specs) != 100 or len({s['seed'] for s in specs}) != 100:
        raise ValueError('fixed monitor must have exactly 100 distinct levels')
    if any(s.get('format') != FORMAT or s.get('generator_version') != GENERATOR_VERSION
           or not 1_000_000 <= int(s['seed']) < 2_000_000 for s in specs):
        raise ValueError('bank must be generated validation levels')
    torch.set_num_threads(1)
    started = time.monotonic()
    report = {'format': 'pebby.ls20-validation-history-probe.v1', 'status': 'running',
              'source': 'generated_only', 'split': 'validation', 'pid': os.getpid(),
              'device': 'cpu', 'workers': 1, 'history': 8, 'max_actions': 120,
              'categories': list(CATEGORIES), 'max_histories_per_level': 8,
              'behavior_checkpoint': {'path': str(args.checkpoint), 'sha256': digest(args.checkpoint)},
              'bank': str(args.bank), 'bank_sha256': digest(args.bank),
              'oracle_actions_in_controller': 0, 'levels': [],
              'public_input_memoization': 'Exact public H8/action/mask bytes, per level; deterministic eval logits reused.',
              'unchanged_definition': 'Byte-identical consecutive public frames, excluding life resets.',
              'limit': 'Later checkpoints scored here are evaluated on fixed epoch8 behavior histories, not their own on-policy distribution.'}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temp = args.report.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.report)
    print('PID', os.getpid(), flush=True)
    persist()
    try:
        policy, _ = load_world_checkpoint(args.checkpoint, 'cpu')
        policy.requires_grad_(False)
        report['behavior_checkpoint'].update(parameters=policy.parameter_count(), config=policy.config())
        rows = []
        for spec in specs:
            if time.monotonic() - started > 600:
                raise TimeoutError('probe exceeded ten-minute budget')
            samples, result = probe_level(spec, policy)
            for event in result['events'].values():
                if event['row_index'] is not None:
                    event['row_index'] += len(rows)
            rows.extend(samples)
            report['levels'].append(result)
            print(f"{len(report['levels'])}/100 seed={spec['seed']} ending={result['ending']} histories={len(samples)}", flush=True)
            persist()
        report['categories_summary'] = {}
        for category in CATEGORIES:
            group = [level['events'][category] for level in report['levels']]
            valid = [event for event in group if event['status'] == 'valid']
            report['categories_summary'][category] = {'statuses': dict(Counter(event['status'] for event in group)),
                   'actor_optimal_count': sum(event['actor_optimal'] for event in valid),
                   'actor_optimal_accuracy': sum(event['actor_optimal'] for event in valid)/len(valid) if valid else None}
        report['checkpoint_unchanged'] = digest(args.checkpoint) == report['behavior_checkpoint']['sha256']
        report['bank_unchanged'] = digest(args.bank) == report['bank_sha256']
        if not report['checkpoint_unchanged'] or not report['bank_unchanged']:
            raise ValueError('behavior checkpoint or bank changed')
        report.update(status='complete', histories=len(rows), endings=dict(Counter(level['ending'] for level in report['levels'])))
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_suffix('.tmp.npz')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays, meta=np.array(json.dumps(report)))
        with np.load(temp, allow_pickle=False) as loaded:
            for key, expected in arrays.items():
                np.testing.assert_array_equal(loaded[key], expected)
        temp.replace(args.out)
        report.update(output=str(args.out), output_sha256=digest(args.out))
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
