"""Generated TRAIN diagnostic: compare actual/imagined scoring on visited states.

Only the public policy drives collection. Actual successors and complete Oracle
labels are used afterward for diagnosis, never for action selection or training.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.model import load_checkpoint
from tools.build_structured_field_cache import actual_histories
from tools.collect_onpolicy_world import collect_level, select
from tools.train_structured_transition import atomic_json, digest


@torch.inference_mode()
def score(policy, rows):
    result = {'states': len(rows), 'branches': 4 * len(rows)}
    if not rows:
        return result
    batch = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    frames = torch.as_tensor(batch['frames'])
    valid = torch.as_tensor(batch['history_valid'])
    previous = torch.as_tensor(batch['previous_actions'])
    current = policy.encoder(frames, valid, previous)
    histories, validity, actions = actual_histories(batch)
    actual = policy.encoder(histories.flatten(0, 1), validity.flatten(0, 1),
                            actions.flatten(0, 1)).reshape(len(rows), 4, 148, 96)
    predicted = policy.successor_fields(current)
    public = policy(frames.long(), history_valid=valid, previous_actions=previous)
    imagined_logits = policy.readout(predicted)
    torch.testing.assert_close(public, imagined_logits, rtol=1e-5, atol=1e-5)
    for name, fields in [('actual', actual), ('imagined', predicted)]:
        logits = policy.readout(fields)
        chosen = logits.argmax(-1).numpy()
        index = np.arange(len(rows))
        bits = (batch['optimal'][:, None] & (1 << np.arange(4))) != 0
        readout = policy.dynamics.readout(fields.flatten(0, 1))
        target_cell = batch['next_player_cell'].reshape(-1, 2)
        correct_player = readout['player_logits'].argmax(-1).numpy() == (
            target_cell[:, 1] * 12 + target_cell[:, 0])
        triple = np.stack([readout['carried_' + key + '_logits'].argmax(-1).numpy()
                           for key in ('shape', 'color', 'rotation')], -1)
        result[name] = {
            'optimal_choices': int(bits[index, chosen].sum()),
            'chosen_life_losses': int(batch['lost_life'][index, chosen].sum()),
            'chosen_unreachable': int((batch['distances'][index, chosen] < 0).sum()),
            'player_correct': int(correct_player.sum()),
            'glyph_joint_correct': int((triple == batch['next_triple'].reshape(-1, 3)).all(-1).sum()),
        }
    result['target_life_losses'] = int(batch['lost_life'].sum())
    result['target_terminal_failures'] = int((batch['terminal'] & ~batch['won']).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--bank', default='data/ls20-verified-train.jsonl', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--levels', type=int, default=10)
    parser.add_argument('--max-actions', type=int, default=32)
    parser.add_argument('--seconds', type=int, default=240)
    args = parser.parse_args()
    if args.report.exists() or not 1 <= args.levels <= 20 or not 1 <= args.max_actions <= 48 or not 1 <= args.seconds <= 600:
        parser.error('new output and bounded counts required')
    torch.set_num_threads(1)
    started = time.monotonic()
    print('PID', os.getpid(), flush=True)
    def expired(*_):
        raise TimeoutError('on-policy diagnostic deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    report = {'status': 'running', 'pid': os.getpid(), 'levels': [], 'official_inputs_used': False,
              'optimization_performed': False, 'scope': 'Generated TRAIN on-policy diagnosis, not held-out completion.',
              'limitations': ['Actual successor scoring is privileged diagnostic only.',
                             'Collection stops on repeated state-history or unreachable state and is capped.',
                             'Small selected TRAIN sample does not establish a generalization rate.']}
    sources = {str(path): digest(path) for path in [args.checkpoint, args.bank, Path(__file__),
               Path('tools/collect_onpolicy_world.py'), Path('tools/build_structured_field_cache.py')]}
    try:
        policy, _ = load_checkpoint(args.checkpoint, 'cpu')
        policy.eval().requires_grad_(False)
        sources.update(policy.sources['code_hashes'])
        selected = select(args.bank, args.levels, 20260912)
        report['selected_seeds'] = [int(spec['seed']) for spec in selected]
        for spec in selected:
            rows, proof, count = collect_level(spec, policy, args.max_actions)
            totals = {'states': 0, 'branches': 0, 'actual': {}, 'imagined': {},
                      'target_life_losses': 0, 'target_terminal_failures': 0}
            for begin in range(0, count, 4):
                metrics = score(policy, rows[begin:min(begin + 4, count)])
                for name, value in metrics.items():
                    if isinstance(value, dict):
                        for key, number in value.items():
                            totals[name][key] = totals[name].get(key, 0) + number
                    else:
                        totals[name] += value
            report['levels'].append({'seed': int(spec['seed']), 'difficulty': spec['difficulty'],
                                     'proof': proof, 'metrics': totals})
            report['elapsed_seconds'] = time.monotonic() - started
            atomic_json(args.report, report)
            print(json.dumps({'seed': int(spec['seed']), 'states': count, 'stop': proof['stop']}), flush=True)
        if any(digest(path) != sha for path, sha in sources.items()):
            raise ValueError('diagnostic sources changed')
        report.update(status='complete', source_unchanged=True, source_hashes=sources)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.report, report)


if __name__ == '__main__':
    main()
