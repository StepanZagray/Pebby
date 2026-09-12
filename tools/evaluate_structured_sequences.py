"""Frozen open-loop H4 field diagnostics; future observations never enter rollout."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.structured_transition import StructuredTransition, steps_targets
from pebby.agent.structured_objective import GROUPS
from tools.build_structured_sequence_cache import FORMAT, digest


def load_cache(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
            or manifest.get('split') != 'validation' or manifest.get('source') != 'generated_only'
            or not manifest.get('history_verified_against_actual_source_rows')):
        raise ValueError('requires verified held-out chronological cache')
    arrays = {}
    for name, info in manifest['arrays'].items():
        source = path / f'{name}.npy'
        if digest(source) != info['sha256']:
            raise ValueError(f'array checksum mismatch: {name}')
        arrays[name] = np.load(source, mmap_mode='r', allow_pickle=False)
        if list(arrays[name].shape) != info['shape'] or str(arrays[name].dtype) != info['dtype']:
            raise ValueError('array shape/dtype mismatch')
        if not np.isfinite(arrays[name]).all():
            raise ValueError('nonfinite cache array')
    seeds = arrays['seeds']; n = len(seeds)
    if len(np.unique(seeds)) != n or not ((seeds >= 1_000_000) & (seeds < 2_000_000)).all():
        raise ValueError('distinct held-out generated levels required')
    if arrays['fields'].shape != (n, 148, 96) or arrays['next_fields'].shape != (n, 4, 148, 96):
        raise ValueError('incorrect chronological field shapes')
    if arrays['actions'].shape != (n, 4) or not np.isin(arrays['actions'], np.arange(4)).all():
        raise ValueError('four actual chronological actions required')
    if any(arrays[key].any() for key in ('terminal', 'won', 'lost_life')):
        raise ValueError('this held-out protocol is live-only')
    return arrays, manifest


def readout_metrics(readout, player, triple, steps):
    targets = {'player': player[..., 1] * 12 + player[..., 0],
               'carried_shape': triple[..., 0], 'carried_color': triple[..., 1],
               'carried_rotation': triple[..., 2], 'steps': steps_targets(steps)}
    metrics = {}; glyph_correct = []
    for name, target in targets.items():
        logits = readout[name + '_logits']
        correct = logits.argmax(-1) == target
        ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none').view_as(target)
        metrics[name] = {'correct': correct.sum(0).tolist(),
                         'accuracy': correct.float().mean(0).tolist(),
                         'cross_entropy': ce.mean(0).tolist()}
        if name.startswith('carried_'):
            glyph_correct.append(correct)
    joint = torch.stack(glyph_correct).all(0)
    metrics['carried_joint'] = {'correct': joint.sum(0).tolist(), 'accuracy': joint.float().mean(0).tolist()}
    return metrics


@torch.inference_mode()
def score(model, arrays, scale, batch_size=16):
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    model.eval()
    predictions = []; actual = []; copies = []; errors = {key: [] for key in GROUPS}; copy_errors = {key: [] for key in GROUPS}
    n = len(arrays['seeds'])
    for begin in range(0, n, batch_size):
        end = min(begin + batch_size, n)
        current = torch.from_numpy(np.array(arrays['fields'][begin:end])).float()
        actions = torch.from_numpy(np.array(arrays['actions'][begin:end])).long()
        # The only transition inputs are observed initial fields and four actions.
        output = model.rollout(current, actions)
        targets = torch.from_numpy(np.array(arrays['next_fields'][begin:end])).float()
        if output['fields'].shape != targets.shape or not torch.isfinite(output['fields']).all():
            raise ValueError('invalid open-loop predicted fields')
        predictions.append(output['readout'])
        actual.append({k: v.reshape(end - begin, 4, *v.shape[1:]) for k, v in model.readout(targets.flatten(0, 1)).items()})
        copies.append({k: v[:, None].expand(-1, 4, *v.shape[1:]) for k, v in model.readout(current).items()})
        for name, (left, right) in GROUPS.items():
            for values, observed in ((errors, output['fields']), (copy_errors, current[:, None])):
                difference = observed[..., left:right] - targets[..., left:right]
                if name == 'core':
                    difference = difference / scale
                values[name].append(difference.square().mean((-1, -2)))
    player, triple, steps = (torch.from_numpy(np.array(arrays[name])).long() for name in
                            ('next_player_cell', 'next_triple', 'next_steps'))
    results = {}
    for name, batches in (('predicted', predictions), ('actual', actual), ('initial_field_copy', copies)):
        combined = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        results[name] = readout_metrics(combined, player, triple, steps)
    groups = {}
    for name in GROUPS:
        error, baseline = (torch.cat(values[name]).mean(0) for values in (errors, copy_errors))
        groups[name] = {'prediction_mse': error.tolist(), 'initial_copy_mse': baseline.tolist(),
                        'ratio': [float(a / b) if b > 0 else None for a, b in zip(error, baseline)]}
    persistent_player = torch.from_numpy(np.array(arrays['player_cell']))[:, None]
    persistent_triple = torch.from_numpy(np.array(arrays['triple']))[:, None]
    persistent_steps = steps_targets(torch.from_numpy(np.array(arrays['steps'])).long())[:, None]
    return {'levels': n, 'horizons': [1, 2, 3, 4], 'readout': results, 'group_mse': groups,
            'exact_initial_label_persistence': {
                'player_accuracy': (player == persistent_player).all(-1).float().mean(0).tolist(),
                'carried_joint_accuracy': (triple == persistent_triple).all(-1).float().mean(0).tolist(),
                'steps_accuracy': (steps_targets(steps) == persistent_steps).float().mean(0).tolist()},
            'underflow_counts_by_horizon': (steps < 0).sum(0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    start = time.monotonic()
    print('PID', os.getpid(), flush=True)
    def timeout(*_):
        raise TimeoutError('frozen H4 scoring exceeded60 seconds')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(60)
    paths = [args.checkpoint, args.cache / 'manifest.json', Path(__file__),
             Path('pebby/agent/structured_transition.py'), Path('pebby/agent/structured_objective.py')]
    hashes = {str(path): digest(path) for path in paths}
    arrays, manifest = load_cache(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'pebby.structured-transition.v1':
        raise ValueError('unsupported checkpoint')
    for training_manifest in checkpoint['cache_manifests'].values():
        if training_manifest['field_encoder'] != manifest['field_encoder']:
            raise ValueError('frozen encoder provenance differs from transition training')
    model = StructuredTransition(checkpoint['config']).eval()
    model.load_state_dict(checkpoint['weights'], strict=True)
    report = score(model, arrays, checkpoint['feature_scale'], args.batch_size)
    if any(digest(path) != sha for path, sha in hashes.items()):
        raise ValueError('input or source changed during scoring')
    report.update(status='complete', pid=os.getpid(), cpu_threads=1,
                  elapsed_seconds=time.monotonic() - start,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  source_hashes=hashes, checkpoint_updates=checkpoint['updates'],
                  parameters=model.parameter_count(), batch_size=args.batch_size,
                  seeds=arrays['seeds'].tolist(), source_rows=arrays['source_rows'].tolist(),
                  difficulty_counts=manifest['difficulty_counts'], event_coverage=manifest['event_coverage'],
                  normalization='core MSE uses saved TRAIN feature std; other groups raw. Initial field is copied unchanged at every horizon.',
                  protocol='Four autoregressive transitions; predicted field feeds next step. No optimizer, target fields, flags or labels enter rollout.',
                  limitations=['Live reachable sequences only: no ending or reset evaluation.',
                               'Actual fields are frozen public semantic distillation, not exact states; exact labels score their readouts separately.',
                               'Generated held-out transition fidelity only; no policy, search or control claim.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    signal.alarm(0)
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds'],
                      'player': report['readout']['predicted']['player']['accuracy']}), flush=True)


if __name__ == '__main__':
    main()
