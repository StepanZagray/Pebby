"""Audit a pinned spatial policy on cached, generated validation histories.

Exact-outcome substitutions diagnose the comparator; they are never gameplay
inputs. Metrics count branches/roots separately and exclude unsupported policy
targets without discarding their physical or event supervision.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_objective import actual_outcomes, targets
from tools.train_reference_spatial_outcomes import player_probabilities
from tools.train_reference_outcomes import ARRAYS, batch, guard, sha, write


def outcome_probabilities(fields, values, events):
    return torch.cat([*[x.float().softmax(-1) for x in fields],
                      values.float().softmax(-1), events.float().sigmoid()], -1)


def score_probabilities(model, probabilities):
    branches = model.outcome_projection(probabilities)
    pooled = branches.mean(1, keepdim=True).expand_as(branches)
    return model.comparator(torch.cat((branches, pooled), -1)).squeeze(-1)


def audit(model, arrays, encoder_weights, *, device='cpu', batch_size=64):
    chunks = {k: [] for k in ('native', 'teacher', 'neutral', 'fields', 'events', 'value')}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays['seeds']), batch_size):
            guard()
            items = batch(arrays, np.arange(start, min(start + batch_size, len(arrays['seeds']))), device)
            pred = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder_weights))
            fields, values, events = actual_outcomes(items)
            results = dict(native=pred['action_logits'], teacher=model.score_outcomes(fields, values, events),
                           neutral=model.score_outcomes(tuple(f[:, :1].expand_as(f) for f in fields), values, events),
                           fields=torch.stack([f.argmax(-1) for f in pred['field_logits']], -1),
                           events=pred['event_logits'].sigmoid(), value=pred['value_logits'].argmax(-1))
            for key, value in results.items():
                chunks[key].append(value.cpu().numpy())
    output = {k: np.concatenate(v) for k, v in chunks.items()}
    optimal = np.asarray(arrays['optimal'])
    valid = optimal != 0
    bits = (optimal[:, None] & (1 << np.arange(4))) != 0
    def count(correct, mask):
        return dict(correct=int((correct & mask).sum()), support=int(mask.sum()),
                    accuracy=float(correct[mask].mean()) if mask.any() else None)
    policy = {key: count(bits[np.arange(len(valid)), output[key].argmax(-1)], valid)
              for key in ('native', 'teacher', 'neutral')}
    distance = np.asarray(arrays['distances'])
    safe = (distance >= 0) & ~np.asarray(arrays['lost_life'])
    choice = np.where(safe, distance, 32767).argmin(-1)
    policy['exact_safe_distance_control'] = count(bits[np.arange(len(valid)), choice], valid)
    pairs = safe[:, :, None] & safe[:, None, :] & valid[:, None, None] & (distance[:, :, None] < distance[:, None, :])
    ordering = {k: count(output[k][:, :, None] > output[k][:, None, :], pairs) for k in ('native', 'teacher', 'neutral')}
    actual_fields = np.stack([x.numpy() for x in targets({k: torch.from_numpy(np.array(v)) for k, v in arrays.items()
                                                        if k not in ('raw', 'state', 'glyph', 'rows', 'seeds')})], -1)
    old = np.asarray(arrays['current_steps'])[:, None]
    actual_budget = np.maximum(arrays['next_steps'], -1)
    predicted_budget = output['fields'][..., 4] - 1
    lost = np.asarray(arrays['lost_life'])
    live = ~lost & ~np.asarray(arrays['terminal'])
    cohorts = dict(live_decrease=live & (arrays['next_steps'] < old), live_unchanged=live & (arrays['next_steps'] == old),
                   live_refill=live & (arrays['next_steps'] > old), life_loss=lost)
    budget = {key: {**count(predicted_budget == actual_budget, mask),
                    'increase_recall': count(predicted_budget > old, mask) if key == 'live_refill' else None,
                    'mae': float(np.abs(predicted_budget - actual_budget)[mask].mean()) if mask.any() else None}
              for key, mask in cohorts.items()}
    fields = {name: count(output['fields'][..., i] == actual_fields[..., i], np.ones_like(lost, dtype=bool))
              for i, name in enumerate(('player', 'shape', 'color', 'rotation', 'steps', 'lives'))}
    fields['lives_on_life_loss'] = count(output['fields'][..., 5] == actual_fields[..., 5], lost)
    event = output['events'] >= .5
    return dict(roots=len(valid), levels=len(np.unique(arrays['seeds'])), unsupported_policy_roots=int((~valid).sum()),
                policy=policy, safe_pair_ordering=ordering, budget=budget, fields=fields,
                event_contradictions=dict(win_without_terminal=int((event[..., 2] & ~event[..., 1]).sum()),
                    life_loss_without_lives_decrease=int((event[..., 0] & (output['fields'][..., 5] >= np.asarray(arrays['current_lives'])[:, None])).sum())),
                limits=['Cached validation histories, not native completion; rows within a level are dependent.',
                        'Teacher and neutral substitutions are privileged diagnostics, not deployment inputs.',
                        'Weighted event outputs are not calibrated success probabilities.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--cache', type=Path, default=Path('data/reference-outcome-inputs-v1'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('checkpoint hash mismatch')
    torch.set_num_threads(1)
    started = time.monotonic()
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if sha(args.cache / 'manifest.json') != cp['cache_manifest_sha256']:
        raise ValueError('cache manifest mismatch')
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    arrays = {n: np.load(args.cache / 'validation' / (n + '.npy'), mmap_mode='r', allow_pickle=False) for n in ARRAYS}
    for name, array in arrays.items():
        expected = manifest['arrays']['validation'][name]
        if (list(array.shape) != expected['shape'] or array.dtype.str != expected['dtype']
                or sha(args.cache / 'validation' / (name + '.npy')) != expected['sha256']):
            raise ValueError(f'validation array differs from pinned manifest: {name}')
    paths = [Path(__file__), args.checkpoint, args.cache / 'manifest.json', *sorted(Path('pebby/agent').glob('*.py')),
             Path('tools/train_reference_spatial_outcomes.py'), Path('tools/train_reference_outcomes.py'),
             *(args.cache / 'validation' / (n + '.npy') for n in ARRAYS)]
    bindings = {str(path.resolve()): sha(path) for path in paths}
    model = SpatialOutcomePlanner(cp['planner_config']).eval()
    model.load_state_dict(cp['planner_weights'], strict=True)
    report = audit(model, arrays, cp['encoder_weights'])
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
                  source_sha256=bindings, device='cpu', elapsed_seconds=time.monotonic()-started)
    if any(sha(path) != digest for path, digest in bindings.items()):
        raise ValueError('source changed during audit')
    report.update(status='complete', sources_unchanged=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
