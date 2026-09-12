"""Held-out four-step latent rollout check on distinct generated validation levels.

No training or official data. Encoder work is chunked, while SIGReg is computed
once across the full selected level population. Terminal/reset clips are absent.
"""
from pebby.ls20.provenance import metadata_difficulty_stages, cache_difficulty_metadata

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from pebby.agent.on_policy_provenance import file_digest
from pebby.agent.world_model import (load_world_checkpoint)
from pebby.agent.world_training_objectives import world_losses
from pebby.agent.world_sequences import load_sidecar, sequence_targets
from pebby.agent.world_train import load_dataset, as_tensors


def select_anchors(data, index, size, seed=42):
    if index.meta.get('mode') != 'explore_validation' or index.meta.get('split') != 'validation':
        raise ValueError('held-out scoring requires validation exploratory sequences')
    if np.any((data['seeds'][index.anchor_row] < 1_000_000) | (data['seeds'][index.anchor_row] >= 2_000_000)):
        raise ValueError('held-out scoring rejects training seeds')
    rng = np.random.default_rng(seed)
    levels = np.unique(data['seeds'][index.anchor_row])
    if size > len(levels) or size < 2:
        raise ValueError('need at least two distinct eligible validation levels')
    difficulty = {int(row['seed']): int(row['difficulty']) for row in data['meta']['levels']}
    groups = [levels[[difficulty[int(level)] == stage for level in levels]] for stage in metadata_difficulty_stages(data['meta'])]
    stage_count = len(metadata_difficulty_stages(data['meta']))
    quotas = [size // stage_count + int(stage < size % stage_count) for stage in range(stage_count)]
    if any(len(group) < count for group, count in zip(groups, quotas)):
        raise ValueError('insufficient validation difficulty coverage for balanced sample')
    chosen = np.concatenate([rng.choice(group, count, replace=False)
                             for group, count in zip(groups, quotas)])
    return np.array([rng.choice(index.anchor_row[data['seeds'][index.anchor_row] == level])
                     for level in chosen], dtype=np.int64)


def score(model, tensors, index, anchors, chunk=16):
    if index.meta.get('mode') != 'explore_validation' or index.meta.get('split') != 'validation':
        raise ValueError('held-out scoring requires a validation index')
    if len(anchors) < 2 or len(np.unique(anchors)) != len(anchors):
        raise ValueError('scoring needs at least two distinct anchors')
    if model.training:
        raise ValueError('scoring requires model.eval()')
    if chunk < 1:
        raise ValueError('chunk must be positive')
    device = next(model.parameters()).device
    current, target, predicted = [], [], []
    correct, ce = 0., 0.
    with torch.inference_mode():
        for start in range(0, len(anchors), chunk):
            rows = torch.from_numpy(anchors[start:start + chunk])
            batch = {key: value[rows] for key, value in tensors.items()}
            batch.update(sequence_targets(tensors, index, rows))
            batch['rollout_mask'] = torch.ones(len(rows), dtype=torch.bool)
            batch = {key: value.to(device) for key, value in batch.items()}
            result = world_losses(model, batch, {'sigreg': 0., 'grounding': 1.,
                                                 'glyph': 1., 'successor_policy': 0.})
            if int(result['diagnostics'].get('rollout_rows', -1)) != len(rows):
                raise ValueError('world loss does not implement chronological rollout rows')
            current.append(result['latent'].float().cpu())
            target.append(result['targets'].float().cpu())
            predicted.append(result['predicted'].float().cpu())
            correct += float(result['diagnostics']['set_accuracy']) * len(rows)
            ce += float(result['losses']['policy']) * len(rows)
        current, target, predicted = map(torch.cat, (current, target, predicted))
        slots = torch.cat((current[None], target.transpose(0, 1)), dim=0).to(device)
        sigreg = float(model.sigreg(slots, generator=torch.Generator(device=device.type).manual_seed(0)))
    mse = (predicted - target).square().mean(dim=(0, 2))
    copy = (current[:, None] - target).square().mean(dim=(0, 2))
    return {'levels': len(anchors), 'policy_set_accuracy': correct / len(anchors),
            'policy_cross_entropy': ce / len(anchors),
            'rollout_prediction_mse_by_horizon': mse.tolist(),
            'initial_state_copy_mse_by_horizon': copy.tolist(),
            'prediction_to_copy_ratio_by_horizon': (mse / copy.clamp_min(1e-12)).tolist(),
            'target_variance_mean_by_horizon': target.var(0, correction=0).mean(-1).tolist(),
            'sigreg_full_population': sigreg, 'sigreg_population': list(slots.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--data-cache-dir', type=Path)
    parser.add_argument('--levels', type=int, default=1024)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    hashes = {str(path): file_digest(path) for path in (
        args.checkpoint, args.validation, args.index, __file__,
        Path(__file__).resolve().parents[1] / 'pebby/agent/world_training_objectives.py')}
    data = load_dataset(args.validation, history=8, cache_dir=args.data_cache_dir)
    index = load_sidecar(args.index, args.validation, data, mode='explore_validation')
    anchors = select_anchors(data, index, args.levels, args.seed)
    model, _ = load_world_checkpoint(args.checkpoint)
    model.to(args.device).eval()
    metrics = score(model, as_tensors(data), index, anchors, args.chunk)
    if any(file_digest(path) != digest for path, digest in hashes.items()):
        raise ValueError('scoring inputs changed')
    report = {'pid': os.getpid(), 'status': 'complete', 'checkpoint': str(args.checkpoint),
              'parameters': model.parameter_count(), 'source_hashes': hashes,
              'metrics': metrics, 'anchor_rows': anchors.tolist(),
              'seeds': data['seeds'][anchors].tolist(), 'selection_seed': args.seed,
              'device': args.device, 'encoder_chunk': args.chunk,
              'elapsed_seconds': time.monotonic() - started,
              'limitations': ['Generated exploratory histories only; no game-completion estimate.',
                              'Four live steps only; terminal and life-reset boundaries excluded.',
                              'Latent MSE depends on learned representation; compare with copy and variance.',
                              'SIGReg computed once on all selected distinct levels; chunk SIGReg discarded.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
