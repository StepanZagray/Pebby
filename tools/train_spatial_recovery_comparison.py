"""Bounded, matched H1 continuation: base-only versus policy-recovery rows.

Both arms restart the same head and AdamW, use the same distinct approved level seeds,
require audited nonzero-policy-target row allowlists, retain the original TRAIN loss weights, and evaluate the fixed validation cache.
Qualification updates are disposable. No encoder, architecture, or objective changes.
"""
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.neural_outcome_policy import ENCODER_RUNTIME, PARENT_SHA, weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_policy import FORMAT
from pebby.agent.spatial_outcome_objective import training_weights
from pebby.agent.world_model import WorldModelConfig, WorldPolicy
from tools.cache_reference_outcome_inputs import Bindings, SCHEMA, stat
from tools.train_reference_outcomes import gpu_available, guard, load_data, optimizer_for, scalar_metrics, sha, write
from tools.train_reference_spatial_outcomes import evaluate, fit_step

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'artifacts/reference-spatial-outcome-v1/fit/model.pt'
CHECKPOINT_SHA = '5ea4cccdd01d7b7d43e75565a72b0109031811f4fe156b356e6b3da232201d57'
BATCH_SIZE = 1024
ARMS = ('control', 'recovery')


def validate_checkpoint(checkpoint):
    required = dict(format=FORMAT, encoder_parent_sha256=PARENT_SHA,
                    encoder_runtime=ENCODER_RUNTIME, encoder_frozen=True,
                    official_training_inputs=False, privileged_inference_inputs=False,
                    planner_horizon=1, planner_refinement_loops=1,
                    learned_voluntary_reset=False, persistent_game_memory=False)
    if any(checkpoint.get(key) != value for key, value in required.items()):
        raise ValueError('continuation requires the frozen generated-only H1 spatial parent')
    if checkpoint['encoder_weights_sha256'] != weights_sha256(checkpoint['encoder_weights']):
        raise ValueError('embedded encoder digest differs')
    if checkpoint['score_weights'] != {'planner': 1., 'direct': 0.}:
        raise ValueError('continuation must preserve planner-only scoring')


def load_supplement(path, checkpoint, base_train, bindings):
    bindings.add(path / 'manifest.json')
    manifest = json.loads((path / 'manifest.json').read_text())
    expected = dict(status='complete', split='train', official_inputs_used=False,
                    source_checkpoint_sha256=CHECKPOINT_SHA,
                    encoder_parent_sha256=checkpoint['encoder_parent_sha256'],
                    encoder_weights_sha256=checkpoint['encoder_weights_sha256'],
                    encoder_runtime=checkpoint['encoder_runtime'])
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('supplement provenance differs from the generated TRAIN parent')
    schema = {**SCHEMA, 'row_kind': ('uint8', ())}
    for name in ('trajectory_id', 'step'):
        if name + '.npy' in manifest.get('files', {}):
            schema[name] = ('int64', ())
    arrays = {}
    for name, (dtype, shape) in schema.items():
        filename = name + '.npy'
        info = manifest.get('files', {}).get(filename)
        if not isinstance(info, dict):
            raise ValueError(f'missing supplemental file metadata: {filename}')
        bindings.add(path / filename, info['sha256'])
        value = np.load(path / filename, mmap_mode='r', allow_pickle=False)
        if (value.dtype != np.dtype(dtype) or value.shape[1:] != shape
                or list(value.shape) != info['shape'] or value.dtype != np.dtype(info['dtype'])):
            raise ValueError(f'supplement schema mismatch: {name}')
        arrays[name] = value
    count = len(arrays['seeds'])
    if not count or any(len(value) != count for value in arrays.values()):
        raise ValueError('supplement arrays need a common nonzero row count')
    if not np.isin(arrays['row_kind'], (0, 1, 2)).all():
        raise ValueError('row_kind must be policy (0), recovery anchor (1), or exhaustion (2)')
    if not np.isin(arrays['seeds'], np.unique(base_train['seeds'])).all():
        raise ValueError('supplement contains non-TRAIN level seeds')
    for name in ('raw', 'state', 'glyph'):
        for start in range(0, count, 1024):
            guard()
            if not np.isfinite(arrays[name][start:start + 1024]).all():
                raise ValueError(f'nonfinite supplemental features: {name}')
    return arrays, manifest



def quality_mask(arrays, rows, train_seeds, name):
    """Validate an explicit audited row allowlist without copying feature arrays."""
    rows = np.asarray(rows)
    if rows.dtype != np.dtype('int64') or rows.ndim != 1 or not len(rows):
        raise ValueError(f'{name} quality rows must be nonempty one-dimensional int64')
    if np.any(rows < 0) or np.any(rows >= len(arrays['seeds'])):
        raise ValueError(f'{name} quality row index out of bounds')
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f'{name} quality rows contain duplicate indices')
    if not np.isin(arrays['seeds'][rows], train_seeds).all():
        raise ValueError(f'{name} quality rows contain non-TRAIN seeds')
    if np.any(arrays['optimal'][rows] == 0):
        raise ValueError(f'{name} quality rows contain zero optimal masks')
    if name == 'supplement' and not np.isin(arrays['row_kind'][rows], (0, 1)).all():
        raise ValueError('supplement quality rows must have kind 0 or 1')
    mask = np.zeros(len(arrays['seeds']), dtype=bool)
    mask[rows] = True
    return mask


def load_quality_manifest(path, base, supplement, report, bindings):
    path = path.resolve()
    quality_hash = bindings.add(path)
    manifest = json.loads(path.read_text())
    expected = dict(status='complete', base_cache_manifest_sha256=report['cache_manifest_sha256'],
                    supplemental_manifest_sha256=report['supplemental_manifest_sha256'])
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('quality manifest is incomplete or bound to different caches')
    rows, files = {}, {}
    train_seeds = np.unique(base['seeds'])
    for name, arrays in (('base', base), ('supplement', supplement)):
        info = manifest.get('files', {}).get(name + '_rows.npy')
        if not isinstance(info, dict) or not isinstance(info.get('path'), str):
            raise ValueError(f'missing {name} quality row file metadata')
        row_path = (path.parent / info['path']).resolve()
        digest = bindings.add(row_path, info['sha256'])
        rows[name] = np.load(row_path, mmap_mode='r', allow_pickle=False)
        quality_mask(arrays, rows[name], train_seeds, name)
        files[str(row_path)] = digest
    report['source_sha256'].update({str(path): quality_hash, **files})
    report['quality_manifest_sha256'] = quality_hash
    report['quality_row_file_sha256'] = files
    report['quality_manifest'] = str(path)
    report['quality_filter'] = {
        name: dict(total_rows=len(arrays['seeds']), allowed_rows=len(rows[name]),
                   excluded_rows=len(arrays['seeds']) - len(rows[name]),
                   allowed_levels=len(np.unique(arrays['seeds'][rows[name]])))
        for name, arrays in (('base', base), ('supplement', supplement))}
    report['quality_filter'].update(required_optimal_nonzero=True, allowed_supplement_row_kinds=[0, 1],
                                    objective_weights_source='unchanged full original TRAIN')
    return rows


def validate_qualification(qualification, report):
    keys = ('source_sha256', 'cache_manifest_sha256', 'supplemental_manifest_sha256', 'objective_weights',
            'initial_planner_weights_sha256', 'learning_rate', 'precision', 'batch_size', 'planned_steps',
            'recovery_fraction', 'policy_row_fraction', 'quality_manifest_sha256',
            'quality_row_file_sha256', 'quality_filter')
    if (qualification.get('status') != 'complete' or qualification.get('qualification_passed') is not True
            or any(key not in report or qualification.get(key) != report[key] for key in keys)):
        raise ValueError('training differs from completed quality-gated matched qualification')


def grouped_rows(seeds, mask=None):
    result = {}
    for row, seed in enumerate(seeds):
        if mask is None or mask[row]:
            result.setdefault(int(seed), []).append(row)
    return result


class MatchedSampler:
    """Reserve policy-covered levels, then sample remaining base levels uniformly.

    This same explicit level distribution applies to both arms. The continuation
    uses uniform levels rather than the original easy-to-hard curriculum.
    """
    def __init__(self, base, supplement, recovery_fraction=.25, policy_row_fraction=.75, *, base_rows, supplement_rows):
        if not 0 < recovery_fraction <= 1 or not 0 <= policy_row_fraction <= 1:
            raise ValueError('invalid recovery or policy-row fraction')
        train_seeds = np.unique(base['seeds'])
        base_mask = quality_mask(base, base_rows, train_seeds, 'base')
        supplement_mask = quality_mask(supplement, supplement_rows, train_seeds, 'supplement')
        self.base = grouped_rows(base['seeds'], base_mask)
        self.policy = grouped_rows(supplement['seeds'], supplement_mask & (supplement['row_kind'] == 0))
        self.anchor = grouped_rows(supplement['seeds'], supplement_mask & (supplement['row_kind'] == 1))
        self.seeds = np.array(sorted(self.base), dtype=np.int64)
        self.eligible = np.array(sorted(self.policy), dtype=np.int64)
        self.reserved = round(BATCH_SIZE * recovery_fraction)
        self.policy_row_fraction = policy_row_fraction
        if not 1 <= self.reserved <= BATCH_SIZE:
            raise ValueError('recovery fraction must reserve at least one slot')
        if not set(self.policy) <= set(self.base) or not set(self.anchor) <= set(self.base):
            raise ValueError('supplement seeds must be in quality-approved base TRAIN levels')
        if len(self.seeds) < BATCH_SIZE or len(self.eligible) < self.reserved:
            raise ValueError('insufficient distinct TRAIN/policy levels for true B1024')

    def sample(self, rng):
        reserved = rng.choice(self.eligible, self.reserved, replace=False)
        remaining = self.seeds[~np.isin(self.seeds, reserved)]
        seeds = np.concatenate((reserved, rng.choice(remaining, BATCH_SIZE - self.reserved, replace=False)))
        base_rows = np.array([rng.choice(self.base[int(seed)]) for seed in seeds], dtype=np.int64)
        replacement = np.full(BATCH_SIZE, -1, dtype=np.int64)
        for index, seed in enumerate(reserved):
            rows = self.policy[int(seed)]
            if int(seed) in self.anchor and rng.random() >= self.policy_row_fraction:
                rows = self.anchor[int(seed)]
            replacement[index] = rng.choice(rows)
        order = rng.permutation(BATCH_SIZE)
        return dict(seeds=seeds[order], base_rows=base_rows[order], replacement_rows=replacement[order])


def matched_batch(base, supplement, selection, arm, device):
    if arm not in ARMS:
        raise ValueError('unknown comparison arm')
    rows, replacements = selection['base_rows'], selection['replacement_rows']
    indices = np.flatnonzero(replacements >= 0)
    if not np.array_equal(base['seeds'][rows], selection['seeds']):
        raise ValueError('base rows differ from matched seeds')
    if not np.array_equal(supplement['seeds'][replacements[indices]], selection['seeds'][indices]):
        raise ValueError('replacement rows differ from matched seeds')
    result = {}
    for key in SCHEMA:
        if key in ('rows', 'seeds'):
            continue
        values = np.array(base[key][rows], copy=True)
        if arm == 'recovery':
            values[indices] = supplement[key][replacements[indices]]
        result[key] = torch.from_numpy(values).to(device)
    return result


def fresh_arm(checkpoint, learning_rate, device):
    torch.manual_seed(42)
    model = SpatialOutcomePlanner(checkpoint['planner_config']).to(device).train()
    model.load_state_dict(checkpoint['planner_weights'], strict=True)
    optimizer = optimizer_for(model, learning_rate)
    if optimizer.state:
        raise ValueError('continuation requires new optimizer moments')
    if weights_sha256(model.state_dict()) != weights_sha256(checkpoint['planner_weights']):
        raise ValueError('warm-start initialization differs')
    return model, optimizer


def schedule(step, steps):
    warmup = max(1, round(.03 * steps))
    return min(1., (step + 1) / warmup) if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def selection_record(selection, supplement):
    slots = np.flatnonzero(selection['replacement_rows'] >= 0)
    return {**{k: v.tolist() for k, v in selection.items()},
            'replacement_slots': slots.tolist(),
            'replacement_row_kinds': supplement['row_kind'][selection['replacement_rows'][slots]].tolist(),
            'distinct_levels': len(np.unique(selection['seeds']))}


def checkpoint_for(checkpoint, model, arm, args, report):
    result = copy.copy(checkpoint)
    result.update(planner_weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  planner_initialization={'kind': 'warm_start', 'source_checkpoint_sha256': CHECKPOINT_SHA,
                                          'weights_sha256': report['initial_planner_weights_sha256']},
                  optimizer_state='new; no resumed moments', optimizer='AdamW fused',
                  optimizer_steps=args.steps, learning_rate=args.lr, batch_size=BATCH_SIZE,
                  train_levels=report['train_levels'],
                  continuation_arm=arm, source_checkpoint_sha256=CHECKPOINT_SHA,
                  training_cache=str(args.cache), supplemental_cache=str(args.supplement),
                  cache_manifest_sha256=report['cache_manifest_sha256'],
                  supplemental_manifest_sha256=report['supplemental_manifest_sha256'],
                  source_sha256=report['source_sha256'], recovery_fraction=args.recovery_fraction,
                  policy_row_fraction=args.policy_row_fraction,
                  quality_manifest=report['quality_manifest'],
                  quality_manifest_sha256=report['quality_manifest_sha256'],
                  quality_row_file_sha256=report['quality_row_file_sha256'], quality_filter=report['quality_filter'],
                  sampling='matched uniform level-first with reserved policy-covered TRAIN seeds')
    return result


def run_arm(args, checkpoint, arrays, supplement, sampler, player_weights, weights, report, arm):
    guard(); gpu_available(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, optimizer = fresh_arm(checkpoint, args.lr, 'cuda')
    rng = np.random.default_rng(42)
    entry = dict(initial_weights_sha256=weights_sha256(model.state_dict()),
                 new_optimizer_moments=True, parameters=model.parameter_count(), updates=[])
    report['arms'][arm] = entry
    started = time.monotonic()
    steps = 1 if args.qualify else args.steps
    for step in range(steps):
        available = guard()
        selection = sampler.sample(rng)
        if len(np.unique(selection['seeds'])) != BATCH_SIZE:
            raise ValueError('true B1024 requires 1024 distinct levels')
        if step == 0:
            entry['first_batch'] = selection_record(selection, supplement)
        for group in optimizer.param_groups:
            group['lr'] = args.lr * schedule(step, args.steps)
        items = matched_batch(arrays['train'], supplement, selection, arm, 'cuda')
        record, norm = fit_step(model, optimizer, items, player_weights, weights, args.precision)
        if (step + 1) % 25 == 0 or step + 1 == steps:
            torch.cuda.synchronize()
            update = dict(step=step + 1, loss=float(record['total']), metrics=scalar_metrics(record),
                          all_parameters_have_finite_gradients=True, unclipped_gradient_norm=float(norm),
                          elapsed_seconds=time.monotonic() - started, available_host_bytes=available,
                          peak_gpu_bytes=torch.cuda.max_memory_allocated())
            entry['updates'].append(update)
            write(args.out_dir / 'report.json', report)
            print(json.dumps(dict(event='progress', arm=arm, **update)), flush=True)
        del items, record
    final = weights_sha256(model.state_dict())
    if final == entry['initial_weights_sha256'] or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('optimizer did not produce changed finite weights')
    entry.update(status='passed', final_weights_sha256=final, elapsed_seconds=time.monotonic() - started,
                 peak_gpu_bytes=torch.cuda.max_memory_allocated(), qualification_updates_discarded=args.qualify)
    if not args.qualify:
        entry['validation'] = evaluate(model, arrays['validation'], player_weights, weights, BATCH_SIZE)
        path = args.out_dir / (arm + '.pending.pt')
        torch.save(checkpoint_for(checkpoint, model, arm, args, report), path)
        entry['checkpoint_sha256'] = sha(path)
    del model, optimizer
    torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--quality-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--steps', type=int, default=780)
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--precision', choices=('bf16', 'float32'), default='bf16')
    parser.add_argument('--recovery-fraction', type=float, default=.25)
    parser.add_argument('--policy-row-fraction', type=float, default=.75)
    args = parser.parse_args(argv)
    if (min(args.steps, args.max_seconds) <= 0 or not math.isfinite(args.lr) or args.lr <= 0
            or not 0 < args.recovery_fraction <= 1 or not 0 <= args.policy_row_fraction <= 1):
        parser.error('invalid finite budget, learning rate, or mixture')
    if not args.qualify and args.qualification is None:
        parser.error('completed matched --qualification required')
    guard(); gpu_available(); args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
                  mode='qualification' if args.qualify else 'training', official_inputs_used=False,
                  encoder_frozen=True, planner_horizon=1, persistent_game_memory=False,
                  learned_voluntary_reset=False, batch_size=BATCH_SIZE, planned_steps=args.steps,
                  learning_rate=args.lr, precision=args.precision, recovery_fraction=args.recovery_fraction,
                  policy_row_fraction=args.policy_row_fraction, arms={})
    def timeout(*_):
        raise TimeoutError('bounded matched continuation deadline reached')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(args.max_seconds)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        bindings = Bindings()
        bindings.add(args.checkpoint, CHECKPOINT_SHA)
        sources = [Path(__file__), args.cache / 'manifest.json', args.supplement / 'manifest.json']
        sources += [ROOT / 'tools' / name for name in ('train_reference_spatial_outcomes.py', 'train_reference_outcomes.py',
                    'cache_reference_outcome_inputs.py')]
        sources += list((ROOT / 'pebby/agent').glob('*.py'))
        for path in sources:
            bindings.add(path)
        report['source_sha256'] = dict(bindings.hashes)
        report['cache_manifest_sha256'] = sha(args.cache / 'manifest.json')
        report['supplemental_manifest_sha256'] = sha(args.supplement / 'manifest.json')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint(checkpoint)
        arrays, hashes, stats = load_data(args.cache)
        report['verified_base_cache_hashes'] = hashes
        weights = training_weights(arrays['train'])
        if weights != checkpoint['objective_weights']:
            raise ValueError('base TRAIN loss weights differ from the current head')
        report['objective_weights'] = weights
        supplement, manifest = load_supplement(args.supplement, checkpoint, arrays['train'], bindings)
        report['verified_supplement_files'] = manifest['files']
        quality_rows = load_quality_manifest(args.quality_manifest, arrays['train'], supplement, report, bindings)
        sampler = MatchedSampler(arrays['train'], supplement, args.recovery_fraction, args.policy_row_fraction,
                                 base_rows=quality_rows['base'], supplement_rows=quality_rows['supplement'])
        encoder = WorldPolicy(WorldModelConfig.from_dict(checkpoint['encoder_config']))
        encoder.load_state_dict(checkpoint['encoder_weights'], strict=True)
        report['frozen_encoder_parameters'] = sum(p.numel() for p in encoder.parameters())
        del encoder
        report.update(initial_planner_weights_sha256=weights_sha256(checkpoint['planner_weights']),
                      frozen_encoder_state_elements=sum(v.numel() for v in checkpoint['encoder_weights'].values()),
                      train_levels=len(sampler.seeds), eligible_policy_levels=len(sampler.eligible),
                      reserved_levels_per_batch=sampler.reserved,
                      sampling='matched uniform level-first; original easy-to-hard curriculum not retained')
        if not args.qualify:
            bindings.add(args.qualification)
            qualification = json.loads(args.qualification.read_text())
            validate_qualification(qualification, report)
            report['qualification_sha256'] = sha(args.qualification)
        player_weights = {k: v.cuda() for k, v in checkpoint['encoder_weights'].items() if k.startswith('player_head.')}
        for arm in ARMS:
            run_arm(args, checkpoint, arrays, supplement, sampler, player_weights, weights, report, arm)
        if report['arms']['control']['first_batch'] != report['arms']['recovery']['first_batch']:
            raise ValueError('comparison first batches differ')
        bindings.verify()
        if any(stat(path) != expected for path, expected in stats.items()):
            raise ValueError('base cache changed during continuation')
        if not args.qualify:
            for arm in ARMS:
                (args.out_dir / (arm + '.pending.pt')).replace(args.out_dir / (arm + '.pt'))
        report.update(status='complete', qualification_passed=args.qualify, sources_unchanged=True,
                      finished_local=datetime.now().astimezone().isoformat())
    except BaseException as error:
        for arm in ARMS:
            (args.out_dir / (arm + '.pending.pt')).unlink(missing_ok=True)
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        write(args.out_dir / 'report.json', report)


if __name__ == '__main__':
    main()
