"""Bounded generated-only spatial transition experiment, independent of policy.

Cache labels supervise outputs; the transition receives only fields and actions.
Validation never selects a checkpoint or determines the number of updates.
"""
from pebby.ls20.provenance import (validate_row_difficulties, metadata_difficulty_stages, metadata_difficulty_version)

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.structured_transition import StructuredTransition
from pebby.agent.structured_objective import objective, diagnostics


LABELS = ('player_cell', 'next_player_cell', 'triple', 'next_triple', 'steps',
          'next_steps', 'lives', 'next_lives', 'lost_life', 'terminal', 'won')
EVENTS = ('lost_life', 'terminal', 'won')
ARRAYS = ('fields', 'next_fields', 'seeds', 'difficulties', *LABELS)


def autocast(device):
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def load_cache(path, split):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('source') != 'generated_only'
            or manifest.get('split') != split
            or manifest.get('format') != 'pebby.structured-field-cache.v1'):
        raise ValueError('cache requires completed generated provenance in requested split')
    if not manifest.get('field_encoder'):
        raise ValueError('cache lacks frozen field encoder provenance')
    arrays = {}
    for name in ARRAYS:
        info = manifest['arrays'][name]
        source = path / (name + '.npy')
        if digest(source) != info['sha256']:
            raise ValueError(f'cache array digest mismatch: {name}')
        arrays[name] = np.load(source, mmap_mode='r', allow_pickle=False)
    n = len(arrays['seeds'])
    if n != len(np.unique(arrays['seeds'])):
        raise ValueError('one row per distinct level required for independent batches')
    low, high = (0, 1_000_000) if split == 'train' else (1_000_000, 2_000_000)
    if not np.all((arrays['seeds'] >= low) & (arrays['seeds'] < high)):
        raise ValueError('generated split namespace violated')
    expected = {'fields': (n, 148, 96), 'next_fields': (n, 4, 148, 96),
                'player_cell': (n, 2), 'next_player_cell': (n, 4, 2),
                'triple': (n, 3), 'next_triple': (n, 4, 3)}
    for name in ARRAYS:
        shape = expected.get(name, (n, 4) if name.startswith('next_') or name in EVENTS else (n,))
        if arrays[name].shape != shape:
            raise ValueError(f'cache shape mismatch: {name}')
        if name in ('fields', 'next_fields'):
            if not np.issubdtype(arrays[name].dtype, np.floating):
                raise ValueError('fields must be floating point')
        elif name not in EVENTS and not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f'noninteger label: {name}')
    validate_row_difficulties(arrays['seeds'], arrays['difficulties'], manifest)
    if not np.all(arrays['terminal'][arrays['won'].astype(bool)]):
        raise ValueError('winning branches must terminate')
    return arrays, manifest


def branch_batch(data, rows, actions, device):
    """Select one action per distinct source level, preserving exact labels."""
    rows, actions = np.asarray(rows), np.asarray(actions)
    if rows.ndim != 1 or actions.shape != rows.shape or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError('row and integer action vectors must agree')
    if np.any((actions < 0) | (actions >= 4)):
        raise ValueError('branch action outside0..3')
    field = torch.as_tensor(np.array(data['fields'][rows]), device=device).float()
    following = torch.as_tensor(np.array(data['next_fields'][rows, actions]), device=device).float()
    labels = {}
    for name in LABELS:
        selected = data[name][rows, actions] if name.startswith('next_') or name in EVENTS else data[name][rows]
        labels[name] = torch.as_tensor(np.array(selected), device=device)
    return field, following, torch.as_tensor(actions, device=device).long(), labels


def training_scale(data):
    # Train current fields only; never read validation statistics.
    total = np.zeros(48, dtype=np.float64)
    squares = np.zeros(48, dtype=np.float64)
    count = 0
    for start in range(0, len(data['seeds']), 64):
        chunk = np.asarray(data['fields'][start:start+64, :, :48], dtype=np.float64)
        total += chunk.sum((0, 1))
        squares += np.square(chunk).sum((0, 1))
        count += chunk.shape[0] * chunk.shape[1]
    if not count:
        raise ValueError('empty training cache')
    return np.sqrt(np.maximum(squares/count - np.square(total/count), .01)).astype(np.float32)


def event_counts(data):
    result = {}
    for name in EVENTS:
        flags = np.asarray(data[name], dtype=bool)
        result[name] = {'positive_branches': int(flags.sum()),
                        'positive_levels': int(flags.any(1).sum()),
                        'branches': int(flags.size), 'levels': int(len(flags))}
    failure = np.asarray(data['terminal'], dtype=bool) & ~np.asarray(data['won'], dtype=bool)
    result['terminal_failure'] = {'positive_branches': int(failure.sum()),
                                  'positive_levels': int(failure.any(1).sum())}
    return result


def sample_rows(data, batch_size, progress, rng):
    if batch_size > len(data['seeds']):
        raise ValueError('batch exceeds distinct training levels')
    difficulty = np.asarray(data['difficulties'], dtype=np.float64)
    # Smooth easy-to-hard shift. Draw without replacement within every update.
    probabilities = np.exp((2 * progress - 1) * (difficulty - 3) * .5)
    probabilities /= probabilities.sum()
    return rng.choice(len(difficulty), batch_size, replace=False, p=probabilities)


def new_model(seed, device, loops):
    torch.manual_seed(seed)
    return StructuredTransition(loops=loops).to(device)


def probe_batch(data, scale, pos_weight, args):
    """Disposable real optimizer updates; descending powers of two up to1024."""
    limit = min(args.max_batch, len(data['seeds']))
    size = 2 ** (limit.bit_length() - 1)
    attempts = []
    while size:
        started = time.monotonic()
        model = optimizer = result = batch = None
        try:
            model = new_model(args.seed, args.device, args.loops).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            batch = branch_batch(data, np.arange(size), np.arange(size) % 4, args.device)
            with autocast(args.device):
                result = objective(model, *batch, scale, pos_weight=pos_weight)
            if not bool(torch.isfinite(result['total'])):
                raise ValueError('nonfinite preflight loss')
            result['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            attempts.append({'batch_size': size, 'status': 'passed',
                             'loss': float(result['total'].detach()), 'gradient_norm': float(norm),
                             'seconds': time.monotonic()-started,
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None})
            return size, attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({'batch_size': size, 'status': 'out_of_memory', 'seconds': time.monotonic()-started})
            size //= 2
        finally:
            del result, batch, optimizer, model
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    raise RuntimeError('no power-of-two batch fits')


@torch.no_grad()
def evaluate(model, data, scale, pos_weight, device, batch_size=128):
    model.eval()
    sums, counts = {}, {}
    losses, rows_seen = {}, 0
    started = time.monotonic()
    for start in range(0, len(data['seeds']), batch_size):
        rows = np.arange(start, min(start+batch_size, len(data['seeds'])))
        for action in range(4):
            batch = branch_batch(data, rows, np.full(len(rows), action, dtype=np.int64), device)
            with autocast(device):
                result = objective(model, *batch, scale, pos_weight=pos_weight)
            field, following, actions, labels = batch
            measured = diagnostics(result, field, following, labels, scale)
            for key, value in measured['sums'].items():
                sums[key] = sums.get(key, 0.) + float(value)
            for key, value in measured['counts'].items():
                counts[key] = counts.get(key, 0.) + float(value)
            for key, value in result['losses'].items():
                losses[key] = losses.get(key, 0.) + float(value) * len(rows)
            # Fixed wrong-action ablation; evaluate against the same actual target.
            with autocast(device):
                wrong = model.predict(field, (actions+1) % 4)
                wrong_player = model.readout(wrong)['player_logits'].argmax(-1)
            target_player = labels['next_player_cell'][:, 1] * 12 + labels['next_player_cell'][:, 0]
            moved = (labels['player_cell'] != labels['next_player_cell']).any(-1)
            key = 'permuted_action_moved_player_accuracy'
            sums[key] = sums.get(key, 0.) + float(((wrong_player == target_player) & moved).sum())
            counts[key] = counts.get(key, 0.) + float(moved.sum())
            rows_seen += len(rows)
    return {'sums': sums, 'counts': counts,
            'metrics': {key: sums[key]/counts[key] if counts.get(key, 0) else None for key in sums},
            'row_weighted_batch_losses': {key: value/rows_seen for key, value in losses.items()},
            'loss_aggregation_note': 'Mask/mass-weighted losses depend on batch partition; use additive diagnostics for population comparisons.',
            'branches': rows_seen, 'levels': len(data['seeds']),
            'events': event_counts(data), 'elapsed_seconds': time.monotonic()-started,
            'scope': 'one-step generated transition fidelity; no control or multi-step success claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--max-batch', type=int, default=1024)
    parser.add_argument('--eval-batch', type=int, default=128)
    parser.add_argument('--loops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_batch <= 1024 or args.max_batch & (args.max_batch-1):
        parser.error('max-batch must be a power of two in1..1024')
    if not 1 <= args.updates <= 2000 or not 1 <= args.seconds <= 1800:
        parser.error('this bounded experiment permits1..2000 updates and1..1800 seconds')
    if Path(args.report).exists() or (not args.preflight_only and Path(args.checkpoint).exists()):
        parser.error('refusing to overwrite experiment artifacts')
    torch.set_num_threads(1)
    started = time.monotonic()
    report = {'status': 'running', 'pid': os.getpid(), 'args': vars(args),
              'control_evaluated': False, 'official_inputs_used': False,
              'precision': 'bfloat16_autocast' if args.device == 'cuda' else 'float32',
              'scope': 'frozen generated features; one-step transition diagnostic'}
    atomic_json(args.report, report)
    def expired(_signum, _frame):
        raise TimeoutError('bounded experiment deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    try:
        train, train_manifest = load_cache(Path(args.cache)/'train', 'train')
        validation, val_manifest = load_cache(Path(args.cache)/'validation', 'validation')
        if np.intersect1d(train['seeds'], validation['seeds']).size:
            raise ValueError('training and validation levels overlap')
        if metadata_difficulty_version(train_manifest) != metadata_difficulty_version(val_manifest):
            raise ValueError('train and validation difficulty versions differ')
        if train_manifest['field_encoder'] != val_manifest['field_encoder']:
            raise ValueError('training and validation field encoders differ')
        sources = {str(Path(args.cache)/split/'manifest.json'): digest(Path(args.cache)/split/'manifest.json')
                   for split in ('train', 'validation')}
        for split, manifest in (('train', train_manifest), ('validation', val_manifest)):
            sources.update({str(Path(args.cache)/split/(name+'.npy')): manifest['arrays'][name]['sha256']
                            for name in ARRAYS})
        for source in (__file__, 'pebby/agent/structured_transition.py', 'pebby/agent/structured_objective.py'):
            sources[str(source)] = digest(source)
        scale = torch.as_tensor(training_scale(train), device=args.device)
        prevalence = np.array([np.asarray(train[name], dtype=bool).mean() for name in EVENTS])
        positive_weight = torch.as_tensor(np.clip((1-prevalence)/np.maximum(prevalence, 1e-8), 1, 20),
                                         device=args.device, dtype=torch.float32)
        report.update(sources=sources, train_levels=len(train['seeds']), validation_levels=len(validation['seeds']),
                      train_events=event_counts(train), validation_events=event_counts(validation),
                      event_positive_weights=positive_weight.tolist(), feature_scale=scale.tolist())
        batch_size, attempts = probe_batch(train, scale, positive_weight, args)
        report.update(batch_size=batch_size, batch_probe=attempts)
        atomic_json(args.report, report)
        if args.preflight_only:
            if any(digest(source) != expected for source, expected in sources.items()):
                raise ValueError('source changed during preflight')
            report.update(status='preflight_complete', source_unchanged=True)
            return
        model = new_model(args.seed, args.device, args.loops).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        rng = np.random.default_rng(args.seed)
        draws = np.zeros(len(metadata_difficulty_stages(train_manifest)), dtype=np.int64)
        report.update(parameters=model.parameter_count(), training=[])
        for step in range(args.updates):
            rows = sample_rows(train, batch_size, step/max(args.updates-1, 1), rng)
            if len(np.unique(train['seeds'][rows])) != batch_size:
                raise RuntimeError('batch lost distinct-level invariant')
            actions = rng.integers(4, size=batch_size)
            batch = branch_batch(train, rows, actions, args.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.device):
                result = objective(model, *batch, scale, pos_weight=positive_weight)
            if not bool(torch.isfinite(result['total'])):
                raise ValueError('nonfinite training loss')
            result['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            draws += np.bincount(train['difficulties'][rows], minlength=len(draws)+1)[1:len(draws)+1]
            if (step+1) % 20 == 0 or step == 0 or step+1 == args.updates:
                event = {'step': step+1, 'loss': float(result['total'].detach()),
                         'losses': {k: float(v.detach()) for k, v in result['losses'].items()},
                         'gradient_norm': float(norm), 'elapsed_seconds': time.monotonic()-started}
                report['training'].append(event)
                report.update(completed_updates=step+1, difficulty_draws=draws.tolist())
                atomic_json(args.report, report)
                print(json.dumps(event), flush=True)
            del result, batch
        # Fixed final checkpoint; selection uses no validation metric.
        if any(digest(source) != expected for source, expected in sources.items()):
            raise ValueError('source changed before checkpoint publication')
        target = Path(args.checkpoint)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+f'.{os.getpid()}.tmp')
        torch.save({'format': 'pebby.structured-transition.v1', 'config': model.config(),
                    'weights': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'sources': sources, 'cache_manifests': {'train': train_manifest, 'validation': val_manifest},
                    'feature_scale': scale.cpu(), 'event_positive_weights': positive_weight.cpu(),
                    'updates': args.updates, 'batch_size': batch_size, 'seed': args.seed,
                    'scope': report['scope'], 'policy_integrated': False}, temporary)
        os.replace(temporary, target)
        report.update(checkpoint=str(target), checkpoint_sha256=digest(target), status='evaluation_running')
        atomic_json(args.report, report)
        report['validation'] = evaluate(model, validation, scale, positive_weight, args.device, args.eval_batch)
        # Training readout fit is necessary to distinguish underfit and transfer.
        report['train_evaluation'] = evaluate(model, train, scale, positive_weight, args.device, args.eval_batch)
        if any(digest(source) != expected for source, expected in sources.items()):
            raise ValueError('source changed during experiment')
        report.update(status='complete', source_unchanged=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(args.report, report)


if __name__ == '__main__':
    main()
