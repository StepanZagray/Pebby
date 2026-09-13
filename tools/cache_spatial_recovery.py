"""Publish frozen public H8 features from a completed generated recovery collection.

Extraction is CUDA FP32 only and never invokes the decision head or a teacher.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from tools import cache_reference_outcome_inputs as base

SOURCE_SHA = '5ea4cccdd01d7b7d43e75565a72b0109031811f4fe156b356e6b3da232201d57'
EXTRA = {'row_kind': ('uint8', ()), 'trajectory_id': ('int64', ()), 'step': ('int64', ())}
SCHEMA = {**base.SCHEMA, **EXTRA}
INPUT_SCHEMA = {**{k: v for k, v in SCHEMA.items() if k not in ('raw', 'state', 'glyph', 'rows')},
                'frames': ('uint8', (8, 64, 64)), 'history_valid': ('bool', (8,)),
                'previous_actions': ('int64', (8,))}


def child(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('source path escapes collection')
    return path


def rehash(bindings):
    bindings.verify()
    for path, expected in bindings.hashes.items():
        if base.sha(path) != expected:
            raise ValueError(f'source hash changed: {path}')
    bindings.verify()


def validate_rows(arrays, seed):
    n = len(arrays['seeds'])
    if n == 0:
        raise ValueError('empty recovery level')
    for key, (dtype, tail) in INPUT_SCHEMA.items():
        value = arrays[key]
        if value.shape != (n, *tail) or not np.can_cast(value.dtype, dtype, casting='safe'):
            raise ValueError(f'recovery source schema mismatch: {key}')
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite source: {key}')
    if not np.all(arrays['seeds'] == seed):
        raise ValueError('level seed and row seeds disagree')
    if np.any(arrays['row_kind'] > 2) or np.any(arrays['trajectory_id'] < 0) or np.any(arrays['step'] < 0):
        raise ValueError('invalid row kind or trajectory index')
    if np.any((arrays['previous_actions'] < -1) | (arrays['previous_actions'] > 3)):
        raise ValueError('invalid public action history')
    if not arrays['history_valid'][:, -1].all():
        raise ValueError('current public history slot must be valid')
    if np.any(arrays['optimal'] > 15) or np.any(arrays['next_optimal'] > 15):
        raise ValueError('invalid four-action target mask')
    return n


def inspect_collection(directory, checkpoint, base_cache, bindings):
    """Bind every input and prove recovery levels belong exclusively to original TRAIN."""
    bindings.add(checkpoint, SOURCE_SHA)
    report_path = directory / 'report.json'
    bindings.add(report_path)
    report = json.loads(report_path.read_text())
    if (report.get('status') != 'complete' or report.get('source_checkpoint_sha256') != SOURCE_SHA
            or report.get('official_inputs_used') is not False
            or report.get('public_inputs') != list(base.PUBLIC)):
        raise ValueError('completed generated-only recovery collection required')
    for path, digest in report.get('source_bindings', {}).items():
        bindings.add(Path(path), digest)
    bank_path = directory / 'train.jsonl'
    bindings.add(bank_path)
    bank = [json.loads(line)['seed'] for line in bank_path.read_text().splitlines() if line.strip()]
    if len(bank) != len(set(bank)):
        raise ValueError('duplicate selected TRAIN seeds')
    manifest_path = base_cache / 'manifest.json'
    bindings.add(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'complete' or manifest.get('parent_sha256') != base.PARENT_SHA
            or not manifest.get('sources_unchanged') or not manifest.get('validation_disjoint')):
        raise ValueError('verified original base cache required')
    sets = {}
    for split, count in [('train', 10000), ('validation', 500)]:
        array = base.open_array(base_cache / split / 'seeds.npy', manifest['arrays'][split]['seeds'], bindings)
        sets[split] = set(map(int, np.unique(array)))
        base.release({'seeds': array}, close=True)
        if len(sets[split]) != count:
            raise ValueError('original TRAIN/VAL level count differs')
    if sets['train'] & sets['validation'] or not set(bank) <= sets['train'] or set(bank) & sets['validation']:
        raise ValueError('recovery bank violates original TRAIN/VAL split')
    levels, seen, total = [], set(), 0
    for entry in report['levels']:
        seed = int(entry['seed'])
        if seed in seen or seed not in bank:
            raise ValueError('unselected or duplicate recovery seed')
        seen.add(seed)
        path = child(directory, entry['array_path'])
        bindings.add(path, entry['sha256'])
        proof_path = child(directory, entry['proof_path'])
        bindings.add(proof_path, entry['proof_sha256'])
        proof = json.loads(proof_path.read_text())
        if (proof.get('status') != 'complete' or proof.get('seed') != seed
                or proof.get('source_checkpoint_sha256') != SOURCE_SHA
                or proof.get('array_sha256') != entry['sha256']):
            raise ValueError('incomplete or mismatched per-level proof')
        for source_path, digest in proof.get('source_bindings', {}).items():
            bindings.add(Path(source_path), digest)
        with np.load(path, allow_pickle=False) as arrays:
            n = validate_rows(arrays, seed)
        if proof.get('rows') != n:
            raise ValueError('per-level proof row count differs')
        levels.append({**entry, 'rows': n, 'offset': total})
        total += n
    if seen != set(bank) or not total:
        raise ValueError('collection does not cover its selected TRAIN bank')
    return levels, total


def payload(encoder, arrays, chosen, offset, device):
    encoded = base.encode_current(encoder, arrays, chosen, device)
    result = {k: encoded[k].cpu().numpy() for k in ('raw', 'state', 'glyph')}
    result.update({k: np.array(arrays[k][chosen], copy=True) for k in (*base.LABELS, *EXTRA)})
    result['rows'] = np.asarray(chosen, dtype=np.int64) + offset
    return result


def feature_parity(encoder, arrays, chosen, cached, offset, device, *, batch_size=64):
    """Check storage against original extraction batches; measure B4/B1 separately.

    CUDA auto kernels can depend on batch shape. A sparse diagnostic batch must
    not be treated as an exact reconstruction of the original B64/tail call.
    """
    import torch
    from pebby.agent.neural_outcome_policy import encoder_execution
    chosen = np.asarray(chosen, dtype=np.int64)
    count = len(arrays['frames'])
    if batch_size < 1 or chosen.ndim != 1 or not len(chosen) or np.any(chosen < 0) or np.any(chosen >= count):
        raise ValueError('valid selected rows and original extraction batch size required')
    def native(rows):
        inputs = [torch.from_numpy(np.array(arrays[k][rows], copy=True)).to(device) for k in base.PUBLIC]
        with encoder_execution(torch.device(device)):
            return encoder.encode(*inputs)
    def metrics(left, right):
        delta = (left - right).float()
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError('nonfinite native parity features')
        return dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                    outside_tolerance=int((delta.abs() > (3e-4 + 3e-4 * right.abs())).sum()),
                    elements=delta.numel())
    keys = ('raw', 'state', 'glyph')
    reference = {k: {} for k in keys}
    batches, maxima = [], dict.fromkeys(keys, 0.)
    for begin in np.unique((chosen // batch_size) * batch_size):
        end = min(int(begin) + batch_size, count)
        rows = np.arange(begin, end)
        encoded = native(rows)
        errors = {}
        for key in keys:
            stored = torch.from_numpy(np.array(cached[key][offset + rows], copy=True)).to(device)
            torch.testing.assert_close(stored, encoded[key], atol=3e-4, rtol=3e-4)
            errors[key] = metrics(stored, encoded[key])
            maxima[key] = max(maxima[key], errors[key]['max_abs'])
            for row in chosen[(chosen >= begin) & (chosen < end)]:
                reference[key][int(row)] = encoded[key][int(row - begin)].clone()
        batches.append(dict(source_start=int(begin), source_stop=end, global_start=offset + int(begin),
                            global_stop=offset + end, batch_rows=end - int(begin), errors=errors))
    expected = {key: torch.stack([reference[key][int(row)] for row in chosen]) for key in keys}
    grouped, single = native(chosen), {key: [] for key in keys}
    for row in chosen:
        encoded = native(np.array([row]))
        for key in keys:
            single[key].append(encoded[key][0])
    return dict(same_batch_max_abs=maxima, original_batches=batches, selected_rows=chosen.tolist(),
                tolerance=dict(atol=3e-4, rtol=3e-4),
                batch_size_sensitivity=dict(
                    diagnostic_only=True, grouped_batch_rows=len(chosen),
                    grouped_vs_original={key: metrics(grouped[key], expected[key]) for key in keys},
                    single_vs_original={key: metrics(torch.stack(single[key]), expected[key]) for key in keys}))


def validate_published(directory):
    directory = Path(directory)
    bindings = base.Bindings()
    bindings.add(directory / 'manifest.json')
    manifest = json.loads((directory / 'manifest.json').read_text())
    from pebby.agent.neural_outcome_policy import ENCODER_RUNTIME
    if (manifest.get('status') != 'complete' or manifest.get('split') != 'train'
            or manifest.get('official_inputs_used') is not False or not manifest.get('sources_unchanged')
            or manifest.get('source_checkpoint_sha256') != SOURCE_SHA
            or manifest.get('encoder_parent_sha256') != base.PARENT_SHA
            or manifest.get('encoder_runtime') != ENCODER_RUNTIME
            or manifest.get('public_inputs') != list(base.PUBLIC)
            or not manifest.get('validation_disjoint') or not manifest.get('parity')
            or len(manifest.get('encoder_weights_sha256', '')) != 64
            or not manifest.get('source_bindings')):
        raise ValueError('incomplete or invalid recovery feature manifest')
    n = manifest['rows']
    if not isinstance(n, int) or n <= 0 or set(manifest['files']) != {k + '.npy' for k in SCHEMA}:
        raise ValueError('invalid recovery output schema')
    for key, (dtype, tail) in SCHEMA.items():
        info = manifest['files'][key + '.npy']
        if info['shape'] != [n, *tail] or info['dtype'] != np.dtype(dtype).str:
            raise ValueError(f'invalid output schema: {key}')
        array = base.open_array(directory / (key + '.npy'), info, bindings)
        try:
            for start in range(0, n, 1024):
                chunk = array[start:start + 1024]
                if not np.isfinite(chunk).all():
                    raise ValueError(f'nonfinite output: {key}')
                if key == 'rows' and not np.array_equal(chunk, np.arange(start, min(start + 1024, n))):
                    raise ValueError('recovery row IDs must preserve global source order')
                if key == 'row_kind' and np.any(chunk > 2):
                    raise ValueError('invalid row kind')
        finally:
            base.release({key: array}, close=True)
    manifest['validated_output_stats'] = bindings.verify()
    manifest['validated_output_hashes'] = bindings.hashes
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base-cache', type=Path, default=base.ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--deadline-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 64 or args.deadline_seconds <= 0:
        parser.error('batch size must be 1..64 and deadline positive')
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    report = dict(status='validating', pid=os.getpid(), start_ticks=base.process_start_ticks(),
                  started_local=datetime.now().astimezone().isoformat(), split='train', official_inputs_used=False,
                  source_checkpoint_sha256=SOURCE_SHA, public_inputs=list(base.PUBLIC), progress_rows=0,
                  extraction_batch_size=args.batch_size)
    bindings, outputs = base.Bindings(), {}
    staging = output.with_name(f'.{output.name}.staging-{os.getpid()}')
    def deadline(*_):
        raise TimeoutError('bounded recovery encoding deadline expired')
    old_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    policy = None
    try:
        base.guard(report, started, args.deadline_seconds)
        levels, n = inspect_collection(args.source_dir.resolve(), args.checkpoint.resolve(), args.base_cache.resolve(), bindings)
        for path in [Path(__file__), Path(base.__file__), *sorted((base.ROOT / 'pebby/agent').glob('*.py'))]:
            bindings.add(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        required = sum(n * np.dtype(dtype).itemsize * int(np.prod(tail)) + 128 for dtype, tail in SCHEMA.values())
        if shutil.disk_usage(output.parent).free < required + 2**30:
            raise OSError('insufficient disk space plus 1 GiB margin')
        staging.mkdir()
        base.write_json(staging / 'progress.json', report)
        print(json.dumps(report), flush=True)
        base.require_no_foreign_cuda()
        import torch
        from pebby.agent.spatial_outcome_policy import load_checkpoint
        from pebby.agent.neural_outcome_policy import ENCODER_RUNTIME, encoder_execution, weights_sha256
        from pebby.agent.world_runtime import configure_execution
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        policy, metadata = load_checkpoint(args.checkpoint, 'cuda')
        policy.eval().requires_grad_(False)
        encoder = policy.encoder
        encoder.encoder_chunk_size = 0
        configure_execution(encoder, compile_core=False, temporal_backend='auto')
        report.update(encoder_parent_sha256=metadata['encoder_parent_sha256'],
                      encoder_weights_sha256=metadata['encoder_weights_sha256'], encoder_runtime=ENCODER_RUNTIME,
                      rows=n, levels=levels, validation_disjoint=True, parity=[])
        outputs = {key: np.lib.format.open_memmap(staging / f'{key}.npy', mode='w+', dtype=dtype, shape=(n, *tail))
                   for key, (dtype, tail) in SCHEMA.items()}
        with torch.inference_mode(), encoder_execution(torch.device('cuda')):
            for entry in levels:
                base.guard(report, started, args.deadline_seconds)
                base.require_no_foreign_cuda()
                with np.load(child(args.source_dir, entry['array_path']), allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in INPUT_SCHEMA}
                validate_rows(arrays, entry['seed'])
                for start in range(0, entry['rows'], args.batch_size):
                    base.guard(report, started, args.deadline_seconds)
                    chosen = np.arange(start, min(start + args.batch_size, entry['rows']))
                    values = payload(encoder, arrays, chosen, entry['offset'], 'cuda')
                    for key, value in values.items():
                        outputs[key][entry['offset'] + chosen] = value
                    report['progress_rows'] += len(chosen)
                if len(report['parity']) < 3:
                    chosen = np.unique(np.linspace(0, entry['rows'] - 1, min(4, entry['rows']), dtype=np.int64))
                    report['parity'].append(dict(seed=entry['seed'], errors=feature_parity(encoder, arrays, chosen, outputs, entry['offset'], 'cuda', batch_size=args.batch_size)))
                del arrays
                base.release(outputs, flush=True)
                base.write_json(staging / 'progress.json', report)
                print(json.dumps(dict(pid=os.getpid(), rows=report['progress_rows'], total=n)), flush=True)
        if weights_sha256(encoder.state_dict()) != report['encoder_weights_sha256']:
            raise ValueError('frozen encoder weights changed')
        if any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
            raise ValueError('encoder gradients enabled')
        base.release(outputs, close=True, flush=True)
        report['files'] = {}
        for key, (dtype, tail) in SCHEMA.items():
            base.guard(report, started, args.deadline_seconds)
            path = staging / f'{key}.npy'
            report['files'][path.name] = dict(sha256=base.sha(path), shape=[n, *tail], dtype=np.dtype(dtype).str)
            path.chmod(0o444)
        rehash(bindings)
        report.update(status='complete', sources_unchanged=True, source_bindings=bindings.hashes,
                      source_stats_after=bindings.verify(), finished_local=datetime.now().astimezone().isoformat())
        base.write_json(staging / 'manifest.json', report)
        validate_published(staging)
        (staging / 'progress.json').unlink()
        (staging / 'manifest.json').chmod(0o444)
        base.publish_directory(staging, output)
        return report
    except BaseException as error:
        if staging.exists():
            base.write_json(staging / 'progress.json', {**report, 'status': 'failed', 'error': repr(error)})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        base.release(outputs, close=True)
        del policy


if __name__ == '__main__':
    main()
