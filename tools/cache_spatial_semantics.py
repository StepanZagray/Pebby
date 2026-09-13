"""Frozen public-current-pixel semantic addon; complete existing row order only.

No labels enter the perceptor. TRAIN quality masks belong to the consumer: this
cache preserves every original and recent row, including zero-optimal failures.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from tools import cache_reference_outcome_inputs as base
from tools import cache_spatial_repair_v3 as recent_cache
from tools.cache_spatial_recovery import child

ROOT = base.ROOT
TEACHER = ROOT / 'artifacts/reference-scene-initial-probe-v1/pixel-control/cell-appearance-initial-candidate.pt'
TEACHER_SHA = '143e0cdde09889ed11e938a004e77b5ed968e93f0cd8353be74f82bf5370082b'
FORMAT = 'pebby.spatial-semantic-features.v1'
CHANNELS = ['wall', 'goal', 'cycler_shape', 'cycler_color', 'cycler_rotation', 'launcher', 'refill', 'player',
            *[f'shape_{i}' for i in range(6)], *[f'color_{i}' for i in range(4)], *[f'rotation_{i}' for i in range(4)]]
SCHEMA = {'semantic': ('float32', (144, 22)), 'rows': ('int64', ()), 'seeds': ('int64', ())}
RUNTIME = dict(precision='float32', storage='float32', execution='native_eager',
               matmul_tf32=False, cudnn_tf32=True, public_input='current frame only',
               probability_api='pebby.agent.spatial_semantic_outcome_policy.semantic_probabilities')
PARITY_ATOL = 3e-6
RUNTIME_SOURCES = tuple(ROOT / 'pebby/agent' / name for name in
    ('cell_appearance.py', 'spatial_semantic_outcome_policy.py', 'neural_outcome_policy.py'))


def current_frames(frames, rows):
    if frames.ndim != 4 or frames.shape[1:] != (8, 64, 64) or frames.dtype != np.uint8:
        raise ValueError('source must be public uint8 H8 frames')
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.dtype != np.int64 or np.any((rows < 0) | (rows >= len(frames))):
        raise ValueError('invalid current frame source rows')
    # Index the history axis before copying, so only current pixels enter RAM.
    result = np.array(frames[rows, -1], copy=True)
    if np.any(result > 15):
        raise ValueError('public palette outside0..15')
    return result


def check_alignment(rows, cached_seeds, source_seeds):
    rows = np.asarray(rows)
    if (rows.dtype != np.int64 or rows.ndim != 1 or len(rows) != len(cached_seeds)
            or len(np.unique(rows)) != len(rows) or np.any((rows < 0) | (rows >= len(source_seeds)))):
        raise ValueError('invalid, duplicate or incomplete source row mapping')
    if not np.array_equal(source_seeds[rows], cached_seeds):
        raise ValueError('public frame seeds disagree with feature row order')


def validate_probabilities(value, count):
    if value.shape != (count, 144, 22) or value.dtype != np.float32:
        raise ValueError('semantic features must be float32[N,144,22]')
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('invalid semantic probabilities')
    for start, stop in ((8, 14), (14, 18), (18, 22)):
        if not np.allclose(value[..., start:stop].sum(-1), 1., atol=5e-6, rtol=0):
            raise ValueError('attribute probabilities must sum to one')


def encode_frames(perceptor, frames, device):
    import torch
    from pebby.agent.spatial_semantic_outcome_policy import semantic_probabilities
    from pebby.agent.neural_outcome_policy import encoder_execution
    if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (64, 64) or np.any(frames > 15):
        raise ValueError('current pixels must be uint8[B,64,64] palette indices')
    with torch.inference_mode(), encoder_execution(torch.device(device)):
        value = semantic_probabilities(perceptor, torch.from_numpy(np.array(frames, copy=True)).to(device)).cpu().numpy()
    validate_probabilities(value, len(frames))
    return value


def read_recent_level(path, entry, cached):
    """Decode each compressed column once and release H8 before returning."""
    offset, count = entry['offset'], entry['rows']
    with np.load(path, allow_pickle=False) as archive:
        histories = archive['frames']
        if len(histories) != count:
            raise ValueError('recent raw frame row count differs')
        frames = current_frames(histories, np.arange(count, dtype=np.int64))
        for key, values in cached.items():
            if key in ('raw', 'state', 'glyph', 'rows'):
                continue
            actual = archive[key]
            if not np.array_equal(actual, values[offset:offset + count]):
                raise ValueError(f'recent raw/cached row alignment differs: {key}')
        if not np.array_equal(cached['seeds'][offset:offset + count], np.full(count, entry['seed'], dtype=np.int64)):
            raise ValueError('recent level seed differs')
    return frames


def original_source(directory, split, manifest, bindings):
    archive = directory / f'{split}.npz'
    digest = manifest['source_sha256'].get(str(archive.resolve()))
    if digest is None:
        raise ValueError('original public archive is not bound by feature manifest')
    candidates = []
    for name, expected in manifest['source_sha256'].items():
        path = Path(name)
        if path.name == 'manifest.json' and path.parent.parent == directory / 'array-cache':
            bindings.add(path, expected)
            info = json.loads(path.read_text())
            if info.get('source_sha256') == digest:
                candidates.append((path, info))
    if len(candidates) != 1:
        raise ValueError('exactly one published frame array cache must match source archive')
    path, info = candidates[0]
    arrays = {key: base.open_array(path.parent / f'{key}.npy', info['arrays'][key], bindings)
              for key in ('frames', 'seeds')}
    return arrays, dict(array_manifest=str(path), array_manifest_sha256=bindings.hashes[str(path)],
                       source_archive_sha256_from_bound_manifest=digest)


def _adopt(bindings, manifest):
    bindings.hashes.update(manifest['validated_output_hashes'])
    bindings.before.update(manifest['validated_output_stats'])


def validate_published(path):
    """Hash addon/runtime/teacher files; full frame hashes remain manifest provenance.

    Consumers receive post-use stat guards. Frame sources are fully hashed at
    extraction, but are not needed or rehashed to load immutable addon features.
    """
    path = Path(path).resolve(); bindings = base.Bindings()
    bindings.add(path / 'manifest.json')
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
            or manifest.get('teacher_sha256') != TEACHER_SHA or manifest.get('channels') != CHANNELS
            or manifest.get('runtime') != RUNTIME or not manifest.get('sources_unchanged')
            or not manifest.get('weights_unchanged') or not manifest.get('all_rows_retained')
            or manifest.get('official_inputs_used') is not False or not manifest.get('validation_disjoint')
            or set(manifest.get('files', {})) != {'train', 'validation', 'recent'}):
        raise ValueError('not a complete frozen public semantic addon')
    teacher = Path(manifest.get('teacher_path', '')).resolve()
    required = {str(p.resolve()) for p in RUNTIME_SOURCES} | {str(teacher)}
    runtime_sources = manifest.get('runtime_source_bindings', {})
    if set(runtime_sources) != required or runtime_sources.get(str(teacher)) != TEACHER_SHA:
        raise ValueError('runtime source/teacher bindings incomplete')
    for source, digest in runtime_sources.items():
        bindings.add(source, digest)
    seeds = {}
    for split, files in manifest['files'].items():
        if set(files) != set(SCHEMA):
            raise ValueError('semantic addon columns differ')
        count = manifest['counts'][split]
        for key, (dtype, tail) in SCHEMA.items():
            info = files[key]
            if info['shape'] != [count, *tail] or info['dtype'] != np.dtype(dtype).str:
                raise ValueError('semantic addon schema differs')
            array = base.open_array(child(path, info['path']), info, bindings)
            try:
                if key == 'semantic':
                    for start in range(0, count, 1024):
                        validate_probabilities(array[start:start + 1024], min(1024, count - start))
                elif key == 'seeds':
                    seeds[split] = np.array(array, copy=True)
                else:
                    if np.any(array < 0) or len(np.unique(array)) != count:
                        raise ValueError('semantic source rows must be nonnegative and distinct')
                    if split == 'recent' and not np.array_equal(array, np.arange(count)):
                        raise ValueError('recent addon must retain complete contiguous source rows')
                if key in ('rows', 'seeds') and base.sha(child(path, info['path'])) != manifest['aligned_source_files'][split][key]['sha256']:
                    raise ValueError('semantic row identity differs from underlying feature cache')
            finally:
                base.release({key: array}, close=True)
        parity = manifest.get('parity', {}).get(split, {})
        if not parity.get('samples') or (not np.isfinite(parity.get('maximum_absolute_error', float('inf')))
                or parity['maximum_absolute_error'] > PARITY_ATOL):
            raise ValueError('single-frame online parity did not pass')
    if np.intersect1d(seeds['train'], seeds['validation']).size or not set(seeds['recent']) <= set(seeds['train']):
        raise ValueError('semantic split leakage or unselected recent TRAIN seeds')
    manifest['validated_output_hashes'] = bindings.hashes
    manifest['validated_output_stats'] = bindings.verify()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, default=TEACHER)
    parser.add_argument('--base-cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--recent-cache', type=Path, default=ROOT / 'data/spatial-repair-v3-current320-inputs')
    parser.add_argument('--frame-source', type=Path, default=base.DATA)
    parser.add_argument('--recent-source', type=Path, default=ROOT / 'data/spatial-repair-v3-current320')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--deadline-seconds', type=int, default=1200)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128 or args.deadline_seconds < 1:
        parser.error('batch1..128 and positive deadline required')
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic(); bindings = base.Bindings(); staging = output.with_name(f'.{output.name}.staging-{os.getpid()}')
    outputs = {}; opened = []; perceptor = None
    report = dict(format=FORMAT, status='validating', pid=os.getpid(), start_ticks=base.process_start_ticks(),
        started_local=datetime.now().astimezone().isoformat(), teacher_sha256=TEACHER_SHA, channels=CHANNELS,
        runtime=RUNTIME, device=args.device, batch_size=args.batch_size, deadline_seconds=args.deadline_seconds,
        memory_reserve_gib=6, abort_memavailable_gib=7, official_inputs_used=False, all_rows_retained=True,
        quality_allowlists_applied=False, quality_allowlists_required_for_training=True,
        input='current public frames only; no privileged labels or successor pixels', progress={}, parity={})
    def expired(*_):
        raise TimeoutError('bounded semantic cache deadline exceeded')
    old_alarm = signal.signal(signal.SIGALRM, expired); signal.alarm(args.deadline_seconds)
    try:
        base.guard(report, started, args.deadline_seconds)
        bindings.add(args.teacher, TEACHER_SHA)
        original = base.validate_published(args.base_cache); _adopt(bindings, original)
        recent = recent_cache.validate_published(args.recent_cache); _adopt(bindings, recent)
        report.update(base_manifest_sha256=bindings.add(args.base_cache / 'manifest.json'),
            recent_manifest_sha256=bindings.add(args.recent_cache / 'manifest.json'),
            base_cache=str(args.base_cache.resolve()), recent_cache=str(args.recent_cache.resolve()),
            counts=dict(train=80000, validation=4000, recent=recent['rows']), aligned_source_files={}, frame_sources={})
        if Path(recent['base_cache']).resolve() != args.base_cache.resolve():
            raise ValueError('recent collection belongs to a different base cache')
        collection_path = args.recent_source.resolve() / 'report.json'
        bindings.add(collection_path, recent['source_bindings'].get(str(collection_path)))
        if str(collection_path) not in recent['source_bindings']:
            raise ValueError('recent raw collection report is not bound by its feature cache')
        collection = json.loads(collection_path.read_text())
        if (collection.get('status') != 'complete' or collection.get('rows') != recent['rows']
                or collection.get('source_checkpoint_sha256') != recent_cache.CHECKPOINT_SHA):
            raise ValueError('raw collection differs from validated cached source')
        source_entries = {e['seed']: e for e in collection['levels']}
        for entry in recent['levels']:
            if any(entry[k] != source_entries.get(entry['seed'], {}).get(k) for k in ('sha256', 'array_path', 'rows')):
                raise ValueError('raw and cached collection level manifests disagree')
        from pebby.agent import spatial_semantic_outcome_policy as online
        for path in [Path(__file__), Path(base.__file__), Path(recent_cache.__file__), ROOT / 'tools/cache_spatial_recovery.py',
                     Path(online.__file__), *[ROOT / 'pebby/agent' / name for name in
                     ('cell_appearance.py', 'spatial_semantic_outcome_planner.py', 'neural_outcome_policy.py',
                      'spatial_outcome_policy.py', 'spatial_outcome_planner.py', 'world_model.py', 'world_runtime.py')]]:
            bindings.add(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        needed = sum(report['counts'].values()) * 144 * 22 * 4 + sum(report['counts'].values()) * 16 + 4096
        if shutil.disk_usage(output.parent).free < needed + 2**30:
            raise OSError('semantic addon needs output bytes plus1GiB margin')
        report['storage_bytes_required'] = needed
        staging.mkdir(); base.write_json(staging / 'progress.json', report)
        print(json.dumps(dict(pid=os.getpid(), started_local=report['started_local'], status='validating', staging=str(staging))), flush=True)
        import torch
        from pebby.agent.neural_outcome_policy import weights_sha256
        torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
        if args.device == 'cuda':
            base.require_no_foreign_cuda(); torch.cuda.reset_peak_memory_stats()
        perceptor = online.load_perceptor(args.teacher).to(args.device)
        before = weights_sha256(perceptor.state_dict())
        cpu_perceptor = copy.deepcopy(perceptor).cpu()
        cached_recent = {key: np.load(args.recent_cache / f'{key}.npy', mmap_mode='r', allow_pickle=False)
                         for key in recent_cache.SCHEMA}
        opened.append(cached_recent)
        for split, count in report['counts'].items():
            folder = staging / split; folder.mkdir()
            source_folder = args.recent_cache if split == 'recent' else args.base_cache / split
            report['aligned_source_files'][split] = {}
            for key in ('rows', 'seeds'):
                source = source_folder / f'{key}.npy'; digest = bindings.add(source)
                shutil.copyfile(source, folder / f'{key}.npy')
                report['aligned_source_files'][split][key] = dict(path=str(source.resolve()), sha256=digest)
            rows = np.load(folder / 'rows.npy', mmap_mode='r', allow_pickle=False)
            seeds = np.load(folder / 'seeds.npy', mmap_mode='r', allow_pickle=False)
            identity = dict(rows=rows, seeds=seeds); opened.append(identity)
            outputs[split] = np.lib.format.open_memmap(folder / 'semantic.npy', mode='w+', dtype=np.float32, shape=(count, 144, 22))
            parity_frames, parity_rows = [], []
            selected_hash = hashlib.sha256()
            if split != 'recent':
                source, info = original_source(args.frame_source.resolve(), split, original, bindings); opened.append(source)
                check_alignment(rows, seeds, source['seeds']); report['frame_sources'][split] = info
                def parts():
                    for start in range(0, count, args.batch_size):
                        chosen = np.arange(start, min(start + args.batch_size, count), dtype=np.int64)
                        yield start, current_frames(source['frames'], np.asarray(rows[chosen]))
            else:
                offset = 0
                for entry in recent['levels']:
                    if entry['offset'] != offset or entry['rows'] < 1:
                        raise ValueError('recent level offsets have gaps or overlaps')
                    bindings.add(child(args.recent_source.resolve(), entry['array_path']), entry['sha256']); offset += entry['rows']
                if offset != count or not np.array_equal(rows, np.arange(count)):
                    raise ValueError('recent cache must retain every raw row')
                report['frame_sources'][split] = dict(collection_report=str(collection_path), levels=recent['levels'])
                def parts():
                    for entry in recent['levels']:
                        base.guard(report, started, args.deadline_seconds)
                        frames = read_recent_level(child(args.recent_source.resolve(), entry['array_path']), entry, cached_recent)
                        for start in range(0, len(frames), args.batch_size):
                            yield entry['offset'] + start, frames[start:start + args.batch_size]
                        del frames
            expected = 0
            for start, frames in parts():
                base.guard(report, started, args.deadline_seconds)
                if start != expected:
                    raise ValueError('frame iterator changed output row order')
                values = encode_frames(perceptor, frames, args.device); outputs[split][start:start + len(frames)] = values
                selected_hash.update(np.asarray(rows[start:start + len(frames)]).tobytes()); selected_hash.update(frames.tobytes())
                if len(parity_rows) < 8 and (len(parity_rows) == 0 or start >= count * len(parity_rows) // 8):
                    parity_rows.append(start); parity_frames.append(frames[0].copy())
                expected += len(frames); report['progress'][split] = expected
                if expected % (args.batch_size * 64) == 0:
                    base.release(outputs, flush=True); base.write_json(staging / 'progress.json', report)
                    print(json.dumps(dict(split=split, rows=expected, elapsed_seconds=time.monotonic() - started)), flush=True)
            if expected != count:
                raise ValueError('semantic cache omitted rows')
            errors, bitwise, cpu_errors, cpu_bitwise = [], True, [], True
            for row, frame in zip(parity_rows, parity_frames):
                direct = encode_frames(perceptor, frame[None], args.device)[0]; saved = np.asarray(outputs[split][row])
                error = float(np.max(np.abs(direct - saved))); errors.append(error); bitwise &= np.array_equal(direct, saved)
                cpu = encode_frames(cpu_perceptor, frame[None], 'cpu')[0]
                cpu_errors.append(float(np.max(np.abs(cpu - direct)))); cpu_bitwise &= np.array_equal(cpu, direct)
                if error > PARITY_ATOL:
                    raise ValueError(f'single-frame online/cache probability parity failed: {error}')
            report['parity'][split] = dict(samples=len(errors), rows=parity_rows, maximum_absolute_error=max(errors),
                                          absolute_tolerance=PARITY_ATOL, bitwise_equal=bool(bitwise),
                                          cpu_device_maximum_absolute_error=max(cpu_errors),
                                          cpu_device_bitwise_equal=bool(cpu_bitwise),
                                          cpu_device_comparison_is_feature_only=True)
            report['frame_sources'][split]['selected_rows_and_current_frames_sha256'] = selected_hash.hexdigest()
            base.release(outputs, flush=True)
            if split != 'recent':
                base.release(source, close=True)
            base.write_json(staging / 'progress.json', report)
        if before != weights_sha256(perceptor.state_dict()) or any(p.requires_grad or p.grad is not None for p in perceptor.parameters()):
            raise ValueError('frozen perceptor changed or received gradients')
        base.release(outputs, close=True, flush=True); report['files'] = {}
        for split, count in report['counts'].items():
            report['files'][split] = {}
            for key, (dtype, tail) in SCHEMA.items():
                path = staging / split / f'{key}.npy'; base.guard(report, started, args.deadline_seconds)
                report['files'][split][key] = dict(path=f'{split}/{key}.npy', sha256=base.sha(path), shape=[count, *tail], dtype=np.dtype(dtype).str)
                path.chmod(0o444)
        bindings.verify()
        report.update(status='complete', source_bindings=bindings.hashes, source_stats=bindings.before, sources_unchanged=True,
            teacher_path=str(args.teacher.resolve()),
            runtime_source_bindings={str(p.resolve()): bindings.hashes[str(p.resolve())]
                                     for p in (*RUNTIME_SOURCES, args.teacher)},
            limits=['Single-frame versus batched parity is numerical, not guaranteed bitwise.',
                    'Teacher was trained on generated initial visible cells; dynamic/fog/overlap accuracy is not established by cache parity.',
                    'CPU/GPU downstream chosen-action equality is not established by this feature-only cache.'],
            weights_unchanged=True, teacher_weights_sha256=before, validation_disjoint=True,
            finished_local=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
        base.write_json(staging / 'manifest.json', report); validate_published(staging)
        (staging / 'progress.json').unlink(); (staging / 'manifest.json').chmod(0o444)
        base.publish_directory(staging, output)
        print(json.dumps(dict(status='complete', output=str(output), elapsed_seconds=time.monotonic() - started)), flush=True)
        return report
    except BaseException as error:
        if staging.exists():
            base.write_json(staging / 'progress.json', {**report, 'status': 'failed', 'error': repr(error)})
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm)
        base.release(outputs, close=True)
        for arrays in opened:
            base.release(arrays, close=True)
        del perceptor


if __name__ == '__main__':
    main()
