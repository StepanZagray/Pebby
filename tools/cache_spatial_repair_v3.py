"""Publish frozen public features and per-objective eligibility for v3 recovery.

All verified raw roots remain stored. A separate TRAIN-only public-H8 audit
excludes conflicting histories and redundant copies from sampling, regardless
of optimal-mask value. Conflict exclusions are possible hidden-state aliasing.
"""
import argparse
from collections import defaultdict
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
from tools import cache_spatial_recovery as legacy
from tools.collect_spatial_repair_v3 import CHECKPOINT, CHECKPOINT_SHA, FORMAT, cohorts, verify_train_membership

EXTRA = {'chosen_action': ('int8', ()), 'policy_valid': ('bool', ()), 'dynamics_valid': ('bool', ())}
SCHEMA = {**legacy.SCHEMA, **EXTRA}
INPUT_SCHEMA = {**legacy.INPUT_SCHEMA, **EXTRA}
LABEL_KEYS = tuple(k for k in base.LABELS if k != 'seeds')


def validate_rows(arrays, seed):
    count = legacy.validate_rows(arrays, seed)
    for key, (dtype, tail) in EXTRA.items():
        if arrays[key].shape != (count, *tail) or arrays[key].dtype != np.dtype(dtype):
            raise ValueError(f'v3 eligibility/action schema mismatch: {key}')
    if not np.array_equal(arrays['policy_valid'], arrays['optimal'] != 0) or not arrays['dynamics_valid'].all():
        raise ValueError('policy validity must mask only undefined actions; every verified row has dynamics')
    if np.any((arrays['chosen_action'] < 0) | (arrays['chosen_action'] > 3)):
        raise ValueError('actual chosen action must be0..3')
    if np.any(arrays['won'] & ~arrays['terminal']) or np.any(arrays['terminal'] & (arrays['next_optimal'] != 0)):
        raise ValueError('terminal target consistency failed')
    loss = arrays['lost_life']
    if not np.array_equal(loss, arrays['next_lives'] < arrays['current_lives'][:, None]):
        raise ValueError('life-loss targets disagree with real successor lives')
    return count


def payload(encoder, arrays, chosen, offset, device):
    result = legacy.payload(encoder, arrays, chosen, offset, device)
    result.update({key: np.array(arrays[key][chosen], copy=True) for key in EXTRA})
    return result


def inspect_collection(directory, checkpoint, base_cache, bindings, *, expected_sha=CHECKPOINT_SHA):
    bindings.add(checkpoint, expected_sha)
    bindings.add(directory / 'report.json')
    report = json.loads((directory / 'report.json').read_text())
    if (report.get('format') != FORMAT or report.get('status') != 'complete'
            or report.get('source_checkpoint_sha256') != expected_sha
            or report.get('official_inputs_used') is not False or not report.get('sources_unchanged')
            or not report.get('workers_reaped') or not report.get('all_engine_verified_rows_retained')
            or report.get('public_inputs') != list(base.PUBLIC)):
        raise ValueError('complete source-bound v3 TRAIN collection required')
    for path, sha in report['source_bindings'].items():
        bindings.add(path, sha)
    bindings.add(directory / 'train.jsonl')
    selected = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines() if line.strip()]
    verify_train_membership(selected, base_cache, bindings)
    by_seed = {int(s['seed']): s for s in selected}
    if report['count'] != len(selected):
        raise ValueError('selected collection count differs')
    levels, seen, total = [], set(), 0
    for entry in report['levels']:
        seed = int(entry['seed'])
        if seed in seen or seed not in by_seed:
            raise ValueError('unselected or duplicate collection level')
        seen.add(seed)
        path = legacy.child(directory, entry['array_path'])
        proof_path = legacy.child(directory, entry['proof_path'])
        bindings.add(path, entry['sha256'])
        bindings.add(proof_path, entry['proof_sha256'])
        proof = json.loads(proof_path.read_text())
        from tools.collect_spatial_recovery import spec_sha
        if (proof.get('status') != 'complete' or proof.get('seed') != seed
                or proof.get('source_checkpoint_sha256') != expected_sha
                or proof.get('array_sha256') != entry['sha256']
                or proof.get('spec_sha256') != spec_sha(by_seed[seed])
                or not proof.get('context_engine_verified') or proof.get('search_truncated')
                or proof.get('oracle_backend') != 'fast' or not proof.get('all_engine_verified_rows_retained')):
            raise ValueError('incomplete or mismatched real-engine source proof')
        for source, sha in proof['source_bindings'].items():
            bindings.add(source, sha)
        with np.load(path, allow_pickle=False) as arrays:
            count = validate_rows(arrays, seed)
            if cohorts(arrays) != proof['cohorts']:
                raise ValueError('source cohort counts changed')
        if count != proof['rows'] or proof['branch_checks']['branches'] != 4 * count:
            raise ValueError('four checked branches per retained root required')
        levels.append({**entry, 'rows': count, 'offset': total})
        total += count
    if seen != set(by_seed) or total != report['rows']:
        raise ValueError('collection omitted selected TRAIN levels or roots')
    return levels, total


def fingerprint(arrays, row, keys):
    result = hashlib.sha256()
    for key in keys:
        dtype = INPUT_SCHEMA[key][0]
        value = np.asarray(arrays[key][row], dtype=dtype)
        result.update(key.encode() + b'\0')
        result.update(str((tuple(value.shape), value.dtype.str)).encode() + b'\0')
        result.update(value.tobytes(order='C'))
    return result.digest()


def approved_rows(public, labels, base_count, priorities):
    """Keep one matching-target representative; exclude all conflicting members."""
    groups = defaultdict(list)
    for index, key in enumerate(public):
        groups[bytes(key)].append(index)
    retained, conflicts, duplicate_count = [], [], 0
    for indices in groups.values():
        if len({bytes(labels[i]) for i in indices}) > 1:
            conflicts.extend(indices)
            continue
        retained.append(min(indices, key=lambda i: (i >= base_count, priorities[i], i)))
        duplicate_count += len(indices) - 1
    return (np.array(sorted(i for i in retained if i < base_count), dtype=np.int64),
            np.array(sorted(i - base_count for i in retained if i >= base_count), dtype=np.int64),
            np.array(sorted(conflicts), dtype=np.int64), duplicate_count)


def fingerprint_level(path):
    """Decode each compressed column once; release H8 after this one level."""
    keys = (*base.PUBLIC, *LABEL_KEYS, 'row_kind', 'seeds')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in keys}
    result = {key: [] for key in ('public', 'labels', 'kinds', 'zero', 'seeds', 'loss_counts', 'refill_counts')}
    for row in range(len(arrays['seeds'])):
        result['public'].append(fingerprint(arrays, row, base.PUBLIC))
        result['labels'].append(fingerprint(arrays, row, LABEL_KEYS))
        result['kinds'].append(int(arrays['row_kind'][row]))
        result['zero'].append(int(arrays['optimal'][row]) == 0)
        result['seeds'].append(int(arrays['seeds'][row]))
        result['loss_counts'].append(int(arrays['lost_life'][row].sum()))
        live = ~arrays['lost_life'][row] & ~arrays['terminal'][row]
        result['refill_counts'].append(int((live & (arrays['next_steps'][row] > arrays['current_steps'][row])).sum()))
    # Returned objects contain hashes/scalars only, so no archive arrays survive.
    return result


def audit_public_histories(directory, levels, staging, bindings):
    audit_path = base.ROOT / 'artifacts/spatial-recovery-v1/base-quality-audit.json'
    bindings.add(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'complete' or audit.get('original_rows') != 80000 or audit.get('domain_or_consistency_invalid_rows') != 0:
        raise ValueError('verified original all-root public/target hashes required')
    if audit['public_hash_fields'] != list(base.PUBLIC) or audit['label_hash_fields'] != list(LABEL_KEYS):
        raise ValueError('original public/target fingerprint field order differs')
    manifest = base.ROOT / 'data/reference-outcome-inputs-v1/manifest.json'
    bindings.add(manifest, audit['input_manifest_sha256'])
    public, labels = [], []
    for key, output in [('public', 'base-public-sha256.npy'), ('labels', 'base-label-sha256.npy')]:
        info = audit['outputs'][output]
        bindings.add(info['path'], info['sha256'])
        value = np.load(info['path'], allow_pickle=False)
        if value.dtype != np.uint8 or value.shape != (80000, 32):
            raise ValueError('original all-root fingerprint schema differs')
        (public if key == 'public' else labels).extend(bytes(row) for row in value)
    priorities, kinds, zero, seeds, loss_counts, refill_counts = [0] * 80000, [], [], [], [], []
    for entry in levels:
        part = fingerprint_level(legacy.child(directory, entry['array_path']))
        if len(part['public']) != entry['rows']:
            raise ValueError('per-level fingerprint row count differs')
        public.extend(part['public'])
        labels.extend(part['labels'])
        priorities.extend(part['kinds'])
        kinds.extend(part['kinds'])
        zero.extend(part['zero'])
        seeds.extend(part['seeds'])
        loss_counts.extend(part['loss_counts'])
        refill_counts.extend(part['refill_counts'])
    base_rows, supplement_rows, conflicting, duplicate_count = approved_rows(public, labels, 80000, priorities)
    if not len(base_rows) or not len(supplement_rows):
        raise ValueError('public-history audit left an empty replay source')
    kinds, zero, seeds = map(np.asarray, (kinds, zero, seeds))
    files = {}
    for name, values in [('base_rows', base_rows), ('supplement_rows', supplement_rows), ('conflicting_rows', conflicting)]:
        path = staging / f'{name}.npy'
        np.save(path, values)
        files[name] = dict(path=path.name, sha256=base.sha(path), shape=list(values.shape), dtype=values.dtype.str)
    return dict(status='complete', files=files, original_base_rows=80000, original_supplement_rows=len(kinds),
                approved_base_rows=len(base_rows), approved_supplement_rows=len(supplement_rows),
                duplicate_rows_removed=duplicate_count, conflicting_rows=len(conflicting),
                conflicting_base_rows=int((conflicting < 80000).sum()),
                conflicting_supplement_rows=int((conflicting >= 80000).sum()),
                retained_zero_policy_supplement_rows=int(zero[supplement_rows].sum()),
                original_zero_policy_supplement_rows=int(zero.sum()),
                original_supplement_life_loss_branches=sum(loss_counts),
                retained_supplement_life_loss_branches=int(np.asarray(loss_counts)[supplement_rows].sum()),
                original_supplement_refill_branches=sum(refill_counts),
                retained_supplement_refill_branches=int(np.asarray(refill_counts)[supplement_rows].sum()),
                retained_supplement_kinds={str(k): int((kinds[supplement_rows] == k).sum()) for k in range(3)},
                retained_supplement_levels=int(len(np.unique(seeds[supplement_rows]))),
                policy_eligibility='optimal != 0 after public-history allowlist; never fabricate actions',
                selection='Same full H8/validity/actions and matching labels: prefer original replay, then policy/recovery/exhaustion. Exclude every member of conflicting target groups.',
                limits=['Conflicts can be genuine hidden-state aliasing, not invalid engine labels; all raw and feature rows remain stored.',
                        'Original all80k raw-H8 hashes reused from bound audit; full original frames are not rehashed here.',
                        'Teacher proofs/array bindings checked; this audit does not independently rerun all engine branches.'])


def validate_published(directory):
    directory = Path(directory)
    bindings = base.Bindings()
    bindings.add(directory / 'manifest.json')
    report = json.loads((directory / 'manifest.json').read_text())
    from pebby.agent.neural_outcome_policy import ENCODER_RUNTIME
    if (report.get('status') != 'complete' or report.get('format') != 'pebby.spatial-repair-features.v3'
            or report.get('source_checkpoint_sha256') != CHECKPOINT_SHA
            or report.get('split') != 'train' or report.get('official_inputs_used') is not False
            or not report.get('sources_unchanged') or not report.get('validation_disjoint')
            or report.get('encoder_runtime') != ENCODER_RUNTIME or not report.get('parity')
            or report.get('encoder_parent_sha256') != base.PARENT_SHA or not report.get('source_bindings')
            or len(report.get('encoder_weights_sha256', '')) != 64):
        raise ValueError('incomplete or invalid v3 feature publication')
    count = report['rows']
    if set(report['files']) != {key + '.npy' for key in SCHEMA}:
        raise ValueError('v3 feature files differ from complete schema')
    for key, (dtype, tail) in SCHEMA.items():
        info = report['files'][key + '.npy']
        if info['shape'] != [count, *tail] or info['dtype'] != np.dtype(dtype).str:
            raise ValueError('v3 feature schema mismatch')
        arrays = {key: base.open_array(directory / (key + '.npy'), info, bindings)}
        try:
            value = arrays[key]
            for start in range(0, count, 1024):
                chunk = value[start:start + 1024]
                if not np.isfinite(chunk).all():
                    raise ValueError('nonfinite v3 feature or target')
                if key == 'rows' and not np.array_equal(chunk, np.arange(start, min(start + 1024, count))):
                    raise ValueError('all original collection rows must remain in source order')
        finally:
            base.release(arrays, close=True)
    quality = report.get('quality', {})
    if quality.get('status') != 'complete' or set(quality.get('files', {})) != {'base_rows', 'supplement_rows', 'conflicting_rows'}:
        raise ValueError('completed public-history quality audit required')
    for key, info in quality['files'].items():
        path = legacy.child(directory, info['path'])
        bindings.add(path, info['sha256'])
        values = np.load(path, allow_pickle=False)
        if (values.dtype != np.int64 or values.ndim != 1 or len(np.unique(values)) != len(values)
                or list(values.shape) != info['shape'] or values.dtype.str != info['dtype']):
            raise ValueError('quality row allowlist schema differs')
        limit = 80000 if key == 'base_rows' else count if key == 'supplement_rows' else 80000 + count
        if np.any(values < 0) or np.any(values >= limit):
            raise ValueError('quality row allowlist out of range')
    verify_train_membership(report['levels'], Path(report['base_cache']), bindings)
    actual_seeds = np.load(directory / 'seeds.npy', allow_pickle=False)
    if set(map(int, actual_seeds)) != {int(level['seed']) for level in report['levels']}:
        raise ValueError('published rows differ from verified TRAIN level set')
    optimal = np.load(directory / 'optimal.npy', allow_pickle=False)
    valid = np.load(directory / 'policy_valid.npy', allow_pickle=False)
    dynamics = np.load(directory / 'dynamics_valid.npy', allow_pickle=False)
    if not np.array_equal(valid, optimal != 0) or not dynamics.all():
        raise ValueError('published objective eligibility differs from verified labels')
    report['validated_output_stats'] = bindings.verify()
    report['validated_output_hashes'] = bindings.hashes
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--base-cache', type=Path, default=base.ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--deadline-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 64 or args.deadline_seconds <= 0:
        parser.error('batch size1..64 and positive deadline required')
    output, directory = args.out_dir.resolve(), args.source_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    staging = output.with_name(f'.{output.name}.staging-{os.getpid()}')
    bindings, outputs, policy = base.Bindings(), {}, None
    report = dict(format='pebby.spatial-repair-features.v3', status='validating', pid=os.getpid(),
                  start_ticks=base.process_start_ticks(), started_local=datetime.now().astimezone().isoformat(),
                  split='train', official_inputs_used=False, source_checkpoint_sha256=CHECKPOINT_SHA,
                  base_cache=str(args.base_cache.resolve()),
                  public_inputs=list(base.PUBLIC), progress_rows=0, extraction_batch_size=args.batch_size)
    def deadline(*_):
        raise TimeoutError('bounded v3 feature publication expired')
    old_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        base.guard(report, started, args.deadline_seconds)
        levels, count = inspect_collection(directory, args.checkpoint.resolve(), args.base_cache.resolve(), bindings)
        for path in [Path(__file__), Path(legacy.__file__), Path(base.__file__),
                     base.ROOT / 'tools/collect_spatial_repair_v3.py',
                     *sorted((base.ROOT / 'pebby/agent').glob('*.py'))]:
            bindings.add(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        required = sum(count * np.dtype(dtype).itemsize * int(np.prod(tail)) + 128 for dtype, tail in SCHEMA.values())
        if shutil.disk_usage(output.parent).free < required + 2**30:
            raise OSError('insufficient cache disk space plus1GiB margin')
        staging.mkdir()
        report['quality'] = audit_public_histories(directory, levels, staging, bindings)
        base.write_json(staging / 'progress.json', report)
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
                      rows=count, levels=levels, validation_disjoint=True, parity=[],
                      all_engine_verified_rows_stored=True, training_requires_quality_allowlists=True)
        outputs = {key: np.lib.format.open_memmap(staging / f'{key}.npy', mode='w+', dtype=dtype, shape=(count, *tail))
                   for key, (dtype, tail) in SCHEMA.items()}
        with torch.inference_mode(), encoder_execution(torch.device('cuda')):
            for entry in levels:
                base.guard(report, started, args.deadline_seconds)
                base.require_no_foreign_cuda()
                with np.load(legacy.child(directory, entry['array_path']), allow_pickle=False) as archive:
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
                    report['parity'].append(dict(seed=entry['seed'], errors=legacy.feature_parity(
                        encoder, arrays, chosen, outputs, entry['offset'], 'cuda', batch_size=args.batch_size)))
                del arrays
                base.release(outputs, flush=True)
                base.write_json(staging / 'progress.json', report)
        if weights_sha256(encoder.state_dict()) != report['encoder_weights_sha256'] or any(
                p.requires_grad or p.grad is not None for p in encoder.parameters()):
            raise ValueError('frozen encoder changed or received gradients')
        base.release(outputs, close=True, flush=True)
        report['files'] = {}
        for key, (dtype, tail) in SCHEMA.items():
            base.guard(report, started, args.deadline_seconds)
            path = staging / f'{key}.npy'
            report['files'][path.name] = dict(sha256=base.sha(path), shape=[count, *tail], dtype=np.dtype(dtype).str)
            path.chmod(0o444)
        legacy.rehash(bindings)
        report.update(status='complete', sources_unchanged=True, source_bindings=bindings.hashes,
                      finished_local=datetime.now().astimezone().isoformat())
        base.write_json(staging / 'manifest.json', report)
        validate_published(staging)
        (staging / 'progress.json').unlink()
        for info in report['quality']['files'].values():
            (staging / info['path']).chmod(0o444)
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
