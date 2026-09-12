"""Add small successor-action label sidecars without rewriting image archives.

Replay the deterministic generated collector, compare every existing row field
by SHA256, and retain only the new four-byte action masks. Original NPY members
are streamed in small row chunks; worker IPC carries hashes and labels, not
images. A complete contextual oracle and actual engine WIN are still required
for every level. No official level is loaded.
"""
from pebby.ls20.provenance import generated_context, validate_difficulty

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import zipfile

import numpy as np

from pebby.agent.world_data import collect_level

FORMAT = 'pebby.ls20-successor-labels.v1'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def row_tag(name, dtype, shape):
    return json.dumps([name, np.dtype(dtype).str, list(shape)], separators=(',', ':')).encode() + b'\0'


def source_fingerprints(path, chunk_rows=128):
    """At most one small decompressed image chunk plus row hash objects in RAM."""
    with np.load(path, allow_pickle=False) as archive:
        if len(set(archive.files)) != len(archive.files):
            raise ValueError('duplicate source archive members')
        meta = json.loads(str(archive['meta'].item()))
        if meta.get('source') != 'generated_only' or meta.get('oracle_search') != 'complete_only':
            raise ValueError('source must be complete-oracle generated data')
        if 'next_optimal' in archive.files:
            raise ValueError('source already contains successor labels')
        seeds = archive['seeds']
        names = [key for key in archive.files if key != 'meta']
    hashes = [hashlib.sha256() for _ in seeds]
    schema = []
    with zipfile.ZipFile(path) as archive:
        for name in names:
            with archive.open(name + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError(f'unsupported NPY header {version}')
                shape, fortran, dtype = reader(stream)
                if fortran or dtype.hasobject or not shape or shape[0] != len(seeds):
                    raise ValueError(f'invalid row array {name}: {shape} {dtype}')
                tail = shape[1:]
                row_bytes = int(np.prod(tail, dtype=np.int64)) * dtype.itemsize
                tag = row_tag(name, dtype, tail)
                schema.append((name, dtype.str, tail))
                for start in range(0, len(seeds), chunk_rows):
                    count = min(chunk_rows, len(seeds) - start)
                    block = stream.read(count * row_bytes)
                    if len(block) != count * row_bytes:
                        raise ValueError(f'truncated NPY member {name}')
                    view = memoryview(block)
                    for offset in range(count):
                        hashes[start + offset].update(tag)
                        hashes[start + offset].update(view[offset * row_bytes:(offset + 1) * row_bytes])
                if stream.read(1):
                    raise ValueError(f'trailing bytes in NPY member {name}')
    return seeds, np.array([h.digest() for h in hashes], dtype='V32'), schema, meta


def recollect(task):
    spec, schema, samples, epsilon, coverage, history = task
    rows, proof = collect_level(spec, history=history, samples=samples, epsilon=epsilon,
                                coverage=coverage, context_index=generated_context(spec))
    if proof.get('excluded') or not proof.get('context_engine_verified'):
        raise ValueError(f"seed {spec['seed']} failed replay: {proof}")
    fingerprints = []
    for row in rows:
        fingerprint = hashlib.sha256()
        for name, dtype, shape in schema:
            value = np.asarray(row[name])
            if value.dtype != np.dtype(dtype) or value.shape != tuple(shape):
                raise ValueError(f"seed {spec['seed']} changed {name} schema")
            fingerprint.update(row_tag(name, dtype, shape))
            fingerprint.update(value.tobytes(order='C'))
        fingerprints.append(fingerprint.digest())
    return int(spec['seed']), np.array(fingerprints, dtype='V32'), np.stack([r['next_optimal'] for r in rows])


def annotate(source, bank, out, *, workers=4, samples=16, epsilon=.15, coverage='mixed', limit_levels=None):
    started = time.monotonic()
    source, bank, out = Path(source), Path(bank), Path(out)
    if out.exists():
        raise ValueError(f'refusing to overwrite {out}')
    source_hash = digest(source)
    seeds, fingerprints, schema, meta = source_fingerprints(source)
    print(f'PID {os.getpid()} | fingerprinted {len(seeds)} source rows in {time.monotonic()-started:.1f}s', flush=True)
    specs = {int(s['seed']): s for s in (json.loads(line) for line in bank.read_text().splitlines() if line)}
    order = np.argsort(seeds, kind='stable')
    levels, starts, counts = np.unique(seeds[order], return_index=True, return_counts=True)
    indices = {int(seed): order[start:start+count] for seed, start, count in zip(levels, starts, counts)}
    if limit_levels:
        chosen = np.linspace(0, len(levels)-1, min(limit_levels, len(levels)), dtype=int)
        levels = levels[chosen]
    history = next(shape[0] for name, _, shape in schema if name == 'frames')
    tasks = [(specs[int(seed)], schema, samples, epsilon, coverage, history) for seed in levels]
    masks = np.zeros((len(seeds), 4), dtype=np.uint8)
    replayed = 0
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for completed, (seed, actual, labels) in enumerate(pool.imap_unordered(recollect, tasks), 1):
            index = indices[seed]
            if len(actual) != len(index) or not np.array_equal(actual, fingerprints[index]):
                raise ValueError(f'seed {seed}: deterministic replay differs from original rows; no labels published')
            masks[index] = labels
            replayed += len(index)
            if completed % 100 == 0 or completed == len(tasks):
                elapsed = time.monotonic() - started
                print(f'{completed}/{len(tasks)} levels | {replayed} exact replay rows | {elapsed:.1f}s', flush=True)
    if digest(source) != source_hash:
        raise ValueError('source file changed during annotation')
    result = {'format': FORMAT, 'source': 'generated_only', 'source_path': str(source),
              'source_sha256': source_hash, 'bank': str(bank), 'bank_sha256': digest(bank),
              'rows': len(seeds), 'levels': len(indices), 'exact_replay_rows': replayed,
              'valid_action_targets': int(np.count_nonzero(masks)),
              'mask_semantics': 'bits0..3: complete-oracle optimal set;0: terminal or unreachable',
              'collection': {'samples': samples, 'epsilon': epsilon, 'coverage': coverage, 'history': history},
              'code_sha256': {p: digest(p) for p in [__file__, 'pebby/agent/world_data.py', 'pebby/ls20/plan.py']},
              'workers': workers, 'elapsed_seconds': time.monotonic()-started,
              'created': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'status': 'pilot' if limit_levels else 'complete'}
    out.parent.mkdir(parents=True, exist_ok=True)
    if not limit_levels:
        if replayed != len(seeds):
            raise ValueError('incomplete label coverage')
        with np.load(source, allow_pickle=False) as archive:
            if np.any(masks[archive['terminal']] != 0):
                raise ValueError('terminal successor received an action target')
        temporary = out.with_name(out.name + '.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, next_optimal=masks, seeds=seeds, meta=np.array(json.dumps(result)))
            os.replace(temporary, out)
        finally:
            temporary.unlink(missing_ok=True)
    report = out.with_suffix('.json')
    report.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--samples-per-level', type=int, default=16)
    parser.add_argument('--epsilon', type=float, default=.15)
    parser.add_argument('--coverage', choices=('prefix', 'mixed'), default='mixed')
    parser.add_argument('--limit-levels', type=int, help='Pilot only: writes report, never a partial label sidecar')
    args = parser.parse_args()
    if args.workers < 1 or args.samples_per_level < 1 or not 0 <= args.epsilon <= 1:
        parser.error('positive counts and epsilon in [0,1] required')
    print(json.dumps(annotate(args.source, args.bank, args.out, workers=args.workers,
                             samples=args.samples_per_level, epsilon=args.epsilon,
                             coverage=args.coverage, limit_levels=args.limit_levels), indent=2))


if __name__ == '__main__':
    main()
