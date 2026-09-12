"""Compare full-update performance probes; never evaluates learned gameplay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(left, right):
    a, b = [json.loads((p / 'report.json').read_text()) for p in (left, right)]
    for report in (a, b):
        if (report['status'] != 'complete' or not report['source_files_unchanged']
                or not report['data_unchanged']):
            raise ValueError('complete unchanged benchmark inputs required')
    for key in ('weights', 'model_config', 'data_sha256_before', 'batch'):
        if a[key] != b[key]:
            raise ValueError(f'comparison input mismatch: {key}')
    for key in ('seed', 'precision', 'tf32', 'warmup_updates', 'timed_updates'):
        if a['settings'][key] != b['settings'][key]:
            raise ValueError(f'comparison runtime mismatch: {key}')
    ca, cb = a['candidate'], b['candidate']
    if ca['batch_size'] != cb['batch_size']:
        raise ValueError('compare speed at the same actual batch size')
    metadata = [json.loads((p / 'parity.json').read_text()) for p in (left, right)]
    for path, meta in zip((left, right), metadata):
        with (path / 'parity.npz').open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != meta['sha256']:
                raise ValueError('parity array hash changed')
    for key in ('names', 'shapes', 'step_count', 'gradient_present'):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f'parity vector mismatch: {key}')
    arrays = [np.load(p / 'parity.npz', allow_pickle=False) for p in (left, right)]
    vectors = {}
    try:
        for key in ('parameters', 'gradients'):
            x, y = [v[key].astype(np.float64) for v in arrays]
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('invalid parity vector')
            delta = y - x
            vectors[key] = dict(exact=bool(np.array_equal(x, y)),
                max_absolute=float(np.abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(x), 1e-30)),
                cosine=float(np.dot(x, y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)))
    finally:
        for archive in arrays:
            archive.close()
    losses = []
    for x, y in zip(ca['updates'], cb['updates'], strict=True):
        values = {}
        for name in ('total', *a['weights']):
            if name not in x['stats']:
                continue
            base, candidate = x['stats'][name], y['stats'][name]
            values[name] = dict(baseline=base, candidate=candidate,
                absolute=abs(candidate-base), relative=abs(candidate-base)/max(abs(base), 1e-8))
        losses.append(dict(phase=x['phase'], index=x['index'], values=values))
    return dict(status='complete', baseline=str(left.resolve()), candidate=str(right.resolve()),
        batch_size=ca['batch_size'], vectors=vectors, losses=losses,
        gradient_presence=[m.get('gradient_present') for m in metadata],
        baseline_seconds=ca['timed_wall_seconds'], candidate_seconds=cb['timed_wall_seconds'],
        median_speedup=ca['median_timed_wall_seconds']/cb['median_timed_wall_seconds'],
        baseline_peak=ca['peak_memory_bytes'], candidate_peak=cb['peak_memory_bytes'],
        limits=['Fixed generated-shard performance probe; not full-bank distinct-level capacity.',
                'Numerical differences require review; no automatic promotion or gameplay claim.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.baseline, args.candidate)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('batch_size', 'vectors', 'median_speedup')}))
