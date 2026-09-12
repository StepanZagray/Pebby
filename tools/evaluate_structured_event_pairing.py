"""TRAIN-only paired event-head diagnosis using actual and imagined successors."""
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from pebby.agent.model import load_checkpoint
from pebby.agent.structured_factored_policy import canonical_metadata
from tools.cache_structured_policy_successors import load_imagined_cache
from tools.calibrate_structured_events import _coherent_metrics, _metrics
from tools.train_structured_policy import load_policy_cache, digest


def json_finite(value):
    """Empty conditional subsets have undefined metrics, represented as null."""
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    out = Path('artifacts/structured-events-paired-train.json')
    arrays_out = Path('data/structured-events-paired-train.npz')
    if out.exists() or arrays_out.exists():
        raise ValueError('refusing output overwrite')
    print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('180s diagnosis limit')))
    signal.alarm(180)
    torch.set_num_threads(1)
    start = time.monotonic()
    checkpoint = Path('checkpoints/ls20-structured-search-ranking-d1.pt')
    policy, _ = load_checkpoint(checkpoint, 'cpu')
    cache = Path('data/structured-field-16384/train')
    data, manifest = load_policy_cache(cache, 'train')
    factory_encoder = policy.sources['factored']['dynamics_metadata']['field_encoder']
    # The factory strictly checked the full encoder metadata and checkpoint hashes.
    if canonical_metadata(manifest['field_encoder']) != canonical_metadata(factory_encoder):
        raise ValueError('encoder metadata mismatch')
    from types import SimpleNamespace
    proxy = SimpleNamespace(sources=policy.sources['factored'])
    imagined, fingerprints = load_imagined_cache('data/structured-policy-imagined-local-h4-400',
                                               'train', cache, data, manifest, proxy)
    n = len(data['seeds'])
    labels = np.stack([data[name] for name in ('lost_life', 'terminal', 'won')], -1).astype(np.int8)
    arrays = {'labels': labels, 'seeds': np.asarray(data['seeds']),
              'source_rows': np.asarray(data['source_rows'])}
    for kind in ('actual', 'imagined'):
        arrays[kind + '_logits'] = np.empty((n, 4, 3), np.float32)
    with torch.inference_mode():
        for first in range(0, n, 256):
            stop = min(first + 256, n)
            current = torch.tensor(np.asarray(data['fields'][first:stop]), dtype=torch.float32)
            summary = policy.dynamics.readout.summary(current)[:, None].expand(-1, 4, -1).reshape(-1, 96)
            actions = torch.arange(4).repeat(stop - first)
            for kind, source in [('actual', data['next_fields']), ('imagined', imagined)]:
                following = torch.tensor(np.asarray(source[first:stop]), dtype=torch.float32).reshape(-1, 148, 96)
                features = torch.cat([summary, policy.dynamics.readout.summary(following),
                                      policy.dynamics.action_embedding(actions)], -1)
                logits = policy.dynamics.event_head(features).reshape(stop - first, 4, 3)
                arrays[kind + '_logits'][first:stop] = logits.numpy()
    # This is a logical no-change proxy, not proof of identical rendered frames.
    same = (~labels[..., 0].astype(bool) & ~labels[..., 1].astype(bool))
    for name in ('player_cell', 'triple', 'steps', 'lives'):
        equal = data['next_' + name] == data[name][:, None]
        if equal.ndim == 3:
            equal = equal.all(-1)
        same &= equal
    arrays['logical_no_change_proxy'] = same
    report = {'status': 'complete', 'pid': os.getpid(), 'scope': 'TRAIN diagnostic only; no optimization or controller evaluation',
              'official_inputs_used': False, 'validation_inputs_used': False,
              'levels': n, 'branches': n * 4, 'logical_no_change_proxy_branches': int(same.sum()),
              'checkpoint_sha256': digest(checkpoint), 'source_manifest_sha256': digest(cache / 'manifest.json'),
              'imagined_cache_fingerprints': fingerprints, 'metrics': {},
              'limitations': ['Actual successor fields are privileged diagnostic inputs and never policy inputs.',
                             'Cached H1 TRAIN states are not search-selected histories.',
                             'No-change proxy uses player/glyph/budget/lives and live events, not rendered-frame identity.',
                             'Comparing views diagnoses sensitivity but cannot alone prove a unique causal bottleneck.']}
    subsets = {'all': np.ones((n, 4), bool), 'logical_no_change_proxy': same,
               'win': labels[..., 2].astype(bool), 'life_loss': labels[..., 0].astype(bool)}
    for kind in ('actual', 'imagined'):
        z = arrays[kind + '_logits']
        report['metrics'][kind] = {}
        for name, selected in subsets.items():
            report['metrics'][kind][name] = {
                'raw': _metrics(z[selected], labels[selected]),
                'calibrated': _coherent_metrics(z[selected], labels[selected], policy.calibrator)}
    np.savez(arrays_out, **arrays)
    report['arrays'] = {'path': str(arrays_out), 'sha256': digest(arrays_out)}
    report['elapsed_seconds'] = time.monotonic() - start
    out.write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
