"""Bounded search-kernel timing with fixed generated fields, no gameplay claim."""
import json
import os
import signal
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from pebby.agent.event_calibration import PositiveSlopePlatt
from pebby.agent.structured_distance import StructuredDistanceReadout
from pebby.agent.structured_factored_policy import StructuredFactoredPolicy, canonical_metadata
from pebby.agent.structured_search import search
from tools.train_structured_policy import digest


def main():
    report_path = Path('artifacts/structured-search-kernel-timing.json')
    if report_path.exists():
        raise ValueError('refusing output overwrite')
    print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('180s deadline')))
    signal.alarm(180)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    policy = StructuredFactoredPolicy.from_checkpoints(
        'checkpoints/ls20-world-cell-recall-b1024.pt',
        'checkpoints/ls20-cell-visibility-initial-200.pt',
        'checkpoints/ls20-factored-local-h4-400.pt').requires_grad_(False)
    head_path = Path('checkpoints/ls20-structured-distance-ranking-200/ranking.pt')
    if digest(head_path) != '11739e8e03fe9730022eb8153508ae6df3f5bbffb569042fcceff87dd5cbd2f7':
        raise ValueError('distance checkpoint drift')
    saved = torch.load(head_path, map_location='cpu', weights_only=True)
    if canonical_metadata(saved['frozen_binding']) != canonical_metadata(policy.sources):
        raise ValueError('distance model binding mismatch')
    head = StructuredDistanceReadout(saved['config']).eval().requires_grad_(False)
    head.load_state_dict(saved['weights'], strict=True)
    cache = Path('data/structured-field-16384/validation')
    manifest = json.loads((cache / 'manifest.json').read_text())
    if digest(cache / 'fields.npy') != manifest['arrays']['fields']['sha256']:
        raise ValueError('field cache drift')
    field = torch.tensor(np.array(np.load(cache / 'fields.npy', mmap_mode='r')[:1]), dtype=torch.float32)
    calibrator = PositiveSlopePlatt().eval().requires_grad_(False)
    report = {'pid': os.getpid(), 'status': 'running', 'official_inputs_used': False,
              'scope': 'Search kernel only; no public encoder timing and no gameplay evidence',
              'calibration': 'identity unfitted, coherent adapter for timing only; never used for gameplay',
              'field_rows': [0], 'repeats': 3, 'warmups_per_device_depth': 1,
              'precision': 'FP32, TF32 disabled', 'devices': {}}
    for device in ('cpu', 'cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        dynamics = policy.dynamics.to(device)
        head.to(device)
        calibrator.to(device)
        current = field.to(device)

        def world(fields, actions):
            output = dynamics(fields, actions)
            events = torch.stack([output['events'][name + '_logits'] for name in ('lost_life', 'terminal', 'won')], -1)
            return output['field'], events

        def outcomes(events):
            result = calibrator.coherent_probabilities(events)
            return torch.stack([result[name] for name in ('loss', 'win', 'continue')], -1)

        results = {}
        for depth in (1, 4):
            elapsed = []
            traces = []
            for repeat in range(4):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = search(current, world, head, outcomes, depth=depth)
                if device == 'cuda':
                    torch.cuda.synchronize()
                duration = time.perf_counter() - start
                if repeat:
                    elapsed.append(duration)
                    traces.append(result['results'])
            results[str(depth)] = {'seconds': elapsed, 'median_seconds': statistics.median(elapsed),
                                   'transitions': result['transition_counts'],
                                   'deterministic_repeats': all(t == traces[0] for t in traces)}
        report['devices'][device] = results
    report.update(status='complete', elapsed_seconds=time.monotonic() - started)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
