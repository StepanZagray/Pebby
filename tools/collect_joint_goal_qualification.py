"""Bounded eight-level generated TRAIN wiring/learnability data, CPU only."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.joint_goal_data import COLLECTIONS, FORMAT, collect_level, qualification_specs, schema, source_paths, validate_arrays
from pebby.ls20 import fastplan


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=180)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 180:
        parser.error('max-seconds must be 1..180')
    if torch.cuda.is_initialized():
        raise RuntimeError('fresh CPU-only process required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('joint qualification collection wall cap exhausted')
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < 7 * 2**30:
            raise MemoryError('seven GiB available-memory reserve breached')

    def timeout(*_):
        raise TimeoutError('joint qualification hard wall-clock alarm')

    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    manifest = dict(format=FORMAT, status='running', pid=os.getpid(), source='generated_only', split='train',
                    requested_levels=8, qualification_only=True, full_seven_tier_coverage=False,
                    intended_use='wiring and learnability qualification only; not generalization or model promotion',
                    device='cpu', cpu_threads=1, memory_reserve_gib=7, wall_cap_seconds=args.max_seconds,
                    started_local=datetime.now().astimezone().isoformat(), schema=schema(), levels=[],
                    official_inputs_used=False, pretrained_models_used=False, cached_features_used=False,
                    label_generation='complete bounded oracle plus actual generated engine transition verification',
                    teacher_routes_training_only=True, horizon=1, branches=4,
                    limits=['Eight compact explicit generated layouts in contexts 0/1; broad seven-tier training remains outstanding.',
                            'Teacher/recovery/exhaustion behavior is a data collector, not a deployable neural policy.',
                            'Sample level identities uniformly first, with replacement when batch size exceeds eight; roots within each level are correlated.',
                            'Valid distance -1 means proven unreachable; invalid distance is unknown/unsupported. Zero optimal mask is not a policy target.',
                            'Post-life-loss finite distance is the reset successor distance; it must not reward death in action ranking.',
                            'Only frames/history_valid/previous_actions enter model forward; every other array is training-only.'])
    temporary = args.out / '.data.npz.tmp'

    def persist():
        manifest['elapsed_seconds'] = time.monotonic() - started
        write_json(args.out / 'manifest.json', manifest)

    print('PID', os.getpid(), flush=True)
    try:
        guard()
        manifest['sources'] = {str(path.resolve()): digest(path) for path in source_paths()}
        specs = qualification_specs()
        write_json(args.out / 'bank.json', specs)
        manifest['bank_sha256'] = digest(args.out / 'bank.json')
        persist()
        records = []
        for spec in specs:
            rows, proof = collect_level(spec, guard)
            records.extend(rows)
            manifest['levels'].append(proof)
            persist()
            print(json.dumps(dict(case=spec['case'], roots=len(rows), verified_route_length=proof['context_optimal_actions'])), flush=True)
        guard()
        arrays = {key: np.stack([row[key] for row in records]) for key in records[0]}
        validate_arrays(arrays, specs)
        manifest['arrays'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}
        manifest['coverage'] = dict(roots=len(records), levels=8,
                                    roots_by_collection=dict(Counter(str(int(x)) for x in arrays['collection'])),
                                    events={key: int(arrays[key].sum()) for key in ('lost_life', 'terminal', 'won')},
                                    roots_with_policy_target=int(arrays['optimal_valid'].sum()),
                                    known_unreachable_root_distances=int((arrays['current_distance_valid'] & (arrays['current_distance'] == -1)).sum()),
                                    known_unreachable_roots_by_collection={name: int(((arrays['collection'] == index) & arrays['current_distance_valid'] & (arrays['current_distance'] == -1)).sum()) for name, index in COLLECTIONS.items()},
                                    known_unreachable_successor_distances=int((arrays['distance_valid'] & (arrays['distances'] == -1)).sum()),
                                    unsupported_successor_distances=int((~arrays['distance_valid']).sum()),
                                    supported_goal_attribute_targets=int(arrays['goal_attribute_valid'].sum()),
                                    solved_goal_labels=int(arrays['goal_solved'].sum()))
        library = fastplan.library_path()
        if any(level['oracle_backend'] == 'fast' for level in manifest['levels']):
            manifest['sources'][str(library.resolve())] = digest(library)
        for path, expected in manifest['sources'].items():
            guard()
            if digest(path) != expected:
                raise ValueError('qualification source changed: ' + path)
        with temporary.open('xb') as stream:
            np.savez(stream, **arrays)
        temporary.rename(args.out / 'data.npz')
        manifest.update(status='complete', data_sha256=digest(args.out / 'data.npz'), sources_unchanged=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        temporary.unlink(missing_ok=True)
        manifest['finished_local'] = datetime.now().astimezone().isoformat()
        persist()
    return manifest


if __name__ == '__main__':
    main()
