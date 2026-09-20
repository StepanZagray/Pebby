"""CPU-only, resumable per-level raw semantic K4 collection on generated banks.

A supervising process enforces memory and wall limits even inside native
solver calls. Each completed shard is immutable; interruption never publishes
partial arrays. Pilot selection intentionally prefers small verified graphs,
and is not a representative generalization evaluation.
"""
import argparse
from datetime import datetime
import gc
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from pebby.agent.joint_goal_replay import (FORMAT, check_spec, collect_level, digest,
                                         schema, spec_digest, write_json, JointGoalReplay)
from pebby.agent.joint_goal_planning import load_checkpoint
from pebby.agent.joint_goal_data import source_paths as qualification_sources
from pebby.ls20 import fastplan


def sources(bank, checkpoint):
    root = Path(__file__).resolve().parents[1]
    paths = set(qualification_sources()) | {Path(__file__).resolve(), Path(bank).resolve(), Path(checkpoint).resolve()}
    # Capture actual loaded behavior/engine Python dependency sources, excluding
    # evolving unrelated trainers/objectives. Include installed ARC engine code.
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and (name.startswith(('pebby.', 'arcengine.')) or name == 'arcengine'):
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py':
                paths.add(path)
    paths.add(root / 'pebby/agent/joint_goal_replay.py')
    if not fastplan.available():
        raise RuntimeError('native solver required: ' + str(fastplan.load_error()))
    paths.add(fastplan.library_path().resolve())
    return {str(path.resolve()): digest(path) for path in sorted(paths)}


def check_sources(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError('collection source/input changed: ' + path)


def archive_sources(directory, bindings):
    """Preserve exact producer code, checkpoint, bank and native solver bytes."""
    archive = Path(directory) / 'sources'
    archive.mkdir(exist_ok=True)
    for original, expected in bindings.items():
        destination = archive / expected
        if not destination.exists():
            temporary = archive / ('.' + expected + '.tmp')
            try:
                with Path(original).open('rb') as source, temporary.open('xb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                if digest(temporary) != expected:
                    raise ValueError('producer input changed while archiving: ' + original)
                temporary.rename(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if digest(destination) != expected:
            raise ValueError('producer source archive checksum mismatch: ' + original)


def choose_specs(bank, split, levels_per_tier):
    selected, seen = [], set()
    groups = {tier: [] for tier in range(1, 8)}
    with Path(bank).open() as stream:
        for line in stream:
            spec = json.loads(line)
            check_spec(spec, split)
            if spec['seed'] in seen:
                raise ValueError('duplicate generated bank seed')
            seen.add(spec['seed'])
            groups[spec['difficulty']].append(spec)
    for tier, specs in groups.items():
        if len(specs) < levels_per_tier:
            raise ValueError('insufficient verified levels for tier ' + str(tier))
        specs.sort(key=lambda spec: (spec.get('reachable_states', 10**12), spec['seed']))
        selected.extend(specs[:levels_per_tier])
    # Complete one level of every tier before second examples.
    return sorted(selected, key=lambda spec: (groups[spec['difficulty']].index(spec), spec['difficulty']))


def _worker(spec, split, checkpoint, max_roots, temporary, destination, bindings):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    temporary, destination = Path(temporary), Path(destination)
    try:
        check_sources(bindings)
        model, _ = load_checkpoint(checkpoint, 'cpu')
        arrays, proof = collect_level(model, spec, split, max_roots=max_roots)
        temporary.mkdir()
        descriptors = {}
        for key, array in arrays.items():
            path = temporary / (key + '.npy')
            with path.open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
            descriptors[key] = dict(sha256=digest(path), bytes=path.stat().st_size,
                                    shape=list(array.shape), dtype=str(array.dtype))
        check_sources(bindings)
        info = dict(format=FORMAT, status='complete', **proof, arrays=descriptors,
                    producer_bindings_sha256=spec_digest(bindings))
        write_json(temporary / 'manifest.json', info)
        temporary.rename(destination)
        print(json.dumps(dict(event='level_complete', pid=os.getpid(), seed=spec['seed'],
                              tier=spec['difficulty'], roots=proof['roots'], events=proof['events'])), flush=True)
    except BaseException as error:
        print(json.dumps(dict(event='level_failed', pid=os.getpid(), seed=spec['seed'],
                              error=f'{type(error).__name__}: {error}')), flush=True)
        raise


def _available_memory():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def _stop(process):
    if process.is_alive():
        process.terminate()
    process.join(2)
    if process.is_alive():
        process.kill()
        process.join(2)
    if process.is_alive():
        raise RuntimeError('failed to stop exact level worker PID ' + str(process.pid))


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--levels-per-tier', type=int, default=2)
    parser.add_argument('--max-roots', type=int, default=128)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--level-seconds', type=int, default=60)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (not 1 <= args.max_seconds <= 180 or not 1 <= args.level_seconds <= 180
            or not 1 <= args.levels_per_tier <= 100 or not 8 <= args.max_roots <= 128):
        parser.error('wall limits must be 1..180; levels per tier 1..100; roots 8..128')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    if _available_memory() < 7 * 2**30:
        raise MemoryError('seven GiB available-memory reserve breached before initialization')
    print(json.dumps(dict(event='collector_start', pid=os.getpid(), device='cpu')), flush=True)
    # Load once to discover the exact behavior dependencies and verify weights.
    model, _ = load_checkpoint(args.checkpoint, 'cpu')
    del model
    gc.collect()
    bindings = sources(args.bank, args.checkpoint)
    specs = choose_specs(args.bank, args.split, args.levels_per_tier)
    configuration = dict(split=args.split, levels_per_tier=args.levels_per_tier, max_roots=args.max_roots,
                         checkpoint_sha256=digest(args.checkpoint), input_bank_sha256=digest(args.bank))
    args.out = args.out.resolve()
    manifest_path = args.out / 'manifest.json'
    if args.resume:
        meta = json.loads(manifest_path.read_text())
        if (meta.get('format') != FORMAT or meta.get('configuration') != configuration
                or meta.get('sources') != bindings or digest(args.out / 'bank.json') != meta['bank_sha256']
                or json.loads((args.out / 'bank.json').read_text()) != specs):
            raise ValueError('resume requires exact same sources, checkpoint, bank and collection configuration')
        # Validate every previous completed shard before admitting more data.
        if meta['shards']:
            with JointGoalReplay(args.out, args.split, allow_incomplete=True):
                pass
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / 'shards').mkdir()
        write_json(args.out / 'bank.json', specs)
        meta = dict(format=FORMAT, status='running', source='generated_only', split=args.split,
                    schema=schema(), configuration=configuration, sources=bindings,
                    bank_sha256=digest(args.out / 'bank.json'), shards=[], attempts=[], invocations=[],
                    device='cpu', threads=1, memory_reserve_gib=7,
                    selection='Within each tier: smallest persisted reachable-state graph, then seed. Bounded pilot; intentionally not representative generalization.',
                    source_archive_contract=dict(version=1, directory='sources', filenames='original SHA256 hex digest',
                                                 binding='sources maps original producer path to SHA256; archive bytes are authoritative for historical producer provenance; current training runtime is bound separately'),
                    cached_features_used=False, teacher_forcing=False, official_inputs_used=False,
                    labels_training_only=True, behavior_checkpoint=str(args.checkpoint.resolve()))
    archive_sources(args.out, bindings)
    invocation = dict(pid=os.getpid(), max_seconds=args.max_seconds, level_seconds=args.level_seconds,
                      started_local=datetime.now().astimezone().isoformat(), workers=[])
    meta['invocations'].append(invocation)
    meta['status'] = 'running'
    write_json(manifest_path, meta)
    active = None
    interrupted = None
    try:
        for spec in specs:
            if any(shard['seed'] == spec['seed'] for shard in meta['shards']):
                continue
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 2:
                interrupted = 'invocation_wall_cap'
                break
            if _available_memory() < 7 * 2**30:
                interrupted = 'memory_reserve'
                break
            destination = args.out / 'shards' / str(spec['seed'])
            temporary = args.out / 'shards' / ('.' + str(spec['seed']) + '.partial')
            # Leftovers can only be uncommitted output from this exact dataset.
            if temporary.exists():
                shutil.rmtree(temporary)
            if destination.exists():
                # A worker may finish atomically before its parent records it.
                info = json.loads((destination / 'manifest.json').read_text())
                if (info.get('status') != 'complete' or info.get('spec_sha256') != spec_digest(spec)
                        or info.get('producer_bindings_sha256') != spec_digest(bindings)):
                    raise ValueError('uncommitted shard has incompatible spec')
            else:
                active = multiprocessing.get_context('spawn').Process(target=_worker, args=(
                    spec, args.split, str(args.checkpoint.resolve()), args.max_roots,
                    str(temporary), str(destination), bindings))
                active.start()
                record = dict(pid=active.pid, seed=spec['seed'], tier=spec['difficulty'])
                invocation['workers'].append(record)
                write_json(manifest_path, meta)
                print(json.dumps(dict(event='worker_start', **record)), flush=True)
                deadline = time.monotonic() + min(args.level_seconds, max(0, remaining - 2))
                failure = None
                while active.is_alive():
                    active.join(.1)
                    if time.monotonic() >= deadline:
                        failure = 'level_or_invocation_timeout'
                        break
                    if _available_memory() < 7 * 2**30:
                        failure = 'memory_reserve'
                        break
                _stop(active)
                record.update(exitcode=active.exitcode, stopped=True, failure=failure)
                meta['attempts'].append(dict(record))
                active = None
                if not destination.exists():
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    write_json(manifest_path, meta)
                    if failure == 'memory_reserve':
                        interrupted = failure
                        break
                    continue
                info = json.loads((destination / 'manifest.json').read_text())
            meta['shards'].append(dict(seed=spec['seed'], roots=info['roots'], spec_sha256=spec_digest(spec),
                                       directory=str(destination.relative_to(args.out)),
                                       manifest_sha256=digest(destination / 'manifest.json')))
            check_sources(bindings)
            write_json(manifest_path, meta)
        check_sources(bindings)
        meta.update(status='complete' if len(meta['shards']) == len(specs) else 'incomplete',
                    sources_unchanged=True)
    except BaseException as error:
        meta.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if active is not None:
            _stop(active)
            invocation['workers'][-1].update(exitcode=active.exitcode, stopped=True)
        invocation.update(elapsed_seconds=time.monotonic() - started, stopped_reason=interrupted,
                          finished_local=datetime.now().astimezone().isoformat())
        meta['coverage'] = dict(requested_levels=len(specs), complete_levels=len(meta['shards']),
                                roots=sum(shard['roots'] for shard in meta['shards']),
                                tiers=sorted({spec['difficulty'] for spec in specs
                                              if any(shard['seed'] == spec['seed'] for shard in meta['shards'])}))
        write_json(manifest_path, meta)
    print(json.dumps(dict(event='collector_end', status=meta['status'], coverage=meta['coverage'],
                          pid=os.getpid())), flush=True)
    return meta


if __name__ == '__main__':
    main()
