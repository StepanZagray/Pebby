"""Bounded, source-bound public-H8 collection from the fresh reference TRAIN bank.

Publish train.npz, train.jsonl and report.json together by renaming a private
sibling directory. Failed staging directories are retained for diagnosis, never
accepted as published data. No training, official layouts or old bank is read.
"""
import argparse
from collections import Counter
import gc
import json
import multiprocessing
from multiprocessing import resource_tracker
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
from pebby.agent import world_data as wd
from pebby.agent.world_model import load_world_checkpoint
from pebby.agent.world_train import (load_dataset, require_verified_data,
    require_winning_coverage, validate_successor_labels)
from pebby.agent.on_policy_provenance import validate_on_policy_provenance
from pebby.ls20.bank import load
from pebby.ls20.plan import Oracle
from pebby.ls20.reference_profiles import DIFFICULTY_VERSION, SEARCH_LIMITS, profile_errors
from tools.collect_onpolicy_world import RowStore, collect_level
from tools.diagnose_reference_policy import memory_available
from tools.train_reference_repair import ROOT, PARENT, PARENT_SHA, digest, validate_parent

BANK = ROOT / 'data/ls20-reference-unequal-v1/train.jsonl'
BANK_SHA = 'f968f9b4a3690be69041eadd1afe129a3ba0527d55829d75687aa5c260532a81'
QUOTAS = (300, 200, 150, 120, 100, 80, 50)
GIB = 2**30


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def start_ticks(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return int(stat[stat.rfind(')') + 2:].split()[19])
    except FileNotFoundError:
        return None


def select_levels(specs, count=1000, seed=20260913, quotas=None):
    quotas = tuple(QUOTAS if quotas is None else quotas)
    if len(quotas) != 7 or any(type(q) is not int or q < 0 for q in quotas) or sum(quotas) != count or not 1 <= count <= 1000:
        raise ValueError('count must be 1..1000 and equal seven nonnegative quotas')
    if len({s['seed'] for s in specs}) != len(specs):
        raise ValueError('duplicate TRAIN seeds')
    for spec in specs:
        if (spec.get('split') != 'train' or spec.get('difficulty_version') != DIFFICULTY_VERSION
                or not 0 <= int(spec['seed']) < 1_000_000 or profile_errors(spec)):
            raise ValueError('current generated reference TRAIN bank required')
    rng = np.random.default_rng(seed)
    selected = []
    for tier, quota in enumerate(quotas, 1):
        group = sorted((s for s in specs if s['difficulty'] == tier), key=lambda s: s['seed'])
        if len(group) < quota:
            raise ValueError('insufficient distinct TRAIN levels in tier')
        selected.extend(group[int(i)] for i in rng.choice(len(group), quota, replace=False))
    return selected


def validate_sources(checkpoint, bank, generation_report):
    if digest(checkpoint) != PARENT_SHA or digest(bank) != BANK_SHA:
        raise ValueError('exact fresh parent and new TRAIN bank hashes required')
    generation = json.loads(generation_report.read_text())
    train = generation.get('banks', {}).get('train', {})
    if (generation.get('status') != 'complete' or generation.get('config', {}).get('profile') != DIFFICULTY_VERSION
            or train.get('sha256') != BANK_SHA or train.get('levels') != 10000):
        raise ValueError('completed 10000-level reference TRAIN publication required')
    specs = load(bank)
    if len(specs) != 10000:
        raise ValueError('expected all 10000 TRAIN records')
    parent = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_parent(parent)
    if set(s['seed'] for s in specs) != set(parent['train_seeds']):
        raise ValueError('new bank must exactly match fresh parent TRAIN seeds')
    return specs, parent


def collect_native(spec, policy, max_actions):
    original = wd.verified_context
    def native(*args, **kwargs):
        if 'engine' in kwargs:
            raise ValueError('unexpected Oracle engine override')
        return Oracle(*args, **kwargs, engine='fast')
    def verified(current, **kwargs):
        with patch.object(wd, 'Oracle', native):
            initial, oracle, proof = original(current, search_limit=SEARCH_LIMITS[current['difficulty'] - 1])
        if (initial is None or oracle is None or oracle.truncated or oracle.engine != 'fast'
                or proof.get('oracle_backend') != 'fast'
                or proof.get('search_limit') != SEARCH_LIMITS[current['difficulty'] - 1]):
            raise ValueError(f'complete native contextual teacher required: {proof}')
        return initial, oracle, proof
    with patch.object(wd, 'verified_context', verified):
        return collect_level(spec, policy, max_actions=max_actions)


def worker(specs, checkpoint, max_actions, shard):
    torch.set_num_threads(1)
    try:
        if digest(checkpoint) != PARENT_SHA:
            raise ValueError('worker checkpoint hash mismatch')
        policy, _ = load_world_checkpoint(checkpoint, 'cpu')
        policy = policy.cpu().float().eval().requires_grad_(False)
        if policy.config().get('history') != 8 or policy.config().get('architecture') != 'world':
            raise ValueError('WorldPolicy H8 required')
        shard.mkdir()
        for spec in specs:
            rows, proof, count = collect_native(spec, policy, max_actions)
            if count < 1:
                raise ValueError('selected seed has no policy rows')
            level_dir = shard / str(spec['seed']); level_dir.mkdir()
            store = RowStore(level_dir, len(rows)); store.append(rows)
            for column in store.columns.values():
                column.flush(); column._mmap.close()
            write(level_dir / 'receipt.json', dict(proof=proof, policy_rows=count, rows=len(rows)))
            del rows, store; gc.collect()
    except BaseException as error:
        write(shard.with_suffix('.error.json'), dict(error=repr(error)))
        raise


def verify_output(data, selected, validation):
    require_verified_data(data); require_winning_coverage(data)
    validate_successor_labels(data, 'reference on-policy output')
    validate_on_policy_provenance(data)
    wanted = {s['seed'] for s in selected}
    if set(map(int, data['seeds'])) != wanted or wanted & set(validation):
        raise ValueError('exact TRAIN seed retention / VAL disjointness failed')
    meta = data['meta']
    if meta.get('difficulty_version') != DIFFICULTY_VERSION:
        raise ValueError('reference difficulty_version required')
    marked, auxiliary = meta['on_policy_rows'], meta['auxiliary_rows']
    if set(marked) & set(auxiliary) or set(marked) | set(auxiliary) != set(range(len(data['seeds']))):
        raise ValueError('policy and auxiliary rows must partition output')
    if set(map(int, data['seeds'][marked])) != wanted:
        raise ValueError('each selected seed must retain a policy row')


def resident_bytes(pid):
    try:
        return next(int(line.split()[1]) * 1024 for line in
                    Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
    except (FileNotFoundError, StopIteration):
        return 0


def admission_bytes(active):
    return 6 * GIB + 3 * GIB + sum(max(0, 3 * GIB - resident_bytes(p.pid)) for p, *_ in active)


def stop_tracker(record):
    if record is None:
        return
    tracker = resource_tracker._resource_tracker
    if tracker._pid != record['pid']:
        raise RuntimeError('resource tracker identity changed')
    tracker._stop()  # Closes our pipe, waits/reaps this exact helper PID.
    record['reaped'] = start_ticks(record['pid']) != record['start_ticks']
    if not record['reaped']:
        raise RuntimeError('resource tracker cleanup incomplete')


def stop_workers(active, records):
    for process, _, _, record in active:
        if process.is_alive() and start_ticks(process.pid) == record['start_ticks']:
            process.terminate()
    for process, _, _, record in active:
        process.join(timeout=3)
        if process.is_alive() and start_ticks(process.pid) == record['start_ticks']:
            process.kill(); process.join(timeout=3)
        record['exitcode'] = process.exitcode
    for record in records:
        record['reaped'] = start_ticks(record['pid']) != record['start_ticks']
    if not all(r['reaped'] for r in records):
        raise RuntimeError('worker cleanup incomplete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=PARENT)
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--generation-report', type=Path, default=BANK.parent / 'generation-report.json')
    parser.add_argument('--data-cache-dir', type=Path, default=ROOT / 'data/reference-world-base-v1/array-cache')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--quotas', type=int, nargs=7)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-actions', type=int, default=150)
    parser.add_argument('--seconds', type=int, default=3600)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32 or not 1 <= args.max_actions <= 150 or not 1 <= args.seconds <= 86400:
        parser.error('workers 1..32, max-actions 1..150, seconds 1..86400 required')
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise FileExistsError('refusing published output overwrite')
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + args.out_dir.name + '.staging-', dir=args.out_dir.parent))
    started = time.monotonic(); active = []; records = []; tracker_record = None
    if resource_tracker._resource_tracker._pid is not None:
        raise RuntimeError('collector must run in a fresh standalone process')
    report = dict(status='running', format='pebby.reference-onpolicy.v1', pid=os.getpid(),
        start_ticks=start_ticks(os.getpid()), official_inputs_used=False, oracle_actions_in_policy_rollout=0,
        history=8, max_policy_actions_per_level=args.max_actions, workers_requested=args.workers,
        device='cpu', torch_threads=1, levels=[], worker_processes=records,
        limits=['Four checked branches per retained state; not exhaustive engine verification.',
                'Zero-policy-mask transition rows are retained. Expert and real failure auxiliaries are separate.',
                'Estimated worker reservations and monitored MemAvailable are not an OS memory guarantee.',
                'Only newly generated TRAIN layouts are read; source generation used aggregate official calibration.'])
    def guard():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError('collection deadline exceeded')
        if memory_available() < 7 * GIB:
            raise MemoryError('MemAvailable below 7 GiB abort threshold (6 GiB reserve plus 1 GiB grace)')
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        write(stage / 'report.json', report)
    def deadline(*_):
        raise TimeoutError('collection deadline exceeded')
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    previous_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('terminated')))
    signal.alarm(args.seconds)
    try:
        guard(); torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
        files = [args.checkpoint, args.bank, args.generation_report, Path(__file__),
                 ROOT / 'tools/collect_onpolicy_world.py', ROOT / 'tools/train_reference_repair.py',
                 ROOT / 'tools/diagnose_reference_policy.py', ROOT / 'tools/validate_extended_collector.py',
                 ROOT / 'third_party/ls20/ls20.py']
        files += sorted((ROOT / 'pebby/agent').glob('*.py'))
        files += sorted((ROOT / 'pebby/ls20').glob('*.py')) + sorted((ROOT / 'pebby/ls20').glob('*.c'))
        files += sorted((ROOT / 'pebby/ls20').glob('*.so'))
        sources = {str(p.resolve()): digest(p) for p in files}
        report['sourcebindings'] = sources
        def unchanged():
            guard()
            if any(digest(p) != sha for p, sha in sources.items()):
                raise ValueError('bound checkpoint/bank/code changed')
        specs, parent = validate_sources(args.checkpoint, args.bank, args.generation_report)
        selected = select_levels(specs, args.count, args.seed, args.quotas)
        validation = parent['validation_seeds']; config = parent['config']; del parent, specs; gc.collect()
        report.update(selected_seeds=[s['seed'] for s in selected], count=len(selected),
            quotas=list(args.quotas or QUOTAS), selection_seed=args.seed,
            policy_checkpoint=str(args.checkpoint.resolve()), policy_sha256=PARENT_SHA)
        unchanged(); persist()
        (stage / 'rows').mkdir(); (stage / 'shards').mkdir()
        store = RowStore(stage / 'rows', max(64, len(selected) * 180))
        marked, auxiliary = [], []
        context = multiprocessing.get_context('spawn'); next_index = 0
        scheduled = sorted(selected, key=lambda s: -s['difficulty'])
        while next_index < len(selected) or active:
            guard()
            # MemAvailable already excludes resident worker memory; reserve only
            # each active worker's remaining headroom plus one new allocation.
            available = memory_available()
            while (next_index < len(selected) and len(active) < args.workers
                   and available >= admission_bytes(active)):
                specs_chunk = scheduled[next_index:next_index + 8]; shard = stage / 'shards' / str(next_index)
                process = context.Process(target=worker, args=(specs_chunk, args.checkpoint, args.max_actions, shard))
                process.start()
                if tracker_record is None:
                    tracker_pid = resource_tracker._resource_tracker._pid
                    tracker_record = dict(pid=tracker_pid, start_ticks=start_ticks(tracker_pid))
                    report['resource_tracker'] = tracker_record
                record = dict(pid=process.pid, start_ticks=start_ticks(process.pid), seeds=[s['seed'] for s in specs_chunk])
                records.append(record); active.append((process, specs_chunk, shard, record)); next_index += len(specs_chunk)
                available = memory_available(); persist()
            if not active:
                raise MemoryError('insufficient admission headroom for one 3 GiB worker plus 6 GiB reserve')
            for item in list(active):
                process, specs_chunk, shard, record = item
                if process.is_alive():
                    continue
                process.join(); record['exitcode'] = process.exitcode
                active.remove(item)
                if process.exitcode != 0:
                    raise RuntimeError(f"worker failed; see {shard.with_suffix('.error.json')}")
                for spec in specs_chunk:
                    level_dir = shard / str(spec['seed'])
                    receipt = json.loads((level_dir / 'receipt.json').read_text())
                    columns = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in level_dir.glob('*.npy')}
                    count = receipt['rows']; policy_count = receipt['policy_rows']
                    marked.extend(range(store.count, store.count + policy_count))
                    auxiliary.extend(range(store.count + policy_count, store.count + count))
                    for offset in range(0, count, 32):
                        store.append([{k: v[i] for k, v in columns.items()} for i in range(offset, min(offset + 32, count))])
                        guard()
                    del columns
                    shutil.rmtree(level_dir)
                    report['levels'].append(receipt['proof']); report['rows'] = store.count
                    persist(); print(f"{len(report['levels'])}/{len(selected)} seed={spec['seed']} rows={count}", flush=True)
                shutil.rmtree(shard)
            time.sleep(.1)
        stop_workers(active, records); stop_tracker(tracker_record); tracker_record = None
        report['workers_exited'] = True
        unchanged()
        arrays = store.arrays()
        report.update(sources_unchanged=True, checkpoint_unchanged=True,
            on_policy_rows=len(marked), auxiliary_rows=len(auxiliary),
            on_policy_levels_by_difficulty=dict(Counter(s['difficulty'] for s in selected)))
        arrays['meta'] = dict(format=wd.FORMAT, source='generated_only', oracle_search='complete_only',
            difficulty_version=DIFFICULTY_VERSION, state_supervision_version=wd.STATE_SUPERVISION_VERSION,
            successor_policy_supervision_version=wd.SUCCESSOR_POLICY_SUPERVISION_VERSION,
            history=8, alternatives_per_state=4, samples=store.count,
            seeds=sorted(s['seed'] for s in selected), accepted_levels=len(selected), win_covered_levels=len(selected),
            coverage='on_policy_with_expert_anchors', on_policy_rows=marked, auxiliary_rows=auxiliary,
            auxiliary_collection='route_proportional_expert_and_real_exhaustion',
            collection_policy='model_greedy', on_policy_provenance=dict(report),
            behavior_checkpoint=dict(path=str(args.checkpoint.resolve()), sha256=PARENT_SHA, config=config),
            levels=report['levels'])
        verify_output(arrays, selected, validation); guard()
        output = stage / 'train.npz'; wd.save(output, arrays); guard()
        loaded = load_dataset(output, cache_dir=args.data_cache_dir)
        if not isinstance(loaded['frames'], np.memmap):
            raise ValueError('publication reload must use mmap cache')
        verify_output(loaded, selected, validation); unchanged()
        bank = stage / 'train.jsonl'
        bank.write_text(''.join(json.dumps(s) + '\n' for s in selected))
        report.update(status='complete', output=str(args.out_dir / output.name), output_sha256=digest(output),
            selected_bank=dict(path=str(args.out_dir / bank.name), sha256=digest(bank), count=len(selected)),
            data_cache_dir=str(args.data_cache_dir.resolve()), validation_disjoint=True,
            verified_reload=True, difficulty_version=DIFFICULTY_VERSION)
        for column in store.columns.values():
            column._mmap.close()
        del arrays, loaded, store; gc.collect(); shutil.rmtree(stage / 'rows'); shutil.rmtree(stage / 'shards')
        unchanged(); persist()
        # An existing nonempty publication cannot be replaced by directory rename.
        # Refuse even an empty output directory created since our initial check.
        if args.out_dir.exists():
            raise FileExistsError('publication target appeared during collection')
        stage.rename(args.out_dir)
        print(json.dumps(dict(status='complete', output=str(args.out_dir / 'train.npz'))), flush=True)
    except BaseException as error:
        signal.alarm(0)
        stop_workers(active, records); stop_tracker(tracker_record); tracker_record = None
        report.update(status='failed_partial', error=repr(error)); persist()
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGTERM, previous_term)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
