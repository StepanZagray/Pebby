"""Resumable bounded-memory collection of complete seven-tier reference banks.

Uses the production collector and checks every expanded action against the exact
Oracle/real engine. Shards bound image memory; final merge streams through disk.
No encoder checkpoint is needed: the cache stores public observations for a fresh
end-to-end model, not embeddings from the obsolete model.
"""
import argparse
import gc
import fcntl
import hashlib
import json
import os
import multiprocessing
from multiprocessing.connection import wait
from types import SimpleNamespace
import traceback
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np
from pebby.agent import world_data
from pebby.agent.world_train import load_dataset, require_verified_data, require_winning_coverage, disjoint_seeds
from pebby.ls20.bank import load
from pebby.ls20.reference_profiles import DIFFICULTY_VERSION, profile_errors
from tools.stream_extended_collection import checked_worker, validate_arrays, ledger_workers, live_workers
from tools.merge_world_data import merge
from tools.regenerate_mechanism_banks import (memory_available, job_memory, active_reservation,
                                            take_pending, MEMORY_RESERVE, TIER_MEMORY_GIB, MAX_HEAVY_WORKERS)

CACHE_WORKER_OVERHEAD=512*1024**2


def checked_native_worker(task):
    """Collect with native-only search; never allocate the Python fallback graph.

    The reduced memory reservation covers the native backend. Patching before
    checked_worker starts also covers verified_context's default-auto call.
    A worker handles one task; restoration remains necessary on all exits.
    """
    original_oracle = world_data.Oracle
    def native_oracle(*args, **kwargs):
        kwargs['engine'] = 'fast'
        return original_oracle(*args, **kwargs)
    with patch.object(world_data, 'Oracle', native_oracle):
        return checked_worker(task)


def _admitted_collection(function, task, connection, marker, job_id):
    """One disposable worker; publish identity before any expensive Oracle work."""
    from tools.stream_extended_collection import process_identity
    identity = process_identity(os.getpid())
    write(Path(marker), dict(**identity, job=job_id))
    try:
        connection.send((True, function(task)))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()
        Path(marker).unlink(missing_ok=True)


class MemoryAdmittedPool:
    """Minimal ordered imap adapter, with admission before each process spawn.

    A finished level may wait in the bounded shard result buffer for an earlier
    level. MemAvailable includes that buffer; outstanding worker growth is
    reserved separately. No task/search budget or source order is changed.
    """
    def __init__(self, workers, directory, event=None):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('workers must be1..16')
        self.workers=workers; self.directory=Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.event=event or (lambda value: None)
        self.active={}; self.started=[]
        self.context=multiprocessing.get_context('spawn')

    def __enter__(self): return self

    def __exit__(self, *args):
        for process,connection,job in self.active.values():
            if process.is_alive(): process.terminate()
        for process,connection,job in self.active.values():
            process.join(timeout=5)
            if process.is_alive(): process.kill(); process.join()
            connection.close()
            (self.directory/(job['id']+'.json')).unlink(missing_ok=True)
        self.active.clear()
        live=[process.pid for process in self.started if process.is_alive()]
        self.event(dict(event='cleanup', pids=[p.pid for p in self.started], live_pids=live))
        if live: raise RuntimeError(f'collection workers remain alive: {live}')
        return False

    def imap(self, function, tasks):
        tasks=list(tasks)
        pending=[]
        for index,(spec,kwargs) in enumerate(tasks):
            if spec.get('difficulty_version') != DIFFICULTY_VERSION or spec.get('difficulty') not in range(1,8):
                raise ValueError('memory admission requires seven-reference metadata')
            pending.append(dict(id=f'level-{index:06d}',profile='ls20-reference-v1',
                                mode=spec['difficulty'], index=index,
                                memory_overhead_bytes=CACHE_WORKER_OVERHEAD))
        results={}; next_index=0
        while pending or self.active:
            while pending and len(self.active)<self.workers:
                active_jobs=[entry[2] for entry in self.active.values()]
                available=memory_available()
                reserved=active_reservation(active_jobs,self.directory)
                job=take_pending(pending,active_jobs,'fair',available=available,reserved=reserved)
                if job is None: break
                receiving,sending=self.context.Pipe(duplex=False)
                process=self.context.Process(target=_admitted_collection,args=(
                    function,tasks[job['index']],sending,str(self.directory/(job['id']+'.json')),job['id']))
                try: process.start()
                except BaseException:
                    receiving.close();sending.close();raise
                sending.close()
                self.started.append(process)
                self.active[job['index']]=(process,receiving,job)
                self.event(dict(event='admitted',pid=process.pid,index=job['index'],tier=job['mode'],
                    available_bytes=available,outstanding_reservation_bytes=reserved,
                    job_reservation_bytes=job_memory(job),active=len(self.active)))
            if not self.active:
                # External memory pressure is not a reason to reduce quality or
                # launch a job without its reservation. The outer deadline stays active.
                self.event(dict(event='waiting_for_memory',available_bytes=memory_available()))
                time.sleep(1)
                continue
            ready=wait([entry[1] for entry in self.active.values()], timeout=.2)
            for index,(process,connection,job) in list(self.active.items()):
                if connection not in ready and not connection.poll():
                    if not process.is_alive(): raise RuntimeError(f'worker{process.pid} exited without result')
                    continue
                try: success,result=connection.recv()
                except EOFError as error: raise RuntimeError(f'worker{process.pid} exited without result') from error
                process.join(timeout=5)
                if process.is_alive(): raise RuntimeError(f'worker{process.pid} did not exit after result')
                connection.close(); del self.active[index]
                if process.exitcode != 0 or not success:
                    raise RuntimeError(f'checked collection failed at source index{index}: {result}')
                results[index]=result
                self.event(dict(event='completed',pid=process.pid,index=index,exitcode=process.exitcode))
            while next_index in results:
                yield results.pop(next_index)
                next_index+=1
        if next_index != len(tasks): raise RuntimeError('ordered result coverage incomplete')


def admitted_build(specs, *, workers, marker_directory, event=None, **kwargs):
    """Keep production build aggregation unchanged, including workers=1 semantics."""
    context=SimpleNamespace(Pool=lambda _workers: MemoryAdmittedPool(workers,marker_directory,event))
    proxy=SimpleNamespace(get_context=lambda method: context)
    with patch.object(world_data,'multiprocessing',proxy), patch.object(world_data,'_worker',checked_native_worker):
        # Force build's ordered imap seam even for a one-worker admitted run.
        return world_data.build(specs,workers=max(2,workers),**kwargs)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def source_hashes():
    files = [Path(__file__).resolve(), Path('tools/stream_extended_collection.py'),
             Path('tools/validate_extended_collector.py'), Path('tools/merge_world_data.py'),
             Path('tools/regenerate_mechanism_banks.py')]
    files += list(Path('pebby/ls20').glob('*.py')) + list(Path('pebby/ls20').glob('*.c'))
    files += [Path('pebby/agent') / n for n in ('world_data.py', 'world_cache.py', 'world_train.py')]
    files += [Path('third_party/ls20/ls20.py')]
    return {str(p.resolve()): digest(p) for p in files}


def verify_sources(sources):
    for p, sha in sources.items():
        if digest(p) != sha:
            raise ValueError(f'source changed: {p}')


def check_bank(specs, split):
    if not specs or len({s['seed'] for s in specs}) != len(specs):
        raise ValueError('nonempty distinct level bank required')
    for s in specs:
        if s.get('split') != split or s.get('difficulty_version') != DIFFICULTY_VERSION:
            raise ValueError('exact seven-tier split required')
        if profile_errors(s):
            raise ValueError(f'profile violation at seed {s["seed"]}: {profile_errors(s)}')
    return specs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, choices=range(1,17), default=16)
    p.add_argument('--shard-levels', type=int, default=100)
    p.add_argument('--samples-per-level', type=int, default=32)
    p.add_argument('--seconds', type=int, default=43200)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if not 1 <= args.shard_levels <= 250 or args.samples_per_level < 16 or args.seconds < 1:
        p.error('shards1..250, samples>=16 and positive deadline required')
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out_dir / '.build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume and not (args.out_dir / 'manifest.json').exists() and any(args.out_dir.rglob('*.npz')):
        lock.close()
        raise ValueError('existing cache files lack original manifest; reuse refused')
    started = time.monotonic()
    print('PID', os.getpid(), flush=True)
    def stop(*_):
        raise InterruptedError('reference cache interrupted or deadline reached')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.alarm(args.seconds)
    generation_path = args.bank_dir / 'generation-report.json'
    generation = json.loads(generation_path.read_text())
    if generation.get('status') != 'complete':
        raise ValueError('complete generation report required before cache build')
    banks = {split: args.bank_dir / (split + '.jsonl') for split in ('train', 'validation')}
    for split, path in banks.items():
        if generation['banks'][split]['sha256'] != digest(path):
            raise ValueError('bank differs from completed generation report')
    sources = source_hashes() | {str(generation_path.resolve()): digest(generation_path)}
    sources.update({str(path.resolve()): digest(path) for path in banks.values()})
    config = dict(history=8, samples=args.samples_per_level, epsilon=.15, coverage='mixed_failure',
                  shard_levels=args.shard_levels, workers=args.workers,
                  scheduler=dict(version='memory-admitted-v2',max_tier6_7=MAX_HEAVY_WORKERS,max_tier5=4,
                                 oracle_backend='fast',
                                 reserve_bytes=MEMORY_RESERVE,tier_memory_gib=TIER_MEMORY_GIB,
                                 cache_worker_overhead_bytes=CACHE_WORKER_OVERHEAD))
    manifest = dict(sources=sources, config=config)
    manifest_path = args.out_dir / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('resume source/config drift')
    else:
        write(manifest_path, manifest)
    report_path = args.out_dir / 'build-report.json'
    report = dict(status='running', pid=os.getpid(), config=config, source='generated_only',
                  sources=sources, splits={}, official_frames_or_routes_used=False,
                  calibration='Generated profiles use official aggregate statistics only.')
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        write(report_path, report)
    persist()
    datasets = {}
    try:
        for split, bank in banks.items():
            specs = check_bank(load(bank), split)
            directory = args.out_dir / split
            directory.mkdir(exist_ok=True)
            ledger = (directory / ('workers-' + str(os.getpid()) + '.jsonl')).resolve()
            os.environ['PEBBY_EXTENDED_COLLECTION_PID_LEDGER'] = str(ledger)
            report['active_split'] = split
            report['worker_ledger'] = str(ledger)
            entry = dict(requested_levels=len(specs), shards=[])
            report['splits'][split] = entry
            proofs, paths = [], []
            for first in range(0, len(specs), args.shard_levels):
                selected = specs[first:first + args.shard_levels]
                path = directory / f'shard-{first:06d}.npz'
                proof_path = path.with_suffix('.json')
                report['active_shard'] = str(path)
                persist()
                if path.exists() or proof_path.exists():
                    if not path.exists() or not proof_path.exists():
                        raise ValueError('partial shard publication; inspect before resume')
                    record = json.loads(proof_path.read_text())
                    if record['sha256'] != digest(path) or record['seeds'] != [s['seed'] for s in selected]:
                        raise ValueError('shard source/hash mismatch')
                    arrays = load_dataset(path, history=8)
                    validate_arrays(arrays, selected)
                else:
                    scheduler_log = directory / f'scheduler-{first:06d}-{os.getpid()}.jsonl'
                    def scheduler_event(value):
                        with scheduler_log.open('a') as stream:
                            stream.write(json.dumps(dict(elapsed_seconds=time.monotonic()-started,**value))+'\n')
                        report['scheduler_latest']=value
                        if value['event'] != 'waiting_for_memory': persist()
                    report['scheduler_log']=str(scheduler_log)
                    arrays = admitted_build(selected, workers=args.workers,
                        marker_directory=directory/f'.active-{os.getpid()}',event=scheduler_event,
                        history=8,samples=args.samples_per_level,epsilon=.15,
                        coverage='mixed_failure',progress=False)
                    counts = validate_arrays(arrays, selected)
                    arrays['meta'].update(split=split, bank=str(bank), bank_sha256=digest(bank))
                    temp = path.with_suffix('.pending')
                    world_data.save(temp, arrays)
                    os.replace(temp, path)
                    record = dict(path=str(path), sha256=digest(path),
                                  seeds=[s['seed'] for s in selected], **counts)
                    write(proof_path, record)
                proofs.extend(arrays['meta']['levels'])
                paths.append(path)
                entry['shards'].append(record)
                entry['collected_levels'] = len(proofs)
                del arrays
                gc.collect()
                persist()
                print(json.dumps(dict(split=split, levels=len(proofs), total=len(specs))), flush=True)
            validity = directory / 'validity.json'
            write(validity, dict(status='complete', levels=proofs))
            merged = args.out_dir / (split + '.npz')
            receipt = directory / 'merged.json'
            if merged.exists():
                if not receipt.exists() or json.loads(receipt.read_text())['sha256'] != digest(merged):
                    raise ValueError('merged output lacks matching publication receipt')
            else:
                merge(paths, merged, validity, split, min_levels=len(specs))
                write(receipt, dict(sha256=digest(merged)))
            data = load_dataset(merged, history=8, cache_dir=args.out_dir / 'array-cache')
            if (data['meta'].get('source_sha256') != {str(path): digest(path) for path in paths}
                    or data['meta'].get('audit_sha256') != digest(validity)):
                raise ValueError('merged cache lineage differs from current shards or validity proofs')
            require_verified_data(data)
            require_winning_coverage(data, split)
            if set(map(int, data['seeds'])) != {s['seed'] for s in specs}:
                raise ValueError('merged cache does not cover exact complete bank')
            datasets[split] = data
            entry.update(status='complete', output=str(merged), sha256=digest(merged),
                         rows=len(data['seeds']), max_distance=int(data['distances'].max()),
                         life_loss_branches=int(data['lost_life'].sum()),
                         terminal_death_branches=int((data['terminal'] & ~data['won']).sum()))
            persist()
        disjoint_seeds(datasets['train'], datasets['validation'])
        verify_sources(sources)
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed_partial', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['active_shard'] = None
        workers = []
        for ledger in args.out_dir.glob(f'*/workers-{os.getpid()}.jsonl'):
            entries = [json.loads(line) for line in ledger.read_text().splitlines() if line]
            if any(w['pid'] != os.getpid() and w['ppid'] != os.getpid() for w in entries):
                raise ValueError('worker ledger has an unrelated process')
            workers.extend({(w['pid'], w['start_ticks']): w for w in entries}.values())
        report['workers'] = workers
        report['workers_stopped'] = not live_workers([w for w in workers if w['pid'] != os.getpid()])
        persist()
        lock.close()


if __name__ == '__main__':
    main()
