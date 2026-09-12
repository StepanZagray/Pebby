"""Regenerate a matched train/validation pair with bounded CPU workers.

Files are published only after complete quotas, exclusion checks and unchanged
source hashes. Existing banks are read for exclusion only, never overwritten.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

from pebby.ls20.generate import FORMAT
from tools.generate_mechanism_pilot import (
    LIMITATIONS, MODES, VERSION, canonical, coverage, digest, generate_one,
)


def _generate(job):
    split, seed, mode, attempts, limit = job
    rejections = []
    row, _ = generate_one(seed, mode, attempts=attempts, limit=limit,
                          seen_seeds=set(), seen_specs=set(), namespace_end=seed+64,
                          record_rejection=rejections.append)
    row.update(split=split, source='generated_only',
               curriculum_version=f'mechanism-{split}-v{VERSION}')
    return row, rejections, os.getpid()


def jobs_for(count, split, occupied, *, attempts=40, limit=600000):
    """Reserve disjoint seed blocks for deterministic independent workers."""
    start, end = (720000, 900000) if split == 'train' else (8100000, 9000000)
    jobs = []
    while len(jobs) < count:
        if start+64 > end:
            raise ValueError(f'{split} seed namespace cannot hold requested quotas')
        if not any(seed in occupied for seed in range(start, start+64)):
            jobs.append((split, start, MODES[len(jobs) % len(MODES)], attempts, limit))
        start += 64
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-count', type=int, default=2004)
    parser.add_argument('--validation-count', type=int, default=504)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--attempts', type=int, default=40)
    parser.add_argument('--limit', type=int, default=600000)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--out-dir', type=Path, default=Path('data/mechanism-bank-v2'))
    args = parser.parse_args(argv)
    if (min(args.train_count, args.validation_count, args.attempts, args.seconds) < 1
            or not 1 <= args.workers <= 4 or not 1 <= args.limit <= 600000):
        parser.error('positive counts/time, 1..4 workers and 1..600000 states required')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.out_dir / 'generation-report.json'
    started = time.monotonic()
    report = dict(status='running', format='pebby.mechanism-banks.v2', pid=os.getpid(),
                  worker_pids=[], requested={'train':args.train_count,'validation':args.validation_count},
                  limits={'workers':args.workers,'seconds':args.seconds,'states':args.limit,
                          'drafts_per_seed':args.attempts,'seeds_per_row':8,'geometry_redraws':16},
                  official_inputs_used=False, limitations=LIMITATIONS,
                  accepted={'train':0,'validation':0}, rejections=[], source_hashes={})
    code = [Path(__file__), Path('tools/generate_mechanism_pilot.py'),
            Path('pebby/agent/world_data.py'), *(Path('pebby/ls20') / p for p in
            ('generate.py','generation_quality.py','plan.py','env.py','layout.py','rails.py','fastplan.py','_fastplan.c'))]
    report['code_hashes'] = {str(p):digest(p) for p in code}
    def persist():
        report['elapsed_seconds'] = time.monotonic()-started
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(report_path)
    def stop(*_):
        raise TimeoutError('bounded regeneration interrupted or deadline reached')
    previous_handlers = {sig:signal.signal(sig, stop) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.alarm(args.seconds)
    pool, processes = None, []
    rows = {'train':[],'validation':[]}
    try:
        occupied, fingerprints = set(), set()
        for path in sorted(Path('data').rglob('*.jsonl')):
            if path.is_symlink():
                continue
            before, used = digest(path), False
            with path.open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get('format') == FORMAT:
                        occupied.add(int(row['seed']))
                        fingerprints.add(canonical(row))
                        used = True
            if used:
                if digest(path) != before:
                    raise RuntimeError(f'inventory changed: {path}')
                report['source_hashes'][str(path)] = before
        jobs = (jobs_for(args.train_count,'train',occupied,attempts=args.attempts,limit=args.limit)
                + jobs_for(args.validation_count,'validation',occupied,attempts=args.attempts,limit=args.limit))
        # Interleave splits for visible progress; every job retains its stable
        # seed/mode assignment regardless of process completion order.
        jobs.sort(key=lambda job: (job[1] % 720000, job[0]))
        persist()
        print(json.dumps({'pid':os.getpid(),'jobs':len(jobs),'workers':args.workers}),flush=True)
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
        futures = [pool.submit(_generate, job) for job in jobs]
        processes = list(pool._processes.values())
        report['worker_pids'] = [p.pid for p in processes]
        persist()
        for future in as_completed(futures):
            row, refusals, worker = future.result()
            fingerprint = canonical(row)
            if row['seed'] in occupied or fingerprint in fingerprints:
                raise RuntimeError(f'duplicate accepted seed/gameplay: {row["seed"]}')
            occupied.add(row['seed'])
            fingerprints.add(fingerprint)
            split = row['split']
            rows[split].append(row)
            report['rejections'].extend(refusals)
            report['accepted'][split] += 1
            with (args.out_dir / f'{split}.pending').open('a') as output:
                output.write(json.dumps(row,separators=(',',':'))+'\n')
            if sum(report['accepted'].values()) % 12 == 0:
                persist()
                print(json.dumps({'accepted':report['accepted'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        pool.shutdown(wait=True)
        pool = None
        report['code_hashes_unchanged'] = all(digest(p)==h for p,h in report['code_hashes'].items())
        report['inventory_hashes_unchanged'] = all(digest(p)==h for p,h in report['source_hashes'].items())
        if not report['code_hashes_unchanged'] or not report['inventory_hashes_unchanged']:
            raise RuntimeError('source or inventory changed during generation')
        if report['accepted'] != report['requested']:
            raise RuntimeError('incomplete split quotas')
        train_geometry = {r['geometry_sha256'] for r in rows['train']}
        if train_geometry & {r['geometry_sha256'] for r in rows['validation']}:
            raise RuntimeError('train/validation geometry overlap')
        report['coverage'] = {split:coverage(values) for split,values in rows.items()}
        report['rejection_counts'] = dict(Counter(r['reason'] for r in report['rejections']))
        report['banks'] = {}
        for split, values in rows.items():
            pending = args.out_dir / f'{split}.pending'
            pending.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'
                                       for r in sorted(values,key=lambda row:row['seed'])))
            path = args.out_dir / f'{split}.jsonl'
            pending.replace(path)
            report['banks'][split] = {'path':str(path),'sha256':digest(path),'levels':len(values)}
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed_closed', error=str(error))
    finally:
        signal.alarm(0)
        if pool is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
            pool.shutdown(wait=True,cancel_futures=True)
        report['workers_stopped'] = all(not p.is_alive() for p in processes)
        for sig, handler in previous_handlers.items():
            signal.signal(sig,handler)
        persist()
        print(json.dumps({k:report[k] for k in ('status','accepted','elapsed_seconds','workers_stopped')}),flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
