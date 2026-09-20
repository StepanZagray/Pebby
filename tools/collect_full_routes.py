"""Collect full-route controller data: every route, learner and recovery state.

Each selected level yields levels/<seed>.npz + levels/<seed>.json (see
``pebby.agent.full_route_data``). Workers run in a spawn pool with one torch
thread each and CUDA disabled; every worker loads the checkpoint itself through
``pebby.agent.model.load_checkpoint`` so any supported policy format works.
Levels whose complete proof already exists are skipped, so an interrupted or
timed-out run can be resumed by re-running the same command.

Example::

    uv run python -m tools.collect_full_routes --bank data/ls20-reference-unequal-v1/train.jsonl \\
        --out-dir /tmp/full-routes --checkpoint artifacts/spatial-recovery-v1/quality-fit/recovery.pt \\
        --quotas 2 2 2 2 2 2 2 --workers 8 --seconds 900
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

# CUDA must be invisible before torch initialises in this process or any child.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from pebby.agent import full_route_data as frd  # noqa: E402
from pebby.ls20.bank import load  # noqa: E402
from tools.diagnose_reference_policy import memory_available  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
DEFAULT_BANK = ROOT / 'data/ls20-reference-unequal-v1/train.jsonl'
GIB = 2**30
TIERS = 7
_worker_state = {}


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


def quotas_for(count, quotas):
    """Seven per-tier quotas: explicit, or ``count`` spread evenly, lowest tiers first."""
    if quotas is not None:
        quotas = tuple(int(q) for q in quotas)
        if len(quotas) != TIERS or any(q < 0 for q in quotas) or not sum(quotas):
            raise ValueError('seven nonnegative quotas with a positive total required')
        if count is not None and count != sum(quotas):
            raise ValueError('count must equal the quota total when both are given')
        return quotas
    if count is None or count < 1:
        raise ValueError('a positive count or seven quotas required')
    base, extra = divmod(count, TIERS)
    return tuple(base + (tier < extra) for tier in range(TIERS))


def select_levels(specs, quotas, seed):
    """Deterministic per-tier sample; a small local stand-in for
    tools.collect_reference_onpolicy.select_levels, which pins bank hashes and
    the TRAIN split and therefore cannot serve validation smoke tests."""
    if len({s['seed'] for s in specs}) != len(specs):
        raise ValueError('duplicate seeds in bank')
    rng = np.random.default_rng(seed)
    selected = []
    for tier, quota in enumerate(quotas, 1):
        group = sorted((s for s in specs if s.get('difficulty') == tier), key=lambda s: s['seed'])
        if len(group) < quota:
            raise ValueError(f'tier {tier} has {len(group)} levels, quota {quota}')
        if quota:
            selected.extend(group[int(i)] for i in rng.choice(len(group), quota, replace=False))
    return selected


def reserve_for(spec, reserve_gib):
    """Host reserve before a search: base plus planner headroom for tiers 6-7."""
    headroom = 2 if int(spec.get('difficulty', 1)) >= 6 else 1
    return (reserve_gib + headroom) * GIB


def _init_worker(checkpoint, learner_cap, reserve_gib):
    import torch
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    policy = None
    if checkpoint is not None:
        from pebby.agent.model import load_checkpoint
        policy, _ = load_checkpoint(checkpoint, 'cpu')
        policy = policy.cpu().float().eval()
        if hasattr(policy, 'requires_grad_'):
            policy.requires_grad_(False)
    _worker_state.update(policy=policy, learner_cap=learner_cap, reserve_gib=reserve_gib)


def _run_level(task):
    spec, out_dir = task
    started = time.monotonic()
    try:
        if memory_available() < reserve_for(spec, _worker_state['reserve_gib']):
            raise MemoryError('host memory reserve exhausted before native teacher search')
        rows, proof = frd.collect_level(spec, _worker_state['policy'], learner_cap=_worker_state['learner_cap'])
        elapsed = time.monotonic() - started
        proof.update(elapsed_seconds=elapsed, worker_pid=os.getpid(),
                     worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        array, proof_path = frd.write_level(out_dir, spec, rows, proof)
        return {'ok': True, 'seed': spec['seed'], 'difficulty': spec.get('difficulty'),
                'summary': summarize(json.loads(proof_path.read_text()), out_dir)}
    except BaseException as error:  # report, do not kill the pool
        return {'ok': False, 'seed': spec['seed'], 'difficulty': spec.get('difficulty'),
                'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
                'elapsed_seconds': time.monotonic() - started}


def summarize(proof, out_dir):
    array, proof_path = frd.level_paths(out_dir, proof['seed'])
    return {'seed': proof['seed'], 'difficulty': proof['difficulty'], 'rows': proof['rows'],
            'route_length': proof['route_length'], 'counts_by_kind': proof['counts_by_kind'],
            'unwinnable_rows': proof['unwinnable_rows'], 'learner_stop': proof['learner_stop'],
            'learner_won': proof['learner_won'], 'learner_actions': proof['learner_actions'],
            'optimal_choice_rate': proof['optimal_choice_rate'], 'recoveries': proof['recoveries'],
            'recoveries_won': proof['recoveries_won'], 'reachable_states': proof.get('reachable_states'),
            'elapsed_seconds': proof.get('elapsed_seconds'), 'worker_peak_rss_mib': proof.get('worker_peak_rss_mib'),
            'array_path': str(array.relative_to(out_dir)), 'proof_path': str(proof_path.relative_to(out_dir)),
            'array_sha256': proof['array_sha256']}


def aggregate(levels):
    tiers = defaultdict(lambda: {'levels': 0, 'rows': 0, 'seconds': 0., 'max_seconds': 0.,
                                 'max_worker_peak_rss_mib': 0., 'counts_by_kind': Counter(),
                                 'unwinnable_rows': 0, 'learner_won': 0, 'recoveries': 0, 'recoveries_won': 0})
    totals = Counter()
    for level in levels:
        tier = tiers[str(level['difficulty'])]
        tier['levels'] += 1
        tier['rows'] += level['rows']
        tier['seconds'] += level['elapsed_seconds'] or 0.
        tier['max_seconds'] = max(tier['max_seconds'], level['elapsed_seconds'] or 0.)
        tier['max_worker_peak_rss_mib'] = max(tier['max_worker_peak_rss_mib'], level['worker_peak_rss_mib'] or 0.)
        tier['counts_by_kind'].update(level['counts_by_kind'])
        tier['unwinnable_rows'] += level['unwinnable_rows']
        tier['learner_won'] += bool(level['learner_won'])
        tier['recoveries'] += level['recoveries']
        tier['recoveries_won'] += level['recoveries_won']
        totals.update(level['counts_by_kind'])
        totals['rows'] += level['rows']
        totals['unwinnable_rows'] += level['unwinnable_rows']
    for tier in tiers.values():
        tier['counts_by_kind'] = dict(tier['counts_by_kind'])
        tier['rows_per_level'] = tier['rows'] / tier['levels'] if tier['levels'] else None
    return {key: tiers[key] for key in sorted(tiers, key=int)}, dict(totals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bank', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='policy checkpoint (any format load_checkpoint supports); omit for route-only rows')
    parser.add_argument('--count', type=int, default=None)
    parser.add_argument('--quotas', type=int, nargs=TIERS, default=None, metavar='Q')
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--workers', type=int, default=default_workers())
    parser.add_argument('--learner-cap', type=int, default=200)
    parser.add_argument('--seconds', type=float, default=3600.)
    parser.add_argument('--memory-reserve-gib', type=float, default=6.,
                        help='base host reserve; +1 GiB per tier<6 search, +2 GiB per tier 6-7 search')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.learner_cap < 1 or args.seconds <= 0:
        parser.error('workers, learner-cap and seconds must be positive')
    quotas = quotas_for(args.count, args.quotas)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'levels').mkdir(exist_ok=True)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    selected = select_levels(load(args.bank), quotas, args.seed)
    (out_dir / 'selected.jsonl').write_text(''.join(json.dumps(s, separators=(',', ':')) + '\n' for s in selected))
    started = time.monotonic()
    report = {'format': frd.FORMAT + '.collection', 'status': 'running', 'pid': os.getpid(),
              'started_local': datetime.now().astimezone().isoformat(), 'bank': str(args.bank.resolve()),
              'bank_sha256': frd.digest(args.bank), 'checkpoint': None if checkpoint is None else str(checkpoint),
              'checkpoint_sha256': None if checkpoint is None else frd.digest(checkpoint),
              'count': sum(quotas), 'quotas': list(quotas), 'seed': args.seed, 'workers': args.workers,
              'learner_cap': args.learner_cap, 'seconds_limit': args.seconds,
              'memory_reserve_gib': args.memory_reserve_gib, 'cuda_visible_devices': '',
              'selected_seeds': [s['seed'] for s in selected], 'progress': {}, 'levels': [], 'failures': [],
              'per_tier': {}, 'counts': {}, 'elapsed_seconds': 0.}
    completed, pending = [], []
    for spec in selected:
        if frd.level_complete(out_dir, spec):
            completed.append(summarize(json.loads(frd.level_paths(out_dir, spec['seed'])[1].read_text()), out_dir))
        else:
            pending.append(spec)
    resumed = len(completed)

    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        report['levels'] = sorted(completed, key=lambda level: (level['difficulty'], level['seed']))
        report['per_tier'], report['counts'] = aggregate(completed)
        report['progress'] = {'total': len(selected), 'completed': len(completed), 'resumed': resumed,
                              'failed': len(report['failures']), 'pending': len(selected) - len(completed) - len(report['failures'])}
        frd.write_json(out_dir / 'report.json', report)
    persist()
    print(json.dumps({'selected': len(selected), 'resumed': resumed, 'pending': len(pending)}), flush=True)
    # Hardest levels first so the long tier 6-7 searches overlap the small ones.
    pending.sort(key=lambda s: (-int(s.get('difficulty', 1)), s['seed']))
    status = 'complete'
    pool = None
    try:
        if pending:
            if memory_available() < args.memory_reserve_gib * GIB:
                raise MemoryError('host reserve already exhausted before starting workers')
            context = multiprocessing.get_context('spawn')
            pool = context.Pool(min(args.workers, len(pending)), initializer=_init_worker,
                                initargs=(checkpoint, args.learner_cap, args.memory_reserve_gib))
            results = pool.imap_unordered(_run_level, [(spec, out_dir) for spec in pending])
            remaining = len(pending)
            while remaining:
                timeout = args.seconds - (time.monotonic() - started)
                if timeout <= 0:
                    raise TimeoutError('collection deadline expired')
                try:
                    result = results.next(timeout=timeout)
                except multiprocessing.TimeoutError:
                    raise TimeoutError('collection deadline expired') from None
                remaining -= 1
                if result['ok']:
                    completed.append(result['summary'])
                    print(json.dumps({'seed': result['seed'], 'difficulty': result['difficulty'],
                                      'rows': result['summary']['rows'], 'stop': result['summary']['learner_stop'],
                                      'seconds': round(result['summary']['elapsed_seconds'], 1),
                                      'peak_rss_mib': round(result['summary']['worker_peak_rss_mib'] or 0)}), flush=True)
                else:
                    report['failures'].append({key: result[key] for key in ('seed', 'difficulty', 'error', 'elapsed_seconds')})
                    print(json.dumps({'seed': result['seed'], 'failed': result['error']}), file=sys.stderr, flush=True)
                    print(result['traceback'], file=sys.stderr, flush=True)
                persist()
            pool.close()
            pool.join()
        if report['failures']:
            status = 'failed'
    except TimeoutError as error:
        status = 'timeout'
        report['error'] = repr(error)
    except BaseException as error:
        status = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        report['children_peak_rss_mib'] = children.ru_maxrss / 1024
        report['status'] = status
        persist()
        print(json.dumps({'status': status, 'completed': len(completed), 'total': len(selected),
                          'rows': report['counts'].get('rows', 0), 'elapsed_seconds': round(report['elapsed_seconds'], 1)}),
              flush=True)
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
