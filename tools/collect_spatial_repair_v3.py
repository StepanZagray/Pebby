"""Collect retained recovery policy TRAIN histories, preserving all verified labels.

Versioned orchestration; the archived v1 collector and checkpoint remain intact.
Only its checkpoint-independent, real-engine checked collect_level is reused.
"""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from pebby.agent.spatial_outcome_policy import load_checkpoint
from pebby.ls20.bank import load
from tools import collect_spatial_recovery as legacy
from tools.cache_reference_outcome_inputs import Bindings, open_array, release
from tools.collect_reference_onpolicy import BANK, BANK_SHA, select_levels, start_ticks, write
from tools.diagnose_reference_policy import digest, memory_available

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
CHECKPOINT_SHA = 'ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8'
PILOT_QUOTAS = (16, 12, 10, 8, 8, 6, 4)
FULL_QUOTAS = (64, 64, 80, 80, 80, 80, 64)
FORMAT = 'pebby.spatial-repair-collection.v3'
GIB = 2**30


def cohorts(arrays):
    loss = np.asarray(arrays['lost_life'], dtype=bool)
    live = ~loss & ~np.asarray(arrays['terminal'], dtype=bool)
    return dict(rows=len(loss), policy_valid_rows=int((arrays['optimal'] != 0).sum()),
                zero_optimal_rows=int((arrays['optimal'] == 0).sum()),
                life_loss_branches=int(loss.sum()),
                live_refill_branches=int((live & (arrays['next_steps'] > arrays['current_steps'][:, None])).sum()),
                exhaustion_rows=int((arrays['row_kind'] == 2).sum()),
                winning_branches=int(arrays['won'].sum()))


def shard_levels(selected, workers):
    if workers not in (1, 2):
        raise ValueError('only one or two bounded CPU workers supported')
    if workers == 1:
        return [selected]
    # Never run two high-tier native oracle searches at once.
    return [group for group in ([s for s in selected if s['difficulty'] >= 6],
                               [s for s in selected if s['difficulty'] < 6]) if group]


def verify_train_membership(selected, cache, bindings):
    manifest_path = cache / 'manifest.json'
    manifest_sha = bindings.add(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete' or not manifest.get('sources_unchanged') or not manifest.get('validation_disjoint'):
        raise ValueError('complete original TRAIN/validation cache required')
    seeds = {}
    for split, count in [('train', 10000), ('validation', 500)]:
        values = open_array(cache / split / 'seeds.npy', manifest['arrays'][split]['seeds'], bindings)
        seeds[split] = set(map(int, np.unique(values)))
        release({'seeds': values}, close=True)
        if len(seeds[split]) != count:
            raise ValueError('original split level counts differ')
    wanted = {int(s['seed']) for s in selected}
    if len(wanted) != len(selected) or seeds['train'] & seeds['validation'] or not wanted <= seeds['train'] or wanted & seeds['validation']:
        raise ValueError('selected levels violate exact original TRAIN membership')
    return manifest_sha


def worker(args):
    torch.set_num_threads(1)
    report = json.loads((args.out_dir / 'report.json').read_text())
    if report['source_checkpoint_sha256'] != args.checkpoint_sha256 or digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('worker checkpoint differs from pinned collection policy')
    legacy.verify_bindings(report['source_bindings'])
    policy, _ = load_checkpoint(args.checkpoint, 'cpu')
    policy.eval().requires_grad_(False)
    for spec in load(args.worker_bank):
        if memory_available() < (10 if spec['difficulty'] >= 6 else 9) * GIB:
            raise MemoryError('insufficient reserve before bounded native teacher search')
        started = time.monotonic()
        rows, proof = legacy.collect_level(spec, policy, args.max_actions, args.recovery_horizon)
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        # Keep every branch label. Eligibility is explicit and never invents a policy action.
        arrays['policy_valid'] = arrays['optimal'] != 0
        arrays['dynamics_valid'] = np.ones(len(rows), dtype=bool)
        path = args.out_dir / 'levels' / f"{spec['seed']}.npz"
        temp = path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
        proof.update(status='complete', source_checkpoint_sha256=args.checkpoint_sha256,
                     spec_sha256=legacy.spec_sha(spec), array_sha256=digest(path),
                     source_bindings=report['source_bindings'], worker_pid=os.getpid(),
                     worker_start_ticks=start_ticks(os.getpid()),
                     elapsed_seconds=time.monotonic() - started,
                     peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     cohorts=cohorts(arrays), all_engine_verified_rows_retained=True)
        write(path.with_suffix('.json'), proof)
        print(json.dumps(dict(seed=spec['seed'], tier=spec['difficulty'], rows=len(rows),
                             seconds=proof['elapsed_seconds'], cohorts=proof['cohorts'])), flush=True)
        del rows, arrays, proof
        gc.collect()
    legacy.verify_bindings(report['source_bindings'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--checkpoint-sha256', default=CHECKPOINT_SHA)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--quotas', type=int, nargs=7, default=PILOT_QUOTAS)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--max-actions', type=int, default=150)
    parser.add_argument('--recovery-horizon', type=int, default=64)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--worker-bank', type=Path)
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.checkpoint_sha256 != CHECKPOINT_SHA:
        parser.error('v3 round is pinned to retained ff883272 recovery policy')
    if not 1 <= args.max_actions <= 150 or not 1 <= args.recovery_horizon <= 64 or args.seconds <= 0:
        parser.error('positive deadline, policy cap1..150 and recovery horizon1..64 required')
    if args.worker_bank:
        return worker(args)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    bindings = Bindings()
    bindings.add(args.checkpoint, args.checkpoint_sha256)
    bindings.add(BANK, BANK_SHA)
    selected = select_levels(load(BANK), args.count, args.seed, args.quotas)
    cache_sha = verify_train_membership(selected, ROOT / 'data/reference-outcome-inputs-v1', bindings)
    metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if metadata.get('cache_manifest_sha256') != cache_sha:
        raise ValueError('checkpoint original TRAIN cache manifest differs')
    del metadata
    paths = [Path(__file__), Path(legacy.__file__), ROOT / 'tools/cache_reference_outcome_inputs.py',
             ROOT / 'tools/validate_extended_collector.py', ROOT / 'tools/collect_reference_onpolicy.py',
             ROOT / 'tools/diagnose_reference_policy.py', ROOT / 'third_party/ls20/ls20.py',
             *sorted((ROOT / 'pebby/agent').glob('*.py')),
             *sorted((ROOT / 'pebby/ls20').glob('*.py')),
             *sorted((ROOT / 'pebby/ls20').glob('*.c')), *sorted((ROOT / 'pebby/ls20').glob('*.so'))]
    for path in paths:
        bindings.add(path)
    if memory_available() < (16 if args.workers == 2 else 10) * GIB:
        raise MemoryError('bounded collection requires preflight host headroom')
    args.out_dir.mkdir(parents=True)
    (args.out_dir / 'levels').mkdir()
    (args.out_dir / 'train.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in selected))
    shards = shard_levels(selected, args.workers)
    banks = []
    for index, specs in enumerate(shards):
        path = args.out_dir / f'worker-{index}.jsonl'
        path.write_text(''.join(json.dumps(s) + '\n' for s in specs))
        banks.append(path)
        bindings.add(path)
    bindings.add(args.out_dir / 'train.jsonl')
    report = dict(format=FORMAT, status='running', pid=os.getpid(), start_ticks=start_ticks(os.getpid()),
                  started_local=datetime.now().astimezone().isoformat(), source_checkpoint_sha256=args.checkpoint_sha256,
                  official_inputs_used=False, public_inputs=legacy.PUBLIC, selected_bank='train.jsonl',
                  source_bindings=bindings.hashes, count=args.count, quotas=list(args.quotas), seed=args.seed,
                  max_actions=args.max_actions, recovery_horizon=args.recovery_horizon, max_recovery_contexts=3,
                  max_seconds=args.seconds, levels=[], workers=[], policy_oracle_actions=0,
                  all_engine_verified_rows_retained=True, policy_eligibility='optimal != 0; no invented action',
                  dynamics_eligibility='all verified rows including optimal=0 and exhaustion',
                  limitations=['One fixed retained-policy collection round; no learned reset or persistent memory.',
                               'Recovery suffixes stop at win or64 actions; partial suffixes remain marked partial.'])
    processes, streams, started = [], [], time.monotonic()
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        write(args.out_dir / 'report.json', report)
    def interrupted(*_):
        raise KeyboardInterrupt('v3 collection interrupted')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    persist()
    try:
        for index, bank in enumerate(banks):
            stream = (args.out_dir / f'worker-{index}.log').open('w')
            streams.append(stream)
            command = [sys.executable, '-m', 'tools.collect_spatial_repair_v3', '--out-dir', str(args.out_dir),
                       '--checkpoint', str(args.checkpoint), '--checkpoint-sha256', args.checkpoint_sha256,
                       '--worker-bank', str(bank), '--max-actions', str(args.max_actions),
                       '--recovery-horizon', str(args.recovery_horizon)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            processes.append(process)
            report['workers'].append(dict(pid=process.pid, start_ticks=start_ticks(process.pid), shard=index))
            persist()
        while True:
            codes = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'collection worker failed: {codes}; see worker logs')
            if time.monotonic() - started > args.seconds:
                raise TimeoutError('v3 collection deadline expired')
            if memory_available() < 6 * GIB:
                raise MemoryError('six GiB host reserve exhausted')
            report['completed_levels'] = len(list((args.out_dir / 'levels').glob('*.json')))
            persist()
            if all(code == 0 for code in codes):
                break
            time.sleep(1)
        legacy.verify_bindings(bindings.hashes)
        for spec in selected:
            relative = Path('levels') / f"{spec['seed']}.npz"
            proof_path = relative.with_suffix('.json')
            proof = json.loads((args.out_dir / proof_path).read_text())
            sha = digest(args.out_dir / relative)
            if (proof.get('status') != 'complete' or proof['array_sha256'] != sha
                    or proof['source_checkpoint_sha256'] != args.checkpoint_sha256
                    or proof['spec_sha256'] != legacy.spec_sha(spec) or not proof['policy_rows']
                    or proof['branch_checks']['branches'] != 4 * proof['rows']):
                raise ValueError('incomplete or mismatched selected level proof')
            report['levels'].append(dict(seed=spec['seed'], difficulty=spec['difficulty'], array_path=str(relative),
                sha256=sha, proof_path=str(proof_path), proof_sha256=digest(args.out_dir / proof_path),
                rows=proof['rows'], policy_rows=proof['policy_rows'], recovery_rows=proof['recovery_rows'],
                stop=proof['stop'], elapsed_seconds=proof['elapsed_seconds'], cohorts=proof['cohorts']))
        report.update(status='complete', sources_unchanged=True, rows=sum(p['rows'] for p in report['levels']))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process, receipt in zip(processes, report['workers']):
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            receipt.update(exitcode=process.returncode, reaped=start_ticks(process.pid) != receipt['start_ticks'])
        for stream in streams:
            stream.close()
        report.update(workers_reaped=all(p['reaped'] for p in report['workers']),
                      finished_local=datetime.now().astimezone().isoformat())
        persist()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return report


if __name__ == '__main__':
    main()
