"""Collect immutable, verified extended-bank snapshots while generation appends.

One spawn pool at a time; at most four workers. Each published NPZ is a complete
mixed_failure/H8/base-16-row shard with route-proportional anchors and failure rows with all requested seeds and winning coverage.
All four actual branches of every expanded state are checked against the Oracle.
"""
from pebby.ls20.provenance import generated_context, row_contexts

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import threading
import time

import numpy as np

from pebby.agent import world_data
from pebby.ls20.extended_curriculum import ContractMismatch, NAMESPACE, VERSION, gameplay_hash
from tools.extend_extended_bank import check_prefix, digest, read_rows
from tools.validate_extended_collector import checked_expansion


def complete_lines(path):
    """Never consume an unterminated row being appended by another process."""
    lines = []
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            if not line.strip():
                raise ValueError(f'{path}: empty bank row')
            json.loads(line)  # malformed complete rows are failures, not pending writes
            lines.append(line)
    return lines


def process_identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        rest = text[text.rfind(')') + 2:].split()
        return {'pid': pid, 'ppid': int(rest[1]), 'start_ticks': int(rest[19])}
    except (OSError, ValueError, IndexError):
        return None


def rss_mib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.


def checked_worker(task):
    """Picklable replacement for world_data's task function; builder unchanged."""
    spec, kwargs = task
    identity = process_identity(os.getpid())
    ledger = os.environ['PEBBY_EXTENDED_COLLECTION_PID_LEDGER']
    # One short O_APPEND write is indivisible for this local regular-file ledger.
    fd = os.open(ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(identity) + '\n').encode())
    finally:
        os.close(fd)
    counts = Counter()
    original = world_data._expand
    def checked(*args, **kw):
        return checked_expansion(original, counts, *args, **kw)
    world_data._expand = checked
    try:
        rows, proof = world_data.collect_level(spec, **kwargs)
        if not rows or 'excluded' in proof:
            raise ContractMismatch(f"seed {spec['seed']}: collector dropped requested level ({proof})")
        proof['branch_verification'] = dict(counts)
        proof['collector_worker'] = identity
        return rows, proof
    finally:
        world_data._expand = original


def ledger_workers(path, parent_pid):
    result = {}
    if Path(path).exists():
        for line in complete_lines(path):
            entry = json.loads(line)
            if entry['ppid'] != parent_pid:
                raise ValueError('worker ledger names a process from another parent')
            result[(entry['pid'], entry['start_ticks'])] = entry
    return list(result.values())


def live_workers(workers):
    return [entry for entry in workers if process_identity(entry['pid']) == entry]


def validate_arrays(arrays, specs):
    # Import only in the parent, after the spawn pool returns: workers do not
    # each load PyTorch merely to check NumPy data provenance.
    from pebby.agent.world_train import require_verified_data, require_winning_coverage, validate_successor_labels
    require_verified_data(arrays)
    require_winning_coverage(arrays, 'extended shard')
    validate_successor_labels(arrays, 'extended shard')
    requested = [s['seed'] for s in specs]
    if len(set(requested)) != len(requested) or set(map(int, arrays['seeds'])) != set(requested):
        raise ContractMismatch('shard omitted, duplicated, or added requested seed')
    proofs = arrays['meta']['levels']
    if [row['seed'] for row in proofs] != requested or any('excluded' in row for row in proofs):
        raise ContractMismatch('shard proof order differs from immutable source order')
    if arrays['next_optimal'].dtype != np.uint8 or arrays['next_optimal'].shape != (len(arrays['seeds']), 4):
        raise ContractMismatch('missing or malformed next_optimal labels')
    if arrays['frames'].shape[1:] != (8, 64, 64) or arrays['next_frames'].shape[1:] != (4, 64, 64):
        raise ContractMismatch('history or actual successor observation contract differs')
    if not np.all(arrays['context_index'] == row_contexts(arrays['seeds'], arrays['meta']['levels'])):
        raise ContractMismatch('wrong per-row gameplay context')
    branches = sum(p['branch_verification']['branches'] for p in proofs)
    expansions = sum(p['branch_verification']['expansions'] for p in proofs)
    if branches != 4 * expansions or expansions < len(arrays['seeds']):
        raise ContractMismatch('not all expanded actions were checked')
    return {'levels': len(specs), 'rows': len(arrays['seeds']), 'checked_branches': branches,
            'checked_expansions': expansions, 'win_covered_levels': arrays['meta']['win_covered_levels'],
            'valid_successor_labels': int((arrays['next_optimal'] != 0).sum())}


def validate_bank_prefix(lines, split, expected_pilot_hash, pilot_count, prior_lines, old_hashes, seen):
    if len(lines) < pilot_count:
        raise ValueError('growing bank lost its complete pilot prefix')
    if hashlib.sha256(b''.join(lines[:pilot_count])).hexdigest() != expected_pilot_hash:
        raise ValueError('growing bank no longer matches original pilot prefix')
    if len(lines) < len(prior_lines) or lines[:len(prior_lines)] != prior_lines:
        raise ValueError('previously completed bank rows were rewritten')
    specs = [json.loads(line) for line in lines]
    check_prefix(specs, split)
    for index in range(len(prior_lines), len(lines)):
        spec = specs[index]
        fingerprint = gameplay_hash(spec)
        if fingerprint in old_hashes or fingerprint in seen:
            raise ValueError('extended gameplay duplicates an old/new/other-split row')
        seen.add(fingerprint)
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-dir', type=Path, default=Path('data/extended-bank-v2'))
    parser.add_argument('--generation-report', type=Path, default=Path('artifacts/world-extended-bank-v2.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/world-extended-v2'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/world-extended-collection.json'))
    parser.add_argument('--group-size', type=int, choices=(250, 500), default=500)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--deadline-seconds', type=int, default=7200)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args(argv)
    if args.deadline_seconds <= 0 or not 1 <= args.poll_seconds <= 30:
        parser.error('deadline must be positive; poll interval must be 1..30 seconds')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if list(args.out_dir.glob('*-part-*.npz')) or list(args.out_dir.glob('*-part-*.jsonl')):
        raise ValueError('output directory already contains shards; refusing to overwrite')
    ledger = args.out_dir / 'worker-pids.jsonl'
    if ledger.exists():
        raise ValueError('worker PID ledger already exists; use a fresh directory')
    ledger.touch(mode=0o600)
    os.environ['PEBBY_EXTENDED_COLLECTION_PID_LEDGER'] = str(ledger.resolve())
    generation = json.loads(args.generation_report.read_text())
    totals = generation['requested_total']
    if totals != {'train': 10000, 'validation': 2000}:
        raise ValueError('expected the authorized 10000/2000 generated bank totals')
    old_hashes = set()
    for path, expected_hash in generation['existing_banks'].items():
        if digest(path) != expected_hash:
            raise ValueError('existing verified source bank changed')
        old_hashes.update(gameplay_hash(row) for row in read_rows(path))
    started = time.monotonic()
    report = {'status': 'running', 'phase': 'starting', 'pid': os.getpid(), 'workers_limit': args.workers,
              'group_size': args.group_size, 'target_levels': totals, 'shards': [], 'captured_levels': {'train': 0, 'validation': 0},
              'bank_dir': str(args.bank_dir), 'generation_report': str(args.generation_report),
              'pid_ledger': str(ledger), 'history': 8, 'samples_per_level': 16, 'epsilon': .15, 'coverage': 'mixed_failure',
              'search_limit': 600000, 'official_gameplay_inputs_used': False, 'active_training_data_changed': False,
              'deadline_seconds': args.deadline_seconds, 'code_sha256': {p: digest(p) for p in
              (__file__, 'tools/validate_extended_collector.py', 'pebby/agent/world_data.py',
               'pebby/ls20/extended_curriculum.py', 'pebby/ls20/plan.py', 'pebby/ls20/rails.py')},
              'coverage_limit': 'All four actions from every collector-expanded state; not all reachable states in each level.'}
    lock = threading.RLock()
    stop_monitor = threading.Event()
    monitored_peak = 0.
    child_identities = {}
    def persist(**changes):
        nonlocal monitored_peak
        with lock:
            report.update(changes)
            workers = ledger_workers(ledger, os.getpid())
            active = live_workers(workers)
            # Track spawn's resource-tracker helper as well as busy workers.
            children_path = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children')
            for child in children_path.read_text().split():
                identity = process_identity(int(child))
                if identity is not None:
                    child_identities[(identity['pid'], identity['start_ticks'])] = identity
            active_children = live_workers(list(child_identities.values()))
            rss = rss_mib(os.getpid()) + sum(rss_mib(w['pid']) for w in active_children)
            monitored_peak = max(monitored_peak, rss)
            report.update(elapsed_seconds=time.monotonic()-started, worker_processes=workers,
                          active_worker_pids=[w['pid'] for w in active], parent_rss_mib=rss_mib(os.getpid()),
                          spawned_child_processes=list(child_identities.values()),
                          active_child_pids=[w['pid'] for w in active_children],
                          current_parent_plus_worker_rss_mib=rss, sampled_peak_parent_plus_worker_rss_mib=monitored_peak,
                          parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temp = args.report.with_suffix('.tmp')
            temp.write_text(json.dumps(report, indent=2)+'\n')
            temp.replace(args.report)
    def monitor():
        while not stop_monitor.wait(3):
            persist()
    def deadline(*_):
        raise TimeoutError('two-hour collection deadline exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    persist()
    print('PID', os.getpid(), 'workers <=', args.workers, flush=True)
    previous = {'train': [], 'validation': []}
    captured = {'train': 0, 'validation': 0}
    seen, completed = set(), []
    temporary = None
    original_worker = world_data._worker
    world_data._worker = checked_worker
    try:
        while captured != totals:
            generation = json.loads(args.generation_report.read_text())
            if generation['status'] not in ('running', 'complete'):
                raise ContractMismatch(f"generation stopped: {generation['status']} {generation.get('error')}")
            available = {}
            for split in ('train', 'validation'):
                lines = complete_lines(args.bank_dir / f'{split}.jsonl')
                if len(lines) > totals[split]:
                    raise ValueError('growing bank exceeds authorized level count')
                available[split] = validate_bank_prefix(lines, split, generation['pilot_prefix_sha256'][split],
                    generation['pilot_prefix_levels'][split], previous[split], old_hashes, seen)
                previous[split] = lines
            ready = next((split for split in ('train', 'validation')
                          if len(available[split]) - captured[split] >= min(args.group_size, totals[split]-captured[split])
                          and captured[split] < totals[split]), None)
            if ready is None:
                if generation['status'] == 'complete':
                    raise ValueError('completed generation has insufficient rows for remaining shards')
                persist(phase='waiting_for_complete_group', available_levels={s:len(v) for s,v in available.items()})
                time.sleep(args.poll_seconds)
                continue
            split = ready
            start = captured[split]
            stop = min(start + args.group_size, totals[split])
            specs = available[split][start:stop]
            part = start // args.group_size
            snapshot = args.out_dir / f'{split}-part-{part:03d}.jsonl'
            output = snapshot.with_suffix('.npz')
            payload = b''.join(previous[split][start:stop])
            with snapshot.open('xb') as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            snapshot.chmod(0o444)
            source_hash = hashlib.sha256(payload).hexdigest()
            persist(phase='collecting', current_shard={'split':split,'part':part,'levels':len(specs),'snapshot':str(snapshot)},
                    available_levels={s:len(v) for s,v in available.items()})
            print('START',split,part,'levels',len(specs),'snapshot',source_hash,flush=True)
            shard_started = time.monotonic()
            arrays = world_data.build(specs, workers=args.workers, progress=True, history=8, samples=16,
                                       epsilon=.15, coverage='mixed_failure', search_limit=600000)
            if live_workers(ledger_workers(ledger, os.getpid())):
                raise RuntimeError('world_data pool left active workers after returning')
            verified = validate_arrays(arrays,specs)
            if digest(snapshot) != source_hash:
                raise ValueError('immutable source snapshot changed during collection')
            arrays['meta'].update(source_bank_snapshot=str(snapshot), source_bank_sha256=source_hash,
                                  extended_curriculum_version=VERSION, generation_namespace=NAMESPACE,
                                  collection_branch_verification=verified)
            temporary = output.with_suffix('.tmp.npz')
            persist(phase='publishing')
            world_data.save(temporary,arrays)
            del arrays
            # Complete validation precedes publication; compressed NPZ CRCs can
            # be checked by downstream readers without another full memory copy.
            temporary.replace(output)
            temporary = None
            elapsed = time.monotonic()-shard_started
            entry = {**verified,'split':split,'part':part,'snapshot':str(snapshot),'snapshot_sha256':source_hash,
                     'npz':str(output),'npz_sha256':digest(output),'elapsed_seconds':elapsed,
                     'levels_per_second':len(specs)/elapsed,'rows_per_second':verified['rows']/elapsed,
                     'first_seed':specs[0]['seed'],'last_seed':specs[-1]['seed']}
            completed.append(entry)
            captured[split] = stop
            persist(phase='shard_complete',shards=list(completed),captured_levels=dict(captured))
            print('DONE',json.dumps(entry),flush=True)
        persist(status='complete',phase='complete')
    except BaseException as error:
        persist(status='blocked_contract_mismatch' if isinstance(error,ContractMismatch) else 'failed',
                phase='error',error=repr(error))
        raise
    finally:
        signal.alarm(0)
        world_data._worker = original_worker
        stop_monitor.set(); thread.join(timeout=5)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        # Pool context normally terminated/joined its children. Escalate only
        # exact still-live worker identities from this parent's own ledger.
        survivors = live_workers(ledger_workers(ledger,os.getpid()))
        for entry in survivors:
            os.kill(entry['pid'],signal.SIGTERM)
        end = time.monotonic()+5
        while survivors and time.monotonic()<end:
            time.sleep(.1)
            survivors=live_workers(survivors)
        for entry in survivors:
            os.kill(entry['pid'],signal.SIGKILL)
        persist(cleanup_workers_remaining=[w['pid'] for w in live_workers(ledger_workers(ledger,os.getpid()))])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
