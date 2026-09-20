#!/usr/bin/env python3
"""Build a frozen, generated-only LS20 confirmation panel.

The panel is selected by a preregistered seed schedule and the reference
generator's proof contract only.  No model, checkpoint, cache, or GPU is
loaded.  Accepted rows are durably written while workers run so a bounded
deadline produces an auditable partial smoke panel rather than a false
70-level result.
"""

from collections import Counter
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORMAT = 'pebby.ls20.level.v1'
GIB = 1024 ** 3
MIN_AVAILABLE = 6 * GIB
PROFILE_LIMITS = (600_000, 600_000, 1_000_000, 2_000_000,
                  4_000_000, 24_000_000, 32_000_000)
FINGERPRINTS = ('seed', 'gameplay_sha256', 'geometry_sha256',
                'geometry_d4_sha256')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_line(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def process_start_ticks(pid):
    """Return Linux process starttime (field 22), or None if unavailable."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        # comm is parenthesized and may itself contain spaces; split after its
        # final closing parenthesis.  The remaining fields begin at field 3.
        fields = stat[stat.rfind(')') + 2:].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return None


def process_entry_exists(pid):
    return Path(f'/proc/{pid}').exists()


def cleanup_record(process, starttime_ticks=None):
    return dict(pid=process.pid, starttime_ticks=starttime_ticks,
                exitcode=process.exitcode, alive=process.is_alive(),
                proc_entry_exists=process_entry_exists(process.pid))


def command_tokens(command):
    if isinstance(command, (str, bytes)):
        return [os.fsdecode(command)]
    return [os.fsdecode(token) for token in command]


def is_compiler_command(command):
    tokens = command_tokens(command)
    if not tokens:
        return False
    compiler = Path(tokens[0]).name in {'cc', 'gcc', 'clang'}
    return compiler or ('-std=c99' in tokens and '-shared' in tokens)


def command_string(command):
    return ' '.join(command_tokens(command))


def fingerprint_subset_errors(candidate, baseline):
    """Return every fingerprint field where candidate escapes baseline."""
    return {field: sorted(candidate[field] - baseline[field])
            for field in FINGERPRINTS if not candidate[field] <= baseline[field]}


def exit_code_for(status, target_count):
    """Only a complete final panel or complete smoke panel is success."""
    if target_count == 70:
        return 0 if status == 'complete' else 1
    return 0 if status == 'smoke_complete' else 1


def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is unavailable')


def rows_from(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inventory(paths):
    rows = []
    for path in paths:
        rows.extend(rows_from(path))
    values = {field: set() for field in FINGERPRINTS}
    for row in rows:
        if row.get('format') != FORMAT:
            continue
        for field in FINGERPRINTS:
            if field in row:
                values[field].add(row[field])
    return values


def seed_schedule(seed_key, quotas, extra, start, end, occupied):
    """Create all candidate seeds before any generation or model evaluation."""
    rng = random.Random(seed_key)
    reserved = set(occupied)
    schedule = []
    for tier in range(1, 8):
        needed = quotas[tier - 1] + extra
        candidates = []
        while len(candidates) < needed:
            seed = rng.randrange(start, end)
            if seed in reserved:
                continue
            reserved.add(seed)
            candidates.append(seed)
        for candidate_index, seed in enumerate(candidates):
            schedule.append(dict(tier=tier, candidate_index=candidate_index,
                                 seed=seed,
                                 ordinal=len(schedule)))
    return schedule


def worker_main(task_queue, result_queue, compiler_queue, attempts):
    # Import inside the child so the parent can inspect and write the schedule
    # before any generator module starts doing work.  fastplan may compile its
    # local C kernel through subprocess.run; wrap Popen only in this worker so
    # the parent can identify and clean up that exact compiler child.
    import subprocess

    original_popen = subprocess.Popen

    def tracked_popen(*popen_args, **popen_kwargs):
        command = popen_kwargs.get('args', popen_args[0] if popen_args else [])
        process = original_popen(*popen_args, **popen_kwargs)
        if is_compiler_command(command):
            compiler_queue.put({
                'event': 'compiler_started', 'pid': process.pid,
                'starttime_ticks': process_start_ticks(process.pid),
                'worker_pid': os.getpid(), 'command': command_string(command),
            })
        return process

    subprocess.Popen = tracked_popen
    try:
        from pebby.ls20.reference_generator import generate_level

        while True:
            task = task_queue.get()
            if task is None:
                return
            started = time.monotonic()
            try:
                row = generate_level(
                    task['seed'], task['tier'], attempts=attempts,
                    search_limit=PROFILE_LIMITS[task['tier'] - 1], split='validation')
                result = dict(status='accepted', row=row)
            except BaseException as error:  # report bounded generator rejection
                result = dict(status='rejected', error_type=type(error).__name__,
                              error=str(error))
            result.update(tier=task['tier'], candidate_index=task['candidate_index'],
                          seed=task['seed'], ordinal=task['ordinal'], pid=os.getpid(),
                          elapsed_seconds=time.monotonic() - started)
            result_queue.put(result)
    finally:
        subprocess.Popen = original_popen


def terminate_workers(processes, starttime_ticks):
    """Terminate, kill if needed, and wait on exactly owned workers."""
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=10)
    return [cleanup_record(process, starttime_ticks.get(process.pid))
            for process in processes]


def compiler_owned_alive(record):
    """Check a compiler PID only when its starttime still identifies our child."""
    pid = record['pid']
    if not process_entry_exists(pid):
        return False
    expected = record.get('starttime_ticks')
    return expected is not None and process_start_ticks(pid) == expected


def terminate_compilers(records):
    """Clean up only compiler PIDs reported by this invocation's workers."""
    for record in records.values():
        if compiler_owned_alive(record):
            try:
                os.kill(record['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(compiler_owned_alive(record)
                                              for record in records.values()):
        time.sleep(0.05)
    for record in records.values():
        if compiler_owned_alive(record):
            try:
                os.kill(record['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(compiler_owned_alive(record)
                                              for record in records.values()):
        time.sleep(0.05)
    cleaned = []
    for record in records.values():
        current_ticks = process_start_ticks(record['pid'])
        cleaned.append({**record, 'alive': compiler_owned_alive(record),
                        'proc_entry_exists': process_entry_exists(record['pid']),
                        'current_starttime_ticks': current_ticks})
    return cleaned


def validate_row(row, tier, known):
    if not isinstance(row, dict):
        return 'row is not an object'
    required = {
        'format': FORMAT, 'source': 'generated_only', 'split': 'validation',
        'difficulty': tier, 'engine_verified': True,
        'context_engine_verified': True, 'engine_win': True,
        'context_index': tier - 1, 'training_context_index': tier - 1,
        'verification_level_index': tier - 1, 'search_truncated': False,
        'levels_completed': 1, 'replay_lives': 3,
    }
    for key, expected in required.items():
        if row.get(key) != expected:
            return f'{key}={row.get(key)!r}, expected {expected!r}'
    proof = row.get('proof')
    if not isinstance(proof, dict) or proof.get('engine_win') is not True:
        return 'missing complete nested proof'
    proof_mirrors = ('seed', 'difficulty', 'difficulty_version', 'context_index',
                     'context_engine_verified', 'context_optimal_actions',
                     'optimal_actions', 'engine_win', 'replay_lives',
                     'levels_completed', 'search_truncated', 'search_limit',
                     'reachable_states', 'oracle_backend')
    for field in proof_mirrors:
        if proof.get(field) != row.get(field):
            return f'proof.{field} does not match outer row'
    for field in FINGERPRINTS:
        if field not in row:
            return f'missing {field}'
        if row[field] in known[field]:
            return f'duplicate {field}'
    # The generator already applies this contract; this second check makes the
    # final bank fail closed if its implementation changes during the run.
    from pebby.ls20.reference_profiles import profile_errors
    errors = profile_errors(row)
    if errors:
        return '; '.join(errors)
    return None


def write_progress(path, payload):
    payload = dict(payload)
    payload['updated_local'] = datetime.now().astimezone().isoformat()
    atomic_json(path, payload)


def run(args):
    if args.workers not in (1, 2):
        raise ValueError('workers must be 1 or 2')
    if args.target_count not in (14, 70):
        raise ValueError('target_count must be 14 or 70')
    if args.target_count % 7:
        raise ValueError('target_count must divide into seven equal tiers')
    if args.candidate_extra < 0:
        raise ValueError('candidate_extra must be nonnegative')
    if args.start_seed < 10_000_000 or args.end_seed <= args.start_seed:
        raise ValueError('fresh namespace must be seed>=10000000 and half-open')
    if args.deadline_seconds < 1 or args.attempts < 1:
        raise ValueError('positive deadline and attempts required')
    if args.out_dir.exists():
        if any(args.out_dir.iterdir()):
            raise ValueError(f'refusing nonempty staging directory: {args.out_dir}')
    else:
        args.out_dir.mkdir(parents=True)
    if not args.initial_selection.is_file():
        raise ValueError(f'missing preserved initial selection: {args.initial_selection}')

    train_path = ROOT / 'data/ls20-reference-unequal-v1/train.jsonl'
    validation_path = ROOT / 'data/ls20-reference-unequal-v1/validation.jsonl'
    development_path = ROOT / 'artifacts/reference-grounding-repair-v1/validation70.jsonl'
    source_paths = [train_path, validation_path, development_path]
    code_paths = [
        Path(__file__).resolve(), ROOT / 'pebby/ls20/reference_generator.py',
        ROOT / 'pebby/ls20/reference_profiles.py', ROOT / 'pebby/ls20/generation_quality.py',
        ROOT / 'pebby/ls20/generate.py', ROOT / 'pebby/ls20/plan.py',
        ROOT / 'pebby/ls20/fastplan.py', ROOT / 'pebby/ls20/_fastplan.c',
        ROOT / 'pebby/ls20/rails.py',
        ROOT / 'pebby/ls20/env.py', ROOT / 'pebby/ls20/layout.py',
        ROOT / 'pebby/ls20/extended_curriculum.py', ROOT / 'pebby/ls20/names.py',
        ROOT / 'pebby/ls20/provenance.py', ROOT / 'third_party/ls20/ls20.py',
    ]
    source_hashes = {str(path): digest(path) for path in source_paths}
    code_hashes = {str(path): digest(path) for path in code_paths}
    known = inventory(source_paths[:2])
    development_rows = rows_from(development_path)
    development_known = inventory([development_path])
    if (not development_rows or
            any(row.get('format') != FORMAT for row in development_rows) or
            any(len(development_known[field]) != len(development_rows)
                for field in FINGERPRINTS)):
        raise ValueError('development70 is missing complete unique fingerprints')
    development_subset_errors = fingerprint_subset_errors(development_known, known)
    if development_subset_errors:
        raise ValueError('development70 is not a subset of validation fingerprints: '
                         f'{development_subset_errors}')
    initial_selection_sha256 = digest(args.initial_selection)
    quotas = [args.target_count // 7] * 7
    schedule = seed_schedule(args.seed_key, quotas, args.candidate_extra,
                             args.start_seed, args.end_seed, known['seed'])
    schedule_path = args.out_dir / 'schedule.json'
    atomic_json(schedule_path, {
        'format': 'pebby.spatial-confirmation-schedule.v1',
        'status': 'preregistered', 'seed_key': args.seed_key,
        'target_count': args.target_count, 'quotas': quotas,
        'candidate_extra': args.candidate_extra,
        'seed_namespace_half_open': [args.start_seed, args.end_seed],
        'model_behavior_used': False,
        'source_hashes': source_hashes,
        'code_hashes': code_hashes,
        'generator': 'pebby.ls20.reference_generator.generate_level',
        'attempts': args.attempts, 'search_limits': list(PROFILE_LIMITS),
        'schedule': schedule,
    })
    tool_path = Path(__file__).resolve()
    process_path = args.out_dir / 'processes.json'
    result_path = args.out_dir / 'results.jsonl'
    partial_path = args.out_dir / 'confirmation-bank.partial.jsonl'
    progress_path = args.out_dir / 'progress.json'
    started_local = datetime.now().astimezone().isoformat()
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    min_available = mem_available()
    if min_available < MIN_AVAILABLE:
        raise RuntimeError(f'available memory {min_available} below 6GiB reserve')

    by_tier = {tier: [item for item in schedule if item['tier'] == tier]
               for tier in range(1, 8)}
    cursors = {tier: 0 for tier in range(1, 8)}
    accepted = {tier: [] for tier in range(1, 8)}
    in_flight = {}
    dispatch_cursor = 1
    results_seen = 0
    rejected = Counter()
    processes = []
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    compiler_queue = ctx.Queue()
    status = 'partial'
    stop_reason = None
    run_error = None
    cleanup = []
    compiler_records = {}
    driver_pid = os.getpid()
    driver_starttime_ticks = process_start_ticks(driver_pid)
    worker_starttime_ticks = {}
    worker_started_local = datetime.now().astimezone().isoformat()

    for index in range(args.workers):
        process = ctx.Process(target=worker_main,
                              args=(task_queue, result_queue, compiler_queue,
                                    args.attempts),
                              name=f'spatial-confirmation-worker-{index + 1}',
                              daemon=True)
        process.start()
        processes.append(process)
        worker_starttime_ticks[process.pid] = process_start_ticks(process.pid)
    atomic_json(process_path, {
        'started_local': worker_started_local,
        'driver': {'pid': driver_pid, 'starttime_ticks': driver_starttime_ticks,
                   'owned_by_this_run': True},
        'workers': [{'pid': p.pid, 'name': p.name} for p in processes],
        'workers_owned_by_this_run': True,
    })

    def selected_rows():
        return [row for tier in range(1, 8) for row in accepted[tier]]

    def progress():
        return {
            'status': status, 'stop_reason': stop_reason,
            'target_count': args.target_count, 'quotas': quotas,
            'accepted_count': sum(len(rows) for rows in accepted.values()),
            'accepted_by_tier': {str(tier): len(accepted[tier])
                                 for tier in range(1, 8)},
            'candidate_cursors': {str(tier): cursors[tier]
                                  for tier in range(1, 8)},
            'results_seen': results_seen,
            'rejections': dict(rejected),
            'min_available_bytes': min_available,
            'elapsed_seconds': time.monotonic() - started,
            'source_hashes': source_hashes,
        }

    def drain_compiler_events(wait_seconds=0):
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                event = compiler_queue.get_nowait()
            except queue.Empty:
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.01)
                continue
            if event.get('event') == 'compiler_started':
                compiler_records[event['pid']] = event

    def fill_workers():
        nonlocal dispatch_cursor
        while len(in_flight) < args.workers:
            chosen = None
            for offset in range(7):
                tier = ((dispatch_cursor - 1 + offset) % 7) + 1
                if (tier not in in_flight and len(accepted[tier]) < quotas[tier - 1]
                        and cursors[tier] < len(by_tier[tier])):
                    chosen = tier
                    dispatch_cursor = (tier % 7) + 1
                    break
            if chosen is None:
                return
            task = by_tier[chosen][cursors[chosen]]
            cursors[chosen] += 1
            in_flight[chosen] = task
            task_queue.put(task)

    try:
        fill_workers()
        write_progress(progress_path, progress())
        while True:
            drain_compiler_events()
            min_available = min(min_available, mem_available())
            if min_available < MIN_AVAILABLE:
                stop_reason = 'minimum available-memory reserve reached'
                break
            if time.monotonic() >= deadline:
                stop_reason = 'deadline reached'
                break
            if all(len(accepted[tier]) >= quotas[tier - 1] for tier in range(1, 8)):
                status = 'complete' if args.target_count == 70 else 'smoke_complete'
                stop_reason = 'requested quotas reached'
                break
            if not in_flight:
                stop_reason = 'candidate schedule exhausted before quotas'
                break
            try:
                result = result_queue.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            except Exception as error:
                if not any(process.is_alive() for process in processes):
                    status = 'failed'
                    stop_reason = f'workers exited without result: {error}'
                    break
                continue
            drain_compiler_events()
            results_seen += 1
            tier = result['tier']
            in_flight.pop(tier, None)
            append_line(result_path, result)
            if result['status'] != 'accepted':
                rejected[result.get('error_type', 'generator_rejection')] += 1
            else:
                reason = validate_row(result['row'], tier, known)
                if reason:
                    rejected[reason] += 1
                else:
                    row = result['row']
                    for field in FINGERPRINTS:
                        known[field].add(row[field])
                    accepted[tier].append(row)
                    atomic_lines(partial_path, selected_rows())
            fill_workers()
            write_progress(progress_path, progress())
        write_progress(progress_path, progress())
    except BaseException as error:
        status = 'failed'
        stop_reason = f'{type(error).__name__}: {error}'
        run_error = dict(type=type(error).__name__, message=str(error))
        write_progress(progress_path, progress())
    finally:
        # A compiler event may be queued just before a worker exits. Drain
        # events before cleanup so an interrupted worker cannot hide its child.
        drain_compiler_events(0.5)
        # A normal completion gets sentinels; deadline/low-memory/failure gets
        # exact termination before join so no worker escapes the bound.
        if status in ('complete', 'smoke_complete'):
            for _ in processes:
                task_queue.put(None)
            for process in processes:
                process.join(timeout=10)
            if any(process.is_alive() or Path(f'/proc/{process.pid}').exists()
                   for process in processes):
                cleanup = terminate_workers(processes, worker_starttime_ticks)
                if any(item['alive'] or item['proc_entry_exists'] for item in cleanup):
                    status = 'failed'
                    stop_reason = 'worker cleanup could not prove all owned PIDs exited'
        else:
            cleanup = terminate_workers(processes, worker_starttime_ticks)
        if status in ('complete', 'smoke_complete'):
            cleanup = [cleanup_record(p, worker_starttime_ticks.get(p.pid))
                       for p in processes]
        if any(item['alive'] or item['proc_entry_exists'] for item in cleanup):
            status = 'failed'
            stop_reason = 'worker cleanup could not prove all owned PIDs exited'
        if status in ('complete', 'smoke_complete') and any(
                item['exitcode'] not in (0, None) for item in cleanup):
            status = 'failed'
            stop_reason = 'worker exited nonzero after producing a panel'
        drain_compiler_events(0.5)
        compiler_cleanup = terminate_compilers(compiler_records)
        if any(item['alive'] or item['proc_entry_exists'] for item in compiler_cleanup):
            status = 'failed'
            stop_reason = 'compiler cleanup could not prove all owned PIDs exited'
        atomic_json(process_path, {
            'started_local': worker_started_local,
            'finished_local': datetime.now().astimezone().isoformat(),
            'driver': {'pid': driver_pid, 'starttime_ticks': driver_starttime_ticks,
                       'owned_by_this_run': True,
                       'proc_entry_exists': process_entry_exists(driver_pid),
                       'current_starttime_ticks': process_start_ticks(driver_pid)},
            'workers': [{'pid': p.pid, 'name': p.name} for p in processes],
            'workers_owned_by_this_run': True,
            'cleanup': cleanup,
            'compiler_processes': compiler_cleanup,
            'all_exited': all(not item['alive'] and not item['proc_entry_exists']
                              for item in cleanup + compiler_cleanup),
        })
        task_queue.close(); result_queue.close(); compiler_queue.close()

    # Source integrity and selected-row counts are checked after cleanup.
    if any(digest(path) != source_hashes[str(path)] for path in source_paths):
        status = 'failed'
        stop_reason = 'source bank changed during generation'
    if any(digest(path) != code_hashes[str(path)] for path in code_paths):
        status = 'failed'
        stop_reason = 'generator, engine, profile, or provenance source changed during generation'
    final_rows = selected_rows()
    expected_complete = (args.target_count == 70
                         and len(final_rows) == 70
                         and all(len(accepted[tier]) == 10 for tier in range(1, 8)))
    if status == 'complete' and not expected_complete:
        status = 'failed'
        stop_reason = 'completion invariant failed: target must be exactly 70 with ten rows per tier'
    receipt = {
        'format': 'pebby.spatial-confirmation-generation.v1',
        'status': status, 'stop_reason': stop_reason,
        'started_local': started_local,
        'finished_local': datetime.now().astimezone().isoformat(),
        'deadline_seconds': args.deadline_seconds,
        'elapsed_seconds': time.monotonic() - started,
        'target_count': args.target_count, 'quotas': quotas,
        'accepted_count': len(final_rows),
        'accepted_by_tier': {str(tier): len(accepted[tier]) for tier in range(1, 8)},
        'candidate_results': results_seen, 'rejections': dict(rejected),
        'bank_path': str(partial_path), 'bank_sha256': digest(partial_path) if partial_path.exists() else None,
        'schedule_path': str(schedule_path), 'schedule_sha256': digest(schedule_path),
        'process_path': str(process_path), 'process_cleanup': json.loads(process_path.read_text()).get('cleanup'),
        'driver': {'pid': driver_pid, 'starttime_ticks': driver_starttime_ticks,
                   'owned_by_this_run': True,
                   'proc_entry_exists': process_entry_exists(driver_pid),
                   'current_starttime_ticks': process_start_ticks(driver_pid)},
        'compiler_processes': json.loads(process_path.read_text()).get('compiler_processes', []),
        'tool_path': str(tool_path), 'tool_sha256': digest(tool_path),
        'source_hashes': source_hashes,
        'code_hashes': code_hashes,
        'initial_selection_path': str(args.initial_selection),
        'initial_selection_sha256': initial_selection_sha256,
        'error': run_error,
        'official_layouts_or_routes_used': False,
        'aggregate_reference_calibration_used': True,
        'model_behavior_used_for_selection': False,
        'fingerprint_fields': list(FINGERPRINTS),
        'search_limits': list(PROFILE_LIMITS), 'attempts': args.attempts,
        'minimum_available_bytes': min_available,
    }
    receipt_path = args.out_dir / 'generation-receipt.json'
    atomic_json(receipt_path, receipt)

    selection = json.loads(args.selection_out.read_text())
    selection['status'] = ('complete_frozen_70' if status == 'complete'
                           else 'partial_smoke14_ready_final70_pending'
                           if status == 'smoke_complete' and len(final_rows) == 14
                           else 'generation_failed_partial')
    selection['updated_local'] = datetime.now().astimezone().isoformat()
    selection['decision'].update(
        confirmation_bank_created=(status == 'complete'),
        confirmation_bank_path=(str(ROOT / 'artifacts/spatial-improvement-v2/confirmation-bank.jsonl')
                                if status == 'complete' else None),
        can_treat_unused_existing_validation_as_confirmation=False,
        reason=('Fresh panel generated and frozen.' if status == 'complete'
                else 'Fresh 14-level smoke panel only; no 70-level confirmation claim.'
                if status == 'smoke_complete' else f'Generation failed or stopped: {stop_reason}'),
    )
    selection.setdefault('fresh_generation_plan', {})
    selection['fresh_generation_plan']['not_run_in_this_task'] = False
    selection['fresh_generation_plan']['scheduler'] = (
        'tools/build_spatial_confirmation.py (preregistered schedule; '
        'bounded spawn workers; generated-only reference API)')
    selection['fresh_generation_plan']['executed_source_snapshot'] = {
        'path': 'artifacts/spatial-improvement-v2/confirmation-generation/executed-source/'
                'build_spatial_confirmation.py',
        'historical_run': 'run-20260913-0920',
        'note': 'Exact pre-repair source is preserved separately; this run receipt hashes the current builder.',
    }
    selection['fresh_generation_plan']['run'] = receipt
    selection['fresh_generation_plan']['schedule'] = {
        'seed_key': args.seed_key,
        'seed_namespace_half_open': [args.start_seed, args.end_seed],
        'target_count': args.target_count, 'quotas': quotas,
        'candidate_extra': args.candidate_extra,
        'schedule_path': str(schedule_path), 'schedule_sha256': digest(schedule_path),
    }
    selection['fresh_generation_plan']['freshness_caveat'] = (
        'Generated rows are fingerprint-disjoint from TRAIN and original v1 validation. '
        'A 14-row smoke panel is not a substitute for the preregistered 70-row confirmation.'
        if status != 'complete' else
        'Generated rows are fingerprint-disjoint from TRAIN and original v1 validation; '
        'the panel is frozen before model evaluation.')
    if status == 'complete':
        final_path = ROOT / 'artifacts/spatial-improvement-v2/confirmation-bank.jsonl'
        if final_path.exists():
            raise RuntimeError(f'refusing overwrite of existing {final_path}')
        atomic_lines(final_path, final_rows)
        selection['decision']['confirmation_bank_sha256'] = digest(final_path)
        selection['decision']['confirmation_bank_levels'] = len(final_rows)
    else:
        selection['decision']['smoke_bank_path'] = str(partial_path)
        selection['decision']['smoke_bank_sha256'] = digest(partial_path) if partial_path.exists() else None
        selection['decision']['smoke_bank_levels'] = len(final_rows)
    atomic_json(args.selection_out, selection)
    return exit_code_for(status, args.target_count)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-count', type=int, default=70)
    parser.add_argument('--candidate-extra', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--attempts', type=int, default=400)
    parser.add_argument('--deadline-seconds', type=int, default=600)
    parser.add_argument('--seed-key', default='Pebby:spatial-improvement-v2:confirmation:v1')
    parser.add_argument('--start-seed', type=int, default=10_000_000)
    parser.add_argument('--end-seed', type=int, default=10_200_000)
    parser.add_argument('--out-dir', type=Path,
                        default=ROOT / 'artifacts/spatial-improvement-v2/confirmation-generation')
    parser.add_argument('--initial-selection', type=Path,
                        default=ROOT / 'artifacts/spatial-improvement-v2/confirmation-generation/initial.json')
    parser.add_argument('--selection-out', type=Path,
                        default=ROOT / 'artifacts/spatial-improvement-v2/confirmation-selection.json')
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
