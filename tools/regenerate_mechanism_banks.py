"""Generate matched banks with bounded workers and durable, resumable job checkpoints.

The default seven-tier profile uses official aggregate references for calibration;
all output layouts are procedural. Legacy v2 generation is an explicit profile.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import hashlib
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import time

# Also support direct execution from an unrelated working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pebby.ls20.extended_curriculum import ContractMismatch, gameplay_hash
from pebby.ls20.generate import FORMAT
from pebby.ls20.provenance import generated_context
from tools import generate_mechanism_pilot as pilot

PROFILES = ('ls20-reference-v1', 'legacy-mechanism-v2')
CHECKPOINT_FORMAT = 'pebby.bank-regeneration.checkpoint.v1'
_SEEDS, _GAMEPLAY = frozenset(), frozenset()
GIB=1024**3
# Fast-search allocation bounds plus interpreter/allocator margin. Collection
# adds its larger PyTorch worker baseline through memory_overhead_bytes.
# Tier 6/7 workers share the validated 1.3125/1.75 GiB working reservations respectively.
# The global 6 GiB reserve remains separate from per-worker reservations.
TIER_MEMORY_GIB=(.3125,.3125,.375,.5,1.,1.3125,1.75)
MEMORY_RESERVE=6*GIB
MAX_HEAVY_WORKERS=8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(_key(value).encode()).hexdigest()


def _write(path, value):
    """Durable atomic replacement: checkpoint first, progress report second."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as out:
        out.write(_key(value) + '\n')
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _initialize(seeds, gameplay):
    global _SEEDS, _GAMEPLAY
    _SEEDS, _GAMEPLAY = frozenset(seeds), frozenset(gameplay)


def _generate(job):
    """One isolated seed attempt; failures return evidence without losing siblings."""
    rejections = []
    seed = job['next_seed']
    def reject(reason):
        rejections.append(reason)
        text = str(reason.get('reason', '')).lower()
        if any(term in text for term in ('state mismatch', 'replay failed', 'contract mismatch', 'contract failure')):
            raise ContractMismatch(text)
    try:
        if seed in _SEEDS:
            raise RuntimeError('occupied seed in worker inventory')
        if job['profile'] == PROFILES[0]:
            from pebby.ls20.reference_generator import generate_level
            row = generate_level(seed, job['mode'], attempts=job['attempts'],
                                 search_limit=job['limit'], split=job['split'],
                                 record_rejection=reject)
        else:
            row, _ = pilot.generate_one(seed, job['mode'], attempts=job['attempts'],
                                        limit=job['limit'] or 600000,
                                        seen_seeds=set(_SEEDS), seen_specs=set(_GAMEPLAY),
                                        namespace_end=job['seed'] + 64, seed_attempts=1,
                                        record_rejection=reject)
            row.update(curriculum_version=f"mechanism-{job['split']}-v{pilot.VERSION}")
        row.update(split=job['split'], source='generated_only')
        if gameplay_hash(row) in _GAMEPLAY:
            raise RuntimeError('gameplay duplicate in worker inventory')
        return dict(status='accepted', row=row, rejections=rejections, pid=os.getpid())
    except Exception as error:
        contract = isinstance(error, ContractMismatch) or 'engine contract failure' in str(error)
        # Unexpected exceptions are quarantined as well; only ordinary bounded
        # generator rejection RuntimeErrors qualify for an automatic seed retry.
        status = 'quarantined' if contract or not isinstance(error, RuntimeError) else 'rejected'
        return dict(status=status, error=str(error), error_type=type(error).__name__,
                    rejections=rejections, pid=os.getpid())


def jobs_for(count, split, occupied, *, attempts=400, limit=None, profile=PROFILES[0],
             quotas=None, seed_range=None):
    """Stable tier assignments and disjoint64-seed retry blocks; old defaults unchanged."""
    if profile not in PROFILES:
        raise ValueError(f'unknown profile: {profile}')
    start, end = (720000, 900000) if split == 'train' else (8100000, 9000000)
    if seed_range is not None:
        start,end=seed_range
        if type(start) is not int or type(end) is not int or start<0 or end<=start:
            raise ValueError('seed range requires0<=start<exclusive end')
    modes = tuple(range(1, 8)) if profile == PROFILES[0] else pilot.MODES
    assignments=None
    if quotas is not None:
        if (profile!=PROFILES[0] or len(quotas)!=7 or any(type(n) is not int or n<1 for n in quotas)
                or sum(quotas)!=count):
            raise ValueError('seven positive reference-tier quotas must sum to count')
        remaining=list(quotas);assignments=[]
        while any(remaining):
            for i,left in enumerate(remaining):
                if left:
                    assignments.append(i+1);remaining[i]-=1
    jobs = []
    while len(jobs) < count:
        if start + 64 > end:
            raise ValueError(f'{split} seed namespace cannot hold requested quotas')
        if not any(seed in occupied for seed in range(start, start + 64)):
            jobs.append(dict(id=f'{split}-{len(jobs):06d}', split=split, seed=start,
                             mode=(assignments[len(jobs)] if assignments is not None else modes[len(jobs) % len(modes)]), attempts=attempts,
                             limit=limit, profile=profile))
        start += 64
    return jobs


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Linux MemAvailable required for reference scheduler')


def job_memory(job):
    overhead=job.get('memory_overhead_bytes',0)
    if type(overhead) is not int or overhead<0:raise ValueError('nonnegative worker memory overhead required')
    base=int(TIER_MEMORY_GIB[job['mode']-1]*GIB) if job['profile']==PROFILES[0] else int(.35*GIB)
    return base+overhead


def active_reservation(active_jobs, directory):
    """Reserve only estimated allocation still outstanding beyond observed RSS.

    Workers publish their exact PID/start time per active job. Missing or stale
    markers reserve the full estimate; observed RSS is already in MemAvailable.
    """
    remaining=0
    for job in active_jobs:
        estimate=job_memory(job);rss=0
        path=directory/(job['id']+'.json')
        try:
            record=json.loads(path.read_text());stat=Path(f'/proc/{record["pid"]}/stat').read_text()
            fields=stat[stat.rfind(')')+2:].split()
            if int(fields[19])==record['start_ticks']:
                rss=int(Path(f'/proc/{record["pid"]}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError,ProcessLookupError,KeyError,ValueError):pass
        remaining+=max(0,estimate-rss)
    return remaining


def _admitted_generate(job, marker):
    stat=Path('/proc/self/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    _write(Path(marker),dict(pid=os.getpid(),start_ticks=int(fields[19]),job=job['id']))
    try:return _generate(job)
    finally:Path(marker).unlink(missing_ok=True)


def take_pending(pending, active_jobs, dispatch, *, available=None, reserved=0):
    """Memory-aware fair dispatch: <=8 tier6/7, <=4 tier5; fill other CPU slots.

    Reservations change admission only, never candidates, proof budgets or seeds.
    Fair mode guarantees a heavy lane by draining cheap work when none is
    active. With one heavy running, cheap backfill remains possible instead of
    waiting for two heavy reservations that may never fit together.
    FIFO remains available; the dynamic memory guard applies to both schedules.
    """
    if dispatch not in ('fifo','fair'):raise ValueError('unknown dispatch policy')
    active=list(active_jobs)
    heavy=sum(j['profile']==PROFILES[0] and j['mode'] in (6,7) for j in active)
    medium=sum(j['profile']==PROFILES[0] and j['mode']==5 for j in active)
    if dispatch=='fair' and heavy<MAX_HEAVY_WORKERS:
        oldest=next((i for i,j in enumerate(pending) if j['profile']==PROFILES[0] and j['mode'] in (6,7)),None)
        if oldest is not None:
            if available is None or available-reserved-job_memory(pending[oldest])>=MEMORY_RESERVE:
                return pending.pop(oldest)
            if heavy==0:
                # Do not continuously refill cheap jobs and starve the heavy
                # queue. Existing jobs finish; no proof or budget is shortened.
                return None
            # If the oldest tier already has a running worker, its progress is
            # protected. A smaller heavy job may use the other lane without
            # starving that tier. In particular,6 can fit beside7 when two7s
            # cannot; keep draining if only6 runs and an older7 is waiting.
            if any(j['profile']==PROFILES[0] and j['mode']==pending[oldest]['mode'] for j in active):
                for index,job in enumerate(pending):
                    if (job['profile']==PROFILES[0] and job['mode'] in (6,7)
                            and available-reserved-job_memory(job)>=MEMORY_RESERVE):
                        return pending.pop(index)
    for index,job in enumerate(pending):
        reference=job['profile']==PROFILES[0]
        # Without an already-running worker of its tier, an oldest heavy must
        # not be bypassed by a cheaper younger heavy.
        if dispatch=='fair' and reference and (job['mode'] in (6,7) or (job['mode']==5 and medium>=4)):continue
        if available is not None and reference and available-reserved-job_memory(job)<MEMORY_RESERVE:continue
        return pending.pop(index)
    return None


def _inventory(out_dir):
    seeds, fingerprints, sources = set(), set(), {}
    for path in sorted((REPO_ROOT / 'data').rglob('*.jsonl')):
        if path.is_symlink() or path.is_relative_to(out_dir):
            continue
        before, used = digest(path), False
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get('format') == FORMAT:
                    seeds.add(int(row['seed']))
                    fingerprints.add(gameplay_hash(row))
                    used = True
        if digest(path) != before:
            raise RuntimeError(f'inventory changed: {path}')
        if used:
            sources[str(path)] = before
    return seeds, fingerprints, sources


def _code_hashes(profile):
    paths = [Path(__file__).resolve(), REPO_ROOT / 'tools/generate_mechanism_pilot.py',
             REPO_ROOT / 'pebby/agent/world_data.py']
    paths += sorted((REPO_ROOT / 'pebby/ls20').glob('*.py'))
    paths.append(REPO_ROOT / 'pebby/ls20/_fastplan.c')
    if profile == PROFILES[0] and not (REPO_ROOT / 'pebby/ls20/reference_generator.py').is_file():
        raise RuntimeError('reference generator is not installed; use explicit legacy profile if intended')
    return {str(path): digest(path) for path in paths}


def _checkpoint(path, job, binding, state):
    payload = dict(format=CHECKPOINT_FORMAT, binding=binding, job=job, state=state)
    _write(path, dict(payload=payload, sha256=_hash(payload)))


def _restore(path, job, binding):
    if not path.exists():
        return dict(status='pending', next_seed=job['seed'], failures=[])
    envelope = json.loads(path.read_text())
    payload = envelope['payload']
    if envelope['sha256'] != _hash(payload):
        raise ValueError(f'checkpoint checksum mismatch: {path}')
    if (payload['format'] != CHECKPOINT_FORMAT or payload['job'] != job
            or payload['binding'] != binding):
        raise ValueError(f'checkpoint source/config binding mismatch: {path}')
    state = payload['state']
    if not job['seed'] <= state['next_seed'] <= job['seed'] + 64:
        raise ValueError(f'checkpoint seed outside reserved block: {path}')
    return state


def _geometry(row):
    if row.get('difficulty_version'):
        from pebby.ls20.generation_quality import geometry_d4_partition
        fingerprint, split = geometry_d4_partition(row)
    else:
        from pebby.ls20.generation_quality import geometry_partition
        fingerprint, split = geometry_partition(row)
    if row.get('geometry_sha256') != fingerprint or row.get('geometry_split') != split or row['split'] != split:
        raise ValueError('generated row has invalid geometry partition proof')
    return fingerprint


def _accept(row, job, seeds, fingerprints, geometries):
    if (row.get('format') != FORMAT or row.get('engine_verified') is not True
            or row.get('context_engine_verified') is not True
            or row.get('search_truncated') is not False
            or not row.get('solution')
            or row.get('training_context_index') != generated_context(row)
            or row.get('verification_level_index') != generated_context(row)
            or row.get('optimal_actions') != len(row.get('solution', []))
            or row.get('context_optimal_actions') != row.get('optimal_actions')):
        raise ValueError('generated row lacks a complete engine-verified contextual route proof')
    if not job['seed'] <= row['seed'] < job['seed'] + 64 or row['split'] != job['split']:
        raise ValueError('generated row escaped assigned job seed/split')
    if job['profile'] == PROFILES[0]:
        if row.get('difficulty_version') != PROFILES[0] or row['difficulty'] != job['mode']:
            raise ValueError('generated row disagrees with reference difficulty assignment')
        from pebby.ls20.reference_profiles import profile_errors
        errors = profile_errors(row)
        proof = row.get('proof')
        if not isinstance(proof, dict):
            errors.append('missing nested contextual proof')
        else:
            paired = {'seed': 'seed', 'difficulty': 'difficulty',
                      'difficulty_version': 'difficulty_version',
                      'context_index': 'training_context_index',
                      'context_engine_verified': 'context_engine_verified',
                      'search_truncated': 'search_truncated',
                      'optimal_actions': 'optimal_actions',
                      'context_optimal_actions': 'context_optimal_actions',
                      'engine_win': 'engine_win', 'replay_lives': 'replay_lives',
                      'levels_completed': 'levels_completed', 'search_limit': 'search_limit',
                      'reachable_states': 'reachable_states', 'oracle_backend': 'oracle_backend'}
            errors += [f'nested proof disagreement: {key}' for key, top in paired.items()
                       if key not in proof or type(proof[key]) is not type(row.get(top))
                       or proof[key] != row.get(top)]
            if (proof.get('engine_win') is not True or type(proof.get('replay_lives')) is not int
                    or proof['replay_lives'] != 3 or type(proof.get('levels_completed')) is not int
                    or proof['levels_completed'] != 1):
                errors.append('nested proof does not certify one win with three lives')
        if errors:
            raise ValueError('; '.join(errors))
    elif row.get('pilot_mode') != job['mode']:
        raise ValueError('generated row disagrees with legacy mode assignment')
    fingerprint = gameplay_hash(row)
    if row['seed'] in seeds or fingerprint in fingerprints:
        return 'duplicate_seed_or_gameplay'
    geometry = _geometry(row)
    opposite = 'validation' if row['split'] == 'train' else 'train'
    if geometry in geometries[opposite]:
        return 'cross_split_geometry_duplicate'
    seeds.add(row['seed'])
    fingerprints.add(fingerprint)
    geometries[row['split']].add(geometry)
    return None


def shuffled_rows(rows, split, profile):
    values = sorted(rows, key=lambda row: row['seed'])
    random.Random(f'bank-final-order:{profile}:{split}:v1').shuffle(values)
    return values


def _coverage(rows):
    return {field: dict(Counter(str(row.get(field)) for row in rows)) for field in
            ('difficulty', 'training_context_index', 'changing_attributes', 'pilot_mode')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default=PROFILES[0])
    parser.add_argument('--train-count', type=int, default=None, help='multiples of seven give equal tier quotas')
    parser.add_argument('--validation-count', type=int, default=None, help='multiples of seven give equal tier quotas')
    parser.add_argument('--train-quotas', type=int, nargs=7, metavar='N', help='positive counts for tiers1..7; overrides default total')
    parser.add_argument('--validation-quotas', type=int, nargs=7, metavar='N', help='positive counts for tiers1..7; overrides default total')
    parser.add_argument('--train-seed-range', type=int, nargs=2, metavar=('START','END'), help='half-open namespace; default720000..900000')
    parser.add_argument('--validation-seed-range', type=int, nargs=2, metavar=('START','END'), help='half-open namespace; default8100000..9000000')
    parser.add_argument('--dispatch', choices=('fifo','fair'), default='fifo', help='fair: at most8 tier6/7 and4 tier5 jobs; memory-aware admission')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--attempts', type=int, default=400)
    parser.add_argument('--limit', type=int, default=None, help='omit for per-tier policy; max 32000000 for reference, 600000 legacy')
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--resume', action='store_true', help='resume checkpoints with identical config, code and inventory')
    args = parser.parse_args(argv)
    for split,default in (('train',196),('validation',56)):
        quotas=getattr(args,split+'_quotas');count=getattr(args,split+'_count')
        if quotas is not None:
            if args.profile!=PROFILES[0] or min(quotas)<1:
                parser.error('explicit quotas require seven positive reference-tier counts')
            if count is not None and count!=sum(quotas):parser.error('explicit count differs from sum of quotas')
            count=sum(quotas)
        setattr(args,split+'_count',default if count is None else count)
    ranges={split:getattr(args,split+'_seed_range') or default for split,default in
            (('train',(720000,900000)),('validation',(8100000,9000000)))}
    if any(start<0 or end<=start for start,end in ranges.values()):parser.error('invalid half-open seed namespace')
    if max(ranges['train'][0],ranges['validation'][0])<min(ranges['train'][1],ranges['validation'][1]):
        parser.error('TRAIN and validation seed namespaces overlap')
    maximum = 32000000 if args.profile == PROFILES[0] else 600000
    max_workers = 16 if args.profile == PROFILES[0] else 4
    if (min(args.train_count, args.validation_count, args.attempts, args.seconds) < 1
            or not 1 <= args.workers <= max_workers or (args.limit is not None and not 1 <= args.limit <= maximum)):
        parser.error(f'positive counts/time, 1..{max_workers} workers and at most {maximum} states required')
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / ('data/ls20-reference-v1' if args.profile == PROFILES[0] else 'data/mechanism-bank-v2')
    args.out_dir = args.out_dir.resolve()
    if args.resume:
        if not args.out_dir.is_dir():
            parser.error('--resume requires an existing output directory')
        if (not (args.out_dir / 'manifest.json').is_file()
                and (any((args.out_dir / 'jobs').glob('*.json')) or any(args.out_dir.glob('*.jsonl')))):
            parser.error('checkpoints or finalized banks exist without their manifest; reuse refused')
    else:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = args.out_dir / 'jobs'
    checkpoint_dir.mkdir(exist_ok=True)
    lock = (args.out_dir / '.regeneration.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        parser.error('another regeneration process owns this output directory')
    report_path = args.out_dir / 'generation-report.json'
    started = time.monotonic()
    report = dict(status='running', format='pebby.mechanism-banks.v3', profile=args.profile,
                  pid=os.getpid(), worker_pids=[], workers_stopped=True,
                  requested={'train': args.train_count, 'validation': args.validation_count},
                  accepted={'train': 0, 'validation': 0}, errors=[],
                  official_inputs_used=args.profile == PROFILES[0],
                  official_input_scope='aggregate calibration only; no official layouts in outputs',
                  limitations=pilot.LIMITATIONS if args.profile == PROFILES[1] else [
                      'Difficulty is calibrated structure/mechanics/action length, not measured controller performance.',
                      'Complete search limits censor drafts; rejected and quarantined jobs remain resumable.'])
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        _write(report_path, report)
    def stop(*_):
        raise TimeoutError('bounded regeneration interrupted or deadline reached')
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.alarm(args.seconds)
    pool, processes, states = None, [], {}
    try:
        seeds, fingerprints, sources = _inventory(args.out_dir)
        config = dict(profile=args.profile, train_count=args.train_count,
                      validation_count=args.validation_count, attempts=args.attempts, limit=args.limit)
        # Omit new defaults to preserve the historical config/job schema. Code
        # fingerprints still fail closed when resuming a different source version.
        for key in ('train_quotas','validation_quotas','train_seed_range','validation_seed_range'):
            if getattr(args,key) is not None:config[key]=getattr(args,key)
        if args.dispatch!='fifo':config['dispatch']=args.dispatch
        manifest = dict(config=config, code_hashes=_code_hashes(args.profile), source_hashes=sources)
        manifest_path = args.out_dir / 'manifest.json'
        if args.resume and manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old != manifest:
                raise ValueError('resume configuration, source code, or inventory changed; checkpoint reuse refused')
        else:
            _write(manifest_path, manifest)
        binding = _hash(manifest)
        report.update(manifest_sha256=binding, **manifest)
        jobs = [job for split, count in report['requested'].items() for job in
                jobs_for(count, split, seeds, attempts=args.attempts, limit=args.limit, profile=args.profile,
                         quotas=getattr(args,split+'_quotas'), seed_range=getattr(args,split+'_seed_range'))]
        jobs.sort(key=lambda job: (int(job['id'].rsplit('-', 1)[-1]), job['split']))
        rows = {'train': [], 'validation': []}
        geometries = {'train': set(), 'validation': set()}
        states = {}
        for job in jobs:
            state = _restore(checkpoint_dir / (job['id'] + '.json'), job, binding)
            states[job['id']] = state
            if state['status'] == 'accepted':
                reason = _accept(state['row'], job, seeds, fingerprints, geometries)
                if reason:
                    raise ValueError(f'duplicate checkpoint: {job["id"]}: {reason}')
                rows[job['split']].append(state['row'])
            elif state['status'] == 'quarantined':
                report['errors'].append(dict(job=job['id'], reason='quarantined contract/worker failure',
                                             detail=state.get('error')))
        report['accepted'] = {split: len(values) for split, values in rows.items()}
        persist()
        active_dir=args.out_dir/'active-workers'
        active_dir.mkdir(exist_ok=True)
        report['scheduler']=dict(max_tier6_7=MAX_HEAVY_WORKERS,max_tier5=4,reserve_bytes=MEMORY_RESERVE,
            tier_estimated_peak_bytes=[int(v*GIB) for v in TIER_MEMORY_GIB],
            limitation='Estimated reservations plus current MemAvailable, not an OS hard memory guarantee.')
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_initialize, initargs=(seeds, fingerprints))
        report['workers_stopped']=False
        futures = {}
        pending = []
        def enqueue(job):
            state = states[job['id']]
            if state['status'] in ('accepted', 'quarantined') or state['next_seed'] >= job['seed'] + 64:
                return
            pending.append(job)
        def fill_workers():
            while len(futures)<args.workers:
                available=memory_available() if args.profile==PROFILES[0] else None
                reserved=active_reservation(futures.values(),active_dir)
                report['memory_admission']=dict(available_bytes=available,outstanding_reserved_bytes=reserved)
                job=take_pending(pending,futures.values(),args.dispatch,available=available,reserved=reserved)
                if job is None:break
                state=states[job['id']]
                futures[pool.submit(_admitted_generate,{**job,'next_seed':state['next_seed']},str(active_dir/(job['id']+'.json')))]=job
            report['worker_pids']=[p.pid for p in pool._processes.values()]
        for job in jobs:
            enqueue(job)
        fill_workers()
        processes = list(pool._processes.values())
        report['worker_pids'] = [process.pid for process in processes]
        persist()
        while futures or pending:
            if not futures:
                persist();time.sleep(1);fill_workers();continue
            # Recreate this iterator after each result so a duplicate can append
            # a retry without losing completed siblings or waiting on job order.
            ready,_ = wait(tuple(futures),timeout=1,return_when=FIRST_COMPLETED)
            if not ready:
                # Reconsider admission when external memory is released; a
                # long search must not keep a newly affordable slot idle.
                fill_workers()
                report['active_jobs']=[j['id'] for j in futures.values()]
                persist()
                continue
            future = next(as_completed(tuple(futures)))
            job = futures.pop(future)
            state = states[job['id']]
            try:
                result = future.result()
            except Exception as error:
                result = dict(status='quarantined', error=str(error), error_type=type(error).__name__, rejections=[])
            failure = dict(seed=state['next_seed'], **{k: v for k, v in result.items() if k != 'row'})
            state['failures'].append(failure)
            state['next_seed'] += 1
            if result['status'] == 'accepted':
                row = result['row']
                state['next_seed'] = max(state['next_seed'], row['seed'] + 1)
                try:
                    reason = _accept(row, job, seeds, fingerprints, geometries)
                except Exception as error:
                    reason = None
                    result = dict(status='quarantined', error=str(error), error_type=type(error).__name__)
                if result['status'] == 'accepted' and reason:
                    result = dict(status='rejected', error=reason)
                if result['status'] == 'accepted':
                    state.update(status='accepted', row=row)
                    # Store the complete verified row before reporting progress.
                    _checkpoint(checkpoint_dir / (job['id'] + '.json'), job, binding, state)
                    rows[job['split']].append(row)
                    report['accepted'][job['split']] += 1
            if result['status'] != 'accepted':
                state.update(status=result['status'], error=result['error'])
                failure.update(status=result['status'], error=result['error'])
                _checkpoint(checkpoint_dir / (job['id'] + '.json'), job, binding, state)
                if result['status'] == 'quarantined':
                    report['errors'].append(dict(job=job['id'], **result))
                else:
                    enqueue(job)
            fill_workers()
            report['active_jobs']=[j['id'] for j in futures.values()]
            persist()
        pool.shutdown(wait=True)
        pool = None
        # Re-enumerate qualifying generated banks and relevant source files:
        # rehashing only the initial list would miss concurrently added inputs.
        if (_inventory(args.out_dir)[2] != manifest['source_hashes']
                or _code_hashes(args.profile) != manifest['code_hashes']):
            raise RuntimeError('source or inventory changed during generation')
        if report['accepted'] != report['requested'] or report['errors']:
            raise RuntimeError('incomplete quotas or quarantined failures; successes remain checkpointed')
        report['coverage'] = {split: _coverage(values) for split, values in rows.items()}
        report['rejection_counts'] = dict(Counter(refusal.get('reason', 'unknown') for state in states.values()
                                                for failure in state['failures'] for refusal in failure.get('rejections', [])))
        report['banks'] = {}
        for split, values in rows.items():
            path = args.out_dir / f'{split}.jsonl'
            content = ''.join(_key(row) + '\n' for row in shuffled_rows(values, split, args.profile))
            if path.exists() and path.read_text() != content:
                raise ValueError(f'refuse to replace different finalized bank: {path}')
            pending = path.with_suffix('.pending')
            with pending.open('w') as out:
                out.write(content)
                out.flush()
                os.fsync(out.fileno())
            pending.replace(path)
            report['banks'][split] = dict(path=str(path), sha256=digest(path), levels=len(values))
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed_closed', error=str(error))
    finally:
        signal.alarm(0)
        if pool is not None:
            # Include children started before an interrupted submission loop.
            processes = list({p.pid: p for p in processes + list((pool._processes or {}).values())}.values())
            report['worker_pids'] = [process.pid for process in processes]
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
            pool.shutdown(wait=True, cancel_futures=True)
        report['workers_stopped'] = all(not process.is_alive() for process in processes)
        report['job_statuses'] = dict(Counter(state['status'] for state in states.values()))
        report['rejection_counts'] = dict(Counter(refusal.get('reason', 'unknown') for state in states.values()
                                                for failure in state['failures'] for refusal in failure.get('rejections', [])))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        persist()
        lock.close()
        print(json.dumps({key: report[key] for key in ('status', 'accepted', 'elapsed_seconds', 'workers_stopped')}), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
