"""Build a tier-balanced generator-v4 bank with a forced-goal-order quota for tier 6.

The default training bank (data/ls20-reference-unequal-v1) is 3 percent tier 6
and 1 percent tier 7. This tool builds train/validation/test banks with an
explicit per-tier quota, using generator version 4 (goal relations, vanishing
rings, three-way geometry holdout) and disjoint seed ranges per split.

Tier 6 draws its two goal cells independently, so most levels can be cleared in
either order, unlike shipped level 6 where the first goal sits in a dead end
behind the second. The optional acceptance filter ``--forced-order-fraction F``
keeps generating tier-6 seeds until at least ``ceil(F * quota)`` retained levels
carry an exact ordering proof: the planner's complete search never reaches a
winning state through the "other goal first" mask. The proof reuses the search
the generator already ran to certify the level, so it costs one vectorized scan
of the packed state keys rather than a second exact search.

Seeds: ``split base + (tier - 1) * 100000 + index``; bases follow
``pebby.ls20.reference_generator_v2.seed_split`` (0 / 1M / 2M). Results are
appended to ``progress/<split>.jsonl`` after every seed, so an interrupted run
resumes with the same seeds and the same deterministic final selection.

Example::

    uv run python tools/build_balanced_bank.py --out-dir data/ls20-balanced-v4 \
        --train-per-tier 700 --validation-per-tier 100 --test-per-tier 100 \
        --workers 12 --generator-version 4 --forced-order-fraction 0.3
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pebby.ls20 import plan  # noqa: E402
from pebby.ls20.generate import build_level, generate_level  # noqa: E402
from pebby.ls20.reference_generator_v2 import (  # noqa: E402
    GEOMETRY_VERSION, MECHANICS_VERSION, SPLITS, seed_split)
from pebby.ls20.reference_profiles import DIFFICULTIES, PROFILES  # noqa: E402

FORMAT = 'pebby.balanced-bank-build.v1'
PROGRESS_FORMAT = 'pebby.balanced-bank-progress.v1'
SPLIT_BASE = {'train': 0, 'validation': 1_000_000, 'test': 2_000_000}
TIER_STRIDE = 100_000          # seeds reserved per tier inside a split's range
GIB = 2 ** 30
GENERATOR_FILES = ('pebby/ls20/generate.py', 'pebby/ls20/reference_generator.py',
                   'pebby/ls20/reference_generator_v2.py', 'pebby/ls20/reference_profiles.py',
                   'pebby/ls20/plan.py', 'pebby/ls20/fastplan.py', 'pebby/ls20/_fastplan.c',
                   'pebby/ls20/layout.py', 'pebby/ls20/generation_quality.py')


# -- seeds ---------------------------------------------------------------------------------------------

def tier_seed(split, tier, index):
    """The ``index``-th seed reserved for ``tier`` inside ``split``'s declared range."""
    if split not in SPLIT_BASE or tier not in DIFFICULTIES or not 0 <= index < TIER_STRIDE:
        raise ValueError('split, tier 1..7 and index below the tier stride required')
    seed = SPLIT_BASE[split] + (tier - 1) * TIER_STRIDE + index
    assert seed_split(seed) == split
    return seed


# -- forced goal order --------------------------------------------------------------------------------

def winning_goal_masks(oracle):
    """Goal masks of every reachable state from which the level can still be won.

    Both search backends keep exactly those states in ``_distance``: the
    reference search's backward BFS from the win states, and the C kernel's
    finite-distance entries. The packed backend is scanned as a numpy view of
    the key array, which costs well under a second for 32M states.
    """
    distance = oracle._distance
    tables = getattr(distance, '_tables', None)
    keys = getattr(distance, '_keys', None)
    if tables is not None and keys is not None:
        count = len(distance)
        if count == 0:
            return set()
        import numpy as np
        with distance._lock:
            array = np.ctypeslib.as_array(keys, shape=(count,))
            masks = np.unique((array >> np.uint64(tables.shift_goals)) & np.uint64(tables.mask_goals))
        return {int(mask) for mask in masks}
    return {state[4] for state in distance}


def goal_order_analysis(oracle):
    """Ordering proof for a two-goal level from a complete, untruncated search.

    ``forced_goal_order`` is True when exactly one single-goal mask appears on
    any winning route, i.e. no winning route clears the other goal first.
    Single-goal levels get ``None``; the masks are recorded for every level.
    """
    if oracle.truncated:
        raise ValueError('goal order needs a complete search')
    goals = [list(cell) for cell, _ in oracle.layout.goals]
    masks = winning_goal_masks(oracle)
    analysis = dict(goal_count=len(goals), winning_goal_masks=sorted(masks),
                    forced_goal_order=None, forced_first_goal=None)
    if len(goals) == 2:
        first_possible = [index for index in range(2) if (1 << index) in masks]
        analysis['forced_goal_order'] = len(first_possible) == 1
        if len(first_possible) == 1:
            analysis['forced_first_goal'] = goals[first_possible[0]]
    return analysis


def analyze_spec_goal_order(spec, *, context_index=None, search_limit=None, engine='auto'):
    """Run the exact planner on ``spec`` and return :func:`goal_order_analysis`."""
    from pebby.ls20.env import Ls20Scenario
    from pebby.ls20.layout import extract
    context = spec.get('training_context_index', spec['difficulty'] - 1) if context_index is None else context_index
    limit = PROFILES[spec['difficulty']]['search_limit'] if search_limit is None else search_limit
    oracle = plan.Oracle(extract(Ls20Scenario(build_level(spec), context)), limit=limit, engine=engine)
    if oracle.truncated or not oracle.solvable:
        raise ValueError('spec is not completely solvable within the search limit')
    try:
        return goal_order_analysis(oracle)
    finally:
        _release(oracle)


def _release(oracle):
    distance = getattr(oracle, '_distance', None)
    if hasattr(distance, 'close'):
        distance.close()


# -- worker --------------------------------------------------------------------------------------------

_LAST_ORACLE = None


class _RecordingOracle(plan.Oracle):
    """The generator's own verify search, kept so the ordering proof is free."""

    def __init__(self, *args, **kwargs):
        global _LAST_ORACLE
        _LAST_ORACLE = None
        super().__init__(*args, **kwargs)
        _LAST_ORACLE = self


def _install_recorder():
    # reference_generator.verify imports Oracle from pebby.ls20.plan at call time.
    plan.Oracle = _RecordingOracle


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def _wait_for_memory(reserve, timeout):
    waited = 0.0
    while memory_available() < reserve:
        if waited >= timeout:
            return False, waited
        time.sleep(1.0)
        waited += 1.0
    return True, waited


def _matches(oracle, row):
    return (oracle is not None and not oracle.truncated
            and oracle._reachable == row['reachable_states']
            and oracle.optimal_actions == row['optimal_actions']
            and {tuple(cell) for cell, _ in oracle.layout.goals} == {tuple(g['cell']) for g in row['goals']})


def generate_one(job):
    """One seed: generate, engine-verify, then prove the goal order if two goals."""
    global _LAST_ORACLE
    split, tier, seed, attempts, reserve, wait_timeout = job
    started = time.perf_counter()
    ok, waited = _wait_for_memory(reserve, wait_timeout)
    result = dict(split=split, tier=tier, seed=seed, wait_seconds=waited, rejections={})
    if not ok:
        result.update(status='retry', reason='host memory reserve exhausted', seconds=time.perf_counter() - started)
        return result
    rejections = Counter()
    _LAST_ORACLE = None
    try:
        row = generate_level(seed, tier, attempts=attempts, generator_version=4, split=split,
                             record_rejection=lambda item: rejections.update([item['reason']]))
    except RuntimeError as error:
        result.update(status='no_level', reason=str(error)[:300])
    except Exception as error:  # noqa: BLE001 - one bad seed must not kill the build
        result.update(status='error', reason=f'{type(error).__name__}: {str(error)[:300]}')
    else:
        oracle = _LAST_ORACLE
        _LAST_ORACLE = None
        reproved = False
        if len(row['goals']) == 2:
            if not _matches(oracle, row):
                _release(oracle)
                from pebby.ls20.env import Ls20Scenario
                from pebby.ls20.layout import extract
                oracle = plan.Oracle(extract(Ls20Scenario(build_level(row), row['training_context_index'])),
                                     limit=row['search_limit'], engine='fast')
                reproved = True
            row.update(goal_order_analysis(oracle))
        else:
            row.update(goal_count=len(row['goals']), forced_goal_order=None, forced_first_goal=None)
        _release(oracle)
        row['goal_order_reproved'] = reproved
        result.update(status='accepted', row=row, forced_goal_order=row['forced_goal_order'],
                      reproved=reproved)
    finally:
        # A seed that exhausted its attempts leaves the last rejected search behind; free it now
        # rather than holding up to a gigabyte idle until this worker's next job.
        if _LAST_ORACLE is not None:
            _release(_LAST_ORACLE)
            _LAST_ORACLE = None
    result['rejections'] = dict(rejections)
    result['seconds'] = time.perf_counter() - started
    return result


# -- selection -----------------------------------------------------------------------------------------

def forced_needed(tier, quota, fraction):
    return math.ceil(fraction * quota) if PROFILES[tier]['goals'] == 2 and quota else 0


def select_rows(rows, quota, need_forced):
    """Deterministic retained subset: seed order, then swap in forced-order rows as needed."""
    rows = sorted(rows, key=lambda r: r['seed'])
    base = rows[:quota]
    if sum(bool(r.get('forced_goal_order')) for r in base) >= need_forced:
        return base
    forced = [r for r in rows if r.get('forced_goal_order')]
    other = [r for r in rows if not r.get('forced_goal_order')]
    take_forced = min(len(forced), max(need_forced, quota - len(other)))
    take_other = min(len(other), quota - take_forced)
    return sorted(forced[:take_forced] + other[:take_other], key=lambda r: r['seed'])


def satisfied(rows, quota, need_forced):
    selected = select_rows(rows, quota, need_forced)
    return len(selected) == quota and sum(bool(r.get('forced_goal_order')) for r in selected) >= need_forced


# -- build state ---------------------------------------------------------------------------------------

class TierState:
    def __init__(self, split, tier, quota, need_forced):
        self.split, self.tier, self.quota, self.need_forced = split, tier, quota, need_forced
        self.rows = []
        self.tried = self.no_level = self.errors = self.forced = 0
        self.seconds = self.wait_seconds = 0.0
        self.next_index = 0
        self.retry = []
        self.inflight = 0
        self.rejections = Counter()

    def record(self, result):
        self.tried += 1
        self.seconds += result['seconds']
        self.wait_seconds += result.get('wait_seconds', 0.0)
        self.rejections.update(result.get('rejections', {}))
        status = result['status']
        if status == 'accepted':
            self.rows.append(result['row'])
            self.forced += bool(result['row'].get('forced_goal_order'))
        elif status == 'no_level':
            self.no_level += 1
            self.rejections['no_level_after_attempts'] += 1
        else:
            self.errors += 1
            self.rejections['worker_error'] += 1

    @property
    def done(self):
        return satisfied(self.rows, self.quota, self.need_forced)

    def seeds_wanted(self):
        """How many more seeds to have in flight, from the observed rates (with a prior)."""
        if self.done:
            return 0
        accepted = len(self.rows)
        acceptance = (accepted + 1) / (self.tried + 2)
        forced_rate = (self.forced + 1) / (accepted + 2)
        missing_total = max(0, self.quota - accepted)
        missing_forced = max(0, self.need_forced - self.forced)
        wanted = max(missing_total / acceptance, missing_forced / (acceptance * forced_rate))
        return max(1, math.ceil(wanted)) - self.inflight

    def next_seed(self):
        if self.retry:
            return self.retry.pop(0)
        seed = tier_seed(self.split, self.tier, self.next_index)
        self.next_index += 1
        return seed

    def summary(self):
        selected = select_rows(self.rows, self.quota, self.need_forced)
        forced_selected = sum(bool(r.get('forced_goal_order')) for r in selected)
        accepted = len(self.rows)
        return dict(quota=self.quota, seeds_tried=self.tried, accepted=accepted, no_level=self.no_level,
                    errors=self.errors, retained=len(selected), forced_order_required=self.need_forced,
                    forced_order_generated=self.forced, forced_order_retained=forced_selected,
                    forced_order_fraction_generated=self.forced / accepted if accepted else None,
                    forced_order_fraction_retained=forced_selected / len(selected) if selected else None,
                    goal_order_reproved=sum(bool(r.get('goal_order_reproved')) for r in self.rows),
                    acceptance_rate=accepted / self.tried if self.tried else None,
                    seconds_total=self.seconds, memory_wait_seconds=self.wait_seconds,
                    seconds_per_seed=self.seconds / self.tried if self.tried else None,
                    seconds_per_accepted=self.seconds / accepted if accepted else None,
                    satisfied=self.done, next_seed_index=self.next_index,
                    rejection_counts=dict(self.rejections))


def load_progress(directory, states):
    """Replay progress logs; seeds already tried are never dispatched again."""
    loaded = 0
    for split in SPLITS:
        path = directory / 'progress' / f'{split}.jsonl'
        if not path.is_file():
            continue
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                result = json.loads(line)
                key = (result['split'], result['tier'])
                if key not in states:
                    continue
                state = states[key]
                state.record(result)
                index = result['seed'] - SPLIT_BASE[split] - (result['tier'] - 1) * TIER_STRIDE
                state.next_index = max(state.next_index, index + 1)
                loaded += 1
    return loaded


def _append_progress(directory, result):
    path = directory / 'progress' / f'{result["split"]}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps({'format': PROGRESS_FORMAT, **result}, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def sha256_of(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_banks(directory, states, tiers):
    banks = {}
    for split in SPLITS:
        rows = []
        for tier in tiers:
            state = states[(split, tier)]
            rows.extend(select_rows(state.rows, state.quota, state.need_forced))
        rows.sort(key=lambda r: (r['difficulty'], r['seed']))
        path = directory / f'{split}.jsonl'
        tmp = path.with_suffix('.jsonl.tmp')
        with tmp.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
        tmp.replace(path)
        banks[split] = dict(path=str(path), sha256=sha256_of(path), levels=len(rows))
    return banks


def run_audit(directory, quotas, tiers, spotcheck, *, generation_report):
    report = directory / 'audit-report.json'
    if report.exists():
        report.unlink()
    command = [sys.executable, str(ROOT / 'tools/audit_generated_banks.py'),
               '--train', str(directory / 'train.jsonl'), '--validation', str(directory / 'validation.jsonl'),
               '--test', str(directory / 'test.jsonl'), '--report', str(report),
               '--min-validation', str(quotas['validation'] * len(tiers)),
               '--min-test', str(max(1, quotas['test'] * len(tiers))),
               '--coverage-policy', 'full' if set(tiers) == set(DIFFICULTIES) else 'sample',
               '--spotcheck', str(spotcheck)]
    if generation_report is not None:
        command += ['--generation-report', str(generation_report)]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    outcome = dict(command=command, returncode=done.returncode, stdout=done.stdout[-2000:], stderr=done.stderr[-2000:])
    if report.is_file():
        audit = json.loads(report.read_text())
        outcome.update(status=audit['status'], errors=audit['errors'][:20], pairwise_overlap=audit['pairwise_overlap'],
                       report=str(report))
    return outcome


# -- main ----------------------------------------------------------------------------------------------

def build(args):
    directory = Path(args.out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    quotas = dict(train=args.train_per_tier, validation=args.validation_per_tier, test=args.test_per_tier)
    tiers = tuple(sorted(set(args.tiers)))
    states = {(split, tier): TierState(split, tier, quotas[split], forced_needed(tier, quotas[split], args.forced_order_fraction))
              for split in SPLITS for tier in tiers}
    loaded = load_progress(directory, states)
    print(f'resumed {loaded} prior seed results' if loaded else 'fresh build', flush=True)
    started = time.perf_counter()
    reserve = int(args.reserve_gib * GIB)
    context = multiprocessing.get_context('fork')
    pending = {}
    stopped_for_time = False
    interrupted = False
    pool = context.Pool(processes=args.workers, initializer=_install_recorder, maxtasksperchild=args.tasks_per_child)
    try:
        while True:
            for handle in list(pending):
                if not handle.ready():
                    continue
                job = pending.pop(handle)
                state = states[(job[0], job[1])]
                state.inflight -= 1
                result = handle.get()
                if result['status'] == 'retry':
                    state.retry.append(result['seed'])
                    print(f'  retry {job[0]} tier {job[1]} seed {result["seed"]}: {result["reason"]}', flush=True)
                    continue
                state.record(result)
                _append_progress(directory, result)
                mark = {'accepted': 'ok', 'no_level': 'none', 'error': 'ERR'}[result['status']]
                forced = result.get('forced_goal_order')
                extra = '' if forced is None else (' forced-order' if forced else ' free-order')
                print(f'  {mark:4} {job[0]:10} tier {job[1]} seed {result["seed"]:8} {result["seconds"]:7.1f}s{extra}'
                      f'  [{len(state.rows)}/{state.quota}{"" if not state.need_forced else f", forced {state.forced}/{state.need_forced}"}]',
                      flush=True)
            elapsed = time.perf_counter() - started
            if args.seconds is not None and elapsed > args.seconds and not stopped_for_time:
                stopped_for_time = True
                print(f'time limit {args.seconds}s reached; draining {len(pending)} in-flight seeds', flush=True)
            if not stopped_for_time:
                # Slow tiers first so the tail is short; never overshoot the estimated need.
                for key in sorted(states, key=lambda k: (-k[1], k[0])):
                    state = states[key]
                    while len(pending) < args.workers and state.seeds_wanted() > 0:
                        seed = state.next_seed()
                        job = (state.split, state.tier, seed, args.attempts, reserve, args.memory_wait_seconds)
                        pending[pool.apply_async(generate_one, (job,))] = job
                        state.inflight += 1
                    if len(pending) >= args.workers:
                        break
            if not pending:
                break
            if stopped_for_time and time.perf_counter() - started > args.seconds + args.drain_seconds:
                print(f'drain limit reached; terminating {len(pending)} seeds (they rerun on resume)', flush=True)
                pool.terminate()
                pending.clear()
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        interrupted = True
        pool.terminate()
    else:
        pool.close()
    finally:
        pool.join()
    elapsed = time.perf_counter() - started
    complete = all(state.done for state in states.values())
    banks = write_banks(directory, states, tiers)
    per_tier = {split: {str(tier): states[(split, tier)].summary() for tier in tiers} for split in SPLITS}
    timing = {}
    rejections = Counter()
    for tier in tiers:
        group = [states[(split, tier)] for split in SPLITS]
        tried = sum(s.tried for s in group)
        accepted = sum(len(s.rows) for s in group)
        seconds = sum(s.seconds for s in group)
        timing[str(tier)] = dict(seeds_tried=tried, accepted=accepted, seconds_total=seconds,
                                 seconds_per_seed=seconds / tried if tried else None,
                                 seconds_per_accepted=seconds / accepted if accepted else None,
                                 acceptance_rate=accepted / tried if tried else None,
                                 forced_order_generated=sum(s.forced for s in group),
                                 forced_order_fraction_generated=sum(s.forced for s in group) / accepted if accepted else None)
        for state in group:
            rejections.update(state.rejections)
            rejections['forced_order_quota'] += len(state.rows) - len(select_rows(state.rows, state.quota, state.need_forced))
    report = dict(
        format=FORMAT, status='complete' if complete else ('interrupted' if interrupted else 'partial'),
        generator_version=args.generator_version, mechanics_version=MECHANICS_VERSION, geometry_version=GEOMETRY_VERSION,
        created=datetime.now(timezone.utc).isoformat(), command=sys.argv, workers=args.workers, attempts=args.attempts,
        forced_order_fraction=args.forced_order_fraction, tiers=list(tiers), quotas=quotas,
        seed_scheme=dict(split_base=SPLIT_BASE, tier_stride=TIER_STRIDE, rule='base + (tier-1)*stride + index'),
        elapsed_seconds=elapsed, stopped_for_time_limit=stopped_for_time, time_limit_seconds=args.seconds,
        banks=banks, per_tier=per_tier, timing_per_tier=timing,
        forced_order_count={split: sum(states[(split, t)].summary()['forced_order_retained'] for t in tiers) for split in SPLITS},
        rejection_counts={reason: int(n) for reason, n in rejections.items()},
        code_hashes={name: sha256_of(ROOT / name) for name in GENERATOR_FILES},
        audit=None)
    report_path = directory / 'generation-report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{report["status"]}: {elapsed:.0f}s, banks {json.dumps({k: v["levels"] for k, v in banks.items()})}', flush=True)
    audit = None
    if not args.skip_audit:
        audit = run_audit(directory, quotas, tiers, args.audit_spotcheck,
                          generation_report=report_path if complete else None)
        print(f'audit: {audit.get("status", "not written")} overlap={audit.get("pairwise_overlap")}', flush=True)
        for error in audit.get('errors', []):
            print(f'  audit error: {error}', flush=True)
        report['audit'] = {k: v for k, v in audit.items() if k not in ('stdout', 'stderr')}
        # The audit hashed the report it was given; keep that copy and record the outcome beside it.
        (directory / 'build-summary.json').write_text(json.dumps(
            dict(status=report['status'], elapsed_seconds=elapsed, banks=banks, audit=report['audit'],
                 forced_order_count=report['forced_order_count']), indent=2) + '\n')
    return report, audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--train-per-tier', type=int, required=True)
    parser.add_argument('--validation-per-tier', type=int, required=True)
    parser.add_argument('--test-per-tier', type=int, required=True)
    parser.add_argument('--tiers', type=int, nargs='+', default=list(DIFFICULTIES))
    parser.add_argument('--workers', type=int, default=max(1, min(12, os.cpu_count() - 4)))
    parser.add_argument('--generator-version', type=int, choices=(4,), default=4,
                        help='only version 4 has a test geometry partition and goal relations')
    parser.add_argument('--forced-order-fraction', type=float, default=0.3,
                        help='minimum fraction of retained two-goal (tier 6) levels with a proved goal order')
    parser.add_argument('--seconds', type=float, default=None, help='stop dispatching new seeds after this wall clock')
    parser.add_argument('--drain-seconds', type=float, default=900.0,
                        help='after the time limit, wait this long for in-flight seeds before terminating them')
    parser.add_argument('--attempts', type=int, default=400, help='draft attempts per seed (generator default)')
    parser.add_argument('--reserve-gib', type=float, default=6.0, help='host MemAvailable a worker waits for before searching')
    parser.add_argument('--memory-wait-seconds', type=float, default=600.0)
    parser.add_argument('--tasks-per-child', type=int, default=8)
    parser.add_argument('--audit-spotcheck', type=int, default=0, help='audit re-search/replay levels per split (0..20)')
    parser.add_argument('--skip-audit', action='store_true')
    args = parser.parse_args(argv)
    if min(args.train_per_tier, args.validation_per_tier, args.test_per_tier) < 1:
        parser.error('every split needs at least one level per tier')
    if any(t not in DIFFICULTIES for t in args.tiers):
        parser.error('tiers must be drawn from 1..7')
    if not 0.0 <= args.forced_order_fraction <= 1.0:
        parser.error('--forced-order-fraction must be in 0..1')
    if args.workers < 1 or args.attempts < 1 or args.tasks_per_child < 1 or not 0 <= args.audit_spotcheck <= 20:
        parser.error('positive workers/attempts/tasks-per-child and spotcheck 0..20 required')
    return args


def main(argv=None):
    args = parse_args(argv)
    report, audit = build(args)
    if report['status'] != 'complete':
        return 2
    return 0 if audit is None or audit.get('status') == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
