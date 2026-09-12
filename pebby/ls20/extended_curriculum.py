"""Optional larger generated lessons, separate from the deterministic v1 curriculum.

This module never reads shipped layouts, frames or routes. Drafts use the shared
sprite builder and are accepted only after complete contextual fast-Oracle search,
real-engine WIN with three lives, and sampled reachable transition agreement.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import time

from . import names
from .generate import FORMAT, GENERATOR_VERSION, RESERVED, _connected, build_level
from .env import Ls20Scenario
from .generation_quality import budget_floor, geometry_partition, route_budget_slack
from .layout import extract
from .plan import Oracle, simulate

VERSION = 2
NAMESPACE = 'ls20-extended-curriculum'
SPLIT_STARTS = {'train': 20000, 'validation': 1020000}
KINDS = ('shape', 'color', 'rotation')
SIZES = (6, 4, 4)


class ContractMismatch(RuntimeError):
    """A planner/engine disagreement is an investigation blocker, never a rejected draft."""


def gameplay_hash(spec):
    # Identical canonicalization to tools.audit_world_contexts.gameplay_hash.
    gameplay = {'walls': sorted(spec['walls']), 'start': spec['start'],
                'start_triple': spec['start_triple'], 'goals': sorted(spec['goals'], key=lambda x: x['cell']),
                'cyclers': sorted(spec['cyclers'], key=lambda x: (x['cell'], x['kind'])),
                'rails': sorted([sorted(r['cells']) for r in spec.get('rails', [])]),
                'launchers': sorted(spec.get('launchers', []), key=lambda x: x['cell']),
                'refills': sorted(spec['refills']), 'step_counter': spec['step_counter'],
                'step_cost': spec['step_cost'], 'fog': spec['fog']}
    return hashlib.sha256(json.dumps(gameplay, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def draft(rng, difficulty, quality_profile='learning'):
    floor = budget_floor({'quality_profile': quality_profile})
    challenge = quality_profile == 'challenge'
    maximum = 7 if difficulty >= 3 and not challenge else 9
    width, height = rng.randint(6, maximum), rng.randint(6, maximum)
    left, top = rng.randint(1, 11 - width), rng.randint(1, 11 - height)
    free = {(left + x, top + y) for x in range(width) for y in range(height)} - RESERVED
    topology = rng.choice(('obstacles', 'corridors', 'corridors') if difficulty == 5 and not challenge
                          else ('room', 'obstacles', 'two_rooms', 'corridors'))
    if topology == 'two_rooms':
        door = top + rng.randrange(height)
        free -= {(left + width // 2, y) for y in range(top, top + height) if y != door}
    elif topology == 'corridors':
        # Alternating full horizontal lanes joined at opposite ends, plus random
        # vertical shortcuts. This creates longer routes without copying a maze.
        keep = set()
        for dy in range(0, height, 2):
            keep.update((left + x, top + dy) for x in range(width))
            if dy + 1 < height:
                keep.add((left + (width - 1 if dy % 4 == 0 else 0), top + dy + 1))
        candidates = sorted(free - keep)
        keep.update(rng.sample(candidates, min(3, len(candidates))))
        free &= keep
    elif topology == 'obstacles':
        for cell in rng.sample(sorted(free), min(len(free) // 5, 12)):
            remaining = free - {cell}
            if len(_connected(remaining, min(remaining))) == len(remaining):
                free = remaining
    if not free or len(_connected(free, min(free))) != len(free):
        return None
    count = 1 if difficulty == 1 else 2 if difficulty in (2, 3) else 3
    kinds = rng.sample(KINDS, count)
    moving = 3 if difficulty == 5 and challenge else 2 if difficulty >= 4 else 1 if difficulty == 3 else 0
    # Three attribute dimensions already expand the graph substantially. The
    # learning profile shares short periods; challenge allows longer walks.
    shared_length = (rng.randint(2, 6) if challenge else 2) if difficulty >= 3 else None
    rails, walked, cyclers = [], set(), []
    for index in range(moving):
        length = shared_length or rng.randint(2, 6)
        options = []
        for x, y in sorted(free - walked):
            for dx, dy in ((1, 0), (0, 1)):
                cells = {(x + i * dx, y + i * dy) for i in range(length)}
                if cells <= free and not cells & walked:
                    options.append(cells)
        if not options:
            return None
        cells = rng.choice(options)
        walked |= cells
        rails.append({'cells': sorted(cells)})
        cyclers.append({'cell': rng.choice(sorted(cells)), 'kind': kinds[index]})
    spots = sorted(free - walked)
    rng.shuffle(spots)
    if len(spots) < count + 12:
        return None
    start = spots.pop()
    for kind in kinds[moving:]:
        cyclers.append({'cell': spots.pop(), 'kind': kind})
    # Extra reachable cyclers are candidates for distractors. Verification counts
    # only those actually avoided by the accepted route, not the requested extras.
    decoy_kinds = [kind for kind in KINDS if kind not in kinds]
    # When all three kinds are required no never-required kind exists. Those
    # rows retain only route-verified unused alternatives, reported separately.
    for _ in range(rng.randint(1, 2)):
        cyclers.append({'cell': spots.pop(), 'kind': rng.choice(decoy_kinds or kinds)})
    refill_count = rng.randint(0, 6) if challenge else rng.randint(0, 2)
    refills = [spots.pop() for _ in range(refill_count)]
    initial = [rng.randrange(size) for size in SIZES]
    goals = []
    for _ in range(2 if difficulty == 5 else 1):
        triple = initial.copy()
        for kind in kinds:
            index = KINDS.index(kind)
            delta = rng.randint(1, 2) if difficulty >= 4 and not challenge else rng.randrange(1, SIZES[index])
            triple[index] = (triple[index] + delta) % SIZES[index]
        if goals and triple == goals[0]['triple']:
            index = KINDS.index(kinds[-1])
            choices = [value for value in range(SIZES[index])
                       if value not in (initial[index], triple[index])]
            triple[index] = rng.choice(choices)
        goals.append({'cell': spots.pop(), 'triple': triple})
    launchers = []
    launcher_count = rng.randint(0, 8) if challenge else rng.randint(0, 1) if difficulty >= 2 else 0
    for _ in range(launcher_count):
        options = []
        goal_cells = {tuple(goal['cell']) for goal in goals}
        for cell in spots:
            for dx, dy in names.ACTION_DELTAS:
                if (cell[0] - dx, cell[1] - dy) in free:
                    continue
                probe, reach = cell, 0
                while True:
                    probe = (probe[0] + dx, probe[1] + dy)
                    if probe not in free or probe in goal_cells:
                        break
                    reach += 1
                if reach >= 2:
                    options.append((cell, (dx, dy)))
        if options:
            cell, delta = rng.choice(options)
            launchers.append({'cell': cell, 'delta': list(delta)})
            spots.remove(cell)
        else:
            break
    # Match the transfer game's full HUD and supported costs. Profile-specific
    # replay checks enforce either eight moves of reserve or a nonnegative tank.
    cost = rng.choice((1, 1, 2)) if not challenge else rng.choice((1, 2))
    budget = 42
    return {'format': FORMAT, 'generator_version': GENERATOR_VERSION,
            'extended_curriculum_version': VERSION, 'generation_namespace': NAMESPACE,
            'quality_profile': quality_profile, 'budget_floor': floor,
            'difficulty': difficulty, 'size': 64, 'topology': topology,
            'room_width': width, 'room_height': height, 'free_cells': len(free),
            'walls': sorted({(x, y) for x in range(12) for y in range(12)} - free),
            'start': start, 'start_triple': initial, 'cyclers': cyclers, 'rails': rails,
            'rail_mode': 'independent_straight' if moving else 'none',
            'goals': goals, 'refills': sorted(refills), 'launchers': launchers,
            'step_counter': budget, 'step_cost': cost, 'fog': difficulty >= 4 and bool(rng.randrange(2))}


def verify(spec, search_limit=600000, transition_samples=8):
    """Return (accepted spec, exclusion); fail loudly on actual mechanics mismatch."""
    from ..agent.world_data import clone_env
    if any(tuple(goal['triple']) == tuple(spec['start_triple']) for goal in spec['goals']):
        return None, 'goal_satisfied_at_spawn'
    floor = budget_floor(spec)
    context = spec['seed'] % 7
    if context == 0 and spec.get('launchers'):
        return None, 'context_zero_launcher_pending_hint'
    fingerprint, partition = geometry_partition(spec)
    split = 'train' if spec['seed'] < 1000000 else 'validation'
    if partition != split:
        return None, 'geometry_split_mismatch'
    env = Ls20Scenario(build_level(spec), context)
    try:
        layout = extract(env)
    except ValueError as error:
        return None, f'inexact_layout:{error}'
    oracle = Oracle(layout, limit=search_limit, engine='fast')
    if oracle.truncated:
        return None, 'search_truncated'
    if not oracle.solvable:
        return None, 'unsolvable'
    solution = oracle.solution()
    if not solution or len(solution) < 10:
        return None, 'route_under_10_actions'
    expected_moving = len(spec.get('rails', []))
    if len(layout.patrollers) != expected_moving:
        raise ContractMismatch(f"seed {spec['seed']}: expected {expected_moving} patrollers, got {len(layout.patrollers)}")
    # Sample random actions at random reachable route states (not only teacher actions).
    rng = random.Random(f"{NAMESPACE}:transitions:{VERSION}:{spec['seed']}")
    checks_at = Counter(rng.randrange(len(solution)) for _ in range(transition_samples))
    checked, moving_used, refill_used = 0, set(), False
    replay = clone_env(env)
    state = oracle.start
    minimum_slack = state[6] // layout.step_cost
    static_contacts = set()
    moving_contacts, launcher_contacts = set(), set()
    launcher_used = False
    result = None
    for step, action in enumerate(solution):
        for _ in range(checks_at[step]):
            chosen = rng.randrange(4)
            branch = clone_env(replay)
            predicted, outcome = simulate(layout, state, chosen, oracle.refills)
            observed = branch.perform(names.ACTION_IDS[chosen])
            if outcome == 'won':
                agrees = observed.won and branch.lives() == 3
            elif outcome == 'died':
                agrees = not observed.finished and branch.lives() == 2 and oracle.state_of(branch) == oracle.start
            else:
                agrees = not observed.finished and branch.lives() == 3 and oracle.state_of(branch) == predicted
            if not agrees:
                raise ContractMismatch(f"seed {spec['seed']} context {context} route_step {step} action {chosen}: "
                                       f"outcome {outcome}, predicted {predicted}, actual {oracle.state_of(branch)}")
            checked += 1
        before = state
        dx, dy = names.ACTION_DELTAS[names.ACTION_IDS.index(action)]
        target = (before[0][0] + dx, before[0][1] + dy)
        if target in layout.cyclers:
            static_contacts.add(target)
        state, outcome = simulate(layout, state, names.ACTION_IDS.index(action), oracle.refills)
        if before[1:4] != state[1:4] and state[0] in layout.moving_cyclers[state[7]]:
            moving_used.add(layout.moving_cyclers[state[7]][state[0]])
            moving_contacts.update(i for i, rider in enumerate(layout.patrollers)
                                   if state[0] in rider['cells'])
        if outcome == 'launched':
            trigger = target if layout.free(target) else before[0]
            launcher_contacts.update(i for i, pad in enumerate(layout.launchers)
                                     if trigger in pad['triggers'])
        if outcome == 'launched' and state[0] in layout.cyclers:
            static_contacts.add(state[0])
        launcher_used |= outcome == 'launched'
        minimum_slack = min(minimum_slack, route_budget_slack(
            layout, before, state, action=names.ACTION_IDS.index(action), outcome=outcome))
        refill_used |= state[5] != before[5]
        result = replay.perform(action)
        if outcome != 'won' and (result.finished or oracle.state_of(replay) != state):
            raise ContractMismatch(f"seed {spec['seed']}: optimal replay state mismatch at step {step}")
    if result is None or not result.won or replay.lives() != 3 or replay.levels_completed != 1:
        raise ContractMismatch(f"seed {spec['seed']}: complete Oracle solution failed real engine WIN")
    if expected_moving and len(moving_used) != expected_moving:
        return None, 'not_all_moving_types_used'
    slack = state[6] // layout.step_cost
    if slack < floor:
        return None, 'under_eight_final_slack_moves' if floor == 8 else 'negative_final_slack_moves'
    if minimum_slack < floor:
        return None, 'under_eight_route_slack_moves' if floor == 8 else 'negative_route_slack_moves'
    distractors = sorted(set(layout.cyclers) - static_contacts)
    if not distractors:
        return None, 'no_unused_distractor_cycler'
    required_kinds = {KINDS[i] for i in range(3)
                      if any(goal['triple'][i] != spec['start_triple'][i] for goal in spec['goals'])}
    non_required = [cell for cell in distractors if layout.cyclers[cell] not in required_kinds]
    if len(required_kinds) < 3 and not non_required:
        return None, 'no_unused_non_required_kind_distractor'
    return {**spec, 'quality_profile': spec.get('quality_profile', 'learning'), 'budget_floor': floor,
            'solution': solution, 'optimal_actions': len(solution), 'slack_moves': slack,
            'geometry_sha256': fingerprint, 'geometry_split': partition,
            'minimum_slack_moves': minimum_slack, 'distractor_count': len(distractors),
            'distractor_cells': distractors, 'non_required_distractor_count': len(non_required),
            'changing_attributes': len(required_kinds),
            'reachable_states': oracle._reachable, 'search_truncated': False, 'search_limit': search_limit,
            'training_context_index': context, 'context_engine_verified': True,
            'context_optimal_actions': len(solution), 'engine_verified': True, 'verification_lives': 3,
            'verification_level_index': context, 'verification_match_hint': context == 0,
            'oracle_backend': oracle.engine, 'patroller_count': len(layout.patrollers),
            'tick_span': layout.tick_span, 'random_transitions_checked': checked,
            'solution_mechanics': {'moving_types': sorted(moving_used), 'refill': refill_used,
                                   'launcher': launcher_used, 'used_patroller_count': len(moving_contacts),
                                   'used_launcher_count': len(launcher_contacts)},
            'gameplay_sha256': gameplay_hash(spec)}, None


def generate_level(seed, difficulty=1, attempts=16, search_limit=600000, quality_profile='learning'):
    budget_floor({'quality_profile': quality_profile})
    if difficulty not in range(1, 6) or attempts < 1 or not 0 < search_limit <= 600000:
        raise ValueError('difficulty 1..5, positive attempts and search_limit <=600000 required')
    if not 20000 <= seed < 1000000 and not 1020000 <= seed < 2000000:
        raise ValueError('seed outside extended train/validation namespace')
    rng = random.Random(f'{NAMESPACE}:{VERSION}:{seed}:{difficulty}:{quality_profile}')
    excluded = Counter()
    for attempt in range(1, attempts + 1):
        # Geometry resampling is cheap and has its own bound; it does not
        # consume the sixteen complete-Oracle verification opportunities.
        split = 'train' if seed < 1000000 else 'validation'
        for _ in range(16):
            spec = draft(rng, difficulty, quality_profile)
            if spec is None:
                excluded['invalid_draft'] += 1
                continue
            if geometry_partition(spec)[1] != split:
                excluded['geometry_split_mismatch'] += 1
                continue
            break
        else:
            continue
        spec.update(seed=seed, generation_attempt=attempt)
        accepted, reason = verify(spec, search_limit)
        if accepted:
            accepted['generation_exclusions'] = dict(excluded)
            return accepted, dict(excluded)
        excluded[reason] += 1
    return None, dict(excluded)


def summarize(rows):
    def distribution(key):
        values = sorted(row[key] for row in rows)
        return {'min': values[0], 'median': values[len(values) // 2], 'max': values[-1],
                'mean': sum(values) / len(values)} if values else None
    return {'levels': len(rows), 'difficulty': dict(Counter(s['difficulty'] for s in rows)),
            'topology': dict(Counter(s['topology'] for s in rows)),
            'unique_geometries': len({s['geometry_sha256'] for s in rows}),
            'geometry_splits': dict(Counter(s['geometry_split'] for s in rows)),
            'quality_profiles': dict(Counter(s['quality_profile'] for s in rows)),
            'changing_attributes': dict(Counter(s['changing_attributes'] for s in rows)),
            'non_required_distractors': distribution('non_required_distractor_count'),
            'rail_lengths': dict(Counter(len(rail['cells']) for s in rows for rail in s['rails'])),
            'used_patrollers': dict(Counter(s['solution_mechanics']['used_patroller_count'] for s in rows)),
            'used_launchers': dict(Counter(s['solution_mechanics']['used_launcher_count'] for s in rows)),
            'patrollers': dict(Counter(s['patroller_count'] for s in rows)),
            'goals': dict(Counter(len(s['goals']) for s in rows)),
            'refills': dict(Counter(len(s['refills']) for s in rows)),
            'costs': dict(Counter(s['step_cost'] for s in rows)),
            'budgets': dict(Counter(s['step_counter'] for s in rows)),
            'launchers': dict(Counter(len(s['launchers']) for s in rows)),
            'distractors': distribution('distractor_count'),
            'minimum_slack_moves': distribution('minimum_slack_moves'),
            'slack_moves': distribution('slack_moves'),
            'contexts': dict(Counter(s['training_context_index'] for s in rows)),
            'optimal_actions': distribution('optimal_actions'), 'free_cells': distribution('free_cells'),
            'reachable_states': distribution('reachable_states'),
            'used_launcher_levels': sum(s['solution_mechanics']['launcher'] for s in rows),
            'used_refill_levels': sum(s['solution_mechanics']['refill'] for s in rows),
            'random_transitions_checked': sum(s['random_transitions_checked'] for s in rows),
            'seeds': [s['seed'] for s in rows]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-profile', choices=('learning', 'challenge'), default='learning')
    parser.add_argument('--train-count', type=int, default=100)
    parser.add_argument('--validation-count', type=int, default=20)
    parser.add_argument('--attempts', type=int, default=16)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--search-limit', type=int, default=600000)
    parser.add_argument('--out-dir', type=Path, default=Path('data'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/world-extended-curriculum-pilot.json'))
    parser.add_argument('--existing-banks', nargs='+', default=['data/ls20-verified-train.jsonl',
                        'data/ls20-verified-validation.jsonl'])
    args = parser.parse_args(argv)
    if min(args.train_count, args.validation_count, args.attempts, args.seconds) < 1:
        parser.error('counts, attempts and seconds must be positive')
    started = time.monotonic()
    report = {'status': 'running', 'pid': os.getpid(), 'workers': 1, 'device': 'cpu',
              'namespace': NAMESPACE, 'version': VERSION, 'quality_profile': args.quality_profile, 'official_gameplay_inputs_used': False,
              'search_limit': args.search_limit, 'attempts_per_seed': args.attempts,
              'requested': {'train': args.train_count, 'validation': args.validation_count},
              'split_summaries': {}, 'exclusions': {}, 'failed_seeds': [], 'existing_banks': {},
              'limits': ['Pilot only, not added to active training.',
                         'Supported wall-mounted launchers; context-zero launcher combination is explicitly excluded.',
                         'Challenge proposes three distinct moving kinds and shared rail lengths2..6; learning uses at most two patrollers on length2 rails.',
                         'Learning d3+ uses compact6..7 rooms; d5 favors obstacles/corridors and d4/5 use one-or-two-step required attribute offsets to bound complete search.',
                         'Counter42/cost1|2; learning reserve8, challenge reserve0. These profiles are separate and do not guarantee strictly ordered difficulty medians.',
                         'Challenge proposes refill counts0..6 and launcher counts0..8; installed counts and route-used counts are reported separately.',
                         'Three-required-kind levels cannot have a never-required-kind distractor; their extras are verified unused alternatives.',
                         'New v2 train/validation geometry is translation-normalized and disjoint; this is not a topology-family or historical-v1 holdout.',
                         'Complete-search cap can bias acceptance toward easier states even when rooms are larger.']}
    hashes, occupied = set(), set()
    for path in args.existing_banks:
        contents = Path(path).read_bytes()
        report['existing_banks'][path] = hashlib.sha256(contents).hexdigest()
        for line in contents.splitlines():
            row = json.loads(line)
            hashes.add(gameplay_hash(row))
            occupied.add(row['seed'])
    baseline_hash_count = len(hashes)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank_rows = {'train': [], 'validation': []}
    exclusions = Counter()
    attempted_seeds, attempted_drafts = 0, 0
    def persist():
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      attempted_seeds=attempted_seeds, attempted_drafts=attempted_drafts,
                      accepted_levels=sum(map(len, bank_rows.values())), exclusions=dict(exclusions))
        accepted = report['accepted_levels']
        report['acceptance_per_seed'] = accepted / max(1, attempted_seeds)
        report['acceptance_per_draft'] = accepted / max(1, attempted_drafts)
        report['levels_per_second'] = accepted / max(.001, report['elapsed_seconds'])
        report['dedup'] = {'existing_unique_gameplay_hashes': baseline_hash_count,
                          'accepted_unique_gameplay_hashes': len(hashes) - baseline_hash_count,
                          'overlap_accepted_vs_existing': 0}
        for split, rows in bank_rows.items():
            report['split_summaries'][split] = summarize(rows)
        temp = args.report.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.report)
    def timeout(*_):
        raise TimeoutError('pilot time cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    print('PID', os.getpid(), flush=True)
    persist()
    try:
        for split, count in (('train', args.train_count), ('validation', args.validation_count)):
            path = args.out_dir / f'ls20-extended-pilot-{split}.jsonl'
            if path.exists():
                raise ValueError(f'refusing to overwrite existing pilot bank {path}')
            # Failed seed attempts are explicit and bounded; accepted difficulty
            # quotas remain balanced by selecting the next missing stage.
            with path.open('x') as stream:
                for offset in range(count * 8):
                    if len(bank_rows[split]) >= count:
                        break
                    seed = SPLIT_STARTS[split] + offset
                    if seed in occupied:
                        raise ValueError(f'seed {seed} overlaps an existing bank')
                    difficulty = len(bank_rows[split]) % 5 + 1
                    attempted_seeds += 1
                    accepted, reasons = generate_level(seed, difficulty, args.attempts, args.search_limit, args.quality_profile)
                    exclusions.update(reasons)
                    attempted_drafts += sum(reasons.values()) + int(accepted is not None)
                    if accepted is None:
                        report['failed_seeds'].append({'seed': seed, 'difficulty': difficulty, 'exclusions': reasons})
                        persist()
                        continue
                    fingerprint = accepted['gameplay_sha256']
                    if fingerprint in hashes:
                        exclusions['duplicate_gameplay'] += 1
                        persist()
                        continue
                    hashes.add(fingerprint)
                    occupied.add(seed)
                    bank_rows[split].append(accepted)
                    stream.write(json.dumps(accepted, separators=(',', ':')) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(split, len(bank_rows[split]), '/', count, 'seed', seed, 'd', difficulty,
                          'moves', accepted['optimal_actions'], 'free', accepted['free_cells'],
                          'moving', accepted['patroller_count'], flush=True)
                    persist()
            if len(bank_rows[split]) < count:
                raise RuntimeError(f'{split}: bounded candidate count exhausted')
        report['status'] = 'complete'
    except TimeoutError:
        report['status'] = 'incomplete_time_cap'
    except Exception as error:
        report['status'] = 'blocked_contract_mismatch' if isinstance(error, ContractMismatch) else 'failed'
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0)
        persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
