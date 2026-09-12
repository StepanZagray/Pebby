"""Bounded, context-verified mechanism and composition training generation."""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import signal
import time

from pebby.agent.world_data import verified_context
from pebby.ls20 import names
from pebby.ls20.generate import FORMAT, GENERATOR_VERSION, RESERVED, _connected
from pebby.ls20.generation_quality import budget_floor, geometry_partition, route_budget_slack
from pebby.ls20.plan import simulate

VERSION = 2
MODES = ('one_attribute', 'two_attributes', 'three_attributes', 'three_goals',
         'four_goals', 'long_rail', 'three_rails', 'three_launchers', 'mixed_mechanisms', 'launcher_network', 'long_route', 'refill_chain')
FIELDS = ('size','walls','start','start_triple','goals','cyclers','rails','launchers','refills','step_counter','step_cost','fog')
KINDS = ('shape', 'color', 'rotation')
SIZES = (6, 4, 4)
MIN_SLACK = 8
MIN_ACTIONS = 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(spec):
    from pebby.ls20.extended_curriculum import gameplay_hash
    return gameplay_hash(spec)


def _long_route(rng, *, challenge=False):
    """Random tree mazes: route length grows without a combinatorial clock.

    Random spanning trees, boundary branches and lattice symmetries vary the
    navigation problem. Starts, locked goals, multi-press cyclers and refills
    are sampled on the resulting tree; the exact verifier decides acceptance.
    """
    nodes = {(x, y) for x in range(1, 10, 2) for y in range(1, 10, 2)}
    root = rng.choice(sorted(nodes))
    free, seen, stack = {root}, {root}, [root]
    while stack:
        x, y = stack[-1]
        options = [(x+2*dx,y+2*dy) for dx,dy in names.ACTION_DELTAS
                   if (x+2*dx,y+2*dy) in nodes - seen]
        if not options:
            stack.pop()
            continue
        cell = rng.choice(options)
        free.update((cell, ((x+cell[0])//2,(y+cell[1])//2)))
        seen.add(cell)
        stack.append(cell)
    boundary = [(10,y) for y in range(1,10,2)] + [(x,10) for x in range(3,10,2)]
    free.update(rng.sample(boundary, rng.randint(4, len(boundary))))
    rotation, mirror = rng.randrange(4), bool(rng.randrange(2))
    def transform(cell):
        x,y = cell
        if mirror:
            x = 11-x
        for _ in range(rotation):
            x,y = 11-y,x
        return x,y
    free = {transform(cell) for cell in free} - RESERVED
    if len(_connected(free, min(free))) != len(free):
        return None
    def neighbors(cell):
        x,y = cell
        return [(x+dx,y+dy) for dx,dy in names.ACTION_DELTAS if (x+dx,y+dy) in free]
    def paths_from(start):
        paths, queue = {start:[start]}, [start]
        for cell in queue:
            for neighbor in neighbors(cell):
                if neighbor not in paths:
                    paths[neighbor] = paths[cell] + [neighbor]
                    queue.append(neighbor)
        return paths
    leaves = [cell for cell in sorted(free) if len(neighbors(cell)) == 1]
    if len(leaves) < 4:
        return None
    start = rng.choice(leaves)
    outward = paths_from(start)
    distant = [cell for cell in leaves if len(outward[cell]) >= (22 if challenge else 34)]
    if not distant:
        return None
    cycler = rng.choice(distant)
    returning = paths_from(cycler)
    presses = rng.randint(2, 3)
    minimum = 39 if challenge else 72
    goals = [cell for cell in leaves if cell not in (start,cycler)
             and len(outward[cycler]) + len(returning[cell]) - 2 + 2*(presses-1) >= minimum]
    if not goals:
        return None
    goal = rng.choice(goals)
    main_route = set(outward[cycler]) | set(returning[goal])
    decoys = [cell for cell in leaves if cell not in main_route]
    if not decoys:
        return None
    decoy = rng.choice(decoys)
    kind = rng.choice(KINDS)
    initial = [rng.randrange(size) for size in SIZES]
    triple = initial.copy()
    dimension = KINDS.index(kind)
    triple[dimension] = (triple[dimension]+presses) % SIZES[dimension]
    available = free - {start,cycler,goal,decoy}
    side_refills = sorted(available - main_route)
    # Off-route tanks can be saved for the return trip. Their locations and the
    # remaining tanks are sampled, not fixed offsets along a memorized path.
    count = 6 if challenge else rng.choice((3,4,5,6))
    refills = rng.sample(side_refills, min(len(side_refills), 2))
    refills += rng.sample(sorted(available-set(refills)), count-len(refills))
    return dict(free=free, start=start, start_triple=initial,
                goals=[{'cell':goal,'triple':triple}],
                cyclers=[{'cell':cycler,'kind':kind},
                         {'cell':decoy,'kind':rng.choice([k for k in KINDS if k != kind])}],
                rails=[], launchers=[], refills=refills, step_counter=42,
                step_cost=2 if challenge else 1, topology='tree_maze', rail_mode='none',
                quality_profile='challenge' if challenge else 'learning', difficulty=5,
                required_cycler_presses=presses)


def _candidate(seed, mode, attempt):
    if mode not in MODES:
        raise ValueError(f'unknown mechanism mode: {mode}')
    rng = random.Random(f'mechanism-v{VERSION}:{seed}:{mode}:{attempt}')
    if mode in ('long_route','refill_chain'):
        draft = _long_route(rng, challenge=mode == 'refill_chain')
        if draft is None:
            return None
    else:
        width, height = rng.choice(((6,4),(7,3),(8,3)) if mode == 'four_goals' else
                                   ((8,4),(7,5),(10,3),(8,6),(9,7)))
        if mode in ('three_rails', 'long_rail', 'mixed_mechanisms'):
            width, height = rng.randint(6, 10), rng.randint(5, 8)
        if mode == 'launcher_network':
            width, height = rng.choice(((8,4),(7,5),(9,4)))
        if mode == 'three_rails':
            width, height = rng.choice(((6,4),(7,4),(8,3)))
        left, top = rng.randint(1,11-width), rng.randint(1,11-height)
        free = {(left+x,top+y) for x in range(width) for y in range(height)} - RESERVED
        rails = []
        if mode == 'long_rail':
            if rng.randrange(2):
                length = rng.randint(6, width)
                x, y = left + rng.randrange(width-length+1), top + rng.randrange(height)
                cells = {(x+i, y) for i in range(length)}
            else:
                rw, rh = rng.randint(3, min(5, width)), rng.randint(3, min(5, height))
                x, y = left + rng.randrange(width-rw+1), top + rng.randrange(height-rh+1)
                cells = {(x+i, y+j) for i in range(rw) for j in range(rh)
                         if i in (0,rw-1) or j in (0,rh-1)}
            rails = [{'cells': sorted(cells)}]
        elif mode in ('three_rails', 'mixed_mechanisms'):
            walked = set()
            for _ in range(3 if mode == 'three_rails' else 1):
                options = []
                length = 2  # synchronized short rails bound three-attribute search
                for x, y in sorted(free - walked):
                    for dx, dy in ((1,0),(0,1)):
                        cells = {(x+i*dx,y+i*dy) for i in range(length)}
                        if cells <= free and not cells & walked:
                            options.append(cells)
                if not options:
                    return None
                cells = rng.choice(options)
                walked |= cells
                rails.append({'cells': sorted(cells)})
        rail_cells = {tuple(cell) for rail in rails for cell in rail['cells']}
        topology = rng.choice(('room', 'obstacles', 'two_rooms'))
        if topology == 'two_rooms':
            door = top + rng.randrange(height)
            cut = {(left+width//2,y) for y in range(top,top+height) if y != door}
            free -= cut - rail_cells
        elif topology == 'obstacles':
            for cell in rng.sample(sorted(free - rail_cells), min(8, len(free)//6)):
                remaining = free - {cell}
                if len(_connected(remaining, min(remaining))) == len(remaining):
                    free = remaining
        if not rail_cells <= free or len(_connected(free, min(free))) != len(free):
            return None
        n_goals = {'three_goals':3, 'four_goals':4, 'three_launchers':2, 'mixed_mechanisms':2}.get(mode, 1)
        # Composition has explicit modes/quotas, so search rejection cannot
        # silently turn the entire bank back into single-attribute lessons.
        count = {'one_attribute':1, 'two_attributes':2, 'three_attributes':3, 'three_rails':3}.get(mode, 2)
        kinds = rng.sample(KINDS, count)
        if n_goals == 4:
            kinds = ['color', 'rotation']  # two independent axes without the larger shape state space
        spots = sorted(free - rail_cells)
        rng.shuffle(spots)
        refill_count = 3 if mode == 'launcher_network' else rng.choice((0,1,2)) if n_goals == 4 else rng.choice((0,1,2,3))
        if len(spots) < 1 + n_goals + len(kinds) + refill_count + 2 + 3:
            return None
        start = spots.pop()
        cyclers = [{'cell': rng.choice(rail['cells']), 'kind': kinds[i % len(kinds)]}
                   for i, rail in enumerate(rails)]
        covered = {c['kind'] for c in cyclers}
        cyclers += [{'cell': spots.pop(), 'kind': kind} for kind in kinds if kind not in covered]
        # Additional tiles are only certified as distractors after route replay.
        cyclers += [{'cell': spots.pop(), 'kind': rng.choice([k for k in KINDS if k not in kinds] or kinds)} for _ in range(rng.randint(1,2))]
        initial = [rng.randrange(size) for size in SIZES]
        options = []
        for values in product(*(range(1, SIZES[KINDS.index(kind)]) for kind in kinds)):
            triple = initial.copy()
            for kind, delta in zip(kinds, values):
                dim = KINDS.index(kind)
                triple[dim] = (initial[dim] + delta) % SIZES[dim]
            options.append(triple)
        goals = [{'cell': spots.pop(), 'triple': triple} for triple in rng.sample(options, n_goals)]
        refills = [spots.pop() for _ in range(refill_count)]
        launchers = []
        if mode in ('three_launchers', 'mixed_mechanisms', 'launcher_network'):
            for _ in range({'three_launchers':3,'mixed_mechanisms':1,'launcher_network':8}[mode]):
                options = []
                for cell in spots:
                    for dx,dy in names.ACTION_DELTAS:
                        if (cell[0]-dx,cell[1]-dy) in free:
                            continue
                        p, n = cell, 0
                        while True:
                            p = (p[0]+dx,p[1]+dy)
                            if p not in free or p in {tuple(g['cell']) for g in goals}:
                                break
                            n += 1
                        if n >= 3:
                            options.append((cell,(dx,dy)))
                if not options:
                    return None
                cell, delta = rng.choice(options)
                spots.remove(cell)
                launchers.append({'cell':cell,'delta':delta})
        cost = 2 if mode == 'launcher_network' else 1 if n_goals == 4 or mode == 'three_rails' else rng.choice((1,1,2))
        draft = dict(free=free, start=start, start_triple=initial, goals=goals,
                     cyclers=cyclers, rails=rails, launchers=launchers, refills=refills,
                     step_counter=42, step_cost=cost, topology=topology,
                     quality_profile='challenge' if mode == 'launcher_network' else 'learning',
                     rail_mode=mode if rails else 'none',
                     difficulty={'one_attribute':1, 'two_attributes':2, 'three_attributes':3,
                                 'three_goals':4, 'four_goals':5, 'long_rail':4,
                                 'three_rails':5, 'three_launchers':5, 'mixed_mechanisms':5, 'launcher_network':5}[mode])
    free = draft.pop('free')
    return dict(format=FORMAT, generator_version=GENERATOR_VERSION,
                curriculum_version=f'mechanism-pilot-v{VERSION}', quality_version=VERSION,
                seed=seed, size=64, pilot_mode=mode, generation_attempt=attempt+1,
                walls=sorted({(x,y) for x in range(12) for y in range(12)}-free),
                free_cells=len(free), fog=bool(rng.randrange(2)), **draft)


def candidate(seed, mode, attempt):
    split = 'train' if 700000 <= seed < 900000 else 'validation'
    for redraw in range(16):
        spec = _candidate(seed, mode, attempt * 16 + redraw)
        if spec is None:
            continue
        fingerprint, partition = geometry_partition(spec)
        if partition == split:
            spec.update(split=split, geometry_sha256=fingerprint, geometry_split=partition,
                        generation_attempt=attempt+1, geometry_attempt=redraw+1)
            return spec
    return None


def verify(spec, limit):
    if spec.get('quality_version') == VERSION or 'geometry_split' in spec:
        fingerprint, partition = geometry_partition(spec)
        expected = 'train' if 700000 <= spec['seed'] < 900000 else 'validation'
        if (partition != expected or spec.get('geometry_split') != partition
                or spec.get('geometry_sha256') != fingerprint or spec.get('split') != partition):
            return None, 'geometry_split_mismatch'
    if any(tuple(g['triple']) == tuple(spec['start_triple']) for g in spec['goals']):
        return None, 'goal_satisfied_at_spawn'
    env, oracle, proof = verified_context(spec, search_limit=limit)
    if env is None:
        return None, proof['excluded']
    layout = oracle.layout
    walked = set().union(*layout.moving_cyclers)
    if walked & {cell for pad in layout.launchers for cell in pad['triggers']}:
        return None, 'rail-launcher overlap'
    solution = oracle.solution(seed=spec['seed'])
    minimum = 72 if spec.get('pilot_mode') == 'long_route' else 39 if spec.get('pilot_mode') == 'refill_chain' else MIN_ACTIONS
    if len(solution) < minimum:
        return None, f'route_under_{minimum}_actions'
    state = oracle.start
    minimum_slack = state[6] // layout.step_cost
    used, static_contacts = Counter(), set()
    moving_contacts, launcher_contacts = set(), set()
    for action in solution:
        before = state
        index = names.ACTION_IDS.index(action)
        state, outcome = simulate(layout, state, index, oracle.refills)
        minimum_slack = min(minimum_slack, route_budget_slack(layout, before, state, action=index, outcome=outcome))
        dx, dy = names.ACTION_DELTAS[index]
        target = (before[0][0]+dx, before[0][1]+dy)
        if target in layout.cyclers:
            static_contacts.add(target)
        destinations = {target} | ({state[0]} if outcome == 'launched' else set())
        tick = layout.next_tick(before[7])
        for number, patroller in enumerate(layout.patrollers):
            own = tick if tick < patroller['tail'] else patroller['tail'] + (tick-patroller['tail']) % patroller['period']
            if patroller['cells'][own] in destinations:
                moving_contacts.add(number)
        if outcome == 'launched':
            entry = target if layout.free(target) else before[0]
            for number, pad in enumerate(layout.launchers):
                if entry in pad['triggers'] and pad['distance'] > 0:
                    launcher_contacts.add(number)
                    break
        if outcome == 'launched' and state[0] in layout.cyclers:
            static_contacts.add(state[0])
        used[outcome] += 1
        used['refills_consumed'] += (state[5]^before[5]).bit_count()
        used['goals_cleared'] += (state[4]^before[4]).bit_count()
        used['moving_cycler_contacts'] += int(state[1:4] != before[1:4] and state[0] in layout.moving_cyclers[state[7]])
        result = env.perform(action)
        if not result.finished and oracle.state_of(env) != state:
            return None, 'stepwise engine oracle state mismatch'
    if not result.won or env.lives() != 3:
        return None, 'extra replay failed'
    slack = state[6] // layout.step_cost
    floor = budget_floor(spec)
    if minimum_slack < floor:
        return None, 'under_eight_route_slack_moves' if floor == 8 else 'negative_route_slack'
    if spec.get('pilot_mode') == 'refill_chain' and used['refills_consumed'] < 3:
        return None, 'fewer_than_three_refills_used'
    if layout.patrollers and not used['moving_cycler_contacts']:
        return None, 'moving cycler unused'
    if spec['launchers'] and not used['launched']:
        return None, 'launchers unused'
    if spec.get('pilot_mode') == 'three_rails' and len(moving_contacts) != 3:
        return None, 'not_all_three_moving_cyclers_used'
    if spec.get('pilot_mode') in ('three_launchers','launcher_network') and len(launcher_contacts) < 3:
        return None, 'fewer_than_three_distinct_launchers_used'
    used['distinct_moving_cyclers'] = len(moving_contacts)
    used['distinct_launchers'] = len(launcher_contacts)
    distractors = len(set(layout.cyclers) - static_contacts)
    if not distractors:
        return None, 'no_unused_distractor_cycler'
    changing = sum(any(g['triple'][i] != spec['start_triple'][i] for g in spec['goals']) for i in range(3))
    return {**spec, 'optimal_actions':len(solution), 'solution':solution, 'reachable_states':oracle._reachable,
            'search_truncated':False, 'search_limit':limit, 'engine_verified':True,
            'training_context_index':spec['seed']%7, 'verification_level_index':spec['seed']%7,
            'context_optimal_actions':len(solution), 'context_engine_verified':True,
            'verification_lives':3, 'budget_floor':floor, 'quality_profile':spec.get('quality_profile','learning'), 'slack_moves':slack, 'minimum_slack_moves':minimum_slack,
            'distractor_count':distractors, 'changing_attributes':changing,
            'nonrequired_distractor_count':sum(layout.cyclers[cell] not in
                {KINDS[i] for i in range(3) if any(g['triple'][i] != spec['start_triple'][i] for g in spec['goals'])}
                for cell in set(layout.cyclers)-static_contacts),
            'oracle_backend':oracle.engine, 'tick_period':layout.tick_period,
            'solution_mechanics':dict(used), 'proof':proof}, None


def generate_one(seed, mode, *, attempts, limit, seen_seeds, seen_specs,
                 record_rejection, namespace_end=None, seed_attempts=8):
    """Fill one mode quota with bounded seed retries, preserving all refusals."""
    for _ in range(seed_attempts):
        while seed in seen_seeds or (seed % 7 == 0 and mode in ('three_launchers','mixed_mechanisms','launcher_network')):
            seed += 1
        if namespace_end is not None and seed >= namespace_end:
            raise RuntimeError('training seed namespace exhausted')
        for attempt in range(attempts):
            spec = candidate(seed, mode, attempt)
            accepted, reason = None, 'invalid_draft'
            if spec is not None:
                if canonical(spec) in seen_specs:
                    reason = 'gameplay_duplicate'
                else:
                    try:
                        accepted, reason = verify(spec, limit)
                    except ValueError as error:
                        reason = f'inexact_layout:{error}'
            if accepted is not None:
                return accepted, seed + 1
            record_rejection({'seed':seed, 'mode':mode, 'attempt':attempt+1, 'reason':reason})
            if reason and ('state mismatch' in reason or 'replay' in reason):
                raise RuntimeError(f'engine contract failure: {reason}')
        record_rejection({'seed':seed, 'mode':mode, 'attempt':attempts, 'reason':'seed_attempts_exhausted'})
        seed += 1
    raise RuntimeError(f'bounded seed attempts exhausted: {mode} {seed}')


LIMITATIONS = [
    'New v2 normalized geometry is split-disjoint; historical v1 banks and topology families are not held out.',
    'Four-goal lessons use compact rooms, cost1 and color/rotation; complete600k-state search still censors harder drafts.',
    'Three-rail lessons require all three kinds to be used, on synchronized length2 rails; long-rail lessons vary length/ring geometry.',
    'Launcher quotas describe installed pads; reports separately count distinct pads used by the verified route.',
    'All-three-attribute lessons have no non-required kind; their extra cyclers are verified unused alternatives.',
    'Learning headroom is not a guarantee arbitrary errors are recoverable; challenge refill-chain rows deliberately allow zero margin.',
    'Context-zero launchers remain outside the exact oracle contract; seed/context is resampled only for launcher modes.',
    'Banks are generated lessons, not controller performance; cached arrays and trained checkpoints must be rebuilt separately.',
]


def coverage(rows):
    axes = {'difficulty':lambda r:r['difficulty'], 'quality_profile':lambda r:r.get('quality_profile','learning'), 'topology':lambda r:r['topology'],
            'changing_attributes':lambda r:r['changing_attributes'],
            'goals':lambda r:len(r['goals']), 'refills':lambda r:len(r['refills']),
            'launchers':lambda r:len(r['launchers']), 'rails':lambda r:len(r['rails']),
            'step_cost':lambda r:r['step_cost'], 'step_counter':lambda r:r['step_counter'],
            'fog':lambda r:r['fog'], 'context':lambda r:r['training_context_index'], 'distractor_count':lambda r:r['distractor_count'],
            'route_band':lambda r:'72+' if r['optimal_actions'] >= 72 else '40-71' if r['optimal_actions'] >= 40 else '10-39'}
    return {key:dict(Counter(fn(row) for row in rows)) for key, fn in axes.items()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-mode',type=int,default=4);parser.add_argument('--attempts',type=int,default=40)
    parser.add_argument('--limit',type=int,default=600000);parser.add_argument('--seconds',type=int,default=270)
    parser.add_argument('--start-seed',type=int,default=8_000_001,
                        help='First candidate seed; existing seeds and unsupported context-zero launchers are skipped')
    parser.add_argument('--out',type=Path,default=Path('data/ls20-mechanism-pilot-v2.jsonl'))
    parser.add_argument('--report',type=Path,default=Path('artifacts/world-mechanism-pilot-v2.json'))
    args=parser.parse_args(argv);started=time.monotonic()
    if args.start_seed < 0 or min(args.per_mode, args.attempts, args.limit, args.seconds) < 1:
        parser.error('nonnegative start seed and positive counts, search limit and seconds required')
    if args.out.exists() or args.report.exists():
        raise FileExistsError('refuse existing output/report')
    # Inventory only generated banks; no shipped inputs or frames. Do not follow symlinks.
    paths=sorted(p for p in Path('data').rglob('*.jsonl') if p!=args.out and not p.is_symlink())
    seen_seeds=set();seen_specs=set();sources={}
    for path in paths:
        used=False
        for line in path.open():
            row=json.loads(line)
            if row.get('format')!=FORMAT:continue
            seen_seeds.add(row['seed']);seen_specs.add(canonical(row));used=True
        if used:sources[str(path)]=digest(path)
    code={str(p):digest(p) for p in [Path(__file__),Path('pebby/ls20/generate.py'),Path('pebby/ls20/generation_quality.py'),Path('pebby/ls20/plan.py'),Path('pebby/ls20/fastplan.py'),Path('pebby/ls20/_fastplan.c'),Path('pebby/ls20/layout.py'),Path('pebby/ls20/rails.py'),Path('pebby/agent/world_data.py')]}
    report={'format':'pebby.mechanism-pilot.v2','status':'running','pid':os.getpid(),'source_hashes':sources,'code_hashes':code,
        'existing_distinct_seeds':len(seen_seeds),'accepted':[],'rejections':[],'limits':vars(args)|{'out':str(args.out),'report':str(args.report)},
        'limitations':LIMITATIONS, 'official_inputs':False,'scope':'Generated pilot; no training, no controller result. Explicit composition and long-route modes; complete-search caps still bias acceptance.'}
    def persist():
        report['runtime_seconds']=time.monotonic()-started
        tmp=args.report.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.report)
    def deadline(_sig,_frame):raise TimeoutError('pilot wall deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(args.seconds)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(args.out)
    seed=args.start_seed
    print(json.dumps({'pid':os.getpid(),'existing_seeds':len(seen_seeds),'limit':args.limit}),flush=True)
    try:
        with args.out.open('x') as out:
            for _ in range(args.per_mode):
                for mode in MODES:
                    accepted, seed = generate_one(seed, mode, attempts=args.attempts, limit=args.limit,
                        seen_seeds=seen_seeds, seen_specs=seen_specs,
                        record_rejection=report['rejections'].append)
                    seen_specs.add(canonical(accepted));seen_seeds.add(accepted['seed'])
                    out.write(json.dumps(accepted,separators=(',',':'))+'\n');out.flush();os.fsync(out.fileno())
                    report['accepted'].append({k:accepted[k] for k in ('seed','pilot_mode','optimal_actions','reachable_states','oracle_backend','tick_period','solution_mechanics')})
                    persist();print(json.dumps(report['accepted'][-1]),flush=True)
        report['status']='complete'
    except (TimeoutError,RuntimeError) as error:report['status']='bounded_partial';report['error']=str(error)
    finally:
        signal.alarm(0);report['bank_sha256']=digest(args.out)
        rows=[json.loads(line) for line in args.out.open()]
        report['coverage']=coverage(rows)
        report['quality_floors']={'route_actions':MIN_ACTIONS,'long_route_actions':72,'learning_slack_moves':MIN_SLACK,'challenge_slack_moves':0}
        report['rail_cell_lengths']=dict(Counter(len(rail['cells']) for row in rows for rail in row['rails']))
        report['rejection_counts']=dict(Counter(r['reason'] for r in report['rejections']))
        report['code_hashes_unchanged']=all(digest(p)==h for p,h in code.items())
        if not report['code_hashes_unchanged']:report['status']='failed_closed'
        persist()
        print(json.dumps({'status':report['status'],'accepted':len(rows),'seconds':report['runtime_seconds'],'coverage':report['coverage'],'rejections':report['rejection_counts']}),flush=True)

    return 0 if report['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
