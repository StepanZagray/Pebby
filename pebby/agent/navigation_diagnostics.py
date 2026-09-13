"""Small generated matching-goal curriculum, verified in the real LS20 engine.

Only ``frames``, ``history_valid`` and ``previous_actions`` are model inputs.
Everything else returned by :func:`collect_examples` is a training target or
an evaluation identifier. Static BFS is a teacher, never an inference input.
Expert-path primitives contain no life-loss positives or attribute changes by
design; deliberately bad learner trajectories can introduce life-loss targets.
"""

from collections import deque
import copy
import hashlib
import random

import numpy as np

from ..ls20 import names
from ..ls20.env import Ls20Scenario
from ..ls20.generate import FORMAT, RESERVED, build_level

SPLITS = ('train', 'development', 'confirmation')
STAGES = ('adjacent', 'open', 'detour')
DIRECTIONS = ('up', 'down', 'left', 'right', 'upper-left', 'upper-right',
              'lower-left', 'lower-right')
OFFSETS = (*names.ACTION_DELTAS, (-1, -1), (1, -1), (-1, 1), (1, 1))
PAIR_INDICES = (0, 0, 1, 1, 2, 3, 3, 2)
MODEL_INPUT_KEYS = ('frames', 'history_valid', 'previous_actions')


def _distance_map(spec):
    if (len(spec['goals']) != 1 or spec['goals'][0]['triple'] != spec['start_triple']
            or any(spec.get(key) for key in ('cyclers', 'refills', 'launchers', 'rails'))
            or spec.get('fog') or spec['step_cost'] != 1):
        raise ValueError('navigation teacher requires one matching goal and static walls')
    walls = set(map(tuple, spec['walls']))
    goal = tuple(spec['goals'][0]['cell'])
    free = {(x, y) for x in range(names.GRID_COLS) for y in range(names.GRID_ROWS)} - walls
    if goal not in free or tuple(spec['start']) not in free:
        raise ValueError('start and goal must be free cells')
    distances, queue = {goal: 0}, deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in names.ACTION_DELTAS:
            cell = (x + dx, y + dy)
            if cell in free and cell not in distances:
                distances[cell] = distances[x, y] + 1
                queue.append(cell)
    return distances


def distance_to_goal(spec, cell):
    """Full remaining static route length; -1 means unreachable/blocked."""
    return _distance_map(spec).get(tuple(cell), -1)


def _mask(distances, cell):
    cell = tuple(cell)
    before = distances.get(cell, -1)
    if before <= 0:
        return 0
    return sum(1 << i for i, (dx, dy) in enumerate(names.ACTION_DELTAS)
               if distances.get((cell[0] + dx, cell[1] + dy), -1) == before - 1)


def optimal_actions(spec, cell):
    """Optimal-action bitmask in up/down/left/right order (indices 0..3)."""
    return _mask(_distance_map(spec), cell)


def _first_action(mask):
    if not mask:
        raise ValueError('no optimal action at a live teacher root')
    return next(i for i in range(4) if mask & (1 << i))


def _verify(spec):
    distances = _distance_map(spec)
    length = distances.get(tuple(spec['start']), -1)
    if not 1 <= length < spec['step_counter'] - 1:
        raise ValueError('navigation route must be nonempty with branch budget slack')
    env = Ls20Scenario(build_level(spec), 0)
    frame = np.asarray(env.render(), dtype=np.uint8)
    if tuple(spec['start']) != env.player_cell() or tuple(spec['start_triple']) != env.triple():
        raise ValueError('engine initial state disagrees with navigation specification')
    for step in range(length):
        result = env.perform(names.ACTION_IDS[_first_action(_mask(distances, env.player_cell()))])
        if result.finished != (step == length - 1) or env.lives() != 3:
            raise ValueError('engine failed navigation route verification')
        if distances.get(env.player_cell()) != length - step - 1:
            raise ValueError('engine movement disagrees with static navigation teacher')
    if not result.won:
        raise ValueError('engine did not win verified navigation route')
    return length, hashlib.sha256(frame.tobytes()).hexdigest()


def _group(seed, group_index, setting, stages):
    x, y, shape, color, rotation = setting
    triple = [shape, color, rotation]
    group_id = f'nav-{seed}-g{group_index:04d}'
    cases = []
    border = {(a, b) for a in range(12) for b in range(12)
              if a in (0, 11) or b in (0, 11)} | RESERVED
    for stage in stages:
        for direction, (dx, dy) in enumerate(OFFSETS[:8 if stage == 'open' else 4]):
            distance = {'adjacent': 1, 'open': 2, 'detour': 3}[stage]
            walls = set(border)
            if stage == 'detour':
                # Opposite goals share both obstacles, start and glyph. The
                # optimal first step must move perpendicular to goal direction.
                walls.update(((x + dx, y + dy), (x - dx, y - dy)))
            spec = dict(format=FORMAT, generator_version=3, seed=seed,
                        source='generated_only', size=64, walls=[list(c) for c in sorted(walls)],
                        start=[x, y], start_triple=triple.copy(),
                        goals=[dict(cell=[x + distance * dx, y + distance * dy], triple=triple.copy())],
                        cyclers=[], refills=[], launchers=[], rails=[], step_counter=42,
                        step_cost=1, fog=False)
            length, digest = _verify(spec)
            cases.append(dict(id=f'{group_id}-{stage}-{DIRECTIONS[direction]}',
                              group_id=group_id, stage=stage, spec=spec,
                              optimal_length=length, direction=DIRECTIONS[direction],
                              pair_id=f'{group_id}-{stage}-p{PAIR_INDICES[direction]}',
                              initial_frame_sha256=digest))
    return cases


def make_cases(seed=0, groups_per_split=4, stages=STAGES, *, guard=None):
    """Generate whole disjoint groups; default is 64 cases per split.

    Each group shares one start/glyph setting across all selected stages and
    includes every opposite direction pair. Adjacent and detour have four
    cardinal goals; open has eight cardinal/diagonal goals. Group settings are
    sampled without replacement. Exact public initial-frame collisions with
    another split reject the entire candidate group, including glyph symmetries.
    This is a controlled primitive curriculum, not benchmark generalization.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if isinstance(groups_per_split, bool) or not isinstance(groups_per_split, int) or not 1 <= groups_per_split <= 32:
        raise ValueError('groups_per_split must be an integer in 1..32')
    stages = tuple(stages)
    if not stages or len(set(stages)) != len(stages) or set(stages) - set(STAGES):
        raise ValueError('stages must be nonempty, unique supported stages')
    settings = [(x, y, s, c, r) for x in range(4, 8) for y in range(4, 8)
                for s in range(6) for c in range(4) for r in range(4)]
    random.Random(seed).shuffle(settings)
    result, seen = {}, {}
    candidate = 0
    for split in SPLITS:
        cases = []
        accepted = 0
        while accepted < groups_per_split:
            if candidate == len(settings):
                raise RuntimeError('unique public group settings exhausted')
            if guard is not None:
                guard()
            group = _group(seed, candidate, settings[candidate], stages)
            candidate += 1
            if any(case['initial_frame_sha256'] in seen and
                   seen[case['initial_frame_sha256']] != split for case in group):
                continue
            for case in group:
                case['split'] = split
                case['spec']['split'] = split
                seen[case['initial_frame_sha256']] = split
            cases.extend(group)
            accepted += 1
        result[split] = cases
    return result


def public_frame_overlap(splits):
    """Re-render specs and report exact cross-split initial-frame overlap.

    Cached case hashes are not trusted. This checks detectable pixel identity,
    not broader geometric similarity or causal-history overlap.
    """
    hashes = {split: {hashlib.sha256(np.asarray(Ls20Scenario(build_level(case['spec']), 0).render(),
                                               dtype=np.uint8).tobytes()).hexdigest()
                       for case in cases} for split, cases in splits.items()}
    keys = list(hashes)
    return {f'{left}/{right}': sorted(hashes[left] & hashes[right])
            for i, left in enumerate(keys) for right in keys[i + 1:]}


def _remaining_distance(env, distances, spec):
    distance = distances.get(env.player_cell(), -1)
    if distance < 0:
        return -1
    # Upstream checks completion before its overdraft death: reaching the goal
    # on the action reducing zero to -1 still wins.
    if distance <= env.steps_left() + 1:
        return distance
    if env.lives() <= 1:
        return -1
    return max(1, env.steps_left() + 1) + distances[tuple(spec['start'])]


def collect_examples(cases, history=8, *, choose_action=None, max_actions=None, guard=None):
    """Every expert-path live root, with causal history and four engine branches.

    ``distances`` is the full route *after* each candidate action, zero on WIN;
    a wall-bump retains the current route length. ``previous_actions`` contains
    0..3 indices of actions producing each observed frame, and -1 for padding
    or the initial frame. Terminal states are branch targets, never roots.
    Optional ``choose_action(**public_arrays)`` visits learner states; its
    return is an integer action index 0..3. It receives copies of the three
    single-root public arrays and no labels. ``max_actions`` caps each episode
    (default: optimal length); history resets on life loss. ``guard()`` runs
    before each case/root for caller-owned deadlines. Learner labels account
    for required budget resets, with -1 for unreachable final-life states.
    Arrays named in MODEL_INPUT_KEYS are the complete public input boundary.
    Other arrays must never be passed to a policy as inputs.
    """
    if isinstance(history, bool) or not isinstance(history, int) or not 1 <= history <= 64:
        raise ValueError('history must be an integer in 1..64')
    if max_actions is not None and (isinstance(max_actions, bool) or
            not isinstance(max_actions, int) or not 1 <= max_actions <= 256):
        raise ValueError('max_actions must be an integer in 1..256')
    rows = []
    for case_index, case in enumerate(cases):
        if guard is not None:
            guard()
        spec = case['spec']
        distances = _distance_map(spec)
        length = distances.get(tuple(spec['start']), -1)
        if length != case['optimal_length'] or not 1 <= length < spec['step_counter'] - 1:
            raise ValueError('case route length or branch budget is invalid')
        env = Ls20Scenario(build_level(spec), 0)
        frames, actions = [env.render()], [-1]
        for step in range(length if max_actions is None else max_actions):
            if guard is not None:
                guard()
            count = min(len(frames), history)
            padding = history - count
            row = dict(frames=np.asarray([frames[-count]] * padding + frames[-count:], dtype=np.uint8),
                       history_valid=np.asarray([False] * padding + [True] * count, dtype=bool),
                       previous_actions=np.asarray([-1] * padding + actions[-count:], dtype=np.int64),
                       player_cell=env.player_cell(), current_triple=env.triple(),
                       current_steps=env.steps_left(), current_lives=env.lives(),
                       case_index=case_index, stage=case['stage'], group_ids=case['group_id'],
                       root_step=step)
            for key in ('next_player_cell', 'next_triple', 'next_steps', 'next_lives',
                        'distances', 'lost_life', 'terminal', 'won'):
                row[key] = []
            mask = 0
            branches = []
            for index, action in enumerate(names.ACTION_IDS):
                branch = copy.deepcopy(env, {id(env.module): env.module})
                outcome = branch.perform(action)
                lost_life = branch.lives() < env.lives()
                after = (0 if outcome.won else -1 if outcome.finished else
                         _remaining_distance(branch, distances, spec))
                before = _remaining_distance(env, distances, spec)
                dx, dy = names.ACTION_DELTAS[index]
                expected = (env.player_cell()[0] + dx, env.player_cell()[1] + dy)
                if expected not in distances:
                    expected = env.player_cell()
                if lost_life and not outcome.finished:
                    expected = tuple(spec['start'])
                if (branch.player_cell() != expected or branch.triple() != env.triple()
                        or outcome.won != (after == 0)
                        or (outcome.finished and not outcome.won and branch.lives() != 0)):
                    raise ValueError('engine branch disagrees with static navigation teacher')
                for key, value in (('next_player_cell', branch.player_cell()), ('next_triple', branch.triple()),
                                   ('next_steps', branch.steps_left()), ('next_lives', branch.lives()),
                                   ('distances', 0 if outcome.won else after),
                                   ('lost_life', branch.lives() < env.lives()),
                                   ('terminal', outcome.finished), ('won', outcome.won)):
                    row[key].append(value)
                if before > 0 and after == before - 1:
                    mask |= 1 << index
                branches.append((branch, outcome))
            if before > 0 and not mask:
                raise ValueError('engine-derived action mask has no optimal action')
            if distances[env.player_cell()] <= env.steps_left() + 1 and mask != _mask(distances, env.player_cell()):
                raise ValueError('engine-derived action mask disagrees with static BFS')
            row['optimal'] = np.uint8(mask)
            rows.append(row)
            choice = (_first_action(mask) if choose_action is None else
                      choose_action(**{key: row[key].copy() for key in MODEL_INPUT_KEYS}))
            if isinstance(choice, (bool, np.bool_)) or not isinstance(choice, (int, np.integer)) or not 0 <= choice < 4:
                raise ValueError('choose_action must return an action index in 0..3')
            old_lives = env.lives()
            env, outcome = branches[choice]
            if outcome.finished:
                if choose_action is None and (step != length - 1 or not outcome.won):
                    raise ValueError('teacher terminated before its final root')
                break
            if outcome.frame is None:
                raise ValueError('live teacher successor has no public frame')
            if env.lives() < old_lives:
                frames, actions = [outcome.frame], [-1]
            else:
                frames.append(outcome.frame)
                actions.append(choice)
    if not rows:
        raise ValueError('cases must contain at least one navigation case')
    arrays = {}
    for key in rows[0]:
        dtype = (np.uint8 if key in ('frames', 'optimal') else bool if key in
                 ('history_valid', 'lost_life', 'terminal', 'won') else str if key in
                 ('stage', 'group_ids') else np.int64)
        arrays[key] = np.asarray([row[key] for row in rows], dtype=dtype)
    arrays['labels'] = arrays['optimal']
    return arrays
