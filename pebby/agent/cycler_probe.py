"""Cycler spoil probe: engine-verified approach/leave/avoid decisions around one cycler.

The retained controller loses shipped levels mainly by stepping back onto a
cycler right after the carried attribute already matches the goal. This probe
isolates that decision in small open rooms holding one goal and one cycler and
scores a public-history policy at three engine-reached states per room:

* ``approach`` (A): the player stands next to the cycler, the attribute does
  not match yet, and the exact oracle says entering the cycler is optimal.
* ``leave`` (B): the player stands on the cycler having just obtained the
  matching attribute. No single action can re-enter a cycler you stand on
  (wall bumps do not re-trigger it; the engine is checked for that), so the
  spoiling action here is the retreat back to the approach cell, which is the
  first half of the observed step-off/step-back loop. The oracle excludes it.
* ``avoid`` (C): the player stepped off the cycler with the attribute already
  matching. The spoiling action steps straight back onto the cycler, which the
  engine confirms breaks the match again. The oracle excludes it.

Ground truth is the exact oracle's optimal-action set at each state, never a
hand rule. Every state is reached by driving the real upstream engine, and the
public history (frames plus the actions that produced them) is built exactly as
``world_data.history_arrays`` builds training inputs. Only ``frames``,
``history_valid`` and ``previous_actions`` ever reach a policy.

This is a development diagnostic on a controlled primitive; it is not a
benchmark and says nothing about shipped-level generalization.
"""

import argparse
from collections import Counter, deque
import json
import random
import time

import numpy as np
import torch

from ..ls20 import names
from ..ls20.env import Ls20Scenario
from ..ls20.generate import FORMAT, RESERVED, build_level
from ..ls20.layout import extract
from ..ls20.plan import Oracle, advance
from .history import PolicyHistory
from .world_data import clone_env, history_arrays

FORMAT_REPORT = 'pebby.cycler-probe.v1'
TYPES = ('approach', 'leave', 'avoid')
KINDS = ('shape', 'color', 'rotation')
SIZES = {'shape': names.SHAPE_COUNT, 'color': names.COLOR_COUNT, 'rotation': names.ROTATION_COUNT}
SPOIL_TYPES = ('leave', 'avoid')
OPPOSITE = (1, 0, 3, 2)
PERPENDICULAR = ((2, 3), (2, 3), (0, 1), (0, 1))
# Step-off patterns after obtaining the match, relative to the approach action.
LEAVE_PATTERNS = ('through', 'retreat', 'perpendicular_a', 'perpendicular_b')
PUBLIC_KEYS = ('frames', 'history_valid', 'previous_actions')
HISTORY = 8
MAX_WALK = HISTORY - 2  # so the whole approach walk can sit inside one H8 window


def optimal_mask(oracle, state):
    """Exact optimal-action bitmask (up/down/left/right bits) at a planner state."""
    best = oracle.distance_for(state)
    if best is None:
        return 0
    mask = 0
    for action in range(4):
        following = advance(oracle.layout, state, action, oracle.refills)
        if following is not None and oracle.distance_for(following) == best - 1:
            mask |= 1 << action
    return mask


def _actions(mask):
    return [names.ACTION_NAMES[i] for i in range(4) if mask & (1 << i)]


def _rect(rng):
    """An open interior rectangle clear of the border and the HUD-reserved cell."""
    width, height = rng.randint(4, 7), rng.randint(4, 7)
    x0 = rng.randint(2, names.GRID_COLS - 1 - width)
    y0 = rng.randint(1, names.GRID_ROWS - 1 - height)
    cells = {(x, y) for x in range(x0, x0 + width) for y in range(y0, y0 + height)}
    if cells & RESERVED:
        raise AssertionError('room overlaps the reserved HUD cell')
    return cells


def _path(free, start, target):
    """Shortest cardinal path as action indices, or None."""
    previous, queue = {start: None}, deque([start])
    while queue:
        cell = queue.popleft()
        if cell == target:
            break
        for index, (dx, dy) in enumerate(names.ACTION_DELTAS):
            following = (cell[0] + dx, cell[1] + dy)
            if following in free and following not in previous:
                previous[following] = (cell, index)
                queue.append(following)
    if target not in previous:
        return None
    actions, cell = [], target
    while previous[cell] is not None:
        cell, index = previous[cell]
        actions.append(index)
    return actions[::-1]


def _draft(rng, kind, approach, leave, context_index):
    """One candidate room; the caller verifies it in the engine and may reject it."""
    room = _rect(rng)
    dx, dy = names.ACTION_DELTAS[approach]
    ex, ey = names.ACTION_DELTAS[leave]
    options = [c for c in sorted(room) if (c[0] - dx, c[1] - dy) in room and (c[0] + ex, c[1] + ey) in room]
    cycler = rng.choice(options)
    before = (cycler[0] - dx, cycler[1] - dy)
    after = (cycler[0] + ex, cycler[1] + ey)
    goals = [c for c in sorted(room) if c not in (cycler, before, after)]
    goal = rng.choice(goals)
    walkable = room - {cycler, goal}
    distances, queue = {before: 0}, deque([before])
    while queue:
        cell = queue.popleft()
        for ddx, ddy in names.ACTION_DELTAS:
            following = (cell[0] + ddx, cell[1] + ddy)
            if following in walkable and following not in distances:
                distances[following] = distances[cell] + 1
                queue.append(following)
    walk = rng.randint(0, MAX_WALK)
    starts = [c for c, d in distances.items() if d == walk]
    if not starts:
        return None
    start = rng.choice(sorted(starts))
    goal_triple = [rng.randrange(SIZES['shape']), rng.randrange(SIZES['color']),
                   rng.randrange(SIZES['rotation'])]
    start_triple = list(goal_triple)
    slot = KINDS.index(kind)
    start_triple[slot] = (goal_triple[slot] - 1) % SIZES[kind]
    walls = sorted({(x, y) for x in range(names.GRID_COLS) for y in range(names.GRID_ROWS)} - room)
    spec = dict(format=FORMAT, generator_version=3, seed=0, source='generated_only',
                size=names.FRAME_SIZE, walls=[list(c) for c in walls], start=list(start),
                start_triple=start_triple, goals=[dict(cell=list(goal), triple=goal_triple)],
                cyclers=[dict(cell=list(cycler), kind=kind)], refills=[], launchers=[], rails=[],
                step_counter=42, step_cost=1, fog=False)
    return spec, dict(room=room, cycler=cycler, before=before, after=after, goal=goal,
                      start=start, walk=_path(walkable, start, before), context_index=context_index)


class _Walk:
    """Drive the real engine and keep the causal public history."""

    def __init__(self, spec, context_index):
        self.spec = spec
        self.env = Ls20Scenario(build_level(spec), context_index)
        self.frames = [np.asarray(self.env.render(), dtype=np.uint8)]
        self.actions = [-1]

    def step(self, action, expected_cell, expected_triple):
        result = self.env.perform(names.ACTION_IDS[action])
        if result.finished or result.frame is None or self.env.lives() != 3:
            raise ValueError('engine terminated or lost a life while driving to a probe state')
        if self.env.player_cell() != tuple(expected_cell) or self.env.triple() != tuple(expected_triple):
            raise ValueError('engine movement disagrees with the probe construction')
        self.frames.append(np.asarray(result.frame, dtype=np.uint8))
        self.actions.append(action)


def _branch_triple(env, action):
    branch = clone_env(env)
    branch.perform(names.ACTION_IDS[action])
    return branch, branch.triple()


def _case(walk, oracle, room, kind, case_type, cycler_direction, index, pattern, room_id):
    env = walk.env
    state = oracle.state_of(env)
    mask = optimal_mask(oracle, state)
    if not mask:
        raise ValueError('oracle has no optimal action at a probe state')
    frames, valid, previous = history_arrays(walk.frames, walk.actions, HISTORY)
    return dict(id=f'{room_id}-{case_type}', room_id=room_id, room_index=index, type=case_type,
                kind=kind, context_index=room['context_index'], leave_pattern=pattern,
                cycler_direction=int(cycler_direction), direction=names.ACTION_NAMES[cycler_direction],
                optimal=int(mask), optimal_actions=_actions(mask),
                oracle_distance=int(oracle.distance_for(state)),
                player_cell=list(env.player_cell()), triple=list(env.triple()),
                goal_triple=list(walk.spec['goals'][0]['triple']),
                steps_left=int(env.steps_left()), walk_length=len(walk.actions) - 1,
                frames=frames, history_valid=valid, previous_actions=previous,
                raw_frames=[frame.copy() for frame in walk.frames], raw_actions=list(walk.actions),
                spec=walk.spec)


def _verify_room(spec, room, kind, approach, leave, index, pattern, engine):
    """Drive the engine through A, B and C; return the three cases or None."""
    context = room['context_index']
    probe = Ls20Scenario(build_level(spec), context)
    if probe.player_cell() != room['start'] or probe.triple() != tuple(spec['start_triple']):
        raise ValueError('engine initial state disagrees with the probe specification')
    oracle = Oracle(extract(probe), engine=engine)
    if oracle.truncated or not oracle.solvable:
        return None
    if oracle.distance_for(oracle.start) > spec['step_counter'] - 8:
        return None
    walk = _Walk(spec, context)
    cell, triple = room['start'], tuple(spec['start_triple'])
    for action in room['walk']:
        dx, dy = names.ACTION_DELTAS[action]
        cell = (cell[0] + dx, cell[1] + dy)
        walk.step(action, cell, triple)
    if walk.env.player_cell() != room['before']:
        raise ValueError('approach walk did not reach the cell next to the cycler')
    room_id = f'cycler-{index:04d}-{kind}-{names.ACTION_NAMES[approach]}-{pattern}-c{context}'
    goal_triple = tuple(spec['goals'][0]['triple'])

    # A: entering must be optimal, and the attribute must not match yet.
    if walk.env.triple() == goal_triple:
        raise ValueError('approach state already matches the goal')
    case_a = _case(walk, oracle, room, kind, 'approach', approach, index, pattern, room_id)
    if not case_a['optimal'] & (1 << approach):
        return None

    # B: on the cycler with the match. Engine: no single action re-triggers the
    # cycler; the retreat-then-re-enter loop does break the match.
    walk.step(approach, room['cycler'], goal_triple)
    for action in range(4):
        _, after = _branch_triple(walk.env, action)
        if after != goal_triple:
            raise ValueError('a single action from the cycler changed the carried attribute')
    retreat, _ = _branch_triple(walk.env, OPPOSITE[approach])
    if retreat.player_cell() != room['before']:
        raise ValueError('retreat from the cycler did not return to the approach cell')
    _, spoiled = _branch_triple(retreat, approach)
    if spoiled == goal_triple:
        raise ValueError('re-entering the cycler did not break the match')
    case_b = _case(walk, oracle, room, kind, 'leave', OPPOSITE[approach], index, pattern, room_id)
    if case_b['optimal'] & (1 << OPPOSITE[approach]):
        return None

    # C: stepped off with the match; stepping back must break it and be excluded.
    walk.step(leave, room['after'], goal_triple)
    back, spoiled = _branch_triple(walk.env, OPPOSITE[leave])
    if back.player_cell() != room['cycler'] or spoiled == goal_triple:
        raise ValueError('stepping back onto the cycler did not re-trigger it')
    case_c = _case(walk, oracle, room, kind, 'avoid', OPPOSITE[leave], index, pattern, room_id)
    if case_c['optimal'] & (1 << OPPOSITE[leave]):
        return None
    return [case_a, case_b, case_c]


def _schedule(index, kinds):
    """Deterministic balance: approach direction cycles fastest, then leave pattern."""
    approach = index % 4
    pattern = LEAVE_PATTERNS[(index // 4) % len(LEAVE_PATTERNS)]
    leave = {'through': approach, 'retreat': OPPOSITE[approach],
             'perpendicular_a': PERPENDICULAR[approach][0],
             'perpendicular_b': PERPENDICULAR[approach][1]}[pattern]
    kind = kinds[index % len(kinds)]
    # (row + column) parity of the 4-wide direction grid: every direction sees
    # both engine contexts once its schedule row is repeated.
    context = (index // 4 + index) % 2
    return kind, approach, leave, pattern, context


def build_cases(seed, count, kinds=KINDS, *, engine='fast', attempts=200, guard=None):
    """``count`` rooms, each yielding one approach, one leave and one avoid case.

    Approach directions cycle up/down/left/right across rooms, so with a
    multiple of four rooms every type sees each direction equally often. The
    step-off pattern after the match cycles through walking through, retreating
    and both perpendicular exits. Cycler kinds cycle through ``kinds`` and the
    engine context alternates between index zero (free matching-cycle hint) and
    one. Rooms whose oracle set disagrees with the intended decision (for
    example a goal placed behind the approach cell) are rejected and redrawn
    with the same schedule slot, so balance survives rejection.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
        raise ValueError('count must be an integer in 1..4096')
    kinds = tuple(kinds)
    if not kinds or len(set(kinds)) != len(kinds) or set(kinds) - set(KINDS):
        raise ValueError(f'kinds must be a nonempty unique subset of {KINDS}')
    rng = random.Random(f'cycler-probe-{seed}')
    cases = []
    for index in range(count):
        kind, approach, leave, pattern, context = _schedule(index, kinds)
        for attempt in range(attempts):
            if guard is not None:
                guard()
            draft = _draft(rng, kind, approach, leave, context)
            if draft is None:
                continue
            spec, room = draft
            spec['seed'] = seed
            verified = _verify_room(spec, room, kind, approach, leave, index, pattern, engine)
            if verified is not None:
                cases.extend(verified)
                break
        else:
            raise RuntimeError(f'no verifiable room for schedule slot {index}')
    return cases


def public_arrays(case, length):
    """The three model inputs for one case at the policy's own history length."""
    return history_arrays(case['raw_frames'], case['raw_actions'], length)


def history_scores(policy, case, device='cpu'):
    """Scores through PolicyHistory, exactly as a live rollout would produce them."""
    history = PolicyHistory(policy, device)
    for frame, action in zip(case['raw_frames'], case['raw_actions']):
        history.observe(frame.tolist(), action)
    with torch.inference_mode():
        return history.scores().detach().cpu()


def _score_cases(policy, cases, device, batch_size):
    length = policy.config().get('history', HISTORY)
    logits = []
    with torch.inference_mode():
        for start in range(0, len(cases), batch_size):
            chunk = cases[start:start + batch_size]
            arrays = [public_arrays(case, length) for case in chunk]
            frames = torch.as_tensor(np.stack([a[0] for a in arrays]), device=device).long()
            valid = torch.as_tensor(np.stack([a[1] for a in arrays]), device=device)
            previous = torch.as_tensor(np.stack([a[2] for a in arrays]), device=device)
            out = policy(frames, history_valid=valid, previous_actions=previous).float()
            if out.shape != (len(chunk), 4) or not bool(torch.isfinite(out).all()):
                raise ValueError('policy must return four finite scores per case')
            logits.append(out.detach().cpu())
    return torch.cat(logits) if logits else torch.zeros(0, 4)


def _summary(rows):
    if not rows:
        return dict(cases=0, accuracy=None, spoil_rate=None, mean_spoil_probability=None,
                    mean_optimal_probability=None)
    spoil = [r for r in rows if r['type'] in SPOIL_TYPES]
    return dict(cases=len(rows), accuracy=float(np.mean([r['correct'] for r in rows])),
                spoil_rate=float(np.mean([r['spoiled'] for r in spoil])) if spoil else None,
                mean_spoil_probability=float(np.mean([r['spoil_probability'] for r in spoil])) if spoil else None,
                mean_optimal_probability=float(np.mean([r['optimal_probability'] for r in rows])))


def evaluate(policy, cases, device='cpu', batch_size=16):
    """Accuracy (argmax in the oracle set) and spoil rate (argmax is the cycler direction).

    ``spoil_rate`` is defined on ``leave`` and ``avoid`` only; for ``approach``
    the cycler direction is the correct move, and ``enter_rate`` reports how
    often it is chosen. ``mean_spoil_probability`` is the softmax mass on the
    spoiling action over leave/avoid cases.
    """
    if not cases:
        raise ValueError('cases must be nonempty')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    was_training = getattr(policy, 'training', False)
    if hasattr(policy, 'eval'):
        policy.eval()
    try:
        logits = _score_cases(policy, cases, device, batch_size)
    finally:
        if was_training and hasattr(policy, 'train'):
            policy.train(True)
    probabilities = logits.softmax(-1)
    rows = []
    for case, score, probability in zip(cases, logits, probabilities):
        choice = int(score.argmax())
        bits = [bool(case['optimal'] & (1 << i)) for i in range(4)]
        rows.append(dict(id=case['id'], type=case['type'], kind=case['kind'], direction=case['direction'],
                         context_index=case['context_index'], leave_pattern=case['leave_pattern'],
                         choice=choice, chosen_action=names.ACTION_NAMES[choice],
                         correct=bits[choice], spoiled=choice == case['cycler_direction'],
                         optimal_probability=float(sum(p for p, b in zip(probability.tolist(), bits) if b)),
                         spoil_probability=float(probability[case['cycler_direction']]),
                         probabilities=probability.tolist(), optimal_actions=case['optimal_actions']))
    per_type = {}
    for case_type in TYPES:
        current = [r for r in rows if r['type'] == case_type]
        per_type[case_type] = _summary(current)
        if case_type == 'approach':
            per_type[case_type]['enter_rate'] = (float(np.mean([r['spoiled'] for r in current]))
                                                 if current else None)
    per_direction = {name: {case_type: _summary([r for r in rows if r['direction'] == name and r['type'] == case_type])
                            for case_type in TYPES} for name in names.ACTION_NAMES}
    per_kind = {kind: _summary([r for r in rows if r['kind'] == kind]) for kind in KINDS
                if any(r['kind'] == kind for r in rows)}
    per_context = {str(c): _summary([r for r in rows if r['context_index'] == c])
                   for c in sorted({r['context_index'] for r in rows})}
    per_pattern = {p: _summary([r for r in rows if r['leave_pattern'] == p and r['type'] == 'avoid'])
                   for p in LEAVE_PATTERNS if any(r['leave_pattern'] == p for r in rows)}
    overall = _summary(rows)
    return dict(format=FORMAT_REPORT, overall=overall, per_type=per_type, per_direction=per_direction,
                per_kind=per_kind, per_context=per_context, avoid_per_leave_pattern=per_pattern,
                rows=rows, chosen_actions=dict(Counter(r['chosen_action'] for r in rows)),
                limits=['Controlled single-cycler primitive; a development diagnostic, not a benchmark.',
                        'Leave spoil is the retreat that starts a step-off/step-back loop; no single action re-enters a cycler you stand on.',
                        'Cases within one room share geometry and history; they are not independent trials.'])


def case_manifest(cases):
    """JSON-safe description of the cases without the frame arrays."""
    skip = {'frames', 'history_valid', 'previous_actions', 'raw_frames', 'raw_actions', 'spec'}
    return [{k: v for k, v in case.items() if k not in skip} for case in cases]


def format_table(report):
    lines = [f"{'type':<10}{'cases':>6}{'accuracy':>10}{'spoil':>8}{'p(spoil)':>10}{'p(opt)':>8}"]
    def fmt(value):
        return '   -' if value is None else f'{value:.3f}'
    for case_type in TYPES:
        s = report['per_type'][case_type]
        lines.append(f"{case_type:<10}{s['cases']:>6}{fmt(s['accuracy']):>10}{fmt(s['spoil_rate']):>8}"
                     f"{fmt(s['mean_spoil_probability']):>10}{fmt(s['mean_optimal_probability']):>8}")
    s = report['overall']
    lines.append(f"{'overall':<10}{s['cases']:>6}{fmt(s['accuracy']):>10}{fmt(s['spoil_rate']):>8}"
                 f"{fmt(s['mean_spoil_probability']):>10}{fmt(s['mean_optimal_probability']):>8}")
    lines.append('per direction (accuracy / spoil):')
    for name, types in report['per_direction'].items():
        parts = [f"{t}={fmt(types[t]['accuracy'])}/{fmt(types[t]['spoil_rate'])}" for t in TYPES]
        lines.append(f"  {name:<6}" + '  '.join(parts))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--count', type=int, default=96, help='rooms; each yields three cases')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--kinds', nargs='+', default=list(KINDS), choices=KINDS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=0, help='torch CPU threads; 0 keeps the default')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    from .model import load_checkpoint
    started = time.monotonic()
    policy, _ = load_checkpoint(args.checkpoint, device=args.device)
    cases = build_cases(args.seed, args.count, tuple(args.kinds))
    built = time.monotonic() - started
    report = evaluate(policy, cases, device=args.device, batch_size=args.batch_size)
    report.update(checkpoint=str(args.checkpoint), seed=args.seed, count=args.count, kinds=list(args.kinds),
                  cases=case_manifest(cases), build_seconds=built,
                  elapsed_seconds=time.monotonic() - started)
    with open(args.out, 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(format_table(report))
    print(f'wrote {args.out}')
    return report


if __name__ == '__main__':
    main()
