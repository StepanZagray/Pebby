"""Check pebby's transition model against the real LS20 engine, action by action.

Run with: PYTHONPATH=. .venv/bin/python tools/differential.py

A uniformly random walk almost never stands on a rail-riding cycler, so levels
5-7 used to pass without their one moving tile ever being exercised. A `hunt`
fraction of moves instead follows a shortest path towards the patrolled cells,
detouring to a refill when the budget would not last the trip. Episodes cycle
through several hunt strengths including zero, because a pure hunt is itself
blind: it paces the rail and stops meeting goal pads and launchers. The report
breaks the compared transitions down by outcome and by whether the player landed
on a moving cycler -- that table, not the episode count, is what says which
mechanics were actually tested.
"""

import collections, random
from pebby.ls20 import names, rails
from pebby.ls20.env import Ls20Env
from pebby.ls20.layout import extract
from pebby.ls20.plan import simulate, _refill_order

def engine_state(env, layout, refills):
    present = set()
    for s in env.game.current_level.get_sprites():
        if s.tags and names.TAG_STEP_REFILL in s.tags:
            present.add(names.pixel_to_cell(s.x, s.y))
    taken = sum(1 << i for i, c in enumerate(refills) if c not in present)
    goals = sum(1 << i for i, g in enumerate(env.goals_solved()) if g)
    # Read the patroller clock back off the sprites: `phase_of` returns the tick
    # whose predicted (position, direction) matches every patroller, so this
    # compares real sprite positions to the planner's table, not tick to tick.
    tick = rails.phase_of(layout.patrollers, layout.tick_tail, layout.tick_period,
                          rails.live_states(env.game))
    return (env.player_cell(), *env.triple(), goals, taken, env.steps_left(), tick)

def targets(layout, refills, state):
    """Where the hunt is heading right now.

    Level 7's rail is 16 moves from the spawn on a 21-move budget, so a hunt that
    walks straight at it dies before it arrives and the moving cycler is barely
    touched. Detour to an unclaimed refill whenever the budget is running short.
    """
    patrolled = frozenset().union(*layout.moving_cyclers) if layout.patrollers else frozenset()
    untaken = frozenset(cell for i, cell in enumerate(refills) if not (state[5] >> i) & 1)
    if untaken and state[6] // layout.step_cost <= 8:
        return untaken
    return patrolled

def chase(layout, refills, state, rng):
    """A move along a shortest walkable path towards the current targets.

    Greedy distance-closing is not enough: level 7's rail sits behind a wall and
    a greedy walk parks against it forever, which is how the moving cycler
    escaped testing in the first place. Flooding backwards from every target
    costs nothing at this scale. Ties are taken at random, and standing on a
    target is a tie with its neighbours, so the walk paces the rail rather than
    leaving it the moment it arrives.
    """
    goals = targets(layout, refills, state)
    if not goals:
        return rng.choice(names.ACTION_IDS)
    cell, distance = state[0], _flood(layout, goals)
    actions = list(range(4))
    rng.shuffle(actions)
    reachable = [a for a in actions
                 if distance.get(_step(cell, a)) is not None]
    if not reachable:
        return rng.choice(names.ACTION_IDS)
    return names.ACTION_IDS[min(reachable, key=lambda a: distance[_step(cell, a)])]

def _step(cell, action):
    dx, dy = names.ACTION_DELTAS[action]
    return (cell[0] + dx, cell[1] + dy)

_floods = {}

def _flood(layout, goals):
    """Cell -> walking distance to the nearest goal cell, cached per target set."""
    key = (layout.level_index, goals)
    if key not in _floods:
        distance = {cell: 0 for cell in goals}
        queue = collections.deque(goals)
        while queue:
            cell = queue.popleft()
            for action in range(4):
                nxt = _step(cell, action)
                if layout.free(nxt) and nxt not in distance:
                    distance[nxt] = distance[cell] + 1
                    queue.append(nxt)
        _floods[key] = distance
    return _floods[key]

HUNTS = (0.0, 0.7, 0.95)  # pure random, mixed, and nearly glued to the rail

def run(level, episodes=600, length=60, seed=0):
    rng = random.Random(seed)
    mismatches = compared = 0
    outcomes = collections.Counter()
    first_bad = None
    for ep in range(episodes):
        hunt = HUNTS[ep % len(HUNTS)]
        env = Ls20Env(); env.set_level(level)
        layout = extract(env); refills = _refill_order(layout)
        state = (layout.start_cell, *layout.start_triple, 0, 0, layout.max_steps, 0)
        history = []
        for _ in range(length):
            action = (chase(layout, refills, state, rng) if rng.random() < hunt
                      else rng.choice(names.ACTION_IDS))
            history.append(action)
            predicted, outcome = simulate(layout, state, action - 1, refills)
            # Where the player ended up, which for a launch is the landing cell.
            moving = layout.moving_cyclers[predicted[7]].get(predicted[0]) is not None
            obs = env.perform(action)
            if outcome in ("died", "won") or obs.finished or env.level_index != level:
                break
            actual = engine_state(env, layout, refills)
            compared += 1
            outcomes[outcome + ("+cycler" if moving else "")] += 1
            if actual != predicted:
                mismatches += 1
                if first_bad is None:
                    first_bad = (ep, list(history), predicted, actual, outcome)
                break
            state = predicted
    return compared, mismatches, outcomes, first_bad

for level in range(7):
    compared, bad, outcomes, first = run(level)
    flag = "OK " if bad == 0 else "DIVERGES"
    mix = " ".join(f"{name}={count}" for name, count in sorted(outcomes.items()))
    print(f"L{level+1}: {flag} {compared - bad}/{compared} transitions matched")
    print(f"     {mix}")
    if first:
        ep, hist, pred, act, outcome = first
        print(f"     first divergence after {len(hist)} actions {hist[-6:]} outcome={outcome}")
        print(f"     predicted cell/triple/goals/taken/steps/tick = {pred}")
        print(f"     engine    cell/triple/goals/taken/steps/tick = {act}")
