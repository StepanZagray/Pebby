"""An exact LS20 planner.

Search runs over the game's *logical* state — cell, carried triple, cleared
goals, consumed refills, remaining budget — rather than over sprites, which
makes it small enough to solve exhaustively. Every rule modelled here was read
out of upstream `step()` and `apply_cell_effects()`; the ones that are easy to
get wrong are called out at their transition.

The planner is only valid for layouts where `Layout.exact` is True. It refuses
otherwise rather than returning a plan that might be unachievable.
"""

from collections import deque
import random

from . import fastplan, names, rails
from .layout import extract

# Enumerated once so a plan never depends on set iteration order.
def _refill_order(layout):
    return tuple(sorted(layout.refills))


class Unplannable(Exception):
    """The layout uses mechanics this planner does not model."""


def simulate(layout, state, action, refills):
    """One action at full fidelity. Returns (next_state, outcome).

    Mirrors upstream `step()` (ls20.py:1912-2014) in its exact order, which is
    easy to get wrong in three places:
      * a wall `break`s out of the cell-effect loop, so a wall never triggers
        the refill or cycler that shares its cell; a rejected goal pad does not;
      * a rejected pad -- and, on level 1 only, a cycle that reveals a match --
        returns before the budget is charged, so those moves are free;
      * a launcher fires only when budget remains, and returns before the win
        check, so a launch can never complete a level.

    `tick` is the shared patroller clock. Upstream advances every patroller
    before it resolves the move (ls20.py:1953-1955) and rewinds them all when
    the move turns out to be blocked (ls20.py:1959-1961) -- where "blocked"
    covers a rejecting goal pad as well as a wall (ls20.py:1878-1885). So cell
    effects are always read at the *next* tick, but only an accepted move keeps
    it. Launcher animation frames return before the advance (ls20.py:1913-1920),
    so a launch still ticks exactly once.
    """
    cell, shape, color, rot, goals, taken, steps, tick = state
    dx, dy = names.ACTION_DELTAS[action]
    target = (cell[0] + dx, cell[1] + dy)
    moved = layout.next_tick(tick)

    rejected = cycled = refilled = False
    if not layout.free(target):
        position = cell                      # wall or edge: no cell effects at all
    else:
        index = layout.goal_at.get(target)
        if (index is not None and not (goals >> index) & 1
                and (shape, color, rot) != layout.goals[index][1]):
            rejected = True
        if target in layout.refills:
            slot = refills.index(target)
            if not (taken >> slot) & 1:
                taken |= 1 << slot
                steps = layout.max_steps
                refilled = True
        kind = layout.cycler_at(target, moved)
        if kind == "shape":
            shape, cycled = (shape + 1) % names.SHAPE_COUNT, True
        elif kind == "color":
            color, cycled = (color + 1) % names.COLOR_COUNT, True
        elif kind == "rotation":
            rot, cycled = (rot + 1) % names.ROTATION_COUNT, True
        position = cell if rejected else target
        tick = tick if rejected else moved

    open_goals = [i for i in range(len(layout.goals)) if not (goals >> i) & 1]
    if rejected:
        return (position, shape, color, rot, goals, taken, steps, tick), "rejected"
    if layout.match_hint and cycled and any(
            (shape, color, rot) == layout.goals[i][1] for i in open_goals):
        return (position, shape, color, rot, goals, taken, steps, tick), "hint"

    if not refilled:
        steps -= layout.step_cost
    exhausted = steps < 0

    if not exhausted:
        for launcher in layout.launchers:
            if position in launcher["triggers"] and launcher["distance"] > 0:
                ddx, ddy = launcher["delta"]
                position = (position[0] + ddx * launcher["distance"],
                            position[1] + ddy * launcher["distance"])
                if position in layout.refills:
                    slot = refills.index(position)
                    if not (taken >> slot) & 1:
                        taken |= 1 << slot
                        steps = layout.max_steps
                kind = layout.cycler_at(position, tick)
                if kind == "shape":
                    shape = (shape + 1) % names.SHAPE_COUNT
                elif kind == "color":
                    color = (color + 1) % names.COLOR_COUNT
                elif kind == "rotation":
                    rot = (rot + 1) % names.ROTATION_COUNT
                return (position, shape, color, rot, goals, taken, steps, tick), "launched"

    index = layout.goal_at.get(position)
    if index is not None and not (goals >> index) & 1 and (shape, color, rot) == layout.goals[index][1]:
        goals |= 1 << index
    result = (position, shape, color, rot, goals, taken, steps, tick)
    if goals == (1 << len(layout.goals)) - 1:
        return result, "won"
    if exhausted:
        return result, "died"
    return result, "moved"


def advance(layout, state, action, refills):
    """The planner's view: `None` for anything that cannot make progress.

    Blocked moves burn budget and change nothing; rejected bumps are free but
    change nothing; running out of budget loses a life, which restores the level
    and so never helps. All three are safe to prune.
    """
    result, outcome = simulate(layout, state, action, refills)
    if outcome in ("rejected", "died") or result == state:
        return None
    return result


class Oracle:
    """Exhaustive shortest-action-count solver with a policy for every state.

    `_distance` holds the true minimum number of actions to finish from any
    reachable state, so `action_at` gives an optimal move off the solution path
    too. That is exactly what oracle-guided imitation needs.

    `limit` caps the reachable set; past it the search stops and `truncated` is
    set, which leaves `solution` possibly None and possibly not optimal. The
    patroller clock multiplies the space by `layout.tick_span`, so the three
    shipped rail levels are far bigger than a generated one (median ~140k):
    level 5 needs 847k states (14s, 0.5 GiB), level 6 needs 12.0M (129s,
    4.3 GiB) and level 7 needs 21.7M (121s, 6.8 GiB). The default is
    deliberately left at the generator's bounded search budget. Generators
    reject truncated drafts; a larger cap changes which drafts they accept.
    Planning the shipped levels 5-7 takes an explicit `limit` and several
    gigabytes of headroom.

    `engine` picks how the search runs: "fast" hands it to the C kernel in
    `fastplan` (same states, same edges, same distances, same truncation point,
    a lot less CPU), "reference" runs the pure-Python search below, and "auto"
    takes the kernel when it is available and falls back otherwise. `engine`
    is recorded on the instance as whichever one actually ran, with
    `fallback_reason` explaining an "auto" that ended up on the reference.
    Either way `_distance` maps planner-state tuples to optimal actions left.
    """

    def __init__(self, layout, limit=600_000, engine="auto"):
        if not layout.exact:
            raise Unplannable("layout uses mechanics this planner does not model")
        if engine not in ("auto", "fast", "reference"):
            raise ValueError("engine must be 'auto', 'fast' or 'reference'")
        self.layout = layout
        self.refills = _refill_order(layout)
        self.limit = limit
        self.full_mask = (1 << len(layout.goals)) - 1
        # The patroller clock starts at 0: upstream places every patroller at its
        # spawn with `_dir` 0 (ls20.py:1678-1681) and puts it back there on death.
        self.start = (layout.start_cell, *layout.start_triple, 0, 0, layout.max_steps, 0)
        self.fallback_reason = None
        if engine == "reference":
            self.engine = "reference"
            self._search()
            return
        reason = self._search_fast()
        if reason is None:
            self.engine = "fast"
        elif engine == "fast":
            raise RuntimeError(f"fast planner unavailable: {reason}")
        else:
            self.engine, self.fallback_reason = "reference", reason
            self._search()

    def _search_fast(self):
        """Run `fastplan`. Returns None on success, else why the reference must run."""
        if not fastplan.available():
            return fastplan.load_error()
        tables = fastplan.tables_for(self.layout, self.refills)
        if tables is None:
            return "layout does not fit the kernel's packed state"
        try:
            distance, reachable, truncated = fastplan.search(tables, self.start, self.limit)
        except (MemoryError, OSError, fastplan.Unsupported) as error:
            return str(error) or type(error).__name__
        self._distance, self._reachable, self.truncated = distance, reachable, truncated
        return None

    def _search(self):
        """The reference search: pure Python over state tuples. `fastplan` mirrors it."""
        layout, refills = self.layout, self.refills
        predecessors = {self.start: []}
        queue = deque([self.start])
        wins = []
        self.truncated = False
        while queue:
            state = queue.popleft()
            if state[4] == self.full_mask:
                wins.append(state)
                continue  # the level advances here; nothing follows
            for action in range(4):
                nxt = advance(layout, state, action, refills)
                if nxt is None:
                    continue
                if nxt not in predecessors:
                    if len(predecessors) >= self.limit:
                        self.truncated = True
                        queue.clear()
                        break
                    predecessors[nxt] = []
                    queue.append(nxt)
                predecessors[nxt].append(state)

        distance = {state: 0 for state in wins}
        frontier = deque(wins)
        while frontier:
            state = frontier.popleft()
            for previous in predecessors.get(state, ()):
                if previous not in distance:
                    distance[previous] = distance[state] + 1
                    frontier.append(previous)
        self._distance = distance
        self._reachable = len(predecessors)

    # -- results --------------------------------------------------------------

    @property
    def solvable(self):
        return self.start in self._distance

    @property
    def optimal_actions(self):
        """Fewest actions needed from the start, or None if unsolvable."""
        return self._distance.get(self.start)

    def action_for(self, state):
        """Best action index (0..3) from a planner state, or None if hopeless."""
        best, choice = self._distance.get(state), None
        if best is None:
            return None
        for action in range(4):
            nxt = advance(self.layout, state, action, self.refills)
            if nxt is not None and self._distance.get(nxt, 1 << 30) == best - 1:
                choice = action
                break
        return choice

    def distance_for(self, state):
        return self._distance.get(state)

    def solution(self, *, seed=None):
        """Optimal action ids; a seed randomizes ties reproducibly.

        Omitting the seed preserves the original first-action tie break.
        Seeded exports vary state visitation without changing optimal distances.
        """
        if not self.solvable:
            return None
        rng = random.Random(seed) if seed is not None else None
        state, path = self.start, []
        while state[4] != self.full_mask:
            if rng is None:
                action = self.action_for(state)
            else:
                remaining = self.distance_for(state)
                choices = []
                for candidate in range(4):
                    nxt = advance(self.layout, state, candidate, self.refills)
                    if nxt is not None and self.distance_for(nxt) == remaining - 1:
                        choices.append(candidate)
                action = rng.choice(choices) if choices else None
            if action is None:
                return None
            path.append(names.ACTION_IDS[action])
            state = advance(self.layout, state, action, self.refills)
        return path

    # -- live game ------------------------------------------------------------

    def state_of(self, env):
        """Map a running game onto a planner state."""
        present = {cell for cell in self.refills}
        for sprite in env.game.current_level.get_sprites():
            if sprite.tags and names.TAG_STEP_REFILL in sprite.tags:
                present.discard(names.pixel_to_cell(sprite.x, sprite.y))
        taken = sum(1 << i for i, cell in enumerate(self.refills) if cell in present)
        goals = sum(1 << i for i, solved in enumerate(env.goals_solved()) if solved)
        # The game keeps no move counter, so the patroller clock has to be read
        # back off the patrollers themselves. `phase_of` returns None only if the
        # game is not on this layout's level, where 0 is as good as anything.
        layout = self.layout
        tick = rails.phase_of(layout.patrollers, layout.tick_tail, layout.tick_period,
                              rails.live_states(env.game))
        return (env.player_cell(), *env.triple(), goals, taken, env.steps_left(),
                0 if tick is None else tick)

    def action_at(self, env):
        """Optimal action id (1..4) from the game's current state, or None."""
        action = self.action_for(self.state_of(env))
        return None if action is None else names.ACTION_IDS[action]


def oracle_for(env, **kwargs):
    return Oracle(extract(env), **kwargs)
