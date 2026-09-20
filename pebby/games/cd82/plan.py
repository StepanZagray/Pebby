"""An exact CD82 planner.

Search runs over the game's logical state -- dial position, selected colour,
colour per atom -- with *macro* edges: "select colour c (0 or 1 click), turn
the dial to d (ring distance presses), then paint". Between two paints nothing
else can usefully happen (a wasted click or a no-op move only costs), and dial
turning and colour selection do not interact, so the macro cost is exactly the
fewest actions between consecutive paints. A* with an admissible heuristic
therefore returns an action-optimal plan; `limit` bounds expansions and sets
`truncated` when exceeded, so a None result is only a proof of unsolvability
when `truncated` is False.
"""

import heapq
import math

from . import names
from .layout import Layout, extract


class Result:
    """Outcome of one search."""

    __slots__ = ("actions", "truncated", "expanded", "reason")

    def __init__(self, actions=None, truncated=False, expanded=0, reason=""):
        self.actions = actions
        self.truncated = truncated
        self.expanded = expanded
        self.reason = reason

    @property
    def solved(self):
        return self.actions is not None

    @property
    def length(self):
        return None if self.actions is None else len(self.actions)


def _heuristic(layout, atoms, color, cover_max):
    """Admissible lower bound on remaining actions.

    Every mismatched atom must receive a final paint in its target colour, and a
    single paint colours at most `cover_max` atoms, so per target colour at
    least ceil(missing / cover_max) paints remain; each target colour other than
    the selected one also needs a swatch click.
    """
    missing = {}
    for atom, want in enumerate(layout.target):
        if atoms[atom] != want:
            missing[want] = missing.get(want, 0) + 1
    if not missing:
        return 0
    paints = sum(math.ceil(n / cover_max) for n in missing.values())
    clicks = len(missing) - (1 if color in missing else 0)
    return paints + clicks


def search(env_or_layout, limit=200_000):
    """Exact A* over macro edges. Returns a Result."""
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if layout.target is None:
        return Result(reason="target is not uniform per atom; no paint sequence can match it")
    if layout.actions_left <= 0:
        return Result(reason="no actions left in the game budget")
    palette = {c: (x, y) for c, x, y in layout.palette}
    if not palette:
        return Result(reason="level has no swatches")
    cover_max = max(len(atoms) for _, _, atoms in layout.ops)
    start = layout.start
    if start[2] == layout.target:
        return Result(actions=[], reason="already matches; the game only checks after a paint")

    # Colours the plan may select: the swatches, plus the current colour (free).
    colours = set(palette) | {start[1]}
    goal = layout.target
    h0 = _heuristic(layout, start[2], start[1], cover_max)
    frontier = [(h0, 0, 0, start)]
    best = {start: 0}
    parent = {start: None}
    expanded = 0
    counter = 1
    while frontier:
        f, g, _, state = heapq.heappop(frontier)
        if g > best.get(state, math.inf):
            continue
        if state[2] == goal:
            return Result(actions=_unwind(layout, palette, parent, state), expanded=expanded)
        expanded += 1
        if expanded > limit:
            return Result(truncated=True, expanded=expanded, reason="expansion limit reached")
        dial, color, atoms = state
        for op_index, (target_dial, kind, region) in enumerate(layout.ops):
            move_cost = names.ring_distance(dial, target_dial)
            for c in colours:
                if c != color and c not in palette:
                    continue
                if all(atoms[a] == c for a in region):
                    continue  # a paint that changes nothing is never useful
                new_atoms = list(atoms)
                for a in region:
                    new_atoms[a] = c
                new_atoms = tuple(new_atoms)
                cost = move_cost + (c != color) + 1
                ng = g + cost
                if ng > layout.actions_left:
                    continue
                nxt = (target_dial, c, new_atoms)
                if ng < best.get(nxt, math.inf):
                    best[nxt] = ng
                    parent[nxt] = (state, op_index, c)
                    h = _heuristic(layout, new_atoms, c, cover_max)
                    heapq.heappush(frontier, (ng + h, ng, counter, nxt))
                    counter += 1
    return Result(expanded=expanded, reason="exhausted: no paint sequence fits the budget")


def _unwind(layout, palette, parent, state):
    steps = []
    while parent[state] is not None:
        prev, op_index, c = parent[state]
        steps.append((prev, op_index, c))
        state = prev
    steps.reverse()
    actions = []
    for (dial, color, _), op_index, c in steps:
        if c != color:
            x, y = palette[c]
            actions.append((names.ACTION_CLICK, x, y))
        target_dial, kind, _ = layout.ops[op_index]
        actions.extend((a, None, None) for a in names.ring_path(dial, target_dial))
        if kind == "paint":
            actions.append((names.ACTION_PAINT, None, None))
        else:
            x, y = names.indicator_click(target_dial)
            actions.append((names.ACTION_CLICK, x, y))
    return actions


def solve(env_or_layout, limit=200_000):
    """A list of (action_id, x, y) completing the current level, or None.

    Exact: None with `solve.truncated == False` means no sequence within the
    remaining budget completes the level. `solve.truncated` and `solve.result`
    describe the most recent call.
    """
    result = search(env_or_layout, limit)
    solve.truncated = result.truncated
    solve.result = result
    return result.actions


solve.truncated = False
solve.result = None
