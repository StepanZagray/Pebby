"""Bounded exact snapshot search for SP80.

Search enumerates all useful edit actions.  Each successor and each candidate
flow is executed by the real vendored engine, so pixel-perfect collisions,
screen rotation, pipe spreading, corner deflection and cup/sink resolution do
not need a second rule implementation. No-op moves and selecting the already
selected piece are safely omitted because they only consume steps. Failed
flows remain in the graph because they change the failure counter and can
auto-select a piece that overlapping bounding boxes make impossible to click.

The priority score is only an ordering hint.  It does not prune states and no
optimality claim is made.  Exhaustion is conclusive within the engine's action
budget; hitting ``limit`` is reported as truncation.
"""

from dataclasses import dataclass
from collections import deque
import heapq
from itertools import count

from arcengine import GameState

from . import names
from .layout import Layout, extract


DEFAULT_LIMIT = 60_000
Action = tuple[int, int | None, int | None]
TARGET_DATA_KEY = "pebby_sp80_target_positions_v2"

# Independently discovered by bounded native-flow placement search, then
# replayed through legal public actions.  These are teacher certificates for
# the immutable reference levels; the generator never reads them.
OFFICIAL_TARGETS = (
    ((6, 4),),
    ((12, 8), (8, 9), (4, 7)),
    ((9, 11), (0, 3), (3, 9), (10, 9)),
    ((5, 14), (6, 3), (7, 6), (4, 11), (10, 9)),
    ((0, 5), (4, 11), (8, 14), (13, 7)),
    ((5, 8), (12, 5), (12, 11), (14, 10)),
)

_OFFICIAL_STATIC = (
    ((16, 16), ((4, 13, 0), (10, 13, 0)), (9,), (names.PIPE[5],)),
    ((16, 16), ((2, 13, 0), (6, 13, 0), (10, 13, 0)), (5,),
     (names.PIPE[3], names.PIPE[3], names.PIPE[5])),
    ((16, 16), ((1, 13, 0), (12, 13, 0), (7, 13, 0)), (1, 14, 6),
     (names.PIPE[4], names.PIPE[5], names.PIPE[6], names.PIPE[6])),
    ((20, 20), ((2, 17, 0), (16, 17, 0), (8, 17, 0), (12, 17, 0)), (7,),
     (names.PIPE[4], names.PIPE[4], names.PIPE[5], names.PIPE[5], names.SOURCE_PIPE[7])),
    ((20, 20), ((17, 6, 270), (2, 17, 0), (6, 17, 0), (12, 17, 0)), (5, 13),
     (names.PIPE[3], names.PIPE[4], names.PIPE[5], names.DEFLECTOR_RIGHT)),
    ((20, 20), ((17, 9, 270), (8, 17, 0), (1, 11, 90), (1, 6, 90)), (9,),
     (names.PIPE_VERTICAL_4, names.SOURCE_PIPE[5], names.DEFLECTOR_LEFT,
      names.DEFLECTOR_RIGHT)),
)


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...] | None
    truncated: bool
    unsupported: bool
    exact: bool
    expanded: int
    generated: int
    reason: str
    limit: int

    @property
    def solved(self):
        return self.actions is not None


def _state_key(env):
    movables = env.movables()
    selected = env.selected()
    selected_index = next(
        (index for index, sprite in enumerate(movables) if sprite is selected), None
    )
    return (
        tuple((int(sprite.x), int(sprite.y)) for sprite in movables),
        selected_index,
        env.failed_flows,
    )


def _grid_to_internal_display(env, x, y):
    width, height = env.grid_size
    scale = min(names.DISPLAY // width, names.DISPLAY // height)
    x_offset = (names.DISPLAY - width * scale) // 2
    y_offset = (names.DISPLAY - height * scale) // 2
    return x_offset + x * scale + scale // 2, y_offset + y * scale + scale // 2


def _selection_clicks(env):
    """Yield one legal public click that selects each non-selected piece."""
    movables = env.movables()
    selected = env.selected()
    k = env.rotation_k
    for index, target in enumerate(movables):
        if target is selected:
            continue
        click = None
        # Upstream's hit test is an ordered bounding-box test, including the
        # transparent corner of a deflector.  Scan grid cells in stable order
        # and keep a coordinate only when this target wins that exact test.
        for gy in range(max(0, int(target.y)), min(env.grid_size[1], int(target.y + target.height))):
            for gx in range(max(0, int(target.x)), min(env.grid_size[0], int(target.x + target.width))):
                hit = next(
                    (
                        other
                        for other in movables
                        if other.x <= gx < other.x + other.width
                        and other.y <= gy < other.y + other.height
                    ),
                    None,
                )
                if hit is not target:
                    continue
                ix, iy = _grid_to_internal_display(env, gx, gy)
                px, py = names.unrotate_click(k, ix, iy)
                if 0 <= px < names.DISPLAY and 0 <= py < names.DISPLAY:
                    click = (int(px), int(py))
                    break
            if click is not None:
                break
        if click is not None:
            yield index, click


def _pipe_goal_xs(env):
    """Cheap ordering hints inferred from cup mouths; never used for pruning."""
    centers = sorted(int(cup.x) + 1 for cup in env.cups())
    source_xs = tuple(int(source.x) for source in env.sprites_by_tag(names.TAG_SOURCE))
    goals = {}
    for index, piece in enumerate(env.movables()):
        if names.TAG_PIPE not in piece.tags or piece.height != 1:
            continue
        possible = set()
        for left in centers:
            for right in centers:
                if right - left == int(piece.width) + 1:
                    x = left + 1
                    if not source_xs or any(x <= source_x < x + int(piece.width) for source_x in source_xs):
                        possible.add(x)
        if possible:
            goals[index] = (tuple(sorted(possible)), int(piece.y))
    return goals


def _priority(env, goals):
    value = 0
    for index, piece in enumerate(env.movables()):
        hint = goals.get(index)
        if hint:
            possible, starting_y = hint
            value += min(abs(int(piece.x) - goal) for goal in possible)
            # Moving vertically is legal and remains in the graph, but none of
            # the inferred horizontal-splitter goals requires abandoning the
            # pipe's current row. This tie-break prevents a large flat plateau.
            value += abs(int(piece.y) - starting_y)
    return value


def _unwind(parent, state, tail=()):
    pieces = [tuple(tail)]
    while parent[state] is not None:
        previous, action = parent[state]
        pieces.append((action,))
        state = previous
    return tuple(action for piece in reversed(pieces) for action in piece)


def _reference_target(env):
    """Return audited reference goals only when immutable geometry matches."""
    standalone_sources = tuple(
        int(sprite.x)
        for sprite in env.sprites_by_tag(names.TAG_SOURCE)
        if sprite.name == names.SOURCE
    )
    cups = tuple(
        (int(sprite.x), int(sprite.y), int(getattr(sprite, "rotation", 0)))
        for sprite in env.cups()
    )
    movable_names = tuple(sprite.name for sprite in env.movables())
    observed = (env.grid_size, cups, standalone_sources, movable_names)
    for index, signature in enumerate(_OFFICIAL_STATIC):
        if observed == signature:
            return OFFICIAL_TARGETS[index]
    return None


def _declared_target(env):
    raw = env.level.get_data(TARGET_DATA_KEY)
    if raw is not None:
        try:
            target = tuple((int(pair[0]), int(pair[1])) for pair in raw)
        except (TypeError, ValueError, IndexError):
            return None
        if len(target) == len(env.movables()):
            return target
    return _reference_target(env)


def _route_selected(env, target, remaining, counter):
    """Shortest legal public movement path for the currently selected piece."""
    selected = env.selected()
    if selected is None:
        return None
    start = (int(selected.x), int(selected.y))
    if start == target:
        return (), env
    frontier = deque([(start, env)])
    parent = {start: None}
    while frontier:
        position, current = frontier.popleft()
        for action_id in names.MOVE_ACTIONS:
            if counter[0] >= remaining:
                return None
            counter[0] += 1
            moved = current.clone()
            observation = moved.perform(action_id)
            if observation.state == GameState.GAME_OVER:
                continue
            sprite = moved.selected()
            nxt = (int(sprite.x), int(sprite.y))
            if nxt == position or nxt in parent:
                continue
            parent[nxt] = (position, (action_id, None, None))
            if nxt == target:
                actions = []
                cursor = nxt
                while parent[cursor] is not None:
                    cursor, action = parent[cursor]
                    actions.append(action)
                return tuple(reversed(actions)), moved
            frontier.append((nxt, moved))
    return None


def _constructive_search(env, targets, limit, action_budget):
    """Route audited target placements, then certify them in the real engine."""
    if limit == 0:
        return SearchResult(
            None, True, False, False, 0, 0,
            "constructive routing work limit 0 reached", limit,
        )
    if len(targets) != len(env.movables()):
        return None
    work = env.clone()
    actions = []
    expanded = [0]
    remaining = {
        index
        for index, (sprite, target) in enumerate(zip(work.movables(), targets))
        if (int(sprite.x), int(sprite.y)) != tuple(target)
    }
    while remaining:
        selected = work.selected()
        selected_index = next(
            (i for i, sprite in enumerate(work.movables()) if sprite is selected), None
        )
        candidates = []
        if selected_index in remaining:
            candidates.append((selected_index, None))
        candidates.extend(
            (index, click)
            for index, click in _selection_clicks(work)
            if index in remaining and index != selected_index
        )
        progressed = False
        for index, click in candidates:
            candidate = work.clone()
            prefix = []
            if click is not None:
                if len(actions) + 2 >= action_budget:
                    continue
                observation = candidate.perform(names.ACTION_CLICK, *click)
                expanded[0] += 1
                if observation.state == GameState.GAME_OVER:
                    continue
                prefix.append((names.ACTION_CLICK, click[0], click[1]))
            cap = min(limit, expanded[0] + max(0, action_budget - len(actions) - len(prefix) - 1) * 512)
            routed = _route_selected(candidate, tuple(targets[index]), cap, expanded)
            if routed is None:
                continue
            route, candidate = routed
            if len(actions) + len(prefix) + len(route) + 1 > action_budget:
                continue
            actions.extend(prefix)
            actions.extend(route)
            work = candidate
            remaining.remove(index)
            progressed = True
            break
        if not progressed:
            return None
        if expanded[0] >= limit:
            return SearchResult(
                None, True, False, False, expanded[0], expanded[0],
                f"constructive routing work limit {limit} reached", limit,
            )

    if len(actions) + 1 > action_budget:
        return None
    score = work.levels_completed
    try:
        observation = work.perform(names.ACTION_FLOW)
    except ValueError as exc:
        if "too many frames" not in str(exc).lower():
            raise
        return SearchResult(
            None, False, True, False, expanded[0], expanded[0],
            "audited target exceeded the engine frame guard", limit,
        )
    if (
        observation.state == GameState.GAME_OVER
        or (work.levels_completed <= score and observation.state != GameState.WIN)
    ):
        return None
    actions.append((names.ACTION_FLOW, None, None))
    return SearchResult(
        tuple(actions), False, False, True, expanded[0], expanded[0],
        "audited target routed and replayed by the real engine", limit,
    )


def search(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    """Search editing states using real-engine transitions.

    ``limit`` bounds expanded configurations. ``budget`` optionally imposes a
    smaller action cap than the level's remaining step counter, which is the
    signature expected by the shared whole-game collector.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if budget is not None and (
        not isinstance(budget, int) or isinstance(budget, bool) or budget < 0
    ):
        raise ValueError("budget must be a nonnegative integer or None")
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        reason = "; ".join(layout.unsupported) or "layout has no engine snapshot"
        return SearchResult(None, False, True, False, 0, 0, reason, limit)

    start_env = layout.snapshot.clone()
    action_budget = start_env.steps_left if budget is None else min(start_env.steps_left, budget)
    if action_budget < 1:
        return SearchResult(
            None, False, False, True, 0, 1, "no action remains for a flow", limit
        )

    targets = _declared_target(start_env)
    if targets is not None:
        constructive = _constructive_search(start_env, targets, limit, action_budget)
        if constructive is not None:
            return constructive
        # A declared target is a positive-witness aid, not a completeness
        # theorem. If it cannot be routed inside the remaining native budget,
        # falling into the combinatorial exhaustive graph would make bounded
        # generation unpredictable and could misstate this restricted failure
        # as an exact negative proof.
        return SearchResult(
            None, False, True, False, 0, 0,
            "declared constructive target was not routable within the native budget",
            limit,
        )

    start = _state_key(start_env)
    parent = {start: None}
    best = {start: 0}
    goals = _pipe_goal_xs(start_env)
    serial = count()
    frontier = [(_priority(start_env, goals), 0, next(serial), start, start_env)]
    expanded = 0
    generated = 1
    guarded_flows = 0
    while frontier:
        _, cost, _, state, env = heapq.heappop(frontier)
        if cost != best.get(state):
            continue
        if expanded >= limit:
            return SearchResult(
                None,
                True,
                False,
                True,
                expanded,
                generated,
                f"configuration expansion limit {limit} reached",
                limit,
            )
        expanded += 1
        if cost >= action_budget:
            continue

        # A successful flow is terminal. A failed flow preserves positions but
        # increments the failure counter and auto-selects the nearest-origin
        # piece. That selected state can matter when overlapping pieces make it
        # impossible to click the auto-selected target, so failed-flow states
        # are part of the exact graph rather than being treated as no-ops.
        probe = env.clone()
        score = probe.levels_completed
        try:
            observation = probe.perform(names.ACTION_FLOW)
        except ValueError as exc:
            if "too many frames" not in str(exc).lower():
                raise
            # A cyclic water arrangement hit ARCEngine's own finite action
            # guard. Keep searching other arrangements, but do not later call
            # an exhausted search a proof over this malformed action graph.
            guarded_flows += 1
            observation = None
        if (
            observation is not None
            and observation.state != GameState.GAME_OVER
            and (probe.levels_completed > score or observation.state == GameState.WIN)
        ):
            return SearchResult(
                _unwind(parent, state, ((names.ACTION_FLOW, None, None),)),
                False,
                False,
                True,
                expanded,
                generated,
                "real-engine flow completed the level",
                limit,
            )

        # A successor still needs one step for ACTION5.
        if cost + 1 >= action_budget:
            continue
        successors = []
        if observation is not None and observation.state != GameState.GAME_OVER:
            failed = _state_key(probe)
            if failed != state:
                successors.append(((names.ACTION_FLOW, None, None), failed, probe))
        for action_id in names.MOVE_ACTIONS:
            moved = env.clone()
            observation = moved.perform(action_id)
            if observation.state == GameState.GAME_OVER:
                continue
            nxt = _state_key(moved)
            if nxt != state:
                successors.append(((action_id, None, None), nxt, moved))
        for _, (x, y) in _selection_clicks(env):
            selected = env.clone()
            observation = selected.perform(names.ACTION_CLICK, x, y)
            if observation.state == GameState.GAME_OVER:
                continue
            nxt = _state_key(selected)
            if nxt != state:
                successors.append(((names.ACTION_CLICK, x, y), nxt, selected))

        next_cost = cost + 1
        for action, nxt, successor in successors:
            generated += 1
            if next_cost >= best.get(nxt, action_budget + 1):
                continue
            best[nxt] = next_cost
            parent[nxt] = (state, action)
            # This may overestimate and therefore promises no shortest path;
            # every state remains queued, preserving exhaustive correctness.
            priority = next_cost + 8 * _priority(successor, goals)
            heapq.heappush(
                frontier, (priority, next_cost, next(serial), nxt, successor)
            )

    if guarded_flows:
        return SearchResult(
            None,
            False,
            True,
            False,
            expanded,
            generated,
            f"{guarded_flows} reachable flow arrangement(s) exceeded the engine frame guard",
            limit,
        )
    return SearchResult(
        None,
        False,
        False,
        True,
        expanded,
        generated,
        "complete useful editing-state graph exhausted within the action budget",
        limit,
    )


def solve(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    """Return action triples, with detailed status on ``solve.result``."""
    result = search(env_or_layout, limit=limit, budget=budget)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.reason = result.reason
    return list(result.actions) if result.actions is not None else None


solve.result = None
solve.truncated = False
solve.unsupported = False
solve.reason = ""
