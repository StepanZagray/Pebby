"""Bounded exact planning for LF52's complete shipped logical ruleset.

The search graph uses native macro-actions: one rail-arrow action or one legal
peg jump (source click plus landing click).  It models entity-stack landing
rules, coloured/non-removing jumps, moving holes, camera shifts, scripted
reset landings, and the three native survivor conditions.  Positive witnesses
are accepted only after replay on a cloned real-engine state.
"""

from collections import Counter, deque
from dataclasses import dataclass
import heapq

from . import names
from .layout import Layout, extract


DEFAULT_LIMIT = 500_000
Action = tuple[int, int | None, int | None]
Cell = tuple[int, int]
PegEntity = tuple[Cell, str]
State = tuple[tuple[PegEntity, ...], frozenset[Cell], frozenset[Cell], tuple[int, int]]

_ARROWS = (
    (names.ACTION_UP, 0, -1),
    (names.ACTION_DOWN, 0, 1),
    (names.ACTION_LEFT, -1, 0),
    (names.ACTION_RIGHT, 1, 0),
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
    mechanic_use: tuple[tuple[str, int], ...] = ()

    @property
    def solved(self):
        return self.actions is not None

    def metadata(self):
        return {
            "solved": self.solved,
            "truncated": self.truncated,
            "unsupported": self.unsupported,
            "exact": self.exact,
            "expanded": self.expanded,
            "generated": self.generated,
            "reason": self.reason,
            "limit": self.limit,
            "solution_length": None if self.actions is None else len(self.actions),
            "mechanic_use": dict(self.mechanic_use),
        }


def _state(layout):
    return (
        tuple(sorted(layout.peg_entities)),
        layout.moving_cells,
        layout.obstacles,
        layout.origin,
    )


def _peg_map(state):
    return dict(state[0])


def _visible(layout, origin, cell):
    return layout.visible(cell, origin)


def _landing_open(layout, state, cell):
    pegs, moving, obstacles, _ = state
    peg_cells = {position for position, _ in pegs}
    count = (
        int(cell in layout.ordinary_cells)
        + int(cell in moving)
        + int(cell in layout.rails)
        + int(cell in peg_cells)
        + int(cell in obstacles)
    )
    return (
        cell in layout.ordinary_cells and count == 1
    ) or (cell in moving and count == 2)


def _jump_options(layout, state):
    pegs, _, obstacles, origin = state
    peg_map = dict(pegs)
    occupied = set(peg_map) | set(obstacles)
    for source, source_kind in pegs:
        if not _visible(layout, origin, source):
            continue
        for dx, dy in names.DIRECTIONS:
            middle = source[0] + dx, source[1] + dy
            destination = source[0] + 2 * dx, source[1] + 2 * dy
            if (
                middle in occupied
                and _landing_open(layout, state, destination)
                and _visible(layout, origin, destination)
            ):
                yield source, source_kind, middle, destination


def _scripted_reset(layout, source_kind, destination, pegs_after):
    level = layout.logical_level
    if level == 1:
        return destination in {(0, 2), (2, 2), (5, 1)}
    if level == 2:
        return destination in {(0, 1), (2, 1), (4, 1)}
    if level == 3:
        if destination in {(1, 0), (0, 3), (10, 0), (13, 2), (13, 4)}:
            return True
        if destination == (10, 2) and len(pegs_after) <= 5:
            return True
        if destination == (10, 4) and len(pegs_after) <= 4:
            return True
    if level == 6 and destination == (16, 2) and source_kind == names.PEG:
        return dict(pegs_after).get((6, 6)) == names.PEG_RED
    return False


def _landing_scroll(level, destination, origin):
    delta = (0, 0)
    if level == 4:
        delta = {(7, 3): (-30, 0), (12, 7): (15, -30)}.get(destination, delta)
    elif level == 5:
        delta = {(7, 3): (-18, 0), (12, 3): (-33, 0)}.get(destination, delta)
    elif level == 6:
        if destination == (7, 6) and origin == (5, 5):
            delta = (-20, 0)
        elif destination == (18, 2) and origin == (-57, 5):
            delta = (-44, 0)
    elif level == 7:
        if destination == (8, 8) and origin == (5, 5):
            delta = (-44, 0)
        elif destination[0] == 16 and origin == (-39, 5):
            delta = (-40, 0)
    elif level == 9 and destination == (6, 5):
        delta = (-20, 0)
    return origin[0] + delta[0], origin[1] + delta[1]


def _goal_count(layout, pegs):
    count = len(pegs)
    if layout.logical_level >= 8:
        count -= layout.initial_blue_count
    return count


def _jump_transition(layout, state, option):
    source, source_kind, middle, destination = option
    pegs, moving, obstacles, origin = state
    peg_map = dict(pegs)
    middle_kind = peg_map.get(middle)
    removed = middle_kind == source_kind and middle_kind != names.PEG_BLUE
    if removed:
        del peg_map[middle]
    del peg_map[source]
    peg_map[destination] = source_kind
    pegs_after = tuple(sorted(peg_map.items()))
    reset = _scripted_reset(layout, source_kind, destination, pegs_after)
    new_origin = _landing_scroll(layout.logical_level, destination, origin)
    event = Counter(jumps=1)
    if removed:
        event["same_color_removals"] += 1
    else:
        event["non_removing_jumps"] += 1
    if middle in obstacles:
        event["blocker_jumps"] += 1
    elif middle_kind is not None and middle_kind != source_kind:
        event["cross_color_jumps"] += 1
    if source_kind == names.PEG_BLUE or middle_kind == names.PEG_BLUE:
        event["blue_peg_interactions"] += 1
    if reset:
        event["scripted_reset_landings"] += 1
    if new_origin != origin:
        event["camera_scrolls"] += 1
    won = _goal_count(layout, pegs_after) == layout.target_nonblue
    return (pegs_after, moving, obstacles, new_origin), won, reset, event


def _rail_scroll(level, dx, dy):
    if level == 3:
        return -dx * 8, 0
    if level in (5, 6):
        return -dx * 6, 0
    if level == 8:
        return 0, -dy * 6
    if level == 9:
        return -dx * 6, -dy * 6
    return 0, 0


def _rail_transition(layout, state, dx, dy):
    pegs_tuple, moving_frozen, obstacles_frozen, origin = state
    pegs = dict(pegs_tuple)
    moving = set(moving_frozen)
    obstacles = set(obstacles_frozen)
    ordered = sorted(
        moving,
        key=(lambda cell: cell[0]) if dx else (lambda cell: cell[1]),
        reverse=(dx > 0 or dy > 0),
    )
    scrolled = False
    moved = 0
    for source in ordered:
        if source not in moving:
            continue
        destination = source[0] + dx, source[1] + dy
        if destination in moving or destination not in layout.rails:
            continue
        moving.remove(source)
        moving.add(destination)
        moved += 1
        peg_kind = pegs.pop(source, None)
        if peg_kind is not None:
            pegs[destination] = peg_kind
        if source in obstacles:
            obstacles.remove(source)
            obstacles.add(destination)
        if peg_kind == names.PEG and not scrolled:
            if layout.logical_level == 4:
                # In the source's elif-chain, level 4 cells below row 11
                # select the generic final ``else`` scroll; row 11+ is the
                # only explicitly suppressed branch.
                delta = (0, 0) if source[1] >= 11 else (-dx * 6, -dy * 6)
            else:
                delta = _rail_scroll(layout.logical_level, dx, dy)
            if delta != (0, 0):
                scrolled = True
                # The source returns before scheduling the camera animation
                # when a right-shift would move an already-positive camera.
                # The yellow cell itself has moved, and later cells in the
                # same arrow action are deliberately not processed.
                if layout.logical_level != 5 and origin[0] >= 5 and delta[0] > 0:
                    break
                origin = origin[0] + delta[0], origin[1] + delta[1]
    nxt = (tuple(sorted(pegs.items())), frozenset(moving), frozenset(obstacles), origin)
    event = Counter()
    if moved:
        event.update(rail_actions=1, moving_hole_moves=moved)
    if scrolled and nxt[3] != state[3]:
        event["camera_scrolls"] += 1
    return nxt, event


def _unwind(parent, state):
    actions = []
    events = Counter()
    while parent[state] is not None:
        previous, edge_actions, edge_events = parent[state]
        actions.append(edge_actions)
        events.update(dict(edge_events))
        state = previous
    actions.reverse()
    return tuple(action for edge in actions for action in edge), events


def _verify_live_witness(env, actions):
    from .env import replay

    try:
        snapshot = env.clone()
        start_score = snapshot.levels_completed
        completed, observation = replay(snapshot, actions)
    except Exception as exc:
        return f"real-engine replay raised {type(exc).__name__}: {exc}"
    if not completed or observation is None or snapshot.levels_completed <= start_score:
        return "real-engine replay did not complete the current level"
    return None


def _negative(layout, truncated, expanded, generated, reason, node_limit):
    exact = not truncated and layout.generated_kind in (
        names.GENERATED_KIND,
        names.FULL_GENERATED_KIND,
    ) and layout.history_depth == 0
    return SearchResult(
        None,
        truncated,
        False,
        exact,
        expanded,
        generated,
        reason,
        node_limit,
    )


class _GuidedFailure(Exception):
    """The snapshot does not match a shipped teacher waypoint."""


class _GuidedCutoff(Exception):
    """The cumulative teacher waypoint search reached its work bound."""


class _TeacherGuide:
    """Exact rail/jump waypoint search for the two largest shipped teachers.

    Levels 7 and 10 are transport puzzles whose unconstrained rail-position
    product overwhelms a useful generic best-first bound.  The guide searches
    each rail configuration between source-derived jump waypoints; it does not
    store an action route.  Every resulting route is still replayed by the
    native engine before it becomes a positive certificate.
    """

    def __init__(self, env, layout, remaining, node_limit):
        self.env = env
        self.layout = layout
        self.remaining = remaining
        self.node_limit = node_limit
        self.state = _state(layout)
        self.actions = []
        self.events = Counter()
        self.expanded = 0
        self.generated = 1

    def _expand(self):
        if self.expanded >= self.node_limit:
            raise _GuidedCutoff
        self.expanded += 1

    def _room(self, extra):
        if len(self.actions) + extra > self.remaining:
            raise _GuidedFailure("teacher waypoint exceeds the native action budget")

    def rail_until(self, predicate):
        if predicate(self.state):
            return
        queue = deque([self.state])
        parent = {self.state: None}
        edge = {}
        depth = {self.state: 0}
        found = None
        while queue:
            state = queue.popleft()
            self._expand()
            for action, dx, dy in _ARROWS:
                nxt, event = _rail_transition(self.layout, state, dx, dy)
                if nxt == state or nxt in parent:
                    continue
                next_depth = depth[state] + 1
                if len(self.actions) + next_depth > self.remaining:
                    continue
                parent[nxt] = state
                edge[nxt] = action, dx, dy, tuple(event.items())
                depth[nxt] = next_depth
                self.generated += 1
                if predicate(nxt):
                    found = nxt
                    queue.clear()
                    break
                queue.append(nxt)
        if found is None:
            raise _GuidedFailure("teacher rail waypoint is unreachable")
        path = []
        cursor = found
        while parent[cursor] is not None:
            path.append(edge[cursor])
            cursor = parent[cursor]
        path.reverse()
        for action, _, _, event in path:
            self.actions.append((action, None, None))
            self.events.update(dict(event))
        self.state = found

    def _take(self, option):
        self._room(2)
        source, _, _, destination = option
        self.actions.extend(
            (
                (names.ACTION_CLICK, *self.layout.click(source, self.state[3])),
                (names.ACTION_CLICK, *self.layout.click(destination, self.state[3])),
            )
        )
        self.state, won, reset, event = _jump_transition(self.layout, self.state, option)
        self.events.update(event)
        self.generated += 1
        if reset:
            raise _GuidedFailure("teacher waypoint reached a scripted reset")
        return won

    def jump(self, source, destination, kind=None):
        options = [
            option
            for option in _jump_options(self.layout, self.state)
            if option[0] == source
            and option[3] == destination
            and (kind is None or option[1] == kind)
        ]
        if not options:
            raise _GuidedFailure(f"teacher jump {source}->{destination} is unavailable")
        return self._take(options[0])

    def jump_search_to_win(self):
        queue = deque([self.state])
        parent = {self.state: None}
        edge = {}
        depth = {self.state: 0}
        goal_state = None
        goal_option = None
        while queue and goal_state is None:
            state = queue.popleft()
            self._expand()
            for option in _jump_options(self.layout, state):
                nxt, won, reset, _ = _jump_transition(self.layout, state, option)
                if reset:
                    continue
                if len(self.actions) + (depth[state] + 1) * 2 > self.remaining:
                    continue
                if won:
                    goal_state = state
                    goal_option = option
                    break
                if nxt not in parent:
                    parent[nxt] = state
                    edge[nxt] = option
                    depth[nxt] = depth[state] + 1
                    self.generated += 1
                    queue.append(nxt)
        if goal_state is None:
            raise _GuidedFailure("teacher board waypoint graph has no winning transition")
        path = []
        cursor = goal_state
        while parent[cursor] is not None:
            path.append(edge[cursor])
            cursor = parent[cursor]
        path.reverse()
        for option in path:
            if self._take(option):
                raise _GuidedFailure("teacher path won before its recorded goal edge")
        if goal_option is None or not self._take(goal_option):
            raise _GuidedFailure("teacher goal edge did not win")

    def result(self):
        actions = tuple(self.actions)
        mismatch = _verify_live_witness(self.env, actions)
        if mismatch is not None:
            return SearchResult(
                None, False, True, False, self.expanded, self.generated,
                mismatch, self.node_limit, tuple(sorted(self.events.items())),
            )
        return SearchResult(
            actions, False, False, True, self.expanded, self.generated,
            "solved by exact teacher waypoints and replayed on a cloned real-engine snapshot",
            self.node_limit, tuple(sorted(self.events.items())),
        )


def _shipped_teacher(layout, level):
    expected = {
        7: (
            (((0, 1), names.PEG), ((6, 1), names.PEG_RED), ((22, 6), names.PEG)),
            (5, 5),
        ),
        10: (
            (
                ((2, 8), names.PEG_BLUE), ((3, 6), names.PEG_BLUE),
                ((4, 0), names.PEG), ((4, 7), names.PEG_BLUE),
                ((5, 5), names.PEG_BLUE), ((6, 5), names.PEG_BLUE),
                ((6, 9), names.PEG), ((8, 9), names.PEG_BLUE),
                ((8, 10), names.PEG_BLUE), ((8, 11), names.PEG_BLUE),
                ((8, 12), names.PEG_BLUE), ((8, 13), names.PEG_BLUE),
            ),
            (5, 3),
        ),
    }
    pegs, origin = expected[level]
    return (
        layout.generated_kind is None
        and layout.logical_level == level
        and layout.action_count == 0
        and layout.history_depth == 0
        and layout.selected is None
        and layout.peg_entities == pegs
        and layout.origin == origin
    )


def _guided_level_seven(guide):
    peg = names.PEG
    red = names.PEG_RED
    guide.rail_until(lambda state: any(option[0] == (0, 1) and option[3] == (0, 3) for option in _jump_options(guide.layout, state)))
    guide.jump((0, 1), (0, 3), peg)
    guide.rail_until(lambda state: any(option[0] == (1, 6) and option[3] == (1, 8) for option in _jump_options(guide.layout, state)))
    guide.jump((1, 6), (1, 8), peg)
    guide.jump((1, 8), (3, 8), peg)
    guide.jump((3, 8), (5, 8), peg)
    guide.rail_until(lambda state: any(option[0] == (6, 1) and option[3] == (6, 3) for option in _jump_options(guide.layout, state)))
    guide.jump((6, 1), (6, 3), red)
    guide.rail_until(lambda state: any(option[0] == (6, 6) and option[3] == (6, 8) for option in _jump_options(guide.layout, state)))
    guide.jump((6, 6), (6, 8), red)
    guide.jump((5, 8), (7, 8), peg)
    guide.jump((6, 8), (8, 8), red)
    guide.rail_until(lambda state: (10, 8) in state[1] and (10, 8) not in dict(state[0]) and (10, 8) not in state[2])
    guide.jump((7, 8), (9, 8), peg)
    guide.jump((8, 8), (10, 8), red)
    guide.jump((9, 8), (11, 8), peg)
    guide.rail_until(lambda state: any(option[0] == (12, 3) and option[3] == (14, 3) and option[1] == red for option in _jump_options(guide.layout, state)))
    guide.jump((12, 3), (14, 3), red)
    guide.rail_until(lambda state: any(option[0] == (11, 8) and option[3] == (11, 6) for option in _jump_options(guide.layout, state)))
    guide.jump((11, 8), (11, 6), peg)
    guide.rail_until(lambda state: any(option[0] == (14, 2) and option[3] == (14, 4) and option[1] == peg for option in _jump_options(guide.layout, state)))
    guide.jump((14, 2), (14, 4), peg)
    guide.jump((14, 3), (14, 5), red)
    guide.jump((14, 4), (14, 6), peg)
    guide.jump((14, 5), (16, 5), red)
    guide.jump((16, 5), (18, 5), red)
    guide.jump((14, 6), (16, 6), peg)
    guide.jump((16, 6), (18, 6), peg)
    guide.rail_until(lambda state: any(option[0] == (18, 6) and option[3] == (18, 4) and option[1] == peg for option in _jump_options(guide.layout, state)))
    guide.jump((18, 6), (18, 4), peg)
    guide.rail_until(lambda state: any(option[0] == (22, 3) and option[3] == (22, 5) and option[1] == peg for option in _jump_options(guide.layout, state)))
    guide.jump((22, 3), (22, 5), peg)
    guide.rail_until(lambda state: (22, 4) in state[1] and (22, 4) not in state[2] and (22, 4) not in dict(state[0]))
    if not guide.jump((22, 6), (22, 4), peg):
        raise _GuidedFailure("level 7 teacher goal edge did not win")


def _guided_level_ten(guide):
    for source, destination in (
        ((6, 5), (4, 5)), ((5, 5), (3, 5)),
        ((3, 5), (3, 7)), ((3, 6), (3, 8)),
        ((2, 8), (4, 8)), ((4, 7), (2, 7)),
    ):
        guide.jump(source, destination, names.PEG_BLUE)
    guide.rail_until(lambda state: any(option[0] == (4, 0) and option[3] == (4, 2) and option[1] == names.PEG for option in _jump_options(guide.layout, state)))
    guide.jump((4, 0), (4, 2), names.PEG)
    for position in ((4, 3), (3, 3), (2, 3), (2, 2), (2, 1), (0, 1), (0, 6), (2, 6)):
        guide.rail_until(lambda state, position=position: dict(state[0]).get(position) == names.PEG)
    guide.jump((2, 6), (2, 8), names.PEG)
    guide.jump_search_to_win()


def _search_guided_teacher(env, layout, remaining, node_limit):
    level = layout.logical_level
    if level not in (7, 10) or not _shipped_teacher(layout, level):
        return None
    guide = _TeacherGuide(env, layout, remaining, node_limit)
    try:
        if level == 7:
            _guided_level_seven(guide)
        else:
            _guided_level_ten(guide)
    except _GuidedCutoff:
        return _negative(
            layout, True, guide.expanded, guide.generated,
            f"configuration expansion limit {node_limit} reached during exact teacher waypoints",
            node_limit,
        )
    except _GuidedFailure as exc:
        return SearchResult(
            None, False, True, False, guide.expanded, guide.generated,
            str(exc), node_limit, tuple(sorted(guide.events.items())),
        )
    return guide.result()


def _search_generated_teacher(env, layout, remaining, node_limit):
    """Return a bounded, native-replayed constructive guide at its exact start."""
    descriptor = env.generated_descriptor
    if (
        layout.generated_kind != names.FULL_GENERATED_KIND
        or layout.action_count != 0
        or layout.history_depth != 0
        or layout.selected is not None
        or layout.reset_prompt
        or not isinstance(descriptor, dict)
    ):
        return None
    raw = descriptor.get("teacher_actions")
    if not isinstance(raw, list) or not raw:
        return None
    actions = []
    for value in raw:
        if not isinstance(value, list) or len(value) != 3:
            return SearchResult(None, False, True, False, 0, 0, "generated guide is malformed", node_limit)
        action, x, y = value
        if not isinstance(action, int) or isinstance(action, bool):
            return SearchResult(None, False, True, False, 0, 0, "generated guide is malformed", node_limit)
        actions.append((action, x, y))
    actions = tuple(actions)
    if len(actions) > remaining:
        return _negative(
            layout, True, 0, 1,
            f"generated positive guide needs {len(actions)} actions beyond bound {remaining}",
            node_limit,
        )
    if node_limit < 1:
        return _negative(layout, True, 0, 1, "configuration expansion limit 0 reached", node_limit)
    mismatch = _verify_live_witness(env, actions)
    if mismatch is not None:
        return SearchResult(None, False, True, False, 1, 1, mismatch, node_limit)
    return SearchResult(
        actions, False, False, True, 1, 1,
        "bounded constructive guide replayed on a cloned real-engine snapshot",
        node_limit,
    )


def _search_forward(env, layout, remaining, node_limit):
    start = _state(layout)
    if node_limit == 0:
        return _negative(layout, True, 0, 1, "configuration expansion limit 0 reached", node_limit)

    queue = []
    counter = 0

    def heuristic(state):
        # A weighted admissibility-free priority is intentional: the contract
        # requires a bounded positive witness, not a fabricated shortest-path
        # claim.  Strongly preferring a real peg removal avoids enumerating the
        # combinatorial rail-position product before taking an available jump.
        remaining = max(0, _goal_count(layout, state[0]) - layout.target_nonblue)
        by_kind = {}
        for cell, kind in state[0]:
            if kind != names.PEG_BLUE:
                by_kind.setdefault(kind, []).append(cell)
        pair_distance = 0
        distances = [
            abs(a[0] - b[0]) + abs(a[1] - b[1])
            for cells in by_kind.values()
            for index, a in enumerate(cells)
            for b in cells[index + 1:]
        ]
        if distances:
            pair_distance = min(distances)
        return remaining * 64 + pair_distance * 4

    heapq.heappush(queue, (heuristic(start), 0, counter, start))
    best = {start: 0}
    parent = {start: None}
    expanded = 0
    generated = 1
    depth_cutoff = False
    while queue:
        _, cost, _, state = heapq.heappop(queue)
        if cost != best.get(state):
            continue
        if expanded >= node_limit:
            return _negative(
                layout, True, expanded, generated,
                f"configuration expansion limit {node_limit} reached", node_limit,
            )
        expanded += 1

        for option in _jump_options(layout, state):
            source, _, _, destination = option
            already_selected = (
                state == start
                and layout.selected == source
                and layout.selected_kind == dict(state[0]).get(source)
            )
            edge_actions = []
            if not already_selected:
                edge_actions.append((names.ACTION_CLICK, *layout.click(source, state[3])))
            edge_actions.append((names.ACTION_CLICK, *layout.click(destination, state[3])))
            next_cost = cost + len(edge_actions)
            if next_cost > remaining:
                depth_cutoff = True
                continue
            nxt, won, reset, event = _jump_transition(layout, state, option)
            if won:
                prefix, events = _unwind(parent, state)
                actions = prefix + tuple(edge_actions)
                events.update(event)
                mismatch = _verify_live_witness(env, actions)
                if mismatch is not None:
                    return SearchResult(
                        None, False, True, False, expanded, generated, mismatch,
                        node_limit, tuple(sorted(events.items())),
                    )
                return SearchResult(
                    actions, False, False, True, expanded, generated,
                    "solved and replayed on a cloned real-engine snapshot",
                    node_limit, tuple(sorted(events.items())),
                )
            if reset:
                continue
            if next_cost < best.get(nxt, remaining + 1):
                best[nxt] = next_cost
                parent[nxt] = state, tuple(edge_actions), tuple(event.items())
                generated += 1
                counter += 1
                heapq.heappush(queue, (heuristic(nxt), next_cost, counter, nxt))

        # Arrow presses clear selection and cost one native action even when no
        # yellow cell moves.  No-op arrows are omitted because they cannot help
        # a shortest positive route.
        for action, dx, dy in _ARROWS:
            next_cost = cost + 1
            if next_cost > remaining:
                depth_cutoff = True
                continue
            nxt, event = _rail_transition(layout, state, dx, dy)
            if nxt == state:
                continue
            if next_cost < best.get(nxt, remaining + 1):
                best[nxt] = next_cost
                parent[nxt] = state, ((action, None, None),), tuple(event.items())
                generated += 1
                counter += 1
                heapq.heappush(queue, (heuristic(nxt), next_cost, counter, nxt))

    reason = (
        f"no route within remaining action budget {remaining}"
        if depth_cutoff else "complete supported logical graph exhausted"
    )
    return _negative(layout, depth_cutoff, expanded, generated, reason, node_limit)


def _recover_then_search(env, layout, remaining, node_limit):
    """Use native undo/reset semantics only after forward search is exhausted."""
    candidates = []
    if layout.pending_auto_undo:
        candidates.append((names.ACTION_UNDO, "resolved pending automatic undo"))
    elif layout.reset_prompt:
        candidates.append((names.ACTION_RESET, "reset scripted failure"))
    elif layout.history_depth:
        candidates.append((names.ACTION_UNDO, "used live undo history"))
    if not candidates or remaining < 1:
        return None
    action, label = candidates[0]
    snapshot = env.clone()
    snapshot.perform(action, None, None)
    recovered = extract(snapshot)
    if not recovered.exact:
        return None
    result = _search_forward(snapshot, recovered, min(remaining - 1, recovered.actions_left), node_limit)
    if result.actions is None:
        return None
    actions = ((action, None, None),) + result.actions
    mismatch = _verify_live_witness(env, actions)
    if mismatch is not None:
        return None
    use = Counter(dict(result.mechanic_use))
    use["undo_actions" if action == names.ACTION_UNDO else "reset_actions"] += 1
    return SearchResult(
        actions, False, False, True, result.expanded, result.generated,
        f"{label}; solved and replayed on a cloned real-engine snapshot",
        node_limit, tuple(sorted(use.items())),
    )


def search(env_or_layout, limit=None, budget=None, node_limit=DEFAULT_LIMIT):
    """Search one live native level under explicit action and work bounds."""
    for value, label in ((limit, "limit"), (budget, "budget")):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{label} must be a nonnegative integer or None")
    if not isinstance(node_limit, int) or isinstance(node_limit, bool) or node_limit < 0:
        raise ValueError("node_limit must be a nonnegative integer")
    if isinstance(env_or_layout, Layout):
        layout = env_or_layout
        live_env = None
    else:
        live_env = env_or_layout
        layout = extract(live_env)
    if not layout.exact:
        return SearchResult(
            None, False, True, False, 0, 0,
            "; ".join(layout.unsupported) or "unsupported LF52 layout", node_limit,
        )
    if live_env is None:
        return SearchResult(
            None, False, True, False, 0, 0,
            "a live Env snapshot is required for real-engine witness verification",
            node_limit,
        )
    remaining = layout.actions_left
    if limit is not None:
        remaining = min(remaining, limit)
    if budget is not None:
        remaining = min(remaining, budget)
    guided = _search_guided_teacher(live_env, layout, remaining, node_limit)
    if guided is not None:
        return guided
    guided = _search_generated_teacher(live_env, layout, remaining, node_limit)
    if guided is not None:
        return guided
    result = _search_forward(live_env, layout, remaining, node_limit)
    if result.actions is not None:
        return result
    recovered = _recover_then_search(live_env, layout, remaining, node_limit)
    return result if recovered is None else recovered


def solve(env_or_layout, limit=None, budget=None, node_limit=DEFAULT_LIMIT):
    """Return action triples or ``None`` and expose the classified result."""
    result = search(env_or_layout, limit=limit, budget=budget, node_limit=node_limit)
    solve.last_result = result
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    return list(result.actions) if result.actions is not None else None


solve.last_result = None
solve.result = None
solve.truncated = False
solve.unsupported = False
solve.exact = True
