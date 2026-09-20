"""Bounded exact-state teacher for all six official CN04 tiers.

The planner models every stateful rule in the shipped engine. Its heuristic
only orders states and never discards a cheaper visit. A cap hit is reported as
truncation, never as proof of impossibility. Returned routes are positive
witnesses, not claims of shortest-path optimality.
"""

import heapq
import itertools

from . import names
from .layout import Layout, UnsupportedLayout, extract


DEFAULT_LIMIT = 1_500_000
GENERATED_ROUTE_LIMIT = 100_000
GENERATED_ASSIGNMENT_LIMIT = 8_000


class SearchResult:
    __slots__ = (
        "actions", "truncated", "unsupported", "expanded", "reason",
        "assignment_caps",
    )

    def __init__(self, actions=None, *, truncated=False, unsupported=False,
                 expanded=0, reason="", assignment_caps=0):
        self.actions = actions
        self.truncated = bool(truncated)
        self.unsupported = bool(unsupported)
        self.expanded = int(expanded)
        self.reason = reason
        self.assignment_caps = int(assignment_caps)

    @property
    def solved(self):
        return self.actions is not None


def _cycle(layout, state, *, clamp):
    transforms, active, selected, forward = state
    group = layout.groups[selected]
    index = active[selected]
    changed_direction = False
    if forward:
        next_index = index + 1
        if next_index >= len(group.alternatives):
            forward = False
            changed_direction = True
            next_index = index - 1
            if next_index < 0:
                forward = True
                changed_direction = False
                next_index = index
    else:
        next_index = index - 1
        if next_index < 0:
            forward = True
            changed_direction = True
            next_index = index + 1
            if next_index >= len(group.alternatives):
                forward = False
                changed_direction = False
                next_index = index
    changed_active = list(active)
    changed_active[selected] = next_index
    piece = group.alternatives[next_index]
    x, y, _ = transforms[selected]
    rotation = piece.initial_rotation
    if clamp:
        width, height = piece.size(rotation)
        x = max(0, min(x, layout.grid_size[0] - width))
        y = max(0, min(y, layout.grid_size[1] - height))
    changed_transforms = list(transforms)
    changed_transforms[selected] = (x, y, rotation)
    return ((tuple(changed_transforms), tuple(changed_active), selected, forward),
            changed_direction)


def _grid_cell(layout, x, y):
    scale, ox, oy = names.display_geometry(layout.grid_size)
    if not (ox <= x < ox + layout.grid_size[0] * scale
            and oy <= y < oy + layout.grid_size[1] * scale):
        return None
    return ((int(x) - ox) // scale, (int(y) - oy) // scale)


def transition(layout, state, action):
    """Apply one action triple and return ``(state, event, checks_win)``."""
    action_id, x_click, y_click = action
    transforms, active, selected, forward = state
    event = {"kind": "noop", "bounce_reversed": False}

    if action_id in names.MOVE_DELTAS:
        if selected is None:
            return state, event, False
        dx, dy = names.MOVE_DELTAS[action_id]
        piece = layout.piece(state, selected)
        x, y, rotation = transforms[selected]
        width, height = piece.size(rotation)
        nx, ny = x + dx, y + dy
        if (nx >= 0 and ny >= 0 and nx + width <= layout.grid_size[0]
                and ny + height <= layout.grid_size[1]):
            changed = list(transforms)
            changed[selected] = (nx, ny, rotation)
            state = (tuple(changed), active, selected, forward)
            event = {"kind": "move", "bounce_reversed": False}
        else:
            event = {"kind": "blocked_move", "bounce_reversed": False}
        return state, event, True
    if action_id == names.ACTION_ROTATE:
        if selected is None:
            return state, event, False
        if layout.groups[selected].is_stack:
            state, reversed_direction = _cycle(layout, state, clamp=True)
            event = {"kind": "stack_cycle_action",
                     "bounce_reversed": reversed_direction}
        else:
            x, y, rotation = transforms[selected]
            rotation = (rotation + 1) % 4
            piece = layout.piece(state, selected)
            width, height = piece.size(rotation)
            x = max(0, min(x, layout.grid_size[0] - width))
            y = max(0, min(y, layout.grid_size[1] - height))
            changed = list(transforms)
            changed[selected] = (x, y, rotation)
            state = (tuple(changed), active, selected, forward)
            event = {"kind": "rotate", "bounce_reversed": False}
        return state, event, True
    if action_id == names.ACTION_CLICK:
        if x_click is None or y_click is None:
            raise ValueError("ACTION6 needs display coordinates")
        cell = _grid_cell(layout, x_click, y_click)
        hit = None if cell is None else layout.first_hit(state, cell)
        if hit is None:
            return state, event, False
        if hit != selected:
            state = (transforms, active, hit, forward)
            event = {"kind": "select", "bounce_reversed": False}
        elif layout.groups[hit].is_stack:
            if layout.is_cycle_cell(state, cell):
                state, reversed_direction = _cycle(layout, state, clamp=False)
                event = {"kind": "stack_cycle_click",
                         "bounce_reversed": reversed_direction}
            else:
                state = (transforms, active, None, forward)
                event = {"kind": "deselect", "bounce_reversed": False}
        else:
            state = (transforms, active, None, forward)
            event = {"kind": "deselect", "bounce_reversed": False}
        return state, event, False
    raise ValueError(f"unsupported action id {action_id}")


def _pin_entries(layout, state):
    entries = []
    counts = {}
    for group_index, (x, y, rotation) in enumerate(state[0]):
        piece = layout.piece(state, group_index)
        for dx, dy, colour in piece.pins(rotation):
            entry = (group_index, x + dx, y + dy, colour)
            entries.append(entry)
            key = (entry[1], entry[2], colour)
            counts[key] = counts.get(key, 0) + 1
    return entries, counts


def _priority(layout, state):
    entries, counts = _pin_entries(layout, state)
    unmatched = [entry for entry in entries if counts[(entry[1], entry[2], entry[3])] != 2]
    distance = 0
    for group, x, y, colour in unmatched:
        distance += min(
            (abs(x - ox) + abs(y - oy)
             for other, ox, oy, other_colour in entries
             if other != group and other_colour == colour),
            default=100,
        )
    return 8 * len(unmatched) + distance


def _unwind(parent, state, final_action=None):
    actions = [] if final_action is None else [final_action]
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return actions


def _candidate_actions(layout, state):
    for action_id in (names.ACTION_UP, names.ACTION_DOWN, names.ACTION_LEFT,
                      names.ACTION_RIGHT, names.ACTION_ROTATE):
        yield (action_id, None, None)
    selected = state[2]
    for target in range(len(layout.groups)):
        if target == selected:
            continue
        click = layout.click_for(state, target)
        if click is not None:
            yield (names.ACTION_CLICK, click[0], click[1])
    click = layout.cycle_click_for(state)
    if click is not None:
        yield (names.ACTION_CLICK, click[0], click[1])


class _GeneratedRouteCap(Exception):
    pass


class _GeneratedAssignmentCap(Exception):
    pass


def _generated_layout(layout):
    """Generated rows have stable nonsemantic names; official search is untouched."""
    return bool(layout.pieces) and all(
        piece.name.startswith("cn04_g") for piece in layout.pieces
    )


def _generated_expected_pin_counts(layout):
    """Return public tier-grammar degrees used only to prune stack choices.

    The two generated stack compositions are distinguishable from the native
    group sizes alone.  The all-three-pin four-member tier-six stack still
    leaves four possible choices; pin count therefore cannot reveal the full
    winning assignment.
    """
    sizes = tuple(len(group.alternatives) for group in layout.groups)
    return {
        (5, 1, 1, 1): (7, 3, 2, 2),
        (6, 4, 1, 1, 1): (2, 3, 3, 2, 2),
    }.get(sizes)


def _generated_winner_sets(layout, tick):
    """Enumerate ambiguous alternate assignments from native sprite data."""
    expected = _generated_expected_pin_counts(layout)
    choices = []
    for group_index, group in enumerate(layout.groups):
        alternatives = range(len(group.alternatives))
        if expected is not None and group.is_stack:
            alternatives = tuple(
                alternate
                for alternate in alternatives
                if len(group.alternatives[alternate].pins(
                    group.alternatives[alternate].initial_rotation,
                )) == expected[group_index]
            )
        choices.append(alternatives)
    if any(not tuple(choice) for choice in choices):
        return
    for active in itertools.product(*choices):
        tick()
        winners = tuple(
            (alternate, group.alternatives[alternate])
            for group, alternate in zip(layout.groups, active)
        )
        colour_counts = {}
        for _, piece in winners:
            for _, _, colour in piece.pins(piece.initial_rotation):
                colour_counts[colour] = colour_counts.get(colour, 0) + 1
        if colour_counts and all(count % 2 == 0 for count in colour_counts.values()):
            yield winners


def _generated_targets(layout, winners, tick):
    """Enumerate connected exact pin assemblies, modulo global translation."""
    group_count = len(layout.groups)
    seen = set()

    def rotations(group_index):
        _, piece = winners[group_index]
        if layout.groups[group_index].is_stack:
            return (piece.initial_rotation,)
        return (0, 1, 2, 3)

    def add_piece(counts, group_index, piece, rotation, x, y):
        changed = {key: list(groups) for key, groups in counts.items()}
        for dx, dy, colour in piece.pins(rotation):
            key = (x + dx, y + dy, colour)
            groups = changed.setdefault(key, [])
            groups.append(group_index)
            if len(groups) > 2:
                return None
        return changed

    def recurse(placed, counts):
        tick()
        if len(placed) == group_count:
            if not counts or any(len(groups) != 2 for groups in counts.values()):
                return
            min_x = min(value[0] for value in placed.values())
            min_y = min(value[1] for value in placed.values())
            key = tuple(
                (placed[index][0] - min_x, placed[index][1] - min_y,
                 placed[index][2], placed[index][3])
                for index in range(group_count)
            )
            if key not in seen:
                seen.add(key)
                yield tuple(placed[index] for index in range(group_count))
            return

        open_pins = [key for key, groups in counts.items() if len(groups) == 1]
        if not open_pins:
            return
        for group_index in range(group_count):
            if group_index in placed:
                continue
            alternate, piece = winners[group_index]
            tried = set()
            for rotation in rotations(group_index):
                for dx, dy, colour in piece.pins(rotation):
                    for px, py, open_colour in open_pins:
                        if colour != open_colour:
                            continue
                        position = (px - dx, py - dy)
                        branch = (rotation, position)
                        if branch in tried:
                            continue
                        tried.add(branch)
                        changed = add_piece(
                            counts, group_index, piece, rotation, *position,
                        )
                        if changed is None:
                            continue
                        placed[group_index] = (
                            position[0], position[1], rotation, alternate,
                        )
                        yield from recurse(placed, changed)
                        del placed[group_index]

    first_alternate, first_piece = winners[0]
    for first_rotation in rotations(0):
        placed = {0: (0, 0, first_rotation, first_alternate)}
        counts = add_piece({}, 0, first_piece, first_rotation, 0, 0)
        yield from recurse(placed, counts)


def _target_translations(layout, target, winners):
    min_x = min_y = 0
    max_x = max_y = 0
    initialized = False
    for group_index, (x, y, rotation, _) in enumerate(target):
        _, piece = winners[group_index]
        width, height = piece.size(rotation)
        if not initialized:
            min_x, min_y, max_x, max_y = x, y, x + width, y + height
            initialized = True
        else:
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + width), max(max_y, y + height)
    x_stop = layout.grid_size[0] - max_x
    y_stop = layout.grid_size[1] - max_y
    if -min_x > x_stop or -min_y > y_stop:
        return ()
    starts = layout.start[0]
    shifts = [
        ((sx, sy), sum(
            abs(starts[index][0] - (value[0] + sx))
            + abs(starts[index][1] - (value[1] + sy))
            for index, value in enumerate(target)
        ))
        for sx in range(-min_x, x_stop + 1)
        for sy in range(-min_y, y_stop + 1)
    ]
    if not shifts:
        return ()
    minimum = min(cost for _, cost in shifts)
    return tuple(shift for shift, cost in shifts if cost == minimum)


def _apply_route_action(layout, state, actions, action, tick):
    tick()
    if len(actions) >= layout.steps_left:
        return state, False, False
    nxt, _, checks_win = transition(layout, state, action)
    actions.append(action)
    return nxt, bool(checks_win and layout.complete(nxt)), True


def _completion_trigger(layout, state, actions, tick):
    """Find a short ACTION1-5 trigger when a live state is already complete."""
    frontier = [(state, tuple())]
    seen = {state}
    for _ in range(6):
        changed = []
        for current, suffix in frontier:
            for action in _candidate_actions(layout, current):
                tick()
                if len(actions) + len(suffix) >= layout.steps_left:
                    continue
                nxt, _, checks_win = transition(layout, current, action)
                candidate = suffix + (action,)
                if checks_win and layout.complete(nxt):
                    return actions + list(candidate)
                if nxt not in seen:
                    seen.add(nxt)
                    changed.append((nxt, candidate))
        frontier = changed
        if not frontier:
            break
    return None


def _route_to_generated_target(layout, target, shift, order, tick):
    transforms = tuple(
        (x + shift[0], y + shift[1], rotation)
        for x, y, rotation, _ in target
    )
    active = tuple(value[3] for value in target)
    target_state = (transforms, active, layout.start[2], layout.start[3])
    if not layout.complete(target_state):
        return None

    state = layout.start
    actions = []
    for group_index in order:
        if state[2] != group_index:
            click = layout.click_for(state, group_index)
            if click is None:
                return None
            state, won, allowed = _apply_route_action(
                layout, state, actions,
                (names.ACTION_CLICK, click[0], click[1]), tick,
            )
            if not allowed:
                return None
            if won:
                return actions

        desired_alternate = active[group_index]
        guard = 0
        while state[1][group_index] != desired_alternate:
            state, won, allowed = _apply_route_action(
                layout, state, actions, (names.ACTION_ROTATE, None, None), tick,
            )
            if not allowed or guard > 2 * len(layout.groups[group_index].alternatives):
                return None
            if won:
                return actions
            guard += 1

        desired_rotation = transforms[group_index][2]
        guard = 0
        while state[0][group_index][2] != desired_rotation:
            if layout.groups[group_index].is_stack:
                return None
            state, won, allowed = _apply_route_action(
                layout, state, actions, (names.ACTION_ROTATE, None, None), tick,
            )
            if not allowed or guard >= 4:
                return None
            if won:
                return actions
            guard += 1

        target_x, target_y, _ = transforms[group_index]
        for action_id, count in (
            (names.ACTION_RIGHT, target_x - state[0][group_index][0]),
            (names.ACTION_LEFT, state[0][group_index][0] - target_x),
            (names.ACTION_DOWN, target_y - state[0][group_index][1]),
            (names.ACTION_UP, state[0][group_index][1] - target_y),
        ):
            for _ in range(max(0, count)):
                state, won, allowed = _apply_route_action(
                    layout, state, actions, (action_id, None, None), tick,
                )
                if not allowed:
                    return None
                if won:
                    return actions
        if state[0][group_index] != transforms[group_index]:
            return None

    if layout.complete(state):
        return _completion_trigger(layout, state, actions, tick)
    return None


def _native_route_replays(env, actions):
    try:
        clone = env.clone()
        before = clone.levels_completed
        for index, action in enumerate(actions):
            observation = clone.perform(*action)
            if observation.levels_completed > before or observation.won:
                return index == len(actions) - 1 and clone.levels_completed == before + 1
            if observation.finished:
                return False
    except (AttributeError, TypeError, ValueError):
        return False
    return False


def _generated_positive_route(env, layout, limit):
    """Return a bounded, native-replayed generated-level witness when found."""
    if limit <= 0 or not _generated_layout(layout):
        return None, 0, False, 0
    cap = min(limit, GENERATED_ROUTE_LIMIT)
    work = 0
    assignment_start = 0
    assignment_capped = False
    assignment_caps = 0

    def tick():
        nonlocal work
        if work >= cap:
            raise _GeneratedRouteCap
        work += 1

    selected = layout.start[2]
    indices = tuple(range(len(layout.groups)))
    if layout.stacks:
        orders = list(itertools.permutations(indices))
        orders.sort(key=lambda order: (order[0] != selected if selected is not None else 1,
                                       order))
    else:
        prefix = () if selected is None else (selected,)
        orders = [prefix + tuple(index for index in indices if index != selected)]
    try:
        for winners in _generated_winner_sets(layout, tick):
            assignment_start = work

            def assignment_tick():
                if work - assignment_start >= GENERATED_ASSIGNMENT_LIMIT:
                    raise _GeneratedAssignmentCap
                tick()

            try:
                for target in _generated_targets(layout, winners, assignment_tick):
                    for shift in _target_translations(layout, target, winners):
                        for order in orders:
                            actions = _route_to_generated_target(
                                layout, target, shift, order, assignment_tick,
                            )
                            if actions is None:
                                continue
                            if _native_route_replays(env, actions):
                                return actions, work, False, assignment_caps
            except _GeneratedAssignmentCap:
                assignment_capped = True
                assignment_caps += 1
                continue
    except _GeneratedRouteCap:
        return None, work, True, assignment_caps
    return None, work, assignment_capped, assignment_caps


def search(env_or_layout, limit=DEFAULT_LIMIT):
    """Find a positive witness for the current level within a hard state cap."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    try:
        layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    except (UnsupportedLayout, AttributeError, TypeError, ValueError) as exc:
        return SearchResult(unsupported=True, reason=str(exc))
    if not layout.exact:
        return SearchResult(unsupported=True, reason="; ".join(layout.unsupported))
    if layout.steps_left < 1:
        return SearchResult(reason="no actions remain before MaxSteps loss")

    generated_work = 0
    if not isinstance(env_or_layout, Layout):
        actions, generated_work, generated_capped, assignment_caps = _generated_positive_route(
            env_or_layout, layout, limit,
        )
        if actions is not None:
            capped_suffix = (
                f" after {assignment_caps} bounded assignment cap(s)"
                if assignment_caps else ""
            )
            return SearchResult(
                actions,
                expanded=generated_work,
                reason=(
                    "floor-independent minimum-translation semantic assembly route "
                    f"won in native replay{capped_suffix}"
                ),
                assignment_caps=assignment_caps,
            )
        if generated_capped:
            return SearchResult(
                truncated=True,
                expanded=generated_work,
                reason="generated-route assignment/work limit reached",
                assignment_caps=assignment_caps,
            )
    exact_limit = max(0, limit - generated_work)

    if not layout.stacks:
        colour_counts = {colour: 0 for colour in names.PIN_COLORS}
        for group in layout.groups:
            for _, _, colour in group.alternatives[0].pins(0):
                colour_counts[colour] += 1
        odd = [colour for colour, count in colour_counts.items() if count % 2]
        if odd:
            return SearchResult(
                expanded=generated_work,
                reason=f"odd total pin count for colours {odd}",
            )

    start = layout.start
    parent = {start: None}
    best = {start: 0}
    counter = 0
    frontier = [(_priority(layout, start), 0, counter, start)]
    expanded = 0

    while frontier:
        _, cost, _, state = heapq.heappop(frontier)
        if cost != best.get(state):
            continue
        if expanded >= exact_limit:
            return SearchResult(truncated=True, expanded=generated_work + expanded,
                                reason="symbolic state expansion limit reached")
        expanded += 1
        if cost >= layout.steps_left:
            continue
        for action in _candidate_actions(layout, state):
            nxt, _, checks_win = transition(layout, state, action)
            if checks_win and layout.complete(nxt):
                return SearchResult(
                    _unwind(parent, state, action),
                    expanded=generated_work + expanded,
                    reason="engine-equivalent completion state reached",
                )
            if nxt == state:
                continue
            next_cost = cost + 1
            if next_cost < best.get(nxt, layout.steps_left + 1):
                best[nxt] = next_cost
                parent[nxt] = (state, action)
                counter += 1
                heapq.heappush(frontier, (next_cost + _priority(layout, nxt),
                                          next_cost, counter, nxt))

    return SearchResult(expanded=generated_work + expanded,
                        reason="exact state graph exhausted within the step budget")


def trace(layout, actions):
    """Replay a symbolic witness and summarize mechanically meaningful events."""
    state = layout.start
    counts = {
        "moves": 0,
        "rotations": 0,
        "selections": 0,
        "stack_cycles_action": 0,
        "stack_cycles_click": 0,
        "bounce_reversals": 0,
        "distinct_stacks_cycled": 0,
    }
    cycled = set()
    for action in actions:
        selected = state[2]
        state, event, _ = transition(layout, state, tuple(action))
        kind = event["kind"]
        if kind == "move":
            counts["moves"] += 1
        elif kind == "rotate":
            counts["rotations"] += 1
        elif kind == "select":
            counts["selections"] += 1
        elif kind == "stack_cycle_action":
            counts["stack_cycles_action"] += 1
            cycled.add(selected)
        elif kind == "stack_cycle_click":
            counts["stack_cycles_click"] += 1
            cycled.add(selected)
        counts["bounce_reversals"] += int(event["bounce_reversed"])
    counts["distinct_stacks_cycled"] = len(cycled)
    counts["won"] = layout.complete(state)
    return counts


def solve(env_or_layout, limit=DEFAULT_LIMIT):
    result = search(env_or_layout, limit=limit)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.reason = result.reason
    return result.actions


solve.result = None
solve.truncated = False
solve.unsupported = False
solve.reason = ""
