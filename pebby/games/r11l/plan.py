"""Bounded constructive teacher for every shipped R11L mechanic.

R11L groups do not collide with one another. Ordinary target groups can
therefore be planned independently while still checking the exact global wall
and hazard masks. Pickup levels use the same exact centroid transition, plus
native replay after every sub-plan so pickup pixel mutation/removal is never
approximated. Successful plans are positive certificates; exhausting the
bounded constructive search is reported as truncation, not impossibility.
"""

from dataclasses import dataclass
import heapq
from itertools import count, permutations

import numpy as np

from . import names
from .layout import Group, Layout, Point, _global_mask, extract


DEFAULT_LIMIT = 20_000
Action = tuple[int, int, int]
State = tuple[tuple[Point, ...], int]


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...] | None
    solved: bool
    expanded: int
    generated: int
    truncated: bool
    unsupported: bool
    exact: bool
    reason: str
    limit: int


def _overlaps(mask: frozenset[Point], position: Point, other: frozenset[Point]) -> bool:
    x, y = position
    return any((x + dx, y + dy) in other for dx, dy in mask)


def _core_position(layout: Layout, positions: tuple[Point, ...], group: Group) -> Point:
    indices = group.fragment_indices
    count_ = len(indices)
    center_x = sum(positions[i][0] + names.HALF for i in indices) // count_
    center_y = sum(positions[i][1] + names.HALF for i in indices) // count_
    return center_x - names.HALF, center_y - names.HALF


def _group_satisfied(layout: Layout, positions: tuple[Point, ...], group: Group) -> bool:
    if not group.required:
        return True
    assert group.target_mask is not None
    return _overlaps(group.core_mask, _core_position(layout, positions, group), group.target_mask)


def _hazard_free(layout: Layout, positions: tuple[Point, ...]) -> bool:
    if not layout.hazards:
        return True
    for group in layout.groups:
        if group.fragment_indices and group.core_mask:
            if _overlaps(group.core_mask, _core_position(layout, positions, group), layout.hazards):
                return False
    return True


def _selection_click(layout: Layout, positions: tuple[Point, ...], index: int) -> Point | None:
    """Return a display click selecting exactly ``index`` under native order."""
    fragment = layout.fragments[index]
    x0, y0 = positions[index]
    candidates = [(x0 + fragment.width // 2, y0 + fragment.height // 2)]
    candidates.extend(
        (x, y)
        for y in range(max(0, y0), min(names.FRAME_SIZE, y0 + fragment.height))
        for x in range(max(0, x0), min(names.FRAME_SIZE, x0 + fragment.width))
        if (x, y) != candidates[0]
    )
    for x, y in candidates:
        if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
            continue
        hit = next(
            (
                other_index
                for other_index, other in enumerate(layout.fragments)
                if positions[other_index][0] <= x < positions[other_index][0] + other.width
                and positions[other_index][1] <= y < positions[other_index][1] + other.height
            ),
            None,
        )
        if hit == index:
            return x, y
    return None


def _mask_centre(mask: frozenset[Point]) -> Point:
    return (
        sum(x for x, _ in mask) // len(mask),
        sum(y for _, y in mask) // len(mask),
    )


def _object_plan(
    layout: Layout,
    positions: tuple[Point, ...],
    selected: int,
    group: Group,
    objective: frozenset[Point],
    *,
    forbidden: tuple[frozenset[Point], ...] = (),
    limit: int,
    action_budget: int,
):
    """A* one group to a target/pickup using exact destination semantics."""
    if limit <= 0:
        return None, 0, 0
    objective_x, objective_y = _mask_centre(objective)

    def goal(candidate):
        return _overlaps(group.core_mask, _core_position(layout, candidate, group), objective)

    def valid(candidate):
        core_position = _core_position(layout, candidate, group)
        return _hazard_free(layout, candidate) and not any(
            _overlaps(group.core_mask, core_position, mask) for mask in forbidden
        )

    def heuristic(candidate):
        core_x, core_y = _core_position(layout, candidate, group)
        return abs(core_x + names.HALF - objective_x) + abs(core_y + names.HALF - objective_y)

    serial = count()
    initial: State = (positions, selected)
    parents: dict[State, tuple[State, tuple[Action, ...]] | None] = {initial: None}
    costs = {initial: 0}
    frontier = [(heuristic(positions), 0, next(serial), initial)]
    expanded = generated = 0
    while frontier and expanded < limit:
        _, cost, _, state = heapq.heappop(frontier)
        if cost != costs.get(state):
            continue
        expanded += 1
        current, current_selected = state
        for index in group.fragment_indices:
            selection = None if current_selected == index else _selection_click(layout, current, index)
            if current_selected != index and selection is None:
                continue
            prefix = () if selection is None else ((names.ACTION_CLICK, *selection),)
            edge_cost = len(prefix) + 1
            if cost + edge_cost > action_budget:
                continue
            fragment = layout.fragments[index]
            preferred = []
            remaining = []
            for click_y in range(names.FRAME_SIZE):
                for click_x in range(names.FRAME_SIZE):
                    if any(
                        px <= click_x < px + other.width
                        and py <= click_y < py + other.height
                        for other, (px, py) in zip(layout.fragments, current)
                    ):
                        continue
                    destination = names.click_to_position(click_x, click_y)
                    if destination == current[index] or _overlaps(fragment.mask, destination, layout.walls):
                        continue
                    moved = list(current)
                    moved[index] = destination
                    moved_tuple = tuple(moved)
                    if not valid(moved_tuple):
                        continue
                    generated += 1
                    item = (moved_tuple, (names.ACTION_CLICK, click_x, click_y))
                    (preferred if goal(moved_tuple) else remaining).append(item)
            for moved, drag in preferred + remaining:
                actions = prefix + (drag,)
                successor = (moved, index)
                new_cost = cost + len(actions)
                if goal(moved):
                    pieces = [actions]
                    cursor = state
                    while parents[cursor] is not None:
                        cursor, previous_actions = parents[cursor]
                        pieces.append(previous_actions)
                    route = tuple(action for piece in reversed(pieces) for action in piece)
                    return (route, moved, index), expanded, generated
                if new_cost >= costs.get(successor, names.MAX_ACTIONS + 1):
                    continue
                costs[successor] = new_cost
                parents[successor] = (state, actions)
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic(moved), new_cost, next(serial), successor),
                )
    return None, expanded, generated


def _ordinary_search(layout: Layout, limit: int) -> SearchResult:
    positions = layout.positions
    selected = layout.selected
    actions: list[Action] = []
    expanded = generated = 0
    for group in (group for group in layout.groups if group.required):
        if _group_satisfied(layout, positions, group):
            continue
        result, used_expanded, used_generated = _object_plan(
            layout,
            positions,
            selected,
            group,
            group.target_mask or frozenset(),
            limit=limit - expanded,
            action_budget=layout.actions_left - len(actions),
        )
        expanded += used_expanded
        generated += used_generated
        if result is None:
            return SearchResult(
                None, False, expanded, generated, True, False, False,
                "bounded constructive group search exhausted", limit,
            )
        route, positions, selected = result
        actions.extend(route)
    if not actions:
        return SearchResult(
            None, False, expanded, generated, False, False, True,
            "no unsatisfied draggable target group", limit,
        )
    return SearchResult(
        tuple(actions), True, expanded, generated, False, False, True,
        "constructive solution found", limit,
    )


def _positive_colours(sprite) -> frozenset[int]:
    return frozenset(int(value) for value in np.unique(sprite.pixels) if value > 0)


def _apply(env, actions: tuple[Action, ...]) -> bool:
    from arcengine import GameState

    for action in actions:
        if env.perform(*action).state == GameState.GAME_OVER:
            return False
    return True


def _pickup_assignment_plan(env, assignment, limit):
    """Plan one absorber/target bijection, replaying every leg natively."""
    work = env.clone()
    actions: list[Action] = []
    expanded = generated = 0
    for absorber_name, (_, target, target_colours) in assignment:
        while True:
            core = work.groups()[absorber_name][names.KEY_CORE]
            current_colours = _positive_colours(core)
            if current_colours == target_colours:
                break
            choices = sorted(
                (
                    pickup
                    for pickup in work.pickups()
                    if _positive_colours(pickup) <= target_colours
                    and not _positive_colours(pickup) <= current_colours
                ),
                key=lambda pickup: (tuple(sorted(_positive_colours(pickup))), pickup.name, pickup.x, pickup.y),
            )
            if not choices:
                return None, expanded, generated
            pickup = choices[0]
            layout = extract(work)
            group = next(group for group in layout.groups if group.name == absorber_name)
            forbidden = tuple(
                _global_mask(other)
                for other in work.pickups()
                if not _positive_colours(other) <= target_colours
            )
            result, used_expanded, used_generated = _object_plan(
                layout,
                layout.positions,
                layout.selected,
                group,
                _global_mask(pickup),
                forbidden=forbidden,
                limit=limit - expanded,
                action_budget=layout.actions_left,
            )
            expanded += used_expanded
            generated += used_generated
            if result is None or not _apply(work, result[0]):
                return None, expanded, generated
            actions.extend(result[0])

        layout = extract(work)
        group = next(group for group in layout.groups if group.name == absorber_name)
        forbidden = tuple(
            _global_mask(other)
            for other in work.pickups()
            if not _positive_colours(other) <= target_colours
        )
        result, used_expanded, used_generated = _object_plan(
            layout,
            layout.positions,
            layout.selected,
            group,
            _global_mask(target),
            forbidden=forbidden,
            limit=limit - expanded,
            action_budget=layout.actions_left,
        )
        expanded += used_expanded
        generated += used_generated
        if result is None or not _apply(work, result[0]):
            return None, expanded, generated
        actions.extend(result[0])
    return tuple(actions), expanded, generated


def _pickup_search(env, limit: int) -> SearchResult:
    groups = env.groups()
    absorbers = sorted(
        name
        for name, data in groups.items()
        if data[names.KEY_CORE] is not None
        and data[names.KEY_CORE].name.startswith(names.PREFIX_ABSORBING_CORE)
    )
    targets = sorted(
        (
            (name, data[names.KEY_TARGET], _positive_colours(data[names.KEY_TARGET]))
            for name, data in groups.items()
            if data[names.KEY_TARGET] is not None and names.DECOY_GROUP_MARKER not in name
        ),
        key=lambda item: item[0],
    )
    if not absorbers or len(absorbers) != len(targets):
        return SearchResult(
            None, False, 0, 0, False, True, False,
            "pickup level needs one absorbing core per required colour target", limit,
        )
    candidates = []
    total_expanded = total_generated = 0
    for target_order in permutations(targets):
        result, expanded, generated = _pickup_assignment_plan(
            env, tuple(zip(absorbers, target_order)), limit
        )
        total_expanded += expanded
        total_generated += generated
        if result is not None:
            candidates.append(result)
    if not candidates:
        return SearchResult(
            None, False, min(total_expanded, limit), total_generated, True, False, False,
            "bounded constructive absorption search exhausted", limit,
        )
    actions = min(candidates, key=lambda route: (len(route), route))
    return SearchResult(
        actions, True, min(total_expanded, limit), total_generated, False, False, True,
        "native-checked absorption solution found", limit,
    )


def search(env_or_layout, limit=DEFAULT_LIMIT) -> SearchResult:
    """Return a bounded positive certificate across all official mechanics."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return SearchResult(
            None, False, 0, 0, True, False, False, "node limit 0 reached", limit
        )
    if not isinstance(env_or_layout, Layout):
        if env_or_layout.pickups() or env_or_layout.absorbing_cores():
            return _pickup_search(env_or_layout, limit)
        layout = extract(env_or_layout)
    else:
        layout = env_or_layout
    if layout.unsupported_reason:
        return SearchResult(
            None, False, 0, 0, False, True, False, layout.unsupported_reason, limit
        )
    if not layout.fragments:
        return SearchResult(
            None, False, 0, 0, False, False, True,
            "level has no draggable fragments", limit,
        )
    return _ordinary_search(layout, limit)


def solve(env_or_layout, limit=DEFAULT_LIMIT):
    """Return action triples or ``None``; detailed metadata stays on the function."""
    result = search(env_or_layout, limit=limit)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    return list(result.actions) if result.actions is not None else None


solve.result = None
solve.truncated = False
solve.unsupported = False
