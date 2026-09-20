"""Bounded exact live-state search for LP85.

Every successor is produced by the real vendored engine.  The action set is
enumerated from all 64x64 public display coordinates and collapsed only when
the ordered sequence of hit button tags is identical.  That preserves native
semantics for transparent button corners, stacked controls, shared cycles,
and controls moved by another cycle.

The visited key contains every sprite position.  Reaching the same positions
in fewer clicks leaves at least as much native budget, so longer histories are
soundly dominated.  If a work or explicit action bound is reached the result
is inconclusive (``truncated=True``); only full graph exhaustion is a negative
exact proof.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

from arcengine import GameState

from . import names
from .layout import Layout, extract


DEFAULT_NODE_LIMIT = 100_000
Action = tuple[int, int | None, int | None]
_CLICK_CACHE = {}


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...] | None
    truncated: bool
    unsupported: bool
    exact: bool
    expanded: int
    generated: int
    reason: str
    node_limit: int

    @property
    def solved(self):
        return self.actions is not None


@dataclass(frozen=True)
class GoalProjection:
    """Fail-closed finite model of LP85's only completion-relevant sprites."""

    paths: dict[str, tuple[tuple[int, int], ...]]
    indices: dict[str, dict[tuple[int, int], int]]
    actions: tuple[tuple[tuple[str, ...], Action, tuple[tuple[str, str], ...]], ...]
    start: tuple[tuple[tuple[int, int], ...], ...]
    targets: tuple[frozenset[tuple[int, int]], ...]
    memberships: dict[tuple[int, int], frozenset[str]]
    directions: dict[str, frozenset[str]]
    tag_counts: dict[str, int]
    passive_groups: frozenset[str]
    nested_groups: frozenset[str]


def _semantic_key(env):
    """All mutable LP85 sprite coordinates, in native list order."""
    return tuple((int(sprite.x), int(sprite.y)) for sprite in env.level._sprites)


def _state_key(env):
    """Public test/debug key including the remaining native budget."""
    return _semantic_key(env), env.steps_left


def _button_geometry(env):
    return (
        tuple(env.level.grid_size or ()),
        (
            int(env.game.camera.x),
            int(env.game.camera.y),
            int(env.game.camera.width),
            int(env.game.camera.height),
        ),
        tuple(
            (
                int(sprite.x),
                int(sprite.y),
                int(sprite.width),
                int(sprite.height),
                tuple(sprite.tags),
            )
            for sprite in env.level._sprites
            if sprite.tags and sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX)
        ),
    )


def _hit_tags(env, grid_x, grid_y):
    tags = []
    for sprite in env.level._sprites:
        if not sprite.tags or not sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX):
            continue
        if (
            grid_x >= sprite.x
            and grid_y >= sprite.y
            and grid_x < sprite.x + sprite.width
            and grid_y < sprite.y + sprite.height
        ):
            tags.append(sprite.tags[0])
    return tuple(tags)


def _button_clicks(env):
    """Yield one public click for every distinct native button-hit sequence."""
    geometry = _button_geometry(env)
    cached = _CLICK_CACHE.get(geometry)
    if cached is not None:
        yield from cached
        return
    representatives = {}
    for y in range(names.FRAME_SIZE):
        for x in range(names.FRAME_SIZE):
            point = env.game.camera.display_to_grid(x, y)
            if point is None:
                continue
            tags = _hit_tags(env, *point)
            if tags and tags not in representatives:
                representatives[tags] = (x, y)
    result = tuple((tags, click) for tags, click in representatives.items())
    _CLICK_CACHE[geometry] = result
    yield from result


def _goal_projection_model(env):
    """Recognize states where filler identity provably cannot affect play.

    LP85 rotates the first sprite whose top-left coordinate occupies each map
    cell.  The abstraction is exact only when every mapped cell contains
    exactly one supported movable sprite, all such sprites are mapped, every
    control is static/off-map, and every public control hit names valid maps.
    These conditions cover all eight official levels and all full generator
    tiers. Anything broader falls back to native-state search.
    """
    compiled = env.compiled_maps()
    if not isinstance(compiled, Mapping) or not compiled:
        return None
    paths = {}
    indices = {}
    union = set()
    for group, entry in compiled.items():
        if not isinstance(group, str) or not group or not isinstance(entry, Mapping):
            return None
        length = entry.get(names.MAP_LENGTH)
        positions = entry.get(names.MAP_POSITIONS)
        if (
            isinstance(length, bool)
            or not isinstance(length, Integral)
            or length < 2
            or not isinstance(positions, Mapping)
            or set(positions) != set(range(1, length + 1))
        ):
            return None
        path = []
        for number in range(1, length + 1):
            point = positions[number]
            if not hasattr(point, "x") or not hasattr(point, "y"):
                return None
            pixel = (int(point.x) * names.GRID_STEP, int(point.y) * names.GRID_STEP)
            if pixel in path:
                return None
            path.append(pixel)
        paths[group] = tuple(path)
        indices[group] = {point: index for index, point in enumerate(path)}
        union.update(path)

    buttons = [
        sprite for sprite in env.level._sprites
        if sprite.tags and isinstance(sprite.tags[0], str)
        and sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX)
    ]
    if not buttons:
        return None
    directions = {group: set() for group in paths}
    tag_counts = {}
    for sprite in buttons:
        tag = sprite.tags[0]
        parts = tag.split("_")
        if (
            len(parts) != 3
            or parts[0] != "button"
            or parts[1] not in paths
            or parts[2] not in ("L", "R")
        ):
            return None
        directions[parts[1]].add(parts[2])
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Empty cells on an unclickable/passive map and decorative tiles outside
    # every map exist in shipped tiers 7 and 8. They are immutable. Every cell
    # that any control can rotate must still be fully occupied by exactly one
    # supported movable, which is the soundness condition the projection needs.
    active_union = set().union(*(
        set(paths[group]) for group, value in directions.items() if value
    ))
    movable_tags = {names.TAG_TILE, names.TAG_GOAL, names.TAG_ALT_GOAL}
    for point in active_union:
        exact = [
            sprite for sprite in env.level._sprites
            if (int(sprite.x), int(sprite.y)) == point
        ]
        if len(exact) != 1:
            return None
        tags = set(exact[0].tags or ())
        if len(tags & movable_tags) != 1:
            return None
    all_movable = [
        sprite for sprite in env.level._sprites
        if set(sprite.tags or ()) & movable_tags
    ]
    for sprite in all_movable:
        position = (int(sprite.x), int(sprite.y))
        tags = set(sprite.tags or ())
        if position not in active_union and tags & {names.TAG_GOAL, names.TAG_ALT_GOAL}:
            return None
    if any((int(sprite.x), int(sprite.y)) in active_union for sprite in buttons):
        return None

    actions = []
    for tags, (x, y) in _button_clicks(env):
        effects = []
        for tag in tags:
            parts = tag.split("_")
            if (
                len(parts) != 3
                or parts[0] != "button"
                or parts[1] not in paths
                or parts[2] not in ("L", "R")
            ):
                return None
            effects.append((parts[1], parts[2]))
        if not effects:
            return None
        actions.append((tuple(tags), (names.ACTION_CLICK, int(x), int(y)), tuple(effects)))
    if not actions:
        return None

    starts = []
    targets = []
    for goal_tag, marker_tag in (
        (names.TAG_GOAL, names.TAG_TARGET_MARKER),
        (names.TAG_ALT_GOAL, names.TAG_ALT_TARGET_MARKER),
    ):
        goal_positions = tuple(sorted(
            (int(sprite.x), int(sprite.y))
            for sprite in env.level.get_sprites_by_tag(goal_tag)
        ))
        target_positions = frozenset(
            (int(sprite.x) + 1, int(sprite.y) + 1)
            for sprite in env.level.get_sprites_by_tag(marker_tag)
        )
        if len(goal_positions) != len(target_positions):
            return None
        if not set(goal_positions).issubset(union) or not target_positions.issubset(union):
            return None
        starts.append(goal_positions)
        targets.append(target_positions)

    memberships = {
        point: frozenset(group for group, path in paths.items() if point in path)
        for point in union
    }
    nested = set()
    groups = tuple(paths)
    path_sets = {group: set(path) for group, path in paths.items()}
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            if path_sets[left] < path_sets[right] or path_sets[right] < path_sets[left]:
                nested.update((left, right))
    return GoalProjection(
        paths=paths,
        indices=indices,
        actions=tuple(actions),
        start=tuple(starts),
        targets=tuple(targets),
        memberships=memberships,
        directions={group: frozenset(value) for group, value in directions.items()},
        tag_counts=tag_counts,
        passive_groups=frozenset(group for group, value in directions.items() if not value),
        nested_groups=frozenset(nested),
    )


def _project_points(points, effects, model):
    result = list(points)
    for group, direction in effects:
        path = model.paths[group]
        index = model.indices[group]
        delta = 1 if direction == "R" else -1
        result = [
            path[(index[point] + delta) % len(path)] if point in index else point
            for point in result
        ]
    return tuple(result)


def _project_state(state, effects, model):
    return tuple(tuple(sorted(_project_points(points, effects, model))) for points in state)


def _projection_complete(state, model):
    return all(model.targets[index].issubset(points) for index, points in enumerate(state))


def _live_goal_state(env):
    return tuple(
        tuple(sorted((int(sprite.x), int(sprite.y)) for sprite in env.level.get_sprites_by_tag(tag)))
        for tag in (names.TAG_GOAL, names.TAG_ALT_GOAL)
    )


def _validated_goal_projection(env):
    """Require every start-state click to agree with one full native step."""
    model = _goal_projection_model(env)
    if model is None:
        return None
    before_score = env.levels_completed
    before_controls = tuple(
        (sprite.tags[0], int(sprite.x), int(sprite.y))
        for sprite in env.level._sprites
        if sprite.tags and isinstance(sprite.tags[0], str)
        and sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX)
    )
    for _, action, effects in model.actions:
        expected = _project_state(model.start, effects, model)
        probe = env.clone()
        observation = probe.perform(*action)
        completed = probe.levels_completed > before_score or observation.state == GameState.WIN
        if completed != _projection_complete(expected, model):
            return None
        if not completed and _live_goal_state(probe) != expected:
            return None
        if not completed:
            after_controls = tuple(
                (sprite.tags[0], int(sprite.x), int(sprite.y))
                for sprite in probe.level._sprites
                if sprite.tags and isinstance(sprite.tags[0], str)
                and sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX)
            )
            if after_controls != before_controls:
                return None
    return model


def _goal_projection_search(layout, limit, node_limit):
    """Shortest path over the exact completion-relevant goal projection."""
    env = layout.snapshot
    model = _validated_goal_projection(env)
    if model is None:
        return None
    start = model.start
    parent = {start: None}
    depth = {start: 0}
    frontier = deque([start])
    expanded = generated = 0
    action_cutoff = False
    while frontier:
        state = frontier.popleft()
        cost = depth[state]
        if expanded >= node_limit:
            return SearchResult(
                None, True, False, True, expanded, generated,
                f"goal-projection expansion limit {node_limit} reached", node_limit,
            )
        expanded += 1
        if limit is not None and cost >= limit:
            action_cutoff = True
            continue
        if cost >= env.steps_left:
            continue
        for _, action, effects in model.actions:
            successor = _project_state(state, effects, model)
            generated += 1
            if _projection_complete(successor, model):
                actions = _unwind(parent, state, action)
                probe = env.clone()
                before = probe.levels_completed
                valid = True
                for action_index, candidate in enumerate(actions):
                    observation = probe.perform(*candidate)
                    completed = probe.levels_completed > before or observation.state == GameState.WIN
                    if completed != (action_index == len(actions) - 1):
                        valid = False
                        break
                if valid:
                    return SearchResult(
                        actions, False, False, True, expanded, generated,
                        "exact shortest goal-projection witness with full native replay",
                        node_limit,
                    )
                return None
            if successor == state or successor in parent:
                continue
            parent[successor] = (state, action)
            depth[successor] = cost + 1
            frontier.append(successor)
    if action_cutoff:
        return SearchResult(
            None, True, False, True, expanded, generated,
            f"action limit {limit} reached before the exact goal graph was exhausted",
            node_limit,
        )
    return SearchResult(
        None, False, False, True, expanded, generated,
        "complete goal-position graph exhausted within the native budget", node_limit,
    )


def solution_mechanics(env, actions):
    """Summarize mechanics actually exercised by an exact native witness."""
    model = _validated_goal_projection(env)
    if model is None:
        raise ValueError("LP85 solution mechanics require an exact goal projection")
    by_action = {action: (tags, effects) for tags, action, effects in model.actions}
    state = model.start
    used_groups = set()
    used_signatures = set()
    stacked_clicks = one_way_clicks = duplicate_clicks = 0
    overlap_transitions = passive_transitions = nested_clicks = 0
    max_stack = 0
    initially_satisfied = {
        point
        for kind, targets in enumerate(model.targets)
        for point in targets
        if point in state[kind]
    }
    displaced = set()
    for raw in actions:
        action = tuple(raw)
        if action not in by_action:
            raise ValueError(f"witness action {action!r} is not a canonical native control hit")
        tags, effects = by_action[action]
        used_signatures.add(tags)
        groups = {group for group, _ in effects}
        used_groups.update(groups)
        max_stack = max(max_stack, len(tags))
        if len(tags) > 1:
            stacked_clicks += 1
        if all(len(model.directions[group]) == 1 for group in groups):
            one_way_clicks += 1
        if any(model.tag_counts.get(tag, 0) > 1 for tag in tags):
            duplicate_clicks += 1
        if groups & model.nested_groups:
            nested_clicks += 1
        projected = []
        for points in state:
            moved = _project_points(points, effects, model)
            for before, after in zip(points, moved):
                if before == after:
                    continue
                if len(model.memberships[before]) > 1 or len(model.memberships[after]) > 1:
                    overlap_transitions += 1
                if (
                    model.memberships[before] & model.passive_groups
                    or model.memberships[after] & model.passive_groups
                ):
                    passive_transitions += 1
            projected.append(tuple(sorted(moved)))
        state = tuple(projected)
        occupied = {point for points in state for point in points}
        displaced.update(point for point in initially_satisfied if point not in occupied)
    occupied = {point for points in state for point in points}
    return {
        "used_groups": sorted(used_groups),
        "used_group_count": len(used_groups),
        "used_effect_signature_count": len(used_signatures),
        "stacked_clicks": stacked_clicks,
        "max_stack_used": max_stack,
        "one_way_clicks": one_way_clicks,
        "duplicate_control_clicks": duplicate_clicks,
        "overlap_goal_transitions": overlap_transitions,
        "passive_overlap_transitions": passive_transitions,
        "nested_cycle_clicks": nested_clicks,
        "initially_satisfied_goals": len(initially_satisfied),
        "temporarily_displaced_initial_target": len(displaced & occupied),
        "all_targets_satisfied": _projection_complete(state, model),
    }


def _unwind(parent, state, tail):
    actions = [tail]
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return tuple(actions)


def _raw_cycle_positions(raw_map):
    """Return a numbered map as ordered ``(x, y)`` positions, or ``None``."""
    if (
        not isinstance(raw_map, Sequence)
        or isinstance(raw_map, (str, bytes))
        or not raw_map
    ):
        return None
    numbered = {}
    width = None
    for y, row in enumerate(raw_map):
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or not row
        ):
            return None
        if width is None:
            width = len(row)
        elif len(row) != width:
            return None
        for x, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            if value == -1:
                continue
            if value < 1 or value in numbered:
                return None
            numbered[value] = (x, y)
    if len(numbered) < 2 or set(numbered) != set(range(1, len(numbered) + 1)):
        return None
    return tuple(numbered[index] for index in range(1, len(numbered) + 1))


def _generated_cycle_model(env):
    """Recognize the exact independent-cycle subset emitted by our generator.

    The private generated-map field is only a signal to attempt recognition.
    Every fact used by the shortcut is checked against the compiled live map,
    current sprites, target markers, and full-pixel native click effects.  A
    mismatch returns ``None`` so broader native search remains authoritative.
    """
    raw_maps = env.level.get_data(names.KEY_GENERATED_MAP)
    compiled = env.compiled_maps()
    if not isinstance(raw_maps, Mapping) or not raw_maps:
        return None
    if not isinstance(compiled, Mapping) or set(raw_maps) != set(compiled):
        return None

    paths = {}
    occupied = set()
    if any(not isinstance(group, str) or not group for group in raw_maps):
        return None
    for group in sorted(raw_maps):
        if not group:
            return None
        path = _raw_cycle_positions(raw_maps[group])
        entry = compiled.get(group)
        if path is None or not isinstance(entry, Mapping):
            return None
        length = entry.get(names.MAP_LENGTH)
        positions = entry.get(names.MAP_POSITIONS)
        if (
            isinstance(length, bool)
            or not isinstance(length, Integral)
            or length != len(path)
            or not isinstance(positions, Mapping)
            or set(positions) != set(range(1, length + 1))
        ):
            return None
        compiled_path = []
        for index in range(1, length + 1):
            point = positions[index]
            if not hasattr(point, "x") or not hasattr(point, "y"):
                return None
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in (point.x, point.y)
            ):
                return None
            compiled_path.append((int(point.x), int(point.y)))
        compiled_path = tuple(compiled_path)
        if compiled_path != path or occupied.intersection(path):
            return None
        occupied.update(path)
        paths[group] = path

    clicks = {}
    for tags, point in _button_clicks(env):
        if len(tags) != 1:
            return None
        parts = tags[0].split("_")
        if (
            len(parts) != 3
            or parts[0] != "button"
            or parts[1] not in paths
            or parts[2] not in ("L", "R")
        ):
            return None
        key = (parts[1], parts[2])
        if key in clicks:
            return None
        clicks[key] = (names.ACTION_CLICK, int(point[0]), int(point[1]))
    expected_clicks = {
        (group, direction) for group in paths for direction in ("L", "R")
    }
    if set(clicks) != expected_clicks:
        return None

    pixel_paths = {
        group: tuple((x * names.GRID_STEP, y * names.GRID_STEP) for x, y in path)
        for group, path in paths.items()
    }
    expected_positions = {point for path in pixel_paths.values() for point in path}
    movable_tags = {names.TAG_TILE, names.TAG_GOAL, names.TAG_ALT_GOAL}
    movable = {}
    markers = []
    controls = {}
    cycle_occupants = {point: 0 for point in expected_positions}
    for sprite in env.level._sprites:
        tags = set(sprite.tags or ())
        position = (int(sprite.x), int(sprite.y))
        if position in cycle_occupants:
            cycle_occupants[position] += 1
        first_tag = sprite.tags[0] if sprite.tags else ""
        if not isinstance(first_tag, str):
            return None
        if "button" in first_tag:
            parts = first_tag.split("_")
            if (
                len(parts) != 3
                or parts[0] != "button"
                or parts[1] not in paths
                or parts[2] not in ("L", "R")
            ):
                return None
            key = (parts[1], parts[2])
            if key in controls or position in expected_positions:
                return None
            controls[key] = sprite
        selected = tags & movable_tags
        if selected:
            if len(selected) != 1:
                return None
            if position in movable:
                return None
            movable[position] = next(iter(selected))
        marker_tags = tags & {names.TAG_TARGET_MARKER, names.TAG_ALT_TARGET_MARKER}
        if marker_tags:
            if len(marker_tags) != 1:
                return None
            markers.append((
                (int(sprite.x) + 1, int(sprite.y) + 1),
                next(iter(marker_tags)),
            ))

    if (
        set(controls) != expected_clicks
        or set(movable) != expected_positions
        or any(count != 1 for count in cycle_occupants.values())
        or len(markers) != len(paths)
    ):
        return None

    model = []
    claimed_markers = set()
    for group, path in pixel_paths.items():
        marker_matches = [
            (index, marker_tag)
            for index, point in enumerate(path)
            for marker_position, marker_tag in markers
            if marker_position == point
        ]
        if len(marker_matches) != 1:
            return None
        target, marker_tag = marker_matches[0]
        marker_key = (path[target], marker_tag)
        if marker_key in claimed_markers:
            return None
        claimed_markers.add(marker_key)
        goal_tag = (
            names.TAG_GOAL
            if marker_tag == names.TAG_TARGET_MARKER
            else names.TAG_ALT_GOAL
        )
        goal_positions = [
            index for index, point in enumerate(path) if movable[point] == goal_tag
        ]
        if len(goal_positions) != 1:
            return None
        if any(
            movable[point] in {names.TAG_GOAL, names.TAG_ALT_GOAL}
            for index, point in enumerate(path)
            if index != goal_positions[0]
        ):
            return None
        model.append((
            group,
            len(path),
            goal_positions[0],
            target,
            clicks[(group, "L")],
            clicks[(group, "R")],
        ))
    if len(claimed_markers) != len(markers):
        return None
    return tuple(model)


def _generated_cycle_search(layout, limit, node_limit):
    """Solve a recognized generated layout algebraically, then replay it live."""
    env = layout.snapshot
    model = _generated_cycle_model(env)
    if model is None:
        return None

    actions = []
    for _, length, current, target, left, right in model:
        left_steps = (current - target) % length
        right_steps = (target - current) % length
        if left_steps <= right_steps:
            actions.extend([left] * left_steps)
        else:
            actions.extend([right] * right_steps)
    if not actions:
        # The native game checks completion only after a control click.  This
        # state is not emitted initially, but can occur in a supplied snapshot.
        # Move one independent ring away and back to trigger that check.
        actions.extend((model[0][5], model[0][4]))

    required = len(actions)
    reachable_depth = min(required, env.steps_left)
    if limit is not None and limit < reachable_depth:
        return SearchResult(
            None,
            True,
            False,
            True,
            1,
            0,
            f"action limit {limit} is below the exact generated minimum {required}",
            node_limit,
        )
    if required > env.steps_left:
        return SearchResult(
            None,
            False,
            False,
            True,
            1,
            0,
            f"exact generated minimum {required} exceeds native remaining budget {env.steps_left}",
            node_limit,
        )

    probe = env.clone()
    before = probe.levels_completed
    for index, action in enumerate(actions):
        observation = probe.perform(*action)
        completed = (
            probe.levels_completed > before or observation.state == GameState.WIN
        )
        if completed != (index == required - 1):
            return None
        if observation.state == GameState.GAME_OVER:
            return None
    return SearchResult(
        tuple(actions),
        False,
        False,
        True,
        1,
        required,
        "minimal live real-engine witness for independent generated cycles",
        node_limit,
    )


def search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Breadth-first search with separate action and expansion bounds."""
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer or None")
    if (
        isinstance(node_limit, bool)
        or not isinstance(node_limit, int)
        or node_limit < 1
    ):
        raise ValueError("node_limit must be a positive integer")
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(
            None,
            False,
            True,
            False,
            0,
            0,
            "; ".join(layout.unsupported) or "unsupported live state",
            node_limit,
        )

    projected_result = _goal_projection_search(layout, limit, node_limit)
    if projected_result is not None:
        return projected_result

    generated_result = _generated_cycle_search(layout, limit, node_limit)
    if generated_result is not None:
        return generated_result

    start_env = layout.snapshot.clone()
    start = _semantic_key(start_env)
    parent = {start: None}
    depth = {start: 0}
    frontier = deque([(start, start_env)])
    expanded = generated = 0
    action_cutoff = False

    while frontier:
        state, env = frontier.popleft()
        cost = depth[state]
        if expanded >= node_limit:
            return SearchResult(
                None,
                True,
                False,
                True,
                expanded,
                generated,
                f"configuration expansion limit {node_limit} reached",
                node_limit,
            )
        expanded += 1
        if limit is not None and cost >= limit:
            action_cutoff = True
            continue

        for _, (x, y) in _button_clicks(env):
            successor = env.clone()
            score = successor.levels_completed
            observation = successor.perform(names.ACTION_CLICK, x, y)
            generated += 1
            action = (names.ACTION_CLICK, x, y)
            if (
                successor.levels_completed > score
                or observation.state == GameState.WIN
            ):
                return SearchResult(
                    _unwind(parent, state, action),
                    False,
                    False,
                    True,
                    expanded,
                    generated,
                    "real-engine click completed the level",
                    node_limit,
                )
            if observation.state == GameState.GAME_OVER:
                continue
            nxt = _semantic_key(successor)
            if nxt == state or nxt in parent:
                continue
            parent[nxt] = (state, action)
            depth[nxt] = cost + 1
            frontier.append((nxt, successor))

    if action_cutoff:
        return SearchResult(
            None,
            True,
            False,
            True,
            expanded,
            generated,
            f"action limit {limit} reached before the exact graph was exhausted",
            node_limit,
        )
    return SearchResult(
        None,
        False,
        False,
        True,
        expanded,
        generated,
        "complete reachable position graph exhausted within the native budget",
        node_limit,
    )


def solve(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Return action triples or ``None`` and expose the detailed last status."""
    result = search(env_or_layout, limit=limit, node_limit=node_limit)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    solve.reason = result.reason
    return list(result.actions) if result.actions is not None else None


solve.result = None
solve.truncated = False
solve.unsupported = False
solve.exact = False
solve.reason = ""
