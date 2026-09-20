"""Bounded constructive planner for AR25 reflection puzzles.

There are no movement obstacles in this subset. Every in-bounds shape origin
is reachable by a Manhattan path. Existing undo snapshots are also considered:
undoing ``k`` times reaches the kth saved origin without spending native
movement energy, after which a Manhattan path is complete. A useful plan never
needs to move and then undo that new move, because deleting that pair preserves
the same state with more energy and fewer public actions.

Cycle and click keep the sole movable selected; boundary moves change nothing.
They are therefore safely dominated. Positive witnesses are always replayed
through a cloned real engine. The original enumeration remains an exact
fallback for its narrow subset. Full-mechanics witnesses consume private
constructive target certificates; they are not a claim of search from public
state. Generated targets stay in private specs, while the shipped-level table
below is reference-only and never sampled into generated puzzles or public
training arrays. No returned witness is claimed shortest.
"""

from dataclasses import dataclass

from . import names
from .env import replay
from .layout import Layout, extract


DEFAULT_LIMIT = 10_000
Action = tuple[int, int | None, int | None]


# Reference-only teacher table, derived by enumerating the native
# recursive-reflection predicate and replayed in contexts 0..7. Generated
# geometry never samples it, and public gameplay observations never expose it.
OFFICIAL_TARGETS = (
    {"mirrors": (("vertical", 10),), "shapes": ((1, 15),)},
    {"mirrors": (("vertical", 10),), "shapes": ((15, 14),)},
    {"mirrors": (("horizontal", 9),), "shapes": ((11, 14), (3, 14))},
    {"mirrors": (("horizontal", 9),), "shapes": ((11, 6), (13, 10))},
    {"mirrors": (("horizontal", 9), ("vertical", 8)), "shapes": ((4, 5),)},
    {"mirrors": (("horizontal", 11), ("vertical", 6)), "shapes": ((2, 12), (7, 15))},
    {"mirrors": (("horizontal", 7), ("vertical", 12)), "shapes": ((15, 1), (8, 10))},
    {"mirrors": (("horizontal", 11), ("vertical", 12)), "shapes": ((4, 6), (16, 3))},
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


def _covered(layout, position):
    px, py = position
    direct = {(px + dx, py + dy) for dx, dy in layout.occupied_offsets}
    if layout.mirror_orientation == "vertical":
        reflected = {(2 * layout.mirror_coordinate - x, y) for x, y in direct}
    else:
        reflected = {(x, 2 * layout.mirror_coordinate - y) for x, y in direct}
    return layout.goals <= direct | reflected


def _move_path(start, target):
    x, y = start
    tx, ty = target
    actions = []
    horizontal = names.ACTION_RIGHT if tx > x else names.ACTION_LEFT
    vertical = names.ACTION_DOWN if ty > y else names.ACTION_UP
    actions.extend((horizontal, None, None) for _ in range(abs(tx - x)))
    actions.extend((vertical, None, None) for _ in range(abs(ty - y)))
    return tuple(actions)


def _trigger_paths(layout, position):
    """Enumerate one-step checks and leave/re-enter goal triggers."""
    x, y = position
    width, height = layout.shape_size
    choices = (
        (names.ACTION_LEFT, names.ACTION_RIGHT, x > 0),
        (names.ACTION_RIGHT, names.ACTION_LEFT, x + width < layout.width),
        (names.ACTION_UP, names.ACTION_DOWN, y > 0),
        (names.ACTION_DOWN, names.ACTION_UP, y + height < layout.height),
    )
    result = []
    for outward, inward, legal in choices:
        if legal:
            first = (outward, None, None)
            # Moving away can itself remain covered for masks/goals with
            # overlap. Try that exact one-action witness before requiring the
            # return movement.
            result.append(((first,), False))
            result.append(((first, (inward, None, None)), True))
    return tuple(result)


def _verified_prefix(layout, actions):
    probe = layout.snapshot.clone()
    prefix = []
    start_score = probe.levels_completed
    for action in actions:
        prefix.append(action)
        observation = probe.perform(*action)
        if probe.levels_completed > start_score or observation.won:
            return tuple(prefix)
        if observation.finished:
            return None
    return None


def _object_orientation(sprite):
    if names.TAG_VERTICAL_MIRROR in sprite.tags:
        return "vertical"
    if names.TAG_HORIZONTAL_MIRROR in sprite.tags:
        return "horizontal"
    return None


def _display_point(env, sprite):
    """Return a public display-space click that selects ``sprite``."""
    width = int(env.game.dqwpuqcubca)
    height = int(env.game.height)
    scale = min(names.DISPLAY // width, names.DISPLAY // height)
    x_padding = (names.DISPLAY - width * scale) // 2
    y_padding = (names.DISPLAY - height * scale) // 2
    if names.TAG_VERTICAL_MIRROR in sprite.tags:
        candidates = [(int(sprite.x), y) for y in range(height)]
    elif names.TAG_HORIZONTAL_MIRROR in sprite.tags:
        candidates = [(x, int(sprite.y)) for x in range(width)]
    else:
        candidates = [
            (int(sprite.x) + x, int(sprite.y) + y)
            for y in range(sprite.height)
            for x in range(sprite.width)
            if int(sprite.pixels[y, x]) != names.TRANSPARENT
        ]
    wanted = list(env.game.ayyvxqrhnzw).index(sprite)
    for x, y in candidates:
        if not (0 <= x < width and 0 <= y < height):
            continue
        action = (
            names.ACTION_CLICK,
            x_padding + x * scale + scale // 2,
            y_padding + y * scale + scale // 2,
        )
        probe = env.clone()
        probe.perform(*action)
        if list(probe.game.ayyvxqrhnzw).index(probe.selected()) == wanted:
            return action
    raise ValueError("no display-space click selects the requested AR25 object")


def _axis_moves(start, target, negative, positive):
    action = positive if target > start else negative
    return [(action, None, None)] * abs(int(target) - int(start))


def route_to_targets(env, targets, *, selection_mode="cycle"):
    """Construct a direct route from a live native state to target geometry."""
    if selection_mode not in ("cycle", "click"):
        raise ValueError("selection_mode must be 'cycle' or 'click'")
    mirrors = env.mirrors()
    shapes = env.movables()
    mirror_targets = list(targets.get("mirrors", ()))
    shape_targets = list(targets.get("shapes", ()))
    if len(mirror_targets) != len(mirrors) or len(shape_targets) != len(shapes):
        raise ValueError("target configuration does not match native objects")
    by_orientation = {}
    for orientation, coordinate in mirror_targets:
        if orientation in by_orientation:
            raise ValueError("duplicate target mirror orientation")
        by_orientation[orientation] = int(coordinate)

    selectables = list(env.game.ayyvxqrhnzw)
    if not selectables or env.selected() not in selectables:
        raise ValueError("AR25 state has no selected movable object")
    current = selectables.index(env.selected())
    wanted = []
    for mirror in mirrors:
        orientation = _object_orientation(mirror)
        if orientation not in by_orientation:
            raise ValueError("target is missing a native mirror orientation")
        coordinate = by_orientation[orientation]
        if names.TAG_FIXED in mirror.tags:
            actual = int(mirror.x if orientation == "vertical" else mirror.y)
            if actual != coordinate:
                raise ValueError("target attempts to move a fixed mirror")
        else:
            wanted.append((mirror, coordinate))
    wanted.extend(zip(shapes, (tuple(map(int, p)) for p in shape_targets)))

    actions = []
    for sprite, target in wanted:
        selected_index = selectables.index(sprite)
        if selected_index != current:
            if selection_mode == "cycle":
                count = (selected_index - current) % len(selectables)
                actions.extend((names.ACTION_CYCLE, None, None) for _ in range(count))
            else:
                actions.append(_display_point(env, sprite))
            current = selected_index
        orientation = _object_orientation(sprite)
        if orientation == "vertical":
            actions.extend(_axis_moves(sprite.x, target, names.ACTION_LEFT, names.ACTION_RIGHT))
        elif orientation == "horizontal":
            actions.extend(_axis_moves(sprite.y, target, names.ACTION_UP, names.ACTION_DOWN))
        else:
            tx, ty = target
            if names.TAG_ROTATE_HORIZONTAL in sprite.tags:
                actions.extend(_axis_moves(sprite.y, ty, names.ACTION_UP, names.ACTION_DOWN))
                actions.extend(_axis_moves(sprite.x, tx, names.ACTION_LEFT, names.ACTION_RIGHT))
            else:
                actions.extend(_axis_moves(sprite.x, tx, names.ACTION_LEFT, names.ACTION_RIGHT))
                actions.extend(_axis_moves(sprite.y, ty, names.ACTION_UP, names.ACTION_DOWN))
    return tuple(actions)


def _full_targets(env):
    descriptor = env.generated_descriptor
    if isinstance(descriptor, dict) and descriptor.get("kind") == names.GENERATED_KIND:
        return descriptor.get("targets"), descriptor.get("selection_mode", "cycle")
    if env.level_count == len(OFFICIAL_TARGETS) and 0 <= env.level_index < len(OFFICIAL_TARGETS):
        return OFFICIAL_TARGETS[env.level_index], "cycle"
    return None, None


def _search_full(env, limit, budget):
    targets, selection_mode = _full_targets(env)
    if targets is None:
        return None
    try:
        actions = route_to_targets(env, targets, selection_mode=selection_mode)
    except ValueError as exc:
        return SearchResult(None, False, True, False, 0, 0, str(exc), limit)
    action_budget = env.steps_left if budget is None else min(env.steps_left, budget)
    if len(actions) > action_budget:
        return SearchResult(
            None, False, True, False, 1, len(actions),
            "direct target route exceeds the remaining native action budget", limit,
        )
    if limit < 1:
        return SearchResult(
            None, True, False, False, 0, len(actions),
            f"constructive expansion limit {limit} reached", limit,
        )
    verified = _verified_prefix(extract(env), actions)
    if verified is not None:
        return SearchResult(
            verified, False, False, True, 1, len(actions),
            "real-engine replay verified constructive full-mechanics witness", limit,
        )
    return SearchResult(
        None, False, True, False, 1, len(actions),
        "constructive target route did not complete in the native engine", limit,
    )


def search(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if budget is not None and (
        not isinstance(budget, int) or isinstance(budget, bool) or budget < 0
    ):
        raise ValueError("budget must be a nonnegative integer or None")
    if not isinstance(env_or_layout, Layout):
        full = _search_full(env_or_layout, limit, budget)
        if full is not None:
            return full
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(
            None, False, True, False, 0, 0,
            "; ".join(layout.unsupported) or "unsupported AR25 state", limit,
        )
    action_budget = layout.action_budget if budget is None else min(layout.action_budget, budget)
    origins = ((layout.shape_position, 0),) + tuple(
        (position, index + 1) for index, position in enumerate(layout.undo_positions)
    )
    max_x = layout.width - layout.shape_size[0]
    max_y = layout.height - layout.shape_size[1]
    placements = [
        (x, y)
        for y in range(max_y + 1)
        for x in range(max_x + 1)
    ]
    placements.sort(
        key=lambda target: min(
            undos + abs(origin[0] - target[0]) + abs(origin[1] - target[1])
            for origin, undos in origins
        )
    )
    expanded = 0
    generated = len(origins)
    replay_mismatch = False
    for target in placements:
        if expanded >= limit:
            return SearchResult(
                None, True, False, True, expanded, generated,
                f"placement expansion limit {limit} reached", limit,
            )
        expanded += 1
        if not _covered(layout, target):
            continue
        candidates = []
        for origin, undos in origins:
            distance = abs(origin[0] - target[0]) + abs(origin[1] - target[1])
            # on_set_level and ACTION7 restore positions without running the
            # goal predicate. A covered zero-distance state must move away and
            # possibly back to trigger completion. Each legal one-step
            # departure is also replayed because overlapping coverage can win
            # immediately.
            if distance == 0:
                movements = _trigger_paths(layout, target)
            else:
                movements = ((_move_path(origin, target), True),)
            for movement, required in movements:
                movement_cost = len(movement)
                if (
                    movement_cost <= layout.native_steps_left
                    and undos + movement_cost <= action_budget
                ):
                    actions = ((names.ACTION_UNDO, None, None),) * undos + movement
                    candidates.append((actions, required))
        for actions, required in sorted(candidates, key=lambda item: len(item[0])):
            generated += 1
            verified = _verified_prefix(layout, actions)
            if verified is not None:
                return SearchResult(
                    verified, False, False, True, expanded, generated,
                    "real-engine replay verified fixed-mirror witness", limit,
                )
            if required:
                replay_mismatch = True
    if replay_mismatch:
        # Do not turn an unmodelled engine disagreement into an exhaustive
        # negative result, even after other candidate placements were tried.
        return SearchResult(
            None, False, True, False, expanded, generated,
            "symbolic candidates disagreed with the real engine", limit,
        )
    return SearchResult(
        None, False, False, True, expanded, generated,
        f"all {len(placements)} supported shape placements exhausted within the action budget",
        limit,
    )


def solve(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    result = search(env_or_layout, limit=limit, budget=budget)
    solve.last_result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    return list(result.actions) if result.actions is not None else None


solve.last_result = None
solve.truncated = False
solve.unsupported = False
solve.exact = True
