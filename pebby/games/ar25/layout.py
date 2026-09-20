"""Extract the exact single-shape, fixed-mirror AR25 subset."""

from dataclasses import dataclass

from arcengine import GameState

from . import names


@dataclass(frozen=True)
class Layout:
    snapshot: object
    width: int
    height: int
    shape_position: tuple[int, int]
    shape_size: tuple[int, int]
    occupied_offsets: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    mirror_orientation: str | None
    mirror_coordinate: int | None
    undo_positions: tuple[tuple[int, int], ...]
    native_steps_left: int
    action_budget: int
    generated: bool
    exact: bool
    unsupported: tuple[str, ...]


def extract(env):
    game = env.game
    movables = env.movables()
    mirrors = env.mirrors()
    goals = frozenset((int(goal.x), int(goal.y)) for goal in env.goals())
    reasons = []
    if env.state != GameState.NOT_FINISHED:
        reasons.append("level is already terminal")
    if len(movables) != 1:
        reasons.append("exact subset requires exactly one movable shape")
    fixed_mirrors = [mirror for mirror in mirrors if names.TAG_FIXED in mirror.tags]
    if len(mirrors) != 1 or len(fixed_mirrors) != 1:
        reasons.append("exact subset requires exactly one fixed mirror")

    shape = movables[0] if len(movables) == 1 else None
    mirror = fixed_mirrors[0] if len(fixed_mirrors) == 1 else None
    if shape is not None:
        forbidden = {
            names.TAG_ROTATE_VERTICAL,
            names.TAG_ROTATE_HORIZONTAL,
            names.TAG_REFLECT_HORIZONTAL_ONLY,
            names.TAG_REFLECT_VERTICAL_ONLY,
        }
        if forbidden.intersection(shape.tags):
            reasons.append("rotation and restricted-reflection shapes are unsupported")
        if env.selected() is not shape:
            reasons.append("the sole movable shape is not selected")
        occupied = frozenset(
            (int(x), int(y))
            for y in range(shape.height)
            for x in range(shape.width)
            if int(shape.pixels[y, x]) != names.TRANSPARENT
        )
        position = (int(shape.x), int(shape.y))
        size = (int(shape.width), int(shape.height))
    else:
        occupied, position, size = frozenset(), (0, 0), (0, 0)

    orientation = None
    coordinate = None
    if mirror is not None:
        vertical = names.TAG_VERTICAL_MIRROR in mirror.tags
        horizontal = names.TAG_HORIZONTAL_MIRROR in mirror.tags
        if vertical == horizontal:
            reasons.append("mirror orientation is ambiguous")
        elif vertical:
            orientation, coordinate = "vertical", int(mirror.x)
        else:
            orientation, coordinate = "horizontal", int(mirror.y)

    undo_positions = []
    if shape is not None:
        for state in reversed(getattr(game, names.ATTR_HISTORY)):
            positions = state.get(names.ATTR_MOVABLES, ())
            if len(positions) != 1:
                reasons.append("undo snapshot is outside the single-shape subset")
                break
            undo_positions.append(tuple(map(int, positions[0])))

    if not goals:
        reasons.append("level has no goal cells")
    if bool(getattr(game, names.ATTR_PENDING_WIN)):
        reasons.append("level completion is already pending")
    descriptor = env.generated_descriptor
    generated = isinstance(descriptor, dict) and descriptor.get("kind") == names.GENERATED_KIND
    return Layout(
        snapshot=env.clone(),
        width=int(game.dqwpuqcubca),
        height=int(game.height),
        shape_position=position,
        shape_size=size,
        occupied_offsets=occupied,
        goals=goals,
        mirror_orientation=orientation,
        mirror_coordinate=coordinate,
        undo_positions=tuple(undo_positions),
        native_steps_left=env.native_steps_left,
        action_budget=env.steps_left,
        generated=generated,
        exact=not reasons,
        unsupported=tuple(reasons),
    )
