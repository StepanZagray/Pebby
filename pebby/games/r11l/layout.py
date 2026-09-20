"""Immutable symbolic view of an R11L level.

Collision masks are extracted from ``Sprite.render()``.  ARCEngine treats
every value except -1 as opaque, including the targets' invisible -2 pixels,
so the planner uses the same rule rather than approximating rectangles.
"""

from dataclasses import dataclass

import numpy as np

from . import names


Point = tuple[int, int]


def _local_mask(sprite) -> frozenset[Point]:
    pixels = np.asarray(sprite.render())
    return frozenset(
        (int(x), int(y))
        for y, x in np.argwhere(pixels != -1)
    )


def _global_mask(sprite) -> frozenset[Point]:
    return frozenset((sprite.x + x, sprite.y + y) for x, y in _local_mask(sprite))


@dataclass(frozen=True)
class Fragment:
    name: str
    group: str
    position: Point
    mask: frozenset[Point]
    width: int
    height: int


@dataclass(frozen=True)
class Group:
    name: str
    fragment_indices: tuple[int, ...]
    core_mask: frozenset[Point]
    target_mask: frozenset[Point] | None
    required: bool


@dataclass(frozen=True)
class Layout:
    fragments: tuple[Fragment, ...]
    groups: tuple[Group, ...]
    walls: frozenset[Point]
    hazards: frozenset[Point]
    positions: tuple[Point, ...]
    selected: int
    actions_left: int
    unsupported_reason: str | None = None


def extract(env) -> Layout:
    """Extract the current action-boundary state from a real-engine ``Env``."""
    engine_fragments = env.fragments()
    index_by_identity = {id(sprite): i for i, sprite in enumerate(engine_fragments)}
    selected_sprite = env.selected()
    selected = index_by_identity.get(id(selected_sprite), -1)

    group_by_fragment = {}
    for group_name, data in env.groups().items():
        for sprite in data[names.KEY_FRAGMENTS]:
            group_by_fragment[id(sprite)] = group_name

    fragments = tuple(
        Fragment(
            name=sprite.name,
            group=group_by_fragment.get(id(sprite), ""),
            position=(sprite.x, sprite.y),
            mask=_local_mask(sprite),
            width=sprite.width,
            height=sprite.height,
        )
        for sprite in engine_fragments
    )

    groups = []
    unsupported = None
    has_absorbers = bool(env.absorbing_cores())

    for group_name, data in sorted(env.groups().items()):
        core = data[names.KEY_CORE]
        target = data[names.KEY_TARGET]
        indices = tuple(
            index_by_identity[id(sprite)]
            for sprite in data[names.KEY_FRAGMENTS]
            if id(sprite) in index_by_identity
        )
        required = target is not None and names.DECOY_GROUP_MARKER not in group_name
        if required and (core is None or not indices) and not has_absorbers:
            unsupported = unsupported or f"required group {group_name!r} has no movable core"
        groups.append(
            Group(
                name=group_name,
                fragment_indices=indices,
                core_mask=frozenset() if core is None else _local_mask(core),
                target_mask=None if target is None else _global_mask(target),
                required=required,
            )
        )

    return Layout(
        fragments=fragments,
        groups=tuple(groups),
        walls=frozenset().union(*(_global_mask(wall) for wall in env.walls())),
        hazards=frozenset().union(*(_global_mask(hazard) for hazard in env.hazards())),
        positions=tuple(fragment.position for fragment in fragments),
        selected=selected,
        actions_left=max(0, env.actions_left),
        unsupported_reason=unsupported,
    )
