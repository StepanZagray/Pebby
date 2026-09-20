"""Fail-closed extraction of RE86's complete stable native state."""

from dataclasses import dataclass
import hashlib

import numpy as np
from arcengine import GameState

from . import names


@dataclass(frozen=True)
class Shape:
    index: int
    name: str
    color: int
    x: int
    y: int
    width: int
    height: int
    color_cells: frozenset[tuple[int, int]]
    occupied_cells: frozenset[tuple[int, int]]
    selected: bool
    flexible: bool
    fixed_center: bool
    pixel_sha256: str


@dataclass(frozen=True)
class Layout:
    snapshot: object
    shapes: tuple[Shape, ...]
    requirements: tuple[tuple[int, int, int], ...]
    active_index: int | None
    steps_left: int
    level_index: int
    exact: bool
    unsupported: tuple[str, ...]
    obstacle_count: int
    dye_count: int


def _main_color(module, sprite):
    try:
        return int(getattr(module, "euqngakkse")(sprite))
    except (IndexError, TypeError, ValueError):
        return None


def extract(env):
    """Capture all mutable pixels; every shipped mechanic is represented."""
    reasons = []
    if env.state != GameState.NOT_FINISHED:
        reasons.append(f"terminal state {env.state.value}")
    if not env.stable():
        reasons.append("a native dye animation is still pending")
    targets = env.targets()
    if len(targets) != 1:
        reasons.append("exact search requires exactly one target sprite")
    elif (targets[0].x, targets[0].y, targets[0].width, targets[0].height) != (
            0, 0, names.FRAME_SIZE, names.FRAME_SIZE):
        reasons.append("the target must be an unshifted 64x64 native mask")

    shapes = []
    for index, sprite in enumerate(env.movables()):
        pixels = np.asarray(sprite.pixels)
        color = _main_color(env.module, sprite)
        if color is None:
            reasons.append(f"movable {index} has no intrinsic colour")
            color = -2
        selected = int(pixels[sprite.height // 2, sprite.width // 2]) == 0
        shapes.append(Shape(
            index=index,
            name=str(sprite.name),
            color=color,
            x=int(sprite.x),
            y=int(sprite.y),
            width=int(sprite.width),
            height=int(sprite.height),
            color_cells=frozenset(
                (int(row), int(col))
                for row, col in np.argwhere(pixels == color)
            ),
            occupied_cells=frozenset(
                (int(row), int(col))
                for row, col in np.argwhere(pixels != names.TRANSPARENT)
            ),
            selected=selected,
            flexible=names.TAG_FLEXIBLE in sprite.tags,
            fixed_center=names.TAG_FIXED_CENTER in sprite.tags,
            pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        ))
    selected = [shape.index for shape in shapes if shape.selected]
    if not shapes:
        reasons.append("level has no movable shapes")
    if len(selected) != 1:
        reasons.append("exact search requires exactly one selected movable")

    requirements = []
    if len(targets) == 1:
        pixels = np.asarray(targets[0].pixels)
        mask = ((pixels != names.TRANSPARENT)
                & (pixels != names.TARGET_GUIDE))
        for row, col in np.argwhere(mask):
            requirements.append((int(row), int(col), int(pixels[row, col])))
    if not requirements:
        reasons.append("target has no coloured anchor cells")
    if env.steps_left <= 0:
        reasons.append("native action budget is exhausted")
    reasons = tuple(dict.fromkeys(reasons))
    return Layout(
        snapshot=env.clone(),
        shapes=tuple(shapes),
        requirements=tuple(requirements),
        active_index=selected[0] if len(selected) == 1 else None,
        steps_left=env.steps_left,
        level_index=env.level_index,
        exact=not reasons,
        unsupported=reasons,
        obstacle_count=len(env.level.get_sprites_by_tag(names.TAG_OBSTACLE)),
        dye_count=len(env.level.get_sprites_by_tag(names.TAG_DYE)),
    )
