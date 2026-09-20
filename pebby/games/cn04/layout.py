"""Exact symbolic extraction for every shipped CN04 mechanic.

CN04 treats sprites that start at the same coordinate as a stack of alternate
pieces. A stack has one visible member, a shared position, fixed per-member
rotations, and all stacks share one bouncing direction bit. Singletons rotate
normally. The state represented here is::

    (group transforms, active alternate indices, selected group, cycle forward)

where each transform is ``(x, y, clockwise quarter turns)``. This is enough to
reproduce movement, rotation, ACTION5 stack cycling, selected-zero-pixel click
cycling, visibility transfer, clamping, click precedence, and pin completion.
"""

from dataclasses import dataclass

import numpy as np

from . import names


class UnsupportedLayout(Exception):
    """The live level uses a representation the exact planner cannot model."""


def _rotated(array, quarter_turns):
    """Render an unscaled pixel array at the engine's clockwise rotation."""
    return np.rot90(array, k=(-quarter_turns) % 4)


@dataclass(frozen=True)
class Piece:
    name: str
    pixels: tuple[tuple[int, ...], ...]
    initial_rotation: int
    sprite_order: int

    def array(self):
        return np.asarray(self.pixels, dtype=int)

    def rendered(self, quarter_turns):
        return _rotated(self.array(), quarter_turns)

    def size(self, quarter_turns):
        height, width = self.rendered(quarter_turns).shape
        return int(width), int(height)

    def opaque(self, quarter_turns):
        rendered = self.rendered(quarter_turns)
        ys, xs = np.nonzero(rendered >= 0)
        return tuple((int(x), int(y)) for y, x in zip(ys, xs))

    def pins(self, quarter_turns):
        rendered = self.rendered(quarter_turns)
        result = []
        for y, x in np.argwhere((rendered == names.PIN_A) | (rendered == names.PIN_B)):
            result.append((int(x), int(y), int(rendered[y, x])))
        return tuple(result)

    def cycle_cells(self, quarter_turns, grey_masking):
        """Rendered cells whose selected display value is exactly zero."""
        rendered = self.rendered(quarter_turns)
        result = []
        for y, x in np.argwhere(rendered >= 0):
            value = int(rendered[y, x])
            if value == 0 or (not grey_masking and value not in (8, 13, 3)):
                result.append((int(x), int(y)))
        return tuple(result)


@dataclass(frozen=True)
class Group:
    index: int
    alternatives: tuple[Piece, ...]

    @property
    def is_stack(self):
        return len(self.alternatives) > 1


class Layout:
    """Static groups plus the exact current dynamic state."""

    def __init__(self, *, groups, start, grid_size, steps_left, grey_masking,
                 unsupported=()):
        self.groups = tuple(groups)
        self.start = start
        self.grid_size = tuple(grid_size)
        self.steps_left = int(steps_left)
        self.grey_masking = bool(grey_masking)
        self.unsupported = tuple(unsupported)

    @property
    def pieces(self):
        """All alternatives, retained as a compatibility/introspection view."""
        return tuple(piece for group in self.groups for piece in group.alternatives)

    @property
    def stacks(self):
        return tuple(tuple(piece.name for piece in group.alternatives)
                     for group in self.groups if group.is_stack)

    @property
    def exact(self):
        return not self.unsupported

    def piece(self, state, group_index):
        return self.groups[group_index].alternatives[state[1][group_index]]

    def visible(self, state):
        transforms = state[0]
        entries = []
        for group_index, transform in enumerate(transforms):
            piece = self.piece(state, group_index)
            entries.append((piece.sprite_order, group_index, piece, transform))
        return tuple(sorted(entries))

    def complete(self, state_or_transforms, active=None):
        """Upstream's true predicate: every colour/location count is two."""
        if active is None:
            transforms, active = state_or_transforms[0], state_or_transforms[1]
        else:
            transforms = state_or_transforms
        counts = {}
        pin_count = 0
        for group, alternate, (px, py, rotation) in zip(self.groups, active, transforms):
            piece = group.alternatives[alternate]
            for dx, dy, colour in piece.pins(rotation):
                key = (px + dx, py + dy, colour)
                counts[key] = counts.get(key, 0) + 1
                pin_count += 1
        return bool(pin_count) and all(count == 2 for count in counts.values())

    def _occupied(self, state):
        occupied = []
        for order, group_index, piece, (x, y, rotation) in self.visible(state):
            cells = {(x + dx, y + dy) for dx, dy in piece.opaque(rotation)}
            occupied.append((order, group_index, cells))
        return occupied

    def first_hit(self, state, cell):
        for _, group_index, cells in self._occupied(state):
            if cell in cells:
                return group_index
        return None

    def click_for(self, state, target):
        """A display click selecting ``target``, or ``None`` if occluded."""
        transforms = state[0]
        piece = self.piece(state, target)
        x, y, rotation = transforms[target]
        for dx, dy in piece.opaque(rotation):
            cell = (x + dx, y + dy)
            if self.first_hit(state, cell) == target:
                return names.grid_to_display(*cell, self.grid_size)
        return None

    def cycle_click_for(self, state):
        """A click that cycles the selected stack through its zero marker."""
        transforms, _, selected, _ = state
        if selected is None or not self.groups[selected].is_stack:
            return None
        piece = self.piece(state, selected)
        x, y, rotation = transforms[selected]
        for dx, dy in piece.cycle_cells(rotation, self.grey_masking):
            cell = (x + dx, y + dy)
            if self.first_hit(state, cell) == selected:
                return names.grid_to_display(*cell, self.grid_size)
        return None

    def is_cycle_cell(self, state, cell):
        """Whether an arbitrary grid cell has selected display value zero."""
        transforms, _, selected, _ = state
        if selected is None or not self.groups[selected].is_stack:
            return False
        piece = self.piece(state, selected)
        x, y, rotation = transforms[selected]
        local = (cell[0] - x, cell[1] - y)
        return (local in piece.cycle_cells(rotation, self.grey_masking)
                and self.first_hit(state, cell) == selected)

    def describe(self):
        visible_pins = sum(len(self.piece(self.start, index).pins(transform[2]))
                           for index, transform in enumerate(self.start[0]))
        alternates = sum(len(group.alternatives) for group in self.groups)
        return (f"{self.grid_size[0]}x{self.grid_size[1]} | groups {len(self.groups)} | "
                f"alternates {alternates} | visible pins {visible_pins} | "
                f"steps left {self.steps_left} | exact={self.exact}")


def extract(env):
    """Extract the current engine state without approximating stack mechanics."""
    sprites = list(env.sprites())
    order = {id(sprite): index for index, sprite in enumerate(sprites)}
    unsupported = []
    groups = []
    transforms = []
    active_indices = []
    selected_group = None
    selected = env.selected()
    seen = set()

    for sprite in sprites:
        stack = tuple(env.stacks().get(sprite, (sprite,)))
        key = tuple(id(member) for member in stack)
        if key in seen:
            continue
        seen.add(key)
        alternatives = []
        visible = []
        for alternate_index, member in enumerate(stack):
            if int(getattr(member, "scale", 1)) != 1:
                unsupported.append(f"scaled sprite {member.name}")
            if getattr(member, "_mirror_ud", False) or getattr(member, "_mirror_lr", False):
                unsupported.append(f"mirrored sprite {member.name}")
            if int(member.rotation) % 90:
                unsupported.append(f"non-quarter rotation on {member.name}")
            pixels = np.asarray(env.original_pixels(member), dtype=int)
            piece = Piece(
                name=member.name,
                pixels=tuple(tuple(int(value) for value in row) for row in pixels.tolist()),
                initial_rotation=(int(member.rotation) // 90) % 4,
                sprite_order=order[id(member)],
            )
            if not piece.pins(piece.initial_rotation):
                unsupported.append(f"pinless alternate {member.name}")
            alternatives.append(piece)
            if member.is_visible:
                visible.append((alternate_index, member))
        if len(visible) != 1:
            unsupported.append(
                f"stack at ({sprite.x},{sprite.y}) has {len(visible)} visible members"
            )
            active_index, active_sprite = ((0, stack[0]) if not visible else visible[0])
        else:
            active_index, active_sprite = visible[0]
        group_index = len(groups)
        groups.append(Group(group_index, tuple(alternatives)))
        transforms.append((int(active_sprite.x), int(active_sprite.y),
                           (int(active_sprite.rotation) // 90) % 4))
        active_indices.append(active_index)
        if active_sprite is selected:
            selected_group = group_index

    if not groups:
        unsupported.append("level has no visible piece groups")
    if selected is not None and selected_group is None:
        unsupported.append("selected sprite is not the visible member of its stack")

    start = (tuple(transforms), tuple(active_indices), selected_group,
             bool(env.cycle_forward()))
    return Layout(
        groups=groups,
        start=start,
        grid_size=env.grid_size,
        steps_left=env.steps_left(),
        grey_masking=env.grey_masking(),
        unsupported=unsupported,
    )
