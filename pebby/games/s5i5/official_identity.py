"""Native semantic identities used only to exclude shipped S5I5 copies.

The generated-spec v3 identity is the public split identity.  This stricter
comparison is derived from initialized engine state so alternate JSON cell
encodings, sprite names, control art, and private certificates cannot hide a
copy of a shipped puzzle.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json

from . import names
from .env import Env


IDENTITY_VERSION = "s5i5-native-semantic-d4-v1"


def _opaque_pixels(sprite):
    pixels = sprite.render()
    return {
        (int(sprite.x) + x, int(sprite.y) + y)
        for y in range(pixels.shape[0])
        for x in range(pixels.shape[1])
        if int(pixels[y, x]) != -1
    }


def _cap_pixels(sprite):
    pixels = sprite.render()
    return {
        (int(sprite.x) + x, int(sprite.y) + y)
        for y in range(pixels.shape[0])
        for x in range(pixels.shape[1])
        if int(pixels[y, x]) == names.CAP_COLOR
    }


def _transform(point, swap, sx, sy):
    x, y = point
    if swap:
        x, y = y, x
    return sx * x, sy * y


def _native_payload(env, transform):
    swap, sx, sy = transform
    rods = list(env.rods())
    rod_set = set(rods)
    children = getattr(env.game, names.ATTR_CHILDREN)
    rail_dispatch = getattr(env.game, names.ATTR_RAIL_RODS)
    buttons = list(env.buttons())

    button_dispatch = {}
    for button in buttons:
        pixels = button.render()
        color = int(pixels[pixels.shape[0] // 2, pixels.shape[1] // 2])
        button_dispatch[button] = [
            rod for rod in rods if int(rod.pixels[1, 1]) == color
        ]

    active = set()
    for controlled in rail_dispatch.values():
        active.update(controlled)
    for controlled in button_dispatch.values():
        active.update(controlled)
    for parent, values in children.items():
        rod_children = {child for child in values if child in rod_set}
        pin_children = {child for child in values if names.TAG_PIN in child.tags}
        if rod_children or pin_children:
            active.add(parent)
            active.update(rod_children)

    all_points = set()
    opaque = {}
    caps = {}
    for rod in rods:
        opaque[rod] = {_transform(point, swap, sx, sy)
                       for point in _opaque_pixels(rod)}
        caps[rod] = {_transform(point, swap, sx, sy)
                     for point in _cap_pixels(rod)} if rod in active else set()
        all_points.update(opaque[rod])
    pin_pixels = [
        {_transform(point, swap, sx, sy) for point in _opaque_pixels(pin)}
        for pin in env.pins()
    ]
    target_pixels = [
        {_transform(point, swap, sx, sy) for point in _opaque_pixels(target)}
        for target in env.targets()
    ]
    for points in pin_pixels + target_pixels:
        all_points.update(points)
    if not all_points:
        raise ValueError("empty native S5I5 state")
    left = min(x for x, _ in all_points)
    top = min(y for _, y in all_points)

    def normalized(points):
        return tuple(sorted((x - left, y - top) for x, y in points))

    actor_key = {
        rod: (normalized(opaque[rod]), normalized(caps[rod]))
        for rod in active
    }
    static_collision = set().union(
        *(opaque[rod] for rod in rods if rod not in active)
    )

    adjacency = {rod: set() for rod in active}
    for parent, values in children.items():
        if parent not in active:
            continue
        for child in values:
            if child in active:
                adjacency[parent].add(child)
                adjacency[child].add(parent)
    component = {}
    for rod in active:
        if rod in component:
            continue
        members = set()
        stack = [rod]
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(adjacency[current])
        marker = tuple(sorted(actor_key[item] for item in members))
        component.update({item: marker for item in members})

    def dispatch(controlled):
        chunks = defaultdict(list)
        for rod in controlled:
            if rod in active:
                chunks[component[rod]].append(actor_key[rod])
        # Native iteration order matters when one control contains multiple
        # actors in the same recursive component. Independent components
        # commute and are canonicalized.
        return tuple(sorted(tuple(rows) for rows in chunks.values()))

    rod_edges = []
    pin_edges = []
    for parent, values in children.items():
        if parent not in active:
            continue
        for child in values:
            if child in active:
                rod_edges.append((actor_key[parent], actor_key[child]))
            elif names.TAG_PIN in child.tags:
                pin_edges.append((actor_key[parent], normalized(_transform_set(
                    _opaque_pixels(child), transform
                ))))

    return {
        "version": IDENTITY_VERSION,
        "budget": int(env.max_steps()),
        "actors": sorted(actor_key.values()),
        "static_collision_union": normalized(static_collision),
        "pins": sorted(normalized(points) for points in pin_pixels),
        "targets": sorted(normalized(points) for points in target_pixels),
        "children": sorted(rod_edges),
        "pin_attachments": sorted(pin_edges),
        "rail_dispatch": sorted(dispatch(values) for values in rail_dispatch.values()),
        "button_dispatch": sorted(dispatch(values) for values in button_dispatch.values()),
    }


def _transform_set(points, transform):
    swap, sx, sy = transform
    return {_transform(point, swap, sx, sy) for point in points}


def native_semantic_hash(env):
    """Hash native semantics under translation and mechanically sound D4.

    Reflections reverse rotation chirality, so levels with rotate buttons use
    the four orientation-preserving square symmetries. Rail-only levels use
    all eight D4 transforms.
    """
    encodings = []
    transforms = [
        (swap, sx, sy)
        for swap in (False, True)
        for sx in (-1, 1)
        for sy in (-1, 1)
        if (not env.buttons()
            or ((sx * sy == 1) if not swap else (sx * sy == -1)))
    ]
    for transform in transforms:
        payload = _native_payload(env, transform)
        encodings.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(min(encodings).encode()).hexdigest()


def native_board_geometry_hash(env):
    """Conservative D4 identity for an official board's collision layout.

    This intentionally ignores actor partitioning, cap texture, colours,
    budget, and control rules. It is an official-layout exclusion, not a claim
    that every matching board has an isomorphic transition system.
    """
    typed_points = set()
    typed_points.update(
        ("collision", *point)
        for rod in env.rods()
        for point in _opaque_pixels(rod)
    )
    typed_points.update(
        ("pin", *point)
        for pin in env.pins()
        for point in _opaque_pixels(pin)
    )
    typed_points.update(
        ("target", *point)
        for target in env.targets()
        for point in _opaque_pixels(target)
    )
    if not typed_points:
        raise ValueError("empty native S5I5 board")
    encodings = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                transformed = []
                for kind, x, y in typed_points:
                    tx, ty = _transform((x, y), swap, sx, sy)
                    transformed.append((kind, tx, ty))
                left = min(x for _, x, _ in transformed)
                top = min(y for _, _, y in transformed)
                normalized = sorted(
                    (kind, x - left, y - top) for kind, x, y in transformed
                )
                encodings.append(json.dumps(normalized, separators=(",", ":")))
    return hashlib.sha256(min(encodings).encode()).hexdigest()


@lru_cache(maxsize=1)
def official_semantic_hashes():
    """Ordered hashes of the eight immutable shipped levels."""
    env = Env()
    hashes = []
    for index in range(env.level_count):
        env.set_level(index)
        hashes.append(native_semantic_hash(env))
    return tuple(hashes)


@lru_cache(maxsize=1)
def official_board_geometry_hashes():
    """Ordered conservative geometry hashes of the shipped levels."""
    env = Env()
    hashes = []
    for index in range(env.level_count):
        env.set_level(index)
        hashes.append(native_board_geometry_hash(env))
    return tuple(hashes)


def official_copy_match(env):
    """Return ``(context, evidence kind)`` for a shipped semantic/board copy."""
    semantic = native_semantic_hash(env)
    try:
        return official_semantic_hashes().index(semantic), "native_semantic"
    except ValueError:
        pass
    board = native_board_geometry_hash(env)
    try:
        return official_board_geometry_hashes().index(board), "board_geometry"
    except ValueError:
        return None


def official_copy_index(env):
    """Return the matching shipped context, or ``None`` for a novel puzzle."""
    match = official_copy_match(env)
    return None if match is None else match[0]
