"""Canonical identities and split assignment for generated S5I5 levels."""

from __future__ import annotations

import hashlib
import json


SPLITS = ("train", "validation", "test")
GEOMETRY_VERSION = "s5i5-pixel-union-d4-v3"
_UNIT = 3


def _rod_cells(rod):
    length = int(rod["length"])
    rotation = int(rod["rotation"])
    x, y = int(rod["x"]), int(rod["y"])
    if rotation == 0:
        base = (x, y + 3 * (length - 1))
        delta = (0, -3)
    elif rotation == 90:
        base = (x, y)
        delta = (3, 0)
    elif rotation == 180:
        base = (x, y)
        delta = (0, 3)
    elif rotation == 270:
        base = (x + 3 * (length - 1), y)
        delta = (-3, 0)
    else:
        raise ValueError("rod rotation must be 0, 90, 180 or 270")
    return [(base[0] + delta[0] * index, base[1] + delta[1] * index)
            for index in range(length)]


def _filled_pixels(cells):
    return {
        (int(x) + dx, int(y) + dy)
        for x, y in cells
        for dx in range(_UNIT)
        for dy in range(_UNIT)
    }


def obstacle_collision_pixels(spec):
    """Exact opaque collision-pixel union rendered by all obstacle sprites."""
    return _filled_pixels(
        cell
        for obstacle in spec["obstacles"]
        for cell in obstacle["cells"]
    )


def _semantic_points(spec):
    """Exact pixel geometry with cosmetic controls and colour IDs removed.

    Points are a set because duplicate/overlapping obstacle cells render the
    same collision mask.  Strict admission rejects those noncanonical source
    encodings, while identity remains bound to the native pixels rather than
    to an arbitrary list representation.
    """
    points = {
        ("obstacle", x, y) for x, y in obstacle_collision_pixels(spec)
    }
    rod_pixels = _filled_pixels(
        cell for rod in spec["rods"] for cell in _rod_cells(rod)
    )
    points.update(("rod", x, y) for x, y in rod_pixels)
    points.update(("pin", int(pin["x"]) + 1, int(pin["y"]) + 1)
                  for pin in spec["pins"])
    target_offsets = ((1, 0), (0, 1), (2, 1), (1, 2))
    points.update(
        ("target", int(target["x"]) + dx, int(target["y"]) + dy)
        for target in spec["targets"]
        for dx, dy in target_offsets
    )
    if not points:
        raise ValueError("empty semantic geometry")
    return sorted(points)


def _normalized(points):
    left = min(x for _, x, _ in points)
    top = min(y for _, _, y in points)
    return sorted((kind, x - left, y - top) for kind, x, y in points)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def geometry_hash(spec):
    """Translation-normalized typed geometry identity."""
    return _digest(_normalized(_semantic_points(spec)))


def geometry_d4_hash(spec):
    """Typed geometry identity normalized under translation and all D4 maps."""
    points = _semantic_points(spec)
    variants = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                transformed = []
                for kind, x, y in points:
                    tx, ty = (y, x) if swap else (x, y)
                    transformed.append((kind, sx * tx, sy * ty))
                variants.append(_normalized(transformed))
    return _digest(min(variants))


def identity_partition(spec):
    fingerprint = geometry_d4_hash(spec)
    return fingerprint, SPLITS[int(fingerprint, 16) % len(SPLITS)]


def _translation_origin(spec):
    points = _semantic_points(spec)
    return min(x for _, x, _ in points), min(y for _, _, y in points)


def _rod_key(rod, origin):
    """Stable rod identity, normalized with the whole puzzle translation."""
    ox, oy = origin
    pixels = _filled_pixels(_rod_cells(rod))
    normalized = tuple(sorted((x - ox, y - oy) for x, y in pixels))
    return normalized, int(rod["rotation"]), int(rod["length"])


def _mechanism(spec):
    origin = _translation_origin(spec)
    by_name = {rod["name"]: _rod_key(rod, origin) for rod in spec["rods"]}
    by_color = {}
    for rod in spec["rods"]:
        by_color.setdefault(int(rod["color"]), []).append(rod["name"])
    controlled = {int(control["color"]) for control in spec["rails"] + spec["buttons"]}
    controlled.update(int(control["secondary_color"]) for control in spec["rails"]
                      if "secondary_color" in control)
    relations = spec["native_relations"]
    adjacency = {name: set() for name in by_name}
    for parent, child in relations["rod_edges"]:
        adjacency[parent].add(child)
        adjacency[child].add(parent)
    component = {}
    for name in adjacency:
        if name in component:
            continue
        members = set()
        stack = [name]
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(adjacency[current])
        marker = min(by_name[item] for item in members)
        component.update({item: marker for item in members})

    def dispatch(colors):
        selected = [rod["name"] for rod in spec["rods"]
                    if int(rod["color"]) in set(colors)]
        chunks = {}
        for name in selected:
            chunks.setdefault(component[name], []).append(by_name[name])
        # Native processes rods in level order. Order is retained inside one
        # recursive component, where ancestor/descendant double-dispatch is
        # observable; independent commuting components are canonicalized.
        return tuple(sorted(tuple(rows) for rows in chunks.values()))

    groups = {color: dispatch((color,)) for color in controlled}
    return by_name, groups, relations, dispatch, origin


def gameplay_hash(spec):
    """Canonical gameplay identity with arbitrary numeric colour labels removed."""
    by_name, color_groups, relations, dispatch, origin = _mechanism(spec)
    groups = sorted(color_groups.values())
    payload = {
        "geometry_d4": geometry_d4_hash(spec),
        "difficulty": int(spec["difficulty"]),
        "budget": int(spec["step_counter"]),
        "rod_lengths": sorted(int(rod["length"]) for rod in spec["rods"]),
        "control_groups": groups,
        "children": sorted((by_name[parent], by_name[child])
                           for parent, child in relations["rod_edges"]),
        "pin_attachments": sorted((by_name[parent],
                                   (int(position[0]) - origin[0],
                                    int(position[1]) - origin[1]))
                                  for parent, position in relations["pin_edges"]),
        "rail_dispatch": sorted(
            (control["orientation"], control.get("style", "compact"),
             dispatch((int(control["color"]),) +
                      ((int(control["secondary_color"]),)
                       if "secondary_color" in control else ())))
            for control in spec["rails"]
        ),
        "button_dispatch": sorted(
            (control.get("style", "compact"), dispatch((int(control["color"]),)))
            for control in spec["buttons"]
        ),
    }
    return _digest(payload)


def solution_semantic_hash(spec):
    """Colour- and placement-independent identity of the certified route.

    Each click is reduced to its mechanic and the canonical set of rod indices
    it controls. This distinguishes genuinely different solution programs from
    cosmetic recolouring or control-panel relocation.
    """
    _, _, _, dispatch, _ = _mechanism(spec)
    controls = []
    for rail in spec["rails"]:
        x, y = int(rail["x"]), int(rail["y"])
        long, short = (13, 7) if rail.get("style") == "large" else (11, 5)
        if rail["orientation"] == "horizontal":
            bounds = (x, y, x + long, y + short)
            classifier = lambda cx, cy, x=x, half=long // 2: (
                "extend" if cx > x + half else "retract"
            )
        else:
            bounds = (x, y, x + short, y + long)
            classifier = lambda cx, cy, y=y, half=long // 2: (
                "extend" if cy > y + half else "retract"
            )
        colors = (int(rail["color"]),)
        if "secondary_color" in rail:
            colors += (int(rail["secondary_color"]),)
        controls.append((bounds, classifier, dispatch(colors)))
    for button in spec["buttons"]:
        x, y = int(button["x"]), int(button["y"])
        size = 7 if button.get("style") == "large" else 5
        controls.append(((x, y, x + size, y + size), lambda cx, cy: "rotate",
                         dispatch((int(button["color"]),))))

    program = []
    for action_id, x, y in spec["solution"]:
        matches = [row for row in controls
                   if row[0][0] <= x < row[0][2] and row[0][1] <= y < row[0][3]]
        if int(action_id) != 6 or len(matches) != 1:
            raise ValueError("solution action does not identify one control")
        bounds, classifier, controlled_rods = matches[0]
        program.append((classifier(int(x), int(y)), controlled_rods))
    return _digest(program)
