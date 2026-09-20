"""Canonical identities and split admission for generated WA30 levels."""

import hashlib
import json

SPLITS = ("train", "validation", "test")
GEOMETRY_VERSION = "wa30-d4-native-sets-v2"
GAMEPLAY_VERSION = "wa30-native-semantic-state-v2"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cell_set(cells):
    return sorted({tuple(cell) for cell in cells})


def _region_union(regions):
    cells = set()
    for col, row, width, height in regions:
        cells.update(
            (col + dx, row + dy)
            for dx in range(width)
            for dy in range(height)
        )
    return sorted(cells)


def semantic_payload(spec):
    """Native initial state, excluding art, rectangle partition and provenance."""
    return {
        "budget": spec["budget"],
        "walls": _cell_set(spec.get("walls", ())),
        "fences": _cell_set(spec.get("fences", ())),
        "goals": _region_union(spec.get("goals", ())),
        "bad": _region_union(spec.get("bad_regions", ())),
        # Native iteration and early returns make these orders observable.
        "boxes": [tuple(cell) for cell in spec.get("boxes", ())],
        "helpers": [tuple(cell) for cell in spec.get("helpers", ())],
        "thieves": [tuple(cell) for cell in spec.get("thieves", ())],
        "player": tuple(spec["player"]),
        # build_level clones the fixed upstream player without consuming a
        # rotation field; every shipped/generated initial player is rotation 0.
        "player_rotation": 0,
    }


def _points(spec):
    value = semantic_payload(spec)
    labelled = []
    for label, key in (("wall", "walls"), ("fence", "fences"),
                       ("goal", "goals"), ("bad", "bad")):
        labelled.extend((label, x, y) for x, y in value[key])
    # Geometry partitioning is intentionally assignment-agnostic. Native actor
    # order remains binding in ``gameplay_hash`` below.
    for label, key in (("box", "boxes"), ("helper", "helpers"), ("thief", "thieves")):
        labelled.extend((label, x, y) for x, y in value[key])
    player_label = (
        "player" if value["player_rotation"] == 0
        else f"player:r{value['player_rotation']}"
    )
    labelled.append((player_label, *value["player"]))
    return labelled


def geometry_d4_hash(spec):
    """Canonical labelled geometry under translation and all D4 transforms."""
    points = _points(spec)
    if not points:
        raise ValueError("empty WA30 geometry")
    variants = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                transformed = [
                    (label, sx * (y if swap else x), sy * (x if swap else y))
                    for label, x, y in points
                ]
                left = min(x for _, x, _ in transformed)
                top = min(y for _, _, y in transformed)
                variants.append(sorted((label, x - left, y - top) for label, x, y in transformed))
    return _digest(min(variants))


def geometry_hash(spec):
    """Exact native geometry, retaining absolute coordinates and actor roles."""
    return _digest(sorted(_points(spec)))


def geometry_partition(spec):
    fingerprint = geometry_d4_hash(spec)
    return fingerprint, SPLITS[int(fingerprint, 16) % len(SPLITS)]


def gameplay_hash(spec):
    """Native-semantic identity excluding route, art and provenance labels."""
    return _digest({"version": GAMEPLAY_VERSION, **semantic_payload(spec)})


def action_hash(solution):
    actions = []
    if type(solution) is not list or not solution:
        raise ValueError("solution must be a nonempty JSON list")
    for step in solution:
        if (type(step) is not list or len(step) != 3 or type(step[0]) is not int
                or step[0] not in (1, 2, 3, 4, 5)
                or step[1] is not None or step[2] is not None):
            raise ValueError("WA30 actions must be exact [action_id, null, null] triples")
        actions.append(step[0])
    return _digest(actions)
