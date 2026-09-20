"""Six CN04 reference contracts measured from the shipped levels.

Each row describes one official level, so ranges below are explicit engineering
tolerances around a scarce reference—not population confidence intervals.  No
official geometry, sprite pixels, or action route is used by generation.
"""

from collections import Counter
from collections.abc import Mapping

import numpy as np

from . import names


DIFFICULTY_VERSION = "cn04-official-six-v2"
QUALITY_PROFILE_VERSION = "cn04-reference-quality-v2"
DIFFICULTIES = tuple(range(1, 7))
NATIVE_PIXEL_VALUES = frozenset(range(names.TRANSPARENT_UNCLICKABLE, 16))
NATIVE_ROTATIONS = frozenset((0, 90, 180, 270))

# Active winning pieces reproduce the shipped per-tier multi-pin constraints.
# Tier six preserves the official repeated three-/four-pin alternatives and
# its all-three-pin second stack, so pin count alone cannot resolve that stack.
WINNING_PIN_DEGREES = {
    1: (2, 2),
    2: (4, 4, 2, 2),
    3: (2, 2, 2),
    4: (2, 2, 2, 2),
    5: (7, 3, 2, 2),
    6: (2, 3, 3, 2, 2),
}
STACK_PIN_SPECTRA = {
    5: ((3, 4, 5, 6, 7),),
    6: ((2, 3, 3, 4, 4, 6), (3, 3, 3, 3)),
}

# Derived once from the 64-action exact teacher witness as documented in the
# family handoff, then independently replayed through the native tier-6 engine.
# Completion occurs on the final action; the route has zero bounce reversals.
OFFICIAL_TIER6_NO_BOUNCE_SOLUTION = (
    (2, None, None), (2, None, None), (2, None, None),
    (4, None, None), (4, None, None), (4, None, None),
    (6, 36, 45),
    (3, None, None), (3, None, None), (3, None, None), (3, None, None),
    (3, None, None), (3, None, None), (3, None, None),
    (6, 48, 6),
    (2, None, None), (2, None, None), (2, None, None),
    (3, None, None), (3, None, None), (3, None, None), (3, None, None),
    (3, None, None), (3, None, None), (2, None, None),
    (3, None, None), (3, None, None), (3, None, None), (3, None, None),
    (2, None, None), (3, None, None), (2, None, None),
    (6, 15, 45),
    (1, None, None), (1, None, None), (1, None, None), (1, None, None),
    (6, 6, 39), (5, None, None), (4, None, None), (1, None, None),
    (6, 27, 21), (6, 15, 33), (1, None, None), (6, 18, 24),
    (2, None, None), (5, None, None), (5, None, None), (5, None, None),
    (6, 21, 30), (5, None, None), (5, None, None), (6, 18, 36),
    (4, None, None),
)

# Exact official measurements from third_party/arc3_games/cn04.py:699-801.
# ``reference_actions`` are bounded exact-state teacher witnesses followed by
# native replay. They are not asserted to be shortest routes.
_ROWS = (
    # groups, sprites, stacks, max, grey, total opaque, visible opaque,
    # total pins, visible pins, witness actions, expanded states.
    (2, 2, (), 75, False, 35, 35, 4, 4, 17, 816),
    (4, 4, (), 100, False, 55, 55, 12, 12, 33, 155),
    (3, 3, (), 125, True, 44, 44, 6, 6, 26, 3_461),
    (4, 4, (), 125, True, 55, 55, 8, 8, 37, 103_910),
    (4, 8, (5,), 150, True, 97, 35, 32, 10, 82, 159_548),
    (5, 13, (6, 4), 200, True, 182, 54, 41, 12, 64, 2_718),
)

_ACTION_RANGES = ((12, 26), (24, 46), (19, 38), (28, 52), (60, 108), (48, 94))
_TOTAL_OPAQUE_RANGES = ((24, 52), (38, 78), (30, 68), (38, 82), (62, 142), (105, 255))
_VISIBLE_OPAQUE_RANGES = ((24, 52), (38, 78), (30, 68), (38, 82), (24, 70), (36, 88))
_TOTAL_PIN_RANGES = ((4, 8), (8, 16), (6, 12), (8, 16), (18, 42), (28, 58))
_VISIBLE_PIN_RANGES = ((4, 8), (8, 16), (6, 12), (8, 16), (8, 16), (10, 20))
_SEARCH_WORK = (180_000, 400_000, 500_000, 900_000, 1_500_000, 1_500_000)

PROFILES = {}
for difficulty, row in enumerate(_ROWS, 1):
    (groups, sprites, stacks, max_steps, grey, total_opaque, visible_opaque,
     total_pins, visible_pins, actions, expanded) = row
    PROFILES[difficulty] = {
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "groups": groups,
        "sprites": sprites,
        "stack_sizes": stacks,
        "max_steps": max_steps,
        "grey_masking": grey,
        "reference_total_opaque": total_opaque,
        "total_opaque": _TOTAL_OPAQUE_RANGES[difficulty - 1],
        "reference_visible_opaque": visible_opaque,
        "visible_opaque": _VISIBLE_OPAQUE_RANGES[difficulty - 1],
        "reference_total_pins": total_pins,
        "total_pins": _TOTAL_PIN_RANGES[difficulty - 1],
        "reference_visible_pins": visible_pins,
        "visible_pins": _VISIBLE_PIN_RANGES[difficulty - 1],
        "reference_actions": actions,
        "reference_expanded_states": expanded,
        "witness_actions": _ACTION_RANGES[difficulty - 1],
        "search_work": _SEARCH_WORK[difficulty - 1],
        "requires_stack_cycle": difficulty >= 5,
        "requires_two_stacks": difficulty == 6,
        # The shared direction reversal is a real stack interaction, but it is
        # not required to solve the shipped tier-six state: a native-replayed
        # 54-action route reaches the winning alternates without reversing.
        # Exercise bounce/recovery separately instead of padding win routes.
        "requires_bounce_reversal": False,
        "pin_colours": (names.PIN_A,) if difficulty == 4 else names.PIN_COLORS,
        "winning_pin_degrees": WINNING_PIN_DEGREES[difficulty],
        "stack_pin_spectra": STACK_PIN_SPECTRA.get(difficulty, ()),
        "winning_parallel_counts": (3, 2, 2) if difficulty == 5 else None,
    }
    if difficulty == 6:
        PROFILES[difficulty].update({
            "reference_no_bounce_actions": 54,
            "reference_no_bounce_reversals": 0,
            "reference_bounce_required": False,
        })


def _piece_schema_errors(spec):
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    errors = []
    pieces = spec.get("pieces")
    if not isinstance(pieces, (list, tuple)) or not pieces:
        errors.append("pieces must be a nonempty sequence")
        return errors
    for piece_index, piece in enumerate(pieces):
        prefix = f"piece {piece_index}"
        if not isinstance(piece, Mapping):
            errors.append(f"{prefix} must be a mapping")
            continue
        for key in ("group", "alternate", "x", "y", "rotation", "layer"):
            if key not in piece or type(piece[key]) is not int:
                errors.append(f"{prefix} {key} must be an integer")
        if (type(piece.get("rotation")) is int
                and piece["rotation"] not in NATIVE_ROTATIONS):
            errors.append(f"{prefix} rotation must be a canonical quarter turn")
        if "visible" not in piece or type(piece["visible"]) is not bool:
            errors.append(f"{prefix} visible must be a boolean")
        if "name" not in piece or type(piece["name"]) is not str or not piece["name"]:
            errors.append(f"{prefix} name must be a nonempty string")

        pixels = piece.get("pixels")
        if not isinstance(pixels, (list, tuple)) or not pixels:
            errors.append(f"{prefix} pixels must be a nonempty rectangular array")
            continue
        width = None
        rectangular = True
        for row in pixels:
            if not isinstance(row, (list, tuple)) or not row:
                rectangular = False
                continue
            if width is None:
                width = len(row)
            elif len(row) != width:
                rectangular = False
            if any(type(value) is not int or value not in NATIVE_PIXEL_VALUES
                   for value in row):
                errors.append(
                    f"{prefix} pixels must be integers in the native -2..15 domain"
                )
                break
        if not rectangular:
            errors.append(f"{prefix} pixels must be a nonempty rectangular array")
    return errors


def primitive_schema_errors(spec):
    """Validate JSON primitives before NumPy or native-engine coercion."""
    errors = _piece_schema_errors(spec)
    if not isinstance(spec, Mapping):
        return errors
    grid_size = spec.get("grid_size")
    if (
        not isinstance(grid_size, (list, tuple))
        or len(grid_size) != 2
        or any(type(value) is not int or value <= 0 for value in grid_size)
    ):
        errors.append("grid_size must contain two positive integers")
    if type(spec.get("background")) is not int:
        errors.append("background must be an integer")
    if type(spec.get("max_steps")) is not int:
        errors.append("max_steps must be an integer")
    if type(spec.get("grey_masking")) is not bool:
        errors.append("grey_masking must be a boolean")
    return errors


def native_semantic_groups(spec):
    """Return native coordinate groups with stable layer/order alternatives.

    CN04 ignores authored group/alternate numbers. It groups sprites by initial
    coordinate in first-occurrence order and stable-sorts each group by layer.
    Labels are accepted only as a consistent bijective annotation; renaming
    them cannot alter identities, while contradictory annotations fail closed.
    """
    schema_errors = _piece_schema_errors(spec)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))
    pieces = spec["pieces"]
    by_coordinate = {}
    label_coordinate = {}
    names_seen = set()
    for order, piece in enumerate(pieces):
        name = piece["name"]
        if name in names_seen:
            raise ValueError("piece names must be distinct nonempty strings")
        names_seen.add(name)
        coordinate = (piece["x"], piece["y"])
        label = piece["group"]
        prior = label_coordinate.setdefault(label, coordinate)
        if prior != coordinate:
            raise ValueError("one group annotation spans multiple native coordinates")
        by_coordinate.setdefault(coordinate, []).append((order, piece))

    groups = []
    coordinate_labels = {}
    for coordinate, entries in by_coordinate.items():
        labels = {piece["group"] for _, piece in entries}
        if len(labels) != 1:
            raise ValueError("one native coordinate has contradictory group annotations")
        label = next(iter(labels))
        if label in coordinate_labels and coordinate_labels[label] != coordinate:
            raise ValueError("group annotation is not bijective with native grouping")
        coordinate_labels[label] = coordinate
        alternates = [piece["alternate"] for _, piece in entries]
        if len(set(alternates)) != len(alternates):
            raise ValueError("alternate annotations must be distinct within a native group")
        ordered = tuple(sorted(entries, key=lambda item: (item[1]["layer"], item[0])))
        if sum(piece["visible"] for _, piece in ordered) != 1:
            raise ValueError("every native group must have exactly one visible alternate")
        groups.append(ordered)
    return tuple(groups)


def structural_metrics(spec):
    pieces = list(spec.get("pieces", ()))
    grouped = native_semantic_groups(spec)
    stack_sizes = tuple(sorted((len(items) for items in grouped if len(items) > 1),
                               reverse=True))
    total_opaque = total_pins = visible_opaque = visible_pins = zero_markers = 0
    visible_colours = set()
    for piece in pieces:
        pixels = np.asarray(piece.get("pixels", ()), dtype=int)
        opaque = int((pixels >= 0).sum())
        pins = int(((pixels == names.PIN_A) | (pixels == names.PIN_B)).sum())
        total_opaque += opaque
        total_pins += pins
        zero_markers += int((pixels == names.CYCLE_PIXEL).sum())
        if piece.get("visible", True):
            visible_opaque += opaque
            visible_pins += pins
            visible_colours.update(int(value) for value in pixels.flat
                                   if value in names.PIN_COLORS)
    targets = list(spec.get("constructed_target", ()))
    if targets:
        xs = [int(item[0]) for item in targets]
        ys = [int(item[1]) for item in targets]
        target_span = [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1]
    else:
        target_span = [0, 0]
    return {
        "groups": len(grouped),
        "sprites": len(pieces),
        "stack_sizes": stack_sizes,
        "total_opaque": total_opaque,
        "visible_opaque": visible_opaque,
        "total_pins": total_pins,
        "visible_pins": visible_pins,
        "visible_pin_colours": tuple(sorted(visible_colours)),
        "zero_markers": zero_markers,
        "target_span": target_span,
        "visual_density": visible_opaque / 400.0,
    }


def profile_errors(spec, *, require_proof=True):
    errors = primitive_schema_errors(spec)
    if errors:
        return errors
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty must be an integer in 1..6"]
    profile = PROFILES[difficulty]
    if spec.get("difficulty_version") != DIFFICULTY_VERSION:
        errors.append("missing calibrated difficulty version")
    if spec.get("quality_profile_version") != QUALITY_PROFILE_VERSION:
        errors.append("missing quality profile version")
    try:
        metrics = structural_metrics(spec)
        for key in ("groups", "sprites", "stack_sizes"):
            if metrics[key] != profile[key]:
                errors.append(f"{key} differs from official tier composition")
        for key in ("total_opaque", "visible_opaque", "total_pins", "visible_pins"):
            low, high = profile[key]
            if not low <= metrics[key] <= high:
                errors.append(f"{key} outside reference tolerance")
        if tuple(spec.get("grid_size", ())) != names.GRID_SIZE:
            errors.append("grid geometry differs from official 20x20 board")
        if spec.get("max_steps") != profile["max_steps"]:
            errors.append("native action budget differs from official tier")
        if spec.get("grey_masking", False) is not profile["grey_masking"]:
            errors.append("grey masking differs from official tier")
        if difficulty >= 5 and metrics["zero_markers"] < sum(profile["stack_sizes"]):
            errors.append("stack alternates lack click-cycle zero markers")
        groups = native_semantic_groups(spec)
        stack_spectra = tuple(
            tuple(sorted(
                int(np.isin(np.asarray(piece["pixels"], dtype=int), names.PIN_COLORS).sum())
                for _, piece in group
            ))
            for group in groups if len(group) > 1
        )
        if stack_spectra != profile["stack_pin_spectra"]:
            errors.append("stack alternate pin spectra differ from the calibrated tier grammar")
        if difficulty == 4:
            colours = {value for piece in spec["pieces"] for row in piece["pixels"]
                       for value in row if value in names.PIN_COLORS}
            if colours != {names.PIN_A}:
                errors.append("tier 4 must preserve the reference single-pin-colour composition")
        if require_proof:
            length = spec.get("solution_length", -1)
            low, high = profile["witness_actions"]
            if not low <= length <= high:
                errors.append("witness action length outside reference tolerance")
            mechanics = spec.get("solution_mechanics", {})
            if not mechanics.get("won"):
                errors.append("mechanic trace does not end in completion")
            if profile["requires_stack_cycle"] and not (
                    mechanics.get("stack_cycles_action", 0)
                    + mechanics.get("stack_cycles_click", 0)):
                errors.append("stack mechanic is not exercised by the witness")
            if profile["requires_two_stacks"] and mechanics.get("distinct_stacks_cycled", 0) < 2:
                errors.append("both generated stack roles must be exercised")
            constraints = spec.get("solution_constraints", {})
            if tuple(constraints.get("winning_pin_counts", ())) != profile["winning_pin_degrees"]:
                errors.append("winning route omits the tier's calibrated multi-pin constraints")
            if constraints.get("connected") is not True:
                errors.append("winning relation topology must be connected")
            if constraints.get("relation_edges") != sum(profile["winning_pin_degrees"]) // 2:
                errors.append("winning relation edge count differs from the tier grammar")
            if (profile["winning_parallel_counts"] is not None
                    and tuple(constraints.get("parallel_relation_counts", ()))
                    != profile["winning_parallel_counts"]):
                errors.append("winning parallel-relation multiplicities differ from the tier grammar")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"malformed CN04 spec: {error}")
    return errors


def official_characterization():
    """JSON-ready aggregate measurements; contains no layouts or routes."""
    result = []
    for difficulty in DIFFICULTIES:
        profile = PROFILES[difficulty]
        result.append({key: value for key, value in profile.items()
                       if key.startswith("reference_") or key in {
                           "difficulty", "context_index", "groups", "sprites",
                           "stack_sizes", "max_steps", "grey_masking", "pin_colours",
                           "winning_pin_degrees", "stack_pin_spectra",
                           "winning_parallel_counts",
                       }})
    return result
