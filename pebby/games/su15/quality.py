"""Reference profiles and recomputed identities for full SU15 generation.

Each profile describes one shipped level.  The ranges below are deliberately
wide engineering tolerances around a single scarce reference, not population
confidence intervals.  Official layouts and routes are never generator input.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json


PROFILE_VERSION = "su15-nine-reference-v2"
MECHANICS_VERSION = "su15-full-mechanics-v3"
GEOMETRY_VERSION = "su15-width-aware-hreflection-v2"
GAMEPLAY_VERSION = "su15-native-object-order-v2"


def _requirements(*values):
    return tuple(values)


# fruit tier counts, enemy class counts, zones, requirements, native steps,
# measured initial non-background pixels, and admitted constructive actions.
_ROWS = (
    ({2: 1}, {}, 1, _requirements(("fruit", 2, 1)), 32, 146, 10, (6, 18)),
    ({0: 8}, {}, 1, _requirements(("fruit", 3, 1)), 32, 173, 14, (10, 24)),
    ({0: 6, 1: 3}, {}, 2, _requirements(("fruit", 3, 1), ("fruit", 2, 1)), 48, 261, 21, (12, 32)),
    ({0: 8}, {1: 1}, 1, _requirements(("fruit", 3, 1)), 48, 181, 24, (14, 34)),
    ({0: 4, 1: 4}, {1: 2}, 1, _requirements(("fruit", 3, 1)), 32, 201, 17, (10, 26)),
    ({5: 1}, {1: 1}, 2, _requirements(("fruit", 3, 1), ("enemy", 1, 1)), 32, 307, 12, (5, 22)),
    ({1: 4, 5: 1}, {1: 2}, 2, _requirements(("fruit", 3, 2)), 32, 339, 8, (5, 24)),
    ({3: 2, 5: 1}, {1: 3}, 4, _requirements(("fruit", 4, 2), ("enemy", 2, 1)), 48, 539, 10, (5, 24)),
    ({1: 2, 5: 1}, {1: 4}, 3, _requirements(("fruit", 4, 1), ("enemy", 3, 1), ("fruit", 2, 1)), 48, 443, 12, (5, 26)),
)

# Top-left bounding boxes of all live fruits, pursuers, and targets in each
# shipped initial state.  Tolerances are intentionally broad around one board;
# they reject collapsed compositions without pretending to estimate a
# population distribution.
_OFFICIAL_BBOX = (
    (42, 48), (36, 35), (57, 32), (52, 35), (56, 50),
    (51, 42), (47, 44), (50, 39), (48, 41),
)

REFERENCE_PROFILES = {
    difficulty: {
        "reference_level": difficulty,
        "context_index": difficulty - 1,
        "fruit_counts": dict(fruits),
        "enemy_counts": dict(enemies),
        "target_count": targets,
        "requirements": requirements,
        "steps": steps,
        # Presentation tolerance around the one measured frame.  Generated
        # legends may legitimately make a sparse tutorial denser.
        "non_background_pixels": (max(80, pixels - 100), min(800, pixels + 180)),
        "actions": actions,
        "reference_witness_actions": witness,
        "search_work": 50_000,
    }
    for difficulty, (fruits, enemies, targets, requirements, steps, pixels, witness, actions)
    in enumerate(_ROWS, 1)
}
for _difficulty, (_width, _height) in enumerate(_OFFICIAL_BBOX, 1):
    REFERENCE_PROFILES[_difficulty].update(
        bbox_reference={"width": _width, "height": _height},
        bbox_width=(max(18, _width - 20), min(64, _width + 20)),
        bbox_height=(max(12, _height - 18), min(53, _height + 18)),
    )


def normalized_requirements(spec):
    result = []
    for value in spec.get("requirements", ()):
        result.append((str(value["kind"]), int(value["tier"]), int(value["count"])))
    return tuple(result)


def structural_metrics(spec):
    fruits = Counter(int(value["tier"]) for value in spec.get("fruits", ()))
    enemies = Counter(int(value["kind"]) for value in spec.get("enemies", ()))
    points = [tuple(map(int, value["position"])) for field in ("fruits", "enemies", "targets")
              for value in spec.get(field, ())]
    if not points:
        raise ValueError("level has no positioned gameplay objects")
    xs, ys = zip(*points)
    return {
        "fruit_counts": dict(sorted(fruits.items())),
        "enemy_counts": dict(sorted(enemies.items())),
        "target_count": len(spec.get("targets", ())),
        "requirements": normalized_requirements(spec),
        "bbox_width": max(xs) - min(xs) + 1,
        "bbox_height": max(ys) - min(ys) + 1,
    }


def _width(label, tier):
    if label == "fruit":
        return int(tier) + 1
    if label == "enemy":
        return 5
    return 9


def _canonical_payload(spec, *, reflect, preserve_native_order=False):
    values = []
    for field, label, type_field in (
        ("fruits", "fruit", "tier"), ("enemies", "enemy", "kind"),
        ("targets", "target", None),
    ):
        family = []
        for value in spec.get(field, ()):
            x, y = map(int, value["position"])
            tier = None if type_field is None else int(value[type_field])
            if reflect:
                x = 64 - _width(label, tier) - x
            family.append((label, tier, x, y))
        values.append(family)
    flattened = [value for family in values for value in family]
    left = min(value[2] for value in flattened)
    top = min(value[3] for value in flattened)
    normalized_families = [
        [(kind, tier, x - left, y - top) for kind, tier, x, y in family]
        for family in values
    ]
    if preserve_native_order:
        # Fruit and pursuer list order is executable state: pursuit tie breaks
        # and knockback collision order use it.  Target order is a union-only
        # completion detail and is normalized as an unordered family.
        objects = {
            "fruits": normalized_families[0],
            "enemies": normalized_families[1],
            "targets": sorted(normalized_families[2]),
        }
        requirements = normalized_requirements(spec)
    else:
        objects = sorted(value for family in normalized_families for value in family)
        requirements = sorted(normalized_requirements(spec))
    return {
        "objects": objects,
        "requirements": requirements,
        "steps": int(spec["steps"]),
    }


def geometry_identity(spec):
    payload = min(
        (_canonical_payload(spec, reflect=False), _canonical_payload(spec, reflect=True)),
        key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def raw_geometry_identity(spec):
    payload = {
        field: sorted((int(value.get("tier", value.get("kind", -1))),
                       *map(int, value["position"])) for value in spec.get(field, ()))
        for field in ("fruits", "enemies", "targets")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gameplay_identity(spec):
    payload = min(
        (_canonical_payload(spec, reflect=False, preserve_native_order=True),
         _canonical_payload(spec, reflect=True, preserve_native_order=True)),
        key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def geometry_partition(spec):
    identity = geometry_identity(spec)
    split = ("train", "validation", "test")[int(identity, 16) % 3]
    return identity, split


def profile_errors(spec, *, require_proof=True):
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in REFERENCE_PROFILES:
        return ["difficulty must be an integer in 1..9"]
    profile = REFERENCE_PROFILES[difficulty]
    try:
        metrics = structural_metrics(spec)
        for field in ("fruit_counts", "enemy_counts", "target_count", "requirements"):
            if metrics[field] != profile[field]:
                errors.append(f"{field} differs from shipped tier {difficulty}")
        if int(spec.get("steps", -1)) != profile["steps"]:
            errors.append("native step budget differs from the shipped tier")
        if spec.get("quality_profile_version") != PROFILE_VERSION:
            errors.append("quality profile version mismatch")
        if spec.get("mechanics_version") != MECHANICS_VERSION:
            errors.append("mechanics version mismatch")
        if not profile["bbox_width"][0] <= metrics["bbox_width"] <= profile["bbox_width"][1]:
            errors.append("playfield width is outside the reference-calibrated tolerance")
        if not profile["bbox_height"][0] <= metrics["bbox_height"] <= profile["bbox_height"][1]:
            errors.append("playfield height is outside the reference-calibrated tolerance")
        if require_proof:
            pixels = spec.get("initial_non_background_pixels")
            low_pixels, high_pixels = profile["non_background_pixels"]
            if type(pixels) is not int or not low_pixels <= pixels <= high_pixels:
                errors.append("initial visual density is outside the reference-calibrated tolerance")
            length = spec.get("solution_length")
            if type(length) is not int or not profile["actions"][0] <= length <= profile["actions"][1]:
                errors.append("constructive witness length is outside the calibrated tolerance")
            mechanics = spec.get("solution_mechanics")
            if not isinstance(mechanics, dict) or not mechanics.get("won"):
                errors.append("missing recomputed winning mechanic trace")
            if difficulty in (2, 3, 4, 5, 7, 8, 9) and mechanics.get("fruit_merges", 0) < 1:
                errors.append("winning route does not exercise ordinary-fruit merging")
            if difficulty in (6, 7, 8, 9) and mechanics.get("fruit_degrades", 0) < 1:
                errors.append("winning route does not exercise pursuer degradation")
            if difficulty in (8, 9) and mechanics.get("enemy_merges", 0) < 1:
                errors.append("winning route does not exercise pursuer-class merging")
            if difficulty >= 4 and mechanics.get("pursuer_motion_actions", 0) < 1:
                errors.append("winning route does not exercise pursuer motion")
            # Native completion is over the union of zones and does not
            # require one object per installed target.  Calibrate only the
            # actually observed multi-zone use, never the installed count.
            expected_zones = 1 if profile["target_count"] == 1 else 2
            if mechanics.get("installed_target_zones") != profile["target_count"]:
                errors.append("stored installed target-zone count is incorrect")
            if mechanics.get("witnessed_target_zones", 0) < expected_zones:
                errors.append("winning route does not witness the calibrated multi-zone composition")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"malformed profile data: {exc}")
    return errors
