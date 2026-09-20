"""Reference-calibrated quality and identity rules for TN36.

There is one shipped example per tier. Ranges below are explicit engineering
tolerances around those scarce references, not population confidence bounds.
"""

from collections import Counter
import hashlib
import json


PROFILE_VERSION = "tn36-seven-reference-v3-public-route-events"
MECHANICS_VERSION = "tn36-full-engine-v2-run-start"
GEOMETRY_VERSION = "tn36-panel-semantic-v2"
GAMEPLAY_VERSION = "tn36-executable-semantics-v2"
DIFFICULTIES = tuple(range(1, 8))
SPLITS = ("train", "validation", "test")

# Native measurements: official slots, bit width, selectors, rollback walls,
# checkpoint platforms, gates, timer budget and replayed teacher action count.
_REFERENCE = (
    (5, 2, 0, 0, 0, 0, 61, 7),
    (4, 6, 2, 0, 0, 0, 61, 9),
    (6, 6, 4, 2, 0, 0, 61, 9),
    (6, 6, 4, 2, 0, 0, 61, 12),
    (6, 6, 5, 2, 0, 0, 61, 16),
    (6, 6, 4, 4, 3, 0, 122, 18),
    (6, 6, 4, 5, 4, 2, 122, 16),
)
_ACTION_RANGES = ((4, 11), (6, 14), (6, 16), (8, 19), (10, 24), (12, 30), (12, 32))
_REFERENCE_DENSITY = (0.263916, 0.665283, 0.690674, 0.705322, 0.717285, 0.718994, 0.708740)
_DENSITY_RANGES = (
    (0.22, 0.58),  # the one-off panned tutorial composition has a broad band
    (0.62, 0.71), (0.65, 0.74), (0.66, 0.75), (0.66, 0.76),
    (0.66, 0.76), (0.66, 0.76),
)
REFERENCE_PROFILES = {
    tier: {
        "reference_level": tier,
        "context_index": tier - 1,
        "slots": row[0],
        "bit_width": row[1],
        "selectors": row[2],
        "walls": row[3],
        "platforms": row[4],
        "gates": row[5],
        "native_budget": row[6],
        "reference_teacher_actions": row[7],
        "actions": _ACTION_RANGES[tier - 1],
        "reference_visual_density": _REFERENCE_DENSITY[tier - 1],
        "visual_density": _DENSITY_RANGES[tier - 1],
        "search_work": 400_000 if tier < 6 else 1_200_000,
    }
    for tier, row in enumerate(_REFERENCE, 1)
}

REQUIRED_MECHANIC_EVENTS = {
    1: ("translation",),
    2: ("translation", "preset_selection"),
    3: ("translation", "preset_selection", "collision_rollback"),
    4: ("translation", "preset_selection", "collision_rollback", "scale"),
    5: ("translation", "preset_selection", "scale", "rotation", "recolor"),
    6: ("translation", "preset_selection", "collision_rollback", "platform_checkpoint"),
    7: ("translation", "preset_selection", "platform_checkpoint", "gate_toggle"),
}


def canonical_geometry(spec):
    """Translation-normalized native collision geometry.

    Signed opcodes make reflections unsafe.  Gate bodies are active collision
    rectangles whenever their barriers are hidden, so both rectangles are
    represented.  Sprite names and control placement are presentation only.
    """
    items = []
    for kind in ("walls", "platforms"):
        for item in spec.get(kind, ()):
            items.append((kind, item["x"], item["y"], item["width"], item["height"]))
    for gate in spec.get("gates", ()):
        items.append(("gate_barrier", gate["x"], gate["y"],
                      gate["width"], gate["height"], gate["visible"]))
        items.append(("gate_body", gate["body_x"], gate["body_y"],
                      gate["body_width"], gate["body_height"],
                      not gate["visible"]))
    anchors = [tuple(spec["actor"][:2]), tuple(spec["target"][:2])]
    if items:
        left = min([value[1] for value in items] + [x for x, _ in anchors])
        top = min([value[2] for value in items] + [y for _, y in anchors])
    else:
        left = min(x for x, _ in anchors)
        top = min(y for _, y in anchors)
    normalized = sorted(
        (kind, x - left, y - top, w, h, *rest)
        for kind, x, y, w, h, *rest in items
    )
    actor = tuple(spec["actor"][:2])
    target = tuple(spec["target"][:2])
    points = ((actor[0] - left, actor[1] - top), (target[0] - left, target[1] - top))
    return normalized, points


def _normalized_transforms(spec):
    geometry, points = canonical_geometry(spec)
    actor = list(spec["actor"])
    target = list(spec["target"])
    actor[:2], target[:2] = list(points[0]), list(points[1])
    if spec["bit_width"] <= 2:
        # Codes 0..3 cannot recolor. Only equality of the two colors affects
        # the native goal, so a joint palette relabel is semantic no-op.
        palette = {}
        actor[4] = palette.setdefault(actor[4], len(palette))
        target[4] = palette.setdefault(target[4], len(palette))
    return geometry, actor, target


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identities(spec):
    raw = {
        "actor": spec["actor"], "target": spec["target"],
        "walls": spec.get("walls", []), "platforms": spec.get("platforms", []),
        "gates": spec.get("gates", []),
    }
    geometry = _sha(raw)
    canonical = _sha(canonical_geometry(spec))
    geometry_state, actor, target = _normalized_transforms(spec)
    semantics = {
        "version": GAMEPLAY_VERSION,
        "actor": actor, "target": target,
        "initial_program": spec["initial_program"],
        "preset_programs": spec.get("preset_programs", []),
        "preset_positions": spec.get("preset_positions", []),
        "preset_rotations": spec.get("preset_rotations", []),
        "preset_scales": spec.get("preset_scales", []),
        "preset_resets": spec.get("preset_resets", []),
        "geometry": geometry_state,
    }
    gameplay = _sha({
        **semantics,
        "difficulty": spec["difficulty"],
        "context_index": spec["context_index"],
    })
    return geometry, canonical, gameplay


def official_equivalence_identity(spec):
    """Presentation-invariant executable puzzle key for copy exclusion."""
    geometry, actor, target = _normalized_transforms(spec)
    return _sha({
        "version": GAMEPLAY_VERSION,
        "actor": actor,
        "target": target,
        "initial_program": spec["initial_program"],
        "preset_programs": spec.get("preset_programs", []),
        "preset_positions": spec.get("preset_positions", []),
        "preset_rotations": spec.get("preset_rotations", []),
        "preset_scales": spec.get("preset_scales", []),
        "preset_resets": spec.get("preset_resets", []),
        "geometry": geometry,
    })


def split_bucket(geometry_d4_sha256):
    return int(geometry_d4_sha256, 16) % 100


def split_accepts(split, bucket):
    return (
        split == "train" and bucket < 80
        or split == "validation" and 80 <= bucket < 90
        or split == "test" and bucket >= 90
    )


def structural_metrics(spec):
    return {
        "slot_count": len(spec.get("initial_program", ())),
        "bit_width": spec.get("bit_width", -1),
        "selector_count": len(spec.get("preset_programs", ())),
        "wall_count": len(spec.get("walls", ())),
        "platform_count": len(spec.get("platforms", ())),
        "gate_count": len(spec.get("gates", ())),
    }


def profile_errors(spec, *, require_proof=True):
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in REFERENCE_PROFILES:
        return ["difficulty must be an integer in 1..7"]
    profile = REFERENCE_PROFILES[difficulty]
    if spec.get("quality_profile_version") != PROFILE_VERSION:
        errors.append("quality profile version mismatch")
    try:
        metrics = structural_metrics(spec)
        expected = {
            "slot_count": profile["slots"], "bit_width": profile["bit_width"],
            "selector_count": profile["selectors"], "wall_count": profile["walls"],
            "platform_count": profile["platforms"], "gate_count": profile["gates"],
        }
        for field, value in expected.items():
            if metrics[field] != value:
                errors.append(f"{field} differs from the official tier")
        if type(spec.get("native_budget")) is not int or spec.get("native_budget") != profile["native_budget"]:
            errors.append("native timer budget differs from the official tier")
        if spec.get("context_index") != profile["context_index"]:
            errors.append("native context differs from the official tier")
        density = spec.get("visual_density")
        if require_proof and (isinstance(density, bool)
                              or not isinstance(density, (int, float))
                              or not profile["visual_density"][0] <= density <= profile["visual_density"][1]):
            errors.append("native frame density is outside the reference tolerance")
        if tuple(spec["actor"]) == tuple(spec["target"]):
            errors.append("level is already solved at spawn")
        if require_proof:
            length = spec.get("solution_length", -1)
            low, high = profile["actions"]
            if type(length) is not int or not low <= length <= high:
                errors.append("public teacher action length is outside the calibrated tolerance")
            mechanics = spec.get("mechanic_mechanics", {})
            if not isinstance(mechanics, dict):
                mechanics = {}
            required = REQUIRED_MECHANIC_EVENTS[difficulty]
            if spec.get("required_solution_events") != list(required):
                errors.append("required_solution_events differs from the fixed tier obligations")
            for event in required:
                if mechanics.get("events", {}).get(event, 0) < 1:
                    errors.append(f"mechanic witness does not exercise {event}")
            if mechanics.get("wins") != 1:
                errors.append("mechanic witness must contain exactly one terminal win")
            public_mechanics = spec.get("solution_mechanics", {})
            if not isinstance(public_mechanics, dict) or public_mechanics.get("wins") != 1:
                errors.append("public teacher route must contain exactly one terminal win")
            if not isinstance(public_mechanics, dict):
                public_mechanics = {}
            public_events = public_mechanics.get("events", {})
            if not isinstance(public_events, dict):
                public_events = {}
            for event in required:
                if public_events.get(event, 0) < 1:
                    errors.append(f"public teacher route does not exercise {event}")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        errors.append(f"malformed full-standard spec: {exc}")
    return errors


def event_counts(events):
    return dict(sorted(Counter(events).items()))
