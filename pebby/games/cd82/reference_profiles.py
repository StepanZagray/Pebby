"""Measured CD82 reference tiers and explicit scarce-reference tolerances.

CD82 ships one level per tier, so these ranges are engineering tolerances
around six concrete references, not population confidence intervals.  Source
geometry is in ``third_party/arc3_games/cd82.py:36-361``; rules and budget are
in ``cd82.py:391-777``.  Action counts and solution mechanics were measured by
the exact planner and replayed sequentially through the native engine.
"""

from copy import deepcopy


DIFFICULTY_VERSION = "cd82-reference-v1"
MECHANICS_INVENTORY_VERSION = "cd82-mechanics-v1"
QUALITY_PROFILE_VERSION = "cd82-full-quality-v1"
DIFFICULTIES = tuple(range(1, 7))

_OFFICIAL = (
    # tier, palette, indicator, colours, density, actions, half, triangle, cap
    (1, 2, False, 2, 0.500, 5, 1, 0, 0),
    (2, 3, False, 3, 0.875, 6, 1, 1, 0),
    (3, 7, True,  4, 1.000, 16, 1, 2, 1),
    (4, 7, True,  4, 1.000, 13, 2, 1, 1),
    (5, 7, True,  4, 1.000, 13, 1, 2, 1),
    (6, 7, True,  5, 0.875, 16, 1, 1, 2),
)

# Wide enough to permit procedural variation while retaining the reference's
# structural/action tier.  With one official example per tier, tighter bounds
# would claim precision the source cannot support.
_ACTION_RANGES = ((4, 8), (5, 9), (13, 19), (10, 16), (10, 16), (13, 20))
_DENSITY_RANGES = ((.30, .70), (.70, 1.0), (.80, 1.0), (.80, 1.0), (.80, 1.0), (.70, 1.0))
_SEARCH_WORK = (20_000, 50_000, 200_000, 200_000, 200_000, 200_000)
_TARGET_COUNTS = (
    {0: 40, 15: 40},
    {0: 10, 12: 40, 15: 30},
    {8: 10, 12: 12, 14: 30, 15: 28},
    {9: 28, 11: 12, 12: 10, 15: 30},
    {8: 12, 9: 8, 12: 40, 14: 20},
    {0: 10, 8: 16, 11: 12, 14: 30, 15: 12},
)
_SEARCH_EXPANDED = (5, 10, 32_814, 1_731, 1_771, 15_506)
_SOLUTION_COUNTS = (
    {"dial_moves": 4, "swatch_clicks": 0, "region_paints": 1},
    {"dial_moves": 3, "swatch_clicks": 1, "region_paints": 2},
    {"dial_moves": 8, "swatch_clicks": 4, "region_paints": 3},
    {"dial_moves": 6, "swatch_clicks": 3, "region_paints": 3},
    {"dial_moves": 5, "swatch_clicks": 4, "region_paints": 3},
    {"dial_moves": 8, "swatch_clicks": 4, "region_paints": 2},
)

PROFILES = {}
for row, actions, density, search_work in zip(
        _OFFICIAL, _ACTION_RANGES, _DENSITY_RANGES, _SEARCH_WORK):
    tier, palette, indicator, colours, reference_density, reference_actions, half, triangle, cap = row
    PROFILES[tier] = {
        "reference_level": tier,
        "context_index": tier - 1,
        "palette_count": palette,
        "indicator": indicator,
        "atoms": 16 if indicator else 8,
        "operations": 12 if indicator else 8,
        "target_colours": colours,
        "reference_density": reference_density,
        "density": density,
        "reference_actions": reference_actions,
        "actions": actions,
        "search_work": search_work,
        "minimum_mechanics": {
            "half_paints": half,
            "triangle_paints": triangle,
            "cap_paints": cap,
        },
        # Compatibility keys used by the original core drafting loop while the
        # full generator is introduced in vertical slices.
        "swatches": (palette, palette),
        "paints": (2, 3) if tier == 1 else ((2, 3) if tier == 2 else (4, 7)),
        "max_actions": actions[1],
        "min_actions": actions[0],
    }


def official_characterization():
    """Return the six measured references without exposing mutable globals."""
    rows = []
    for index, (tier, palette, indicator, colours, density, actions, half, triangle, cap) in enumerate(_OFFICIAL):
        mechanics = dict(_SOLUTION_COUNTS[index])
        mechanics.update(half_paints=half, triangle_paints=triangle, cap_paints=cap)
        rows.append({
            "tier": tier,
            "context_index": tier - 1,
            "frame": [64, 64],
            "canvas": [10, 10],
            "compared_cells": 80,
            "native_budget": 100,
            "usable_actions": 99,
            "palette_count": palette,
            "indicator": indicator,
            "atoms": 16 if indicator else 8,
            "operations": 12 if indicator else 8,
            "target_colours": colours,
            "target_colour_counts": dict(_TARGET_COUNTS[index]),
            "target_nonblack_fraction": density,
            "optimal_actions": actions,
            "search_expanded": _SEARCH_EXPANDED[index],
            "controls": {"dial_actions": 4, "paint_action": 1, "click_action": 1},
            "objects": {
                "target": 1, "canvas": 1, "basket": 1, "cursor": 1,
                "swatches": palette, "indicator": int(indicator),
            },
            "constraints": {"ignored_diagonal_cells": 20, "loss_threshold": 100},
            "solution_mechanics": mechanics,
            "source": "third_party/arc3_games/cd82.py:36-777; exact native replay",
        })
    return deepcopy(rows)


def structural_metrics(spec):
    """Metrics that can be checked before spending any planner work."""
    from . import names
    from .layout import atom_colours, atomise

    target = spec["target"]
    if len(target) != 10 or any(len(row) != 10 for row in target):
        raise ValueError("target must be 10x10")
    indicator = bool(spec["indicator"])
    operations, atoms = atomise(indicator)
    if atom_colours(target, atoms) is None:
        raise ValueError("target is not uniform on native paint atoms")
    compared = [int(target[row][col]) for row in range(10) for col in range(10)
                if row != col and row + col != 9]
    colours = set(compared)
    return {
        "palette_count": len(spec["palette"]),
        "indicator": indicator,
        "atoms": len(atoms),
        "operations": len(operations),
        "target_colours": len(colours),
        "target_nonblack_fraction": sum(value != 0 for value in compared) / len(compared),
        "compared_cells": int(names.compare_mask().sum()),
    }


def profile_errors(spec, *, require_proof=True):
    """Human-readable violations of the measured tier contract."""
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty must be an integer in 1..6"]
    profile = PROFILES[difficulty]
    try:
        if type(spec.get("indicator")) is not bool:
            errors.append("indicator must be a boolean")
        palette = spec["palette"]
        if (not isinstance(palette, list)
                or any(type(value) is not int or not 0 <= value <= 15 for value in palette)):
            errors.append("palette must be a JSON list of ARC colour integers")
        target = spec["target"]
        if (not isinstance(target, list)
                or any(not isinstance(row, list) for row in target)
                or any(type(value) is not int or not 0 <= value <= 15
                       for row in target for value in row)):
            errors.append("target must contain ARC colour integers")
        metrics = structural_metrics(spec)
        if len(set(palette)) != len(palette):
            errors.append("palette colours must be unique")
        if 0 not in palette or 15 not in palette:
            errors.append("palette must contain blank and initially selected colours")
        target_colours = {int(spec["target"][row][col])
                          for row in range(10) for col in range(10)
                          if row != col and row + col != 9}
        if not target_colours <= set(palette):
            errors.append("target uses a colour absent from the palette")
        for key in ("palette_count", "indicator", "atoms", "operations", "target_colours"):
            if metrics[key] != profile[key]:
                errors.append(f"{key} differs from reference tier")
        low, high = profile["density"]
        if not low <= metrics["target_nonblack_fraction"] <= high:
            errors.append("target density outside scarce-reference tolerance")
        if metrics["compared_cells"] != 80:
            errors.append("native comparison geometry differs")
        context = difficulty - 1
        if (type(spec.get("reference_level")) is not int
                or spec.get("reference_level") != difficulty
                or type(spec.get("context_index")) is not int
                or spec.get("context_index") != context):
            errors.append("reference/native context differs from tier")
        if spec.get("difficulty_version") != DIFFICULTY_VERSION:
            errors.append("missing calibrated difficulty version")
        if spec.get("mechanics_inventory_version") != MECHANICS_INVENTORY_VERSION:
            errors.append("missing mechanics inventory version")
        if spec.get("quality_profile_version") != QUALITY_PROFILE_VERSION:
            errors.append("missing quality profile version")
        if spec.get("source") != "generated_only":
            errors.append("source must be generated_only")
        if spec.get("format") != "pebby-cd82-generated-v2" or spec.get("generator_version") != 2:
            errors.append("generated format/version differs")
        if require_proof:
            from .generation_quality import (
                IDENTITY_VERSION,
                canonical_identities,
                geometry_split,
                raw_geometry_identity,
            )

            actions = spec.get("optimal_actions", -1)
            if (type(spec.get("seed")) is not int or spec.get("seed", -1) < 0
                    or type(spec.get("generation_attempt")) is not int
                    or spec.get("generation_attempt", 0) < 1):
                errors.append("seed/generation attempt metadata is malformed")
            if type(actions) is not int or not profile["actions"][0] <= actions <= profile["actions"][1]:
                errors.append("optimal action length outside scarce-reference tolerance")
            solution_length = spec.get("solution_length")
            if (type(solution_length) is not int
                    or solution_length != len(spec.get("solution", []))
                    or actions != solution_length):
                errors.append("solution/action certificate is inconsistent")
            if spec.get("search_truncated") is not False:
                errors.append("exact search was truncated")
            if (type(spec.get("search_limit")) is not int
                    or spec.get("search_limit") != profile["search_work"]
                    or type(spec.get("search_expanded")) is not int
                    or not 0 <= spec.get("search_expanded", -1) <= profile["search_work"]):
                errors.append("search work certificate differs from tier cap")
            if (spec.get("engine_verified") is not True
                    or spec.get("context_engine_verified") is not True):
                errors.append("native context replay certificate is missing")
            if (type(spec.get("engine_budget")) is not int
                    or spec.get("engine_budget") != 100
                    or type(spec.get("usable_actions")) is not int
                    or spec.get("usable_actions") != 99
                    or type(spec.get("max_actions")) is not int
                    or spec.get("max_actions") != profile["max_actions"]):
                errors.append("native budget certificate differs")
            if (type(spec.get("training_context_index")) is not int
                    or spec.get("training_context_index") != context
                    or type(spec.get("verification_level_index")) is not int
                    or spec.get("verification_level_index") != context):
                errors.append("explicit native context certificate differs")
            mechanics = spec.get("solution_mechanics", {})
            for mechanic, minimum in profile["minimum_mechanics"].items():
                if mechanics.get(mechanic, 0) < minimum:
                    errors.append(f"solution does not exercise enough {mechanic}")
            for field in ("gameplay_sha256", "geometry_sha256"):
                value = spec.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    errors.append(f"missing canonical {field}")
            gameplay, geometry_d4 = canonical_identities(spec)
            geometry = raw_geometry_identity(spec)
            if (spec.get("gameplay_sha256") != gameplay
                    or spec.get("geometry_sha256") != geometry
                    or spec.get("geometry_d4_sha256") != geometry_d4):
                errors.append("canonical identity does not match target composition")
            if spec.get("split") not in ("train", "validation", "test"):
                errors.append("missing declared data split")
            elif geometry_split(geometry_d4) != spec.get("split"):
                errors.append("declared split differs from canonical geometry partition")
            if spec.get("official_copy") is not False:
                errors.append("official-copy rejection certificate is missing")
            if spec.get("geometry_version") != IDENTITY_VERSION:
                errors.append("geometry identity version differs")
            proof = spec.get("proof", {})
            proof_expected = {
                "format": spec.get("format"),
                "generator_version": spec.get("generator_version"),
                "difficulty_version": spec.get("difficulty_version"),
                "mechanics_inventory_version": spec.get("mechanics_inventory_version"),
                "quality_profile_version": spec.get("quality_profile_version"),
                "seed": spec.get("seed"),
                "difficulty": difficulty,
                "split": spec.get("split"),
                "gameplay_sha256": spec.get("gameplay_sha256"),
                "geometry_sha256": spec.get("geometry_sha256"),
                "geometry_d4_sha256": spec.get("geometry_d4_sha256"),
                "geometry_version": spec.get("geometry_version"),
                "context_index": context,
                "level_count": len(PROFILES),
                "native_budget": 100,
                "usable_actions": 99,
                "optimal_actions": actions,
                "search_limit": spec.get("search_limit"),
                "search_expanded": spec.get("search_expanded"),
                "search_truncated": False,
                "level_advanced": True,
                "levels_completed": 1,
                "actual_display_coordinates": True,
            }
            if any(type(proof.get(field)) is not type(expected)
                   or proof.get(field) != expected
                   for field, expected in proof_expected.items()):
                errors.append("nested proof fields differ from the native certificate")
            episode_top_fields = {
                "episode_engine_verified", "episode_index",
                "episode_levels_completed", "episode_sha256",
            }
            episode_proof_fields = episode_top_fields | {"forced_transitions"}
            if (episode_top_fields & spec.keys()) or (episode_proof_fields & proof.keys()):
                episode_sha256 = spec.get("episode_sha256")
                sha_is_valid = (isinstance(episode_sha256, str)
                                and len(episode_sha256) == 64
                                and all(character in "0123456789abcdef"
                                        for character in episode_sha256))
                episode_top_expected = {
                    "episode_engine_verified": True,
                    "episode_index": context,
                    "episode_levels_completed": len(PROFILES),
                }
                episode_proof_expected = {
                    **episode_top_expected,
                    "episode_sha256": episode_sha256,
                    "forced_transitions": 0,
                }
                if (not sha_is_valid
                        or any(type(spec.get(field)) is not type(expected)
                               or spec.get(field) != expected
                               for field, expected in episode_top_expected.items())
                        or any(type(proof.get(field)) is not type(expected)
                               or proof.get(field) != expected
                               for field, expected in episode_proof_expected.items())):
                    errors.append("episode certificate fields differ from the full game")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        errors.append(f"malformed reference spec: {error}")
    return errors
