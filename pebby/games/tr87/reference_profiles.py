"""Reference-calibrated contracts for all six shipped TR87 tiers.

There is one official level per tier, so the ranges below are deliberate
tolerances around scarce references, not population confidence intervals.
The source facts come from ``third_party/arc3_games/tr87.py``:

* levels and object geometry: lines 472-877;
* native budgets and level-index seeds: lines 910-969;
* controls and editable groups: lines 974-1022;
* direct, double and tree translation: lines 1044-1105.

Reference action counts for tiers 1-5 are exhaustive symbolic minima.  Tier 6
is explicitly a constructive 35-action replay witness; no optimality claim is
made for it.  See the family characterization report for the measured rows.
"""

from .generation_quality import gameplay_identity, geometry_identity

# Structural/difficulty tolerances are unchanged by the v3 generator fix;
# retaining this version also leaves tiers 1-5 seed mappings untouched.
DIFFICULTY_VERSION = "tr87-reference-v2"
MECHANICS_INVENTORY_VERSION = "tr87-mechanics-v2"
DIFFICULTIES = tuple(range(1, 7))

# The arities are multisets: rule order is gameplay-relevant and remains
# procedural, while the amount of visual and grammatical structure is fixed to
# the corresponding official tier.
_ROWS = (
    # mode, split, rows, rules, arities, source, target, tiles, density,
    # measured reference actions, calibrated accepted range, budget
    ("plain", 34, 3, 6, ((1, 1),) * 6, 5, 5, 22, 1156, 14, (12, 17), 128),
    ("plain", 34, 3, 6,
     ((1, 1), (1, 1), (1, 2), (1, 2), (1, 3), (1, 3)),
     4, 7, 29, 1499, 25, (22, 29), 128),
    ("plain", 34, 3, 6,
     ((1, 1), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1)),
     8, 7, 33, 1695, 21, (18, 25), 128),
    ("double", 37, 4, 8, ((1, 1),) * 8, 7, 7, 30, 1548, 21, (18, 25), 128),
    ("alter", 38, 2, 4, ((1, 1), (1, 1), (1, 2), (2, 1)),
     5, 5, 20, 1058, 14, (12, 20), 128),
    ("tree_alter_double", 41, 3, 6,
     ((1, 1), (1, 1), (1, 1), (1, 2), (1, 2), (1, 2)),
     3, 6, 24, 1254, 35, (30, 42), 256),
)


def _flags(mode):
    return {
        "alter_rules": mode in ("alter", "tree_alter_double"),
        "double_translation": mode in ("double", "tree_alter_double"),
        "tree_translation": mode == "tree_alter_double",
    }


PROFILES = {}
for difficulty, row in enumerate(_ROWS, 1):
    (mode, split_y, rule_rows, rule_count, arities, source_count,
     target_count, tile_count, density, actions, action_range, budget) = row
    PROFILES[difficulty] = {
        "reference_level": difficulty,
        "context_index": difficulty - 1,
        "mode": mode,
        "flags": _flags(mode),
        "split_y": split_y,
        "rule_rows": rule_rows,
        "rule_count": rule_count,
        "rule_arities": tuple(sorted(arities)),
        "source_count": source_count,
        "target_count": target_count,
        "tile_count": tile_count,
        # Pixel count includes the live cursor and HUD.  +/-160 covers glyph
        # ink variation while still rejecting sparse/core-shaped boards.
        "visual_nonbackground_pixels": (density - 160, density + 160),
        "reference_actions": actions,
        "action_range": action_range,
        "reference_actions_optimal": difficulty <= 5,
        "budget": budget,
        "search_work": 400_000,
    }


def structural_metrics(spec):
    """Return gameplay/geometry counts without trusting stored metadata."""
    rules = spec["rules"]
    rule_rows = len({rule["y"] for rule in rules})
    arities = tuple(sorted((len(rule["lhs"]), len(rule["rhs"])) for rule in rules))
    source_count = len(spec["source"]["symbols"])
    target_count = len(spec["target"]["symbols"])
    tile_count = source_count + target_count + sum(a + b for a, b in arities)
    return {
        "split_y": spec["split_y"],
        "rule_rows": rule_rows,
        "rule_count": len(rules),
        "rule_arities": arities,
        "source_count": source_count,
        "target_count": target_count,
        "tile_count": tile_count,
    }


def profile_errors(spec, *, require_proof=True):
    """Human-readable violations of the declared official-tier contract."""
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty must be an integer in 1..6"]
    profile = PROFILES[difficulty]
    if spec.get("difficulty_version") != DIFFICULTY_VERSION:
        errors.append("missing calibrated difficulty version")
    if spec.get("quality_profile_version") != "tr87-reference-quality-v3":
        errors.append("missing quality profile version")
    if spec.get("generator_version") != 3 or spec.get("format") != "pebby.tr87.full-level.v3":
        errors.append("unsupported full generator format/version")
    if spec.get("mode") != profile["mode"]:
        errors.append("mode differs from reference tier")
    try:
        metrics = structural_metrics(spec)
        for key in ("split_y", "rule_rows", "rule_count", "rule_arities",
                    "source_count", "target_count", "tile_count"):
            if metrics[key] != profile[key]:
                errors.append(f"{key} differs from reference tier")
        flags = spec.get("mechanics", {})
        for key, expected in profile["flags"].items():
            if flags.get(key) is not expected:
                errors.append(f"{key} differs from reference tier")
    except (KeyError, TypeError, ValueError):
        errors.append("malformed structural specification")
        return errors
    if not require_proof:
        return errors
    if (spec.get("context_index") != profile["context_index"]
            or spec.get("training_context_index") != profile["context_index"]):
        errors.append("verification context differs from reference tier")
    if spec.get("native_budget") != profile["budget"]:
        errors.append("native action budget differs from reference tier")
    if not spec.get("context_engine_verified") or not spec.get("engine_win"):
        errors.append("missing real-engine context replay certificate")
    actions = spec.get("solution_length", -1)
    low, high = profile["action_range"]
    if not low <= actions <= high:
        errors.append("witness action length outside calibrated range")
    if (spec.get("budget_remaining") != profile["budget"] - actions
            or spec.get("budget_remaining", -1) < 0):
        errors.append("witness/native budget accounting disagrees")
    density = spec.get("visual_nonbackground_pixels", -1)
    low, high = profile["visual_nonbackground_pixels"]
    if not low <= density <= high:
        errors.append("visual density outside calibrated range")
    mechanics = spec.get("solution_mechanics", {})
    solution = spec.get("solution", [])
    if spec.get("context_solution") != solution:
        errors.append("context witness differs from stored solution")
    if len(solution) != actions or any(
            not isinstance(step, (list, tuple)) or len(step) != 3
            or step[0] not in (1, 2, 3, 4) or step[1:] not in ([None, None], (None, None))
            for step in solution):
        errors.append("solution does not use actual display action semantics")
    if mechanics.get("cycle_actions") != sum(step[0] in (1, 2) for step in solution):
        errors.append("cycle-action evidence disagrees with witness")
    if mechanics.get("select_actions") != sum(step[0] in (3, 4) for step in solution):
        errors.append("selection-action evidence disagrees with witness")
    if mechanics.get("cycle_actions", 0) < 1:
        errors.append("witness does not exercise a cycle action")
    if difficulty in (4, 6) and mechanics.get("translation_depth", 0) < 2:
        errors.append("witness does not exercise composed translation")
    if profile["flags"]["alter_rules"] and mechanics.get("edited_rule_groups", 0) < 4:
        errors.append("witness edits too few rule groups")
    if difficulty == 6:
        if mechanics.get("tree_branch_exercised") is not True:
            errors.append("tier 6 witness does not exercise tree translation")
        mixed = mechanics.get("tree_mixed_child_expansions", 0)
        repeated = mechanics.get("tree_repeated_child_expansions", 0)
        if mixed < 1:
            errors.append("tier 6 winning trace has no mixed-child tree expansion")
        if repeated < 1:
            errors.append("tier 6 winning trace has no repeated-child tree expansion")
        if mixed + repeated != mechanics.get("translation_segments"):
            errors.append("tier 6 child-shape evidence disagrees with winning trace segments")
    for key in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256"):
        if not isinstance(spec.get(key), str) or len(spec[key]) != 64:
            errors.append(f"missing canonical {key}")
    try:
        if spec.get("geometry_sha256") != geometry_identity(spec):
            errors.append("geometry identity does not match content")
        if spec.get("geometry_d4_sha256") != geometry_identity(spec):
            errors.append("D4 geometry identity does not match content")
        if spec.get("gameplay_sha256") != gameplay_identity(spec):
            errors.append("gameplay identity does not match content")
    except (KeyError, TypeError, ValueError):
        errors.append("canonical identity cannot be recomputed")
    if spec.get("official_copy") is not False:
        errors.append("official-copy gate missing or failed")
    if spec.get("split") not in ("train", "validation", "test"):
        errors.append("invalid split")
    if spec.get("geometry_split") != spec.get("split"):
        errors.append("geometry identity is in a different split")
    return errors
