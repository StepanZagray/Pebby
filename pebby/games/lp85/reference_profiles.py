"""Measured LP85 reference tiers and executable generation tolerances.

The measurements below come from the eight shipped levels in
``third_party/arc3_games/lp85.py``:

* level objects and budgets: lines 1091-1581;
* numbered-map compiler and rotation direction: lines 21242-21286;
* click, budget, and completion semantics: lines 21339-21450.

There is one official level per tier.  Consequently the ranges are explicit
engineering tolerances around scarce references, not population confidence
intervals.  Only aggregate measurements are retained here; official layouts
and routes are never generator inputs.
"""

from __future__ import annotations

from collections.abc import Mapping


DIFFICULTIES = tuple(range(1, 9))
QUALITY_PROFILE_VERSION = "lp85-official-eight-tier-v1"
MECHANICS_INVENTORY_VERSION = "lp85-native-cycle-mechanics-v1"


# Exact aggregate measurements. ``reference_actions`` is the shortest length
# found by goal-projection BFS and then replayed in one uninterrupted native
# eight-level episode.  It does not store or expose the official route.
_REFERENCE_ROWS = (
    # grid, budget, cycles, lengths, union, overlap pairs/incidences/max,
    # nested, controls/sites/effects, stacked/max, duplicate extras,
    # controlled/passive/right-only/bidirectional, normal/alternate goals,
    # initially satisfied, shortest actions, shortest-route used effects/groups.
    ((32, 19), 13, 1, (20,), 20, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0,
     1, 0, 0, 1, 1, 0, 0, 5, 1, 1),
    ((41, 41), 60, 3, (10, 10, 26), 42, 2, 4, 2, 0, 6, 6, 6, 0, 1, 0,
     3, 0, 0, 3, 2, 0, 0, 8, 2, 2),
    ((39, 31), 80, 2, (16, 16), 30, 1, 2, 2, 0, 4, 4, 4, 0, 1, 0,
     2, 0, 0, 2, 1, 1, 0, 16, 2, 2),
    ((57, 57), 150, 2, (20, 20), 36, 1, 4, 4, 0, 16, 16, 4, 0, 1, 12,
     2, 0, 0, 2, 1, 1, 0, 12, 2, 2),
    ((27, 32), 80, 2, (5, 21), 21, 1, 5, 5, 1, 4, 4, 4, 0, 1, 0,
     2, 0, 0, 2, 2, 0, 0, 9, 2, 2),
    ((60, 64), 80, 36, (2, 2, 2) + (3,) * 24 + (8,) * 9, 75,
     78, 78, 1, 0, 36, 7, 7, 7, 8, 0, 36, 0, 36, 0, 3, 0, 0,
     19, 7, 36),
    ((48, 36), 80, 4, (3, 4, 4, 8), 17, 2, 2, 1, 0, 6, 4, 4, 2, 2, 0,
     3, 1, 0, 3, 2, 0, 1, 5, 4, 3),
    ((63, 63), 80, 6, (2, 6, 7, 14, 15, 16), 45, 3, 15, 7, 3,
     12, 8, 8, 2, 3, 0, 6, 0, 0, 6, 3, 0, 0, 5, 3, 5),
)


_FIELDS = (
    "grid", "step_budget", "cycle_count", "cycle_lengths", "union_cells",
    "overlap_pairs", "overlap_incidences", "max_pair_overlap", "nested_pairs",
    "control_sprites", "control_sites", "effect_signatures", "stacked_sites",
    "max_stack", "duplicate_control_extras", "controlled_groups", "passive_cycles",
    "right_only_groups", "bidirectional_groups", "normal_goals", "alternate_goals",
    "initially_satisfied_goals", "reference_actions", "reference_used_effects",
    "reference_used_groups",
)


REFERENCE = {
    tier: dict(zip(_FIELDS, row, strict=True))
    for tier, row in enumerate(_REFERENCE_ROWS, 1)
}


# Generated admission contracts. Structural signatures that define a mechanic
# are exact; geometry/action ranges allow new layouts while staying near the
# single available reference. Tier 6 deliberately retains the exact 36-cycle,
# seven-stack composition because weakening it would remove the official rule.
_ACTION_RANGES = ((3, 7), (4, 12), (12, 20), (6, 18), (5, 14), (12, 28), (4, 7), (3, 8))
_UNION_RANGES = ((16, 24), (36, 48), (26, 36), (32, 42), (18, 24), (72, 78), (15, 21), (40, 50))
_GRID_RANGES = (
    ((28, 36), (17, 23)), ((37, 45), (37, 45)), ((35, 43), (29, 37)),
    ((51, 63), (51, 63)), ((25, 35), (28, 38)), ((57, 63), (60, 64)),
    ((42, 54), (33, 43)), ((57, 63), (57, 63)),
)


PROFILES = {}
for tier in DIFFICULTIES:
    reference = REFERENCE[tier]
    profile = dict(reference)
    profile.update(
        difficulty=tier,
        context_index=tier - 1,
        actions=_ACTION_RANGES[tier - 1],
        union_cells_range=_UNION_RANGES[tier - 1],
        grid_width_range=_GRID_RANGES[tier - 1][0],
        grid_height_range=_GRID_RANGES[tier - 1][1],
        search_work=(200_000 if tier == 6 else 100_000),
    )
    PROFILES[tier] = profile


def _in_range(value, bounds):
    return bounds[0] <= value <= bounds[1]


def profile_errors(spec, metrics, *, require_proof=True):
    """Return fail-closed profile errors for a generated JSON specification."""
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty must be an integer in 1..8"]
    profile = PROFILES[difficulty]
    errors = []
    if spec.get("quality_profile_version") != QUALITY_PROFILE_VERSION:
        errors.append("quality profile version mismatch")
    if spec.get("mechanics_inventory_version") != MECHANICS_INVENTORY_VERSION:
        errors.append("mechanics inventory version mismatch")
    if type(spec.get("step_budget")) is not int or spec.get("step_budget") != profile["step_budget"]:
        errors.append("native step budget differs from the reference tier")
    if not _in_range(metrics.get("grid_width", -1), profile["grid_width_range"]):
        errors.append("grid width outside reference tolerance")
    if not _in_range(metrics.get("grid_height", -1), profile["grid_height_range"]):
        errors.append("grid height outside reference tolerance")
    if not _in_range(metrics.get("union_cells", -1), profile["union_cells_range"]):
        errors.append("visual object density outside reference tolerance")

    exact = (
        "cycle_count", "overlap_pairs", "overlap_incidences", "max_pair_overlap",
        "nested_pairs", "control_sprites", "control_sites", "effect_signatures",
        "stacked_sites", "max_stack", "duplicate_control_extras", "controlled_groups",
        "passive_cycles", "right_only_groups", "bidirectional_groups", "normal_goals",
        "alternate_goals", "initially_satisfied_goals",
    )
    for key in exact:
        if metrics.get(key) != profile[key]:
            errors.append(f"{key} differs from the reference mechanic composition")

    if require_proof:
        length = spec.get("solution_length")
        if type(length) is not int or not _in_range(length, profile["actions"]):
            errors.append("solution length outside reference tolerance")
        if type(spec.get("training_context_index")) is not int:
            errors.append("training context index must be an integer")
        elif spec.get("training_context_index") != profile["context_index"]:
            errors.append("training context differs from the official tier")
        if type(spec.get("verification_level_index")) is not int:
            errors.append("verification level index must be an integer")
        elif spec.get("verification_level_index") != profile["context_index"]:
            errors.append("native verification context differs from the official tier")
        budget = spec.get("native_budget")
        if not isinstance(budget, Mapping):
            errors.append("native budget must be a mapping")
            budget = {}
        if type(budget.get("initial")) is not int or budget.get("initial") != profile["step_budget"]:
            errors.append("native budget proof is missing or inconsistent")
        used = spec.get("solution_mechanics", {})
        if not isinstance(used, Mapping):
            errors.append("solution mechanics must be a mapping")
            used = {}
        if not used.get("all_targets_satisfied"):
            errors.append("solution did not satisfy every target kind")
        minimum_effects = (1, 2, 2, 2, 2, 7, 3, 3)[difficulty - 1]
        minimum_groups = (1, 2, 2, 2, 2, 36, 3, 5)[difficulty - 1]
        if used.get("used_effect_signature_count", 0) < minimum_effects:
            errors.append("too few distinct native control effects exercised")
        if used.get("used_group_count", 0) < minimum_groups:
            errors.append("too few movement groups exercised")
        if difficulty in (2, 3, 4, 5) and used.get("overlap_goal_transitions", 0) < 1:
            errors.append("shared-cycle interaction was not exercised")
        if difficulty == 4 and used.get("duplicate_control_clicks", 0) < 1:
            errors.append("duplicated control presentation was not exercised")
        if difficulty == 5 and used.get("nested_cycle_clicks", 0) < 1:
            errors.append("fully nested cycle interaction was not exercised")
        if difficulty == 6:
            if used.get("stacked_clicks", 0) < 7 or used.get("max_stack_used", 0) < 8:
                errors.append("tier-6 compound one-way controls were not fully exercised")
            if used.get("one_way_clicks", 0) != length:
                errors.append("tier-6 witness contains a non-one-way click")
        if difficulty == 7:
            if used.get("stacked_clicks", 0) < 1:
                errors.append("tier-7 coupled control was not exercised")
            if used.get("passive_overlap_transitions", 0) < 1:
                errors.append("tier-7 passive-cycle overlap was not exercised")
            if used.get("temporarily_displaced_initial_target", 0) < 1:
                errors.append("tier-7 initially satisfied constraint was not displaced and restored")
        if difficulty == 8:
            if used.get("stacked_clicks", 0) < 1 or used.get("nested_cycle_clicks", 0) < 2:
                errors.append("tier-8 nested/stacked interaction was not exercised")
    return errors
