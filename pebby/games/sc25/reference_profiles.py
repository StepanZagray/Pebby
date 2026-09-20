"""Measured one-reference-per-tier contracts for the six shipped SC25 levels.

Each tier has one official level, so the ranges below are explicit tolerances
around scarce references, not population confidence intervals.  Witness action
counts come from bounded symbolic construction plus real-engine replay; they
are deliberately not labelled optimal.
"""

from . import names


DIFFICULTIES = tuple(range(1, 7))
PROFILE_VERSION = "sc25-official-six-v2-target-covered-pickups"

# Source: third_party/arc3_games/sc25.py:1378-1640, measured through Layout.
REFERENCE_PROFILES = {
    1: dict(budget=50, spells=(names.SPELL_GROW,), pads=0, small_pads=0,
            pickups=0, primary=0, alternate=0, rings=0,
            density=(0.78, 0.98), witness_actions=(10, 24),
            required=("tutorial_demo_actions", "shrink_casts")),
    2: dict(budget=25, spells=(names.SPELL_TELEPORT,), pads=1, small_pads=0,
            pickups=0, primary=0, alternate=0, rings=0,
            density=(0.78, 0.98), witness_actions=(4, 16),
            required=("large_teleports",)),
    3: dict(budget=50, spells=(names.SPELL_FIRE,), pads=0, small_pads=0,
            pickups=0, primary=1, alternate=0, rings=0,
            density=(0.74, 0.97), witness_actions=(8, 25),
            required=("primary_targets_hit",)),
    4: dict(budget=35, spells=(names.SPELL_FIRE, names.SPELL_GROW), pads=0,
            small_pads=0, pickups=1, primary=1, alternate=0, rings=0,
            density=(0.68, 0.93), witness_actions=(15, 38),
            required=("shrink_casts", "grow_casts", "primary_targets_hit",
                      "pickups_consumed")),
    5: dict(budget=65, spells=(names.SPELL_FIRE, names.SPELL_TELEPORT, names.SPELL_GROW),
            pads=1, small_pads=1, pickups=1, primary=1, alternate=1, rings=0,
            density=(0.70, 0.96), witness_actions=(22, 58),
            required=("shrink_casts", "grow_casts", "large_teleports",
                      "small_teleports", "primary_targets_hit",
                      "alternate_targets_hit", "pickups_consumed")),
    6: dict(budget=60, spells=(names.SPELL_FIRE, names.SPELL_TELEPORT, names.SPELL_GROW),
            pads=2, small_pads=1, pickups=2, primary=1, alternate=1, rings=0,
            density=(0.70, 0.97), witness_actions=(24, 62),
            required=("shrink_casts", "grow_casts", "large_teleports",
                      "small_teleports", "primary_targets_hit",
                      "alternate_targets_hit", "pickups_consumed",
                      "target_covered_pickups",
                      "target_clear_before_pickup_pairs")),
}


OFFICIAL_MEASUREMENTS = {
    1: dict(density=0.9065, witness_actions=13, moves=9, clicks=4),
    2: dict(density=0.8708, witness_actions=5, moves=2, clicks=3),
    3: dict(density=0.8625, witness_actions=11, moves=8, clicks=3),
    4: dict(density=0.7732, witness_actions=22, moves=11, clicks=11),
    5: dict(density=0.8264, witness_actions=36, moves=16, clicks=20),
    6: dict(density=0.8823, witness_actions=35, moves=12, clicks=23),
}


def profile_errors(spec, *, require_proof=True):
    """Return reference-profile violations for a JSON level spec."""
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in REFERENCE_PROFILES:
        return ["difficulty is not an official tier"]
    profile = REFERENCE_PROFILES[difficulty]
    errors = []
    for field in ("budget", "pads", "small_pads", "pickups"):
        value = len(spec.get(field, [])) if field != "budget" else spec.get(field)
        if value != profile[field]:
            errors.append(f"{field}={value!r}, expected {profile[field]!r}")
    primary = sum(item.get("family", "primary") == "primary" for item in spec.get("targets", []))
    alternate = sum(item.get("family") == "alternate" for item in spec.get("targets", []))
    rings = len(spec.get("rings", []))
    for field, value in (("primary", primary), ("alternate", alternate), ("rings", rings)):
        if value != profile[field]:
            errors.append(f"{field}={value}, expected {profile[field]}")
    if tuple(spec.get("spells", ())) != profile["spells"]:
        errors.append("spell composition differs from the official tier")
    density = spec.get("visual_density")
    if density is None or not profile["density"][0] <= density <= profile["density"][1]:
        errors.append(f"visual_density={density!r} outside {profile['density']}")
    if require_proof:
        length = spec.get("solution_length")
        if length is None or not profile["witness_actions"][0] <= length <= profile["witness_actions"][1]:
            errors.append(f"solution_length={length!r} outside {profile['witness_actions']}")
        mechanics = spec.get("solution_mechanics", {})
        for key in profile["required"]:
            minimum = 2 if difficulty == 6 and key in (
                "large_teleports", "pickups_consumed", "target_covered_pickups",
                "target_clear_before_pickup_pairs",
            ) else 1
            if mechanics.get(key, 0) < minimum:
                errors.append(f"solution does not exercise {key} >= {minimum}")
    return errors
