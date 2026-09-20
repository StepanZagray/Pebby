"""Official-derived eight-tier RE86 characterization and admission profiles.

There is one shipped level per tier.  Ranges below are explicit engineering
tolerances around those scarce references, not population confidence bounds.
"""

from pathlib import Path
import hashlib

import numpy as np

from . import names


DIFFICULTIES = tuple(range(1, 9))
DIFFICULTY_VERSION = "re86-official-eight-tier-v1"
MECHANICS_INVENTORY_VERSION = "re86-full-engine-2158-v1"
QUALITY_PROFILE_VERSION = "re86-reference-quality-v1"
SOURCE_SHA256 = "cf3c1520e17f0b70e7a3cea0e680601a591befce306b4dd1926c735a7d3b65e8"


_REFERENCE = {
    1: (100, 2, 0, 0, 0, 0, 8, 64, 234, (12, 40), 20_000),
    2: (100, 3, 1, 0, 0, 0, 10, 80, 287, (24, 58), 30_000),
    3: (200, 3, 1, 0, 0, 0, 8, 64, 269, (32, 78), 30_000),
    4: (200, 2, 0, 1, 0, 6, 6, 48, 424, (28, 76), 100_000),
    5: (250, 3, 1, 1, 0, 5, 10, 80, 461, (38, 105), 120_000),
    6: (200, 2, 1, 0, 1, 0, 8, 64, 298, (28, 90), 160_000),
    7: (300, 3, 1, 0, 1, 5, 9, 72, 443, (35, 135), 300_000),
    8: (400, 2, 2, 0, 2, 14, 8, 64, 606, (40, 165), 400_000),
}


def _profile(difficulty, values):
    (budget, movables, flexible, fixed, obstacles, dyes, anchors, guides,
     density, action_range, search_work) = values
    mechanics = ["translation", "cyclic_selection", "target_anchors"]
    if flexible:
        mechanics += ["flexible_center"]
    if fixed:
        mechanics += ["fixed_center"]
    if dyes:
        mechanics += ["dye", "flood_fill_animation", "secondary_color_state"]
    if obstacles:
        mechanics += ["obstacle_collision", "collision_deformation"]
    if difficulty >= 3:
        mechanics += ["ambiguous_target_assignment"]
    if difficulty >= 6:
        mechanics += ["flexible_resizing"]
    return {
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "native_budget": budget,
        "movables": movables,
        "flexible": flexible,
        "fixed_center": fixed,
        "obstacles": obstacles,
        "dyes": dyes,
        "target_colored": anchors,
        "target_guides": guides,
        "visual_nonbackground_reference": density,
        "visual_nonbackground_range": (
            max(80, int(density * 0.62)), int(density * 1.38)
        ),
        "action_range": action_range,
        "search_work": search_work,
        "required_mechanics": tuple(mechanics),
    }


PROFILES = {difficulty: _profile(difficulty, values)
            for difficulty, values in _REFERENCE.items()}


OFFICIAL_CHARACTERIZATION = {
    "source": "third_party/arc3_games/re86.py",
    "source_sha256": SOURCE_SHA256,
    "level_definitions": "lines 1615-1753",
    "win_and_selection": "lines 1894-1941",
    "movement_deformation": "lines 1943-2093",
    "dye_flood_animation": "lines 2076-2122",
    "dispatch_budget_completion": "lines 2124-2158",
    "reference_scope": "one shipped level per tier; tolerances are not confidence intervals",
    "known_positive_witness_actions": {
        1: 20,
        2: 37,
        3: 49,
        4: 43,
        5: 63,
        6: 46,
        7: 99,
        8: 158,
    },
    "witness_claim": (
        "Tier 1 is the legacy exact-subset optimum. Tiers 2-6 are bounded "
        "factored native positive witnesses, not optimality claims. Tiers 7 "
        "and 8 were positively replayed at 234,779 and 27,376 measured work "
        "units respectively."
    ),
}


def verify_source_identity(root=None):
    root = Path(__file__).resolve().parents[3] if root is None else Path(root)
    path = root / "third_party" / "arc3_games" / "re86.py"
    return hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256


def structural_metrics(env):
    level = env.level
    target = env.targets()[0]
    frame = np.asarray(env.render())
    target_pixels = np.asarray(target.pixels)
    return {
        "native_budget": env.max_steps,
        "movables": len(env.movables()),
        "flexible": sum(names.TAG_FLEXIBLE in sprite.tags
                        for sprite in env.movables()),
        "fixed_center": sum(names.TAG_FIXED_CENTER in sprite.tags
                            for sprite in env.movables()),
        "obstacles": len(level.get_sprites_by_tag(names.TAG_OBSTACLE)),
        "dyes": len(level.get_sprites_by_tag(names.TAG_DYE)),
        "target_colored": int(np.count_nonzero(
            (target_pixels != names.TRANSPARENT)
            & (target_pixels != names.TARGET_GUIDE)
        )),
        "target_guides": int(np.count_nonzero(
            target_pixels == names.TARGET_GUIDE
        )),
        "visual_nonbackground": int(np.count_nonzero(
            (frame != 5) & (frame != 3)
        )),
    }


def profile_errors(spec, metrics=None):
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty is not an official tier"]
    profile = PROFILES[difficulty]
    if metrics is None:
        metrics = spec.get("structural_metrics")
    if not isinstance(metrics, dict):
        return ["structural metrics are missing"]
    for key in ("native_budget", "movables", "flexible", "fixed_center",
                "obstacles", "dyes", "target_colored", "target_guides"):
        if metrics.get(key) != profile[key]:
            errors.append(f"{key} differs from tier reference")
    density = metrics.get("visual_nonbackground")
    low, high = profile["visual_nonbackground_range"]
    if type(density) is not int or not low <= density <= high:
        errors.append("visual density is outside the reference tolerance")
    actions = spec.get("solution_length")
    low, high = profile["action_range"]
    if type(actions) is not int or not low <= actions <= high:
        errors.append("solution witness length is outside the reference tolerance")
    mechanics = spec.get("solution_mechanics")
    if not isinstance(mechanics, dict) or mechanics.get("engine_win") is not True:
        errors.append("solution mechanic certificate is missing a native win")
        return errors
    if mechanics.get("selection_actions", 0) < (1 if profile["movables"] > 1 else 0):
        errors.append("cyclic selection is not exercised")
    if profile["dyes"] and mechanics.get("dye_events", 0) < 1:
        errors.append("dye/flood-fill is not exercised")
    if profile["dyes"] and mechanics.get("dye_animation_frames", 0) < 1:
        errors.append("dye animation frames are not witnessed")
    if profile["fixed_center"] and mechanics.get("fixed_center_selections", 0) < 1:
        errors.append("fixed-centre selection behavior is not exercised")
    if profile["fixed_center"] and mechanics.get(
            "target_constrained_selection_centers", 0) < 1:
        errors.append("target-constrained selection centre is not exercised")
    if profile["flexible"] and mechanics.get("flexible_selections", 0) < 1:
        errors.append("flexible selection behavior is not exercised")
    if difficulty >= 6 and mechanics.get("resize_events", 0) < 1:
        errors.append("flexible obstacle resizing is not exercised")
    if difficulty == 3 and mechanics.get("ambiguous_target_assignment") is not True:
        errors.append("same-colour ambiguous target assignment is not exercised")
    if profile["obstacles"] and (
            mechanics.get("resize_events", 0)
            + mechanics.get("deformation_events", 0) < 1):
        errors.append("obstacle collision deformation is not exercised")
    if difficulty == 7 and mechanics.get("deformation_events", 0) < 1:
        errors.append("ordinary obstacle pixel deformation is not exercised")
    return errors
