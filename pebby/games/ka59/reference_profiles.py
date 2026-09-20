"""Official-derived KA59 tier contracts.

There is one shipped level per tier, so every range below is an explicit
engineering tolerance around a scarce reference, never a population estimate.
No official geometry or official action sequence is used by generation.

Source: ``third_party/arc3_games/ka59.py`` at snapshot
``3e8341f4c317ceb9992ca54e8ad453ee4e8a04547b27a907f4a560a9a30d857e``.
Level definitions are at lines 40935-41036 and mechanics at 41051-41461.
"""

from collections import Counter

from . import names


DIFFICULTY_VERSION = "ka59-official-reference-v2"
MECHANICS_VERSION = "ka59-full-native-body-retained-effects-v3"
QUALITY_VERSION = "ka59-reference-active-blast-quality-v3"
GEOMETRY_VERSION = "ka59-semantic-d4-extents-three-way-v2"
DIFFICULTIES = tuple(range(1, 8))

# Exact observations from the seven shipped levels.  ``reference_actions`` is
# a positive native witness measured in this worktree, not an optimal length.
# The values are positive witnesses only; none is labeled optimal.
_ROWS = (
    # grid, budget, boxes, targets, players, player targets,
    # explosives (prototype counts), wall pixels, witness, generated range.
    (45, 100, (names.BOX_3X3, names.BOX_3X3), 2, (), 0, (), 126, 11, (12, 22)),
    (63, 127, names.BOX_PROTOTYPES, 4, (), 0, (), 1611, 30, (28, 60)),
    (54, 100, (names.BOX_3X3,), 1, (names.PLAYER_LARGE, names.PLAYER_LARGE), 2, (), 0, 35, (18, 45)),
    (54, 127, (names.BOX_3X3, names.BOX_3X3), 2, (names.PLAYER_LARGE,), 1, (), 234, 46, (28, 65)),
    (63, 100, (names.BOX_3X3,), 1, (names.PLAYER_SMALL,), 0,
     ((names.EXPLOSIVE_SMALL, 4), (names.EXPLOSIVE_MIXED, 1)), 594, 24, (14, 32)),
    (63, 150, (names.BOX_3X3,), 1, (names.PLAYER_LARGE,), 1,
     ((names.EXPLOSIVE_LARGE, 3),), 360, 51, (20, 75)),
    (63, 200, (names.BOX_3X6, names.BOX_6X3), 2, (names.PLAYER_LARGE,), 1,
     ((names.EXPLOSIVE_LARGE, 2),), 210, 65, (35, 110)),
)

_REQUIRED_USE = (
    {"recursive_pushes": 1, "selection_clicks": 0},
    {"recursive_pushes": 0, "selection_clicks": 3},
    {"player_pushes": 2},
    {"player_pushes": 1, "selection_clicks": 1},
    {"explosions": 1, "blast_pushes": 1},
    {"explosions": 1, "blast_pushes": 1},
    {"explosions": 1, "blast_pushes": 1, "selection_clicks": 1},
)

PROFILES = {}
for difficulty, row in enumerate(_ROWS, 1):
    (
        grid,
        budget,
        boxes,
        targets,
        players,
        player_targets,
        explosives,
        wall_pixels,
        reference_actions,
        actions,
    ) = row
    # Wall tolerances reflect one irregular mask per tier.  Tier 3 has no
    # internal wall; other procedural masks may differ substantially while
    # retaining the reference's sparse/dense character.
    if wall_pixels == 0:
        wall_range = (0, 0)
    elif difficulty == 2:
        wall_range = (700, 2400)
    else:
        wall_range = (max(24, wall_pixels // 3), wall_pixels * 2 + 180)
    PROFILES[difficulty] = {
        "reference_level": difficulty,
        "context_index": difficulty - 1,
        "grid_size": grid,
        "step_budget": budget,
        "boxes": tuple(boxes),
        "targets": targets,
        "players": tuple(players),
        "player_targets": player_targets,
        "explosives": dict(explosives),
        "enemies": 0,
        "wall_pixels": wall_pixels,
        "wall_pixels_range": wall_range,
        "reference_actions": reference_actions,
        "generated_witness_actions": actions,
        "required_use": dict(_REQUIRED_USE[difficulty - 1]),
        "search_limit": 250_000,
    }

for difficulty, visible in enumerate((0, 0, 0, 0, 2, 3, 2), 1):
    PROFILES[difficulty]["visible_explosives"] = visible

PROFILES[5]["required_detonation_variants"] = (
    names.EXPLOSIVE_SMALL,
    names.EXPLOSIVE_MIXED,
)
PROFILES[5]["required_arranged_variants"] = (names.EXPLOSIVE_MIXED,)
PROFILES[5]["required_detonation_channels"] = ("detonation_changes_boxes",)
PROFILES[6]["required_detonation_variants"] = (names.EXPLOSIVE_LARGE,)
PROFILES[6]["required_arranged_variants"] = (names.EXPLOSIVE_LARGE,)
PROFILES[6]["required_detonation_channels"] = (
    "detonation_changes_boxes", "detonation_changes_players",
)
PROFILES[7]["required_detonation_variants"] = (names.EXPLOSIVE_LARGE,)
PROFILES[7]["required_arranged_variants"] = (names.EXPLOSIVE_LARGE,)
PROFILES[7]["required_detonation_channels"] = (
    "detonation_changes_boxes", "detonation_changes_players",
)


_SIZES = {
    names.BOX_3X3: (3, 3),
    names.BOX_3X6: (3, 6),
    names.BOX_6X3: (6, 3),
    names.BOX_6X6: (6, 6),
    names.PLAYER_LARGE: (9, 9),
    names.PLAYER_SMALL: (3, 3),
    names.EXPLOSIVE_SMALL: (3, 3),
    names.EXPLOSIVE_LARGE: (6, 6),
    names.EXPLOSIVE_MIXED: (6, 6),
}


def structural_metrics(spec):
    """Recompute structural facts from a JSON-ready procedural spec."""
    wall_rows = spec.get("wall_rows", ())
    wall_pixels = sum(row.count("1") for row in wall_rows)
    boxes = sorted(item["prototype"] for item in spec["boxes"])
    players = sorted(item["prototype"] for item in spec.get("players", ()))
    explosives = Counter(item["prototype"] for item in spec.get("explosives", ()))
    grid = int(spec["grid_size"])
    visible_explosives = sum(
        item["start"][0] < grid
        and item["start"][1] < grid
        and item["start"][0] + _SIZES[item["prototype"]][0] > 0
        and item["start"][1] + _SIZES[item["prototype"]][1] > 0
        for item in spec.get("explosives", ())
    )
    return {
        "grid_size": int(spec["grid_size"]),
        "step_budget": int(spec["step_budget"]),
        "box_count": len(spec["boxes"]),
        "boxes": boxes,
        "target_count": len(spec["boxes"]),
        "players": players,
        "player_target_count": sum(item.get("target") is not None for item in spec.get("players", ())),
        "explosives": dict(explosives),
        "visible_explosives": visible_explosives,
        "enemy_count": len(spec.get("enemies", ())),
        "wall_pixels": wall_pixels,
    }


def profile_errors(spec, *, require_proof=True):
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty must be an integer in 1..7"]
    profile = PROFILES[difficulty]
    if spec.get("difficulty_version") != DIFFICULTY_VERSION:
        errors.append("missing calibrated difficulty version")
    try:
        metrics = structural_metrics(spec)
        for key in ("grid_size", "step_budget"):
            if metrics[key] != profile[key]:
                errors.append(f"{key} differs from the official tier")
        if metrics["boxes"] != sorted(profile["boxes"]):
            errors.append("box shape composition differs from the official tier")
        if metrics["target_count"] != profile["targets"]:
            errors.append("ordinary target count differs from the official tier")
        if metrics["players"] != sorted(profile["players"]):
            errors.append("player composition differs from the official tier")
        if metrics["player_target_count"] != profile["player_targets"]:
            errors.append("player-target count differs from the official tier")
        if metrics["explosives"] != profile["explosives"]:
            errors.append("explosive prototype composition differs from the official tier")
        if metrics["visible_explosives"] != profile["visible_explosives"]:
            errors.append("visible explosive count differs from the official composition")
        if metrics["enemy_count"] != profile["enemies"]:
            errors.append("official tiers contain no pursuing enemies")
        low, high = profile["wall_pixels_range"]
        if not low <= metrics["wall_pixels"] <= high:
            errors.append("internal-wall visual density is outside the reference tolerance")
        grid = profile["grid_size"]
        for family in ("boxes", "players"):
            for index, item in enumerate(spec.get(family, ())):
                width, height = _SIZES[item["prototype"]]
                x, y = item["start"]
                if not (0 <= x and 0 <= y and x + width <= grid and y + height <= grid):
                    errors.append(f"{family}[{index}] start is clipped outside the native frame")
                target = item.get("target")
                if target is not None:
                    tx, ty = target
                    if not (
                        1 <= tx
                        and 1 <= ty
                        and tx + width + 1 <= grid
                        and ty + height + 1 <= grid
                    ):
                        errors.append(f"{family}[{index}] target outline is clipped outside the native frame")
                    if [x, y] == target:
                        errors.append(f"{family}[{index}] starts already satisfying its target")
        if require_proof:
            actions = spec.get("solution_length")
            action_low, action_high = profile["generated_witness_actions"]
            if type(actions) is not int or not action_low <= actions <= action_high:
                errors.append("constructive witness length is outside the reference tolerance")
            used = spec.get("solution_mechanics", {})
            for mechanic, minimum in profile["required_use"].items():
                if used.get(mechanic, 0) < minimum:
                    errors.append(f"winning route does not exercise enough {mechanic}")
            required_variants = profile.get("required_detonation_variants", ())
            effects = used.get("explosive_device_effects", ())
            if not isinstance(effects, list):
                errors.append("body-retaining explosive evidence is missing")
                effects = ()
            for prototype in required_variants:
                if not any(
                    isinstance(effect, dict)
                    and effect.get("prototype") == prototype
                    and effect.get("detonation_causal") is True
                    for effect in effects
                ):
                    errors.append(
                        f"winning route lacks a body-retaining detonation effect for {prototype}"
                    )
            for prototype in profile.get("required_arranged_variants", ()):
                if not any(
                    isinstance(effect, dict)
                    and effect.get("prototype") == prototype
                    and effect.get("detonation_causal") is True
                    and effect.get("arranged_before_detonation") is True
                    for effect in effects
                ):
                    errors.append(
                        f"winning route lacks an actively arranged detonation effect for {prototype}"
                    )
            for channel in profile.get("required_detonation_channels", ()):
                if not any(
                    isinstance(effect, dict) and effect.get(channel) is True
                    for effect in effects
                ):
                    errors.append(f"winning route lacks detonation evidence for {channel}")
            if required_variants and used.get("active_explosive_setups", 0) < 1:
                errors.append("winning route does not actively reposition an explosive")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"malformed reference spec: {exc}")
    return errors
