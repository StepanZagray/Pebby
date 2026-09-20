"""Measured DC22 reference facts and explicit one-level tolerances.

There is one shipped puzzle per tier, so these are engineering tolerances around
six individual references, not population confidence intervals.  Every tier
has a measured positive witness, replayed in its real native context.  These
witness lengths are deliberately not presented as official optima.
"""

from collections import Counter

from . import names


DIFFICULTY_VERSION = "dc22-official-six-v1"
QUALITY_VERSION = "dc22-reference-tolerances-v4"
DIFFICULTIES = tuple(range(1, 7))
STEP_BUDGETS = (128, 192, 192, 192, 512, 1024)
SEARCH_LIMITS = (25_000, 50_000, 75_000, 100_000, 1_000_000, 500_000)

# Measured by constructing each official level in its real context and sampling
# all even player top-left coordinates in the playfield.  ``reference_actions``
# records positive native-replayed witnesses; no shortest-route claim is made.
_REFERENCE = (
    ((64, 44), 14, 17, 36, (8, 8, 26, 32), 2, 2, 3, 0, 0, 0, 0, 0, 0, 20),
    ((64, 48), 19, 24, 68, (4, 4, 34, 46), 3, 3, 5, 1, 0, 0, 0, 0, 0, 42),
    ((64, 48), 24, 33, 48, (4, 8, 32, 38), 4, 4, 9, 1, 2, 0, 0, 0, 0, 45),
    ((64, 48), 26, 52, 56, (4, 10, 30, 36), 3, 3, 6, 0, 2, 0, 0, 0, 0, 62),
    ((64, 64), 37, 55, 63, (10, 6, 32, 54), 5, 10, 8, 1, 2, 1, 1, 0, 0, 108),
    ((64, 64), 72, 97, 136, (0, 4, 38, 60), 9, 8, 9, 2, 4, 1, 0, 4, 19, 141),
)
_SUPPORT_RANGES = ((24, 48), (40, 90), (30, 70), (38, 76), (42, 90), (100, 170))
_ACTION_RANGES = ((12, 48), (16, 68), (18, 80), (24, 104), (24, 144), (28, 192))
_MECHANICS = (
    ("colour_control", "surface_cycle", "fall_recovery"),
    ("colour_control", "surface_cycle", "key_pickup", "key_unlocked_control", "fall_recovery"),
    ("colour_control", "surface_cycle", "key_pickup", "key_unlocked_control", "bridge_teleport", "fall_recovery"),
    ("colour_control", "expanding_surface", "moving_surface", "bridge_teleport", "fall_recovery"),
    ("colour_control", "surface_cycle", "expanding_surface", "moving_surface", "key_pickup", "key_unlocked_control", "bridge_teleport", "crusher_move", "object_grab", "object_carry", "fall_recovery"),
    ("colour_control", "surface_cycle", "moving_surface", "key_pickup", "key_unlocked_control", "bridge_teleport", "track_crusher_move", "bridge_grab", "bridge_carry", "bridge_colour_cycle", "pressure_control", "fall_recovery"),
)

PROFILES = {}
for difficulty, values in enumerate(_REFERENCE, 1):
    (
        grid_size,
        raw_sprites,
        engine_sprites,
        support_samples,
        support_bbox,
        colour_controls,
        click_controls,
        toggle_instances,
        keys,
        bridges,
        crushers,
        carried_objects,
        sensors,
        track_cells,
        reference_actions,
    ) = values
    PROFILES[difficulty] = {
        "reference_level": difficulty,
        "context_index": difficulty - 1,
        "grid_size": grid_size,
        "step_budget": STEP_BUDGETS[difficulty - 1],
        "raw_sprites": raw_sprites,
        "engine_sprites": engine_sprites,
        "support_samples": support_samples,
        "support_bbox": support_bbox,
        "colour_controls": colour_controls,
        "click_controls": click_controls,
        "toggle_instances": toggle_instances,
        "keys": keys,
        "bridges": bridges,
        "crushers": crushers,
        "carried_objects": carried_objects,
        "sensors": sensors,
        "track_cells": track_cells,
        "reference_actions": reference_actions,
        "generated_support_range": _SUPPORT_RANGES[difficulty - 1],
        "generated_action_range": _ACTION_RANGES[difficulty - 1],
        "required_mechanics": _MECHANICS[difficulty - 1],
        "required_winning_mechanics": _MECHANICS[difficulty - 1],
        "required_interaction_mechanics": (),
        "search_limit": SEARCH_LIMITS[difficulty - 1],
    }


def _support_samples(level):
    """Count even top-left points supported by initially active native sprites."""
    # Count global player-aligned pixels in the x<40 playfield. Irregular bridge
    # masks have support on odd sprite-local offsets, so local ``range(..., 2)``
    # sampling would undercount them even though their global coordinates are
    # valid two-pixel player positions.
    points = set()
    for sprite in level.get_sprites():
        tags = set(sprite.tags)
        if names.TAG_CRUSHER in tags or names.TAG_FALL_BLOCKER in tags:
            continue
        initialized_support = (
            sprite.interaction.name == "INTANGIBLE"
            or names.TAG_TOGGLE in tags and ("omvz" in tags or names.TAG_BUTTON in tags)
        )
        if not initialized_support:
            continue
        pixels = sprite.render()
        for y in range(sprite.height):
            for x in range(sprite.width):
                gx, gy = int(sprite.x) + x, int(sprite.y) + y
                if 0 <= gx < 40 and gx % 2 == 0 and gy % 2 == 0 and pixels[y, x] >= 0:
                    points.add((gx, gy))
    return points


def structural_metrics(level):
    sprites = level.get_sprites()
    counts = Counter()
    for sprite in sprites:
        tags = set(sprite.tags)
        counts["colour_controls"] += names.TAG_BUTTON in tags
        counts["click_controls"] += names.TAG_CLICK in tags
        counts["toggle_instances"] += names.TAG_TOGGLE in tags
        counts["keys"] += names.TAG_GATE_KEY in tags
        counts["bridges"] += names.TAG_BRIDGE in tags
        counts["crushers"] += names.TAG_CRUSHER in tags
        counts["carried_objects"] += names.TAG_BRIDGE_OBJECT in tags
        counts["sensors"] += "njvd-rolo" in tags
        counts["track_cells"] += names.TAG_FALL_BLOCKER in tags
    support = _support_samples(level)
    bbox = None
    if support:
        bbox = (
            min(x for x, _ in support),
            min(y for _, y in support),
            max(x for x, _ in support),
            max(y for _, y in support),
        )
    return {
        "grid_size": tuple(level.grid_size),
        "sprite_count": len(sprites),
        "support_samples": len(support),
        "support_bbox": bbox,
        **{key: int(counts[key]) for key in (
            "colour_controls", "click_controls", "toggle_instances", "keys",
            "bridges", "crushers", "carried_objects", "sensors", "track_cells",
        )},
    }
