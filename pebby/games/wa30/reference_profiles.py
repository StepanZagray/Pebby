"""Measured WA30 reference tiers and explicit one-reference tolerances.

Each row comes from one shipped level in ``third_party/arc3_games/wa30.py``.
The ranges are engineering tolerances around scarce references, not population
confidence intervals. ``reference_witness_actions`` records a replayed positive
witness found by the exact transition teacher; it is not an optimality claim.
"""

DIFFICULTY_VERSION = "wa30-official-reference-v1"
DIFFICULTIES = tuple(range(1, 10))

# Exact source measurements: boxes, helpers, thieves, tagged wall cells,
# fences, goal cells, bad cells, native budget, positive witness actions.
_MEASURED = (
    (3, 0, 0, 0, 0, 3, 0, 200, 26),
    (5, 1, 0, 0, 0, 6, 0, 70, 65),
    (5, 1, 0, 0, 16, 8, 0, 100, 99),
    (7, 3, 0, 13, 22, 7, 0, 100, 84),
    (6, 1, 0, 20, 0, 8, 0, 125, 108),
    (2, 0, 1, 30, 0, 4, 4, 75, 48),
    (2, 0, 1, 34, 0, 2, 2, 125, 36),
    (13, 2, 2, 28, 0, 17, 18, 150, 106),
    (9, 2, 1, 50, 6, 13, 4, 70, 65),
)
REFERENCE_MEASUREMENTS = {
    difficulty: {
        key: value
        for key, value in zip(
            ("boxes", "helpers", "thieves", "walls", "fences", "goal_cells", "bad_cells", "budget", "witness_actions"),
            measured,
        )
    }
    for difficulty, measured in enumerate(_MEASURED, 1)
}

# Action intervals deliberately include constructive procedural variants while
# staying centered on the measured witness and below the native budget.
_ACTION_RANGES = (
    (18, 40), (40, 70), (60, 100), (45, 100), (40, 125),
    (25, 75), (20, 65), (65, 150), (40, 70),
)
_SEARCH_LIMITS = (600_000, 300_000, 600_000, 1_000_000, 2_200_000,
                  750_000, 750_000, 2_200_000, 2_000_000)


def _count_range(value, *, floor=0, tolerance=0):
    return max(floor, value - tolerance), value + tolerance


PROFILES = {}
for difficulty, measured in enumerate(_MEASURED, 1):
    boxes, helpers, thieves, walls, fences, goals, bad, budget, witness = measured
    PROFILES[difficulty] = {
        "reference_level": difficulty,
        "reference_witness_actions": witness,
        "actions": _ACTION_RANGES[difficulty - 1],
        "boxes": (boxes, boxes),
        "helpers": (helpers, helpers),
        "thieves": (thieves, thieves),
        "walls": (walls, walls),
        "fences": (fences, fences),
        "goal_cells": (goals, goals),
        "bad_cells": (bad, bad),
        "budget": budget,
        "context_index": difficulty - 1,
        "search_limit": _SEARCH_LIMITS[difficulty - 1],
    }

# Defining native-origin topology measured from the shipped tiers.  These are
# relational constraints, not coordinate templates: procedural drafts may
# translate/rotate their chambers and corridors while retaining the causal
# separation or confinement that makes the mechanic meaningful.
PROFILES[3]["topology"] = {
    "ranges": {
        "player_component_cells": (96, 144),
        "player_component_goal_cells": (0, 0),
        "helpers_in_player_component": (0, 0),
        "boxes_in_player_component": (3, 3),
        "boxes_on_fences": (2, 2),
    },
    "minimums": {"fence_removed_component_gain": 96},
}
PROFILES[4]["topology"] = {
    "ranges": {
        "player_component_cells": (20, 20),
        "player_component_goal_cells": (0, 0),
        "helpers_in_player_component": (0, 0),
        "boxes_in_player_component": (6, 6),
        "boxes_on_fences": (0, 0),
    },
    "minimums": {"fence_removed_component_gain": 180},
}
PROFILES[7]["topology"] = {
    "ranges": {
        "player_component_cells": (76, 80),
        "player_component_goal_cells": (2, 2),
        "thieves_in_player_component": (1, 1),
    },
    "minimums": {"wall_removed_component_gain": 170},
}
for difficulty in (5, 6, 8, 9):
    PROFILES[difficulty].setdefault("topology", {}).setdefault("minimums", {})["wall_route_detour"] = 1


MECHANIC_INVENTORY = {
    "movement_and_facing": {"source": "wa30.py:987-992,1203-1226", "tiers": DIFFICULTIES},
    "manual_grab_drag_drop": {"source": "wa30.py:996-1001,1227-1236", "tiers": DIFFICULTIES},
    "native_action_budget": {"source": "wa30.py:772-805,1203-1248", "tiers": DIFFICULTIES},
    "goal_completion_unheld": {"source": "wa30.py:1003-1005,1194-1201", "tiers": DIFFICULTIES},
    "helper_target_cache_and_bfs": {"source": "wa30.py:934-948,1009-1054,1142-1163", "tiers": (2, 3, 4, 5, 8, 9)},
    "fence_actor_block_box_entry": {"source": "wa30.py:916-927,993-1001", "tiers": (3, 4, 9)},
    "walls_and_frame_border": {"source": "wa30.py:904-914,993-1001", "tiers": (4, 5, 6, 7, 8, 9)},
    "thief_target_cache_and_bad_delivery": {"source": "wa30.py:949-963,1056-1101,1165-1192", "tiers": (6, 7, 8, 9)},
    "holder_steal_and_player_thief_destruction": {"source": "wa30.py:1103-1118,1227-1242", "tiers": (6, 7, 8, 9)},
    "ordered_helper_then_thief_phase": {"source": "wa30.py:1198-1201", "tiers": (8, 9)},
    "state_recolour_cues": {"source": "wa30.py:1119-1140", "tiers": DIFFICULTIES},
}


def in_range(value, bounds):
    return bounds[0] <= value <= bounds[1]
