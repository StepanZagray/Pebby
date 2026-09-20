"""Scarce-reference profiles for the eight shipped S5I5 levels.

There is one official level per tier.  These measurements are therefore
reference points with explicit engineering tolerances, not population
confidence intervals.  The official layouts and routes are never generator
inputs.

Source locations in ``third_party/arc3_games/s5i5.py``:

* levels and native budgets: lines 1728-1944;
* pin attachment and explicit ``Children`` graphs: lines 2051-2072;
* recursive translation/resize/rotation: lines 2086-2161;
* collision rollback and shared-colour control dispatch: lines 2163-2240;
* win-before-budget-exhaustion ordering: lines 2241-2245.

Official shortest paths for tiers 1-6 were measured with the native-snapshot
BFS in :mod:`pebby.games.s5i5.plan` and replayed in the native engine.  A
400,000-state native pass timed out while expanding tier 7, so tiers 7-8
deliberately have no claimed official optimum.  Bounded guided searches later
found 45- and 38-action positive witnesses respectively; both were replayed on
fresh native clones. Their generated action bands are bounded constructive-
witness targets, clearly labelled below.
"""

DIFFICULTY_VERSION = "s5i5-official-reference-v1"
MECHANICS_INVENTORY_VERSION = "s5i5-full-mechanics-v1"
DIFFICULTIES = tuple(range(1, 9))


# Counts describe mechanically moving rods (controlled roots plus recursive
# descendants), directly controlled rods, immobile obstacle sprites, connected
# component sizes, total explicit rod-to-rod edges, maximum branch factor,
# maximum graph depth, pins/targets, rails and buttons. ``rod_units`` is the
# sum of initial three-pixel length units over mechanically moving rods.
REFERENCE_CHARACTERIZATION = {
    1: dict(budget=50, mechanical_rods=2, controlled_rods=2, obstacles=2,
            component_sizes=(1, 1), edges=0, branch=0, depth=1,
            pins=2, targets=2, rails=2, buttons=0,
            rail_orientations={"horizontal": 1, "vertical": 1},
            shared_color_groups=(), rod_units=5, obstacle_pixels=18,
            arena_obstacle_pixels=0,
            occupied_rod_pixels=63, official_optimal_actions=13,
            official_action_evidence="exact exhaustive native-snapshot BFS"),
    2: dict(budget=150, mechanical_rods=4, controlled_rods=4, obstacles=2,
            component_sizes=(4,), edges=3, branch=1, depth=4,
            pins=1, targets=1, rails=4, buttons=0,
            rail_orientations={"horizontal": 4}, shared_color_groups=(),
            rod_units=4, obstacle_pixels=744, arena_obstacle_pixels=195,
            occupied_rod_pixels=780,
            official_optimal_actions=26,
            official_action_evidence="exact exhaustive native-snapshot BFS"),
    3: dict(budget=200, mechanical_rods=8, controlled_rods=6, obstacles=2,
            component_sizes=(1, 1, 3, 3), edges=4, branch=1, depth=3,
            pins=2, targets=2, rails=6, buttons=0,
            rail_orientations={"horizontal": 6}, shared_color_groups=(),
            rod_units=31, obstacle_pixels=813, arena_obstacle_pixels=9,
            occupied_rod_pixels=1092,
            official_optimal_actions=37,
            official_action_evidence="exact exhaustive native-snapshot BFS"),
    4: dict(budget=100, mechanical_rods=9, controlled_rods=9, obstacles=1,
            component_sizes=(1, 2, 2, 2, 2), edges=4, branch=1, depth=2,
            pins=1, targets=1, rails=5, buttons=0,
            rail_orientations={"horizontal": 5}, shared_color_groups=(5,),
            rod_units=17, obstacle_pixels=393, arena_obstacle_pixels=0,
            occupied_rod_pixels=546,
            official_optimal_actions=30,
            official_action_evidence="exact exhaustive native-snapshot BFS"),
    5: dict(budget=150, mechanical_rods=7, controlled_rods=6, obstacles=3,
            component_sizes=(1, 1, 2, 3), edges=3, branch=1, depth=3,
            pins=2, targets=2, rails=4, buttons=0,
            rail_orientations={"horizontal": 4}, shared_color_groups=(2, 2),
            rod_units=24, obstacle_pixels=858, arena_obstacle_pixels=54,
            occupied_rod_pixels=1074,
            official_optimal_actions=28,
            official_action_evidence="exact exhaustive native-snapshot BFS"),
    6: dict(budget=150, mechanical_rods=3, controlled_rods=3, obstacles=1,
            component_sizes=(3,), edges=2, branch=1, depth=3,
            pins=1, targets=1, rails=3, buttons=3,
            rail_orientations={"horizontal": 3}, shared_color_groups=(),
            rod_units=7, obstacle_pixels=252, arena_obstacle_pixels=234,
            occupied_rod_pixels=315,
            official_optimal_actions=25,
            official_action_evidence="exact exhaustive native-snapshot BFS"),
    7: dict(budget=200, mechanical_rods=6, controlled_rods=6, obstacles=1,
            component_sizes=(1, 1, 4), edges=3, branch=1, depth=4,
            pins=2, targets=2, rails=5, buttons=4,
            rail_orientations={"horizontal": 5}, shared_color_groups=(),
            rod_units=16, obstacle_pixels=708, arena_obstacle_pixels=216,
            occupied_rod_pixels=708,
            official_optimal_actions=None, official_witness_actions=45,
            official_action_evidence="guided exact-transition witness replayed natively; no optimum claimed"),
    8: dict(budget=200, mechanical_rods=12, controlled_rods=10, obstacles=2,
            component_sizes=(1, 3, 3, 5), edges=8, branch=4, depth=3,
            pins=1, targets=1, rails=6, buttons=1,
            rail_orientations={"horizontal": 6}, shared_color_groups=(2, 2, 2),
            rod_units=31, obstacle_pixels=417, arena_obstacle_pixels=162,
            occupied_rod_pixels=774,
            official_optimal_actions=None, official_witness_actions=38,
            official_action_evidence="guided exact-transition witness replayed natively; no optimum claimed"),
}


# Explicit tolerances around the scarce references.  Tiers 7-8 are generated
# with bounded, loop-free constructive witnesses and make no optimality claim.
PROFILES = {
    1: dict(action_range=(8, 18), rod_units=(4, 7), obstacle_cells=(2, 5),
            occupied_density=(0.008, 0.050), required_kinds=("extend", "retract"),
            require_vertical=True, distinct_controls=2, shortcut_bound=8,
            search_work=400_000),
    2: dict(action_range=(18, 38), rod_units=(4, 8), obstacle_cells=(66, 92),
            occupied_density=(0.140, 0.250), required_kinds=("extend",),
            require_linked=True, distinct_controls=4, shortcut_bound=12,
            search_work=400_000),
    3: dict(action_range=(27, 48), rod_units=(22, 35),
            interior_obstacle_cells=(1, 1), obstacle_pixels=(813, 813),
            arena_obstacle_pixels=(9, 9), required_kinds=("extend", "retract"),
            require_linked=True, distinct_controls=4, shortcut_bound=12,
            search_work=600_000),
    4: dict(action_range=(19, 40), rod_units=(12, 22),
            interior_obstacle_cells=(0, 0), obstacle_pixels=(393, 393),
            arena_obstacle_pixels=(0, 0), required_kinds=("extend", "retract"),
            require_linked=True, require_shared=True, require_constraint=True,
            constraint_controls=1, constraint_events=3, distinct_controls=3,
            shortcut_bound=8,
            search_work=600_000),
    5: dict(action_range=(19, 38), rod_units=(17, 29),
            interior_obstacle_cells=(6, 6), obstacle_pixels=(858, 858),
            arena_obstacle_pixels=(54, 54), required_kinds=("extend", "retract"),
            require_linked=True, require_shared=True, distinct_controls=3,
            shortcut_bound=12,
            search_work=800_000),
    6: dict(action_range=(17, 35), rod_units=(6, 10), obstacle_cells=(22, 35),
            occupied_density=(0.050, 0.130), required_kinds=("extend", "retract", "rotate"),
            require_linked=True, distinct_controls=4, shortcut_bound=14,
            search_work=800_000),
    7: dict(action_range=(20, 55), rod_units=(12, 22), obstacle_cells=(62, 88),
            occupied_density=(0.150, 0.300), required_kinds=("extend", "retract", "rotate"),
            require_linked=True, distinct_controls=5, shortcut_bound=14,
            search_work=1_000_000),
    8: dict(action_range=(24, 65), rod_units=(23, 36), obstacle_cells=(36, 56),
            occupied_density=(0.080, 0.210), required_kinds=("extend", "retract", "rotate"),
            require_linked=True, require_shared=True, require_branch=True,
            require_constraint=True, constraint_controls=3, constraint_events=3,
            distinct_controls=5, shortcut_bound=10, search_work=1_000_000),
}


SUPPORTED_MECHANICS = (
    "ACTION6 display-coordinate clicks and native per-level budgets",
    "horizontal and vertical rail extension/retraction",
    "rotation buttons and the child three-quarter-turn exception",
    "shared-colour controls acting on multiple rods",
    "implicit pin attachment and multiple pin/target completion",
    "recursive chains through depth four and four-way branching",
    "internal obstacle collision with next-frame rollback",
    "native sequential advancement through all eight contexts",
)
