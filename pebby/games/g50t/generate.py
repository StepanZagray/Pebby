"""Full seven-tier procedural G50T generator.

Every accepted row is a new corridor grammar instance, is assigned to a
canonical geometry split, is solved through :mod:`pebby.games.g50t.plan`, and
is replayed at its native level index. Reference ranges are tolerances around
one scarce shipped level per tier, not population confidence intervals.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from numbers import Integral
import random

import numpy as np
from arcengine import Level, Sprite

from . import names
from .env import Env, replay, upstream
from .layout import extract
from .plan import DEFAULT_NODE_LIMIT, search, transition


FORMAT = "pebby.g50t.level.v2"
GENERATOR_VERSION = 4
MECHANICS_VERSION = "g50t-complete-mechanics-v4"
QUALITY_VERSION = "g50t-seven-reference-tiers-v3"
GEOMETRY_VERSION = "g50t-relational-d4-v3"
DIFFICULTIES = tuple(range(1, 8))
DEFAULT_ATTEMPTS = 96
MAX_ATTEMPTS = 256
MAX_GENERATED_PER_EXPANSION = len(names.ACTION_IDS)
# Preserve the reviewed seed schedule so fixed-seed quality comparisons remain
# apples-to-apples.  Generator/mechanics metadata still versions the changed
# grammar and prevents stale rows from passing current validation.
RNG_NAMESPACE = "g50t-full-v3"

# Direct measurements from the seven shipped levels. Witness actions are
# positive native replays found by this teacher; they are not shortest-route
# claims. Action tolerances deliberately remain broad because each tier has one
# reference, so sampling uncertainty cannot be estimated.
_REFERENCE = {
    1: (20, 2, 1, 1, 0, 0, 17, (10, 28)),
    2: (36, 3, 2, 2, 0, 0, 31, (24, 42)),
    3: (36, 3, 3, 3, 0, 0, 64, (50, 78)),
    4: (28, 3, 2, 1, 1, 0, 31, (22, 46)),
    5: (42, 3, 4, 3, 1, 0, 52, (30, 68)),
    6: (44, 3, 5, 5, 0, 1, 49, (32, 66)),
    7: (42, 3, 4, 2, 2, 1, 43, (25, 60)),
}
_REFERENCE_STRUCTURE = {
    # corridor ppm, D4-invariant short/long extent, branch-cell count
    1: (900_000, 6, 8, 2),
    2: (916_667, 8, 8, 3),
    3: (972_222, 8, 8, 1),
    4: (964_286, 8, 8, 1),
    5: (880_952, 9, 10, 5),
    6: (931_818, 9, 10, 3),
    7: (928_571, 9, 10, 3),
}
_STRUCTURE_BOUNDS = {
    # Explicit engineering tolerances around the single shipped reference.
    # They are not confidence intervals.  The generated lattice is at most
    # 9x9, so tiers 5-7 admit the closest native extent (9 rather than 10).
    1: ((700_000, 1_000_000), (4, 9), (7, 9), (1, 8)),
    2: ((730_000, 1_000_000), (7, 9), (8, 9), (2, 12)),
    3: ((850_000, 1_000_000), (8, 9), (8, 9), (1, 5)),
    4: ((780_000, 1_000_000), (7, 9), (7, 9), (1, 8)),
    5: ((760_000, 1_000_000), (8, 9), (9, 9), (3, 10)),
    6: ((780_000, 1_000_000), (8, 9), (9, 9), (3, 10)),
    7: ((800_000, 1_000_000), (8, 9), (9, 9), (2, 8)),
}
_SEARCH_WORK = {1: 100_000, 2: 150_000, 3: 500_000, 4: 150_000,
                5: 200_000, 6: 150_000, 7: 200_000}
PROFILES = {}
for _difficulty, _values in _REFERENCE.items():
    (_free, _slots, _switches, _doors, _teleports, _enemies,
     _actions, _action_range) = _values
    PROFILES[_difficulty] = {
        "reference_level": _difficulty,
        "reference_free_cells": _free,
        "free_cells": (max(14, _free - 12), min(55, _free + 10)),
        "timeline_slots": _slots,
        "switches": _switches,
        "doors": _doors,
        "teleports": _teleports,
        "enemies": _enemies,
        "reference_witness_actions": _actions,
        "witness_actions": _action_range,
        "context_index": _difficulty - 1,
        "search_work": _SEARCH_WORK[_difficulty],
    }
    (_corridor, _short, _long, _branches) = _REFERENCE_STRUCTURE[_difficulty]
    (_corridor_bounds, _short_bounds, _long_bounds,
     _branch_bounds) = _STRUCTURE_BOUNDS[_difficulty]
    PROFILES[_difficulty].update(
        reference_corridor_fraction_ppm=_corridor,
        corridor_fraction_ppm=_corridor_bounds,
        reference_extent_short=_short,
        reference_extent_long=_long,
        extent_short=_short_bounds,
        extent_long=_long_bounds,
        reference_branch_cells=_branches,
        branch_cells=_branch_bounds,
        toggle_detour=(2, 8) if _difficulty == 5 else None,
    )

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": "g50t-5849a774",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_VERSION,
    "curriculum": [
        {"difficulty": difficulty, "context_index": difficulty - 1,
         "search_work": PROFILES[difficulty]["search_work"]}
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "g50t.md#official-reference-characterization",
        "solution_mechanics": "g50t-route-certificate-v2",
        "native_budget": "g50t-native-129-action-countdown",
        "context_engine_replay": "tests/games/test_g50t_quality.py",
        "novelty_split": GEOMETRY_VERSION,
        "bounded_rejections": "g50t.md#bounded-quality-audit",
        "root_acceptance": "external-astra-review1/g50t-final-recheck-detailed.md",
        "rewind_boundary_differential": "native full-world prefixes 35/35, 33/33, 26/26 plus real enemy teleport",
        "three_split_collector": "root 7WIN train/validation/test actions 243/236/253",
        "native_visuals": "root-preserved seven-tier native frame review",
    },
    "caveats": [
        "generation uses a finite authored procedural grammar and does not cover every valid topology",
        "each tier has one shipped reference; ranges are explicit tolerances, not confidence intervals",
        "tier-five and tier-six witnesses remain shorter than the constructive shipped-level witnesses",
        "positive routes are bounded constructive witnesses and make no shortest-route or universal-search claim",
        "structural bounds are engineering tolerances around one shipped level per tier",
        "D4 canonicalization does not claim general graph-isomorphism normalization",
    ],
}

_ROTATION_OF_DELTA = {(0, 1): 0, (-1, 0): 90, (0, -1): 180, (1, 0): 270}
_SPLITS = ("train", "validation", "test")
_OFFICIAL_GEOMETRY_SHA256 = frozenset({
    # Recomputed from the native layouts with GEOMETRY_VERSION's relational,
    # translation-normalized D4 representation.  These are data identities,
    # not generation templates.
    "cc961b80f45e60de0a9c8afc90d8d4b58bc142e1b0c01c73db7c149ca7074bc3",
    "1b97cc6ceb3878ac6380cc6cb43a9d28c2e483685e3606c7bc9576fc8b1dccc4",
    "2027930aa7757af9dbbcfb55e4483755be043db73b041a0b0541936f9abc62d5",
    "8cf0972642bd3bb708af4a1aded08fc882e1622cd209cab4c3a0787b54651f8c",
    "528e273e73fe9e4e579a26d2f37a91dc1ba3184a2747cf7e3e5828038d5f0b87",
    "eaa0e32267d8c8a9f4b7eabf7d427ab3965b9630f57e1c50497d4bf5c1ea56dd",
    "fb150eaee73d3a7a3f34e66b526fffcf95add859aa3cef7d3019d838a8ee58b9",
})


def _integer(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _cell(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-integer cell")
    return _integer(value[0], label), _integer(value[1], label)


def _walk(start, actions):
    cells = {start}
    here = start
    for action in actions:
        dx, dy = names.ACTION_DELTA[action]
        here = here[0] + dx, here[1] + dy
        if not (0 <= here[0] <= 8 and 0 <= here[1] <= 8):
            raise AssertionError(f"template route left the 9x9 lattice at {here}")
        cells.add(here)
    return cells, here


def _door(cell, circuit, delta, *, toggle=False):
    return {"cell": list(cell), "circuit": circuit, "open_delta": list(delta),
            "toggle": bool(toggle)}


def _switch(cell, circuit, kind="ordinary"):
    return {"cell": list(cell), "circuit": circuit, "kind": kind}


def _base_template(difficulty):
    """Return authored mechanic grammar, not an official layout or route."""
    U, D, L, R = (names.ACTION_UP, names.ACTION_DOWN,
                  names.ACTION_LEFT, names.ACTION_RIGHT)
    if difficulty == 1:
        start = (1, 4)
        ghost, ordinary = _walk(start, (U, R))
        final, goal = _walk(start, (R, D, R, R, U, R, R, D, R, R))
        cells = ghost | final
        switches = [_switch(ordinary, 0)]
        doors = [_door((4, 5), 0, (0, 1))]
        teleports = []
        enemies = []
    elif difficulty == 2:
        start = (4, 4)
        route_a, switch_a = _walk(start, (U, U, L, L, D, D, L, L))
        route_b, switch_b = _walk(start, (D, D, L, L, D, D))
        final_actions = (R, U, R, U, R, R, D, D, D, D, L, L)
        final, goal = _walk(start, final_actions)
        cells = route_a | route_b | final
        switches = [_switch(switch_a, 0), _switch(switch_b, 1)]
        doors = [_door((8, 5), 0, (1, 0)), _door((7, 6), 1, (0, 1))]
        teleports = []
        enemies = []
    elif difficulty == 3:
        # A long common stem is traversed in all three timelines, matching the
        # shipped tier's substantially larger action profile without padding:
        # two branches park ordinary-switch ghosts and the third is a gated
        # route whose own toggle opens the final door.
        start = (0, 8)
        stem, junction = _walk(start, (R, R, R, R, R, R, U, U))
        branch_a, switch_a = _walk(junction, (R, R, U, U, U, U))
        branch_b, switch_b = _walk(junction, (L, L, L, L, L, L))
        final, goal = _walk(junction, (
            U, U, L, L, L, L, L, L, U, U, U, U, R, R, R, R, R,
        ))
        cells = stem | branch_a | branch_b | final
        switches = [
            _switch(switch_a, 0), _switch(switch_b, 1),
            _switch((2, 4), 2, "toggle"),
        ]
        doors = [
            _door((6, 5), 0, (1, 0)),
            _door((4, 4), 1, (0, 1)),
            _door((0, 2), 2, (1, 0), toggle=True),
        ]
        teleports = []
        enemies = []
    elif difficulty == 4:
        start = (1, 4)
        route_a, switch_a = _walk(start, (U, R))
        route_b, switch_b = _walk(start, (D, D, R, R, D, D, R, R))
        final, pad_a = _walk(start, (R, R, R, U, R, U, R, R))
        pad_b, goal = (0, 8), (0, 7)
        cells = route_a | route_b | final | {pad_b, goal}
        switches = [_switch(switch_a, 0), _switch(switch_b, 1, "teleport")]
        doors = [_door((4, 4), 0, (0, 1))]
        teleports = [{"pads": [list(pad_a), list(pad_b)], "circuit": 1}]
        enemies = []
    elif difficulty == 5:
        start = (1, 4)
        route_a, switch_a = _walk(start, (U, R))
        route_b, switch_b = _walk(start, (D, D, L, D, D, R, R, R))
        final, pad_a = _walk(start, (R, R, R, D, D, R, R, R))
        pad_b, goal = (8, 0), (8, 1)
        # Both toggle controls are genuine two-step side trips from the
        # door-free approach to their doors.  They are moved among equivalent
        # detour branches by ``_vary_device_relations`` below.
        detour_a, detour_b = (3, 3), (4, 7)
        cells = route_a | route_b | final | {pad_b, goal, detour_a, detour_b}
        switches = [
            _switch(switch_a, 0), _switch(detour_a, 1, "toggle"),
            _switch(detour_b, 2, "toggle"), _switch(switch_b, 3, "teleport"),
        ]
        doors = [
            _door((4, 4), 0, (0, -1)),
            _door((4, 5), 1, (-1, 0), toggle=True),
            _door((5, 6), 2, (0, 1), toggle=True),
        ]
        teleports = [{"pads": [list(pad_a), list(pad_b)], "circuit": 3}]
        enemies = []
    elif difficulty == 6:
        start = (1, 4)
        route_a, switch_a = _walk(start, (U, R))
        route_b, switch_b = _walk(start, (D, R))
        final, goal = _walk(start, (R, R, R, R, R, D, D, R, R, U, U))
        guide = [(x, 8) for x in range(1, 9)]
        cells = route_a | route_b | final | set(guide)
        switches = [
            _switch(switch_a, 0), _switch(switch_b, 1),
            _switch((3, 4), 2, "toggle"), _switch((5, 8), 3, "toggle"),
            _switch((1, 8), 4, "toggle"),
        ]
        doors = [
            # Each ordinary player/ghost circuit gates the enemy before one
            # independently required toggle: 0 -> door 6 -> toggle 3 and
            # 1 -> door 2 -> toggle 4.
            _door((6, 8), 0, (0, -1)), _door((2, 8), 1, (0, -1)),
            _door((6, 4), 2, (0, -1), toggle=True),
            _door((6, 6), 3, (-1, 0), toggle=True),
            _door((8, 5), 4, (-1, 0), toggle=True),
        ]
        teleports = []
        enemies = [{"cell": [8, 8], "path": [list(cell) for cell in guide]}]
    else:
        # The first replay holds the ordinary switch.  On the next timeline
        # the player reaches pad A exactly as the enemy reaches circuit 2;
        # that enemy-operated teleport is the only scheduled route to the
        # toggle and second teleport switch.  The resulting replay reaches
        # circuit 3 exactly as the final player reaches pad C, transporting
        # the player to the goal island.  Both teleport relations therefore
        # participate in the winning schedule rather than being decoration.
        start = (1, 4)
        route_a, switch_a = _walk(start, (U, U, R, R))
        first, pad_a = _walk(start, (R, R, R))
        second, switch_b = _walk((8, 4), (U, U, L, L, D))
        final, pad_c = _walk(start, (D, D, R, R, U, U, R, R))
        guide = [(x, 0) for x in range(1, 8)]
        pad_b = (8, 4)
        # ``_vary_required_routes`` grows a fresh induced goal island out of
        # this teleport destination on every attempt.
        pad_d = goal = (8, 8)
        cells = route_a | first | second | final | {pad_b, pad_d} | set(guide)
        switches = [
            _switch(switch_a, 0), _switch((8, 3), 1, "toggle"),
            _switch((4, 0), 2, "teleport"), _switch(switch_b, 3, "teleport"),
        ]
        doors = [_door((2, 6), 0, (0, 1)),
                 _door((3, 5), 1, (-1, 0), toggle=True)]
        teleports = [
            {"pads": [list(pad_a), list(pad_b)], "circuit": 2},
            {"pads": [list(pad_c), list(pad_d)], "circuit": 3},
        ]
        enemies = [{"cell": [7, 0], "path": [list(cell) for cell in guide]}]
    return {
        "start": list(start), "goal": list(goal), "cells": set(cells),
        "switches": switches, "doors": doors, "teleports": teleports,
        "enemies": enemies,
    }


def _adjacent(cell):
    for dx, dy in names.ACTION_DELTA.values():
        nxt = cell[0] + dx, cell[1] + dy
        if 0 <= nxt[0] <= 8 and 0 <= nxt[1] <= 8:
            yield nxt


def _augment(rng, cells, target, reserved):
    cells = set(cells)
    while len(cells) < target:
        candidates = sorted({nxt for cell in cells for nxt in _adjacent(cell)
                             if nxt not in cells and nxt not in reserved
                             and sum(neighbor in cells for neighbor in _adjacent(nxt)) == 1})
        if not candidates:
            break
        cells.add(rng.choice(candidates))
    return cells


def _grow_required_tail(rng, template, start, steps):
    """Grow an induced corridor and return its last cell.

    Requiring each new cell to have exactly one existing neighbour preserves
    the authored graph relation: the corridor cannot introduce a shortcut to
    another device branch.  Early boundary exhaustion is harmless; the
    structural/profile checks still decide whether the attempt is admissible.
    """
    start = _cell(start, "tail start")
    cells = set(template["cells"])
    tip = start
    for _ in range(steps):
        candidates = [cell for cell in _adjacent(tip) if cell not in cells
                      and sum(neighbor in cells for neighbor in _adjacent(cell)) == 1]
        if not candidates:
            break
        tip = rng.choice(sorted(candidates))
        cells.add(tip)
    template["cells"] = cells
    return tip


def _vary_device_relations(rng, template, difficulty):
    """Vary active switch/gate relations without changing mechanic counts."""
    if difficulty == 5:
        choices = (((3, 3), (2, 5)), ((4, 7), (3, 6)))
        for switch, options in zip(template["switches"][1:3], choices):
            old = _cell(switch["cell"], "toggle switch")
            selected = rng.choice(options)
            if selected != old:
                # These are leaf detours in the authored skeleton, so removing
                # the unused option cannot disconnect any other active device.
                template["cells"].discard(old)
                template["cells"].add(selected)
                switch["cell"] = list(selected)
    elif difficulty == 6:
        first_gate = rng.choice((6, 7))
        second_gate = rng.choice((2, 3))
        template["doors"][0]["cell"] = [first_gate, 8]
        template["switches"][3]["cell"] = [first_gate - 1, 8]
        template["doors"][1]["cell"] = [second_gate, 8]
        template["switches"][4]["cell"] = [1, 8]


def _vary_required_routes(rng, template, difficulty):
    """Vary solution-bearing branches, never only irrelevant free cells."""
    if difficulty == 7:
        template["goal"] = list(_grow_required_tail(
            rng, template, template["goal"], rng.randint(7, 12),
        ))
        return

    start_ranges = {
        1: (0, 2), 2: (0, 3), 3: (4, 6), 4: (0, 3),
        5: (4, 7), 6: (3, 6),
    }
    low, high = start_ranges[difficulty]
    template["start"] = list(_grow_required_tail(
        rng, template, template["start"], rng.randint(low, high),
    ))

    switch_ranges = {
        1: ((0, 2),),
        2: ((0, 2), (0, 2)),
        3: ((0, 2), (0, 2), (0, 0)),
        4: ((0, 2), (0, 2)),
        5: ((0, 0), (0, 0), (0, 0), (0, 0)),
        6: ((1, 3), (1, 3), (0, 1), (0, 0), (0, 0)),
    }
    for switch, (minimum, maximum) in zip(template["switches"],
                                           switch_ranges[difficulty]):
        switch["cell"] = list(_grow_required_tail(
            rng, template, switch["cell"], rng.randint(minimum, maximum),
        ))

    goal_ranges = {1: (0, 2), 2: (0, 3), 3: (4, 7),
                   4: (0, 3), 5: (6, 10), 6: (4, 7)}
    low, high = goal_ranges[difficulty]
    template["goal"] = list(_grow_required_tail(
        rng, template, template["goal"], rng.randint(low, high),
    ))


def _transform_cell(cell, transform):
    x, y = cell
    if transform >= 4:
        x = 8 - x
    for _ in range(transform % 4):
        x, y = 8 - y, x
    return x, y


def _transform_delta(delta, transform):
    x, y = delta
    if transform >= 4:
        x = -x
    for _ in range(transform % 4):
        x, y = -y, x
    return x, y


def canonical_action_identity(actions):
    """Return the lexicographically minimum action tuple over all D4 views."""
    variants = []
    action_of_delta = {delta: action for action, delta in names.ACTION_DELTA.items()}
    for transform in range(8):
        route = []
        for step in actions:
            if (not isinstance(step, (list, tuple)) or len(step) != 3
                    or type(step[0]) is not int or step[0] not in names.ACTION_IDS):
                raise ValueError("action identity requires legal native action triples")
            action = step[0]
            if action == names.ACTION_REWIND:
                route.append(action)
            else:
                route.append(action_of_delta[
                    _transform_delta(names.ACTION_DELTA[action], transform)
                ])
        variants.append(tuple(route))
    return min(variants)


def _transformed(template, transform):
    result = {
        "start": list(_transform_cell(template["start"], transform)),
        "goal": list(_transform_cell(template["goal"], transform)),
        "cells": {_transform_cell(cell, transform) for cell in template["cells"]},
        "switches": [], "doors": [], "teleports": [], "enemies": [],
    }
    for switch in template["switches"]:
        result["switches"].append({**switch,
            "cell": list(_transform_cell(switch["cell"], transform))})
    for door in template["doors"]:
        result["doors"].append({**door,
            "cell": list(_transform_cell(door["cell"], transform)),
            "open_delta": list(_transform_delta(door["open_delta"], transform))})
    for link in template["teleports"]:
        result["teleports"].append({**link,
            "pads": [list(_transform_cell(cell, transform)) for cell in link["pads"]]})
    for enemy in template["enemies"]:
        result["enemies"].append({
            "cell": list(_transform_cell(enemy["cell"], transform)),
            "path": [list(_transform_cell(cell, transform)) for cell in enemy["path"]],
        })
    return result


def _draft(seed, difficulty, attempt):
    rng = random.Random(f"{RNG_NAMESPACE}:{seed}:{difficulty}:{attempt}")
    template = _base_template(difficulty)
    _vary_device_relations(rng, template, difficulty)
    _vary_required_routes(rng, template, difficulty)
    reserved = {_cell(value, "reserved") for value in (template["start"], template["goal"])}
    reserved.update(_cell(s["cell"], "switch") for s in template["switches"])
    reserved.update(_cell(d["cell"], "door") for d in template["doors"])
    reserved.update((_cell(d["cell"], "door")[0] + _cell(d["open_delta"], "open delta")[0],
                     _cell(d["cell"], "door")[1] + _cell(d["open_delta"], "open delta")[1])
                    for d in template["doors"])
    reserved.update(_cell(pad, "pad") for link in template["teleports"] for pad in link["pads"])
    reserved.update(_cell(e["cell"], "enemy") for e in template["enemies"])
    low, high = PROFILES[difficulty]["free_cells"]
    minimum = max(low, len(template["cells"]))
    # Oversized required-route drafts are ordinary bounded rejections, not an
    # exception that escapes the generator and drops its rejection evidence.
    target = len(template["cells"]) if minimum > high else rng.randint(minimum, high)
    template["cells"] = _augment(rng, template["cells"], target, reserved)
    transformed = _transformed(template, rng.randrange(8))
    transformed["cells"] = [list(cell) for cell in sorted(
        transformed["cells"], key=lambda value: (value[1], value[0]))]
    return {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION, "quality_profile_version": QUALITY_VERSION,
        "geometry_version": GEOMETRY_VERSION, "game": "g50t", "seed": seed,
        "source_id": FULL_STANDARD_CONTRACT["source_id"],
        "difficulty": difficulty, "reference_level": difficulty,
        "reference_witness_actions": PROFILES[difficulty]["reference_witness_actions"],
        "origin": [1, 1], "timeline_slots": PROFILES[difficulty]["timeline_slots"],
        "native_action_budget": names.NATIVE_MAX_ACTIONS,
        "reference_calibration": "aggregate per-tier measurements; no official geometry or route copied",
        "topology": "procedural induced-branch corridor grammar",
        **transformed,
    }


def _to_pixel(origin, cell):
    return origin[0] + cell[0] * names.GRID_STEP, origin[1] + cell[1] * names.GRID_STEP


def _endpoint_sprite(points, tag, name, color, *, forbidden_points=()):
    left = min(x for x, _ in points)
    top = min(y for _, y in points)
    right = max(x for x, _ in points) + 1
    bottom = max(max(y for _, y in points), points[0][1] + 1)
    pixels = np.full((bottom - top + 1, right - left + 1), -1, dtype=np.int8)
    # Draw readable one-pixel routed wires. The +1 routing lane avoids every
    # other actor's exact centre, which is the native engine's linkage test;
    # only the intended endpoint pixels can create circuit membership.
    source_x, source_y = points[0]
    for target_x, target_y in points:
        lane_y = source_y + 1
        lane_x = target_x + 1
        for y in range(min(source_y, lane_y), max(source_y, lane_y) + 1):
            pixels[y - top, source_x - left] = color
        for x in range(min(source_x, lane_x), max(source_x, lane_x) + 1):
            pixels[lane_y - top, x - left] = color
        for y in range(min(lane_y, target_y), max(lane_y, target_y) + 1):
            pixels[y - top, lane_x - left] = color
        for x in range(min(lane_x, target_x), max(lane_x, target_x) + 1):
            pixels[target_y - top, x - left] = color
    # Lines are purely visual between their native endpoint pixels.  Erase a
    # crossing at every foreign device centre so a decorative wire cannot
    # silently associate that device with two circuits.
    for point_x, point_y in forbidden_points:
        if left <= point_x <= right and top <= point_y <= bottom:
            pixels[point_y - top, point_x - left] = -1
    return Sprite(pixels=pixels, name=name, visible=True, collidable=False,
                  tags=[tag], layer=0).set_position(left, top)


def _path_sprite(origin, cells):
    pixels = np.full((64, 64), -1, dtype=np.int8)
    for cell in cells:
        x, y = _to_pixel(origin, cell)
        # Movement lattice origins are six pixels apart.  A seven-pixel tile
        # leaks one opaque pixel into the neighbouring lattice cell, silently
        # adding guide branches that are absent from the semantic spec.
        pixels[y:y + names.GRID_STEP, x:x + names.GRID_STEP] = 9
    return Sprite(pixels=pixels, name="pebby-g50t-enemy-guide", visible=True,
                  collidable=False, tags=[names.TAG_ENEMY_PATH], layer=-1)


def build_level(spec):
    """Reconstruct one native level from a strict JSON-compatible v2 spec."""
    errors = structural_errors(spec, require_identity=False)
    if errors:
        raise ValueError("invalid generated G50T level: " + "; ".join(errors))
    origin = _cell(spec["origin"], "origin")
    cells = {_cell(value, "cell") for value in spec["cells"]}
    floor = np.full((63, 64), -1, dtype=np.int8)
    for cell in cells:
        x, y = _to_pixel(origin, cell)
        floor[y:y + 7, x:x + 7] = 5
    boundary = Sprite(pixels=floor, name="pebby-g50t-walkable-graph", visible=True,
                      collidable=True, tags=[names.TAG_BOUNDARY], layer=-1)
    module = upstream()
    prototypes = module.sprites
    start_px = _to_pixel(origin, spec["start"])
    goal_px = _to_pixel(origin, spec["goal"])
    sprites = [
        boundary,
        prototypes[names.SPRITE_GOAL].clone().set_position(goal_px[0] - 1, goal_px[1] - 1),
    ]
    for number in range(spec["timeline_slots"]):
        sprites.append(prototypes[names.SPRITE_CHECKPOINT].clone().set_position(4 * number, 0))

    circuits = {}
    circuit_kinds = {}
    for switch in spec["switches"]:
        cell = _cell(switch["cell"], "switch cell")
        position = _to_pixel(origin, cell)
        sprite = prototypes[names.SPRITE_SWITCH].clone().set_position(*position)
        color = {"ordinary": 8, "toggle": names.TOGGLE_COLOR, "teleport": 15}[switch["kind"]]
        if switch["kind"] != "ordinary":
            sprite.color_remap(None, color)
        sprites.append(sprite)
        circuits.setdefault(switch["circuit"], []).append((position[0] + 3, position[1] + 3))
        circuit_kinds[switch["circuit"]] = switch["kind"]
    for door in spec["doors"]:
        cell = _cell(door["cell"], "door cell")
        position = _to_pixel(origin, cell)
        delta = _cell(door["open_delta"], "door open delta")
        sprite = prototypes[names.SPRITE_DOOR].clone().set_position(*position).set_rotation(
            _ROTATION_OF_DELTA[delta])
        if door["toggle"]:
            sprite.color_remap(None, names.TOGGLE_COLOR)
        sprites.append(sprite)
        circuits.setdefault(door["circuit"], []).append((position[0] + 3, position[1] + 3))
    for number, link in enumerate(spec["teleports"]):
        pad_points = []
        for pad in link["pads"]:
            position = _to_pixel(origin, pad)
            sprites.append(prototypes[names.SPRITE_TELEPORT_PAD].clone().set_position(*position))
            pad_points.append((position[0] + 3, position[1] + 3))
        sprites.append(_endpoint_sprite(pad_points, names.TAG_TELEPORT_LINK,
                                        f"pebby-g50t-teleport-link-{number}", 15))
        circuits.setdefault(link["circuit"], []).extend(pad_points)
    for circuit, points in sorted(circuits.items()):
        color = {"ordinary": 8, "toggle": names.TOGGLE_COLOR,
                 "teleport": 15}[circuit_kinds[circuit]]
        foreign = {point for other, values in circuits.items() if other != circuit
                   for point in values}
        sprites.append(_endpoint_sprite(points, names.TAG_CIRCUIT,
                                        f"pebby-g50t-circuit-{circuit}", color,
                                        forbidden_points=foreign))
    for enemy in spec["enemies"]:
        position = _to_pixel(origin, enemy["cell"])
        sprites.append(_path_sprite(origin, enemy["path"]))
        sprites.append(prototypes[names.SPRITE_ENEMY].clone().set_position(*position))
    sprites.extend([
        prototypes[names.SPRITE_CURSOR].clone(),
        prototypes[names.SPRITE_TIMER].clone().set_position(0, 63),
        prototypes[names.SPRITE_PLAYER].clone().set_position(*start_px),
        prototypes[names.SPRITE_TIMER_BACKGROUND].clone().set_position(0, 63),
    ])
    return Level(sprites=sprites, grid_size=(64, 64),
                 name=f"generated-g50t-d{spec['difficulty']}-s{spec.get('seed', 0)}")


def _normalize_geometry(spec, transform, *, normalize_translation):
    transformed_cells = [_transform_cell(_cell(value, "geometry cell"), transform)
                         for value in spec["cells"]]
    offset_x = min(x for x, _ in transformed_cells) if normalize_translation else 0
    offset_y = min(y for _, y in transformed_cells) if normalize_translation else 0

    def cell(value):
        x, y = _transform_cell(_cell(value, "geometry cell"), transform)
        return x - offset_x, y - offset_y

    delta = lambda value: _transform_delta(_cell(value, "geometry delta"), transform)
    switches = list(spec["switches"])
    doors = list(spec["doors"])
    teleports = list(spec["teleports"])
    circuit_ids = {value["circuit"] for value in switches}
    circuits = []
    for circuit in circuit_ids:
        circuit_switches = [value for value in switches if value["circuit"] == circuit]
        circuit_doors = [value for value in doors if value["circuit"] == circuit]
        circuit_teleports = [value for value in teleports if value["circuit"] == circuit]
        circuits.append({
            "switches": sorted((cell(value["cell"]), value["kind"])
                               for value in circuit_switches),
            "doors": sorted((cell(value["cell"]), value["toggle"],
                             delta(value["open_delta"])) for value in circuit_doors),
            "teleports": sorted(tuple(sorted(cell(pad) for pad in value["pads"]))
                                for value in circuit_teleports),
        })
    return {
        "cells": sorted(cell(value) for value in spec["cells"]),
        "start": cell(spec["start"]), "goal": cell(spec["goal"]),
        "timeline_slots": spec["timeline_slots"],
        # Numeric circuit IDs are private names.  Grouping each switch with its
        # actual outputs binds the playable relation without making a relabel
        # change geometry or split.
        "circuits": sorted(circuits, key=lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"),
        )),
        "enemies": sorted((cell(value["cell"]), tuple(sorted(cell(p) for p in value["path"])))
                          for value in spec["enemies"]),
    }


def geometry_hashes(spec):
    raw_payload = json.dumps(
        _normalize_geometry(spec, 0, normalize_translation=False),
        sort_keys=True, separators=(",", ":"),
    )
    raw_digest = hashlib.sha256(raw_payload.encode()).hexdigest()
    variants = []
    for transform in range(8):
        payload = json.dumps(_normalize_geometry(
            spec, transform, normalize_translation=True,
        ), sort_keys=True,
                             separators=(",", ":"))
        variants.append((payload, transform))
    payload, transform = min(variants)
    d4_digest = hashlib.sha256(payload.encode()).hexdigest()
    return raw_digest, d4_digest, transform


def geometry_identity(spec):
    _, digest, transform = geometry_hashes(spec)
    bucket = int(digest[:8], 16) % 100
    split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
    return digest, split, transform


def device_relation_identity(spec):
    """D4/translation canonical identity of active actor/device relations."""
    variants = []
    for transform in range(8):
        points = [spec["start"], spec["goal"]]
        points += [value["cell"] for value in spec.get("switches", ())]
        points += [value["cell"] for value in spec.get("doors", ())]
        points += [pad for value in spec.get("teleports", ()) for pad in value["pads"]]
        points += [value["cell"] for value in spec.get("enemies", ())]
        points += [cell for value in spec.get("enemies", ()) for cell in value["path"]]
        transformed = [_transform_cell(_cell(value, "device cell"), transform)
                       for value in points]
        offset_x = min(x for x, _ in transformed)
        offset_y = min(y for _, y in transformed)

        def cell(value):
            x, y = _transform_cell(_cell(value, "device cell"), transform)
            return x - offset_x, y - offset_y

        def delta(value):
            return _transform_delta(_cell(value, "device delta"), transform)

        circuits = []
        for circuit in {value["circuit"] for value in spec.get("switches", ())}:
            circuits.append({
                "switches": sorted((cell(value["cell"]), value["kind"])
                                   for value in spec["switches"]
                                   if value["circuit"] == circuit),
                "doors": sorted((cell(value["cell"]), value["toggle"],
                                 delta(value["open_delta"]))
                                for value in spec.get("doors", ())
                                if value["circuit"] == circuit),
                "teleports": sorted(tuple(sorted(cell(pad) for pad in value["pads"]))
                                    for value in spec.get("teleports", ())
                                    if value["circuit"] == circuit),
            })
        payload = {
            "start": cell(spec["start"]), "goal": cell(spec["goal"]),
            "circuits": sorted(circuits, key=lambda value: json.dumps(
                value, sort_keys=True, separators=(",", ":"),
            )),
            "enemies": sorted((cell(value["cell"]),
                               tuple(sorted(cell(path) for path in value["path"])))
                              for value in spec.get("enemies", ())),
        }
        variants.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def gameplay_identity(spec):
    _, geometry, _ = geometry_hashes(spec)
    payload = json.dumps({
        "source_id": FULL_STANDARD_CONTRACT["source_id"],
        "difficulty": spec["difficulty"],
        "native_action_budget": spec["native_action_budget"],
        "geometry_d4_sha256": geometry,
    },
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _layout_geometry_spec(layout, difficulty):
    """Project a native layout into the identity schema used for exclusions."""
    origin_x = min(x for x, _ in layout.allowed)
    origin_y = min(y for _, y in layout.allowed)

    def cell(value):
        x, y = value
        if ((x - origin_x) % names.GRID_STEP
                or (y - origin_y) % names.GRID_STEP):
            raise ValueError("native layout does not use one movement lattice")
        return [(x - origin_x) // names.GRID_STEP,
                (y - origin_y) // names.GRID_STEP]

    teleport_circuits = {link.circuit for link in layout.teleports}
    toggle_circuits = {door.circuit for door in layout.doors if door.toggle}
    return {
        "difficulty": difficulty,
        "native_action_budget": names.NATIVE_MAX_ACTIONS,
        "timeline_slots": layout.stage_count,
        "cells": [cell(value) for value in layout.allowed],
        "start": cell(layout.start), "goal": cell(layout.goal),
        "switches": [
            {"cell": cell(position), "circuit": circuit,
             "kind": ("teleport" if circuit in teleport_circuits else
                      "toggle" if circuit in toggle_circuits else "ordinary")}
            for position, circuit in layout.switches
        ],
        "doors": [
            {"cell": cell(door.closed), "circuit": door.circuit,
             "open_delta": [
                 (door.opened[0] - door.closed[0]) // names.GRID_STEP,
                 (door.opened[1] - door.closed[1]) // names.GRID_STEP,
             ], "toggle": door.toggle}
            for door in layout.doors
        ],
        "teleports": [
            {"pads": [cell(pad) for pad in link.pads], "circuit": link.circuit}
            for link in layout.teleports
        ],
        "enemies": [
            {"cell": cell(enemy.start), "path": [cell(value) for value in enemy.path]}
            for enemy in layout.enemies
        ],
    }


def _canonical_spec_circuits(spec):
    """Assign stable IDs from circuit relations, never private numeric labels."""
    payloads = []
    for circuit in {value["circuit"] for value in spec.get("switches", ())}:
        payload = {
            "switches": sorted((tuple(value["cell"]), value["kind"])
                               for value in spec["switches"]
                               if value["circuit"] == circuit),
            "doors": sorted((tuple(value["cell"]), value["toggle"],
                             tuple(value["open_delta"]))
                            for value in spec.get("doors", ())
                            if value["circuit"] == circuit),
            "teleports": sorted(tuple(sorted(tuple(pad) for pad in value["pads"]))
                                for value in spec.get("teleports", ())
                                if value["circuit"] == circuit),
        }
        payloads.append((json.dumps(payload, sort_keys=True, separators=(",", ":")), circuit))
    return {private: canonical for canonical, (_, private) in enumerate(sorted(payloads))}


def _tier_six_gate_relations(spec):
    """Pair each ordinary enemy-path gate with its nearest blocked toggle."""
    if spec.get("difficulty") != 6 or len(spec.get("enemies", ())) != 1:
        return []
    enemy = spec["enemies"][0]
    canonical = _canonical_spec_circuits(spec)
    start = _cell(enemy["cell"], "enemy start")
    path = {_cell(value, "enemy path") for value in enemy["path"]}
    toggles = {
        _cell(value["cell"], "toggle switch"): canonical[value["circuit"]]
        for value in spec.get("switches", ()) if value.get("kind") == "toggle"
    }

    def distances(origin, blocked=None):
        if origin not in path or origin == blocked:
            return {}
        result = {origin: 0}
        queue = [origin]
        for here in queue:
            for neighbor in _adjacent(here):
                if (neighbor in path and neighbor != blocked
                        and neighbor not in result):
                    result[neighbor] = result[here] + 1
                    queue.append(neighbor)
        return result

    relations = []
    for switch in spec.get("switches", ()):
        if switch.get("kind") != "ordinary":
            continue
        gates = [value for value in spec.get("doors", ())
                 if value.get("circuit") == switch.get("circuit")
                 and not value.get("toggle")]
        if len(gates) != 1:
            continue
        gate = _cell(gates[0]["cell"], "ordinary enemy gate")
        reachable_without_gate = distances(start, gate)
        from_gate = distances(gate)
        candidates = [
            (from_gate[target], target, circuit)
            for target, circuit in toggles.items()
            if target not in reachable_without_gate and target in from_gate
        ]
        if not candidates:
            continue
        _, target, circuit = min(candidates)
        relations.append({
            "ordinary_circuit": canonical[switch["circuit"]],
            "gate_cell": list(gate),
            "toggle_circuit": circuit,
            "toggle_cell": list(target),
        })
    return sorted(relations, key=lambda value: value["ordinary_circuit"])


def structural_metrics(spec):
    cells = {_cell(value, "cell") for value in spec["cells"]}
    degree = Counter(sum(neighbor in cells for neighbor in _adjacent(cell)) for cell in cells)
    corridor_cells = degree[0] + degree[1] + degree[2]
    width = max(x for x, _ in cells) - min(x for x, _ in cells) + 1
    height = max(y for _, y in cells) - min(y for _, y in cells) + 1

    def distances(start):
        result = {start: 0}
        queue = [start]
        for here in queue:
            for neighbor in _adjacent(here):
                if neighbor in cells and neighbor not in result:
                    result[neighbor] = result[here] + 1
                    queue.append(neighbor)
        return result

    start = _cell(spec["start"], "start")
    from_start = distances(start)
    toggle_detours = []
    for switch in spec.get("switches", ()):
        if switch.get("kind") != "toggle":
            continue
        switch_cell = _cell(switch["cell"], "toggle switch")
        from_switch = distances(switch_cell)
        outputs = [_cell(door["cell"], "toggle door")
                   for door in spec.get("doors", ())
                   if door.get("circuit") == switch.get("circuit")]
        if (switch_cell not in from_start or not outputs
                or any(value not in from_start or value not in from_switch
                       for value in outputs)):
            toggle_detours.append(-1)
            continue
        toggle_detours.append(min(
            from_start[switch_cell] + from_switch[door] - from_start[door]
            for door in outputs
        ))
    return {
        "free_cells": len(cells),
        "corridor_cells": corridor_cells,
        "corridor_fraction_ppm": corridor_cells * 1_000_000 // max(1, len(cells)),
        "extent_short": min(width, height),
        "extent_long": max(width, height),
        "branch_cells": degree[3] + degree[4],
        "toggle_detours": toggle_detours,
        "enemy_gate_relations": _tier_six_gate_relations(spec),
    }


def structural_errors(spec, *, require_identity=True):
    errors = []
    if not isinstance(spec, Mapping):
        return ["generated spec must be an object"]
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["difficulty must be an integer in 1..7"]
    profile = PROFILES[difficulty]
    try:
        if spec.get("format") != FORMAT:
            errors.append("format differs from the full generator format")
        if _cell(spec.get("origin"), "origin") != (1, 1):
            errors.append("origin must be [1, 1]")
        cells_list = [_cell(value, "cell") for value in spec.get("cells", ())]
        cells = set(cells_list)
        if not cells or len(cells) != len(cells_list):
            errors.append("cells must be nonempty and unique")
        if any(not (0 <= x <= 8 and 0 <= y <= 8) for x, y in cells):
            errors.append("cells must fit the 9x9 lattice")
        required = [_cell(spec.get("start"), "start"), _cell(spec.get("goal"), "goal")]
        required += [_cell(value["cell"], "switch") for value in spec.get("switches", ())]
        required += [_cell(value["cell"], "door") for value in spec.get("doors", ())]
        required += [_cell(pad, "pad") for value in spec.get("teleports", ()) for pad in value["pads"]]
        required += [_cell(value["cell"], "enemy") for value in spec.get("enemies", ())]
        if any(value not in cells for value in required):
            errors.append("every actor must occupy a walkable cell")
        if (type(spec.get("timeline_slots")) is not int
                or spec.get("timeline_slots") != profile["timeline_slots"]):
            errors.append("timeline slot count differs from the reference tier")
        for key in ("switches", "doors", "teleports", "enemies"):
            if len(spec.get(key, ())) != profile[key]:
                errors.append(f"{key} count differs from the reference tier")
        if (type(spec.get("native_action_budget")) is not int
                or spec.get("native_action_budget") != names.NATIVE_MAX_ACTIONS):
            errors.append("native countdown budget differs from the engine")
        metrics = structural_metrics(spec)
        if not profile["free_cells"][0] <= metrics["free_cells"] <= profile["free_cells"][1]:
            errors.append("free cell count is outside the reference tolerance")
        for key, label in (
            ("corridor_fraction_ppm", "corridor fraction"),
            ("extent_short", "short room extent"),
            ("extent_long", "long room extent"),
            ("branch_cells", "branch-cell count"),
        ):
            low, high = profile[key]
            if not low <= metrics[key] <= high:
                errors.append(f"{label} is outside the reference tolerance")
        detour_bounds = profile["toggle_detour"]
        if detour_bounds is not None and (
                len(metrics["toggle_detours"]) != 2
                or any(not detour_bounds[0] <= value <= detour_bounds[1]
                       for value in metrics["toggle_detours"])):
            errors.append("tier-five toggle controls must be bounded side detours")
        switch_circuit_values = [value["circuit"] for value in spec.get("switches", ())]
        if any(type(value) is not int for value in switch_circuit_values):
            errors.append("switch circuit identifiers must be integers")
        circuits = set(switch_circuit_values)
        if circuits != set(range(len(spec.get("switches", ())))):
            errors.append("switch circuits must be distinct contiguous identifiers")
        circuit_kinds = {value["circuit"]: value.get("kind")
                         for value in spec.get("switches", ())}
        for value in spec.get("switches", ()):
            if value.get("kind") not in ("ordinary", "toggle", "teleport"):
                errors.append("unknown switch kind")
        for value in spec.get("doors", ()):
            delta = _cell(value.get("open_delta"), "door open delta")
            if delta not in _ROTATION_OF_DELTA:
                errors.append("door open delta must be cardinal")
            if value.get("circuit") not in circuits or type(value.get("toggle")) is not bool:
                errors.append("door circuit/toggle metadata is malformed")
            elif ((circuit_kinds.get(value["circuit"]) == "toggle")
                  is not value["toggle"]):
                errors.append("door toggle state must match its switch/wire association")
        for value in spec.get("teleports", ()):
            if len(value.get("pads", ())) != 2 or value.get("circuit") not in circuits:
                errors.append("teleport must have two pads and a declared circuit")
            elif circuit_kinds.get(value["circuit"]) != "teleport":
                errors.append("teleport output must use a teleport-colored circuit")
        for circuit, kind in circuit_kinds.items():
            if kind == "ordinary" and not any(
                value.get("circuit") == circuit and not value.get("toggle")
                for value in spec.get("doors", ())
            ):
                errors.append("ordinary circuit must drive an ordinary door")
            if kind == "toggle" and not any(
                value.get("circuit") == circuit and value.get("toggle")
                for value in spec.get("doors", ())
            ):
                errors.append("toggle circuit must drive a toggle door")
            if kind == "teleport" and not any(
                value.get("circuit") == circuit
                for value in spec.get("teleports", ())
            ):
                errors.append("teleport circuit must drive a paired link")
        enemy_paths = []
        for value in spec.get("enemies", ()):
            start = _cell(value.get("cell"), "enemy cell")
            path_list = [_cell(cell, "enemy path cell")
                         for cell in value.get("path", ())]
            path = set(path_list)
            if (not path or len(path) != len(path_list) or start not in path
                    or any(cell not in cells for cell in path)):
                errors.append("enemy path must be unique, walkable, and contain its start")
            elif path:
                reached = {next(iter(path))}
                frontier = list(reached)
                while frontier:
                    current = frontier.pop()
                    for neighbor in _adjacent(current):
                        if neighbor in path and neighbor not in reached:
                            reached.add(neighbor)
                            frontier.append(neighbor)
                if reached != path:
                    errors.append("enemy guide path must be cardinally connected")
            enemy_paths.append(path)
        if difficulty == 6 and enemy_paths:
            toggle_switches = {
                _cell(value["cell"], "toggle switch"): value["circuit"]
                for value in spec.get("switches", ()) if value.get("kind") == "toggle"
            }
            touched = {circuit for cell, circuit in toggle_switches.items()
                       if any(cell in path for path in enemy_paths)}
            if len(touched) < 2:
                errors.append("tier-six enemy guide must contain two toggle switches")
            relations = metrics["enemy_gate_relations"]
            if (len(relations) != 2
                    or len({value["ordinary_circuit"] for value in relations}) != 2
                    or len({value["toggle_circuit"] for value in relations}) != 2):
                errors.append(
                    "tier-six ordinary circuits must independently gate two enemy toggles"
                )
        if difficulty == 7 and enemy_paths:
            teleport_circuits = {value["circuit"] for value in spec.get("teleports", ())}
            if not any(
                value.get("circuit") in teleport_circuits
                and any(_cell(value["cell"], "teleport switch") in path
                        for path in enemy_paths)
                for value in spec.get("switches", ())
            ):
                errors.append("tier-seven enemy guide must contain a teleport switch")
        if require_identity:
            if not _same_typed_value(spec.get("structure"), metrics):
                errors.append("stored structural evidence differs from recomputation")
            raw_geometry, d4_geometry, _ = geometry_hashes(spec)
            _, split, _ = geometry_identity(spec)
            if spec.get("geometry_sha256") != raw_geometry:
                errors.append("stored raw geometry identity differs from recomputation")
            if spec.get("geometry_d4_sha256") != d4_geometry:
                errors.append("stored D4 geometry identity differs from recomputation")
            if spec.get("geometry_split") != split or spec.get("split") != split:
                errors.append("stored split differs from canonical geometry partition")
            if d4_geometry in _OFFICIAL_GEOMETRY_SHA256:
                errors.append("generated geometry is a canonical copy of an official level")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"malformed generated spec: {exc}")
    return errors


def _world_signature(world):
    return (world.player, world.history,
            tuple((ghost.position, ghost.path) for ghost in world.ghosts),
            tuple((enemy.position, enemy.orientation, enemy.dead, enemy.history)
                  for enemy in world.enemies), world.doors, world.stage, world.dead)


def _canonical_circuit_map(spec, layout):
    """Map native enumeration indices to private, relabel-safe spec IDs."""
    native_at = {position: circuit for position, circuit in layout.switches}
    stable = _canonical_spec_circuits(spec)
    result = {}
    for switch in spec["switches"]:
        position = _to_pixel(_cell(spec["origin"], "origin"), switch["cell"])
        if position not in native_at:
            raise ValueError("native extraction omitted a declared switch")
        native = native_at[position]
        canonical = stable[switch["circuit"]]
        if native in result and result[native] != canonical:
            raise ValueError("one native circuit maps to multiple canonical IDs")
        result[native] = canonical
    if len(result) != len({value["circuit"] for value in spec["switches"]}):
        raise ValueError("native/canonical circuit mapping is not bijective")
    return result


def _route_certificate(spec, actions):
    context = spec["difficulty"] - 1
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context + 1)])
    env.set_level(context)
    layout = extract(env)
    if not layout.exact:
        raise ValueError("generated native layout is outside the complete planner model")
    canonical_circuit = _canonical_circuit_map(spec, layout)
    world = layout.world
    counts = Counter()
    enemy_circuits = set()
    enemy_teleport_triggers = set()
    minimum_steps_left = env.steps_left
    observation = None
    for number, step in enumerate(actions):
        action, x, y = step
        outcome = transition(layout, world, action)
        if outcome is None:
            raise ValueError(f"stored route has an invalid symbolic action at step {number}")
        expected, won, events = outcome
        entered_circuits = set()
        for before, after in zip(world.enemies, expected.enemies):
            if before.position == after.position or after.dead:
                continue
            entered_circuits.update(
                canonical_circuit[circuit] for cell, circuit in layout.switches
                if after.position == cell
            )
        enemy_circuits.update(entered_circuits)
        if "teleport" in events:
            teleport_circuits = {
                canonical_circuit[link.circuit] for link in layout.teleports
            }
            enemy_teleport_triggers.update(entered_circuits & teleport_circuits)
        counts.update(events)
        counts["actions"] += 1
        counts["enemy_steps"] += sum(a.position != b.position
                                     for a, b in zip(world.enemies, expected.enemies))
        counts["toggle_activations"] += sum(
            before != after and door.toggle
            for before, after, door in zip(world.doors, expected.doors, layout.doors))
        counts["ordinary_door_activations"] += sum(
            not before and after and not door.toggle
            for before, after, door in zip(world.doors, expected.doors, layout.doors))
        observation = env.perform(action, x, y)
        minimum_steps_left = min(minimum_steps_left, env.steps_left)
        completed = env.levels_completed > 0 or observation.won
        if not completed:
            actual = extract(env)
            if not actual.exact or _world_signature(actual.world) != _world_signature(expected):
                raise ValueError(f"symbolic/native state disagreement at route step {number}")
        world = expected
        if completed:
            if not won or number != len(actions) - 1:
                raise ValueError("native completion disagrees with the stored route boundary")
    if observation is None or not observation.won or env.levels_completed != 1:
        raise ValueError("stored route does not win at its native context")
    result = dict(counts)
    result.update(won=True, rewinds=counts["rewind"], ghosts_created=counts["ghost_created"],
                  teleports=counts["teleport"], enemy_exercised=counts["enemy_steps"] > 0,
                  enemy_circuit_activations=sorted(enemy_circuits),
                  enemy_triggered_teleport_circuits=sorted(enemy_teleport_triggers))
    return result, minimum_steps_left


def _native_enemy_contacts(spec, actions):
    """Replay one native counterfactual and return win plus canonical contacts."""
    context = spec["difficulty"] - 1
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context + 1)])
    env.set_level(context)
    initial = extract(env)
    if not initial.exact:
        raise ValueError("counterfactual native layout is outside the planner model")
    canonical_circuit = _canonical_circuit_map(spec, initial)
    contacts = set()
    for step in actions:
        observation = env.perform(*step)
        snapshot = extract(env)
        for enemy in snapshot.world.enemies:
            contacts.update(
                canonical_circuit[circuit]
                for position, circuit in snapshot.switches
                if enemy.position == position
            )
        if observation.won or env.levels_completed:
            break
    return env.levels_completed > 0, contacts


def _native_enemy_gate_dependencies(spec, actions, baseline_certificate):
    """Disable each player control in turn and replay the same native route."""
    dependencies = []
    baseline = set(baseline_certificate.get("enemy_circuit_activations", ()))
    canonical = _canonical_spec_circuits(spec)
    for relation in _tier_six_gate_relations(spec):
        altered = deepcopy(spec)
        control = next(
            value for value in altered["switches"]
            if canonical[value["circuit"]] == relation["ordinary_circuit"]
        )
        # The goal is reached only at native completion, so moving this pressure
        # switch there disables the control throughout the attempted route while
        # preserving the native circuit, door, enemy, and toggle implementations.
        control["cell"] = deepcopy(altered["goal"])
        won, contacts = _native_enemy_contacts(altered, actions)
        dependencies.append({
            **relation,
            "probe": "ordinary-switch-at-goal-native-nonwinning-replay",
            "baseline_target_triggered": relation["toggle_circuit"] in baseline,
            "disabled_switch_native_win": won,
            "disabled_switch_target_triggered": relation["toggle_circuit"] in contacts,
        })
    return dependencies


def _mechanic_errors(spec, certificate):
    difficulty = spec["difficulty"]
    errors = []
    required_rewinds = 1 if difficulty == 1 else 2
    if certificate.get("rewinds", 0) < required_rewinds:
        errors.append("winning route does not create the reference number of replay ghosts")
    toggle_count = sum(value["kind"] == "toggle" for value in spec["switches"])
    if certificate.get("toggle_activations", 0) < toggle_count:
        errors.append("winning route does not exercise every toggle circuit")
    if certificate.get("teleports", 0) < len(spec["teleports"]):
        errors.append("winning route does not exercise every teleport link")
    if spec["enemies"] and not certificate.get("enemy_exercised"):
        errors.append("winning route does not advance the moving enemy")
    enemy_circuits = set(certificate.get("enemy_circuit_activations", ()))
    canonical = _canonical_spec_circuits(spec)
    if difficulty == 6:
        toggle_circuits = {
            canonical[value["circuit"]] for value in spec["switches"]
            if value["kind"] == "toggle"
        }
        if len(enemy_circuits & toggle_circuits) < 2:
            errors.append("tier-six enemy does not actuate two toggle circuits")
        dependencies = certificate.get("enemy_gate_dependencies", ())
        if (not isinstance(dependencies, list) or len(dependencies) != 2
                or len({value.get("ordinary_circuit") for value in dependencies}) != 2
                or len({value.get("toggle_circuit") for value in dependencies}) != 2
                or any(value.get("probe") !=
                       "ordinary-switch-at-goal-native-nonwinning-replay"
                       or value.get("baseline_target_triggered") is not True
                       or value.get("disabled_switch_native_win") is not False
                       or value.get("disabled_switch_target_triggered") is not False
                       for value in dependencies)):
            errors.append("tier-six native enemy-gate interventions are incomplete")
    if difficulty == 7:
        teleport_circuits = {
            canonical[value["circuit"]] for value in spec["teleports"]
        }
        triggered = set(certificate.get("enemy_triggered_teleport_circuits", ()))
        if not triggered & teleport_circuits:
            errors.append("tier-seven enemy does not trigger an occupied teleport link")
    if certificate.get("ordinary_door_activations", 0) < sum(
            not value["toggle"] for value in spec["doors"]):
        errors.append("winning route does not activate every ordinary door")
    return errors


def _jsonable(value):
    return json.loads(json.dumps(value, separators=(",", ":")))


def _same_typed_value(left, right):
    """JSON evidence equality that does not equate bool/int or int/float."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (left.keys() == right.keys()
                and all(_same_typed_value(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (len(left) == len(right)
                and all(_same_typed_value(a, b) for a, b in zip(left, right)))
    return left == right


def generate(seed, difficulty=1, attempts=DEFAULT_ATTEMPTS,
             node_limit=DEFAULT_NODE_LIMIT, action_limit=names.NATIVE_MAX_ACTIONS,
             *, split="train", record_rejection=None):
    seed = _integer(seed, "seed", minimum=0)
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts", minimum=1)
    node_limit = _integer(node_limit, "node_limit", minimum=1)
    action_limit = _integer(action_limit, "action_limit", minimum=1)
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be an integer in 1..7")
    if attempts > MAX_ATTEMPTS:
        raise ValueError(f"attempts must not exceed the declared cap {MAX_ATTEMPTS}")
    if split not in _SPLITS:
        raise ValueError("split must be train, validation, or test")
    if record_rejection is not None and not callable(record_rejection):
        raise ValueError("record_rejection must be callable or None")
    rejections = Counter()
    total_expanded = total_generated = 0
    search_limit = min(node_limit, PROFILES[difficulty]["search_work"])

    def reject(reason, attempt, **details):
        rejections[reason] += 1
        if record_rejection is not None:
            record_rejection({
                "seed": seed, "difficulty": difficulty, "attempt": attempt,
                "split": split, "reason": reason,
                "generator_version": GENERATOR_VERSION, **details,
            })

    for attempt in range(attempts):
        spec = _draft(seed, difficulty, attempt)
        raw_geometry, geometry, _ = geometry_hashes(spec)
        _, partition, _ = geometry_identity(spec)
        if partition != split:
            reject("geometry_split", attempt)
            continue
        spec.update(attempt=attempt, requested_seed=seed, effective_seed=seed,
                    split=split, effective_split=split, geometry_split=partition,
                    geometry_sha256=raw_geometry, geometry_d4_sha256=geometry,
                    context_index=difficulty - 1, training_context_index=difficulty - 1,
                    verification_level_index=difficulty - 1, source="generated_only")
        spec["structure"] = structural_metrics(spec)
        spec["device_relation_sha256"] = device_relation_identity(spec)
        if structural_errors(spec):
            reject("profile_structure", attempt)
            continue
        try:
            level = build_level(spec)
            env = Env([level.clone() for _ in range(difficulty)])
            env.set_level(difficulty - 1)
            result = search(env, limit=min(action_limit, names.NATIVE_MAX_ACTIONS),
                            node_limit=search_limit)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            reject(f"native_build:{type(exc).__name__}", attempt)
            continue
        total_expanded += result.expanded
        total_generated += result.generated
        if result.actions is None:
            reason = "search_truncated" if result.truncated else "search_no_witness"
            reject(reason, attempt, search_expanded=result.expanded,
                   search_generated=result.generated)
            continue
        actions = [list(action) for action in result.actions]
        spec["solution"] = actions
        spec["context_solution"] = actions
        spec["solution_length"] = len(actions)
        try:
            mechanics, minimum_steps_left = _route_certificate(spec, actions)
            if difficulty == 6:
                mechanics["enemy_gate_dependencies"] = _native_enemy_gate_dependencies(
                    spec, actions, mechanics,
                )
        except ValueError as exc:
            reject("native_replay_disagreement", attempt)
            continue
        profile = PROFILES[difficulty]
        if not profile["witness_actions"][0] <= len(actions) <= profile["witness_actions"][1]:
            reject("reference_action_tolerance", attempt)
            continue
        mechanic_errors = _mechanic_errors(spec, mechanics)
        if mechanic_errors:
            reason = "mechanic_use:" + mechanic_errors[0]
            reject(reason, attempt)
            continue
        spec.update(
            engine_verified=True, context_engine_verified=True, engine_win=True,
            search_exact=True, search_truncated=False, search_limit=search_limit,
            search_expanded=result.expanded, search_generated=result.generated,
            generation_search_expanded=total_expanded,
            generation_search_generated=total_generated,
            generation_attempts=attempt + 1,
            generation_attempt_limit=attempts,
            levels_completed=1, witness_actions=len(actions), optimality_claim=False,
            minimum_steps_left=minimum_steps_left, native_steps_used=len(actions),
            solution_mechanics=mechanics, generation_exclusions=dict(rejections),
            limitations="positive bounded witness; no shortest-route or exhaustive-negative claim",
            proof={
                "kind": "native-context-route-replay", "difficulty": difficulty,
                "context_index": difficulty - 1, "engine_win": True,
                "search_truncated": False, "search_limit": search_limit,
                "search_expanded": result.expanded,
                "search_generated": result.generated,
                "generation_search_expanded": total_expanded,
                "generation_search_generated": total_generated,
                "generation_attempts": attempt + 1,
                "generation_attempt_limit": attempts,
                "witness_actions": len(actions), "optimality_claim": False,
                "split": split, "geometry_sha256": raw_geometry,
                "geometry_d4_sha256": geometry,
                "device_relation_sha256": spec["device_relation_sha256"],
                "structure": spec["structure"],
            },
        )
        spec["gameplay_sha256"] = gameplay_identity(spec)
        generate.last_rejections = dict(rejections)
        generate.last_work = {
            "attempts": attempt + 1, "expanded": total_expanded,
            "generated": total_generated,
        }
        return _jsonable(spec)
    generate.last_rejections = dict(rejections)
    generate.last_work = {
        "attempts": attempts, "expanded": total_expanded,
        "generated": total_generated,
    }
    return None


generate.last_rejections = {}
generate.last_work = {"attempts": 0, "expanded": 0, "generated": 0}


def _game_level_seed(game_seed, level_index, difficulty):
    seed = _integer(game_seed, "game seed", minimum=0)
    material = f"g50t-5849a774:{seed}:{level_index}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split="train", difficulties=None,
                  attempts=DEFAULT_ATTEMPTS, node_limit=DEFAULT_NODE_LIMIT):
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not selected or any(isinstance(value, bool) or not isinstance(value, Integral)
                           for value in selected):
        raise ValueError("game difficulties must be a nonempty integer sequence")
    selected = tuple(int(value) for value in selected)
    if any(value not in DIFFICULTIES for value in selected):
        raise ValueError("game difficulties must be drawn from 1..7")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("game difficulties must be strictly increasing and distinct")
    specs = []
    for level_index, difficulty in enumerate(selected):
        child_seed = _game_level_seed(seed, level_index, difficulty)
        spec = generate(child_seed, difficulty, attempts=attempts,
                        node_limit=node_limit, split=split)
        if spec is None:
            return None
        spec.update(game_seed=int(seed), game_level_index=level_index)
        specs.append(spec)
    return specs


def build_game(specs):
    if (not isinstance(specs, Sequence) or isinstance(specs, (str, bytes))
            or len(specs) != len(DIFFICULTIES)):
        raise ValueError("G50T build_game requires exactly seven ordered specs")
    levels = []
    split = None
    game_seed = None
    has_game_metadata = [
        isinstance(spec, Mapping) and ("game_seed" in spec or "game_level_index" in spec)
        for spec in specs
    ]
    if any(has_game_metadata) and not all(has_game_metadata):
        raise ValueError("game specs must either all carry parent metadata or all omit it")
    geometries = set()
    gameplays = set()
    for index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if not isinstance(spec, Mapping) or spec.get("difficulty") != difficulty:
            raise ValueError("game specs must use difficulties 1..7 in order")
        if any(spec.get(key) != index for key in
               ("context_index", "training_context_index", "verification_level_index")):
            raise ValueError("game spec has a shifted native context")
        split = spec.get("split") if split is None else split
        if spec.get("split") != split:
            raise ValueError("game specs must use one canonical split")
        if all(has_game_metadata):
            if type(spec.get("game_seed")) is not int or spec["game_seed"] < 0:
                raise ValueError("game specs must bind a nonnegative integer parent seed")
            game_seed = spec["game_seed"] if game_seed is None else game_seed
            if spec["game_seed"] != game_seed or spec.get("game_level_index") != index:
                raise ValueError("game specs must share one parent seed and ordered positions")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"spec {index} fails the full contract: " + "; ".join(errors))
        geometry = spec["geometry_d4_sha256"]
        gameplay = spec["gameplay_sha256"]
        if geometry in geometries or gameplay in gameplays:
            raise ValueError("game specs contain duplicate semantic identities")
        geometries.add(geometry)
        gameplays.add(gameplay)
        levels.append(build_level(spec))
    # Validate the public builder as one uninterrupted native game, including
    # real level-index advancement.  Per-row validation already replays each
    # witness in isolation; this catches sequence/context corruption.
    env = Env(levels)
    for index, spec in enumerate(specs):
        if env.level_index != index or not replay(env, spec["solution"]):
            raise ValueError(f"game spec {index} does not replay in sequence")
        if env.levels_completed != index + 1:
            raise ValueError(f"game spec {index} has incorrect native completion state")
    return levels


def validate_full_standard(spec, curriculum_entry):
    errors = []
    if not isinstance(spec, Mapping):
        return ["generated spec must be an object"]
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be an object"]
    difficulty = curriculum_entry.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty must be an integer in 1..7"]
    profile = PROFILES[difficulty]
    if (type(curriculum_entry.get("context_index")) is not int
            or curriculum_entry.get("context_index") != difficulty - 1):
        errors.append("curriculum context differs from the official tier")
    if (type(curriculum_entry.get("search_work")) is not int
            or curriculum_entry.get("search_work") != profile["search_work"]):
        errors.append("curriculum search work differs from the calibrated cap")
    errors.extend(structural_errors(spec))
    for key, expected in (
        ("generator_version", GENERATOR_VERSION), ("mechanics_version", MECHANICS_VERSION),
        ("quality_profile_version", QUALITY_VERSION), ("geometry_version", GEOMETRY_VERSION),
        ("difficulty", difficulty), ("context_index", difficulty - 1),
        ("training_context_index", difficulty - 1),
        ("verification_level_index", difficulty - 1), ("source", "generated_only"),
        ("engine_verified", True), ("context_engine_verified", True),
        ("engine_win", True), ("search_exact", True), ("search_truncated", False),
        ("optimality_claim", False), ("search_limit", profile["search_work"]),
        ("levels_completed", 1), ("source_id", FULL_STANDARD_CONTRACT["source_id"]),
    ):
        actual = spec.get(key)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{key} is missing or inconsistent")
    solution = spec.get("solution")
    route_length = len(solution) if isinstance(solution, list) else None
    if (not isinstance(solution, list)
            or not _same_typed_value(solution, spec.get("context_solution"))):
        errors.append("solution and contextual solution must exactly match")
    elif any(not isinstance(step, list) or len(step) != 3 or type(step[0]) is not int
             or step[0] not in names.ACTION_IDS or step[1:] != [None, None]
             for step in solution):
        errors.append("solution must contain legal native action triples")
    elif (type(spec.get("solution_length")) is not int
          or type(spec.get("witness_actions")) is not int
          or spec.get("solution_length") != len(solution)
          or spec.get("witness_actions") != len(solution)):
        errors.append("stored witness action counts differ from the route")
    elif not profile["witness_actions"][0] <= len(solution) <= profile["witness_actions"][1]:
        errors.append("witness length is outside the reference tolerance")
    if (type(spec.get("native_action_budget")) is not int
            or spec.get("native_action_budget") != names.NATIVE_MAX_ACTIONS):
        errors.append("native action budget is inconsistent")
    if (type(spec.get("native_steps_used")) is not int
            or route_length is None
            or spec.get("native_steps_used") != route_length):
        errors.append("native_steps_used differs from the replayed route")
    if (type(spec.get("minimum_steps_left")) is not int
            or not 0 <= spec.get("minimum_steps_left", -1) <= names.NATIVE_MAX_ACTIONS):
        errors.append("minimum_steps_left must be an in-budget integer")
    for key in ("search_expanded", "search_generated",
                "generation_search_expanded", "generation_search_generated"):
        value = spec.get(key)
        if type(value) is not int or value < 0:
            errors.append(f"{key} must be a nonnegative integer")
    generation_attempt_limit = spec.get("generation_attempt_limit")
    if (type(generation_attempt_limit) is not int
            or not 1 <= generation_attempt_limit <= MAX_ATTEMPTS):
        errors.append("generation_attempt_limit must respect the declared cap")
    generation_attempts = spec.get("generation_attempts")
    if (type(generation_attempts) is not int
            or type(generation_attempt_limit) is not int
            or not 1 <= generation_attempts <= generation_attempt_limit):
        errors.append("generation_attempts must respect the declared attempt cap")
    if (type(spec.get("attempt")) is not int or spec.get("attempt", -1) < 0
            or spec.get("generation_attempts") != spec.get("attempt", -1) + 1):
        errors.append("generation attempt evidence is inconsistent")
    if (type(spec.get("generation_search_expanded")) is int
            and type(spec.get("search_expanded")) is int
            and spec["generation_search_expanded"] < spec["search_expanded"]):
        errors.append("cumulative expanded work is smaller than accepted-attempt work")
    if (type(spec.get("generation_search_generated")) is int
            and type(spec.get("search_generated")) is int
            and spec["generation_search_generated"] < spec["search_generated"]):
        errors.append("cumulative generated work is smaller than accepted-attempt work")
    if (type(spec.get("search_expanded")) is int
            and type(spec.get("search_limit")) is int
            and spec["search_expanded"] > spec["search_limit"]):
        errors.append("accepted-attempt expanded work exceeds its search cap")
    if (type(spec.get("search_generated")) is int
            and type(spec.get("search_expanded")) is int
            and spec["search_generated"] >
            MAX_GENERATED_PER_EXPANSION * spec["search_expanded"]):
        errors.append("accepted-attempt generated work exceeds its branching bound")
    if (type(spec.get("generation_search_expanded")) is int
            and type(spec.get("generation_attempts")) is int
            and type(spec.get("search_limit")) is int
            and spec["generation_search_expanded"] >
            spec["generation_attempts"] * spec["search_limit"]):
        errors.append("cumulative expanded work exceeds the attempt/search caps")
    if (type(spec.get("generation_search_generated")) is int
            and type(spec.get("generation_search_expanded")) is int
            and spec["generation_search_generated"] >
            MAX_GENERATED_PER_EXPANSION * spec["generation_search_expanded"]):
        errors.append("cumulative generated work exceeds its branching bound")
    exclusions = spec.get("generation_exclusions")
    if not isinstance(exclusions, dict) or any(type(key) is not str or type(value) is not int or value < 0
                                                for key, value in (exclusions.items() if isinstance(exclusions, dict) else ())):
        errors.append("bounded rejection counters are missing or malformed")
    elif type(spec.get("attempt")) is int and sum(exclusions.values()) != spec["attempt"]:
        errors.append("bounded rejection counters do not cover every rejected attempt")
    try:
        if solution is not None:
            mechanics, minimum_steps_left = _route_certificate(spec, solution)
            if difficulty == 6:
                mechanics["enemy_gate_dependencies"] = _native_enemy_gate_dependencies(
                    spec, solution, mechanics,
                )
            if not _same_typed_value(mechanics, spec.get("solution_mechanics")):
                errors.append("solution mechanic evidence differs from native replay")
            errors.extend(_mechanic_errors(spec, mechanics))
            if minimum_steps_left != spec.get("minimum_steps_left"):
                errors.append("native budget evidence differs from replay")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"stored route could not be replayed: {type(exc).__name__}: {exc}")
    try:
        gameplay = gameplay_identity(spec)
        if spec.get("gameplay_sha256") != gameplay:
            errors.append("stored gameplay identity differs from recomputation")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"gameplay identity could not be recomputed: {exc}")
    try:
        relation = device_relation_identity(spec)
        if type(spec.get("device_relation_sha256")) is not str or spec.get(
                "device_relation_sha256") != relation:
            errors.append("stored device-relation identity differs from recomputation")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"device-relation identity could not be recomputed: {exc}")
    proof = spec.get("proof")
    if not isinstance(proof, dict):
        errors.append("nested proof is missing")
    else:
        mirrors = {
            "kind": "native-context-route-replay",
            "difficulty": difficulty, "context_index": difficulty - 1,
            "engine_win": True, "search_truncated": False,
            "search_limit": profile["search_work"],
            "search_expanded": spec.get("search_expanded"),
            "search_generated": spec.get("search_generated"),
            "generation_search_expanded": spec.get("generation_search_expanded"),
            "generation_search_generated": spec.get("generation_search_generated"),
            "generation_attempts": spec.get("generation_attempts"),
            "generation_attempt_limit": spec.get("generation_attempt_limit"),
            "witness_actions": spec.get("witness_actions"),
            "optimality_claim": False, "split": spec.get("split"),
            "geometry_sha256": spec.get("geometry_sha256"),
            "geometry_d4_sha256": spec.get("geometry_d4_sha256"),
            "device_relation_sha256": spec.get("device_relation_sha256"),
            "structure": spec.get("structure"),
        }
        for key, expected in mirrors.items():
            actual = proof.get(key)
            if not _same_typed_value(actual, expected):
                errors.append(f"proof.{key} does not mirror the top-level certificate")
    return errors
