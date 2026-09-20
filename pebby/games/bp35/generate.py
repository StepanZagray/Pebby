"""Full nine-tier BP35 generation with native-engine route certificates."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from numbers import Integral
import random

from arcengine import GameState, Level
import numpy as np

from pebby.multigame import source_for

from . import names
from .env import UPSTREAM, Env, replay, upstream


SOURCE_SHA256 = "e9aecb52c629c3e742276c2db04b81f91555051d8617b6cbc58bac410824867f"
if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("vendored BP35 source bytes do not match the calibrated generator")
SOURCE_ID = source_for("bp35").source_id
FORMAT = "pebby.bp35.level.v3"
GENERATOR_VERSION = 3
MECHANICS_VERSION = "bp35-full-mechanics-v2"
QUALITY_VERSION = "bp35-reference-nine-tier-v2"
GEOMETRY_VERSION = "bp35-d4-split-v2"
DIFFICULTIES = tuple(range(1, 10))
DEFAULT_ATTEMPTS = 96
DEFAULT_LIMIT = 1_000_000
MAX_ATTEMPTS = 256
SPLITS = ("train", "validation", "test")
SEARCH_WORK = (20_000, 40_000, 80_000, 120_000, 180_000, 300_000, 500_000, 700_000, 1_000_000)

PROFILES = {
    1: {"grammar": "destructible_descent", "layers": 3, "hazards": True, "spikes": ()},
    2: {"grammar": "destructible_descent", "layers": 7, "hazards": True, "spikes": ("v",)},
    3: {"grammar": "bridge_build_descent", "layers": 3, "hazards": True, "spikes": ("v",)},
    4: {"grammar": "gravity_lattice", "layers": 3, "passes": 3, "gate": "x", "spikes": ("v",)},
    5: {"grammar": "gravity_lattice", "layers": 5, "passes": 3, "gate": "mixed_x_bridge", "spikes": ("v", "u")},
    6: {"grammar": "gravity_lattice", "layers": 8, "passes": 3, "gate": "1", "spikes": ("v", "u")},
    7: {"grammar": "gravity_lattice", "layers": 10, "passes": 3, "gate": "1", "spikes": ("v", "u")},
    8: {"grammar": "growth_bridge", "growth": True, "spikes": ("v",), "gravity": True, "bridge": True},
    9: {"grammar": "growth_bridge", "growth": True, "spikes": ("v", "u"), "gravity": True, "bridge": True, "destructible": True},
}

# These route-topology families were bounded-native-replayed while the compact
# lattice grammar was calibrated. Random generation combines them with
# independently varied supports; every accepted instance is replayed again.
GRAVITY_PASS_PATTERNS = {
    4: ((2, 4, 8), (6, 2, 8), (2, 7, 3), (3, 8, 7), (6, 8, 3),
        (8, 7, 3), (3, 8, 2), (2, 4, 7), (4, 8, 3), (2, 8, 4), (4, 2, 8),
        (3, 7, 2), (6, 3, 7), (7, 2, 8), (8, 3, 2), (4, 7, 3), (3, 4, 8)),
    5: ((3, 8, 4), (2, 8, 4), (7, 6, 3), (7, 6, 2), (2, 3, 7),
        (8, 2, 6), (2, 4, 3), (6, 8, 3), (4, 7, 2), (6, 3, 7),
        (4, 2, 8), (6, 2, 8), (2, 7, 3), (2, 4, 8),
        (7, 2, 8), (7, 3, 8), (2, 3, 8)),
    6: ((4, 2, 8),),
    7: ((6, 2, 8),),
}
GRAVITY_SUPPORT_PATTERNS = {
    4: ((23, 15, 7), (24, 16, 7), (23, 14, 6), (24, 15, 6)),
    5: ((26, 21, 16, 11, 6), (27, 22, 17, 12, 7),
        (25, 20, 15, 10, 5), (26, 20, 15, 10, 5)),
    6: ((26, 23, 20, 17, 14, 11, 8, 5), (27, 24, 21, 18, 15, 12, 9, 5),
        (27, 24, 21, 18, 15, 12, 8, 5),
        (26, 23, 20, 17, 14, 10, 8, 5),
        (27, 24, 21, 18, 14, 11, 8, 5),
        (26, 23, 20, 16, 13, 10, 8, 5),
        (27, 24, 20, 17, 14, 11, 8, 5),
        (26, 22, 19, 16, 13, 10, 8, 5),
        (26, 24, 22, 18, 13, 11, 8, 5)),
    7: ((27, 25, 23, 21, 19, 17, 15, 13, 10, 5),
        (27, 24, 22, 20, 18, 16, 14, 12, 9, 5),
        (26, 24, 22, 20, 18, 16, 14, 12, 9, 5),
        (27, 25, 23, 21, 19, 17, 15, 12, 9, 5),
        (27, 24, 21, 19, 17, 15, 13, 11, 9, 5),
        (26, 23, 21, 19, 17, 15, 13, 11, 9, 5),
        (27, 25, 22, 20, 18, 16, 14, 12, 9, 5),
        (26, 24, 21, 19, 17, 15, 13, 11, 9, 5),
        (27, 25, 23, 19, 17, 15, 13, 10, 8, 5)),
}
GROWTH_COLUMN_PATTERNS = {
    8: ((3, 8, 2), (3, 7, 2), (4, 8, 2), (4, 7, 2),
        (6, 2, 8), (7, 2, 8), (6, 3, 8), (7, 3, 8),
        (3, 8, 4), (7, 2, 6), (4, 8, 3), (6, 2, 7),
        (3, 8, 6), (4, 6, 2), (4, 8, 7), (5, 7, 3), (6, 3, 2), (6, 8, 4), (7, 3, 6),
        (4, 5, 2), (4, 6, 3), (4, 7, 8), (5, 6, 2), (5, 8, 2)),
    9: ((3, 8, 2), (4, 8, 2), (6, 2, 8), (7, 2, 8),
        (3, 8, 4), (7, 2, 6), (4, 8, 3), (6, 2, 7),
        (5, 2, 7), (4, 2, 8), (5, 8, 3)),
}
MIN_ACTIONS = (10, 16, 18, 18, 24, 28, 34, 45, 55)
DESCENT_SUPPORT_PATTERNS = {
    1: ((20, 13, 6), (20, 12, 5), (19, 12, 5),
        (21, 14, 7), (21, 13, 6), (19, 11, 4)),
    2: ((38, 33, 28, 23, 18, 13, 8),),
    3: ((27, 18, 9),),
}

# Close-reference ranges are intentionally explicit.  They are derived from
# the nine native maps but leave room for procedural relation layouts rather
# than copying a shipped puzzle.  Counts describe initial interactive cells.
STRUCTURAL_RANGES = {
    1: {"height": (28, 34), "destructibles": (18, 28), "moving_hazard_bands": (1, 1)},
    2: {"height": (42, 50), "destructibles": (42, 56), "down_spikes": (1, 7), "moving_hazard_bands": (1, 1)},
    3: {"height": (33, 41), "solid_bridges": (7, 12), "open_bridges": (6, 12), "destructibles": (2, 6), "moving_hazard_bands": (1, 1)},
    4: {"height": (28, 36), "destructibles": (8, 18), "gravity_switches": (3, 6)},
    5: {"height": (28, 36), "destructibles": (8, 16), "open_bridges": (4, 6), "gravity_switches": (2, 6)},
    6: {"height": (29, 37), "solid_bridges": (15, 17), "open_bridges": (8, 10), "down_spikes": (5, 5), "up_spikes": (3, 3), "gravity_switches": (6, 6)},
    7: {"height": (27, 35), "solid_bridges": (19, 21), "open_bridges": (13, 15), "down_spikes": (7, 7), "up_spikes": (4, 4), "gravity_switches": (10, 10)},
    8: {"height": (31, 39), "growth_seeds": (2, 4), "solid_bridges": (5, 12), "open_bridges": (2, 10), "down_spikes": (6, 14), "gravity_switches": (1, 3)},
    9: {"height": (38, 46), "growth_seeds": (2, 4), "solid_bridges": (5, 12), "open_bridges": (2, 10), "down_spikes": (6, 14), "destructibles": (2, 8), "gravity_switches": (8, 18)},
}

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_VERSION,
    "curriculum": [
        {"difficulty": d, "context_index": d - 1, "search_work": SEARCH_WORK[d - 1]}
        for d in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "docs/generator-evidence/bp35.md#native-reference-inventory",
        "solution_mechanics": "docs/generator-evidence/bp35.md#generated-curriculum-and-modes",
        "native_budget": "docs/generator-evidence/bp35.md#native-reference-inventory",
        "context_engine_replay": "docs/generator-evidence/bp35-final.md",
        "novelty_split": "docs/generator-evidence/bp35.md#bounded-correction-evidence",
        "bounded_rejections": "docs/generator-evidence/bp35.md#bounded-correction-evidence",
        "structural_chamber_replay": "docs/generator-evidence/native-astra-bp35-chambers/review.md#independent-native-execution",
        "root_train_replay": "docs/generator-evidence/bp35-chambers-root-train.json",
        "root_validation_replay": "docs/generator-evidence/bp35-chambers-root-validation.json",
        "root_test_replay": "docs/generator-evidence/bp35-chambers-root-test.json",
        "native_frame_review": "docs/generator-evidence/bp35-render-comparison.json",
        "official_copy_deny": "docs/generator-evidence/bp35.md#bounded-correction-evidence",
    },
    "caveats": [
        "UNDO is certified through recovery probes and is not padded into winning teacher routes",
        "spike participation is a native losing counterfactual rather than contact in a winning route",
        "the constructive planner certifies positive routes but does not prove shortest paths",
        "the bounded greedy deletion probe is an alternate-route guard, not an optimality proof",
        "tiers 6-7 use two causally necessary gravity reversals, below the independently measured shipped within-level census of 3 and 11",
        "tiers 6-7 each use a finite nine-layout support grammar; their restricted adjacent-action search excludes remote clicks, reversals, closures, and undo",
        "native initial entity-grid D4 identity rejects the nine official layouts but is not a graph-isomorphism or population-novelty proof",
        "official live-prefix teacher recovery remains unsupported outside pristine pinned witnesses; generated recovery is separate",
    ],
}


def _int(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _typed_equal(left, right):
    """Compare JSON-like certificates without Python's bool/int aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            match = next(
                (
                    index for index, (right_key, _) in enumerate(unmatched)
                    if _typed_equal(left_key, right_key)
                ),
                None,
            )
            if match is None:
                return False
            _, right_value = unmatched.pop(match)
            if not _typed_equal(left_value, right_value):
                return False
        return not unmatched
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _typed_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _cells(rows):
    return sorted((x, y, char) for y, row in enumerate(rows) for x, char in enumerate(row) if char != " ")


def _d4_cells(rows):
    points = _cells(rows)
    transforms = []
    for mirror in (False, True):
        for rotation in range(4):
            mapped = []
            for x, y, char in points:
                if mirror:
                    x = -x
                for _ in range(rotation):
                    x, y = -y, x
                mapped.append((x, y, char))
            min_x = min(x for x, _, _ in mapped)
            min_y = min(y for _, y, _ in mapped)
            norm = sorted((x - min_x, y - min_y, char) for x, y, char in mapped)
            transforms.append(norm)
    return min(transforms, key=_canonical)


def _identity_atom(value):
    """Encode a native scalar without collapsing bool/int/string aliases."""
    if value is None:
        return ["none", None]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return ["int", int(value)]
    if type(value) is str:
        return ["str", value]
    raise TypeError(f"unsupported native identity scalar {type(value).__name__}")


def _d4_typed_points(points):
    """Canonicalize typed point roles under translation and D4 symmetry."""
    unique = {}
    for point in points:
        if not isinstance(point, (tuple, list)) or len(point) != 3:
            raise TypeError("native identity points must be (x, y, role) triples")
        x, y, role = point
        if type(x) is not int or type(y) is not int:
            raise TypeError("native identity coordinates must be exact integers")
        role_key = _canonical(role)
        unique[(x, y, role_key)] = copy.deepcopy(role)
    if not unique:
        raise ValueError("native identity needs at least one occupied point")

    source = [(x, y, role) for (x, y, _), role in unique.items()]
    transforms = []
    for mirror in (False, True):
        for rotation in range(4):
            mapped = []
            for source_x, source_y, role in source:
                x, y = source_x, source_y
                if mirror:
                    x = -x
                for _ in range(rotation):
                    x, y = -y, x
                mapped.append((x, y, role))
            min_x = min(x for x, _, _ in mapped)
            min_y = min(y for _, y, _ in mapped)
            norm = sorted(
                ([x - min_x, y - min_y, role] for x, y, role in mapped),
                key=_canonical,
            )
            transforms.append(norm)
    return min(transforms, key=_canonical)


def geometry_identities(spec):
    rows = tuple(spec["rows_bottom_up"])
    raw = _sha({"version": GEOMETRY_VERSION, "cells": _cells(rows)})
    d4 = _sha({"version": GEOMETRY_VERSION, "cells": _d4_cells(rows)})
    gameplay = _sha({
        "version": MECHANICS_VERSION,
        "difficulty": spec["difficulty"],
        "context_index": spec["context_index"],
        "native_action_limit": names.native_action_limit(spec["difficulty"]),
        "cells": _cells(rows),
    })
    return raw, d4, gameplay


def geometry_split(d4_hash):
    bucket = int(d4_hash[:16], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def _child_seed(game_seed, index, difficulty):
    game_seed = _int(game_seed, "game seed", minimum=0)
    data = f"{SOURCE_ID}:{game_seed}:{index}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _effective_seed(seed, difficulty, attempt):
    data = f"{int(seed)}:{int(difficulty)}:{int(attempt)}".encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _blank(width, height):
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x in range(width):
        grid[0][x] = grid[-1][x] = "o"
    for y in range(height):
        # Official BP35 rooms use two-cell side walls. Matching that readable
        # frame keeps the playable shaft and sprites visually proportioned.
        grid[y][0] = grid[y][1] = grid[y][-2] = grid[y][-1] = "o"
    return grid


def _row(grid, y, char="o"):
    for x in range(len(grid[y])):
        grid[y][x] = char


def _choose_columns(rng, count):
    choices = list(range(2, 9))
    cols = [rng.choice(choices)]
    while len(cols) < count:
        far = [x for x in choices if x != cols[-1] and abs(x - cols[-1]) >= 2]
        cols.append(rng.choice(far))
    return cols


def _move_semantic(route, start, target):
    action = names.ACTION_RIGHT if target > start else names.ACTION_LEFT
    route.extend((action, None) for _ in range(abs(target - start)))


def _base_draft(seed, difficulty, attempt, rows, profile, construction, route, probes):
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_version": QUALITY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "source_id": SOURCE_ID,
        "source_sha256": SOURCE_SHA256,
        "source": "generated_only",
        "seed": int(seed),
        "attempt": attempt,
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "rows_bottom_up": rows,
        "legend": copy.deepcopy(names.LEGEND),
        "groups": copy.deepcopy(names.GROUPS),
        "kind": names.GENERATED_KIND,
        "profile": profile,
        "construction": construction,
        "_semantic_solution": route,
        "_counterfactuals": probes,
    }


def _descent_draft(rng, seed, difficulty, attempt):
    profile = PROFILES[difficulty]
    if difficulty == 1:
        height, supports = 30, list(rng.choice(DESCENT_SUPPORT_PATTERNS[1]))
    elif difficulty == 2:
        height, supports = 46, [38, 33, 28, 23, 18, 13, 8]
    else:
        height, supports = 37, [27, 18, 9]
    grid = _blank(11, height)
    current = rng.choice((4, 5, 6))
    start = current
    targets = []
    route = []
    open_spans = []
    for index, support in enumerate(supports):
        far = [x for x in range(2, 9) if abs(x - current) in (3, 4)]
        target = rng.choice(far)
        targets.append(target)
        if difficulty < 3:
            for x in range(2, 9):
                grid[support][x] = "x"
        else:
            for x in range(2, 9):
                grid[support][x] = "1"
            direction = 1 if target > current else -1
            span = list(range(current + direction, target, direction))
            for x in span:
                grid[support][x] = "2"
                route.append((names.ACTION_CLICK, (x, support)))
            # One alternate destructible per bridge-building row preserves the
            # official mixed-device cue and blocks identity-only padding.
            spare = next(x for x in range(2, 9) if x not in span and x not in (current, target))
            grid[support][spare] = "x"
            open_spans.append(span)
            if index == len(supports) - 1:
                grid[support][target] = "x"
        _move_semantic(route, current, target)
        route.append((names.ACTION_CLICK, (target, support)))
        current = target
    grid[supports[0] + 1][start] = "n"
    grid[1][current] = "+"
    probes = []
    if difficulty >= 2:
        first_direction = 1 if targets[0] > start else -1
        spike_x = start - first_direction
        grid[supports[0]][spike_x] = "v"
        probes.append({
            "name": "down_spike", "prefix": 0,
            "action": names.ACTION_LEFT if first_direction > 0 else names.ACTION_RIGHT,
            "cell": [spike_x, supports[0]],
        })
    hazard_y = height - 3
    _row(grid, hazard_y, "m")
    _row(grid, hazard_y + 1, "w")
    rows = ["".join(row) for row in grid]
    return _base_draft(
        seed, difficulty, attempt, rows,
        {"grammar": profile["grammar"], "layers": profile["layers"],
         "hazards": True, "spikes": list(profile["spikes"])},
        {"supports": supports, "route_columns": targets, "start_column": start,
         "hazard_start_y": hazard_y, "open_spans": open_spans},
        route, probes,
    )


def _gravity_draft(rng, seed, difficulty, attempt):
    profile = PROFILES[difficulty]
    layers = profile["layers"]
    height = {4: 32, 5: 32, 6: 33, 7: 33}[difficulty]
    supports = list(rng.choice(GRAVITY_SUPPORT_PATTERNS[difficulty]))
    grid = _blank(11, height)
    passes = profile["passes"]
    start = 5
    pass_columns = list(rng.choice(GRAVITY_PASS_PATTERNS[difficulty]))
    first_direction = 1 if pass_columns[0] > start else -1
    down_spike_x = start - first_direction
    gate_columns = []
    gate_initial = []
    initial_open = []
    incoming = start
    for layer in range(layers):
        choices = list(range(2, 9))
        rng.shuffle(choices)
        gate_columns.append(list(pass_columns))
        for x in range(2, 9):
            grid[supports[layer]][x] = "o"
        layer_gate_initial = []
        for pass_index, x in enumerate(gate_columns[layer]):
            if difficulty in (6, 7):
                # The descending and final columns are initially solid.  The
                # middle column remains open for the reversed-gravity ascent.
                char = "2" if pass_index == 1 else "1"
            elif difficulty == 5:
                char = "2" if pass_index == passes - 1 else "x"
            else:
                char = profile["gate"]
            grid[supports[layer]][x] = char
            layer_gate_initial.append(char)
        gate_initial.append(layer_gate_initial)
        open_count = 0
        candidates = [
            x for x in range(2, 9)
            if x != incoming and x not in gate_columns[layer]
        ]
        rng.shuffle(candidates)
        opened = candidates[:open_count]
        for x in opened:
            grid[supports[layer]][x] = "2"
        initial_open.append(opened)
        incoming = gate_columns[layer][0]

    turn_bridge_cells = []
    turn_bridge_spikes = []
    arrival_dividers = []
    ascent_dividers = []
    reversal_traps = []
    if difficulty in (6, 7):
        first_column, ascent_column, final_column = pass_columns
        direction = 1 if ascent_column > first_column else -1
        turn_bridge_cells = list(range(first_column + direction, ascent_column + direction, direction))
        for x in turn_bridge_cells:
            grid[supports[-1]][x] = "2"
            grid[supports[-1] - 1][x] = "v"
            turn_bridge_spikes.append((x, supports[-1] - 1))
        grid[supports[-1] - 1][first_column] = "v"
        turn_bridge_spikes.append((first_column, supports[-1] - 1))
        ascent_divider_x = ascent_column + (1 if first_column > ascent_column else -1)
        final_divider_x = final_column - (1 if final_column > first_column else -1)
        bottom_support_y = supports[-1]
        bottom_transfer_y = bottom_support_y + 1
        top_transfer_y = height - 2

        # Two continuous, visible barriers form three real chambers under
        # either gravity direction.  The ascent barrier has one stateful
        # bottom doorway: an open bridge over an immutable spike, with empty
        # standing space above it.  Both barriers have one ceiling aperture.
        # No per-shelf gap remains for an underside or camera-shifted bypass.
        for y in range(height):
            ascent_cell = (ascent_divider_x, y)
            if y == bottom_support_y:
                if grid[y][ascent_divider_x] != "2":
                    return None
            elif y == bottom_support_y - 1:
                if grid[y][ascent_divider_x] != "v":
                    return None
            elif y in (bottom_transfer_y, top_transfer_y):
                if grid[y][ascent_divider_x] != " ":
                    return None
            else:
                grid[y][ascent_divider_x] = "o"
                ascent_dividers.append(ascent_cell)

            final_cell = (final_divider_x, y)
            if y == top_transfer_y:
                if grid[y][final_divider_x] != " ":
                    return None
            else:
                grid[y][final_divider_x] = "o"
                arrival_dividers.append(final_cell)

        # A full immutable spike barrier caps the middle chamber.  The player
        # can cross above it only after ascending in the separate x=2 shaft.
        trap_stop = final_divider_x + (1 if difficulty == 7 else 0)
        for x in range(ascent_divider_x + 1, trap_stop):
            cell = (x, height - 3)
            existing = grid[cell[1]][cell[0]]
            if existing != " " and not (
                difficulty == 7 and x == final_divider_x and existing == "o"
            ):
                return None
            grid[cell[1]][cell[0]] = "u"
            reversal_traps.append(cell)

    route = []
    probes = []
    current = start
    grid[supports[0] + 1][start] = "n"
    spike_x = down_spike_x
    grid[supports[0]][spike_x] = "v"
    probes.append({
        "name": "down_spike", "prefix": 0,
        "action": names.ACTION_RIGHT if spike_x > start else names.ACTION_LEFT,
        "cell": [spike_x, supports[0]],
    })

    # Banks are alternative consumable switches, as in the shipped gravity
    # tiers.  Route switches are installed later at the actual turn columns so
    # they are removed from the new gravity direction before native motion.
    switch_target = {4: 4, 5: 4, 6: 6, 7: 10}[difficulty]
    switch_cells = []

    if difficulty in (6, 7):
        first_column, ascent_column, final_column = pass_columns
        _move_semantic(route, current, first_column)
        current = first_column
        # Leave the last first-column gate closed. The player must land above
        # it and build the bottom turn bridge instead of falling to the floor.
        for support in supports[:-1]:
            route.append((names.ACTION_CLICK, (first_column, support)))
        for x in turn_bridge_cells:
            route.append((names.ACTION_CLICK, (x, supports[-1])))
        _move_semantic(route, current, ascent_column)
        current = ascent_column
        bottom_switch = (1, 2)
        top_switch = (1, height - 3)
        for switch in (bottom_switch, top_switch):
            if grid[switch[1]][switch[0]] not in (" ", "o"):
                return None
            grid[switch[1]][switch[0]] = "g"
            switch_cells.append(switch)
        route.append((names.ACTION_CLICK, bottom_switch))
        # The open ascent gates carry the player to the ceiling. At the top,
        # cross to a distinct final column and reverse gravity again.
        _move_semantic(route, current, final_column)
        current = final_column
        route.append((names.ACTION_CLICK, top_switch))
        for support in supports:
            route.append((names.ACTION_CLICK, (final_column, support)))
    else:
        bottom_turns = 0
        top_turns = 0
        up_probe_added = False
        for pass_index in range(passes):
            order = range(layers) if pass_index % 2 == 0 else range(layers - 1, -1, -1)
            for layer in order:
                if pass_index == 0:
                    initially_open_gates = [
                        x for index, (x, char) in enumerate(zip(gate_columns[layer], gate_initial[layer]))
                        if char == "2"
                    ]
                    for x in [*initial_open[layer], *initially_open_gates]:
                        route.append((names.ACTION_CLICK, (x, supports[layer])))
                target = gate_columns[layer][pass_index]
                _move_semantic(route, current, target)
                route.append((names.ACTION_CLICK, (target, supports[layer])))
                current = target
            if pass_index == passes - 1:
                break
            next_layer = layers - 1 if pass_index % 2 == 0 else 0
            next_target = gate_columns[next_layer][pass_index + 1]
            _move_semantic(route, current, next_target)
            current = next_target
            if pass_index % 2 == 0:
                switch = (1, 2 + bottom_turns)
                bottom_turns += 1
            else:
                switch = (1, height - 3 - top_turns)
                top_turns += 1
            if grid[switch[1]][switch[0]] not in (" ", "o"):
                return None
            grid[switch[1]][switch[0]] = "g"
            switch_cells.append(switch)
            route.append((names.ACTION_CLICK, switch))
            if not up_probe_added and pass_index == 0 and "u" in profile["spikes"]:
                direction = 1 if gate_columns[next_layer][pass_index + 1] <= 5 else -1
                up_spike_x = current + direction
                if not 2 <= up_spike_x <= 8:
                    up_spike_x = current - direction
                up_spike_y = supports[-1]
                grid[up_spike_y][up_spike_x] = "u"
                probes.append({
                    "name": "up_spike", "prefix": len(route),
                    "action": names.ACTION_RIGHT if up_spike_x > current else names.ACTION_LEFT,
                    "cell": [up_spike_x, up_spike_y],
                })
                up_probe_added = True

    if difficulty in (6, 7):
        # Alternative resources stay in visible side-wall banks beside the
        # two turn chambers. They can be clicked remotely whenever the camera
        # shows them, but premature reversal remains behind the immutable cap.
        count_per_bank = switch_target // 2
        bottom_cells = tuple((1, 2 + index) for index in range(count_per_bank))
        top_cells = tuple((1, height - 3 - index) for index in range(count_per_bank))
        for x, y in (*bottom_cells, *top_cells):
            if grid[y][x] not in (" ", "o", "g"):
                return None
            if grid[y][x] != "g":
                grid[y][x] = "g"
                switch_cells.append((x, y))
    else:
        # Remaining switches occupy the protected second side-wall strip.
        for y in range(4, height - 4):
            if len(switch_cells) >= switch_target:
                break
            if grid[y][1] == "o":
                grid[y][1] = "g"
                switch_cells.append((1, y))

    # The goal is entered vertically through the final pass.  Directional
    # spike guards isolate it from earlier bottom/top turn corridors, so the
    # player cannot replace the gravity sequence with a horizontal walk.
    if passes % 2:
        goal_y = 2
        guard_y, guard_char = 1, "v"
    else:
        goal_y = height - 2
        guard_y, guard_char = height - 2, "u"
    grid[goal_y][current] = "+"
    goal_guards = []
    guard_x = current - 3 if current >= 6 else current + 3
    grid[guard_y][guard_x] = guard_char
    goal_guards.append([guard_x, guard_y, guard_char])
    rows = ["".join(row) for row in grid]
    return _base_draft(
        seed, difficulty, attempt, rows,
        {"grammar": profile["grammar"], "layers": layers, "passes": passes,
         "gate": profile["gate"], "spikes": list(profile["spikes"])},
        {"supports": supports, "gate_columns": gate_columns, "start_column": start,
         "gate_initial": gate_initial, "initial_open": initial_open,
         "switch_cells": [list(cell) for cell in switch_cells],
         "turn_bridge_cells": [[x, supports[-1]] for x in turn_bridge_cells],
         "turn_bridge_spikes": [list(cell) for cell in turn_bridge_spikes],
         "arrival_dividers": [list(cell) for cell in arrival_dividers],
         "ascent_dividers": [list(cell) for cell in ascent_dividers],
         "reversal_traps": [list(cell) for cell in reversal_traps],
         "goal_cell": [current, goal_y], "goal_guards": goal_guards},
        route, probes,
    )


def _growth_draft(rng, seed, difficulty, attempt):
    # Three native seed chains are joined by bridge-building traversals.  A
    # chain supplies the only support above a spike in its old shaft; each
    # transition closes an open span, crosses it, then reopens the last cell.
    height = 39 if difficulty == 8 else 44
    grid = _blank(11, height)
    columns = list(rng.choice(GROWTH_COLUMN_PATTERNS[difficulty]))
    start_y = 35 if difficulty == 8 else 40
    transition_y = [27, 17] if difficulty == 8 else [31, 20]
    next_seed_y = [22, 12] if difficulty == 8 else [26, 15]
    exit_y = 7 if difficulty == 8 else 4
    grid[start_y][columns[0]] = "n"
    for column, seed_y in zip(columns, [start_y - 1, *next_seed_y]):
        grid[seed_y][column] = "y"

    route = []
    bridge_spans = []
    bridge_initial = []
    bridge_gap_spikes = []
    growth_lane_guards = []
    growth_shafts = []
    current_x, current_y = columns[0], start_y
    for stage in range(3):
        stop_y = transition_y[stage] if stage < 2 else exit_y
        growth_shafts.append([current_x, current_y, stop_y])
        while current_y > stop_y:
            route.append((names.ACTION_CLICK, (current_x, current_y - 1)))
            current_y -= 1
        if stage == 2:
            break
        target_x = columns[stage + 1]
        direction = 1 if target_x > current_x else -1
        span = list(range(current_x + direction, target_x + direction, direction))
        # A visible solid ledge binds the exact vertical descent to the
        # bridge crossing.  If a camera-relative click is deleted and the
        # player stops a row high, the ledge blocks the tempting all-growth
        # horizontal shortcut instead of hiding a collision surface.
        for y in (current_y + 1,):
            for x in span:
                if grid[y][x] == " ":
                    grid[y][x] = "o"
                    growth_lane_guards.append([x, y])
        bridge_y = current_y - 1
        initial = []
        for offset, x in enumerate(span):
            # Mixed solid/open spans match the visual language of the native
            # bridge tiers.  Only open cells are closed before crossing; the
            # target remains solid so opening it is what drops the player to
            # the next growth seed.
            char = "1" if x == target_x or (offset + stage) % 2 == 0 else "2"
            grid[bridge_y][x] = char
            initial.append([x, bridge_y, char])
            if char == "2":
                grid[bridge_y - 1][x] = "v"
                bridge_gap_spikes.append([x, bridge_y - 1])
                route.append((names.ACTION_CLICK, (x, bridge_y)))
        # The vertical chain surrounds the player with new growth cells.  Each
        # click consumes the next growth front and each move advances across
        # the now-solid bridge deck.  Closing every initially open bridge cell
        # is therefore required support, rather than decorative route work.
        for x in span:
            route.append((names.ACTION_CLICK, (x, current_y)))
            route.append((names.ACTION_RIGHT if direction > 0 else names.ACTION_LEFT, None))
        route.append((names.ACTION_CLICK, (target_x, bridge_y)))
        bridge_spans.append([[x, bridge_y] for x in span])
        bridge_initial.append(initial)
        current_x = target_x
        current_y = next_seed_y[stage] + 1

    # The final growth-created side cell opens onto a bridge-supported turn
    # chamber.  Gravity then lifts the player to a ceiling and the gem; the
    # opposite move meets the named up-spike in tier 9.
    goal_column = 7 if current_x <= 5 else 3
    direction = 1 if goal_column > current_x else -1
    final_span = list(range(current_x + direction, goal_column + direction, direction))
    for y in (current_y + 1,):
        for x in final_span:
            if x == goal_column:
                continue
            if grid[y][x] == " ":
                grid[y][x] = "o"
                growth_lane_guards.append([x, y])
    final_bridge_y = current_y - 1
    final_initial = []
    for offset, x in enumerate(final_span):
        char = "1" if x == goal_column or offset % 2 == 0 else "2"
        grid[final_bridge_y][x] = char
        final_initial.append([x, final_bridge_y, char])
        if char == "2":
            grid[final_bridge_y - 1][x] = "v"
            bridge_gap_spikes.append([x, final_bridge_y - 1])
            route.append((names.ACTION_CLICK, (x, final_bridge_y)))
    for x in final_span:
        route.append((names.ACTION_CLICK, (x, current_y)))
        route.append((names.ACTION_RIGHT if direction > 0 else names.ACTION_LEFT, None))
    current_x = goal_column

    # Consume the growth cell directly above the player.  Its next growth
    # front becomes the visible ceiling stop; without this click reversed
    # gravity cannot lift the player into the goal row.
    route.append((names.ACTION_CLICK, (current_x, current_y + 1)))
    switch_cells = [(1, current_y)]
    grid[current_y][1] = "g"
    switch_target = 1 if difficulty == 8 else 10
    for y in range(3, height - 3):
        if len(switch_cells) >= switch_target:
            break
        if grid[y][1] == "o":
            grid[y][1] = "g"
            switch_cells.append((1, y))
    switch_prefix = len(route)
    route.append((names.ACTION_CLICK, switch_cells[0]))
    ceiling_y = current_y + (2 if difficulty == 8 else 3)
    grid[ceiling_y][current_x] = "o"
    goal_x = current_x + (-1 if current_x >= 5 else 1)
    grid[ceiling_y - 1][goal_x] = "+"
    # A visible canopy blocks an uncontrolled downward fall into the goal
    # column.  The certified route reaches the gem horizontally after gravity
    # lifts it beneath the canopy, so this removes the old walk-only shortcut
    # without hiding collision geometry or padding the teacher route.
    canopy = [goal_x, ceiling_y]
    grid[canopy[1]][canopy[0]] = "o"
    _move_semantic(route, current_x, goal_x)

    # Spikes are tied to the exact native collision they certify.  Additional
    # down-spikes sit beneath abandoned shafts, so removing growth causes the
    # first click to fall into a real hazard before another seed is reachable.
    down_spikes = []
    for stage in range(2):
        spike_y = next_seed_y[stage] + 2
        grid[spike_y][columns[stage]] = "v"
        down_spikes.append([columns[stage], spike_y])
    first_direction = 1 if columns[1] > columns[0] else -1
    side_spike_x = columns[0] - first_direction
    grid[start_y - 1][side_spike_x] = "v"
    other_spike_x = columns[0] + first_direction
    grid[start_y - 1][other_spike_x] = "v"
    probes = [{
        "name": "down_spike", "prefix": 0,
        "action": names.ACTION_LEFT if side_spike_x < columns[0] else names.ACTION_RIGHT,
        "cell": [side_spike_x, start_y - 1],
    }]
    if difficulty == 9:
        up_spike_x = current_x + (1 if goal_x < current_x else -1)
        grid[ceiling_y - 1][up_spike_x] = " "
        grid[ceiling_y][up_spike_x] = "u"
        grid[ceiling_y + 1][up_spike_x] = "o"
        probes.append({
            "name": "up_spike", "prefix": switch_prefix + 1,
            "action": names.ACTION_RIGHT if up_spike_x > current_x else names.ACTION_LEFT,
            "cell": [up_spike_x, ceiling_y],
        })
        # Destructible barriers form alternative active cells beside the
        # transition spans; the winning route must remove the final barrier.
        destruct_y = ceiling_y - 1
        for x in range(2, 9):
            if x not in (*columns, current_x, goal_x):
                grid[destruct_y][x] = "x"
        barrier_x = current_x
        grid[destruct_y][barrier_x] = "x"
        route.insert(switch_prefix, (names.ACTION_CLICK, (barrier_x, destruct_y)))
        switch_prefix += 1
        probes[1]["prefix"] += 1

    rows = ["".join(row) for row in grid]
    return _base_draft(
        seed, difficulty, attempt, rows,
        {"grammar": "growth_bridge", "growth": True, "gravity": True,
         "bridge": True, "destructible": difficulty == 9,
         "spikes": list(PROFILES[difficulty]["spikes"])},
        {"growth_columns": columns, "growth_start_y": start_y,
         "growth_shafts": growth_shafts, "bridge_spans": bridge_spans,
         "final_bridge_span": [[x, final_bridge_y] for x in final_span],
         "bridge_initial": bridge_initial, "final_bridge_initial": final_initial,
         "bridge_gap_spikes": bridge_gap_spikes,
         "growth_lane_guards": growth_lane_guards,
         "gravity_switches": [list(cell) for cell in switch_cells],
         "ceiling_y": ceiling_y, "goal_canopy": canopy,
         "growth_ablation_spikes": down_spikes,
         "initial_escape_spikes": [[side_spike_x, start_y - 1], [other_spike_x, start_y - 1]],
         "up_spike_backstop": [up_spike_x, ceiling_y + 1] if difficulty == 9 else None,
         "destructible_y": ceiling_y - 1 if difficulty == 9 else None},
        route, probes,
    )


def _platform_draft(rng, seed, difficulty, attempt):
    if difficulty <= 3:
        return _descent_draft(rng, seed, difficulty, attempt)
    return _gravity_draft(rng, seed, difficulty, attempt)


def _draft(seed, difficulty, attempt):
    rng = random.Random(f"bp35-full:{int(seed)}:{difficulty}:{attempt}")
    return _growth_draft(rng, seed, difficulty, attempt) if difficulty >= 8 else _platform_draft(rng, seed, difficulty, attempt)


def build_level(spec):
    if not isinstance(spec, Mapping) or spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _int(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if type(spec.get("context_index")) is not int or spec.get("context_index") != difficulty - 1:
        raise ValueError("context_index must equal difficulty-1")
    rows = spec.get("rows_bottom_up")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("rows_bottom_up must be a nonempty sequence")
    rows = tuple(rows)
    if any(not isinstance(row, str) for row in rows):
        raise ValueError("grid rows must be strings")
    width = len(rows[0])
    if width != 11 or any(len(row) != width for row in rows):
        raise ValueError("BP35 grids must have rectangular width 11")
    if not (8 <= len(rows) <= 60):
        raise ValueError("BP35 grid height must be in 8..60")
    allowed = set(names.LEGEND) | {" "}
    if any(char not in allowed for row in rows for char in row):
        raise ValueError("grid contains an unknown tile")
    flat = "".join(rows)
    if flat.count("n") != 1 or flat.count("+") != 1:
        raise ValueError("grid needs exactly one player and gem")
    if any(char != "o" for char in rows[0]) or any(char != "o" for char in rows[-1]):
        raise ValueError("grid needs solid bottom and top boundaries")
    descriptor = {
        "kind": names.GENERATED_KIND,
        "rows_bottom_up": list(rows),
        "legend": copy.deepcopy(names.LEGEND),
        "groups": copy.deepcopy(names.GROUPS),
        "difficulty": difficulty,
    }
    # The private route remains available to the package planner, never in the
    # public model observation.
    if "solution" in spec:
        descriptor["solution"] = copy.deepcopy(spec["solution"])
    module = upstream()
    placeholder = module.sprites["sprite-1"].clone().set_position(4, 3)
    return Level(
        sprites=[placeholder], grid_size=(8, 8),
        data={names.LEVEL_GRID_DATA: descriptor},
        name=f"generated-bp35-full-d{difficulty}-s{spec.get('seed', 0)}",
    )


def _context_env(spec):
    difficulty = int(spec["difficulty"])
    levels = [build_level(spec) for _ in range(difficulty)]
    env = Env(levels)
    env.set_level(difficulty - 1)
    return env


def _native_initial_points(env):
    grid = getattr(env.world, names.ATTR_GRID)
    points = []
    for entity in getattr(grid, names.ATTR_ENTITIES):
        occupied = tuple(tuple(map(int, cell)) for cell in entity.hrlzbohbpn)
        if not occupied:
            occupied = ((int(entity.grid_x), int(entity.grid_y)),)
        role = [
            ["name", _identity_atom(entity.name)],
            ["material", _identity_atom(entity.flrpnczugo)],
            ["collidable", _identity_atom(bool(entity.collidable))],
        ]
        points.extend((x, y, role) for x, y in occupied)
    return points


def _native_points_d4_sha256(points):
    return _sha({
        "version": "bp35-native-initial-entity-d4-v1",
        "points": _d4_typed_points(points),
    })


def _native_initial_d4_sha256(env):
    return _native_points_d4_sha256(_native_initial_points(env))


_OFFICIAL_NATIVE_D4 = None


def _official_native_d4_identities():
    global _OFFICIAL_NATIVE_D4
    if _OFFICIAL_NATIVE_D4 is None:
        identities = []
        for index in range(len(upstream().levels)):
            env = Env()
            env.set_level(index)
            identities.append(_native_initial_d4_sha256(env))
        _OFFICIAL_NATIVE_D4 = tuple(identities)
    return _OFFICIAL_NATIVE_D4


def _target_name(env, cell):
    grid = getattr(env.world, names.ATTR_GRID)
    entities = grid.jhzcxkveiw(*cell)
    return entities[0].name if len(entities) == 1 else None


def _hazard_y(env):
    grid = getattr(env.world, names.ATTR_GRID)
    hazards = grid.wwkbcxznzg(names.HAZARD_A)
    return int(hazards[0].grid_y) if hazards else None


def _materialize_and_replay(spec, semantic):
    env = _context_env(spec)
    base = env.levels_completed
    actions = []
    events = Counter()
    min_hazard_distance = None
    min_click_margin = 64
    semantic = tuple(semantic)
    for action_index, (action_id, cell) in enumerate(semantic):
        frame = np.asarray(env.render())
        if frame.shape != (64, 64) or frame.dtype.kind not in "iu" or frame.min() < 0 or frame.max() > 15:
            return None
        events["visual_frames_checked"] += 1
        if action_id == names.ACTION_CLICK:
            if cell is None:
                return None
            camera_y = int(getattr(getattr(env.world, names.ATTR_CAMERA), names.ATTR_CAMERA_OFFSET)[1])
            x, y = int(cell[0]) * 6 + 3, int(cell[1]) * 6 - camera_y + 3
            if not (0 <= x < 64 and 0 <= y < 63):
                return None
            margin = min(x, y, 63 - x, 62 - y)
            min_click_margin = min(min_click_margin, margin)
            if margin < 3 or len(np.unique(frame[y - 3 : y + 4, x - 3 : x + 4])) < 2:
                return None
            events["visible_click_cues"] += 1
            before_name = _target_name(env, cell)
            before_gravity = bool(getattr(env.world, names.ATTR_GRAVITY_DOWN))
            action = (action_id, x, y)
        else:
            before_name = None
            before_gravity = bool(getattr(env.world, names.ATTR_GRAVITY_DOWN))
            action = (action_id, None, None)
        before_hazard = _hazard_y(env)
        observation = env.perform(*action)
        actions.append(action)
        after_hazard = _hazard_y(env)
        if before_hazard is not None and after_hazard != before_hazard:
            events["hazard_moves"] += 1
        if after_hazard is not None:
            player_y = int(getattr(env.world, names.ATTR_PLAYER).grid_y)
            distance = abs(after_hazard - player_y)
            min_hazard_distance = distance if min_hazard_distance is None else min(min_hazard_distance, distance)
        if before_name == names.DESTRUCTIBLE:
            events["destructible_clicks"] += 1
        elif before_name == names.GROWER:
            events["growth_clicks"] += 1
        elif before_name == names.BRIDGE_SOLID:
            events["bridge_toggles"] += 1
            events["bridge_solid_to_open"] += 1
        elif before_name == names.BRIDGE_OPEN:
            events["bridge_toggles"] += 1
            events["bridge_open_to_solid"] += 1
        elif before_name == names.GRAVITY_SWITCH:
            events["gravity_toggles"] += 1
        if bool(getattr(env.world, names.ATTR_GRAVITY_DOWN)) != before_gravity:
            events["gravity_changes"] += 1
        events[f"action_{action_id}"] += 1
        if observation.state == GameState.GAME_OVER:
            return None
        if env.levels_completed > base or observation.state == GameState.WIN:
            if action_index != len(semantic) - 1:
                return None
            events["won"] = 1
            break
    if not events["won"]:
        return None
    if min_hazard_distance is not None:
        events["minimum_hazard_distance"] = min_hazard_distance
    events["minimum_click_margin"] = min_click_margin
    return tuple(actions), dict(events)


def _installed_counts(spec):
    counts = Counter("".join(spec["rows_bottom_up"]))
    return {
        "destructibles": counts["x"], "growth_seeds": counts["y"],
        "solid_bridges": counts["1"], "open_bridges": counts["2"],
        "gravity_switches": counts["g"], "down_spikes": counts["v"],
        "up_spikes": counts["u"], "moving_hazard_bands": int(bool(counts["m"] and counts["w"])),
    }


def _profile_events_ok(difficulty, events, installed):
    if installed.get("destructibles", 0) and events.get("destructible_clicks", 0) < 1:
        return False
    if (installed.get("solid_bridges", 0) or installed.get("open_bridges", 0)) and events.get("bridge_toggles", 0) < 1:
        return False
    required_gravity = {4: 2, 5: 2, 6: 2, 7: 2, 8: 1, 9: 1}.get(difficulty, 0)
    if events.get("gravity_changes", 0) < required_gravity:
        return False
    if difficulty >= 8 and events.get("growth_clicks", 0) < 20:
        return False
    if difficulty in (3, 5, 6, 7, 8, 9):
        if events.get("bridge_open_to_solid", 0) < 1:
            return False
        if events.get("bridge_solid_to_open", 0) < 1:
            return False
    if difficulty <= 3 and events.get("hazard_moves", 0) < 1:
        return False
    if difficulty <= 3 and events.get("minimum_hazard_distance", 99) > 6:
        return False
    if difficulty >= 2 and installed.get("down_spikes", 0) < 1:
        return False
    if difficulty in (5, 9) and installed.get("up_spikes", 0) < 1:
        return False
    return True


def _constraint_probe_results(spec, actions, probes):
    results = {}
    for probe in probes:
        if not isinstance(probe, Mapping):
            return None
        name = probe.get("name")
        prefix = probe.get("prefix")
        action = probe.get("action")
        cell = probe.get("cell")
        if (
            name not in ("down_spike", "up_spike")
            or type(prefix) is not int
            or not 0 <= prefix <= len(actions)
            or action not in (names.ACTION_LEFT, names.ACTION_RIGHT)
            or not isinstance(cell, list)
            or len(cell) != 2
            or any(type(value) is not int for value in cell)
        ):
            return None
        spike_char = "v" if name == "down_spike" else "u"
        x, y = cell
        rows = spec.get("rows_bottom_up")
        if (
            not isinstance(rows, Sequence)
            or not 0 <= y < len(rows)
            or not 0 <= x < len(rows[y])
            or rows[y][x] != spike_char
        ):
            return None
        env = _context_env(spec)
        for certified in actions[:prefix]:
            observation = env.perform(*certified)
            if observation.state != GameState.NOT_FINISHED:
                return None
        observation = env.perform(action, None, None)
        baseline_loss = bool(
            observation.state == GameState.GAME_OVER
            and getattr(env.world, names.ATTR_WORLD_LOSS)
        )
        ablated = copy.deepcopy(spec)
        ablated_rows = [list(row) for row in ablated["rows_bottom_up"]]
        ablated_rows[y][x] = " "
        ablated["rows_bottom_up"] = ["".join(row) for row in ablated_rows]
        ablated_env = _context_env(ablated)
        for certified in actions[:prefix]:
            observation = ablated_env.perform(*certified)
            if observation.state != GameState.NOT_FINISHED:
                return None
        ablated_observation = ablated_env.perform(action, None, None)
        ablated_loss = bool(
            ablated_observation.state == GameState.GAME_OVER
            and getattr(ablated_env.world, names.ATTR_WORLD_LOSS)
        )
        results[f"{name}_loss"] = baseline_loss and not ablated_loss
    return results


def _native_probe_key(env):
    grid = getattr(env.world, names.ATTR_GRID)
    entities = tuple(sorted(
        (entity.name, int(entity.grid_x), int(entity.grid_y))
        for entity in getattr(grid, names.ATTR_ENTITIES)
    ))
    player = getattr(env.world, names.ATTR_PLAYER)
    return (
        int(player.grid_x), int(player.grid_y),
        bool(getattr(env.world, names.ATTR_GRAVITY_DOWN)),
        entities,
    )


def _visible_non_growth_actions(env):
    actions = [(names.ACTION_LEFT, None, None), (names.ACTION_RIGHT, None, None)]
    grid = getattr(env.world, names.ATTR_GRID)
    camera_y = int(getattr(getattr(env.world, names.ATTR_CAMERA), names.ATTR_CAMERA_OFFSET)[1])
    clickable = {
        names.DESTRUCTIBLE, names.BRIDGE_SOLID, names.BRIDGE_OPEN,
        names.GRAVITY_SWITCH,
    }
    for entity in getattr(grid, names.ATTR_ENTITIES):
        if entity.name not in clickable:
            continue
        x = int(entity.grid_x) * 6 + 3
        y = int(entity.grid_y) * 6 - camera_y + 3
        if 0 <= x < 64 and 0 <= y < 64:
            actions.append((names.ACTION_CLICK, x, y))
    return tuple(dict.fromkeys(actions))


def _bounded_no_growth_probe(spec, *, horizon=12, max_expanded=2_000):
    queue = deque([(_context_env(spec), 0)])
    seen = {_native_probe_key(queue[0][0])}
    expanded = 0
    while queue:
        env, depth = queue.popleft()
        if depth >= horizon:
            continue
        if expanded >= max_expanded:
            return {"horizon": horizon, "expanded": expanded, "truncated": True, "win": False}
        expanded += 1
        start_score = env.levels_completed
        for action in _visible_non_growth_actions(env):
            child = env.clone()
            observation = child.perform(*action)
            if child.levels_completed > start_score or observation.state == GameState.WIN:
                return {"horizon": horizon, "expanded": expanded, "truncated": False, "win": True}
            if observation.state == GameState.GAME_OVER:
                continue
            key = _native_probe_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, depth + 1))
    return {"horizon": horizon, "expanded": expanded, "truncated": False, "win": False}


def _growth_dependency_evidence(spec, actions):
    ablated = copy.deepcopy(spec)
    ablated["rows_bottom_up"] = [row.replace("y", " ") for row in spec["rows_bottom_up"]]
    env = _context_env(ablated)
    start_score = env.levels_completed
    ablated_win = False
    for action in actions:
        observation = env.perform(*action)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            ablated_win = True
            break
        if observation.state == GameState.GAME_OVER:
            break
    evidence = _bounded_no_growth_probe(spec)
    evidence["growth_removed_certified_route_wins"] = ablated_win
    return evidence


def _route_target_names(spec, actions):
    env = _context_env(spec)
    names_seen = []
    for action in actions:
        if action[0] == names.ACTION_CLICK:
            camera_y = int(getattr(getattr(env.world, names.ATTR_CAMERA), names.ATTR_CAMERA_OFFSET)[1])
            cell = (int(action[1]) // 6, (int(action[2]) + camera_y) // 6)
            names_seen.append(_target_name(env, cell))
        else:
            names_seen.append(None)
        observation = env.perform(*action)
        if observation.state != GameState.NOT_FINISHED:
            break
    return names_seen


def _route_wins(spec, actions):
    env = _context_env(spec)
    start = env.levels_completed
    for action in actions:
        try:
            observation = env.perform(*action)
        except (TypeError, ValueError, RuntimeError, IndexError):
            return False
        if env.levels_completed > start or observation.state == GameState.WIN:
            return True
        if observation.state == GameState.GAME_OVER:
            return False
    return False


def _perform_cell_click(env, cell):
    camera_y = int(getattr(getattr(env.world, names.ATTR_CAMERA), names.ATTR_CAMERA_OFFSET)[1])
    x, y = int(cell[0]) * 6 + 3, int(cell[1]) * 6 - camera_y + 3
    if not (0 <= x < 64 and 0 <= y < 64):
        return None
    return env.perform(names.ACTION_CLICK, x, y)


def _gravity_chamber_certificate(spec):
    """Recompute the tier-6/7 three-chamber invariant on the native world."""
    rows = tuple(spec["rows_bottom_up"])
    construction = spec["construction"]
    supports = tuple(construction["supports"])
    first_column, ascent_column, final_column = construction["gate_columns"][0]
    ascent_partition = ascent_column + (1 if first_column > ascent_column else -1)
    final_partition = final_column - (1 if final_column > first_column else -1)
    bottom_support = supports[-1]
    bottom_transfer = bottom_support + 1
    top_transfer = len(rows) - 2
    trap_y = len(rows) - 3
    trap_stop = final_partition + (1 if spec["difficulty"] == 7 else 0)
    trap_cells = tuple((x, trap_y) for x in range(ascent_partition + 1, trap_stop))

    expected = {}
    for y in range(len(rows)):
        if y == bottom_support:
            expected[(ascent_partition, y)] = "2"
        elif y == bottom_support - 1:
            expected[(ascent_partition, y)] = "v"
        elif y in (bottom_transfer, top_transfer):
            expected[(ascent_partition, y)] = " "
        else:
            expected[(ascent_partition, y)] = "o"
        expected[(final_partition, y)] = " " if y == top_transfer else "o"
    for cell in trap_cells:
        expected[cell] = "u"
    for x, y in construction["turn_bridge_cells"]:
        expected[(int(x), int(y))] = "2"
    for x, y in construction["turn_bridge_spikes"]:
        expected[(int(x), int(y))] = "v"

    row_match = all(rows[y][x] == char for (x, y), char in expected.items())
    top_corridor_clear = all(rows[top_transfer][x] == " " for x in range(ascent_column, final_column + 1))
    env = _context_env(spec)
    native_names = {
        " ": None,
        "o": names.WALL,
        "2": names.BRIDGE_OPEN,
        "v": names.SPIKE_A,
        "u": names.SPIKE_B,
    }
    native_match = all(_target_name(env, cell) == native_names[char] for cell, char in expected.items())
    player = getattr(env.world, names.ATTR_PLAYER)
    goal_cell = tuple(map(int, construction["goal_cell"]))
    player_in_middle = ascent_partition < int(player.grid_x) < final_partition
    goal_in_final = goal_cell[0] > final_partition and _target_name(env, goal_cell) == names.GEM
    immutable_chars = {"o", "v", "u"}
    allowed_mutable = {
        tuple(map(int, cell)) for cell in construction["turn_bridge_cells"]
    } | {
        (ascent_partition, bottom_transfer),
        (ascent_partition, top_transfer),
        (final_partition, top_transfer),
    }
    mutable_barriers = [
        [x, y, char]
        for (x, y), char in expected.items()
        if char not in immutable_chars and (x, y) not in allowed_mutable
    ]
    valid = bool(
        row_match
        and native_match
        and top_corridor_clear
        and player_in_middle
        and goal_in_final
        and not mutable_barriers
    )
    return {
        "method": "native_three_chamber_all_rows_v1",
        "checked_rows": len(rows),
        "partition_cells_checked": 2 * len(rows),
        "ascent_partition_column": ascent_partition,
        "final_partition_column": final_partition,
        "bottom_bridge_aperture": [ascent_partition, bottom_support],
        "bottom_transfer_cell": [ascent_partition, bottom_transfer],
        "top_transfer_cells": [
            [ascent_partition, top_transfer],
            [final_partition, top_transfer],
        ],
        "middle_trap_cells": [list(cell) for cell in trap_cells],
        "row_match": row_match,
        "native_entity_match": native_match,
        "top_corridor_clear": top_corridor_clear,
        "player_in_middle_chamber": player_in_middle,
        "goal_in_final_chamber": goal_in_final,
        "unexpected_mutable_barriers": mutable_barriers,
        "valid": valid,
    }


def _bounded_static_adjacent_probe(spec, *, max_expanded=512):
    """Exhaust moves plus adjacent solid-gate openings from the initial state.

    The restricted action set has no undo, gravity reversal, remote click, or
    open-to-solid bridge transition.  State deduplication therefore gives a
    finite reachability check; ``truncated`` is true unless its whole frontier
    was exhausted below the explicit expansion cap.
    """
    queue = deque([_context_env(spec)])
    seen = {_native_probe_key(queue[0])}
    expanded = 0
    while queue:
        if expanded >= max_expanded:
            return {
                "method": "native_initial_static_adjacent_reachability_v1",
                "action_classes": ["left", "right", "adjacent_solid_gate_open"],
                "max_expanded": max_expanded,
                "expanded": expanded,
                "frontier_exhausted": False,
                "truncated": True,
                "win": False,
            }
        env = queue.popleft()
        expanded += 1
        player = getattr(env.world, names.ATTR_PLAYER)
        gravity_down = bool(getattr(env.world, names.ATTR_GRAVITY_DOWN))
        adjacent = (int(player.grid_x), int(player.grid_y) + (-1 if gravity_down else 1))
        actions = [
            (names.ACTION_LEFT, None, None),
            (names.ACTION_RIGHT, None, None),
        ]
        if _target_name(env, adjacent) == names.BRIDGE_SOLID:
            camera_y = int(getattr(getattr(env.world, names.ATTR_CAMERA), names.ATTR_CAMERA_OFFSET)[1])
            click = (adjacent[0] * 6 + 3, adjacent[1] * 6 - camera_y + 3)
            if 0 <= click[0] < 64 and 0 <= click[1] < 64:
                actions.append((names.ACTION_CLICK, *click))
        start_score = env.levels_completed
        for action in actions:
            child = env.clone()
            observation = child.perform(*action)
            if child.levels_completed > start_score or observation.state == GameState.WIN:
                return {
                    "method": "native_initial_static_adjacent_reachability_v1",
                    "action_classes": ["left", "right", "adjacent_solid_gate_open"],
                    "max_expanded": max_expanded,
                    "expanded": expanded,
                    "frontier_exhausted": False,
                    "truncated": False,
                    "win": True,
                }
            if observation.state == GameState.GAME_OVER:
                continue
            key = _native_probe_key(child)
            if key not in seen:
                seen.add(key)
                queue.append(child)
    return {
        "method": "native_initial_static_adjacent_reachability_v1",
        "action_classes": ["left", "right", "adjacent_solid_gate_open"],
        "max_expanded": max_expanded,
        "expanded": expanded,
        "frontier_exhausted": True,
        "truncated": False,
        "win": False,
    }


def _gravity_dependency_evidence(spec, actions):
    """Native bounded checks for the tier-6/7 adjacent escape classes."""
    construction = spec["construction"]
    supports = tuple(construction["supports"])
    first_column, _, final_column = construction["gate_columns"][0]
    old_shortcut = (
        (names.ACTION_LEFT, None, None),
        (names.ACTION_CLICK, 27, 33),
        *((names.ACTION_RIGHT, None, None),) * 4,
    )
    initial_final_shortcut = (
        *((names.ACTION_RIGHT, None, None),) * 3,
        *((names.ACTION_CLICK, 51, 33),) * 10,
    )
    remote_ascent_shortcut = (
        (names.ACTION_LEFT, None, None),
        *((names.ACTION_CLICK, 27, 33),) * 6,
        (names.ACTION_CLICK, 15, 15),
        (names.ACTION_LEFT, None, None),
        (names.ACTION_LEFT, None, None),
        (names.ACTION_CLICK, 9, 15),
        *((names.ACTION_RIGHT, None, None),) * 6,
        (names.ACTION_CLICK, 9, 23),
        *((names.ACTION_CLICK, 51, 33),) * 8,
    )
    underside_zero_closure_shortcut = (
        (names.ACTION_LEFT, None, None),
        *((names.ACTION_CLICK, 27, 33),) * 7,
        (names.ACTION_RIGHT, None, None),
        (names.ACTION_RIGHT, None, None),
        (names.ACTION_CLICK, 9, 15),
        (names.ACTION_RIGHT, None, None),
        (names.ACTION_RIGHT, None, None),
        *((names.ACTION_CLICK, 51, 35),) * 7,
        (names.ACTION_CLICK, 9, 23),
        (names.ACTION_CLICK, 51, 33),
    )
    underside_shifted_shortcut = (
        (names.ACTION_RIGHT, None, None),
        *((names.ACTION_CLICK, 39, 33),) * 9,
        (names.ACTION_CLICK, 33, 33),
        (names.ACTION_LEFT, None, None),
        (names.ACTION_CLICK, 39, 57),
        (names.ACTION_CLICK, 9, 15),
        *((names.ACTION_RIGHT, None, None),) * 3,
        *((names.ACTION_CLICK, 51, 35),) * 9,
        (names.ACTION_CLICK, 9, 23),
        (names.ACTION_CLICK, 51, 33),
    )
    target_names = _route_target_names(spec, actions)
    gravity_indices = [index for index, target in enumerate(target_names) if target == names.GRAVITY_SWITCH]
    closure_indices = [index for index, target in enumerate(target_names) if target == names.BRIDGE_OPEN]

    arrivals = []
    ascent_transfers = []
    start_column = int(construction["start_column"])
    direction = names.ACTION_RIGHT if first_column > start_column else names.ACTION_LEFT
    for opened_gates in range(len(supports)):
        for target_column, rows in ((final_column, arrivals), (construction["gate_columns"][0][1], ascent_transfers)):
            env = _context_env(spec)
            observation = None
            for _ in range(abs(first_column - start_column)):
                observation = env.perform(direction, None, None)
            for support in supports[:opened_gates]:
                observation = _perform_cell_click(env, (first_column, support))
                if observation is None or observation.state != GameState.NOT_FINISHED:
                    break
            transfer_action = names.ACTION_RIGHT if target_column > first_column else names.ACTION_LEFT
            start = env.levels_completed
            for _ in range(abs(target_column - first_column) + 1):
                observation = env.perform(transfer_action, None, None)
                if observation.state != GameState.NOT_FINISHED:
                    break
            player = getattr(env.world, names.ATTR_PLAYER)
            rows.append({
                "opened_first_gates": opened_gates,
                "target_column": int(target_column),
                "reached_target_column": int(player.grid_x) == target_column,
                "win": bool(
                    env.levels_completed > start
                    or (observation is not None and observation.state == GameState.WIN)
                ),
            })

    def premature_reversal_loss(probe_spec):
        env = _context_env(probe_spec)
        for _ in range(abs(first_column - start_column)):
            env.perform(direction, None, None)
        for support in supports[:-1]:
            observation = _perform_cell_click(env, (first_column, support))
            if observation is None or observation.state != GameState.NOT_FINISHED:
                return False
        observation = _perform_cell_click(env, (1, 2))
        return bool(
            observation is not None
            and observation.state == GameState.GAME_OVER
            and getattr(env.world, names.ATTR_WORLD_LOSS)
        )

    ablated = copy.deepcopy(spec)
    trap_x, trap_y = first_column, len(spec["rows_bottom_up"]) - 3
    ablated_rows = [list(row) for row in ablated["rows_bottom_up"]]
    ablated_rows[trap_y][trap_x] = " "
    ablated["rows_bottom_up"] = ["".join(row) for row in ablated_rows]

    without_gravity = [action for index, action in enumerate(actions) if index not in gravity_indices]
    without_closures = [action for index, action in enumerate(actions) if index not in closure_indices]
    return {
        "method": "native_structural_adjacent_strategies_v2",
        "expanded_templates": len(arrivals) + len(ascent_transfers) + 7,
        "truncated": False,
        "old_six_action_shortcut_wins": _route_wins(spec, old_shortcut),
        "initial_final_column_shortcut_wins": _route_wins(spec, initial_final_shortcut),
        "remote_ascent_shortcut_wins": _route_wins(spec, remote_ascent_shortcut),
        "underside_zero_closure_shortcut_wins": _route_wins(
            spec, underside_zero_closure_shortcut
        ),
        "underside_shifted_shortcut_wins": _route_wins(spec, underside_shifted_shortcut),
        "certified_without_gravity_changes_wins": _route_wins(spec, without_gravity),
        "certified_without_bridge_closures_wins": _route_wins(spec, without_closures),
        "gravity_actions_removed": len(gravity_indices),
        "closure_actions_removed": len(closure_indices),
        "premature_reversal_loses": premature_reversal_loss(spec),
        "trap_ablated_premature_reversal_loses": premature_reversal_loss(ablated),
        "arrival_transfers": arrivals,
        "ascent_transfers": ascent_transfers,
        "initial_static_adjacent_search": _bounded_static_adjacent_probe(spec),
    }


_DELETION_PROBE_CACHE = {}


def _actions_first_win(spec, actions):
    env = _context_env(spec)
    base = env.levels_completed
    for index, action in enumerate(actions):
        try:
            observation = env.perform(*action)
        except (TypeError, ValueError, RuntimeError, IndexError):
            return False
        if observation.state == GameState.GAME_OVER:
            return False
        if env.levels_completed > base or observation.state == GameState.WIN:
            return index == len(actions) - 1
    return False


def _greedy_deletion_probe(spec, actions, *, max_checks=1_000):
    key = _sha({
        "rows": list(spec["rows_bottom_up"]),
        "actions": [list(action) for action in actions],
        "max_checks": max_checks,
    })
    cached = _DELETION_PROBE_CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    working = [(original_index, tuple(action)) for original_index, action in enumerate(actions)]
    checks = 0
    index = 0
    while index < len(working):
        if checks >= max_checks:
            result = {
                "method": "native_greedy_single_action_deletion_v1",
                "original_length": len(actions), "reduced_length": len(working),
                "removed_actions": len(actions) - len(working),
                "removed_original_indices": sorted(set(range(len(actions))) - {item[0] for item in working}),
                "checks": checks, "truncated": True,
                "reduced_solution_mechanics": None,
            }
            _DELETION_PROBE_CACHE[key] = copy.deepcopy(result)
            return result
        checks += 1
        candidate = working[:index] + working[index + 1 :]
        if _actions_first_win(spec, [item[1] for item in candidate]):
            working = candidate
            index = 0
        else:
            index += 1
    replay_spec = copy.deepcopy(spec)
    replay_spec["solution"] = [list(action) for _, action in working]
    mechanics = _replay_certificate(replay_spec)
    result = {
        "method": "native_greedy_single_action_deletion_v1",
        "original_length": len(actions), "reduced_length": len(working),
        "removed_actions": len(actions) - len(working),
        "removed_original_indices": sorted(set(range(len(actions))) - {item[0] for item in working}),
        "checks": checks, "truncated": False,
        "reduced_solution_mechanics": mechanics,
    }
    _DELETION_PROBE_CACHE[key] = copy.deepcopy(result)
    return result


def _certify(draft, *, split, search_limit, exclusions):
    candidate = copy.deepcopy(draft)
    semantic = candidate.pop("_semantic_solution")
    probes = candidate.pop("_counterfactuals", ())
    raw, d4, gameplay = geometry_identities(candidate)
    actual_split = geometry_split(d4)
    if actual_split != split:
        exclusions["split_partition"] += 1
        return None
    native_initial_d4 = _native_initial_d4_sha256(_context_env(candidate))
    if native_initial_d4 in _official_native_d4_identities():
        exclusions["official_native_duplicate"] += 1
        return None
    proof = _materialize_and_replay(candidate, semantic)
    if proof is None:
        exclusions["engine_replay"] += 1
        return None
    actions, events = proof
    installed = _installed_counts(candidate)
    if not _profile_events_ok(candidate["difficulty"], events, installed):
        exclusions["mechanic_participation"] += 1
        return None
    chamber_structure = None
    if candidate["difficulty"] in (6, 7):
        chamber_structure = _gravity_chamber_certificate(candidate)
        if not chamber_structure["valid"]:
            exclusions["chamber_structure"] += 1
            return None
    constraint_probes = _constraint_probe_results(candidate, actions, probes)
    required_probes = {"down_spike_loss"} if candidate["difficulty"] >= 2 else set()
    if candidate["difficulty"] in (5, 9):
        required_probes.add("up_spike_loss")
    if constraint_probes is None or any(not constraint_probes.get(key) for key in required_probes):
        exclusions["constraint_probe"] += 1
        return None
    strategy_probes = None
    if candidate["difficulty"] >= 8:
        strategy_probes = _growth_dependency_evidence(candidate, actions)
        if (
            strategy_probes["win"]
            or strategy_probes["truncated"]
            or strategy_probes["growth_removed_certified_route_wins"]
        ):
            exclusions["growth_dependency"] += 1
            return None
    lattice_strategy_probes = None
    if candidate["difficulty"] in (6, 7):
        lattice_strategy_probes = _gravity_dependency_evidence(candidate, actions)
        if (
            lattice_strategy_probes["truncated"]
            or lattice_strategy_probes["old_six_action_shortcut_wins"]
            or lattice_strategy_probes["initial_final_column_shortcut_wins"]
            or lattice_strategy_probes["remote_ascent_shortcut_wins"]
            or lattice_strategy_probes["underside_zero_closure_shortcut_wins"]
            or lattice_strategy_probes["underside_shifted_shortcut_wins"]
            or lattice_strategy_probes["certified_without_gravity_changes_wins"]
            or lattice_strategy_probes["certified_without_bridge_closures_wins"]
            or lattice_strategy_probes["gravity_actions_removed"] < 2
            or lattice_strategy_probes["closure_actions_removed"] < 1
            or not lattice_strategy_probes["premature_reversal_loses"]
            or lattice_strategy_probes["trap_ablated_premature_reversal_loses"]
            or lattice_strategy_probes["initial_static_adjacent_search"]["truncated"]
            or not lattice_strategy_probes["initial_static_adjacent_search"]["frontier_exhausted"]
            or lattice_strategy_probes["initial_static_adjacent_search"]["win"]
            or any(
                row["reached_target_column"] or row["win"]
                for key in ("arrival_transfers", "ascent_transfers")
                for row in lattice_strategy_probes[key]
            )
        ):
            exclusions["lattice_dependency"] += 1
            return None
    shortcut_probe = _greedy_deletion_probe(candidate, actions)
    if (
        shortcut_probe["truncated"]
        or shortcut_probe["reduced_length"] < MIN_ACTIONS[candidate["difficulty"] - 1]
        or not _profile_events_ok(
            candidate["difficulty"], shortcut_probe["reduced_solution_mechanics"], installed
        )
    ):
        exclusions["shortcut_probe"] += 1
        return None
    limit = names.native_action_limit(candidate["difficulty"])
    if len(actions) > search_limit:
        exclusions["search_work"] += 1
        return None
    if len(actions) < MIN_ACTIONS[candidate["difficulty"] - 1]:
        exclusions["route_depth"] += 1
        return None
    if len(actions) >= limit:
        exclusions["native_budget"] += 1
        return None
    candidate.update(
        split=split,
        effective_split=split,
        requested_seed=int(candidate["seed"]),
        effective_seed=_effective_seed(candidate["seed"], candidate["difficulty"], candidate["attempt"]),
        geometry_sha256=raw,
        geometry_d4_sha256=d4,
        gameplay_sha256=gameplay,
        native_initial_d4_sha256=native_initial_d4,
        geometry_split=actual_split,
        solution=[list(action) for action in actions],
        context_solution=[list(action) for action in actions],
        solution_length=len(actions),
        solution_mechanics=events,
        constraint_probe_plan=copy.deepcopy(list(probes)),
        constraint_probes=constraint_probes,
        strategy_probes=strategy_probes,
        lattice_strategy_probes=lattice_strategy_probes,
        chamber_structure=chamber_structure,
        shortcut_probe=shortcut_probe,
        installed_mechanics=installed,
        native_action_limit=limit,
        budget_slack=limit - len(actions),
        search_limit=search_limit,
        search_expanded=len(actions),
        search_generated=len(actions),
        search_truncated=False,
        planner_exact=True,
        planner_optimality="not claimed; constructive positive witness",
        witness_optimality_claimed=False,
        engine_verified=True,
        engine_win=True,
        context_engine_verified=True,
        levels_completed=1,
        verification_level_index=candidate["difficulty"] - 1,
        generation_exclusions=dict(exclusions),
    )
    candidate["proof"] = {
        key: candidate[key]
        for key in (
            "source_id", "source_sha256", "seed", "requested_seed", "effective_seed", "attempt",
            "difficulty", "context_index", "split",
            "geometry_sha256", "geometry_d4_sha256", "gameplay_sha256",
            "native_initial_d4_sha256",
            "solution_length", "solution_mechanics", "constraint_probe_plan", "constraint_probes",
            "strategy_probes", "lattice_strategy_probes", "chamber_structure", "shortcut_probe",
            "native_action_limit",
            "budget_slack", "search_limit", "search_truncated",
            "engine_verified", "engine_win", "levels_completed",
        )
    }
    candidate["proof"].update(
        exact=True, unsupported=False, witness_optimality_claimed=False,
    )
    return candidate


def generate(seed, difficulty, attempts=DEFAULT_ATTEMPTS, limit=DEFAULT_LIMIT, *, split="train", search_limit=None, node_limit=None, max_attempts=None):
    seed = _int(seed, "seed", minimum=0)
    difficulty = _int(difficulty, "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    attempts = attempts if max_attempts is None else max_attempts
    attempts = _int(attempts, "attempts", minimum=1)
    if attempts > MAX_ATTEMPTS:
        raise ValueError(f"attempts cannot exceed {MAX_ATTEMPTS}")
    if node_limit is not None:
        limit = node_limit
    limit = limit if search_limit is None else search_limit
    limit = _int(limit, "limit", minimum=0)
    generate.last_rejections = {}
    if limit == 0:
        return None
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    exclusions = Counter()
    for attempt in range(attempts):
        draft = _draft(seed, difficulty, attempt)
        if draft is None:
            exclusions["draft_layout"] += 1
            continue
        candidate = _certify(
            draft, split=split,
            search_limit=min(limit, SEARCH_WORK[difficulty - 1]), exclusions=exclusions,
        )
        if candidate is not None:
            generate.last_rejections = dict(exclusions)
            return candidate
    generate.last_rejections = dict(exclusions)
    return None


generate.last_rejections = {}


def generate_game(seed, *, split="train", difficulties=None, attempts=DEFAULT_ATTEMPTS, limit=DEFAULT_LIMIT, node_limit=None):
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not selected or any(isinstance(d, bool) or not isinstance(d, Integral) for d in selected):
        raise ValueError("game difficulties must be a nonempty integer sequence")
    selected = tuple(int(d) for d in selected)
    if any(d not in DIFFICULTIES for d in selected) or selected != tuple(sorted(set(selected))):
        raise ValueError(f"game difficulties must be increasing members of {DIFFICULTIES}")
    if node_limit is not None:
        limit = node_limit
    specs = []
    full_game = selected == DIFFICULTIES
    for index, difficulty in enumerate(selected):
        spec = generate(_child_seed(seed, index, difficulty), difficulty, attempts=attempts, limit=limit, split=split)
        if spec is None:
            return None
        spec.update(
            game_seed=int(seed), game_level_index=index,
            game_scope="full_standard" if full_game else "explicit_smoke_subset",
            is_full_standard_game=full_game,
        )
        specs.append(spec)
    return specs


def build_game(specs):
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or len(specs) != len(DIFFICULTIES):
        raise ValueError("BP35 build_game requires exactly nine ordered specs")
    levels = []
    split = None
    for index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if not isinstance(spec, Mapping) or spec.get("difficulty") != difficulty or spec.get("context_index") != index:
            raise ValueError("game specs must preserve native difficulties and contexts 1..9")
        # generate_game adds private whole-game annotations, while the shared
        # collector legitimately builds an ordered game from nine independent
        # generate(...) calls.  When annotations are present they remain
        # strict; their absence cannot invalidate an otherwise certified game.
        if "game_level_index" in spec and spec.get("game_level_index") != index:
            raise ValueError("game_level_index does not match the native context")
        if "game_scope" in spec and spec.get("game_scope") != "full_standard":
            raise ValueError("build_game accepts only a complete full-standard game")
        if "is_full_standard_game" in spec and spec.get("is_full_standard_game") is not True:
            raise ValueError("build_game accepts only a complete full-standard game")
        if split is None:
            split = spec.get("split")
        elif spec.get("split") != split:
            raise ValueError("game specs must use one split")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError("spec does not satisfy BP35 full contract: " + "; ".join(errors))
        levels.append(build_level(spec))
    return levels


def _replay_certificate(spec):
    # Stored display triples are authoritative for the validation replay.
    solution = spec.get("solution")
    if not isinstance(solution, list) or not solution:
        raise ValueError("stored route must be a nonempty action list")
    for raw in solution:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError("solution action must be a triple")
        action, x, y = raw
        if type(action) is not int or action not in names.AVAILABLE_ACTIONS:
            raise ValueError("solution contains an illegal action")
        if action == names.ACTION_CLICK:
            if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
                raise ValueError("click coordinates are invalid")
        elif x is not None or y is not None:
            raise ValueError("non-click actions must have null coordinates")
    env = _context_env(spec)
    base = env.levels_completed
    events = Counter()
    min_hazard_distance = None
    min_click_margin = 64
    for action_index, raw in enumerate(solution):
        frame = np.asarray(env.render())
        if frame.shape != (64, 64) or frame.dtype.kind not in "iu" or frame.min() < 0 or frame.max() > 15:
            raise ValueError("native frame shape, dtype, or palette is invalid")
        events["visual_frames_checked"] += 1
        action, x, y = raw
        if action == names.ACTION_CLICK:
            margin = min(x, y, 63 - x, 62 - y)
            min_click_margin = min(min_click_margin, margin)
            if margin < 3 or len(np.unique(frame[y - 3 : y + 4, x - 3 : x + 4])) < 2:
                raise ValueError("clicked native sprite is clipped or lacks a visible cue")
            events["visible_click_cues"] += 1
            grid = getattr(env.world, names.ATTR_GRID)
            camera_y = int(getattr(getattr(env.world, names.ATTR_CAMERA), names.ATTR_CAMERA_OFFSET)[1])
            cell = grid.hyntnfvpgl(x, y + camera_y)
            before_name = _target_name(env, cell)
        else:
            before_name = None
        before_gravity = bool(getattr(env.world, names.ATTR_GRAVITY_DOWN))
        before_hazard = _hazard_y(env)
        obs = env.perform(action, x, y)
        after_hazard = _hazard_y(env)
        if before_hazard is not None and after_hazard != before_hazard:
            events["hazard_moves"] += 1
        if after_hazard is not None:
            distance = abs(after_hazard - int(getattr(env.world, names.ATTR_PLAYER).grid_y))
            min_hazard_distance = distance if min_hazard_distance is None else min(min_hazard_distance, distance)
        if before_name == names.DESTRUCTIBLE: events["destructible_clicks"] += 1
        elif before_name == names.GROWER: events["growth_clicks"] += 1
        elif before_name == names.BRIDGE_SOLID:
            events["bridge_toggles"] += 1
            events["bridge_solid_to_open"] += 1
        elif before_name == names.BRIDGE_OPEN:
            events["bridge_toggles"] += 1
            events["bridge_open_to_solid"] += 1
        elif before_name == names.GRAVITY_SWITCH: events["gravity_toggles"] += 1
        if bool(getattr(env.world, names.ATTR_GRAVITY_DOWN)) != before_gravity: events["gravity_changes"] += 1
        events[f"action_{action}"] += 1
        if obs.state == GameState.GAME_OVER:
            raise ValueError("stored route loses in the native engine")
        if env.levels_completed > base or obs.state == GameState.WIN:
            if action_index != len(solution) - 1:
                raise ValueError("stored route continues after the first native win")
            events["won"] = 1
            break
    if not events["won"]:
        raise ValueError("stored route does not complete its native level")
    if min_hazard_distance is not None: events["minimum_hazard_distance"] = min_hazard_distance
    events["minimum_click_margin"] = min_click_margin
    return dict(events)


def _expected_profile(difficulty):
    profile = PROFILES[difficulty]
    if difficulty <= 3:
        return {
            "grammar": profile["grammar"], "layers": profile["layers"],
            "hazards": bool(profile["hazards"]),
            "spikes": list(profile["spikes"]),
        }
    if difficulty <= 7:
        return {
            "grammar": profile["grammar"], "layers": profile["layers"],
            "passes": profile["passes"], "gate": profile["gate"],
            "spikes": list(profile["spikes"]),
        }
    return {
        "grammar": "growth_bridge", "growth": True, "gravity": True, "bridge": True,
        "destructible": difficulty == 9,
        "spikes": list(profile["spikes"]),
    }


def _structure_errors(spec, difficulty):
    errors = []
    if not _typed_equal(spec.get("profile"), _expected_profile(difficulty)):
        errors.append("stored tier profile differs from the calibrated profile")
    if not _typed_equal(spec.get("legend"), names.LEGEND) or not _typed_equal(
        spec.get("groups"), names.GROUPS
    ):
        errors.append("native legend or collision groups differ from BP35")
    if spec.get("kind") != names.GENERATED_KIND:
        errors.append("generated level kind is invalid")
    rows = spec.get("rows_bottom_up")
    construction = spec.get("construction")
    if not isinstance(construction, Mapping) or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return errors + ["construction certificate is missing"]

    ranges = STRUCTURAL_RANGES[difficulty]
    installed = _installed_counts(spec)
    observed = {"height": len(rows), **installed}
    for key, (minimum, maximum) in ranges.items():
        value = observed.get(key)
        if type(value) is not int or not minimum <= value <= maximum:
            errors.append(f"{key} is outside the calibrated close-reference range")

    # The versioned grammar is deterministic from the bound seed and attempt.
    # Rebuilding it here catches metadata-only certificates, unreachable
    # padding, hidden/background collision ink, and mutated active topology.
    seed, attempt = spec.get("seed"), spec.get("attempt")
    if type(seed) is int and seed >= 0 and type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS:
        expected = _draft(seed, difficulty, attempt)
        for key in ("rows_bottom_up", "profile", "construction"):
            if not _typed_equal(spec.get(key), expected.get(key)):
                errors.append(f"{key} differs from the versioned tier grammar")
        expected_probes = copy.deepcopy(expected.get("_counterfactuals", []))
        if not _typed_equal(spec.get("constraint_probe_plan"), expected_probes):
            errors.append("constraint probe plan differs from the versioned tier grammar")
    else:
        errors.append("versioned tier grammar cannot be rebuilt from seed and attempt")
    return errors


def validate_full_standard(spec, curriculum_entry):
    errors = []
    if not isinstance(spec, Mapping):
        return ["generated spec must be an object"]
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be an object"]
    difficulty = curriculum_entry.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty must be in 1..9"]
    if not _typed_equal(curriculum_entry.get("context_index"), difficulty - 1) or not _typed_equal(
        curriculum_entry.get("search_work"), SEARCH_WORK[difficulty - 1]
    ):
        errors.append("curriculum row differs from the calibrated tier")
    expected = {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION, "quality_version": QUALITY_VERSION,
        "geometry_version": GEOMETRY_VERSION, "source_id": SOURCE_ID,
        "source_sha256": SOURCE_SHA256,
        "source": "generated_only", "difficulty": difficulty, "context_index": difficulty - 1,
        "verification_level_index": difficulty - 1, "search_limit": SEARCH_WORK[difficulty - 1],
        "search_truncated": False, "planner_exact": True, "engine_verified": True,
        "engine_win": True, "context_engine_verified": True, "levels_completed": 1,
        "native_action_limit": names.native_action_limit(difficulty),
    }
    for key, value in expected.items():
        if not _typed_equal(spec.get(key), value):
            errors.append(f"{key} is missing or inconsistent")
    if len(upstream().levels) != len(DIFFICULTIES) or tuple(DIFFICULTIES) != tuple(range(1, len(upstream().levels) + 1)):
        errors.append("declared curriculum no longer matches the native official level count")
    errors.extend(_structure_errors(spec, difficulty))
    seed, requested_seed, attempt = spec.get("seed"), spec.get("requested_seed"), spec.get("attempt")
    if (
        type(seed) is not int
        or seed < 0
        or not _typed_equal(requested_seed, seed)
        or type(attempt) is not int
        or not 0 <= attempt < MAX_ATTEMPTS
        or not _typed_equal(
            spec.get("effective_seed"), _effective_seed(seed, difficulty, attempt)
        )
    ):
        errors.append("requested/effective seed or attempt mapping is invalid")
    if spec.get("split") not in SPLITS or not _typed_equal(
        spec.get("effective_split"), spec.get("split")
    ):
        errors.append("split declaration is invalid")
    try:
        build_level(spec)
        raw, d4, gameplay = geometry_identities(spec)
        native_initial_d4 = _native_initial_d4_sha256(_context_env(spec))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"geometry cannot be rebuilt: {type(exc).__name__}: {exc}")
    else:
        if not _typed_equal(spec.get("geometry_sha256"), raw) or not _typed_equal(
            spec.get("geometry_d4_sha256"), d4
        ) or not _typed_equal(spec.get("gameplay_sha256"), gameplay):
            errors.append("stored identities differ from recomputation")
        if not _typed_equal(spec.get("geometry_split"), geometry_split(d4)) or not _typed_equal(
            spec.get("split"), geometry_split(d4)
        ):
            errors.append("geometry partition differs from requested split")
        if not _typed_equal(spec.get("native_initial_d4_sha256"), native_initial_d4):
            errors.append("native initial entity-grid identity differs from recomputation")
        if native_initial_d4 in _official_native_d4_identities():
            errors.append("generated native initial entity-grid duplicates an official level")
    solution = spec.get("solution")
    if not isinstance(solution, list) or not _typed_equal(
        spec.get("context_solution"), solution
    ) or not _typed_equal(spec.get("solution_length"), len(solution)):
        errors.append("stored shared solutions are missing or inconsistent")
    elif len(solution) >= names.native_action_limit(difficulty) or not _typed_equal(
        spec.get("budget_slack"), names.native_action_limit(difficulty) - len(solution)
    ):
        errors.append("solution exceeds or misstates the native budget")
    elif len(solution) < MIN_ACTIONS[difficulty - 1]:
        errors.append("solution is below the calibrated route-depth floor")
    if (
        not _typed_equal(spec.get("search_expanded"), spec.get("solution_length"))
        or not _typed_equal(spec.get("search_generated"), spec.get("solution_length"))
        or not _typed_equal(
            spec.get("planner_optimality"), "not claimed; constructive positive witness"
        )
        or spec.get("witness_optimality_claimed") is not False
    ):
        errors.append("constructive search accounting or optimality disclaimer is invalid")
    exclusions = spec.get("generation_exclusions")
    if not isinstance(exclusions, Mapping) or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in (exclusions.items() if isinstance(exclusions, Mapping) else ())):
        errors.append("bounded rejection counts are malformed")
    try:
        installed = _installed_counts(spec)
        events = _replay_certificate(spec)
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
        errors.append(f"stored route replay failed: {type(exc).__name__}: {exc}")
    else:
        if not _typed_equal(spec.get("installed_mechanics"), installed):
            errors.append("installed mechanic census differs from recomputation")
        if not _typed_equal(spec.get("solution_mechanics"), events):
            errors.append("solution mechanic census differs from native replay")
        if not _profile_events_ok(difficulty, events, installed):
            errors.append("route does not meet the tier mechanic-use profile")
        probe_plan = spec.get("constraint_probe_plan")
        probes = _constraint_probe_results(spec, solution, probe_plan if isinstance(probe_plan, list) else ())
        if probes is None or not _typed_equal(spec.get("constraint_probes"), probes):
            errors.append("spike constraint probes differ from native replay")
        required_probes = {"down_spike_loss"} if difficulty >= 2 else set()
        if difficulty in (5, 9):
            required_probes.add("up_spike_loss")
        if probes is not None and any(not probes.get(key) for key in required_probes):
            errors.append("tier spike constraint probe is missing")
        if difficulty >= 8:
            strategy = _growth_dependency_evidence(spec, solution)
            if not _typed_equal(spec.get("strategy_probes"), strategy):
                errors.append("growth dependency probes differ from native replay")
            if (
                strategy.get("win")
                or strategy.get("truncated")
                or strategy.get("growth_removed_certified_route_wins")
            ):
                errors.append("growth is not causally required by the bounded native probes")
        if difficulty in (6, 7):
            chamber = _gravity_chamber_certificate(spec)
            if not _typed_equal(spec.get("chamber_structure"), chamber):
                errors.append("three-chamber structure differs from native all-row recomputation")
            if not chamber.get("valid"):
                errors.append("three-chamber structure has an unguarded or mutable partition")
            lattice = _gravity_dependency_evidence(spec, solution)
            if not _typed_equal(spec.get("lattice_strategy_probes"), lattice):
                errors.append("gravity-lattice dependency probes differ from native replay")
            if (
                lattice.get("truncated")
                or lattice.get("old_six_action_shortcut_wins")
                or lattice.get("initial_final_column_shortcut_wins")
                or lattice.get("remote_ascent_shortcut_wins")
                or lattice.get("underside_zero_closure_shortcut_wins")
                or lattice.get("underside_shifted_shortcut_wins")
                or lattice.get("certified_without_gravity_changes_wins")
                or lattice.get("certified_without_bridge_closures_wins")
                or lattice.get("gravity_actions_removed", 0) < 2
                or lattice.get("closure_actions_removed", 0) < 1
                or not lattice.get("premature_reversal_loses")
                or lattice.get("trap_ablated_premature_reversal_loses")
                or (lattice.get("initial_static_adjacent_search") or {}).get("truncated")
                or not (lattice.get("initial_static_adjacent_search") or {}).get("frontier_exhausted")
                or (lattice.get("initial_static_adjacent_search") or {}).get("win")
                or any(
                    row.get("reached_target_column") or row.get("win")
                    for key in ("arrival_transfers", "ascent_transfers")
                    for row in lattice.get(key, ())
                    if isinstance(row, Mapping)
                )
            ):
                errors.append("gravity and bridge relations are bypassable in bounded native probes")
        shortcut = _greedy_deletion_probe(spec, solution)
        if not _typed_equal(spec.get("shortcut_probe"), shortcut):
            errors.append("native shortcut probe differs from recomputation")
        if (
            shortcut.get("truncated")
            or shortcut.get("reduced_length", -1) < MIN_ACTIONS[difficulty - 1]
            or not _profile_events_ok(
                difficulty, shortcut.get("reduced_solution_mechanics") or {}, installed
            )
        ):
            errors.append("native shortcut probe falls below the tier mechanic/depth gate")
    proof = spec.get("proof")
    if not isinstance(proof, Mapping):
        errors.append("nested proof is missing")
    else:
        for key in ("source_id", "source_sha256", "seed", "requested_seed", "effective_seed", "attempt", "difficulty", "context_index", "split", "geometry_sha256", "geometry_d4_sha256", "gameplay_sha256", "native_initial_d4_sha256", "solution_length", "solution_mechanics", "constraint_probe_plan", "constraint_probes", "strategy_probes", "lattice_strategy_probes", "chamber_structure", "shortcut_probe", "native_action_limit", "budget_slack", "search_limit", "search_truncated", "engine_verified", "engine_win", "levels_completed"):
            if not _typed_equal(proof.get(key), spec.get(key)):
                errors.append(f"proof.{key} does not mirror the top-level certificate")
        if proof.get("exact") is not True or proof.get("unsupported") is not False or proof.get("witness_optimality_claimed") is not False:
            errors.append("nested proof status or optimality disclaimer is invalid")
    return errors


def replays_to_completion(spec):
    try:
        _replay_certificate(spec)
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError):
        return False
    return True
