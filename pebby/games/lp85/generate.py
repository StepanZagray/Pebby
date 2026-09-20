"""Full eight-tier procedural LP85 generator with native certificates.

Generation uses only aggregate measurements from the shipped levels. Fresh
cycle geometry, token assignments, controls, and targets are sampled for every
candidate. Admission requires reference-profile structure, an exact shortest
goal-projection witness, mechanic-use evidence, the requested canonical split,
and replay in the native level index used by the full game.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
import hashlib
import json
import random

from arcengine import GameState, Level

from . import names
from .env import Env, official_levels, upstream
from .plan import search, solution_mechanics
from .reference_profiles import (
    DIFFICULTIES,
    MECHANICS_INVENTORY_VERSION,
    PROFILES,
    QUALITY_PROFILE_VERSION,
    profile_errors,
)


SOURCE_ID = "lp85-305b61c3"
FORMAT = "pebby.lp85.full-level.v2"
GENERATOR_VERSION = 3
GEOMETRY_VERSION = "lp85-action-d4-three-way-v2"
GAMEPLAY_IDENTITY_VERSION = "lp85-native-transition-identity-v2"
PRESENTATION_IDENTITY_VERSION = "lp85-visible-presentation-v1"
SPLITS = ("train", "validation", "test")
DEFAULT_ATTEMPTS = 48
GOAL_KIND = {
    "normal": (names.GOAL_TOKEN, names.TARGET_MARKER),
    "alternate": (names.ALT_GOAL_TOKEN, names.ALT_TARGET_MARKER),
}


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": SOURCE_ID,
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": [
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": PROFILES[difficulty]["search_work"],
        }
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "pebby/games/lp85/reference_profiles.py:REFERENCE",
        "solution_mechanics": "pebby/games/lp85/plan.py:solution_mechanics",
        "native_budget": "pebby/games/lp85/generate.py:verify",
        "context_engine_replay": "pebby/games/lp85/generate.py:_context_env",
        "novelty_split": "pebby/games/lp85/generate.py:geometry_d4_sha256",
        "bounded_rejections": "pebby/games/lp85/generate.py:generate.last_report",
    },
    "caveats": (
        "One shipped level calibrates each tier; tolerances are engineering bounds, not confidence intervals.",
        "The bounded generated grammar and finite audits do not establish population equivalence.",
        "D4 canonicalization rejects transformed copies but is not a general graph-isomorphism proof.",
        "Generation proves projected shortestness and native replay; publication validation does not independently re-search stored optimality.",
    ),
}


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _rectangle_cycle(x, y, width, height):
    result = [[column, y] for column in range(x, x + width)]
    result.extend([[x + width - 1, row] for row in range(y + 1, y + height)])
    result.extend([[column, y + height - 1] for column in range(x + width - 2, x - 1, -1)])
    result.extend([[x, row] for row in range(y + height - 2, y, -1)])
    return result


def _vary_path(path, rng):
    path = [list(point) for point in path]
    offset = rng.randrange(len(path))
    path = path[offset:] + path[:offset]
    if rng.randrange(2):
        path.reverse()
    return path


def _rect_candidates(width, height, cell_width, cell_height):
    return [
        _rectangle_cycle(x, y, width, height)
        for y in range(1, cell_height - height)
        for x in range(1, cell_width - width)
    ]


def _overlap_rectangles(rng, grid, dimensions, desired):
    cell_width, cell_height = grid[0] // names.GRID_STEP, grid[1] // names.GRID_STEP
    first_width, first_height = dimensions[0]
    first_candidates = _rect_candidates(first_width, first_height, cell_width, cell_height)
    rng.shuffle(first_candidates)
    for first in first_candidates:
        paths = [first]
        candidates_by_index = []
        for width, height in dimensions[1:]:
            values = _rect_candidates(width, height, cell_width, cell_height)
            rng.shuffle(values)
            candidates_by_index.append(values)

        def place(index):
            if index == len(candidates_by_index):
                return True
            candidate_index = index + 1
            for candidate in candidates_by_index[index]:
                valid = True
                for previous_index, previous in enumerate(paths):
                    key = tuple(sorted((previous_index, candidate_index)))
                    expected = desired.get(key, 0)
                    if len({tuple(point) for point in previous} & {tuple(point) for point in candidate}) != expected:
                        valid = False
                        break
                if valid:
                    paths.append(candidate)
                    if place(index + 1):
                        return True
                    paths.pop()
            return False

        if place(0):
            return [_vary_path(path, rng) for path in paths]
    raise ValueError("bounded rectangle placement could not meet the overlap profile")


def _site(position, effects):
    return {
        "position": [int(position[0]), int(position[1])],
        "effects": [
            {"group": str(group), "direction": str(direction)}
            for group, direction in effects
        ],
    }


def _simple_controls(groups, grid):
    width, height = grid
    spacing = max(6, (height - 8) // max(1, len(groups)))
    sites = []
    for index, group in enumerate(groups):
        y = min(height - 6, 3 + index * spacing)
        sites.append(_site((1, y), ((group, "L"),)))
        sites.append(_site((width - 4, y), ((group, "R"),)))
    return sites


def _tier_topology(rng, difficulty):
    """Sample fresh geometry matching one official mechanic composition."""
    if difficulty == 1:
        grid = (32, 19)
        width = rng.choice((6, 7))
        height = rng.choice((4, 5))
        x = rng.randint(2, 3 if width == 6 else 2)
        y = rng.randint(1, 2 if height == 4 else 1)
        cycles = [{"group": "A", "path": _vary_path(_rectangle_cycle(x, y, width, height), rng)}]
        controls = _simple_controls(("A",), grid)
        # Keep the tutorial close to the official five-click shortest route,
        # while requiring the learner to inspect both the direction and the
        # distance encoded by this particular puzzle.  These distances are
        # strictly below half of every sampled 16..20-cell cycle, so the
        # opposite control cannot tie the intended shortest route.
        signed_distance = rng.choice((-6, -5, -4, 4, 5, 6))
        scramble = [0 if signed_distance < 0 else 1] * abs(signed_distance)
        kinds = ("normal",)
    elif difficulty == 2:
        grid = (41, 41)
        paths = _overlap_rectangles(rng, grid, ((9, 6), (4, 3), (4, 3)), {(0, 1): 2, (0, 2): 2})
        cycles = [{"group": group, "path": path} for group, path in zip(("A", "B", "C"), paths)]
        controls = _simple_controls(("A", "B", "C"), grid)
        scramble = [1, 5, 1, 1, 1, 5, 5, 5]
        kinds = ("normal", "normal")
    elif difficulty == 3:
        grid = (39, 31)
        paths = _overlap_rectangles(rng, grid, ((5, 5), (5, 5)), {(0, 1): 2})
        cycles = [{"group": group, "path": path} for group, path in zip(("A", "B"), paths)]
        controls = _simple_controls(("A", "B"), grid)
        scramble = [2] * 10 + [0] * 6
        kinds = ("normal", "alternate")
    elif difficulty == 4:
        grid = (57, 57)
        paths = _overlap_rectangles(rng, grid, ((7, 5), (7, 5)), {(0, 1): 4})
        cycles = [{"group": group, "path": path} for group, path in zip(("A", "B"), paths)]
        controls = []
        ys = (5, 17, 29, 41)
        xs = (6, 18, 30, 42)
        controls.extend(_site((1, y), (("A", "L"),)) for y in ys)
        controls.extend(_site((53, y), (("A", "R"),)) for y in ys)
        controls.extend(_site((x, 1), (("B", "L"),)) for x in xs)
        controls.extend(_site((x, 52), (("B", "R"),)) for x in xs)
        scramble = [8] * 4 + [0] * 8
        kinds = ("normal", "alternate")
    elif difficulty == 5:
        grid = (27, 32)
        outer = _rectangle_cycle(2, 1, 7, 5)
        outer.insert(rng.randrange(1, len(outer)), [5, 3])
        outer = _vary_path(outer, rng)
        offset = rng.randrange(len(outer))
        inner = [outer[(offset + index) % len(outer)] for index in range(5)]
        cycles = [
            {"group": "A", "path": outer},
            {"group": "B", "path": _vary_path(inner, rng)},
        ]
        controls = _simple_controls(("A", "B"), grid)
        scramble = [1, 1, 2, 0, 3, 0, 0, 0, 0]
        kinds = ("normal", "normal")
    elif difficulty == 6:
        grid = (60, 64)
        groups = tuple("ABCDEFGHI")
        origins = [(2 + 6 * column, 2 + 6 * row) for row in range(3) for column in range(3)]
        rng.shuffle(origins)
        paths = {}
        for group, (x, y) in zip(groups, origins):
            paths[group] = _vary_path(_rectangle_cycle(x, y, 3, 3), rng)
        numeric = []
        number = 1
        for cluster in (("A", "B", "C"), ("D", "E", "F"), ("G", "H", "I")):
            offsets = [rng.randrange(8) for _ in cluster]
            for index in range(8):
                path = [paths[group][(index + offset) % 8] for group, offset in zip(cluster, offsets)]
                numeric.append({"group": str(number), "path": _vary_path(path, rng)})
                number += 1
        satellites = ((18, 3), (18, 9), (18, 15))
        for group, carrier, satellite in zip(("25", "26", "27"), ("C", "F", "I"), satellites):
            numeric.append({"group": group, "path": _vary_path([paths[carrier][rng.randrange(8)], list(satellite)], rng)})
        cycles = [{"group": group, "path": path} for group, path in paths.items()] + numeric
        stacks = [
            tuple((str(value), "R") for value in range(1, 9)),
            tuple((str(value), "R") for value in range(9, 17)),
            tuple((str(value), "R") for value in range(17, 25)),
            tuple((str(value), "R") for value in range(25, 28)),
            tuple((group, "R") for group in "ABC"),
            tuple((group, "R") for group in "DEF"),
            tuple((group, "R") for group in "GHI"),
        ]
        stacks = [tuple(rng.sample(list(stack), len(stack))) for stack in stacks]
        controls = [
            _site((1, 10), stacks[0]), _site((56, 10), stacks[1]),
            _site((1, 46), stacks[2]), _site((56, 46), stacks[3]),
            _site((15, 60), stacks[4]), _site((27, 60), stacks[5]),
            _site((39, 60), stacks[6]),
        ]
        scramble = [0, 0, 1, 1, 4, 4, 5, 5, 5, 5, 2, 2, 6, 6, 6, 6, 6, 6, 3]
        kinds = ("normal", "normal", "normal")
    elif difficulty == 7:
        grid = (48, 36)
        a = _vary_path(_rectangle_cycle(rng.randint(5, 7), 3, 3, 3), rng)
        a_shared = a[rng.randrange(len(a))]
        b_unique = [[3, 7], [4, 7]]
        b = _vary_path([a_shared] + b_unique, rng)
        c = _vary_path([b_unique[-1], [3, 9], [4, 9], [5, 9]], rng)
        d = _vary_path([[10, 8], [11, 8], [11, 9], [10, 9]], rng)
        cycles = [
            {"group": "A", "path": a}, {"group": "B", "path": b},
            {"group": "C", "path": c}, {"group": "D", "path": d},
        ]
        controls = [
            _site((19, 31), (("A", "L"), ("D", "L"))),
            _site((25, 31), (("D", "R"), ("A", "R"))),
            _site((1, 6), (("B", "L"),)),
            _site((44, 6), (("B", "R"),)),
        ]
        scramble = [1, 3, 0, 2, 0]
        kinds = ("normal", "normal")
    else:
        grid = (63, 63)
        f = _vary_path(_rectangle_cycle(2, 2, 5, 4), rng)
        e = _vary_path(_rectangle_cycle(8, 3, 5, 5), rng)
        d = _vary_path(_rectangle_cycle(14, 5, 5, 4), rng)
        a = _vary_path([f[index] for index in rng.sample(range(len(f)), 2)], rng)
        b_offset = rng.randrange(len(e))
        b = _vary_path([e[(b_offset + index) % len(e)] for index in range(6)], rng)
        d_offset = rng.randrange(len(d))
        c = _vary_path([d[(d_offset + index) % len(d)] for index in range(7)], rng)
        cycles = [
            {"group": "A", "path": a}, {"group": "B", "path": b},
            {"group": "C", "path": c}, {"group": "D", "path": d},
            {"group": "E", "path": e}, {"group": "F", "path": f},
        ]
        controls = [
            _site((59, 8), (("A", "L"),)), _site((59, 14), (("A", "R"),)),
            _site((59, 23), (("B", "L"),)), _site((59, 29), (("B", "R"),)),
            _site((59, 38), (("C", "L"),)), _site((59, 44), (("C", "R"),)),
            _site((25, 58), (("D", "L"), ("E", "L"), ("F", "L"))),
            _site((35, 58), (("F", "R"), ("E", "R"), ("D", "R"))),
        ]
        scramble = [3, 3, 5, 5, 6]
        kinds = ("normal", "normal", "normal")
    return {
        "grid_size": [*grid],
        "cycles": cycles,
        "controls": controls,
        "goal_kinds": kinds,
        "scramble": scramble,
    }


def _effect_tuple(site):
    return tuple((effect["group"], effect["direction"]) for effect in site["effects"])


def _move_cell(cell, effects, paths):
    point = tuple(cell)
    for group, direction in effects:
        path = paths[group]
        try:
            index = path.index(point)
        except ValueError:
            continue
        point = path[(index + (1 if direction == "R" else -1)) % len(path)]
    return point


def _goal_candidates(topology, difficulty):
    paths = {cycle["group"]: tuple(map(tuple, cycle["path"])) for cycle in topology["cycles"]}
    active = {group for site in topology["controls"] for group, _ in _effect_tuple(site)}
    cells = sorted(set().union(*(set(paths[group]) for group in active)))
    if difficulty == 2:
        overlap = [cell for cell in cells if sum(cell in set(path) for path in paths.values()) > 1]
        return overlap + [cell for cell in cells if cell not in overlap]
    if difficulty == 6:
        return list(paths["A"] + paths["D"] + paths["G"])
    if difficulty == 7:
        passive = set(paths["C"])
        return [cell for cell in cells if cell in passive] + cells
    if difficulty == 8:
        return list(paths["A"] + paths["B"] + paths["C"])
    return cells


def _assign_goals(rng, topology, difficulty):
    paths = {cycle["group"]: tuple(map(tuple, cycle["path"])) for cycle in topology["cycles"]}
    effects = [_effect_tuple(site) for site in topology["controls"]]
    sequence = [effects[index] for index in topology["scramble"]]
    kinds = topology["goal_kinds"]
    candidates = _goal_candidates(topology, difficulty)
    if difficulty == 3:
        # Select a real shortest-state depth around the official depth 16.
        # If a sampled topology has no qualifying state at this exact depth,
        # this candidate fails and the outer bounded attempt loop resamples it.
        desired_depth = rng.choice((15, 16, 17))
        memberships = Counter(point for path in paths.values() for point in path)
        overlap_cells = [point for point, count in memberships.items() if count > 1]
        starts_to_try = [
            (first, second)
            for first in overlap_cells + candidates
            for second in candidates
            if first != second
        ]
        rng.shuffle(starts_to_try)
        starts_to_try.sort(key=lambda state: state[0] not in overlap_cells)
        for start in starts_to_try:
            frontier = deque([start])
            depth = {start: 0}
            masks = {start: 0}
            crossed = {start: False}
            while frontier:
                state = frontier.popleft()
                current_depth = depth[state]
                if (
                    current_depth == desired_depth
                    and masks[state] == 0b11
                    and crossed[state]
                    and all(source != target for source, target in zip(start, state))
                ):
                    return [
                        {"kind": kind, "start": list(source), "target": list(target)}
                        for kind, source, target in zip(kinds, start, state)
                    ]
                if current_depth >= desired_depth:
                    continue
                for effect in effects:
                    successor = tuple(_move_cell(point, effect, paths) for point in state)
                    if successor == state or successor in depth:
                        continue
                    effect_groups = {group for group, _ in effect}
                    mask = masks[state]
                    if "A" in effect_groups:
                        mask |= 1
                    if "B" in effect_groups:
                        mask |= 2
                    transition_crossed = crossed[state] or any(
                        before != after
                        and (memberships[before] > 1 or memberships[after] > 1)
                        for before, after in zip(state, successor)
                    )
                    depth[successor] = current_depth + 1
                    masks[successor] = mask
                    crossed[successor] = transition_crossed
                    frontier.append(successor)
        raise ValueError(
            f"bounded tier-3 search found no shared-cycle target at depth {desired_depth}"
        )
    if difficulty == 6:
        # Pick a target state whose first breadth-first witness uses every one
        # of the seven compound controls. This is constructive difficulty, not
        # an official route: geometry and starts were freshly sampled above.
        desired_depth = rng.choice((18, 19, 20))
        for _ in range(24):
            start = tuple(sorted((
                rng.choice(paths["A"]), rng.choice(paths["D"]), rng.choice(paths["G"]),
            )))
            if len(set(start)) != 3:
                continue
            frontier = deque([start])
            depth = {start: 0}
            masks = {start: 0}
            target = None
            while frontier:
                state = frontier.popleft()
                current_depth = depth[state]
                if (
                    current_depth == desired_depth
                    and masks[state] == (1 << len(effects)) - 1
                    and not set(state) & set(start)
                ):
                    target = state
                    break
                if current_depth >= desired_depth:
                    continue
                for effect_index, effect in enumerate(effects):
                    successor = tuple(sorted(_move_cell(point, effect, paths) for point in state))
                    if successor == state or successor in depth:
                        continue
                    depth[successor] = current_depth + 1
                    masks[successor] = masks[state] | (1 << effect_index)
                    frontier.append(successor)
            if target is not None:
                return [
                    {"kind": "normal", "start": list(source), "target": list(destination)}
                    for source, destination in zip(start, target)
                ]
        raise ValueError(
            f"bounded tier-6 search found no all-control target at depth {desired_depth}"
        )
    if difficulty == 7:
        ordered_effects = [
            _effect_tuple(site)
            for site in sorted(topology["controls"], key=lambda site: (site["position"][1], site["position"][0]))
        ]
        active_cells = sorted(set().union(*(set(paths[group]) for group in ("A", "B", "D"))))
        passive_cells = set(paths["C"])
        preferred = [cell for cell in active_cells if cell in passive_cells]
        starts_to_try = []
        for first in preferred + active_cells:
            for second in active_cells:
                if first != second:
                    starts_to_try.append(tuple(sorted((first, second))))
        rng.shuffle(starts_to_try)
        starts_to_try.sort(key=lambda state: not bool(set(state) & passive_cells))
        for start in starts_to_try:
            frontier = deque([start])
            depth = {start: 0}
            parent = {start: None}
            for_target = None
            while frontier:
                state = frontier.popleft()
                current_depth = depth[state]
                if 4 <= current_depth <= 7 and len(set(start) & set(state)) == 1:
                    route = []
                    cursor = state
                    while parent[cursor] is not None:
                        previous, effect_index = parent[cursor]
                        route.append(effect_index)
                        cursor = previous
                    route.reverse()
                    used_groups = {
                        group for effect_index in route for group, _ in ordered_effects[effect_index]
                    }
                    common = next(iter(set(start) & set(state)))
                    trace = start
                    displaced = False
                    passive_crossed = False
                    for effect_index in route:
                        effect = ordered_effects[effect_index]
                        moved = tuple(_move_cell(point, effect, paths) for point in trace)
                        passive_crossed |= any(
                            before != after and (before in passive_cells or after in passive_cells)
                            for before, after in zip(trace, moved)
                        )
                        trace = tuple(sorted(moved))
                        displaced |= common not in trace
                    if (
                        len(set(route)) >= 3
                        and used_groups == {"A", "B", "D"}
                        and passive_crossed
                        and displaced
                    ):
                        for_target = state
                        break
                if current_depth >= 8:
                    continue
                for effect_index, effect in enumerate(ordered_effects):
                    successor = tuple(sorted(_move_cell(point, effect, paths) for point in state))
                    if successor == state or successor in depth:
                        continue
                    depth[successor] = current_depth + 1
                    parent[successor] = (state, effect_index)
                    frontier.append(successor)
            if for_target is not None:
                return [
                    {"kind": "normal", "start": list(source), "target": list(destination)}
                    for source, destination in zip(start, for_target)
                ]
        raise ValueError("bounded tier-7 search found no passive/stacked target assignment")
    for _ in range(1_024 if difficulty == 7 else 256):
        starts = rng.sample(candidates, len(kinds))
        targets = []
        for start in starts:
            point = tuple(start)
            for effect in sequence:
                point = _move_cell(point, effect, paths)
            targets.append(point)
        if len(set(starts)) != len(starts) or len(set(targets)) != len(targets):
            continue
        if all(start == target for start, target in zip(starts, targets)):
            continue
        initial_satisfied = sum(
            start in {targets[index] for index, target_kind in enumerate(kinds) if target_kind == kind}
            for start, kind in zip(starts, kinds)
        )
        if initial_satisfied != (1 if difficulty == 7 else 0):
            continue
        if difficulty == 7:
            passive = set(paths["C"])
            crossed = False
            for start in starts:
                point = tuple(start)
                for effect in sequence:
                    after = _move_cell(point, effect, paths)
                    crossed |= point != after and (point in passive or after in passive)
                    point = after
            if not crossed:
                continue
        return [
            {"kind": kind, "start": list(start), "target": list(target)}
            for kind, start, target in zip(kinds, starts, targets)
        ]
    raise ValueError("bounded goal assignment could not express the tier constraints")


def _map_material(spec):
    return {
        "grid_size": spec["grid_size"],
        "cycles": spec["cycles"],
        "controls": spec["controls"],
        "goals": spec["goals"],
    }


def _map_id(spec):
    payload = json.dumps(_map_material(spec), sort_keys=True, separators=(",", ":")).encode()
    return "pebby-lp85-full-" + hashlib.sha256(payload).hexdigest()[:24]


def _minimum_cycle_rotation(path):
    """Canonicalize only a cycle's arbitrary numbered origin, not its direction."""
    path = tuple(path)
    return min(path[index:] + path[:index] for index in range(len(path)))


def _action_relevant_material(spec, transform=lambda point: point, *, normalize_translation=False):
    """Return the executable transition geometry, excluding passive decoration.

    Native completion uses unordered start and target sets within each goal
    kind. Group names and the numbered origin of a cyclic path are construction
    labels. A compound control's effect order is native behavior and is kept.
    """
    cycles = {
        cycle["group"]: tuple((int(x) * 3, int(y) * 3) for x, y in cycle["path"])
        for cycle in spec["cycles"]
    }
    active_groups = {
        effect["group"]
        for site in spec["controls"]
        for effect in site["effects"]
    }
    control_points = [tuple(map(int, site["position"])) for site in spec["controls"]]
    goals_by_kind = {
        kind: {
            "starts": [
                (int(goal["start"][0]) * 3, int(goal["start"][1]) * 3)
                for goal in spec["goals"]
                if goal["kind"] == kind
            ],
            "targets": [
                (int(goal["target"][0]) * 3, int(goal["target"][1]) * 3)
                for goal in spec["goals"]
                if goal["kind"] == kind
            ],
        }
        for kind in GOAL_KIND
    }
    anchors = [point for group in active_groups for point in cycles[group]]
    anchors.extend(control_points)
    for points in goals_by_kind.values():
        anchors.extend(points["starts"])
        anchors.extend(points["targets"])
    transformed = [transform(point) for point in anchors]
    left = min((point[0] for point in transformed), default=0) if normalize_translation else 0
    top = min((point[1] for point in transformed), default=0) if normalize_translation else 0

    def project(point):
        x, y = transform(point)
        return x - left, y - top

    canonical_paths = {
        group: _minimum_cycle_rotation(tuple(project(point) for point in cycles[group]))
        for group in active_groups
    }
    return {
        "cycles": sorted(canonical_paths.values()),
        "controls": sorted(
            (
                project(point),
                tuple(
                    (canonical_paths[effect["group"]], effect["direction"])
                    for effect in site["effects"]
                ),
            )
            for point, site in zip(control_points, spec["controls"])
        ),
        "goals": [
            (
                kind,
                sorted(project(point) for point in goals_by_kind[kind]["starts"]),
                sorted(project(point) for point in goals_by_kind[kind]["targets"]),
            )
            for kind in GOAL_KIND
        ],
    }


def _json_sha256(material):
    payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(material):
    return json.dumps(material, sort_keys=True, separators=(",", ":"))


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def geometry_d4_sha256(spec):
    """Canonical action-relevant geometry under translation and D4 transforms."""
    variants = []
    for swap in (False, True):
        for sign_x in (-1, 1):
            for sign_y in (-1, 1):
                def transform(point):
                    x, y = point
                    return sign_x * (y if swap else x), sign_y * (x if swap else y)

                material = _action_relevant_material(
                    spec, transform, normalize_translation=True
                )
                variants.append(json.dumps(material, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def geometry_sha256(spec):
    """Hash exact-coordinate action geometry with construction labels removed."""
    return _json_sha256(_action_relevant_material(spec))


def gameplay_sha256(spec):
    """Hash the exact native transition puzzle, including its click budget."""
    return _json_sha256({
        "action_geometry": _action_relevant_material(spec),
        "step_budget": spec.get("step_budget"),
    })


def presentation_sha256(spec):
    """Hash visible placement separately from transition/split identity."""
    goals = {
        kind: {
            "starts": sorted(
                tuple(map(int, goal["start"]))
                for goal in spec["goals"]
                if goal["kind"] == kind
            ),
            "targets": sorted(
                tuple(map(int, goal["target"]))
                for goal in spec["goals"]
                if goal["kind"] == kind
            ),
        }
        for kind in GOAL_KIND
    }
    material = {
        "grid_size": tuple(map(int, spec["grid_size"])),
        "controls": sorted(
            (
                tuple(map(int, site["position"])),
                tuple(effect["direction"] for effect in site["effects"]),
            )
            for site in spec["controls"]
        ),
        "goals": goals,
        "fillers": sorted(
            (tuple(map(int, filler["cell"])), filler["prototype"])
            for filler in spec.get("fillers", ())
        ),
    }
    return _json_sha256(material)


def geometry_partition(fingerprint):
    return SPLITS[int(fingerprint, 16) % len(SPLITS)]


def _official_geometry_hashes():
    cached = getattr(_official_geometry_hashes, "cached", None)
    if cached is not None:
        return cached
    hashes = set()
    for level in official_levels():
        env = Env([level])
        compiled = env.compiled_maps()
        cycles = []
        for group, entry in compiled.items():
            positions = entry[names.MAP_POSITIONS]
            cycles.append({
                "group": group,
                "path": [
                    [positions[index].x, positions[index].y]
                    for index in range(1, entry[names.MAP_LENGTH] + 1)
                ],
            })
        by_position = defaultdict(list)
        for sprite in env.level._sprites:
            if sprite.tags and sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX):
                _, group, direction = sprite.tags[0].split("_")
                by_position[(sprite.x, sprite.y)].append({"group": group, "direction": direction})
        controls = [
            {"position": list(position), "effects": effects}
            for position, effects in by_position.items()
        ]
        goals = []
        for kind, goal_tag, marker_tag in (
            ("normal", names.TAG_GOAL, names.TAG_TARGET_MARKER),
            ("alternate", names.TAG_ALT_GOAL, names.TAG_ALT_TARGET_MARKER),
        ):
            starts = sorted((sprite.x // 3, sprite.y // 3) for sprite in env.level.get_sprites_by_tag(goal_tag))
            targets = sorted(
                ((sprite.x + 1) // 3, (sprite.y + 1) // 3)
                for sprite in env.level.get_sprites_by_tag(marker_tag)
            )
            goals.extend(
                {"kind": kind, "start": list(start), "target": list(target)}
                for start, target in zip(starts, targets)
            )
        hashes.add(geometry_d4_sha256({"cycles": cycles, "controls": controls, "goals": goals}))
    _official_geometry_hashes.cached = frozenset(hashes)
    return _official_geometry_hashes.cached


def _validated_components(spec):
    if not isinstance(spec, Mapping) or spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be 1..8")
    grid = spec.get("grid_size")
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or len(grid) != 2:
        raise ValueError("grid_size must contain width and height")
    width, height = _integer(grid[0], "grid width"), _integer(grid[1], "grid height")
    if width < 8 or height < 8:
        raise ValueError("grid is too small")
    cycles = []
    groups = set()
    for raw in spec.get("cycles", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("cycle entries must be objects")
        group = raw.get("group")
        if not isinstance(group, str) or not group or group in groups:
            raise ValueError("cycle groups must be distinct nonempty strings")
        groups.add(group)
        path = []
        for point in raw.get("path", ()):
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
                raise ValueError("cycle points must contain x and y")
            x, y = _integer(point[0], "cycle x"), _integer(point[1], "cycle y")
            if not (0 <= x and x * 3 + 2 < width and 0 <= y and y * 3 + 2 < height):
                raise ValueError("cycle point lies outside the native grid")
            path.append((x, y))
        if len(path) < 2 or len(set(path)) != len(path):
            raise ValueError("each cycle needs at least two distinct points")
        cycles.append({"group": group, "path": path})
    if not cycles:
        raise ValueError("at least one cycle is required")
    controls = []
    rectangles = []
    for raw in spec.get("controls", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("control sites must be objects")
        position = raw.get("position")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 2:
            raise ValueError("control position must contain x and y")
        x, y = _integer(position[0], "control x"), _integer(position[1], "control y")
        if not (0 <= x <= width - 3 and 0 <= y <= height - 4):
            raise ValueError("control does not fit the grid")
        if any(x < bx + 3 and bx < x + 3 and y < by + 4 and by < y + 4 for bx, by in rectangles):
            raise ValueError("distinct control sites must not partially overlap")
        rectangles.append((x, y))
        effects = []
        for effect in raw.get("effects", ()):
            if not isinstance(effect, Mapping):
                raise ValueError("control effects must be objects")
            group, direction = effect.get("group"), effect.get("direction")
            if group not in groups or direction not in ("L", "R"):
                raise ValueError("control references an unknown group or direction")
            effects.append((group, direction))
        if not effects or len(set(effects)) != len(effects):
            raise ValueError("a control site needs distinct movement effects")
        controls.append({"position": (x, y), "effects": effects})
    if not controls:
        raise ValueError("at least one control is required")
    union = set().union(*(set(cycle["path"]) for cycle in cycles))
    goals = []
    starts = set()
    for raw in spec.get("goals", ()):
        if not isinstance(raw, Mapping) or raw.get("kind") not in GOAL_KIND:
            raise ValueError("goal kind must be normal or alternate")
        start, target = raw.get("start"), raw.get("target")
        if not all(
            isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) == 2
            for point in (start, target)
        ):
            raise ValueError("goal start/target must contain x and y")
        start = (_integer(start[0], "goal start x"), _integer(start[1], "goal start y"))
        target = (_integer(target[0], "goal target x"), _integer(target[1], "goal target y"))
        if start not in union or target not in union or start in starts:
            raise ValueError("goals require distinct starts and mapped targets")
        starts.add(start)
        goals.append({"kind": raw["kind"], "start": start, "target": target})
    for kind in GOAL_KIND:
        selected = [goal for goal in goals if goal["kind"] == kind]
        if len({goal["target"] for goal in selected}) != len(selected):
            raise ValueError("same-kind target markers must be distinct")
    fillers = spec.get("fillers")
    if not isinstance(fillers, Sequence) or isinstance(fillers, (str, bytes)):
        raise ValueError("fillers must be a sequence")
    filler_by_cell = {}
    for raw in fillers:
        if not isinstance(raw, Mapping) or raw.get("prototype") not in names.TILE_PROTOTYPES:
            raise ValueError("invalid filler prototype")
        cell = raw.get("cell")
        if not isinstance(cell, Sequence) or isinstance(cell, (str, bytes)) or len(cell) != 2:
            raise ValueError("filler cell must contain x and y")
        point = (_integer(cell[0], "filler x"), _integer(cell[1], "filler y"))
        if point in filler_by_cell:
            raise ValueError("duplicate filler cell")
        filler_by_cell[point] = raw["prototype"]
    if set(filler_by_cell) != union - starts:
        raise ValueError("fillers must occupy every mapped non-goal cell exactly once")
    return difficulty, (width, height), cycles, controls, goals, filler_by_cell


def structural_metrics(spec):
    difficulty, grid, cycles, controls, goals, _ = _validated_components(spec)
    paths = {cycle["group"]: set(cycle["path"]) for cycle in cycles}
    union = set().union(*paths.values())
    overlaps = []
    nested = 0
    groups = tuple(paths)
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            shared = paths[left] & paths[right]
            if shared:
                overlaps.append(len(shared))
                nested += paths[left] < paths[right] or paths[right] < paths[left]
    directions = defaultdict(set)
    tags = Counter()
    signatures = set()
    for site in controls:
        signature = tuple(site["effects"])
        signatures.add(signature)
        for group, direction in signature:
            directions[group].add(direction)
            tags[(group, direction)] += 1
    by_kind_starts = defaultdict(set)
    by_kind_targets = defaultdict(set)
    for goal in goals:
        by_kind_starts[goal["kind"]].add(goal["start"])
        by_kind_targets[goal["kind"]].add(goal["target"])
    return {
        "difficulty": difficulty,
        "grid_width": grid[0], "grid_height": grid[1],
        "cycle_count": len(cycles),
        "cycle_lengths": tuple(sorted(len(path) for path in paths.values())),
        "union_cells": len(union),
        "overlap_pairs": len(overlaps),
        "overlap_incidences": sum(overlaps),
        "max_pair_overlap": max(overlaps, default=0),
        "nested_pairs": nested,
        "control_sprites": sum(len(site["effects"]) for site in controls),
        "control_sites": len(controls),
        "effect_signatures": len(signatures),
        "stacked_sites": sum(len(site["effects"]) > 1 for site in controls),
        "max_stack": max(len(site["effects"]) for site in controls),
        "duplicate_control_extras": sum(count - 1 for count in tags.values()),
        "controlled_groups": len(directions),
        "passive_cycles": len(paths) - len(directions),
        "right_only_groups": sum(value == {"R"} for value in directions.values()),
        "bidirectional_groups": sum(value == {"L", "R"} for value in directions.values()),
        "normal_goals": len(by_kind_starts["normal"]),
        "alternate_goals": len(by_kind_starts["alternate"]),
        "initially_satisfied_goals": sum(
            len(by_kind_starts[kind] & by_kind_targets[kind]) for kind in GOAL_KIND
        ),
    }


def _raw_maps(cycles, grid):
    cells_wide = (grid[0] + names.GRID_STEP - 1) // names.GRID_STEP
    cells_high = (grid[1] + names.GRID_STEP - 1) // names.GRID_STEP
    result = {}
    for cycle in cycles:
        numbered = [[-1 for _ in range(cells_wide)] for _ in range(cells_high)]
        for number, (x, y) in enumerate(cycle["path"], 1):
            numbered[y][x] = number
        result[cycle["group"]] = numbered
    return result


def _button_prototype(group, direction):
    tag = f"button_{group}_{direction}"
    for sprite in upstream().sprites.values():
        if sprite.tags and sprite.tags[0] == tag:
            return sprite
    raise ValueError(f"vendored LP85 has no prototype for {tag}")


def build_level(spec):
    """Reconstruct one native level from a validated JSON-safe full spec."""
    difficulty, grid, cycles, controls, goals, fillers = _validated_components(spec)
    if spec.get("map_id") != _map_id(spec):
        raise ValueError("map_id does not match the full gameplay definition")
    sprites = []
    for site in controls:
        for group, direction in site["effects"]:
            sprites.append(_button_prototype(group, direction).clone().set_position(*site["position"]))
    prototypes = upstream().sprites
    for goal in goals:
        _, marker_name = GOAL_KIND[goal["kind"]]
        x, y = goal["target"]
        sprites.append(prototypes[marker_name].clone().set_position(x * 3 - 1, y * 3 - 1))
    start_by_cell = {goal["start"]: goal for goal in goals}
    union = sorted(set().union(*(set(cycle["path"]) for cycle in cycles)))
    for cell in union:
        if cell in start_by_cell:
            prototype = GOAL_KIND[start_by_cell[cell]["kind"]][0]
        else:
            prototype = fillers[cell]
        sprites.append(prototypes[prototype].clone().set_position(cell[0] * 3, cell[1] * 3))
    return Level(
        sprites=sprites,
        grid_size=grid,
        data={
            names.KEY_STEPS: int(spec["step_budget"]),
            names.KEY_LEVEL_NAME: spec["map_id"],
            names.KEY_GENERATED_MAP: _raw_maps(cycles, grid),
        },
        name=f"generated-lp85-full-d{difficulty}-s{spec.get('seed', 0)}",
    )


def _draft(seed, difficulty, attempt):
    seed = _integer(seed, "seed")
    difficulty = _integer(difficulty, "difficulty")
    attempt = _integer(attempt, "attempt")
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be 1..8")
    rng = random.Random(f"lp85-full-v{GENERATOR_VERSION}:{seed}:{difficulty}:{attempt}")
    topology = _tier_topology(rng, difficulty)
    goals = _assign_goals(rng, topology, difficulty)
    union = sorted({tuple(point) for cycle in topology["cycles"] for point in cycle["path"]})
    starts = {tuple(goal["start"]) for goal in goals}
    fillers = [
        {"cell": list(cell), "prototype": rng.choice(names.TILE_PROTOTYPES)}
        for cell in union if cell not in starts
    ]
    spec = {
        "format": FORMAT,
        "source": "generated_only",
        "source_id": SOURCE_ID,
        "generator_version": GENERATOR_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "gameplay_identity_version": GAMEPLAY_IDENTITY_VERSION,
        "presentation_identity_version": PRESENTATION_IDENTITY_VERSION,
        "seed": seed,
        "generation_attempt": attempt,
        "difficulty": difficulty,
        "training_context_index": difficulty - 1,
        "grid_size": topology["grid_size"],
        "step_budget": PROFILES[difficulty]["step_budget"],
        "cycles": topology["cycles"],
        "controls": topology["controls"],
        "goals": goals,
        "fillers": fillers,
        "official_mechanics_complete": True,
        "omitted_mechanics": [],
    }
    spec["map_id"] = _map_id(spec)
    spec["structural_metrics"] = structural_metrics(spec)
    fingerprint = geometry_d4_sha256(spec)
    spec.update(
        geometry_d4_sha256=fingerprint,
        geometry_sha256=geometry_sha256(spec),
        gameplay_sha256=gameplay_sha256(spec),
        presentation_sha256=presentation_sha256(spec),
        geometry_split=geometry_partition(fingerprint),
        official_geometry_copy=fingerprint in _official_geometry_hashes(),
    )
    return spec


def _context_env(level, context_index):
    env = Env([level.clone() for _ in DIFFICULTIES])
    env.set_level(context_index)
    return env


def verify(spec, node_limit=None):
    """Return ``(accepted, reason)`` after exact contextual search and replay."""
    try:
        difficulty = _integer(spec.get("difficulty"), "difficulty")
        metrics = structural_metrics(spec)
        errors = profile_errors(spec, metrics, require_proof=False)
        if errors:
            return None, "profile:" + errors[0]
        if spec.get("official_geometry_copy"):
            return None, "official_geometry_copy"
        if spec.get("geometry_split") != spec.get("split"):
            return None, "geometry_split_mismatch"
        level = build_level(spec)
        work = PROFILES[difficulty]["search_work"]
        if node_limit is not None and _integer(node_limit, "node_limit") != work:
            raise ValueError(f"node_limit for difficulty {difficulty} must equal contract search_work {work}")
        context_index = difficulty - 1
        search_env = _context_env(level, context_index)
        result = search(search_env, limit=int(spec["step_budget"]), node_limit=work)
        if result.unsupported or not result.exact:
            return None, "unsupported_exact_model"
        if result.truncated:
            return None, "search_truncated"
        if result.actions is None:
            return None, "proven_unsolvable"
        actions = tuple(result.actions)
        mechanics = solution_mechanics(_context_env(level, context_index), actions)

        replay_env = _context_env(level, context_index)
        before_score = replay_env.levels_completed
        if replay_env.level_index != context_index or replay_env.steps_left != spec["step_budget"]:
            return None, "context_initialization_mismatch"
        observation = None
        for action_index, action in enumerate(actions):
            observation = replay_env.perform(*action)
            completed = replay_env.levels_completed > before_score or observation.state == GameState.WIN
            if completed != (action_index == len(actions) - 1):
                return None, "native_replay_transition_mismatch"
        if observation is None:
            return None, "empty_witness"
        terminal_remaining = int(spec["step_budget"]) - max(0, len(actions) - 1)
        accepted = dict(spec)
        accepted.update(
            solution=[list(action) for action in actions],
            solution_length=len(actions),
            optimal_actions=len(actions),
            solution_mechanics=mechanics,
            engine_verified=True,
            search_exact=True,
            search_truncated=False,
            verification_level_index=context_index,
            native_budget={
                "initial": int(spec["step_budget"]),
                "charged_nonterminal_actions": max(0, len(actions) - 1),
                "winning_click_is_uncharged": True,
                "remaining_before_level_transition": terminal_remaining,
            },
            context_engine_replay={
                "level_index": context_index,
                "levels_completed_delta": replay_env.levels_completed - before_score,
                "native_state": observation.state.value,
                "fresh_engine": True,
            },
            proof={
                "kind": "exact-guarded-goal-projection-and-contextual-native-replay",
                "expanded": int(result.expanded), "generated": int(result.generated),
                "search_work": int(work), "action_count": len(actions),
                "seed": int(spec["seed"]), "difficulty": difficulty, "split": spec["split"],
                "generator_version": GENERATOR_VERSION,
                "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
                "quality_profile_version": QUALITY_PROFILE_VERSION,
                "geometry_version": GEOMETRY_VERSION,
                "gameplay_identity_version": GAMEPLAY_IDENTITY_VERSION,
                "presentation_identity_version": PRESENTATION_IDENTITY_VERSION,
                "training_context_index": context_index,
                "verification_level_index": context_index,
                "geometry_sha256": spec["geometry_sha256"],
                "geometry_d4_sha256": spec["geometry_d4_sha256"],
                "gameplay_sha256": spec["gameplay_sha256"],
                "presentation_sha256": spec["presentation_sha256"],
                "engine_verified": True, "search_exact": True, "search_truncated": False,
            },
            planner_optimality=(
                "breadth-first shortest path over a fail-closed static-control, fully occupied "
                "active-map goal projection; every control transition and the final route are natively replayed"
            ),
        )
        errors = profile_errors(accepted, metrics, require_proof=True)
        if errors:
            return None, "profile:" + errors[0]
        return accepted, "accepted"
    except (IndexError, KeyError, TypeError, ValueError) as error:
        return None, f"invalid_candidate:{type(error).__name__}:{error}"


def generate(seed, difficulty=1, attempts=DEFAULT_ATTEMPTS, node_limit=None, *, split, record_rejection=None):
    """Generate one split-bound, contextual, replay-certified LP85 level."""
    seed = _integer(seed, "seed")
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts")
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be 1..8")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if node_limit is not None and (
        isinstance(node_limit, bool) or not isinstance(node_limit, int) or not 1 <= node_limit <= 32_000_000
    ):
        raise ValueError("node_limit must be 1..32,000,000")
    exclusions = Counter()
    records = []
    for attempt in range(attempts):
        try:
            candidate = _draft(seed, difficulty, attempt)
            candidate["split"] = split
            if candidate["geometry_split"] != split:
                accepted, reason = None, "geometry_split_mismatch"
            elif candidate["official_geometry_copy"]:
                accepted, reason = None, "official_geometry_copy"
            else:
                accepted, reason = verify(candidate, node_limit=node_limit)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            accepted, reason = None, f"invalid_candidate:{type(error).__name__}:{error}"
        if accepted is not None:
            accepted["generation_exclusions"] = dict(exclusions)
            accepted["proof"]["rejections_before_accept"] = sum(exclusions.values())
            generate.last_report = {
                "seed": seed, "difficulty": difficulty, "split": split,
                "attempts_used": attempt + 1, "accepted": True,
                "rejections": dict(exclusions), "records": records,
            }
            return accepted
        exclusions[reason] += 1
        record = {
            "seed": seed, "difficulty": difficulty, "split": split,
            "attempt": attempt, "reason": reason,
            "generator_version": GENERATOR_VERSION,
            "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        }
        records.append(record)
        if record_rejection is not None:
            record_rejection(dict(record))
    generate.last_report = {
        "seed": seed, "difficulty": difficulty, "split": split,
        "attempts_used": attempts, "accepted": False,
        "rejections": dict(exclusions), "records": records,
    }
    return None


generate.last_report = None


def _child_seed(game_seed, ordinal, difficulty):
    material = f"{SOURCE_ID}:game:{int(game_seed)}:{int(ordinal)}:{int(difficulty)}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split, difficulties=None, attempts=DEFAULT_ATTEMPTS, node_limit=None):
    """Generate a complete increasing-difficulty game, or an explicit reduced sequence."""
    seed = _integer(seed, "seed")
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not selected or any(type(value) is not int or value not in DIFFICULTIES for value in selected):
        raise ValueError("difficulties must be a nonempty sequence drawn from 1..8")
    if tuple(sorted(selected)) != selected or len(set(selected)) != len(selected):
        raise ValueError("difficulties must be strictly increasing")
    specs = []
    for ordinal, difficulty in enumerate(selected):
        child = _child_seed(seed, ordinal, difficulty)
        spec = generate(child, difficulty, attempts=attempts, node_limit=node_limit, split=split)
        if spec is None:
            generate_game.last_report = {
                "seed": seed, "split": split, "failed_ordinal": ordinal,
                "difficulty": difficulty, "child_seed": child,
                "generator_report": generate.last_report,
            }
            return None
        spec["game_seed"] = seed
        spec["game_ordinal"] = ordinal
        spec["game_child_seed"] = child
        specs.append(spec)
    if selected == DIFFICULTIES:
        levels = build_game(specs)
        env = Env(levels)
        for index, spec in enumerate(specs):
            if env.level_index != index or env.levels_completed != index:
                raise RuntimeError("generated full game entered the wrong native context")
            before = env.levels_completed
            for action in spec["solution"]:
                env.perform(*action)
            if env.levels_completed != before + 1:
                raise RuntimeError("stored full-game witness failed native sequential replay")
            spec["proof"]["full_game_replay"] = True
            spec["proof"]["full_game_level_index"] = index
        if env.state != GameState.WIN:
            raise RuntimeError("eight accepted levels did not produce one native game win")
    generate_game.last_report = {
        "seed": seed, "split": split, "difficulties": list(selected),
        "accepted": True, "count": len(specs),
    }
    return specs


generate_game.last_report = None


def validate_full_standard(spec, curriculum_entry):
    """Validate one accepted row against the finalized family contract."""
    errors = []
    if not isinstance(spec, Mapping):
        return ("spec must be a mapping",)
    if not isinstance(curriculum_entry, Mapping):
        return ("curriculum entry must be a mapping",)
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ("difficulty must be an integer in 1..8",)
    expected = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
    for key in ("difficulty", "context_index", "search_work"):
        value = curriculum_entry.get(key)
        if type(value) is not int:
            errors.append(f"curriculum {key} must be an integer")
        elif value != expected[key]:
            errors.append(f"curriculum {key} mismatch")
    if type(spec.get("seed")) is not int:
        errors.append("seed must be an integer")
    if type(spec.get("generation_attempt")) is not int or spec.get("generation_attempt", -1) < 0:
        errors.append("generation attempt must be a nonnegative integer")
    if type(spec.get("training_context_index")) is not int:
        errors.append("training context index must be an integer")
    elif spec.get("training_context_index") != expected["context_index"]:
        errors.append("training context differs from the curriculum")
    if type(spec.get("verification_level_index")) is not int:
        errors.append("verification level index must be an integer")
    elif spec.get("verification_level_index") != expected["context_index"]:
        errors.append("verification context differs from the curriculum")
    if spec.get("source") != "generated_only" or spec.get("source_id") != SOURCE_ID:
        errors.append("generated source identity mismatch")
    if type(spec.get("generator_version")) is not int or spec.get("generator_version") != GENERATOR_VERSION:
        errors.append("generator version mismatch")
    if spec.get("geometry_version") != GEOMETRY_VERSION:
        errors.append("geometry version mismatch")
    if spec.get("gameplay_identity_version") != GAMEPLAY_IDENTITY_VERSION:
        errors.append("gameplay identity version mismatch")
    if spec.get("presentation_identity_version") != PRESENTATION_IDENTITY_VERSION:
        errors.append("presentation identity version mismatch")
    if spec.get("split") not in SPLITS or spec.get("geometry_split") != spec.get("split"):
        errors.append("explicit split/geometry partition mismatch")
    for key in (
        "geometry_sha256", "geometry_d4_sha256", "gameplay_sha256", "presentation_sha256"
    ):
        if not _is_sha256(spec.get(key)):
            errors.append(f"missing {key}")
    if spec.get("official_geometry_copy") is not False:
        errors.append("official geometry novelty rejection is missing")
    if spec.get("omitted_mechanics") != [] or spec.get("official_mechanics_complete") is not True:
        errors.append("full mode declares omitted official mechanics")
    for key, label in (
        ("structural_metrics", "structural metrics"),
        ("solution_mechanics", "solution mechanics"),
        ("native_budget", "native budget"),
        ("context_engine_replay", "context engine replay"),
        ("proof", "proof"),
        ("generation_exclusions", "generation exclusions"),
    ):
        if not isinstance(spec.get(key), Mapping):
            errors.append(f"{label} must be a mapping")
    level = None
    metrics = None
    recomputed_mechanics = None
    replay_facts = None
    recomputed_budget = None
    try:
        metrics = structural_metrics(spec)
        stored_metrics = spec.get("structural_metrics")
        # JSON round-trips tuples as lists, so compare canonical JSON values.
        if _canonical_json(metrics) != _canonical_json(stored_metrics):
            errors.append("stored structural metrics do not recompute")
        expected_d4 = geometry_d4_sha256(spec)
        expected_raw = geometry_sha256(spec)
        expected_gameplay = gameplay_sha256(spec)
        expected_presentation = presentation_sha256(spec)
        if spec.get("geometry_d4_sha256") != expected_d4:
            errors.append("D4 geometry identity does not recompute")
        if spec.get("geometry_sha256") != expected_raw:
            errors.append("raw geometry identity does not recompute")
        if spec.get("gameplay_sha256") != expected_gameplay:
            errors.append("gameplay identity does not recompute")
        if spec.get("presentation_sha256") != expected_presentation:
            errors.append("presentation identity does not recompute")
        if spec.get("geometry_split") != geometry_partition(expected_d4):
            errors.append("canonical geometry partition does not recompute")
        if expected_d4 in _official_geometry_hashes():
            errors.append("generated geometry duplicates an official level under D4")
        errors.extend(profile_errors(spec, metrics, require_proof=True))
        level = build_level(spec)

        actions = spec.get("solution")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or not actions:
            errors.append("solution must be a nonempty action sequence")
        else:
            context_index = expected["context_index"]
            mechanics_env = _context_env(level, context_index)
            recomputed_mechanics = solution_mechanics(mechanics_env, actions)
            if _canonical_json(recomputed_mechanics) != _canonical_json(
                spec.get("solution_mechanics")
            ):
                errors.append("stored solution mechanics do not recompute")
            replay_env = _context_env(level, context_index)
            before_score = replay_env.levels_completed
            if replay_env.level_index != context_index:
                errors.append("native replay initialized at the wrong context")
            if replay_env.steps_left != spec.get("step_budget"):
                errors.append("native replay initialized with the wrong budget")
            observation = None
            early = False
            for action_index, action in enumerate(actions):
                observation = replay_env.perform(*action)
                completed = (
                    replay_env.levels_completed > before_score
                    or observation.state == GameState.WIN
                )
                if completed != (action_index == len(actions) - 1):
                    early = True
                    break
            if early or observation is None or replay_env.levels_completed != before_score + 1:
                errors.append("stored witness does not produce exactly one contextual native win")
            else:
                replay_facts = {
                    "level_index": context_index,
                    "levels_completed_delta": 1,
                    "native_state": observation.state.value,
                    "fresh_engine": True,
                }
                if _canonical_json(replay_facts) != _canonical_json(spec.get("context_engine_replay")):
                    errors.append("contextual native replay evidence does not recompute")
            recomputed_budget = {
                "initial": int(spec["step_budget"]),
                "charged_nonterminal_actions": max(0, len(actions) - 1),
                "winning_click_is_uncharged": True,
                "remaining_before_level_transition": int(spec["step_budget"]) - max(0, len(actions) - 1),
            }
            if _canonical_json(recomputed_budget) != _canonical_json(spec.get("native_budget")):
                errors.append("native budget evidence does not recompute")
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError) as error:
        errors.append(f"malformed full spec: {error}")
    proof = spec.get("proof")
    if isinstance(proof, Mapping):
        if proof.get("kind") != "exact-guarded-goal-projection-and-contextual-native-replay":
            errors.append("proof kind is missing or unsupported")
        for key in (
            "seed", "difficulty", "generator_version", "training_context_index",
            "verification_level_index", "search_work", "action_count", "expanded", "generated",
        ):
            value = proof.get(key)
            if type(value) is not int or (key in ("search_work", "action_count", "expanded", "generated") and value < 0):
                qualifier = "nonnegative " if key in ("search_work", "action_count", "expanded", "generated") else ""
                errors.append(f"proof {key.replace('_', ' ')} must be a {qualifier}integer")
        for key in (
            "split", "mechanics_inventory_version", "quality_profile_version", "geometry_version",
            "gameplay_identity_version", "presentation_identity_version",
        ):
            if not isinstance(proof.get(key), str) or not proof.get(key):
                errors.append(f"proof {key.replace('_', ' ')} must be a nonempty string")
        for key in (
            "geometry_sha256", "geometry_d4_sha256", "gameplay_sha256", "presentation_sha256"
        ):
            if not _is_sha256(proof.get(key)):
                errors.append(f"proof {key.replace('_', ' ')} must be a SHA-256 string")
        for key in ("engine_verified", "search_exact", "search_truncated"):
            if type(proof.get(key)) is not bool:
                errors.append(f"proof {key.replace('_', ' ')} must be a boolean")
        if "rejections_before_accept" in proof:
            value = proof.get("rejections_before_accept")
            if type(value) is not int or value < 0:
                errors.append("proof rejections before accept must be a nonnegative integer")
        if "full_game_replay" in proof and proof.get("full_game_replay") is not True:
            errors.append("proof full game replay must be true when present")
        if "full_game_level_index" in proof and (
            type(proof.get("full_game_level_index")) is not int
            or proof.get("full_game_level_index") != expected["context_index"]
        ):
            errors.append("proof full game level index must match the curriculum when present")
        mirrors = {
            "seed": spec.get("seed"), "difficulty": difficulty, "split": spec.get("split"),
            "generator_version": spec.get("generator_version"),
            "mechanics_inventory_version": spec.get("mechanics_inventory_version"),
            "quality_profile_version": spec.get("quality_profile_version"),
            "geometry_version": spec.get("geometry_version"),
            "gameplay_identity_version": spec.get("gameplay_identity_version"),
            "presentation_identity_version": spec.get("presentation_identity_version"),
            "training_context_index": spec.get("training_context_index"),
            "verification_level_index": spec.get("verification_level_index"),
            "geometry_sha256": spec.get("geometry_sha256"),
            "geometry_d4_sha256": spec.get("geometry_d4_sha256"),
            "gameplay_sha256": spec.get("gameplay_sha256"),
            "presentation_sha256": spec.get("presentation_sha256"),
            "engine_verified": spec.get("engine_verified"),
            "search_exact": spec.get("search_exact"),
            "search_truncated": spec.get("search_truncated"),
        }
        for key, value in mirrors.items():
            if type(proof.get(key)) is not type(value) or proof.get(key) != value:
                errors.append(f"proof does not mirror {key}")
        if proof.get("search_work") != expected["search_work"]:
            errors.append("proof search work differs from curriculum")
    native_budget = spec.get("native_budget")
    if isinstance(native_budget, Mapping):
        for key in ("initial", "charged_nonterminal_actions", "remaining_before_level_transition"):
            if type(native_budget.get(key)) is not int:
                errors.append(f"native budget {key.replace('_', ' ')} must be an integer")
        if type(native_budget.get("initial")) is int and native_budget["initial"] <= 0:
            errors.append("native budget initial must be positive")
        if (
            type(native_budget.get("charged_nonterminal_actions")) is int
            and native_budget["charged_nonterminal_actions"] < 0
        ):
            errors.append("native budget charged actions must be nonnegative")
        if native_budget.get("winning_click_is_uncharged") is not True:
            errors.append("native budget must mark the winning click uncharged")
    replay_evidence = spec.get("context_engine_replay")
    if isinstance(replay_evidence, Mapping):
        for key in ("level_index", "levels_completed_delta"):
            if type(replay_evidence.get(key)) is not int:
                errors.append(f"context engine replay {key.replace('_', ' ')} must be an integer")
        if not isinstance(replay_evidence.get("native_state"), str):
            errors.append("context engine replay native state must be a string")
        if replay_evidence.get("fresh_engine") is not True:
            errors.append("context engine replay must declare a fresh engine")
    solution_value = spec.get("solution")
    solution_count = (
        len(solution_value)
        if isinstance(solution_value, Sequence) and not isinstance(solution_value, (str, bytes))
        else -1
    )
    if type(spec.get("solution_length")) is not int or spec.get("solution_length") != solution_count:
        errors.append("solution length mismatch")
    if type(spec.get("optimal_actions")) is not int or spec.get("optimal_actions") != spec.get("solution_length"):
        errors.append("stored optimal action count differs from the certified witness")
    if isinstance(proof, Mapping) and proof.get("action_count") != spec.get("solution_length"):
        errors.append("proof action count differs from the certified witness")
    if spec.get("engine_verified") is not True or spec.get("search_exact") is not True:
        errors.append("positive exact native certificate is missing")
    if spec.get("search_truncated") is not False:
        errors.append("truncated search cannot certify a row")
    exclusions = spec.get("generation_exclusions")
    if isinstance(exclusions, Mapping):
        if any(
            not isinstance(reason, str)
            or not reason
            or type(count) is not int
            or count < 0
            for reason, count in exclusions.items()
        ):
            errors.append("rejection counts must be nonnegative integers with nonempty reasons")
        elif isinstance(proof, Mapping) and proof.get("rejections_before_accept") != sum(exclusions.values()):
            errors.append("proof rejection count does not match generation exclusions")
    return tuple(dict.fromkeys(errors))


def build_game(specs):
    """Build exactly one ordered native eight-level full-standard episode."""
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
        raise ValueError("specs must be a sequence")
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("full-standard LP85 build_game requires exactly eight specs")
    splits = {spec.get("split") for spec in specs if isinstance(spec, Mapping)}
    if len(splits) != 1:
        raise ValueError("full game specs must use one explicit split")
    levels = []
    child_seeds = []
    seeds = []
    for index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if not isinstance(spec, Mapping) or spec.get("difficulty") != difficulty:
            raise ValueError("full game specs must be ordered difficulties 1..8")
        if (
            type(spec.get("training_context_index")) is not int
            or type(spec.get("verification_level_index")) is not int
            or spec.get("training_context_index") != index
            or spec.get("verification_level_index") != index
        ):
            raise ValueError("full game spec context/index mismatch")
        if type(spec.get("game_ordinal", index)) is not int or spec.get("game_ordinal", index) != index:
            raise ValueError("full game has a duplicate or out-of-order ordinal")
        if type(spec.get("seed")) is not int:
            raise ValueError("full game level seeds must be integers")
        seeds.append(spec["seed"])
        if "game_child_seed" in spec:
            if type(spec["game_child_seed"]) is not int:
                raise ValueError("full game child seeds must be integers")
            child_seeds.append(spec["game_child_seed"])
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"difficulty {difficulty} is not full-standard: {errors[0]}")
        levels.append(build_level(spec))
    if len(set(seeds)) != len(seeds):
        raise ValueError("full game has duplicate level seeds")
    if child_seeds and (len(child_seeds) != len(specs) or len(set(child_seeds)) != len(child_seeds)):
        raise ValueError("full game has missing or duplicate child seeds")
    # Construction itself proves the stored witnesses work in one actual
    # uninterrupted native episode; callers cannot bypass this by validating
    # eight isolated rows and then reordering them.
    replay_env = Env([level.clone() for level in levels])
    for index, spec in enumerate(specs):
        if replay_env.level_index != index or replay_env.levels_completed != index:
            raise ValueError("full game entered the wrong native level context")
        before = replay_env.levels_completed
        for action in spec["solution"]:
            replay_env.perform(*action)
        if replay_env.levels_completed != before + 1:
            raise ValueError(f"difficulty {index + 1} witness failed sequential native replay")
    if replay_env.state != GameState.WIN:
        raise ValueError("complete eight-level game did not reach native WIN")
    return levels
