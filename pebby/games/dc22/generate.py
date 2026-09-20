"""Full six-tier procedural DC22 generation with native certificates.

Drafts combine a random tree-maze navigation prefix with mechanic stages made
from the vendored game's own sprites. The stage route is constructed from the
sampled topology; acceptance still requires replay in the real engine at the
official native context. Constructive routes are not claimed shortest.
"""

from collections import Counter, deque
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from numbers import Integral
import random

from arcengine import Level, Sprite

from . import names
from .env import Env, official_levels, upstream
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    PROFILES,
    QUALITY_VERSION,
    structural_metrics,
)


FORMAT = "pebby.dc22.level.v2"
GENERATOR_VERSION = 6
MECHANICS_VERSION = "dc22-full-mechanics-v4"
GEOMETRY_VERSION = "dc22-tagged-d4-three-way-v3"
DEFAULT_ATTEMPTS = 24
DEFAULT_NODE_LIMIT = max(profile["search_limit"] for profile in PROFILES.values())
SPLITS = ("train", "validation", "test")
SOURCE_ID = "dc22-fdcac232"
MAZE_ORIGIN = (4, 4)
MAZE_PITCH = 4
MAZE_WIDTH = 4
MAZE_HEIGHT = 3


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_VERSION,
    "curriculum": [
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": PROFILES[difficulty]["search_limit"],
        }
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "dc22.md#official-reference-characterization",
        "solution_mechanics": "dc22-native-event-certificate-v2",
        "mechanic_stage_diversity": "dc22-canonical-stage-relations-v1",
        "native_budget": "dc22-context-budget-replay-v1",
        "context_engine_replay": "tests/games/test_dc22_quality.py",
        "novelty_split": GEOMETRY_VERSION,
        "bounded_rejections": "dc22.md#bounded-quality-audit",
        "version_6_closure_review": (
            "dc22-v6 closure: 10 tier-4 rows; 40/40 independent device deletions "
            "both prevented completion and changed the player trace; four actual support "
            "contacts per row; 10/10 late-phase omissions prevented completion; focused "
            "nested malformed inputs returned clean diagnostic lists"
        ),
        "root_integration_review": (
            "root acceptance: three complete six-level native games won in "
            "300/293/300 actions in 7.8444 seconds; current native frames reviewed"
        ),
    },
    "caveats": [
        "all six official action counts are measured positive witnesses, not shortest-route optima",
        "constructive generated witnesses are native-replayed but are not claimed shortest",
        "tier-6 bridge colour routing uses a selectively recoloured source and heterogeneous fixed destinations",
        "D4 identity is conservative and does not hold out graph-isomorphic topology families",
        "the complete 48-row/303-click audit is historical version-5 evidence; current version-6 evidence is scoped",
        "generation uses a finite serial-stage and blueprint grammar",
        "reference calibration uses one scarce official reference per tier with engineering tolerances",
        "tier 1 deliberately has small tutorial diversity",
        "constructive witnesses are nonoptimal and distinguish replay work from search limits",
        "failure-diagnostic persistence is opt-in for callers",
    ],
}


_ALLOWED_PROTOTYPES = {
    "tovemc-plelvb1", "tovemc-plelvb_plong_1", "tovemc-plelvb-p-1",
    "buezna-blrmbx", "buezna-matkhq", "buezna-pueite",
    "drfztmbrixto-1", "drfztmbrixto-2", "drfztmbrixto-3",
    "drfztmbrixto-buezna", "moxubw-plelvb-1", "sprite-6",
    "tewfut1", "tewfut2", "piyqze-buezna-pueite",
    "piyqze-buezna-pueite-1", "piyqze-buezna-refgps-1",
    "crzsjq-1", "crzsjq-up", "crzsjq-dowlja", "crzsjq-lersnf",
    "crzsjq-riidpd", "crzsjq-grawwq", "sprite-10",
    "brixtocrzsjq-1", "brixto-orckhi2", "crzsjq-lersnf-1",
    "crzsjq-riidpd-1", "crzsjq-up-1", "crzsjq-grawwq-1",
    "crzsjq-lersnf-2", "sprite_81", "sprite_81-1", "sprite_81-2",
    "sprite_81-4", "bg", "renrjo-buezna", "tewfutpibpar1", "tewfutpibpar2",
    "coorbs-bg", "coorbs-bg-1", "merged-sprite", "merged-sprite-1",
    "drfztmbrixto-5", "tewfutyefmyf1",
}


def _integer(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if positive and value < 1:
        raise ValueError(f"{label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _component(prototype, x, y, *, extra_tags=()):
    if prototype not in _ALLOWED_PROTOTYPES:
        raise ValueError(f"unsupported generated prototype {prototype!r}")
    value = {"prototype": prototype, "position": [int(x), int(y)]}
    if extra_tags:
        value["extra_tags"] = list(extra_tags)
    return value


def _neighbors(index, width, height):
    x, y = index % width, index // width
    if y:
        yield index - width
    if y + 1 < height:
        yield index + width
    if x:
        yield index - 1
    if x + 1 < width:
        yield index + 1


def _maze(rng, width, height, start):
    seen = {start}
    stack = [start]
    passages = []
    while stack:
        current = stack[-1]
        options = [cell for cell in _neighbors(current, width, height) if cell not in seen]
        if not options:
            stack.pop()
            continue
        nxt = rng.choice(options)
        seen.add(nxt)
        passages.append([min(current, nxt), max(current, nxt)])
        stack.append(nxt)
    return passages


def _adjacency(cell_count, passages):
    graph = [[] for _ in range(cell_count)]
    for left, right in passages:
        graph[left].append(right)
        graph[right].append(left)
    return graph


def _cell_position(cell):
    return (
        MAZE_ORIGIN[0] + (cell % MAZE_WIDTH) * MAZE_PITCH,
        MAZE_ORIGIN[1] + (cell // MAZE_WIDTH) * MAZE_PITCH,
    )


def _cell_route(graph, start, goal):
    parent = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nxt in graph[current]:
            if nxt not in parent:
                parent[nxt] = current
                queue.append(nxt)
    if goal not in parent:
        raise ValueError("maze route is disconnected")
    route = []
    current = goal
    while current is not None:
        route.append(current)
        current = parent[current]
    return list(reversed(route))


def _actions_between(start, goal):
    x, y = start
    gx, gy = goal
    if x != gx and y != gy:
        raise ValueError("constructed route segments must be orthogonal")
    actions = []
    while x != gx:
        action = names.ACTION_RIGHT if gx > x else names.ACTION_LEFT
        actions.append([action, None, None])
        x += 2 if gx > x else -2
    while y != gy:
        action = names.ACTION_DOWN if gy > y else names.ACTION_UP
        actions.append([action, None, None])
        y += 2 if gy > y else -2
    return actions


def _route_actions(points):
    actions = []
    for start, goal in zip(points, points[1:]):
        actions.extend(_actions_between(start, goal))
    return actions


def _click_action(prototype, position, grid_size):
    sprite = upstream().sprites[prototype]
    pixels = sprite.render()
    center = ((sprite.width - 1) / 2, (sprite.height - 1) / 2)
    candidates = [
        (x, y)
        for y in range(sprite.height)
        for x in range(sprite.width)
        if pixels[y, x] >= 0
    ]
    if not candidates:
        raise ValueError(f"click prototype {prototype!r} has no visible pixel")
    dx, dy = min(candidates, key=lambda point: abs(point[0] - center[0]) + abs(point[1] - center[1]))
    y_offset = (names.FRAME_SIZE - int(grid_size[1])) // 2
    return [names.ACTION_CLICK, int(position[0]) + dx, int(position[1]) + dy + y_offset]


def _add_line(floors, start, goal):
    x, y = start
    gx, gy = goal
    floors.add((x, y))
    if x != gx and y != gy:
        raise ValueError("floor lines must be orthogonal")
    while x != gx:
        x += 2 if gx > x else -2
        floors.add((x, y))
    while y != gy:
        y += 2 if gy > y else -2
        floors.add((x, y))


def _add_annex(rng, floors, anchor, origin, width, height):
    """Add a connected random side maze without extending the winning route."""
    passages = _maze(rng, width, height, rng.randrange(width * height))

    def point(cell):
        return (
            origin[0] + (cell % width) * MAZE_PITCH,
            origin[1] + (cell // width) * MAZE_PITCH,
        )

    cells = [point(cell) for cell in range(width * height)]
    floors.update(cells)
    for left, right in passages:
        lx, ly = point(left)
        rx, ry = point(right)
        floors.add(((lx + rx) // 2, (ly + ry) // 2))
    closest = min(cells, key=lambda value: abs(value[0] - anchor[0]) + abs(value[1] - anchor[1]))
    bend = (anchor[0], closest[1])
    _add_line(floors, anchor, bend)
    _add_line(floors, bend, closest)


def _maze_prefix(rng, stage_start):
    exit_row = rng.randrange(MAZE_HEIGHT)
    exit_cell = exit_row * MAZE_WIDTH + MAZE_WIDTH - 1
    provisional = rng.randrange(MAZE_WIDTH * MAZE_HEIGHT)
    passages = _maze(rng, MAZE_WIDTH, MAZE_HEIGHT, provisional)
    graph = _adjacency(MAZE_WIDTH * MAZE_HEIGHT, passages)
    distances = {exit_cell: 0}
    queue = deque([exit_cell])
    while queue:
        current = queue.popleft()
        for nxt in graph[current]:
            if nxt not in distances:
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    farthest = max(distances.values())
    start_cell = rng.choice(sorted(cell for cell, distance in distances.items() if distance == farthest))
    cell_route = _cell_route(graph, start_cell, exit_cell)
    points = [_cell_position(cell) for cell in cell_route]
    exit_position = points[-1]
    if exit_position[0] != stage_start[0] and exit_position[1] != stage_start[1]:
        points.append((exit_position[0], stage_start[1]))
    points.append(stage_start)
    floors = {_cell_position(cell) for cell in range(MAZE_WIDTH * MAZE_HEIGHT)}
    for left, right in passages:
        lx, ly = _cell_position(left)
        rx, ry = _cell_position(right)
        floors.add(((lx + rx) // 2, (ly + ry) // 2))
    for left, right in zip(points[-3:], points[-2:]):
        _add_line(floors, left, right)
    return {
        "maze_start": start_cell,
        "maze_exit": exit_cell,
        "maze_passages": passages,
        "start": list(points[0]),
        "prefix_actions": _route_actions(points),
        "floors": floors,
    }


def _draft(seed, difficulty, attempt):
    profile = PROFILES[difficulty]
    rng = random.Random(f"{MECHANICS_VERSION}:{seed}:{difficulty}:{attempt}")
    grid_size = tuple(profile["grid_size"])
    components = []
    interaction_probes = {}

    def click(prototype, position):
        return _click_action(prototype, position, grid_size)

    if difficulty == 1:
        stage = (16, 20)
        base = _maze_prefix(rng, stage)
        floors = base["floors"]
        gate_count = rng.choice((1, 2))
        floors.add((18, 20))
        floors.add((24, 20))
        goal = (30, 20)
        _add_line(floors, (30 if gate_count == 2 else 24, 20), goal)
        button = (46, 8)
        components.extend([
            _component("tovemc-plelvb1", 20, 20),
            _component("buezna-blrmbx", *button),
        ])
        if gate_count == 2:
            components.append(_component("tovemc-plelvb1", 26, 20))
        solution = base["prefix_actions"] + _actions_between(stage, (18, 20))
        solution += [click("buezna-blrmbx", button)] + _actions_between((18, 20), goal)
        fall_probe = base["prefix_actions"] + _actions_between(stage, (18, 20))
        fall_probe += [click("buezna-blrmbx", button)] + _actions_between((18, 20), (20, 20))
        fall_probe += [click("buezna-blrmbx", button)]
        stage_blueprint = {"sequential_surface_gates": gate_count}
    elif difficulty == 2:
        stage = (16, 20)
        base = _maze_prefix(rng, stage)
        floors = base["floors"]
        _add_annex(rng, floors, stage, (16, 28), 3, 3)
        turn_gate = bool(rng.randrange(2))
        if turn_gate:
            floors.update(((28, 20), (28, 30)))
            goal = (28, 30)
        else:
            _add_line(floors, (28, 20), (32, 20))
            goal = (32, 20)
        button = (46, 8)
        components.extend([
            _component("piyqze-buezna-pueite", *stage),
            _component("tovemc-plelvb_plong_1", 18, 20),
            _component("buezna-pueite", *button),
        ])
        if turn_gate:
            components.append(_component("tovemc-plelvb-p-1", 28, 22))
        solution = base["prefix_actions"] + [click("buezna-pueite", button)]
        solution += _actions_between(stage, (28, 20))
        if turn_gate:
            solution += _actions_between((28, 20), goal)
        else:
            solution += _actions_between((28, 20), goal)
        fall_probe = base["prefix_actions"] + [click("buezna-pueite", button)]
        fall_probe += _actions_between(stage, (18, 20)) + [click("buezna-pueite", button)]
        stage_blueprint = {"unlocked_gate_topology": "L-double" if turn_gate else "straight-single"}
    elif difficulty == 3:
        stage = (16, 20)
        base = _maze_prefix(rng, stage)
        floors = base["floors"]
        long_gate = bool(rng.randrange(2))
        gate_prototype = "tovemc-plelvb_plong_1" if long_gate else "tovemc-plelvb-p-1"
        bridge_source = (34, 20) if long_gate else (28, 20)
        floors.add((22, 20))
        _add_line(floors, (20, 32), (32, 32))
        b_button, d_button, c_button = (46, 6), (46, 16), (46, 28)
        components.extend([
            _component("tovemc-plelvb1", 18, 20), _component("buezna-blrmbx", *b_button),
            _component("piyqze-buezna-pueite", 22, 20),
            _component(gate_prototype, 24, 20), _component("buezna-pueite", *d_button),
            _component("tewfut1", *bridge_source), _component("tewfut2", 18, 32),
            _component("buezna-matkhq", *c_button),
        ])
        solution = base["prefix_actions"] + [click("buezna-blrmbx", b_button)]
        solution += _actions_between(stage, (22, 20)) + [click("buezna-pueite", d_button)]
        solution += _actions_between((22, 20), bridge_source) + [click("buezna-matkhq", c_button)]
        solution += _actions_between((18, 32), (32, 32))
        fall_probe = base["prefix_actions"] + [click("buezna-blrmbx", b_button)]
        fall_probe += _actions_between(stage, (18, 20)) + [click("buezna-blrmbx", b_button)]
        goal = (32, 32)
        stage_blueprint = {"unlocked_gate_extent": "long" if long_gate else "short"}
    elif difficulty == 4:
        route_y = rng.choice((12, 16))
        stage = (16, route_y)
        base = _maze_prefix(rng, stage)
        floors = base["floors"]
        _add_annex(rng, floors, (4, route_y), (4, 24), 3, 3)
        expansion_clicks = 4
        moving_steps = rng.choice((2, 3, 4, 5))
        moving_last = 22 + 2 * moving_steps
        right_connector = moving_last + 2
        lower_exit = moving_last - 6
        second_expander_x = lower_exit - 16
        goal = (second_expander_x - 2, 42)
        floors.add((34, route_y))
        _add_line(floors, (right_connector, 30), (right_connector, 34))
        _add_line(floors, (lower_exit, 34), (lower_exit, 42))
        floors.add(goal)
        e_button, f_button, c_button = (52, 5), (46, 16), (46, 29)
        components.extend([
            _component("drfztmbrixto-1", 18, route_y),
            _component("drfztmbrixto-5", second_expander_x, 42),
            _component("drfztmbrixto-buezna", *e_button),
            _component("tewfut1", 34, route_y), _component("tewfut2", 16, 30),
            _component("buezna-matkhq", *c_button),
            _component("moxubw-plelvb-1", 18, 30),
            _component("moxubw-plelvb-1", 18, 34),
            _component("sprite-6", *f_button),
        ])
        solution = list(base["prefix_actions"])
        solution += [click("drfztmbrixto-buezna", e_button)] * expansion_clicks
        solution += _actions_between(stage, (34, route_y)) + [click("buezna-matkhq", c_button)]
        solution += _actions_between((16, 30), (22, 30))
        for x in range(24, moving_last + 1, 2):
            solution.append(click("sprite-6", f_button))
            solution += _actions_between((x - 2, 30), (x, 30))
        solution += _actions_between((moving_last, 30), (right_connector, 30))
        solution += _actions_between((right_connector, 30), (right_connector, 34))
        solution += _actions_between((right_connector, 34), (lower_exit, 34))
        solution += _actions_between((lower_exit, 34), (lower_exit, 42))
        solution += [click("drfztmbrixto-buezna", e_button)] * 4
        solution += _actions_between((lower_exit, 42), goal)
        fall_probe = list(base["prefix_actions"])
        fall_probe += [click("drfztmbrixto-buezna", e_button)] * expansion_clicks
        fall_probe += _actions_between(stage, (24, route_y))
        fall_probe += [click("drfztmbrixto-buezna", e_button)]
        stage_blueprint = {
            "expansion_click_depth": expansion_clicks,
            "paired_expansion_cycles": 2,
            "expansion_initial_phases": [1, 5],
            "expansion_route_row": route_y,
            "second_expansion_row": 42,
            "moving_surface_steps": moving_steps,
            "joint_moving_surfaces": 2,
            "paired_route": "expand-outward/move-out-and-back/expand-return",
        }
    elif difficulty == 5:
        stage = (16, 34)
        base = _maze_prefix(rng, stage)
        floors = base["floors"]
        carry_right = rng.choice((1, 2, 3))
        bridge_landing = 6 + 4 * carry_right
        floors.add((bridge_landing + 2, 42))
        moving_base = bridge_landing + 8
        moving_steps = rng.choice(tuple(range(2, min(4, (36 - moving_base - 4) // 2) + 1)))
        moving_last = moving_base + 4 + 2 * moving_steps
        goal = (moving_last + 2, 42)
        floors.add(goal)
        d_button, c_button, e_button, f_button = (46, 5), (46, 14), (54, 21), (44, 19)
        grab, up, down, left, right = (48, 34), (48, 28), (58, 28), (43, 28), (53, 28)
        components.extend([
            _component("piyqze-buezna-pueite", *stage),
            _component("tovemc-plelvb_plong_1", 18, 34), _component("buezna-pueite", *d_button),
            _component("drfztmbrixto-1", 28, 34),
            _component("drfztmbrixto-buezna", *e_button),
            _component("tewfut1", 36, 34), _component("tewfut2", bridge_landing, 42),
            _component("buezna-matkhq", *c_button), _component("crzsjq-1", 2, 46),
            _component("sprite-10", 8, 44), _component("crzsjq-grawwq", *grab),
            _component("crzsjq-up", *up), _component("crzsjq-dowlja", *down),
            _component("crzsjq-lersnf", *left), _component("crzsjq-riidpd", *right),
            _component("moxubw-plelvb-1", moving_base, 42),
            _component("sprite-6", *f_button),
        ])
        solution = base["prefix_actions"] + [click("buezna-pueite", d_button)]
        solution += _actions_between(stage, (26, 34))
        solution += [click("drfztmbrixto-buezna", e_button)] * 4
        solution += _actions_between((26, 34), (36, 34))
        solution += [click("crzsjq-grawwq", grab)] + [click("crzsjq-up", up)] * 3
        solution += [click("crzsjq-riidpd", right)] * carry_right
        solution += [click("buezna-matkhq", c_button)]
        solution += _actions_between((bridge_landing, 42), (moving_base + 4, 42))
        for x in range(moving_base + 6, moving_last + 1, 2):
            solution.append(click("sprite-6", f_button))
            solution += _actions_between((x - 2, 42), (x, 42))
        solution += _actions_between((moving_last, 42), goal)
        fall_probe = base["prefix_actions"] + [click("buezna-pueite", d_button)]
        fall_probe += _actions_between(stage, (18, 34)) + [click("buezna-pueite", d_button)]
        stage_blueprint = {
            "crusher_carry_right_steps": carry_right,
            "carried_object_required_gap": [bridge_landing + 4, bridge_landing + 6],
            "expansion_click_depth": 4,
            "moving_surface_steps": moving_steps,
        }
    else:
        stage = (6, 28)
        base = _maze_prefix(rng, stage)
        floors = base["floors"]
        _add_annex(rng, floors, (10, 52), (2, 40), 3, 6)
        _add_annex(rng, floors, (28, 52), (28, 40), 3, 6)
        _add_line(floors, stage, (16, 28))
        track_order = rng.choice(("right-up", "up-right"))
        moving_steps = rng.choice((3, 4, 5))
        moving_exit = 16 + 2 * moving_steps
        _add_line(floors, (moving_exit + 2, 52), (32, 52))
        c_button, colour_button, f_button = (46, 5), (46, 14), (44, 45)
        left, right, up, grab, down = (45, 24), (53, 24), (49, 20), (47, 34), (49, 38)
        special_tags = (names.TAG_BRIDGE_COLOR_CYCLE, "d")
        right_sensor_x, up_sensor_x = ((12, 14) if track_order == "right-up" else (14, 12))
        track_cells = ((28, 32), (28, 28)) if track_order == "right-up" else ((24, 28), (28, 28))
        components.extend([
            _component("piyqze-buezna-refgps-1", 8, 28),
            _component("piyqze-buezna-pueite-1", 10, 28),
            _component("sprite_81", 6, 28), _component("sprite_81-2", right_sensor_x, 28),
            _component("sprite_81-1", up_sensor_x, 28), _component("sprite_81-4", 16, 28),
            _component("brixtocrzsjq-1", 20, 28), _component("brixto-orckhi2", 14, 22),
            *[_component("bg", *position) for position in track_cells],
            _component("crzsjq-lersnf-1", *left), _component("crzsjq-riidpd-1", *right),
            _component("crzsjq-up-1", *up), _component("crzsjq-grawwq-1", *grab),
            _component("crzsjq-lersnf-2", *down), _component("buezna-matkhq", *c_button),
            _component("renrjo-buezna", *colour_button),
            _component("tewfutpibpar1", 38, 28, extra_tags=special_tags),
            _component("tewfutpibpar1", 34, 4),
            _component("tewfutyefmyf1", 10, 52),
            _component("moxubw-plelvb-1", 12, 52), _component("sprite-6", *f_button),
        ])
        solution = base["prefix_actions"] + _actions_between(stage, (8, 28))
        solution += [click("crzsjq-grawwq-1", grab)]
        if track_order == "right-up":
            solution += _actions_between((8, 28), (12, 28)) + [click("crzsjq-riidpd-1", right)]
            solution += _actions_between((12, 28), (14, 28)) + [click("crzsjq-up-1", up)]
        else:
            solution += _actions_between((8, 28), (12, 28)) + [click("crzsjq-up-1", up)]
            solution += _actions_between((12, 28), (14, 28)) + [click("crzsjq-riidpd-1", right)]
        solution += [click("buezna-matkhq", c_button)]
        solution += [click("renrjo-buezna", colour_button)] * 2
        solution += _actions_between((14, 28), (38, 28)) + [click("buezna-matkhq", c_button)]
        solution += _actions_between((10, 52), (16, 52))
        for x in range(18, moving_exit + 1, 2):
            solution.append(click("sprite-6", f_button))
            solution += _actions_between((x - 2, 52), (x, 52))
        solution += _actions_between((moving_exit, 52), (32, 52))
        fall_probe = base["prefix_actions"] + _actions_between(stage, (8, 28))
        fall_probe += [click("crzsjq-grawwq-1", grab)]
        if track_order == "right-up":
            fall_probe += _actions_between((8, 28), (12, 28)) + [click("crzsjq-riidpd-1", right)]
            fall_probe += _actions_between((12, 28), (14, 28)) + [click("crzsjq-up-1", up)]
        else:
            fall_probe += _actions_between((8, 28), (12, 28)) + [click("crzsjq-up-1", up)]
            fall_probe += _actions_between((12, 28), (14, 28)) + [click("crzsjq-riidpd-1", right)]
        fall_probe += [click("buezna-matkhq", c_button)]
        fall_probe += _actions_between((14, 28), (18, 28)) + [click("buezna-matkhq", c_button)]
        goal = (32, 52)
        stage_blueprint = {
            "pressure_track_order": track_order,
            "moving_surface_steps": moving_steps,
            "bridge_colour_routing": "two-cycles-to-yefmyf",
        }

    spec = {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION, "difficulty_version": DIFFICULTY_VERSION,
        "quality_version": QUALITY_VERSION, "geometry_version": GEOMETRY_VERSION,
        "source": "generated_only", "seed": int(seed), "requested_seed": int(seed),
        "generation_attempt": int(attempt), "difficulty": int(difficulty),
        "context_index": int(difficulty - 1), "training_context_index": int(difficulty - 1),
        "verification_level_index": int(difficulty - 1), "grid_size": list(grid_size),
        "step_budget": int(profile["step_budget"]), "maze_width": MAZE_WIDTH,
        "maze_height": MAZE_HEIGHT, "maze_origin": list(MAZE_ORIGIN), "maze_pitch": MAZE_PITCH,
        "maze_start": base["maze_start"], "maze_exit": base["maze_exit"],
        "maze_passages": base["maze_passages"], "start": list(base["start"]),
        "goal": list(goal), "floors": [list(point) for point in sorted(floors)],
        "components": components, "constructed_solution": solution, "fall_probe": fall_probe,
        "prefix_solution_length": len(base["prefix_actions"]),
        "stage_blueprint": stage_blueprint,
        "interaction_probes": interaction_probes,
        "reference_level": difficulty, "reference_actions": profile["reference_actions"],
        "reference_action_status": "measured positive native witness; optimality not established",
        "action_tolerance_kind": "generated engineering bound",
        "required_mechanics": list(profile["required_mechanics"]),
        "required_winning_mechanics": list(profile["required_winning_mechanics"]),
        "required_interaction_mechanics": list(profile["required_interaction_mechanics"]),
        "reference_calibration": (
            "aggregate structure and native budget measured from one official level per tier; "
            "explicit tolerances are not population confidence intervals"
        ),
    }
    stage_hash, action_hash = stage_identity_hashes(spec)
    spec["mechanic_stage_sha256"] = stage_hash
    spec["stage_action_sha256"] = action_hash
    return spec


def _validated_spec(spec):
    if not isinstance(spec, Mapping):
        raise ValueError("generated spec must be an object")
    if spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if spec.get("generator_version") != GENERATOR_VERSION or spec.get("mechanics_version") != MECHANICS_VERSION:
        raise ValueError("generator/mechanics version mismatch")
    profile = PROFILES[difficulty]
    if list(spec.get("grid_size", ())) != list(profile["grid_size"]):
        raise ValueError("grid size differs from the official tier")
    if _integer(spec.get("step_budget"), "step_budget") != profile["step_budget"]:
        raise ValueError("native step budget differs from the official tier")
    for key in ("context_index", "training_context_index", "verification_level_index"):
        if _integer(spec.get(key), key) != difficulty - 1:
            raise ValueError(f"{key} differs from the native tier context")
    start, goal = tuple(spec.get("start", ())), tuple(spec.get("goal", ()))
    if len(start) != 2 or len(goal) != 2 or start == goal:
        raise ValueError("start and goal must be distinct coordinate pairs")
    floors, seen = [], set()
    for index, raw in enumerate(spec.get("floors", ())):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"floors[{index}] must be a coordinate pair")
        point = (_integer(raw[0], f"floors[{index}][0]"), _integer(raw[1], f"floors[{index}][1]"))
        if point in seen:
            raise ValueError("duplicate generated floor")
        if point[0] % 2 or point[1] % 2 or not (0 <= point[0] < 40 and 0 <= point[1] < profile["grid_size"][1]):
            raise ValueError("generated floor is outside the aligned playfield")
        seen.add(point)
        floors.append(point)
    if start not in seen or goal not in seen:
        raise ValueError("start and goal require ordinary generated support")
    components = []
    for index, raw in enumerate(spec.get("components", ())):
        if not isinstance(raw, Mapping):
            raise ValueError(f"components[{index}] must be an object")
        prototype, position = raw.get("prototype"), raw.get("position")
        if prototype not in _ALLOWED_PROTOTYPES or not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError(f"components[{index}] is malformed")
        x, y = _integer(position[0], f"components[{index}].x"), _integer(position[1], f"components[{index}].y")
        tags = raw.get("extra_tags", ())
        if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"components[{index}].extra_tags must be strings")
        components.append((prototype, x, y, tuple(tags)))
    return difficulty, profile, start, goal, floors, components


def build_level(spec):
    """Reconstruct one JSON-safe generated spec as an actual native level."""
    difficulty, profile, start, goal, floors, components = _validated_spec(spec)
    native = upstream().sprites
    sprites = [
        Sprite(pixels=[[2, 2], [2, 2]], name=f"generated-dc22-floor-{index}",
               visible=True, collidable=False, layer=-2).set_position(x, y)
        for index, (x, y) in enumerate(floors)
    ]
    for prototype, x, y, tags in components:
        sprite = native[prototype].clone().set_position(x, y)
        for tag in tags:
            if tag not in sprite.tags:
                sprite.tags.append(tag)
        sprites.append(sprite)
    # Preserve the shipped game's visual grammar: a patterned control panel
    # and hard separator keep click controls distinct from the playfield.
    if difficulty == 1:
        panel = ("coorbs-bg", "merged-sprite", 32)
    elif difficulty == 6:
        panel = ("coorbs-bg-1", "merged-sprite-1", 40)
    else:
        panel = ("coorbs-bg", "merged-sprite", 38)
    sprites.extend([
        native[panel[0]].clone().set_position(panel[2], 0),
        native[panel[1]].clone().set_position(panel[2], -2),
    ])
    sprites.append(native[names.SPRITE_GOAL].clone().set_position(*goal))
    sprites.append(native[names.SPRITE_PLAYER].clone().set_position(*start))
    return Level(
        sprites=sprites, grid_size=tuple(profile["grid_size"]),
        data={names.KEY_STEPS: profile["step_budget"], names.KEY_GENERATED_FORMAT: FORMAT,
              names.KEY_GENERATED_SPEC: copy.deepcopy(dict(spec))},
        name=f"generated-dc22-d{difficulty}-s{spec.get('seed', 0)}-a{spec.get('generation_attempt', 0)}",
    )


def _context_env(level, context_index):
    env = Env([level.clone() for _ in range(context_index + 1)])
    if context_index:
        env.set_level(context_index)
    return env


def _sprite_signature(sprite):
    return (sprite.name, int(sprite.x), int(sprite.y), int(sprite.interaction.value), bool(sprite.is_visible))


def _active_toggle_signature(env):
    return tuple(sorted(
        _sprite_signature(sprite) for sprite in env.level.get_sprites_by_tag(names.TAG_TOGGLE)
        if sprite.interaction.name != "REMOVED"
    ))


def _clicked_sprite(env, x, y):
    point = env.game.camera.display_to_grid(x, y)
    if point is None:
        return None
    return getattr(env.game, names.METHOD_HIT_VISIBLE)(*point, names.TAG_CLICK)


def _mechanic_certificate(level, context_index, actions):
    env = _context_env(level, context_index)
    counts, distinct_controls, distinct_bridges = Counter(), set(), set()
    unlocked_colors = set()
    changed_expanders, changed_movers = set(), set()
    minimum_steps, result, first_completion_action = env.steps_left, None, None
    for action_index, (action_id, x, y) in enumerate(actions, 1):
        before_player = (int(env.player.x), int(env.player.y))
        before_steps = env.steps_left
        before_keys = {
            (sprite.name, int(sprite.x), int(sprite.y), next(
                (tag for tag in sprite.tags if len(tag) == 1), None
            ))
            for sprite in env.level.get_sprites_by_tag(names.TAG_GATE_KEY)
        }
        before_toggles = _active_toggle_signature(env)
        crushers = env.level.get_sprites_by_tag(names.TAG_CRUSHER)
        before_crusher = (int(crushers[0].x), int(crushers[0].y)) if crushers else None
        before_attachment = str(getattr(env.game, "svxnnbpjl", "none"))
        before_bridge_names = tuple(sorted(
            sprite.name for sprite in env.level.get_sprites_by_tag(names.TAG_BRIDGE)
            if sprite.interaction.name != "REMOVED"
        ))
        clicked = _clicked_sprite(env, x, y) if action_id == names.ACTION_CLICK else None
        if clicked is not None:
            distinct_controls.add((clicked.name, int(clicked.x), int(clicked.y)))
            clicked_color = next((tag for tag in clicked.tags if len(tag) == 1), None)
            if clicked_color in unlocked_colors:
                counts["key_unlocked_control"] += 1
            one_letter = {tag for tag in clicked.tags if len(tag) == 1}
            if any(
                tag in one_letter and env.game.alugfbupso(env.player, sensor)
                for sensor in env.level.get_sprites_by_tag(names.TAG_PRESSURE)
                for tag in sensor.tags if len(tag) == 1
            ):
                counts["pressure_control"] += 1
        result = env.perform(action_id, x, y)
        after_player = (int(env.player.x), int(env.player.y))
        after_keys = {
            (sprite.name, int(sprite.x), int(sprite.y), next(
                (tag for tag in sprite.tags if len(tag) == 1), None
            ))
            for sprite in env.level.get_sprites_by_tag(names.TAG_GATE_KEY)
        }
        after_toggles = _active_toggle_signature(env)
        if after_toggles != before_toggles:
            counts["surface_cycle"] += 1
            counts["colour_control"] += 1
            changed_names = {
                value[0] for value in set(before_toggles).symmetric_difference(after_toggles)
            }
            if any(name.startswith("drfztmbrixto-") for name in changed_names):
                counts["expanding_surface"] += 1
                changed_expanders.update(
                    (value[1], value[2]) for value in set(before_toggles).symmetric_difference(after_toggles)
                    if value[0].startswith("drfztmbrixto-")
                )
            if any(name.startswith("moxubw-plelvb-") for name in changed_names):
                counts["moving_surface"] += 1
                changed_movers.update(
                    (value[1], value[2]) for value in set(before_toggles).symmetric_difference(after_toggles)
                    if value[0].startswith("moxubw-plelvb-")
                )
        removed_keys = before_keys - after_keys
        if removed_keys:
            counts["key_pickup"] += len(removed_keys)
            unlocked_colors.update(value[3] for value in removed_keys if value[3] is not None)
        if action_id == names.ACTION_CLICK and max(abs(after_player[0] - before_player[0]), abs(after_player[1] - before_player[1])) > 2:
            counts["bridge_teleport"] += 1
            distinct_bridges.add((before_player, after_player))
        crushers = env.level.get_sprites_by_tag(names.TAG_CRUSHER)
        moved = bool(crushers and before_crusher != (int(crushers[0].x), int(crushers[0].y)))
        if moved:
            counts["track_crusher_move" if getattr(env.game, "qnlqkldrl", False) else "crusher_move"] += 1
        after_attachment = str(getattr(env.game, "svxnnbpjl", "none"))
        if before_attachment != after_attachment:
            counts["object_grab" if after_attachment == "grawwq-object" else "bridge_grab"] += 1
        if moved and before_attachment == after_attachment == "grawwq-object":
            counts["object_carry"] += 1
        if moved and before_attachment == after_attachment == "brixto":
            counts["bridge_carry"] += 1
        after_bridge_names = tuple(sorted(
            sprite.name for sprite in env.level.get_sprites_by_tag(names.TAG_BRIDGE)
            if sprite.interaction.name != "REMOVED"
        ))
        if clicked is not None and names.TAG_BRIDGE_COLOR_BUTTON in clicked.tags and after_bridge_names != before_bridge_names:
            counts["bridge_colour_cycle"] += 1
        minimum_steps = min(minimum_steps, before_steps, env.steps_left)
        if result.won or env.levels_completed > 0:
            first_completion_action = action_index
            break
        if result.state.name == "GAME_OVER":
            break
    counts["distinct_controls"] = len(distinct_controls)
    counts["distinct_bridge_transfers"] = len(distinct_bridges)
    return {
        "won": bool(result and (result.won or env.levels_completed > 0)),
        "levels_completed": int(env.levels_completed), "final_steps": int(env.steps_left),
        "minimum_steps": int(minimum_steps),
        "first_completion_action": first_completion_action,
        "expanding_surface_instances": len(changed_expanders),
        "moving_surface_instances": len(changed_movers),
        **{key: int(value) for key, value in sorted(counts.items())},
    }


def _fall_certificate(level, context_index, actions):
    if not actions or actions[-1][0] != names.ACTION_CLICK:
        raise ValueError("fall probe must end in a click")
    env = _context_env(level, context_index)
    for action in actions[:-1]:
        observation = env.perform(*action)
        if observation.state.name == "GAME_OVER" or env.levels_completed:
            raise ValueError("fall probe prefix terminated unexpectedly")
    before_player, before_steps = (int(env.player.x), int(env.player.y)), env.steps_left
    observation = env.perform(*actions[-1])
    after_player = (int(env.player.x), int(env.player.y))
    return {"recovered": observation.state.name != "GAME_OVER" and before_player == after_player,
            "penalty": int(before_steps - env.steps_left), "player": list(after_player)}


def _native_route_trace(level, context_index, actions):
    """Replay public actions and retain player/budget/completion observations."""
    env = _context_env(level, context_index)
    trace = []
    for action in actions:
        observation = env.perform(*action)
        trace.append([
            int(env.player.x), int(env.player.y), int(env.steps_left),
            observation.state.name, int(env.levels_completed),
        ])
    return trace, bool(env.levels_completed), int(env.levels_completed)


def _paired_use_certificate(spec, actions):
    """Prove the tier-4 witness depends on each member of both shared-control pairs."""
    if spec.get("difficulty") != 4:
        return {}
    baseline, won, completed = _native_route_trace(build_level(spec), 3, actions)
    if not won or completed != 1:
        raise ValueError("tier 4 paired-use baseline must win exactly once")
    roles = {}
    for component in spec["components"]:
        prototype = component["prototype"]
        y = component["position"][1]
        if prototype == "drfztmbrixto-1":
            roles["phase1_expander"] = component
        elif prototype == "drfztmbrixto-5":
            roles["phase5_expander"] = component
        elif prototype == "moxubw-plelvb-1" and y == 30:
            roles["upper_mover"] = component
        elif prototype == "moxubw-plelvb-1" and y == 34:
            roles["lower_mover"] = component
    expected = {"phase1_expander", "phase5_expander", "upper_mover", "lower_mover"}
    if set(roles) != expected:
        raise ValueError("tier 4 paired-use components are incomplete")
    evidence = {}
    for role, target in roles.items():
        ablated = copy.deepcopy(dict(spec))
        removed = False
        kept = []
        for component in ablated["components"]:
            if not removed and component == target:
                removed = True
            else:
                kept.append(component)
        ablated["components"] = kept
        changed, changed_won, changed_completed = _native_route_trace(
            build_level(ablated), 3, actions,
        )
        divergence = next(
            (index for index, (left, right) in enumerate(zip(baseline, changed), 1) if left != right),
            None,
        )
        evidence[role] = {
            "component": copy.deepcopy(target),
            "effect": "native player/support access trace",
            "first_trace_divergence": divergence,
            "won": bool(changed_won),
            "levels_completed": int(changed_completed),
        }
    return evidence


def _canonical_points(spec):
    points = [("floor", int(x), int(y)) for x, y in spec["floors"]]
    for component in spec["components"]:
        prototype, (x, y) = component["prototype"], component["position"]
        if x < 40:
            points.append((prototype, int(x), int(y)))
    points.extend((label, *map(int, spec[label])) for label in ("start", "goal"))
    return points


def _normalized_variant(points, swap, sx, sy):
    transformed = [(label, sx * (y if swap else x), sy * (x if swap else y)) for label, x, y in points]
    left, top = min(x for _, x, _ in transformed), min(y for _, _, y in transformed)
    return sorted((label, x - left, y - top) for label, x, y in transformed)


def geometry_hashes(spec):
    points = _canonical_points(spec)
    raw = _normalized_variant(points, False, 1, 1)
    variants = [_normalized_variant(points, swap, sx, sy)
                for swap in (False, True) for sx in (-1, 1) for sy in (-1, 1)]
    encode = lambda value: json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encode(raw)).hexdigest(), hashlib.sha256(encode(min(variants))).hexdigest()


def geometry_partition(spec):
    _, fingerprint = geometry_hashes(spec)
    return fingerprint, SPLITS[int(fingerprint, 16) % len(SPLITS)]


def gameplay_hash(spec):
    payload = {key: spec[key] for key in (
        "difficulty", "grid_size", "step_budget", "start", "goal", "floors",
        "components",
    )}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _semantic_stage_actions(spec):
    click_labels = {}
    grid_size = tuple(spec["grid_size"])
    for component in spec["components"]:
        prototype = component["prototype"]
        sprite = upstream().sprites[prototype]
        if names.TAG_CLICK not in sprite.tags:
            continue
        action = tuple(_click_action(prototype, component["position"], grid_size))
        click_labels[action] = prototype
    prefix_length = spec.get("prefix_solution_length")
    if type(prefix_length) is not int or not 0 <= prefix_length <= len(spec["constructed_solution"]):
        raise ValueError("prefix solution length is invalid")
    semantic = []
    for raw in spec["constructed_solution"][prefix_length:]:
        action = tuple(raw)
        if action[0] == names.ACTION_CLICK:
            label = click_labels.get(action)
            if label is None:
                raise ValueError("stage click has no generated control")
            semantic.append(("click", label))
        else:
            semantic.append(("move", int(action[0])))
    return semantic


def derive_stage_blueprint(spec):
    """Recompute the human-readable mechanic blueprint from puzzle and suffix."""
    difficulty = spec["difficulty"]
    prototypes = [component["prototype"] for component in spec["components"]]
    semantic = _semantic_stage_actions(spec)
    clicks = [value[1] for value in semantic if value[0] == "click"]
    if difficulty == 1:
        return {"sequential_surface_gates": prototypes.count("tovemc-plelvb1")}
    if difficulty == 2:
        return {
            "unlocked_gate_topology": (
                "L-double" if "tovemc-plelvb-p-1" in prototypes else "straight-single"
            )
        }
    if difficulty == 3:
        return {
            "unlocked_gate_extent": (
                "long" if "tovemc-plelvb_plong_1" in prototypes else "short"
            )
        }
    if difficulty == 4:
        return {
            "expansion_click_depth": clicks.count("drfztmbrixto-buezna") // 2,
            "paired_expansion_cycles": 2,
            "expansion_initial_phases": sorted(
                int(prototype.rsplit("-", 1)[1]) for prototype in prototypes
                if prototype.startswith("drfztmbrixto-") and prototype[-1].isdigit()
            ),
            "expansion_route_row": next(
                int(component["position"][1]) for component in spec["components"]
                if component["prototype"] == "drfztmbrixto-1"
            ),
            "second_expansion_row": next(
                int(component["position"][1]) for component in spec["components"]
                if component["prototype"] == "drfztmbrixto-5"
            ),
            "moving_surface_steps": clicks.count("sprite-6"),
            "joint_moving_surfaces": prototypes.count("moxubw-plelvb-1"),
            "paired_route": "expand-outward/move-out-and-back/expand-return",
        }
    if difficulty == 5:
        landing = next(
            int(component["position"][0])
            for component in spec["components"]
            if component["prototype"] == "tewfut2"
        )
        return {
            "crusher_carry_right_steps": clicks.count("crzsjq-riidpd"),
            "carried_object_required_gap": [landing + 4, landing + 6],
            "expansion_click_depth": clicks.count("drfztmbrixto-buezna"),
            "moving_surface_steps": clicks.count("sprite-6"),
        }
    right = clicks.index("crzsjq-riidpd-1")
    up = clicks.index("crzsjq-up-1")
    return {
        "pressure_track_order": "right-up" if right < up else "up-right",
        "moving_surface_steps": clicks.count("sprite-6"),
        "bridge_colour_routing": "two-cycles-to-yefmyf",
    }


def stage_identity_hashes(spec):
    """Return canonical mechanic-relation and semantic suffix identities."""
    points = []
    for component in spec["components"]:
        prototype, (x, y) = component["prototype"], component["position"]
        if x >= 40:
            continue
        tags = ",".join(sorted(component.get("extra_tags", ())))
        points.append((f"{prototype}|{tags}", int(x), int(y)))
    points.append(("goal", *map(int, spec["goal"])))
    variants = [
        _normalized_variant(points, swap, sx, sy)
        for swap in (False, True)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    actions = _semantic_stage_actions(spec)
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    action_hash = hashlib.sha256(encode(actions)).hexdigest()
    relation_hash = hashlib.sha256(encode({
        "difficulty": spec["difficulty"],
        "blueprint": derive_stage_blueprint(spec),
        "components": min(variants),
        "semantic_suffix": actions,
    })).hexdigest()
    return relation_hash, action_hash


def _profile_errors(spec, level, *, require_proof):
    difficulty, profile = spec["difficulty"], PROFILES[spec["difficulty"]]
    metrics, errors = structural_metrics(level), []
    if metrics["grid_size"] != tuple(profile["grid_size"]):
        errors.append("grid_size outside reference profile")
    low, high = profile["generated_support_range"]
    if not low <= metrics["support_samples"] <= high:
        errors.append("initial support density outside explicit reference tolerance")
    if metrics["colour_controls"] < 1:
        errors.append("tier has no colour control")
    if difficulty in (2, 3, 5, 6) and metrics["keys"] < 1:
        errors.append("tier is missing its key/unlocked-control mechanic")
    if difficulty in (3, 4, 5, 6) and metrics["bridges"] < 2:
        errors.append("tier is missing paired bridges")
    if difficulty >= 5 and metrics["crushers"] != 1:
        errors.append("tier must contain exactly one native crusher")
    if difficulty == 5 and metrics["carried_objects"] != 1:
        errors.append("tier 5 must contain one carried object")
    prototypes = [component["prototype"] for component in spec.get("components", ())]
    if difficulty == 4:
        if prototypes.count("drfztmbrixto-1") != 1 or prototypes.count("drfztmbrixto-5") != 1:
            errors.append("tier 4 must contain the official phase-1/phase-5 expansion pair")
        if prototypes.count("moxubw-plelvb-1") != 2:
            errors.append("tier 4 must contain two jointly controlled moving surfaces")
    if difficulty == 5 and not {
        "drfztmbrixto-1", "drfztmbrixto-buezna", "moxubw-plelvb-1", "sprite-6",
    }.issubset(prototypes):
        errors.append("tier 5 is missing its coupled expansion/movement composition")
    if difficulty == 6:
        source_count = sum(
            component["prototype"] == "tewfutpibpar1"
            and names.TAG_BRIDGE_COLOR_CYCLE in component.get("extra_tags", ())
            for component in spec.get("components", ())
        )
        if source_count != 1 or "tewfutyefmyf1" not in prototypes:
            errors.append("tier 6 is missing selective source recolouring or heterogeneous destinations")
    if difficulty == 6 and (metrics["sensors"] < 4 or metrics["track_cells"] < 2):
        errors.append("tier 6 pressure controls or crusher track are incomplete")
    if require_proof:
        length = spec.get("solution_length")
        lo, hi = profile["generated_action_range"]
        if type(length) is not int or not lo <= length <= hi:
            errors.append("constructive action length outside generated engineering bound")
        mechanics = spec.get("solution_mechanics", {})
        if difficulty == 4 and (
            mechanics.get("expanding_surface_instances", 0) < 2
            or mechanics.get("moving_surface_instances", 0) < 2
        ):
            errors.append("tier 4 route did not natively exercise both coupled surface instances")
        if difficulty == 4:
            paired = spec.get("paired_use")
            expected_roles = {
                "phase1_expander", "phase5_expander", "upper_mover", "lower_mover",
            }
            if not isinstance(paired, Mapping) or set(paired) != expected_roles:
                errors.append("tier 4 paired-use certificate is missing or incomplete")
            else:
                for role, evidence in paired.items():
                    if (
                        not isinstance(evidence, Mapping)
                        or type(evidence.get("first_trace_divergence")) is not int
                        or evidence["first_trace_divergence"] < 1
                        or evidence.get("won") is not False
                        or evidence.get("levels_completed") != 0
                    ):
                        errors.append(f"tier 4 {role} lacks native causal ablation evidence")
        for mechanic in profile["required_winning_mechanics"]:
            if mechanic == "fall_recovery":
                fall = spec.get("fall_recovery", {})
                if fall.get("recovered") is not True or fall.get("penalty") != 20:
                    errors.append("fall rollback/20-step recovery certificate is missing")
            elif not isinstance(mechanics, Mapping) or mechanics.get(mechanic, 0) < 1:
                errors.append(f"winning route did not exercise {mechanic}")
        interaction = spec.get("interaction_mechanics", {})
        for mechanic in profile["required_interaction_mechanics"]:
            certificate = interaction.get(mechanic, {}) if isinstance(interaction, Mapping) else {}
            if not isinstance(certificate, Mapping) or certificate.get(mechanic, 0) < 1:
                errors.append(f"interaction probe did not exercise {mechanic}")
    return errors, metrics


def _recompute_draft_rejection(seed, difficulty, attempt, split):
    """Re-evaluate one prior deterministic draft and return its first rejection."""
    spec = _draft(seed, difficulty, attempt)
    raw_hash, d4_hash = geometry_hashes(spec)
    partition = SPLITS[int(d4_hash, 16) % len(SPLITS)]
    if partition != split:
        return "geometry_split"
    spec.update(
        split=split, geometry_sha256=raw_hash, geometry_d4_sha256=d4_hash,
        geometry_split=partition,
    )
    try:
        level = build_level(spec)
        errors, _ = _profile_errors(spec, level, require_proof=False)
        if errors:
            return "structural_profile"
        actions = [list(action) for action in spec["constructed_solution"]]
        mechanics = _mechanic_certificate(level, difficulty - 1, actions)
        if not mechanics["won"]:
            return "native_replay"
        fall = _fall_certificate(level, difficulty - 1, spec["fall_probe"])
        interactions = {
            mechanic: _mechanic_certificate(level, difficulty - 1, probe_actions)
            for mechanic, probe_actions in spec["interaction_probes"].items()
        }
        spec.update(
            solution_length=len(actions), solution_mechanics=mechanics,
            interaction_mechanics=interactions, fall_recovery=fall,
        )
        if difficulty == 4:
            spec["paired_use"] = _paired_use_certificate(spec, actions)
        errors, _ = _profile_errors(spec, level, require_proof=True)
        return "mechanic_or_action_profile" if errors else None
    except (AssertionError, IndexError, KeyError, TypeError, ValueError):
        return "native_transition"


def generate(seed, difficulty=1, attempts=DEFAULT_ATTEMPTS, node_limit=DEFAULT_NODE_LIMIT, *, split="train", diagnostics=None):
    """Generate one tier with bounded redraws and exact-context native replay."""
    seed = _integer(seed, "seed", nonnegative=True)
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts", positive=True)
    node_limit = _integer(node_limit, "node_limit", positive=True)
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError("diagnostics must be a mutable dict or None")
    if diagnostics is not None:
        diagnostics.clear()
    profile, exclusions = PROFILES[difficulty], Counter()
    search_limit = min(node_limit, profile["search_limit"])
    for attempt in range(1, attempts + 1):
        spec = _draft(seed, difficulty, attempt)
        raw_hash, d4_hash = geometry_hashes(spec)
        partition = SPLITS[int(d4_hash, 16) % len(SPLITS)]
        if partition != split:
            exclusions["geometry_split"] += 1
            continue
        spec.update(split=split, geometry_sha256=raw_hash, geometry_d4_sha256=d4_hash,
                    geometry_split=partition)
        try:
            level = build_level(spec)
            errors, metrics = _profile_errors(spec, level, require_proof=False)
            if errors:
                exclusions["structural_profile"] += 1
                continue
            actions = [list(action) for action in spec["constructed_solution"]]
            mechanics = _mechanic_certificate(level, difficulty - 1, actions)
            if not mechanics["won"]:
                exclusions["native_replay"] += 1
                continue
            fall = _fall_certificate(level, difficulty - 1, spec["fall_probe"])
            interaction_mechanics = {
                mechanic: _mechanic_certificate(level, difficulty - 1, actions)
                for mechanic, actions in spec["interaction_probes"].items()
            }
            paired_use = _paired_use_certificate(spec, actions) if difficulty == 4 else {}
        except (AssertionError, IndexError, KeyError, TypeError, ValueError):
            exclusions["native_transition"] += 1
            continue
        spec.update(
            solution=actions, context_solution=copy.deepcopy(actions), solution_length=len(actions),
            optimality="not_claimed_constructive_witness", optimal_actions=None,
            context_optimal_actions=None, search_backend="constructive-native-replay",
            search_limit=search_limit, search_expanded=0, search_generated=0,
            constructive_replay_actions=len(actions), reachable_states=len(actions) + 1,
            work_kind="constructive-native-replay-not-search", search_truncated=False,
            engine_verified=True, context_engine_verified=True, engine_win=True, levels_completed=1,
            solution_mechanics=mechanics, interaction_mechanics=interaction_mechanics,
            fall_recovery=fall,
            paired_use=paired_use,
            minimum_steps_remaining=mechanics["minimum_steps"],
            final_steps_remaining=mechanics["final_steps"],
            first_completion_action=mechanics["first_completion_action"],
            structural_metrics=json.loads(json.dumps(metrics)),
            generation_exclusions=dict(exclusions),
            generation_attempt_cap=attempts,
            requested_node_limit=node_limit,
        )
        spec["gameplay_sha256"] = gameplay_hash(spec)
        spec["proof"] = {
            "kind": "constructive-route-native-context-replay", "seed": seed,
            "requested_seed": seed, "generation_attempt": attempt,
            "generation_attempt_cap": attempts,
            "requested_node_limit": node_limit,
            "difficulty": difficulty, "context_index": difficulty - 1,
            "generator_version": GENERATOR_VERSION, "mechanics_version": MECHANICS_VERSION,
            "quality_version": QUALITY_VERSION, "split": split,
            "geometry_sha256": raw_hash, "geometry_d4_sha256": d4_hash,
            "gameplay_sha256": spec["gameplay_sha256"], "search_limit": search_limit,
            "search_expanded": 0, "search_generated": 0,
            "constructive_replay_actions": len(actions),
            "reachable_states": len(actions) + 1,
            "work_kind": "constructive-native-replay-not-search",
            "mechanic_stage_sha256": spec["mechanic_stage_sha256"],
            "stage_action_sha256": spec["stage_action_sha256"],
            "search_truncated": False, "optimality": "not claimed", "engine_win": True,
            "levels_completed": 1, "native_step_budget": profile["step_budget"],
            "minimum_steps_remaining": mechanics["minimum_steps"],
            "final_steps_remaining": mechanics["final_steps"],
            "first_completion_action": mechanics["first_completion_action"],
            "generation_exclusions": dict(exclusions),
            "paired_use_sha256": hashlib.sha256(json.dumps(
                paired_use, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
        }
        errors, _ = _profile_errors(spec, level, require_proof=True)
        if errors:
            exclusions["mechanic_or_action_profile"] += 1
            continue
        spec["generation_exclusions"] = dict(exclusions)
        if diagnostics is not None:
            diagnostics.update({
                "status": "accepted", "requested_seed": seed, "difficulty": difficulty,
                "attempts": attempt, "generator_version": GENERATOR_VERSION,
                "reasons": dict(exclusions),
            })
        return spec
    if diagnostics is not None:
        diagnostics.update({
            "status": "rejected", "requested_seed": seed, "difficulty": difficulty,
            "attempts": attempts, "generator_version": GENERATOR_VERSION,
            "reasons": dict(exclusions),
        })
    return None


def _game_level_seed(game_seed, level_index, difficulty):
    game_seed = _integer(game_seed, "game seed", nonnegative=True)
    material = f"{SOURCE_ID}:{game_seed}:{level_index}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split="train", difficulties=None, attempts=DEFAULT_ATTEMPTS, node_limit=DEFAULT_NODE_LIMIT, diagnostics=None):
    """Generate exactly one complete increasing six-level native game."""
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in selected):
        raise ValueError("game difficulties must be integers")
    selected = tuple(int(value) for value in selected)
    if selected != DIFFICULTIES:
        raise ValueError(f"full-standard DC22 games require exactly {DIFFICULTIES}")
    seed = _integer(seed, "game seed", nonnegative=True)
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError("diagnostics must be a mutable dict or None")
    if diagnostics is not None:
        diagnostics.clear()
    specs = []
    for level_index, difficulty in enumerate(selected):
        child_seed = _game_level_seed(seed, level_index, difficulty)
        child_diagnostic = {}
        spec = generate(
            child_seed, difficulty, attempts=attempts, node_limit=node_limit,
            split=split, diagnostics=child_diagnostic,
        )
        if spec is None:
            if diagnostics is not None:
                diagnostics.update({
                    "status": "rejected", "requested_seed": seed, "split": split,
                    "failed_level_index": level_index, "failed_difficulty": difficulty,
                    "child": child_diagnostic,
                })
            return None
        spec.update(game_seed=int(seed), game_level_index=level_index)
        specs.append(spec)
    env = Env(build_game(specs))
    for index, spec in enumerate(specs):
        if env.level_index != index:
            raise ValueError(f"generated whole-game context shifted before tier {index + 1}")
        before = env.levels_completed
        for action_index, action in enumerate(spec["solution"], 1):
            observation = env.perform(*action)
            if env.levels_completed > before and action_index != len(spec["solution"]):
                raise ValueError(f"generated tier {index + 1} completes before its final action")
        if env.levels_completed != before + 1:
            raise ValueError(f"generated whole-game replay failed at tier {index + 1}")
    if env.levels_completed != len(DIFFICULTIES):
        raise ValueError("generated whole game did not complete all six tiers")
    if diagnostics is not None:
        diagnostics.update({"status": "accepted", "requested_seed": seed, "split": split})
    return specs


def build_game(specs):
    """Build only a complete, correctly ordered, single-split six-level game."""
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or len(specs) != len(DIFFICULTIES):
        raise ValueError(f"DC22 build_game requires exactly {len(DIFFICULTIES)} ordered specs")
    split, levels = None, []
    for index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if not isinstance(spec, Mapping) or spec.get("difficulty") != difficulty:
            raise ValueError("game specs must use difficulties 1..6 in order")
        if spec.get("context_index") != index:
            raise ValueError("game specs have shifted native contexts")
        if split is None:
            split = spec.get("split")
            if split not in SPLITS:
                raise ValueError("game specs must declare a valid split")
        elif spec.get("split") != split:
            raise ValueError("game specs must all use the same split")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"spec {index} does not satisfy the full contract: {'; '.join(errors)}")
        levels.append(build_level(spec))
    return levels


def _official_geometry_hashes():
    hashes = set()
    for level in official_levels():
        points = []
        for sprite in level.get_sprites():
            if sprite.x >= 40:
                continue
            tags = set(sprite.tags)
            if names.TAG_PLAYER in tags:
                label = "start"
            elif names.TAG_GOAL in tags:
                label = "goal"
            elif names.TAG_TOGGLE in tags:
                label = sprite.name
            elif not sprite.is_collidable:
                label = "floor"
            else:
                continue
            points.append((label, int(sprite.x), int(sprite.y)))
        if points:
            variants = [_normalized_variant(points, swap, sx, sy)
                        for swap in (False, True) for sx in (-1, 1) for sy in (-1, 1)]
            encoded = json.dumps(min(variants), separators=(",", ":"), sort_keys=True).encode()
            hashes.add(hashlib.sha256(encoded).hexdigest())
    return hashes


def validate_full_standard(spec, curriculum_entry):
    """Fail closed by rebuilding identities, profiles, route events, and replay."""
    errors = []
    if not isinstance(spec, Mapping):
        return ["generated spec must be an object"]
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be an object"]
    difficulty = curriculum_entry.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty must be in 1..6"]
    profile = PROFILES[difficulty]
    mandatory = (
        "seed", "requested_seed", "generation_attempt", "generation_attempt_cap",
        "requested_node_limit", "search_limit", "search_expanded", "search_generated",
        "constructive_replay_actions", "reachable_states", "work_kind",
        "generation_exclusions", "step_budget", "minimum_steps_remaining",
        "final_steps_remaining", "first_completion_action", "levels_completed", "engine_verified",
        "context_engine_verified", "engine_win", "search_truncated", "proof",
    )
    for key in mandatory:
        if key not in spec:
            errors.append(f"{key} is mandatory")
    for key in ("engine_verified", "context_engine_verified", "engine_win", "search_truncated"):
        if key in spec and type(spec[key]) is not bool:
            errors.append(f"{key} must be a boolean")
    for key in (
        "seed", "requested_seed", "generation_attempt", "generation_attempt_cap",
        "requested_node_limit", "search_limit", "search_expanded", "search_generated",
        "constructive_replay_actions", "reachable_states", "step_budget",
        "minimum_steps_remaining", "final_steps_remaining", "levels_completed",
        "first_completion_action",
    ):
        if key in spec and type(spec[key]) is not int:
            errors.append(f"{key} must be an integer")
    seed = spec.get("seed")
    requested_seed = spec.get("requested_seed")
    if type(seed) is not int or seed < 0 or seed != requested_seed:
        errors.append("seed provenance is invalid or inconsistent")
    attempt = spec.get("generation_attempt")
    attempt_cap = spec.get("generation_attempt_cap")
    if type(attempt) is not int or type(attempt_cap) is not int or not 1 <= attempt <= attempt_cap:
        errors.append("generation attempt provenance is invalid")
    requested_node_limit = spec.get("requested_node_limit")
    search_limit = spec.get("search_limit")
    if (
        type(requested_node_limit) is not int or requested_node_limit < 1
        or type(search_limit) is not int
        or search_limit != min(requested_node_limit, profile["search_limit"])
    ):
        errors.append("constructive verification work cap is invalid")
    if spec.get("search_expanded") != 0 or spec.get("search_generated") != 0:
        errors.append("constructive replay must not claim search expansions")
    if spec.get("work_kind") != "constructive-native-replay-not-search":
        errors.append("constructive/search work kind is missing or inconsistent")
    exclusions = spec.get("generation_exclusions")
    allowed_exclusions = {
        "geometry_split", "structural_profile", "native_replay",
        "native_transition", "mechanic_or_action_profile",
    }
    if not isinstance(exclusions, Mapping):
        errors.append("generation rejection facts must be an object")
    else:
        valid_exclusions = not any(
            key not in allowed_exclusions or type(value) is not int or value < 0
            for key, value in exclusions.items()
        )
        if not valid_exclusions:
            errors.append("generation rejection facts contain invalid reasons or counts")
        if valid_exclusions and type(attempt) is int and sum(exclusions.values()) != attempt - 1:
            errors.append("generation rejection counts do not match the accepted attempt")
        if valid_exclusions and type(seed) is int and seed >= 0 and type(attempt) is int and attempt > 0 and spec.get("split") in SPLITS:
            expected_rejections = Counter()
            for prior_attempt in range(1, attempt):
                reason = _recompute_draft_rejection(seed, difficulty, prior_attempt, spec.get("split"))
                if reason is None:
                    errors.append("accepted attempt skipped an earlier admissible deterministic draft")
                    break
                expected_rejections[reason] += 1
            if dict(exclusions) != dict(expected_rejections):
                errors.append("stored rejection facts differ from deterministic redraws")
    if curriculum_entry.get("context_index") != difficulty - 1:
        errors.append("curriculum context differs from tier")
    if curriculum_entry.get("search_work") != profile["search_limit"]:
        errors.append("curriculum search work differs from calibrated cap")
    for key, expected in (
        ("format", FORMAT), ("generator_version", GENERATOR_VERSION),
        ("mechanics_version", MECHANICS_VERSION), ("difficulty_version", DIFFICULTY_VERSION),
        ("quality_version", QUALITY_VERSION), ("geometry_version", GEOMETRY_VERSION),
        ("source", "generated_only"), ("difficulty", difficulty),
        ("context_index", difficulty - 1), ("training_context_index", difficulty - 1),
        ("verification_level_index", difficulty - 1), ("split", spec.get("geometry_split")),
        ("required_mechanics", list(profile["required_mechanics"])),
        ("required_winning_mechanics", list(profile["required_winning_mechanics"])),
        ("required_interaction_mechanics", list(profile["required_interaction_mechanics"])),
        ("engine_verified", True), ("context_engine_verified", True), ("engine_win", True),
        ("search_truncated", False), ("optimality", "not_claimed_constructive_witness"),
    ):
        if spec.get(key) != expected:
            errors.append(f"{key} is missing or inconsistent")
    if spec.get("split") not in SPLITS:
        errors.append("split must be train, validation, or test")
    try:
        level = build_level(spec)
        if type(seed) is int and seed >= 0 and type(attempt) is int and attempt > 0:
            drafted = _draft(seed, difficulty, attempt)
            draft_keys = (
                "grid_size", "step_budget", "maze_width", "maze_height", "maze_origin",
                "maze_pitch", "maze_start", "maze_exit", "maze_passages", "start", "goal",
                "floors", "components", "constructed_solution", "fall_probe",
                "prefix_solution_length", "stage_blueprint", "interaction_probes",
                "required_mechanics", "required_winning_mechanics",
                "required_interaction_mechanics", "mechanic_stage_sha256", "stage_action_sha256",
            )
            if any(spec.get(key) != drafted.get(key) for key in draft_keys):
                errors.append("accepted row differs from its deterministic seed/attempt draft")
        profile_errors, metrics = _profile_errors(spec, level, require_proof=True)
        errors.extend(profile_errors)
        if spec.get("structural_metrics") != json.loads(json.dumps(metrics)):
            errors.append("stored structural metrics do not match rebuilt level")
        raw_hash, d4_hash = geometry_hashes(spec)
        partition = SPLITS[int(d4_hash, 16) % len(SPLITS)]
        if spec.get("geometry_sha256") != raw_hash:
            errors.append("raw geometry identity mismatch")
        if spec.get("geometry_d4_sha256") != d4_hash:
            errors.append("D4 geometry identity mismatch")
        if spec.get("geometry_split") != partition or spec.get("split") != partition:
            errors.append("canonical geometry partition differs from requested split")
        if spec.get("gameplay_sha256") != gameplay_hash(spec):
            errors.append("gameplay identity mismatch")
        stage_hash, action_hash = stage_identity_hashes(spec)
        if spec.get("stage_blueprint") != derive_stage_blueprint(spec):
            errors.append("stored mechanic-stage blueprint differs from rebuilt relations")
        if spec.get("mechanic_stage_sha256") != stage_hash:
            errors.append("mechanic-stage relation identity mismatch")
        if spec.get("stage_action_sha256") != action_hash:
            errors.append("semantic stage-action identity mismatch")
        if d4_hash in _official_geometry_hashes():
            errors.append("generated geometry matches an official level under D4")
        solution = spec.get("solution")
        if not isinstance(solution, list) or solution != spec.get("context_solution") or solution != spec.get("constructed_solution"):
            errors.append("stored route fields differ")
        else:
            mechanics = _mechanic_certificate(level, difficulty - 1, solution)
            if mechanics != spec.get("solution_mechanics"):
                errors.append("stored mechanic certificate differs from native replay")
            if not mechanics["won"] or mechanics["levels_completed"] != 1:
                errors.append("stored route does not win exactly one native level")
            if mechanics.get("first_completion_action") != len(solution):
                errors.append("first completion must occur on the final certified action")
            if spec.get("first_completion_action") != mechanics.get("first_completion_action"):
                errors.append("stored first completion index differs from native replay")
            if spec.get("solution_length") != len(solution):
                errors.append("solution length mismatch")
            if spec.get("constructive_replay_actions") != len(solution):
                errors.append("constructive replay action work differs from the certified route")
            if spec.get("reachable_states") != len(solution) + 1:
                errors.append("constructive replay state count differs from the certified route")
            if spec.get("minimum_steps_remaining") != mechanics["minimum_steps"]:
                errors.append("minimum native budget differs from replay")
            if spec.get("final_steps_remaining") != mechanics["final_steps"]:
                errors.append("final native budget differs from replay")
            if spec.get("levels_completed") != mechanics["levels_completed"]:
                errors.append("completion count differs from replay")
            if difficulty == 4:
                paired_use = _paired_use_certificate(spec, solution)
                if spec.get("paired_use") != paired_use:
                    errors.append("stored tier 4 paired-use certificate differs from native ablation")
        probes = spec.get("interaction_probes")
        stored_interactions = spec.get("interaction_mechanics")
        expected_probes = set(profile["required_interaction_mechanics"])
        if not isinstance(probes, Mapping) or set(probes) != expected_probes:
            errors.append("interaction probe set differs from the tier profile")
        elif not isinstance(stored_interactions, Mapping) or set(stored_interactions) != expected_probes:
            errors.append("interaction mechanic certificate set differs from the tier profile")
        else:
            for mechanic, actions in probes.items():
                certificate = _mechanic_certificate(level, difficulty - 1, actions)
                if certificate != stored_interactions.get(mechanic):
                    errors.append(f"stored {mechanic} interaction certificate differs from native replay")
        fall = _fall_certificate(level, difficulty - 1, spec.get("fall_probe", ()))
        if fall != spec.get("fall_recovery"):
            errors.append("fall recovery certificate differs from native replay")
        proof = spec.get("proof")
        if not isinstance(proof, Mapping):
            errors.append("nested proof is missing")
        else:
            mirrors = {
                "seed": spec.get("seed"), "requested_seed": spec.get("requested_seed"),
                "generation_attempt": spec.get("generation_attempt"),
                "generation_attempt_cap": spec.get("generation_attempt_cap"),
                "requested_node_limit": spec.get("requested_node_limit"),
                "difficulty": difficulty, "context_index": difficulty - 1,
                "generator_version": GENERATOR_VERSION, "mechanics_version": MECHANICS_VERSION,
                "quality_version": QUALITY_VERSION, "split": spec.get("split"),
                "geometry_sha256": raw_hash, "geometry_d4_sha256": d4_hash,
                "gameplay_sha256": spec.get("gameplay_sha256"), "search_limit": spec.get("search_limit"),
                "search_expanded": 0, "search_generated": 0,
                "constructive_replay_actions": spec.get("constructive_replay_actions"),
                "reachable_states": spec.get("reachable_states"),
                "work_kind": "constructive-native-replay-not-search",
                "mechanic_stage_sha256": stage_hash,
                "stage_action_sha256": action_hash,
                "search_truncated": False, "optimality": "not claimed", "engine_win": True,
                "levels_completed": 1, "native_step_budget": profile["step_budget"],
                "minimum_steps_remaining": spec.get("minimum_steps_remaining"),
                "final_steps_remaining": spec.get("final_steps_remaining"),
                "first_completion_action": spec.get("first_completion_action"),
                "generation_exclusions": spec.get("generation_exclusions"),
                "paired_use_sha256": hashlib.sha256(json.dumps(
                    spec.get("paired_use", {}), sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest(),
            }
            for key in mirrors:
                if key not in proof:
                    errors.append(f"proof.{key} is mandatory")
            for key, expected in mirrors.items():
                if proof.get(key) != expected:
                    errors.append(f"proof.{key} does not mirror the accepted row")
            for key in ("search_truncated", "engine_win"):
                if key in proof and type(proof[key]) is not bool:
                    errors.append(f"proof.{key} must be a boolean")
            for key in (
                "seed", "requested_seed", "generation_attempt", "generation_attempt_cap",
                "requested_node_limit", "difficulty", "context_index", "generator_version",
                "search_limit", "search_expanded", "search_generated",
                "constructive_replay_actions", "reachable_states", "levels_completed",
                "native_step_budget", "minimum_steps_remaining", "final_steps_remaining",
                "first_completion_action",
            ):
                if key in proof and type(proof[key]) is not int:
                    errors.append(f"proof.{key} must be an integer")
    except (AssertionError, IndexError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"native validation failed: {type(exc).__name__}: {exc}")
    return errors
