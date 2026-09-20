"""Full seven-tier, reference-calibrated KA59 procedural generation.

Official levels supply aggregate counts and tolerances only.  Drafts construct
new routes, wall masks, placements, and assignments, then derive a constructive
witness and replay it through the unmodified native engine at the intended
level index.  No official geometry or route is stored or sampled.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import random

import numpy as np
from arcengine import GameState, Level, Sprite

from . import names
from .env import Env, upstream
from .plan import DEFAULT_NODE_LIMIT, _selection_clicks
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    GEOMETRY_VERSION,
    MECHANICS_VERSION,
    PROFILES,
    QUALITY_VERSION,
    profile_errors,
    structural_metrics,
)


FORMAT = "pebby.ka59.level.v4"
GENERATOR_VERSION = 4
DEFAULT_ATTEMPTS = 64
SPLITS = ("train", "validation", "test")
_OFFICIAL_FRAME_HASHES = None
_OFFICIAL_GEOMETRY_HASHES = None


@dataclass(frozen=True)
class GenerationOutcome:
    spec: dict | None
    failure: dict | None


@dataclass(frozen=True)
class GameGenerationOutcome:
    specs: list[dict] | None
    failures: tuple[dict, ...]

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": "ka59-38d34dbb",
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
        "official_tier_characterization": "ka59-reference-profiles-v1:third_party-lines-40935-41461",
        "solution_mechanics": "ka59-native-body-retained-device-certificate-v2",
        "native_budget": "ka59-stepcounter-replay-v1",
        "context_engine_replay": "ka59-seven-context-constructive-replay-v1",
        "novelty_split": GEOMETRY_VERSION,
        "bounded_rejections": "ka59-generation-exclusions-v1",
        "independent_blast_type_closure": "ka59-astra-21-row-native-reset-and-14-type-mutations-2026-09-19",
        "root_native_frame_review": "ka59-root-official-and-generated-seven-frame-review-2026-09-19",
        "root_primary_collector": "ka59-root-three-split-seven-level-win-226-217-238-2026-09-19",
        "root_acceptance": "ka59-root-readiness-authorization-2026-09-19",
    },
    "caveats": [
        "one official level exists per tier, so tolerances are engineering bounds rather than population intervals",
        "official tier action lengths are bounded positive witnesses, not optimality claims",
        "pursuing enemies are engine-supported but occur in zero shipped levels and are excluded from the calibrated default",
        "D4 identity does not prove graph-isomorphism novelty",
        "tier 7 retains the official-calibrated passive player-goal relation: repeated box-route LEFT actions let a large-bomb blast displace the player onto its goal",
        "the procedural grammar is finite and bounded and does not establish distribution equivalence to the official puzzles",
        "off-frame and redundant devices may be nonessential; no every-device indispensability claim is made",
        "tier 6 bomb-1 suppression produced earlier counterfactual wins in three independent review rows, so its real blast effect may hinder and is not evidence of universal necessity, helpfulness, or route optimality",
        "the historical 168-row audit predates the blast correction; current blast evidence is the author's 27-row affected cohort plus an independent 21-row closure cohort",
        "rejection diagnostics are retained by diagnostic APIs and the bank sidecar but remain opt-in for compatibility callers",
    ],
}


def _integer(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _type_sensitive_equal(left, right):
    """JSON evidence equality where bool/int/float are never interchangeable."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return (
            left.keys() == right.keys()
            and all(_type_sensitive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _type_sensitive_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _json_primitive_errors(value, path="spec"):
    """Reject non-JSON and floating evidence before constructing native state."""
    errors = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                errors.append(f"{path} contains a non-string object key")
                continue
            errors.extend(_json_primitive_errors(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_json_primitive_errors(child, f"{path}[{index}]"))
    elif type(value) not in (str, int, bool, type(None)):
        errors.append(
            f"{path} must use exact JSON primitive types (floats are not accepted)"
        )
    return errors


def effective_seed(seed, split="train"):
    seed = _integer(seed, "seed", minimum=0)
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    material = f"ka59-v{GENERATOR_VERSION}:{split}:{seed}".encode()
    mapped = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")
    return mapped & ((1 << 63) - 1), split


def _game_level_seed(game_seed, level_index, difficulty):
    game_seed = _integer(game_seed, "game seed", minimum=0)
    material = f"ka59-38d34dbb:{game_seed}:{level_index}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )


def _empty_rows(grid):
    return [["0"] * grid for _ in range(grid)]


def _rows_from_rects(grid, rects):
    rows = _empty_rows(grid)
    for x, y, width, height in rects:
        for py in range(max(0, y), min(grid, y + height)):
            for px in range(max(0, x), min(grid, x + width)):
                rows[py][px] = "1"
    return ["".join(row) for row in rows]


def _route_cells(points, width, height, grid, margin=1):
    clear = set()
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 != x2 and y1 != y2:
            raise ValueError("route segments must be axis aligned")
        dx = 0 if x1 == x2 else (3 if x2 > x1 else -3)
        dy = 0 if y1 == y2 else (3 if y2 > y1 else -3)
        x, y = x1, y1
        while True:
            for py in range(max(0, y - margin), min(grid, y + height + margin)):
                for px in range(max(0, x - margin), min(grid, x + width + margin)):
                    clear.add((px, py))
            if (x, y) == (x2, y2):
                break
            x, y = x + dx, y + dy
    return clear


def _dense_wall_rows(grid, routes, rng, target_pixels):
    clear = set()
    for points, width, height in routes:
        clear.update(_route_cells(points, width, height, grid))
    candidates = set()
    for y in range(0, grid - 2, 3):
        for x in range(0, grid - 2, 3):
            block = {(px, py) for py in range(y, y + 3) for px in range(x, x + 3)}
            if not block & clear:
                candidates.add((x, y))
    target_blocks = min(len(candidates), target_pixels // 9)
    selected = set()
    frontier = set()
    while len(selected) < target_blocks:
        if not frontier:
            remaining = sorted(candidates - selected)
            if not remaining:
                break
            seed = rng.choice(remaining)
            selected.add(seed)
            x, y = seed
            frontier.update(
                (x + dx, y + dy) for dx, dy in ((0, -3), (0, 3), (-3, 0), (3, 0))
                if (x + dx, y + dy) in candidates and (x + dx, y + dy) not in selected
            )
            continue
        point = rng.choice(sorted(frontier))
        frontier.remove(point)
        if point in selected:
            continue
        selected.add(point)
        x, y = point
        frontier.update(
            (x + dx, y + dy) for dx, dy in ((0, -3), (0, 3), (-3, 0), (3, 0))
            if (x + dx, y + dy) in candidates and (x + dx, y + dy) not in selected
        )
    walls = {
        (px, py)
        for x, y in selected
        for py in range(y, y + 3)
        for px in range(x, x + 3)
    }
    rows = _empty_rows(grid)
    for x, y in walls:
        rows[y][x] = "1"
    return ["".join(row) for row in rows]


def _moves(points):
    actions = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        if dx and dy:
            raise ValueError("route segments must be axis aligned")
        if dx % 3 or dy % 3:
            raise ValueError("route segments must use the native lattice")
        if dx > 0:
            actions.extend([names.ACTION_RIGHT] * (dx // 3))
        elif dx < 0:
            actions.extend([names.ACTION_LEFT] * (-dx // 3))
        elif dy > 0:
            actions.extend([names.ACTION_DOWN] * (dy // 3))
        else:
            actions.extend([names.ACTION_UP] * (-dy // 3))
    return actions


def _entity(prototype, start, *, target=None, rotation=0):
    value = {"prototype": prototype, "start": list(start)}
    if target is not None:
        value["target"] = list(target)
    if rotation:
        value["rotation"] = int(rotation)
    return value


def _base_spec(seed, difficulty, attempt):
    profile = PROFILES[difficulty]
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_version": QUALITY_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "source": "generated_only",
        "seed": int(seed),
        "generation_attempt": int(attempt),
        "difficulty": difficulty,
        "grid_size": profile["grid_size"],
        "step_budget": profile["step_budget"],
        "boxes": [],
        "players": [],
        "explosives": [],
        "enemies": [],
        "wall_rows": [],
    }


def _draft(seed, difficulty, attempt):
    rng = random.Random(f"{MECHANICS_VERSION}:{seed}:{difficulty}:{attempt}")
    spec = _base_spec(seed, difficulty, attempt)
    commands = []

    if difficulty == 1:
        y = rng.choice((21, 24, 27))
        start_x = rng.choice((3, 6, 9))
        pushed_x = start_x + 18
        wall_x = pushed_x + 3
        goal_x = wall_x + rng.choice(
            [gap for gap in (6, 9, 12) if wall_x + gap <= 39]
        )
        detour = rng.choice((6, 9, 12))
        detour_y = y - detour if y - detour >= 3 else y + detour
        spec["boxes"] = [
            _entity(names.BOX_3X3, (start_x, y), target=(3, 3)),
            _entity(names.BOX_3X3, (start_x + 3, y), target=(6, 3)),
        ]
        rects = [
            (wall_x, y - 3, 6, 9),
            (rng.choice((3, 12, 21)), 36, 21, 3),
            (rng.choice((3, 15, 27)), 3, 9, 3),
        ]
        spec["wall_rows"] = _rows_from_rects(45, rects)
        commands = [
            ("move", names.ACTION_RIGHT),
            *[("move", action) for action in _moves(
                [(start_x, y), (start_x, detour_y), (goal_x, detour_y), (goal_x, y)]
            )],
        ]

    elif difficulty == 2:
        prototypes = list(names.BOX_PROTOTYPES)
        rng.shuffle(prototypes)
        lanes = (6, 18, 33, 48)
        routes = []
        for index, (prototype, y) in enumerate(zip(prototypes, lanes)):
            width = upstream().sprites[prototype].width
            height = upstream().sprites[prototype].height
            left = rng.choice((3, 6, 9))
            right = left + rng.choice((18, 21, 24))
            if index % 2:
                start, goal = (right, y), (left, y)
            else:
                start, goal = (left, y), (right, y)
            offset = rng.choice((3, 6)) * (-1 if index in (0, 2) else 1)
            waypoint_y = y + offset
            waypoint_y = max(3, min(57 - height + 3, waypoint_y))
            points = [start, (start[0], waypoint_y), (goal[0], waypoint_y), goal]
            spec["boxes"].append(_entity(prototype, start, target=(3 + index * 9, 3)))
            routes.append((points, width, height))
            commands.append(("select", index))
            commands.extend(("move", action) for action in _moves(points))
        spec["wall_rows"] = _dense_wall_rows(
            63, routes, rng, rng.randrange(1250, 1851, 9)
        )

    elif difficulty == 3:
        y = rng.choice((33, 36))
        player_one_x = rng.choice((9, 12))
        player_two_x = rng.choice((30, 33))
        player_offset = rng.choice((12, 15))
        player_two_y = y - player_offset
        # Finish on the opposite side of both players.  Besides making the
        # open-board tier a genuine two-player routing problem, this keeps the
        # final vertical leg from accidentally pushing player one a third time
        # and clipping its derived target outline at the lower frame edge.
        goal_x = rng.choice((3, 6, 9))
        goal_y = rng.choice((42, 45, 48))
        spec["boxes"] = [_entity(names.BOX_3X3, (6, y), target=(3, 3))]
        spec["players"] = [
            _entity(names.PLAYER_LARGE, (player_one_x, y), target=(42, 3)),
            _entity(names.PLAYER_LARGE, (player_two_x, player_two_y), target=(42, 15)),
        ]
        spec["wall_rows"] = []
        commands = [
            ("move", names.ACTION_RIGHT)
            for _ in range((player_one_x - 6) // 3 + 1)
        ]
        commands.extend(("move", names.ACTION_RIGHT) for _ in range(6))
        commands.append(("move", names.ACTION_UP))
        commands.extend(
            ("move", names.ACTION_RIGHT)
            for _ in range((player_two_x - (player_one_x + 15)) // 3)
        )
        commands.extend(
            ("move", names.ACTION_UP)
            for _ in range((player_offset - 3) // 3 - 1)
        )
        route_start = (player_two_x, y - player_offset + 6)
        commands.extend(
            ("move", action)
            for action in _moves([route_start, (goal_x, route_start[1]), (goal_x, goal_y)])
        )

    elif difficulty == 4:
        y = rng.choice((18, 21, 24))
        detour = rng.choice((6, 9))
        box_zero_goal = (rng.choice((42, 45, 48)), y - detour)
        box_one_start = (rng.choice((42, 45, 48)), 42)
        box_one_goal = (rng.choice((3, 6, 9)), 42)
        box_one_detour_y = 51
        spec["boxes"] = [
            _entity(names.BOX_3X3, (6, y), target=(3, 3)),
            _entity(names.BOX_3X3, box_one_start, target=(6, 3)),
        ]
        spec["players"] = [_entity(names.PLAYER_LARGE, (9, y), target=(42, 6))]
        rects = [(0, 33, 54, 6), (24, 39, 9, 9)]
        spec["wall_rows"] = _rows_from_rects(54, rects)
        commands = [("move", names.ACTION_RIGHT), ("move", names.ACTION_RIGHT)]
        commands.extend(
            ("move", action)
            for action in _moves([(9, y), (9, y - detour), box_zero_goal])
        )
        commands.append(("select", 1))
        commands.extend(
            ("move", action)
            for action in _moves([
                box_one_start,
                (box_one_start[0], box_one_detour_y),
                (box_one_goal[0], box_one_detour_y),
                box_one_goal,
            ])
        )

    elif difficulty == 5:
        box_x = rng.choice((21, 24, 27))
        spec["boxes"] = [_entity(names.BOX_3X3, (box_x, 30), target=(3, 3))]
        spec["players"] = [_entity(names.PLAYER_SMALL, (rng.choice((3, 6, 9)), 9))]
        spec["explosives"] = [
            # This one-frame timer and the mixed timer form a coupled native
            # interaction: both blasts change the selected-box trajectory.
            _entity(names.EXPLOSIVE_SMALL, (box_x - 9, 30), rotation=270),
            _entity(names.EXPLOSIVE_SMALL, (-9, 15), rotation=180),
            _entity(names.EXPLOSIVE_SMALL, (-12, -9), rotation=270),
            _entity(names.EXPLOSIVE_SMALL, (9, -12)),
            # The box actively arranges this same-footprint mixed timer before
            # its fourth charge.  Suppressing only its blast (while preserving
            # its body and recharge reset) changes the box route and completion.
            _entity(
                names.EXPLOSIVE_MIXED,
                (box_x + 3, 30),
                rotation=90,
            ),
        ]
        rects = [
            (0, 45, rng.choice((36, 39, 42)), 6),
            (48, 3, 9, rng.choice((18, 21, 24))),
        ]
        spec["wall_rows"] = _rows_from_rects(63, rects)
        right_steps, down_steps = rng.choice(((7, 3), (7, 5), (8, 3)))
        commands = [
            *[("move", names.ACTION_RIGHT) for _ in range(right_steps)],
            *[("move", names.ACTION_DOWN) for _ in range(down_steps)],
            *[("move", names.ACTION_UP) for _ in range(4)],
        ]

    elif difficulty == 6:
        box_start_x = rng.choice((3, 6, 9))
        box_start_y = rng.choice((48, 51, 54))
        box_goal_y = 45
        first_x = rng.choice((21, 24, 27))
        upper_y = rng.choice((9, 12, 15))
        second_x = rng.choice((45, 48))
        lower_y = rng.choice((42, 45, 48))
        box_goal_x = rng.choice((54, 57))
        player_x = rng.choice((27, 30))
        spec["boxes"] = [_entity(names.BOX_3X3, (box_start_x, box_start_y), target=(3, 3))]
        spec["players"] = [_entity(names.PLAYER_LARGE, (player_x, 30), target=(45, 45))]
        spec["explosives"] = [
            _entity(names.EXPLOSIVE_LARGE, (player_x, 24)),
            # The first horizontal box transition pushes this upward-facing
            # bomb into a new box/player relation before its sixth-move blast.
            _entity(
                names.EXPLOSIVE_LARGE,
                (box_start_x + 6, box_start_y - 3),
                rotation=90,
            ),
            _entity(names.EXPLOSIVE_LARGE, (54, 24), rotation=rng.choice((90, 270))),
        ]
        rects = [(15, 0, 3, 45), (39, 18, 3, 45), (51, 0, 3, 39)]
        spec["wall_rows"] = _rows_from_rects(63, rects)
        commands = [
            *[("move", names.ACTION_RIGHT) for _ in range((first_x - box_start_x) // 3)],
            *[("move", names.ACTION_UP) for _ in range((box_start_y - upper_y) // 3)],
            *[("move", names.ACTION_RIGHT) for _ in range((second_x - first_x) // 3)],
            *[("move", names.ACTION_DOWN) for _ in range((lower_y - upper_y) // 3)],
            *[("move", names.ACTION_RIGHT) for _ in range((box_goal_x - second_x) // 3)],
        ]

    else:
        box_zero_x = rng.choice((3, 6, 9))
        box_zero_y = rng.choice((48, 51, 54))
        box_zero_goal_y = rng.choice((6, 9, 12))
        box_zero_goal_x = rng.choice((51, 54, 57))
        box_one_start_x = rng.choice((48, 51, 54))
        box_one_goal_x = rng.choice((3, 6, 9))
        spec["boxes"] = [
            _entity(names.BOX_3X6, (box_zero_x, box_zero_y), target=(3, 3)),
            _entity(names.BOX_6X3, (box_one_start_x, 51), target=(12, 3)),
        ]
        spec["players"] = [_entity(names.PLAYER_LARGE, (30, 30), target=(45, 45))]
        spec["explosives"] = [
            _entity(names.EXPLOSIVE_LARGE, (30, 24)),
            # Box zero must recursively reposition this bomb on its opening
            # vertical route; its blast then changes the box trajectory while
            # the other large bomb establishes the player-target displacement.
            _entity(
                names.EXPLOSIVE_LARGE,
                (box_zero_x, box_zero_y - 6),
                rotation=0,
            ),
        ]
        rects = [
            (18, 39, 24, 6),
            (21, 15, 6, 18),
            (rng.choice((39, 42, 45)), 33, 12, 6),
            (rng.choice((9, 12, 15)), 24, 9, 9),
        ]
        spec["wall_rows"] = _rows_from_rects(63, rects)
        commands = [
            *[("move", names.ACTION_UP) for _ in range((box_zero_y - box_zero_goal_y) // 3)],
            *[("move", names.ACTION_RIGHT) for _ in range((box_zero_goal_x - box_zero_x) // 3)],
            ("select", 1),
            ("move", names.ACTION_DOWN),
            ("move", names.ACTION_DOWN),
            *[("move", names.ACTION_LEFT) for _ in range((box_one_start_x - box_one_goal_x) // 3)],
            ("move", names.ACTION_UP),
            ("move", names.ACTION_UP),
        ]
    return spec, commands


def _position(value, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain two integers")
    return tuple(_integer(item, label) for item in value)


def _wall_sprite(spec):
    grid = _integer(spec.get("grid_size"), "grid_size", minimum=1)
    rows = spec.get("wall_rows", ())
    if not isinstance(rows, list):
        raise ValueError("wall_rows must be a list")
    if not rows:
        return None
    if len(rows) != grid or any(
        not isinstance(row, str) or len(row) != grid or set(row) - {"0", "1"}
        for row in rows
    ):
        raise ValueError("wall_rows must be a square 0/1 mask matching grid_size")
    pixels = np.full((grid, grid), -1, dtype=np.int8)
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value == "1":
                pixels[y, x] = 15
    if not np.any(pixels != -1):
        return None
    return Sprite(
        pixels=pixels,
        name="generated-ka59-internal-wall",
        visible=True,
        collidable=True,
        tags=[names.TAG_WALL],
    )


def build_level(spec):
    """Rebuild a generated native level from a JSON-safe specification."""
    if not isinstance(spec, Mapping) or spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    grid = _integer(spec.get("grid_size"), "grid_size", minimum=1)
    budget = _integer(spec.get("step_budget"), "step_budget", minimum=1)
    module = upstream()
    prototypes = module.sprites
    sprites = []

    boxes = spec.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("boxes must be a nonempty list")
    for index, item in enumerate(boxes):
        prototype = item.get("prototype")
        if prototype not in names.BOX_PROTOTYPES:
            raise ValueError(f"boxes[{index}] has an unknown prototype")
        start = _position(item.get("start"), f"boxes[{index}].start")
        target = _position(item.get("target"), f"boxes[{index}].target")
        sprites.append(
            prototypes[names.BOX_TO_TARGET[prototype]].clone().set_position(
                target[0] - 1, target[1] - 1
            )
        )
        sprites.append(prototypes[prototype].clone().set_position(*start))

    players = spec.get("players", ())
    if not isinstance(players, list):
        raise ValueError("players must be a list")
    for index, item in enumerate(players):
        prototype = item.get("prototype")
        if prototype not in (names.PLAYER_LARGE, names.PLAYER_SMALL):
            raise ValueError(f"players[{index}] has an unknown prototype")
        start = _position(item.get("start"), f"players[{index}].start")
        sprites.append(prototypes[prototype].clone().set_position(*start))
        if item.get("target") is not None:
            target = _position(item["target"], f"players[{index}].target")
            sprites.append(
                prototypes[names.PLAYER_TARGET].clone().set_position(
                    target[0] - 1, target[1] - 1
                )
            )

    explosives = spec.get("explosives", ())
    if not isinstance(explosives, list):
        raise ValueError("explosives must be a list")
    for index, item in enumerate(explosives):
        prototype = item.get("prototype")
        if prototype not in (
            names.EXPLOSIVE_SMALL,
            names.EXPLOSIVE_LARGE,
            names.EXPLOSIVE_MIXED,
        ):
            raise ValueError(f"explosives[{index}] has an unknown prototype")
        start = _position(item.get("start"), f"explosives[{index}].start")
        rotation = _integer(item.get("rotation", 0), f"explosives[{index}].rotation")
        if rotation not in (0, 90, 180, 270):
            raise ValueError("explosive rotations must be multiples of 90 degrees")
        sprites.append(
            prototypes[prototype].clone().set_position(*start).set_rotation(rotation)
        )
    if spec.get("enemies", ()):
        raise ValueError("calibrated official profiles have reference enemy count zero")

    wall = _wall_sprite(spec)
    if wall is not None:
        sprites.append(wall)
    boundary_pixels = np.full((grid + 6, grid + 6), -1, dtype=np.int8)
    boundary_pixels[:3, :] = 2
    boundary_pixels[-3:, :] = 2
    boundary_pixels[:, :3] = 2
    boundary_pixels[:, -3:] = 2
    sprites.append(
        Sprite(
            pixels=boundary_pixels,
            name="generated-ka59-boundary",
            visible=True,
            collidable=True,
            tags=[names.TAG_BOUNDARY],
        ).set_position(-3, -3)
    )
    return Level(
        sprites=sprites,
        grid_size=(grid, grid),
        data={names.KEY_STEPS: budget},
        name=f"generated-ka59-v{GENERATOR_VERSION}-d{difficulty}-s{spec.get('seed', 0)}",
    )


def _execute_commands(spec, commands):
    env = Env([build_level(spec)])
    actions = []
    for kind, value in commands:
        if env.state != GameState.NOT_FINISHED:
            raise ValueError("draft completed before its constructive route ended")
        if kind == "select":
            choices = dict(_selection_clicks(env))
            if value not in choices:
                if env.boxes().index(env.selected()) == value:
                    continue
                raise ValueError(f"box {value} has no visible native click")
            x, y = choices[value]
            action = (names.ACTION_CLICK, x, y)
        elif kind == "move":
            action = (value, None, None)
        else:
            raise ValueError(f"unknown constructive command {kind!r}")
        env.perform(*action)
        actions.append(action)
    return env, actions


def _derive_targets(spec, commands):
    env, _ = _execute_commands(spec, commands)
    for item, sprite in zip(spec["boxes"], env.boxes()):
        item["target"] = [int(sprite.x), int(sprite.y)]
    players = env.level.get_sprites_by_tag(names.TAG_PLAYER)
    for item, sprite in zip(spec.get("players", ()), players):
        if item.get("target") is not None:
            item["target"] = [int(sprite.x), int(sprite.y)]


def _snapshot(env):
    return {
        "boxes": tuple((sprite.x, sprite.y) for sprite in env.boxes()),
        "players": tuple(
            (sprite.x, sprite.y) for sprite in env.level.get_sprites_by_tag(names.TAG_PLAYER)
        ),
        "explosives": tuple(
            (sprite.x, sprite.y, np.asarray(sprite.pixels).tobytes())
            for sprite in env.level.get_sprites_by_tag(names.TAG_EXPLOSIVE)
        ),
        "selected": env.boxes().index(env.selected()),
        "steps": env.steps_left,
    }


def _trajectory_signature(spec, actions, context_index, disabled_explosives=()):
    """Replay a route with selected blasts suppressed but bomb bodies retained.

    Suppressed bombs still advance their native timers and are reset to the
    native spent/recharge state whenever they would detonate.  This isolates
    the blast channel without deleting collidable geometry.
    """
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context_index + 1)])
    env.set_level(context_index)
    explosives = tuple(env.level.get_sprites_by_tag(names.TAG_EXPLOSIVE))
    disabled = frozenset(disabled_explosives)
    if any(type(index) is not int or not 0 <= index < len(explosives) for index in disabled):
        raise ValueError("disabled explosive index is outside the native level")
    detonation_attempts = {index: [] for index in disabled}
    action_number = 0
    if disabled:
        native_charge = env.game.lflcissmce

        def charge_without_disabled_blasts():
            ready = native_charge()
            retained = []
            for bomb in ready:
                index = explosives.index(bomb)
                if index in disabled:
                    detonation_attempts[index].append(action_number)
                    env.game.pxqdkrdaye(bomb)
                else:
                    retained.append(bomb)
            return retained

        env.game.lflcissmce = charge_without_disabled_blasts
    before_score = env.levels_completed
    states = []
    first_win = None
    initial_explosives = tuple((int(bomb.x), int(bomb.y)) for bomb in explosives)
    for action_number, action in enumerate(actions, 1):
        if env.state != GameState.NOT_FINISHED or env.levels_completed > before_score:
            break
        observation = env.perform(*action)
        states.append((
            tuple((int(box.x), int(box.y)) for box in env.boxes()),
            tuple(
                (int(player.x), int(player.y))
                for player in env.level.get_sprites_by_tag(names.TAG_PLAYER)
            ),
            tuple((int(bomb.x), int(bomb.y)) for bomb in explosives),
        ))
        if env.levels_completed > before_score or observation.state == GameState.WIN:
            first_win = action_number
            break
    return {
        "states": tuple(states),
        "first_win": first_win,
        "initial_explosives": initial_explosives,
        "detonation_attempts": {
            index: tuple(attempts) for index, attempts in detonation_attempts.items()
        },
    }


def _first_trace_difference(baseline, counterfactual):
    for action_number, (left, right) in enumerate(
        zip(baseline["states"], counterfactual["states"]), 1
    ):
        if left[:2] != right[:2]:
            return action_number
    if (
        len(baseline["states"]) != len(counterfactual["states"])
        or baseline["first_win"] != counterfactual["first_win"]
    ):
        return min(len(baseline["states"]), len(counterfactual["states"])) + 1
    return None


def _explosive_device_effects(spec, actions, context_index):
    """Attribute body arrangement and detonation effects to each native bomb."""
    baseline = _trajectory_signature(spec, actions, context_index)
    indexes = tuple(range(len(spec.get("explosives", ()))))
    body_only = _trajectory_signature(
        spec, actions, context_index, disabled_explosives=indexes
    )
    effects = []
    for index, item in enumerate(spec.get("explosives", ())):
        suppressed = _trajectory_signature(
            spec, actions, context_index, disabled_explosives=(index,)
        )
        first_difference = _first_trace_difference(baseline, suppressed)
        box_effect = any(
            left[0] != right[0]
            for left, right in zip(baseline["states"], suppressed["states"])
        )
        player_effect = any(
            left[1] != right[1]
            for left, right in zip(baseline["states"], suppressed["states"])
        )
        positions = [body_only["initial_explosives"][index]] + [
            state[2][index] for state in body_only["states"]
        ]
        body_moves = [
            action
            for action, (before, after) in enumerate(zip(positions, positions[1:]), 1)
            if before != after
        ]
        attempts = body_only["detonation_attempts"].get(index, ())
        arranged = bool(body_moves and attempts and body_moves[0] <= attempts[0])
        effects.append({
            "index": index,
            "prototype": item["prototype"],
            "counterfactual_retains_body": True,
            "counterfactual_uses_native_recharge_reset": True,
            "body_repositions_without_blast": len(body_moves),
            "body_reposition_actions": body_moves,
            "body_push_or_collision_participation": bool(body_moves),
            "first_body_reposition_action": body_moves[0] if body_moves else None,
            "detonation_attempt_actions": list(attempts),
            "arranged_before_detonation": arranged,
            "detonation_causal": first_difference is not None,
            "detonation_changes_boxes": box_effect,
            "detonation_changes_players": player_effect,
            "detonation_changes_first_win": (
                baseline["first_win"] != suppressed["first_win"]
            ),
            "first_detonation_difference_action": first_difference,
            "baseline_first_win": baseline["first_win"],
            "suppressed_first_win": suppressed["first_win"],
        })
    return effects


def _will_explode(sprite):
    for index in range(sprite.height):
        if sprite.pixels[index, 0] != 12:
            return index == sprite.height - 1
    return False


def _route_certificate(spec, actions, context_index):
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context_index + 1)])
    env.set_level(context_index)
    mechanics = Counter()
    semantic_actions = []
    minimum_steps = env.steps_left
    observation = None
    for action_number, action in enumerate(actions):
        if env.state != GameState.NOT_FINISHED:
            raise ValueError("stored route contains actions after native completion")
        before = _snapshot(env)
        action_id, x, y = action
        exploding = 0
        if action_id in names.MOVE_ACTIONS:
            exploding = sum(
                _will_explode(sprite)
                for sprite in env.level.get_sprites_by_tag(names.TAG_EXPLOSIVE)
            )
            delta = {
                names.ACTION_UP: (0, -3),
                names.ACTION_DOWN: (0, 3),
                names.ACTION_LEFT: (-3, 0),
                names.ACTION_RIGHT: (3, 0),
            }[action_id]
            semantic_actions.append(["move", *delta])
        else:
            point = env.game.camera.display_to_grid(x, y)
            clicked = env.level.get_sprite_at(*point, names.TAG_BOX) if point else None
            if clicked not in env.boxes():
                raise ValueError("stored click no longer selects a native box")
            clicked_index = env.boxes().index(clicked)
            semantic_actions.append(["select", clicked_index])
        observation = env.perform(action_id, x, y)
        after = _snapshot(env)
        minimum_steps = min(minimum_steps, after["steps"])
        if action_id == names.ACTION_CLICK and after["selected"] != before["selected"]:
            mechanics["selection_clicks"] += 1
        moved_other_boxes = sum(
            old != new
            for index, (old, new) in enumerate(zip(before["boxes"], after["boxes"]))
            if index != before["selected"]
        )
        moved_players = sum(
            old != new for old, new in zip(before["players"], after["players"])
        )
        moved_explosives = sum(
            old[:2] != new[:2]
            for old, new in zip(before["explosives"], after["explosives"])
        )
        if moved_explosives:
            # The selected-box transition is applied before any charge-triggered
            # blast in this native action, so a changed bomb position records
            # an actively established relation even when detonation follows.
            mechanics["active_explosive_setups"] += moved_explosives
        if exploding:
            mechanics["explosions"] += exploding
            selected_before = before["boxes"][before["selected"]]
            selected_after = after["boxes"][before["selected"]]
            expected = selected_before
            if action_id in names.MOVE_ACTIONS:
                expected = (
                    selected_before[0] + delta[0],
                    selected_before[1] + delta[1],
                )
            if moved_other_boxes or moved_players or selected_after not in (selected_before, expected):
                mechanics["blast_pushes"] += 1
        else:
            if moved_other_boxes:
                mechanics["recursive_pushes"] += moved_other_boxes
            if moved_players:
                mechanics["player_pushes"] += moved_players
                mechanics["recursive_pushes"] += moved_players
        if observation.state == GameState.GAME_OVER:
            raise ValueError(f"native route lost at action {action_number}")
    if (
        observation is None
        or env.levels_completed != 1
        or observation.state not in (GameState.WIN, GameState.NOT_FINISHED)
    ):
        raise ValueError("constructive route did not complete exactly one native level")
    mechanics.update(
        won=True,
        native_budget_spent=spec["step_budget"] - env.steps_left,
        final_steps_left=env.steps_left,
        minimum_steps_left=minimum_steps,
    )
    mechanics["explosive_device_effects"] = (
        _explosive_device_effects(spec, actions, context_index)
        if spec.get("explosives") else []
    )
    mechanics["active_explosive_setups"] = sum(
        effect["arranged_before_detonation"]
        for effect in mechanics["explosive_device_effects"]
    )
    return dict(mechanics), semantic_actions


def _features(spec):
    features = []
    for y, row in enumerate(spec.get("wall_rows", ())):
        for x, value in enumerate(row):
            if value == "1":
                features.append((("wall",), x, y))
    prototypes = upstream().sprites
    for item in spec["boxes"]:
        sprite = prototypes[item["prototype"]]
        for kind, point in (("box", item["start"]), ("box_target", item["target"])):
            label = (kind, sprite.width, sprite.height)
            for y in range(point[1], point[1] + sprite.height):
                for x in range(point[0], point[0] + sprite.width):
                    features.append((label, x, y))
    for item in spec.get("players", ()):
        sprite = prototypes[item["prototype"]]
        for y in range(item["start"][1], item["start"][1] + sprite.height):
            for x in range(item["start"][0], item["start"][0] + sprite.width):
                features.append((("player", sprite.width, sprite.height), x, y))
        if item.get("target") is not None:
            for y in range(item["target"][1], item["target"][1] + sprite.height):
                for x in range(item["target"][0], item["target"][0] + sprite.width):
                    features.append((("player_target", sprite.width, sprite.height), x, y))
    direction = {0: (0, 1), 90: (-1, 0), 180: (0, -1), 270: (1, 0)}
    for item in spec.get("explosives", ()):
        sprite = prototypes[item["prototype"]]
        dx, dy = direction[item.get("rotation", 0)]
        # Prototype is behavior-bearing: the 6x6 large and mixed timers have
        # the same footprint but different initial charge rows.
        label = ("explosive", item["prototype"], sprite.width, sprite.height, dx, dy)
        for y in range(item["start"][1], item["start"][1] + sprite.height):
            for x in range(item["start"][0], item["start"][0] + sprite.width):
                features.append((label, x, y))
    return features


def _transform_point(x, y, swap, sx, sy):
    return (sx * (y if swap else x), sy * (x if swap else y))


def _canonical_feature_variants(features, semantic_actions=None):
    variants = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                transformed = []
                for raw_label, x, y in features:
                    label = raw_label
                    kind = label[0]
                    if kind in ("box", "box_target", "player", "player_target"):
                        _, width, height = label
                        label = (kind, height, width) if swap else label
                    elif kind == "explosive":
                        _, prototype, width, height, dx, dy = label
                        tdx, tdy = _transform_point(dx, dy, swap, sx, sy)
                        if swap:
                            width, height = height, width
                        label = (kind, prototype, width, height, tdx, tdy)
                    tx, ty = _transform_point(x, y, swap, sx, sy)
                    transformed.append((label, tx, ty))
                left = min(x for _, x, _ in transformed)
                top = min(y for _, _, y in transformed)
                geometry = sorted((label, x - left, y - top) for label, x, y in transformed)
                actions = []
                for token in semantic_actions or ():
                    if token[0] == "move":
                        dx, dy = _transform_point(token[1], token[2], swap, sx, sy)
                        actions.append(["move", dx, dy])
                    else:
                        actions.append(list(token))
                variants.append((geometry, actions))
    return variants


def _canonical_variants(spec, semantic_actions=None):
    return _canonical_feature_variants(_features(spec), semantic_actions)


def _geometry_hash(features):
    geometry_json = min(
        json.dumps(geometry, separators=(",", ":"))
        for geometry, _ in _canonical_feature_variants(features)
    )
    return hashlib.sha256(geometry_json.encode()).hexdigest()


def _identities(spec, semantic_actions):
    variants = _canonical_variants(spec, semantic_actions)
    geometry_json = min(
        json.dumps(geometry, separators=(",", ":")) for geometry, _ in variants
    )
    geometry = hashlib.sha256(geometry_json.encode()).hexdigest()
    gameplay_json = min(
        json.dumps([candidate, actions], separators=(",", ":"))
        for candidate, actions in variants
    )
    gameplay = hashlib.sha256(gameplay_json.encode()).hexdigest()
    action_json = min(
        json.dumps(actions, separators=(",", ":")) for _, actions in variants
    )
    action = hashlib.sha256(action_json.encode()).hexdigest()
    partition = SPLITS[int(geometry, 16) % len(SPLITS)]
    return geometry, gameplay, action, partition


def _level_features(level):
    """Extract the same behavior-bearing identity features from a native level."""
    features = []
    for wall in level.get_sprites_by_tag(names.TAG_WALL):
        pixels = np.asarray(wall.pixels)
        for py, px in np.argwhere(pixels != -1):
            features.append((("wall",), int(wall.x + px), int(wall.y + py)))
    for box in level.get_sprites_by_tag(names.TAG_BOX):
        label = ("box", int(box.width), int(box.height))
        for y in range(box.y, box.y + box.height):
            for x in range(box.x, box.x + box.width):
                features.append((label, x, y))
    for target in level.get_sprites_by_tag(names.TAG_TARGET):
        width, height = target.width - 2, target.height - 2
        label = ("box_target", int(width), int(height))
        for y in range(target.y + 1, target.y + 1 + height):
            for x in range(target.x + 1, target.x + 1 + width):
                features.append((label, x, y))
    for player in level.get_sprites_by_tag(names.TAG_PLAYER):
        label = ("player", int(player.width), int(player.height))
        for y in range(player.y, player.y + player.height):
            for x in range(player.x, player.x + player.width):
                features.append((label, x, y))
    for target in level.get_sprites_by_tag(names.TAG_PLAYER_TARGET):
        width, height = target.width - 2, target.height - 2
        label = ("player_target", int(width), int(height))
        for y in range(target.y + 1, target.y + 1 + height):
            for x in range(target.x + 1, target.x + 1 + width):
                features.append((label, x, y))
    direction = {0: (0, 1), 90: (-1, 0), 180: (0, -1), 270: (1, 0)}
    for explosive in level.get_sprites_by_tag(names.TAG_EXPLOSIVE):
        dx, dy = direction[int(explosive.rotation)]
        label = (
            "explosive", explosive.name, int(explosive.width), int(explosive.height), dx, dy,
        )
        for y in range(explosive.y, explosive.y + explosive.height):
            for x in range(explosive.x, explosive.x + explosive.width):
                features.append((label, x, y))
    return features


def generate_with_diagnostics(
    seed,
    difficulty=1,
    attempts=DEFAULT_ATTEMPTS,
    node_limit=DEFAULT_NODE_LIMIT,
    *,
    split=None,
):
    """Generate one tier while retaining a typed receipt on total rejection."""
    seed = _integer(seed, "seed", minimum=0)
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts", minimum=1)
    node_limit = _integer(node_limit, "node_limit", minimum=1)
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    requested_split = "train" if split is None else split
    mapped_seed, effective_split = effective_seed(seed, requested_split)
    exclusions = Counter()
    for attempt in range(1, attempts + 1):
        try:
            spec, commands = _draft(mapped_seed, difficulty, attempt)
            _derive_targets(spec, commands)
            _, actions = _execute_commands(spec, commands)
            mechanics, semantic_actions = _route_certificate(
                spec, actions, difficulty - 1
            )
            geometry, gameplay, action, partition = _identities(
                spec, semantic_actions
            )
            if partition != effective_split:
                exclusions["geometry_split"] += 1
                continue
            spec.update(
                requested_seed=seed,
                original_seed=seed,
                effective_seed=mapped_seed,
                split=effective_split,
                effective_split=effective_split,
                context_index=difficulty - 1,
                training_context_index=difficulty - 1,
                verification_level_index=difficulty - 1,
                solution=[list(action_value) for action_value in actions],
                context_solution=[list(action_value) for action_value in actions],
                semantic_solution=semantic_actions,
                solution_length=len(actions),
                solution_mechanics=mechanics,
                context_engine_verified=True,
                engine_verified=True,
                engine_win=True,
                levels_completed=1,
                geometry_sha256=geometry,
                geometry_d4_sha256=geometry,
                geometry_split=partition,
                gameplay_sha256=gameplay,
                action_sha256=action,
                search_limit=min(node_limit, PROFILES[difficulty]["search_limit"]),
                search_performed=False,
                search_exact=False,
                search_truncated=None,
                oracle_backend="constructive-native-witness",
                generation_attempt_limit=attempts,
                generation_exclusions=dict(exclusions),
                coverage={
                    "difficulty_profile": difficulty,
                    "generated_only": True,
                    "full_reference_profile": True,
                    "official_mechanics": [
                        "selectable box routing",
                        "recursive native pushing",
                        "internal walls" if spec["wall_rows"] else "open board",
                        "player routing" if spec["players"] else "no player",
                        "timed explosive blast pushes" if spec["explosives"] else "no explosives",
                    ],
                },
            )
            spec["structural_metrics"] = structural_metrics(spec)
            errors = profile_errors(spec)
            if errors:
                exclusions["profile:" + errors[0]] += 1
                continue
            spec["proof"] = {
                "kind": "constructive-real-engine-replay",
                "seed": mapped_seed,
                "difficulty": difficulty,
                "context_index": difficulty - 1,
                "context_engine_verified": True,
                "engine_verified": True,
                "engine_win": True,
                "levels_completed": 1,
                "action_count": len(actions),
                "mechanics": dict(mechanics),
                "search_limit": spec["search_limit"],
                "search_performed": False,
                "search_exact": False,
                "search_truncated": None,
                "oracle_backend": spec["oracle_backend"],
                "generator_version": GENERATOR_VERSION,
                "mechanics_version": MECHANICS_VERSION,
                "split": effective_split,
                "geometry_version": GEOMETRY_VERSION,
                "geometry_split": partition,
                "generation_attempt": attempt,
                "generation_attempt_limit": attempts,
                "generation_exclusions": dict(exclusions),
            }
            spec["generation_exclusions"] = dict(exclusions)
            return GenerationOutcome(spec, None)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            exclusions[f"draft:{type(exc).__name__}:{exc}"] += 1
    failure = {
        "format": "pebby.ka59.generation-failure.v1",
        "requested_seed": seed,
        "effective_seed": mapped_seed,
        "difficulty": difficulty,
        "split": effective_split,
        "attempt_limit": attempts,
        "node_limit": node_limit,
        "reasons": dict(exclusions),
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_version": QUALITY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
    }
    return GenerationOutcome(None, failure)


def generate(
    seed,
    difficulty=1,
    attempts=DEFAULT_ATTEMPTS,
    node_limit=DEFAULT_NODE_LIMIT,
    *,
    split=None,
):
    """Compatibility API returning a spec or ``None`` after bounded rejection."""
    return generate_with_diagnostics(
        seed,
        difficulty,
        attempts=attempts,
        node_limit=node_limit,
        split=split,
    ).spec


def generate_game_with_diagnostics(
    seed,
    *,
    split="train",
    difficulties=None,
    attempts=DEFAULT_ATTEMPTS,
    node_limit=DEFAULT_NODE_LIMIT,
):
    """Generate an ordered curriculum with retained per-child failure receipts."""
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not selected:
        raise ValueError("a generated game needs at least one difficulty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in selected):
        raise ValueError("game difficulties must be integers")
    selected = tuple(int(value) for value in selected)
    if any(value not in DIFFICULTIES for value in selected):
        raise ValueError(f"game difficulties must be drawn from {DIFFICULTIES}")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("game difficulties must be strictly increasing and distinct")
    specs = []
    for level_index, difficulty in enumerate(selected):
        child_seed = _game_level_seed(seed, level_index, difficulty)
        outcome = generate_with_diagnostics(
            child_seed,
            difficulty,
            attempts=attempts,
            node_limit=node_limit,
            split=split,
        )
        spec = outcome.spec
        if spec is None:
            failure = dict(outcome.failure or {})
            failure.update(
                game_seed=int(seed),
                game_level_index=level_index,
                child_seed=child_seed,
            )
            return GameGenerationOutcome(None, (failure,))
        spec.update(game_seed=int(seed), game_level_index=level_index)
        specs.append(spec)
    return GameGenerationOutcome(specs, ())


def generate_game(
    seed,
    *,
    split="train",
    difficulties=None,
    attempts=DEFAULT_ATTEMPTS,
    node_limit=DEFAULT_NODE_LIMIT,
):
    """Generate the full ordered seven-level curriculum by default."""
    return generate_game_with_diagnostics(
        seed,
        split=split,
        difficulties=difficulties,
        attempts=attempts,
        node_limit=node_limit,
    ).specs


def build_game(specs):
    """Build exactly seven ordered levels without shifting native contexts."""
    if (
        not isinstance(specs, Sequence)
        or isinstance(specs, (str, bytes))
        or len(specs) != len(DIFFICULTIES)
    ):
        raise ValueError(f"KA59 build_game requires exactly {len(DIFFICULTIES)} ordered specs")
    split = None
    levels = []
    for index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if (
            not isinstance(spec, Mapping)
            or type(spec.get("difficulty")) is not int
            or spec.get("difficulty") != difficulty
        ):
            raise ValueError("game specs must use difficulties 1..7 in order")
        for key in ("context_index", "training_context_index", "verification_level_index"):
            if type(spec.get(key)) is not int or spec.get(key) != index:
                raise ValueError(f"spec {index} has a shifted native {key}")
        if split is None:
            split = spec.get("split")
        elif spec.get("split") != split:
            raise ValueError("game specs must all use the same split")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"spec {index} violates the full contract: {'; '.join(errors)}")
        levels.append(build_level(spec))
    return levels


def validate_full_standard(spec, curriculum_entry):
    """Recompute structure, split identities, and native winning-route use."""
    errors = []
    if not isinstance(spec, Mapping):
        return ["generated spec must be an object"]
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be an object"]
    primitive_errors = _json_primitive_errors(spec)
    primitive_errors.extend(_json_primitive_errors(curriculum_entry, "curriculum"))
    if primitive_errors:
        return primitive_errors
    difficulty = curriculum_entry.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty must be an integer in 1..7"]
    expected_context = difficulty - 1
    profile = PROFILES[difficulty]
    if type(curriculum_entry.get("context_index")) is not int or curriculum_entry.get("context_index") != expected_context:
        errors.append("curriculum context differs from tier-1")
    if type(curriculum_entry.get("search_work")) is not int or curriculum_entry.get("search_work") != profile["search_limit"]:
        errors.append("curriculum search work differs from the declared tier cap")
    try:
        errors.extend(profile_errors(spec))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"profile validation failed: {exc}")
    try:
        metrics = structural_metrics(spec)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"structural metrics could not be recomputed: {exc}")
    else:
        if not _type_sensitive_equal(spec.get("structural_metrics"), metrics):
            errors.append("stored structural metrics differ from recomputation")
    expected_versions = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_version": QUALITY_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "source": "generated_only",
    }
    for key, expected in expected_versions.items():
        if not _type_sensitive_equal(spec.get(key), expected):
            errors.append(f"{key} differs from the pending-audit full-standard version")
    required_outcomes = {
        "context_engine_verified": True,
        "engine_verified": True,
        "engine_win": True,
        "levels_completed": 1,
        "search_performed": False,
        "search_exact": False,
        "search_truncated": None,
        "oracle_backend": "constructive-native-witness",
    }
    for key, expected in required_outcomes.items():
        if key not in spec:
            errors.append(f"required outcome field {key} is missing")
        elif type(spec[key]) is not type(expected) or spec[key] != expected:
            errors.append(f"{key} contradicts recomputed constructive proof")
    search_limit = spec.get("search_limit")
    if (
        isinstance(search_limit, bool)
        or not isinstance(search_limit, Integral)
        or not 1 <= int(search_limit) <= profile["search_limit"]
    ):
        errors.append("search_limit must be a positive integer within the tier cap")
    generation_attempt = spec.get("generation_attempt")
    attempt_limit = spec.get("generation_attempt_limit")
    if (
        isinstance(generation_attempt, bool)
        or not isinstance(generation_attempt, Integral)
        or isinstance(attempt_limit, bool)
        or not isinstance(attempt_limit, Integral)
        or not 1 <= int(generation_attempt) <= int(attempt_limit)
    ):
        errors.append("generation attempt and bound are missing or inconsistent")
    if type(spec.get("difficulty")) is not int or spec.get("difficulty") != difficulty:
        errors.append("spec difficulty differs from curriculum")
    for key in ("context_index", "training_context_index", "verification_level_index"):
        if type(spec.get(key)) is not int or spec.get(key) != expected_context:
            errors.append(f"{key} differs from the calibrated context")
    split = spec.get("split")
    requested_seed = spec.get("requested_seed")
    effective = spec.get("effective_seed")
    try:
        mapped, _ = effective_seed(requested_seed, split)
    except (TypeError, ValueError) as exc:
        errors.append(f"requested seed/split is malformed: {exc}")
    else:
        if effective != mapped or spec.get("seed") != mapped or spec.get("effective_split") != split:
            errors.append("effective seed does not match requested seed/split mapping")
    solution = spec.get("solution")
    if not isinstance(solution, list) or not _type_sensitive_equal(
        solution, spec.get("context_solution")
    ):
        errors.append("solution and context_solution must be identical action lists")
        actions = None
    else:
        try:
            actions = [
                (
                    _integer(value[0], "action id"),
                    value[1],
                    value[2],
                )
                for value in solution
                if isinstance(value, list) and len(value) == 3
            ]
            if len(actions) != len(solution):
                raise ValueError("malformed action triple")
        except (TypeError, ValueError, IndexError) as exc:
            errors.append(f"solution actions are malformed: {exc}")
            actions = None
    if actions is not None:
        try:
            mechanics, semantic_actions = _route_certificate(spec, actions, expected_context)
            geometry, gameplay, action, partition = _identities(spec, semantic_actions)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"stored route/geometry could not be recomputed: {type(exc).__name__}: {exc}")
        else:
            if spec.get("solution_length") != len(actions):
                errors.append("solution_length differs from the stored route")
            if not _type_sensitive_equal(spec.get("semantic_solution"), semantic_actions):
                errors.append("semantic action sequence differs from native replay")
            if not _type_sensitive_equal(spec.get("solution_mechanics"), mechanics):
                errors.append("solution mechanics differ from native replay")
            expected_identities = {
                "geometry_sha256": geometry,
                "geometry_d4_sha256": geometry,
                "gameplay_sha256": gameplay,
                "action_sha256": action,
                "geometry_split": partition,
            }
            for key, expected in expected_identities.items():
                if not _type_sensitive_equal(spec.get(key), expected):
                    errors.append(f"{key} differs from recomputation")
            if split != partition:
                errors.append("canonical geometry partition differs from requested split")
            if geometry in _official_geometry_hashes():
                errors.append("generated D4 geometry duplicates an official level")
            try:
                frame = np.asarray(Env([build_level(spec)]).render(), dtype=np.uint8)
                frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                if frame_hash in _official_start_frame_hashes():
                    errors.append("generated start frame duplicates an official level")
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                errors.append(f"official-copy check could not render the spec: {exc}")
    proof = spec.get("proof")
    if not isinstance(proof, Mapping):
        errors.append("nested proof is missing")
    else:
        mirrors = {
            "seed": effective,
            "difficulty": difficulty,
            "context_index": expected_context,
            "context_engine_verified": spec.get("context_engine_verified"),
            "engine_verified": spec.get("engine_verified"),
            "engine_win": spec.get("engine_win"),
            "levels_completed": spec.get("levels_completed"),
            "action_count": spec.get("solution_length"),
            "mechanics": spec.get("solution_mechanics"),
            "search_limit": spec.get("search_limit"),
            "search_performed": spec.get("search_performed"),
            "search_exact": spec.get("search_exact"),
            "search_truncated": spec.get("search_truncated"),
            "oracle_backend": spec.get("oracle_backend"),
            "generator_version": GENERATOR_VERSION,
            "mechanics_version": MECHANICS_VERSION,
            "split": split,
            "geometry_version": GEOMETRY_VERSION,
            "geometry_split": spec.get("geometry_split"),
            "generation_attempt": spec.get("generation_attempt"),
            "generation_attempt_limit": spec.get("generation_attempt_limit"),
            "generation_exclusions": spec.get("generation_exclusions"),
        }
        if proof.get("kind") != "constructive-real-engine-replay":
            errors.append("proof.kind is missing or incorrect")
        for key, expected in mirrors.items():
            if key not in proof or not _type_sensitive_equal(proof[key], expected):
                errors.append(f"proof.{key} does not mirror top-level evidence")
    exclusions = spec.get("generation_exclusions")
    if not isinstance(exclusions, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
        for key, value in (exclusions.items() if isinstance(exclusions, dict) else ())
    ):
        errors.append("bounded generation rejection counts are missing or malformed")
    elif isinstance(generation_attempt, Integral) and not isinstance(generation_attempt, bool):
        if sum(int(value) for value in exclusions.values()) != int(generation_attempt) - 1:
            errors.append("generation exclusions do not account for every rejected attempt")
    return errors


def _official_start_frame_hashes():
    global _OFFICIAL_FRAME_HASHES
    if _OFFICIAL_FRAME_HASHES is None:
        from .env import official_levels

        _OFFICIAL_FRAME_HASHES = frozenset(
            hashlib.sha256(
                np.asarray(Env([level]).render(), dtype=np.uint8).tobytes()
            ).hexdigest()
            for level in official_levels()
        )
    return _OFFICIAL_FRAME_HASHES


def _official_geometry_hashes():
    global _OFFICIAL_GEOMETRY_HASHES
    if _OFFICIAL_GEOMETRY_HASHES is None:
        from .env import official_levels

        _OFFICIAL_GEOMETRY_HASHES = frozenset(
            _geometry_hash(_level_features(level)) for level in official_levels()
        )
    return _OFFICIAL_GEOMETRY_HASHES


def replays_to_completion(spec, *, context_index=None):
    context = spec.get("verification_level_index", spec["difficulty"] - 1)
    if context_index is not None:
        context = context_index
    try:
        _route_certificate(
            spec,
            [tuple(action) for action in spec["solution"]],
            int(context),
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return True
