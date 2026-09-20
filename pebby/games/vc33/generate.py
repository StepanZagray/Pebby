"""Full-mechanic, reference-calibrated VC33 procedural generation.

The seven generated tiers follow the seven shipped native contexts without
copying their layouts or routes. Candidates use fresh support-chain geometry,
are solved by the exact compact teacher, and are replayed in the unmodified
engine at their intended native level index.
"""

from collections import Counter
from functools import lru_cache
import hashlib
import json
import random

import numpy as np
from arcengine import GameState, Level, Sprite

from . import names
from .env import Env, official_levels
from .plan import search


FORMAT = "pebby.vc33.level.v3"
GENERATOR_VERSION = 3
MECHANICS_VERSION = "vc33-full-native-v2"
QUALITY_PROFILE_VERSION = "vc33-seven-reference-tiers-v2"
IDENTITY_VERSION = "vc33-d4-gameplay-v2"
SOURCE_ID = "vc33-5430563c"
DIFFICULTIES = tuple(range(1, len(official_levels()) + 1))
SPLITS = ("train", "validation", "test")
DEFAULT_ATTEMPTS = 48

# One shipped level exists at each tier. These are explicit engineering
# tolerances around those scarce references, not population confidence bounds.
# Structural facts come from third_party/arc3_games/vc33.py:1553-1742. Native
# mechanics come from :1829-2123. Exact action lengths were independently
# recomputed by plan.search and replayed in the native engine on 2026-09-18.
REFERENCE_PROFILES = {
    1: {"grid": 32, "budget": 50, "gravity": (2, 0), "loads": 1,
        "targets": 1, "buttons": 2, "swaps": 0, "floors": 0,
        "walls": 1, "supports": 2, "density": (0.28, 0.44),
        "reference_density": 0.362305, "reference_actions": 3,
        "actions": (2, 6), "search_work": 20_000},
    2: {"grid": 32, "budget": 50, "gravity": (-2, 0), "loads": 1,
        "targets": 1, "buttons": 4, "swaps": 0, "floors": 0,
        "walls": 2, "supports": 3, "density": (0.32, 0.52),
        "reference_density": 0.444336, "reference_actions": 7,
        "actions": (5, 11), "search_work": 50_000},
    3: {"grid": 52, "budget": 75, "gravity": (0, 2), "loads": 3,
        "targets": 3, "buttons": 8, "swaps": 0, "floors": 0,
        "walls": 4, "supports": 5, "density": (0.13, 0.31),
        "reference_density": 0.200074, "reference_actions": 23,
        "actions": (16, 30), "search_work": 200_000},
    4: {"grid": 64, "budget": 50, "gravity": (0, 3), "loads": 1,
        "targets": 1, "buttons": 6, "swaps": 2, "floors": 0,
        "walls": 4, "supports": 5, "density": (0.16, 0.34),
        "reference_density": 0.235352, "reference_actions": 21,
        "actions": (13, 29), "search_work": 300_000},
    5: {"grid": 64, "budget": 200, "gravity": (3, 0), "loads": 2,
        "targets": 2, "buttons": 6, "swaps": 3, "floors": 0,
        "walls": 3, "supports": 4, "density": (0.23, 0.44),
        "reference_density": 0.426025, "reference_actions": 44,
        "actions": (28, 56), "search_work": 700_000},
    6: {"grid": 64, "budget": 50, "gravity": (-3, 0), "loads": 1,
        "targets": 1, "buttons": 4, "swaps": 2, "floors": 1,
        "walls": 1, "supports": 3, "density": (0.20, 0.40),
        "reference_density": 0.338623, "reference_actions": 20,
        "actions": (14, 28), "search_work": 400_000},
    7: {"grid": 48, "budget": 200, "gravity": (0, -2), "loads": 3,
        "targets": 3, "buttons": 8, "swaps": 3, "floors": 2,
        "walls": 2, "supports": 5, "density": (0.32, 0.54),
        "reference_density": 0.443142, "reference_actions": 49,
        "actions": (28, 62), "search_work": 1_000_000},
}

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": SOURCE_ID,
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": tuple(
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": REFERENCE_PROFILES[difficulty]["search_work"],
        }
        for difficulty in DIFFICULTIES
    ),
    "evidence": {
        "official_tier_characterization":
            "vc33-final.md#official-reference-characterization",
        "solution_mechanics": "spec.proof.mechanic_use",
        "native_budget": "spec.proof.native_budget",
        "context_engine_replay": "spec.proof.context_replay",
        "novelty_split": "spec.identities",
        "bounded_rejections": "spec.generation.rejections",
    },
    "caveats": (
        "Each calibration tier contains one official level; tolerances are engineering bounds.",
        "Root acceptance is based on bounded native replay, frame review, and sampled-quality evidence; compact-solver optimality lacks an independent proof.",
    ),
}

_TAG_CHANNELS = (
    names.TAG_FLOOR,
    names.TAG_SWAP,
    names.TAG_TARGET,
    names.TAG_LOAD,
    names.TAG_BUTTON,
    names.TAG_WALL,
    names.TAG_SUPPORT,
)


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _strict_equal(value, expected):
    """Compare JSON-like evidence without Python's bool/int/float aliases."""
    if isinstance(expected, dict):
        return (
            isinstance(value, dict)
            and value.keys() == expected.keys()
            and all(_strict_equal(value[key], item)
                    for key, item in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(_strict_equal(actual, item)
                    for actual, item in zip(value, expected))
        )
    return type(value) is type(expected) and value == expected


def _split(value):
    if value not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return value


def _rect_shape(horizontal, cross_size, primary_size):
    # Horizontal native levels rotate their prototypes by 90/270 degrees, so
    # Sprite.width is the array's first dimension.  Using one canonical
    # (primary, cross) array order preserves the engine's resize semantics.
    return (primary_size, cross_size)


def _position(horizontal, cross, primary):
    return (primary, cross) if horizontal else (cross, primary)


def _sprite(name, shape, color, tags, position, *, layer=0, pixels=None):
    array = np.full(shape, color, dtype=np.int8) if pixels is None else pixels
    return Sprite(
        pixels=array,
        name=name,
        visible=True,
        collidable=True,
        tags=list(tags),
        layer=layer,
    ).set_position(*position)


def _load_pixels(horizontal, size, color):
    if size == 3:
        pixels = np.array(((-1, 4, -1), (4, 4, 4),
                           (color, color, color)), dtype=np.int8)
    else:
        pixels = np.full((6, 6), 4, dtype=np.int8)
        pixels[:2, :2] = -1
        pixels[:2, 4:] = -1
        pixels[-2:, :] = color
    return pixels


def _target_pixels(horizontal, cross_size, primary_size, color):
    pixels = np.full((primary_size, cross_size), -2, dtype=np.int8)
    pixels[-1:, :] = color
    return pixels


def _wall_pixels(primary_size, cross_size):
    """Return one native-like contiguous structural wall segment."""
    return np.full((primary_size, cross_size), 5, dtype=np.int8)


def _partition_widths(rng, count, total, minimum):
    widths = [minimum] * count
    remaining = total - minimum * count
    if remaining < 0:
        raise ValueError("support width budget is too small")
    for _ in range(remaining):
        widths[rng.randrange(count)] += 1
    return widths


def _tier_plan(rng, difficulty):
    """Return a fresh support-chain state and intended target state."""
    profile = REFERENCE_PROFILES[difficulty]
    grid = profile["grid"]
    gravity = profile["gravity"]
    magnitude = max(abs(value) for value in gravity)
    support_count = profile["supports"]
    button_size = 3 if difficulty in (4, 5, 6) else 2
    gaps = support_count - 1
    horizontal = bool(gravity[0])

    support_totals = {1: 20, 2: 21, 3: 18, 4: 32, 5: 42, 6: 46, 7: 31}
    usable = support_totals[difficulty] + rng.randint(-1, 1)
    minimum = 6 if difficulty in (4, 5, 6) else 3
    widths = _partition_widths(rng, support_count, usable, minimum)
    chain_width = sum(widths) + button_size * gaps
    cross_origin = rng.randint(0, grid - chain_width)
    support_cross = []
    cursor = cross_origin
    for width in widths:
        support_cross.append(cursor)
        cursor += width + button_size

    colors = rng.sample((11, 14, 15), profile["loads"])
    swaps = ()
    floors = ()
    wall_gaps = tuple(range(gaps))
    button_gaps = tuple(range(gaps))
    load_size = 6 if difficulty in (4, 5, 6) else 3

    if difficulty == 1:
        base = rng.randint(17, 20)
        clicks = rng.randint(2, 4)
        initial_edges = [base, base]
        assignments = [1]
        goal_assignments = [1]
        goal_edges = {1: base - clicks * magnitude}
    elif difficulty == 2:
        base = rng.randint(14, 16)
        clicks = rng.randint(5, 6)
        initial_edges = [base] * 3
        assignments = [2]
        goal_assignments = [2]
        goal_edges = {2: base + clicks * magnitude}
    elif difficulty == 3:
        base = rng.randint(23, 25)
        counts = [rng.randint(5, 7), rng.randint(6, 8), rng.randint(5, 7)]
        initial_edges = [base] * 5
        assignments = [0, 2, 4]
        goal_assignments = list(assignments)
        goal_edges = {
            support: base - clicks * magnitude
            for support, clicks in zip(assignments, counts)
        }
    elif difficulty == 4:
        base = 36
        align = rng.randint(6, 8)
        finish = rng.randint(8, 10)
        initial_edges = [base + align * magnitude,
                         base - align * magnitude, base, base, base]
        assignments = [0]
        goal_assignments = [2]
        goal_edges = {2: base - finish * magnitude}
        button_gaps = (0, 2, 3)
        swaps = (0, 1)
    elif difficulty == 5:
        base = rng.randint(34, 35)
        align_left = rng.randint(5, 6)
        align_right = rng.randint(5, 6)
        finish_left = rng.randint(8, 9)
        finish_right = rng.randint(8, 9)
        initial_edges = [base + align_left * magnitude,
                         base - align_left * magnitude,
                         base - align_right * magnitude,
                         base + align_right * magnitude]
        assignments = [0, 3]
        goal_assignments = [3, 0]
        goal_edges = {
            0: base - finish_left * magnitude,
            3: base - finish_right * magnitude,
        }
        swaps = (0, 1, 2)
    elif difficulty == 6:
        base = 27
        align = rng.randint(6, 7)
        finish = rng.randint(9, 10)
        initial_edges = [base - align * magnitude,
                         base + align * magnitude, base]
        assignments = [0]
        goal_assignments = [2]
        goal_edges = {2: base + finish * magnitude}
        swaps = (0, 1)
        floors = ((0, base + 6),)
        wall_gaps = (1,)
    else:
        base = 24
        align_left = rng.randint(9, 10)
        align_right = rng.randint(9, 10)
        finish = rng.randint(7, 9)
        initial_edges = [base - align_left * magnitude,
                         base + align_left * magnitude,
                         base - align_right * magnitude,
                         base + align_right * magnitude, base]
        assignments = [0, 2, 3]
        goal_assignments = [2, 3, 1]
        goal_edges = {
            1: base + finish * magnitude,
            2: base - finish * magnitude,
        }
        swaps = (0, 1, 2)
        # Both floor-backed receivers carry loads during their alignment.
        floors = ((0, base + 4), (2, base + 4))
        wall_gaps = (1, 3)

    target_walls = []
    for load_index, support in enumerate(goal_assignments):
        candidates = [gap for gap in wall_gaps if support in (gap, gap + 1)]
        if not candidates:
            raise ValueError("goal support has no target-bearing wall")
        target_walls.append(candidates[load_index % len(candidates)])

    goal_edge_values = list(initial_edges)
    for support, edge in goal_edges.items():
        goal_edge_values[support] = edge
    return {
        "horizontal": horizontal,
        "button_size": button_size,
        "widths": widths,
        "support_cross": support_cross,
        "initial_edges": initial_edges,
        "assignments": assignments,
        "goal_assignments": goal_assignments,
        "goal_edges": goal_edge_values,
        "colors": colors,
        "load_size": load_size,
        "button_gaps": list(button_gaps),
        "wall_gaps": list(wall_gaps),
        "swaps": list(swaps),
        "floors": [list(value) for value in floors],
        "target_walls": target_walls,
        "swap_edge": base,
    }


def _draft(seed, difficulty, attempt):
    namespace = f"{MECHANICS_VERSION}:{seed}:{difficulty}:{attempt}".encode()
    rng_seed = int.from_bytes(hashlib.blake2b(namespace, digest_size=16).digest(), "big")
    rng = random.Random(rng_seed)
    profile = REFERENCE_PROFILES[difficulty]
    plan = _tier_plan(rng, difficulty)
    goal_assignments = plan.pop("goal_assignments")
    goal_edges = plan.pop("goal_edges")
    target_walls = plan.pop("target_walls")
    load_size = plan["load_size"]
    targets = []
    positive = profile["gravity"][0] > 0 or profile["gravity"][1] > 0
    for support, wall, color in zip(
            goal_assignments, target_walls, plan["colors"]):
        edge = goal_edges[support]
        primary = edge - load_size if positive else edge
        targets.append({
            "wall_gap": wall,
            "primary": primary,
            "color": color,
        })
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "identity_version": IDENTITY_VERSION,
        "source": "generated_only",
        "source_id": SOURCE_ID,
        "seed": int(seed),
        "attempt": int(attempt),
        "difficulty": int(difficulty),
        "training_context_index": int(difficulty - 1),
        "verification_level_index": int(difficulty - 1),
        "grid_size": profile["grid"],
        "step_budget": profile["budget"],
        "gravity": list(profile["gravity"]),
        "plan": plan,
        "targets": targets,
    }


def build_level(spec):
    """Build one native level from a JSON-round-trippable VC33 v3 spec."""
    if spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    profile = REFERENCE_PROFILES[difficulty]
    grid = _integer(spec.get("grid_size"), "grid_size")
    budget = _integer(spec.get("step_budget"), "step_budget")
    gravity = tuple(_integer(value, "gravity") for value in spec.get("gravity", ()))
    if grid != profile["grid"] or budget != profile["budget"]:
        raise ValueError("grid or budget does not match the calibrated tier")
    if gravity != profile["gravity"]:
        raise ValueError("gravity does not match the calibrated tier")
    if _integer(spec.get("training_context_index"), "training_context_index") != difficulty - 1:
        raise ValueError("training context must equal difficulty - 1")
    if _integer(spec.get("verification_level_index"), "verification_level_index") != difficulty - 1:
        raise ValueError("verification context must equal difficulty - 1")

    plan = spec.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    horizontal = bool(gravity[0])
    if plan.get("horizontal") is not horizontal:
        raise ValueError("plan orientation disagrees with gravity")
    button_size = _integer(plan.get("button_size"), "button_size")
    widths = [_integer(value, "support width") for value in plan.get("widths", ())]
    cross = [_integer(value, "support cross") for value in plan.get("support_cross", ())]
    edges = [_integer(value, "support edge") for value in plan.get("initial_edges", ())]
    if len(widths) != profile["supports"] or len(cross) != len(widths) or len(edges) != len(widths):
        raise ValueError("support geometry count does not match the tier")
    if min(widths) < 3 or min(edges) < 1 or max(edges) >= grid:
        raise ValueError("support geometry is out of bounds")
    for index in range(1, len(widths)):
        if cross[index] != cross[index - 1] + widths[index - 1] + button_size:
            raise ValueError("supports must form a chain separated by button-sized gaps")
    if cross[0] < 0 or cross[-1] + widths[-1] > grid:
        raise ValueError("support chain leaves the grid")

    sprites = []
    positive = gravity[0] > 0 or gravity[1] > 0
    rotation = 270 if gravity[0] > 0 else 90 if gravity[0] < 0 else 0
    for index, (cross_start, width, edge) in enumerate(zip(cross, widths, edges)):
        primary_start = edge if positive else 0
        primary_size = grid - edge if positive else edge
        sprites.append(_sprite(
            f"generated-vc33-support-{index}",
            _rect_shape(horizontal, width, primary_size), 0,
            (names.TAG_SUPPORT,), _position(horizontal, cross_start, primary_start),
            layer=-1,
        ))

    wall_gaps = [_integer(value, "wall gap") for value in plan.get("wall_gaps", ())]
    if len(set(wall_gaps)) != len(wall_gaps):
        raise ValueError("wall gaps must be distinct")
    load_size = _integer(plan.get("load_size"), "load_size")
    targets = list(spec.get("targets", ()))
    if len(targets) != profile["targets"]:
        raise ValueError("target count does not match the tier")
    target_primary_size = 3 if load_size == 6 else 2
    target_geometry = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"targets[{index}] must be an object")
        gap = _integer(target.get("wall_gap"), f"targets[{index}].wall_gap")
        primary = _integer(target.get("primary"), f"targets[{index}].primary")
        if gap not in wall_gaps:
            raise ValueError("target must use a declared wall")
        if not (0 <= primary <= grid - target_primary_size):
            raise ValueError("target leaves the grid")
        target_geometry.append((gap, primary))
    for gap in wall_gaps:
        if not (0 <= gap < len(widths) - 1):
            raise ValueError("wall gap is out of range")
        wall_cross = cross[gap] + widths[gap]
        wall_targets = [primary for target_gap, primary in target_geometry
                        if target_gap == gap]
        if positive:
            wall_start = 0
            wall_end = max(
                [button_size]
                + [primary + target_primary_size for primary in wall_targets]
            )
        else:
            wall_start = min(
                [grid - button_size]
                + wall_targets
            )
            wall_end = grid
        wall_size = wall_end - wall_start
        sprites.append(_sprite(
            f"generated-vc33-wall-{gap}",
            _rect_shape(horizontal, button_size, wall_size), 5,
            (names.TAG_WALL,),
            _position(horizontal, wall_cross, wall_start),
            pixels=_wall_pixels(wall_size, button_size),
        ))

    floors = list(plan.get("floors", ()))
    for floor_index, value in enumerate(floors):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("each floor needs support index and primary coordinate")
        support = _integer(value[0], "floor support")
        primary = _integer(value[1], "floor primary")
        if not (0 <= support < len(widths)) or not (0 <= primary < grid - 1):
            raise ValueError("floor is out of range")
        sprites.append(_sprite(
            f"generated-vc33-floor-{floor_index}",
            _rect_shape(horizontal, widths[support], 2), 5,
            (names.TAG_FLOOR,), _position(horizontal, cross[support], primary),
        ))

    assignments = [_integer(value, "load assignment")
                   for value in plan.get("assignments", ())]
    colors = [_integer(value, "load color") for value in plan.get("colors", ())]
    if len(assignments) != profile["loads"] or len(colors) != len(assignments):
        raise ValueError("load assignment count does not match the tier")
    if len(set(colors)) != len(colors) or any(color not in (11, 14, 15) for color in colors):
        raise ValueError("loads need distinct supported colors")
    for load_index, (support, color) in enumerate(zip(assignments, colors)):
        if not (0 <= support < len(widths)) or widths[support] < load_size:
            raise ValueError("load does not fit its support")
        load_cross = cross[support] + (widths[support] - load_size) // 2
        load_primary = edges[support] - load_size if positive else edges[support]
        if not (0 <= load_primary <= grid - load_size):
            raise ValueError("load leaves the grid")
        sprites.append(_sprite(
            f"generated-vc33-load-{load_index}",
            _rect_shape(horizontal, load_size, load_size), color,
            (names.TAG_LOAD,), _position(horizontal, load_cross, load_primary),
            layer=2, pixels=_load_pixels(horizontal, load_size, color),
        ))

    for index, target in enumerate(targets):
        gap = _integer(target.get("wall_gap"), f"targets[{index}].wall_gap")
        primary = _integer(target.get("primary"), f"targets[{index}].primary")
        color = _integer(target.get("color"), f"targets[{index}].color")
        if gap not in wall_gaps or color not in colors:
            raise ValueError("target must use a declared wall and load color")
        primary_size = 3 if load_size == 6 else 2
        if not (0 <= primary <= grid - primary_size):
            raise ValueError("target leaves the grid")
        wall_cross = cross[gap] + widths[gap]
        sprites.append(_sprite(
            f"generated-vc33-target-{index}",
            _rect_shape(horizontal, button_size, primary_size), color,
            (names.TAG_TARGET,), _position(horizontal, wall_cross, primary),
            layer=1,
            pixels=_target_pixels(horizontal, button_size, primary_size, color),
        ))

    button_primary = grid - button_size if positive else 0
    button_gaps = [_integer(value, "button gap")
                   for value in plan.get("button_gaps", ())]
    for gap in button_gaps:
        if not (0 <= gap < len(widths) - 1):
            raise ValueError("button gap is out of range")
        left_end = cross[gap] + widths[gap]
        right_start = cross[gap + 1]
        for direction, button_cross in (
                ("forward", right_start),
                ("reverse", left_end - button_size)):
            sprites.append(_sprite(
                f"generated-vc33-button-{gap}-{direction}",
                _rect_shape(horizontal, button_size, button_size), 9,
                (names.TAG_BUTTON, names.TAG_CLICK),
                _position(horizontal, button_cross, button_primary), layer=3,
            ))

    swaps = [_integer(value, "swap gap") for value in plan.get("swaps", ())]
    for gap in swaps:
        if not (0 <= gap < len(widths) - 1):
            raise ValueError("swap gap is out of range")
        swap_cross = cross[gap] + widths[gap]
        swap_edge = _integer(plan.get("swap_edge"), "swap edge")
        primary_size = 8 if button_size == 2 else 12
        swap_primary = swap_edge - primary_size if positive else swap_edge
        if not (0 <= swap_primary <= grid - primary_size):
            raise ValueError("swap bar leaves the grid")
        sprites.append(_sprite(
            f"generated-vc33-swap-{gap}",
            _rect_shape(horizontal, button_size, primary_size), 1,
            (names.TAG_SWAP, names.TAG_CLICK),
            _position(horizontal, swap_cross, swap_primary), layer=3,
        ))

    if horizontal:
        for sprite in sprites:
            sprite.set_rotation(rotation)

    level = Level(
        sprites=sprites,
        grid_size=(grid, grid),
        data={names.KEY_STEPS: budget, names.KEY_GRAVITY: list(gravity)},
        name=f"generated-vc33-d{difficulty}-s{spec.get('seed', 0)}-a{spec.get('attempt', 0)}",
    )
    env = Env([level])
    if len(getattr(env.game, names.ATTR_BUTTON_PAIRS)) != len(button_gaps) * 2:
        raise ValueError("native engine did not discover every generated button pair")
    return level


def _metrics(level):
    grid = int(level.grid_size[0])
    counts = {
        "loads": len(level.get_sprites_by_tag(names.TAG_LOAD)),
        "targets": len(level.get_sprites_by_tag(names.TAG_TARGET)),
        "buttons": len(level.get_sprites_by_tag(names.TAG_BUTTON)),
        "swaps": len(level.get_sprites_by_tag(names.TAG_SWAP)),
        "floors": len(level.get_sprites_by_tag(names.TAG_FLOOR)),
        "walls": len(level.get_sprites_by_tag(names.TAG_WALL)),
        "supports": len(level.get_sprites_by_tag(names.TAG_SUPPORT)),
    }
    occupied = np.zeros((grid, grid), dtype=bool)
    for sprite in level.get_sprites():
        pixels = _oriented_pixels(sprite)
        ys, xs = np.where(pixels != -1)
        for dy, dx in zip(ys, xs):
            x, y = int(sprite.x + dx), int(sprite.y + dy)
            if 0 <= x < grid and 0 <= y < grid:
                occupied[y, x] = True
    return {
        "grid_size": grid,
        "step_budget": _integer(level.get_data(names.KEY_STEPS), "StepCounter"),
        "gravity": [int(value) for value in level.get_data(names.KEY_GRAVITY)],
        **counts,
        "occupied_pixels": int(occupied.sum()),
        "visual_density": round(float(occupied.mean()), 6),
    }


def _geometry_tensor(level):
    grid = int(level.grid_size[0])
    channels = np.zeros((len(_TAG_CHANNELS), grid, grid), dtype=np.uint8)
    tag_index = {tag: index for index, tag in enumerate(_TAG_CHANNELS)}
    for sprite in level.get_sprites():
        relevant = [tag_index[tag] for tag in sprite.tags if tag in tag_index]
        if not relevant:
            continue
        pixels = _oriented_pixels(sprite)
        ys, xs = np.where(pixels != -1)
        for dy, dx in zip(ys, xs):
            x, y = int(sprite.x + dx), int(sprite.y + dy)
            if 0 <= x < grid and 0 <= y < grid:
                channels[relevant, y, x] = 1
    used = np.any(channels, axis=0)
    ys, xs = np.where(used)
    if len(xs):
        channels = channels[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return channels


def _oriented_pixels(sprite):
    pixels = np.asarray(sprite.pixels)
    turns = (int(sprite.rotation) // 90) % 4
    return np.rot90(pixels, k=-turns) if turns else pixels


def _d4_hash(level):
    tensor = _geometry_tensor(level)
    variants = []
    for reflected in (False, True):
        value = tensor[:, :, ::-1] if reflected else tensor
        for turns in range(4):
            transformed = np.rot90(value, turns, axes=(1, 2))
            header = f"{transformed.shape[1]}x{transformed.shape[2]}:".encode()
            variants.append(header + transformed.tobytes())
    return hashlib.sha256(min(variants)).hexdigest()


def _gameplay_hash(spec, level=None):
    plan = spec["plan"]
    level = build_level(spec) if level is None else level
    payload = {
        "identity_version": IDENTITY_VERSION,
        # The public, translation/D4-normalized executed board keeps gameplay
        # identities split-safe without hashing private construction metadata.
        "geometry_d4_sha256": _d4_hash(level),
        "difficulty": spec["difficulty"],
        "grid_size": spec["grid_size"],
        "step_budget": spec["step_budget"],
        "gravity": spec["gravity"],
        # Only fields executed by the stable native transition system belong
        # here. Absolute chain translation, drawing dimensions, construction
        # helpers, solution, and proof metadata are deliberately excluded.
        "state": {
            "support_count": len(plan["widths"]),
            "initial_edges": plan["initial_edges"],
            "assignments": plan["assignments"],
            "load_colors": plan["colors"],
            "load_size": plan["load_size"],
            "button_gaps": plan["button_gaps"],
            "wall_gaps": plan["wall_gaps"],
            "swaps": plan["swaps"],
            "floors": plan["floors"],
            "swap_edge": plan["swap_edge"],
        },
        "targets": sorted(
            [{
                "wall_gap": target["wall_gap"],
                "primary": target["primary"],
                "color": target["color"],
            } for target in spec["targets"]],
            key=lambda target: (
            target["wall_gap"], target["primary"], target["color"]
            ),
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_split(identity):
    return SPLITS[int(identity[:16], 16) % len(SPLITS)]


@lru_cache(maxsize=1)
def _official_hashes():
    return frozenset(_d4_hash(level) for level in official_levels())


def _support_assignment(env, load):
    supports = env.level.get_sprites_by_tag(names.TAG_SUPPORT)
    matches = [index for index, support in enumerate(supports)
               if env.game.bcpuwqzpxw(load, support)]
    return matches[0] if len(matches) == 1 else None


def _edge(env, support):
    return int(env.game.hpakcxndwy(support))


def _context_replay(spec, actions):
    """Replay at the declared real native index and derive mechanic evidence."""
    difficulty = spec["difficulty"]
    levels = [build_level(spec) for _ in range(difficulty)]
    env = Env(levels)
    for prefix in range(difficulty - 1):
        before = env.levels_completed
        for action in actions:
            observation = env.perform(*action)
            if env.levels_completed > before:
                break
        else:
            return None, f"prefix replay did not complete context {prefix}"
        if env.level_index != prefix + 1:
            return None, "native prefix advanced to the wrong level index"

    if env.level_index != difficulty - 1:
        return None, "candidate did not start at its declared native context"
    before_score = env.levels_completed
    minimum_budget = env.steps_left
    used_buttons = set()
    used_swaps = set()
    floor_transfers = 0
    floor_limits = set()
    coupled_multi_load_transfers = 0
    animation_frames = 0
    changed_actions = 0

    for ordinal, action in enumerate(actions):
        point = env.game.camera.display_to_grid(action[1], action[2])
        hit = env.level.get_sprite_at(*point) if point is not None else None
        supports = env.level.get_sprites_by_tag(names.TAG_SUPPORT)
        loads = env.level.get_sprites_by_tag(names.TAG_LOAD)
        before_edges = tuple(_edge(env, support) for support in supports)
        before_positions = tuple((int(load.x), int(load.y)) for load in loads)
        before_assignments = tuple(_support_assignment(env, load) for load in loads)
        receiver = None
        receiver_limit = None
        coupled_loads = 0
        positive = env.game.qhmwbtpcsk()
        if hit is not None and names.TAG_BUTTON in hit.tags:
            pair = getattr(env.game, names.ATTR_BUTTON_PAIRS).get(hit)
            if pair:
                receiver = pair[1]
                pair_indices = {supports.index(support) for support in pair}
                coupled_loads = sum(
                    assignment in pair_indices for assignment in before_assignments
                )
                floors = env.game.kectayqmfn(receiver)
                if floors:
                    floor_transfers += 1
                    receiver_limit = int(env.game.ysoqxdegud(receiver))
            used_buttons.add(hit.name)
        elif hit is not None and names.TAG_SWAP in hit.tags:
            used_swaps.add(hit.name)

        observation = env.perform(*action)
        minimum_budget = min(minimum_budget, env.steps_left)
        animation_frames += max(0, len(observation.frames) - 1)
        won = env.levels_completed > before_score or observation.state == GameState.WIN
        if won:
            after_edges = before_edges
            after_positions = before_positions
            after_assignments = before_assignments
            # Native completion proves that this final action changed the
            # just-finished level even though the engine has already replaced
            # it with the next context, so its post-state is no longer visible.
            changed_actions += 1
        else:
            after_supports = env.level.get_sprites_by_tag(names.TAG_SUPPORT)
            after_loads = env.level.get_sprites_by_tag(names.TAG_LOAD)
            after_edges = tuple(_edge(env, support) for support in after_supports)
            after_positions = tuple((int(load.x), int(load.y)) for load in after_loads)
            after_assignments = tuple(_support_assignment(env, load) for load in after_loads)
        if (after_edges, after_positions, after_assignments) != (
                before_edges, before_positions, before_assignments):
            changed_actions += 1
        if receiver is not None and receiver_limit is not None:
            receiver_index = supports.index(receiver)
            after_value = after_edges[receiver_index]
            if ((positive and after_value <= receiver_limit)
                    or (not positive and after_value >= receiver_limit)):
                floor_limits.add(receiver.name)
        if before_positions != after_positions and coupled_loads >= 2:
            coupled_multi_load_transfers += 1
        if won:
            if ordinal != len(actions) - 1:
                return None, "stored route wins before its final action"
            break
        if observation.state == GameState.GAME_OVER:
            return None, "stored route exhausted the native budget"
    else:
        return None, "stored route did not complete the candidate"

    return {
        "action_count": len(actions),
        "changed_actions": changed_actions,
        "distinct_buttons": len(used_buttons),
        "distinct_swaps": len(used_swaps),
        "floor_transfers": floor_transfers,
        "floor_limit_contacts": len(floor_limits),
        "coupled_multi_load_transfers": coupled_multi_load_transfers,
        "animation_frames": animation_frames,
        "minimum_steps_left": minimum_budget,
        "final_steps_left": env.steps_left,
        "levels_completed_before": before_score,
        "levels_completed_after": env.levels_completed,
        "verification_level_index": difficulty - 1,
        "won": True,
    }, None


def _mechanic_errors(difficulty, metrics, trace):
    errors = []
    if trace["changed_actions"] != trace["action_count"]:
        errors.append("winning route contains a no-op action")
    if metrics["buttons"] and trace["distinct_buttons"] == 0:
        errors.append("winning route does not use a balance button")
    required_swaps = 3 if difficulty == 7 else 1 if metrics["swaps"] else 0
    if trace["distinct_swaps"] < required_swaps:
        errors.append(f"winning route uses fewer than {required_swaps} required swap bars")
    if metrics["swaps"] and trace["animation_frames"] == 0:
        errors.append("winning route does not exercise native swap animation")
    if metrics["floors"] and trace["floor_limit_contacts"] < metrics["floors"]:
        errors.append("winning route does not reach every floor-constrained limit")
    if difficulty == 7 and trace["coupled_multi_load_transfers"] == 0:
        errors.append("tier 7 route does not exercise a balance pair carrying two loads")
    return errors


def _profile_errors(difficulty, metrics, action_count):
    profile = REFERENCE_PROFILES[difficulty]
    errors = []
    for field in ("grid_size", "step_budget"):
        expected = profile["grid" if field == "grid_size" else "budget"]
        if metrics[field] != expected:
            errors.append(f"{field}={metrics[field]} expected {expected}")
    if tuple(metrics["gravity"]) != profile["gravity"]:
        errors.append("gravity differs from the reference tier")
    for field in ("loads", "targets", "buttons", "swaps", "floors",
                  "walls", "supports"):
        if metrics[field] != profile[field]:
            errors.append(f"{field}={metrics[field]} expected {profile[field]}")
    low, high = profile["density"]
    if not low <= metrics["visual_density"] <= high:
        errors.append(
            f"visual_density={metrics['visual_density']} outside [{low}, {high}]"
        )
    low, high = profile["actions"]
    if not low <= action_count <= high:
        errors.append(f"action_count={action_count} outside [{low}, {high}]")
    return errors


def validate_full_standard(spec, curriculum_entry):
    """Recompute full admission facts; never trust stored proof booleans."""
    errors = []
    if not isinstance(spec, dict):
        return ("spec must be an object",)
    try:
        difficulty = _integer(spec.get("difficulty"), "difficulty")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        expected = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        if not isinstance(curriculum_entry, dict):
            raise ValueError("curriculum_entry must be an object")
        for key in ("difficulty", "context_index", "search_work"):
            value = _integer(curriculum_entry.get(key), f"curriculum {key}")
            if value != expected[key]:
                errors.append(f"curriculum {key} does not match the contract")
        _split(spec.get("split"))
        if spec.get("format") != FORMAT:
            errors.append("format mismatch")
        for key, expected_value in (
                ("generator_version", GENERATOR_VERSION),
                ("mechanics_version", MECHANICS_VERSION),
                ("quality_profile_version", QUALITY_PROFILE_VERSION),
                ("identity_version", IDENTITY_VERSION),
                ("source", "generated_only"),
                ("source_id", SOURCE_ID),
                ("context_index", difficulty - 1),
                ("training_context_index", difficulty - 1),
                ("verification_level_index", difficulty - 1)):
            if not _strict_equal(spec.get(key), expected_value):
                errors.append(f"{key} mismatch")
        for key, expected_value in (
                ("engine_verified", True),
                ("search_exact", True),
                ("search_truncated", False)):
            if spec.get(key) is not expected_value:
                errors.append(f"{key} mismatch")
        seed = _integer(spec.get("seed"), "seed")
        attempt = _integer(spec.get("attempt"), "attempt")
        expected_effective = _effective_seed(seed, difficulty, attempt)
        if not _strict_equal(spec.get("effective_seed"), expected_effective):
            errors.append("effective_seed does not recompute")
        level = build_level(spec)
        metrics = _metrics(level)
        if not _strict_equal(spec.get("metrics"), metrics):
            errors.append("stored structural metrics do not recompute")
        solution = spec.get("solution")
        if not isinstance(solution, list) or not solution:
            errors.append("solution must be a nonempty action list")
            actions = ()
        else:
            actions = tuple(
                tuple(_integer(value, f"solution[{index}]") for value in action)
                for index, action in enumerate(solution)
            )
            if any(len(action) != 3 or action[0] != names.ACTION_CLICK
                   for action in actions):
                errors.append("solution contains a non-click action")
        if not _strict_equal(spec.get("solution_length"), len(actions)):
            errors.append("solution_length does not match solution")
        errors.extend(_profile_errors(difficulty, metrics, len(actions)))

        geometry = _d4_hash(level)
        gameplay = _gameplay_hash(spec, level)
        identities = spec.get("identities")
        if not isinstance(identities, dict):
            errors.append("identities must be an object")
        else:
            if not _strict_equal(identities.get("geometry_d4"), geometry):
                errors.append("geometry D4 identity does not recompute")
            if not _strict_equal(identities.get("gameplay"), gameplay):
                errors.append("gameplay identity does not recompute")
            if not _strict_equal(identities.get("partition"), spec.get("split")):
                errors.append("nested geometry partition does not recompute")
            if not _strict_equal(identities.get("version"), IDENTITY_VERSION):
                errors.append("nested identity version mismatch")
        for key, value in (
                ("geometry_sha256", geometry),
                ("geometry_d4_sha256", geometry),
                ("geometry_split", spec.get("split")),
                ("geometry_version", IDENTITY_VERSION),
                ("gameplay_sha256", gameplay),
                ("gameplay_identity_version", IDENTITY_VERSION)):
            if not _strict_equal(spec.get(key), value):
                errors.append(f"{key} does not recompute")
        if geometry in _official_hashes():
            errors.append("generated geometry duplicates an official level")
        if _canonical_split(geometry) != spec.get("split"):
            errors.append("geometry partition does not match split")

        expected_mechanics = {
            "balance_transfers": True,
            "negative_gravity": min(metrics["gravity"]) < 0,
            "swap_bars": metrics["swaps"],
            "floor_limits": metrics["floors"],
            "coupled_multi_load_supports": difficulty == 7,
        }
        if not _strict_equal(spec.get("mechanics"), expected_mechanics):
            errors.append("mechanics inventory does not recompute")

        generation = spec.get("generation")
        if not isinstance(generation, dict):
            errors.append("generation evidence must be an object")
        else:
            attempts_used = generation.get("attempts_used")
            bounded_attempts = generation.get("bounded_attempts")
            rejections = generation.get("rejections")
            if (type(attempts_used) is not int or attempts_used < 1
                    or attempts_used != attempt + 1):
                errors.append("generation attempts do not match accepted attempt")
            if (type(bounded_attempts) is not int
                    or bounded_attempts < attempts_used):
                errors.append("bounded generation attempts are invalid")
            if (not isinstance(rejections, dict)
                    or any(not isinstance(key, str) or not key
                           or type(value) is not int or value < 0
                           for key, value in rejections.items())):
                errors.append("generation rejection counters are invalid")
            elif sum(rejections.values()) != attempt:
                errors.append("generation rejection counters do not cover prior attempts")
            if not _strict_equal(spec.get("generation_exclusions"), rejections):
                errors.append("generation exclusion counters do not mirror rejections")

        if actions:
            trace, replay_error = _context_replay(spec, actions)
            if replay_error:
                errors.append(replay_error)
            else:
                errors.extend(_mechanic_errors(difficulty, metrics, trace))
                proof = spec.get("proof")
                if not isinstance(proof, dict):
                    errors.append("proof must be an object")
                else:
                    for key, expected_value in (
                            ("kind", "exact-compact-search-plus-native-context-replay"),
                            ("search_backend", "vc33-compact-exact-v1"),
                            ("difficulty", difficulty),
                            ("split", spec.get("split")),
                            ("context_index", difficulty - 1),
                            ("geometry_d4_sha256", geometry),
                            ("gameplay_sha256", gameplay),
                            ("generator_version", GENERATOR_VERSION),
                            ("mechanics_version", MECHANICS_VERSION),
                            ("quality_profile_version", QUALITY_PROFILE_VERSION),
                            ("context_engine_verified", True),
                            ("engine_win", True)):
                        if not _strict_equal(proof.get(key), expected_value):
                            errors.append(f"proof {key} mismatch")
                    if not _strict_equal(proof.get("mechanic_use"), trace):
                        errors.append("mechanic-use evidence does not recompute")
                    expected_budget = {
                        "initial": metrics["step_budget"],
                        "minimum": trace["minimum_steps_left"],
                        "final": trace["final_steps_left"],
                    }
                    if not _strict_equal(proof.get("native_budget"), expected_budget):
                        errors.append("native-budget evidence does not recompute")
                    expected_context = {
                        "verification_level_index": difficulty - 1,
                        "levels_completed_before": difficulty - 1,
                        "levels_completed_after": difficulty,
                        "won": True,
                    }
                    if not _strict_equal(proof.get("context_replay"), expected_context):
                        errors.append("context-replay evidence does not recompute")
                    if not _strict_equal(
                            proof.get("search_limit"), expected["search_work"]):
                        errors.append("search limit differs from curriculum")
                    if proof.get("search_truncated") is not False:
                        errors.append("proof must record a non-truncated search")
                    if not _strict_equal(proof.get("optimal_actions"), len(actions)):
                        errors.append("stored optimal action count differs from route")
                    expanded = proof.get("search_expanded")
                    generated = proof.get("search_generated")
                    if (type(expanded) is not int or not 1 <= expanded <= expected["search_work"]
                            or not _strict_equal(proof.get("search_work"), expanded)):
                        errors.append("proof search-work evidence is invalid")
                    if type(generated) is not int or generated < 0:
                        errors.append("proof generated-state count is invalid")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return tuple(dict.fromkeys(errors))


def generate(seed, difficulty=1, *, split=None, attempts=DEFAULT_ATTEMPTS,
             node_limit=None):
    """Generate one full-standard level for an explicit canonical split."""
    seed = _integer(seed, "seed")
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts")
    split = _split(split)
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    search_work = REFERENCE_PROFILES[difficulty]["search_work"]
    if node_limit is not None and _integer(node_limit, "node_limit") != search_work:
        raise ValueError("full mode node_limit must equal the curriculum search_work")

    rejections = Counter()
    for attempt in range(attempts):
        try:
            spec = _draft(seed, difficulty, attempt)
            spec["split"] = split
            level = build_level(spec)
            metrics = _metrics(level)
        except (IndexError, KeyError, TypeError, ValueError):
            rejections["invalid_geometry"] += 1
            continue
        geometry = _d4_hash(level)
        if geometry in _official_hashes():
            rejections["official_copy"] += 1
            continue
        if _canonical_split(geometry) != split:
            rejections["geometry_split_mismatch"] += 1
            continue
        structural = _profile_errors(difficulty, metrics, 0)
        structural = [error for error in structural if not error.startswith("action_count=")]
        if structural:
            rejections["structural_profile"] += 1
            continue

        result = search(Env([level]), limit=metrics["step_budget"],
                        node_limit=search_work)
        if result.unsupported:
            rejections["teacher_unsupported"] += 1
            continue
        if result.truncated:
            rejections["search_truncated"] += 1
            continue
        if result.actions is None:
            rejections["proven_unsolvable"] += 1
            continue
        actions = tuple(result.actions)
        if _profile_errors(difficulty, metrics, len(actions)):
            rejections["action_profile"] += 1
            continue
        trace, replay_error = _context_replay(spec, actions)
        if replay_error:
            rejections["native_context_replay"] += 1
            continue
        if _mechanic_errors(difficulty, metrics, trace):
            rejections["mechanic_use"] += 1
            continue

        spec["solution"] = [list(action) for action in actions]
        spec["solution_length"] = len(actions)
        spec["metrics"] = metrics
        spec["effective_seed"] = _effective_seed(seed, difficulty, attempt)
        spec["context_index"] = difficulty - 1
        spec["geometry_sha256"] = geometry
        spec["geometry_d4_sha256"] = geometry
        spec["geometry_split"] = split
        spec["geometry_version"] = IDENTITY_VERSION
        spec["gameplay_sha256"] = _gameplay_hash(spec, level)
        spec["gameplay_identity_version"] = IDENTITY_VERSION
        spec["identities"] = {
            "geometry_d4": geometry,
            "gameplay": spec["gameplay_sha256"],
            "partition": split,
            "version": IDENTITY_VERSION,
        }
        spec["generation"] = {
            "attempts_used": attempt + 1,
            "rejections": dict(sorted(rejections.items())),
            "bounded_attempts": attempts,
        }
        spec["generation_exclusions"] = dict(sorted(rejections.items()))
        spec["engine_verified"] = True
        spec["search_exact"] = True
        spec["search_truncated"] = False
        spec["mechanics"] = {
            "balance_transfers": True,
            "negative_gravity": min(spec["gravity"]) < 0,
            "swap_bars": metrics["swaps"],
            "floor_limits": metrics["floors"],
            "coupled_multi_load_supports": difficulty == 7,
        }
        spec["proof"] = {
            "kind": "exact-compact-search-plus-native-context-replay",
            "search_backend": "vc33-compact-exact-v1",
            "search_limit": search_work,
            "search_expanded": int(result.expanded),
            "search_generated": int(result.generated),
            "search_truncated": False,
            "search_work": int(result.expanded),
            "optimal_actions": len(actions),
            "difficulty": difficulty,
            "split": split,
            "context_index": difficulty - 1,
            "geometry_d4_sha256": geometry,
            "gameplay_sha256": spec["gameplay_sha256"],
            "generator_version": GENERATOR_VERSION,
            "mechanics_version": MECHANICS_VERSION,
            "quality_profile_version": QUALITY_PROFILE_VERSION,
            "context_engine_verified": True,
            "engine_win": True,
            "mechanic_use": trace,
            "native_budget": {
                "initial": metrics["step_budget"],
                "minimum": trace["minimum_steps_left"],
                "final": trace["final_steps_left"],
            },
            "context_replay": {
                "verification_level_index": difficulty - 1,
                "levels_completed_before": difficulty - 1,
                "levels_completed_after": difficulty,
                "won": True,
            },
        }
        errors = validate_full_standard(
            spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        )
        if errors:
            rejections["validator_rejection"] += 1
            continue
        generate.last_report = {
            "seed": seed,
            "difficulty": difficulty,
            "split": split,
            "accepted": True,
            "attempts_used": attempt + 1,
            "rejections": dict(sorted(rejections.items())),
        }
        return spec

    generate.last_report = {
        "seed": seed,
        "difficulty": difficulty,
        "split": split,
        "accepted": False,
        "attempts_used": attempts,
        "rejections": dict(sorted(rejections.items())),
    }
    return None


generate.last_report = None


def _effective_seed(seed, difficulty, attempt):
    payload = f"{MECHANICS_VERSION}:{seed}:{difficulty}:{attempt}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _child_seed(game_seed, ordinal, difficulty):
    payload = f"{SOURCE_ID}:{game_seed}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split, difficulties=None, attempts=DEFAULT_ATTEMPTS,
                  node_limit=None):
    """Generate an ordered full game, or an explicit reduced ergonomic sample."""
    seed = _integer(seed, "seed")
    split = _split(split)
    if difficulties is None:
        selected = DIFFICULTIES
    else:
        try:
            selected = tuple(
                _integer(value, "difficulty") for value in difficulties
            )
        except TypeError as exc:
            raise ValueError("difficulties must be an increasing sequence") from exc
    if not selected or any(value not in DIFFICULTIES for value in selected):
        raise ValueError(f"difficulties must be a nonempty subset of {DIFFICULTIES}")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("difficulties must be strictly increasing and unique")
    specs = []
    child_reports = []
    for ordinal, difficulty in enumerate(selected):
        child = _child_seed(seed, ordinal, difficulty)
        spec = generate(child, difficulty, split=split, attempts=attempts,
                        node_limit=node_limit)
        child_report = dict(generate.last_report or {})
        child_reports.append(child_report)
        if spec is None:
            generate_game.last_report = {
                "accepted": False,
                "game_seed": seed,
                "split": split,
                "difficulties": list(selected),
                "failure_stage": "child_generation",
                "failed_ordinal": ordinal,
                "failed_difficulty": difficulty,
                "child_seed": child,
                "child_report": child_report,
            }
            return None
        spec["game_seed"] = seed
        spec["game_ordinal"] = ordinal
        spec["child_seed"] = child
        specs.append(spec)
    if selected == DIFFICULTIES:
        try:
            build_game(specs)
        except ValueError as exc:
            generate_game.last_report = {
                "accepted": False,
                "game_seed": seed,
                "split": split,
                "difficulties": list(selected),
                "failure_stage": "full_game_replay",
                "error": str(exc),
                "children": child_reports,
            }
            return None
    generate_game.last_report = {
        "accepted": True,
        "game_seed": seed,
        "split": split,
        "difficulties": list(selected),
        "children": child_reports,
    }
    return specs


generate_game.last_report = None


def build_game(specs):
    """Build an exact seven-level native sequence; reject shifted curricula."""
    if specs is None:
        raise ValueError("full game specs must be a sequence")
    try:
        specs = list(specs)
    except TypeError as exc:
        raise ValueError("full game specs must be a sequence") from exc
    if len(specs) != len(DIFFICULTIES):
        raise ValueError(f"full VC33 games need exactly {len(DIFFICULTIES)} levels")
    if any(not isinstance(spec, dict) for spec in specs):
        raise ValueError("every full game level spec must be an object")
    try:
        splits = {_split(spec.get("split")) for spec in specs}
    except (TypeError, ValueError) as exc:
        raise ValueError("every full game level needs a valid split") from exc
    if len(splits) != 1:
        raise ValueError("full game levels must use one split")
    seen_geometry = set()
    seen_gameplay = set()
    levels = []
    for index, (difficulty, spec) in enumerate(zip(DIFFICULTIES, specs)):
        if not isinstance(spec, dict) or spec.get("difficulty") != difficulty:
            raise ValueError("full game difficulties must be ordered 1..N")
        if spec.get("training_context_index") != index or spec.get("verification_level_index") != index:
            raise ValueError("full game context indices must be ordered 0..N-1")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError("invalid full-standard level: " + "; ".join(errors))
        identities = spec["identities"]
        geometry = identities["geometry_d4"]
        gameplay = identities["gameplay"]
        if geometry in seen_geometry or gameplay in seen_gameplay:
            raise ValueError("full game contains a duplicate level identity")
        seen_geometry.add(geometry)
        seen_gameplay.add(gameplay)
        levels.append(build_level(spec))
    episode = Env(levels)
    for index, spec in enumerate(specs):
        if episode.level_index != index or episode.levels_completed != index:
            raise ValueError("native episode entered the wrong curriculum context")
        before = episode.levels_completed
        observation = None
        for ordinal, action in enumerate(spec["solution"]):
            observation = episode.perform(*action)
            if episode.levels_completed > before or observation.state == GameState.WIN:
                if ordinal != len(spec["solution"]) - 1:
                    raise ValueError(
                        "native episode won before its final stored action"
                    )
                break
        if episode.levels_completed != index + 1:
            raise ValueError("native episode witness did not advance exactly one tier")
    if observation is None or observation.state != GameState.WIN:
        raise ValueError("native seven-tier episode did not reach WIN")
    return levels
