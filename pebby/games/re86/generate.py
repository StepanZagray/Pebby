"""Eight-tier, full-mechanics procedural RE86 generator.

Official levels contribute aggregate profiles only.  Drafts use fresh sprite
assignments, positions, paths, targets, and colour choices.  Every accepted
witness is replayed through the vendored engine at its native context; it is a
constructive positive certificate and is not labelled shortest.
"""

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import random

import numpy as np
from arcengine import Level, Sprite

from . import names
from .env import Env, official_levels, upstream
from .plan import normalize_unselected, solution_mechanics
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    MECHANICS_INVENTORY_VERSION,
    PROFILES,
    QUALITY_PROFILE_VERSION,
    SOURCE_SHA256,
    profile_errors,
    structural_metrics,
)


FORMAT = "pebby.re86.full-level.v2"
SOURCE_ID = "re86-8af5384d"
GENERATOR_VERSION = 4
GEOMETRY_VERSION = "re86-ordered-fixed-board-d4-v2"
GAMEPLAY_IDENTITY_VERSION = "re86-public-transition-d4-v2"
SOLUTION_SEMANTIC_VERSION = "re86-solution-actions-events-d4-v1"
DEFAULT_ATTEMPTS = 96
SPLITS = ("train", "validation", "test")

RIGID = (
    "0041edqtyiekev", "0030rxzjuipynt", "0043ingegpwaik",
    "0045rflckndtdm", "0033iipgjezqam", "0042qffokapnyc",
)
FLEXIBLE = (
    "0035hkenprijlo", "0038kytlejjbbe", "0039ihlvdjkxyx",
    "0037ycofafbdtv", "0044bkfjtphmea", "0051gzxbzqbgog",
)
FIXED = ("0048nilhpyjmsb", "0050jzuinqsedg")
DYES = (
    "0016wenbbqfzrp", "0017yjjvpgmubx", "0018nrmmstjbnh",
    "0019cjajxjizmd", "0024yhkhjpnoey", "0025nxtvkfnydw",
    "0026eaadiarevl", "0029fpsoedgzyh", "0006vrywevjjuj",
    "0008hdkmsraeai", "0009ltuoosgpsn", "0010khksgqmupu",
    "0011xlmwdzuxex", "0012zusobywdet", "0013gtwtquvamd",
    "0014ylhoroarjl",
)
ADMISSIBLE_DYES = frozenset((*DYES, "0023ggyzglpjdy"))
OBSTACLES = ("0004wovqeugbap", "0002evmlgoerxd", "0005gulbaugmeh")


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
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
        "official_tier_characterization": (
            ".scratch/multigame-resume/full-standard/re86.md + "
            "pebby/games/re86/reference_profiles.py"
        ),
        "solution_mechanics": "spec.solution_mechanics recomputed by native replay",
        "native_budget": "spec.structural_metrics.native_budget + route replay",
        "context_engine_replay": "spec.proof.context_engine_verified",
        "novelty_split": (
            "public spec.gameplay_sha256 partition; ordered geometry copy check; "
            "private spec.solution_semantic_sha256 diversity"
        ),
        "bounded_rejections": "spec.generation_exclusions",
    },
    "caveats": [
        "One official level exists per tier; tolerances are engineering bounds, not confidence intervals.",
        "Constructive witnesses are native positive certificates and do not claim global optimality.",
        "The procedural grammar is finite; route identity is separate diagnostic evidence, not a generalization proof.",
        "D4 public-gameplay partitioning and color-blind ordered geometry answer different identity questions.",
        "Tier 7 has a longer optional native win that avoids rigid deformation; required generated witnesses exercise it.",
        "Teacher node_limit counts joined work but can report three per-movable expansions at the tiny limit of one.",
    ],
}


def _integer(value, label):
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _point(value, label):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(item) is not int for item in value)):
        raise ValueError(f"{label} must contain two integers")
    return value[0], value[1]


def _main_color(sprite):
    return int(getattr(upstream(), "euqngakkse")(sprite))


def _component(component, expected_tag, index):
    if not isinstance(component, Mapping):
        raise ValueError(f"component {index} must be an object")
    prototype = component.get("prototype")
    module = upstream()
    if prototype not in module.sprites:
        raise ValueError(f"component {index} has an unknown prototype")
    sprite = module.sprites[prototype].clone()
    if expected_tag not in sprite.tags:
        raise ValueError(f"component {index} has the wrong native tag")
    x, y = _point(component.get("position"), f"component {index}.position")
    rotation = _integer(component.get("rotation", 0), f"component {index}.rotation")
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"component {index}.rotation must be 0, 90, 180, or 270")
    sprite.set_position(x, y).set_rotation(rotation)
    recolor = component.get("recolor")
    if recolor is not None:
        recolor = _integer(recolor, f"component {index}.recolor")
        if not 0 <= recolor <= 15:
            raise ValueError(f"component {index}.recolor is outside palette 0..15")
        old = _main_color(sprite)
        sprite.pixels[sprite.pixels == old] = recolor
    return sprite


def _target_pixels(spec, *, sentinel=False):
    pixels = np.full((64, 64), names.TRANSPARENT, dtype=np.int8)
    if sentinel:
        pixels[0, 0] = 15
        return pixels
    target = spec.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("target must be an object")
    for index, value in enumerate(target.get("guides", ())):
        row, col = _point(value, f"target.guides[{index}]")
        if not 0 <= row < 64 or not 0 <= col < 64:
            raise ValueError("target guide is outside the frame")
        pixels[row, col] = names.TARGET_GUIDE
    for index, value in enumerate(target.get("colored", ())):
        if (not isinstance(value, (list, tuple)) or len(value) != 3
                or any(type(item) is not int for item in value)):
            raise ValueError(f"target.colored[{index}] must be row, col, color")
        row, col, color = value
        if not 0 <= row < 64 or not 0 <= col < 64:
            raise ValueError("target color is outside the frame")
        if (not 0 <= color <= 15
                or color in (names.TARGET_GUIDE, names.SELECTED_CENTER)):
            raise ValueError("target color is not a movable palette value")
        pixels[row, col] = color
    return pixels


def _level(spec, *, sentinel=False):
    if spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    profile = PROFILES[difficulty]
    if _integer(spec.get("step_budget"), "step_budget") != profile["native_budget"]:
        raise ValueError("step budget differs from the official tier")
    movables = [
        _component(value, names.TAG_MOVABLE, index)
        for index, value in enumerate(spec.get("movables", ()))
    ]
    dyes = [
        _component(value, names.TAG_DYE, index)
        for index, value in enumerate(spec.get("dyes", ()))
    ]
    obstacles = [
        _component(value, names.TAG_OBSTACLE, index)
        for index, value in enumerate(spec.get("obstacles", ()))
    ]
    if len(movables) != profile["movables"]:
        raise ValueError("movable count differs from the official tier")
    if not movables:
        raise ValueError("a level needs a movable")
    for sprite in movables:
        normalize_unselected(sprite)
    first = movables[0]
    first.pixels[first.height // 2, first.width // 2] = names.SELECTED_CENTER
    target = Sprite(
        pixels=_target_pixels(spec, sentinel=sentinel),
        name="generated-re86-target" if not sentinel else "re86-draft-sentinel",
        visible=not sentinel,
        collidable=True,
        tags=[names.TAG_TARGET, names.TAG_BACKGROUND],
        layer=-1,
    )
    return Level(
        sprites=[*obstacles, *dyes, *movables, target],
        grid_size=(64, 64),
        data={names.KEY_STEP_COUNTER: profile["native_budget"]},
        name=f"generated-re86-d{difficulty}-s{spec.get('seed', 0)}",
    )


def build_level(spec):
    """Reconstruct one JSON-safe full-mechanics spec as a native level."""
    return _level(spec, sentinel=False)


def env_for(spec, context_index=None):
    context = (spec["difficulty"] - 1 if context_index is None
               else _integer(context_index, "context_index"))
    if not 0 <= context < len(DIFFICULTIES):
        raise ValueError("context index must be in 0..7")
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context + 1)])
    env.set_level(context)
    return env


def _actions_between(start, goal, horizontal_first=True):
    sx, sy = start
    gx, gy = goal
    if (gx - sx) % 3 or (gy - sy) % 3:
        raise ValueError("constructive points must share the native 3-pixel lattice")
    horizontal = ((names.ACTION_RIGHT if gx > sx else names.ACTION_LEFT),
                  abs(gx - sx) // 3)
    vertical = ((names.ACTION_DOWN if gy > sy else names.ACTION_UP),
                abs(gy - sy) // 3)
    axes = (horizontal, vertical) if horizontal_first else (vertical, horizontal)
    return [action for action, count in axes for _ in range(count)]


def _switch(route, selected, target, count):
    for _ in range((target - selected) % count):
        route.append(names.ACTION_NEXT)
    return target


def _position(prototype):
    sprite = upstream().sprites[prototype]
    return sprite.width, sprite.height


def _special_positions(count, forbidden):
    positions = []
    candidates = (
        [(x, 1) for x in range(1, 59, 8)]
        + [(x, 57) for x in range(1, 59, 8)]
        + [(57, y) for y in range(9, 50, 8)]
    )
    for x, y in candidates:
        if all(abs(x - fx) > 7 or abs(y - fy) > 7 for fx, fy in forbidden):
            positions.append((x, y))
        if len(positions) == count:
            break
    if len(positions) != count:
        return None
    return positions


def _draft_components(rng, difficulty, route_variant):
    profile = PROFILES[difficulty]
    kinds_by_tier = {
        1: ("rigid", "rigid"),
        2: ("rigid", "rigid", "flexible"),
        3: ("rigid", "rigid", "flexible"),
        4: ("rigid", "fixed"),
        5: ("rigid", "fixed", "flexible"),
        6: ("rigid", "flexible"),
        7: ("rigid", "rigid", "flexible"),
        8: ("flexible", "flexible"),
    }
    pools = {"rigid": list(RIGID), "fixed": list(FIXED),
             "flexible": list(FLEXIBLE)}
    for pool in pools.values():
        rng.shuffle(pool)
    kinds = kinds_by_tier[difficulty]
    prototypes = [pools[kind].pop() for kind in kinds]
    if difficulty == 6:
        prototypes[1] = rng.choice(FLEXIBLE[:3])
    elif difficulty == 7:
        prototypes[1] = "0041edqtyiekev"
        prototypes[2] = rng.choice(FLEXIBLE[:3])
    elif difficulty == 8:
        prototypes = rng.sample(list(FLEXIBLE[:3]), 2)

    components = []
    routes = []
    used_dyes = []
    obstacles = []
    if difficulty <= 5:
        starts = [(3, 3), (3, 24), (3, 42)][:len(kinds)]
        x_goal = rng.choice((24, 27, 30)) if difficulty <= 3 else rng.choice((30, 33, 36))
        goal_ys = {
            # Eight seed-derived route families have genuinely different
            # target distances (15..22 native actions including selection),
            # rather than relying on random palette/prototype jitter.
            1: [3 + 3 * (route_variant // 4), 24],
            2: [rng.choice((15, 18, 21)), rng.choice((30, 33, 36)), rng.choice((3, 6, 9))],
            3: [rng.choice((30, 33, 36)), rng.choice((3, 6, 9)), rng.choice((18, 21, 24))],
            4: [rng.choice((21, 24, 27)), rng.choice((3, 6, 9))],
            5: [rng.choice((27, 30, 33)), rng.choice((3, 6, 9)), rng.choice((18, 21, 24))],
        }[difficulty]
        if difficulty == 1:
            x_goal = 24 + 3 * (route_variant % 4)
        for index, (kind, prototype, start) in enumerate(
                zip(kinds, prototypes, starts)):
            component = {"prototype": prototype, "position": list(start), "rotation": 0}
            if difficulty == 3:
                component["recolor"] = 8
            components.append(component)
            goal = (x_goal, goal_ys[index])
            routes.append(_actions_between(start, goal, rng.choice((True, False))))
            if index < min(len(kinds), profile["dyes"]):
                width, height = _position(prototype)
                used_dyes.append((start[0] + width, start[1] + height // 2 - 2))
    else:
        if difficulty == 6:
            starts = [(3, 3), (3, 33)]
            components = [
                {"prototype": prototypes[0], "position": list(starts[0]), "rotation": 0},
                {"prototype": prototypes[1], "position": list(starts[1]), "rotation": 0},
            ]
            routes = [
                _actions_between(
                    starts[0], (rng.choice((27, 30, 33)), rng.choice((12, 15, 18, 21))),
                    rng.choice((True, False)),
                ),
                [names.ACTION_RIGHT] * 3
                + [names.ACTION_UP] * rng.choice((4, 5, 6, 7))
                + [names.ACTION_RIGHT] * rng.choice((4, 5, 6, 7, 8, 9)),
            ]
            obstacles = [("0004wovqeugbap", (18, 37))]
        elif difficulty == 7:
            starts = [(3, 3), (3, 42), (3, 24)]
            components = [
                {"prototype": prototypes[i], "position": list(starts[i]), "rotation": 0}
                for i in range(3)
            ]
            routes = [
                _actions_between(starts[0], (rng.choice((27, 30, 33)), rng.choice((12, 15, 18))), rng.choice((True, False))),
                [names.ACTION_UP] * 4
                + [names.ACTION_RIGHT] * rng.choice((8, 9, 10))
                + [names.ACTION_DOWN] * rng.choice((1, 2, 3)),
                [names.ACTION_RIGHT] * 4
                + [names.ACTION_UP] * rng.choice((4, 5, 6))
                + [names.ACTION_RIGHT] * rng.choice((4, 5, 6, 7, 8)),
            ]
            obstacles = [("0004wovqeugbap", (24, 28))]
            used_dyes = [(16, 7), (16, 30)]
        else:
            starts = [(3, 3), (3, 39)]
            components = [
                {"prototype": prototypes[i], "position": list(starts[i]), "rotation": 0}
                for i in range(2)
            ]
            routes = [
                [names.ACTION_RIGHT] * 4
                + [names.ACTION_DOWN] * 4
                + [names.ACTION_RIGHT] * 11
                + [names.ACTION_DOWN] * route_variant,
                [names.ACTION_RIGHT] * 4
                + [names.ACTION_UP] * 3
                + [names.ACTION_RIGHT]
                + [names.ACTION_UP]
                + [names.ACTION_RIGHT] * 10
                + [names.ACTION_UP],
            ]
            obstacles = [
                ("0004wovqeugbap", (24, 7)),
                ("0004wovqeugbap", (24, 43)),
            ]
            used_dyes = [(16, 7), (16, 43)]

    route = []
    selected = 0
    order = list(range(len(components)))
    if any(kinds[index] == "flexible" for index in order):
        flexible_index = next(index for index in reversed(order)
                              if kinds[index] == "flexible")
        order.remove(flexible_index)
        order.append(flexible_index)
    elif "fixed" in kinds:
        fixed_index = kinds.index("fixed")
        order.remove(fixed_index)
        order.insert(0, fixed_index)
    for index in order:
        selected = _switch(route, selected, index, len(components))
        route.extend(routes[index])

    dye_components = []
    dye_names = list(DYES)
    rng.shuffle(dye_names)
    preferred = {
        7: ("0023ggyzglpjdy", "0019cjajxjizmd"),  # palette 9, 11
        8: ("0018nrmmstjbnh", "0019cjajxjizmd"),  # palette 6, 11
    }.get(difficulty, ())
    forbidden = [tuple(value["position"]) for value in components]
    for used_index, position in enumerate(used_dyes):
        prototype = (preferred[used_index] if used_index < len(preferred)
                     else dye_names.pop())
        if prototype in dye_names:
            dye_names.remove(prototype)
        dye_components.append({
            "prototype": prototype, "position": list(position),
            "rotation": rng.choice((0, 90, 180, 270)),
        })
        forbidden.append(position)
    extras = profile["dyes"] - len(dye_components)
    extra_positions = _special_positions(extras, forbidden)
    if extra_positions is None:
        return None
    for position in extra_positions:
        dye_components.append({
            "prototype": dye_names.pop(), "position": list(position),
            "rotation": rng.choice((0, 90, 180, 270)),
        })
    obstacle_components = [
        {"prototype": prototype, "position": list(position), "rotation": 0}
        for prototype, position in obstacles
    ]
    return components, dye_components, obstacle_components, route


def _advance(level, route):
    env = Env([level])
    observation = None
    for action_id in route:
        try:
            observation = env.perform(action_id)
        except (IndexError, RuntimeError, ValueError):
            return None
        if observation.finished:
            return None
    return env


def _composite_with_owners(movables):
    values = np.full((64, 64), names.TRANSPARENT, dtype=np.int8)
    owners = np.full((64, 64), -1, dtype=np.int8)
    for index, sprite in enumerate(movables):
        y0, y1 = max(0, sprite.y), min(64, sprite.y + sprite.height)
        x0, x1 = max(0, sprite.x), min(64, sprite.x + sprite.width)
        if y0 >= y1 or x0 >= x1:
            continue
        source = sprite.pixels[y0 - sprite.y:y1 - sprite.y,
                               x0 - sprite.x:x1 - sprite.x]
        mask = source != names.TRANSPARENT
        values[y0:y1, x0:x1][mask] = source[mask]
        owners[y0:y1, x0:x1][mask] = index
    return values, owners


def _derive_target(env, profile, rng):
    for sprite in env.movables():
        for row, col in np.argwhere(sprite.pixels != names.TRANSPARENT):
            if not (0 <= sprite.y + int(row) <= 62
                    and 0 <= sprite.x + int(col) < 64):
                return None
    values, owners = _composite_with_owners(env.movables())
    count = profile["target_colored"]
    candidates = {index: [] for index in range(len(env.movables()))}
    for row, col in np.argwhere((values > 0) & (values != names.TARGET_GUIDE)):
        row, col = int(row), int(col)
        if not (1 <= row <= 62 and 1 <= col <= 62):
            continue
        owner = int(owners[row, col])
        if owner >= 0:
            candidates[owner].append((row, col, int(values[row, col])))
    for values_for_owner in candidates.values():
        rng.shuffle(values_for_owner)
    for owner, values_for_owner in candidates.items():
        sprite = env.movables()[owner]
        center = (sprite.y + sprite.height // 2,
                  sprite.x + sprite.width // 2)
        values_for_owner.sort(
            key=lambda value: abs(value[0] - center[0]) + abs(value[1] - center[1]),
            reverse=True,
        )
    chosen = []
    active = env.selected_index()
    # Fixed/intrinsic centres are mechanically informative: when selected they
    # are palette 0, so a target marker here forces ACTION5 normalization.
    for index, sprite in enumerate(env.movables()):
        if index == active:
            continue
        row = sprite.y + sprite.height // 2
        col = sprite.x + sprite.width // 2
        if (1 <= row <= 62 and 1 <= col <= 62
                and int(owners[row, col]) == index and int(values[row, col]) > 0):
            chosen.append((row, col, int(values[row, col])))
    quotas = {
        1: (4, 4), 2: (4, 3, 3), 3: (3, 2, 3), 4: (3, 3),
        5: (4, 3, 3), 6: (4, 4), 7: (2, 4, 3), 8: (4, 4),
    }[profile["difficulty"]]
    for owner, quota in enumerate(quotas):
        already = sum(int(owners[row, col]) == owner for row, col, _ in chosen)
        for candidate in candidates[owner]:
            if already >= quota:
                break
            row, col, _ = candidate
            if all(max(abs(row - old_row), abs(col - old_col)) >= 3
                   for old_row, old_col, _ in chosen):
                chosen.append(candidate)
                already += 1
        if already != quota:
            return None
    if len(chosen) != count:
        return None
    guides = set()
    colored_points = {(row, col) for row, col, _ in chosen}
    for row, col, _ in chosen:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                point = row + dr, col + dc
                if point not in colored_points:
                    guides.add(point)
    if len(guides) != profile["target_guides"]:
        return None
    return {
        "colored": [list(value) for value in sorted(chosen)],
        "guides": [list(value) for value in sorted(guides)],
    }


def _draft(seed, difficulty, attempt, split):
    rng = random.Random(
        f"{DIFFICULTY_VERSION}:{GENERATOR_VERSION}:{seed}:{difficulty}:{split}:{attempt}"
    )
    built = _draft_components(rng, difficulty, seed % 8)
    if built is None:
        return None, "invalid_geometry"
    movables, dyes, obstacles, planned = built
    spec = {
        "format": FORMAT,
        "game": "re86",
        "source_id": SOURCE_ID,
        "vendored_source_sha256": SOURCE_SHA256,
        "generator_version": GENERATOR_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "gameplay_identity_version": GAMEPLAY_IDENTITY_VERSION,
        "solution_semantic_version": SOLUTION_SEMANTIC_VERSION,
        "source": "generated_only",
        "seed": seed,
        "difficulty": difficulty,
        "step_budget": PROFILES[difficulty]["native_budget"],
        "movables": movables,
        "dyes": dyes,
        "obstacles": obstacles,
        "target": {"colored": [[0, 0, 15]], "guides": []},
    }
    draft_env = _advance(_level(spec, sentinel=True), planned)
    if draft_env is None or not draft_env.stable():
        return None, "constructive_route_terminal"
    target = _derive_target(draft_env, PROFILES[difficulty], rng)
    if target is None:
        return None, "target_geometry"
    spec["target"] = target
    return (spec, tuple((action, None, None) for action in planned)), None


def _transforms(row, col):
    return (
        (row, col), (row, -col), (-row, col), (-row, -col),
        (col, row), (col, -row), (-col, row), (-col, -row),
    )


def _transform_rect(pixels, x, y, transform_index):
    """Transform a complete sprite rectangle on the fixed 64x64 board."""
    height, width = pixels.shape
    if transform_index == 0:
        transformed, new_x, new_y = pixels, x, y
    elif transform_index == 1:
        transformed, new_x, new_y = np.fliplr(pixels), 64 - x - width, y
    elif transform_index == 2:
        transformed, new_x, new_y = np.flipud(pixels), x, 64 - y - height
    elif transform_index == 3:
        transformed = np.flipud(np.fliplr(pixels))
        new_x, new_y = 64 - x - width, 64 - y - height
    elif transform_index == 4:
        transformed, new_x, new_y = pixels.T, y, x
    elif transform_index == 5:
        transformed = np.rot90(pixels, k=-1)
        new_x, new_y = 64 - y - height, x
    elif transform_index == 6:
        transformed = np.rot90(pixels, k=1)
        new_x, new_y = y, 64 - x - width
    else:
        transformed = np.flipud(np.fliplr(pixels.T))
        new_x, new_y = 64 - y - height, 64 - x - width
    return np.asarray(transformed), int(new_x), int(new_y)


def _identity_pixels(sprite, role, *, geometry):
    pixels = np.asarray(sprite.render(), dtype=np.int16).copy()
    if geometry:
        occupied = pixels != names.TRANSPARENT
        selected = pixels == names.SELECTED_CENTER
        guide = pixels == names.TARGET_GUIDE
        pixels[:] = names.TRANSPARENT
        pixels[occupied] = 1
        pixels[selected] = names.SELECTED_CENTER
        pixels[guide] = names.TARGET_GUIDE
    elif role == "obstacle":
        pixels = np.where(pixels == names.TRANSPARENT,
                          names.TRANSPARENT, 1).astype(np.int16)
    elif role == "target":
        pixels[pixels == names.TARGET_GUIDE] = names.TRANSPARENT
    return pixels


def _sprite_identity(sprite, role, transform_index, *, geometry):
    pixels = _identity_pixels(sprite, role, geometry=geometry)
    pixels, x, y = _transform_rect(
        pixels, int(sprite.x), int(sprite.y), transform_index
    )
    cells = [
        [int(row), int(col), int(pixels[row, col])]
        for row, col in np.argwhere(pixels != names.TRANSPARENT)
    ]
    record = {
        "position": [x, y],
        "size": [int(pixels.shape[0]), int(pixels.shape[1])],
        "cells": cells,
    }
    if role == "movable":
        record["kind"] = (
            "flexible" if names.TAG_FLEXIBLE in sprite.tags
            else "fixed" if names.TAG_FIXED_CENTER in sprite.tags
            else "rigid"
        )
    if role == "dye" and not geometry:
        record["dye_color"] = int(sprite.pixels[1, 1])
    return record


def _identity_payload(level, transform_index, *, geometry, difficulty=None):
    # List order is native behavior: ACTION5 cycles movables in this order and
    # overlapping obstacle/dye dispatch iterates each tag list in this order.
    payload = {
        "obstacles": [
            _sprite_identity(sprite, "obstacle", transform_index,
                             geometry=geometry)
            for sprite in level.get_sprites_by_tag(names.TAG_OBSTACLE)
        ],
        "dyes": [
            _sprite_identity(sprite, "dye", transform_index,
                             geometry=geometry)
            for sprite in level.get_sprites_by_tag(names.TAG_DYE)
        ],
        "movables": [
            _sprite_identity(sprite, "movable", transform_index,
                             geometry=geometry)
            for sprite in level.get_sprites_by_tag(names.TAG_MOVABLE)
        ],
        "targets": [
            _sprite_identity(sprite, "target", transform_index,
                             geometry=geometry)
            for sprite in level.get_sprites_by_tag(names.TAG_TARGET)
        ],
    }
    if not geometry:
        payload.update({
            "context_index": difficulty - 1,
            "native_budget": int(level.get_data(names.KEY_STEP_COUNTER)),
        })
    return payload


def _level_identity(level, *, geometry, difficulty=None):
    representations = [
        json.dumps(
            _identity_payload(level, transform_index, geometry=geometry,
                              difficulty=difficulty),
            sort_keys=True,
            separators=(",", ":"),
        )
        for transform_index in range(8)
    ]
    return hashlib.sha256(min(representations).encode()).hexdigest()


def geometry_identity(spec):
    return _level_identity(build_level(spec), geometry=True)


_DELTA_TO_ACTION = {value: key for key, value in names.ACTION_DELTAS.items()}


def _action_transforms(actions):
    results = [[] for _ in range(8)]
    for action in actions:
        if action == names.ACTION_NEXT:
            for result in results:
                result.append(action)
            continue
        dx, dy = names.ACTION_DELTAS[action]
        for index, (new_dy, new_dx) in enumerate(_transforms(dy, dx)):
            results[index].append(_DELTA_TO_ACTION[(new_dx, new_dy)])
    return tuple(tuple(value) for value in results)


def gameplay_identity(spec):
    """Hash only the canonical public transition system, never its witness."""
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be an official tier")
    return _level_identity(
        build_level(spec), geometry=False, difficulty=difficulty
    )


def solution_semantic_identity(spec):
    """D4-normalized private route identity for diversity reporting only."""
    actions = [int(value[0]) for value in spec.get("solution", ())]
    mechanics = spec.get("solution_mechanics", {})
    semantic = {
        "actions": min(_action_transforms(actions)) if actions else (),
        "events": {key: mechanics.get(key, 0) for key in (
            "selection_actions", "dye_events", "resize_events",
            "deformation_events", "fixed_center_selections",
            "flexible_selections",
        )},
        "tier": spec.get("difficulty"),
    }
    return hashlib.sha256(json.dumps(
        semantic, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def semantic_partition(identity):
    return SPLITS[int(identity, 16) % len(SPLITS)]


# Compatibility name for callers that only need the deterministic three-way
# partition. Generation and validation pass the public gameplay identity.
geometry_partition = semantic_partition


@lru_cache(maxsize=1)
def _official_geometry_hashes():
    return frozenset(
        _level_identity(level, geometry=True) for level in official_levels()
    )


def _replay_witness(spec, actions, context):
    env = env_for(spec, context)
    before = env.levels_completed
    observation = None
    for index, action in enumerate(actions):
        observation = env.perform(*action)
        if env.levels_completed > before or observation.won:
            return index == len(actions) - 1, observation, env
        if observation.finished:
            return False, observation, env
    return False, observation, env


def generate(seed, difficulty=1, attempts=DEFAULT_ATTEMPTS,
             node_limit=None, stats=None, *, split="train"):
    """Generate one split-qualified official-tier level."""
    seed = _integer(seed, "seed")
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts")
    if seed < 0 or difficulty not in DIFFICULTIES:
        raise ValueError("seed must be nonnegative and difficulty must be in 1..8")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    if node_limit is not None:
        if _integer(node_limit, "node_limit") < 1:
            raise ValueError("node_limit must be positive")
    search_limit = min(
        PROFILES[difficulty]["search_work"],
        node_limit if node_limit is not None else PROFILES[difficulty]["search_work"],
    )
    exclusions = {}
    for attempt in range(1, attempts + 1):
        drafted, reason = _draft(seed, difficulty, attempt, split)
        if drafted is None:
            _reject(exclusions, stats, reason)
            continue
        spec, actions = drafted
        if len(actions) > search_limit:
            _reject(exclusions, stats, "constructive_work_cap")
            continue
        spec.update({
            "generation_attempt": attempt,
            "search_limit": search_limit,
            "split": split,
            "context_index": difficulty - 1,
            "training_context_index": difficulty - 1,
            "verification_level_index": difficulty - 1,
        })
        try:
            geometry = geometry_identity(spec)
            gameplay = gameplay_identity(spec)
            partition = semantic_partition(gameplay)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            _reject(exclusions, stats, "identity:" + type(error).__name__)
            continue
        if partition != split:
            _reject(exclusions, stats, "public_gameplay_split")
            continue
        spec.update({
            "geometry_sha256": geometry,
            "geometry_d4_sha256": geometry,
            "gameplay_sha256": gameplay,
            "geometry_split": partition,
            "split_partition_bucket": int(gameplay, 16) % len(SPLITS),
            "official_copy": geometry in _official_geometry_hashes(),
        })
        if spec["official_copy"]:
            _reject(exclusions, stats, "official_copy")
            continue
        won, observation, _ = _replay_witness(spec, actions, difficulty - 1)
        if not won or observation is None:
            _reject(exclusions, stats, "native_replay")
            continue
        mechanics = solution_mechanics(env_for(spec, difficulty - 1), actions)
        spec.update({
            "solution": [list(action) for action in actions],
            "context_solution": [list(action) for action in actions],
            "solution_length": len(actions),
            "solution_mechanics": mechanics,
            "structural_metrics": structural_metrics(env_for(spec, difficulty - 1)),
            "native_budget": PROFILES[difficulty]["native_budget"],
            "budget_remaining": PROFILES[difficulty]["native_budget"] - len(actions),
            "search_exact": False,
            "search_truncated": False,
            "engine_verified": True,
            "optimality_claim": "none-constructive-native-witness",
        })
        spec["solution_semantic_sha256"] = solution_semantic_identity(spec)
        errors = profile_errors(spec) + _presentation_errors(spec)
        if errors:
            _reject(exclusions, stats, "profile:" + "|".join(errors))
            continue
        spec["generation_exclusions"] = dict(exclusions)
        exclusions_sha256 = hashlib.sha256(json.dumps(
            spec["generation_exclusions"], sort_keys=True,
            separators=(",", ":"),
        ).encode()).hexdigest()
        spec["proof"] = {
            "kind": "constructive-real-engine-replay",
            "format": FORMAT,
            "source_id": SOURCE_ID,
            "vendored_source_sha256": SOURCE_SHA256,
            "seed": seed,
            "generation_attempt": attempt,
            "difficulty": difficulty,
            "split": split,
            "context_index": difficulty - 1,
            "training_context_index": difficulty - 1,
            "verification_level_index": difficulty - 1,
            "native_budget": spec["native_budget"],
            "action_count": len(actions),
            "search_limit": search_limit,
            "search_work": len(actions),
            "search_truncated": False,
            "optimal": False,
            "context_engine_verified": True,
            "engine_win": True,
            "geometry_d4_sha256": geometry,
            "gameplay_sha256": spec["gameplay_sha256"],
            "solution_semantic_sha256": spec["solution_semantic_sha256"],
            "generation_exclusions_sha256": exclusions_sha256,
            "generator_version": GENERATOR_VERSION,
            "difficulty_version": DIFFICULTY_VERSION,
            "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
            "quality_profile_version": QUALITY_PROFILE_VERSION,
            "geometry_version": GEOMETRY_VERSION,
            "gameplay_identity_version": GAMEPLAY_IDENTITY_VERSION,
            "solution_semantic_version": SOLUTION_SEMANTIC_VERSION,
        }
        if stats is not None:
            stats["accepted"] = stats.get("accepted", 0) + 1
        return spec
    return None


def _reject(local, external, reason):
    local[reason] = local.get(reason, 0) + 1
    if external is not None:
        external[reason] = external.get(reason, 0) + 1


def _child_seed(seed, ordinal, difficulty):
    payload = f"{SOURCE_ID}:{seed}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big"
    ) & ((1 << 63) - 1)


def generate_game(seed, *, split="train", difficulties=None,
                  attempts=DEFAULT_ATTEMPTS, node_limit=None, stats=None):
    """Generate all eight increasing native contexts (or an explicit smoke subset)."""
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    tiers = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not tiers or tuple(sorted(set(tiers))) != tiers:
        raise ValueError("difficulties must be unique and increasing")
    if any(type(value) is not int or value not in DIFFICULTIES for value in tiers):
        raise ValueError("difficulties must be drawn from 1..8")
    specs = []
    for ordinal, difficulty in enumerate(tiers):
        spec = generate(
            _child_seed(seed, ordinal, difficulty), difficulty,
            attempts=attempts, node_limit=node_limit, stats=stats, split=split,
        )
        if spec is None:
            return None
        spec["game_seed"] = seed
        spec["game_ordinal"] = ordinal
        specs.append(spec)
    return specs


def _presentation_errors(spec):
    """Check native sprite extents and every visible target/tutorial cue."""
    errors = []
    level = build_level(spec)
    for sprite in level.get_sprites():
        raw = np.asarray(sprite.pixels)
        if (not np.issubdtype(raw.dtype, np.integer)
                or np.any(raw < names.TRANSPARENT) or np.any(raw > 15)):
            errors.append(f"sprite {sprite.name} contains a non-native palette value")
        if names.TAG_TARGET in sprite.tags:
            continue
        for row, col in np.argwhere(sprite.pixels != names.TRANSPARENT):
            global_row = sprite.y + int(row)
            global_col = sprite.x + int(col)
            if not (0 <= global_row <= 62 and 0 <= global_col < 64):
                errors.append(f"sprite {sprite.name} clips or enters the step-counter row")
                break
    target = np.asarray(level.get_sprites_by_tag(names.TAG_TARGET)[0].pixels)
    colored = np.argwhere(
        (target != names.TRANSPARENT) & (target != names.TARGET_GUIDE)
    )
    for row, col in colored:
        color = int(target[row, col])
        if color in (0, 1, 2, 3, 4, 5, 15):
            errors.append("target anchor lacks readable contrast from UI/guide colours")
        neighbors = [
            int(target[row + dr, col + dc])
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if dr or dc
        ]
        if neighbors.count(names.TARGET_GUIDE) != 8:
            errors.append("target anchor does not have its complete 3x3 guide cue")
    # Native rendering itself is part of admission: it must remain a complete
    # public palette frame without hidden/off-canvas construction errors.
    frame = np.asarray(Env([level]).render())
    if frame.shape != (64, 64) or not np.issubdtype(frame.dtype, np.integer):
        errors.append("native render is not a 64x64 integer palette frame")
    elif np.any(frame < 0) or np.any(frame > 15):
        errors.append("native render contains a palette value outside 0..15")
    route = spec.get("solution", ())
    if isinstance(route, list):
        probe = env_for(spec, spec.get("training_context_index"))
        for action_index, action in enumerate(route):
            if action_index == len(route) - 1:
                break
            observation = probe.perform(*action)
            if any(
                    np.any(np.asarray(value) < 0)
                    or np.any(np.asarray(value) > 15)
                    for value in observation.frames):
                errors.append("witness frame contains a palette value outside 0..15")
            for sprite in probe.movables():
                for row, col in np.argwhere(sprite.pixels != names.TRANSPARENT):
                    if not (0 <= sprite.y + int(row) <= 62
                            and 0 <= sprite.x + int(col) < 64):
                        errors.append("witness produces clipped intermediate sprite pixels")
                        break
    return list(dict.fromkeys(errors))


_TOP_LEVEL_FIELDS = frozenset({
    "format", "game", "source_id", "vendored_source_sha256",
    "generator_version", "difficulty_version", "quality_profile_version",
    "mechanics_inventory_version", "geometry_version",
    "gameplay_identity_version", "solution_semantic_version", "source",
    "seed", "difficulty", "step_budget", "movables", "dyes", "obstacles",
    "target", "generation_attempt", "search_limit", "split", "context_index",
    "training_context_index", "verification_level_index", "geometry_sha256",
    "geometry_d4_sha256", "gameplay_sha256", "solution_semantic_sha256",
    "geometry_split", "split_partition_bucket", "official_copy", "solution",
    "context_solution", "solution_length", "solution_mechanics",
    "structural_metrics", "native_budget", "budget_remaining", "search_exact",
    "search_truncated", "engine_verified", "optimality_claim",
    "generation_exclusions", "proof",
})
_OPTIONAL_GAME_FIELDS = frozenset({"game_seed", "game_ordinal"})
_CURRICULUM_FIELDS = frozenset({"difficulty", "context_index", "search_work"})
_METRIC_FIELDS = frozenset({
    "native_budget", "movables", "flexible", "fixed_center", "obstacles",
    "dyes", "target_colored", "target_guides", "visual_nonbackground",
})
_MECHANIC_INTEGER_FIELDS = frozenset({
    "move_actions", "selection_actions", "dye_events", "dye_animation_frames",
    "resize_events", "deformation_events", "blocked_moves",
    "fixed_center_selections", "flexible_selections",
    "target_constrained_selection_centers", "same_color_shape_pairs",
    "actions_replayed",
})
_MECHANIC_FIELDS = _MECHANIC_INTEGER_FIELDS | frozenset({
    "distinct_selected", "ambiguous_target_assignment", "engine_win",
})
_PROOF_FIELDS = frozenset({
    "kind", "format", "source_id", "vendored_source_sha256", "seed",
    "generation_attempt", "difficulty", "split", "context_index",
    "training_context_index", "verification_level_index", "native_budget",
    "action_count", "search_limit", "search_work", "search_truncated", "optimal",
    "context_engine_verified", "engine_win", "geometry_d4_sha256",
    "gameplay_sha256", "solution_semantic_sha256",
    "generation_exclusions_sha256", "generator_version", "difficulty_version",
    "mechanics_inventory_version", "quality_profile_version", "geometry_version",
    "gameplay_identity_version", "solution_semantic_version",
})


def _is_sha256(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _field_names(values):
    """Render even non-JSON mapping keys as a finite validation diagnostic."""
    return ", ".join(sorted(
        value if type(value) is str else repr(value) for value in values
    ))


def _strict_component_errors(values, label, allowed_prototypes, *, rotations,
                             recolor=False):
    errors = []
    if type(values) is not list:
        return [f"{label} must be a list"]
    for index, component in enumerate(values):
        prefix = f"{label}[{index}]"
        if type(component) is not dict:
            errors.append(f"{prefix} must be an object")
            continue
        allowed = {"prototype", "position", "rotation"}
        if recolor:
            allowed.add("recolor")
        if set(component) - allowed:
            errors.append(f"{prefix} has unsupported fields")
        prototype = component.get("prototype")
        if type(prototype) is not str or prototype not in allowed_prototypes:
            errors.append(f"{prefix}.prototype is unsupported")
        position = component.get("position")
        if (type(position) is not list or len(position) != 2
                or any(type(value) is not int for value in position)):
            errors.append(f"{prefix}.position must be two JSON integers")
        rotation = component.get("rotation")
        if type(rotation) is not int or rotation not in rotations:
            errors.append(f"{prefix}.rotation is unsupported")
        if "recolor" in component:
            value = component["recolor"]
            if not recolor or type(value) is not int or not 0 <= value <= 15:
                errors.append(f"{prefix}.recolor is outside palette 0..15")
    return errors


def _strict_target_errors(target):
    if type(target) is not dict or set(target) != {"colored", "guides"}:
        return ["target must contain exactly colored and guides lists"]
    errors = []
    guides = target["guides"]
    if type(guides) is not list:
        errors.append("target.guides must be a list")
    else:
        for index, value in enumerate(guides):
            if (type(value) is not list or len(value) != 2
                    or any(type(item) is not int for item in value)
                    or not all(0 <= item < 64 for item in value)):
                errors.append(f"target.guides[{index}] is not a frame coordinate")
    colored = target["colored"]
    if type(colored) is not list:
        errors.append("target.colored must be a list")
    else:
        for index, value in enumerate(colored):
            valid = (type(value) is list and len(value) == 3
                     and all(type(item) is int for item in value))
            if not valid:
                errors.append(f"target.colored[{index}] must be three JSON integers")
                continue
            row, col, color = value
            if not (0 <= row < 64 and 0 <= col < 64):
                errors.append(f"target.colored[{index}] is outside the frame")
            if not (0 <= color <= 15) or color in (
                    names.SELECTED_CENTER, names.TARGET_GUIDE):
                errors.append(f"target.colored[{index}] is outside the movable palette")
    return errors


def _strict_action_errors(actions, label):
    if type(actions) is not list:
        return [f"{label} must be a list"]
    errors = []
    for index, action in enumerate(actions):
        if (type(action) is not list or len(action) != 3
                or type(action[0]) is not int or action[0] not in names.ACTION_IDS
                or action[1] is not None or action[2] is not None):
            errors.append(f"{label}[{index}] is not a native display action")
    return errors


def _preflight_errors(spec, curriculum_entry):
    """Fail closed on JSON/provenance/certificate facts before native code."""
    if type(spec) is not dict:
        return ["spec must be a JSON object"]
    if type(curriculum_entry) is not dict:
        return ["curriculum entry must be a JSON object"]
    errors = []
    keys = set(spec)
    missing = _TOP_LEVEL_FIELDS - keys
    extra = keys - _TOP_LEVEL_FIELDS - _OPTIONAL_GAME_FIELDS
    if missing:
        errors.append("missing required fields: " + _field_names(missing))
    if extra:
        errors.append("unsupported top-level fields: " + _field_names(extra))
    if missing:
        return errors

    difficulty = spec["difficulty"]
    canonical_entry = (FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
                       if type(difficulty) is int and difficulty in DIFFICULTIES
                       else None)
    curriculum_schema_valid = set(curriculum_entry) == _CURRICULUM_FIELDS
    if not curriculum_schema_valid:
        errors.append(
            "curriculum entry must contain exactly difficulty, context_index, "
            "and search_work"
        )
    else:
        for key in sorted(_CURRICULUM_FIELDS):
            if type(curriculum_entry[key]) is not int:
                errors.append(f"curriculum {key} must be a JSON integer")
    curriculum_types_valid = curriculum_schema_valid and all(
        type(curriculum_entry[key]) is int for key in _CURRICULUM_FIELDS
    )
    if (canonical_entry is not None and curriculum_types_valid
            and curriculum_entry != canonical_entry):
        errors.append("curriculum entry differs from the canonical tier entry")
    if canonical_entry is None:
        return errors + ["difficulty is not an official tier"]
    profile = PROFILES[difficulty]
    context = difficulty - 1

    exact = {
        "format": FORMAT,
        "game": "re86",
        "source_id": SOURCE_ID,
        "vendored_source_sha256": SOURCE_SHA256,
        "generator_version": GENERATOR_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "gameplay_identity_version": GAMEPLAY_IDENTITY_VERSION,
        "solution_semantic_version": SOLUTION_SEMANTIC_VERSION,
        "source": "generated_only",
        "step_budget": profile["native_budget"],
        "native_budget": profile["native_budget"],
        "split": spec["split"] if spec["split"] in SPLITS else None,
        "geometry_split": spec["split"] if spec["split"] in SPLITS else None,
        "context_index": context,
        "training_context_index": context,
        "verification_level_index": context,
        "search_exact": False,
        "search_truncated": False,
        "engine_verified": True,
        "optimality_claim": "none-constructive-native-witness",
    }
    for key, expected in exact.items():
        if spec[key] != expected or type(spec[key]) is not type(expected):
            errors.append(f"{key} differs from the canonical generated value")
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        errors.append("seed must be a nonnegative JSON integer")
    if type(spec["generation_attempt"]) is not int or spec["generation_attempt"] < 1:
        errors.append("generation_attempt must be a positive JSON integer")
    if (type(spec["search_limit"]) is not int
            or not 1 <= spec["search_limit"] <= canonical_entry["search_work"]):
        errors.append("search_limit is outside the canonical tier cap")
    for key in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256",
                "solution_semantic_sha256"):
        if not _is_sha256(spec[key]):
            errors.append(f"{key} is not a lowercase SHA-256")
    if spec["geometry_sha256"] != spec["geometry_d4_sha256"]:
        errors.append("raw/canonical geometry identity mismatch")
    if type(spec["split_partition_bucket"]) is not int or not 0 <= spec[
            "split_partition_bucket"] < len(SPLITS):
        errors.append("split_partition_bucket is invalid")
    if type(spec["official_copy"]) is not bool:
        errors.append("official_copy must be boolean")

    errors.extend(_strict_component_errors(
        spec["movables"], "movables", set(RIGID) | set(FLEXIBLE) | set(FIXED),
        rotations={0}, recolor=True,
    ))
    errors.extend(_strict_component_errors(
        spec["dyes"], "dyes", ADMISSIBLE_DYES,
        rotations={0, 90, 180, 270}, recolor=False,
    ))
    errors.extend(_strict_component_errors(
        spec["obstacles"], "obstacles", set(OBSTACLES),
        rotations={0}, recolor=False,
    ))
    errors.extend(_strict_target_errors(spec["target"]))
    errors.extend(_strict_action_errors(spec["solution"], "solution"))
    errors.extend(_strict_action_errors(spec["context_solution"], "context_solution"))
    if spec["context_solution"] != spec["solution"]:
        errors.append("context_solution differs from solution")
    action_count = len(spec["solution"]) if type(spec["solution"]) is list else -1
    if type(spec["solution_length"]) is not int or spec["solution_length"] != action_count:
        errors.append("solution_length differs from the executed route")
    expected_remaining = profile["native_budget"] - action_count
    if (type(spec["budget_remaining"]) is not int
            or spec["budget_remaining"] != expected_remaining
            or expected_remaining < 0):
        errors.append("budget_remaining differs from native budget minus route length")

    metrics = spec["structural_metrics"]
    if type(metrics) is not dict or set(metrics) != _METRIC_FIELDS:
        errors.append("structural_metrics has an invalid schema")
    elif any(type(value) is not int or value < 0 for value in metrics.values()):
        errors.append("structural_metrics values must be nonnegative JSON integers")
    mechanics = spec["solution_mechanics"]
    if type(mechanics) is not dict or set(mechanics) != _MECHANIC_FIELDS:
        errors.append("solution_mechanics has an invalid schema")
    else:
        for key in _MECHANIC_INTEGER_FIELDS:
            if type(mechanics[key]) is not int or mechanics[key] < 0:
                errors.append(f"solution_mechanics.{key} is invalid")
        selected = mechanics["distinct_selected"]
        if (type(selected) is not list
                or any(type(value) is not int or not 0 <= value < profile["movables"]
                       for value in selected)
                or selected != sorted(set(selected))):
            errors.append("solution_mechanics.distinct_selected is invalid")
        for key in ("ambiguous_target_assignment", "engine_win"):
            if type(mechanics[key]) is not bool:
                errors.append(f"solution_mechanics.{key} must be boolean")

    exclusions = spec["generation_exclusions"]
    if type(exclusions) is not dict:
        errors.append("generation_exclusions must be an object")
        exclusions = {}
    valid_exclusions = not any(
        type(key) is not str or not key or type(value) is not int or value < 1
        for key, value in exclusions.items()
    )
    if not valid_exclusions:
        errors.append("generation_exclusions must contain positive typed counts")
    if (type(spec["generation_attempt"]) is int
            and valid_exclusions
            and sum(exclusions.values()) != spec["generation_attempt"] - 1):
        errors.append("generation exclusions do not account for prior attempts")

    game_fields = keys & _OPTIONAL_GAME_FIELDS
    if game_fields and game_fields != _OPTIONAL_GAME_FIELDS:
        errors.append("game_seed and game_ordinal must appear together")
    elif game_fields == _OPTIONAL_GAME_FIELDS:
        game_seed, ordinal = spec["game_seed"], spec["game_ordinal"]
        if (type(game_seed) is not int or game_seed < 0
                or type(ordinal) is not int or not 0 <= ordinal < len(DIFFICULTIES)
                or spec["seed"] != _child_seed(game_seed, ordinal, difficulty)):
            errors.append("whole-game provenance does not match the child seed/context")

    proof = spec["proof"]
    if type(proof) is not dict or set(proof) != _PROOF_FIELDS:
        errors.append("proof has an invalid exact schema")
        return errors
    exclusions_sha256 = None
    if valid_exclusions:
        exclusions_sha256 = hashlib.sha256(json.dumps(
            exclusions, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
    expected_proof = {
        "kind": "constructive-real-engine-replay",
        "format": FORMAT,
        "source_id": SOURCE_ID,
        "vendored_source_sha256": SOURCE_SHA256,
        "seed": spec["seed"],
        "generation_attempt": spec["generation_attempt"],
        "difficulty": difficulty,
        "split": spec["split"],
        "context_index": context,
        "training_context_index": context,
        "verification_level_index": context,
        "native_budget": profile["native_budget"],
        "action_count": action_count,
        "search_limit": spec["search_limit"],
        "search_work": action_count,
        "search_truncated": False,
        "optimal": False,
        "context_engine_verified": True,
        "engine_win": True,
        "geometry_d4_sha256": spec["geometry_d4_sha256"],
        "gameplay_sha256": spec["gameplay_sha256"],
        "solution_semantic_sha256": spec["solution_semantic_sha256"],
        "generation_exclusions_sha256": exclusions_sha256,
        "generator_version": GENERATOR_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "gameplay_identity_version": GAMEPLAY_IDENTITY_VERSION,
        "solution_semantic_version": SOLUTION_SEMANTIC_VERSION,
    }
    for key, expected in expected_proof.items():
        if proof[key] != expected or type(proof[key]) is not type(expected):
            errors.append(f"proof {key} differs from the executed/canonical fact")
    if type(spec["search_limit"]) is int and action_count > spec["search_limit"]:
        errors.append("constructive work exceeds the declared search limit")
    return errors


def validate_full_standard(spec, curriculum_entry):
    """Recompute structure, identities, partition, replay, and mechanic use."""
    errors = _preflight_errors(spec, curriculum_entry)
    if errors:
        return list(dict.fromkeys(errors))
    difficulty = spec["difficulty"]
    context = difficulty - 1
    try:
        env = env_for(spec, context)
        if env.level_index != context or env.max_steps != PROFILES[difficulty]["native_budget"]:
            errors.append("native context or budget differs from the canonical tier")
        metrics = structural_metrics(env)
        if metrics != spec["structural_metrics"]:
            errors.append("stored structural metrics differ from native reconstruction")
        errors.extend(profile_errors(spec, metrics))
        errors.extend(_presentation_errors(spec))
        geometry = geometry_identity(spec)
        if geometry != spec["geometry_d4_sha256"]:
            errors.append("geometry identity mismatch")
        if (geometry in _official_geometry_hashes()) != spec["official_copy"]:
            errors.append("official-copy declaration differs from canonical geometry")
        if spec["official_copy"]:
            errors.append("official geometry copies are not admissible")
        gameplay = gameplay_identity(spec)
        if gameplay != spec["gameplay_sha256"]:
            errors.append("gameplay identity mismatch")
        if semantic_partition(gameplay) != spec["split"]:
            errors.append("public gameplay partition mismatch")
        if int(gameplay, 16) % len(SPLITS) != spec["split_partition_bucket"]:
            errors.append("public gameplay partition bucket mismatch")
        route_semantic = solution_semantic_identity(spec)
        if route_semantic != spec["solution_semantic_sha256"]:
            errors.append("solution semantic identity mismatch")
        normalized = tuple(tuple(action) for action in spec["solution"])
        won, _, replayed = _replay_witness(spec, normalized, context)
        if not won:
            errors.append("solution does not win exactly on its final action")
        elif replayed.actions_used != len(normalized):
            errors.append("native action count differs from the certificate length")
        mechanics = solution_mechanics(env_for(spec, context), normalized)
        if mechanics != spec["solution_mechanics"]:
            errors.append("solution mechanic evidence differs from native replay")
    except Exception as error:
        errors.append(f"native validation failed: {type(error).__name__}: {error}")
    return list(dict.fromkeys(errors))


def build_game(specs):
    """Validate and build exactly one ordered eight-context native episode."""
    if not isinstance(specs, (list, tuple)) or len(specs) != len(DIFFICULTIES):
        raise ValueError("full-standard RE86 games require exactly eight specs")
    splits = {spec.get("split") for spec in specs if isinstance(spec, Mapping)}
    if len(splits) != 1:
        raise ValueError("all eight specs must use one split")
    game_seeds = {
        spec.get("game_seed") for spec in specs
        if isinstance(spec, Mapping) and "game_seed" in spec
    }
    if game_seeds and (len(game_seeds) != 1
                       or any("game_seed" not in spec for spec in specs)):
        raise ValueError("whole-game provenance must be present and consistent")
    geometries = set()
    gameplays = set()
    for index, (difficulty, spec, curriculum) in enumerate(zip(
            DIFFICULTIES, specs, FULL_STANDARD_CONTRACT["curriculum"])):
        if (not isinstance(spec, Mapping) or spec.get("difficulty") != difficulty
                or spec.get("training_context_index") != index):
            raise ValueError("specs must be ordered difficulties 1..8 at contexts 0..7")
        if game_seeds and spec.get("game_ordinal") != index:
            raise ValueError("whole-game ordinal differs from native context order")
        errors = validate_full_standard(spec, curriculum)
        if errors:
            raise ValueError("invalid full-standard spec: " + "; ".join(errors))
        if spec["geometry_d4_sha256"] in geometries:
            raise ValueError("duplicate geometry identity within a game")
        if spec["gameplay_sha256"] in gameplays:
            raise ValueError("duplicate gameplay identity within a game")
        geometries.add(spec["geometry_d4_sha256"])
        gameplays.add(spec["gameplay_sha256"])
    levels = [build_level(spec) for spec in specs]
    episode = Env(levels)
    observation = None
    for index, spec in enumerate(specs):
        if episode.level_index != index or episode.levels_completed != index:
            raise ValueError("native episode entered the wrong RE86 context")
        for action in spec["solution"]:
            observation = episode.perform(*action)
        if episode.levels_completed != index + 1:
            raise ValueError("witness did not advance exactly one RE86 tier")
    if observation is None or not observation.won:
        raise ValueError("eight-tier RE86 episode did not reach WIN")
    return levels
