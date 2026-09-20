"""Full-mechanics, reference-calibrated procedural R11L generation.

The six tiers correspond one-for-one with the six shipped levels.  Geometry is
newly sampled; shipped sprites are used only as the native mechanic alphabet.
Every accepted row carries a bounded constructive witness replayed by the real
engine at the matching native level index.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from hashlib import blake2b, sha256
import json
from numbers import Integral
import random

import numpy as np
from arcengine import GameState, Level, Sprite

from pebby import multigame as M

from . import names
from .env import Env, official_levels
from .layout import _global_mask, extract
from .plan import (
    DEFAULT_LIMIT,
    _core_position,
    _group_satisfied,
    _hazard_free,
    _overlaps,
    search,
)


FORMAT = "pebby.r11l.level.v3"
GENERATOR_VERSION = 3
MECHANICS_INVENTORY_VERSION = "r11l-full-mechanics-v2"
QUALITY_PROFILE_VERSION = "r11l-official-reference-v2"
GEOMETRY_VERSION = "r11l-d4-semantic-relations-v2"
DIFFICULTIES = tuple(range(1, 7))
SPLITS = ("train", "validation", "test")
MAX_ATTEMPTS = 36


# One shipped level exists per tier. These are descriptive measurements of
# those six scarce references, so tolerances below are explicit engineering
# bands rather than population confidence intervals. ``reference_actions`` is
# the measured constructive-teacher witness, never claimed to be optimal.
REFERENCE_PROFILES = {
    1: dict(reference_level=1, reference_actions=4, action_range=(3, 5),
            fragment_counts=(2,), required_targets=1, visual_targets=1,
            visual_decoys=0, absorbers=0, pickups=0, useful_pickups=0,
            wall_pixels=660, wall_range=(590, 730),
            hazard_pixels=0, hazard_range=(0, 0),
            occupied_visual_pixels=778, occupied_visual_range=(660, 895),
            reference_wall_constrained_drags=1, min_wall_constrained_drags=1,
            reference_hazard_constrained_drags=0, min_hazard_constrained_drags=0),
    2: dict(reference_level=2, reference_actions=7, action_range=(5, 9),
            fragment_counts=(2, 3), required_targets=2, visual_targets=2,
            visual_decoys=0, absorbers=0, pickups=0, useful_pickups=0,
            wall_pixels=565, wall_range=(500, 630),
            hazard_pixels=691, hazard_range=(620, 770),
            occupied_visual_pixels=1349, occupied_visual_range=(1180, 1520),
            reference_wall_constrained_drags=4, min_wall_constrained_drags=3,
            reference_hazard_constrained_drags=3, min_hazard_constrained_drags=2),
    3: dict(reference_level=3, reference_actions=11, action_range=(9, 13),
            fragment_counts=(2, 4), required_targets=2, visual_targets=2,
            visual_decoys=0, absorbers=0, pickups=0, useful_pickups=0,
            wall_pixels=587, wall_range=(520, 655),
            hazard_pixels=1002, hazard_range=(900, 1100),
            occupied_visual_pixels=1657, occupied_visual_range=(1450, 1870),
            reference_wall_constrained_drags=4, min_wall_constrained_drags=3,
            reference_hazard_constrained_drags=5, min_hazard_constrained_drags=4),
    4: dict(reference_level=4, reference_actions=14, action_range=(12, 16),
            fragment_counts=(2, 2, 3), required_targets=3, visual_targets=8,
            visual_decoys=5, absorbers=0, pickups=0, useful_pickups=0,
            wall_pixels=371, wall_range=(330, 420),
            hazard_pixels=223, hazard_range=(195, 255),
            occupied_visual_pixels=902, occupied_visual_range=(790, 1015),
            reference_wall_constrained_drags=6, min_wall_constrained_drags=5,
            reference_hazard_constrained_drags=4, min_hazard_constrained_drags=3),
    5: dict(reference_level=5, reference_actions=17, action_range=(15, 19),
            fragment_counts=(2, 3), required_targets=2, visual_targets=5,
            visual_decoys=3, absorbers=2, pickups=4, useful_pickups=4,
            wall_pixels=401, wall_range=(355, 450),
            hazard_pixels=0, hazard_range=(0, 0),
            occupied_visual_pixels=692, occupied_visual_range=(610, 795),
            reference_wall_constrained_drags=10, min_wall_constrained_drags=8,
            reference_hazard_constrained_drags=0, min_hazard_constrained_drags=0),
    6: dict(reference_level=6, reference_actions=18, action_range=(16, 20),
            fragment_counts=(2, 3), required_targets=2, visual_targets=3,
            visual_decoys=1, absorbers=2, pickups=9, useful_pickups=6,
            wall_pixels=423, wall_range=(375, 475),
            hazard_pixels=0, hazard_range=(0, 0),
            occupied_visual_pixels=802, occupied_visual_range=(700, 910),
            reference_wall_constrained_drags=11, min_wall_constrained_drags=9,
            reference_hazard_constrained_drags=0, min_hazard_constrained_drags=0),
}


ORDINARY_TIERS = {
    1: (("pumlzd", 2),),
    2: (("orrqlj", 3), ("pumlzd", 2)),
    3: (("grhcew", 4), ("pumlzd", 2)),
    4: (
        ("blxuubrengnt", 2),
        ("orrqljliwocqblxuubpumlzd", 2),
        ("yeogyfgrhcew", 3),
    ),
}

TARGET_COLOUR_SETS = {
    5: (("blxuubrengnt", (8, 9)), ("yeogyfgrhcew", (11, 14))),
    6: (("pumlzdgrhcewblxuub", (9, 14, 15)),
        ("yeogyfpiocnkblxuub", (6, 10, 11))),
}

PICKUP_FOR_TIER = {
    5: {
        8: "hawffurengnt", 9: "hawffublxuub",
        11: "hawffuyeogyf", 14: "hawffugrhcew",
    },
    6: {
        6: "hawffupiocnk", 8: "blxuubcoffvq",
        9: "hawffublxuubbonbid", 10: "liwocqblxuubcoffvq",
        11: "yeogyfcoffvq", 12: "orrqlj", 13: "madctp",
        14: "grhcew", 15: "pumlzdcoffvq",
    },
}


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": M.source_for("r11l").source_id,
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": tuple(
        {"difficulty": difficulty, "context_index": difficulty - 1,
         "search_work": DEFAULT_LIMIT}
        for difficulty in DIFFICULTIES
    ),
    "evidence": {
        "official_tier_characterization": ".scratch/multigame-resume/full-standard/r11l.md#reference-characterization",
        "solution_mechanics": "pebby/games/r11l/generate.py:solution_mechanics",
        "native_budget": "third_party/arc3_games/r11l.py:1407,1807-1811",
        "context_engine_replay": "tests/games/test_r11l.py:test_full_generated_game_replays_without_forced_transitions",
        "novelty_split": "pebby/games/r11l/generate.py:canonical_identity",
        "bounded_rejections": "pebby/games/r11l/generate.py:generate",
    },
    "caveats": (
        "Each tier has one shipped reference; tolerance bands are not population confidence intervals.",
        "Teacher witnesses are constructive positive certificates and are not claimed shortest.",
        "Generated geometry uses a rectangular lane/obstacle grammar; action/profile similarity is based on one official reference per tier, not population equivalence.",
    ),
}


def _rect(x, y, width, height):
    return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}


def _validate_rect(rect, label):
    values = tuple(int(rect[key]) for key in ("x", "y", "width", "height"))
    x, y, width, height = values
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 64 or y + height > 64:
        raise ValueError(f"{label} rectangle is outside the 64x64 board")
    return values


def _base_walls(rng, difficulty):
    thickness = 2 if difficulty <= 3 else 1
    walls = [
        _rect(0, 0, 64, thickness), _rect(0, 64 - thickness, 64, thickness),
        _rect(0, 0, thickness, 64), _rect(64 - thickness, 0, thickness, 64),
    ]
    x_offset = rng.randint(-2, 2)
    if difficulty == 1:
        walls.extend((_rect(28 + x_offset, 3, 3, 14), _rect(30 - x_offset, 46, 3, 14),
                      _rect(17 + rng.randint(-1, 1), 24, 10, 8)))
    elif difficulty == 2:
        walls.extend((_rect(28 + x_offset, 3, 3, 12), _rect(30 - x_offset, 49, 3, 11)))
    elif difficulty == 3:
        walls.extend((_rect(28 + x_offset, 3, 3, 16), _rect(30 - x_offset, 46, 3, 14)))
    else:
        walls.append(_rect(29 + x_offset, 25, 3, 14))
        if difficulty == 4:
            walls.append(_rect(17 + rng.randint(-1, 1), 3, 7, 11))
        elif difficulty == 5:
            walls.append(_rect(25 + rng.randint(-1, 1), 3, 10, 11))
        else:
            walls.extend((_rect(24 + rng.randint(-1, 1), 3, 11, 11), _rect(57, 28, 2, 4)))
    return walls


def _ordinary_hazards(rng, difficulty, groups):
    if difficulty == 1:
        return []
    if difficulty == 2:
        hazards = [_rect(5, 25 + rng.randint(-1, 1), 52, 13)]
    elif difficulty == 3:
        hazards = [_rect(2, 24 + rng.randint(-1, 1), 60, 16)]
    else:
        hazards = [_rect(22, 17 + rng.randint(-1, 1), 22, 8)]
    for group in groups:
        tx, ty = group["target"]
        side = 1 if tx < 20 else -1
        hx = tx + 8 if side > 0 else tx - 6
        if 0 <= hx <= 61:
            hazards.append(_rect(hx, max(2, min(57, ty + 1)), 3, 5))
    return hazards


def _draft_ordinary(rng, seed, difficulty, attempt):
    templates = ORDINARY_TIERS[difficulty]
    lanes = {1: (31,), 2: (13, 47), 3: (13, 47), 4: (9, 31, 51)}[difficulty]
    groups = []
    for ordinal, ((prototype, fragment_count), base_y) in enumerate(zip(templates, lanes)):
        target_left = difficulty == 4 and ordinal in (0, 2)
        target_x = rng.randint(4, 9) if target_left else rng.randint(47, 53)
        target_y = max(3, min(54, base_y + rng.randint(-2, 2)))
        source_right = target_left
        start_x = rng.randint(43, 49) if source_right else rng.randint(4, 8)
        fragments = []
        for index in range(fragment_count):
            x = start_x - index * 7 if source_right else start_x + index * 7
            y = max(2, min(57, target_y + rng.randint(-5, 5)))
            fragments.append([x, y])
        groups.append({
            "prototype": prototype,
            "target": [target_x, target_y],
            "fragments": fragments,
            "ordinal": ordinal,
        })
    decoys = []
    if difficulty == 4:
        decoy_positions = ((9, 15), (51, 51), (17, 5), (11, 40), (50, 28))
        for ordinal, (x, y) in enumerate(decoy_positions):
            decoys.append({"position": [x + rng.randint(-1, 1), y + rng.randint(-1, 1)],
                            "colour": (7, 9, 12, 13, 14)[ordinal]})
    spec = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "seed": int(seed),
        "generation_attempt": int(attempt),
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "mode": "ordinary-centroid",
        "groups": groups,
        "absorbers": [],
        "targets": [],
        "pickups": [],
        "decoys": decoys,
        "walls": _base_walls(rng, difficulty),
    }
    spec["hazards"] = _ordinary_hazards(rng, difficulty, groups)
    return spec


def _draft_absorption(rng, seed, difficulty, attempt):
    lane_ys = [13 + rng.randint(-2, 2), 46 + rng.randint(-2, 2)]
    target_templates = list(TARGET_COLOUR_SETS[difficulty])
    rng.shuffle(target_templates)
    targets = []
    pickups = []
    # Alternating objectives force real centroid reconfiguration comparable to
    # the shipped pickup routes; monotone lanes collapse to one drag per item.
    pickup_xs = (48, 20) if difficulty == 5 else (46, 18, 40)
    for target_index, ((prototype, colours), lane_y) in enumerate(zip(target_templates, lane_ys)):
        target_core_x = 50 + rng.randint(-2, 1)
        targets.append({
            "prototype": prototype,
            "position": [target_core_x - 1, lane_y - 1],
            "colours": list(colours),
            "target_index": target_index,
        })
        for colour, base_x in zip(colours, pickup_xs):
            pickups.append({
                "prototype": PICKUP_FOR_TIER[difficulty][colour],
                "position": [base_x + rng.randint(-1, 1), lane_y],
                "colour": colour,
                "role": "useful",
                "target_index": target_index,
            })
    if difficulty == 6:
        for colour, x in zip((8, 12, 13), (14, 31, 49)):
            pickups.append({
                "prototype": PICKUP_FOR_TIER[difficulty][colour],
                "position": [x + rng.randint(-1, 1), 29 + rng.randint(-2, 2)],
                "colour": colour,
                "role": "decoy",
                "target_index": None,
            })
    absorbers = [
        {"prototype": "whkxtx", "fragments": [[5, lane_ys[0] - 3], [15, lane_ys[0] + 3]]},
        {"prototype": "whkxtx-2", "fragments": [[4, lane_ys[1]], [13, lane_ys[1] - 5],
                                                   [22, lane_ys[1] + 4]]},
    ]
    visual_decoys = 3 if difficulty == 5 else 1
    decoys = [
        {"position": [8 + index * 20 + rng.randint(-1, 1), 29 + rng.randint(-1, 1)],
         "colour": (7, 12, 14)[index]}
        for index in range(visual_decoys)
    ]
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "seed": int(seed),
        "generation_attempt": int(attempt),
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "mode": "pickup-colour-set",
        "groups": [],
        "absorbers": absorbers,
        "targets": targets,
        "pickups": pickups,
        "decoys": decoys,
        "walls": _base_walls(rng, difficulty),
        "hazards": [],
    }


def _draft(rng, seed, difficulty, attempt):
    if difficulty <= 4:
        return _draft_ordinary(rng, seed, difficulty, attempt)
    return _draft_absorption(rng, seed, difficulty, attempt)


def _solid_sprite(sprite_name, rect, colour):
    x, y, width, height = _validate_rect(rect, sprite_name)
    return Sprite(
        [[colour] * width for _ in range(height)],
        name=sprite_name, x=x, y=y, visible=True, collidable=True,
    )


def build_level(spec):
    """Reconstruct a real ARCEngine level from a JSON-round-trippable spec."""
    if spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    module = __import__("pebby.games.r11l.env", fromlist=["upstream"]).upstream()
    prototypes = module.sprites
    sprites = []

    seen = set()
    for group in spec.get("groups", []):
        prototype = str(group["prototype"])
        if prototype in seen or names.PREFIX_CORE + prototype not in prototypes:
            raise ValueError("ordinary groups need distinct native prototypes")
        seen.add(prototype)
        target_x, target_y = map(int, group["target"])
        fragments = [tuple(map(int, point)) for point in group["fragments"]]
        if len(fragments) < 2:
            raise ValueError("every centroid group needs at least two fragments")
        sprites.append(prototypes[names.PREFIX_TARGET + prototype].clone().set_position(target_x, target_y))
        sprites.append(prototypes[names.PREFIX_CORE + prototype].clone().set_position(0, 0))
        sprites.extend(
            prototypes[names.PREFIX_FRAGMENT + prototype].clone().set_position(x, y)
            for x, y in fragments
        )

    for absorber in spec.get("absorbers", []):
        prototype = str(absorber["prototype"])
        fragments = [tuple(map(int, point)) for point in absorber["fragments"]]
        if prototype in seen or prototype not in ("whkxtx", "whkxtx-2") or len(fragments) < 2:
            raise ValueError("invalid absorbing-core group")
        seen.add(prototype)
        sprites.append(prototypes[names.PREFIX_CORE + prototype].clone().set_position(0, 0))
        sprites.extend(
            prototypes[names.PREFIX_FRAGMENT + prototype].clone().set_position(x, y)
            for x, y in fragments
        )

    for target in spec.get("targets", []):
        prototype = str(target["prototype"])
        x, y = map(int, target["position"])
        sprites.append(prototypes[names.PREFIX_TARGET + prototype].clone().set_position(x, y))
    for pickup in spec.get("pickups", []):
        prototype = str(pickup["prototype"])
        x, y = map(int, pickup["position"])
        sprites.append(prototypes[names.PREFIX_PICKUP + prototype].clone().set_position(x, y))
    for decoy in spec.get("decoys", []):
        x, y = map(int, decoy["position"])
        colour = int(decoy["colour"])
        target = prototypes[names.PREFIX_TARGET + names.DECOY_GROUP_MARKER].clone().set_position(x, y)
        target.color_remap(None, colour)
        sprites.append(target)
        decor_name = names.PREFIX_DECOR + names.DECOY_GROUP_MARKER
        if decor_name in prototypes:
            decor = prototypes[decor_name].clone().set_position(x + 4, y)
            decor.color_remap(None, colour)
            sprites.append(decor)

    wall_pixels = [[-1] * 64 for _ in range(64)]
    for rect in spec.get("walls", []):
        x, y, width, height = _validate_rect(rect, "wall")
        for py in range(y, y + height):
            for px in range(x, x + width):
                wall_pixels[py][px] = 2
    sprites.append(Sprite(
        wall_pixels, name=names.PREFIX_WALL + "generated", x=0, y=0,
        visible=True, collidable=True,
    ))
    for index, rect in enumerate(spec.get("hazards", [])):
        sprites.append(_solid_sprite(
            names.PREFIX_HAZARD + f"-generated-{index}", rect, 10
        ))
    return Level(
        sprites=sprites, grid_size=(64, 64),
        name=f"generated-r11l-d{difficulty}-s{spec.get('seed', 0)}",
    )


def _geometry_tokens(spec):
    tokens = []
    for rect in spec.get("walls", []):
        x, y, width, height = _validate_rect(rect, "wall")
        tokens.extend(("wall", px, py) for py in range(y, y + height) for px in range(x, x + width))
    for rect in spec.get("hazards", []):
        x, y, width, height = _validate_rect(rect, "hazard")
        tokens.extend(("hazard", px, py) for py in range(y, y + height) for px in range(x, x + width))
    for group in spec.get("groups", []):
        label = f"group-{len(group['fragments'])}"
        tokens.append((label + "-target", *map(int, group["target"])))
        tokens.extend((label + "-fragment", *map(int, point)) for point in group["fragments"])
    for absorber in spec.get("absorbers", []):
        label = f"absorber-{len(absorber['fragments'])}"
        tokens.extend((label + "-fragment", *map(int, point)) for point in absorber["fragments"])
    for target in spec.get("targets", []):
        colours = _prototype_colours(names.PREFIX_TARGET + str(target["prototype"]))
        tokens.append((f"colour-target-{len(colours)}", *map(int, target["position"])))
    for pickup in spec.get("pickups", []):
        colours = _prototype_colours(names.PREFIX_PICKUP + str(pickup["prototype"]))
        target_sets = [
            _prototype_colours(names.PREFIX_TARGET + str(target["prototype"]))
            for target in spec.get("targets", [])
        ]
        role = "useful" if any(colours <= target for target in target_sets) else "decoy"
        tokens.append((f"pickup-{role}", *map(int, pickup["position"])))
    for decoy in spec.get("decoys", []):
        tokens.append(("visual-decoy", *map(int, decoy["position"])))
    return list(set(tokens))


def _canonical_geometry_hash(tokens):
    if not tokens:
        raise ValueError("cannot identify empty geometry")
    variants = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                changed = []
                for label, x, y in tokens:
                    tx, ty = (y, x) if swap else (x, y)
                    changed.append((label, sx * tx, sy * ty))
                left = min(x for _, x, _ in changed)
                top = min(y for _, _, y in changed)
                variants.append(sorted((label, x - left, y - top) for label, x, y in changed))
    return sha256(json.dumps(min(variants), separators=(",", ":")).encode()).hexdigest()


def _prototype(name):
    module = __import__("pebby.games.r11l.env", fromlist=["upstream"]).upstream()
    try:
        return module.sprites[name]
    except KeyError as error:
        raise ValueError(f"unknown native sprite prototype {name!r}") from error


def _prototype_colours(name):
    return frozenset(int(value) for value in np.unique(_prototype(name).pixels) if value > 0)


def _raw_pattern(name):
    pixels = np.asarray(_prototype(name).pixels)
    return tuple(tuple(int(value) for value in row) for row in pixels)


def _pattern_shape(pattern):
    """Colour-independent ordering key; colour equality is normalized later."""
    return tuple(tuple(value if value <= 0 else 1 for value in row) for row in pattern)


def _normalise_pattern_colours(pattern, colour_ids):
    rows = []
    for row in pattern:
        changed = []
        for value in row:
            if value <= 0:
                changed.append(value)
            else:
                changed.append(f"c{colour_ids.setdefault(value, len(colour_ids))}")
        rows.append(changed)
    return rows


def _semantic_variant(spec, *, swap, sx, sy):
    transformed_geometry = []
    for label, x, y in _geometry_tokens(spec):
        tx, ty = (y, x) if swap else (x, y)
        transformed_geometry.append((label, sx * tx, sy * ty))
    left = min(x for _, x, _ in transformed_geometry)
    top = min(y for _, _, y in transformed_geometry)

    def point(value):
        x, y = map(int, value)
        tx, ty = (y, x) if swap else (x, y)
        return [sx * tx - left, sy * ty - top]

    geometry = sorted((label, x - left, y - top) for label, x, y in transformed_geometry)
    ordinary = []
    for group in spec.get("groups", []):
        prototype = str(group["prototype"])
        patterns = {
            "target": _raw_pattern(names.PREFIX_TARGET + prototype),
            "core": _raw_pattern(names.PREFIX_CORE + prototype),
            "fragment": _raw_pattern(names.PREFIX_FRAGMENT + prototype),
        }
        ordinary.append({
            "target": point(group["target"]),
            "fragments": sorted(point(value) for value in group["fragments"]),
            "patterns": patterns,
        })
    ordinary.sort(key=lambda group: (
        group["target"], group["fragments"],
        tuple((name, _pattern_shape(pattern)) for name, pattern in sorted(group["patterns"].items())),
    ))

    absorbers = []
    for group in spec.get("absorbers", []):
        prototype = str(group["prototype"])
        patterns = {
            "core": _raw_pattern(names.PREFIX_CORE + prototype),
            "fragment": _raw_pattern(names.PREFIX_FRAGMENT + prototype),
        }
        absorbers.append({
            "fragments": sorted(point(value) for value in group["fragments"]),
            "patterns": patterns,
        })
    absorbers.sort(key=lambda group: (
        group["fragments"],
        tuple((name, _pattern_shape(pattern)) for name, pattern in sorted(group["patterns"].items())),
    ))

    targets = [
        {
            "position": point(target["position"]),
            "pattern": _raw_pattern(names.PREFIX_TARGET + str(target["prototype"])),
        }
        for target in spec.get("targets", [])
    ]
    targets.sort(key=lambda target: (target["position"], _pattern_shape(target["pattern"])))
    pickups = [
        {
            "position": point(pickup["position"]),
            "pattern": _raw_pattern(names.PREFIX_PICKUP + str(pickup["prototype"])),
        }
        for pickup in spec.get("pickups", [])
    ]
    pickups.sort(key=lambda pickup: (pickup["position"], _pattern_shape(pickup["pattern"])))

    # Normalize positive colour IDs globally in canonical spatial order. This
    # removes palette cosmetics while preserving equality and assignment
    # relationships across cores, targets, fragments, and exact pickup masks.
    colour_ids = {}
    for group in ordinary:
        group["patterns"] = {
            name: _normalise_pattern_colours(pattern, colour_ids)
            for name, pattern in sorted(group["patterns"].items())
        }
    for group in absorbers:
        group["patterns"] = {
            name: _normalise_pattern_colours(pattern, colour_ids)
            for name, pattern in sorted(group["patterns"].items())
        }
    for target in targets:
        target["pattern"] = _normalise_pattern_colours(target["pattern"], colour_ids)
    for pickup in pickups:
        pickup["pattern"] = _normalise_pattern_colours(pickup["pattern"], colour_ids)
    return {
        "difficulty": spec["difficulty"],
        "mode": spec["mode"],
        "geometry": geometry,
        "ordinary_groups": ordinary,
        "absorber_groups": absorbers,
        "colour_targets": targets,
        "pickups": pickups,
    }


def canonical_identity(spec):
    """Canonical public gameplay under translation, D4, and palette renaming."""
    tokens = _geometry_tokens(spec)
    geometry = _canonical_geometry_hash(tokens)
    variants = [
        _semantic_variant(spec, swap=swap, sx=sx, sy=sy)
        for swap in (False, True)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    canonical = min(json.dumps(value, sort_keys=True, separators=(",", ":")) for value in variants)
    gameplay = sha256(canonical.encode()).hexdigest()
    return geometry, gameplay


def _native_geometry_tokens(level):
    env = Env([level])
    env.reset()
    board = {(x, y) for y in range(64) for x in range(64)}
    tokens = set()
    for wall in env.walls():
        tokens.update(("wall", x, y) for x, y in _global_mask(wall) & board)
    for hazard in env.hazards():
        tokens.update(("hazard", x, y) for x, y in _global_mask(hazard) & board)
    required_colour_sets = []
    for group_name, data in env.groups().items():
        fragments = data[names.KEY_FRAGMENTS]
        core = data[names.KEY_CORE]
        target = data[names.KEY_TARGET]
        if fragments:
            absorbing = core is not None and core.name.startswith(names.PREFIX_ABSORBING_CORE)
            label = f"absorber-{len(fragments)}" if absorbing else f"group-{len(fragments)}"
            tokens.update((label + "-fragment", int(sprite.x), int(sprite.y)) for sprite in fragments)
            if target is not None and not absorbing:
                tokens.add((label + "-target", int(target.x), int(target.y)))
        elif target is not None and names.DECOY_GROUP_MARKER not in group_name:
            colours = {int(value) for value in np.unique(target.pixels) if value > 0}
            required_colour_sets.append(frozenset(colours))
            tokens.add((f"colour-target-{len(colours)}", int(target.x), int(target.y)))
    for sprite in env.game.current_level.get_sprites():
        if sprite.name.startswith(names.PREFIX_TARGET + names.DECOY_GROUP_MARKER):
            tokens.add(("visual-decoy", int(sprite.x), int(sprite.y)))
    for pickup in env.pickups():
        colours = frozenset(int(value) for value in np.unique(pickup.pixels) if value > 0)
        role = "useful" if any(colours <= target for target in required_colour_sets) else "decoy"
        tokens.add((f"pickup-{role}", int(pickup.x), int(pickup.y)))
    return tokens


@lru_cache(maxsize=1)
def official_geometry_hashes():
    """D4/translation-normalized identities of all six shipped layouts."""
    return tuple(
        _canonical_geometry_hash(_native_geometry_tokens(level))
        for level in official_levels()
    )


def geometry_partition(spec):
    geometry, _ = canonical_identity(spec)
    return geometry, SPLITS[int(geometry, 16) % len(SPLITS)]


def structural_metrics(spec):
    level = build_level(spec)
    env = Env([level])
    frame = np.asarray(env.reset())
    sprites = env.game.current_level.get_sprites()
    targets = [sprite for sprite in sprites if sprite.name.startswith(names.PREFIX_TARGET)]
    required_targets = [
        data[names.KEY_TARGET]
        for group_name, data in env.groups().items()
        if data[names.KEY_TARGET] is not None and names.DECOY_GROUP_MARKER not in group_name
    ]
    target_colour_sets = [
        frozenset(int(value) for value in np.unique(target.pixels) if value > 0)
        for target in required_targets
    ]
    pickup_colour_sets = [
        frozenset(int(value) for value in np.unique(pickup.pixels) if value > 0)
        for pickup in env.pickups()
    ]
    board = {(x, y) for y in range(64) for x in range(64)}

    def covered(items):
        if not items:
            return 0
        return len(frozenset().union(*(_global_mask(item) for item in items)) & board)

    return {
        "fragment_counts": sorted(
            len(data[names.KEY_FRAGMENTS])
            for data in env.groups().values()
            if data[names.KEY_FRAGMENTS]
        ),
        "fragments": len(env.fragments()),
        "required_targets": len(required_targets),
        "visual_targets": len(targets),
        "visual_decoys": sum(names.DECOY_GROUP_MARKER in sprite.name for sprite in targets),
        "absorbers": len(env.absorbing_cores()),
        "pickups": len(env.pickups()),
        "useful_pickups": sum(
            any(colours <= target for target in target_colour_sets)
            for colours in pickup_colour_sets
        ),
        "decoy_pickups": sum(
            not any(colours <= target for target in target_colour_sets)
            for colours in pickup_colour_sets
        ),
        "multicolour_targets": sum(
            len({int(value) for value in np.unique(target.pixels) if value > 0}) > 1
            for target in required_targets
        ),
        "wall_pixels": covered(env.walls()),
        "hazard_pixels": covered(env.hazards()),
        "occupied_visual_pixels": int(np.count_nonzero(frame != 5)),
        "board_width": 64,
        "board_height": 64,
    }


def _mask_centre(mask):
    return (
        sum(x for x, _ in mask) // len(mask),
        sum(y for _, y in mask) // len(mask),
    )


def _progress_score(layout, positions, group, objective):
    core_x, core_y = _core_position(layout, positions, group)
    objective_x, objective_y = _mask_centre(objective)
    return abs(core_x + names.HALF - objective_x) + abs(core_y + names.HALF - objective_y)


def _native_constraint_rollback(env, action, kind):
    probe = env.clone()
    before_positions = [(fragment.x, fragment.y) for fragment in probe.fragments()]
    before_hazards = probe.hazards_hit()
    outcome = probe.perform(*action)
    after_positions = [(fragment.x, fragment.y) for fragment in probe.fragments()]
    if kind == "hazard":
        return (
            outcome.state != GameState.GAME_OVER
            and after_positions == before_positions
            and probe.hazards_hit() == before_hazards + 1
        )
    return after_positions == before_positions and probe.hazards_hit() == before_hazards


def _route_constraint_witnesses(env, action, route_action_index):
    """Find native-confirmed constraints relevant to this live witness drag.

    An alternative must improve the selected group's current objective (or
    satisfy its target), not merely reach an arbitrary remote wall/hazard.
    Pickup legs use the witness successor core as their current objective.
    """
    layout = extract(env)
    selected = layout.selected
    _, actual_x, actual_y = action
    if selected < 0 or any(
        px <= actual_x < px + fragment.width and py <= actual_y < py + fragment.height
        for fragment, (px, py) in zip(layout.fragments, layout.positions)
    ):
        return []
    group = next(group for group in layout.groups if selected in group.fragment_indices)
    if group.required and group.target_mask:
        objective = group.target_mask
    else:
        successor = list(layout.positions)
        successor[selected] = names.click_to_position(actual_x, actual_y)
        objective = frozenset({_core_position(layout, tuple(successor), group)})
    before_score = _progress_score(layout, layout.positions, group, objective)
    fragment = layout.fragments[selected]
    found = {}
    for click_y in range(names.FRAME_SIZE):
        for click_x in range(names.FRAME_SIZE):
            if (click_x, click_y) == (actual_x, actual_y) or any(
                px <= click_x < px + other.width and py <= click_y < py + other.height
                for other, (px, py) in zip(layout.fragments, layout.positions)
            ):
                continue
            destination = names.click_to_position(click_x, click_y)
            if destination == layout.positions[selected]:
                continue
            moved = list(layout.positions)
            moved[selected] = destination
            moved = tuple(moved)
            satisfies = bool(group.required and _group_satisfied(layout, moved, group))
            after_score = _progress_score(layout, moved, group, objective)
            if not satisfies and after_score >= before_score:
                continue
            if _overlaps(fragment.mask, destination, layout.walls):
                kind = "wall"
                label = "wall_destination_block"
            elif not _hazard_free(layout, moved):
                kind = "hazard"
                label = "hazard_core_rollback"
            else:
                continue
            alternative = (names.ACTION_CLICK, click_x, click_y)
            if kind in found or not _native_constraint_rollback(env, alternative, kind):
                continue
            found[kind] = {
                "kind": label,
                "route_action_index": int(route_action_index),
                "alternative": list(alternative),
                "progress_before": int(before_score),
                "progress_after": int(after_score),
                "goal_satisfying": satisfies,
                "native_rollback": True,
            }
            if len(found) == (2 if layout.hazards else 1):
                return [found[key] for key in ("wall", "hazard") if key in found]
    return [found[key] for key in ("wall", "hazard") if key in found]


def _pickup_key(sprite):
    return sprite.name, int(sprite.x), int(sprite.y)


def _solution_trace(env, actions, spec):
    start_index = env.level_index
    start_score = env.levels_completed
    selections = drags = wall_blocks = hazard_hits = 0
    removals = []
    core_colours = {}
    replayed_actions = 0
    constraint_witnesses = []
    for action in actions:
        if env.level_index != start_index:
            break
        for name, data in env.groups().items():
            core = data[names.KEY_CORE]
            if core is not None and core.name.startswith(names.PREFIX_ABSORBING_CORE):
                core_colours[name] = sorted(int(value) for value in np.unique(core.pixels) if value > 0)
        fragments = env.fragments()
        selected = env.selected()
        _, x, y = action
        hit = next(
            (fragment for fragment in fragments
             if fragment.x <= x < fragment.x + fragment.width
             and fragment.y <= y < fragment.y + fragment.height),
            None,
        )
        before_pickups = {_pickup_key(pickup) for pickup in env.pickups()}
        before_hazards = env.hazards_hit()
        before_level = env.level_index
        if hit is not None:
            selections += 1
        elif selected is not None:
            drags += 1
            constraint_witnesses.extend(
                _route_constraint_witnesses(env, action, replayed_actions)
            )
        observation = env.perform(*action)
        replayed_actions += 1
        if observation.state == GameState.GAME_OVER:
            return None
        if env.level_index == before_level:
            after_pickups = {_pickup_key(pickup) for pickup in env.pickups()}
            removals.extend(sorted(before_pickups - after_pickups))
            hazard_hits += max(0, env.hazards_hit() - before_hazards)
            if hit is None and selected is not None:
                expected = names.click_to_position(x, y)
                if (selected.x, selected.y) != expected and env.hazards_hit() == before_hazards:
                    wall_blocks += 1
    completed = env.levels_completed == start_score + 1
    if not completed:
        return None
    useful_positions = {
        (names.PREFIX_PICKUP + str(pickup["prototype"]), *map(int, pickup["position"]))
        for pickup in spec.get("pickups", []) if pickup.get("role") == "useful"
    }
    removed = set(removals)
    useful_absorbed = len(removed & useful_positions)
    decoy_absorbed = len(removed - useful_positions)
    mechanics = {
        "selection_actions": selections,
        "drag_actions": drags,
        "wall_blocked_actions": wall_blocks,
        "hazard_hits": hazard_hits,
        "hazard_constrained_witness_drags": sum(
            witness["kind"] == "hazard_core_rollback" for witness in constraint_witnesses
        ),
        "wall_constrained_witness_drags": sum(
            witness["kind"] == "wall_destination_block" for witness in constraint_witnesses
        ),
        "route_constraint_witnesses": constraint_witnesses,
        "pickups_absorbed": len(removals),
        "useful_pickups_absorbed": useful_absorbed,
        "decoy_pickups_absorbed": decoy_absorbed,
        "absorber_colour_sets": core_colours,
        "exact_colour_targets_satisfied": len(spec.get("targets", [])) if completed else 0,
        "required_targets_satisfied": (
            len(spec.get("groups", [])) if spec["mode"] == "ordinary-centroid"
            else len(spec.get("targets", []))
        ),
        "visual_decoys_ignored": len(spec.get("decoys", [])),
        "replayed_actions": replayed_actions,
    }
    replay_data = {
        "context_index": spec["difficulty"] - 1,
        "start_level_index": start_index,
        "score_delta": env.levels_completed - start_score,
        "final_level_index": env.level_index,
        "final_state": env.state.value,
        "forced_transitions": 0,
    }
    return mechanics, replay_data


def _assignment_errors(spec):
    errors = []
    targets = spec.get("targets", [])
    target_sets = []
    for index, target in enumerate(targets):
        if type(target.get("target_index")) is not int or target["target_index"] != index:
            errors.append("target indices must be exact ordered integers")
            continue
        actual = _prototype_colours(names.PREFIX_TARGET + str(target["prototype"]))
        declared = target.get("colours")
        if (
            not isinstance(declared, list)
            or any(type(value) is not int for value in declared)
            or frozenset(declared) != actual
            or len(declared) != len(actual)
        ):
            errors.append("declared target colours differ from the native pixel pattern")
        target_sets.append(actual)
    for pickup in spec.get("pickups", []):
        actual = _prototype_colours(names.PREFIX_PICKUP + str(pickup["prototype"]))
        if len(actual) != 1 or type(pickup.get("colour")) is not int or {pickup["colour"]} != actual:
            errors.append("declared pickup colour differs from the native pixel pattern")
            continue
        role = pickup.get("role")
        target_index = pickup.get("target_index")
        compatible = [index for index, colours in enumerate(target_sets) if actual <= colours]
        if role == "useful":
            if type(target_index) is not int or target_index not in compatible:
                errors.append("useful pickup assignment differs from native target colours")
        elif role == "decoy":
            if target_index is not None or compatible:
                errors.append("decoy pickup is compatible with a native target")
        else:
            errors.append("pickup role must be useful or decoy")
    return errors


def profile_errors(spec, *, require_proof=True):
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in REFERENCE_PROFILES:
        return ["difficulty must be an integer in 1..6"]
    profile = REFERENCE_PROFILES[difficulty]
    if spec.get("quality_profile_version") != QUALITY_PROFILE_VERSION:
        errors.append("quality profile version differs from the measured reference")
    try:
        metrics = structural_metrics(spec)
        exact_fields = (
            "fragment_counts", "required_targets", "visual_targets", "visual_decoys",
            "absorbers", "pickups", "useful_pickups",
        )
        for field in exact_fields:
            expected = list(profile[field]) if field == "fragment_counts" else profile[field]
            if metrics[field] != expected:
                errors.append(f"{field} differs from reference tier")
        errors.extend(_assignment_errors(spec))
        wall_low, wall_high = profile["wall_range"]
        if not wall_low <= metrics["wall_pixels"] <= wall_high:
            errors.append("wall pixel density outside explicit reference band")
        hazard_low, hazard_high = profile["hazard_range"]
        if not hazard_low <= metrics["hazard_pixels"] <= hazard_high:
            errors.append("hazard pixel density outside explicit reference band")
        visual_low, visual_high = profile["occupied_visual_range"]
        if not visual_low <= metrics["occupied_visual_pixels"] <= visual_high:
            errors.append("initial visual density outside explicit reference band")
        if difficulty == 4 and metrics["multicolour_targets"] != 3:
            errors.append("tier 4 must retain all three multicolour targets")
        if require_proof:
            low, high = profile["action_range"]
            if not low <= spec.get("solution_length", -1) <= high:
                errors.append("constructive action length outside explicit reference band")
            if spec.get("max_actions") != names.MAX_ACTIONS or spec.get("usable_actions") != names.USABLE_ACTIONS:
                errors.append("native 60-action/59-usable budget is missing")
            if spec.get("context_index") != difficulty - 1:
                errors.append("native context index differs from official tier")
            if not spec.get("engine_verified"):
                errors.append("native positive replay certificate missing")
            replay_data = spec.get("native_replay", {})
            if replay_data.get("context_index") != difficulty - 1 or replay_data.get("score_delta") != 1:
                errors.append("native context replay did not complete exactly one level")
            mechanics = spec.get("solution_mechanics", {})
            if mechanics.get("required_targets_satisfied") != profile["required_targets"]:
                errors.append("not all required target mechanics were certified")
            drags = mechanics.get("drag_actions", -1)
            wall_drags = mechanics.get("wall_constrained_witness_drags", -1)
            hazard_drags = mechanics.get("hazard_constrained_witness_drags", -1)
            if (
                type(drags) is not int or type(wall_drags) is not int
                or not profile["min_wall_constrained_drags"] <= wall_drags <= drags
            ):
                errors.append("walls lack route-relevant native witness constraints")
            if (
                type(hazard_drags) is not int
                or not profile["min_hazard_constrained_drags"] <= hazard_drags <= drags
            ):
                errors.append("hazards lack route-relevant native rollback constraints")
            witnesses = mechanics.get("route_constraint_witnesses")
            if (
                not isinstance(witnesses, list)
                or any(
                    not isinstance(witness, Mapping)
                    or witness.get("native_rollback") is not True
                    or witness.get("kind") not in ("wall_destination_block", "hazard_core_rollback")
                    for witness in witnesses
                )
            ):
                errors.append("native route-constraint witness details are malformed")
            if difficulty >= 5:
                if mechanics.get("useful_pickups_absorbed") != profile["useful_pickups"]:
                    errors.append("not every required pickup was absorbed")
                if mechanics.get("decoy_pickups_absorbed") != 0:
                    errors.append("winning witness polluted a core with a decoy pickup")
                if mechanics.get("exact_colour_targets_satisfied") != profile["required_targets"]:
                    errors.append("colour-set target equality was not certified")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"malformed spec: {error}")
    return errors


def verify(spec, limit=DEFAULT_LIMIT):
    """Return ``(accepted, reason)`` after symbolic construction and native replay."""
    structural = profile_errors(spec, require_proof=False)
    if structural:
        return None, "profile_structure:" + structural[0]
    difficulty = spec["difficulty"]
    levels = official_levels()
    levels[difficulty - 1] = build_level(spec)
    planning_env = Env(levels)
    planning_env.reset()
    planning_env.set_level(difficulty - 1)
    result = search(planning_env, limit=limit)
    if not result.solved or result.actions is None:
        reason = "search_truncated" if result.truncated else "search_unsupported" if result.unsupported else "search_failed"
        return None, reason
    replay_env = Env(levels)
    replay_env.reset()
    replay_env.set_level(difficulty - 1)
    traced = _solution_trace(replay_env, result.actions, spec)
    if traced is None:
        return None, "native_replay_failed"
    mechanics, replay_data = traced
    metrics = structural_metrics(spec)
    geometry, gameplay = canonical_identity(spec)
    accepted = dict(spec)
    accepted.update({
        "source": "generated_only",
        "reference_level": difficulty,
        "reference_teacher_actions": REFERENCE_PROFILES[difficulty]["reference_actions"],
        "structural_metrics": metrics,
        "solution": [list(action) for action in result.actions],
        "solution_length": len(result.actions),
        "solution_mechanics": mechanics,
        "search_expanded": result.expanded,
        "search_generated": result.generated,
        "search_limit": int(limit),
        "search_truncated": result.truncated,
        "search_exact_positive": result.exact,
        "engine_verified": True,
        "native_replay": replay_data,
        "max_actions": names.MAX_ACTIONS,
        "usable_actions": names.USABLE_ACTIONS,
        "remaining_usable_actions": names.USABLE_ACTIONS - len(result.actions),
        "geometry_sha256": geometry,
        "geometry_d4_sha256": geometry,
        "gameplay_sha256": gameplay,
        "geometry_version": GEOMETRY_VERSION,
    })
    errors = profile_errors(accepted)
    if errors:
        return None, "profile_proof:" + errors[0]
    return accepted, None


def _validate_request(seed, difficulty, attempts, limit, split):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be an integer in 1..6")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= 32_000_000:
        raise ValueError("limit must be an integer in 1..32,000,000")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")


def generate(
    seed,
    difficulty=1,
    attempts=MAX_ATTEMPTS,
    limit=DEFAULT_LIMIT,
    *,
    split,
    record_rejection=None,
):
    """Generate one deterministic full-standard level in an explicit split."""
    _validate_request(seed, difficulty, attempts, limit, split)
    rng = random.Random(f"{MECHANICS_INVENTORY_VERSION}:{int(seed)}:{difficulty}")
    exclusions = Counter()
    for attempt in range(1, attempts + 1):
        spec = _draft(rng, int(seed), difficulty, attempt)
        geometry, gameplay = canonical_identity(spec)
        partition = SPLITS[int(geometry, 16) % len(SPLITS)]
        if partition != split:
            reason = "geometry_split"
            exclusions[reason] += 1
        elif geometry in official_geometry_hashes():
            reason = "official_geometry_copy"
            exclusions[reason] += 1
        else:
            spec.update({
                "split": split,
                "geometry_split": partition,
                "geometry_sha256": geometry,
                "geometry_d4_sha256": geometry,
                "gameplay_sha256": gameplay,
                "geometry_version": GEOMETRY_VERSION,
            })
            accepted, reason = verify(spec, limit=limit)
            if accepted is not None:
                accepted["generation_exclusions"] = dict(exclusions)
                accepted["generation_attempts_used"] = attempt
                generate.last_report = {
                    "accepted": True, "attempts": attempt,
                    "rejections": dict(exclusions), "reason": None,
                }
                return accepted
            exclusions[reason] += 1
        if record_rejection is not None:
            record_rejection({
                "seed": int(seed), "difficulty": difficulty, "attempt": attempt,
                "split": split, "reason": reason,
                "generator_version": GENERATOR_VERSION,
            })
    generate.last_report = {
        "accepted": False, "attempts": attempts,
        "rejections": dict(exclusions), "reason": "attempts_exhausted",
    }
    return None


generate.last_report = None


def _child_seed(game_seed, ordinal, difficulty):
    payload = f"r11l:{int(game_seed)}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")


def generate_game(
    seed,
    *,
    split,
    difficulties=None,
    attempts=MAX_ATTEMPTS,
    limit=DEFAULT_LIMIT,
    record_rejection=None,
):
    """Generate an increasing-difficulty sequence; defaults to all six tiers."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    chosen = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not chosen or any(type(value) is not int or value not in DIFFICULTIES for value in chosen):
        raise ValueError("difficulties must be a nonempty sequence drawn from 1..6")
    if tuple(sorted(set(chosen))) != chosen:
        raise ValueError("difficulties must be strictly increasing and unique")
    specs = []
    for ordinal, difficulty in enumerate(chosen):
        child_seed = _child_seed(seed, ordinal, difficulty)
        spec = generate(
            child_seed, difficulty, attempts=attempts, limit=limit,
            split=split, record_rejection=record_rejection,
        )
        if spec is None:
            return None
        spec.update({
            "game_seed": int(seed), "game_ordinal": ordinal,
            "game_child_seed": child_seed,
        })
        specs.append(spec)
    return specs


def _validated_actions(spec):
    solution = spec.get("solution")
    if not isinstance(solution, list) or not solution:
        raise ValueError("solution must be a nonempty list")
    actions = []
    for action in solution:
        if (
            not isinstance(action, list)
            or len(action) != 3
            or any(type(value) is not int for value in action)
            or action[0] != names.ACTION_CLICK
            or not 0 <= action[1] < names.FRAME_SIZE
            or not 0 <= action[2] < names.FRAME_SIZE
        ):
            raise ValueError("malformed stored ACTION6 route")
        actions.append(tuple(action))
    return tuple(actions)


def validate_full_standard(spec, curriculum_entry):
    """Return all family-local contract/profile errors for one accepted row."""
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    errors = []
    difficulty = spec.get("difficulty")
    if spec.get("format") != FORMAT:
        errors.append("level format is not the full R11L format")
    if spec.get("generator_version") != GENERATOR_VERSION:
        errors.append("generator version mismatch")
    if spec.get("mechanics_inventory_version") != MECHANICS_INVENTORY_VERSION:
        errors.append("mechanics inventory version mismatch")
    if spec.get("quality_profile_version") != QUALITY_PROFILE_VERSION:
        errors.append("quality profile version mismatch")
    if spec.get("source") != "generated_only":
        errors.append("source must be generated_only")
    if spec.get("split") not in SPLITS:
        errors.append("explicit split missing")
    try:
        if type(difficulty) is not int or difficulty not in DIFFICULTIES:
            raise ValueError("invalid difficulty")
        expected = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        if not isinstance(curriculum_entry, Mapping):
            raise ValueError("curriculum entry must be a mapping")
        curriculum = dict(curriculum_entry)
        if any(type(curriculum.get(field)) is not int for field in ("difficulty", "context_index", "search_work")):
            errors.append("curriculum difficulty, context_index, and search_work must be strict integers")
        if curriculum != expected:
            errors.append("curriculum entry differs from contract")
        if type(spec.get("context_index")) is not int or spec["context_index"] != expected["context_index"]:
            errors.append("context index differs from curriculum")
        search_limit = spec.get("search_limit")
        if type(search_limit) is not int or not 1 <= search_limit <= expected["search_work"]:
            errors.append("search work is outside curriculum bound")
    except (IndexError, KeyError, TypeError, ValueError):
        errors.append("invalid curriculum/difficulty relation")
    try:
        geometry, gameplay = canonical_identity(spec)
        if spec.get("geometry_d4_sha256") != geometry or spec.get("geometry_sha256") != geometry:
            errors.append("geometry identity mismatch")
        if spec.get("gameplay_sha256") != gameplay:
            errors.append("gameplay identity mismatch")
        if geometry_partition(spec)[1] != spec.get("split"):
            errors.append("geometry is assigned to another split")
        if geometry in official_geometry_hashes():
            errors.append("geometry duplicates a shipped official level")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid canonical identity: {error}")
    try:
        measured = structural_metrics(spec)
        if spec.get("structural_metrics") != measured:
            errors.append("stored structural metrics differ from rebuilt native level")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"could not rebuild structural metrics: {error}")
    try:
        actions = _validated_actions(spec)
        if (isinstance(spec.get("solution_length"), bool)
                or spec.get("solution_length") != len(actions)):
            errors.append("solution length metadata differs from stored route")
        if spec.get("remaining_usable_actions") != names.USABLE_ACTIONS - len(actions):
            errors.append("remaining native budget differs from stored route")
        levels = official_levels()
        levels[difficulty - 1] = build_level(spec)
        replay_env = Env(levels)
        replay_env.reset()
        replay_env.set_level(difficulty - 1)
        traced = _solution_trace(replay_env, tuple(actions), spec)
        if traced is None:
            errors.append("stored route fails native context replay")
        else:
            mechanics, replay_data = traced
            if mechanics.get("replayed_actions") != len(actions):
                errors.append("stored route contains actions after native completion")
            if spec.get("solution_mechanics") != mechanics:
                errors.append("stored solution mechanics differ from replayed events")
            if spec.get("native_replay") != replay_data:
                errors.append("stored native replay summary differs from replay")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"could not validate stored route: {error}")
    try:
        for field in ("search_expanded", "search_generated", "search_limit"):
            value = spec.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{field} must be a nonnegative integer")
        if spec.get("search_truncated") is not False or spec.get("search_exact_positive") is not True:
            errors.append("positive bounded-search flags are invalid")
    except TypeError as error:
        errors.append(f"malformed bounded-search metadata: {error}")
    try:
        errors.extend(profile_errors(spec))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"could not validate reference profile: {error}")
    return errors


def _replay_full_game(specs, levels):
    """Replay all stored witnesses through native automatic progression."""
    env = Env(levels)
    env.reset()
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index or env.state != GameState.NOT_FINISHED:
            raise ValueError(f"sequential replay entered tier {index + 1} in an invalid native state")
        actions = _validated_actions(spec)
        for action_index, action in enumerate(actions):
            before_score = env.levels_completed
            outcome = env.perform(*action)
            if outcome.state == GameState.GAME_OVER:
                raise ValueError(f"sequential replay lost on tier {index + 1}")
            if env.levels_completed > before_score and action_index != len(actions) - 1:
                raise ValueError(f"tier {index + 1} stored route has actions after native completion")
        if env.levels_completed != index + 1:
            raise ValueError(f"sequential replay did not complete tier {index + 1}")
        if index + 1 < len(specs):
            if env.level_index != index + 1 or env.state != GameState.NOT_FINISHED:
                raise ValueError(f"sequential replay did not advance naturally after tier {index + 1}")
    if env.state != GameState.WIN or env.levels_completed != len(specs):
        raise ValueError("sequential replay did not finish the complete six-tier game")


def build_game(specs):
    """Validate and build exactly one native six-level increasing game list."""
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
        raise ValueError("full R11L games require a sequence of level mappings")
    specs = list(specs)
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("full R11L games require exactly six levels")
    if any(not isinstance(spec, Mapping) for spec in specs):
        raise ValueError("every full R11L level spec must be a mapping")
    if [spec.get("difficulty") for spec in specs] != list(DIFFICULTIES):
        raise ValueError("full R11L games must be ordered difficulties 1..6")
    splits = {spec.get("split") for spec in specs}
    if len(splits) != 1 or None in splits:
        raise ValueError("full R11L games require one explicit common split")
    identities = [spec.get("gameplay_sha256") for spec in specs]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("full R11L games require unique gameplay identities")
    for index, spec in enumerate(specs):
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"tier {index + 1} is not full-standard: {errors[0]}")
    levels = [build_level(spec) for spec in specs]
    _replay_full_game(specs, levels)
    return levels
