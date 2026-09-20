"""Full six-tier SP80 generator with native-engine replay certificates.

Each accepted level is built from a tier-specific water-routing topology, is
scrambled into a legal editing puzzle, and is accepted only after its public
action witness wins in the unmodified vendored engine.  The six difficulties
follow the six reference levels in order; higher tiers introduce cascaded
splitters, multiple sources, embedded sources, side cups, deflectors, a
vertical pipe, and three-edge sink hazards.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import copy
import functools
import hashlib
import json
import random
from collections.abc import Mapping, Sequence

from arcengine import ActionInput, GameAction, GameState, Level

from . import names
from .env import Env, official_levels, replay, upstream
from .plan import DEFAULT_LIMIT, TARGET_DATA_KEY, search


FORMAT = "pebby.sp80.level.v3"
GENERATOR_VERSION = 3
SOURCE_ID = "sp80-589a99af"
DIFFICULTIES = tuple(range(1, 7))
SPLITS = ("train", "validation", "test")
MAX_ATTEMPTS = 24

_LAST_LEVEL_GENERATION_REPORT = ContextVar("sp80_last_level_generation_report", default=None)
_LAST_GAME_GENERATION_REPORT = ContextVar("sp80_last_game_generation_report", default=None)

# Exact structural anchors come from the six immutable reference levels.
# Solution ranges admit procedural geometry while keeping native action demand
# near the audited reference teacher lengths (4, 18, 32, 49, 42, 43).
REFERENCE_PROFILES = {
    1: {"size": 16, "steps": 30, "rotation": 0, "pipes": 1, "sources": 1,
        "cups": 2, "sinks": 1, "reference_teacher_length": 4,
        "solution_range": (4, 7), "flow_frames": (12, 30)},
    2: {"size": 16, "steps": 45, "rotation": 180, "pipes": 3, "sources": 1,
        "cups": 3, "sinks": 1, "reference_teacher_length": 18,
        "solution_range": (14, 23), "flow_frames": (14, 35)},
    3: {"size": 16, "steps": 100, "rotation": 180, "pipes": 4, "sources": 3,
        "cups": 3, "sinks": 1, "reference_teacher_length": 32,
        "solution_range": (26, 38), "flow_frames": (13, 38)},
    4: {"size": 20, "steps": 120, "rotation": 0, "pipes": 5, "sources": 1,
        "cups": 4, "sinks": 1, "reference_teacher_length": 49,
        "solution_range": (41, 57), "flow_frames": (17, 45)},
    5: {"size": 20, "steps": 100, "rotation": 180, "pipes": 3, "sources": 2,
        "cups": 4, "sinks": 3, "reference_teacher_length": 42,
        "solution_range": (36, 51), "flow_frames": (16, 45)},
    6: {"size": 20, "steps": 120, "rotation": 0, "pipes": 2, "sources": 1,
        "cups": 4, "sinks": 3, "reference_teacher_length": 43,
        "solution_range": (35, 53), "flow_frames": (17, 50)},
}

CURRICULUM = tuple(
    {"difficulty": difficulty, "context_index": difficulty - 1, "search_work": 80_000}
    for difficulty in DIFFICULTIES
)

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": "sp80-mechanics-source-audit-v4",
    "quality_profile_version": "sp80-reference-profile-v3",
    "curriculum": CURRICULUM,
    "evidence": {
        "official_tier_characterization": "sp80-reference-six-tier-audit-v3",
        "solution_mechanics": "sp80-native-flow-event-and-ablation-certificate-v4",
        "native_budget": "sp80-native-action-budget-audit-v3",
        "context_engine_replay": "sp80-six-level-context-replay-v3",
        "novelty_split": "sp80-semantic-sha256-and-d4-partition-v3",
        "bounded_rejections": "sp80-typed-bounded-rejections-v3",
    },
    "caveats": [
        "Root accepted the tier-4 connected source-pipe mechanics after independent audit.",
        "Positive teachers are native-replayed witnesses; shortest-path optimality is not claimed.",
        "Bounded nearer-parking comparators shorten official tiers 4/5 to 45/26 actions; witness length is not objective hardness.",
        "Default 24-attempt generation can exhaust tier-3 seed 906 and whole-game seed 888/train.",
        "Tier 1 has finite tutorial diversity and produced 7/8 distinct D4 identities in root sampling.",
        "Negative fallback search remains bounded and reports truncation or unsupported engine cycles.",
    ],
}

_KIND_TO_NAME = {
    "pipe_h": lambda p: names.PIPE[p["length"]],
    "pipe_v": lambda p: names.PIPE_VERTICAL_4,
    "source_pipe": lambda p: names.SOURCE_PIPE[p["length"]],
    "deflector_left": lambda p: names.DEFLECTOR_LEFT,
    "deflector_right": lambda p: names.DEFLECTOR_RIGHT,
}


def _int(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _piece(kind, length, target, role, initial=None):
    item = {"kind": kind, "x": target[0], "y": target[1],
            "target_x": target[0], "target_y": target[1], "role": role}
    if length is not None:
        item["length"] = length
    if initial is not None:
        item["x"], item["y"] = initial
    return item


def _cup(x, y, rotation=0):
    return {"x": x, "y": y, "rotation": rotation}


def _sink(x, y, rotation=0):
    return {"x": x, "y": y, "rotation": rotation}


def _bottom_and_side_sinks(size):
    return [_sink(0, size - 1), _sink(size - 1, 0, 90), _sink(-1, 0, 90)]


def _base_spec(seed, difficulty, split, attempt, pieces, sources, cups, sinks, topology):
    profile = REFERENCE_PROFILES[difficulty]
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "source_id": SOURCE_ID,
        "seed": int(seed),
        "attempt": int(attempt),
        "split": split,
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "grid_size": profile["size"],
        "rotation": profile["rotation"],
        "steps": profile["steps"],
        "topology": topology,
        "pieces": pieces,
        "sources": list(sources),
        "cups": cups,
        "sinks": sinks,
    }


def _draft_tier(rng, seed, difficulty, split, attempt):
    if difficulty == 1:
        length = rng.choice((3, 4, 5, 6))
        target_x = rng.randint(3, 16 - length - 3)
        target_y = rng.randint(4, 9)
        source = rng.randint(target_x, target_x + length - 1)
        pieces = [_piece("pipe_h", length, (target_x, target_y), "primary-splitter")]
        cups = [_cup(target_x - 2, 13), _cup(target_x + length - 1, 13)]
        return _base_spec(seed, 1, split, attempt, pieces, [source], cups,
                          [_sink(0, 15)], "single-splitter-variable-span")

    if difficulty == 2:
        dx = rng.choice((-1, 0, 1))
        top_y = rng.randint(4, 6)
        lower_y = rng.randint(top_y + 3, 10)
        pieces = [
            _piece("pipe_h", 3, (7 + dx, top_y), "cascade-root"),
            _piece("pipe_h", 3, (10 + dx, lower_y), "cascade-child"),
            _piece("pipe_h", 5, (0, 10), "movable-flow-blocker",
                   (4 + dx, rng.randint(10, 11))),
        ]
        cups = [_cup(5 + dx, 13), _cup(8 + dx, 13), _cup(12 + dx, 13)]
        return _base_spec(seed, 2, split, attempt, pieces, [8 + dx], cups,
                          [_sink(0, 15)], "two-stage-cascade-with-blocker")

    if difficulty == 3:
        dx = rng.choice((0, 1))
        root_y = rng.randint(5, 7)
        child_y = rng.randint(root_y + 2, 10)
        pieces = [
            _piece("pipe_h", 4, (8 + dx, child_y), "merged-source-child"),
            _piece("pipe_h", 5, (2 + dx, 10), "left-output-blocker", (0, 10)),
            _piece("pipe_h", 6, (2 + dx, root_y), "shared-source-root"),
            _piece("pipe_h", 6, (9, 4), "right-output-blocker", (9, 10)),
        ]
        cups = [_cup(0 + dx, 13), _cup(6 + dx, 13), _cup(11 + dx, 13)]
        return _base_spec(seed, 3, split, attempt, pieces,
                          [4 + dx, 6 + dx, 8 + dx], cups, [_sink(0, 15)],
                          "multi-source-merge-and-cascade")

    if difficulty == 4:
        root_x = rng.choice((3, 4, 5))
        root_y = rng.randint(3, 4)
        cascade_y = root_y + 3
        source_y = cascade_y + 3
        lower_y = source_y + 2
        pieces = [
            _piece("pipe_h", 4, (root_x, root_y), "top-source-splitter"),
            _piece("pipe_h", 4, (root_x, 14), "source-emission-blocker",
                   (root_x + 8, min(14, source_y + 4))),
            _piece("pipe_h", 5, (root_x + 1, cascade_y),
                   "source-incoming-cascade"),
            _piece("pipe_h", 5, (root_x, lower_y), "left-output-splitter"),
            _piece("source_pipe", 7, (root_x + 6, source_y),
                   "incoming-emitting-source-splitter"),
        ]
        cups = [_cup(root_x - 2, 17), _cup(root_x + 4, 17),
                _cup(root_x + 8, 17), _cup(root_x + 12, 17)]
        return _base_spec(
            seed, 4, split, attempt, pieces, [root_x], cups, [_sink(0, 19)],
            "connected-incoming-emitting-source-network",
        )

    if difficulty == 5:
        left_x = rng.choice((3, 4))
        deflector_x = rng.choice((12, 13, 14))
        flow_y = rng.randint(7, 10)
        pieces = [
            _piece("pipe_h", 3, (left_x, flow_y), "bottom-pair-splitter"),
            _piece("pipe_h", 4, (7, 3), "left-flow-blocker", (0, 12)),
            _piece("pipe_h", 5, (7, 4), "right-flow-blocker", (5, 13)),
            _piece("deflector_right", None, (deflector_x, flow_y), "side-cup-turn"),
        ]
        # The side cup ends before the column-19 sink, as in the reference level.
        side_cup_x = min(deflector_x + 4, 20 - 1 - _cup_size(270)[0])
        cups = [_cup(left_x - 2, 17), _cup(left_x + 2, 17),
                _cup(deflector_x - 2, 17), _cup(side_cup_x, flow_y - 1, 270)]
        return _base_spec(seed, 5, split, attempt, pieces,
                          [left_x + 1, deflector_x], cups, _bottom_and_side_sinks(20),
                          "splitter-plus-branching-deflector")

    bottom_source = rng.choice((7, 8, 9, 10))
    emitter_x = rng.choice((13, 14))
    turn_y = rng.choice((9, 10, 11))
    vertical_x = rng.choice((4, 5)) if bottom_source == 7 else rng.choice((4, 5, 6))
    pieces = [
        _piece("pipe_v", 4, (vertical_x, turn_y - 2), "vertical-side-splitter"),
        _piece("source_pipe", 5, (emitter_x - 2, turn_y - 5), "embedded-source-emitter"),
        _piece("deflector_left", None, (emitter_x - 2, turn_y + 1), "left-turn"),
        _piece("deflector_right", None, (emitter_x, turn_y), "right-turn"),
    ]
    cups = [_cup(emitter_x + 3, turn_y - 1, 270), _cup(bottom_source - 1, 17),
            _cup(1, turn_y + 1, 90), _cup(1, turn_y - 4, 90)]
    return _base_spec(seed, 6, split, attempt, pieces, [bottom_source], cups,
                      _bottom_and_side_sinks(20),
                      "embedded-source-double-turn-vertical-split")


def _prototype_name(piece):
    try:
        return _KIND_TO_NAME[piece["kind"]](piece)
    except (KeyError, TypeError):
        raise ValueError(f"unsupported piece description {piece!r}") from None


def _sprite_size(piece):
    sprite = upstream().sprites[_prototype_name(piece)]
    return int(sprite.width), int(sprite.height)


def _cup_size(rotation):
    sprite = upstream().sprites[names.CUP].clone().set_rotation(rotation)
    return int(sprite.width), int(sprite.height)


@functools.lru_cache(maxsize=None)
def _sink_cells_for(size, sinks):
    """Grid cells the edge sinks paint, from the native sprite's own geometry.

    Sinks are drawn last, so anything placed on these cells is overdrawn.  The
    official levels keep every piece, cup and source clear of them.
    """
    prototype = upstream().sprites[names.SINK]
    cells = set()
    for x, y, rotation in sinks:
        if rotation not in (0, 90, 180, 270):
            raise ValueError("sink rotation must be a quarter turn")
        sprite = prototype.clone().set_position(x, y).set_rotation(rotation)
        pixels = sprite.render()
        for row in range(pixels.shape[0]):
            for column in range(pixels.shape[1]):
                cell_x, cell_y = int(sprite.x) + column, int(sprite.y) + row
                if int(pixels[row, column]) >= 0 and 0 <= cell_x < size and 0 <= cell_y < size:
                    cells.add((cell_x, cell_y))
    return frozenset(cells)


def _sink_cells(spec):
    sinks = tuple(
        (_int(sink.get("x"), "sink x"), _int(sink.get("y"), "sink y"),
         _int(sink.get("rotation", 0), "sink rotation"))
        for sink in spec["sinks"]
    )
    return _sink_cells_for(_int(spec["grid_size"], "grid_size"), sinks)


def _overlaps_sinks(spec, x, y, width, height):
    cells = _sink_cells(spec)
    return any((cell_x, cell_y) in cells
               for cell_y in range(y, y + height) for cell_x in range(x, x + width))


def _position_allowed(spec, piece, x, y):
    width, height = _sprite_size(piece)
    size = spec["grid_size"]
    if x < 0 or x + width > size or y < names.MIN_PIECE_ROW or y + height > size:
        return False
    if _overlaps_sinks(spec, x, y, width, height):
        return False
    for cup in spec["cups"]:
        cup_width, cup_height = _cup_size(cup["rotation"])
        if (x < cup["x"] + cup_width + 1 and x + width > cup["x"] - 1
                and y < cup["y"] + cup_height + 1 and y + height > cup["y"] - 1):
            return False
    return True


def _scramble(rng, spec):
    """Move active pieces off-target; preserve deliberate blocker starts."""
    active = [i for i, piece in enumerate(spec["pieces"])
              if "blocker" not in piece["role"]]
    distance_ranges = {
        1: (3, 5),
        2: (3, 5),
        3: (6, 9),
        4: (6, 9),
        5: (4, 7),
        6: (7, 10),
    }
    minimum, maximum = distance_ranges[spec["difficulty"]]
    for index in active:
        piece = spec["pieces"][index]
        tx, ty = piece["target_x"], piece["target_y"]
        candidates = []
        width, height = _sprite_size(piece)
        for y in range(3, spec["grid_size"] - height + 1):
            for x in range(0, spec["grid_size"] - width + 1):
                distance = abs(x - tx) + abs(y - ty)
                if minimum <= distance <= maximum and _position_allowed(spec, piece, x, y):
                    candidates.append((x, y, distance))
        if not candidates:
            return False
        x, y, _ = rng.choice(candidates)
        piece["x"], piece["y"] = x, y
    return True


def _validate_structure(spec):
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a mapping")
    difficulty = _int(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    profile = REFERENCE_PROFILES[difficulty]
    if spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    if spec.get("source_id") != SOURCE_ID:
        raise ValueError("source_id is not the canonical SP80 source id")
    if spec.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("generator version differs from the audited format")
    if spec.get("split") not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    for key in ("seed", "attempt"):
        if _int(spec.get(key), key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("grid_size", "steps", "rotation", "context_index"):
        _int(spec.get(key), key)
    if spec["grid_size"] != profile["size"] or spec["steps"] != profile["steps"]:
        raise ValueError("grid size or native step budget differs from the reference tier")
    if spec["rotation"] != profile["rotation"] or spec["context_index"] != difficulty - 1:
        raise ValueError("rotation or context index differs from the ordered reference tier")
    collections = {}
    for key in ("pieces", "sources", "cups", "sinks"):
        value = spec.get(key)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a sequence")
        collections[key] = list(value)
    pieces = collections["pieces"]
    sources = collections["sources"]
    cups = collections["cups"]
    sinks = collections["sinks"]
    if any(not isinstance(piece, Mapping) for piece in pieces):
        raise ValueError("every piece must be a mapping")
    if any(not isinstance(row, Mapping) for row in cups + sinks):
        raise ValueError("every cup and sink must be a mapping")
    pipe_count = sum(piece.get("kind") in ("pipe_h", "pipe_v", "source_pipe") for piece in pieces)
    if pipe_count != profile["pipes"] or len(sources) != profile["sources"]:
        raise ValueError("piece or top-source count differs from the reference tier")
    if len(cups) != profile["cups"] or len(sinks) != profile["sinks"]:
        raise ValueError("cup or sink count differs from the reference tier")
    inventory = Counter(piece.get("kind") for piece in pieces)
    expected_inventory = {
        1: Counter({"pipe_h": 1}),
        2: Counter({"pipe_h": 3}),
        3: Counter({"pipe_h": 4}),
        4: Counter({"pipe_h": 4, "source_pipe": 1}),
        5: Counter({"pipe_h": 3, "deflector_right": 1}),
        6: Counter({"pipe_v": 1, "source_pipe": 1,
                    "deflector_left": 1, "deflector_right": 1}),
    }[difficulty]
    if inventory != expected_inventory:
        raise ValueError("movable mechanic inventory differs from the reference tier")
    horizontal_lengths = sorted(
        piece.get("length") for piece in pieces if piece.get("kind") == "pipe_h"
    )
    expected_lengths = {
        1: None,
        2: [3, 3, 5],
        3: [4, 5, 6, 6],
        4: [4, 4, 5, 5],
        5: [3, 4, 5],
        6: [],
    }[difficulty]
    if difficulty == 1:
        if horizontal_lengths[0] not in (3, 4, 5, 6):
            raise ValueError("tier-1 splitter length is outside its calibrated range")
    elif horizontal_lengths != expected_lengths:
        raise ValueError("horizontal pipe lengths differ from the reference tier")
    source_lengths = [piece.get("length") for piece in pieces
                      if piece.get("kind") == "source_pipe"]
    if source_lengths != ({4: [7], 6: [5]}.get(difficulty, [])):
        raise ValueError("embedded-source pipe length differs from the reference tier")
    saw_deflector = False
    for index, piece in enumerate(pieces):
        kind = piece.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"pieces[{index}].kind must be a string")
        if kind.startswith("deflector"):
            saw_deflector = True
        elif saw_deflector:
            raise ValueError("all pipes must precede deflectors in movable order")
        _prototype_name(piece)
        for prefix in ("", "target_"):
            x = _int(piece.get(prefix + "x"), f"pieces[{index}].{prefix}x")
            y = _int(piece.get(prefix + "y"), f"pieces[{index}].{prefix}y")
            if not _position_allowed(spec, piece, x, y):
                raise ValueError(f"pieces[{index}] {prefix or 'initial '}position is illegal")
    size = profile["size"]
    sink_cells = _sink_cells(spec)
    for index, source in enumerate(sources):
        x = _int(source, f"sources[{index}]")
        if not 0 <= x < size:
            raise ValueError("top source is outside the board")
        if (x, 0) in sink_cells or (x, 1) in sink_cells:
            raise ValueError("top source sits on an edge sink")
    for index, cup in enumerate(cups):
        rotation = _int(cup.get("rotation"), f"cups[{index}].rotation")
        if rotation not in (0, 90, 180, 270):
            raise ValueError("cup rotation must be a quarter turn")
        width, height = _cup_size(rotation)
        x, y = _int(cup.get("x"), "cup x"), _int(cup.get("y"), "cup y")
        if not (0 <= x and x + width <= size and 0 <= y and y + height <= size):
            raise ValueError("cup is outside the board")
        if _overlaps_sinks(spec, x, y, width, height):
            raise ValueError("cup overlaps an edge sink")
    cup_boxes = []
    for cup in cups:
        width, height = _cup_size(cup["rotation"])
        box = (cup["x"], cup["y"], cup["x"] + width, cup["y"] + height)
        if any(box[0] < other[2] and box[2] > other[0]
               and box[1] < other[3] and box[3] > other[1] for other in cup_boxes):
            raise ValueError("cups overlap")
        cup_boxes.append(box)
    sink_signature = []
    for index, sink in enumerate(sinks):
        x = _int(sink.get("x"), f"sinks[{index}].x")
        y = _int(sink.get("y"), f"sinks[{index}].y")
        rotation = _int(sink.get("rotation"), f"sinks[{index}].rotation")
        if rotation not in (0, 90):
            raise ValueError("sink rotation must match a horizontal or vertical edge")
        sink_signature.append((x, y, rotation))
    expected_sinks = ([(0, size - 1, 0)] if difficulty < 5 else
                      [(0, size - 1, 0), (size - 1, 0, 90), (-1, 0, 90)])
    if sink_signature != expected_sinks:
        raise ValueError("sink placement does not match the tier's edge hazards")
    return profile


def build_level(spec):
    """Rebuild an ARCEngine ``Level`` from a JSON-round-trippable spec."""
    _validate_structure(spec)
    module = upstream()
    prototypes = module.sprites
    size = spec["grid_size"]
    sprites = [prototypes[names.FRAME[size]].clone().set_position(-1, -1)]
    for source_x in spec["sources"]:
        sprites.append(prototypes[names.WATER].clone().set_position(source_x, 1))
    for piece in spec["pieces"]:
        sprites.append(
            prototypes[_prototype_name(piece)].clone().set_position(piece["x"], piece["y"])
        )
    for cup in spec["cups"]:
        sprites.append(
            prototypes[names.CUP].clone().set_position(cup["x"], cup["y"])
            .set_rotation(cup["rotation"])
        )
    for source_x in spec["sources"]:
        sprites.append(prototypes[names.SOURCE].clone().set_position(source_x, 0))
    for sink in spec["sinks"]:
        sprites.append(
            prototypes[names.SINK].clone().set_position(sink["x"], sink["y"])
            .set_rotation(sink["rotation"])
        )
    targets = [[piece["target_x"], piece["target_y"]] for piece in spec["pieces"]]
    return Level(
        sprites=sprites,
        grid_size=(size, size),
        data={names.KEY_STEPS: spec["steps"], names.KEY_ROTATION: spec["rotation"],
              TARGET_DATA_KEY: targets},
        name=f"generated-sp80-d{spec['difficulty']}-{spec['split']}-s{spec['seed']}",
    )


def _category(sprite):
    tags = set(sprite.tags)
    for tag, label in ((names.TAG_PIPE, "pipe"), (names.TAG_DEFLECTOR, "deflector"),
                       (names.TAG_CUP, "cup"), (names.TAG_WATER, "water"),
                       (names.TAG_SOURCE, "source"), (names.TAG_SINK, "sink")):
        if tag in tags:
            return label
    return "frame"


def _digest(value):
    payload = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _level_geometry_forms(level):
    """Visual object geometry under D4, excluding frame/HUD and duplicate water heads.

    Source sprites retain source placement in the identity.  The one-cell
    static water sprites directly below them are presentation duplicates, and
    the frame/HUD depends only on the separately encoded grid size.
    """
    size = int(level.grid_size[0])
    objects = []
    for sprite in level.get_sprites():
        if sprite.name in names.FRAME.values() or names.TAG_WATER in sprite.tags:
            continue
        pixels = sprite.render()
        points = tuple(
            (int(sprite.x + x), int(sprite.y + y), int(pixels[y, x]))
            for y in range(pixels.shape[0]) for x in range(pixels.shape[1])
            if int(pixels[y, x]) >= 0
        )
        objects.append((_category(sprite), points))

    def transform(x, y, code):
        if code >= 4:
            x = size - 1 - x
            code -= 4
        for _ in range(code):
            x, y = size - 1 - y, x
        return x, y

    forms = []
    for code in range(8):
        encoded = []
        for category, points in objects:
            encoded.append((category, tuple(sorted((*transform(x, y, code), color)
                                                    for x, y, color in points))))
        forms.append(json.dumps(
            {"grid_size": size, "objects": sorted(encoded)}, separators=(",", ":")
        ))
    return forms


def _level_gameplay_payload(level):
    """Actual initial native semantics, without private targets or certificates."""
    sprites = [sprite for sprite in level.get_sprites()
               if sprite.name not in names.FRAME.values()]
    movables = list(level.get_sprites_by_tag(names.TAG_PIPE))
    movables += list(level.get_sprites_by_tag(names.TAG_DEFLECTOR))
    selected = min(
        range(len(movables)),
        key=lambda index: movables[index].x ** 2 + movables[index].y ** 2,
        default=None,
    )

    def semantic(sprite):
        return [
            str(sprite.name), int(sprite.x), int(sprite.y),
            int(getattr(sprite, "rotation", 0)),
        ]

    return {
        "grid_size": [int(value) for value in level.grid_size],
        "native_budget": int(level.get_data(names.KEY_STEPS) or 50),
        "presentation_rotation": int(level.get_data(names.KEY_ROTATION) or 0),
        # Source/cup/sink overlap and first-hit behavior can depend on native
        # insertion order, so this is deliberately not D4- or order-normalized.
        "sprite_order": [semantic(sprite) for sprite in sprites],
        "movable_order": [semantic(sprite) for sprite in movables],
        "initial_selected_index": selected,
        "initial_failed_flows": 0,
        "initial_mode": "change",
    }


def _identities(spec):
    level = build_level(spec)
    forms = _level_geometry_forms(level)
    return _digest(forms[0]), _digest(min(forms)), _digest(_level_gameplay_payload(level))


def _geometry_partition(geometry_d4_sha256):
    return SPLITS[int(geometry_d4_sha256, 16) % len(SPLITS)]


def _official_identity_sets():
    exact, d4, gameplay = set(), set(), set()
    for level in official_levels():
        forms = _level_geometry_forms(level)
        exact.add(_digest(forms[0]))
        d4.add(_digest(min(forms)))
        gameplay.add(_digest(_level_gameplay_payload(level)))
    return frozenset(exact), frozenset(d4), frozenset(gameplay)


_OFFICIAL_GEOMETRY, _OFFICIAL_D4_GEOMETRY, _OFFICIAL_GAMEPLAY = _official_identity_sets()


def _trace_target(spec):
    """Derive mechanic-use events from native spill frames at target placement."""
    env = Env([build_level(spec)])
    env.reset()
    for sprite, piece in zip(env.movables(), spec["pieces"]):
        sprite.set_position(piece["target_x"], piece["target_y"])
    game = env.game
    movables = env.movables()
    index_for = {id(sprite): index for index, sprite in enumerate(movables)}
    source_pipe_indices = {
        index for index, piece in enumerate(spec["pieces"])
        if piece["kind"] == "source_pipe"
    }
    events = Counter()
    used = set()
    game._set_action(ActionInput(id=GameAction.ACTION5, data={}))
    previous_filled = 0
    for frame in range(1, 1001):
        mode = env.mode
        heads = list(getattr(game, names.ATTR_HEADS)) if mode == "spill" else []
        if mode == "spill":
            for water, dx, dy in heads:
                hit = env.level.get_sprite_at(water.x + dx, water.y + dy)
                if hit is None:
                    continue
                tags = set(getattr(hit, "tags", ()))
                if names.TAG_PIPE in tags:
                    index = index_for.get(id(hit))
                    if index is not None:
                        used.add(index)
                        if index in source_pipe_indices:
                            events["source_pipe_incoming_hits"] += 1
                    events["vertical_pipe_hits" if hit.height > hit.width else "horizontal_pipe_hits"] += 1
                elif names.TAG_DEFLECTOR in tags:
                    index = index_for.get(id(hit))
                    if index is not None:
                        used.add(index)
                    key = "deflector_left_hits" if hit.name == names.DEFLECTOR_LEFT else "deflector_right_hits"
                    events[key] += 1
                elif names.TAG_SINK in tags:
                    events["sink_contacts"] += 1
        game.step()
        filled = len(getattr(game, names.ATTR_FILLED_CUPS))
        if filled > previous_filled:
            events["cup_fill_events"] += filled - previous_filled
            previous_filled = filled
        if frame == 1:
            initial_water = len(spec["sources"])
            events["embedded_source_emissions"] = max(
                0, len(getattr(game, names.ATTR_HEADS)) - initial_water
            )
        if getattr(game, names.ATTR_FLOW_DONE):
            return {
                "flow_frames": frame,
                "cups_filled": filled,
                "sink_hit": bool(getattr(game, names.ATTR_SINK_HIT)),
                "used_piece_indices": sorted(used),
                "source_pipe_tag_ablation_prevents_win": (
                    _source_pipe_tag_ablation_prevents_win(spec)
                ),
                **{key: int(events[key]) for key in (
                    "horizontal_pipe_hits", "vertical_pipe_hits",
                    "source_pipe_incoming_hits", "embedded_source_emissions",
                    "deflector_left_hits", "deflector_right_hits",
                    "cup_fill_events", "sink_contacts")},
            }
    raise ValueError("target flow exceeded the explicit 1,000-frame certificate guard")


def _source_pipe_tag_ablation_prevents_win(spec):
    """Whether removing only the target source pipe's splitter role breaks the win."""
    source_pipe_indices = [
        index for index, piece in enumerate(spec["pieces"])
        if piece["kind"] == "source_pipe"
    ]
    if not source_pipe_indices:
        return False
    env = Env([build_level(spec)])
    env.reset()
    for sprite, piece in zip(env.movables(), spec["pieces"]):
        sprite.set_position(piece["target_x"], piece["target_y"])
    for index in source_pipe_indices:
        source_pipe = env.movables()[index]
        if names.TAG_SOURCE not in source_pipe.tags or names.TAG_PIPE not in source_pipe.tags:
            raise ValueError("source pipe does not expose both native source and pipe tags")
        source_pipe.tags.remove(names.TAG_PIPE)
    observation = env.perform(names.ACTION_FLOW)
    return observation.state != GameState.WIN and env.levels_completed == 0


def _mechanic_errors(spec, trace):
    difficulty = spec["difficulty"]
    errors = []
    if trace["cups_filled"] != len(spec["cups"]) or trace["sink_hit"]:
        errors.append("target flow does not fill every cup without a sink hit")
    if difficulty <= 5 and trace["horizontal_pipe_hits"] < 1:
        errors.append("horizontal splitter was not used")
    if difficulty in (4, 6):
        if trace["embedded_source_emissions"] < 1:
            errors.append("embedded source did not emit")
    if difficulty == 4:
        if trace["source_pipe_incoming_hits"] < 1:
            errors.append("embedded source pipe did not receive incoming native flow")
        if not trace["source_pipe_tag_ablation_prevents_win"]:
            errors.append("embedded source pipe splitter role is not consequential")
    if difficulty == 5 and trace["deflector_right_hits"] < 1:
        errors.append("right deflector was not used")
    if difficulty == 6:
        if trace["vertical_pipe_hits"] < 1:
            errors.append("vertical pipe was not used")
        if trace["deflector_left_hits"] < 1 or trace["deflector_right_hits"] < 1:
            errors.append("both deflector directions were not used")
    if difficulty >= 5:
        side_cups = sum(cup["rotation"] in (90, 270) for cup in spec["cups"])
        if side_cups < (1 if difficulty == 5 else 3):
            errors.append("oriented side-cup coverage is below the reference tier")
        if len(spec["sinks"]) != 3:
            errors.append("three-edge sink hazard is absent")
    profile = REFERENCE_PROFILES[difficulty]
    if not profile["flow_frames"][0] <= trace["flow_frames"] <= profile["flow_frames"][1]:
        errors.append("native flow-frame count is outside the calibrated tier tolerance")
    return errors


class _CriticalityFrameGuard(RuntimeError):
    pass


def _critical_piece_indices(spec):
    """Native one-away test: leaving this piece scrambled must break the win."""
    critical = []
    for held_index, held in enumerate(spec["pieces"]):
        if (held["x"], held["y"]) == (held["target_x"], held["target_y"]):
            continue
        env = Env([build_level(spec)])
        env.reset()
        for index, (sprite, piece) in enumerate(zip(env.movables(), spec["pieces"])):
            if index == held_index:
                sprite.set_position(piece["x"], piece["y"])
            else:
                sprite.set_position(piece["target_x"], piece["target_y"])
        try:
            observation = env.perform(names.ACTION_FLOW)
        except ValueError as exc:
            if "too many frames" not in str(exc).lower():
                raise
            raise _CriticalityFrameGuard(
                f"one-away replay for piece {held_index} exceeded the engine frame guard"
            ) from exc
        if observation.state != GameState.WIN and env.levels_completed == 0:
            critical.append(held_index)
    return critical


def _replay_at_context(spec, actions):
    index = spec["context_index"]
    levels = [build_level(spec) for _ in DIFFICULTIES]
    env = Env(levels)
    env.reset()
    env.set_level(index)
    score = env.levels_completed
    won, observation = replay(env, actions)
    return {
        "won": bool(won),
        "start_index": index,
        "end_index": env.level_index,
        "score_delta": env.levels_completed - score,
        "terminal_state": observation.state.name if observation is not None else None,
    }


def _work_limit(limit, node_limit, search_work):
    values = [value for value in (limit, node_limit, search_work) if value is not None]
    if any(type(value) is not int or not 1 <= value <= 32_000_000 for value in values):
        raise ValueError("each search work limit must be an integer in 1..32000000")
    work = DEFAULT_LIMIT if not values else min(values)
    return work


def last_generation_report():
    """Return this context's terminal one-level generation report."""
    report = _LAST_LEVEL_GENERATION_REPORT.get()
    return None if report is None else copy.deepcopy(report)


def last_game_generation_report():
    """Return this context's terminal whole-game generation report."""
    report = _LAST_GAME_GENERATION_REPORT.get()
    return None if report is None else copy.deepcopy(report)


def verify(spec, limit=DEFAULT_LIMIT, rejection_counts=None):
    """Return an enriched spec after structural, mechanic and native proof checks."""
    try:
        profile = _validate_structure(spec)
        trace = _trace_target(spec)
        mechanic_errors = _mechanic_errors(spec, trace)
        critical = _critical_piece_indices(spec)
    except _CriticalityFrameGuard:
        if rejection_counts is not None:
            rejection_counts["criticality_frame_guard"] += 1
        return None
    except (TypeError, ValueError, KeyError):
        if rejection_counts is not None:
            rejection_counts["malformed_or_guarded"] += 1
        return None
    if mechanic_errors:
        if rejection_counts is not None:
            rejection_counts["mechanic_use"] += 1
        return None
    moved = [
        index for index, piece in enumerate(spec["pieces"])
        if (piece["x"], piece["y"]) != (piece["target_x"], piece["target_y"])
    ]
    if critical != moved:
        if rejection_counts is not None:
            rejection_counts["noncritical_moves"] += 1
        return None
    env = Env([build_level(spec)])
    env.reset()
    result = search(env, limit=limit)
    if not result.solved or result.truncated or result.unsupported or not result.exact:
        if rejection_counts is not None:
            rejection_counts["teacher_route"] += 1
        return None
    actions = [list(action) for action in result.actions]
    length = len(actions)
    if not profile["solution_range"][0] <= length <= profile["solution_range"][1]:
        if rejection_counts is not None:
            rejection_counts["quality_profile"] += 1
        return None
    context = _replay_at_context(spec, actions)
    if not context["won"] or context["score_delta"] != 1:
        if rejection_counts is not None:
            rejection_counts["context_replay"] += 1
        return None
    enriched = json.loads(json.dumps(spec))
    geometry_sha256, geometry_d4_sha256, gameplay_sha256 = _identities(enriched)
    geometry_split = _geometry_partition(geometry_d4_sha256)
    if (geometry_split != enriched["split"]
            or geometry_sha256 in _OFFICIAL_GEOMETRY
            or geometry_d4_sha256 in _OFFICIAL_D4_GEOMETRY
            or gameplay_sha256 in _OFFICIAL_GAMEPLAY):
        if rejection_counts is not None:
            rejection_counts["identity_or_official_copy"] += 1
        return None
    enriched.update({
        "solution": actions,
        "solution_length": length,
        "native_steps_used": length,
        "native_steps_remaining": enriched["steps"] - length,
        "search_work_limit": limit,
        "search_expanded": result.expanded,
        "search_generated": result.generated,
        "search_work": result.expanded,
        "search_truncated": False,
        "search_unsupported": False,
        "search_exact_positive": True,
        "planner_optimality": "not claimed; constructive shortest per-piece routing with native replay",
        "engine_verified": True,
        "proof_level_index": enriched["context_index"],
        "context_replay": context,
        "mechanic_use": trace,
        "critical_piece_indices": critical,
        "geometry_sha256": geometry_sha256,
        "geometry_d4_sha256": geometry_d4_sha256,
        "gameplay_sha256": gameplay_sha256,
        "geometry_split": geometry_split,
        "geometry_version": "sp80-palette-d4-static-water-excluded-v3",
        "gameplay_identity_version": "sp80-native-initial-semantics-v3",
        "official_copy": False,
        "quality_profile_version": FULL_STANDARD_CONTRACT["quality_profile_version"],
        "mechanics_inventory_version": FULL_STANDARD_CONTRACT["mechanics_inventory_version"],
        "reference_tolerance": {
            "solution_length": list(profile["solution_range"]),
            "flow_frames": list(profile["flow_frames"]),
            "reference_teacher_length": profile["reference_teacher_length"],
        },
        "proof": {
            "kind": "constructive-target-native-replay",
            "difficulty": enriched["difficulty"],
            "context_index": enriched["context_index"],
            "native_budget": enriched["steps"],
            "context_engine_verified": True,
            "engine_win": True,
            "search_truncated": False,
            "search_unsupported": False,
            "search_work": result.expanded,
            "search_work_limit": limit,
            "split": enriched["split"],
            "geometry_d4_sha256": geometry_d4_sha256,
            "gameplay_sha256": gameplay_sha256,
            "generator_version": GENERATOR_VERSION,
            "quality_profile_version": FULL_STANDARD_CONTRACT["quality_profile_version"],
        },
    })
    return enriched


def generate(seed, difficulty=1, attempts=MAX_ATTEMPTS, limit=None, node_limit=None,
             search_work=None, split="train", max_attempts=None):
    """Generate one tier level; every positive result carries a native witness."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if max_attempts is not None:
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        attempts = max_attempts
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    work = _work_limit(limit, node_limit, search_work)
    rng = random.Random(f"sp80-full-v3:{split}:{seed}:{difficulty}")
    rejected = Counter()
    for attempt in range(attempts):
        spec = _draft_tier(rng, int(seed), difficulty, split, attempt)
        if not _scramble(rng, spec):
            rejected["scramble"] += 1
            continue
        try:
            geometry_sha256, geometry_d4_sha256, gameplay_sha256 = _identities(spec)
        except (KeyError, TypeError, ValueError):
            rejected["identity"] += 1
            continue
        geometry_split = _geometry_partition(geometry_d4_sha256)
        if geometry_split != split:
            rejected["geometry_split"] += 1
            continue
        if (geometry_sha256 in _OFFICIAL_GEOMETRY
                or geometry_d4_sha256 in _OFFICIAL_D4_GEOMETRY
                or gameplay_sha256 in _OFFICIAL_GAMEPLAY):
            rejected["official_copy"] += 1
            continue
        spec.update({
            "geometry_sha256": geometry_sha256,
            "geometry_d4_sha256": geometry_d4_sha256,
            "gameplay_sha256": gameplay_sha256,
            "geometry_split": geometry_split,
            "geometry_version": "sp80-palette-d4-static-water-excluded-v3",
            "gameplay_identity_version": "sp80-native-initial-semantics-v3",
            "official_copy": False,
        })
        accepted = verify(spec, limit=work, rejection_counts=rejected)
        if accepted is not None:
            accepted["generation_exclusions"] = dict(sorted(rejected.items()))
            report = {
                "scope": "level",
                "status": "accepted",
                "seed": seed,
                "difficulty": difficulty,
                "split": split,
                "attempts_allowed": attempts,
                "attempts_used": attempt + 1,
                "search_work_bound": work,
                "rejections": dict(sorted(rejected.items())),
                "terminal_reason": "accepted_native_replay",
            }
            _LAST_LEVEL_GENERATION_REPORT.set(report)
            accepted["generation_diagnostics"] = copy.deepcopy(report)
            return accepted
    _LAST_LEVEL_GENERATION_REPORT.set({
        "scope": "level",
        "status": "exhausted",
        "seed": seed,
        "difficulty": difficulty,
        "split": split,
        "attempts_allowed": attempts,
        "attempts_used": attempts,
        "search_work_bound": work,
        "rejections": dict(sorted(rejected.items())),
        "terminal_reason": "bounded_attempts_exhausted",
    })
    return None


def _child_seed(seed, split, difficulty):
    payload = f"{SOURCE_ID}|whole-game-v2|{split}|{seed}|{difficulty}"
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")


def _requested_tiers(difficulties):
    if difficulties is None:
        return DIFFICULTIES
    if (not isinstance(difficulties, Sequence)
            or isinstance(difficulties, (str, bytes))):
        raise ValueError("difficulties must be an ordered integer sequence")
    tiers = tuple(difficulties)
    if not tiers or any(type(value) is not int or value not in DIFFICULTIES for value in tiers):
        raise ValueError("difficulties must be a nonempty sequence drawn from DIFFICULTIES")
    if tuple(value for value in DIFFICULTIES if value in tiers) != tiers:
        raise ValueError("explicit difficulties must be unique and in official order")
    return tiers


def _sequential_replay(specs):
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    env.reset()
    records = []
    observation = None
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            raise ValueError("native game entered the wrong curriculum context")
        won, observation = replay(env, spec["solution"])
        if not won or env.levels_completed != index + 1:
            raise ValueError("stored solution did not advance exactly one native level")
        records.append({
            "start_index": index,
            "end_index": env.level_index,
            "score_after": env.levels_completed,
            "terminal_state": observation.state.name,
        })
    if observation is None or observation.state != GameState.WIN:
        raise ValueError("complete native game did not reach WIN")
    return levels, records


def generate_game(seed, *, split="train", difficulties=None, attempts=MAX_ATTEMPTS,
                  limit=None, node_limit=None, search_work=None):
    """Generate ordered levels; omitted ``difficulties`` means the full six-tier game."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    requested = _requested_tiers(difficulties)
    specs = []
    for difficulty in requested:
        spec = generate(
            _child_seed(seed, split, difficulty), difficulty, attempts=attempts,
            limit=limit, node_limit=node_limit, search_work=search_work, split=split,
        )
        if spec is None:
            _LAST_GAME_GENERATION_REPORT.set({
                "scope": "whole_game",
                "status": "failed",
                "game_seed": seed,
                "split": split,
                "requested_difficulties": list(requested),
                "completed_difficulties": [row["difficulty"] for row in specs],
                "failed_difficulty": difficulty,
                "failed_child_seed": _child_seed(seed, split, difficulty),
                "terminal_reason": "level_generation_failed",
                "level_report": last_generation_report(),
            })
            return None
        spec["game_seed"] = seed
        spec["child_seed"] = spec["seed"]
        specs.append(spec)
    if requested == DIFFICULTIES:
        try:
            _, records = _sequential_replay(specs)
        except ValueError:
            _LAST_GAME_GENERATION_REPORT.set({
                "scope": "whole_game",
                "status": "failed",
                "game_seed": seed,
                "split": split,
                "requested_difficulties": list(requested),
                "completed_difficulties": [row["difficulty"] for row in specs],
                "failed_difficulty": None,
                "terminal_reason": "sequential_native_replay_failed",
            })
            return None
        for spec, record in zip(specs, records):
            spec["whole_game_context_replay"] = record
    _LAST_GAME_GENERATION_REPORT.set({
        "scope": "whole_game",
        "status": "accepted",
        "game_seed": seed,
        "split": split,
        "requested_difficulties": list(requested),
        "completed_difficulties": [row["difficulty"] for row in specs],
        "terminal_reason": "accepted_native_replay",
    })
    return specs


def build_game(specs):
    """Build only a complete, correctly ordered six-level native game."""
    if not isinstance(specs, (list, tuple)):
        raise ValueError("full-standard SP80 game specs must be a list or tuple")
    specs = list(specs)
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("build_game requires the complete six-tier sequence")
    if any(not isinstance(spec, Mapping) for spec in specs):
        raise ValueError("every game spec must be a mapping")
    difficulties = tuple(spec.get("difficulty") for spec in specs)
    if difficulties != DIFFICULTIES or len(set(difficulties)) != len(difficulties):
        raise ValueError("game difficulties must be unique and ordered 1 through 6")
    splits = {spec.get("split") for spec in specs}
    if len(splits) != 1:
        raise ValueError("all game levels must belong to the same split")
    geometries, gameplays = set(), set()
    for index, (spec, entry) in enumerate(zip(specs, CURRICULUM)):
        if spec.get("context_index") != index:
            raise ValueError("context indices must match native level indices")
        if spec.get("proof_level_index") != index:
            raise ValueError("proof context does not match native level index")
        errors = validate_full_standard(spec, entry)
        if errors:
            raise ValueError("invalid full-standard spec: " + "; ".join(errors))
        geometry = spec["geometry_d4_sha256"]
        gameplay = spec["gameplay_sha256"]
        if geometry in geometries or gameplay in gameplays:
            raise ValueError("duplicate geometry or gameplay identity in game")
        geometries.add(geometry)
        gameplays.add(gameplay)
    levels, _ = _sequential_replay(specs)
    return levels


def validate_full_standard(spec, curriculum_entry):
    """Recompute structure, identities, mechanics, budget and native proof parity."""
    errors = []
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    try:
        profile = _validate_structure(spec)
    except (TypeError, ValueError, KeyError) as exc:
        return [f"malformed spec: {exc}"]
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be a mapping"]
    expected_difficulty = curriculum_entry.get("difficulty")
    expected_context = curriculum_entry.get("context_index")
    expected_work = curriculum_entry.get("search_work")
    if any(type(value) is not int for value in
           (expected_difficulty, expected_context, expected_work)):
        return ["curriculum difficulty, context_index and search_work must be integers"]
    if spec["difficulty"] != expected_difficulty:
        errors.append("difficulty does not match curriculum entry")
    if spec["context_index"] != expected_context:
        errors.append("context index does not match curriculum entry")
    if not 1 <= expected_work <= 32_000_000:
        errors.append("curriculum search_work is outside the contract bounds")
    stored_work = spec.get("search_work_limit")
    if (isinstance(stored_work, bool) or not isinstance(stored_work, int)
            or stored_work < 1 or stored_work > expected_work):
        errors.append("stored search work does not honor the curriculum bound")
    try:
        geometry_sha256, geometry_d4_sha256, gameplay_sha256 = _identities(spec)
        if spec.get("geometry_sha256") != geometry_sha256:
            errors.append("exact geometry SHA-256 mismatch")
        if spec.get("geometry_d4_sha256") != geometry_d4_sha256:
            errors.append("D4 geometry SHA-256 mismatch")
        if spec.get("gameplay_sha256") != gameplay_sha256:
            errors.append("native gameplay SHA-256 mismatch")
        geometry_split = _geometry_partition(geometry_d4_sha256)
        if spec.get("geometry_split") != geometry_split or spec.get("split") != geometry_split:
            errors.append("canonical geometry belongs to a different split")
        if (geometry_sha256 in _OFFICIAL_GEOMETRY
                or geometry_d4_sha256 in _OFFICIAL_D4_GEOMETRY
                or gameplay_sha256 in _OFFICIAL_GAMEPLAY):
            errors.append("recomputed initial identity matches an official level")
        if spec.get("official_copy") is not False:
            errors.append("official-copy exclusion flag is missing")
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"identity recomputation failed: {exc}")

    try:
        trace = _trace_target(spec)
        errors.extend(_mechanic_errors(spec, trace))
        if spec.get("mechanic_use") != trace:
            errors.append("stored mechanic-use certificate differs from native trace")
        critical = _critical_piece_indices(spec)
        if spec.get("critical_piece_indices") != critical:
            errors.append("stored critical-piece certificate differs from native one-away replay")
        moved = [
            index for index, piece in enumerate(spec["pieces"])
            if (piece["x"], piece["y"]) != (piece["target_x"], piece["target_y"])
        ]
        if critical != moved:
            errors.append("the certified route contains a noncritical piece move")
    except _CriticalityFrameGuard as exc:
        errors.append(f"critical-piece proof is unknown: {exc}")
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"mechanic trace failed: {exc}")

    actions = spec.get("solution")
    if not isinstance(actions, list):
        errors.append("solution must be a JSON action list")
        actions = []
    else:
        for action in actions:
            if not isinstance(action, list) or len(action) != 3:
                errors.append("solution contains a malformed action triple")
                break
            action_id, x, y = action
            if type(action_id) is not int or action_id not in names.AVAILABLE_ACTIONS:
                errors.append("solution contains an unavailable action")
                break
            if action_id == names.ACTION_CLICK:
                if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
                    errors.append("solution contains an illegal display click")
                    break
            elif x is not None or y is not None:
                errors.append("non-click solution actions must use null coordinates")
                break
    if spec.get("solution_length") != len(actions):
        errors.append("solution length metadata mismatch")
    if not profile["solution_range"][0] <= len(actions) <= profile["solution_range"][1]:
        errors.append("solution length is outside the calibrated tier tolerance")
    if len(actions) > spec["steps"]:
        errors.append("solution exceeds the native action budget")
    if spec.get("native_steps_used") != len(actions):
        errors.append("native step accounting mismatch")
    if spec.get("native_steps_remaining") != spec["steps"] - len(actions):
        errors.append("remaining native step accounting mismatch")
    try:
        fresh = Env([build_level(spec)])
        fresh.reset()
        recomputed = search(fresh, limit=stored_work if type(stored_work) is int else 0)
        recomputed_actions = ([list(action) for action in recomputed.actions]
                              if recomputed.actions is not None else None)
        if (not recomputed.solved or recomputed.truncated or recomputed.unsupported
                or recomputed_actions != actions):
            errors.append("stored route differs from bounded constructive recomputation")
        if spec.get("search_work") != recomputed.expanded:
            errors.append("stored search work differs from measured routing work")
        context = _replay_at_context(spec, actions)
        if not context["won"] or context["score_delta"] != 1:
            errors.append("stored route does not win in the real engine")
        if spec.get("context_replay") != context:
            errors.append("stored context replay differs from native replay")
        target_positions = [(p["target_x"], p["target_y"]) for p in spec["pieces"]]
        env = Env([build_level(spec)])
        env.reset()
        for action in actions[:-1]:
            env.perform(*action)
        if tuple((int(s.x), int(s.y)) for s in env.movables()) != tuple(target_positions):
            errors.append("route does not reach its certified target placement")
        if not actions or actions[-1] != [names.ACTION_FLOW, None, None]:
            errors.append("route does not terminate with the certified flow action")
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        errors.append(f"native proof replay failed: {exc}")
    proof = spec.get("proof")
    if not isinstance(proof, Mapping):
        errors.append("proof must be a mapping")
    else:
        mirrors = {
            "difficulty": spec.get("difficulty"),
            "context_index": spec.get("context_index"),
            "native_budget": spec.get("steps"),
            "context_engine_verified": True,
            "engine_win": True,
            "search_truncated": False,
            "search_unsupported": False,
            "search_work": spec.get("search_work"),
            "search_work_limit": spec.get("search_work_limit"),
            "split": spec.get("split"),
            "geometry_d4_sha256": spec.get("geometry_d4_sha256"),
            "gameplay_sha256": spec.get("gameplay_sha256"),
            "generator_version": GENERATOR_VERSION,
            "quality_profile_version": FULL_STANDARD_CONTRACT["quality_profile_version"],
        }
        for key, expected in mirrors.items():
            if proof.get(key) != expected:
                errors.append(f"proof.{key} does not mirror the certificate")
    if spec.get("engine_verified") is not True:
        errors.append("engine verification flag is missing")
    exclusions = spec.get("generation_exclusions")
    diagnostics = spec.get("generation_diagnostics")
    if (not isinstance(exclusions, Mapping)
            or any(not isinstance(key, str) or type(value) is not int or value < 0
                   for key, value in (exclusions.items() if isinstance(exclusions, Mapping) else ()))):
        errors.append("bounded generation exclusions are missing or malformed")
    if not isinstance(diagnostics, Mapping):
        errors.append("bounded generation diagnostics are missing")
    else:
        if (type(diagnostics.get("attempts_allowed")) is not int
                or type(diagnostics.get("attempts_used")) is not int
                or not 1 <= diagnostics.get("attempts_used", 0) <= diagnostics.get("attempts_allowed", 0)):
            errors.append("bounded attempt diagnostics are malformed")
        if diagnostics.get("search_work_bound") != stored_work:
            errors.append("generation search-work bound differs from the proof")
        diagnostic_rejections = diagnostics.get("rejections")
        if not isinstance(diagnostic_rejections, Mapping):
            errors.append("generation rejection diagnostics are malformed")
        elif (isinstance(exclusions, Mapping)
              and dict(diagnostic_rejections) != dict(exclusions)):
            errors.append("generation rejection diagnostics differ from exclusions")
    return errors
