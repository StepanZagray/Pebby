"""Full seven-tier TN36 procedural generation and native certification."""

from collections import Counter
import hashlib
import json
import random

from arcengine import GameState, Level

from . import names
from .env import Env, UPSTREAM, upstream
from .layout import extract
from .plan import execute_program, program_actions, program_end_state, search
from .quality import (
    DIFFICULTIES, GAMEPLAY_VERSION, GEOMETRY_VERSION, MECHANICS_VERSION,
    PROFILE_VERSION, REFERENCE_PROFILES, REQUIRED_MECHANIC_EVENTS, SPLITS,
    event_counts, identities, official_equivalence_identity, profile_errors,
    split_accepts, split_bucket,
)


SOURCE_SHA256 = "43c30052cdeb230017eb1ec23877050d96297756b86071417868ec61138c9fb7"
if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("vendored TN36 source bytes differ from the calibrated engine")

FORMAT = "pebby.tn36.full-level.v4"
GENERATOR_VERSION = 5
# Draft rejections (invalid geometry, wrong split bucket) are cheap; the
# validation/test buckets each accept 10% of canonical gameplay, and tier 5
# drafts additionally lose most candidates to frame clipping, so the default
# budget must cover a few hundred drafts to reach tier 5 in every split.
DEFAULT_ATTEMPTS = 600

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": names.SOURCE_ID,
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": PROFILE_VERSION,
    "curriculum": [
        {"difficulty": d, "context_index": d - 1,
         "search_work": REFERENCE_PROFILES[d]["search_work"]}
        for d in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "tn36.md#official-reference-characterization",
        "solution_mechanics": "public solution, mechanic_solution, and tier-7 recovery route are independently native-replayed; the public solution itself exercises every tier-required mechanic event (selector click, rollback, scale, rotation, recolor, checkpoint, gate toggle) and is the constrained search output",
        "native_budget": "quality.py:REFERENCE_PROFILES and native timer replay",
        "context_engine_replay": "root v4 three-split full-game replay: 7 WIN with 67/66/71 actions; current glyph-only revision preserves generator semantics",
        "novelty_split": "translation-normalized executable gameplay partition, deterministic provenance, and official-equivalence exclusion",
        "bounded_rejections": "generate.last_report typed bounded causes; default cap can honestly exhaust",
    },
    "caveats": [
        "Each tier has one shipped reference; tolerances are engineering bands, not population estimates.",
        "Tiers 6-7 use constructive checkpoint routes; no shortest-route claim is made for multi-run solutions.",
        "Preset selectors are pedagogical locked-panel demonstrations and do not directly alter the goal panel; the public teacher clicks the selector demonstrating an opcode effect its own programs use before editing.",
        "Per-preset editable history exists in engine code but has zero shipped-level incidence because every shipped preset panel is locked.",
        "Generated public routes satisfy the action bands, but no global shortest-route claim is made.",
        "Generation uses a finite procedural grammar and bounded attempts, so generate can return None.",
        "Preset effect glyphs cover the generated unambiguous opcode families; ambiguous or no-op-only programs intentionally receive no glyph.",
    ],
}


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _selector_access_points(selector_xs):
    """Return one native first-hit point that reaches each ordered selector."""
    rectangles = [(x, 54, 9, 9) for x in selector_xs]
    points = []
    for index, (x, y, width, height) in enumerate(rectangles):
        point = next((
            (px, py)
            for py in range(y, y + height)
            for px in range(x, x + width)
            if not any(
                left <= px < left + other_width
                and top <= py < top + other_height
                for left, top, other_width, other_height in rectangles[:index]
            )
        ), None)
        if point is None:
            raise ValueError("selector hit regions must each have an independently accessible point")
        points.append(point)
    return tuple(points)


def _valid_target(state, difficulty=None):
    x, y, _, scale, _ = state
    if difficulty == 1:
        return 14 <= x - scale and x + 5 * scale <= 50 and 9 <= y - scale and y + 5 * scale <= 40
    return 33 <= x - scale and x + 5 * scale <= 62 and 4 <= y - scale and y + 5 * scale <= 31


def _valid_actor(state, difficulty=None):
    x, y, _, scale, _ = state
    if difficulty == 1:
        return 13 <= x and x + 4 * scale <= 51 and 8 <= y and y + 4 * scale <= 41
    return 31 <= x and x + 4 * scale <= 64 and 2 <= y and y + 4 * scale <= 32


def _transform_sprite(sprite, state, *, target=False):
    x, y, rotation, scale, color = state
    if target:
        sprite.set_position(x - scale, y - scale)
        if color != 11:
            sprite.color_remap(11, color)
    else:
        sprite.set_position(x, y)
        if color != 11:
            sprite.color_remap(None, color)
    sprite.set_rotation(rotation)
    sprite.set_scale(scale)
    return sprite


def _wall(source, name, x, y, rotation=0):
    sprite = source[name].clone().set_position(x, y).set_rotation(rotation)
    return sprite, {
        "sprite": name, "x": int(x), "y": int(y), "rotation": int(rotation),
        "width": int(sprite.width), "height": int(sprite.height),
    }


def _program_sprites(source, *, start_x, slots, width, values):
    sprites = []
    for slot in range(slots):
        x = start_x + 5 * slot
        sprites.append(source["inwola"].clone().set_position(x, 32))
        value = int(values[slot])
        for bit in range(width):
            button = source["Maidxz"].clone().set_position(x + 1, 32 + 3 * bit)
            if bit % 2:
                button.set_rotation(90)
            if value & (1 << bit):
                button.color_remap(None, 5)
            sprites.append(button)
    return sprites


_PRESET_EFFECT_GLYPHS = {
    ("dx", -4): ("iczcramoqjhw", 0, 3, 180),
    ("dx", 4): ("iczcramoqjhw", 2, 3, 0),
    ("dy", -4): ("iczcramoqjhw", 3, 0, 270),
    ("dy", 4): ("iczcramoqjhw", 3, 2, 90),
    ("scale", 1): ("iczcrascvkkwuqelhb", 2, 2, 0),
    ("scale", -1): ("iczcrascvkkwdoylbb", 4, 4, 0),
    ("rotation", 90): ("iczcraroumnb", 2, 2, 0),
    ("color", 15): ("iczcrapumzpq", 3, 3, 0),
}


def _preset_effect_glyph(source, program, selector_x):
    """Build the native cue for a preset's single public opcode effect."""
    effects = {
        names.OPCODE_EFFECTS.get(value, ("noop", 0))
        for value in program
        if names.OPCODE_EFFECTS.get(value, ("noop", 0))[0] != "noop"
    }
    descriptor = _PRESET_EFFECT_GLYPHS.get(next(iter(effects))) if len(effects) == 1 else None
    if descriptor is None:
        return None
    name, offset_x, offset_y, rotation = descriptor
    return source[name].clone().set_position(
        selector_x + offset_x, 54 + offset_y,
    ).set_rotation(rotation)


def _validated_obstacle(item, expected_kind):
    if not isinstance(item, dict):
        raise ValueError(f"{expected_kind} entry must be a mapping")
    source = upstream().sprites
    name = item.get("sprite")
    if type(name) is not str:
        raise ValueError(f"{expected_kind}.sprite must be a string")
    allowed = {
        "wall": {"wauzms", "wauzms-1", "wauzms-2", "wauzms-4", "wauzms-5"},
        "platform": {"chrccc", "chrccc-2"},
    }[expected_kind]
    if name not in allowed:
        raise ValueError(f"invalid {expected_kind} sprite")
    x = _integer(item.get("x"), f"{expected_kind}.x")
    y = _integer(item.get("y"), f"{expected_kind}.y")
    rotation = _integer(item.get("rotation", 0), f"{expected_kind}.rotation")
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"{expected_kind}.rotation must be a quarter turn")
    sprite = source[name].clone().set_position(x, y).set_rotation(rotation)
    if (type(item.get("width")) is not int or type(item.get("height")) is not int
            or item["width"] != sprite.width or item["height"] != sprite.height):
        raise ValueError(f"{expected_kind} dimensions do not match its native sprite")
    if not (31 <= x and 0 <= y and x + sprite.width <= 64 and y + sprite.height <= 32):
        raise ValueError(f"{expected_kind} clips or leaves the goal playfield")
    return sprite


def build_level(spec):
    """Rebuild one native level from a fail-closed JSON-safe specification."""
    if not isinstance(spec, dict) or spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    profile = REFERENCE_PROFILES[difficulty]
    if _integer(spec.get("context_index"), "context_index") != difficulty - 1:
        raise ValueError("context_index does not match difficulty")
    if type(spec.get("actor")) is not list or type(spec.get("target")) is not list:
        raise ValueError("actor/target transforms must be JSON lists")
    actor = tuple(_integer(value, "actor") for value in spec["actor"])
    target = tuple(_integer(value, "target") for value in spec["target"])
    if (len(actor) != 5 or len(target) != 5
            or not _valid_actor(actor, difficulty)
            or not _valid_target(target, difficulty)):
        raise ValueError("actor/target transforms are malformed or out of frame")
    if (actor[2] not in (0, 90, 180, 270) or target[2] not in (0, 90, 180, 270)
            or not 1 <= actor[3] <= 4 or not 1 <= target[3] <= 4
            or actor[4] not in (8, 9, 11, 15) or target[4] not in (8, 9, 11, 15)):
        raise ValueError("actor/target rotation, scale, or color is outside the native schema")
    slots = profile["slots"]
    width = profile["bit_width"]
    if type(spec.get("bit_width")) is not int or spec["bit_width"] != width:
        raise ValueError("bit_width does not match the official tier")
    if type(spec.get("native_budget")) is not int or spec["native_budget"] != profile["native_budget"]:
        raise ValueError("native_budget does not match the official tier")
    if type(spec.get("initial_program")) is not list:
        raise ValueError("initial_program must be a JSON list")
    initial = [_integer(value, "initial_program") for value in spec["initial_program"]]
    if len(initial) != slots or any(not 0 <= value < 1 << width for value in initial):
        raise ValueError("initial_program does not match the tier program geometry")
    preset_programs = spec.get("preset_programs")
    if type(preset_programs) is not list or any(type(values) is not list for values in preset_programs):
        raise ValueError("preset_programs must be a list of JSON lists")
    left_slots = _integer(spec.get("left_slot_count", 0), "left_slot_count")
    if len(preset_programs) != profile["selectors"]:
        raise ValueError("preset count does not match the official tier")
    if preset_programs:
        if not 1 <= left_slots <= 6 or any(len(values) != left_slots for values in preset_programs):
            raise ValueError("preset programs have inconsistent slot counts")
        if any(any(type(value) is not int or not 0 <= value < 64 for value in values)
               for values in preset_programs):
            raise ValueError("preset opcode is outside the six-bit program domain")
    elif left_slots != 0:
        raise ValueError("left_slot_count must be zero without presets")
    preset_positions = spec.get("preset_positions")
    preset_rotations = spec.get("preset_rotations")
    preset_scales = spec.get("preset_scales")
    preset_resets = spec.get("preset_resets")
    selector_xs = spec.get("selector_xs")
    values = (preset_positions, preset_rotations, preset_scales, preset_resets, selector_xs)
    if any(type(value) is not list for value in values):
        raise ValueError("preset metadata and selector_xs must be JSON lists")
    if any(len(value) != len(preset_programs) for value in values):
        raise ValueError("preset metadata lengths do not match preset_programs")
    for position in preset_positions:
        if (type(position) is not list or len(position) != 2
                or any(type(value) is not int for value in position)
                or any(not -8 <= value <= 8 for value in position)):
            raise ValueError("preset position is outside the native grid-offset schema")
    if any(type(value) is not int or value not in (0, 90, 180, 270)
           for value in preset_rotations):
        raise ValueError("preset rotations must be exact quarter-turn integers")
    if any(type(value) is not int or not 1 <= value <= 4 for value in preset_scales):
        raise ValueError("preset scales must be exact bounded integers")
    if any(type(value) is not bool for value in preset_resets):
        raise ValueError("preset reset flags must be booleans")
    if any(type(value) is not int for value in selector_xs):
        raise ValueError("selector positions must be exact integers")
    _selector_access_points(selector_xs)
    if selector_xs != sorted(selector_xs) or len(set(selector_xs)) != len(selector_xs):
        raise ValueError("selector positions must preserve strict native left-to-right order")
    if type(spec.get("walls")) is not list or type(spec.get("platforms")) is not list:
        raise ValueError("walls/platforms must be JSON lists")
    if type(spec.get("gates")) is not list:
        raise ValueError("gates must be a JSON list")

    source = upstream().sprites
    if difficulty == 1:
        sprites = [
            source["pavecd"].clone(),
            source["plljmx_2"].clone().set_position(-33, 14),
            source["plljmx_4"].clone().set_position(13, 8),
            source["grsysj4"].clone().set_position(-31, 16),
            source["grsysj5"].clone().set_position(14, 9),
            source["dixdujbaatdvbizzhq"].clone().set_position(-32, 15),
            source["bltjrl4dixduj"].clone().set_position(-23, 24),
            _transform_sprite(source["bltjrl4"].clone(), actor),
            _transform_sprite(source["taptxx"].clone(), target, target=True),
            source["reooaotakfnbsmurct"].clone().set_position(-32, 44),
            source["takfnbsmurct"].clone().set_position(19, 41),
            source["buojsacorqds"].clone().set_position(36, 48),
            source["sucqgkbuojsa"].clone().set_position(32, 51),
            source["sthpyhbaatdvbizzhq"].clone().set_position(1, 1),
            source["sthpyhbizzhq"].clone().set_position(1, 1),
        ]
        for slot, value in enumerate(initial):
            x = 19 + 5 * slot
            sprites.append(source["inwolasmurct"].clone().set_position(x, 41))
            horizontal = source["Maidxzhowygv"].clone().set_position(x, 41)
            vertical = source["Maidxzvenlef"].clone().set_position(x, 44)
            if value & 1:
                horizontal.color_remap(None, 5)
            if value & 2:
                vertical.color_remap(None, 5)
            sprites.extend((horizontal, vertical))
    else:
        sprites = [
            source["pavecd3"].clone().set_position(-1, 0),
            source["plljmx_2"].clone().set_position(0, 2),
            source["plljmx_3"].clone().set_position(31, 2),
            source["grsysj4"].clone().set_position(2, 4),
            source["grsysj4"].clone().set_position(33, 4),
            source["bltjrl4dixduj"].clone().set_position(14, 16),
            _transform_sprite(source["bltjrl4"].clone(), actor),
            _transform_sprite(source["taptxx"].clone(), target, target=True),
            source["reooaotakfnb"].clone().set_position(1, 32),
            source["takfnb"].clone().set_position(32, 32),
            source["sucqgkbuojsa"].clone().set_position(53, 54),
            source["sthpyhbaatdvbizzhq"].clone().set_position(1, 1),
            source["sthpyhbizzhq"].clone().set_position(1, 1),
            source["dixdujbaatdvbizzhq"].clone().set_position(1, 3),
        ]
        # The shipped advanced tiers cover the left demonstration grid with a
        # solid tutorial card. Besides matching visual density, this makes the
        # locked preset program read as a worked example rather than a second
        # unsolved board.
        sprites.extend(_program_sprites(source, start_x=32, slots=slots, width=width, values=initial))
    if preset_programs:
        left_x = max(1, (31 - 5 * left_slots) // 2)
        sprites.extend(_program_sprites(
            source, start_x=left_x, slots=left_slots, width=6,
            values=preset_programs[0],
        ))
        if len(selector_xs) != len(preset_programs):
            raise ValueError("selector positions do not match presets")
        for index, x in enumerate(selector_xs):
            x = _integer(x, "selector_x")
            if not 1 <= x <= 31:
                raise ValueError("selector clips the left control panel")
            glyph = _preset_effect_glyph(source, preset_programs[index], x)
            if glyph is not None:
                sprites.append(glyph)
            name = "tozzsfsedoig" if index == 0 else "tozzsf1"
            sprites.append(source[name].clone().set_position(x, 54))

    for item in spec.get("walls", []):
        sprites.append(_validated_obstacle(item, "wall"))
    for item in spec.get("platforms", []):
        sprites.append(_validated_obstacle(item, "platform"))
    for gate in spec.get("gates", []):
        if not isinstance(gate, dict):
            raise ValueError("gate entry must be a mapping")
        rotation = _integer(gate.get("rotation"), "gate.rotation") % 360
        if rotation not in (0, 180):
            raise ValueError("gates support horizontal rotations 0 or 180")
        x = _integer(gate.get("x"), "gate.x")
        y = _integer(gate.get("y"), "gate.y")
        body_x = _integer(gate.get("body_x"), "gate.body_x")
        body_y = _integer(gate.get("body_y"), "gate.body_y")
        barrier = source["laycmuommm"].clone().set_position(x, y).set_rotation(rotation)
        body = source["laycmuofkkgm"].clone().set_position(body_x, body_y).set_rotation(rotation)
        if type(gate.get("visible")) is not bool:
            raise ValueError("gate.visible must be a boolean")
        if (type(gate.get("width")) is not int or type(gate.get("height")) is not int
                or type(gate.get("body_width")) is not int
                or type(gate.get("body_height")) is not int
                or gate["width"] != barrier.width or gate["height"] != barrier.height
                or gate["body_width"] != body.width or gate["body_height"] != body.height):
            raise ValueError("gate dimensions do not match the native barrier")
        if not (x < body_x + body.width and body_x < x + barrier.width
                and y < body_y + body.height and body_y < y + barrier.height):
            raise ValueError("gate body must overlap its native barrier for pairing")
        if not (31 <= min(x, body_x) and max(x + barrier.width, body_x + body.width) <= 64
                and 0 <= min(y, body_y) and max(y + barrier.height, body_y + body.height) <= 32):
            raise ValueError("gate clips or leaves the goal playfield")
        barrier.set_visible(gate["visible"])
        sprites.extend((body, barrier))

    data = {
        "Programs": [list(values) for values in preset_programs] or None,
        "Positions": [list(value) for value in preset_positions] or None,
        "Rotations": list(preset_rotations) or None,
        "scvkkws": list(preset_scales) or None,
        "Reset": list(preset_resets) or None,
    }
    return Level(
        sprites=sprites, grid_size=(64, 64), data=data,
        name=f"generated-tn36-d{difficulty}-s{spec.get('seed', 0)}-a{spec.get('generation_attempt', 0)}",
    )


def _context_env(spec):
    level = build_level(spec)
    env = Env([level.clone() for _ in DIFFICULTIES])
    env.set_level(spec["context_index"])
    return env


def _item(source, name, x, y, rotation=0):
    _, value = _wall(source, name, x, y, rotation)
    return value


def _preset_fields(rng, difficulty):
    count = REFERENCE_PROFILES[difficulty]["selectors"]
    if not count:
        return {
            "left_slot_count": 0, "preset_programs": [], "preset_positions": [],
            "preset_rotations": [], "preset_scales": [], "preset_resets": [],
            "selector_xs": [],
        }
    left_slots = 4 if difficulty in (2, 3) else 3
    opcode_families = [1, 33, 2, 3, 8, 9, 5, 63]
    rng.shuffle(opcode_families)
    programs = [[opcode_families[index]] * left_slots for index in range(count)]
    positions = []
    rotations = []
    scales = []
    for index in range(count):
        positions.append(list(rng.choice(((0, -2), (0, 2), (2, 0), (-2, 0), (0, 0)))))
        rotations.append(rng.choice((0, 90, 180, 270)))
        scales.append(rng.choice((1, 1, 1, 2)))
    selector_xs = [int(round(1 + index * 30 / max(1, count - 1))) for index in range(count)]
    return {
        "left_slot_count": left_slots, "preset_programs": programs,
        "preset_positions": positions, "preset_rotations": rotations,
        "preset_scales": scales, "preset_resets": [False] * count,
        "selector_xs": selector_xs,
    }


def _base(rng, difficulty):
    profile = REFERENCE_PROFILES[difficulty]
    spec = {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "game": "tn36", "source_id": names.SOURCE_ID,
        "vendored_source_sha256": SOURCE_SHA256,
        "mechanics_version": MECHANICS_VERSION,
        "quality_profile_version": PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION, "gameplay_version": GAMEPLAY_VERSION,
        "difficulty": difficulty, "context_index": difficulty - 1,
        "bit_width": profile["bit_width"], "native_budget": profile["native_budget"],
        "walls": [], "platforms": [], "gates": [], "source": "generated_only",
    }
    spec.update(_preset_fields(rng, difficulty))
    return spec


def _program_at_distance(rng, wanted, width, distance):
    bits = [(slot, bit) for slot in range(len(wanted)) for bit in range(width)]
    rng.shuffle(bits)
    values = list(wanted)
    for slot, bit in bits[:distance]:
        values[slot] ^= 1 << bit
    return values


def _draft(rng, difficulty):
    source = upstream().sprites
    spec = _base(rng, difficulty)
    move = rng.choice((1, 2))
    opposite = 2 if move == 1 else 1
    color = rng.choice((8, 9, 15))
    construction = []

    if difficulty == 1:
        program = list(rng.choice((
            (3, 3, 3, 3, 3), (2, 2, 3, 3, 3), (1, 1, 3, 3, 3),
            (2, 3, 2, 3, 2), (1, 3, 1, 3, 1),
        )))
        dx = sum(names.OPCODE_EFFECTS[code][1] for code in program
                 if names.OPCODE_EFFECTS[code][0] == "dx")
        actor_x = 22 if dx >= 0 else 42
        actor = [actor_x, 13, 0, 1, 11]
        construction = [program]
        initial = _program_at_distance(rng, program, 2, 6)
    elif difficulty == 2:
        actor = [rng.choice((41, 45, 49)), 24, rng.choice((0, 90, 180, 270)), 1, 11]
        program = list(rng.choice((
            (33, 33, 33, 33), (1, 33, 33, 33), (2, 33, 33, 33),
            (33, 5, 33, 33), (33, 6, 33, 33), (33, 33, 7, 33),
        )))
        construction = [program]
        initial = _program_at_distance(rng, program, 6, 7)
    elif difficulty == 3:
        actor = [37 if move == 2 else 53, 20, rng.choice((0, 180)), 1, 11]
        wall_x = actor[0] + (8 if move == 2 else -4)
        spec["walls"] = [
            _item(source, "wauzms", wall_x, 20, 90),
            _item(source, "wauzms-1", rng.choice((33, 57)), 4),
        ]
        tail = list(rng.choice(((33, 33, move, move), (33, move, 33, move),
                                (33, move, move, 33))))
        program = [move, move, *tail]
        construction = [program]
        initial = _program_at_distance(rng, program, 6, 7)
    elif difficulty == 4:
        actor = [45, 8, rng.choice((0, 180)), 2, 11]
        spec["walls"] = [
            _item(source, "wauzms-4", 33, 20),
            _item(source, "wauzms-2", 53, 20),
        ]
        program = list(rng.choice((
            (3, 9, 3, 3, 2, 2), (9, 3, 3, 3, 2, 2),
            (3, 3, 9, 3, 2, 2), (3, 3, 3, 9, 2, 2),
        )))
        construction = [program]
        # The reference tier has a twelve-action teacher.  A ten-bit draft
        # distance correlated with sub-band public routes strongly enough to
        # exhaust legitimate test-split candidates.  Sixteen preserves the
        # native program grammar while producing independently searched routes
        # inside the unchanged 8..19 action band.
        initial = _program_at_distance(rng, program, 6, 16)
    elif difficulty == 5:
        actor = [49, 8, 270, 1, 11]
        wall_configuration = rng.choice((
            (("wauzms", 33, 20, 0), ("wauzms-1", 57, 20, 0)),
            (("wauzms-2", 33, 24, 0), ("wauzms-2", 53, 20, 0)),
            (("wauzms-1", 33, 16, 0), ("wauzms-4", 41, 24, 0)),
            (("wauzms-2", 33, 20, 0), ("wauzms", 53, 24, 0)),
        ))
        spec["walls"] = [_item(source, *value) for value in wall_configuration]
        operations = [3, 8, rng.choice((5, 6, 7)), 14 if color == 9 else 15, move, 0]
        if color == 15:
            operations[3] = 63
        # All chosen operations commute for this obstacle placement; ordering
        # changes the executed program semantics without adding inert clicks.
        rng.shuffle(operations)
        program = operations
        construction = [program]
        initial = _program_at_distance(rng, program, 6, 14)
    elif difficulty == 6:
        actor = [33, 28, 90, 1, 11]
        side_configuration = rng.choice((
            (("wauzms-1", 33, 12, 0), ("wauzms-1", 57, 24, 0), ("wauzms-2", 49, 4, 0)),
            (("wauzms-2", 33, 8, 0), ("wauzms-1", 57, 20, 0), ("wauzms-1", 33, 24, 0)),
            (("wauzms-1", 37, 8, 0), ("wauzms-2", 53, 24, 90), ("wauzms-1", 33, 16, 0)),
        ))
        spec["walls"] = [
            _item(source, "wauzms", 45, 20, 90),
            *[_item(source, *value) for value in side_configuration],
        ]
        extra_platforms = rng.choice((((37, 8), (53, 28)), ((33, 8), (49, 24)), ((37, 12), (53, 24))))
        spec["platforms"] = [
            _item(source, "chrccc", 41, 20),
            *[_item(source, "chrccc", x, y) for x, y in extra_platforms],
        ]
        suffix = [33, 33, 0]
        rng.shuffle(suffix)
        second = [10, 2, 33, 33, 33, 0]
        rng.shuffle(second)
        construction = [[2, 2, 2, *suffix], second]
        initial = _program_at_distance(rng, construction[0], 6, 5)
    else:
        actor = [37, 28, 180, 1, 11]
        wall_configuration = rng.choice((
            (("wauzms", 33, 4, 0), ("wauzms-1", 57, 8, 0), ("wauzms-1", 33, 16, 0),
             ("wauzms-1", 57, 24, 0), ("wauzms-2", 49, 28, 0)),
            (("wauzms-2", 33, 4, 0), ("wauzms-1", 57, 12, 0), ("wauzms-1", 33, 20, 0),
             ("wauzms-1", 57, 24, 0), ("wauzms-2", 49, 28, 0)),
            (("wauzms", 41, 4, 0), ("wauzms-1", 33, 8, 0), ("wauzms-1", 57, 16, 0),
             ("wauzms-1", 33, 24, 0), ("wauzms-2", 49, 28, 0)),
        ))
        spec["walls"] = [_item(source, *value) for value in wall_configuration]
        other_platforms = rng.choice((((53, 24), (33, 12)), ((49, 24), (33, 8)), ((53, 20), (37, 12))))
        spec["platforms"] = [
            _item(source, "chrccc", 41, 20),
            *[_item(source, "chrccc", x, y) for x, y in other_platforms],
            _item(source, "chrccc-2", 37, 28),
        ]
        spec["gates"] = [
            {"x": 37, "y": 16, "body_x": 47, "body_y": 16, "rotation": 0,
             "visible": False, "width": 14, "height": 4,
             "body_width": 4, "body_height": 4},
            {"x": 47, "y": 12, "body_x": 47, "body_y": 12, "rotation": 180,
             "visible": False, "width": 14, "height": 4,
             "body_width": 4, "body_height": 4},
        ]
        suffix = [33, 33, 0]
        rng.shuffle(suffix)
        ending = [10, 2, 0]
        rng.shuffle(ending)
        construction = [[2, 2, 1, *suffix], [33, 33, 33, *ending]]
        initial = list(construction[0])

    dummy_target = [30, 33, 270, 1, 15] if difficulty == 1 else [57, 24, 270, 1, 15]
    spec.update(actor=actor, target=dummy_target, initial_program=initial)
    try:
        layout = extract(_context_env(spec))
        world = (layout.current, layout.initial, layout.actor_alive,
                 tuple(gate.visible for gate in layout.gates))
        for program in construction[:-1]:
            world, won, _ = execute_program(layout, world, program)
            if won:
                return None
        end_world, _ = program_end_state(layout, world, construction[-1])
        target = end_world[0]
        if not end_world[2] or target == tuple(actor) or not _valid_target(target, difficulty):
            return None
        spec["target"] = list(target)
    except (KeyError, TypeError, ValueError, IndexError):
        return None

    spec["required_solution_events"] = list(REQUIRED_MECHANIC_EVENTS[difficulty])
    spec["_construction_programs"] = construction
    return spec


def _trace(spec, actions):
    env = _context_env(spec)
    if type(actions) is not list or not 1 <= len(actions) <= env.clicks_left:
        raise ValueError("route must be a nonempty JSON list within the native click budget")
    events = []
    programs = []
    selectors = []
    start_budget = env.clicks_left
    minimum_budget = start_budget
    pre_win_budget = None
    first_win = None
    observation = None
    for index, action in enumerate(actions):
        if type(action) is not list or len(action) != 3:
            raise ValueError(f"solution action {index} must be a three-item JSON list")
        action_id, x, y = action
        if type(action_id) is not int or action_id != names.ACTION_CLICK:
            raise ValueError(f"solution action {index} is not a TN36 click")
        if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
            raise ValueError(f"solution action {index} has invalid coordinates")
        layout = extract(env)
        selected = None
        predicted = None
        if (x, y) in [preset.click for preset in layout.presets]:
            events.append("preset_selection")
            selected = [preset.click for preset in layout.presets].index((x, y))
            selectors.append(selected)
        for clicks in layout.bit_clicks:
            if (x, y) in clicks:
                events.append("bit_toggle")
                break
        if layout.run_click == (x, y):
            events.append("program_run")
            world = (layout.current, layout.initial, layout.actor_alive,
                     tuple(gate.visible for gate in layout.gates))
            predicted, _, emitted = execute_program(layout, world, layout.program)
            events.extend(emitted)
            programs.append(list(layout.program))
        budget_before_action = env.clicks_left
        observation = env.perform(action_id, x, y)
        minimum_budget = min(minimum_budget, env.clicks_left)
        if env.levels_completed > 0 or observation.state == GameState.WIN:
            first_win = index
            pre_win_budget = budget_before_action
            events.append("native_win")
            break
        # Ground the modelled events natively: a selector click must select
        # its preset, and a non-winning run must leave the engine exactly in
        # the world the event model predicted (rollbacks, checkpoints, gates).
        if selected is not None and env.controller.jwmpcflifn != selected:
            raise ValueError(f"solution action {index} did not select preset {selected}")
        if predicted is not None:
            settled = extract(env)
            native = (settled.current, settled.initial, settled.actor_alive,
                      tuple(gate.visible for gate in settled.gates))
            if native != predicted:
                raise ValueError(f"native run outcome after action {index} diverges from the event model")
    if first_win is None:
        raise ValueError("route did not win in its intended native context")
    return {
        "events": event_counts(events),
        "programs_executed": programs,
        "distinct_programs": len({tuple(program) for program in programs}),
        "selector_indices": selectors,
        "wins": 1,
        "first_win_action": first_win,
        "start_budget": start_budget,
        "remaining_budget_before_win": pre_win_budget,
        "end_budget": env.clicks_left,
        "minimum_budget": minimum_budget,
        "levels_completed": env.levels_completed,
        "end_level_index": env.level_index,
        "engine_state": observation.state.name,
    }


def _public_search(spec, work):
    """The one public teacher search: bounded, and constrained to the tier's required events."""
    difficulty = spec["difficulty"]
    return search(
        _context_env(spec),
        limit=REFERENCE_PROFILES[difficulty]["native_budget"],
        node_limit=work,
        required_events=REQUIRED_MECHANIC_EVENTS[difficulty],
    )


def _official_equivalence_ids():
    if not hasattr(_official_equivalence_ids, "value"):
        values = set()
        env = Env()
        for context in range(len(DIFFICULTIES)):
            env.set_level(context)
            layout = extract(env)
            value = {
                "difficulty": context + 1, "context_index": context,
                "bit_width": layout.bit_widths[0],
                "actor": list(layout.initial), "target": list(layout.target),
                "initial_program": list(layout.program),
                "preset_programs": [list(p.program) for p in layout.presets],
                "preset_positions": [list(p.position) for p in layout.presets],
                "preset_rotations": [p.rotation for p in layout.presets],
                "preset_scales": [p.scale for p in layout.presets],
                "preset_resets": [p.reset for p in layout.presets],
                "walls": [{"x": r.x, "y": r.y, "width": r.width, "height": r.height} for r in layout.walls],
                "platforms": [{"x": r.x, "y": r.y, "width": r.width, "height": r.height} for r in layout.platforms],
                "gates": [{"x": g.barrier.x, "y": g.barrier.y, "width": g.barrier.width,
                           "height": g.barrier.height, "body_x": g.body.x,
                           "body_y": g.body.y, "body_width": g.body.width,
                           "body_height": g.body.height,
                           "visible": g.visible} for g in layout.gates],
            }
            try:
                values.add(official_equivalence_identity(value))
            except (KeyError, TypeError, ValueError):
                pass
        _official_equivalence_ids.value = frozenset(values)
    return _official_equivalence_ids.value


def _action_digest(actions):
    return hashlib.sha256(json.dumps(actions, separators=(",", ":")).encode()).hexdigest()


RNG_PROVENANCE_VERSION = "tn36-sequential-draft-v2"
RECOVERY_PROOF_KIND = "native-destructive-prefix-plus-positive-live-recovery-v1"
_DRAFT_FIELDS = (
    "format", "generator_version", "game", "source_id", "vendored_source_sha256",
    "mechanics_version", "quality_profile_version", "geometry_version",
    "gameplay_version", "difficulty", "context_index", "bit_width",
    "native_budget", "walls", "platforms", "gates", "source",
    "left_slot_count", "preset_programs", "preset_positions", "preset_rotations",
    "preset_scales", "preset_resets", "selector_xs", "actor", "target",
    "initial_program", "required_solution_events", "construction_programs",
)


def _typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _draft_payload(spec):
    return {field: spec.get(field) for field in _DRAFT_FIELDS}


def _draft_digest(seed, difficulty, attempt, spec):
    payload = {
        "version": RNG_PROVENANCE_VERSION,
        "seed": seed,
        "difficulty": difficulty,
        "attempt": attempt,
        "draft": _draft_payload(spec),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _recomputed_draft(seed, difficulty, attempt):
    rng = random.Random(f"{MECHANICS_VERSION}:{seed}:{difficulty}")
    candidate = None
    for _ in range(attempt):
        candidate = _draft(rng, difficulty)
    if candidate is None:
        return None
    candidate = dict(candidate)
    programs = candidate.pop("_construction_programs")
    candidate["construction_programs"] = [list(program) for program in programs]
    return candidate


def _recovery_prefix(spec):
    env = _context_env(spec)
    layout = extract(env)
    destructive_program = (10, 33, 33, 33, 0, 0)
    return [list(action) for action in program_actions(
        layout, layout.program, destructive_program
    )]


def _selector_access_errors(spec):
    errors = []
    expected_programs = spec.get("preset_programs", [])
    try:
        access_points = _selector_access_points(spec.get("selector_xs", []))
    except (TypeError, ValueError) as exc:
        return [f"selector access geometry failed: {exc}"]
    for index, (expected, access_point) in enumerate(zip(expected_programs, access_points)):
        try:
            env = _context_env(spec)
            layout = extract(env)
            if index >= len(layout.presets):
                errors.append(f"selector {index} has no extracted preset")
                continue
            env.perform(names.ACTION_CLICK, *access_point)
            if env.controller.jwmpcflifn != index:
                errors.append(f"selector {index} does not select its own preset")
            actual = env.controller.mvqheosngn.vupcwzjtxu.vkuvtkaerv
            if list(actual) != expected:
                errors.append(f"selector {index} exposes the wrong preset program")
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f"selector {index} access check failed: {exc}")
    return errors


def _full_schema_errors(spec, canonical_entry):
    """Cheap strict metadata/proof boundary before native reconstruction."""
    errors = []
    exact_ints = ("seed", "generation_attempt", "difficulty", "context_index",
                  "bit_width", "native_budget", "solution_length", "search_limit",
                  "search_expanded", "search_generated", "split_partition_bucket")
    for field in exact_ints:
        if type(spec.get(field)) is not int:
            errors.append(f"{field} must be an exact integer")
    if (type(spec.get("generation_attempt")) is int
            and not 1 <= spec["generation_attempt"] <= 10_000):
        errors.append("generation_attempt is outside the bounded generator cap")
    for field, expected in (
        ("engine_verified", True), ("search_truncated", False),
    ):
        if spec.get(field) is not expected:
            errors.append(f"{field} must be {expected}")
    if type(spec.get("planner_exact")) is not bool:
        errors.append("planner_exact must be a boolean")
    for field in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256"):
        value = spec.get(field)
        if (type(value) is not str or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            errors.append(f"{field} must be a lowercase SHA-256 hex digest")
    if spec.get("rng_provenance_version") != RNG_PROVENANCE_VERSION:
        errors.append("rng_provenance_version mismatch")
    provenance = spec.get("draft_provenance_sha256")
    if (type(provenance) is not str or len(provenance) != 64
            or any(character not in "0123456789abcdef" for character in provenance)):
        errors.append("draft_provenance_sha256 must be a lowercase SHA-256 hex digest")
    if (type(spec.get("construction_programs")) is not list
            or any(type(program) is not list for program in spec.get("construction_programs", []))):
        errors.append("construction_programs must be a list of JSON lists")
    difficulty = spec.get("difficulty")
    if type(difficulty) is int and difficulty in REFERENCE_PROFILES:
        expected_events = list(REQUIRED_MECHANIC_EVENTS[difficulty])
        if spec.get("required_solution_events") != expected_events:
            errors.append("required_solution_events differs from the fixed tier obligations")
        work = canonical_entry["search_work"]
        expanded = spec.get("search_expanded")
        generated = spec.get("search_generated")
        if type(spec.get("search_limit")) is not int or spec.get("search_limit") != work:
            errors.append("search_limit differs from the canonical tier work cap")
        if type(expanded) is not int or not 1 <= expanded <= work:
            errors.append("search_expanded is outside the canonical work cap")
        generated_cap = work * (1 << REFERENCE_PROFILES[difficulty]["bit_width"])
        if type(generated) is not int or not 1 <= generated <= generated_cap:
            errors.append("search_generated is outside the bounded transition cap")
    for field in ("solution_mechanics", "mechanic_mechanics"):
        if type(spec.get(field)) is not dict:
            errors.append(f"{field} must be a mapping")
    if type(spec.get("mechanic_solution")) is not list or not spec.get("mechanic_solution"):
        errors.append("mechanic_solution must be a nonempty list")
    if difficulty == 7:
        if type(spec.get("recovery_solution")) is not list or not spec.get("recovery_solution"):
            errors.append("recovery_solution must be a nonempty JSON list")
        if type(spec.get("recovery_mechanics")) is not dict:
            errors.append("recovery_mechanics must be a mapping")
    proof = spec.get("proof")
    if type(proof) is not dict:
        errors.append("proof must be a mapping")
    else:
        for field in ("context_engine_verified", "search_truncated",
                      "planner_exact", "shortest_route_claimed"):
            if type(proof.get(field)) is not bool:
                errors.append(f"proof.{field} must be a boolean")
        replay = proof.get("context_engine_replay")
        if type(replay) is not dict:
            errors.append("proof.context_engine_replay must be a mapping")
        else:
            if type(replay.get("levels_completed")) is not int:
                errors.append("proof.context_engine_replay.levels_completed must be an exact integer")
            if type(replay.get("end_level_index")) is not int:
                errors.append("proof.context_engine_replay.end_level_index must be an exact integer")
            if type(replay.get("engine_state")) is not str:
                errors.append("proof.context_engine_replay.engine_state must be a string")
        for field in ("context_index", "action_count", "search_work",
                      "search_expanded", "search_generated", "program_runs",
                      "mechanic_action_count"):
            if type(proof.get(field)) is not int:
                errors.append(f"proof.{field} must be an exact integer")
    return errors


def validate_full_standard(spec, curriculum_entry):
    """Recompute identities, structure, route events, timer and native win."""
    errors = []
    if not isinstance(spec, dict) or not isinstance(curriculum_entry, dict):
        return ["spec and curriculum entry must be mappings"]
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["difficulty must be an integer in 1..7"]
    canonical_entry = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
    if not _typed_equal(curriculum_entry, canonical_entry):
        errors.append("curriculum entry differs from the canonical tier")
    errors.extend(_full_schema_errors(spec, canonical_entry))
    expected = {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "game": "tn36",
        "mechanics_version": MECHANICS_VERSION,
        "quality_profile_version": PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION, "gameplay_version": GAMEPLAY_VERSION,
        "context_index": difficulty - 1, "source": "generated_only",
        "source_id": names.SOURCE_ID, "vendored_source_sha256": SOURCE_SHA256,
    }
    for field, value in expected.items():
        if type(spec.get(field)) is not type(value) or spec.get(field) != value:
            errors.append(f"{field} does not match the full-standard value")
    try:
        seed = spec.get("seed")
        attempt = spec.get("generation_attempt")
        if type(seed) is int and type(attempt) is int and 1 <= attempt <= 10_000:
            actual_digest = _draft_digest(seed, difficulty, attempt, spec)
            candidate = _recomputed_draft(seed, difficulty, attempt)
            expected_digest = None if candidate is None else _draft_digest(
                seed, difficulty, attempt, candidate
            )
            if spec.get("draft_provenance_sha256") != actual_digest:
                errors.append("draft provenance does not match the stored generated content")
            if expected_digest is None or spec.get("draft_provenance_sha256") != expected_digest:
                errors.append("seed/generation_attempt does not reproduce the stored draft")
        mechanic_certificate = spec.get("mechanic_mechanics")
        if isinstance(mechanic_certificate, dict) and not _typed_equal(
            mechanic_certificate.get("programs_executed"),
            spec.get("construction_programs"),
        ):
            errors.append("construction_programs differs from the native mechanic witness")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        errors.append(f"draft provenance validation failed: {exc}")
    split = spec.get("split")
    if split not in SPLITS or spec.get("geometry_split") != split:
        errors.append("split/geometry_split is invalid")
    try:
        build_level(spec)
        env = _context_env(spec)
        layout = extract(env)
        if not layout.exact:
            errors.append("rebuilt layout is unsupported: " + "; ".join(layout.unsupported))
        profile = REFERENCE_PROFILES[difficulty]
        rebuilt_counts = (
            len(layout.program), layout.bit_widths[0], len(layout.presets),
            len(layout.walls), len(layout.platforms), len(layout.gates),
        )
        expected_counts = (
            profile["slots"], profile["bit_width"], profile["selectors"],
            profile["walls"], profile["platforms"], profile["gates"],
        )
        if rebuilt_counts != expected_counts:
            errors.append("rebuilt native controls/obstacles differ from the tier profile")
        if env.clicks_left != REFERENCE_PROFILES[difficulty]["native_budget"]:
            errors.append("rebuilt native timer budget differs from the tier")
        geometry, canonical, gameplay = identities(spec)
        if spec.get("geometry_sha256") != geometry:
            errors.append("geometry_sha256 does not match rebuilt content")
        if spec.get("geometry_d4_sha256") != canonical:
            errors.append("geometry_d4_sha256 does not match canonical content")
        if spec.get("gameplay_sha256") != gameplay:
            errors.append("gameplay_sha256 does not match executable semantics")
        bucket = split_bucket(gameplay)
        if spec.get("split_partition_bucket") != bucket or not split_accepts(split, bucket):
            errors.append("canonical gameplay belongs to a different split")
        if official_equivalence_identity(spec) in _official_equivalence_ids():
            errors.append("generated executable puzzle duplicates an official tier up to presentation translation")
        frame = env.render()
        density = round(sum(pixel != 5 for row in frame for pixel in row) / 4096, 6)
        if spec.get("visual_density") != density:
            errors.append("visual density does not match the native start frame")
        if len(frame) != 64 or any(len(row) != 64 for row in frame):
            errors.append("native frame is not 64x64")
        errors.extend(_selector_access_errors(spec))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        errors.append(f"build/identity validation failed: {exc}")

    raw_actions = spec.get("solution")
    if not isinstance(raw_actions, list) or not raw_actions:
        errors.append("solution must be a nonempty list")
    else:
        try:
            trace = _trace(spec, raw_actions)
            if not _typed_equal(trace, spec.get("solution_mechanics")):
                errors.append("solution mechanic certificate does not match native replay")
            if trace["first_win_action"] != len(raw_actions) - 1:
                errors.append("native route wins before its stored final action")
            if spec.get("solution_length") != len(raw_actions):
                errors.append("solution_length does not match the route")
            difficulty_result = _public_search(spec, canonical_entry["search_work"])
            parsed_actions = tuple(tuple(action) for action in raw_actions)
            if (not difficulty_result.solved or difficulty_result.truncated
                    or difficulty_result.unsupported):
                errors.append("bounded public difficulty search did not reproduce a solution")
            elif difficulty_result.actions != parsed_actions:
                errors.append("stored solution differs from the independently recomputed public teacher")
            if spec.get("search_expanded") != difficulty_result.expanded:
                errors.append("search_expanded differs from recomputed public search")
            if spec.get("search_generated") != difficulty_result.generated:
                errors.append("search_generated differs from recomputed public search")
            if spec.get("planner_exact") is not difficulty_result.exact:
                errors.append("planner_exact differs from recomputed public search")
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f"solution replay failed: {exc}")
    mechanic_actions = spec.get("mechanic_solution")
    if not isinstance(mechanic_actions, list) or not mechanic_actions:
        errors.append("mechanic_solution must be a nonempty list")
    else:
        try:
            mechanic_trace = _trace(spec, mechanic_actions)
            if not _typed_equal(mechanic_trace, spec.get("mechanic_mechanics")):
                errors.append("mechanic certificate does not match native replay")
            if mechanic_trace["first_win_action"] != len(mechanic_actions) - 1:
                errors.append("mechanic route wins before its stored final action")
            for event in REQUIRED_MECHANIC_EVENTS[difficulty]:
                if mechanic_trace["events"].get(event, 0) < 1:
                    errors.append(f"mechanic route does not exercise {event}")
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f"mechanic replay failed: {exc}")
    recovery_trace = None
    if difficulty == 7:
        recovery = spec.get("recovery_solution")
        if not isinstance(recovery, list) or not recovery:
            errors.append("tier 7 requires a gate-destruction recovery certificate")
        else:
            try:
                recovery_trace = _trace(spec, recovery)
                if not _typed_equal(recovery_trace, spec.get("recovery_mechanics")):
                    errors.append("recovery mechanic certificate does not match native replay")
                recovery_events = recovery_trace["events"]
                for event in ("gate_destroy", "failed_run_reset", "native_win"):
                    if recovery_events.get(event, 0) < 1:
                        errors.append(f"recovery route does not exercise {event}")
                if recovery_trace["first_win_action"] != len(recovery) - 1:
                    errors.append("recovery route wins before its stored final action")
            except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
                errors.append(f"recovery replay failed: {exc}")
    try:
        errors.extend(profile_errors(spec))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"profile validation failed: {exc}")
    proof = spec.get("proof")
    if isinstance(proof, dict):
        required_proof = {
            "context_index": difficulty - 1,
            "context_engine_verified": True,
            "search_truncated": False,
            "action_count": spec.get("solution_length"),
            "search_work": canonical_entry["search_work"],
            "geometry_split": split,
            "shortest_route_claimed": False,
            "solution_sha256": _action_digest(raw_actions) if isinstance(raw_actions, list) else None,
            "mechanic_action_count": len(mechanic_actions) if isinstance(mechanic_actions, list) else None,
            "mechanic_solution_sha256": (
                _action_digest(mechanic_actions) if isinstance(mechanic_actions, list) else None
            ),
        }
        for field, value in required_proof.items():
            if not _typed_equal(proof.get(field), value):
                errors.append(f"proof.{field} mismatch")
        for field in ("search_expanded", "search_generated"):
            if type(proof.get(field)) is not int or proof[field] <= 0:
                errors.append(f"proof.{field} must be a positive integer")
            if proof.get(field) != spec.get(field):
                errors.append(f"proof.{field} differs from top-level evidence")
        if proof.get("planner_exact") is not spec.get("planner_exact"):
            errors.append("proof.planner_exact differs from top-level evidence")
        solution_certificate = spec.get("solution_mechanics")
        replayed_program_runs = None
        if isinstance(solution_certificate, dict):
            replayed_events = solution_certificate.get("events")
            if isinstance(replayed_events, dict):
                replayed_program_runs = replayed_events.get("program_run")
        if proof.get("program_runs") != replayed_program_runs:
            errors.append("proof.program_runs differs from the replayed route")
        expected_replay = None
        if isinstance(spec.get("solution_mechanics"), dict):
            expected_replay = {
                "levels_completed": spec["solution_mechanics"].get("levels_completed"),
                "end_level_index": spec["solution_mechanics"].get("end_level_index"),
                "engine_state": spec["solution_mechanics"].get("engine_state"),
            }
        if not _typed_equal(proof.get("context_engine_replay"), expected_replay):
            errors.append("proof.context_engine_replay differs from native route replay")
        if difficulty == 7:
            recovery_proof = proof.get("recovery")
            if not isinstance(recovery_proof, dict):
                errors.append("proof.recovery must be a mapping")
            else:
                recovery_actions = spec.get("recovery_solution")
                recovery_count = len(recovery_actions) if type(recovery_actions) is list else None
                expected_prefix = None
                try:
                    expected_prefix = _recovery_prefix(spec)
                except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
                    errors.append(f"recovery prefix reconstruction failed: {exc}")
                recovery_expected = {
                    "kind": RECOVERY_PROOF_KIND,
                    "action_count": recovery_count,
                    "solution_sha256": (
                        _action_digest(recovery_actions)
                        if type(recovery_actions) is list else None
                    ),
                    "destructive_prefix_action_count": (
                        len(expected_prefix) if expected_prefix is not None else None
                    ),
                    "destructive_prefix_sha256": (
                        _action_digest(expected_prefix) if expected_prefix is not None else None
                    ),
                    "context_engine_verified": True,
                    "gate_destroyed": True,
                    "native_win": True,
                }
                for field, value in recovery_expected.items():
                    if not _typed_equal(recovery_proof.get(field), value):
                        errors.append(f"proof.recovery.{field} mismatch")
                for field in ("search_work", "search_expanded", "search_generated", "search_truncated"):
                    if field in recovery_proof:
                        errors.append(f"proof.recovery.{field} is an unsupported recovery-work claim")
                if (type(recovery_actions) is list and expected_prefix is not None
                        and recovery_actions[:len(expected_prefix)] != expected_prefix):
                    errors.append("recovery route does not begin with the reconstructed destructive prefix")
                if recovery_trace is not None:
                    if recovery_trace["events"].get("gate_destroy", 0) < 1:
                        errors.append("proof.recovery gate claim lacks native trace evidence")
    return errors


def generate(seed, difficulty=1, attempts=DEFAULT_ATTEMPTS, node_limit=None, *, split="train",
             record_rejection=None):
    """Generate one bounded, reference-tiered, native-replayed TN36 level."""
    seed = _integer(seed, "seed")
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts")
    if difficulty not in DIFFICULTIES or not 1 <= attempts <= 10_000 or split not in SPLITS:
        raise ValueError("difficulty/split/attempts are outside the full generator contract")
    tier_work = REFERENCE_PROFILES[difficulty]["search_work"]
    requested_work = tier_work if node_limit is None else _integer(node_limit, "node_limit")
    if requested_work < tier_work:
        raise ValueError("node_limit cannot lower the full-standard tier search gate")
    if requested_work > 32_000_000:
        raise ValueError("node_limit exceeds the bounded full-standard cap")
    work = tier_work
    rng = random.Random(f"{MECHANICS_VERSION}:{seed}:{difficulty}")
    rejected = Counter()
    for attempt in range(1, attempts + 1):
        spec = _draft(rng, difficulty)
        if spec is None:
            rejected["invalid_geometry"] += 1
            continue
        construction_programs = [list(program) for program in spec.pop("_construction_programs")]
        spec.update(
            seed=seed, generation_attempt=attempt, split=split,
            construction_programs=construction_programs,
            rng_provenance_version=RNG_PROVENANCE_VERSION,
        )
        spec["draft_provenance_sha256"] = _draft_digest(seed, difficulty, attempt, spec)
        try:
            geometry, canonical, gameplay = identities(spec)
            bucket = split_bucket(gameplay)
            if not split_accepts(split, bucket):
                rejected["gameplay_split"] += 1
                continue
            if official_equivalence_identity(spec) in _official_equivalence_ids():
                rejected["official_equivalence"] += 1
                continue
            spec.update(
                geometry_sha256=geometry, geometry_d4_sha256=canonical,
                gameplay_sha256=gameplay, geometry_split=split,
                split_partition_bucket=bucket,
            )
            result = _public_search(spec, work)
            if result.truncated:
                rejected["search_truncated"] += 1
                continue
            if result.unsupported:
                rejected["unsupported_layout"] += 1
                continue
            if not result.solved:
                rejected["proven_unreachable"] += 1
                continue
            public_actions = list(result.actions)
            low, high = REFERENCE_PROFILES[difficulty]["actions"]
            if not low <= len(public_actions) <= high:
                rejected["public_route_band"] += 1
                continue

            # The public teacher and its search proof are exactly this fresh-
            # state route, and the search is constrained so the route itself
            # exercises the tier's required mechanic events. The separate
            # construction replay below is an additional witness only.
            spec["solution"] = [list(action) for action in public_actions]
            spec["solution_length"] = len(public_actions)
            spec["solution_mechanics"] = _trace(spec, spec["solution"])
            if any(spec["solution_mechanics"]["events"].get(event, 0) < 1
                   for event in REQUIRED_MECHANIC_EVENTS[difficulty]):
                rejected["public_route_events"] += 1
                continue

            witness_env = _context_env(spec)
            mechanic_actions = []
            mechanic_layout = extract(witness_env)
            prefix = []
            if mechanic_layout.presets:
                selector = mechanic_layout.presets[(seed + attempt) % len(mechanic_layout.presets)].click
                prefix.append((names.ACTION_CLICK, *selector))
            for action in prefix:
                witness_env.perform(*action)
                mechanic_actions.append(action)
            early_win = False
            for program_index, program in enumerate(construction_programs):
                live = extract(witness_env)
                encoded = program_actions(live, live.program, program)
                for action_index, action in enumerate(encoded):
                    observation = witness_env.perform(*action)
                    mechanic_actions.append(action)
                    won = witness_env.levels_completed > 0 or observation.state == GameState.WIN
                    final = (program_index == len(construction_programs) - 1
                             and action_index == len(encoded) - 1)
                    if won and not final:
                        early_win = True
                        break
                if early_win:
                    break
            if early_win or witness_env.levels_completed != 1:
                rejected["construction_replay"] += 1
                continue
            spec["mechanic_solution"] = [list(action) for action in mechanic_actions]
            spec["mechanic_mechanics"] = _trace(spec, spec["mechanic_solution"])
            spec["visual_density"] = round(
                sum(pixel != 5 for row in _context_env(spec).render() for pixel in row) / 4096, 6)
            spec.update(
                engine_verified=True, search_truncated=False,
                planner_exact=bool(result.exact), search_expanded=result.expanded,
                search_generated=result.generated, search_limit=work,
            )
            recovery_result = None
            recovery_prefix = None
            if difficulty == 7:
                recovery_env = _context_env(spec)
                recovery_prefix = _recovery_prefix(spec)
                for action in recovery_prefix:
                    observation = recovery_env.perform(*action)
                if recovery_env.levels_completed or observation.state == GameState.WIN:
                    rejected["recovery_probe_won"] += 1
                    continue
                recovery_result = search(
                    recovery_env, limit=recovery_env.clicks_left, node_limit=work)
                if not recovery_result.solved or recovery_result.truncated:
                    rejected["recovery_search"] += 1
                    continue
                recovery_actions = recovery_prefix + [list(action) for action in recovery_result.actions]
                spec["recovery_solution"] = recovery_actions
                spec["recovery_mechanics"] = _trace(spec, spec["recovery_solution"])
            spec["proof"] = {
                "context_index": difficulty - 1, "context_engine_verified": True,
                "search_truncated": False, "planner_exact": bool(result.exact),
                "shortest_route_claimed": False,
                "action_count": len(public_actions), "search_work": work,
                "search_expanded": result.expanded, "search_generated": result.generated,
                "program_runs": spec["solution_mechanics"]["events"].get("program_run", 0),
                "geometry_split": split,
                "solution_sha256": hashlib.sha256(json.dumps(
                    spec["solution"], separators=(",", ":")
                ).encode()).hexdigest(),
                "mechanic_action_count": len(spec["mechanic_solution"]),
                "mechanic_solution_sha256": hashlib.sha256(json.dumps(
                    spec["mechanic_solution"], separators=(",", ":")
                ).encode()).hexdigest(),
                "context_engine_replay": {
                    "levels_completed": spec["solution_mechanics"]["levels_completed"],
                    "end_level_index": spec["solution_mechanics"]["end_level_index"],
                    "engine_state": spec["solution_mechanics"]["engine_state"],
                },
            }
            if recovery_result is not None:
                spec["proof"]["recovery"] = {
                    "kind": RECOVERY_PROOF_KIND,
                    "action_count": len(spec["recovery_solution"]),
                    "solution_sha256": _action_digest(spec["recovery_solution"]),
                    "destructive_prefix_action_count": len(recovery_prefix),
                    "destructive_prefix_sha256": _action_digest(recovery_prefix),
                    "context_engine_verified": True,
                    "gate_destroyed": True,
                    "native_win": True,
                }
            errors = validate_full_standard(
                spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1])
            if errors:
                rejected["validation"] += 1
                continue
            spec["generation_rejections"] = dict(rejected)
            generate.last_report = {
                "accepted": True, "seed": seed, "difficulty": difficulty,
                "split": split, "attempt": attempt, "search_work": work,
                "rejections": dict(rejected),
            }
            return spec
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            rejected[f"exception:{type(exc).__name__}"] += 1
            if record_rejection:
                record_rejection({"seed": seed, "difficulty": difficulty,
                                  "attempt": attempt, "reason": type(exc).__name__})
    generate.last_report = {
        "accepted": False, "seed": seed, "difficulty": difficulty,
        "split": split, "attempts": attempts, "search_work": work,
        "rejections": dict(rejected),
    }
    return None


generate.last_report = None


def _child_seed(parent, ordinal, difficulty):
    payload = f"{MECHANICS_VERSION}:{parent}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split="train", difficulties=None, attempts=DEFAULT_ATTEMPTS,
                  node_limit=None):
    """Generate all seven increasing native tiers, or an explicit smoke subset."""
    seed = _integer(seed, "seed")
    requested = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if (not requested or any(type(value) is not int or value not in DIFFICULTIES for value in requested)
            or tuple(sorted(set(requested))) != requested):
        raise ValueError("difficulties must be a nonempty, unique, increasing tier sequence")
    specs = []
    reports = []
    for ordinal, difficulty in enumerate(requested):
        child = _child_seed(seed, ordinal, difficulty)
        spec = generate(child, difficulty, attempts=attempts, node_limit=node_limit, split=split)
        reports.append(generate.last_report)
        if spec is None:
            generate_game.last_report = {
                "accepted": False, "game_seed": seed, "split": split,
                "failed_difficulty": difficulty, "tier_reports": reports,
            }
            return None
        spec.update(game_seed=seed, game_position=ordinal, child_seed=child)
        specs.append(spec)
    if requested != DIFFICULTIES:
        for spec in specs:
            spec["sequence_kind"] = "explicit-smoke-subset"
        generate_game.last_report = {"accepted": True, "smoke": True, "tier_reports": reports}
        return specs
    try:
        build_game(specs)
    except ValueError as exc:
        generate_game.last_report = {
            "accepted": False, "game_seed": seed, "split": split,
            "reason": str(exc), "tier_reports": reports,
        }
        return None
    sequence_hash = hashlib.sha256(json.dumps(
        [spec["gameplay_sha256"] for spec in specs], separators=(",", ":")
    ).encode()).hexdigest()
    for spec in specs:
        spec.update(sequence_kind="full-official-context", game_sequence_sha256=sequence_hash)
    generate_game.last_report = {"accepted": True, "smoke": False, "tier_reports": reports}
    return specs


generate_game.last_report = None


def build_game(specs):
    """Validate and replay exactly seven independently generated ordered tiers."""
    if not isinstance(specs, (list, tuple)) or len(specs) != len(DIFFICULTIES):
        raise ValueError("a full TN36 game requires exactly seven specs")
    splits = {spec.get("split") for spec in specs if isinstance(spec, dict)}
    if len(splits) != 1 or next(iter(splits), None) not in SPLITS:
        raise ValueError("full-game specs must share one valid split")
    seen_gameplay = set()
    for index, (spec, entry) in enumerate(zip(specs, FULL_STANDARD_CONTRACT["curriculum"])):
        errors = validate_full_standard(spec, entry)
        if errors:
            raise ValueError(f"invalid tier {index + 1}: {'; '.join(errors)}")
        if spec["difficulty"] != index + 1:
            raise ValueError("full-game difficulties must be exactly 1..7")
        if spec["gameplay_sha256"] in seen_gameplay:
            raise ValueError("full game contains duplicate executable gameplay")
        seen_gameplay.add(spec["gameplay_sha256"])
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    env.reset()
    observation = None
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            raise ValueError(f"native episode did not enter tier {index + 1}")
        start = env.levels_completed
        for action_index, action in enumerate(spec["solution"]):
            observation = env.perform(*action)
            if env.levels_completed > start:
                if action_index != len(spec["solution"]) - 1:
                    raise ValueError(f"tier {index + 1} wins before its final action")
                break
        if env.levels_completed != index + 1:
            raise ValueError(f"native episode failed at tier {index + 1}")
    if observation is None or observation.state != GameState.WIN:
        raise ValueError("native seven-level episode did not finish with WIN")
    return [build_level(spec) for spec in specs]
