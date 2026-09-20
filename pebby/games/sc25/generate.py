"""Full six-tier, reference-calibrated SC25 procedural generation.

The default curriculum mirrors the mechanic composition and native budgets of
the six shipped levels without copying their geometry or routes. Every accepted
row has a bounded symbolic witness, an actual-context native replay, a mechanic
use certificate, and canonical geometry/gameplay identities.
"""

from collections import Counter
from functools import lru_cache
import hashlib
import json
import random

from arcengine import Level, Sprite

from . import names
from .env import Env, official_levels, upstream
from .layout import KIND_PICKUP, KIND_TARGET, KIND_TARGET_ALT, PAD, extract
from .plan import _cast, _move, _spell_actions, search
from .reference_profiles import (
    DIFFICULTIES,
    PROFILE_VERSION,
    REFERENCE_PROFILES,
    profile_errors,
)


FORMAT = "pebby.sc25.level.v2"
GENERATOR_VERSION = 3
MECHANICS_VERSION = "sc25-full-six-v2-target-covered-pickups"
# Keep the already-reviewed tiers 1--5 on their prior deterministic streams.
# Tier 6 has a new grammar below, so using the same stream key there does not
# preserve an old puzzle or certificate.
SEED_VERSION = "sc25-full-six-v1"
GEOMETRY_VERSION = "sc25-wall-d4-v1"
GAMEPLAY_VERSION = "sc25-gameplay-v1"
CAUSALITY_VERSION = "sc25-target-covered-pickup-native-v1"
MAX_ATTEMPTS = 72
SPLITS = ("train", "validation", "test")

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": "sc25-635fd71a",
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": PROFILE_VERSION,
    "curriculum": tuple(
        {"difficulty": difficulty, "context_index": difficulty - 1, "search_work": 400_000}
        for difficulty in DIFFICULTIES
    ),
    "evidence": {
        "official_tier_characterization": "sc25.md#official-reference-characterization",
        "solution_mechanics": "spec.solution_mechanics and tests/games/test_sc25_quality.py",
        "native_budget": "reference_profiles.py:REFERENCE_PROFILES",
        "context_engine_replay": "spec.proof.context_engine_verified",
        "novelty_split": "geometry_d4_sha256/geometry_split/gameplay_sha256",
        "bounded_rejections": "spec.generation_rejections and generate(attempts=...)",
        "tier6_covered_pickup_causality": (
            "spec.solution_causality/sc25-target-covered-pickup-native-v1"
        ),
        "root_acceptance": "sc25.md#root-acceptance",
    },
    "caveats": (
        "Each tier has one official reference; tolerances are explicit scarce-reference bands.",
        "Witness lengths are constructive replay lengths, not shortest-route claims.",
        "Target-covered-pickup causality is certified for the checked route prefixes and suffixes; it is not a universal impossibility proof over every alternate route.",
        "Pickup refunds are replay-verified, but the constructive witnesses do not prove each refund is required to remain within the native budget.",
        "Procedural diversity is finite and deliberately limited to the documented room, corridor, and stage grammars.",
        "The engine's crzdcq ring and failed casts occur in no shipped level and are not assigned an official frequency.",
    ),
}

_FRAME_PARTS = (
    ("clcbko-1", 22, 47), ("clcbko-1", 33, 47),
    ("clcbko-1", 22, 58), ("clcbko-1", 33, 58),
    ("clcbko-2", 28, 47), ("clcbko-2", 28, 58),
    ("clcbko-3", 22, 53), ("clcbko-3", 33, 53),
    ("clcbko-4", 28, 53),
)


def _wall_canvas(filled=True):
    value = "#" if filled else "."
    return [[value for _ in range(64)] for _ in range(64)]


def _carve(wall, x, y, width, height):
    for py in range(max(0, y), min(64, y + height)):
        for px in range(max(0, x), min(64, x + width)):
            wall[py][px] = "."


def _finish(wall, difficulty, **fields):
    spec = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_profile_version": PROFILE_VERSION,
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "wall": ["".join(row) for row in wall],
        "blocks": [], "targets": [], "pickups": [], "pads": [],
        "small_pads": [], "rings": [], "source": "generated_only",
        **fields,
    }
    spec["visual_density"] = round(
        sum(row.count("#") for row in spec["wall"]) / (64 * 64), 6
    )
    return spec


def _vertical_spur(wall, rng, xs, y, width, *, upward=True):
    """Carve a reachable optional branch off a known route or room edge."""
    x = rng.choice(tuple(xs))
    length = rng.choice((2, 4, 6, 8))
    if upward:
        _carve(wall, x, y - length, width, length + 1)
    else:
        _carve(wall, x, y - 1, width, length + 1)


def _monotone_cells(rng, start, horizontal, vertical, unit):
    """Random shortest path with signed horizontal/vertical step counts."""
    moves = []
    sx = 1 if horizontal >= 0 else -1
    sy = 1 if vertical >= 0 else -1
    moves.extend((sx * unit, 0) for _ in range(abs(horizontal)))
    moves.extend((0, sy * unit) for _ in range(abs(vertical)))
    rng.shuffle(moves)
    cells = [start]
    x, y = start
    for dx, dy in moves:
        x, y = x + dx, y + dy
        cells.append((x, y))
    return cells


def _draft_tier1(rng):
    wall = _wall_canvas()
    horizontal = rng.choice((-9, -8, -7, 7, 8, 9))
    vertical = rng.choice((-4, -3, -2, 2, 3, 4))
    cells = _monotone_cells(rng, (30, 28), horizontal, vertical, 2)
    for x, y in cells:
        _carve(wall, x, y, 2, 2)
    start_x, start_y = cells[0]
    door_x, door_y = cells[-1]
    _carve(wall, start_x - 1, start_y - 1, 6, 6)
    _carve(wall, door_x - 2, door_y - 2, 7, 7)
    # Reachable branch geometry varies navigable topology without padding.
    branch_x, branch_y = cells[len(cells) // 2]
    _carve(wall, branch_x - 2, branch_y + (2 if vertical < 0 else -6), 5, 6)
    _carve(wall, start_x + (-12 if horizontal > 0 else 4), start_y - 6, 10, 12)
    return _finish(wall, 1, player={"x": start_x, "y": start_y, "scale": 2},
                   door={"x": door_x, "y": door_y}, spells=[names.SPELL_GROW],
                   budget=50, mechanic="shrink-corridor")


def _draft_tier2(rng):
    wall = _wall_canvas()
    horizontal = rng.choice((2, 3, 4, 5))
    vertical = rng.choice((-3, -2, -1, 1, 2, 3))
    cells = _monotone_cells(rng, (10, 28), horizontal, vertical, 4)
    for x, y in cells:
        _carve(wall, x, y, 4, 4)
    pad_x, pad_y = cells[0]
    door_x, door_y = cells[-1]
    _carve(wall, pad_x - 3, pad_y - 3, 10, 10)
    _carve(wall, door_x - 3, door_y - 3, 10, 10)
    mid_x, mid_y = cells[len(cells) // 2]
    _carve(wall, mid_x, mid_y + (4 if vertical < 0 else -8), 12, 8)
    start_y = rng.choice((8, 12, 36, 40))
    _carve(wall, 48, start_y, 12, 10)
    return _finish(wall, 2, player={"x": 52, "y": start_y + 2, "scale": 2},
                   door={"x": door_x, "y": door_y}, pads=[{"x": pad_x, "y": pad_y}],
                   spells=[names.SPELL_TELEPORT], budget=25,
                   mechanic="large-teleport")


def _draft_tier3(rng):
    wall = _wall_canvas()
    y = rng.choice((18, 22, 26, 30))
    start_offset = rng.choice((-8, -4, 0, 4, 8))
    door_offset = rng.choice((-8, -4, 0, 4, 8))
    target_x = rng.choice((16, 20))
    block_x = target_x + rng.choice((8, 12))
    door_x = block_x + rng.choice((8, 12))
    _carve(wall, 8, min(y, y + start_offset), 4, abs(start_offset) + 4)
    _carve(wall, 8, y, door_x - 4, 4)
    _carve(wall, door_x, min(y, y + door_offset), 4, abs(door_offset) + 4)
    _carve(wall, 5, y + start_offset - 3, 10, 10)
    _carve(wall, door_x - 3, y + door_offset - 3, 10, 10)
    _carve(wall, 24, y + (4 if rng.randrange(2) else -10), 12, 10)
    return _finish(
        wall, 3, player={"x": 8, "y": y + start_offset, "scale": 2},
        door={"x": door_x, "y": y + door_offset},
        blocks=[{"x": block_x, "y": y, "rotation": 0, "family": "primary"}],
        targets=[{"x": target_x, "y": y, "family": "primary"}],
        spells=[names.SPELL_FIRE], budget=50, mechanic="primary-fire-gate")


def _draft_tier4(rng):
    wall = _wall_canvas()
    horizontal = rng.choice((-1, 1))
    vertical = rng.choice((-1, 1))
    # Multi-spell icon frames occupy x=0..10 and are collidable.  Keep every
    # gameplay chamber clear of that native UI strip.
    x = 50 if horizontal < 0 else 14
    start_y = 8 if vertical > 0 else 40
    shaft_length = rng.choice((12, 14, 16, 18))
    bottom_y = start_y + vertical * shaft_length
    path_y = bottom_y - 2
    _carve(wall, x - 2, start_y - 2, 8, 8)
    _carve(wall, x, min(start_y, bottom_y), 2, abs(bottom_y - start_y) + 2)
    _carve(wall, x - 3, bottom_y - 3, 10, 10)
    door_x = 4 if horizontal < 0 else 55
    _carve(wall, min(door_x, x), path_y, abs(x - door_x) + 6, 4)
    _carve(wall, door_x - 2, path_y - 2, 9, 9)
    target_x = x + horizontal * rng.choice((12, 16, 20))
    block_x = x + horizontal * rng.choice((28, 32, 36))
    room_anchor = x + horizontal * 8
    room_y = path_y + (2 if vertical > 0 else -16)
    _carve(wall, room_anchor - 6, room_y, 14, 16)
    _vertical_spur(wall, rng, range(22, 39, 4), path_y, 4,
                   upward=bool(rng.randrange(2)))
    return _finish(
        wall, 4, player={"x": x, "y": start_y, "scale": 2},
        door={"x": door_x, "y": path_y - 1},
        blocks=[{"x": block_x, "y": path_y, "rotation": 0, "family": "primary"}],
        targets=[{"x": target_x, "y": path_y, "family": "primary"}],
        # The two-pixel shaft is the only route out after shrinking; putting
        # the pickup at its midpoint makes the native budget refund causal.
        pickups=[{"x": x, "y": start_y + vertical * (shaft_length // 2)}],
        spells=[names.SPELL_FIRE, names.SPELL_GROW], budget=35,
        mechanic="shrink-grow-pickup-fire")


def _draft_tier5(rng):
    wall = _wall_canvas()
    # A and B are disconnected target chambers; C is the only exit chamber.
    # The two variants reverse which scale-gated teleport must be used first,
    # so this is interaction-order diversity rather than decorative noise.
    small_first = bool(rng.randrange(2))
    ax = rng.choice((12, 14))
    aw = rng.choice((24, 26))
    ah = rng.choice((20, 22))
    _carve(wall, ax, 4, aw, ah)
    bx = rng.choice((16, 18))
    bw = rng.choice((26, 28))
    _carve(wall, bx, 28, bw, 18)

    c_left = rng.choice((50, 52))
    pad_y = rng.choice((28, 32))
    _carve(wall, c_left, 2, 4, pad_y + 6)
    door_x = c_left

    a_sx, a_sy = rng.choice((-1, 1)), rng.choice((-1, 1))
    a_player = (ax + (4 if a_sx > 0 else aw - 8), 8 if a_sy > 0 else 18)
    a_target = (a_player[0] + 12 * a_sx, a_player[1] + 12 * a_sy)
    b_sx, b_sy = rng.choice((-1, 1)), rng.choice((-1, 1))
    b_start = (bx + (6 if b_sx > 0 else bw - 10), 32 if b_sy > 0 else 40)
    b_target = (b_start[0] + 16 * b_sx, b_start[1] + 8 * b_sy)

    first_family = "primary" if small_first else "alternate"
    second_family = "alternate" if small_first else "primary"
    pads = [{"x": door_x, "y": pad_y}] if small_first else [
        {"x": b_start[0], "y": b_start[1]}]
    small_pads = [{"x": b_start[0], "y": b_start[1]}] if small_first else [
        {"x": door_x, "y": pad_y}]
    # When B is reached small, it must grow before the large jump to C.  When
    # B is reached large, it must shrink before the small jump to C.
    return _finish(
        wall, 5, player={"x": a_player[0], "y": a_player[1], "scale": 2},
        door={"x": door_x, "y": 3},
        blocks=[
            {"x": door_x, "y": 10, "rotation": 0, "family": "primary"},
            {"x": door_x, "y": 14, "rotation": 0, "family": "alternate"}],
        targets=[
            {"x": a_target[0], "y": a_target[1], "family": first_family},
            {"x": b_target[0], "y": b_target[1], "family": second_family}],
        pickups=[{"x": door_x,
                  "y": pad_y - (12 if small_first else 10)}], pads=pads,
        small_pads=small_pads,
        spells=[names.SPELL_FIRE, names.SPELL_TELEPORT, names.SPELL_GROW],
        budget=65,
        mechanic=("full-composition-small-first" if small_first
                  else "full-composition-large-first"))


def _draft_tier6(rng):
    wall = _wall_canvas()
    # A is a scale-2 start room.  The only useful first destination is the
    # scale-1 pad in B, so reaching it requires the shrink spell.
    a_left = rng.choice((14, 16, 18))
    a_top = rng.choice((4, 6))
    _carve(wall, a_left, a_top, 10, 10)
    a_start = (a_left + 3, a_top + 3)

    # B is a two-pixel corridor ending at a target-covered pickup.  The target
    # blocks the route into the only four-pixel growth chamber.  Consequently
    # the native order is fire target -> move onto/collect pickup -> grow.
    b_sx = rng.choice((-1, 1))
    b_y = rng.choice((20, 24, 28))
    b_steps = rng.choice((3, 4, 5))
    if b_sx > 0:
        b_pad = (12, b_y + 1)
        b_target = (b_pad[0] + 2 * b_steps, b_y)
        _carve(wall, b_pad[0], b_y + 1, b_target[0] - b_pad[0] + 2, 2)
        _carve(wall, b_target[0], b_y - 1, 12, 8)
    else:
        b_pad = (50, b_y + 1)
        b_target = (b_pad[0] - 2 * b_steps - 2, b_y)
        _carve(wall, b_target[0], b_y + 1, b_pad[0] - b_target[0] + 2, 2)
        _carve(wall, b_target[0] - 8, b_y - 1, 12, 8)

    # The scale-2 cursor must visit isolated D before C.  C is a broad final
    # corridor whose second target-covered pickup and the two family blocks
    # physically gate the exit.  Signed direction and spacing vary while all
    # sprites remain fully inside the 64x64 frame.
    d_pad = (rng.choice((40, 42, 44)), rng.choice((6, 8)))
    _carve(wall, d_pad[0] - 3, d_pad[1] - 3, 10, 10)
    c_sx = rng.choice((-1, 1))
    c_y = rng.choice((36, 38, 40))
    c_steps = rng.choice((2, 3, 4))
    if c_sx > 0:
        c_pad = (12, c_y)
        c_target = (c_pad[0] + 4 * c_steps, c_y)
        block_positions = ((c_target[0] + 8, c_y), (c_target[0] + 16, c_y))
        door = (56, c_y - 1)
    else:
        c_pad = (48, c_y)
        c_target = (c_pad[0] - 4 * c_steps, c_y)
        block_positions = ((c_target[0] - 8, c_y), (c_target[0] - 16, c_y))
        door = (3, c_y - 1)
    _carve(wall, 3, c_y, 58, 4)

    first_family = rng.choice(("primary", "alternate"))
    second_family = "alternate" if first_family == "primary" else "primary"
    return _finish(
        wall, 6, player={"x": a_start[0], "y": a_start[1], "scale": 2},
        door={"x": door[0], "y": door[1]},
        blocks=[
            {"x": block_positions[0][0], "y": block_positions[0][1],
             "rotation": 0, "family": "primary"},
            {"x": block_positions[1][0], "y": block_positions[1][1],
             "rotation": 0, "family": "alternate"}],
        targets=[
            {"x": b_target[0], "y": b_target[1], "family": first_family},
            {"x": c_target[0], "y": c_target[1], "family": second_family}],
        pickups=[
            {"x": b_target[0], "y": b_target[1]},
            {"x": c_target[0], "y": c_target[1]}],
        pads=[{"x": d_pad[0], "y": d_pad[1]},
              {"x": c_pad[0], "y": c_pad[1]}],
        small_pads=[{"x": b_pad[0], "y": b_pad[1]}],
        spells=[names.SPELL_FIRE, names.SPELL_TELEPORT, names.SPELL_GROW],
        budget=60, mechanic="full-composition-target-covered-pickups")


_DRAFTERS = {1: _draft_tier1, 2: _draft_tier2, 3: _draft_tier3,
             4: _draft_tier4, 5: _draft_tier5, 6: _draft_tier6}


def _spell_ui(spells):
    if not spells:
        return []
    protos = upstream().sprites
    result = []
    for name, x, y in _FRAME_PARTS:
        result.append(protos[name].clone().set_position(x, y))
    for row in range(3):
        for col in range(3):
            result.append(protos[names.SPRITE_GRID_CELL].clone()
                          .set_position(24 + 5 * col, 49 + 5 * row))
    positions = [(11, 50)] if len(spells) == 1 else [(0, 11 * index) for index in range(len(spells))]
    for spell, (x, y) in zip(spells, positions):
        result.append(protos[names.SPRITE_ICON_FRAME].clone().set_position(x, y))
        icon_x = x + (4 if spell == names.SPELL_FIRE else 1)
        result.append(protos[names.SPRITE_ICON_PREFIX + spell].clone().set_position(icon_x, y + 1))
    result.append(protos[names.SPRITE_GRID_PANEL].clone().set_position(22, 47))
    return result


def build_level(spec):
    """Reconstruct one native ARCEngine level from a JSON-round-trippable spec."""
    if spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    wall = spec.get("wall")
    if not isinstance(wall, list) or len(wall) != 64 or any(
        not isinstance(row, str) or len(row) != 64 or set(row) - {"#", "."} for row in wall
    ):
        raise ValueError("wall must be 64 strings of 64 '#' or '.' characters")
    spells = list(spec.get("spells", []))
    if len(spells) != len(set(spells)) or any(spell not in names.PATTERNS for spell in spells):
        raise ValueError("spec contains duplicate or unknown spells")
    protos = upstream().sprites
    sprites = [protos[names.SPRITE_BUDGET_BAR].clone().set_position(62, 0).set_scale(2)]
    sprites.extend(_spell_ui(spells))
    pixels = [[names.WALL_COLOR if ch == "#" else -1 for ch in row] for row in wall]
    sprites.append(Sprite(pixels, name=names.SPRITE_WALL_PREFIX + "generated", x=0, y=0,
                          visible=True, collidable=True))
    door = spec["door"]
    sprites.append(protos[names.SPRITE_DOOR].clone().set_position(int(door["x"]), int(door["y"])))
    player = spec["player"]
    sprites.append(protos[names.SPRITE_PLAYER].clone()
                   .set_position(int(player["x"]), int(player["y"]))
                   .set_scale(int(player.get("scale", 2))))
    block_names = {"primary": names.SPRITE_BLOCK, "alternate": names.SPRITE_BLOCK_ALT}
    target_names = {"primary": names.SPRITE_TARGET, "alternate": names.SPRITE_TARGET_ALT}
    for block in spec.get("blocks", []):
        family = block.get("family", "primary")
        if family not in block_names:
            raise ValueError("unknown block family")
        sprites.append(protos[block_names[family]].clone()
                       .set_position(int(block["x"]), int(block["y"]))
                       .set_rotation(int(block.get("rotation", 0))))
    for target in spec.get("targets", []):
        family = target.get("family", "primary")
        if family not in target_names:
            raise ValueError("unknown target family")
        sprites.append(protos[target_names[family]].clone()
                       .set_position(int(target["x"]), int(target["y"])))
    for pickup in spec.get("pickups", []):
        sprites.append(protos[names.SPRITE_PICKUP].clone()
                       .set_position(int(pickup["x"]), int(pickup["y"])))
    pads = list(spec.get("pads", []))
    if pads:
        sprites.append(protos[names.SPRITE_TELEPORT_INDICATOR].clone()
                       .set_position(int(pads[0]["x"]) - 1, int(pads[0]["y"]) - 1))
    for pad in pads:
        sprites.append(protos[names.SPRITE_TELEPORT_PAD].clone()
                       .set_position(int(pad["x"]), int(pad["y"])))
    small_pads = list(spec.get("small_pads", []))
    if small_pads:
        sprites.append(protos[names.SPRITE_TELEPORT_INDICATOR_SMALL].clone()
                       .set_position(int(small_pads[0]["x"]) - 2, int(small_pads[0]["y"]) - 2))
    for pad in small_pads:
        sprites.append(protos[names.SPRITE_TELEPORT_PAD_SMALL].clone()
                       .set_position(int(pad["x"]), int(pad["y"])))
    for ring in spec.get("rings", []):
        sprites.append(protos[names.SPRITE_RING_OBSTACLE].clone()
                       .set_position(int(ring["x"]), int(ring["y"])))
    data = {names.KEY_BUDGET: int(spec["budget"]), names.KEY_SPELLS: spells}
    return Level(sprites=sprites, grid_size=(64, 64), data=data,
                 name=f"generated-sc25-d{spec['difficulty']}")


def _d4_rows(rows):
    grid = tuple(tuple(ch == "#" for ch in row) for row in rows)
    variants = []
    current = grid
    for _ in range(4):
        variants.append(current)
        variants.append(tuple(tuple(reversed(row)) for row in current))
        current = tuple(tuple(current[63 - col][row] for col in range(64)) for row in range(64))
    return min("".join("1" if bit else "0" for row in item for bit in row) for item in variants)


def geometry_identity(spec):
    return hashlib.sha256(_d4_rows(spec["wall"]).encode()).hexdigest()


def geometry_partition(spec):
    digest = geometry_identity(spec)
    return digest, SPLITS[int(digest, 16) % len(SPLITS)]


@lru_cache(maxsize=1)
def _official_geometry_hashes():
    result = set()
    for index in range(len(official_levels())):
        env = Env()
        env.set_level(index)
        mask = extract(env).wall_raw[PAD:PAD + 64, PAD:PAD + 64]
        rows = ["".join("#" if value else "." for value in row) for row in mask]
        result.add(geometry_identity({"wall": rows}))
    return result


def gameplay_identity(spec):
    body = {
        "geometry": spec["geometry_d4_sha256"], "difficulty": spec["difficulty"],
        "budget": spec["budget"], "spells": spec["spells"],
        "player": spec["player"], "door": spec["door"],
        "blocks": spec.get("blocks", []), "targets": spec.get("targets", []),
        "pickups": spec.get("pickups", []), "pads": spec.get("pads", []),
        "small_pads": spec.get("small_pads", []), "rings": spec.get("rings", []),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _mechanic_trace(layout, actions):
    actions = tuple(actions)
    state = layout.start
    winning_used = None
    counts = Counter(move_actions=0, click_actions=0, shrink_casts=0, grow_casts=0,
                     fire_casts=0, teleports=0, large_teleports=0,
                     small_teleports=0, primary_targets_hit=0,
                     alternate_targets_hit=0, pickups_consumed=0,
                     target_covered_pickups=0,
                     target_clear_before_pickup_pairs=0,
                     tutorial_demo_actions=0, failed_casts=0)
    targets_by_position = {
        (item.x, item.y): item for item in layout.removables
        if item.kind in (KIND_TARGET, KIND_TARGET_ALT)
    }
    covered_pickups = {
        item.bit: targets_by_position[(item.x, item.y)]
        for item in layout.pickups if (item.x, item.y) in targets_by_position
    }
    counts["target_covered_pickups"] = len(covered_pickups)
    spell_masks = {
        sum(1 << (row * 3 + col) for row, col in names.PATTERNS[spell]): spell
        for spell in layout.spells
    }
    won = False
    for action_index, (action, x, y) in enumerate(actions):
        if state[8]:
            state = state[:8] + (False, state[9])
            counts["tutorial_demo_actions"] += 1
            continue
        if action in names.MOVE_ACTIONS:
            before_state = state
            before_removed = state[6]
            state, won = _move(layout, state, action)
            counts["move_actions"] += 1
            if won:
                winning_used = before_state[7]
                if action_index != len(actions) - 1:
                    raise ValueError("witness completes before its final action")
                break
            if state is None:
                raise ValueError("witness exceeds native budget")
            for item in layout.pickups:
                if not before_removed & (1 << item.bit) and state[6] & (1 << item.bit):
                    counts["pickups_consumed"] += 1
                    target = covered_pickups.get(item.bit)
                    if target is not None and before_removed & (1 << target.bit):
                        counts["target_clear_before_pickup_pairs"] += 1
            continue
        if action != 6 or x is None or y is None:
            raise ValueError("invalid witness action")
        counts["click_actions"] += 1
        col_num = (int(x) - (names.GRID_ORIGIN[0] + 1)) // names.GRID_PITCH
        row_num = (int(y) - (names.GRID_ORIGIN[1] + 1)) // names.GRID_PITCH
        if not (0 <= row_num < 3 and 0 <= col_num < 3):
            raise ValueError("witness click is outside the spell grid")
        new_grid = state[9] ^ (1 << (row_num * 3 + col_num))
        spell = spell_masks.get(new_grid)
        if spell is None:
            used = state[7] + 1
            if layout.budget is not None and used > layout.budget:
                raise ValueError("witness exceeds native budget")
            state = state[:7] + (used, state[8], new_grid)
            continue
        before = state
        after = _cast(layout, state, spell)
        if after is None:
            counts["failed_casts"] += 1
            raise ValueError("winning witness contains an unmodelled failed cast")
        if spell == names.SPELL_GROW:
            counts["shrink_casts" if after[2] < before[2] else "grow_casts"] += 1
        elif spell == names.SPELL_TELEPORT:
            counts["teleports"] += 1
            counts["small_teleports" if before[2] == 1 else "large_teleports"] += 1
        elif spell == names.SPELL_FIRE:
            counts["fire_casts"] += 1
            changed = after[6] & ~before[6]
            for item in layout.removables:
                if changed & (1 << item.bit):
                    if item.kind == KIND_TARGET:
                        counts["primary_targets_hit"] += 1
                    elif item.kind == KIND_TARGET_ALT:
                        counts["alternate_targets_hit"] += 1
        changed = after[6] & ~before[6]
        for item in layout.removables:
            if item.kind == KIND_PICKUP and changed & (1 << item.bit):
                counts["pickups_consumed"] += 1
                target = covered_pickups.get(item.bit)
                if target is not None and before[6] & (1 << target.bit):
                    counts["target_clear_before_pickup_pairs"] += 1
        state = after
    counts["won"] = won
    counts["final_used"] = winning_used if won else (state[7] if state is not None else None)
    counts["remaining_budget"] = (
        None if layout.budget is None or counts["final_used"] is None
        else layout.budget - counts["final_used"]
    )
    return dict(counts)


def _sprite_present(env, name, position):
    return any(
        sprite.name == name and (sprite.x, sprite.y) == position
        for sprite in env.game.current_level.get_sprites()
    )


def _native_covered_pickup_evidence(spec, actions):
    """Recompute tier-6 target/pickup ordering and its blocked counterfactual.

    Each counterfactual replays the real native prefix up to (but excluding)
    the click that completes a fire cast, then applies the route's subsequent
    movement suffix.  With the target still present, that suffix must leave the
    paired pickup present and fail to reach the position reached by the actual
    target-cleared route.
    """
    targets = list(spec.get("targets", []))
    pickups = list(spec.get("pickups", []))
    if len(targets) != 2 or len(pickups) != 2:
        raise ValueError("tier 6 requires exactly two targets and two pickups")
    pickup_positions = [(item.get("x"), item.get("y")) for item in pickups]
    target_positions = [(item.get("x"), item.get("y")) for item in targets]
    if len(set(target_positions)) != 2 or sorted(pickup_positions) != sorted(target_positions):
        raise ValueError("each tier-6 target must have one same-coordinate pickup")

    target_names = {
        "primary": names.SPRITE_TARGET,
        "alternate": names.SPRITE_TARGET_ALT,
    }
    parsed = [tuple(action) for action in actions]
    env = _context_env(spec)
    target_events = {}
    pickup_events = {}
    pickup_budget = {}
    pickup_positions_after = {}
    for action_index, (action, x, y) in enumerate(parsed):
        before_targets = {
            (item["x"], item["y"]): _sprite_present(
                env, target_names[item["family"]], (item["x"], item["y"])
            )
            for item in targets
        }
        before_pickups = {
            (item["x"], item["y"]): _sprite_present(
                env, names.SPRITE_PICKUP, (item["x"], item["y"])
            )
            for item in pickups
        }
        used_before = env.used()
        env.perform(action, x, y)
        for item in targets:
            position = (item["x"], item["y"])
            if before_targets[position] and not _sprite_present(
                    env, target_names[item["family"]], position):
                if position in target_events:
                    raise ValueError("a covered target disappeared more than once")
                target_events[position] = action_index
        for item in pickups:
            position = (item["x"], item["y"])
            if before_pickups[position] and not _sprite_present(
                    env, names.SPRITE_PICKUP, position):
                if position in pickup_events:
                    raise ValueError("a covered pickup disappeared more than once")
                pickup_events[position] = action_index
                pickup_budget[position] = (used_before, env.used())
                pickup_positions_after[position] = (env.player.x, env.player.y)

    evidence = []
    for item in targets:
        position = (item["x"], item["y"])
        clear_index = target_events.get(position)
        pickup_index = pickup_events.get(position)
        if clear_index is None or pickup_index is None or clear_index >= pickup_index:
            raise ValueError("covered pickup was not collected after its target cleared")
        movement_suffix = parsed[clear_index + 1:pickup_index + 1]
        if not movement_suffix or any(action not in names.MOVE_ACTIONS for action, _x, _y in movement_suffix):
            raise ValueError("covered pickup must follow its fire clear through movement only")

        counterfactual = _context_env(spec)
        for action, x, y in parsed[:clear_index]:
            counterfactual.perform(action, x, y)
        if not _sprite_present(counterfactual, target_names[item["family"]], position):
            raise ValueError("covered target is absent before its completing fire click")
        if not _sprite_present(counterfactual, names.SPRITE_PICKUP, position):
            raise ValueError("covered pickup is absent before its target clears")
        for action, x, y in movement_suffix:
            counterfactual.perform(action, x, y)
        counter_position = (counterfactual.player.x, counterfactual.player.y)
        if not _sprite_present(counterfactual, target_names[item["family"]], position):
            raise ValueError("counterfactual movement unexpectedly cleared its target")
        if not _sprite_present(counterfactual, names.SPRITE_PICKUP, position):
            raise ValueError("counterfactual movement collected a target-covered pickup")
        if counter_position == pickup_positions_after[position]:
            raise ValueError("counterfactual movement was not blocked by the uncleared target")
        used_before, used_after = pickup_budget[position]
        if used_after != max(0, used_before + 1 - names.PICKUP_REFUND):
            raise ValueError("covered pickup did not apply the native budget refund")
        evidence.append({
            "family": item["family"],
            "position": [position[0], position[1]],
            "target_clear_action": clear_index + 1,
            "pickup_collect_action": pickup_index + 1,
            "movement_suffix": [action for action, _x, _y in movement_suffix],
            "counterfactual_target_present": True,
            "counterfactual_pickup_present": True,
            "counterfactual_progress_blocked": True,
            "used_before_pickup_action": used_before,
            "used_after_pickup_action": used_after,
        })
    return {
        "version": CAUSALITY_VERSION,
        "native_dependencies_verified": 2,
        "pairs": evidence,
    }


def _context_env(spec):
    env = Env([build_level(spec) for _ in DIFFICULTIES])
    env.set_level(spec["context_index"])
    return env


def verify(spec, limit=None):
    """Return an enriched row only after profile gates and native-context replay."""
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return None, "difficulty"
    limit = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]["search_work"] if limit is None else limit
    if type(limit) is not int or not 0 < limit <= 32_000_000:
        raise ValueError("limit must be an integer in 1..32000000")
    preliminary = profile_errors(spec, require_proof=False)
    if preliminary:
        return None, "profile: " + "; ".join(preliminary)
    env = _context_env(spec)
    layout = extract(env)
    result = search(layout, limit=limit)
    if result.unsupported:
        return None, "unsupported: " + result.reason
    if result.truncated:
        return None, "search_cutoff"
    if not result.solved:
        return None, "exhausted_without_solution"
    try:
        mechanics = _mechanic_trace(layout, result.actions)
    except ValueError as exc:
        return None, "trace: " + str(exc)
    candidate = dict(spec)
    candidate.update(solution=[list(action) for action in result.actions],
                     context_solution=[list(action) for action in result.actions],
                     solution_length=len(result.actions), solution_mechanics=mechanics,
                     search_expanded=result.expanded, search_limit=limit,
                     search_truncated=False)
    errors = profile_errors(candidate)
    if errors:
        return None, "profile: " + "; ".join(errors)
    observation = None
    start_score = env.levels_completed
    for action_index, (action, x, y) in enumerate(result.actions):
        if action not in env.available_actions:
            return None, "native_action_unavailable"
        observation = env.perform(action, x, y)
        if env.levels_completed > start_score and action_index != len(result.actions) - 1:
            return None, "native_route_completes_before_final_action"
    context = difficulty - 1
    advanced = env.levels_completed == 1 and (
        (context == len(DIFFICULTIES) - 1 and observation is not None and observation.won)
        or (context < len(DIFFICULTIES) - 1 and env.level_index == context + 1))
    if not advanced:
        return None, "native_context_replay"
    if difficulty == 6:
        try:
            candidate["solution_causality"] = _native_covered_pickup_evidence(
                candidate, result.actions
            )
        except ValueError as exc:
            return None, "tier6_causality: " + str(exc)
    candidate.update(
        engine_verified=True, context_engine_verified=True,
        proof_level_index=context, verification_level_index=context,
        native_budget=layout.budget,
        proof={"context_index": context, "context_engine_verified": True,
               "search_truncated": False, "search_limit": limit,
               "search_expanded": result.expanded,
               "witness_kind": "constructive-not-claimed-optimal",
               "levels_completed": 1})
    candidate["gameplay_sha256"] = gameplay_identity(candidate)
    return candidate, None


def validate_full_standard(spec, curriculum_entry):
    """Recompute and return full-contract violations for one accepted row.

    Stored proof flags and counters are assertions, not trusted evidence. This
    validator rebuilds the native level, traces the stored route, and replays it
    in its intended engine context. Malformed rows produce errors, not crashes.
    """
    errors = []
    actual_density = None
    if not isinstance(spec, dict):
        return ["spec must be a mapping"]
    if not isinstance(curriculum_entry, dict):
        return ["curriculum entry must be a mapping"]
    difficulty = curriculum_entry.get("difficulty")
    context = curriculum_entry.get("context_index")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty is invalid"]
    if type(curriculum_entry.get("search_work")) is not int or not 0 < curriculum_entry["search_work"] <= 32_000_000:
        errors.append("curriculum search_work is invalid")
    if type(context) is not int or context != difficulty - 1:
        errors.append("curriculum context_index is invalid")
    if type(spec.get("difficulty")) is not int or spec.get("difficulty") != difficulty:
        errors.append("difficulty does not match curriculum")
    if type(spec.get("context_index")) is not int or spec.get("context_index") != context or context != difficulty - 1:
        errors.append("native context does not match curriculum")
    if spec.get("split") not in SPLITS or spec.get("geometry_split") != spec.get("split"):
        errors.append("split/geometry partition mismatch")
    if spec.get("generator_version") != GENERATOR_VERSION:
        errors.append("generator version mismatch")
    if spec.get("mechanics_version") != MECHANICS_VERSION:
        errors.append("mechanics version mismatch")
    if spec.get("quality_profile_version") != PROFILE_VERSION:
        errors.append("quality profile version mismatch")
    if spec.get("source") != "generated_only":
        errors.append("source is not generated_only")
    for field in ("geometry_d4_sha256", "gameplay_sha256"):
        if not isinstance(spec.get(field), str) or not spec[field]:
            errors.append(f"missing {field}")
    proof = spec.get("proof", {})
    if not isinstance(proof, dict):
        errors.append("proof must be a mapping")
        proof = {}
    if not spec.get("engine_verified") or not proof.get("context_engine_verified"):
        errors.append("missing native context replay proof")
    if proof.get("context_index") != context or proof.get("search_truncated") is not False:
        errors.append("proof context/cutoff mismatch")
    if spec.get("native_budget") != REFERENCE_PROFILES.get(difficulty, {}).get("budget"):
        errors.append("native budget mismatch")
    try:
        actual_density = round(sum(row.count("#") for row in spec["wall"]) / (64 * 64), 6)
        if spec.get("visual_density") != actual_density:
            errors.append("stored visual density does not match geometry")
        geometry_hash, partition = geometry_partition(spec)
        if spec.get("geometry_d4_sha256") != geometry_hash or spec.get("geometry_sha256") != geometry_hash:
            errors.append("stored geometry identity does not match geometry")
        if partition != spec.get("split"):
            errors.append("canonical geometry partition does not match split")
        if geometry_hash in _official_geometry_hashes():
            errors.append("generated geometry matches an official wall geometry")
        identity_spec = dict(spec, geometry_d4_sha256=geometry_hash)
        if spec.get("gameplay_sha256") != gameplay_identity(identity_spec):
            errors.append("stored gameplay identity does not match content")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"malformed geometry/content: {exc}")

    actions = spec.get("solution")
    parsed_actions = []
    actions_are_list = isinstance(actions, list)
    if not actions_are_list or not actions:
        errors.append("solution must be a nonempty list")
    else:
        for index, action in enumerate(actions):
            if not isinstance(action, (list, tuple)) or len(action) != 3:
                errors.append(f"solution action {index} is malformed")
                continue
            action_id, x, y = action
            if type(action_id) is not int or action_id not in names.ACTION_IDS:
                errors.append(f"solution action {index} has invalid id")
                continue
            if action_id == 6:
                if type(x) is not int or type(y) is not int:
                    errors.append(f"solution click {index} has invalid coordinates")
                    continue
            elif x is not None or y is not None:
                errors.append(f"solution movement {index} must have null coordinates")
                continue
            parsed_actions.append((action_id, x, y))
    action_count = len(actions) if actions_are_list else 0
    if spec.get("solution_length") != len(parsed_actions) or len(parsed_actions) != action_count:
        errors.append("stored solution length does not match the route")

    stored_mechanics = spec.get("solution_mechanics")
    if not isinstance(stored_mechanics, dict):
        errors.append("solution_mechanics must be a mapping")
    if difficulty == 6 and not isinstance(spec.get("solution_causality"), dict):
        errors.append("tier 6 solution_causality must be a mapping")

    recomputed = dict(spec)
    if parsed_actions and len(parsed_actions) == action_count:
        try:
            env = _context_env(spec)
            layout = extract(env)
            if layout.budget != REFERENCE_PROFILES[difficulty]["budget"]:
                errors.append("rebuilt native budget differs from the profile")
            mechanics = _mechanic_trace(layout, parsed_actions)
            if mechanics != spec.get("solution_mechanics"):
                errors.append("stored solution mechanics do not match the route")
            recomputed.update(solution_length=len(parsed_actions), solution_mechanics=mechanics,
                              visual_density=actual_density)
            observation = None
            start_score = env.levels_completed
            for action_index, (action_id, x, y) in enumerate(parsed_actions):
                if action_id not in env.available_actions:
                    errors.append("route contains an unavailable native action")
                    break
                observation = env.perform(action_id, x, y)
                if env.levels_completed > start_score:
                    if action_index != len(parsed_actions) - 1:
                        errors.append("stored route completes before its final action")
                    break
            advanced = env.levels_completed == 1 and (
                (context == len(DIFFICULTIES) - 1 and observation is not None and observation.won)
                or (context < len(DIFFICULTIES) - 1 and env.level_index == context + 1))
            if not advanced:
                errors.append("stored route does not win in the intended native context")
            if difficulty == 6:
                causality = _native_covered_pickup_evidence(spec, parsed_actions)
                if causality != spec.get("solution_causality"):
                    errors.append("stored tier-6 covered-pickup causality does not match native replay")
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f"route rebuild/trace/replay failed: {exc}")
    try:
        errors.extend(profile_errors(recomputed))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        errors.append(f"profile validation failed: {exc}")
    return errors


def generate(seed, difficulty=1, attempts=MAX_ATTEMPTS, limit=None, *, split,
             record_rejection=None):
    """Generate one deterministic full-standard tier for an explicit split."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    rng = random.Random(f"{SEED_VERSION}:{seed}:{difficulty}:{split}")
    rejected = Counter()
    for attempt in range(1, attempts + 1):
        spec = _DRAFTERS[difficulty](rng)
        geometry_hash, partition = geometry_partition(spec)
        if partition != split:
            reason = "geometry_split"
        elif geometry_hash in _official_geometry_hashes():
            reason = "official_geometry_copy"
        else:
            spec.update(seed=seed, generation_attempt=attempt, split=split,
                        geometry_d4_sha256=geometry_hash, geometry_sha256=geometry_hash,
                        geometry_split=partition, geometry_version=GEOMETRY_VERSION,
                        gameplay_version=GAMEPLAY_VERSION)
            accepted, reason = verify(spec, limit=limit)
            if accepted is not None:
                accepted["generation_rejections"] = dict(rejected)
                errors = validate_full_standard(
                    accepted, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1])
                if not errors:
                    return accepted
                reason = "contract: " + "; ".join(errors)
        rejected[reason] += 1
        if record_rejection is not None:
            record_rejection({"seed": seed, "difficulty": difficulty,
                              "attempt": attempt, "split": split, "reason": reason})
    return None


def _child_seed(seed, ordinal, difficulty):
    payload = f"sc25:{SEED_VERSION}:{seed}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _verify_sequential(specs, limit=None):
    env = Env([build_level(spec) for spec in specs])
    env.reset()
    enriched = []
    observation = None
    for index, original in enumerate(specs):
        if env.level_index != index:
            return None
        layout = extract(env)
        work = FULL_STANDARD_CONTRACT["curriculum"][index]["search_work"] if limit is None else limit
        result = search(layout, limit=work)
        if not result.solved or result.truncated or result.unsupported:
            return None
        mechanics = _mechanic_trace(layout, result.actions)
        row = dict(original)
        row.update(solution=[list(action) for action in result.actions],
                   context_solution=[list(action) for action in result.actions],
                   solution_length=len(result.actions), solution_mechanics=mechanics,
                   sequential_engine_verified=True)
        if row["difficulty"] == 6:
            try:
                row["solution_causality"] = _native_covered_pickup_evidence(
                    row, result.actions
                )
            except ValueError:
                return None
        row["proof"] = dict(row["proof"], sequential_context_index=index,
                            sequential_engine_verified=True)
        if profile_errors(row):
            return None
        for action, x, y in result.actions:
            observation = env.perform(action, x, y)
        if env.levels_completed != index + 1:
            return None
        row["gameplay_sha256"] = gameplay_identity(row)
        enriched.append(row)
    if observation is None or not observation.won:
        return None
    return enriched


def generate_game(seed, *, split, difficulties=None, attempts=MAX_ATTEMPTS, limit=None):
    """Generate a complete increasing six-level native curriculum.

    A reduced ``difficulties`` sequence is useful for explicit smoke collection,
    but it cannot be passed to :func:`build_game` because native indices shift.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    requested = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not requested or any(
        type(difficulty) is not int or difficulty not in DIFFICULTIES
        for difficulty in requested
    ):
        raise ValueError("difficulties must be a nonempty subset of official tiers")
    if len(set(requested)) != len(requested) or tuple(sorted(requested)) != requested:
        raise ValueError("difficulties must be unique and increasing")
    specs = []
    for ordinal, difficulty in enumerate(requested):
        child = _child_seed(seed, ordinal, difficulty)
        spec = generate(child, difficulty, attempts=attempts, limit=limit, split=split)
        if spec is None:
            return None
        spec = dict(spec)
        spec.update(game_seed=seed, game_ordinal=ordinal, child_seed=child)
        specs.append(spec)
    if requested != DIFFICULTIES:
        for spec in specs:
            spec["sequence_kind"] = "explicit-smoke-subset"
        return specs
    sequential = _verify_sequential(specs, limit=limit)
    if sequential is None:
        return None
    for row, entry in zip(sequential, FULL_STANDARD_CONTRACT["curriculum"]):
        if validate_full_standard(row, entry):
            return None
    sequence_hash = hashlib.sha256(
        json.dumps([row["gameplay_sha256"] for row in sequential], separators=(",", ":")).encode()
    ).hexdigest()
    for row in sequential:
        row.update(sequence_kind="full-official-context", game_sequence_sha256=sequence_hash)
    return sequential


def build_game(specs):
    """Validate and natively replay exactly six ordered levels before building."""
    if not isinstance(specs, (list, tuple)):
        raise ValueError("full SC25 game specs must be a list or tuple")
    specs = list(specs)
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("full SC25 game requires exactly six levels")
    if any(not isinstance(spec, dict) for spec in specs):
        raise ValueError("every full-game spec must be a mapping")
    splits = [spec.get("split") for spec in specs]
    if any(split not in SPLITS for split in splits) or any(
            split != splits[0] for split in splits[1:]):
        raise ValueError("all game levels must use the same split")
    identities = set()
    for index, (spec, entry) in enumerate(zip(specs, FULL_STANDARD_CONTRACT["curriculum"])):
        errors = validate_full_standard(spec, entry)
        if errors:
            raise ValueError(f"invalid tier {index + 1}: {'; '.join(errors)}")
        identity = spec["gameplay_sha256"]
        if identity in identities:
            raise ValueError("duplicate gameplay identity in full game")
        identities.add(identity)
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    env.reset()
    observation = None
    for index, spec in enumerate(specs):
        if env.level_index != index:
            raise ValueError(f"native sequence did not enter tier {index + 1}")
        start_score = env.levels_completed
        actions = [tuple(value) for value in spec["solution"]]
        for action_index, (action, x, y) in enumerate(actions):
            observation = env.perform(action, x, y)
            if env.levels_completed > start_score:
                if action_index != len(actions) - 1:
                    raise ValueError(
                        f"tier {index + 1} route completes before its final action"
                    )
                break
        if env.levels_completed != index + 1:
            raise ValueError(f"tier {index + 1} route failed in the full native sequence")
    if observation is None or not observation.won:
        raise ValueError("full native sequence did not reach WIN")
    return [build_level(spec) for spec in specs]


def generate_engine_variant(seed, *, split, attempts=MAX_ATTEMPTS, limit=400_000):
    """Generate the non-reference ``ring-failed-fire`` recovery lesson.

    ``crzdcq`` rings and failed casts occur in the engine but in none of the six
    shipped levels. This opt-in row therefore never passes the official full
    profile. Its certificate deliberately fires into a ring, confirms that the
    target remains, then solves and replays the level from that live prefix.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    rng = random.Random(f"{MECHANICS_VERSION}:engine-ring:{seed}:{split}")
    for attempt in range(1, attempts + 1):
        spec = _draft_tier3(rng)
        y = spec["player"]["y"]
        rows = [list(row) for row in spec["wall"]]
        _carve(rows, 8, y - 8, 4, 8)
        spec["wall"] = ["".join(row) for row in rows]
        spec["rings"] = [{"x": 8, "y": y - 6}]
        spec["visual_density"] = round(
            sum(row.count("#") for row in spec["wall"]) / (64 * 64), 6)
        spec.update(profile_kind="engine-extension-nonreference",
                    engine_only_mechanics=["ring_fire_block", "failed_fire_cast"],
                    source="generated_engine_extension")
        geometry_hash, partition = geometry_partition(spec)
        if partition != split or geometry_hash in _official_geometry_hashes():
            continue
        spec.update(seed=seed, generation_attempt=attempt, split=split,
                    geometry_d4_sha256=geometry_hash, geometry_sha256=geometry_hash,
                    geometry_split=partition, geometry_version=GEOMETRY_VERSION,
                    gameplay_version=GAMEPLAY_VERSION)
        env = _context_env(spec)
        before_targets = len(env.game.current_level.get_sprites_by_name(names.SPRITE_TARGET))
        prefix = [(1, None, None), *_spell_actions(names.SPELL_FIRE)]
        env.perform(*prefix[0])
        probe_layout = extract(env)
        hit = probe_layout.fire_hit(
            probe_layout.start[0], probe_layout.start[1], probe_layout.start[2],
            probe_layout.start[3], probe_layout.start[6])
        if hit is None or hit.name != names.SPRITE_RING_OBSTACLE:
            continue
        for action, x, click_y in prefix[1:]:
            env.perform(action, x, click_y)
        after_targets = len(env.game.current_level.get_sprites_by_name(names.SPRITE_TARGET))
        grid = getattr(env.game, names.ATTR_GRID)
        if before_targets != 1 or after_targets != 1 or any(any(row) for row in grid):
            continue
        result = search(env, limit=limit)
        if not result.solved or result.truncated or result.unsupported:
            continue
        observation = None
        for action, x, click_y in result.actions:
            observation = env.perform(action, x, click_y)
        if env.levels_completed != 1 or env.level_index != 3:
            continue
        spec.update(
            engine_variant_verified=True,
            proof_level_index=2,
            probe_prefix=[list(action) for action in prefix],
            recovery_solution=[list(action) for action in result.actions],
            recovery_solution_length=len(result.actions),
            probe_mechanics={"failed_fire_casts": 1, "ring_fire_blocks": 1,
                             "target_survived_probe": True,
                             "native_recovery_completed": observation is not None},
        )
        spec["gameplay_sha256"] = gameplay_identity(spec)
        return spec
    return None
