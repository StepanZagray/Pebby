"""Full eight-tier, split-safe, real-engine-certified SB26 generation."""

from collections import Counter
import hashlib
import json
from numbers import Integral
import random

from arcengine import Level

from . import names
from .env import UPSTREAM, Env, official_levels, prototype
from .layout import Tile, extract
from .plan import assignment_from_layout, search, traverse
from .reference_profiles import DIFFICULTIES, REFERENCE_PROFILES, profile_errors


FORMAT = "pebby.sb26.level.v3"
SOURCE_SHA256 = "dbb4877853a8d30f84e28d26f4a3d6ad7d2d1018602e82ccb5a62284043984f3"
if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("vendored SB26 source bytes do not match the calibrated generator")
GENERATOR_VERSION = 3
MECHANICS_VERSION = "sb26-full-recursive-grammar-v1"
DIFFICULTY_VERSION = "sb26-official-eight-tier-v1"
GEOMETRY_VERSION = "native-role-semantics-v2"
QUALITY_VERSION = "sb26-reference-proof-v2"
SPLITS = ("train", "validation", "test")
DEFAULT_ATTEMPTS = 96
DEFAULT_NODE_LIMIT = 500_000
MAX_NODE_LIMIT = 1_000_000
MAX_GENERATION_ATTEMPTS = 10_000
MAX_ACTION_LIMIT = 64
GENERATION_REJECTION_REASONS = frozenset({
    "invalid_geometry",
    "geometry_split",
    "profile_mismatch",
    "official_copy",
    "search_truncated",
    "unsupported",
    "proven_unsolvable",
    "native_replay",
    "mechanic_use",
    "action_length",
})


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": "sb26-7fbdac44",
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_VERSION,
    "curriculum": [
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": DEFAULT_NODE_LIMIT,
        }
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "sb26.md#official-reference-characterization",
        "solution_mechanics": "spec.solution_mechanics and tests/games/test_sb26_quality.py",
        "native_budget": "third_party/arc3_games/sb26.py:721-913",
        "context_engine_replay": "spec.proof.context_engine_verified plus sequential replay tests",
        "novelty_split": "geometry_sha256/geometry_d4_sha256/gameplay_sha256",
        "bounded_rejections": "generation_exclusions and generation_limits",
        "independent_closure": ".scratch/multigame-resume/full-standard/external-astra-re86-sb26/sb26-closure-detailed.md",
        "enriched_provenance": "tests/games/test_sb26.py::test_generate_game_is_exactly_eight_increasing_contexts_and_build_replays_it",
    },
    "caveats": (
        "Each tier is calibrated to one shipped reference, so tolerances are engineering bounds, not confidence intervals.",
        "Constructive witnesses are native-replayed but are not claimed shortest routes.",
        "Tier 8 certifies a winning cyclic goal prefix; the separate first-slot guard test is negative/recovery evidence.",
        "Canonical identities encode native frame roles, link targets, regular-output equality, constraints, and goal order; presentation coordinates and unrelated frame-border/regular colour equality are excluded.",
        "geometry_split is a legacy field name whose value is derived from gameplay_sha256; raw_start_frame_sha256 separately binds exact rendered layout identity.",
        "Tier 1 has finite generated support of 23 semantic states partitioned 9 train, 10 validation, and 4 test after official-copy exclusion.",
    ),
}


class ContractMismatch(ValueError):
    """A stored generated row failed fail-closed recomputation."""


def _same_exact_json(actual, expected):
    """Compare JSON values without Python's bool/int/float coercions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(_same_exact_json(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (len(actual) == len(expected)
                and all(_same_exact_json(left, right) for left, right in zip(actual, expected)))
    return actual == expected


def _require_exact(mapping, field, expected):
    if type(mapping) is not dict or field not in mapping:
        raise ContractMismatch(f"{field} is missing")
    if not _same_exact_json(mapping[field], expected):
        raise ContractMismatch(f"{field}={mapping[field]!r}, expected exact {expected!r}")


def _require_int(mapping, field, *, minimum=None, maximum=None):
    if type(mapping) is not dict or field not in mapping:
        raise ContractMismatch(f"{field} is missing")
    value = mapping[field]
    if type(value) is not int:
        raise ContractMismatch(f"{field} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ContractMismatch(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ContractMismatch(f"{field} cannot exceed {maximum}")
    return value


def _jsonable(value):
    return json.loads(json.dumps(value, separators=(",", ":")))


def _integer(value, field, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _position(value, field):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be [x, y]")
    return _integer(value[0], f"{field}[0]"), _integer(value[1], f"{field}[1]")


def _colour(value, field):
    value = _integer(value, field)
    if value not in names.COLOURS:
        raise ValueError(f"{field} must come from {names.COLOURS}")
    return value


def _item(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    kind = value.get("kind")
    if kind not in ("regular", "link"):
        raise ValueError(f"{field}.kind must be regular or link")
    return kind, _colour(value.get("colour"), f"{field}.colour")


def _sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _frame_bytes_hash(frame):
    return hashlib.sha256(bytes(int(pixel) for row in frame for pixel in row)).hexdigest()


def effective_seed(requested_seed, split):
    """Map public seeds into disjoint deterministic split namespaces."""
    requested_seed = _integer(requested_seed, "seed", minimum=0)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return requested_seed % 1_000_000 + SPLITS.index(split) * 1_000_000


def _normalised_colours(sequence):
    mapping = {}
    output = []
    for colour in sequence:
        if colour not in mapping:
            mapping[colour] = len(mapping)
        output.append(mapping[colour])
    return mapping, output


def _semantic_item(tile, regular_colours, frame_targets):
    """Encode only the equality relation that has native gameplay meaning."""
    if tile.kind == "regular":
        return ["regular", regular_colours[tile.colour]]
    return ["link", frame_targets[tile.colour]]


def _identity_payloads(layout):
    """Return position-free native constraint and transition identities.

    Frame colours form a link-target namespace; regular tile/goal colours form
    an independent output namespace.  Collisions between those namespaces are
    graphical accidents, while a link's equality with its target frame is a
    real transition and is encoded as the target's native frame index.
    """
    frame_targets = {frame.colour: index for index, frame in enumerate(layout.frames)}
    regular_order = []
    for frame in layout.frames:
        regular_order.extend(
            tile.colour for tile in frame.occupants
            if tile is not None and tile.kind == "regular"
        )
    tray = sorted(
        (tile for tile in layout.tiles if tile.position not in layout.slots),
        key=lambda tile: (tile.position[1], tile.position[0]),
    )
    regular_order.extend(tile.colour for tile in tray if tile.kind == "regular")
    regular_order.extend(layout.goals)
    regular_colours, _ = _normalised_colours(regular_order)

    fixed = []
    for frame_index, frame in enumerate(layout.frames):
        for slot_index, tile in enumerate(frame.occupants):
            if tile is not None and not tile.movable:
                fixed.append([frame_index, slot_index, *_semantic_item(
                    tile, regular_colours, frame_targets
                )])
    tray_payload = [_semantic_item(tile, regular_colours, frame_targets) for tile in tray]
    goals = [regular_colours[colour] for colour in layout.goals]
    constraints = {
        "root_frame": 0,
        "frame_arities": [frame.arity for frame in layout.frames],
        "fixed_slots": fixed,
        "movable_tray": tray_payload,
        "ordered_goals": goals,
    }
    frame_programs = []
    fixed_by_slot = {(item[0], item[1]): item[2:] for item in fixed}
    for frame_index, frame in enumerate(layout.frames):
        frame_programs.append([
            fixed_by_slot.get((frame_index, slot_index), ["assignable"])
            for slot_index in range(frame.arity)
        ])
    gameplay = {
        "root_frame": 0,
        "native_frame_programs": frame_programs,
        "movable_tray": tray_payload,
        "ordered_goals": goals,
    }
    return constraints, gameplay, regular_colours, frame_targets, tray


def identity_hashes(layout, frame):
    constraints, gameplay, _, _, _ = _identity_payloads(layout)
    geometry_sha = _sha({"version": GEOMETRY_VERSION, "constraints": constraints})
    gameplay_sha = _sha({"version": GEOMETRY_VERSION, "transition": gameplay})
    return {
        "geometry_sha256": geometry_sha,
        "geometry_d4_sha256": geometry_sha,
        "geometry_split": SPLITS[int(gameplay_sha, 16) % len(SPLITS)],
        "gameplay_sha256": gameplay_sha,
        "raw_start_frame_sha256": _frame_bytes_hash(frame),
    }


def _official_hashes():
    cache = getattr(_official_hashes, "cache", None)
    if cache is None:
        geometry = set()
        gameplay = set()
        raw = set()
        for level in official_levels():
            env = Env([level])
            frame = env.reset()
            identities = identity_hashes(extract(env), frame)
            geometry.add(identities["geometry_sha256"])
            gameplay.add(identities["gameplay_sha256"])
            raw.add(identities["raw_start_frame_sha256"])
        cache = geometry, gameplay, raw
        _official_hashes.cache = cache
    return cache


def _native_context(level, context_index):
    levels = [level.clone() for _ in DIFFICULTIES]
    env = Env(levels)
    env.reset()
    if context_index:
        env.set_level(context_index)
    return env


def build_level(spec):
    """Rebuild a generated native level from JSON-compatible data."""
    frames_data = spec.get("frames")
    if not isinstance(frames_data, list) or not frames_data:
        raise ValueError("frames must be a nonempty list")
    frames = []
    frame_rectangles = []
    frame_colours = []
    for index, data in enumerate(frames_data):
        if not isinstance(data, dict):
            raise ValueError(f"frames[{index}] must be an object")
        arity = _integer(data.get("arity"), f"frames[{index}].arity")
        if arity not in names.FRAME_FOR_ARITY:
            raise ValueError(f"frames[{index}].arity must be in 1..7")
        colour = _colour(data.get("colour"), f"frames[{index}].colour")
        x, y = _position(data.get("position"), f"frames[{index}].position")
        frame = prototype(names.FRAME_FOR_ARITY[arity]).set_position(x, y).color_remap(None, colour)
        if x < 0 or y < 9 or x + frame.width > names.FRAME_SIZE or y + frame.height > names.TRAY_MIN_Y:
            raise ValueError(f"frames[{index}] does not fit above the tray")
        rectangle = (x, y, x + frame.width - 1, y + frame.height - 1)
        if any(not (rectangle[2] < other[0] or other[2] < rectangle[0]
                       or rectangle[3] < other[1] or other[3] < rectangle[1])
               for other in frame_rectangles):
            raise ValueError("frames overlap")
        frames.append(frame)
        frame_rectangles.append(rectangle)
        frame_colours.append(colour)
    if len(set(frame_colours)) != len(frame_colours):
        raise ValueError("frame colours must be distinct")
    if [(frame.y, frame.x) for frame in frames] != sorted((frame.y, frame.x) for frame in frames):
        raise ValueError("frames must be listed in native (y, x) traversal order")

    cells = {(frame_index, slot_index): position
             for frame_index, frame in enumerate(frames)
             for slot_index, position in enumerate(names.frame_cells(frame))}
    fixed_data = spec.get("fixed", [])
    tray_data = spec.get("tray")
    if not isinstance(fixed_data, list) or not isinstance(tray_data, list):
        raise ValueError("fixed and tray must be lists")
    sprites = list(frames)
    occupied = set()
    for index, data in enumerate(fixed_data):
        kind, colour = _item(data, f"fixed[{index}]")
        frame_index = _integer(data.get("frame"), f"fixed[{index}].frame")
        slot_index = _integer(data.get("slot"), f"fixed[{index}].slot")
        key = frame_index, slot_index
        if key not in cells or key in occupied:
            raise ValueError(f"fixed[{index}] has an invalid or duplicate frame slot")
        if kind == "link" and colour not in frame_colours:
            raise ValueError(f"fixed[{index}] link has no matching frame colour")
        item_name = names.SPRITE_TILE if kind == "regular" else names.SPRITE_LINK
        sprites.append(prototype(item_name).set_position(*cells[key]).color_remap(None, colour))
        occupied.add(key)
    tray_positions = set()
    for index, data in enumerate(tray_data):
        kind, colour = _item(data, f"tray[{index}]")
        x, y = _position(data.get("position"), f"tray[{index}].position")
        if not (0 <= x <= names.FRAME_SIZE - names.TILE_SIZE and names.TRAY_MIN_Y <= y <= names.FRAME_SIZE - names.TILE_SIZE):
            raise ValueError(f"tray[{index}] does not fit in the tray")
        if (x, y) in tray_positions:
            raise ValueError("tray positions must be distinct")
        if kind == "link" and colour not in frame_colours:
            raise ValueError(f"tray[{index}] link has no matching frame colour")
        tray_positions.add((x, y))
        item_name = names.SPRITE_TILE if kind == "regular" else names.SPRITE_LINK
        sprites.append(prototype(item_name).set_position(x, y).color_remap(None, colour))
    if len(cells) != len(fixed_data) + len(tray_data):
        raise ValueError("fixed plus tray item count must equal total frame slots")

    connections = spec.get("connections", [])
    if not isinstance(connections, list):
        raise ValueError("connections must be a list")
    fixed_by_slot = {(int(item["frame"]), int(item["slot"])): item for item in fixed_data}
    seen_connections = set()
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            raise ValueError(f"connections[{index}] must be an object")
        source = (
            _integer(connection.get("from_frame"), f"connections[{index}].from_frame"),
            _integer(connection.get("from_slot"), f"connections[{index}].from_slot"),
        )
        target = _integer(connection.get("to_frame"), f"connections[{index}].to_frame")
        item = fixed_by_slot.get(source)
        if (source in seen_connections or item is None or item.get("kind") != "link"
                or not 0 <= target < len(frames)
                or int(item.get("colour")) != frame_colours[target]):
            raise ValueError(f"connections[{index}] must join one fixed link to its target frame")
        source_x, source_y = cells[source]
        line_x, line_y = source_x + 2, source_y + 4
        if frames[target].y - line_y != 8:
            raise ValueError(f"connections[{index}] requires the native eight-pixel vertical cue")
        sprites.append(
            prototype(names.SPRITE_CONNECTOR)
            .set_position(line_x, line_y)
            .color_remap(None, frame_colours[target])
        )
        seen_connections.add(source)

    for key, position in cells.items():
        if key not in occupied:
            sprites.append(prototype(names.SPRITE_SLOT).set_position(*position))

    goals = spec.get("goals")
    goal_positions = spec.get("goal_positions")
    if not isinstance(goals, list) or not isinstance(goal_positions, list) or len(goals) != len(goal_positions):
        raise ValueError("goals and goal_positions must be equal-length lists")
    seen_goals = set()
    for index, (raw_colour, raw_position) in enumerate(zip(goals, goal_positions)):
        colour = _colour(raw_colour, f"goals[{index}]")
        x, y = _position(raw_position, f"goal_positions[{index}]")
        if not (1 <= x <= 58 and 0 <= y <= 46) or (x, y) in seen_goals:
            raise ValueError(f"goal_positions[{index}] is invalid or duplicated")
        seen_goals.add((x, y))
        sprites.append(prototype(names.SPRITE_GOAL).set_position(x, y).color_remap(None, colour))
        sprites.append(prototype(names.SPRITE_GOAL_BACKDROP).set_position(x - 1, y - 1))
    sprites.append(prototype(names.SPRITE_ENERGY_LINE).set_position(0, 53))
    return Level(sprites=sprites, grid_size=(names.FRAME_SIZE, names.FRAME_SIZE),
                 name=f"SB26 generated d{spec.get('difficulty', '?')}")


def _regular_colours(rng, difficulty):
    colours = list(names.COLOURS)
    rng.shuffle(colours)
    if difficulty == 1:
        return colours[:4]
    if difficulty in (2, 3, 4):
        return colours
    if difficulty == 5:
        return colours[:5] + [colours[0]]
    if difficulty == 6:
        return colours[:6] + colours[:3]
    if difficulty == 7:
        return colours[:4] + colours[:3]
    return colours[:6]


def _frame_positions(rng, difficulty):
    if difficulty == 1:
        return [(rng.choice((15, 18, 21)), rng.choice((20, 25, 30)))]
    if difficulty == 2:
        jitter = rng.choice((-3, 0, 3))
        return [(18 + jitter, 18), (18 + jitter, 32)]
    if difficulty == 3:
        shift = rng.choice((-2, 0, 2))
        return [(15 + shift, 18), (8 + shift, 33), (40 + shift, 33)]
    if difficulty in (4, 5):
        shift = rng.choice((-3, 0, 3))
        return [(15 + shift, 18), (21 - shift, 33)]
    if difficulty == 6:
        shift = rng.choice((-2, 0, 2))
        return [(7 + shift, 18), (35 + shift, 18), (7 - shift, 32), (35 - shift, 32)]
    if difficulty == 7:
        return [(rng.choice((14, 21, 28)), 12),
                (rng.choice((14, 21, 28)), 25),
                (rng.choice((14, 21, 28)), 38)]
    shift = rng.choice((-3, 0, 3))
    return [(18 + shift, 22), (18 - shift, 36)]


def _roles(rng, difficulty, frame_colours):
    """Return slot -> (kind, colour-or-None, fixed) before regular colours."""
    arities = REFERENCE_PROFILES[difficulty]["frame_arities"]
    roles = {(frame, slot): ["regular", None, False]
             for frame, arity in enumerate(arities) for slot in range(arity)}
    if difficulty == 2:
        roles[(0, rng.choice((1, 2))) ] = ["link", frame_colours[1], True]
    elif difficulty == 3:
        link_slots = rng.choice(((0, 2), (0, 3), (1, 3), (1, 4), (2, 4)))
        targets = [1, 2]
        rng.shuffle(targets)
        for slot, target in zip(link_slots, targets):
            roles[(0, slot)] = ["link", frame_colours[target], True]
    elif difficulty == 4:
        roles[(0, rng.randrange(5))] = ["link", frame_colours[1], False]
        roles[(1, rng.randrange(3))][2] = True
    elif difficulty == 5:
        for slot in rng.sample(range(5), 2):
            roles[(0, slot)] = ["link", frame_colours[1], False]
    elif difficulty == 6:
        targets = [1, 2, 3]
        rng.shuffle(targets)
        for slot, target in enumerate(targets):
            roles[(0, slot)] = ["link", frame_colours[target], False]
        for child in (1, 2, 3):
            roles[(child, rng.randrange(3))][2] = True
    elif difficulty == 7:
        roles[(0, rng.randrange(3))] = ["link", frame_colours[1], False]
        roles[(1, rng.randrange(3))] = ["link", frame_colours[2], False]
        roles[(2, rng.randrange(3))][2] = True
    elif difficulty == 8:
        roles[(0, rng.choice((1, 2, 3)))] = ["link", frame_colours[1], False]
        roles[(1, rng.choice((1, 2, 3)))] = ["link", frame_colours[0], False]
    return roles


def _emit_goals(arities, frame_colours, roles, goal_count):
    colour_to_frame = {colour: index for index, colour in enumerate(frame_colours)}
    stack = [(0, 0)]
    goals = []
    seen = set()
    while len(goals) < goal_count:
        state = tuple(stack), len(goals)
        if state in seen:
            raise ValueError("draft traversal loops without emitting a colour")
        seen.add(state)
        frame, slot = stack[-1]
        if slot >= arities[frame]:
            if len(stack) == 1:
                raise ValueError("draft traversal ended before the reference goal count")
            stack.pop()
            parent, parent_slot = stack[-1]
            stack[-1] = parent, parent_slot + 1
            continue
        kind, colour, _ = roles[(frame, slot)]
        if kind == "regular":
            goals.append(colour)
            stack[-1] = frame, slot + 1
        else:
            target = colour_to_frame[colour]
            if slot == 0 and (frame, slot) in stack[:-1] and len(stack) > 1 and stack[-2][1] == 0:
                raise ValueError("draft triggers the first-slot cycle guard")
            stack.append((target, 0))
    return goals


def _goal_positions(goal_count):
    if goal_count <= 9:
        start = (64 - (7 * goal_count - 1)) // 2 + 1
        return [[start + 7 * index, 1] for index in range(goal_count)]
    per_row = 6
    start = (64 - (7 * per_row - 1)) // 2 + 1
    return [[start + 7 * (index % per_row), 1 + 7 * (index // per_row)]
            for index in range(goal_count)]


def _draft(rng, requested_seed, split, difficulty):
    profile = REFERENCE_PROFILES[difficulty]
    arities = profile["frame_arities"]
    child_colours = [colour for colour in names.COLOURS if colour != 8]
    rng.shuffle(child_colours)
    frame_colours = [8] + child_colours[:len(arities) - 1]
    roles = _roles(rng, difficulty, frame_colours)
    regular_colours = _regular_colours(rng, difficulty)
    rng.shuffle(regular_colours)
    regular_slots = [slot for slot in roles if roles[slot][0] == "regular"]
    if len(regular_slots) != len(regular_colours):
        raise AssertionError("profile regular count differs from construction roles")
    for slot, colour in zip(regular_slots, regular_colours):
        roles[slot][1] = colour

    positions = _frame_positions(rng, difficulty)
    frames = [
        {"arity": arity, "position": list(position), "colour": colour}
        for arity, position, colour in zip(arities, positions, frame_colours)
    ]
    fixed = []
    movable = []
    for (frame, slot), (kind, colour, is_fixed) in roles.items():
        item = {"kind": kind, "colour": colour}
        if is_fixed:
            fixed.append({**item, "frame": frame, "slot": slot})
        else:
            movable.append(item)
    rng.shuffle(movable)
    tray_width = 7 * len(movable) - 1
    tray_start = (64 - tray_width) // 2
    tray = [{**item, "position": [tray_start + 7 * index, 56]}
            for index, item in enumerate(movable)]
    goals = _emit_goals(arities, frame_colours, roles, profile["goals"])
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "quality_version": QUALITY_VERSION,
        "source": "generated_only",
        "game": "sb26",
        "requested_seed": int(requested_seed),
        "original_seed": int(requested_seed),
        "seed": effective_seed(requested_seed, split),
        "effective_seed": effective_seed(requested_seed, split),
        "effective_split": split,
        "difficulty": int(difficulty),
        "split": split,
        "context_index": difficulty - 1,
        "training_context_index": difficulty - 1,
        "verification_level_index": difficulty - 1,
        "frames": frames,
        "fixed": fixed,
        "tray": tray,
        "goals": goals,
        "goal_positions": _goal_positions(len(goals)),
        "connections": ([{"from_frame": 0,
                           "from_slot": next(slot for (frame, slot), role in roles.items()
                                             if frame == 0 and role[0] == "link"),
                           "to_frame": 1}] if difficulty == 2 else []),
        "coverage": "full shipped SB26 frame/link/colour-sharing/cyclic-prefix grammar",
        "omitted_mechanics": [],
    }


def _structural_metrics(layout, frame):
    fixed = Counter(tile.kind for tile in layout.fixed_tiles)
    movable = Counter(tile.kind for tile in layout.movable_tiles)
    regular_colours = {tile.colour for tile in layout.tiles if tile.kind == "regular"}
    visible = [(x, y) for y, row in enumerate(frame) for x, colour in enumerate(row) if colour != 4]
    return {
        "frame_count": len(layout.frames),
        "frame_arities": [frame.arity for frame in layout.frames],
        "connector_count": layout.connector_count,
        "fixed_regular": fixed["regular"],
        "fixed_links": fixed["link"],
        "movable_regular": movable["regular"],
        "movable_links": movable["link"],
        "regular_tiles": fixed["regular"] + movable["regular"],
        "link_tiles": fixed["link"] + movable["link"],
        "goals": len(layout.goals),
        "distinct_regular_colours": len(regular_colours),
        "shared_regular_colours": fixed["regular"] + movable["regular"] - len(regular_colours),
        "initial_energy": layout.initial_energy,
        "visual_density": round(len(visible) / (64 * 64), 6),
        "visible_bbox": [min(x for x, _ in visible), min(y for _, y in visible),
                         max(x for x, _ in visible), max(y for _, y in visible)],
    }


def _route_certificate(level, context_index, actions):
    """Recompute an unpadded placement trace, traversal use, and native win."""
    actions = [tuple(action) for action in actions]
    if not actions or actions[-1] != (names.ACTION_SUBMIT, None, None):
        raise ContractMismatch("winning route must end with ACTION5")
    if any(action[0] != names.ACTION_CLICK for action in actions[:-1]) or len(actions[:-1]) % 2:
        raise ContractMismatch("generated witness must contain click pairs followed by one submit")
    env = _native_context(level, context_index)
    initial = extract(env)
    constraints, _, regular_colours, frame_targets, tray = _identity_payloads(initial)
    del constraints
    tray_index = {tile.position: index for index, tile in enumerate(tray)}
    slot_index = {position: (frame_index, index)
                  for frame_index, frame in enumerate(initial.frames)
                  for index, position in enumerate(frame.slots)}
    live_positions = {tile.position: tile for tile in initial.tiles}
    placement_trace = []
    for first, second in zip(actions[:-1:2], actions[1:-1:2]):
        source = (first[1] - names.CLICK_INSET, first[2] - names.CLICK_INSET)
        destination = (second[1] - names.CLICK_INSET, second[2] - names.CLICK_INSET)
        tile = live_positions.get(source)
        if tile is None or not tile.movable or source not in tray_index:
            raise ContractMismatch("route source is not an unused movable tray tile")
        if destination not in slot_index or destination in live_positions:
            raise ContractMismatch("route destination is not an empty frame slot")
        placement_trace.append([
            tray_index[source], *_semantic_item(tile, regular_colours, frame_targets),
            slot_index[destination][0], slot_index[destination][1],
        ])
        del live_positions[source]
        live_positions[destination] = Tile(tile.kind, tile.colour, destination, True)

    score = env.levels_completed
    for action in actions[:-1]:
        observation = env.perform(*action)
        if observation.finished or env.levels_completed != score:
            raise ContractMismatch("placement prefix terminated the native level")
    filled = extract(env)
    assignment = assignment_from_layout(filled)
    if assignment is None:
        raise ContractMismatch("winning route leaves a frame slot empty")
    mechanic_use = traverse(filled, assignment).metadata()
    observation = env.perform(*actions[-1])
    if env.levels_completed != score + 1 or not observation.won and env.level_index == context_index:
        raise ContractMismatch("route did not win the real engine in its declared context")
    expected_energy = initial.energy - len(placement_trace) - 1
    # The next level resets visible energy, so bind the route-side arithmetic
    # rather than trusting the post-transition field.
    mechanic_use.update({
        "placements": len(placement_trace),
        "submit_actions": 1,
        "undo_actions": 0,
        "energy_cost": len(placement_trace) + 1,
        "route_minimum_energy": expected_energy,
        "first_completion_action_index": len(actions) - 1,
        "placement_trace": placement_trace,
    })
    return mechanic_use, _sha(placement_trace), observation


def certify(spec, *, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Return a full proof row only after profile gates and native replay."""
    candidate = _jsonable(spec)
    difficulty = _integer(candidate.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    node_limit = _integer(node_limit, "node_limit", minimum=0)
    if node_limit > MAX_NODE_LIMIT:
        raise ValueError(f"node_limit cannot exceed {MAX_NODE_LIMIT}")
    action_limit = REFERENCE_PROFILES[difficulty]["reference_actions"] if limit is None else _integer(limit, "limit", minimum=1)
    if action_limit > MAX_ACTION_LIMIT:
        raise ValueError(f"limit cannot exceed {MAX_ACTION_LIMIT}")
    level = build_level(candidate)
    context = difficulty - 1
    env = _native_context(level, context)
    initial_frame = env.render()
    layout = extract(env)
    metrics = _structural_metrics(layout, initial_frame)
    errors = profile_errors(difficulty, metrics)
    if errors:
        certify.last_reason = "profile_mismatch"
        return None
    identities = identity_hashes(layout, initial_frame)
    if identities["geometry_split"] != candidate.get("split"):
        certify.last_reason = "geometry_split"
        return None
    official_geometry, official_gameplay, official_raw = _official_hashes()
    if (identities["geometry_sha256"] in official_geometry
            or identities["gameplay_sha256"] in official_gameplay
            or identities["raw_start_frame_sha256"] in official_raw):
        certify.last_reason = "official_copy"
        return None
    result = search(env, limit=action_limit, node_limit=node_limit)
    if not result.solved:
        if result.truncated:
            certify.last_reason = "search_truncated"
        elif result.unsupported:
            certify.last_reason = "unsupported"
        else:
            certify.last_reason = "proven_unsolvable"
        return None
    try:
        mechanics, action_hash, observation = _route_certificate(level, context, result.actions)
    except ContractMismatch as exc:
        certify.last_reason = "native_replay"
        return None
    errors = profile_errors(difficulty, metrics, mechanics)
    if errors:
        certify.last_reason = "mechanic_use"
        return None
    if len(result.actions) != REFERENCE_PROFILES[difficulty]["reference_actions"]:
        certify.last_reason = "action_length"
        return None

    candidate.update(identities)
    candidate.update({
        "solution": [[action, x, y] for action, x, y in result.actions],
        "context_solution": [[action, x, y] for action, x, y in result.actions],
        "solution_length": len(result.actions),
        "action_sequence_sha256": action_hash,
        "structural_metrics": metrics,
        "solution_mechanics": mechanics,
        "engine_verified": True,
        "context_engine_verified": True,
        "engine_win": True,
        "engine_budget": layout.initial_energy,
        "native_budget": layout.initial_energy,
        "proof_level_index": context,
        "solution_energy_cost": mechanics["energy_cost"],
        "teacher_model_exact": True,
        "solution_optimality": "not_claimed_constructive",
        "search_truncated": False,
        "search_unsupported": False,
        "search_expanded": result.expanded,
        "search_limit": node_limit,
        "generation_limits": {"action_limit": action_limit, "node_limit": node_limit},
    })
    proof = {
        "format": FORMAT,
        "source": candidate["source"],
        "game": candidate["game"],
        "seed": candidate["seed"],
        "requested_seed": candidate["requested_seed"],
        "original_seed": candidate["original_seed"],
        "effective_seed": candidate["effective_seed"],
        "difficulty": difficulty,
        "split": candidate["split"],
        "effective_split": candidate["effective_split"],
        "context_index": context,
        "training_context_index": context,
        "verification_level_index": context,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "quality_version": QUALITY_VERSION,
        "context_engine_verified": True,
        "engine_win": True,
        "levels_completed": 1,
        "search_truncated": False,
        "search_unsupported": False,
        "teacher_model_exact": True,
        "solution_optimality": "not_claimed_constructive",
        "action_limit": action_limit,
        "search_limit": node_limit,
        "search_expanded": result.expanded,
        "solution_length": len(result.actions),
        "native_budget": layout.initial_energy,
        "engine_budget": layout.initial_energy,
        "proof_level_index": context,
        "solution_energy_cost": mechanics["energy_cost"],
        "first_completion_action_index": mechanics["first_completion_action_index"],
        "geometry_sha256": identities["geometry_sha256"],
        "geometry_d4_sha256": identities["geometry_d4_sha256"],
        "gameplay_sha256": identities["gameplay_sha256"],
        "raw_start_frame_sha256": identities["raw_start_frame_sha256"],
        "action_sequence_sha256": action_hash,
    }
    candidate["proof"] = proof
    certify.last_reason = None
    return candidate


certify.last_reason = None


def generate(seed, difficulty=1, attempts=DEFAULT_ATTEMPTS, limit=None,
             node_limit=DEFAULT_NODE_LIMIT, *, split="train", record_rejection=None):
    """Generate one deterministic tier with canonical three-way partitioning."""
    requested_seed = _integer(seed, "seed", minimum=0)
    difficulty = _integer(difficulty, "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    attempts = _integer(attempts, "attempts", minimum=0)
    if attempts > MAX_GENERATION_ATTEMPTS:
        raise ValueError(f"attempts cannot exceed {MAX_GENERATION_ATTEMPTS}")
    node_limit = _integer(node_limit, "node_limit", minimum=0)
    if node_limit > MAX_NODE_LIMIT:
        raise ValueError(f"node_limit cannot exceed {MAX_NODE_LIMIT}")
    seed_value = effective_seed(requested_seed, split)
    rng = random.Random(f"{MECHANICS_VERSION}:{seed_value}:{difficulty}")
    exclusions = Counter()
    for attempt in range(1, attempts + 1):
        try:
            draft = _draft(rng, requested_seed, split, difficulty)
            level = build_level(draft)
            probe = _native_context(level, difficulty - 1)
            identities = identity_hashes(extract(probe), probe.render())
        except (ValueError, AssertionError) as exc:
            reason = "invalid_geometry"
            exclusions[reason] += 1
            if record_rejection:
                record_rejection({"seed": requested_seed, "difficulty": difficulty,
                                  "attempt": attempt, "reason": reason, "detail": str(exc)})
            continue
        if identities["geometry_split"] != split:
            reason = "geometry_split"
            exclusions[reason] += 1
            if record_rejection:
                record_rejection({"seed": requested_seed, "difficulty": difficulty,
                                  "attempt": attempt, "reason": reason})
            continue
        verified = certify(draft, limit=limit, node_limit=node_limit)
        if verified is not None:
            verified["generation_attempt"] = attempt
            verified["generation_exclusions"] = dict(exclusions)
            verified["generation_rejections"] = dict(exclusions)
            verified["generation_limits"]["attempts"] = attempts
            generate.last_report = {
                "accepted": True,
                "seed": requested_seed,
                "effective_seed": seed_value,
                "difficulty": difficulty,
                "split": split,
                "attempt": attempt,
                "rejections": dict(exclusions),
            }
            return verified
        reason = certify.last_reason or "certification_failed"
        exclusions[reason] += 1
        if record_rejection:
            record_rejection({"seed": requested_seed, "difficulty": difficulty,
                              "attempt": attempt, "reason": reason})
    generate.last_report = {
        "accepted": False,
        "seed": requested_seed,
        "effective_seed": seed_value,
        "difficulty": difficulty,
        "split": split,
        "attempts": attempts,
        "rejections": dict(exclusions),
    }
    return None


generate.last_report = None


def _child_seed(game_seed, difficulty):
    payload = f"{MECHANICS_VERSION}:game:{int(game_seed)}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def generate_game(game_seed, *, split="train", difficulties=None,
                  attempts=DEFAULT_ATTEMPTS, limit=None,
                  node_limit=DEFAULT_NODE_LIMIT):
    """Generate the full game, or an explicitly labelled increasing smoke subset."""
    game_seed = _integer(game_seed, "game_seed", minimum=0)
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if (not selected or any(type(value) is not int or value not in DIFFICULTIES for value in selected)
            or tuple(sorted(set(selected))) != selected):
        raise ValueError("difficulties must be a nonempty, unique, increasing subset of 1..8")
    specs = []
    for ordinal, difficulty in enumerate(selected):
        child_seed = _child_seed(game_seed, difficulty)
        spec = generate(
            child_seed,
            difficulty,
            attempts=attempts,
            limit=limit,
            node_limit=node_limit,
            split=split,
        )
        if spec is None:
            generate_game.last_report = {
                "accepted": False, "game_seed": game_seed, "split": split,
                "child_ordinal": ordinal, "child_difficulty": difficulty,
                "child_seed": child_seed, "child_report": generate.last_report,
            }
            return None
        spec["game_seed"] = game_seed
        spec["game_ordinal"] = ordinal
        spec["child_seed"] = child_seed
        specs.append(spec)
    if selected != DIFFICULTIES:
        for spec in specs:
            spec["sequence_kind"] = "explicit-smoke-subset"
        generate_game.last_report = {"accepted": True, "sequence_kind": "explicit-smoke-subset"}
        return specs

    env = Env([build_level(spec) for spec in specs])
    env.reset()
    observation = None
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            generate_game.last_report = {"accepted": False, "reason": "sequential_context",
                                         "child_difficulty": index + 1}
            return None
        start_score = env.levels_completed
        for action_index, (action, x, y) in enumerate(spec["solution"]):
            observation = env.perform(action, x, y)
            if env.levels_completed > start_score and action_index != len(spec["solution"]) - 1:
                generate_game.last_report = {"accepted": False, "reason": "early_native_win",
                                             "child_difficulty": index + 1}
                return None
        if env.levels_completed != index + 1:
            generate_game.last_report = {"accepted": False, "reason": "sequential_native_replay",
                                         "child_difficulty": index + 1}
            return None
        spec["proof"] = dict(spec["proof"], sequential_context_index=index,
                             sequential_engine_verified=True)
    if observation is None or not observation.won:
        generate_game.last_report = {"accepted": False, "reason": "final_state_not_win"}
        return None
    sequence_hash = _sha([spec["gameplay_sha256"] for spec in specs])
    for spec in specs:
        spec["sequence_kind"] = "full-official-context"
        spec["game_sequence_sha256"] = sequence_hash
    generate_game.last_report = {"accepted": True, "sequence_kind": "full-official-context",
                                 "game_sequence_sha256": sequence_hash}
    return specs


generate_game.last_report = None


def _validate_full_standard_strict(spec, curriculum_entry=None):
    """Fail closed on schema, provenance, identity, and native replay claims."""
    row = _jsonable(spec)
    if type(row) is not dict:
        raise ContractMismatch("generated row must be a JSON object")
    if type(curriculum_entry) is not dict:
        raise ContractMismatch("curriculum entry must be a mapping")

    required = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "quality_version": QUALITY_VERSION,
        "source": "generated_only",
        "game": "sb26",
        "coverage": "full shipped SB26 frame/link/colour-sharing/cyclic-prefix grammar",
        "omitted_mechanics": [],
    }
    for field, expected in required.items():
        _require_exact(row, field, expected)
    difficulty = _require_int(row, "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ContractMismatch("difficulty is outside the eight-tier curriculum")
    context = difficulty - 1
    for field in ("context_index", "training_context_index", "verification_level_index"):
        _require_exact(row, field, context)
    if (type(curriculum_entry.get("difficulty")) is not int
            or type(curriculum_entry.get("context_index")) is not int
            or type(curriculum_entry.get("search_work")) is not int
            or not 0 < curriculum_entry["search_work"] <= MAX_NODE_LIMIT
            or curriculum_entry["difficulty"] != difficulty
            or curriculum_entry["context_index"] != context):
        raise ContractMismatch("curriculum entry does not match the row")

    split = row.get("split")
    if type(split) is not str or split not in SPLITS:
        raise ContractMismatch("split is missing or invalid")
    requested_seed = _require_int(row, "requested_seed", minimum=0)
    mapped = effective_seed(requested_seed, split)
    _require_exact(row, "original_seed", requested_seed)
    _require_exact(row, "effective_split", split)
    _require_exact(row, "effective_seed", mapped)
    _require_exact(row, "seed", mapped)

    structural_schema = {
        "frames": {"arity", "position", "colour"},
        "fixed": {"kind", "colour", "frame", "slot"},
        "tray": {"kind", "colour", "position"},
        "connections": {"from_frame", "from_slot", "to_frame"},
    }
    for field, keys in structural_schema.items():
        value = row.get(field)
        if type(value) is not list:
            raise ContractMismatch(f"{field} must be a present list")
        for index, item in enumerate(value):
            if type(item) is not dict or set(item) != keys:
                raise ContractMismatch(f"{field}[{index}] has malformed fields")
    for field in ("goals", "goal_positions"):
        if field not in row or type(row[field]) is not list:
            raise ContractMismatch(f"{field} must be a present list")

    top_claims = {
        "engine_verified": True,
        "context_engine_verified": True,
        "engine_win": True,
        "teacher_model_exact": True,
        "solution_optimality": "not_claimed_constructive",
        "search_truncated": False,
        "search_unsupported": False,
        "engine_budget": 64,
        "native_budget": 64,
        "proof_level_index": context,
    }
    for field, expected in top_claims.items():
        _require_exact(row, field, expected)

    level = build_level(row)
    env = _native_context(level, context)
    frame = env.render()
    layout = extract(env)
    if not layout.exact:
        raise ContractMismatch("native layout is unsupported: " + "; ".join(layout.unsupported))
    metrics = _structural_metrics(layout, frame)
    if not _same_exact_json(row.get("structural_metrics"), metrics):
        raise ContractMismatch("stored structural metrics do not match native recomputation")
    errors = profile_errors(difficulty, metrics)
    if errors:
        raise ContractMismatch("profile mismatch: " + "; ".join(errors))
    identities = identity_hashes(layout, frame)
    for field, actual in identities.items():
        _require_exact(row, field, actual)
    if identities["geometry_split"] != split:
        raise ContractMismatch("canonical semantic partition differs from requested split")
    official_geometry, official_gameplay, official_raw = _official_hashes()
    if (identities["geometry_sha256"] in official_geometry
            or identities["gameplay_sha256"] in official_gameplay
            or identities["raw_start_frame_sha256"] in official_raw):
        raise ContractMismatch("generated row duplicates an official identity")

    solution = row.get("solution")
    if type(solution) is not list or not solution:
        raise ContractMismatch("solution must be a nonempty list")
    for index, action in enumerate(solution):
        if type(action) is not list or len(action) != 3 or type(action[0]) is not int:
            raise ContractMismatch(f"solution action {index} is malformed")
        action_id, x, y = action
        if action_id == names.ACTION_CLICK:
            if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
                raise ContractMismatch(f"solution click {index} has invalid coordinates")
        elif action_id in (names.ACTION_SUBMIT, names.ACTION_UNDO):
            if x is not None or y is not None:
                raise ContractMismatch(f"solution action {index} must use null coordinates")
        else:
            raise ContractMismatch(f"solution action {index} has an invalid id")
    mechanics, action_hash, _ = _route_certificate(level, context, solution)
    if not _same_exact_json(row.get("context_solution"), solution):
        raise ContractMismatch("context_solution must exactly mirror solution")
    if not _same_exact_json(row.get("solution_mechanics"), mechanics):
        raise ContractMismatch("stored mechanic-use certificate does not match route replay")
    _require_exact(row, "action_sequence_sha256", action_hash)
    _require_exact(row, "solution_length", len(solution))
    _require_exact(row, "solution_length", REFERENCE_PROFILES[difficulty]["reference_actions"])
    _require_exact(row, "solution_energy_cost", mechanics["energy_cost"])
    if mechanics["route_minimum_energy"] != 64 - mechanics["energy_cost"]:
        raise ContractMismatch("route energy arithmetic is inconsistent with native budget")
    if mechanics["first_completion_action_index"] != len(solution) - 1:
        raise ContractMismatch("stored route does not first complete on its final action")
    errors = profile_errors(difficulty, metrics, mechanics)
    if errors:
        raise ContractMismatch("mechanic-use mismatch: " + "; ".join(errors))

    limits = row.get("generation_limits")
    if type(limits) is not dict or set(limits) != {"action_limit", "node_limit", "attempts"}:
        raise ContractMismatch("generation limits do not match the curriculum")
    action_limit = _require_int(
        limits, "action_limit", minimum=len(solution), maximum=MAX_ACTION_LIMIT
    )
    node_limit = _require_int(limits, "node_limit", minimum=1, maximum=MAX_NODE_LIMIT)
    attempts = _require_int(
        limits, "attempts", minimum=1, maximum=MAX_GENERATION_ATTEMPTS
    )
    if node_limit != curriculum_entry["search_work"]:
        raise ContractMismatch("generation node limit differs from the curriculum")

    proof = row.get("proof")
    if type(proof) is not dict:
        raise ContractMismatch("proof object is missing")
    mirrors = {
        "format": FORMAT,
        "source": "generated_only",
        "game": "sb26",
        "seed": row["seed"],
        "requested_seed": row["requested_seed"],
        "original_seed": row["original_seed"],
        "effective_seed": row["effective_seed"],
        "difficulty": difficulty,
        "split": split,
        "effective_split": split,
        "context_index": context,
        "training_context_index": context,
        "verification_level_index": context,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "quality_version": QUALITY_VERSION,
        "context_engine_verified": True,
        "engine_win": True,
        "levels_completed": 1,
        "search_truncated": False,
        "search_unsupported": False,
        "teacher_model_exact": True,
        "solution_optimality": "not_claimed_constructive",
        "geometry_sha256": identities["geometry_sha256"],
        "geometry_d4_sha256": identities["geometry_d4_sha256"],
        "gameplay_sha256": identities["gameplay_sha256"],
        "raw_start_frame_sha256": identities["raw_start_frame_sha256"],
        "action_sequence_sha256": action_hash,
        "action_limit": action_limit,
        "search_limit": curriculum_entry["search_work"],
        "search_expanded": row.get("search_expanded"),
        "solution_length": row.get("solution_length"),
        "native_budget": 64,
        "engine_budget": 64,
        "proof_level_index": context,
        "solution_energy_cost": mechanics["energy_cost"],
        "first_completion_action_index": len(solution) - 1,
    }
    for field, expected in mirrors.items():
        _require_exact(proof, field, expected)
    proof_keys = set(mirrors)
    sequential_keys = {"sequential_context_index", "sequential_engine_verified"}
    present_sequential = sequential_keys & set(proof)
    if present_sequential:
        if present_sequential != sequential_keys:
            raise ContractMismatch("sequential proof fields must be present together")
        _require_exact(proof, "sequential_context_index", context)
        _require_exact(proof, "sequential_engine_verified", True)
        proof_keys |= sequential_keys
    if set(proof) != proof_keys:
        raise ContractMismatch("proof contains unrecognized or stale fields")

    _require_int(row, "search_expanded", minimum=1, maximum=curriculum_entry["search_work"])
    _require_exact(row, "search_limit", curriculum_entry["search_work"])
    exclusions = row.get("generation_exclusions")
    if (type(exclusions) is not dict
            or any(type(key) is not str or key not in GENERATION_REJECTION_REASONS
                   or type(value) is not int or value < 1
                   for key, value in (exclusions.items() if type(exclusions) is dict else ()))):
        raise ContractMismatch("bounded generation rejection counts are missing or malformed")
    if not _same_exact_json(row.get("generation_rejections"), exclusions):
        raise ContractMismatch("generation rejection summaries disagree")
    generation_attempt = _require_int(row, "generation_attempt", minimum=1, maximum=attempts)
    if sum(exclusions.values()) != generation_attempt - 1:
        raise ContractMismatch("generation rejection counts do not precede the accepted attempt")


def validate_full_standard(spec, curriculum_entry):
    """Return full-contract violations; malformed rows fail closed."""
    try:
        _validate_full_standard_strict(spec, curriculum_entry)
    except (ContractMismatch, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        return [str(exc)]
    return []


def build_game(specs):
    """Validate and build only a complete ordered eight-level game."""
    specs = list(specs)
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("a full SB26 game must contain exactly eight levels")
    splits = [spec.get("split") for spec in specs if type(spec) is dict]
    if len(splits) != len(specs) or any(split not in SPLITS for split in splits) or len(set(splits)) != 1:
        raise ValueError("all full-game levels must be mappings in the same declared split")
    game_fields = {"game_seed", "game_ordinal", "child_seed", "sequence_kind",
                   "game_sequence_sha256"}
    enriched = [game_fields & set(spec) for spec in specs]
    if any(enriched) and any(fields != game_fields for fields in enriched):
        raise ValueError("full-game provenance fields must be all present or all absent")
    has_game_provenance = bool(enriched[0])
    if has_game_provenance:
        game_seed = specs[0]["game_seed"]
        if type(game_seed) is not int or game_seed < 0:
            raise ValueError("game_seed must be an exact nonnegative integer")
    levels = []
    identities = set()
    for expected, spec in zip(DIFFICULTIES, specs):
        if type(spec.get("difficulty")) is not int or spec["difficulty"] != expected:
            raise ValueError("SB26 game difficulties must be exactly 1..8 in order")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][expected - 1])
        if errors:
            raise ValueError(f"invalid tier {expected}: {'; '.join(errors)}")
        sequential_keys = {"sequential_context_index", "sequential_engine_verified"}
        has_sequential_proof = sequential_keys <= set(spec["proof"])
        if has_game_provenance:
            if (type(spec["game_seed"]) is not int or spec["game_seed"] != game_seed
                    or type(spec["game_ordinal"]) is not int
                    or spec["game_ordinal"] != expected - 1
                    or type(spec["child_seed"]) is not int
                    or spec["child_seed"] != _child_seed(game_seed, expected)
                    or spec["child_seed"] != spec["requested_seed"]
                    or spec["sequence_kind"] != "full-official-context"
                    or not has_sequential_proof):
                raise ValueError("full-game provenance does not match its ordered child")
        elif has_sequential_proof:
            raise ValueError("standalone-generated specs cannot carry sequential proof claims")
        if spec["gameplay_sha256"] in identities:
            raise ValueError("duplicate gameplay identity in full game")
        identities.add(spec["gameplay_sha256"])
        levels.append(build_level(spec))
    if has_game_provenance:
        sequence_hash = _sha([spec["gameplay_sha256"] for spec in specs])
        if any(type(spec["game_sequence_sha256"]) is not str
               or spec["game_sequence_sha256"] != sequence_hash for spec in specs):
            raise ValueError("game sequence identity does not match ordered gameplay identities")
    env = Env(levels)
    env.reset()
    observation = None
    for index, spec in enumerate(specs):
        if env.level_index != index:
            raise ValueError(f"native sequence did not enter tier {index + 1}")
        start_score = env.levels_completed
        for action_index, (action, x, y) in enumerate(spec["solution"]):
            observation = env.perform(action, x, y)
            if env.levels_completed > start_score:
                if action_index != len(spec["solution"]) - 1:
                    raise ValueError(f"tier {index + 1} route completes before final action")
                break
        if env.levels_completed != index + 1:
            raise ValueError(f"tier {index + 1} route failed in the full native sequence")
    if observation is None or not observation.won:
        raise ValueError("full native sequence did not reach WIN")
    return [build_level(spec) for spec in specs]
