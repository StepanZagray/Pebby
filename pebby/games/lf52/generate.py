"""Full-mechanics, reference-calibrated LF52 procedural generation.

Every tier is built for its actual native logical-level context.  A private
generated teacher guide supplies the accepted program, which is then replayed
in the native context for its certificate.  Proof ``expanded``/``generated``
counts record program length/length + 1; they do not claim a fresh independent
shortest-path search.  Official layouts and routes are never copied into
generated data.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import json
from numbers import Integral
import random

from arcengine import GameState, Level

from . import names
from .env import Env, official_levels, replay, upstream
from .layout import extract
from .plan import (
    DEFAULT_LIMIT,
    _jump_options,
    _jump_transition,
    _scripted_reset,
    _state,
    search,
)


FORMAT = "pebby.lf52.level.v5"
GENERATOR_VERSION = 5
MECHANICS_VERSION = "lf52-full-source-semantics-v5"
QUALITY_VERSION = "lf52-official-ten-tier-v4"
GEOMETRY_VERSION = "lf52-executed-typed-d4-three-way-v4"
SOURCE_ID = "lf52-271a04aa"
DIFFICULTIES = tuple(range(1, 11))
MAX_ATTEMPTS = 64
SEARCH_LIMITS = {difficulty: 250_000 for difficulty in DIFFICULTIES}

# One shipped level exists per tier.  These are exact source/engine
# measurements, not population estimates.  ``reference_solution_actions`` is
# null where the bounded teacher audit has not yet produced a positive route;
# generated gates never fabricate those missing measurements.
REFERENCE_PROFILES = {
    1: dict(width=7, height=7, ordinary=33, moving=0, rails=0, obstacles=0, pegs=5, visible_cells=33, reference_solution_actions=8),
    2: dict(width=9, height=8, ordinary=25, moving=1, rails=19, obstacles=0, pegs=5, visible_cells=26, reference_solution_actions=34),
    3: dict(width=14, height=9, ordinary=47, moving=2, rails=14, obstacles=0, pegs=14, visible_cells=26, reference_solution_actions=46),
    4: dict(width=18, height=12, ordinary=72, moving=3, rails=18, obstacles=12, pegs=6, visible_cells=25, reference_solution_actions=50),
    5: dict(width=20, height=9, ordinary=40, moving=3, rails=53, obstacles=7, pegs=6, visible_cells=13, reference_solution_actions=89),
    6: dict(width=26, height=8, ordinary=66, moving=3, rails=24, obstacles=3, pegs=8, visible_cells=37, reference_solution_actions=91),
    7: dict(width=23, height=9, ordinary=32, moving=6, rails=59, obstacles=15, pegs=3, visible_cells=16, reference_solution_actions=144),
    8: dict(width=9, height=11, ordinary=55, moving=5, rails=22, obstacles=7, pegs=6, visible_cells=48, reference_solution_actions=68),
    9: dict(width=21, height=7, ordinary=68, moving=1, rails=15, obstacles=5, pegs=9, visible_cells=29, reference_solution_actions=106),
    10: dict(width=9, height=14, ordinary=27, moving=8, rails=36, obstacles=0, pegs=12, visible_cells=31, reference_solution_actions=58),
}

REQUIRED_MECHANICS = {
    1: ("same_color_removals", "scripted_reset_choices"),
    2: ("same_color_removals", "rail_actions", "moving_hole_moves", "scripted_reset_choices"),
    3: ("same_color_removals", "rail_actions", "moving_hole_moves", "scripted_reset_choices", "camera_scrolls"),
    4: ("same_color_removals", "rail_actions", "moving_hole_moves", "blocker_jumps", "camera_scrolls"),
    5: ("same_color_removals", "rail_actions", "moving_hole_moves", "blocker_jumps", "camera_scrolls"),
    6: ("same_color_removals", "rail_actions", "moving_hole_moves", "blocker_jumps", "cross_color_jumps", "scripted_reset_choices", "camera_scrolls"),
    7: ("same_color_removals", "rail_actions", "moving_hole_moves", "blocker_jumps", "cross_color_jumps", "camera_scrolls"),
    8: ("same_color_removals", "rail_actions", "moving_hole_moves", "blocker_jumps", "cross_color_jumps", "blue_peg_interactions", "camera_scrolls"),
    9: ("same_color_removals", "rail_actions", "moving_hole_moves", "blocker_jumps", "cross_color_jumps", "blue_peg_interactions", "camera_scrolls"),
    10: ("same_color_removals", "rail_actions", "moving_hole_moves", "cross_color_jumps", "blue_peg_interactions"),
}

# Explicit engineering tolerances around one measured reference route per
# tier.  These are generated-quality bounds, not population intervals or
# shortest-route claims.
ACTION_TOLERANCES = {
    1: (6, 16), 2: (20, 50), 3: (20, 60), 4: (32, 80), 5: (28, 110),
    6: (28, 120), 7: (40, 180), 8: (12, 90), 9: (24, 130), 10: (24, 100),
}

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_VERSION,
    "curriculum": [
        {"difficulty": difficulty, "context_index": difficulty - 1, "search_work": SEARCH_LIMITS[difficulty]}
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "lf52-source-lines-4630-4997-and-native-profile-v1",
        "solution_mechanics": "lf52-root-astra-native-mechanics-acceptance-v5",
        "native_budget": "lf52-source-lines-5763-5771",
        "context_engine_replay": "lf52-root-three-split-sequential-replay-v5",
        "novelty_split": "lf52-author-three-split-30-row-identity-census-v5",
        "bounded_rejections": "lf52-generation-and-bank-bounds-reviewed-v5",
    },
    "caveats": [
        "each tier has one official reference, so tolerances are engineering bounds rather than population intervals",
        "official teacher routes are bounded positive witnesses and do not claim shortest-path optimality",
        "D4 identity does not claim graph-isomorphism novelty",
        "generation uses finite constructive grammars rather than sampling the full space of valid LF52 boards",
        "accepted routes are certificate-guided native replays, not fresh independent solver results",
        "tiny early tutorial classes have less structural diversity than later rail/color tiers",
        "tier 4 witnesses traverse six of twelve blockers and expose all twelve as legal choices, without claiming every installed object is indispensable",
        "bounded whole-game generation can return None; outer parent-seed resampling remains part of the shared collection contract",
        "the revised v5 grammar has complete three-split game evidence but no new population-scale route-diversity estimate",
        "tier 10 native evidence makes the transported blue payload causally necessary for the stored route, while deleting the separate leader can still leave a winning route",
    ],
}


def _integer(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def effective_seed(seed, split="train"):
    seed = _integer(seed, "seed", minimum=0)
    if split not in names.SPLITS:
        raise ValueError("split must be train, validation, or test")
    material = f"{SOURCE_ID}:{split}:{seed}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _cell_list(cells):
    return [list(cell) for cell in sorted(cells)]


def _peg_list(pegs):
    return [{"cell": list(cell), "kind": kind} for cell, kind in sorted(pegs.items())]


def _board(spec):
    board = spec.get("board")
    if not isinstance(board, Mapping):
        raise ValueError("board must be an object")

    def cells(key):
        raw = board.get(key)
        if not isinstance(raw, list):
            raise ValueError(f"board.{key} must be a list")
        result = []
        for index, value in enumerate(raw):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"board.{key}[{index}] must be [x, y]")
            result.append((_integer(value[0], f"board.{key}[{index}][0]", minimum=0), _integer(value[1], f"board.{key}[{index}][1]", minimum=0)))
        if len(result) != len(set(result)):
            raise ValueError(f"board.{key} must not contain duplicates")
        return set(result)

    ordinary = cells("ordinary")
    moving = cells("moving")
    rails = cells("rails")
    obstacles = cells("obstacles")
    raw_pegs = board.get("pegs")
    if not isinstance(raw_pegs, list):
        raise ValueError("board.pegs must be a list")
    pegs = {}
    for index, value in enumerate(raw_pegs):
        if not isinstance(value, Mapping):
            raise ValueError(f"board.pegs[{index}] must be an object")
        raw_cell = value.get("cell")
        if not isinstance(raw_cell, list) or len(raw_cell) != 2:
            raise ValueError(f"board.pegs[{index}].cell must be [x, y]")
        cell = (_integer(raw_cell[0], "peg x", minimum=0), _integer(raw_cell[1], "peg y", minimum=0))
        kind = value.get("kind")
        if kind not in names.PEG_KINDS:
            raise ValueError(f"board.pegs[{index}].kind is not a native peg kind")
        if cell in pegs:
            raise ValueError("board.pegs must not stack pegs")
        pegs[cell] = kind
    if ordinary & moving:
        raise ValueError("ordinary and moving holes must be disjoint")
    if ordinary & rails:
        raise ValueError("ordinary holes and rails cannot share a native cell")
    if not moving <= rails:
        raise ValueError("every moving hole must sit on a rail")
    if not set(pegs) <= ordinary | moving:
        raise ValueError("every peg must sit on a hole")
    if not obstacles <= ordinary | moving:
        raise ValueError("every obstacle must sit on a hole")
    if set(pegs) & obstacles:
        raise ValueError("pegs and obstacles must not share a cell")
    return ordinary, moving, rails, obstacles, pegs


def _descriptor_from_board(spec):
    ordinary, moving, rails, obstacles, pegs = _board(spec)
    occupied = ordinary | moving | rails | obstacles | set(pegs)
    width = max(x for x, _ in occupied) + 1
    height = max(y for _, y in occupied) + 1
    rows = []
    ordinary_symbols = {
        names.PEG: "x", names.PEG_RED: "r", names.PEG_BLUE: "b", names.PEG_GRAY: "g",
    }
    legend = {}

    def rail_sprite(cell):
        x, y = cell
        up = (x, y - 1) in rails
        right = (x + 1, y) in rails
        down = (x, y + 1) in rails
        left = (x - 1, y) in rails
        mask = (up, right, down, left)
        # The vendored artwork contains the straight pieces, all four corners,
        # and the two horizontal junctions used by the official boards.
        return {
            (False, True, False, True): names.RAIL_PREFIX,
            (True, False, True, False): f"{names.RAIL_PREFIX}-up",
            (True, True, False, False): f"{names.RAIL_PREFIX}-L",
            (True, False, False, True): f"{names.RAIL_PREFIX}-3",
            (False, True, True, False): f"{names.RAIL_PREFIX}-<",
            (False, False, True, True): f"{names.RAIL_PREFIX}->",
            (False, True, True, True): f"{names.RAIL_PREFIX}-T",
            (True, True, False, True): f"{names.RAIL_PREFIX}-t",
        }.get(mask, f"{names.RAIL_PREFIX}-up" if up or down else names.RAIL_PREFIX)

    def rail_symbol(cell):
        stack = []
        if cell in obstacles:
            stack.append(names.OBSTACLE)
        elif cell in pegs:
            stack.append(pegs[cell])
        if cell in moving:
            stack.append(names.MOVING_HOLE)
        stack.append(rail_sprite(cell))
        symbol = chr(0xE000 + len(legend))
        legend[symbol] = stack
        return symbol
    for y in range(height):
        row = []
        for x in range(width):
            cell = x, y
            if cell in moving:
                char = rail_symbol(cell)
            elif cell in ordinary:
                if cell in obstacles:
                    char = "p"
                elif cell in pegs:
                    char = ordinary_symbols[pegs[cell]]
                else:
                    char = "."
            elif cell in rails:
                char = rail_symbol(cell)
            else:
                char = " "
            row.append(char)
        rows.append("".join(row))
    return {
        "kind": names.FULL_GENERATED_KIND,
        "rows_top_down": rows,
        "legend": legend,
    }


def build_level(spec):
    """Reconstruct one generated native placeholder and logical board."""
    if not isinstance(spec, Mapping) or spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    descriptor = _descriptor_from_board(spec)
    if isinstance(spec.get("solution"), list):
        # A constructive guide is planner metadata, not native geometry.  The
        # public planner replays it on a clone before returning a witness, and
        # validation independently replays it from the executable board.
        descriptor["teacher_actions"] = json.loads(json.dumps(spec["solution"]))
    placeholder = upstream().sprites[names.PLACEHOLDER_SPRITE].clone().set_position(3, 2)
    return Level(
        sprites=[placeholder],
        grid_size=(8, 8),
        data={names.LEVEL_LAYOUT_DATA: descriptor},
        name=f"generated-lf52-d{difficulty}-s{spec.get('effective_seed', 0)}",
    )


def _context_env(spec):
    difficulty = int(spec["difficulty"])
    levels = official_levels()
    levels[difficulty - 1] = build_level(spec)
    env = Env(levels)
    env.reset()
    env.set_level(difficulty - 1)
    return env


def _corridor_to_target(target, other_x, rows):
    """Return an induced snake rail path whose first cell is ``target``.

    Parallel runs are separated by one empty row, so the moving entity cannot
    shortcut between them.  Generation places the entity at the final cell;
    the native arrow sequence must traverse the entire corridor to reach the
    mechanic-bearing target.
    """
    path = [target]
    current_x, current_y = target
    target_x = target[0]
    for row in rows:
        step_y = 1 if row > current_y else -1
        for y in range(current_y + step_y, row + step_y, step_y):
            path.append((current_x, y))
        current_y = row
        destination_x = other_x if current_x == target_x else target_x
        step_x = 1 if destination_x > current_x else -1
        for x in range(current_x + step_x, destination_x + step_x, step_x):
            path.append((x, current_y))
        current_x = destination_x
    if len(path) != len(set(path)):
        raise AssertionError("rail corridor must be a simple path")
    return path


def _install_corridor(moving, rails, target, other_x, rows):
    path = _corridor_to_target(target, other_x, rows)
    rails.update(path)
    moving.add(path[-1])
    return path[-1]


def _tier2_corridor(variant):
    """Return one of 32 task-carrying tutorial rail programs.

    Every path has the same native destination at ``(1, 1)`` but independently
    varies its three horizontal turns.  The carried green therefore has to
    traverse the changed topology; this is gameplay variation rather than an
    ordinary-cell decoration around a fixed four-shape core.
    """
    first_x = 7 + variant % 2
    second_x = 5 + (variant // 2) % 4
    third_x = 5 + (variant // 8) % 4
    path = [(1, 1)]
    current_x, current_y = path[-1]
    for row, destination_x in zip((3, 5, 7), (first_x, second_x, third_x)):
        for y in range(current_y + 1, row + 1):
            path.append((current_x, y))
        step_x = 1 if destination_x > current_x else -1
        for x in range(current_x + step_x, destination_x + step_x, step_x):
            path.append((x, row))
        current_x, current_y = destination_x, row
    if len(path) != len(set(path)):
        raise AssertionError("tier-2 active corridor must be a simple path")
    return path


def _tier8_corridor(variant):
    """Approach the jump lane from above so camera tracking reveals it."""
    path = [(1, 2), (1, 1), (1, 0)]
    path.extend((x, 0) for x in range(2, 6 + variant % 4 + 1))
    return path


def _base_mechanic_board(difficulty, variant):
    ordinary, moving, rails, obstacles, pegs = set(), set(), set(), set(), {}
    if difficulty == 1:
        # After the first correct jump, 4->2 is a legal scripted-reset choice;
        # 3->5 is the winning continuation.
        ordinary.update((x, 2) for x in range(1, 6))
        pegs.update({(1, 2): names.PEG, (2, 2): names.PEG, (4, 2): names.PEG})
        expansions = 1 + variant % 2
        target = (1, 2)
        for _ in range(expansions):
            source = target[0], target[1] + 2
            middle = target[0], target[1] + 1
            del pegs[target]
            pegs[source] = pegs[middle] = names.PEG
            ordinary.update((source, middle, target))
            target = source
    elif difficulty == 2:
        path = _tier2_corridor(variant)
        rails.update(path)
        start = path[-1]
        moving.add(start)
        pegs[start] = names.PEG
        ordinary.update((x, 1) for x in range(2, 6))
        pegs.update({(2, 1): names.PEG, (4, 1): names.PEG})
    elif difficulty == 3:
        start = _install_corridor(moving, rails, (2, 5), 9 + variant % 4, (3, 1))
        pegs[start] = names.PEG
        ordinary.update((x, 5) for x in range(3, 7))
        pegs.update({(3, 5): names.PEG, (5, 5): names.PEG})
    elif difficulty in (4, 5):
        if difficulty == 4:
            start = _install_corridor(moving, rails, (1, 3), 7 + variant % 4, (5, 7, 9, 11))
            pegs[start] = names.PEG
            obstacles.update({(2, 3), (4, 3)})
        else:
            start = _install_corridor(moving, rails, (4, 3), 19 - variant % 2, (1, 5, 7))
            obstacles.add(start)
            obstacles.add((2, 3)); pegs[(1, 3)] = names.PEG
        moving_destination_x = 1 if difficulty == 4 else 4
        ordinary.update((x, 3) for x in range(1, 10) if x != moving_destination_x)
        pegs.update({(6, 3): names.PEG, (8, 3): names.PEG})
        if difficulty == 4:
            obstacles.add((4, 3))
    elif difficulty == 6:
        start = _install_corridor(moving, rails, (4, 6), 11 + variant % 4, (4, 2, 0))
        obstacles.add(start)
        ordinary.update((x, 6) for x in range(1, 8) if x != 4)
        pegs.update({(1, 6): names.PEG, (2, 6): names.PEG_RED, (6, 6): names.PEG})
    elif difficulty == 7:
        start = _install_corridor(moving, rails, (5, 8), 14 + variant % 4, (6, 4, 2, 0))
        obstacles.add(start)
        ordinary.update((x, 8) for x in range(2, 9) if x != 5)
        pegs.update({(2, 8): names.PEG, (3, 8): names.PEG_RED, (7, 8): names.PEG})
    elif difficulty == 8:
        path = _tier8_corridor(variant)
        rails.update(path); moving.add(path[-1]); start = path[-1]
        pegs[start] = names.PEG
        ordinary.update((1, y) for y in range(3, 9))
        obstacles.add((1, 3)); pegs.update({(1, 5): names.PEG_BLUE, (1, 7): names.PEG})
    elif difficulty == 9:
        _install_corridor(moving, rails, (6, 5), 15 + variant % 4, (3, 1))
        ordinary.update((x, 5) for x in range(0, 6))
        obstacles.add((1, 5)); pegs.update({(0, 5): names.PEG, (3, 5): names.PEG_BLUE, (5, 5): names.PEG})
    else:
        # Keep two cells after the penultimate turn.  A one-cell final leg
        # strands the adjacent leading/trailing pair at the corner under the
        # native simultaneous-movement rule; rows 13/14 retain the intended
        # topology variation while both remain constructively reachable.
        final_row = 14 if (variant // 6) % 2 else 13
        path = _corridor_to_target(
            (2, 3), 3 + variant % 6, (5, 7, 9, 11, final_row),
        )
        rails.update(path)
        # The empty leading actor clears each occupied rail cell before the
        # trailing blue carrier advances.  A one-cell spur above the target
        # lets the leader peel away so the carrier reaches (2, 3), where it
        # participates in the green solution rather than remaining a detached
        # motion demonstration.
        rails.add((2, 2))
        moving.update((path[-2], path[-1]))
        pegs[path[-1]] = names.PEG_BLUE
        ordinary.update({(1, 3), (3, 3), (4, 3), (5, 3)})
        pegs.update({(1, 3): names.PEG, (4, 3): names.PEG})
    return ordinary, moving, rails, obstacles, pegs


_TRAP_DESTINATIONS = {
    1: {(0, 2), (2, 2), (5, 1)},
    2: {(0, 1), (2, 1), (4, 1)},
    3: {(1, 0), (0, 3), (10, 0), (13, 2), (13, 4), (10, 2), (10, 4)},
    6: {(16, 2)},
}
_CAMERA_DESTINATIONS = {
    4: {(7, 3), (12, 7)}, 5: {(7, 3), (12, 3)},
    6: {(7, 6), (18, 2)}, 7: {(8, 8)}, 9: {(6, 5)},
}


def _within(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _visible_at_origin(origin, cell):
    x, y = cell
    return 0 <= origin[0] + x * names.TILE and origin[0] + (x + 1) * names.TILE <= names.DISPLAY and 0 <= origin[1] + y * names.TILE and origin[1] + (y + 1) * names.TILE <= names.DISPLAY


def _candidate_steps(destination, difficulty, ordinary, moving, rails, obstacles, pegs, rng, origin):
    width = REFERENCE_PROFILES[difficulty]["width"]
    height = REFERENCE_PROFILES[difficulty]["height"]
    occupied = set(pegs) | obstacles | moving | rails
    choices = []
    directions = list(names.DIRECTIONS)
    rng.shuffle(directions)
    for dx, dy in directions:
        middle = destination[0] - dx, destination[1] - dy
        source = destination[0] - 2 * dx, destination[1] - 2 * dy
        if (
            _within(source, width, height)
            and _within(middle, width, height)
            and _visible_at_origin(origin, source)
            and _visible_at_origin(origin, middle)
            and source not in occupied
            and middle not in occupied
            and source != middle
        ):
            choices.append((source, middle, destination))
    return choices


def _reverse_removals(difficulty, count, ordinary, moving, rails, obstacles, pegs, rng, origin, forbidden=()):
    """Add ``count`` same-colour pegs and return their exact forward jumps."""
    if difficulty == 3 and count:
        # Tier 3's rule is not represented by a flag: admission requires a
        # real legal alternative into one of its unconditional reset cells.
        unconditional = {(1, 0), (0, 3), (10, 0), (13, 2), (13, 4)}
        for _ in range(512):
            trial_ordinary = set(ordinary)
            trial_pegs = dict(pegs)
            reversed_jumps = []
            for _step in range(count):
                choices = []
                for destination, kind in trial_pegs.items():
                    if kind != names.PEG or destination in moving or destination in forbidden:
                        continue
                    if destination in _TRAP_DESTINATIONS.get(difficulty, ()) or destination in _CAMERA_DESTINATIONS.get(difficulty, ()):
                        continue
                    choices.extend(_candidate_steps(
                        destination, difficulty, trial_ordinary, moving, rails,
                        obstacles, trial_pegs, rng, origin,
                    ))
                if not choices:
                    break
                source, middle, target = rng.choice(choices)
                del trial_pegs[target]
                trial_pegs[source] = names.PEG; trial_pegs[middle] = names.PEG
                trial_ordinary.update((source, middle, target))
                reversed_jumps.append((source, target))
            if len(reversed_jumps) != count:
                continue
            route = list(reversed(reversed_jumps))
            state = set(trial_pegs)
            trap = None
            for source, destination in route:
                for reset_cell in unconditional:
                    delta = reset_cell[0] - source[0], reset_cell[1] - source[1]
                    if (abs(delta[0]) == 2 and delta[1] == 0) or (abs(delta[1]) == 2 and delta[0] == 0):
                        middle = (source[0] + reset_cell[0]) // 2, (source[1] + reset_cell[1]) // 2
                        if middle in state and reset_cell not in state and reset_cell not in rails | moving | obstacles:
                            trap = reset_cell
                            break
                if trap is not None:
                    break
                middle = (source[0] + destination[0]) // 2, (source[1] + destination[1]) // 2
                state.remove(source); state.remove(middle); state.add(destination)
            if trap is not None:
                trial_ordinary.add(trap)
                ordinary.clear(); ordinary.update(trial_ordinary)
                pegs.clear(); pegs.update(trial_pegs)
                return route
        raise ValueError("could not grow a tier-3 removal program with a native reset alternative")

    def grow(remaining, trial_ordinary, trial_pegs, reversed_jumps):
        if remaining == 0:
            return trial_ordinary, trial_pegs, reversed_jumps
        destinations = [cell for cell, kind in trial_pegs.items() if kind == names.PEG and cell not in moving and cell not in forbidden]
        rng.shuffle(destinations)
        for destination in destinations:
            if destination in _TRAP_DESTINATIONS.get(difficulty, ()) or destination in _CAMERA_DESTINATIONS.get(difficulty, ()):
                continue
            choices = _candidate_steps(destination, difficulty, trial_ordinary, moving, rails, obstacles, trial_pegs, rng, origin)
            for source, middle, target in choices:
                next_ordinary = set(trial_ordinary); next_ordinary.update((source, middle, target))
                next_pegs = dict(trial_pegs); del next_pegs[target]
                next_pegs[source] = names.PEG; next_pegs[middle] = names.PEG
                found = grow(remaining - 1, next_ordinary, next_pegs, reversed_jumps + [(source, target)])
                if found is not None:
                    return found
        return None

    found = grow(count, set(ordinary), dict(pegs), [])
    if found is None:
        raise ValueError("could not grow the active peg-removal program")
    trial_ordinary, trial_pegs, reversed_jumps = found
    ordinary.clear(); ordinary.update(trial_ordinary)
    pegs.clear(); pegs.update(trial_pegs)
    return list(reversed(reversed_jumps))


def _relocate_over_blockers(
    difficulty, count, ordinary, moving, rails, obstacles, pegs, rng, origin,
    forbidden=(), reserved=(),
):
    """Move one route-critical green backward through ``count`` new blockers."""
    if count <= 0:
        return [], []
    reserved = set(reserved)
    destinations = [cell for cell, kind in pegs.items() if kind == names.PEG and cell not in moving and cell not in forbidden]
    rng.shuffle(destinations)
    def grow(destination, remaining, trial_ordinary, trial_obstacles, trial_pegs, reversed_jumps, added):
        if remaining == 0:
            return trial_ordinary, trial_obstacles, trial_pegs, reversed_jumps, added
        if destination in _TRAP_DESTINATIONS.get(difficulty, ()) or destination in _CAMERA_DESTINATIONS.get(difficulty, ()):
            return None
        choices = _candidate_steps(
            destination, difficulty, trial_ordinary, moving, rails,
            trial_obstacles, trial_pegs, rng, origin,
        )
        for source, middle, target in choices:
            if source in reserved or middle in reserved:
                continue
            next_ordinary = set(trial_ordinary); next_ordinary.update((source, middle, target))
            next_obstacles = set(trial_obstacles); next_obstacles.add(middle)
            next_pegs = dict(trial_pegs); kind = next_pegs.pop(target); next_pegs[source] = kind
            found = grow(
                source, remaining - 1, next_ordinary, next_obstacles,
                next_pegs, reversed_jumps + [(source, target)], added + [middle],
            )
            if found is not None:
                return found
        return None

    for original in destinations:
        found = grow(original, count, set(ordinary), set(obstacles), dict(pegs), [], [])
        if found is not None:
            trial_ordinary, trial_obstacles, trial_pegs, reversed_jumps, added = found
            ordinary.clear(); ordinary.update(trial_ordinary)
            obstacles.clear(); obstacles.update(trial_obstacles)
            pegs.clear(); pegs.update(trial_pegs)
            return list(reversed(reversed_jumps)), added
    raise ValueError("could not place the active blocker traversal")


def _blue_vacancy_chain(
    difficulty, critical, count, ordinary, moving, rails, obstacles, pegs,
    rng, origin, protected=(),
):
    """Block ``critical`` with blue pegs until a required rearrangement vacates it."""
    moves = count // 2
    if moves == 0:
        return [], []
    width = REFERENCE_PROFILES[difficulty]["width"]
    height = REFERENCE_PROFILES[difficulty]["height"]
    occupied = set(pegs) | obstacles | moving | rails

    def extend(anchors, middles):
        if len(anchors) == moves + 1:
            return anchors, middles
        current = anchors[-1]
        directions = list(names.DIRECTIONS)
        rng.shuffle(directions)
        for dx, dy in directions:
            middle = current[0] + dx, current[1] + dy
            nxt = current[0] + 2 * dx, current[1] + 2 * dy
            used = set(anchors) | set(middles) | occupied
            if (
                _within(middle, width, height)
                and _within(nxt, width, height)
                and _visible_at_origin(origin, middle)
                and _visible_at_origin(origin, nxt)
                and middle not in used
                and nxt not in used
                and nxt not in _TRAP_DESTINATIONS.get(difficulty, ())
                and nxt not in _CAMERA_DESTINATIONS.get(difficulty, ())
            ):
                found = extend(anchors + [nxt], middles + [middle])
                if found is not None:
                    return found
        return None

    # ``critical`` is an empty destination in the original core.
    if critical in occupied:
        raise ValueError("blue rearrangement target is not initially empty")
    found = extend([critical], [])
    if found is None:
        raise ValueError("could not place the active blue rearrangement")
    anchors, middles = found
    # Every native jump destination must be a hole.  The terminal anchor has
    # no peg, but omitting its ordinary hole made acceptance depend on whether
    # later envelope filling happened to select that coordinate.
    ordinary.update(anchors)
    blue_cells = anchors[:-1] + middles
    for cell in blue_cells:
        ordinary.add(cell)
        pegs[cell] = names.PEG_BLUE
    jumps = [(anchors[index], anchors[index + 1]) for index in range(moves - 1, -1, -1)]
    if count % 2:
        candidates = [
            (x, y) for y in range(height) for x in range(width)
            if _visible_at_origin(origin, (x, y))
            and (x, y) not in set(pegs) | obstacles | moving | rails
            and (x, y) not in set(protected)
        ]
        rng.shuffle(candidates)
        if not candidates:
            raise ValueError("could not place the remaining blue peg")
        cell = candidates[0]
        ordinary.add(cell); pegs[cell] = names.PEG_BLUE
        blue_cells.append(cell)
    return jumps, blue_cells


def _install_auxiliary_movers(difficulty, target_count, final_arrow, candidates, ordinary, moving, rails, obstacles, pegs, rng):
    needed = target_count - len(moving)
    if needed <= 0:
        return []
    if final_arrow not in (names.ACTION_UP, names.ACTION_DOWN):
        raise ValueError("active auxiliary movers require a vertical final transport action")
    dy = -1 if final_arrow == names.ACTION_UP else 1
    width = REFERENCE_PROFILES[difficulty]["width"]
    height = REFERENCE_PROFILES[difficulty]["height"]
    pool = list(dict.fromkeys(candidates))
    rng.shuffle(pool)
    pool.sort(key=lambda cell: pegs.get(cell) == names.PEG)
    installed = []
    auxiliary_track_cells = set()
    for target in pool:
        start = target[0], target[1] - dy
        pair = {target, start}
        neighboring_rails = {
            (cell[0] + dx, cell[1] + step_y)
            for cell in pair
            for dx, step_y in names.DIRECTIONS
        } & (rails | auxiliary_track_cells)
        if (
            len(installed) >= needed
            or target not in ordinary
            or target in moving
            or not _within(start, width, height)
            or start in ordinary | moving | rails | obstacles | set(pegs)
            or (target[0], target[1] + dy) in rails
            or neighboring_rails
            or any(
                abs(cell[0] - other[0]) + abs(cell[1] - other[1]) <= 1
                for cell in pair for other in auxiliary_track_cells
            )
        ):
            continue
        ordinary.remove(target)
        rails.update((target, start))
        moving.add(start)
        if target in obstacles:
            obstacles.remove(target); obstacles.add(start)
        if target in pegs:
            pegs[start] = pegs.pop(target)
        installed.append((start, target))
        auxiliary_track_cells.update(pair)
    remaining = needed - len(installed)
    if remaining:
        # A compact convoy is itself an active global-order/collision device:
        # the leading hole must move first to open each following destination.
        # It is used only when isolated route landings cannot host every
        # reference-count mover without joining unrelated rails.
        found = None
        for x in range(width):
            for top in range(height - remaining):
                lane = {(x, top + offset) for offset in range(remaining + 1)}
                neighbors = {
                    (cell[0] + dx, cell[1] + step_y)
                    for cell in lane for dx, step_y in names.DIRECTIONS
                }
                if not lane & (ordinary | moving | rails | obstacles | set(pegs)) and not neighbors & rails:
                    found = sorted(lane, key=lambda cell: cell[1])
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError("could not place the interacting moving-hole convoy")
        rails.update(found)
        initial = found[1:] if dy < 0 else found[:-1]
        targets = found[:-1] if dy < 0 else found[1:]
        moving.update(initial)
        installed.extend(zip(initial, targets))
    return installed


def _add_constraint_blockers(difficulty, count, ordinary, moving, rails, obstacles, pegs, rng, origin, forbidden=()):
    """Place remaining blockers on visible branch holes, never as cosmetics."""
    if count <= 0:
        return []
    width = REFERENCE_PROFILES[difficulty]["width"]
    height = REFERENCE_PROFILES[difficulty]["height"]
    candidates = [
        (x, y) for y in range(height) for x in range(width)
        if _visible_at_origin(origin, (x, y))
        and (x, y) not in moving | rails | obstacles | set(pegs)
        and (x, y) not in forbidden
    ]
    rng.shuffle(candidates)
    placed = []
    for cell in candidates:
        if len(placed) == count:
            break
        ordinary.add(cell); obstacles.add(cell); placed.append(cell)
    if len(placed) != count:
        raise ValueError("could not place all visible branch blockers")
    return placed


def _post_core_green_extension(difficulty, base_cell, count, ordinary, moving, rails, obstacles, pegs, rng, origin):
    """Add pegs reduced only after the original core reaches ``base_cell``."""
    if count <= 0:
        return []
    width = REFERENCE_PROFILES[difficulty]["width"]
    height = REFERENCE_PROFILES[difficulty]["height"]
    occupied = moving | rails | obstacles | set(pegs)
    directions = list(names.DIRECTIONS); rng.shuffle(directions)
    for dx, dy in directions:
        middle = base_cell[0] + dx, base_cell[1] + dy
        target = base_cell[0] + 2 * dx, base_cell[1] + 2 * dy
        if (
            not _within(middle, width, height) or not _within(target, width, height)
            or not _visible_at_origin(origin, middle) or not _visible_at_origin(origin, target)
            or middle in occupied or target in occupied
            or target in _TRAP_DESTINATIONS.get(difficulty, ())
            or target in _CAMERA_DESTINATIONS.get(difficulty, ())
        ):
            continue
        pseudo_ordinary = {base_cell, middle, target}
        pseudo_pegs = {base_cell: names.PEG, middle: names.PEG}
        # The first reverse expansion is forced so the core survivor is one
        # of the extension's initial pegs rather than an extra padded actor.
        extra_obstacles = set(obstacles) | (set(pegs) - {base_cell})
        try:
            later = _reverse_removals(
                difficulty, count - 1, pseudo_ordinary, set(), set(rails),
                extra_obstacles, pseudo_pegs, rng, origin, {base_cell},
            )
        except ValueError:
            continue
        for cell, kind in pseudo_pegs.items():
            if cell != base_cell:
                if cell in occupied:
                    break
                pegs[cell] = kind
        else:
            ordinary.update(pseudo_ordinary)
            return later + [(base_cell, target)]
    raise ValueError("could not place the post-core peg-removal extension")


def _complete_active_envelope(difficulty, ordinary, moving, rails, obstacles, pegs, rng):
    """Add connected choice topology after the mechanic-bearing program exists."""
    profile = REFERENCE_PROFILES[difficulty]
    width, height = profile["width"], profile["height"]
    target = max(len(ordinary), max(6, profile["ordinary"] * 3 // 4))
    occupied = ordinary | moving | rails | obstacles | set(pegs)
    frontier = [
        (x, y) for y in range(height) for x in range(width)
        if (x, y) not in occupied
        and any((x + dx, y + dy) in ordinary | moving for dx, dy in names.DIRECTIONS)
    ]
    rng.shuffle(frontier)
    while frontier and len(ordinary) < target:
        cell = frontier.pop()
        if cell in ordinary | moving | rails | obstacles | set(pegs):
            continue
        ordinary.add(cell)
        for dx, dy in names.DIRECTIONS:
            neighbor = cell[0] + dx, cell[1] + dy
            if _within(neighbor, width, height):
                frontier.append(neighbor)
    # The envelope is a measured presentation property.  These anchors are
    # excluded from seed-diversity claims; active program hashes are separate.
    for cell in ((0, 0), (width - 1, height - 1)):
        if cell not in moving | rails | obstacles | set(pegs):
            ordinary.add(cell)


def _jump_cells(jumps):
    cells = set()
    for source, destination in jumps:
        cells.update((
            source,
            ((source[0] + destination[0]) // 2, (source[1] + destination[1]) // 2),
            destination,
        ))
    return cells


def _induced_rail_path(start, target, length, forbidden, rng, width, height):
    """Find a bounded simple rail path with no nonconsecutive rail contacts."""
    forbidden = set(forbidden) - {start, target}
    visits = 0
    cap = 100_000

    def grow(path):
        nonlocal visits
        visits += 1
        if visits > cap:
            return None
        current = path[-1]
        if len(path) == length:
            return path if current == target else None
        remaining = length - len(path)
        distance = abs(current[0] - target[0]) + abs(current[1] - target[1])
        if distance > remaining or (remaining - distance) % 2:
            return None
        directions = list(names.DIRECTIONS)
        rng.shuffle(directions)
        directions.sort(
            key=lambda delta: abs(current[0] + delta[0] - target[0])
            + abs(current[1] + delta[1] - target[1])
        )
        for dx, dy in directions:
            nxt = current[0] + dx, current[1] + dy
            if (
                not _within(nxt, width, height)
                or nxt in path
                or nxt in forbidden
                or any(
                    (nxt[0] + ax, nxt[1] + ay) in path[:-1]
                    for ax, ay in names.DIRECTIONS
                )
            ):
                continue
            result = grow(path + [nxt])
            if result is not None:
                return result
        return None

    result = grow([start])
    if result is None:
        raise ValueError("could not construct the bounded active rail path")
    return result


def _path_actions(path):
    action_for_delta = {
        (0, -1): names.ACTION_UP,
        (0, 1): names.ACTION_DOWN,
        (-1, 0): names.ACTION_LEFT,
        (1, 0): names.ACTION_RIGHT,
    }
    return [
        (action_for_delta[(right[0] - left[0], right[1] - left[1])], None, None)
        for left, right in zip(path, path[1:])
    ]


def _auxiliary_track(length, direction, forbidden_rails, forbidden_cells, width, height):
    """Place one globally controlled mover on an isolated straight rail."""
    dx, dy = direction
    for y in range(height):
        for x in range(width):
            track = [(x + index * dx, y + index * dy) for index in range(length)]
            if any(not _within(cell, width, height) for cell in track):
                continue
            if set(track) & (set(forbidden_rails) | set(forbidden_cells)):
                continue
            if any(
                (cell[0] + ax, cell[1] + ay) in forbidden_rails
                for cell in track for ax, ay in names.DIRECTIONS
            ):
                continue
            return track
    raise ValueError("could not place an isolated auxiliary moving-hole track")


def _random_reverse_tree(rng, *, count, x_range, y_range, required, forbidden_destinations=()):
    """Return a varied same-colour program satisfying a semantic state gate."""
    x_values = tuple(x_range)
    y_values = tuple(y_range)
    forbidden_destinations = set(forbidden_destinations)
    for _trial in range(8192):
        occupied = {(rng.choice(x_values), rng.choice(y_values))}
        reversed_jumps = []
        for _depth in range(count - 1):
            choices = []
            for destination in occupied:
                for dx, dy in names.DIRECTIONS:
                    middle = destination[0] - dx, destination[1] - dy
                    source = destination[0] - 2 * dx, destination[1] - 2 * dy
                    if (
                        source[0] in x_values
                        and source[1] in y_values
                        and source not in occupied
                        and middle not in occupied
                    ):
                        choices.append((source, middle, destination))
            if not choices:
                break
            source, middle, destination = rng.choice(choices)
            occupied.remove(destination)
            occupied.update((source, middle))
            reversed_jumps.append((source, destination))
        if len(occupied) != count or not set(required) <= occupied:
            continue
        jumps = list(reversed(reversed_jumps))
        if any(destination in forbidden_destinations for _, destination in jumps):
            continue
        return set(occupied), jumps
    raise ValueError("could not construct the conditional same-colour program")


def _relocate_with_active_blockers(occupied, jumps, count, rng, width, height, protected=()):
    """Replace selected initial pegs by route-critical blocker approaches."""
    occupied = set(occupied)
    protected = set(protected)
    used = _jump_cells(jumps) | occupied
    preludes = []
    blockers = set()
    destinations = sorted(occupied - protected)
    rng.shuffle(destinations)
    for destination in destinations:
        directions = list(names.DIRECTIONS)
        rng.shuffle(directions)
        for dx, dy in directions:
            middle = destination[0] - dx, destination[1] - dy
            source = destination[0] - 2 * dx, destination[1] - 2 * dy
            if (
                not _within(source, width, height)
                or source[0] < 9
                or source in used
                or middle in used
            ):
                continue
            occupied.remove(destination)
            occupied.add(source)
            used.update((source, middle))
            blockers.add(middle)
            preludes.append((source, destination))
            break
        if len(preludes) == count:
            return occupied, preludes, blockers
    raise ValueError("could not make every reference blocker route-critical")


def _conditional_tier3_tree(rng):
    """Grow a varied 14-peg tree around a guaranteed late-count trap choice.

    The five-peg tail exposes ``(8,2)->(10,2)`` as a native reset alternative
    while the stored route takes a different reduction.  Three forced reverse
    expansions make the rail-delivered peg at ``(8,7)`` participate in that
    same reduction tree.  The remaining six reverse expansions are sampled by
    bounded backtracking, so geometry remains procedural without relying on a
    roughly one-in-three-hundred accidental semantic match.
    """
    tail_state = {(8, 2), (9, 2), (9, 4), (9, 5), (10, 1)}
    tail_jumps = [
        ((9, 5), (9, 3)),
        ((9, 3), (9, 1)),
        ((10, 1), (8, 1)),
        ((8, 2), (8, 0)),
    ]
    occupied = set(tail_state)
    reverse_expansions = []
    for source, middle, destination in (
        ((8, 4), (8, 3), (8, 2)),
        ((8, 6), (8, 5), (8, 4)),
        ((8, 8), (8, 7), (8, 6)),
    ):
        occupied.remove(destination)
        occupied.update((source, middle))
        reverse_expansions.append((source, destination))

    traps = set(_TRAP_DESTINATIONS[3])

    def grow(state, reversed_jumps, remaining):
        if remaining == 0:
            return state, reversed_jumps
        choices = []
        for destination in state:
            if destination == (8, 7) or destination in traps:
                continue
            for dx, dy in names.DIRECTIONS:
                middle = destination[0] - dx, destination[1] - dy
                source = destination[0] - 2 * dx, destination[1] - 2 * dy
                if (
                    8 <= source[0] < 14
                    and 0 <= source[1] < 9
                    and source not in state
                    and middle not in state
                ):
                    choices.append((source, middle, destination))
        rng.shuffle(choices)
        for source, middle, destination in choices:
            next_state = set(state)
            next_state.remove(destination)
            next_state.update((source, middle))
            found = grow(
                next_state,
                reversed_jumps + [(source, destination)],
                remaining - 1,
            )
            if found is not None:
                return found
        return None

    found = grow(occupied, reverse_expansions, 6)
    if found is None:
        raise ValueError("could not extend the guided tier-3 conditional tree")
    initial, reversed_jumps = found
    jumps = list(reversed(reversed_jumps)) + tail_jumps
    if len(initial) != 14 or (8, 7) not in initial:
        raise AssertionError("guided tier-3 tree lost its rail-delivered actor")
    return initial, jumps


def _conditional_tier3_recipe(rng):
    """Build tier 3 around its late-count conditional reset branch."""
    carrier_start = (2, 7)
    carrier_target = (8, 7)
    for _ in range(32):
        occupied, jumps = _conditional_tier3_tree(rng)
        state = set(occupied)
        conditional_witness = False
        for source, destination in jumps:
            if len(state) <= 5 and {(8, 2), (9, 2)} <= state and (10, 2) not in state:
                conditional_witness = True
            if len(state) <= 4 and {(8, 4), (9, 4)} <= state and (10, 4) not in state:
                conditional_witness = True
            middle = ((source[0] + destination[0]) // 2, (source[1] + destination[1]) // 2)
            state.difference_update((source, middle)); state.add(destination)
        if not conditional_witness or carrier_target not in _jump_cells(jumps):
            continue
        program_cells = _jump_cells(jumps)
        try:
            path = _induced_rail_path(
                carrier_start, carrier_target, 11,
                program_cells | {(10, 2), (10, 4)}, rng, 14, 9,
            )
        except ValueError:
            continue
        arrows = _path_actions(path)
        first_delta = (
            path[1][0] - path[0][0],
            path[1][1] - path[0][1],
        )
        auxiliary = _auxiliary_track(3, first_delta, set(path), program_cells, 14, 9)
        rails = set(path) | set(auxiliary)
        moving = {carrier_start, auxiliary[0]}
        pegs = {cell: names.PEG for cell in occupied - {carrier_target}}
        pegs[carrier_start] = names.PEG
        ordinary = program_cells - rails
        ordinary.update({(10, 2), (10, 4)})
        _complete_active_envelope(3, ordinary, moving, rails, set(), pegs, rng)
        return ordinary, moving, rails, set(), pegs, arrows, jumps, {
            "conditional_reset": "tier3-late-green-count",
            "conditional_witness_cells": [[8, 2], [9, 2], [10, 2]],
        }
    raise ValueError("could not combine the tier-3 conditional reset and rail program")


def _conditional_tier6_recipe(rng):
    """Build tier 6 around both native camera and red-position trap branches."""
    carrier_start = (5, 6)
    carrier_target = (15, 2)
    # Moving red over the carrier first puts red at the source-defined sentinel
    # and exposes the official 5->7 landing-camera branch before rail travel.
    red_relocation = ((4, 6), (6, 6))
    for _ in range(64):
        desired, jumps = _random_reverse_tree(
            rng,
            count=7,
            x_range=range(9, 18),
            y_range=range(8),
            required={(14, 2), carrier_target},
            forbidden_destinations={(16, 2)},
        )
        # The trap branch must be open once red reaches its source-defined
        # sentinel cell; the stored winning program deliberately avoids it.
        if (16, 2) in desired:
            continue
        try:
            relocated, blocker_jumps, blockers = _relocate_with_active_blockers(
                desired, jumps, 3, rng, 26, 8,
                protected={(14, 2), carrier_target},
            )
        except ValueError:
            continue
        all_jump_cells = _jump_cells(jumps + blocker_jumps + [red_relocation])
        try:
            path = _induced_rail_path(
                carrier_start, carrier_target, 19,
                all_jump_cells | blockers | {(7, 6), (16, 2)}, rng, 26, 8,
            )
        except ValueError:
            continue
        arrows = _path_actions(path)
        first_delta = (
            path[1][0] - path[0][0],
            path[1][1] - path[0][1],
        )
        track_a = _auxiliary_track(2, first_delta, set(path), all_jump_cells | blockers, 26, 8)
        track_b = _auxiliary_track(
            3, first_delta, set(path) | set(track_a), all_jump_cells | blockers, 26, 8,
        )
        rails = set(path) | set(track_a) | set(track_b)
        moving = {carrier_start, track_a[0], track_b[0]}
        pegs = {cell: names.PEG for cell in relocated - {carrier_target}}
        pegs[carrier_start] = names.PEG
        pegs[(4, 6)] = names.PEG_RED
        ordinary = all_jump_cells - rails
        ordinary.update({(7, 6), (16, 2)})
        obstacles = set(blockers)
        _complete_active_envelope(6, ordinary, moving, rails, obstacles, pegs, rng)
        steps = [red_relocation]
        return ordinary, moving, rails, obstacles, pegs, arrows, blocker_jumps, jumps, steps, {
            "conditional_reset": "tier6-red-at-6-6",
            "conditional_witness_cells": [[14, 2], [15, 2], [16, 2]],
            "camera_branch": [[5, 6], [7, 6]],
        }
    raise ValueError("could not combine tier-6 conditional reset, camera, and rail programs")


def _tier7_required_blocker_path(rng, forbidden=()):
    """Move red through all fourteen static blockers before the core unlocks."""
    goal = (3, 8)
    forbidden = set(forbidden)
    vertices = {
        (x, y) for x in (1, 3, 5, 7) for y in (0, 2, 4, 6, 8)
    } - {(7, 8)} - forbidden
    # The base program needs x=4/6 empty and its moving blocker at x=5.
    forbidden_middles = {(2, 8), (4, 8), (5, 8), (6, 8), (7, 8)} | forbidden
    starts = sorted(vertices - {goal})
    rng.shuffle(starts)

    def grow(path):
        if len(path) == 15:
            return path if path[-1] == goal else None
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        rng.shuffle(directions)
        current = path[-1]
        for dx, dy in directions:
            nxt = current[0] + dx, current[1] + dy
            middle = current[0] + dx // 2, current[1] + dy // 2
            if nxt in vertices and nxt not in path and middle not in forbidden_middles:
                result = grow(path + [nxt])
                if result is not None:
                    return result
        return None

    for start in starts:
        path = grow([start])
        if path is not None:
            jumps = list(zip(path, path[1:]))
            blockers = {
                ((source[0] + destination[0]) // 2, (source[1] + destination[1]) // 2)
                for source, destination in jumps
            }
            return path[0], jumps, blockers
    raise ValueError("could not construct the tier-7 blocker dependency path")


def _ordered_pair_transport(rails, leader_start, task_start, leader_goal, task_goal):
    """Find a bounded arrow program for a leading actor and task carrier.

    The task carrier begins immediately behind the leader.  Native
    leading-first updates let it enter a cell vacated earlier in the same arrow
    action.  Goals keep the actors distinct so the returned relation can be
    checked against retained native entity identities during certification.
    """
    rails = frozenset(rails)
    start = (frozenset((leader_start, task_start)), task_start)
    queue = [start]
    parent = {start: None}
    edge = {}
    for state in queue:
        moving_frozen, task = state
        if leader_goal in moving_frozen and task == task_goal:
            actions = []
            cursor = state
            while parent[cursor] is not None:
                actions.append(edge[cursor])
                cursor = parent[cursor]
            return list(reversed(actions))
        for action, dx, dy in (
            (names.ACTION_UP, 0, -1),
            (names.ACTION_DOWN, 0, 1),
            (names.ACTION_LEFT, -1, 0),
            (names.ACTION_RIGHT, 1, 0),
        ):
            moving = set(moving_frozen)
            next_task = task
            ordered = sorted(
                moving,
                key=(lambda cell: cell[0]) if dx else (lambda cell: cell[1]),
                reverse=(dx > 0 or dy > 0),
            )
            for source in ordered:
                if source not in moving:
                    continue
                destination = source[0] + dx, source[1] + dy
                if destination in moving or destination not in rails:
                    continue
                moving.remove(source)
                moving.add(destination)
                if source == next_task:
                    next_task = destination
            nxt = (frozenset(moving), next_task)
            if frozenset(moving) != moving_frozen and nxt not in parent:
                parent[nxt] = state
                edge[nxt] = (action, None, None)
                queue.append(nxt)
    raise ValueError("could not route the ordered task-carrying mover pair")


def _active_tier7_recipe(rng):
    """Require every blocker and all six globally ordered movers on the route."""
    main_path = [
        (5, 8), (5, 7), (6, 7), (7, 7), (8, 7),
        *_corridor_to_target((9, 7), 22, (5, 3, 1)),
    ]
    # The leader peels left at the end of the main track, allowing the trailing
    # blocker carrier to drop onto (5, 8).  That blocker is then the middle of
    # the required 4->6 core jump.  Four additional movers share the isolated
    # seven-cell track, preserving the official six-actor inventory.
    leader_goal = (4, 8)
    task_goal = (5, 8)
    auxiliary_track = [(x, 8) for x in range(16, 23)]
    rails = set(main_path) | {leader_goal} | set(auxiliary_track)
    if len(main_path) != 51 or len(rails) != 59:
        raise AssertionError("tier-7 active rail topology must contain 59 distinct cells")
    start, blocker_jumps, blockers = _tier7_required_blocker_path(rng, rails)
    leader_start = main_path[-2]
    task_start = main_path[-1]
    moving_positions = {leader_start, task_start} | {(x, 8) for x in range(19, 23)}
    transport = _ordered_pair_transport(
        set(main_path) | {leader_goal},
        leader_start,
        task_start,
        leader_goal,
        task_goal,
    )
    ordinary = _jump_cells(blocker_jumps) | {
        (2, 8), (3, 8), (6, 8), (7, 8), (8, 8),
    }
    moving = set(moving_positions)
    obstacles = set(blockers) | {task_start}
    pegs = {start: names.PEG_RED, (2, 8): names.PEG, (7, 8): names.PEG}
    _complete_active_envelope(7, ordinary, moving, rails, obstacles, pegs, rng)
    core_jumps = [((2, 8), (4, 8)), ((4, 8), (6, 8)), ((6, 8), (8, 8))]
    steps = (
        [("jump", jump) for jump in blocker_jumps]
        + [("action", action) for action in transport]
        + [("jump", jump) for jump in core_jumps]
    )
    return ordinary, moving, rails, obstacles, pegs, steps, {
        "grammar": "active-dependency-native-program-v2",
        "blocker_traversal_steps": len(blocker_jumps) + 1,
        "moving_hole_transport_actions": len(transport),
        "auxiliary_moving_holes": 4,
        "global_ordered_convoy": True,
        "order_leader_start": list(leader_start),
        "order_task_start": list(task_start),
        "order_leader_goal": list(leader_goal),
        "order_task_goal": list(task_goal),
        "order_task_kind": "blocker",
    }


def _base_recipe(difficulty, variant):
    ordinary, moving, rails, obstacles, pegs = _base_mechanic_board(difficulty, variant)
    spec = {
        "format": FORMAT, "difficulty": difficulty, "effective_seed": 0,
        "board": {
            "ordinary": _cell_list(ordinary), "moving": _cell_list(moving),
            "rails": _cell_list(rails), "obstacles": _cell_list(obstacles),
            "pegs": _peg_list(pegs),
        },
    }
    levels = official_levels(); levels[difficulty - 1] = build_level(spec)
    base_env = Env(levels); base_env.reset(); base_env.set_level(difficulty - 1)
    if difficulty == 10:
        task_starts = [cell for cell in moving if pegs.get(cell) == names.PEG_BLUE]
        leader_starts = [cell for cell in moving if cell not in task_starts]
        if len(task_starts) != 1 or len(leader_starts) != 1:
            raise ValueError("tier-10 base pair is malformed")
        prefix = _ordered_pair_transport(
            rails,
            leader_starts[0],
            task_starts[0],
            (2, 2),
            (2, 3),
        )
        for action in prefix:
            base_env.perform(*action)
        boundary_layout = extract(base_env)
        suffix = []
        for source, destination in (((1, 3), (3, 3)), ((3, 3), (5, 3))):
            before = extract(base_env)
            options = list(_jump_options(before, _state(before)))
            if not any(option[0] == source and option[3] == destination for option in options):
                raise ValueError("tier-10 ordered carrier did not open its green jump program")
            first = (names.ACTION_CLICK, *before.click(source))
            base_env.perform(*first); suffix.append(first)
            selected = extract(base_env)
            second = (names.ACTION_CLICK, *selected.click(destination))
            base_env.perform(*second); suffix.append(second)
        if base_env.levels_completed != 1:
            raise ValueError("tier-10 ordered carrier base program did not win")
        actions = tuple(prefix + suffix)
        return (
            ordinary, moving, rails, obstacles, pegs, actions, len(prefix),
            set(), boundary_layout, [(3, 3), (5, 3)],
        )
    result = search(base_env, node_limit=SEARCH_LIMITS[difficulty])
    if result.actions is None:
        raise ValueError("native mechanic core has no bounded positive witness")
    actions = tuple(result.actions)
    arrow_indices = [index for index, action in enumerate(actions) if action[0] in (1, 2, 3, 4)]
    boundary = max(arrow_indices) + 1 if arrow_indices else 0
    probe = Env(levels); probe.reset(); probe.set_level(difficulty - 1)
    touched = set()
    selected = None
    for action, x, y in actions[:boundary]:
        before = extract(probe)
        if action == names.ACTION_CLICK:
            cell = ((x - before.origin[0]) // names.TILE, (y - before.origin[1]) // names.TILE)
            if before.selected is None:
                selected = cell
            else:
                touched.update((selected, ((selected[0] + cell[0]) // 2, (selected[1] + cell[1]) // 2), cell))
                selected = None
        probe.perform(action, x, y)
    boundary_layout = extract(probe)
    suffix_destinations = []
    selected = None
    for action, x, y in actions[boundary:]:
        if action != names.ACTION_CLICK:
            continue
        cell = ((x - boundary_layout.origin[0]) // names.TILE, (y - boundary_layout.origin[1]) // names.TILE)
        if selected is None:
            selected = cell
        else:
            suffix_destinations.append(cell); selected = None
    return ordinary, moving, rails, obstacles, pegs, actions, boundary, touched, boundary_layout, suffix_destinations


def _materialize_program(spec, prefix, logical_groups, suffix, post_groups=()):
    env = _context_env(spec)
    start_score = env.levels_completed
    actions = []

    def take(action):
        action = tuple(action)
        observation = env.perform(*action)
        actions.append(action)
        if observation.state == GameState.GAME_OVER:
            raise ValueError("constructive program lost the native episode")
        if env.levels_completed > start_score and len(actions) != expected_total:
            raise ValueError("constructive program won before its final action")

    expected_total = len(prefix) + len(suffix) + 2 * sum(len(group) for group in logical_groups + post_groups)
    for action in prefix:
        take(action)
    for group in logical_groups:
        for source, destination in group:
            before = extract(env)
            if not before.visible(source) or not before.visible(destination):
                raise ValueError("constructive jump is clipped by the native camera")
            options = list(_jump_options(before, _state(before)))
            if not any(option[0] == source and option[3] == destination for option in options):
                raise ValueError(
                    f"constructive jump {source}->{destination} is not native-legal; "
                    f"pegs={before.peg_entities!r}, obstacles={sorted(before.obstacles)!r}, "
                    f"moving={sorted(before.moving_cells)!r}"
                )
            take((names.ACTION_CLICK, *before.click(source)))
            selected = extract(env)
            take((names.ACTION_CLICK, *selected.click(destination)))
    for action in suffix:
        take(action)
    for group in post_groups:
        for source, destination in group:
            before = extract(env)
            if not before.visible(source) or not before.visible(destination):
                raise ValueError("post-core constructive jump is clipped by the native camera")
            options = list(_jump_options(before, _state(before)))
            if not any(option[0] == source and option[3] == destination for option in options):
                raise ValueError(f"post-core constructive jump {source}->{destination} is not native-legal")
            take((names.ACTION_CLICK, *before.click(source)))
            take((names.ACTION_CLICK, *extract(env).click(destination)))
    if env.levels_completed <= start_score:
        raise ValueError("constructive program did not complete its native context")
    return [list(action) for action in actions]


def _materialize_steps(spec, steps):
    """Execute a mixed arrow/jump program and require its first win at EOF."""
    env = _context_env(spec)
    start_score = env.levels_completed
    actions = []
    expected_total = sum(1 if step[0] == "action" else 2 for step in steps)

    def take(action):
        observation = env.perform(*action)
        actions.append(tuple(action))
        if observation.state == GameState.GAME_OVER:
            raise ValueError("constructive mixed program lost the native episode")
        if env.levels_completed > start_score and len(actions) != expected_total:
            raise ValueError("constructive mixed program won before its final action")

    for kind, value in steps:
        if kind == "action":
            take(tuple(value))
            continue
        if kind != "jump":
            raise ValueError("unknown constructive mixed-program step")
        source, destination = value
        before = extract(env)
        if not before.visible(source) or not before.visible(destination):
            raise ValueError(f"constructive jump {source}->{destination} is clipped by the native camera")
        options = list(_jump_options(before, _state(before)))
        if not any(option[0] == source and option[3] == destination for option in options):
            raise ValueError(f"constructive jump {source}->{destination} is not native-legal")
        take((names.ACTION_CLICK, *before.click(source)))
        take((names.ACTION_CLICK, *extract(env).click(destination)))
    if env.levels_completed <= start_score:
        raise ValueError("constructive mixed program did not complete its native context")
    return [list(action) for action in actions]


def _reachable_alternative_blockers(spec, actions, count, rng):
    """Choose static blockers that are legal alternatives on a native route.

    Candidate middles must remain unoccupied for the complete positive trace,
    and their landing cells must already be native-open at the witness state.
    The stored route avoids these choices; final certification independently
    replays the augmented board and records each physical blocker entity that
    is actually offered.
    """
    if count <= 0:
        return [], []
    _ordinary, _moving, static_rails, _obstacles, _pegs = _board(spec)
    env = _context_env(spec)
    start_score = env.levels_completed
    snapshots = []
    dynamically_occupied = set()
    for action in actions:
        before = extract(env)
        dynamically_occupied.update(before.pegs | before.moving_cells | before.obstacles)
        if before.selected is None:
            snapshots.append(before)
        env.perform(*action)
    if env.levels_completed <= start_score:
        raise ValueError("preliminary tier-4 route did not reach its native goal")

    candidates = []
    for witness_index, layout in enumerate(snapshots):
        peg_cells = set(layout.pegs)
        occupied = peg_cells | set(layout.obstacles)
        for source, _kind in layout.peg_entities:
            for dx, dy in names.DIRECTIONS:
                middle = source[0] + dx, source[1] + dy
                destination = source[0] + 2 * dx, source[1] + 2 * dy
                stack_count = (
                    int(destination in layout.ordinary_cells)
                    + int(destination in layout.moving_cells)
                    + int(destination in layout.rails)
                    + int(destination in peg_cells)
                    + int(destination in layout.obstacles)
                )
                landing_open = (
                    destination in layout.ordinary_cells and stack_count == 1
                ) or (
                    destination in layout.moving_cells and stack_count == 2
                )
                destination_can_be_added = (
                    _within(destination, REFERENCE_PROFILES[4]["width"], REFERENCE_PROFILES[4]["height"])
                    and destination not in dynamically_occupied
                    # A rail under a moving hole is intentionally absent from
                    # the extracted visible rail set.  Admission nevertheless
                    # forbids adding an ordinary hole there, so use the native
                    # descriptor's complete static rail set.
                    and destination not in static_rails
                )
                if (
                    _within(middle, REFERENCE_PROFILES[4]["width"], REFERENCE_PROFILES[4]["height"])
                    and middle not in dynamically_occupied
                    and middle not in static_rails
                    and middle not in occupied
                    and (landing_open or destination_can_be_added)
                    and layout.visible(source)
                    and layout.visible(middle)
                    and layout.visible(destination)
                    and destination not in _CAMERA_DESTINATIONS.get(4, ())
                ):
                    candidates.append((middle, destination, witness_index, source, landing_open))
    pairs = list({(middle, destination) for middle, destination, _, _, _ in candidates})
    open_pairs = {
        (middle, destination)
        for middle, destination, _, _, landing_open in candidates
        if landing_open
    }
    rng.shuffle(pairs)
    visits = 0

    def choose(index, middles, destinations, selected_pairs):
        nonlocal visits
        visits += 1
        if visits > 50_000:
            return None
        if len(middles) == count:
            added_destinations = {
                destination
                for middle, destination in selected_pairs
                if (middle, destination) not in open_pairs
            }
            return list(middles), list(added_destinations)
        if len(pairs) - index < count - len(middles):
            return None
        for offset in range(index, len(pairs)):
            middle, destination = pairs[offset]
            if middle in middles or middle in destinations or destination in middles:
                continue
            found = choose(
                offset + 1,
                middles + (middle,),
                destinations | {destination},
                selected_pairs + ((middle, destination),),
            )
            if found is not None:
                return found
        return None

    chosen = choose(0, (), set(), ())
    if chosen is not None:
        return chosen
    raise ValueError("could not place every tier-4 blocker on a reachable legal branch")


def _draft(seed, effective, split, difficulty, attempt):
    rng = random.Random(f"lf52-v3:{effective}:{difficulty}:{attempt}")
    variant_count = 32 if difficulty == 2 else 12 if difficulty == 10 else 4
    variant = rng.randrange(variant_count)
    if difficulty in (3, 6, 7):
        if difficulty == 3:
            ordinary, moving, rails, obstacles, pegs, arrows, jumps, relations = _conditional_tier3_recipe(rng)
            steps = (
                [("action", action) for action in arrows]
                + [("jump", jump) for jump in jumps]
            )
            construction = {
                "grammar": "conditional-active-native-program-v2",
                "variant": variant,
                "green_removal_steps": len(jumps),
                "blocker_traversal_steps": 0,
                "auxiliary_moving_holes": 1,
                **relations,
            }
        elif difficulty == 6:
            (
                ordinary, moving, rails, obstacles, pegs, arrows,
                blocker_jumps, jumps, initial_jumps, relations,
            ) = _conditional_tier6_recipe(rng)
            steps = (
                [("jump", jump) for jump in initial_jumps]
                + [("action", action) for action in arrows]
                + [("jump", jump) for jump in blocker_jumps + jumps]
            )
            construction = {
                "grammar": "conditional-active-native-program-v2",
                "variant": variant,
                "green_removal_steps": len(jumps) + 1,
                "blocker_traversal_steps": len(blocker_jumps),
                "auxiliary_moving_holes": 2,
                **relations,
            }
        else:
            ordinary, moving, rails, obstacles, pegs, steps, construction = _active_tier7_recipe(rng)
        candidate = {
            "format": FORMAT,
            "generator_version": GENERATOR_VERSION,
            "mechanics_version": MECHANICS_VERSION,
            "quality_version": QUALITY_VERSION,
            "geometry_version": GEOMETRY_VERSION,
            "game": "lf52",
            "source": "generated_only",
            "seed": effective,
            "requested_seed": seed,
            "effective_seed": effective,
            "split": split,
            "effective_split": split,
            "attempt": attempt,
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "training_context_index": difficulty - 1,
            "verification_level_index": difficulty - 1,
            "board": {
                "ordinary": _cell_list(ordinary),
                "moving": _cell_list(moving),
                "rails": _cell_list(rails),
                "obstacles": _cell_list(obstacles),
                "pegs": _peg_list(pegs),
            },
            "required_mechanics": list(REQUIRED_MECHANICS[difficulty]),
            "generation_exclusions": ["official layouts", "official routes", "held-out m0r0"],
            "construction": construction,
        }
        candidate["_construction_actions"] = _materialize_steps(candidate, steps)
        return candidate

    (
        ordinary, moving, rails, obstacles, pegs, base_actions, boundary,
        touched, boundary_layout, suffix_destinations,
    ) = _base_recipe(difficulty, variant)
    ordinary, moving, rails, obstacles, pegs = set(ordinary), set(moving), set(rails), set(obstacles), dict(pegs)
    ordered_relation = {}
    if difficulty == 10:
        task_starts = [cell for cell in moving if pegs.get(cell) == names.PEG_BLUE]
        leader_starts = [cell for cell in moving if cell not in task_starts]
        if len(task_starts) != 1 or len(leader_starts) != 1:
            raise ValueError("tier-10 ordered blue carrier pair is malformed")
        ordered_relation = {
            "global_ordered_convoy": True,
            "order_leader_start": list(leader_starts[0]),
            "order_task_start": list(task_starts[0]),
            "order_task_kind": "blue_peg",
        }

    phase_origin = boundary_layout.origin
    # For colour tiers the reference count includes additional blues below;
    # grow only the missing non-blue population here.
    target_blue = {8: 4, 9: 6, 10: 10}.get(difficulty, 0)
    green_needed = REFERENCE_PROFILES[difficulty]["pegs"] - target_blue - sum(kind != names.PEG_BLUE for kind in pegs.values())
    post_core_jumps = []
    if difficulty == 2:
        # The native tutorial traps every initial core peg's reverse landing.
        # Two route-critical pegs therefore extend the core after its second
        # removal instead of padding a trapped prelude.
        ordinary.update({(6, 1), (7, 1), (8, 1)})
        pegs[(6, 1)] = names.PEG
        pegs[(8, 1)] = names.PEG
        post_core_jumps = [((5, 1), (7, 1)), ((8, 1), (6, 1))]
        green_jumps = []
    elif difficulty == 6:
        green_jumps = []
        post_core_jumps = _post_core_green_extension(
            difficulty, (7, 6), green_needed, ordinary, moving, rails,
            obstacles, pegs, rng, (-15, 5),
        )
    else:
        green_jumps = _reverse_removals(
            difficulty, green_needed, ordinary, moving, rails, obstacles, pegs,
            rng, phase_origin, touched,
        )

    blocker_needed = REFERENCE_PROFILES[difficulty]["obstacles"] - len(obstacles)
    active_caps = {4: 4, 6: 0, 7: 0}
    traversed_blockers = min(blocker_needed, active_caps.get(difficulty, blocker_needed))
    blocker_jumps, added_blockers = _relocate_over_blockers(
        difficulty, traversed_blockers, ordinary, moving, rails, obstacles, pegs,
        rng, phase_origin, touched,
        _jump_cells(green_jumps + post_core_jumps),
    )
    active_jump_cells = {
        cell
        for source, destination in blocker_jumps + green_jumps + post_core_jumps
        for cell in (source, ((source[0] + destination[0]) // 2, (source[1] + destination[1]) // 2), destination)
    }
    branch_needed = blocker_needed - traversed_blockers
    branch_blockers = [] if difficulty == 4 else _add_constraint_blockers(
        difficulty, branch_needed, ordinary, moving, rails, obstacles, pegs,
        rng, phase_origin, active_jump_cells | set(suffix_destinations),
    )

    blue_needed = target_blue - sum(kind == names.PEG_BLUE for kind in pegs.values())
    blue_jumps = []
    blue_cells = []
    if blue_needed:
        critical_candidates = (
            [destination for _, destination in blocker_jumps]
            + [destination for _, destination in green_jumps]
            + suffix_destinations
        )
        critical_candidates = list(dict.fromkeys(
            cell for cell in critical_candidates
            if cell in ordinary and cell not in pegs and cell not in obstacles and cell not in rails and cell not in moving
        ))
        rng.shuffle(critical_candidates)
        if not critical_candidates:
            raise ValueError("colour tier has no route-critical landing to guard")
        last_blue_error = None
        for critical in critical_candidates:
            try:
                blue_jumps, blue_cells = _blue_vacancy_chain(
                    difficulty, critical, blue_needed, ordinary, moving,
                    rails, obstacles, pegs, rng, phase_origin,
                    active_jump_cells | set(suffix_destinations),
                )
                break
            except ValueError as exc:
                last_blue_error = exc
        else:
            raise ValueError("could not place the active blue rearrangement") from last_blue_error

    arrows = [action[0] for action in base_actions[:boundary] if action[0] in (1, 2, 3, 4)]
    mover_candidates = (
        added_blockers + branch_blockers + blue_cells + suffix_destinations
        + [destination for _, destination in blocker_jumps + green_jumps + blue_jumps]
    )
    if difficulty in (4, 5, 10):
        # Tier 10's eight-hole convoy is the mechanic: keeping the six new
        # holes together preserves native leading-first collision semantics
        # and avoids turning certificate metadata into actor routing.
        # Tiers 4/5 already have a task-carrying primary mover; keeping the two
        # inventory companions on an isolated shared lane prevents them from
        # silently relocating blocker/removal-program actors before their
        # certified jumps.
        mover_candidates = []
    installed_movers = _install_auxiliary_movers(
        difficulty, REFERENCE_PROFILES[difficulty]["moving"], arrows[-1] if arrows else None,
        mover_candidates, ordinary, moving, rails, obstacles, pegs, rng,
    )
    _complete_active_envelope(difficulty, ordinary, moving, rails, obstacles, pegs, rng)

    candidate = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_version": QUALITY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "game": "lf52",
        "source": "generated_only",
        "seed": effective,
        "requested_seed": seed,
        "effective_seed": effective,
        "split": split,
        "effective_split": split,
        "attempt": attempt,
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "training_context_index": difficulty - 1,
        "verification_level_index": difficulty - 1,
        "board": {
            "ordinary": _cell_list(ordinary),
            "moving": _cell_list(moving),
            "rails": _cell_list(rails),
            "obstacles": _cell_list(obstacles),
            "pegs": _peg_list(pegs),
        },
        "required_mechanics": list(REQUIRED_MECHANICS[difficulty]),
        "generation_exclusions": ["official layouts", "official routes", "held-out m0r0"],
        "construction": {
            "grammar": "active-backward-native-program-v1",
            "variant": variant,
            "green_reverse_steps": len(green_jumps),
            "blocker_traversal_steps": len(blocker_jumps),
            "branch_constraint_blockers": len(branch_blockers),
            "blue_vacancy_steps": len(blue_jumps),
            "auxiliary_moving_holes": len(installed_movers),
            **ordered_relation,
        },
    }
    if difficulty == 4 and branch_needed:
        preliminary_actions = _materialize_program(
            candidate,
            base_actions[:boundary],
            (blue_jumps, blocker_jumps, green_jumps),
            base_actions[boundary:],
            (post_core_jumps,),
        )
        branch_blockers, branch_landings = _reachable_alternative_blockers(
            candidate, preliminary_actions, branch_needed, rng,
        )
        ordinary.update(branch_blockers + branch_landings)
        obstacles.update(branch_blockers)
        candidate["board"]["ordinary"] = _cell_list(ordinary)
        candidate["board"]["obstacles"] = _cell_list(obstacles)
        candidate["construction"]["branch_constraint_blockers"] = len(branch_blockers)
    candidate["_construction_actions"] = _materialize_program(
        candidate,
        base_actions[:boundary],
        (blue_jumps, blocker_jumps, green_jumps),
        base_actions[boundary:],
        (post_core_jumps,),
    )
    return candidate


def _canonical_board(spec, transform=None, normalize=False):
    ordinary, moving, rails, obstacles, pegs = _board(spec)
    typed = []
    for label, cells in (("ordinary", ordinary), ("moving", moving), ("rail", rails), ("obstacle", obstacles)):
        typed.extend((label, x, y) for x, y in cells)
    typed.extend((kind, x, y) for (x, y), kind in pegs.items())
    if transform is not None:
        converted = [(label, *transform(x, y)) for label, x, y in typed]
    else:
        converted = typed
    if normalize and converted:
        min_x = min(x for _, x, _ in converted); min_y = min(y for _, _, y in converted)
        converted = [(label, x - min_x, y - min_y) for label, x, y in converted]
    return sorted(converted)


def _sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def geometry_identities(spec):
    exact = _sha({"difficulty": spec["difficulty"], "typed": _canonical_board(spec)})
    transforms = (
        lambda x, y: (x, y), lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (-x, -y),
        lambda x, y: (y, x), lambda x, y: (-y, x), lambda x, y: (y, -x), lambda x, y: (-y, -x),
    )
    canonical = min(_canonical_board(spec, transform, True) for transform in transforms)
    d4 = _sha({"difficulty": spec["difficulty"], "typed": canonical})
    return exact, d4


def geometry_partition(d4):
    bucket = int(d4[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


@lru_cache(maxsize=1)
def _official_geometry_d4_identities():
    """Canonical typed identities for rejecting exact/D4 official copies."""
    env = Env()
    env.reset()
    identities = set()
    for index in range(len(DIFFICULTIES)):
        env.set_level(index)
        layout = extract(env)
        pseudo = {
            "difficulty": index + 1,
            "board": {
                "ordinary": _cell_list(layout.ordinary_cells),
                "moving": _cell_list(layout.moving_cells),
                "rails": _cell_list(layout.rails),
                "obstacles": _cell_list(layout.obstacles),
                "pegs": [
                    {"cell": list(cell), "kind": kind}
                    for cell, kind in layout.peg_entities
                ],
            },
        }
        identities.add(geometry_identities(pseudo)[1])
    return frozenset(identities)


def _validated_actions(solution):
    if not isinstance(solution, list) or not solution:
        raise ValueError("solution must be a nonempty list of action triples")
    normalized = []
    allowed = {
        names.ACTION_UP, names.ACTION_DOWN, names.ACTION_LEFT,
        names.ACTION_RIGHT, names.ACTION_CLICK,
    }
    for index, raw in enumerate(solution):
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError(f"solution[{index}] must be an action triple")
        action, x, y = raw
        if type(action) is not int or action not in allowed:
            raise ValueError(f"solution[{index}] has an unavailable action id")
        if action == names.ACTION_CLICK:
            if type(x) is not int or type(y) is not int:
                raise ValueError(f"solution[{index}] click coordinates must be integers")
            if not (0 <= x < names.DISPLAY and 0 <= y < names.DISPLAY):
                raise ValueError(f"solution[{index}] click is outside the native display")
        elif x is not None or y is not None:
            raise ValueError(f"solution[{index}] non-click coordinates must be null")
        normalized.append((action, x, y))
    return tuple(normalized)


def _normalized_action_identity(solution):
    solution = _validated_actions(solution)
    clicks = [(action[1], action[2]) for action in solution if action[0] == names.ACTION_CLICK]
    min_x = min((x for x, _ in clicks), default=0); min_y = min((y for _, y in clicks), default=0)
    normalized = [
        [action, None, None] if action != names.ACTION_CLICK else [action, x - min_x, y - min_y]
        for action, x, y in solution
    ]
    return _sha(normalized)


def _gameplay_identity(difficulty, geometry_d4_sha256):
    """Identify the puzzle/rules independently of any solver certificate."""
    return _sha(
        {
            "difficulty": difficulty,
            "geometry_d4_sha256": geometry_d4_sha256,
            "mechanics_version": MECHANICS_VERSION,
            "target_nonblue": 2 if difficulty in (6, 7) else 1,
            "native_action_budget": names.native_action_limit(difficulty),
        }
    )


def _actual_distribution(spec):
    ordinary, moving, rails, obstacles, pegs = _board(spec)
    occupied = ordinary | moving | rails | obstacles | set(pegs)
    kinds = Counter(pegs.values())
    difficulty = int(spec["difficulty"])
    origin = (10, 5) if difficulty == 1 else (6, 8) if difficulty == 2 else (5, 3) if difficulty == 10 else (5, 5)
    visible_cells = sum(
        1
        for x, y in ordinary | moving
        if 0 <= origin[0] + x * names.TILE
        and origin[0] + (x + 1) * names.TILE <= names.DISPLAY
        and 0 <= origin[1] + y * names.TILE
        and origin[1] + (y + 1) * names.TILE <= names.DISPLAY
    )
    return {
        "width": max(x for x, _ in occupied) - min(x for x, _ in occupied) + 1,
        "height": max(y for _, y in occupied) - min(y for _, y in occupied) + 1,
        "ordinary_cells": len(ordinary),
        "moving_cells": len(moving),
        "rail_cells": len(rails),
        "obstacles": len(obstacles),
        "pegs": len(pegs),
        "green_pegs": kinds[names.PEG],
        "red_pegs": kinds[names.PEG_RED],
        "blue_pegs": kinds[names.PEG_BLUE],
        "visible_cells": visible_cells,
    }


def profile_errors(spec):
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["difficulty must be an integer in 1..10"]
    try:
        actual = _actual_distribution(spec)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"board characterization failed: {exc}"]
    ref = REFERENCE_PROFILES[difficulty]
    if abs(actual["width"] - ref["width"]) > 1 or abs(actual["height"] - ref["height"]) > 1:
        errors.append("board envelope differs from the single official tier beyond ±1 cell")
    ordinary_low = max(5, ref["ordinary"] // 2)
    if not ordinary_low <= actual["ordinary_cells"] <= ref["ordinary"] + 20:
        errors.append("ordinary-cell count is outside the explicit single-reference tolerance")
    if actual["moving_cells"] != ref["moving"]:
        errors.append("moving-hole count differs from the official tier inventory")
    if ref["rails"] == 0 and actual["rail_cells"] != 0:
        errors.append("tier without official rails gained rails")
    if ref["rails"] and not 2 <= actual["rail_cells"] <= ref["rails"] + 30:
        errors.append("rail-cell count is outside the explicit single-reference tolerance")
    if actual["obstacles"] != ref["obstacles"]:
        errors.append("blocker count differs from the official tier inventory")
    if actual["pegs"] != ref["pegs"]:
        errors.append("peg count differs from the official tier inventory")
    if difficulty in (6, 7) and actual["red_pegs"] < 1:
        errors.append("red survivor role is absent")
    if difficulty >= 8 and actual["blue_pegs"] < 1:
        errors.append("blue non-goal peg role is absent")
    expected_blue = {8: 4, 9: 6, 10: 10}.get(difficulty, 0)
    if actual["blue_pegs"] != expected_blue:
        errors.append("blue-peg count differs from the official tier inventory")
    visible_low = max(5, ref["visible_cells"] // 2)
    if not visible_low <= actual["visible_cells"] <= ref["visible_cells"] + 20:
        errors.append("initial visible-cell density is outside the explicit single-reference tolerance")
    return errors


def _route_certificate(spec, actions):
    actions = _validated_actions(actions)
    env = _context_env(spec)
    start_score = env.levels_completed
    use = Counter()
    mover_positions = {
        id(entity): tuple(map(int, getattr(entity, names.PROP_GRID_POSITION)))
        for entity in getattr(env.grid, names.METHOD_ENTITIES_NAMED)(names.MOVING_HOLE)
    }
    moved_movers = set()
    traversed_blockers = set()
    offered_blockers = set()
    ordered_task_actors = set()
    used_ordered_task_actors = set()
    probed_traps = set()

    def entity_positions(entity_name):
        return {
            tuple(map(int, getattr(entity, names.PROP_GRID_POSITION))): id(entity)
            for entity in getattr(env.grid, names.METHOD_ENTITIES_NAMED)(entity_name)
        }

    def peg_positions():
        positions = {}
        for peg_kind in names.PEG_KINDS:
            positions.update(entity_positions(peg_kind))
        return positions

    def scan_visible_alternatives(before):
        if before.selected is not None:
            return
        state = _state(before)
        blocker_entities = entity_positions(names.OBSTACLE)
        for option in _jump_options(before, state):
            source, source_kind, middle, destination = option
            if middle in before.obstacles:
                use["blocker_jump_choices"] += 1
                blocker_id = blocker_entities.get(middle)
                if blocker_id is not None:
                    offered_blockers.add(blocker_id)
            nxt, _, _, event = _jump_transition(before, state, option)
            if event.get("camera_scrolls", 0):
                use["landing_camera_choices"] += 1
                probe_key = ("camera", source, destination, before.origin)
                if probe_key not in probed_traps:
                    probe = env.clone()
                    probe.perform(names.ACTION_CLICK, *before.click(source))
                    probe.perform(names.ACTION_CLICK, *before.click(destination))
                    if extract(probe).origin != before.origin:
                        use["native_landing_camera_probes"] += 1
                    probed_traps.add(probe_key)
            if _scripted_reset(before, source_kind, destination, nxt[0]):
                use["scripted_reset_choices"] += 1
                probe_key = ("reset", source, destination)
                if probe_key not in probed_traps:
                    probe = env.clone()
                    probe.perform(names.ACTION_CLICK, *before.click(source))
                    probe.perform(names.ACTION_CLICK, *before.click(destination))
                    probed = extract(probe)
                    if (
                        probed.reset_prompt
                        or probed.pending_auto_undo
                        or probed.action_count < before.action_count + 2
                    ):
                        use["native_scripted_reset_probes"] += 1
                        if (
                            (before.logical_level == 3 and destination in {(10, 2), (10, 4)})
                            or (before.logical_level == 6 and destination == (16, 2))
                        ):
                            use["native_conditional_reset_probes"] += 1
                    probed_traps.add(probe_key)

    for index, (action, x, y) in enumerate(actions):
        before = extract(env)
        if not before.exact:
            raise ValueError("route left the exactly modelled active level")
        scan_visible_alternatives(before)
        if action in (names.ACTION_UP, names.ACTION_DOWN, names.ACTION_LEFT, names.ACTION_RIGHT):
            before_moving = before.moving_cells
            before_origin = before.origin
            dx, dy = {
                names.ACTION_UP: (0, -1), names.ACTION_DOWN: (0, 1),
                names.ACTION_LEFT: (-1, 0), names.ACTION_RIGHT: (1, 0),
            }[action]
            mover_at = {position: key for key, position in mover_positions.items()}
            pegs_at = peg_positions()
            blockers_at = entity_positions(names.OBSTACLE)
            order_candidates = []
            for source, mover_id in mover_at.items():
                destination = source[0] + dx, source[1] + dy
                if destination not in mover_at:
                    continue
                payload = {
                    value for value in (pegs_at.get(source), blockers_at.get(source))
                    if value is not None
                }
                if payload:
                    order_candidates.append((mover_id, destination, payload))
            if any(
                (cell[0] + dx, cell[1] + dy) in before_moving
                and (cell[0] + 2 * dx, cell[1] + 2 * dy) in before.rails
                and (cell[0] + 2 * dx, cell[1] + 2 * dy) not in before_moving
                for cell in before_moving
            ):
                use["moving_hole_order_events"] += 1
            use["rail_actions"] += 1
            observation = env.perform(action, None, None)
            if env.levels_completed == start_score:
                after = extract(env)
                use["camera_scrolls"] += int(after.origin != before_origin)
                new_mover_positions = {}
                for entity in getattr(env.grid, names.METHOD_ENTITIES_NAMED)(names.MOVING_HOLE):
                    key = id(entity)
                    position = tuple(map(int, getattr(entity, names.PROP_GRID_POSITION)))
                    if mover_positions.get(key) != position:
                        moved_movers.add(key)
                        use["moving_hole_moves"] += 1
                    new_mover_positions[key] = position
                for mover_id, destination, payload in order_candidates:
                    if new_mover_positions.get(mover_id) == destination:
                        use["moving_hole_order_task_relations"] += 1
                        ordered_task_actors.update(payload)
                mover_positions = new_mover_positions
        elif action == names.ACTION_CLICK:
            if type(x) is not int or type(y) is not int:
                raise ValueError("click route entries require integer display coordinates")
            if before.selected is not None:
                destination = ((x - before.origin[0]) // names.TILE, (y - before.origin[1]) // names.TILE)
                option = next((value for value in _jump_options(before, _state(before)) if value[0] == before.selected and value[3] == destination), None)
                if option is not None:
                    source, source_kind, middle, destination = option
                    _, _, _, event = _jump_transition(before, _state(before), option)
                    use.update(event)
                    current_pegs = peg_positions()
                    current_blockers = entity_positions(names.OBSTACLE)
                    related = {
                        value for value in (
                            current_pegs.get(source), current_pegs.get(middle),
                            current_blockers.get(middle),
                        ) if value is not None
                    } & ordered_task_actors
                    if related:
                        use["ordered_task_actor_jump_relations"] += 1
                        used_ordered_task_actors.update(related)
                    if middle in before.obstacles:
                        traversed_blockers.add(middle)
                    if source_kind == names.PEG_BLUE:
                        use["blue_source_jumps"] += 1
                    if {source, middle, destination} & set(before.moving_cells):
                        use["moving_hole_jump_relations"] += 1
            observation = env.perform(action, x, y)
        if observation.state == GameState.GAME_OVER:
            raise ValueError("route loses the native episode")
        if env.levels_completed > start_score:
            if index != len(actions) - 1:
                raise ValueError("route wins before its declared final action")
            use["native_actions_consumed"] = len(actions)
    if env.levels_completed <= start_score:
        raise ValueError("route does not complete its intended native context")
    use["distinct_moving_holes_moved"] = len(moved_movers)
    use["distinct_blockers_traversed"] = len(traversed_blockers)
    use["distinct_blockers_offered"] = len(offered_blockers)
    use["distinct_ordered_task_actors_used"] = len(used_ordered_task_actors)
    return dict(sorted(use.items()))


def _presentation_certificate(spec, actions):
    """Check native frame palette, full-tile click extents, and rule cues."""
    env = _context_env(spec)
    start_score = env.levels_completed
    cues = set()
    palette_sizes = []
    click_tiles_checked = 0
    for action, x, y in _validated_actions(actions):
        layout = extract(env)
        frame = env.render()
        values = {int(value) for row in frame for value in row}
        if not values or min(values) < 0 or max(values) > 15:
            raise ValueError("native frame leaves the 16-colour ARC palette")
        palette_sizes.append(len(values))
        if any(layout.visible(cell) for cell in layout.moving_cells):
            cues.add("moving_hole")
        if any(layout.visible(cell) for cell in layout.obstacles):
            cues.add("blocker")
        for cell, kind in layout.peg_entities:
            if layout.visible(cell):
                if kind == names.PEG_RED:
                    cues.add("red_peg")
                elif kind == names.PEG_BLUE:
                    cues.add("blue_peg")
                elif kind == names.PEG:
                    cues.add("green_peg")
        if action == names.ACTION_CLICK:
            cell = ((x - layout.origin[0]) // names.TILE, (y - layout.origin[1]) // names.TILE)
            if not layout.visible(cell):
                raise ValueError(f"route clicks a clipped logical tile {cell!r}")
            left = layout.origin[0] + cell[0] * names.TILE
            top = layout.origin[1] + cell[1] * names.TILE
            tile_values = {
                int(frame[row][column])
                for row in range(top, top + names.TILE)
                for column in range(left, left + names.TILE)
            }
            if len(tile_values) < 2:
                raise ValueError(f"clickable tile {cell!r} has no readable sprite/background contrast")
            click_tiles_checked += 1
        env.perform(action, x, y)
    required = {"green_peg"}
    difficulty = int(spec["difficulty"])
    if difficulty >= 2:
        required.add("moving_hole")
    if 4 <= difficulty <= 9:
        required.add("blocker")
    if difficulty in (6, 7):
        required.add("red_peg")
    if difficulty >= 8:
        required.add("blue_peg")
    missing = sorted(required - cues)
    if missing:
        raise ValueError("native route never presents required visible cues: " + ", ".join(missing))
    return {
        "frames_checked": len(palette_sizes),
        "minimum_palette_colors": min(palette_sizes),
        "click_tiles_checked": click_tiles_checked,
        "visible_cues": sorted(cues),
        "full_tile_click_extents": True,
    }


def _mechanic_relation_errors(difficulty, mechanics):
    errors = []
    if mechanics.get("scripted_reset_landings", 0):
        errors.append("winning route itself lands on a scripted reset cell")
    moving_expected = REFERENCE_PROFILES[difficulty]["moving"]
    if mechanics.get("distinct_moving_holes_moved", 0) != moving_expected:
        errors.append("not every native moving-hole actor moved")
    if moving_expected and mechanics.get("moving_hole_jump_relations", 0) < 1:
        errors.append("winning route never used a moved hole in a jump relation")
    if difficulty in (1, 2, 3, 6) and mechanics.get("native_scripted_reset_probes", 0) < 1:
        errors.append("native reset alternative was not observed from a winning-route witness state")
    if difficulty in (3, 6) and mechanics.get("native_conditional_reset_probes", 0) < 1:
        errors.append("native conditional reset was not observed from a winning-route witness state")
    if difficulty == 6 and mechanics.get("native_landing_camera_probes", 0) < 1:
        errors.append("tier-6 landing camera branch was not observed in the native engine")
    if difficulty in (6, 7) and mechanics.get("distinct_blockers_traversed", 0) != REFERENCE_PROFILES[difficulty]["obstacles"]:
        errors.append("route did not traverse every dependency blocker in the active grammar")
    if difficulty == 4 and mechanics.get("distinct_blockers_offered", 0) != REFERENCE_PROFILES[difficulty]["obstacles"]:
        errors.append("tier-4 route states did not expose every installed blocker as a legal alternative")
    if difficulty >= 8 and mechanics.get("blue_source_jumps", 0) < 1:
        errors.append("blue role was never rearranged as a jump source")
    if difficulty in (7, 10) and mechanics.get("moving_hole_order_events", 0) < 1:
        errors.append("multi-hole tier has no leading-first collision-order event")
    if difficulty in (7, 10) and mechanics.get("moving_hole_order_task_relations", 0) < 1:
        errors.append("multi-hole tier has no task-carrying leading-order movement")
    if difficulty in (7, 10) and mechanics.get("ordered_task_actor_jump_relations", 0) < 1:
        errors.append("ordered task carrier never participates in a native jump relation")
    return errors


def certify(spec, limit=DEFAULT_LIMIT):
    candidate = json.loads(json.dumps(spec, separators=(",", ":")))
    construction_actions = candidate.pop("_construction_actions", None)
    errors = profile_errors(candidate)
    if errors:
        return None, "profile", errors
    try:
        solution = [list(action) for action in _validated_actions(construction_actions)]
    except ValueError as exc:
        return None, "construction", [str(exc)]
    if len(solution) > limit:
        return None, "search_cap", [f"constructive witness needs {len(solution)} actions beyond bound {limit}"]
    action_low, action_high = ACTION_TOLERANCES[candidate["difficulty"]]
    if not action_low <= len(solution) <= action_high:
        return None, "action_profile", [
            f"winning route length {len(solution)} is outside generated tier tolerance "
            f"{action_low}..{action_high}"
        ]
    try:
        mechanics = _route_certificate(candidate, solution)
        presentation = _presentation_certificate(candidate, solution)
    except ValueError as exc:
        return None, "native_replay", [str(exc)]
    missing = [name for name in REQUIRED_MECHANICS[candidate["difficulty"]] if mechanics.get(name, 0) < 1]
    if missing:
        return None, "mechanic_use", ["winning route did not exercise: " + ", ".join(missing)]
    relation_errors = _mechanic_relation_errors(candidate["difficulty"], mechanics)
    if relation_errors:
        return None, "mechanic_relations", relation_errors
    exact, d4 = geometry_identities(candidate)
    if d4 in _official_geometry_d4_identities():
        return None, "official_copy", ["typed D4 geometry matches a shipped level"]
    partition = geometry_partition(d4)
    if partition != candidate["split"]:
        return None, "split_partition", [f"canonical geometry belongs to {partition}"]
    action_identity = _normalized_action_identity(solution)
    gameplay = _gameplay_identity(candidate["difficulty"], d4)
    distribution = _actual_distribution(candidate)
    proof = {
        "method": "bounded-constructive-program-plus-context-native-replay",
        "search_limit": int(limit),
        "expanded": len(solution),
        "generated": len(solution) + 1,
        "truncated": False,
        "unsupported": False,
        "context_index": candidate["difficulty"] - 1,
        "route_sha256": _sha(solution),
        "geometry_sha256": exact,
        "gameplay_sha256": gameplay,
    }
    candidate.update(
        solution=solution,
        context_solution=solution,
        solution_length=len(solution),
        solution_mechanics=mechanics,
        action_sequence_sha256=action_identity,
        engine_verified=True,
        context_engine_verified=True,
        engine_win=True,
        planner_exact=True,
        search_truncated=False,
        search_limit=int(limit),
        native_action_budget=names.native_action_limit(candidate["difficulty"]),
        budget_slack=names.native_action_limit(candidate["difficulty"]) - len(solution),
        geometry_sha256=exact,
        geometry_d4_sha256=d4,
        geometry_split=partition,
        gameplay_sha256=gameplay,
        actual_distribution=distribution,
        presentation=presentation,
        proof=proof,
        proof_sha256=_sha(proof),
        coverage={
            "all_official_tier_mechanics": True,
            "required": list(REQUIRED_MECHANICS[candidate["difficulty"]]),
            "exercised": sorted(name for name, count in mechanics.items() if count),
        },
        quality_profile={
            "single_official_reference_actions": REFERENCE_PROFILES[candidate["difficulty"]]["reference_solution_actions"],
            "generated_action_tolerance": [action_low, action_high],
            "optimality_claimed": False,
        },
    )
    return json.loads(json.dumps(candidate, separators=(",", ":"))), "accepted", []


def generate(seed, difficulty=1, attempts=MAX_ATTEMPTS, limit=None, *, max_attempts=None, search_limit=None, node_limit=None, split="train"):
    """Return one deterministic full-mechanics tier, or ``None`` within bounds."""
    seed = _integer(seed, "seed", minimum=0)
    difficulty = _integer(difficulty, "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in names.SPLITS:
        raise ValueError("split must be train, validation, or test")
    if max_attempts is not None:
        attempts = max_attempts
    attempts = _integer(attempts, "attempts", minimum=1)
    if attempts > MAX_ATTEMPTS:
        raise ValueError(f"attempts must be at most {MAX_ATTEMPTS}")
    if search_limit is not None:
        limit = search_limit
    if node_limit is not None:
        limit = node_limit
    if limit is None:
        limit = SEARCH_LIMITS[difficulty]
    limit = _integer(limit, "limit", minimum=1)
    effective = effective_seed(seed, split)
    rejected = Counter()
    details = []
    seen_d4 = set()
    for attempt in range(attempts):
        try:
            candidate = _draft(seed, effective, split, difficulty, attempt)
            draft_d4 = geometry_identities(candidate)[1]
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            reason = "construction"
            messages = [str(exc)]
            rejected[reason] += 1
            if len(details) < 8:
                details.append({"attempt": attempt, "reason": reason, "details": messages})
            continue
        if draft_d4 in seen_d4:
            reason = "duplicate_geometry"
            messages = ["candidate repeats an earlier canonical D4 geometry in this bounded call"]
            rejected[reason] += 1
            if len(details) < 8:
                details.append({"attempt": attempt, "reason": reason, "details": messages})
            continue
        seen_d4.add(draft_d4)
        # The split is derived entirely from public typed geometry.  Reject it
        # before the repeated presentation/mechanic replay; ``certify`` still
        # recomputes and enforces the same partition for accepted candidates.
        draft_partition = geometry_partition(draft_d4)
        if draft_partition != split:
            reason = "split_partition"
            messages = [f"canonical geometry belongs to {draft_partition}"]
            rejected[reason] += 1
            if len(details) < 8:
                details.append({"attempt": attempt, "reason": reason, "details": messages})
            continue
        verified, reason, messages = certify(candidate, limit=limit)
        if verified is not None:
            verified["generation_diagnostics"] = {
                "attempted": attempt + 1,
                "accepted": 1,
                "rejected": dict(sorted(rejected.items())),
                "bounded_attempts": attempts,
            }
            generate.last_diagnostics = verified["generation_diagnostics"]
            return verified
        rejected[reason] += 1
        if len(details) < 8:
            details.append({"attempt": attempt, "reason": reason, "details": messages})
    generate.last_diagnostics = {
        "attempted": attempts,
        "accepted": 0,
        "rejected": dict(sorted(rejected.items())),
        "examples": details,
        "bounded_attempts": attempts,
    }
    return None


generate.last_diagnostics = None


def _game_level_seed(game_seed, level_index, difficulty):
    game_seed = _integer(game_seed, "game seed", minimum=0)
    material = f"{SOURCE_ID}:{game_seed}:{level_index}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split="train", difficulties=None, attempts=MAX_ATTEMPTS, node_limit=max(SEARCH_LIMITS.values())):
    seed = _integer(seed, "game seed", minimum=0)
    if difficulties is None:
        selected = DIFFICULTIES
    else:
        if isinstance(difficulties, (str, bytes)):
            raise ValueError("game difficulties must be an integer sequence")
        try:
            selected = tuple(difficulties)
        except TypeError as exc:
            raise ValueError("game difficulties must be an integer sequence") from exc
    if not selected:
        raise ValueError("a generated game needs at least one difficulty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in selected):
        raise ValueError("game difficulties must be integers")
    selected = tuple(int(value) for value in selected)
    if any(value not in DIFFICULTIES for value in selected):
        raise ValueError(f"game difficulties must be drawn from {DIFFICULTIES}")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("game difficulties must be strictly increasing and distinct")
    mode = "full_game" if selected == DIFFICULTIES else "reduced_smoke"
    specs = []
    rows = []
    for level_index, difficulty in enumerate(selected):
        child_seed = _game_level_seed(seed, level_index, difficulty)
        spec = generate(child_seed, difficulty, attempts=attempts, node_limit=node_limit, split=split)
        if spec is None:
            generate_game.last_diagnostics = {
                "game_seed": seed,
                "split": split,
                "mode": mode,
                "requested_difficulties": list(selected),
                "completed": len(specs),
                "failed_level_index": level_index,
                "failed_difficulty": difficulty,
                "failed_child_seed": child_seed,
                "failure": json.loads(json.dumps(generate.last_diagnostics)),
                "levels": rows,
            }
            return None
        spec.update(
            game_seed=seed,
            game_level_index=level_index,
            game_requested_difficulty=difficulty,
            game_difficulties=list(selected),
            game_mode=mode,
        )
        specs.append(spec)
        rows.append({
            "level_index": level_index,
            "difficulty": difficulty,
            "child_seed": child_seed,
            "generation": json.loads(json.dumps(spec["generation_diagnostics"])),
        })
    generate_game.last_diagnostics = {
        "game_seed": seed,
        "split": split,
        "mode": mode,
        "requested_difficulties": list(selected),
        "completed": len(specs),
        "accepted": True,
        "levels": rows,
    }
    return specs


generate_game.last_diagnostics = None


def build_game(specs):
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or len(specs) != len(DIFFICULTIES):
        raise ValueError(f"LF52 build_game requires exactly {len(DIFFICULTIES)} ordered specs")
    game_fields = {
        "game_mode", "game_difficulties", "game_seed", "game_level_index",
        "game_requested_difficulty",
    }
    field_sets = [
        {key for key in game_fields if isinstance(spec, Mapping) and key in spec}
        for spec in specs
    ]
    if any(field_sets) and any(fields != game_fields for fields in field_sets):
        raise ValueError("game provenance must be absent from every row or complete on every row")
    enriched = bool(field_sets and field_sets[0] == game_fields)
    split = None
    levels = []
    geometry_seen = set()
    gameplay_seen = set()
    game_seed = None
    for index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if not isinstance(spec, Mapping) or type(spec.get("difficulty")) is not int or spec.get("difficulty") != difficulty:
            raise ValueError("game specs must use difficulties 1..10 in order")
        if (
            type(spec.get("context_index")) is not int
            or type(spec.get("verification_level_index")) is not int
            or spec.get("context_index") != index
            or spec.get("verification_level_index") != index
        ):
            raise ValueError(f"spec {index} has a shifted native context")
        if split is None:
            split = spec.get("split")
        elif spec.get("split") != split:
            raise ValueError("game specs must all use the same split")
        if enriched:
            if spec.get("game_mode") != "full_game" or spec.get("game_difficulties") != list(DIFFICULTIES):
                raise ValueError("enriched build_game provenance must describe the full ten-tier mode")
            if type(spec.get("game_seed")) is not int or spec.get("game_seed") < 0:
                raise ValueError("game seed metadata is malformed")
            if game_seed is None:
                game_seed = spec["game_seed"]
            elif spec["game_seed"] != game_seed:
                raise ValueError("game specs have inconsistent parent seeds")
            if (
                type(spec.get("game_level_index")) is not int
                or spec.get("game_level_index") != index
                or type(spec.get("game_requested_difficulty")) is not int
                or spec.get("game_requested_difficulty") != difficulty
                or spec.get("requested_seed") != _game_level_seed(game_seed, index, difficulty)
            ):
                raise ValueError("game child seed/position/tier metadata is malformed")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"spec {index} does not satisfy the full contract: " + "; ".join(errors))
        geometry = spec.get("geometry_d4_sha256")
        gameplay = spec.get("gameplay_sha256")
        if geometry in geometry_seen or gameplay in gameplay_seen:
            raise ValueError("game specs contain duplicate semantic geometry/gameplay identities")
        geometry_seen.add(geometry); gameplay_seen.add(gameplay)
        levels.append(build_level(spec))
    # A complete builder also proves that the stored per-context routes remain
    # valid when levels advance naturally without forced transitions.
    proof_env = Env(levels)
    proof_env.reset()
    for index, spec in enumerate(specs):
        if proof_env.level_index != index or not replay(proof_env, spec["solution"])[0]:
            raise ValueError(f"spec {index} failed sequential native replay")
        if proof_env.levels_completed != index + 1:
            raise ValueError(f"spec {index} did not advance exactly one native level")
    if proof_env.state != GameState.WIN:
        raise ValueError("complete generated game did not reach native WIN")
    return levels


def _typed_tree_equal(stored, expected):
    """Compare generated JSON evidence without bool/int/float equivalence."""
    if isinstance(expected, Mapping):
        return (
            isinstance(stored, Mapping)
            and stored.keys() == expected.keys()
            and all(_typed_tree_equal(stored[key], value) for key, value in expected.items())
        )
    if type(expected) is list:
        return (
            type(stored) is list
            and len(stored) == len(expected)
            and all(_typed_tree_equal(left, right) for left, right in zip(stored, expected))
        )
    return type(stored) is type(expected) and stored == expected


def validate_full_standard(spec, curriculum_entry):
    """Recompute structure, partition, identities, and native route evidence."""
    errors = []
    if not isinstance(spec, Mapping):
        return ["generated spec must be an object"]
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be an object"]
    difficulty = curriculum_entry.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty must be in 1..10"]
    expected_context = difficulty - 1
    if type(curriculum_entry.get("context_index")) is not int or curriculum_entry.get("context_index") != expected_context:
        errors.append("curriculum context differs from tier-1")
    if type(curriculum_entry.get("search_work")) is not int or curriculum_entry.get("search_work") != SEARCH_LIMITS[difficulty]:
        errors.append("curriculum search work differs from the declared tier cap")
    if spec.get("format") != FORMAT or type(spec.get("generator_version")) is not int or spec.get("generator_version") != GENERATOR_VERSION:
        errors.append("format/generator version differs from the full generator")
    for key, expected in (("mechanics_version", MECHANICS_VERSION), ("quality_version", QUALITY_VERSION), ("geometry_version", GEOMETRY_VERSION), ("source", "generated_only")):
        if spec.get(key) != expected:
            errors.append(f"{key} is missing or inconsistent")
    if type(spec.get("difficulty")) is not int or spec.get("difficulty") != difficulty:
        errors.append("spec difficulty differs from curriculum")
    if spec.get("game") != "lf52":
        errors.append("game identity is missing or inconsistent")
    for key in ("context_index", "training_context_index", "verification_level_index"):
        if type(spec.get(key)) is not int or spec.get(key) != expected_context:
            errors.append(f"{key} differs from the native context")
    if spec.get("required_mechanics") != list(REQUIRED_MECHANICS[difficulty]):
        errors.append("required_mechanics differs from the audited tier inventory")
    if spec.get("generation_exclusions") != ["official layouts", "official routes", "held-out m0r0"]:
        errors.append("generation exclusions are malformed")
    try:
        errors.extend(profile_errors(spec))
        actual_distribution = _actual_distribution(spec)
        exact, d4 = geometry_identities(spec)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"geometry/profile recomputation failed: {exc}")
        exact = d4 = None
        actual_distribution = None
    if actual_distribution is not None and not _typed_tree_equal(
        spec.get("actual_distribution"), actual_distribution
    ):
        errors.append("stored structural distribution differs from recomputation")
    if exact is not None:
        partition = geometry_partition(d4)
        if spec.get("geometry_sha256") != exact or spec.get("geometry_d4_sha256") != d4:
            errors.append("stored geometry identity differs from recomputation")
        if spec.get("geometry_split") != partition or spec.get("split") != partition:
            errors.append("stored geometry partition differs from recomputation")
        if d4 in _official_geometry_d4_identities():
            errors.append("generated typed geometry is an official/D4 copy")
    split = spec.get("split")
    requested = spec.get("requested_seed")
    if split not in names.SPLITS or type(requested) is not int or requested < 0:
        errors.append("requested seed and split are malformed")
    else:
        expected_effective = effective_seed(requested, split)
        if (
            type(spec.get("effective_seed")) is not int
            or type(spec.get("seed")) is not int
            or spec.get("effective_seed") != expected_effective
            or spec.get("seed") != expected_effective
            or spec.get("effective_split") != split
        ):
            errors.append("effective seed/split differs from canonical mapping")
    solution = spec.get("solution")
    try:
        checked_actions = _validated_actions(solution)
    except (TypeError, ValueError) as exc:
        checked_actions = None
        errors.append(f"solution schema is malformed: {exc}")
    if (
        checked_actions is None
        or not _typed_tree_equal(spec.get("context_solution"), solution)
        or type(spec.get("solution_length")) is not int
        or spec.get("solution_length") != len(solution)
    ):
        errors.append("solution/context_solution/length are inconsistent")
    else:
        route_sha = _sha(solution)
        if not isinstance(spec.get("proof"), Mapping) or spec["proof"].get("route_sha256") != route_sha:
            errors.append("proof route identity differs from the stored solution")
        if spec.get("action_sequence_sha256") != _normalized_action_identity(solution):
            errors.append("canonical action-sequence identity differs from recomputation")
        action_low, action_high = ACTION_TOLERANCES[difficulty]
        if not action_low <= len(solution) <= action_high:
            errors.append("winning route length is outside the explicit generated tier tolerance")
        expected_quality = {
            "single_official_reference_actions": REFERENCE_PROFILES[difficulty]["reference_solution_actions"],
            "generated_action_tolerance": [action_low, action_high],
            "optimality_claimed": False,
        }
        if not _typed_tree_equal(spec.get("quality_profile"), expected_quality):
            errors.append("stored action-quality profile differs from calibrated tier metadata")
        try:
            mechanics = _route_certificate(spec, solution)
            presentation = _presentation_certificate(spec, solution)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f"native route replay failed: {exc}")
        else:
            if not _typed_tree_equal(spec.get("solution_mechanics"), mechanics):
                errors.append("stored solution mechanics differ from native recomputation")
            missing = [name for name in REQUIRED_MECHANICS[difficulty] if mechanics.get(name, 0) < 1]
            if missing:
                errors.append("native winning route does not exercise: " + ", ".join(missing))
            errors.extend(_mechanic_relation_errors(difficulty, mechanics))
            if not _typed_tree_equal(spec.get("presentation"), presentation):
                errors.append("stored presentation certificate differs from native recomputation")
            expected_coverage = {
                "all_official_tier_mechanics": True,
                "required": list(REQUIRED_MECHANICS[difficulty]),
                "exercised": sorted(name for name, count in mechanics.items() if count),
            }
            if not _typed_tree_equal(spec.get("coverage"), expected_coverage):
                errors.append("stored mechanic coverage differs from native recomputation")
            native_consumed = mechanics.get("native_actions_consumed")
            budget = names.native_action_limit(difficulty)
            if native_consumed != len(solution):
                errors.append("native route accounting differs from the complete action list")
            if type(spec.get("native_action_budget")) is not int or spec.get("native_action_budget") != budget:
                errors.append("native_action_budget differs from the source threshold")
            if type(spec.get("budget_slack")) is not int or spec.get("budget_slack") != budget - native_consumed:
                errors.append("budget_slack differs from native execution")
            if native_consumed >= budget:
                errors.append("winning route reaches or exceeds the native loss threshold")
            if d4 is not None:
                gameplay = _gameplay_identity(difficulty, d4)
                if spec.get("gameplay_sha256") != gameplay:
                    errors.append("stored gameplay identity differs from recomputation")
    proof = spec.get("proof")
    if not isinstance(proof, Mapping):
        errors.append("proof must be an object")
    else:
        if spec.get("proof_sha256") != _sha(proof):
            errors.append("stored proof identity differs from proof recomputation")
        expanded = proof.get("expanded")
        generated = proof.get("generated")
        counters_are_integers = type(expanded) is int and type(generated) is int
        if not counters_are_integers or expanded < 0 or generated < expanded:
            errors.append("proof search counters are malformed")
        elif checked_actions is not None and (
            expanded != len(checked_actions)
            or generated != len(checked_actions) + 1
        ):
            errors.append("proof search counters differ from the audited constructive work")
        if proof.get("method") != "bounded-constructive-program-plus-context-native-replay":
            errors.append("proof.method is not the audited bounded method")
        if proof.get("truncated") is not False:
            errors.append("proof.truncated must be false")
        if proof.get("unsupported") is not False:
            errors.append("proof.unsupported must be false")
        proof_search_limit = proof.get("search_limit")
        if type(proof_search_limit) is not int or proof_search_limit != SEARCH_LIMITS[difficulty]:
            errors.append("proof.search_limit differs from the curriculum bound")
        elif counters_are_integers and expanded > proof_search_limit:
            errors.append("proof.expanded exceeds its declared work bound")
        mirrors = {
            "search_limit": spec.get("search_limit"),
            "context_index": expected_context,
            "geometry_sha256": spec.get("geometry_sha256"),
            "gameplay_sha256": spec.get("gameplay_sha256"),
        }
        for key, expected in mirrors.items():
            if not _typed_tree_equal(proof.get(key), expected):
                errors.append(f"proof.{key} does not mirror recomputed/top-level evidence")
    for key in ("engine_verified", "context_engine_verified", "engine_win", "planner_exact"):
        if spec.get(key) is not True:
            errors.append(f"{key} is missing")
    if spec.get("search_truncated") is not False:
        errors.append("accepted spec claims a truncated search")
    if type(spec.get("search_limit")) is not int or spec.get("search_limit") != SEARCH_LIMITS[difficulty]:
        errors.append("search_limit differs from the full curriculum bound")
    diagnostics = spec.get("generation_diagnostics")
    if not isinstance(diagnostics, Mapping):
        errors.append("generation_diagnostics must be an object")
    else:
        attempted = diagnostics.get("attempted")
        accepted = diagnostics.get("accepted")
        bounded = diagnostics.get("bounded_attempts")
        rejected = diagnostics.get("rejected")
        if type(attempted) is not int or type(accepted) is not int or type(bounded) is not int:
            errors.append("generation diagnostic counters must be integers")
        elif not (1 <= attempted <= bounded <= MAX_ATTEMPTS) or accepted != 1:
            errors.append("generation diagnostic counters violate their hard bounds")
        if type(spec.get("attempt")) is not int or spec.get("attempt") < 0:
            errors.append("accepted attempt index is malformed")
        elif type(attempted) is int and attempted != spec.get("attempt") + 1:
            errors.append("accepted attempt index differs from generation diagnostics")
        if not isinstance(rejected, Mapping) or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in (rejected.items() if isinstance(rejected, Mapping) else ())
        ):
            errors.append("generation rejection counters are malformed")
        elif type(attempted) is int and sum(rejected.values()) != attempted - 1:
            errors.append("generation rejection counters do not account for prior attempts")
    return errors
