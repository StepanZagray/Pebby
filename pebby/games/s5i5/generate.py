"""Full, reference-calibrated procedural generation for all eight S5I5 tiers.

The generator samples new rod geometry, attachment graphs, colour assignments,
obstacles and targets. Accepted rows carry a loop-free constructive witness,
replayed in the real vendored engine at the intended native context. The
witness is a positive certificate only: no shortest-route claim is made.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from numbers import Integral
import random

from arcengine import GameState, Level, Sprite

from . import names
from .env import Env, replay
from .generation_quality import (
    GEOMETRY_VERSION,
    SPLITS,
    gameplay_hash,
    geometry_hash,
    identity_partition,
    solution_semantic_hash,
)
from .layout import extract
from .official_identity import official_copy_match
from .plan import _Engine, _FastEngine, search
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    MECHANICS_INVENTORY_VERSION,
    PROFILES,
    REFERENCE_CHARACTERIZATION,
    SUPPORTED_MECHANICS,
)


FORMAT = "pebby.s5i5.level.v2"
GENERATOR_VERSION = 2
QUALITY_PROFILE_VERSION = DIFFICULTY_VERSION
SOURCE_ID = "s5i5-18d95033"
DEFAULT_ATTEMPTS = 512
MAX_ATTEMPTS = 512
UNIT = names.ROD_THICKNESS
ROD_COLORS = (7, 8, 9, 10, 11, 12, 14)
OBSTACLE_COLOR = 15
OBSTACLE_PALETTES = {
    1: (15, 6), 2: (15, 15), 3: (15, 15), 4: (15,),
    5: (15, 15, 15), 6: (15,), 7: (15,), 8: (0, 15),
}


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": [
        {"difficulty": difficulty, "context_index": difficulty - 1,
         "search_work": PROFILES[difficulty]["search_work"]}
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "docs/generator-evidence/s5i5.md#official-reference-characterization",
        "solution_mechanics": "pebby/games/s5i5/generate.py:_solution_mechanics",
        "native_budget": "third_party/arc3_games/s5i5.py:1952-1987,2181-2245",
        "context_engine_replay": "tests/games/test_s5i5.py:test_complete_generated_game_replays_all_native_contexts",
        "novelty_split": GEOMETRY_VERSION,
        "bounded_rejections": "tests/games/test_s5i5_quality.py:test_bounded_all_tier_quality_audit",
        "official_copy_exclusion": "tests/games/test_s5i5_official_copy.py",
        "tier8_recipe_variation": "tests/games/test_s5i5_tier8_variation.py",
        "three_split_full_games": "docs/generator-evidence/s5i5-final-root-{train,validation,test}.json",
        "native_frame_comparison": "docs/generator-evidence/s5i5-render-comparison.json",
    },
    "caveats": [
        "One shipped level exists per tier; tolerances are engineering bands, not population intervals.",
        "Generated rows use replayed constructive witnesses and do not claim shortest routes.",
        "Official tiers 7 and 8 use guided native-positive witnesses; no exhaustive optimum is claimed.",
        "Tier 8 has four finite causal recipes with one attachment topology; they use 35-36 rod units and maximum rod lengths 7-9 versus the official tier's 31 units and maximum 7.",
        "Tier 5 preserves the official 54 visible obstacle pixels, but generation uses one connected component rather than the official 27/18/9-pixel components.",
        "Official board-geometry denial is intentionally conservative and is not a transition-system isomorphism claim.",
    ],
}


_COMPONENT_LENGTHS = {
    1: ((2,), (3,)),
    2: ((1, 1, 1, 1),),
    3: ((6,), (5,), (1, 7, 1), (1, 7, 1)),
    # The native tier-four pin carrier is the standalone one-unit component.
    # Its four same-colour companions are the children of four independent
    # parents, including the long nine-unit parent. Keeping that topology is
    # what permits genuine shared-child collision gates instead of putting an
    # unsupported recursive edge on the pin carrier itself.
    4: ((1,), (1, 1), (1, 1), (1, 1), (9, 1)),
    5: ((6,), (5,), (1, 5), (1, 5, 1)),
    6: ((2, 3, 2),),
    7: ((5,), (4,), (1, 2, 2, 1)),
    8: ((1,), (1, 6, 6), (1, 6, 1), (1, 1, 1, 9, 1)),
}
_UNCONTROLLED_LOCAL = {
    1: (), 2: (), 3: ((0, 0), (1, 0)), 4: (), 5: ((1, 0),),
    6: (), 7: (), 8: ((2, 2), (3, 2)),
}
_CONTROL_GROUP_SIZES = {
    1: (1, 1), 2: (1, 1, 1, 1), 3: (1, 1, 1, 1, 1, 1),
    4: (5, 1, 1, 1, 1), 5: (2, 2, 1, 1), 6: (1, 1, 1),
    7: (1, 1, 1, 1, 1, 1), 8: (2, 2, 2, 1, 1, 1, 1),
}
_PIN_LOCAL = {
    1: ((0, 0), (1, 0)), 2: ((0, 3),), 3: ((2, 2), (3, 2)),
    4: ((0, 0),), 5: ((0, 0), (3, 2)), 6: ((0, 2),),
    7: ((0, 0), (2, 3)), 8: ((0, 0),),
}

# Each tier-8 recipe changes collision clearances and the number of physical
# retracts needed at several causal gates.  These are deliberately finite: the
# native interaction is brittle enough that unconstrained coordinate sampling
# mostly produces inert or unsolvable layouts.  Entries are
# (first/second/third gate x, first/second/third/branch-gate rod length).
_TIER8_CAUSAL_RECIPES = (
    ((24, 27, 21), (7, 7, 6, 7)),
    ((21, 27, 24), (7, 6, 7, 8)),
    ((27, 21, 27), (6, 7, 6, 8)),
    ((24, 24, 24), (6, 6, 6, 9)),
)


def _select_tier8_recipe(rng):
    """Select a recipe without perturbing downstream colour/obstacle draws."""
    state = repr(rng.getstate()).encode("utf-8")
    index = int.from_bytes(hashlib.sha256(state).digest()[:8], "big")
    return _TIER8_CAUSAL_RECIPES[index % len(_TIER8_CAUSAL_RECIPES)]


def _apply_tier8_recipe(rows, recipe):
    """Apply one reviewed causal geometry to an existing tier-8 rod forest."""
    gate_xs, gate_lengths = recipe
    first_x, second_x, third_final_x = gate_xs
    first_length = gate_lengths[0]
    first_clearance = (30 - first_x) // UNIT
    first_retracts = first_length - first_clearance
    causal_geometry = {
        0: (30, 12, 0),
        1: (first_x - UNIT, 9, 90),
        2: (first_x, 9, 90),
        # rod3 is translated by every rod2 retract.  Its initial position
        # therefore encodes the desired third-gate clearance after stage1.
        3: (third_final_x + UNIT * first_retracts, 3, 90),
        4: (second_x - UNIT, 6, 90), 5: (second_x, 6, 90),
        6: (45, 33, 90),
        7: (30, 24, 90), 8: (21, 12, 90), 9: (30, 21, 0),
        10: (15, 15, 90), 11: (3, 3, 180),
    }
    for index, (x, y, rotation) in causal_geometry.items():
        rows[index].update(x=x, y=y, rotation=rotation)
    for index, length in zip((2, 5, 3, 10), gate_lengths):
        rows[index]["length"] = length


# -- sprite construction -------------------------------------------------------

def rod_pixels(color, rotation, length):
    n = int(length) * UNIT
    if rotation in (0, 180):
        pixels = [[int(color)] * UNIT for _ in range(n)]
        pixels[-1 if rotation == 0 else 0] = [names.CAP_COLOR] * UNIT
    elif rotation in (90, 270):
        pixels = [[int(color)] * n for _ in range(UNIT)]
        column = 0 if rotation == 90 else n - 1
        for row in pixels:
            row[column] = names.CAP_COLOR
    else:
        raise ValueError("rotation must be 0, 90, 180 or 270")
    return pixels


def rod_sprite(name, color, x, y, rotation, length):
    return Sprite(rod_pixels(color, rotation, length), name=name, x=int(x), y=int(y),
                  layer=0, tags=[names.TAG_ROD])


def obstacle_sprite(name, cells, color=OBSTACLE_COLOR, ink_stride=1):
    points = [(int(x), int(y)) for x, y in cells]
    if not points:
        raise ValueError("obstacle must contain at least one cell")
    left, top = min(x for x, _ in points), min(y for _, y in points)
    right, bottom = max(x for x, _ in points) + UNIT, max(y for _, y in points) + UNIT
    pixels = [[-1] * (right - left) for _ in range(bottom - top)]
    for x, y in points:
        for yy in range(y - top, y - top + UNIT):
            for xx in range(x - left, x - left + UNIT):
                pixels[yy][xx] = int(color)
    return Sprite(pixels, name=name, x=left, y=top, layer=0, tags=[names.TAG_ROD])


def _canonical_boundary_cells(difficulty):
    """Native generated boundary cells, including clipped endpoint anchors."""
    if difficulty not in (3, 4, 5):
        return None

    def covering_positions(start, end):
        positions = list(range(start, end - 1, UNIT))
        final = end - UNIT + 1
        if positions[-1] != final:
            positions.append(final)
        return positions

    if difficulty == 4:
        horizontal = covering_positions(-UNIT, names.FRAME - 1)
        vertical = covering_positions(0, names.FRAME - 1)
        return ([[x, -UNIT] for x in horizontal]
                + [[-UNIT, y] for y in vertical])
    horizontal = covering_positions(-UNIT, names.FRAME + UNIT - 1)
    vertical = covering_positions(0, names.FRAME - 1)
    return ([[x, -UNIT] for x in horizontal]
            + [[x, names.FRAME] for x in horizontal]
            + [[-UNIT, y] for y in vertical]
            + [[names.FRAME, y] for y in vertical])


def pin_sprite(name, x, y):
    c = names.PIN_COLOR
    return Sprite([[-2, -2, -2], [-2, c, -2], [-2, -2, -2]], name=name,
                  x=int(x), y=int(y), layer=1, tags=[names.TAG_PIN])


def target_sprite(name, x, y):
    c = names.PIN_COLOR
    return Sprite([[-2, c, -2], [c, -2, c], [-2, c, -2]], name=name,
                  x=int(x), y=int(y), layer=1, tags=[names.TAG_TARGET])


def rail_sprite(name, color, x, y, orientation="horizontal", style="compact",
                secondary_color=None):
    c = int(color)
    if style == "compact":
        pixels = [[2] * 11,
                  [2, c, c, c, 4, 3, 4, c, c, c, 2],
                  [2, c, 4, 4, 4, 3, 4, 4, 4, c, 2],
                  [2, c, c, c, 4, 3, 4, c, c, c, 2],
                  [2] * 11]
    elif style == "large":
        pixels = [[2] * 13,
                  [2, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 2],
                  [2, 4, c, c, c, 4, 3, 4, c, c, c, 4, 2],
                  [2, 4, c, 4, 4, 4, 3, 4, 4, 4, c, 4, 2],
                  [2, 4, c, c, c, 4, 3, 4, c, c, c, 4, 2],
                  [2, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 2],
                  [2] * 13]
    else:
        raise ValueError("rail style must be compact or large")
    if secondary_color is not None:
        pixels[1][1] = int(secondary_color)
    if orientation == "vertical":
        pixels = [list(row) for row in zip(*pixels)]
    elif orientation != "horizontal":
        raise ValueError("rail orientation must be horizontal or vertical")
    return Sprite(pixels, name=name, x=int(x), y=int(y), layer=2, tags=[names.TAG_RAIL])


def button_sprite(name, color, x, y, style="compact"):
    c = int(color)
    if style == "compact":
        pixels = [[2, 2, 2, 2, 2], [2, 4, c, 4, 2], [2, c, c, c, 2],
                  [2, 4, c, 4, 2], [2, 2, 2, 2, 2]]
    elif style == "large":
        pixels = [[2] * 7, [2, 4, 4, 4, 4, 4, 2],
                  [2, 4, 4, c, 4, 4, 2], [2, 4, c, c, c, 4, 2],
                  [2, 4, 4, c, 4, 4, 2], [2, 4, 4, 4, 4, 4, 2], [2] * 7]
    else:
        raise ValueError("button style must be compact or large")
    return Sprite(pixels, name=name, x=int(x), y=int(y), layer=2,
                  tags=[names.TAG_BUTTON])


def build_level(spec):
    """Build the actual ARCEngine level described by a full JSON spec."""
    sprites = [obstacle_sprite(obstacle["name"], obstacle["cells"],
                               obstacle.get("color", OBSTACLE_COLOR),
                               obstacle.get("ink_stride", 1))
               for obstacle in spec.get("obstacles", [])]
    for rod in spec["rods"]:
        sprites.append(rod_sprite(rod["name"], rod["color"], rod["x"], rod["y"],
                                  rod["rotation"], rod["length"]))
    for index, pin in enumerate(spec["pins"]):
        sprites.append(pin_sprite(f"pin{index}", pin["x"], pin["y"]))
    for index, target in enumerate(spec["targets"]):
        sprites.append(target_sprite(f"target{index}", target["x"], target["y"]))
    for rail in spec["rails"]:
        sprites.append(rail_sprite(rail["name"], rail["color"], rail["x"], rail["y"],
                               rail["orientation"], rail.get("style", "compact"),
                               rail.get("secondary_color")))
    for button in spec["buttons"]:
        sprites.append(button_sprite(button["name"], button["color"],
                                     button["x"], button["y"],
                                     button.get("style", "compact")))
    data = {
        names.KEY_STEP_COUNTER: int(spec["step_counter"]),
        names.KEY_CHILDREN: [list(edge) for edge in spec.get("children", [])],
    }
    return Level(sprites=sprites, grid_size=(64, 64), data=data,
                 name=spec.get("name", "generated-s5i5"))


def solution_actions(spec):
    return [tuple(int(value) for value in action) for action in spec["solution"]]


# -- procedural grammar --------------------------------------------------------

_DIRECTIONS = ((0, -1, 0), (1, 0, 90), (0, 1, 180), (-1, 0, 270))


def _origin(base, rotation, length):
    dx, dy = names.EXTEND_DIRECTION[rotation]
    cells = [(base[0] + dx * index, base[1] + dy * index)
             for index in range(length)]
    return min(x for x, _ in cells) * UNIT, min(y for _, y in cells) * UNIT


def _place_chain(rng, lengths, occupied):
    for _ in range(500):
        base = (rng.randint(2, 18), rng.randint(2, 11))
        local = []
        current = base
        previous = None
        valid = True
        for length in lengths:
            directions = list(_DIRECTIONS)
            rng.shuffle(directions)
            if previous is not None:
                directions.sort(key=lambda item: (item[0] == -previous[0]
                                                   and item[1] == -previous[1]))
            chosen = None
            for dx, dy, rotation in directions:
                cells = [(current[0] + dx * step, current[1] + dy * step)
                         for step in range(length)]
                used = {cell for row in local for cell in row[3]}
                if (all(1 <= x <= 19 and 1 <= y <= 12 for x, y in cells)
                        and not set(cells) & (occupied | used)):
                    chosen = (dx, dy, rotation, cells)
                    break
            if chosen is None:
                valid = False
                break
            dx, dy, rotation, cells = chosen
            local.append((current, rotation, length, cells))
            current = (current[0] + dx * length, current[1] + dy * length)
            previous = (dx, dy)
        if valid:
            return local
    return None


def _place_branch(rng, occupied):
    for _ in range(300):
        center = (rng.randint(3, 17), rng.randint(3, 10))
        cells = {center, (center[0], center[1] - 1), (center[0] + 1, center[1]),
                 (center[0], center[1] + 1), (center[0] - 1, center[1])}
        if cells & occupied:
            continue
        directions = list(_DIRECTIONS)
        rng.shuffle(directions)
        rows = [(center, directions[0][2], 1, [center])]
        for dx, dy, rotation in _DIRECTIONS:
            cell = (center[0] + dx, center[1] + dy)
            rows.append((cell, rotation, 1, [cell]))
        return rows
    return None


def _tier4_geometry(rng):
    """Sample one calibrated tier-4 causal layout, in rod units.

    The topology is pinned to the native tier: an independent one-unit pin
    carrier extends rightwards along a corridor row through four one-unit
    shared-colour children, each the explicit child of its own parent above
    the corridor.  Three parents are short downward rods; the fourth is a
    long upward rod that has to retract.  The seed chooses the carrier
    origin, the corridor row, the gate spacing, which parent owns which gate,
    every parent column and row, and the long parent's length.  Row indices
    follow ``_COMPONENT_LENGTHS[4]``: rod0 is the carrier, rods 1/3/5/7 are
    parents and rods 2/4/6/8 their children.
    """
    carrier_x = rng.randint(1, 2)
    corridor_y = rng.randint(9, 10)
    gate_xs = [carrier_x + offset for offset in sorted(rng.sample(range(1, 6), 4))]
    gate_order = list(range(4))
    rng.shuffle(gate_order)
    # Parents live to the right of the corridor sweep, in distinct columns.
    columns = rng.sample(range(8, 20), 4)
    long_top = rng.randint(0, 2)
    long_length = rng.randint(6, min(11, corridor_y - long_top))
    geometry = {0: (carrier_x, corridor_y, 90, 1)}
    for component in range(4):
        parent, child = 2 * component + 1, 2 * component + 2
        if component == 3:
            geometry[parent] = (columns[component], long_top, 0, long_length)
        else:
            geometry[parent] = (columns[component], rng.randint(1, 3), 180, 1)
        geometry[child] = (gate_xs[gate_order[component]], corridor_y, 180, 1)
    return geometry


def _place_rods(rng, difficulty):
    occupied = set()
    rows = []
    components = []
    children = []
    for lengths in _COMPONENT_LENGTHS[difficulty]:
        start = len(rows)
        branch = difficulty == 8 and len(lengths) == 5
        placed = _place_branch(rng, occupied) if branch else _place_chain(rng, lengths, occupied)
        if placed is None:
            return None
        for base, rotation, length, cells in placed:
            name = f"rod{len(rows)}"
            x, y = _origin(base, rotation, length)
            rows.append({"name": name, "color": 1, "x": x, "y": y,
                         "rotation": rotation, "length": length})
            occupied.update(cells)
        indices = list(range(start, len(rows)))
        components.append(indices)
        if branch:
            children.extend((rows[indices[0]]["name"], rows[index]["name"])
                            for index in indices[1:])
        else:
            children.extend((rows[parent]["name"], rows[child]["name"])
                            for parent, child in zip(indices, indices[1:]))

    if difficulty == 4:
        # Native tier 4 has no in-arena obstacle pixels. Its standalone pin
        # carrier shares a control with four children of independently acted
        # parents. Each child blocks the carrier's next extension until its
        # parent moves it vertically out of the horizontal pin corridor. The
        # long final parent can retract, while the other three extend, so the
        # route needs both rail directions and five physical controls.
        for index, (x, y, rotation, length) in _tier4_geometry(rng).items():
            rows[index].update(x=x * UNIT, y=y * UNIT, rotation=rotation,
                               length=length)
        occupied = set()
        for rod in rows:
            x, y = int(rod["x"]) // UNIT, int(rod["y"]) // UNIT
            length, rotation = int(rod["length"]), int(rod["rotation"])
            if rotation in (0, 180):
                cells = {(x, y + step) for step in range(length)}
            else:
                cells = {(x + step, y) for step in range(length)}
            if occupied & cells:
                return None
            occupied.update(cells)

    if difficulty == 8:
        # The pin carrier is independent, as in the native reference. Three
        # chain rods successively gate its upward rail; a shared-colour branch
        # companion gates the final extension until the branch is rotated.
        # The long branch child makes every preparatory retract necessary.
        _apply_tier8_recipe(rows, _select_tier8_recipe(rng))
        occupied = set()
        for rod in rows:
            x, y = int(rod["x"]) // UNIT, int(rod["y"]) // UNIT
            length, rotation = int(rod["length"]), int(rod["rotation"])
            if rotation in (0, 180):
                occupied.update((x, y + step) for step in range(length))
            else:
                occupied.update((x + step, y) for step in range(length))
        if sum(int(rod["length"]) for rod in rows) != len(occupied):
            return None
        # Obstacle sampling must not silently occupy the swept volume of the
        # staged branch and its clearing chains. These are placement reserves,
        # not hidden collision cells and are not emitted as geometry.
        occupied.update((x, y) for x in range(5, 20) for y in range(0, 12))

    uncontrolled = {
        components[component][local]
        for component, local in _UNCONTROLLED_LOCAL[difficulty]
    }
    controlled = [index for index in range(len(rows)) if index not in uncontrolled]
    shuffled = controlled[:]
    rng.shuffle(shuffled)
    # On shared-colour tiers, make the first shared group contain a pinned
    # ancestor/descendant pair. One click then changes the pin equation twice;
    # shared dispatch is a real dependency rather than incidental animation.
    shared_pair = None
    if difficulty == 4:
        # The pin carrier and every paired child share one rail. Their four
        # parents retain distinct rails, matching the native causal roles.
        shared_members = [components[0][0], components[1][1],
                          components[2][1], components[3][1],
                          components[4][1]]
        shuffled = shared_members + [index for index in shuffled
                                     if index not in shared_members]
    elif difficulty == 5:
        shared_pair = (components[3][0], components[3][1])
    if shared_pair is not None:
        shuffled = list(shared_pair) + [index for index in shuffled
                                        if index not in shared_pair]
    elif difficulty == 8:
        # group0 couples the independent pin carrier with one off-pin branch
        # companion. Groups1-4 are distinct native collision gates; group6
        # rotates the four-way branch. Two harmless leaves remain uncontrolled.
        shuffled = [components[0][0], components[3][1],
                    components[1][1], components[2][0],
                    components[2][1], components[1][0],
                    components[1][2], components[3][3],
                    components[3][4], components[3][0]]
    colors = list(ROD_COLORS)
    rng.shuffle(colors)
    cursor = 0
    groups = []
    for color, size in zip(colors, _CONTROL_GROUP_SIZES[difficulty]):
        group = shuffled[cursor:cursor + size]
        cursor += size
        groups.append((color, group))
        for index in group:
            rows[index]["color"] = color
    if cursor != len(controlled):
        return None
    pin_indices = [components[component][local] for component, local in _PIN_LOCAL[difficulty]]
    return rows, children, components, groups, pin_indices, occupied


def _control_specs(difficulty, groups):
    reference = REFERENCE_CHARACTERIZATION[difficulty]
    colors = [color for color, _ in groups]
    rail_colors = colors[:reference["rails"]]
    rail_style = "large" if difficulty in (1, 2, 3, 4, 6) else "compact"
    if difficulty == 1:
        rails = [
            {"name": "rail0", "color": rail_colors[0], "x": 5, "y": 54,
             "orientation": "horizontal", "style": rail_style},
            {"name": "rail1", "color": rail_colors[1], "x": 56, "y": 45,
             "orientation": "vertical", "style": rail_style},
        ]
    else:
        if rail_style == "large":
            slots = [(3, 54), (18, 54), (33, 54), (48, 54), (10, 45), (38, 45)]
        else:
            slots = [(2, 49), (19, 49), (36, 49), (2, 56), (19, 56), (36, 56)]
        rails = [
            {"name": f"rail{index}", "color": color, "x": slots[index][0],
             "y": slots[index][1], "orientation": "horizontal", "style": rail_style}
            for index, color in enumerate(rail_colors)
        ]
    button_count = reference["buttons"]
    if difficulty == 7:
        button_colors = [colors[0], colors[1], colors[2], colors[5]]
    elif difficulty == 8:
        button_colors = [colors[6]]
    else:
        button_colors = colors[:button_count]
    button_style = "large" if difficulty in (6, 8) else "compact"
    button_y = 41 if difficulty == 8 else 43
    button_slots = [(5 + 10 * index, button_y) for index in range(button_count)]
    buttons = [
        {"name": f"button{index}", "color": color, "x": x, "y": y,
         "style": button_style}
        for index, (color, (x, y)) in enumerate(zip(button_colors, button_slots))
    ]
    return rails, buttons


def _obstacle_specs(rng, difficulty, occupied, rods, groups):
    reference = REFERENCE_CHARACTERIZATION[difficulty]
    profile = PROFILES[difficulty]
    calibrated_topology = difficulty in (3, 4, 5)
    if calibrated_topology:
        low, high = profile["interior_obstacle_cells"]
    else:
        low, high = profile["obstacle_cells"]
    count = rng.randint(low, high)
    # Reserve a clean lower control panel instead of letting obstacle pixels
    # compete with buttons/rails for attention.
    candidates = [(x, y) for y in range(13) for x in range(21) if (x, y) not in occupied]
    forced = None
    controlled_colors = {color for color, _ in groups}
    candidates_rods = [rod for rod in rods if rod["color"] in controlled_colors]
    rng.shuffle(candidates_rods)
    for rod in candidates_rods:
        length = int(rod["length"])
        rotation = int(rod["rotation"])
        if rotation == 0:
            base = (rod["x"] // UNIT, rod["y"] // UNIT + length - 1)
        elif rotation in (90, 180):
            base = (rod["x"] // UNIT, rod["y"] // UNIT)
        else:
            base = (rod["x"] // UNIT + length - 1, rod["y"] // UNIT)
        dx, dy = names.EXTEND_DIRECTION[rotation]
        gap = rng.randint(3, 5)
        probe = (base[0] + dx * (length + gap), base[1] + dy * (length + gap))
        if probe in candidates:
            forced = probe
            break

    if calibrated_topology:
        def covering_positions(start, end):
            positions = list(range(start, end - 1, UNIT))
            final = end - UNIT + 1
            if positions[-1] != final:
                positions.append(final)
            return positions

        def full_boundary():
            horizontal = covering_positions(-UNIT, names.FRAME + UNIT - 1)
            vertical = covering_positions(0, names.FRAME - 1)
            return ([[x, -UNIT] for x in horizontal]
                    + [[x, names.FRAME] for x in horizontal]
                    + [[-UNIT, y] for y in vertical]
                    + [[names.FRAME, y] for y in vertical])

        def top_left_boundary():
            horizontal = covering_positions(-UNIT, names.FRAME - 1)
            vertical = covering_positions(0, names.FRAME - 1)
            return ([[x, -UNIT] for x in horizontal]
                    + [[-UNIT, y] for y in vertical])

        boundary = top_left_boundary() if difficulty == 4 else full_boundary()
        obstacles = [{"name": "boundary0", "cells": boundary,
                      "color": OBSTACLE_COLOR, "ink_stride": 1}]
        interior_count = profile["interior_obstacle_cells"][0]
        if interior_count == 0:
            return obstacles
        available = set(candidates)
        start = forced if forced in available else rng.choice(sorted(available))
        selected = [start]
        available.remove(start)
        while len(selected) < interior_count:
            frontier = set()
            for x, y in selected:
                frontier.update({(x - 1, y), (x + 1, y),
                                 (x, y - 1), (x, y + 1)})
            frontier &= available
            if not frontier:
                return None
            cell = rng.choice(sorted(frontier))
            selected.append(cell)
            available.remove(cell)
        sizes = (1,) if difficulty == 3 else (5, 1)
        offset = 0
        for index, size in enumerate(sizes, 1):
            cells = [[x * UNIT, y * UNIT]
                     for x, y in selected[offset:offset + size]]
            obstacles.append({"name": f"obstacle{index}", "cells": cells,
                              "color": OBSTACLE_COLOR, "ink_stride": 1})
            offset += size
        return obstacles

    candidate_set = set(candidates)
    perimeter = ([(x, 0) for x in range(21)]
                 + [(20, y) for y in range(1, 13)]
                 + [(x, 12) for x in range(19, -1, -1)]
                 + [(0, y) for y in range(11, 0, -1)])
    offset = rng.randrange(len(perimeter))
    perimeter = perimeter[offset:] + perimeter[:offset]
    if rng.random() < 0.5:
        perimeter.reverse()
    border = [cell for cell in perimeter if cell in candidate_set]
    interior_set = candidate_set - set(border)
    interior = []
    if rng.random() < 0.5:
        lines = list(range(1, 12))
        rng.shuffle(lines)
        for y in lines:
            row = [(x, y) for x in range(1, 20) if (x, y) in interior_set]
            if rng.random() < 0.5:
                row.reverse()
            interior.extend(row)
    else:
        lines = list(range(1, 20))
        rng.shuffle(lines)
        for x in lines:
            column = [(x, y) for y in range(1, 12) if (x, y) in interior_set]
            if rng.random() < 0.5:
                column.reverse()
            interior.extend(column)
    selected = [] if forced is None else [forced]
    for cell in border + interior:
        if cell not in selected:
            selected.append(cell)
        if len(selected) == count:
            break
    if len(selected) != count:
        return None
    obstacle_count = reference["obstacles"]
    groups_of_cells = [[] for _ in range(obstacle_count)]
    for index, (x, y) in enumerate(selected):
        group = min(index * obstacle_count // len(selected), obstacle_count - 1)
        groups_of_cells[group].append([x * UNIT, y * UNIT])
    return [{"name": f"obstacle{index}", "cells": cells,
             "color": OBSTACLE_PALETTES[difficulty][index],
             "ink_stride": 1}
            for index, cells in enumerate(groups_of_cells)]


def _draft(rng, difficulty):
    placed = _place_rods(rng, difficulty)
    if placed is None:
        return None, "rod_geometry"
    rods, children, _, groups, pin_indices, occupied = placed
    rails, buttons = _control_specs(difficulty, groups)
    obstacles = _obstacle_specs(rng, difficulty, occupied, rods, groups)
    if obstacles is None:
        return None, "obstacle_geometry"
    pins = [{"x": rods[index]["x"], "y": rods[index]["y"]} for index in pin_indices]
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "source": "generated_only",
        "difficulty": difficulty,
        "reference_level": difficulty,
        "context_index": difficulty - 1,
        "step_counter": REFERENCE_CHARACTERIZATION[difficulty]["budget"],
        "obstacles": obstacles,
        "rods": rods,
        "pins": pins,
        "targets": [],
        "rails": rails,
        "buttons": buttons,
        "children": [list(edge) for edge in children],
        "reference_calibration": "aggregate per-tier measurements; no official geometry or route copied",
    }, None


# -- witness construction and evidence ----------------------------------------

def _rod_tuple_changed(before, after):
    if before[0] != after[0] or before[1] != after[1]:
        return True
    return before[2].shape != after[2].shape or before[2].tobytes() != after[2].tobytes()


def _descendants(spec):
    children = defaultdict(list)
    for parent, child in spec["children"]:
        children[parent].append(child)
    result = {}
    for root in [rod["name"] for rod in spec["rods"]]:
        seen = set()
        stack = list(children[root])
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            stack.extend(children[name])
        result[root] = seen
    return result, children


def _pin_path_rods(spec):
    parents = {child: parent for parent, child in spec["children"]}
    relevant = set()
    for pin in spec["pins"]:
        matches = [rod["name"] for rod in spec["rods"]
                   if int(rod["x"]) == int(pin["x"]) and int(rod["y"]) == int(pin["y"])]
        for name in matches:
            while name not in relevant:
                relevant.add(name)
                if name not in parents:
                    break
                name = parents[name]
    return relevant


def _transition_features(spec, control, before, after):
    offset = len(spec["obstacles"])
    changed = [index for index, (left, right) in enumerate(zip(before[0], after[0]))
               if _rod_tuple_changed(left, right)]
    changed_rods = [index - offset for index in changed if index >= offset]
    rods = spec["rods"]
    control_spec = next((row for row in spec["rails"] + spec["buttons"]
                         if row["name"] == control.name), None)
    acted_colors = ({int(control_spec["color"])}
                    if control_spec is not None else {int(control.color)})
    if control_spec is not None and "secondary_color" in control_spec:
        acted_colors.add(int(control_spec["secondary_color"]))
    direct = {index for index, rod in enumerate(rods)
              if int(rod["color"]) in acted_colors}
    descendants, child_map = _descendants(spec)
    linked_names = set().union(*(descendants[rods[index]["name"]] for index in direct)) if direct else set()
    changed_names = {rods[index]["name"] for index in changed_rods}
    rail = next((row for row in spec["rails"] if row["name"] == control.name), None)
    direct_names = {rods[index]["name"] for index in direct}
    moved_pins = [index for index, (left, right) in enumerate(zip(before[1], after[1]))
                  if left != right]
    return {
        "kind": control.kind,
        "control_name": control.name,
        "control": [names.ACTION_CLICK, int(control.click[0]), int(control.click[1])],
        "changed_rods": len(changed_rods),
        "shared": len(direct) > 1 and len(direct & set(changed_rods)) > 1,
        "linked": bool(changed_names & linked_names),
        "branch": any(len(child_map[name]) > 1 and name in direct_names for name in child_map),
        "vertical": rail is not None and rail["orientation"] == "vertical",
        "pin_motion": before[1] != after[1],
        "moved_pins": moved_pins,
        "pin_path": bool(changed_names & _pin_path_rods(spec)),
        "constraint_motion": bool(changed_names) and not bool(
            changed_names & _pin_path_rods(spec)
        ),
    }


def _fast_transition_features(spec, engine, control, before, after):
    """The same construction facts from an exact compact action boundary."""
    changed_names = {
        rod.name for rod, left, right in zip(engine.native_rods, before[0], after[0])
        if left != right
    }
    rods = spec["rods"]
    control_spec = next((row for row in spec["rails"] + spec["buttons"]
                         if row["name"] == control.name), None)
    acted_colors = set(control.colors)
    direct_names = {rod["name"] for rod in rods
                    if int(rod["color"]) in acted_colors}
    descendants, child_map = _descendants(spec)
    linked_names = set().union(*(descendants[name] for name in direct_names)) if direct_names else set()
    rail = next((row for row in spec["rails"] if row["name"] == control.name), None)
    moved_pins = [index for index, (left, right) in enumerate(zip(before[1], after[1]))
                  if left != right]
    return {
        "kind": control.kind,
        "control_name": control.name,
        "control": [names.ACTION_CLICK, int(control.click[0]), int(control.click[1])],
        "changed_rods": len(changed_names),
        "shared": len(direct_names) > 1 and len(direct_names & changed_names) > 1,
        "linked": bool(changed_names & linked_names),
        "branch": any(len(child_map[name]) > 1 and name in direct_names for name in child_map),
        "vertical": rail is not None and rail["orientation"] == "vertical",
        "pin_motion": before[1] != after[1],
        "moved_pins": moved_pins,
        "pin_path": bool(changed_names & _pin_path_rods(spec)),
        "constraint_motion": bool(changed_names) and not bool(
            changed_names & _pin_path_rods(spec)
        ),
    }


def _copy_facts(facts):
    return {
        key: (set(value) if isinstance(value, set) else value)
        for key, value in facts.items()
    }


def _record_feature(facts, feature, *, constraint=False):
    facts["kinds"].add(feature["kind"])
    facts["controls"].add(feature["control_name"])
    facts["moved_pins"].update(feature["moved_pins"])
    facts["pin_motion_events"] += int(feature["pin_motion"])
    for flag in ("vertical", "linked", "shared", "branch", "pin_motion"):
        facts[flag] = facts[flag] or feature[flag]
    if constraint:
        facts["constraint"] = True
        facts["constraint_controls"].add(feature["control_name"])
        facts["constraint_events"] += 1


def _tier8_rank(facts, actions, pin_set, compact_gap=None, compact_length=0):
    """Prefer distinct causal stages and mechanics before route length."""
    return (
        min(len(facts["constraint_controls"]), 3),
        min(facts["constraint_events"], 3),
        min(len(facts["controls"]), 5),
        min(len(facts["kinds"]), 3),
        int(facts["shared"]) + int(facts["linked"]) + int(facts["branch"]),
        -(1000 if compact_gap is None else compact_gap),
        compact_length,
        int(all(0 <= x <= 61 and 0 <= y <= 38 for x, y in pin_set)),
        sum(min(x, 61 - x, y, 38 - y) for x, y in pin_set),
        min(facts["pin_motion_events"], 24),
        len(facts["moved_pins"]),
        len(set(pin_set)),
        -len(actions),
    )


def _greedy_fast_reduction(engine, action_indices, actions, targets):
    """Cheap independent prefilter; native deletion remains authoritative."""
    def first_completion(route):
        state = engine.start
        for index, action in enumerate(route, 1):
            state, _ = engine.step(state, action_indices[tuple(action)])
            if all(target in state[1] for target in targets):
                return index
        return None

    route = list(actions)
    changed = True
    while changed:
        changed = False
        for width in range(min(8, len(route)), 0, -1):
            for index in range(len(route) - width, -1, -1):
                candidate = route[:index] + route[index + width:]
                first = first_completion(candidate)
                if first is not None:
                    route = candidate[:first]
                    changed = True
                    break
            if changed:
                break
    return route


def _construct_tier8_witness(rng, spec, transition_limit):
    """Build tier 8 from explicit native-verified unblock/use stages.

    The ordinary constructor follows one scored path. Tier 8 needs several
    distinct off-path controls to clear successive pin-path moves, so retain a
    small bounded beam. An off-path action enters the beam only together with a
    pin-path action that was blocked immediately before it and moves a pin
    immediately after it. This prevents free rod travel from accumulating as
    alleged constraint work.
    """
    high = PROFILES[8]["action_range"][1]
    probe = dict(spec, targets=[{"x": -99, "y": -99}])
    env = Env([build_level(probe)])
    layout = extract(env)
    controls = {(names.ACTION_CLICK, control.click[0], control.click[1]): control
                for control in layout.controls}
    engine = _FastEngine(env, layout)
    start = engine.start
    action_indices = {action: index for index, action in enumerate(engine.actions)}
    pin_path = _pin_path_rods(spec)
    pin_path_colors = {int(rod["color"]) for rod in spec["rods"]
                       if rod["name"] in pin_path}
    pin_actions = [action for action, control in controls.items()
                   if pin_path_colors.intersection(control.colors)]
    initial_facts = {
        "kinds": set(), "controls": set(), "vertical": False,
        "linked": False, "shared": False, "branch": False,
        "pin_motion": False, "constraint": False,
        "constraint_controls": set(), "constraint_events": 0,
        "moved_pins": set(), "pin_motion_events": 0,
    }
    # state, actions, facts, selected pin-rail directions, rotations, and
    # path-local seen states
    frontier = [(start, [], initial_facts, {}, Counter(), frozenset((start,)))]
    work = 0
    beam_width = 8
    for _ in range(high):
        expanded = []
        for state, actions, facts, rail_directions, rotations, seen in frontier:
            blocked = []
            for pin_action in pin_actions:
                if work >= transition_limit:
                    return None, "verification_work_exhausted", work
                work += 1
                pin_state, _ = engine.step(state, action_indices[pin_action])
                pin_feature = _fast_transition_features(
                    spec, engine, controls[pin_action], state, pin_state
                )
                if pin_state == state or not pin_feature["pin_motion"]:
                    blocked.append(pin_action)

            candidate_actions = list(layout.actions)
            rng.shuffle(candidate_actions)
            for action in candidate_actions:
                if len(actions) >= high or work >= transition_limit:
                    break
                control = controls[action]
                if control.kind == "rotate" and rotations[action] >= 3:
                    continue
                if control.kind in ("extend", "retract"):
                    selected = rail_directions.get(control.name)
                    if (pin_path_colors.intersection(control.colors)
                            and selected is not None and selected != control.kind):
                        continue
                work += 1
                mid_state, _ = engine.step(state, action_indices[action])
                if mid_state == state or mid_state in seen:
                    continue
                feature = _fast_transition_features(spec, engine, control, state, mid_state)
                off_path = feature["constraint_motion"] and not feature["pin_path"]
                stages = []
                if off_path:
                    for unlocked_action in blocked:
                        if work >= transition_limit:
                            return None, "verification_work_exhausted", work
                        work += 1
                        end_state, _ = engine.step(
                            mid_state, action_indices[unlocked_action]
                        )
                        unlocked = _fast_transition_features(
                            spec, engine, controls[unlocked_action], mid_state, end_state
                        )
                        if (end_state not in (mid_state, state) and end_state not in seen
                                and unlocked["pin_motion"]):
                            stages.append((
                                [action, unlocked_action], end_state,
                                [(feature, True), (unlocked, False)],
                                {mid_state, end_state},
                            ))
                    # A native reference constraint may need one preparatory
                    # off-path move before the distinct move that opens the
                    # pin path. Admit only the complete setup/unlock/use
                    # triple; never retain free-standing setup travel.
                    if len(actions) + 3 <= high:
                        blocked_mid = []
                        for pin_action in pin_actions:
                            work += 1
                            pin_state, _ = engine.step(
                                mid_state, action_indices[pin_action]
                            )
                            pin_feature = _fast_transition_features(
                                spec, engine, controls[pin_action], mid_state, pin_state
                            )
                            if pin_state == mid_state or not pin_feature["pin_motion"]:
                                blocked_mid.append(pin_action)
                        for second_action in candidate_actions:
                            second_control = controls[second_action]
                            if (second_control.kind == "rotate"
                                    and rotations[second_action]
                                    + int(second_action == action) >= 3):
                                continue
                            work += 1
                            second_state, _ = engine.step(
                                mid_state, action_indices[second_action]
                            )
                            if second_state in (state, mid_state) or second_state in seen:
                                continue
                            second_feature = _fast_transition_features(
                                spec, engine, second_control, mid_state, second_state
                            )
                            if not (second_feature["constraint_motion"]
                                    and not second_feature["pin_path"]):
                                continue
                            for unlocked_action in blocked_mid:
                                work += 1
                                end_state, _ = engine.step(
                                    second_state, action_indices[unlocked_action]
                                )
                                unlocked = _fast_transition_features(
                                    spec, engine, controls[unlocked_action],
                                    second_state, end_state
                                )
                                if (end_state not in (state, mid_state, second_state)
                                        and end_state not in seen
                                        and unlocked["pin_motion"]):
                                    stages.append((
                                        [action, second_action, unlocked_action],
                                        end_state,
                                        [(feature, False), (second_feature, True),
                                         (unlocked, False)],
                                        {mid_state, second_state, end_state},
                                    ))
                elif feature["pin_path"] or feature["pin_motion"]:
                    stages.append((
                        [action], mid_state, [(feature, False)], {mid_state}
                    ))
                for added, end_state, recorded, stage_states in stages:
                    if len(actions) + len(added) > high:
                        continue
                    next_facts = _copy_facts(facts)
                    for recorded_feature, causal in recorded:
                        _record_feature(next_facts, recorded_feature, constraint=causal)
                    next_directions = dict(rail_directions)
                    next_rotations = rotations.copy()
                    for used in added:
                        used_control = controls[used]
                        if used_control.kind == "rotate":
                            next_rotations[used] += 1
                        elif pin_path_colors.intersection(used_control.colors):
                            next_directions.setdefault(used_control.name, used_control.kind)
                    next_actions = actions + added
                    pin_set = tuple(sorted(end_state[1]))
                    next_seen = seen | stage_states
                    expanded.append((end_state, next_actions, next_facts,
                                     next_directions, next_rotations, next_seen, pin_set))
        if not expanded:
            return None, "constructive_dead_end", work
        # Keep semantically different fact signatures even at identical native
        # geometry; the route evidence is part of the construction state.
        best = {}
        for row in expanded:
            state, actions, facts, _, _, _, pin_set = row
            signature = (state, frozenset(facts["kinds"]),
                         frozenset(facts["controls"]),
                         frozenset(facts["constraint_controls"]),
                         facts["constraint_events"], facts["pin_motion_events"],
                         facts["shared"],
                         facts["linked"], facts["branch"])
            rank = _tier8_rank(facts, actions, pin_set)
            if signature not in best or rank > best[signature][0]:
                best[signature] = (rank, row[:-1])
        ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)
        frontier = [row for _, row in ranked[:beam_width]]
        # Native deletion is deliberately expensive. Check only the strongest
        # two semantic candidates in a layer, never every compact expansion.
        for state, actions, facts, _, _, _ in frontier[:2]:
            pin_set = tuple(sorted(state[1]))
            if (not _requirements_met(8, facts)
                    or len(actions) < PROFILES[8]["action_range"][0]
                    or len(set(pin_set)) != len(spec["pins"])
                    or not all(0 <= x <= 61 and 0 <= y <= 38 for x, y in pin_set)):
                continue
            targets = [{"x": x, "y": y} for x, y in pin_set]
            candidate_spec = dict(spec, targets=targets)
            try:
                if _greedy_fast_reduction(
                        engine, action_indices, actions, tuple(pin_set)) != actions:
                    continue
                if _greedy_native_reduction(candidate_spec, actions) == actions:
                    return (actions, targets), None, work
            except (AssertionError, KeyError, TypeError, ValueError,
                    IndexError, RecursionError):
                continue
    return None, "constructive_profile", work


def _tier8_recipe_stages(spec):
    """Derive the finite recipe from actual collision-bearing geometry."""
    rods = {rod["name"]: rod for rod in spec["rods"]}
    try:
        pin_x = int(spec["pins"][0]["x"])
        first = rods["rod2"]
        second = rods["rod5"]
        third = rods["rod3"]
        branch = rods["rod10"]
        if (len(spec["pins"]) != 1
                or any(int(rod["rotation"]) != 90
                       for rod in (first, second, third, branch))
                or int(rods["rod0"]["rotation"]) != 0):
            return None

        def clearance(anchor):
            gap = pin_x - anchor
            return gap // UNIT if gap > 0 and gap % UNIT == 0 else 0

        first_count = int(first["length"]) - clearance(int(first["x"]))
        second_count = int(second["length"]) - clearance(int(second["x"]))
        # The first gate is a parent of rod3, so shrinking rod2 translates the
        # third gate left before it is used.
        third_anchor = int(third["x"]) - UNIT * first_count
        third_count = int(third["length"]) - clearance(third_anchor)
        branch_count = int(branch["length"]) - 1
        counts = (first_count, second_count, third_count, branch_count)
        if any(count < 1 for count in counts):
            return None
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return (
        ("rail1", "retract", first_count),
        ("rail0", "extend", 1),
        ("rail2", "retract", second_count),
        ("rail0", "extend", 1),
        ("rail3", "retract", third_count),
        ("rail4", "retract", branch_count),
        ("button0", "rotate", 1),
        ("rail0", "extend", 1),
    )


def _construct_tier8_recipe(spec, transition_limit):
    """Replay one selected independent-pin/branch-companion construction."""
    stages = _tier8_recipe_stages(spec)
    if stages is None:
        return None, "constructive_profile", 0
    required_work = sum(count for _, _, count in stages)
    if not PROFILES[8]["action_range"][0] <= required_work <= PROFILES[8]["action_range"][1]:
        return None, "constructive_profile", 0
    if transition_limit < required_work:
        return None, "verification_work_exhausted", 0

    probe = dict(spec, targets=[{"x": -99, "y": -99}])
    env = Env([build_level(probe)])
    layout = extract(env)
    engine = _FastEngine(env, layout)
    controls = {
        (control.name, control.kind): (
            names.ACTION_CLICK, int(control.click[0]), int(control.click[1])
        )
        for control in layout.controls
    }
    action_indices = {action: index for index, action in enumerate(engine.actions)}
    actions = []
    state = engine.start
    work = 0
    try:
        for control_name, kind, count in stages:
            action = controls[(control_name, kind)]
            for _ in range(count):
                next_state, _ = engine.step(state, action_indices[action])
                work += 1
                if next_state == state:
                    return None, "constructive_dead_end", work
                state = next_state
                actions.append(action)
    except KeyError:
        return None, "constructive_profile", work

    pins = tuple(sorted(state[1]))
    if (len(pins) != 1
            or not all(0 <= x <= 61 and 0 <= y <= 38 for x, y in pins)):
        return None, "constructive_profile", work
    targets = [{"x": int(x), "y": int(y)} for x, y in pins]
    return (actions, targets), None, work


def _requirements_met(difficulty, facts):
    profile = PROFILES[difficulty]
    if not set(profile["required_kinds"]) <= facts["kinds"]:
        return False
    if len(facts["controls"]) < profile["distinct_controls"]:
        return False
    if profile.get("require_vertical") and not facts["vertical"]:
        return False
    if profile.get("require_linked") and not facts["linked"]:
        return False
    if profile.get("require_shared") and not facts["shared"]:
        return False
    if profile.get("require_branch") and not facts["branch"]:
        return False
    if profile.get("require_constraint") and not facts["constraint"]:
        return False
    if len(facts["constraint_controls"]) < profile.get("constraint_controls", 0):
        return False
    if facts["constraint_events"] < profile.get("constraint_events", 0):
        return False
    return facts["moved_pins"] == set(range(REFERENCE_CHARACTERIZATION[difficulty]["pins"]))


def _construct_witness(rng, spec, transition_limit):
    difficulty = spec["difficulty"]
    if difficulty == 8:
        return _construct_tier8_recipe(spec, transition_limit)
    _, high = PROFILES[difficulty]["action_range"]
    probe = dict(spec, targets=[{"x": -99, "y": -99}])
    env = Env([build_level(probe)])
    layout = extract(env)
    controls = {(names.ACTION_CLICK, control.click[0], control.click[1]): control
                for control in layout.controls}
    engine = _Engine(env)
    key, data = engine.snapshot()
    seen = {key}
    pin_sets = {tuple(sorted((pin.x, pin.y) for pin in engine.pins))}
    actions = []
    facts = {"kinds": set(), "controls": set(), "vertical": False,
             "linked": False, "shared": False, "branch": False,
             "pin_motion": False, "constraint": False,
             "constraint_controls": set(), "constraint_events": 0,
             "moved_pins": set()}
    rail_directions = {}
    rotation_counts = Counter()
    work = 0
    forced_action = None
    pin_path_colors = {int(rod["color"]) for rod in spec["rods"]
                       if rod["name"] in _pin_path_rods(spec)}
    pin_control_actions = [action for action, control in controls.items()
                           if pin_path_colors.intersection(control.colors)]

    while len(actions) < high:
        blocked_pin_actions = []
        if PROFILES[difficulty].get("require_constraint"):
            for probe_action in pin_control_actions:
                if work >= transition_limit:
                    return None, "verification_work_exhausted", work
                work += 1
                engine.restore(data)
                engine.click(probe_action[1], probe_action[2])
                _, probe_data = engine.snapshot()
                probe_feature = _transition_features(
                    spec, controls[probe_action], data, probe_data
                )
                if not probe_feature["pin_motion"]:
                    blocked_pin_actions.append(probe_action)
        candidates = ([forced_action] if forced_action is not None
                      else list(layout.actions))
        forced_action = None
        rng.shuffle(candidates)
        scored = []
        for action in candidates:
            if work >= transition_limit:
                return None, "verification_work_exhausted", work
            work += 1
            engine.restore(data)
            engine.click(action[1], action[2])
            next_key, next_data = engine.snapshot()
            if next_key in seen:
                continue
            feature = _transition_features(spec, controls[action], data, next_data)
            feature["constraint_unlock"] = False
            feature["unlocked_action"] = None
            if (feature["constraint_motion"] and blocked_pin_actions
                    and PROFILES[difficulty].get("require_constraint")):
                for probe_action in blocked_pin_actions:
                    if work >= transition_limit:
                        return None, "verification_work_exhausted", work
                    work += 1
                    engine.restore(next_data)
                    engine.click(probe_action[1], probe_action[2])
                    _, unlocked_data = engine.snapshot()
                    unlocked = _transition_features(
                        spec, controls[probe_action], next_data, unlocked_data
                    )
                    if unlocked["pin_motion"]:
                        feature["constraint_unlock"] = True
                        feature["unlocked_action"] = probe_action
                        break
            if not (feature["pin_motion"] or feature["pin_path"]
                    or (PROFILES[difficulty].get("require_constraint")
                        and feature["constraint_motion"]
                        and feature["constraint_unlock"])):
                continue
            control = controls[action]
            if control.kind in ("extend", "retract"):
                selected = rail_directions.get(control.name)
                restrict_direction = (
                    difficulty != 8
                    or (difficulty == 8
                        and bool(pin_path_colors.intersection(control.colors)))
                )
                if (restrict_direction and selected is not None
                        and selected != control.kind):
                    continue
            elif rotation_counts[tuple(feature["control"])] >= 3:
                continue
            score = 0
            score += 12 * (feature["kind"] in PROFILES[difficulty]["required_kinds"]
                           and feature["kind"] not in facts["kinds"])
            score += 8 * (tuple(feature["control"]) not in facts["controls"])
            for flag in ("vertical", "linked", "shared", "branch", "pin_motion"):
                score += 10 * (feature[flag] and not facts[flag])
            score += 30 * (feature["constraint_unlock"] and not facts["constraint"])
            # Preserve maneuvering room instead of greedily walking a pin into
            # a board edge. This makes long reference-band witnesses arise
            # from several interacting pin paths, not reversible rail travel.
            score += 0.25 * sum(min(x, 61 - x, y, 38 - y)
                                for x, y in next_data[1])
            score += rng.random()
            scored.append((score, action, next_key, next_data, feature))
        if not scored:
            return None, "constructive_dead_end", work
        _, action, key, data, feature = max(scored, key=lambda row: row[0])
        engine.restore(data)
        actions.append(action)
        seen.add(key)
        facts["kinds"].add(feature["kind"])
        facts["controls"].add(tuple(feature["control"]))
        facts["moved_pins"].update(feature["moved_pins"])
        chosen_control = controls[action]
        if chosen_control.kind in ("extend", "retract"):
            restrict_direction = (
                difficulty != 8
                or (difficulty == 8 and bool(
                    pin_path_colors.intersection(chosen_control.colors)
                ))
            )
            if restrict_direction:
                rail_directions.setdefault(chosen_control.name, chosen_control.kind)
        else:
            rotation_counts[tuple(feature["control"])] += 1
        for flag in ("vertical", "linked", "shared", "branch", "pin_motion"):
            facts[flag] = facts[flag] or feature[flag]
        facts["constraint"] = facts["constraint"] or feature["constraint_unlock"]
        if feature["constraint_unlock"]:
            facts["constraint_controls"].add(feature["control_name"])
            facts["constraint_events"] += 1
        if feature["unlocked_action"] is not None:
            forced_action = feature["unlocked_action"]
        pin_set = tuple(sorted((pin.x, pin.y) for pin in engine.pins))
        unique_pins = pin_set not in pin_sets
        pin_sets.add(pin_set)
        # Target placement is not keyed to a desired certificate length. Once
        # all required relations have acted, sampling may stop at any new pin
        # state; independent deletion/search gates decide whether it is hard.
        requirements_met = unique_pins and _requirements_met(difficulty, facts)
        constrained = PROFILES[difficulty].get("require_constraint")
        stop_floor = (PROFILES[difficulty]["shortcut_bound"] if constrained
                      else PROFILES[difficulty]["action_range"][0])
        sampled_stop = not constrained or rng.random() < 0.10
        if (requirements_met and len(actions) >= stop_floor and sampled_stop):
            if len(set(pin_set)) != len(spec["pins"]):
                continue
            if all(0 <= x <= 61 and 0 <= y <= 38 for x, y in pin_set):
                targets = [{"x": x, "y": y} for x, y in pin_set]
                if constrained:
                    probe_spec = dict(spec, targets=targets)
                    try:
                        if _greedy_native_reduction(probe_spec, actions) != actions:
                            continue
                    except (AssertionError, KeyError, TypeError, ValueError,
                            IndexError, RecursionError):
                        continue
                return (actions, targets), None, work
    return None, "constructive_profile", work


def _native_snapshot(env):
    rods = tuple((sprite.x, sprite.y, sprite.pixels.copy()) for sprite in env.rods())
    pins = tuple((pin.x, pin.y) for pin in env.pins())
    return rods, pins


def _native_relations(spec):
    """Relationships actually installed by the native initialization pass."""
    env = Env([build_level(spec)])
    rods = {rod["name"]: next(sprite for sprite in env.rods()
                               if sprite.name == rod["name"])
            for rod in spec["rods"]}
    reverse = {sprite: name for name, sprite in rods.items()}
    pin_positions = {pin: [int(pin.x), int(pin.y)] for pin in env.pins()}
    native_children = getattr(env.game, names.ATTR_CHILDREN)
    rod_edges = []
    pin_edges = []
    for parent_name, parent in rods.items():
        for child in native_children.get(parent, ()):
            if child in reverse:
                rod_edges.append([parent_name, reverse[child]])
            elif child in pin_positions:
                pin_edges.append([parent_name, pin_positions[child]])
    return {"rod_edges": sorted(rod_edges), "pin_edges": sorted(pin_edges)}


def _native_episode(spec, actions):
    """Replay the target-bearing episode and measure only pre-WIN actions."""
    context = int(spec["context_index"])
    env = Env([build_level(spec) for _ in range(context + 1)])
    env.set_level(context)
    layout = extract(env)
    controls = {(names.ACTION_CLICK, control.click[0], control.click[1]): control
                for control in layout.controls}
    counts = Counter()
    used_controls = set()
    changed_rods_max = 0
    per_pin_motion = [0] * len(spec["pins"])
    budget_start = env.steps_left()
    before_score = env.levels_completed
    for index, action in enumerate(actions, 1):
        action = tuple(action)
        control = controls.get(action)
        if control is None:
            raise ValueError("solution contains a click outside the canonical controls")
        before = _native_snapshot(env)
        observation = env.perform(*action)
        after = _native_snapshot(env)
        feature = _transition_features(spec, control, before, after)
        counts["unchanged_actions" if feature["changed_rods"] == 0 else "changed_actions"] += 1
        counts[feature["kind"]] += 1
        counts["vertical_rail_actions"] += int(feature["vertical"])
        counts["linked_actions"] += int(feature["linked"])
        counts["shared_color_actions"] += int(feature["shared"])
        counts["branch_actions"] += int(feature["branch"])
        counts["pin_motion_actions"] += int(feature["pin_motion"])
        counts["pin_path_actions"] += int(feature["pin_path"])
        counts["constraint_motion_actions"] += int(feature["constraint_motion"])
        changed_rods_max = max(changed_rods_max, feature["changed_rods"])
        used_controls.add(feature["control_name"])
        for pin_index in feature["moved_pins"]:
            per_pin_motion[pin_index] += 1
        completed = env.levels_completed > before_score
        if completed and index != len(actions):
            raise ValueError("native completion occurred before the final stored action")
        if observation.state == GameState.GAME_OVER:
            raise ValueError("native episode exhausted its budget")
        if completed:
            break
    if env.levels_completed != before_score + 1:
        raise ValueError("native episode did not complete")
    result = dict(counts)
    result.update(
        distinct_controls=len(used_controls),
        max_rods_changed_by_one_action=changed_rods_max,
        all_actions_changed=counts["unchanged_actions"] == 0,
        all_actions_pin_relevant=counts["pin_path_actions"] == len(actions),
        all_actions_relevant=(counts["pin_path_actions"]
                              + counts["constraint_motion_actions"] == len(actions)),
        per_pin_motion_actions=per_pin_motion,
        first_completion_action=len(actions),
        native_budget_start=budget_start,
        native_budget_consumed=budget_start - env.steps_left(),
        native_budget_remaining=env.steps_left(),
        won=True,
    )
    return result, env


def _solution_mechanics(spec, actions):
    return _native_episode(spec, actions)[0]


def _apply_relation_ablation(env, spec, ablation):
    kind, parent_name, child_value = ablation
    rods = {rod.name: rod for rod in env.rods() if rod.name in {
        row["name"] for row in spec["rods"]
    }}
    if kind == "rod_edge":
        getattr(env.game, names.ATTR_CHILDREN)[rods[parent_name]].discard(rods[child_value])
    elif kind == "pin_edge":
        pin = next(pin for pin in env.pins()
                   if [int(pin.x), int(pin.y)] == list(child_value))
        getattr(env.game, names.ATTR_CHILDREN)[rods[parent_name]].discard(pin)
    elif kind == "shared_direct":
        rod = rods[parent_name]
        for controlled in getattr(env.game, names.ATTR_RAIL_RODS).values():
            if rod in controlled:
                controlled.remove(rod)
        old_color = int(next(row["color"] for row in spec["rods"]
                            if row["name"] == parent_name))
        rod.pixels[rod.pixels == old_color] = 1
    else:
        raise ValueError("unknown relation ablation")


def _first_native_completion(spec, actions, *, ablation=None):
    context = int(spec["context_index"])
    env = Env([build_level(spec) for _ in range(context + 1)])
    env.set_level(context)
    if ablation is not None:
        _apply_relation_ablation(env, spec, ablation)
    before = env.levels_completed
    for index, action in enumerate(actions, 1):
        observation = env.perform(*action)
        if observation.state == GameState.GAME_OVER:
            return None
        if env.levels_completed > before:
            return index
    return None


def _greedy_native_reduction(spec, actions, *, ablation=None):
    context = int(spec["context_index"])
    base = Env([build_level(spec) for _ in range(context + 1)])
    base.set_level(context)
    if ablation is not None:
        _apply_relation_ablation(base, spec, ablation)
    route = [tuple(action) for action in actions]
    changed = True
    while changed:
        changed = False
        prefix_engine = _Engine(base)
        _, start = prefix_engine.snapshot()
        prefixes = [start]
        # A deletion start is at most len(route)-1. The final native-positive
        # action therefore never needs to be applied while building prefixes.
        for action in route[:-1]:
            if prefix_engine.click(action[1], action[2]):
                break
            _, prefix = prefix_engine.snapshot()
            prefixes.append(prefix)
        trial = _Engine(base)
        max_width = 8 if int(spec["difficulty"]) == 8 else 3
        for width in range(min(max_width, len(route)), 0, -1):
            if width > len(route):
                continue
            for index in range(len(route) - width, -1, -1):
                candidate = route[:index] + route[index + width:]
                trial.restore(prefixes[index])
                first = None
                for suffix_index, action in enumerate(route[index + width:], 1):
                    if trial.click(action[1], action[2]):
                        first = index + suffix_index
                        break
                if first is not None:
                    route = candidate[:first]
                    changed = True
                    break
            if changed:
                break
    return route


def _tier8_edge_checks(spec, actions, relations, pin_paths=None):
    """Measure indirect tier-8 recursive effects, including off-pin branches.

    The shipped tier-8 pin carrier is independent: its recursive mechanism is
    causal through collisions and a shared-colour off-pin companion. For that
    tier, inspect every native rod edge and accept either a changed completion
    point or a deletion shortcut after edge ablation. This is stored-route
    evidence, not a universal necessity or optimality claim.
    """
    route = [tuple(action) for action in actions]
    checks = []
    pin_paths = {} if pin_paths is None else pin_paths
    for parent, child in sorted(map(tuple, relations["rod_edges"])):
        ablation = ("rod_edge", parent, child)
        first = _first_native_completion(spec, route, ablation=ablation)
        reduced = (route if first != len(route) else
                   _greedy_native_reduction(spec, route, ablation=ablation))
        checks.append({
            "parent": parent,
            "child": child,
            "pins": [list(position) for position, path in sorted(pin_paths.items())
                     if (parent, child) in path],
            "first_completion": first,
            "reduced_actions": len(reduced),
            "essential": first != len(route) or reduced != route,
        })
    return checks


def _dependency_evidence(spec, actions):
    relations = _native_relations(spec)
    if len(relations["pin_edges"]) != len(spec["pins"]):
        return None
    pin_checks = []
    for parent, position in relations["pin_edges"]:
        essential = _first_native_completion(
            spec, actions, ablation=("pin_edge", parent, position)
        ) is None
        pin_checks.append({"parent": parent, "pin": position, "essential": essential})
    parents = {child: parent for parent, child in relations["rod_edges"]}
    relevant_edges = set()
    pin_paths = {}
    for parent, position in relations["pin_edges"]:
        path_edges = []
        child = parent
        while child in parents:
            edge = (parents[child], child)
            relevant_edges.add(edge)
            path_edges.append(edge)
            child = parents[child]
        pin_paths[tuple(position)] = path_edges
    difficulty = spec["difficulty"]
    # Official tier four has a standalone pin carrier. Its linked work is in
    # the four off-pin shared children, each moved by an independent parent to
    # clear a later shared-control collision. Audit those actual native
    # attachments instead of inventing a recursive pin ancestry edge.
    if difficulty == 4:
        pin_path = _pin_path_rods(spec)
        shared_colors = {
            int(rod["color"]) for rod in spec["rods"]
            if rod["name"] in pin_path
        }
        rod_colors = {rod["name"]: int(rod["color"]) for rod in spec["rods"]}
        relevant_edges.update(
            (parent, child) for parent, child in relations["rod_edges"]
            if child not in pin_path and rod_colors[child] in shared_colors
        )
    if difficulty == 8:
        edge_checks = _tier8_edge_checks(spec, actions, relations, pin_paths)
    else:
        edge_checks = []
        for parent, child in sorted(relevant_edges):
            essential = _first_native_completion(
                spec, actions, ablation=("rod_edge", parent, child)
            ) is None
            edge_checks.append({
                "parent": parent,
                "child": child,
                "pins": [list(position) for position, path in sorted(pin_paths.items())
                         if (parent, child) in path],
                "essential": essential,
            })
    color_groups = defaultdict(list)
    for rod in spec["rods"]:
        color_groups[int(rod["color"])].append(rod["name"])
    relevant_rods = {parent for parent, _ in relations["pin_edges"]}
    relevant_rods.update(parent for parent, _ in relevant_edges)
    relevant_rods.update(child for _, child in relevant_edges)
    shared_checks = []
    for group in color_groups.values():
        if len(group) < 2 or not (set(group) & relevant_rods):
            continue
        for rod_name in sorted(set(group) & relevant_rods):
            essential = _first_native_completion(
                spec, actions, ablation=("shared_direct", rod_name, None)
            ) is None
            shared_checks.append({"rod": rod_name, "essential": essential})
    pin_path = _pin_path_rods(spec)
    shared_colors = {int(rod["color"]) for rod in spec["rods"]
                     if rod["name"] in pin_path}
    companion_checks = []
    for rod in spec["rods"]:
        if rod["name"] in pin_path or int(rod["color"]) not in shared_colors:
            continue
        reduced_spec = deepcopy(spec)
        reduced_spec["rods"] = [row for row in reduced_spec["rods"]
                                if row["name"] != rod["name"]]
        reduced_spec["children"] = [edge for edge in reduced_spec["children"]
                                    if rod["name"] not in edge]
        try:
            first = _first_native_completion(reduced_spec, actions)
            reduced_route = _greedy_native_reduction(reduced_spec, actions)
        except (AssertionError, KeyError, TypeError, ValueError, IndexError, RecursionError):
            return None
        companion_checks.append({
            "rod": rod["name"],
            "first_completion": first,
            "reduced_actions": len(reduced_route),
            "essential": first != len(actions) or reduced_route != list(actions),
        })
    stripped = deepcopy(spec)
    removed_rods = sorted(rod["name"] for rod in stripped["rods"]
                          if rod["name"] not in pin_path)
    stripped["rods"] = [rod for rod in stripped["rods"] if rod["name"] in pin_path]
    stripped["children"] = [edge for edge in stripped["children"]
                            if set(edge) <= pin_path]
    removed_obstacles = len(stripped["obstacles"])
    stripped["obstacles"] = []
    try:
        stripped_first = _first_native_completion(stripped, actions)
        stripped_route = _greedy_native_reduction(stripped, actions)
    except (AssertionError, KeyError, TypeError, ValueError, IndexError, RecursionError):
        return None
    constraint_gate = {
        "removed_rods": removed_rods,
        "removed_obstacles": removed_obstacles,
        "first_completion": stripped_first,
        "reduced_actions": len(stripped_route),
        "essential": (stripped_first != len(actions)
                      or stripped_route != list(actions)),
    }
    layout = extract(Env([build_level(dict(spec, targets=[]))]))
    rods_by_color = defaultdict(set)
    for rod in spec["rods"]:
        rods_by_color[int(rod["color"])].add(rod["name"])
    action_roles = {}
    for control in layout.controls:
        controlled = set().union(*(rods_by_color[color] for color in control.colors))
        action_roles[(names.ACTION_CLICK, *control.click)] = {
            "pin": bool(controlled & pin_path),
            "companion": bool(controlled - pin_path),
        }
    constraint_indices = [index for index, action in enumerate(actions)
                          if not action_roles[tuple(action)]["pin"]]
    shared_indices = [index for index, action in enumerate(actions)
                      if action_roles[tuple(action)]["pin"]
                      and action_roles[tuple(action)]["companion"]]
    order_checks = []
    for left in constraint_indices:
        for right in shared_indices:
            if tuple(actions[left]) == tuple(actions[right]):
                continue
            swapped = list(actions)
            swapped[left], swapped[right] = swapped[right], swapped[left]
            first = _first_native_completion(spec, swapped)
            order_checks.append({
                "constraint_index": left,
                "shared_index": right,
                "first_completion": first,
                "essential": first != len(actions),
            })
            if order_checks[-1]["essential"] or len(order_checks) >= 12:
                break
        if order_checks and (order_checks[-1]["essential"] or len(order_checks) >= 12):
            break
    evidence = {
        "native_relations": relations,
        "pin_attachments": pin_checks,
        "recursive_edges": edge_checks,
        "shared_direct_rods": shared_checks,
        "shared_companions": companion_checks,
        "constraint_gate": constraint_gate,
        "constraint_order": order_checks,
        "branch_edges": [],
    }
    if not all(row["essential"] for row in pin_checks):
        return None
    if PROFILES[difficulty].get("require_linked"):
        essential_edges = {(row["parent"], row["child"])
                           for row in edge_checks if row["essential"]}
        if difficulty == 8:
            if not essential_edges:
                return None
        else:
            missing_linked_evidence = (not essential_edges if difficulty == 4
                                       else not edge_checks)
            if (missing_linked_evidence
                    or any(path and not essential_edges.intersection(path)
                           for path in pin_paths.values())):
                return None
    if (PROFILES[difficulty].get("require_shared")
            and not any(row["essential"] for row in companion_checks)):
        return None
    if (PROFILES[difficulty].get("require_constraint")
            and not constraint_gate["essential"]):
        return None
    if (PROFILES[difficulty].get("require_constraint")
            and not any(row["essential"] for row in order_checks)):
        return None
    if PROFILES[difficulty].get("require_branch"):
        child_counts = Counter(parent for parent, _ in relations["rod_edges"])
        evidence["branch_edges"] = [
            row for row in edge_checks if child_counts[row["parent"]] > 1
        ]
        if not any(row["essential"] for row in evidence["branch_edges"]):
            return None
    return evidence


def _collision_witness(spec):
    probe = dict(spec, targets=[{"x": -99, "y": -99}],
                 step_counter=max(64, int(spec["step_counter"])))
    layout = extract(Env([build_level(probe)]))
    extends = [(names.ACTION_CLICK, control.click[0], control.click[1])
               for control in layout.controls if control.kind == "extend"]
    for action in extends:
        engine = _Engine(Env([build_level(probe)]))
        key, _ = engine.snapshot()
        route = []
        changed = 0
        for _ in range(32):
            engine.click(action[1], action[2])
            next_key, _ = engine.snapshot()
            route.append(list(action))
            if next_key == key:
                if changed:
                    return {"actions": route, "rollback_action": list(action),
                            "changed_prefix_actions": changed,
                            "native_state_restored": True}
                break
            changed += 1
            key = next_key
    return None


def _replay_in_context(spec, actions):
    try:
        _, env = _native_episode(spec, actions)
    except (AssertionError, KeyError, TypeError, ValueError, IndexError):
        return False, None
    return env.state != GameState.GAME_OVER, env


# -- structural/profile admission ---------------------------------------------

def structural_metrics(spec):
    rods = list(spec["rods"])
    names_to_index = {rod["name"]: index for index, rod in enumerate(rods)}
    if len(names_to_index) != len(rods):
        raise ValueError("rod names must be unique")
    children = defaultdict(list)
    parents = {}
    for parent, child in spec["children"]:
        if parent not in names_to_index or child not in names_to_index:
            raise ValueError("children edge references an unknown rod")
        if child in parents:
            raise ValueError("a rod cannot have two explicit parents")
        parents[child] = parent
        children[parent].append(child)
    roots = [rod["name"] for rod in rods if rod["name"] not in parents]

    visiting = set()
    visited = set()
    def walk(name):
        if name in visiting:
            raise ValueError("children graph contains a cycle")
        if name in visited:
            return (0, 0)
        visiting.add(name)
        descendants = [walk(child) for child in children[name]]
        visiting.remove(name)
        visited.add(name)
        return (1 + sum(size for size, _ in descendants),
                1 + max((depth for _, depth in descendants), default=0))

    components = [walk(root) for root in roots]
    if len(visited) != len(rods):
        raise ValueError("children graph is not a forest")
    control_colors = {int(row["color"]) for row in spec["rails"] + spec["buttons"]}
    control_colors.update(int(row["secondary_color"]) for row in spec["rails"]
                          if "secondary_color" in row)
    controlled = [rod for rod in rods if int(rod["color"]) in control_colors]
    shared = Counter(int(rod["color"]) for rod in controlled)
    obstacle_cells = {tuple(cell) for obstacle in spec["obstacles"] for cell in obstacle["cells"]}
    obstacle_pixels = {
        (int(x) + dx, int(y) + dy)
        for x, y in obstacle_cells
        for dx in range(UNIT)
        for dy in range(UNIT)
    }
    arena_obstacle_pixels = {
        point for point in obstacle_pixels
        if 0 <= point[0] < names.FRAME and 0 <= point[1] < 41
    }
    interior_obstacle_cells = sum(
        1 for x, y in obstacle_cells
        if 0 <= int(x) and int(x) + UNIT <= names.FRAME
        and 0 <= int(y) and int(y) + UNIT <= 41
    )
    rod_cells = set()
    for rod in rods:
        length, rotation = int(rod["length"]), int(rod["rotation"])
        if rotation == 0:
            base = (int(rod["x"]), int(rod["y"]) + UNIT * (length - 1))
        elif rotation in (90, 180):
            base = (int(rod["x"]), int(rod["y"]))
        elif rotation == 270:
            base = (int(rod["x"]) + UNIT * (length - 1), int(rod["y"]))
        else:
            raise ValueError("invalid rod rotation")
        dx, dy = names.EXTEND_DIRECTION[rotation]
        rod_cells.update((base[0] + dx * UNIT * step, base[1] + dy * UNIT * step)
                         for step in range(length))
    rod_pixels = {
        (int(x) + dx, int(y) + dy)
        for x, y in rod_cells
        for dx in range(UNIT)
        for dy in range(UNIT)
    }
    occupied_pixels = obstacle_pixels | rod_pixels
    occupied_frame_pixels = {
        point for point in occupied_pixels
        if 0 <= point[0] < names.FRAME and 0 <= point[1] < names.FRAME
    }
    orientations = Counter(row["orientation"] for row in spec["rails"])
    return {
        "mechanical_rods": len(rods),
        "controlled_rods": len(controlled),
        "obstacles": len(spec["obstacles"]),
        "component_sizes": sorted(size for size, _ in components),
        "edges": len(spec["children"]),
        "branch": max((len(value) for value in children.values()), default=0),
        "depth": max((depth for _, depth in components), default=1),
        "pins": len(spec["pins"]),
        "targets": len(spec["targets"]),
        "rails": len(spec["rails"]),
        "buttons": len(spec["buttons"]),
        "rail_orientations": dict(sorted(orientations.items())),
        "shared_color_groups": sorted(value for value in shared.values() if value > 1),
        "rod_units": sum(int(rod["length"]) for rod in rods),
        "obstacle_cells": len(obstacle_cells),
        "interior_obstacle_cells": interior_obstacle_cells,
        "obstacle_pixels": len(obstacle_pixels),
        "arena_obstacle_pixels": len(arena_obstacle_pixels),
        "occupied_rod_pixels": len(occupied_pixels),
        "occupied_density": len(occupied_frame_pixels) / (64 * 64),
    }


def _profile_errors(spec, *, require_evidence=True):
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["difficulty must be an integer in 1..8"]
    reference, profile = REFERENCE_CHARACTERIZATION[difficulty], PROFILES[difficulty]
    try:
        metrics = structural_metrics(spec)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return [f"malformed structure: {error}"]
    exact = ["mechanical_rods", "controlled_rods", "obstacles", "edges", "branch", "depth",
             "pins", "rails", "buttons", "rail_orientations"]
    if require_evidence:
        exact.append("targets")
    for key in exact:
        if metrics[key] != reference[key]:
            errors.append(f"{key} differs from official tier")
    if tuple(metrics["component_sizes"]) != tuple(reference["component_sizes"]):
        errors.append("component sizes differ from official tier")
    if tuple(metrics["shared_color_groups"]) != tuple(reference["shared_color_groups"]):
        errors.append("shared-colour grouping differs from official tier")
    calibrated_ranges = ["rod_units"]
    if "interior_obstacle_cells" in profile:
        calibrated_ranges.extend(("interior_obstacle_cells", "obstacle_pixels",
                                  "arena_obstacle_pixels"))
    else:
        calibrated_ranges.extend(("obstacle_cells", "occupied_density"))
    for key in calibrated_ranges:
        if not profile[key][0] <= metrics[key] <= profile[key][1]:
            errors.append(f"{key} outside reference tolerance")
    if int(spec.get("step_counter", -1)) != reference["budget"]:
        errors.append("native budget differs from official tier")
    if require_evidence:
        actions = spec.get("solution", ())
        if len(actions) < profile["action_range"][0]:
            errors.append("constructive action length is below the reference-calibrated tier band")
        if len(actions) > profile["action_range"][1]:
            errors.append("constructive action length exceeds tier tolerance")
        if len(actions) < profile["shortcut_bound"]:
            errors.append("certificate is below the independent dependency bound")
        mechanics = spec.get("solution_mechanics", {})
        for kind in profile["required_kinds"]:
            if mechanics.get(kind, 0) < 1:
                errors.append(f"winning route does not exercise {kind}")
        if mechanics.get("distinct_controls", 0) < profile["distinct_controls"]:
            errors.append("winning route uses too few distinct controls")
        for required, counter in (("require_vertical", "vertical_rail_actions"),
                                  ("require_linked", "linked_actions"),
                                  ("require_shared", "shared_color_actions"),
                                  ("require_branch", "branch_actions")):
            if profile.get(required) and mechanics.get(counter, 0) < 1:
                errors.append(f"winning route lacks {counter}")
        if mechanics.get("pin_motion_actions", 0) < 1:
            errors.append("winning route never moves a pin")
        if mechanics.get("all_actions_changed") is not True:
            errors.append("winning route contains padding/no-op actions")
        if mechanics.get("all_actions_relevant") is not True:
            errors.append("winning route contains actions unrelated to pins or constraints")
        if (profile.get("require_constraint")
                and mechanics.get("constraint_motion_actions", 0) < 1):
            errors.append("winning route does not exercise an off-pin constraint")
        if (not profile.get("require_constraint")
                and mechanics.get("all_actions_pin_relevant") is not True):
            errors.append("winning route contains actions unrelated to target pin paths")
        if (not isinstance(mechanics.get("per_pin_motion_actions"), list)
                or len(mechanics["per_pin_motion_actions"]) != reference["pins"]
                or any(type(value) is not int or value < 1
                       for value in mechanics["per_pin_motion_actions"])):
            errors.append("winning route does not move every required pin")
        if mechanics.get("first_completion_action") != len(actions):
            errors.append("native completion is not exactly the final action")
        collision = spec.get("collision_rollback", {})
        if not collision.get("native_state_restored") or collision.get("changed_prefix_actions", 0) < 1:
            errors.append("native collision rollback witness is missing")
    expected_rail_style = "large" if difficulty in (1, 2, 3, 4, 6) else "compact"
    expected_button_style = "large" if difficulty in (6, 8) else "compact"
    if any(row.get("style") != expected_rail_style for row in spec["rails"]):
        errors.append("rail presentation style differs from official tier")
    if any(row.get("style") != expected_button_style for row in spec["buttons"]):
        errors.append("button presentation style differs from official tier")
    if tuple(row.get("color") for row in spec["obstacles"]) != OBSTACLE_PALETTES[difficulty]:
        errors.append("obstacle palette differs from official tier")
    return errors


def _schema_errors(spec):
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    errors = []
    string_fields = (
        "format", "mechanics_inventory_version", "quality_profile_version",
        "geometry_version", "source", "split", "certificate_type",
        "geometry_sha256", "geometry_d4_sha256", "geometry_partition",
        "gameplay_sha256", "solution_semantic_sha256",
    )
    integer_fields = (
        "generator_version", "difficulty", "reference_level", "context_index",
        "step_counter", "seed", "generation_attempt", "solution_length",
        "verification_work", "verification_limit",
    )
    list_fields = ("obstacles", "rods", "pins", "targets", "rails", "buttons",
                   "children", "solution")
    mapping_fields = ("solution_mechanics", "collision_rollback", "dependency_evidence",
                      "shortcut_evidence", "native_relations", "proof",
                      "generation_exclusions")
    for key in string_fields:
        if type(spec.get(key)) is not str:
            errors.append(f"{key} must be a string")
    for key in integer_fields:
        if type(spec.get(key)) is not int:
            errors.append(f"{key} must be an integer")
    for key in list_fields:
        if type(spec.get(key)) is not list:
            errors.append(f"{key} must be a list")
    for key in mapping_fields:
        if not isinstance(spec.get(key), Mapping):
            errors.append(f"{key} must be a mapping")
    for key in ("optimality_claim", "shortest_search_performed"):
        if type(spec.get(key)) is not bool:
            errors.append(f"{key} must be a boolean")
    if errors:
        return errors
    if spec["seed"] < 0 or spec["step_counter"] < 1:
        errors.append("seed and native budget must be positive-domain integers")
    if spec["reference_level"] != spec["difficulty"]:
        errors.append("reference_level must equal difficulty")
    if spec["certificate_type"] != "constructive_native_witness":
        errors.append("certificate_type must be constructive_native_witness")
    if spec["solution_length"] != len(spec["solution"]):
        errors.append("solution_length does not match solution")
    if spec["optimality_claim"] is not False or spec["shortest_search_performed"] is not False:
        errors.append("generated certificate must make no optimality claim")
    for index, action in enumerate(spec["solution"]):
        if (type(action) is not list or len(action) != 3
                or any(type(value) is not int for value in action)):
            errors.append(f"solution action {index} must be three exact integers")
        elif action[0] != names.ACTION_CLICK or not all(0 <= value < names.FRAME
                                                        for value in action[1:]):
            errors.append(f"solution action {index} is outside the action contract")
    for key in ("obstacles", "rods", "pins", "targets", "rails", "buttons"):
        if any(not isinstance(row, Mapping) for row in spec[key]):
            errors.append(f"{key} entries must be mappings")
    if any(type(edge) is not list or len(edge) != 2 for edge in spec["children"]):
        errors.append("children entries must be two-item lists")
    if errors:
        return errors

    def exact_row(row, fields, label):
        for field, expected_type in fields.items():
            if type(row.get(field)) is not expected_type:
                errors.append(f"{label} {field} has the wrong primitive type")

    for row in spec["obstacles"]:
        exact_row(row, {"name": str, "cells": list, "color": int,
                        "ink_stride": int}, "obstacle")
        if set(row) != {"name", "cells", "color", "ink_stride"}:
            errors.append("obstacle entries must use only canonical schema fields")
        cells = row.get("cells")
        if type(cells) is list and (not cells or any(
            type(cell) is not list or len(cell) != 2
            or any(type(value) is not int for value in cell)
            for cell in cells
        )):
            errors.append("obstacle cells must be nonempty integer coordinate pairs")
        if type(row.get("ink_stride")) is int and row["ink_stride"] != 1:
            errors.append("obstacle ink_stride must be exactly 1")
    obstacle_rows_valid = all(
        type(row.get("name")) is str
        and type(row.get("cells")) is list
        and row["cells"]
        and all(
            type(cell) is list and len(cell) == 2
            and all(type(value) is int for value in cell)
            for cell in row["cells"]
        )
        for row in spec["obstacles"]
    )
    difficulty = spec.get("difficulty")
    if obstacle_rows_valid and type(difficulty) is int and difficulty in DIFFICULTIES:
        boundary = _canonical_boundary_cells(difficulty)
        first_interior = 0
        if boundary is not None:
            first_interior = 1
            if (not spec["obstacles"]
                    or spec["obstacles"][0]["name"] != "boundary0"
                    or spec["obstacles"][0]["cells"] != boundary):
                errors.append("boundary0 cells must exactly match the native canonical boundary")
        expected_names = (
            (["boundary0"] + [f"obstacle{index}"
                              for index in range(1, len(spec["obstacles"]))])
            if boundary is not None else
            [f"obstacle{index}" for index in range(len(spec["obstacles"]))]
        )
        if [row["name"] for row in spec["obstacles"]] != expected_names:
            errors.append("obstacle names/order differ from the canonical generated schema")

        occupied_pixels = set()
        if boundary is not None and spec["obstacles"]:
            occupied_pixels.update(
                (x + dx, y + dy)
                for x, y in boundary
                for dx in range(UNIT)
                for dy in range(UNIT)
            )
        seen_cells = set()
        for row in spec["obstacles"][first_interior:]:
            for raw_cell in row["cells"]:
                x, y = raw_cell
                cell = (x, y)
                if cell in seen_cells:
                    errors.append("interior obstacle cells must not be duplicated")
                seen_cells.add(cell)
                if (x % UNIT or y % UNIT
                        or not 0 <= x <= names.FRAME - UNIT - 1
                        or not 0 <= y <= 36):
                    errors.append(
                        "interior obstacle cells must use the generated 3-pixel arena lattice"
                    )
                pixels = {
                    (x + dx, y + dy)
                    for dx in range(UNIT)
                    for dy in range(UNIT)
                }
                if occupied_pixels & pixels:
                    errors.append("obstacle cells must not encode overlapping collision pixels")
                occupied_pixels.update(pixels)
    for row in spec["rods"]:
        exact_row(row, {"name": str, "color": int, "x": int, "y": int,
                        "rotation": int, "length": int}, "rod")
        if type(row.get("rotation")) is int and row["rotation"] not in (0, 90, 180, 270):
            errors.append("rod rotation is outside the native quarter turns")
        if type(row.get("length")) is int and not 1 <= row["length"] <= 21:
            errors.append("rod length is outside native board bounds")
    for label in ("pins", "targets"):
        for row in spec[label]:
            exact_row(row, {"x": int, "y": int}, label[:-1])
    for row in spec["rails"]:
        exact_row(row, {"name": str, "color": int, "x": int, "y": int,
                        "orientation": str, "style": str}, "rail")
        if "secondary_color" in row and type(row["secondary_color"]) is not int:
            errors.append("rail secondary_color must be an integer")
        if row.get("orientation") not in ("horizontal", "vertical"):
            errors.append("rail orientation is invalid")
        if row.get("style") not in ("compact", "large"):
            errors.append("rail style is invalid")
    for row in spec["buttons"]:
        exact_row(row, {"name": str, "color": int, "x": int, "y": int,
                        "style": str}, "button")
        if row.get("style") not in ("compact", "large"):
            errors.append("button style is invalid")
    if errors:
        return errors

    rod_names = [row["name"] for row in spec["rods"]]
    if len(set(rod_names)) != len(rod_names):
        errors.append("rod names must be unique")
    known_rods = set(rod_names)
    parents = {}
    graph = defaultdict(list)
    for edge in spec["children"]:
        if any(type(value) is not str for value in edge):
            errors.append("children endpoints must be strings")
            continue
        parent, child = edge
        if parent not in known_rods or child not in known_rods:
            errors.append("children edge references an unknown rod")
        if parent == child:
            errors.append("children graph contains a self cycle")
        if child in parents:
            errors.append("a rod cannot have two explicit parents")
        parents[child] = parent
        graph[parent].append(child)
    visiting, visited = set(), set()
    def visit(name):
        if name in visiting:
            return False
        if name in visited:
            return True
        visiting.add(name)
        valid = all(visit(child) for child in graph[name])
        visiting.remove(name)
        visited.add(name)
        return valid
    if not all(visit(name) for name in rod_names):
        errors.append("children graph contains a cycle")
    if errors:
        return errors
    relations = spec["native_relations"]
    if type(relations.get("rod_edges")) is not list or type(relations.get("pin_edges")) is not list:
        errors.append("native_relations must contain rod_edges and pin_edges lists")
    proof = spec["proof"]
    proof_types = {
        "seed": int, "difficulty": int, "context_index": int, "split": str,
        "geometry_d4_sha256": str, "gameplay_sha256": str,
        "solution_semantic_sha256": str, "generator_version": int,
        "mechanics_inventory_version": str, "quality_profile_version": str,
        "certificate_type": str, "optimality_claim": bool,
        "native_context_replayed": bool, "engine_win": bool,
        "budget_start": int, "budget_actions": int, "budget_remaining": int,
        "verification_work": int, "verification_limit": int,
    }
    for key, expected_type in proof_types.items():
        if type(proof.get(key)) is not expected_type:
            errors.append(f"proof {key} has the wrong primitive type")
    for key in ("optimality_claim",):
        if proof.get(key) is not False:
            errors.append(f"proof {key} must be exactly false")
    for key in ("native_context_replayed", "engine_win"):
        if proof.get(key) is not True:
            errors.append(f"proof {key} must be exactly true")
    mechanics = spec["solution_mechanics"]
    for key in ("all_actions_changed", "all_actions_pin_relevant",
                "all_actions_relevant", "won"):
        if type(mechanics.get(key)) is not bool:
            errors.append(f"solution_mechanics {key} must be a boolean")
    mechanic_counts = (
        "changed_actions", "distinct_controls", "max_rods_changed_by_one_action",
        "pin_motion_actions", "pin_path_actions", "first_completion_action",
        "native_budget_start", "native_budget_consumed", "native_budget_remaining",
    )
    for key in mechanic_counts:
        if type(mechanics.get(key)) is not int or mechanics.get(key, -1) < 0:
            errors.append(f"solution_mechanics {key} must be a nonnegative integer")
    for key in ("extend", "retract", "rotate", "vertical_rail_actions",
                "linked_actions", "shared_color_actions", "branch_actions",
                "constraint_motion_actions", "unchanged_actions"):
        if key in mechanics and (type(mechanics[key]) is not int or mechanics[key] < 0):
            errors.append(f"solution_mechanics {key} must be a nonnegative integer when present")
    if type(mechanics.get("per_pin_motion_actions")) is not list:
        errors.append("solution_mechanics per_pin_motion_actions must be a list")
    elif any(type(value) is not int or value < 0
             for value in mechanics["per_pin_motion_actions"]):
        errors.append("solution_mechanics per_pin_motion_actions entries must be nonnegative integers")
    collision = spec["collision_rollback"]
    if (type(collision.get("actions")) is not list
            or type(collision.get("rollback_action")) is not list
            or type(collision.get("changed_prefix_actions")) is not int
            or type(collision.get("native_state_restored")) is not bool):
        errors.append("collision rollback evidence has invalid container or primitive types")
    shortcut = spec["shortcut_evidence"]
    if (type(shortcut.get("optimality_claim")) is not bool
            or shortcut.get("optimality_claim") is not False):
        errors.append("shortcut evidence optimality claim must be exactly false")
    for key in ("greedy_fixed_point_actions", "bound", "compact_nodes"):
        if type(shortcut.get(key)) is not int or shortcut.get(key, -1) < 0:
            errors.append(f"shortcut evidence {key} must be a nonnegative integer")
    if type(shortcut.get("compact_outcome")) is not str:
        errors.append("shortcut evidence compact_outcome must be a string")
    if any(type(value) is not int or value < 0
           for value in spec["generation_exclusions"].values()):
        errors.append("generation exclusion counters must be nonnegative integers")
    for section in ("pin_attachments", "recursive_edges", "shared_direct_rods",
                    "shared_companions", "constraint_order", "branch_edges"):
        rows = spec["dependency_evidence"].get(section)
        if type(rows) is not list:
            errors.append(f"dependency evidence {section} must be a list")
        elif (any(not isinstance(row, Mapping) for row in rows)
              or any(type(row.get("essential")) is not bool for row in rows)):
            errors.append(f"dependency evidence {section} entries must be mappings with boolean essentials")
    gate = spec["dependency_evidence"].get("constraint_gate")
    if not isinstance(gate, Mapping) or type(gate.get("essential")) is not bool:
        errors.append("dependency evidence constraint_gate must contain a boolean essential")
    return errors


def validate_full_standard(spec, curriculum_entry):
    """Recompute family-local structure, identities and native positive proofs."""
    errors = _schema_errors(spec)
    if errors:
        return errors
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be a mapping"]
    expected_versions = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "source": "generated_only",
    }
    for key, expected in expected_versions.items():
        if type(spec.get(key)) is not type(expected) or spec.get(key) != expected:
            errors.append(f"{key} mismatch")
    difficulty = spec.get("difficulty")
    try:
        expected_curriculum = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        if dict(curriculum_entry) != expected_curriculum:
            errors.append("curriculum entry differs from contract")
        if (type(spec.get("context_index")) is not int
                or spec.get("context_index") != expected_curriculum["context_index"]):
            errors.append("native context differs from curriculum")
        if not 1 <= int(spec.get("verification_work", 0)) <= expected_curriculum["search_work"]:
            errors.append("verification work is outside the curriculum bound")
        if not 1 <= int(spec.get("verification_limit", 0)) <= expected_curriculum["search_work"]:
            errors.append("verification limit is outside the curriculum bound")
        if int(spec.get("verification_work", 0)) > int(spec.get("verification_limit", 0)):
            errors.append("verification work exceeds its declared limit")
    except (IndexError, KeyError, TypeError, ValueError):
        errors.append("invalid difficulty/curriculum relation")
    if spec.get("split") not in SPLITS:
        errors.append("explicit split missing")
    errors.extend(_profile_errors(spec))
    if errors:
        return errors
    try:
        official_match = official_copy_match(Env([build_level(spec)]))
        if official_match is not None:
            official_index, evidence_kind = official_match
            errors.append(
                f"official {evidence_kind} copy shipped tier {official_index + 1}"
            )
    except (AssertionError, KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"official-copy identity failed: {error}")
    if errors:
        return errors
    try:
        raw_identity = geometry_hash(spec)
        d4_identity, partition = identity_partition(spec)
        play_identity = gameplay_hash(spec)
        route_identity = solution_semantic_hash(spec)
        if spec.get("geometry_sha256") != raw_identity:
            errors.append("raw geometry identity mismatch")
        if spec.get("geometry_d4_sha256") != d4_identity:
            errors.append("D4 geometry identity mismatch")
        if spec.get("gameplay_sha256") != play_identity:
            errors.append("gameplay identity mismatch")
        if spec.get("solution_semantic_sha256") != route_identity:
            errors.append("solution semantic identity mismatch")
        if partition != spec.get("split") or spec.get("geometry_partition") != partition:
            errors.append("geometry belongs to another canonical split")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"identity recomputation failed: {error}")
    try:
        actions = solution_actions(spec)
        mechanics = _solution_mechanics(spec, actions)
        if mechanics != spec.get("solution_mechanics"):
            errors.append("solution mechanic evidence mismatch")
        relations = _native_relations(spec)
        if relations != spec.get("native_relations"):
            errors.append("native attachment relation evidence mismatch")
        reduced = _greedy_native_reduction(spec, actions)
        if reduced != actions:
            errors.append("winning route has a native-positive deletion shortcut")
        dependency = _dependency_evidence(spec, actions)
        if dependency is None or dependency != spec.get("dependency_evidence"):
            errors.append("relation dependency evidence mismatch")
        shortcut_bound = PROFILES[difficulty]["shortcut_bound"]
        shortcut_env = Env([build_level(spec) for _ in range(difficulty)])
        shortcut_env.set_level(difficulty - 1)
        shortcut = search(shortcut_env, limit=shortcut_bound - 1, max_nodes=4_000)
        expected_shortcut = {
            "greedy_fixed_point_actions": len(reduced),
            "bound": shortcut_bound,
            "compact_nodes": shortcut.nodes,
            "compact_outcome": ("exhaustive_clear" if shortcut.exact and not shortcut.truncated
                                else "bounded_unknown"),
            "optimality_claim": False,
        }
        if shortcut.actions is not None:
            errors.append("independent compact search found a short solution")
        if expected_shortcut != spec.get("shortcut_evidence"):
            errors.append("shortcut search evidence mismatch")
        completed, env = _replay_in_context(spec, actions)
        if not completed or env.state == GameState.GAME_OVER:
            errors.append("winning route failed native context replay")
        collision = _collision_witness(spec)
        if collision != spec.get("collision_rollback"):
            errors.append("collision rollback evidence mismatch")
    except (AssertionError, KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"native replay failed: {error}")
    proof = spec["proof"]
    action_count = len(spec["solution"])
    mirrors = {
        "seed": spec.get("seed"), "difficulty": difficulty,
        "context_index": spec.get("context_index"), "split": spec.get("split"),
        "geometry_d4_sha256": spec.get("geometry_d4_sha256"),
        "gameplay_sha256": spec.get("gameplay_sha256"),
        "solution_semantic_sha256": spec.get("solution_semantic_sha256"),
        "generator_version": GENERATOR_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "certificate_type": "constructive_native_witness",
        "optimality_claim": False,
        "native_context_replayed": True,
        "engine_win": True,
        "budget_start": spec.get("step_counter"),
        "verification_work": spec.get("verification_work"),
        "verification_limit": spec.get("verification_limit"),
        "budget_actions": spec["solution_mechanics"]["native_budget_consumed"],
        "budget_remaining": spec["solution_mechanics"]["native_budget_remaining"],
    }
    for key, value in mirrors.items():
        if type(proof.get(key)) is not type(value) or proof.get(key) != value:
            errors.append(f"proof {key} mismatch")
    if action_count > int(spec.get("step_counter", -1)):
        errors.append("winning route exceeds the native budget")
    return errors


# -- public generation modes ---------------------------------------------------

def _reject(exclusions, reason, *, seed, difficulty, attempt, record_rejection):
    exclusions[reason] += 1
    if record_rejection is not None:
        record_rejection({
            "seed": int(seed), "difficulty": int(difficulty), "attempt": int(attempt),
            "reason": reason, "generator_version": GENERATOR_VERSION,
            "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        })


def generate(seed, difficulty, attempts=DEFAULT_ATTEMPTS, *, split,
             max_transitions=None, record_rejection=None):
    """Generate one full-standard tier in an explicit canonical split."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be an integer in 1..8")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    if type(attempts) is not int or not 1 <= attempts <= MAX_ATTEMPTS:
        raise ValueError(f"attempts must be an integer in 1..{MAX_ATTEMPTS}")
    work_limit = PROFILES[difficulty]["search_work"] if max_transitions is None else max_transitions
    if type(work_limit) is not int or not 1 <= work_limit <= PROFILES[difficulty]["search_work"]:
        raise ValueError("max_transitions must be a positive integer within the tier contract")

    rng = random.Random(f"{MECHANICS_INVENTORY_VERSION}:{int(seed)}:{difficulty}")
    exclusions = Counter()
    verification_work = 0
    for attempt in range(1, attempts + 1):
        spec, reason = _draft(rng, difficulty)
        if spec is None:
            _reject(exclusions, reason, seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        spec.update(seed=int(seed), split=split, generation_attempt=attempt,
                    name=f"s5i5-full-d{difficulty}-{int(seed)}")
        draft_errors = _profile_errors(spec, require_evidence=False)
        if draft_errors:
            _reject(exclusions, "profile_structure:" + "|".join(sorted(draft_errors)),
                    seed=seed, difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            continue
        witness, reason, work = _construct_witness(
            rng, spec, work_limit - verification_work
        )
        verification_work += work
        if witness is None:
            _reject(exclusions, reason, seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            if reason == "verification_work_exhausted":
                break
            continue
        actions, targets = witness
        spec["targets"] = targets
        spec["native_relations"] = _native_relations(spec)
        official_match = official_copy_match(Env([build_level(spec)]))
        if official_match is not None:
            official_index, evidence_kind = official_match
            _reject(
                exclusions,
                f"official_{evidence_kind}_copy:tier{official_index + 1}",
                seed=seed,
                difficulty=difficulty,
                attempt=attempt,
                record_rejection=record_rejection,
            )
            continue
        if Env([build_level(spec)]).is_won_position():
            _reject(exclusions, "already_solved", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        reduced = _greedy_native_reduction(spec, actions)
        shortcut_bound = PROFILES[difficulty]["shortcut_bound"]
        if len(reduced) != len(actions) or len(reduced) < shortcut_bound:
            _reject(exclusions, "native_action_deletion_shortcut", seed=seed,
                    difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            continue
        dependency = _dependency_evidence(spec, actions)
        if dependency is None:
            _reject(exclusions, "relation_dependency", seed=seed,
                    difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            continue
        try:
            mechanics = _solution_mechanics(spec, actions)
            collision = _collision_witness(spec)
        except (AssertionError, KeyError, TypeError, ValueError, IndexError):
            mechanics, collision = {}, None
        if collision is None:
            _reject(exclusions, "collision_witness", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        spec.update(
            solution=[list(action) for action in actions],
            solution_length=len(actions),
            solution_mechanics=mechanics,
            collision_rollback=collision,
            certificate_type="constructive_native_witness",
            optimality_claim=False,
            shortest_search_performed=False,
            dependency_evidence=dependency,
            verification_work=verification_work,
            verification_limit=work_limit,
            generation_exclusions=dict(exclusions),
        )
        errors = _profile_errors(spec)
        if errors:
            _reject(exclusions, "profile_evidence:" + "|".join(sorted(errors)),
                    seed=seed, difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            continue
        raw_identity = geometry_hash(spec)
        d4_identity, partition = identity_partition(spec)
        if partition != split:
            _reject(exclusions, "geometry_split", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        spec.update(
            geometry_sha256=raw_identity,
            geometry_d4_sha256=d4_identity,
            geometry_partition=partition,
        )
        spec["gameplay_sha256"] = gameplay_hash(spec)
        spec["solution_semantic_sha256"] = solution_semantic_hash(spec)
        shortcut_env = Env([build_level(spec) for _ in range(difficulty)])
        shortcut_env.set_level(difficulty - 1)
        shortcut = search(shortcut_env, limit=shortcut_bound - 1, max_nodes=4_000)
        verification_work += shortcut.nodes
        if verification_work > work_limit:
            _reject(exclusions, "verification_work_exhausted", seed=seed,
                    difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            break
        if shortcut.actions is not None:
            _reject(exclusions, "independent_compact_shortcut", seed=seed,
                    difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            continue
        spec["shortcut_evidence"] = {
            "greedy_fixed_point_actions": len(reduced),
            "bound": shortcut_bound,
            "compact_nodes": shortcut.nodes,
            "compact_outcome": ("exhaustive_clear" if shortcut.exact and not shortcut.truncated
                                else "bounded_unknown"),
            "optimality_claim": False,
        }
        spec["verification_work"] = verification_work
        completed, env = _replay_in_context(spec, actions)
        if not completed or env.state == GameState.GAME_OVER:
            _reject(exclusions, "context_native_replay", seed=seed,
                    difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            continue
        spec["proof"] = {
            "seed": int(seed), "difficulty": difficulty,
            "context_index": difficulty - 1, "split": split,
            "geometry_d4_sha256": d4_identity,
            "gameplay_sha256": spec["gameplay_sha256"],
            "solution_semantic_sha256": spec["solution_semantic_sha256"],
            "generator_version": GENERATOR_VERSION,
            "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
            "quality_profile_version": QUALITY_PROFILE_VERSION,
            "certificate_type": "constructive_native_witness",
            "optimality_claim": False,
            "native_context_replayed": True,
            "engine_win": True,
            "budget_start": mechanics["native_budget_start"],
            "budget_actions": mechanics["native_budget_consumed"],
            "budget_remaining": mechanics["native_budget_remaining"],
            "verification_work": verification_work,
            "verification_limit": work_limit,
        }
        spec["generation_exclusions"] = dict(exclusions)
        validation = validate_full_standard(
            spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        )
        if validation:
            _reject(exclusions, "validator:" + "|".join(sorted(set(validation))),
                    seed=seed, difficulty=difficulty, attempt=attempt,
                    record_rejection=record_rejection)
            continue
        spec["generation_exclusions"] = dict(exclusions)
        return spec
    return None


def _child_seed(game_seed, ordinal, difficulty):
    if isinstance(game_seed, bool) or not isinstance(game_seed, Integral) or game_seed < 0:
        raise ValueError("game seed must be a nonnegative integer")
    material = f"{SOURCE_ID}:{int(game_seed)}:{int(ordinal)}:{int(difficulty)}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split, difficulties=None, attempts=DEFAULT_ATTEMPTS,
                  max_transitions=None, record_rejection=None):
    """Generate all eight increasing tiers, or an explicit smoke subset."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("game seed must be a nonnegative integer")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if (not selected or any(type(value) is not int or value not in DIFFICULTIES for value in selected)
            or tuple(sorted(set(selected))) != selected):
        raise ValueError("difficulties must be a strictly increasing unique subset of 1..8")
    specs = []
    for ordinal, difficulty in enumerate(selected):
        child_seed = _child_seed(seed, ordinal, difficulty)
        spec = generate(child_seed, difficulty, attempts=attempts, split=split,
                        max_transitions=max_transitions,
                        record_rejection=record_rejection)
        if spec is None:
            return None
        spec.update(game_seed=int(seed), game_ordinal=ordinal,
                    game_child_seed=child_seed)
        specs.append(spec)
    return specs


def build_game(specs):
    """Validate and build exactly one ordered eight-context native game."""
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
        raise ValueError("specs must be a sequence")
    specs = list(specs)
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("full S5I5 games require exactly eight specs")
    if any(not isinstance(spec, Mapping) for spec in specs):
        raise ValueError("every game spec must be a mapping")
    if tuple(spec.get("difficulty") for spec in specs) != DIFFICULTIES:
        raise ValueError("full S5I5 games require difficulties 1..8 in order")
    if tuple(spec.get("context_index") for spec in specs) != tuple(range(8)):
        raise ValueError("full S5I5 games must preserve native contexts 0..7")
    splits = {spec.get("split") for spec in specs}
    if len(splits) != 1 or next(iter(splits), None) not in SPLITS:
        raise ValueError("full S5I5 games require one common explicit split")
    for identity_key in ("geometry_d4_sha256", "gameplay_sha256"):
        identities = [spec.get(identity_key) for spec in specs]
        if any(not value for value in identities) or len(set(identities)) != len(identities):
            raise ValueError(f"full S5I5 games require distinct {identity_key} values")
    for index, spec in enumerate(specs):
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"tier {index + 1} is not full-standard: {errors[0]}")
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            raise ValueError("native game shifted away from the declared context sequence")
        before = env.levels_completed
        actions = solution_actions(spec)
        for action_index, action in enumerate(actions, 1):
            observation = env.perform(*action)
            if observation.state == GameState.GAME_OVER:
                raise ValueError(f"native sequential replay lost at context {index}")
            if env.levels_completed > before:
                if action_index != len(actions):
                    raise ValueError(f"native context {index} completed before its final action")
                break
        if env.levels_completed != index + 1:
            raise ValueError(f"native sequential replay failed at context {index}")
    if env.state != GameState.WIN:
        raise ValueError("native eight-tier replay did not finish in WIN")
    return levels
