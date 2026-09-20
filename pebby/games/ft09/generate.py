"""Full, reference-calibrated procedural generator for all six FT09 tiers.

Drafts use new role geometry and assignments. A draft is accepted only when
an untruncated exact search lands inside its official-tier action tolerance,
the required tier mechanics participate in that solution, and the actions win
in the real engine at the intended native level index. Search cutoffs are
rejections, never impossibility claims or positive certificates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import random

from arcengine import Level, Sprite

from . import names
from .env import Env, replay
from .generation_quality import SPLITS, gameplay_hash, geometry_hash, identity_partition
from .layout import extract
from .plan import search, verify
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    MECHANICS_INVENTORY_VERSION,
    PROFILES,
    profile_errors,
    structural_metrics,
)

FORMAT = "pebby.ft09.level.v3"
GENERATOR_VERSION = 3
QUALITY_PROFILE_VERSION = DIFFICULTY_VERSION
MAX_SEARCH_WORK = 2_000_000
SOURCE_ID = "ft09-0d8bbf25"

# The shipped game establishes these six vivid colours as readable against its
# charcoal background and its white/grey rule flags and magenta stencil marks.
# Sampling their order still provides 30 two-colour and 120 three-colour cycles.
COLOURS = (8, 9, 11, 12, 14, 15)
MISMATCH_PIXELS = (2, 3)
TUTORIAL_PANEL_POSITIONS = ((1, 1), (18, 1), (1, 18))

# Tier layouts deliberately vary around the one scarce reference. The exact
# acceptance ranges remain centralized in reference_profiles.py.
_DRAFT = {
    1: dict(cells=(7, 8, 9), constraints=1,
            canvas=((3, 3), (3, 4), (4, 3), (4, 4)), witness=(3, 6)),
    2: dict(cells=(12, 13), constraints=2, canvas=((3, 5), (5, 3)), witness=(5, 9)),
    3: dict(cells=(22, 23, 24), constraints=4, canvas=((5, 7), (7, 5)), witness=(11, 17)),
    4: dict(cells=(17, 18, 19), constraints=3, canvas=((5, 5),), witness=(13, 20)),
    5: dict(cells=(29, 30, 31), constraints=8, canvas=((7, 7),), witness=(18, 24)),
    6: dict(cells=(21, 22, 23), constraints=4, canvas=((7, 6), (6, 7)), witness=(11, 16)),
}

SUPPORTED_MECHANICS = (
    "click-only native display action 6 and free empty/rule no-ops",
    "match and differ constraints over eight neighbouring slots",
    "native per-tier click budgets and loss behavior",
    "two-colour and three-colour cyclic cell states",
    "ordinary centre-only cells",
    "mixed ordinary and cross-stencil special cells",
    "all-special boards with directional per-cell stencils",
    "tier-1 invalid-click tutorial flash animation",
    "sequential native level advancement",
)

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": [
        {"difficulty": difficulty, "context_index": PROFILES[difficulty]["context_index"],
         "search_work": PROFILES[difficulty]["search_work"]}
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "ft09-full-reference-v1#official-characterization",
        "solution_mechanics": "ft09-full-reference-v1#mechanic-participation",
        "native_budget": "ft09-full-reference-v1#native-budgets",
        "context_engine_replay": "ft09-full-reference-v1#context-and-sequential-replay",
        "novelty_split": "ft09-full-reference-v1#novelty-and-identities",
        "bounded_rejections": "ft09-full-reference-v1#bounded-quality-audit",
    },
    "caveats": [
        "One shipped level exists per tier; tolerances are explicit design bounds, not population intervals."
    ],
}


def _in_bounds(value, bounds):
    return bounds[0] <= value <= bounds[1]


def _slot_neighbours(slot, width, height):
    x, y = slot
    return {
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0) and 0 <= x + dx < width and 0 <= y + dy < height
    }


def _role_geometry(rng, difficulty):
    """Sample cell/rule slots with the reference span and rule-edge density."""
    settings = _DRAFT[difficulty]
    edge_bounds = PROFILES[difficulty]["rule_edges"]
    for _ in range(600):
        width, height = rng.choice(settings["canvas"])
        n_cells = rng.choice(settings["cells"])
        interior = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
        if len(interior) < settings["constraints"]:
            continue
        rules = set(rng.sample(interior, settings["constraints"]))
        neighbour_candidates = set().union(*(_slot_neighbours(rule, width, height) for rule in rules)) - rules
        candidates = sorted(
            ({(x, y) for x in range(width) for y in range(height)} - rules)
            if difficulty == 1 else neighbour_candidates
        )
        if len(candidates) < n_cells:
            continue
        for _ in range(100):
            cells = set(rng.sample(candidates, n_cells))
            anchors = cells | rules
            if not ({0, width - 1} <= {x for x, _ in anchors}
                    and {0, height - 1} <= {y for _, y in anchors}):
                continue
            degrees = {cell: sum(cell in _slot_neighbours(rule, width, height) for rule in rules)
                       for cell in cells}
            if difficulty != 1 and not all(degrees.values()):
                continue
            active_edges = sum(degrees.values())
            if not _in_bounds(active_edges, edge_bounds):
                continue
            if any(not (cells & _slot_neighbours(rule, width, height)) for rule in rules):
                continue
            occupied = cells | rules
            reached = {next(iter(occupied))}
            while True:
                expanded = reached | {
                    point for point in occupied
                    if any(max(abs(point[0] - q[0]), abs(point[1] - q[1])) == 1 for q in reached)
                }
                if expanded == reached:
                    break
                reached = expanded
            if reached == occupied:
                return sorted(cells), sorted(rules)
    return None


def _mask_with(*offsets):
    mask = [[0, 0, 0] for _ in range(3)]
    mask[1][1] = 1
    for dx, dy in offsets:
        mask[dy + 1][dx + 1] = 1
    return mask


def _witness_and_specials(rng, difficulty, cells):
    n = len(cells)
    cell_set = set(cells)
    special_masks = {}
    special_indices = set()
    if difficulty == 5:
        live = [i for i, (x, y) in enumerate(cells)
                if any((x + dx, y + dy) in cell_set
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        if len(live) < 3:
            return None
        special_indices = set(rng.sample(live, 3))
        for index in special_indices:
            special_masks[index] = _mask_with((-1, 0), (1, 0), (0, -1), (0, 1))
    elif difficulty == 6:
        special_indices = set(range(n))
        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
        live_counts = [(sum((x + dx, y + dy) in cell_set for x, y in cells), (dx, dy))
                       for dx, dy in directions]
        best = max(value for value, _ in live_counts)
        direction = rng.choice([direction for value, direction in live_counts if value == best])
        for index in special_indices:
            special_masks[index] = _mask_with(direction)

    if difficulty == 4:
        target_actions = rng.randint(*_DRAFT[difficulty]["witness"])
        twos = rng.randint(3, min(6, target_actions // 2))
        ones = target_actions - 2 * twos
        if ones < 2 or twos + ones > n:
            return None
        counts = [2] * twos + [1] * ones + [0] * (n - twos - ones)
        rng.shuffle(counts)
        return special_indices, special_masks, list(counts)

    target_actions = min(rng.randint(*_DRAFT[difficulty]["witness"]), n)
    mandatory = set()
    if difficulty == 5:
        mandatory = set(special_indices)
    elif difficulty == 6:
        live = {i for i, mask in special_masks.items()
                if any(int(mask[row][col]) and (row, col) != (1, 1)
                       and (cells[i][0] + col - 1, cells[i][1] + row - 1) in cell_set
                       for row in range(3) for col in range(3))}
        mandatory = set(rng.sample(sorted(live), min(7, len(live))))
    if len(mandatory) > target_actions:
        return None
    selected = mandatory | set(rng.sample(sorted(set(range(n)) - mandatory), target_actions - len(mandatory)))
    if difficulty <= 3:
        target = [int(i in selected) for i in range(n)]
    else:
        index = {slot: number for number, slot in enumerate(cells)}
        target = [0] * n
        for number in selected:
            x, y = cells[number]
            mask = special_masks.get(number, names.IDENTITY_STENCIL)
            for row in range(3):
                for col in range(3):
                    if int(mask[row][col]):
                        affected = index.get((x + col - 1, y + row - 1))
                        if affected is not None:
                            target[affected] ^= 1
    return special_indices, special_masks, target


def _draft(rng, difficulty):
    geometry = _role_geometry(rng, difficulty)
    if geometry is None:
        return None
    cell_slots, rule_slots = geometry
    witness = _witness_and_specials(rng, difficulty, cell_slots)
    if witness is None:
        return None
    special_indices, special_masks, target = witness
    palette = rng.sample(COLOURS, PROFILES[difficulty]["palette_size"])

    if difficulty == 1:
        offset_x, offset_y = 17, 17
    else:
        offset_x, offset_y = rng.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
    position = lambda slot: (offset_x + names.PITCH * slot[0], offset_y + names.PITCH * slot[1])
    cells = []
    for index, slot in enumerate(cell_slots):
        x, y = position(slot)
        row = {"x": x, "y": y}
        if index in special_indices:
            row["stencil"] = [list(values) for values in special_masks[index]]
        cells.append(row)

    cell_index = {slot: index for index, slot in enumerate(cell_slots)}
    if difficulty == 4:
        centre_indices = [0, 1, 2]
        rng.shuffle(centre_indices)
    else:
        centre_indices = [rng.randrange(len(palette)) for _ in rule_slots]
    constraints = []
    for rule_number, slot in enumerate(rule_slots):
        centre_index = centre_indices[rule_number]
        mask = [[rng.choice(MISMATCH_PIXELS) for _ in range(3)] for _ in range(3)]
        mask[1][1] = 0
        for row in range(3):
            for col in range(3):
                if (row, col) == (1, 1):
                    continue
                neighbour = cell_index.get((slot[0] + col - 1, slot[1] + row - 1))
                if neighbour is not None:
                    mask[row][col] = (0 if target[neighbour] == centre_index
                                      else rng.choice(MISMATCH_PIXELS))
        x, y = position(slot)
        constraints.append({"x": x, "y": y, "colour": palette[centre_index], "mask": mask})

    occupied = set(cell_slots + rule_slots)
    if difficulty == 1:
        empty_slots = [(x, y) for x in range(4) for y in range(4)
                       if (x, y) not in occupied]
    else:
        empty_slots = [(x, y) for x in range(7) for y in range(7)
                       if (x, y) not in occupied]
    empty = rng.choice(empty_slots)
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "source_id": SOURCE_ID,
        "difficulty": difficulty,
        "context_index": PROFILES[difficulty]["context_index"],
        "grid": names.GRID,
        "palette": palette,
        "stencil": [list(row) for row in names.IDENTITY_STENCIL],
        "budget": PROFILES[difficulty]["budget"],
        "cells": cells,
        "constraints": constraints,
        "tutorial_hint": difficulty == 1,
        "no_op_actions": [
            [names.ACTION_CLICK, *names.cell_click(constraints[0]["x"], constraints[0]["y"])],
            [names.ACTION_CLICK, *names.cell_click(*position(empty))],
        ],
        "supported_mechanics": list(SUPPORTED_MECHANICS),
        "omitted_mechanics": [],
        "source": "generated_only",
    }


def _hint_sprite(spec):
    anchors = [(int(row["x"]), int(row["y"])) for key in ("cells", "constraints")
               for row in spec[key]]
    left = max(0, min(x for x, _ in anchors) - 1)
    top = max(0, min(y for _, y in anchors) - 1)
    right = min(names.GRID - 1, max(x for x, _ in anchors) + names.CELL_SIZE)
    bottom = min(names.GRID - 1, max(y for _, y in anchors) + names.CELL_SIZE)
    width, height = right - left + 1, bottom - top + 1
    pixels = [[2 if row in (0, height - 1) or col in (0, width - 1) else -1
               for col in range(width)] for row in range(height)]
    return Sprite(pixels=pixels, name="generated_hint", x=left, y=top,
                  visible=True, collidable=False, tags=[names.TAG_HINT])


def _palette_legend(spec):
    pixels = []
    for colour in spec["palette"]:
        pixels.extend([[int(colour), int(colour)], [int(colour), int(colour)]])
    return Sprite(pixels=pixels, name="generated_palette_legend", x=30, y=0,
                  visible=True, collidable=False)


def _rotate_mask(mask):
    return tuple(tuple(row) for row in zip(*mask[::-1]))


def _tutorial_example_sprites(spec):
    constraint = spec["constraints"][0]
    source = tuple(
        tuple(0 if int(value) == 0 else MISMATCH_PIXELS[0] for value in row)
        for row in constraint["mask"]
    )
    rotated = _rotate_mask(source)
    masks = (source, rotated, tuple(tuple(reversed(row)) for row in source))
    centre = int(constraint["colour"])
    different = next(int(colour) for colour in spec["palette"] if int(colour) != centre)
    examples = []
    for index, ((x, y), mask) in enumerate(zip(TUTORIAL_PANEL_POSITIONS, masks)):
        pixels = [[5] * 13 for _ in range(13)]
        for offset in range(13):
            pixels[0][offset] = pixels[12][offset] = 3
            pixels[offset][0] = pixels[offset][12] = 3
        for row in range(3):
            for col in range(3):
                if (row, col) == (1, 1):
                    continue
                colour = centre if mask[row][col] == names.MATCH_FLAG else different
                top, left = 1 + 4 * row, 1 + 4 * col
                for py in range(top, top + 3):
                    for px in range(left, left + 3):
                        pixels[py][px] = colour
        for row in range(3):
            for col in range(3):
                pixels[5 + row][5 + col] = (centre if (row, col) == (1, 1)
                                             else mask[row][col])
        examples.append(Sprite(
            pixels=pixels, name=f"generated_tutorial_example_{index}", x=x, y=y,
            visible=True, collidable=False,
        ))
    return examples


def build_level(spec):
    """Build a native ARCEngine level without changing the vendored engine."""
    sprites = []
    for index, cell in enumerate(spec["cells"]):
        if "stencil" in cell:
            pixels = [[names.STENCIL_MARKER if int(cell["stencil"][row][col]) else names.SPECIAL_BODY
                       for col in range(3)] for row in range(3)]
            pixels[1][1] = names.SPECIAL_CENTRE
            tags = [names.TAG_SPECIAL_CELL, names.TAG_ANY_CELL]
        else:
            pixels = [[names.CELL_BODY] * 3 for _ in range(3)]
            tags = [names.TAG_CELL, names.TAG_ANY_CELL]
        sprites.append(Sprite(pixels=pixels, name=f"cell{index}", x=int(cell["x"]), y=int(cell["y"]),
                              visible=True, collidable=True, tags=tags))
    for index, constraint in enumerate(spec["constraints"]):
        pixels = [[names.MATCH_FLAG if int(constraint["mask"][row][col]) == 0 else MISMATCH_PIXELS[0]
                   for col in range(3)] for row in range(3)]
        pixels[1][1] = int(constraint["colour"])
        sprites.append(Sprite(pixels=pixels, name=f"rule{index}", x=int(constraint["x"]),
                              y=int(constraint["y"]), visible=True, collidable=True,
                              tags=[names.TAG_CONSTRAINT]))
    if spec.get("tutorial_hint") and spec.get("constraints") and len(spec.get("palette", ())) >= 2:
        sprites.extend(_tutorial_example_sprites(spec))
    if spec.get("tutorial_hint"):
        sprites.append(_hint_sprite(spec))
    if type(spec.get("difficulty")) is int and int(spec["difficulty"]) >= 2:
        sprites.append(_palette_legend(spec))
    data = {
        names.KEY_BUDGET: int(spec["budget"]),
        names.KEY_PALETTE: [int(value) for value in spec["palette"]],
        names.KEY_STENCIL: [[int(value) for value in row] for row in spec["stencil"]],
    }
    return Level(sprites=sprites, grid_size=(int(spec["grid"]), int(spec["grid"])), data=data,
                 name=spec.get("name", "generated"))


def _sprite_bounds(sprite):
    return sprite.x, sprite.y, sprite.x + sprite.width, sprite.y + sprite.height


def _bounds_overlap(first, second):
    return (first[0] < second[2] and second[0] < first[2]
            and first[1] < second[3] and second[1] < first[3])


def visual_errors(spec):
    """Validate the complete generated composition, including decorative cues."""
    errors = []
    try:
        palette = tuple(spec["palette"])
        if (not palette or any(type(colour) is not int for colour in palette)
                or len(set(palette)) != len(palette)
                or any(colour not in COLOURS for colour in palette)):
            errors.append("palette must use distinct high-contrast official FT09 colours")
    except Exception as exc:
        return [f"visual palette data is malformed: {exc}"]

    try:
        level = build_level(spec)
        sprites = list(level.get_sprites())
        grid = int(spec["grid"])
    except Exception as exc:
        return errors + [f"visual composition could not be built: {exc}"]

    for sprite in sprites:
        left, top, right, bottom = _sprite_bounds(sprite)
        if left < 0 or top < 0 or right > grid or bottom > grid:
            errors.append(
                f"sprite {sprite.name} is clipped: rendered extent "
                f"({left}, {top})..({right}, {bottom}) exceeds {grid}x{grid}"
            )

    examples = [sprite for sprite in sprites
                if sprite.name.startswith("generated_tutorial_example_")]
    legends = [sprite for sprite in sprites if sprite.name == "generated_palette_legend"]
    active = [sprite for sprite in sprites
              if (sprite.name.startswith("cell") or sprite.name.startswith("rule")
                  or sprite.name == "generated_hint")]
    difficulty = spec.get("difficulty")

    if difficulty == 1:
        if len(examples) != 3 or legends:
            errors.append("tier 1 must render exactly three examples and no palette legend")
        try:
            source = tuple(
                tuple(0 if int(value) == 0 else MISMATCH_PIXELS[0] for value in row)
                for row in spec["constraints"][0]["mask"]
            )
            d4_masks = set()
            transformed = source
            for _ in range(4):
                d4_masks.add(transformed)
                d4_masks.add(tuple(tuple(reversed(row)) for row in transformed))
                transformed = _rotate_mask(transformed)
            centre = int(spec["constraints"][0]["colour"])
            for example in examples:
                pixels = example.render().tolist()
                border = (pixels[0] + pixels[-1]
                          + [row[0] for row in pixels] + [row[-1] for row in pixels])
                if len(pixels) != 13 or any(len(row) != 13 for row in pixels) or set(border) != {3}:
                    errors.append(f"tutorial example {example.name} lacks its complete static panel")
                    continue
                if example.is_collidable:
                    errors.append(f"tutorial example {example.name} must be static")
                observed = tuple(
                    tuple(0 if (row, col) == (1, 1) or pixels[5 + row][5 + col] == 0
                          else MISMATCH_PIXELS[0] for col in range(3))
                    for row in range(3)
                )
                if observed not in d4_masks or pixels[6][6] != centre:
                    errors.append(f"tutorial example {example.name} is not derived from its puzzle rule")
                for row in range(3):
                    for col in range(3):
                        if (row, col) == (1, 1):
                            continue
                        flag = pixels[5 + row][5 + col]
                        cell_colour = pixels[2 + 4 * row][2 + 4 * col]
                        if (flag not in (names.MATCH_FLAG, MISMATCH_PIXELS[0])
                                or cell_colour not in palette
                                or ((cell_colour == centre) != (flag == names.MATCH_FLAG))):
                            errors.append(
                                f"tutorial example {example.name} does not truthfully show match/differ"
                            )
                            break
                    else:
                        continue
                    break
        except Exception as exc:
            errors.append(f"tutorial examples could not be verified: {exc}")
    elif type(difficulty) is int and difficulty >= 2:
        if examples or len(legends) != 1:
            errors.append("tiers 2..6 must render one palette legend and no tutorial examples")
        elif legends[0].render().tolist() != [
                [colour, colour] for colour in palette for _ in range(2)]:
            errors.append("palette legend does not show the generated cycle in order")

    for index, first in enumerate(examples):
        for second in examples[index + 1:]:
            if _bounds_overlap(_sprite_bounds(first), _sprite_bounds(second)):
                errors.append(f"tutorial examples {first.name} and {second.name} overlap")
        for other in active:
            if _bounds_overlap(_sprite_bounds(first), _sprite_bounds(other)):
                errors.append(f"tutorial example {first.name} overlaps the active puzzle")
    for legend in legends:
        for other in sprites:
            if other is not legend and _bounds_overlap(_sprite_bounds(legend), _sprite_bounds(other)):
                errors.append(f"sprite {other.name} overlaps the palette legend")
    return errors


def replay_in_context(spec):
    """Replay the stored solution at its actual native level index."""
    context = int(spec.get("context_index", 0))
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context + 1)])
    env.set_level(context)
    if env.level_index != context:
        return False
    actions = [tuple(action) for action in spec["solution"]]
    if any(action[0] not in env.available_actions for action in actions):
        return False
    completed, _ = replay(env, actions, expect_level=context)
    return completed and (env.state.name == "WIN" if context == env.level_count - 1 else True)


def replays_to_completion(spec):
    """Compatibility name: verification now honors the declared context."""
    return replay_in_context(spec)


def _solution_mechanics(layout, result):
    return _mechanics_from_actions(layout, result.actions)


def _mechanics_from_actions(layout, actions):
    by_click = {click: index for index, click in enumerate(layout.clicks)}
    normalized = []
    clicks = []
    for action in actions:
        if (not isinstance(action, (list, tuple)) or len(action) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in action)):
            raise ValueError("solution actions must be integer (id, x, y) triples")
        parsed = tuple(action)
        if parsed[0] != names.ACTION_CLICK or parsed[1:] not in by_click:
            raise ValueError("solution action does not click a live cell")
        normalized.append(parsed)
        clicks.append(by_click[parsed[1:]])
    if not normalized or not verify(layout, normalized):
        raise ValueError("solution does not win symbolically at its final action")
    state = layout.initial
    counts = Counter(clicks)
    special = {index for index, cell in enumerate(layout.cells) if cell[2]}
    coupled = {index for index, affects in enumerate(layout.affects)
               if any(target != index for target in affects)}
    third_colour_actions = 0
    for index in clicks:
        state = layout.click(state, index)
        if 2 in state:
            third_colour_actions += 1
    position_to_index = {cell[:2]: index for index, cell in enumerate(layout.cells)}
    exercised_constraints = 0
    for x, y, _centre, _mask in layout.constraints:
        neighbours = [position_to_index[(x + dx, y + dy)]
                      for dx, dy in names.NEIGHBOUR_OFFSETS.values()
                      if (x + dx, y + dy) in position_to_index]
        if any(state[index] != layout.initial[index] for index in neighbours):
            exercised_constraints += 1
    return {
        "distinct_clicked_cells": len(counts),
        "repeat_actions": sum(value - 1 for value in counts.values()),
        "changed_cells": sum(a != b for a, b in zip(layout.initial, state)),
        "third_colour_actions": third_colour_actions,
        "third_colour_cells_final": sum(value == 2 for value in state),
        "special_clicks": sum(counts[index] for index in special),
        "distinct_special_cells_clicked": len(set(counts) & special),
        "coupled_special_clicks": sum(counts[index] for index in special & coupled),
        "distinct_coupled_special_cells_clicked": len(set(counts) & special & coupled),
        "exercised_constraints": exercised_constraints,
        "constraint_count": len(layout.constraints),
    }


def _spec_from_env(env):
    """Read official semantics only for exact-copy rejection."""
    layout = extract(env)
    cells = []
    for sprite, (_x, _y, special) in zip(env.cells(), layout.cells):
        row = {"x": int(sprite.x), "y": int(sprite.y)}
        if special:
            stencil = [[1 if int(sprite.pixels[r][c]) == names.STENCIL_MARKER else 0
                        for c in range(3)] for r in range(3)]
            stencil[1][1] = 1
            row["stencil"] = stencil
        cells.append(row)
    constraints = [
        {"x": x, "y": y, "colour": centre, "mask": [list(row) for row in mask]}
        for x, y, centre, mask in layout.constraints
    ]
    return {"palette": list(layout.palette), "budget": layout.budget,
            "stencil": [list(row) for row in layout.stencil], "cells": cells,
            "constraints": constraints}


@lru_cache(maxsize=1)
def _official_identities():
    geometries, gameplays = set(), set()
    env = Env()
    for index in range(env.level_count):
        env.set_level(index)
        spec = _spec_from_env(env)
        geometries.add(geometry_hash(spec))
        gameplays.add(gameplay_hash(spec))
    return frozenset(geometries), frozenset(gameplays)


def _reject(exclusions, reason, *, seed, difficulty, attempt, record_rejection):
    exclusions[reason] += 1
    if record_rejection:
        record_rejection({"seed": seed, "difficulty": difficulty, "attempt": attempt,
                          "reason": reason, "generator_version": GENERATOR_VERSION})


def validate_full_standard(spec, curriculum_entry):
    """Return human-readable failures for the approved shared family contract."""
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    try:
        errors = profile_errors(spec)
    except Exception as exc:  # validator boundary: malformed public JSON must not escape
        errors = [f"malformed spec/profile data: {exc}"]
    errors.extend(visual_errors(spec))
    if not isinstance(curriculum_entry, Mapping):
        return errors + ["curriculum entry must be a mapping"]
    difficulty = spec.get("difficulty")
    valid_difficulty = type(difficulty) is int and difficulty in PROFILES
    if curriculum_entry.get("difficulty") != difficulty:
        errors.append("curriculum difficulty and spec difficulty disagree")
    if valid_difficulty:
        if curriculum_entry.get("context_index") != PROFILES[difficulty]["context_index"]:
            errors.append("curriculum context index differs from official tier order")
        if curriculum_entry.get("search_work") != PROFILES[difficulty]["search_work"]:
            errors.append("curriculum search work differs from calibrated bound")
    if spec.get("format") != FORMAT or spec.get("generator_version") != GENERATOR_VERSION:
        errors.append("wrong generated format/version")
    if spec.get("difficulty_version") != DIFFICULTY_VERSION:
        errors.append("wrong reference profile version")
    if spec.get("mechanics_inventory_version") != MECHANICS_INVENTORY_VERSION:
        errors.append("wrong mechanic inventory version")
    if spec.get("quality_profile_version") != QUALITY_PROFILE_VERSION:
        errors.append("wrong quality profile version")
    if spec.get("source_id") != SOURCE_ID or spec.get("source") != "generated_only":
        errors.append("wrong generated source provenance")
    split = spec.get("split")
    if split not in SPLITS:
        errors.append("full-standard spec must declare train, validation, or test split")
    elif spec.get("geometry_partition") != split:
        errors.append("canonical D4 geometry is not in the requested split partition")
    computed_geometry = computed_gameplay = None
    try:
        computed_geometry = geometry_hash(spec)
        computed_gameplay = gameplay_hash(spec)
        if spec.get("geometry_sha256") != computed_geometry:
            errors.append("geometry identity does not match canonical content")
        if spec.get("geometry_d4_sha256") != computed_geometry:
            errors.append("D4 geometry identity does not match canonical content")
        if spec.get("geometry_d4_sha256") != spec.get("geometry_sha256"):
            errors.append("D4 geometry identity and geometry identity disagree")
        if spec.get("gameplay_sha256") != computed_gameplay:
            errors.append("gameplay identity does not match canonical content")
        if split in SPLITS and identity_partition(computed_geometry) != split:
            errors.append("recomputed canonical geometry is outside the requested split")
        official_geometry, official_gameplay = _official_identities()
        if computed_geometry in official_geometry:
            errors.append("recomputed canonical geometry matches an official level")
        if computed_gameplay in official_gameplay:
            errors.append("recomputed gameplay matches an official level")
    except Exception as exc:  # validator boundary
        errors.append(f"identity could not be recomputed: {exc}")
    try:
        recomputed_metrics = structural_metrics(spec)
        for field, value in recomputed_metrics.items():
            if spec.get(field) != value:
                errors.append(f"stored {field} does not match recomputed structure")
    except Exception as exc:  # validator boundary
        errors.append(f"structure could not be recomputed: {exc}")
    if spec.get("omitted_mechanics") != []:
        errors.append("full FT09 mode may not declare omitted mechanics")
    proof = spec.get("proof")
    if not isinstance(proof, Mapping):
        errors.append("missing structured proof")
    elif valid_difficulty:
        expected_search = curriculum_entry.get("search_work")
        expected = {
            "difficulty": difficulty,
            "context_index": PROFILES[difficulty]["context_index"],
            "native_budget": PROFILES[difficulty]["budget"],
            "optimal_actions": spec.get("optimal_actions"),
            "search_limit": expected_search,
            "search_work": spec.get("search_work"),
            "search_truncated": False,
            "symbolic_verified": True,
            "context_engine_verified": True,
            "engine_win": True,
            "split": split,
            "geometry_sha256": spec.get("geometry_sha256"),
            "gameplay_sha256": spec.get("gameplay_sha256"),
            "generator_version": GENERATOR_VERSION,
            "difficulty_version": DIFFICULTY_VERSION,
            "source_id": SOURCE_ID,
        }
        for field, value in expected.items():
            if proof.get(field) != value:
                errors.append(f"proof {field} does not mirror certified top-level data")
        if spec.get("search_limit") != expected_search:
            errors.append("accepted full row did not use the calibrated search limit")
        measured_work = spec.get("search_work")
        if (type(measured_work) is not int or type(spec.get("search_limit")) is not int
                or measured_work < 1 or measured_work > spec["search_limit"]):
            errors.append("measured search work is missing or outside its bound")
    if (valid_difficulty
            and spec.get("context_index") == PROFILES[difficulty]["context_index"]):
        try:
            layout = extract(Env([build_level(spec)]))
            recomputed_mechanics = _mechanics_from_actions(layout, spec.get("solution", ()))
            if spec.get("solution_mechanics") != recomputed_mechanics:
                errors.append("stored solution mechanics do not match route simulation")
            if spec.get("engine_verified") is not True or spec.get("context_engine_verified") is not True:
                errors.append("top-level native proof flags must be exact booleans")
            if not replay_in_context(spec):
                errors.append("stored solution does not win in the declared native context")
        except Exception as exc:  # validator boundary
            errors.append(f"route/native replay could not be recomputed: {exc}")
    return errors


def generate(seed, difficulty, attempts=200, limit=None, *, split=None, record_rejection=None):
    """Generate a bounded, exact, real-engine-certified FT09 level.

    ``split`` is mandatory for full collection and must be explicit. Omitting
    it retains deterministic legacy/smoke compatibility but such a row will
    intentionally fail ``validate_full_standard``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if limit is not None and (type(limit) is not int or not 0 < limit <= MAX_SEARCH_WORK):
        raise ValueError(f"limit must be a positive integer <= {MAX_SEARCH_WORK}")
    if split is not None and split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    search_limit = min(limit or PROFILES[difficulty]["search_work"],
                       PROFILES[difficulty]["search_work"])
    stream = split if split is not None else "smoke"
    rng = random.Random(f"{DIFFICULTY_VERSION}:{GENERATOR_VERSION}:{stream}:{seed}:{difficulty}")
    exclusions = Counter()
    official_geometry, official_gameplay = _official_identities()

    for attempt in range(1, attempts + 1):
        spec = _draft(rng, difficulty)
        if spec is None:
            _reject(exclusions, "invalid_geometry", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        layout = extract(Env([build_level(spec)]))
        result = search(layout, limit=search_limit)
        if not result:
            reason = "search_truncated" if result.truncated else "no_solution"
            _reject(exclusions, reason, seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        if result.truncated:
            _reject(exclusions, "search_truncated", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        actions = len(result)
        if not _in_bounds(actions, PROFILES[difficulty]["actions"]):
            _reject(exclusions, "action_profile", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue

        mechanics = _solution_mechanics(layout, result)
        if mechanics["exercised_constraints"] != mechanics["constraint_count"]:
            _reject(exclusions, "inactive_constraint", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        spec.update(
            seed=seed,
            generation_attempt=attempt,
            generation_exclusions=dict(exclusions),
            name=f"ft09-full-d{difficulty}-{seed}",
            split=split,
            solution=[list(action) for action in result.actions],
            solution_length=actions,
            optimal_actions=actions,
            search_limit=search_limit,
            search_work=result.explored,
            search_truncated=False,
            solution_mechanics=mechanics,
            engine_verified=False,
            context_engine_verified=False,
        )
        spec.update(structural_metrics(spec))
        spec["geometry_sha256"] = geometry_hash(spec)
        spec["geometry_d4_sha256"] = spec["geometry_sha256"]
        spec["gameplay_sha256"] = gameplay_hash(spec)
        spec["geometry_partition"] = identity_partition(spec["geometry_sha256"])
        spec["gameplay_partition"] = identity_partition(spec["gameplay_sha256"])
        if spec["geometry_sha256"] in official_geometry or spec["gameplay_sha256"] in official_gameplay:
            _reject(exclusions, "official_copy", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        if split is not None and spec["geometry_partition"] != split:
            _reject(exclusions, "split_partition", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        if not verify(layout, result.actions):
            _reject(exclusions, "symbolic_replay", seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        spec.update(engine_verified=True, context_engine_verified=replay_in_context(spec))
        spec["proof"] = {
            "difficulty": difficulty,
            "context_index": PROFILES[difficulty]["context_index"],
            "native_budget": PROFILES[difficulty]["budget"],
            "optimal_actions": actions,
            "search_limit": search_limit,
            "search_work": result.explored,
            "search_truncated": False,
            "symbolic_verified": True,
            "context_engine_verified": spec["context_engine_verified"],
            "engine_win": spec["context_engine_verified"],
            "split": split,
            "geometry_sha256": spec["geometry_sha256"],
            "gameplay_sha256": spec["gameplay_sha256"],
            "generator_version": GENERATOR_VERSION,
            "difficulty_version": DIFFICULTY_VERSION,
            "source_id": SOURCE_ID,
        }
        errors = profile_errors(spec)
        if split is not None:
            errors.extend(validate_full_standard(
                spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            ))
        if errors:
            reason = "profile:" + "|".join(sorted(set(errors)))
            _reject(exclusions, reason, seed=seed, difficulty=difficulty,
                    attempt=attempt, record_rejection=record_rejection)
            continue
        spec["generation_exclusions"] = dict(exclusions)
        return spec
    return None


def _child_seed(seed, ordinal, difficulty):
    material = f"{SOURCE_ID}:{seed}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split, difficulties=None, attempts=200, limit=None,
                  record_rejection=None):
    """Generate an ordered full native curriculum from stable per-tier seeds."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not selected or any(type(value) is not int or value not in DIFFICULTIES for value in selected):
        raise ValueError(f"difficulties must be a nonempty sequence drawn from {DIFFICULTIES}")
    if difficulties is not None and selected != tuple(sorted(set(selected))):
        raise ValueError("explicit smoke difficulties must be distinct and in increasing order")
    rows = []
    for ordinal, difficulty in enumerate(selected):
        row = generate(
            _child_seed(seed, ordinal, difficulty), difficulty,
            attempts=attempts, limit=limit, split=split,
            record_rejection=record_rejection,
        )
        if row is None:
            return None
        row["game_seed"] = seed
        row["game_ordinal"] = ordinal
        rows.append(row)
    return rows


def build_game(specs):
    """Validate and build exactly one ordered six-tier native level list."""
    if not isinstance(specs, (list, tuple)) or len(specs) != len(DIFFICULTIES):
        raise ValueError(f"full FT09 game requires exactly {len(DIFFICULTIES)} specs")
    if any(not isinstance(spec, Mapping) for spec in specs):
        raise ValueError("every game spec must be a mapping")
    difficulties = tuple(spec.get("difficulty") for spec in specs)
    contexts = tuple(spec.get("context_index") for spec in specs)
    if difficulties != DIFFICULTIES:
        raise ValueError("game specs must contain difficulties 1..6 in order")
    if contexts != tuple(range(len(DIFFICULTIES))):
        raise ValueError("game specs must preserve native context indices 0..5")
    split_values = [spec.get("split") for spec in specs]
    if (any(split not in SPLITS for split in split_values)
            or any(split != split_values[0] for split in split_values[1:])):
        raise ValueError("game specs must share one explicit full split")
    gameplay_identities = [spec.get("gameplay_sha256") for spec in specs]
    geometry_identities = [spec.get("geometry_sha256") for spec in specs]
    if (any(not isinstance(value, str) or not value for value in gameplay_identities + geometry_identities)
            or len(set(gameplay_identities)) != len(gameplay_identities)
            or len(set(geometry_identities)) != len(geometry_identities)):
        raise ValueError("game specs must have distinct geometry and gameplay identities")
    errors = []
    for spec, curriculum in zip(specs, FULL_STANDARD_CONTRACT["curriculum"]):
        errors.extend(validate_full_standard(spec, curriculum))
    if errors:
        raise ValueError("invalid full-standard game: " + "; ".join(sorted(set(errors))))
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    for index, spec in enumerate(specs):
        if env.level_index != index:
            raise ValueError("native game shifted away from the declared context sequence")
        actions = [tuple(action) for action in spec["solution"]]
        completed, _ = replay(env, actions, expect_level=index)
        if not completed or env.levels_completed != index + 1:
            raise ValueError(f"native sequential replay failed at context {index}")
    if env.state.name != "WIN":
        raise ValueError("native six-tier replay did not finish in WIN")
    return levels
