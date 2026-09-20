"""Full six-tier procedural CN04 generator.

Generation uses only aggregate official measurements. Geometry, pixels,
assignments, starts, and constructive candidates are newly sampled. Every
accepted row passes a family-local reference profile, a canonical three-way
geometry split, symbolic/native transition differential replay in its intended
level index, and an official-sprite-set novelty check.
"""

from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import random

import numpy as np
from arcengine import Level, Sprite

from . import names
from .env import Env, official_levels
from .layout import extract
from .plan import GENERATED_ASSIGNMENT_LIMIT, search, trace, transition
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    PROFILES,
    QUALITY_PROFILE_VERSION,
    STACK_PIN_SPECTRA,
    WINNING_PIN_DEGREES,
    native_semantic_groups,
    primitive_schema_errors,
    profile_errors,
    structural_metrics,
)


FORMAT = "pebby.cn04.level.v3"
GENERATOR_VERSION = 4
MECHANICS_VERSION = "cn04-full-multipin-relations-v2"
GEOMETRY_VERSION = "cn04-native-grouped-initial-d4-v3"
SPLIT_VERSION = "cn04-three-way-native-grouped-d4-v3"
SOURCE_ID = "cn04-2fe56bfb"
SPLITS = ("train", "validation", "test")
BODY_COLOURS = (9, 10, 11, 12, 14, 15)
BACKGROUND_COLOURS = (4, 9, 10, 11, 12, 14, 15)

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": SOURCE_ID,
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": tuple({
        "difficulty": difficulty,
        "context_index": difficulty - 1,
        "search_work": PROFILES[difficulty]["search_work"],
    } for difficulty in DIFFICULTIES),
    "evidence": {
        "official_tier_characterization": "pebby/games/cn04/reference_profiles.py: official_characterization",
        "solution_mechanics": "spec.solution_mechanics from exact symbolic witness trace",
        "native_budget": "spec.native_budget preserves MaxSteps and usable-action boundary",
        "context_engine_replay": "spec.proof native differential replay at difficulty-1",
        "novelty_split": "semantic D4 hash, gameplay hash, and official sprite-set rejection",
        "bounded_rejections": "spec.generation_exclusions and generate.last_rejections",
        "root_acceptance": "docs/generator-evidence/cn04.md: independent source, schema, native episode, and visual checks",
    },
    "caveats": (
        "Each tier is calibrated around one shipped level; tolerances are engineering bounds, not confidence intervals.",
        "Minimum translation is conditional on the selected assembly, not a global shortest-path proof.",
        "The teacher uses public tier degree information to prune alternate assignments.",
        "Tier five has a fixed unlabeled parallel-star topology despite colour and layout diversity.",
        "Assembly and assignment search are finitely bounded; truncation is inconclusive.",
        "Ready records generator-contract acceptance, not trained-controller performance.",
    ),
}


def _jsonable(value):
    return json.loads(json.dumps(value))


def _exact_tree_equal(actual, expected):
    """Compare JSON-like values without bool/int or int/float equivalence."""
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and actual.keys() == expected.keys()
            and all(_exact_tree_equal(actual[key], value)
                    for key, value in expected.items())
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, type(expected))
            and len(actual) == len(expected)
            and all(_exact_tree_equal(left, right)
                    for left, right in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _rendered(pixels, rotation):
    return np.rot90(np.asarray(pixels, dtype=int), k=(-(rotation // 90)) % 4)


def _relation_edges(rng, degrees):
    """Sample a connected loop-free multigraph with the requested degrees."""
    if tuple(degrees) == (2, 2):
        return ((0, 1), (0, 1))
    for _ in range(800):
        stubs = [group for group, count in enumerate(degrees) for _ in range(count)]
        rng.shuffle(stubs)
        edges = []
        valid = True
        while stubs:
            left = stubs.pop()
            candidates = [index for index, right in enumerate(stubs) if right != left]
            if not candidates:
                valid = False
                break
            chosen = rng.choice(candidates)
            right = stubs.pop(chosen)
            edge = tuple(sorted((left, right)))
            # Official tier five necessarily has three simultaneous relations
            # between its seven-pin stack winner and the three-pin singleton.
            if edges.count(edge) >= 3:
                valid = False
                break
            edges.append(edge)
        if not valid:
            continue
        reached = {0}
        while True:
            changed = reached | {
                right for left, right in edges if left in reached
            } | {
                left for left, right in edges if right in reached
            }
            if changed == reached:
                break
            reached = changed
        if len(reached) == len(degrees):
            rng.shuffle(edges)
            return tuple(edges)
    return None


def _relation_points(rng, count):
    """Fresh edge locations whose layout changes relation geometry, not travel."""
    width = rng.randint(5, 8)
    height = rng.randint(5, 8)
    cells = [(x, y) for y in range(height) for x in range(width)]
    rng.shuffle(cells)
    chosen = []
    for cell in cells:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 2
               for other in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        return None
    ox = rng.randint(4, 20 - width - 4)
    oy = rng.randint(4, 20 - height - 4)
    return tuple((x + ox, y + oy) for x, y in chosen)


def _piece_pixels(rng, pins, body_colour, minimum_opaque):
    """Create one asymmetric connected piece through all relation pins."""
    xs = [x for x, _, _ in pins]
    ys = [y for _, y, _ in pins]
    left, top = rng.randint(1, 2), rng.randint(1, 2)
    right, bottom = rng.randint(1, 2), rng.randint(1, 2)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 1 + left + right
    height = max_y - min_y + 1 + top + bottom
    local = tuple((x - min_x + left, y - min_y + top, colour)
                  for x, y, colour in pins)
    pixels = [[names.TRANSPARENT for _ in range(width)] for _ in range(height)]
    anchor_x, anchor_y, _ = local[0]
    for x, y, _ in local:
        if rng.randrange(2):
            for px in range(min(anchor_x, x), max(anchor_x, x) + 1):
                pixels[anchor_y][px] = body_colour
            for py in range(min(anchor_y, y), max(anchor_y, y) + 1):
                pixels[py][x] = body_colour
        else:
            for py in range(min(anchor_y, y), max(anchor_y, y) + 1):
                pixels[py][anchor_x] = body_colour
            for px in range(min(anchor_x, x), max(anchor_x, x) + 1):
                pixels[y][px] = body_colour
    path = {(x, y) for y, row in enumerate(pixels) for x, value in enumerate(row)
            if value >= 0}
    wanted = min(width * height, minimum_opaque + rng.randint(0, 4))
    while len(path) < wanted:
        candidates = sorted({(nx, ny) for x, y in path
                             for nx, ny in ((x + 1, y), (x - 1, y),
                                            (x, y + 1), (x, y - 1))
                             if 0 <= nx < width and 0 <= ny < height
                             and pixels[ny][nx] < 0})
        if not candidates:
            break
        nx, ny = rng.choice(candidates)
        pixels[ny][nx] = body_colour
        path.add((nx, ny))
    for x, y, colour in local:
        pixels[y][x] = colour
    return pixels, (min_x - left, min_y - top)


def _add_cycle_marker(pixels):
    changed = [list(row) for row in pixels]
    for y, row in enumerate(changed):
        for x, value in enumerate(row):
            if value >= 0 and value not in names.PIN_COLORS:
                changed[y][x] = names.CYCLE_PIXEL
                return changed
    return changed


def _random_alternate(rng, body_colour, pin_colours, pin_count,
                      *, pin_sequence=None):
    width, height = rng.randint(4, 7), rng.randint(4, 7)
    wanted = rng.randint(max(9, pin_count + 2), min(22, width * height))
    x, y = rng.randrange(width), rng.randrange(height)
    cells = {(x, y)}
    while len(cells) < wanted:
        x, y = rng.choice(tuple(cells))
        dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        x, y = max(0, min(width - 1, x + dx)), max(0, min(height - 1, y + dy))
        cells.add((x, y))
    pixels = [[names.TRANSPARENT for _ in range(width)] for _ in range(height)]
    for x, y in cells:
        pixels[y][x] = body_colour
    pin_cells = rng.sample(sorted(cells), pin_count)
    colours = list(pin_sequence) if pin_sequence is not None else [
        pin_colours[index % len(pin_colours)] for index in range(pin_count)
    ]
    if len(colours) != pin_count:
        raise ValueError("alternate pin colour sequence has the wrong length")
    rng.shuffle(colours)
    for index, (x, y) in enumerate(pin_cells):
        pixels[y][x] = colours[index]
    return _add_cycle_marker(pixels)


def _inverse_store(pixels, rotation):
    if rotation == 0:
        return [list(row) for row in pixels]
    stored = np.rot90(np.asarray(pixels, dtype=int), k=(rotation // 90) % 4)
    return stored.astype(int).tolist()


def _minimum_target_travel(starts, alternatives, rotations, targets, winning):
    min_x = min_y = 0
    max_x = max_y = 0
    initialized = False
    for group_index, target in enumerate(targets):
        winner = winning[group_index]
        rendered = _rendered(
            alternatives[group_index][winner], rotations[group_index][winner],
        )
        height, width = rendered.shape
        x, y = target
        if not initialized:
            min_x, min_y, max_x, max_y = x, y, x + width, y + height
            initialized = True
        else:
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + width), max(max_y, y + height)
    values = []
    for sx in range(-min_x, 20 - max_x + 1):
        for sy in range(-min_y, 20 - max_y + 1):
            values.append(sum(
                abs(starts[index][0] - (targets[index][0] + sx))
                + abs(starts[index][1] - (targets[index][1] + sy))
                for index in range(len(starts))
            ))
    return min(values) if values else -1


def _scatter_groups(
    rng, alternatives, rotations, target_positions, winning, minimum_travel,
):
    # The stacked tiers need a genuinely dispersed initial state.  A bounded
    # thousand-candidate placement search is still tiny compared with the
    # teacher bound and avoids manufacturing length by choosing a remote final
    # translation after the puzzle has already been fixed.
    for _ in range(1_000):
        occupied = set()
        occupied_origins = set()
        transforms = []
        valid = True
        for group_index, group in enumerate(alternatives):
            placed = None
            for trial in range(220):
                rotation = rotations[group_index][0]
                rendered = _rendered(group[0], rotation)
                height, width = rendered.shape
                if trial == 0 and group_index == 0:
                    x, y = 1, 2
                    if x + width > 20 or y + height > 20:
                        continue
                else:
                    x = rng.randint(0, 20 - width)
                    y = rng.randint(1, 20 - height)
                # Native CN04 forms stacks solely from coincident initial
                # top-left coordinates.  Distinct generated groups must never
                # accidentally merge even when their opaque cells do not
                # overlap.
                if (x, y) in occupied_origins:
                    continue
                cells = {(x + int(px), y + int(py))
                         for py, px in np.argwhere(rendered >= 0)}
                if cells.isdisjoint(occupied):
                    placed = (x, y, cells)
                    break
            if placed is None:
                valid = False
                break
            x, y, cells = placed
            occupied.update(cells)
            occupied_origins.add((x, y))
            transforms.append((x, y))
        if valid and _minimum_target_travel(
            transforms, alternatives, rotations, target_positions, winning,
        ) >= minimum_travel:
            return transforms
    return None


def _stack_composition(rng, difficulty):
    if difficulty == 5:
        return (5, 1, 1, 1), (rng.randint(1, 4), 0, 0, 0)
    if difficulty == 6:
        return (6, 4, 1, 1, 1), (rng.randint(1, 5), rng.randint(1, 3), 0, 0, 0)
    groups = PROFILES[difficulty]["groups"]
    return (1,) * groups, (0,) * groups


def _initial_foreground_colours(pieces, grey_masking):
    """Visible native colours after CN04 selects its nearest initial sprite."""
    visible = [piece for piece in pieces if piece.get("visible", True)]
    if not visible:
        return set()
    selected = min(visible, key=lambda piece: int(piece["x"]) ** 2 + int(piece["y"]) ** 2)
    colours = set()
    for piece in visible:
        values = {
            int(value)
            for row in piece["pixels"]
            for value in row
            if int(value) >= 0
        }
        if grey_masking and piece is not selected:
            colours.add(names.GREY)
        elif grey_masking:
            colours.update(names.PIN_A if value == names.PIN_B else value
                           for value in values)
        elif piece is selected:
            colours.update(
                value if value in (names.PIN_A, names.PIN_B, names.MATCHED)
                else names.SELECTED_BODY
                for value in values
            )
            if names.PIN_B in colours:
                colours.remove(names.PIN_B)
                colours.add(names.PIN_A)
        else:
            colours.update(names.PIN_A if value == names.PIN_B else value
                           for value in values)
    return colours


def _draft(seed, difficulty, attempt):
    rng = random.Random(f"{MECHANICS_VERSION}:{int(seed)}:{difficulty}:{attempt}")
    profile = PROFILES[difficulty]
    group_count = profile["groups"]
    degrees = WINNING_PIN_DEGREES[difficulty]
    relation_edges = _relation_edges(rng, degrees)
    if relation_edges is None:
        return None
    points = _relation_points(rng, len(relation_edges))
    if points is None:
        return None
    if difficulty == 4:
        edge_colours = [names.PIN_A] * len(relation_edges)
    else:
        edge_colours = [
            names.PIN_A if index % 2 == 0 else names.PIN_B
            for index in range(len(relation_edges))
        ]
        rng.shuffle(edge_colours)
    incident = [[] for _ in range(group_count)]
    for edge, point, colour in zip(relation_edges, points, edge_colours):
        for group_index in edge:
            incident[group_index].append((*point, colour))
    if tuple(len(pins) for pins in incident) != degrees:
        raise AssertionError("relation graph degrees differ from tier grammar")
    target_pixels = []
    targets = []
    minimum_opaque = {1: 14, 2: 11, 3: 11, 4: 11, 5: 8, 6: 8}[difficulty]
    for index, pins in enumerate(incident):
        pixels, target = _piece_pixels(
            rng, pins,
            BODY_COLOURS[index % len(BODY_COLOURS)],
            minimum_opaque,
        )
        target_pixels.append(pixels)
        targets.append(target)

    stack_sizes, winning = _stack_composition(rng, difficulty)
    alternatives = []
    rotations = []
    stack_ordinal = 0
    for group_index, stack_size in enumerate(stack_sizes):
        spectrum = None
        if stack_size > 1:
            spectrum = list(STACK_PIN_SPECTRA[difficulty][stack_ordinal])
            stack_ordinal += 1
            spectrum.remove(degrees[group_index])
            rng.shuffle(spectrum)
            spectrum.insert(winning[group_index], degrees[group_index])
        group = []
        group_rotations = []
        for alternate in range(stack_size):
            fixed_rotation = 90 * rng.randrange(4) if stack_size > 1 else 0
            if alternate == winning[group_index]:
                pixels = target_pixels[group_index]
                if stack_size > 1:
                    pixels = _add_cycle_marker(pixels)
                pixels = _inverse_store(pixels, fixed_rotation)
            elif stack_size > 1:
                # Equal-count decoys also preserve the winner's colour
                # multiset, so neither total pin count nor per-colour spectrum
                # identifies the correct alternate.
                same_degree_colours = None
                if spectrum[alternate] == degrees[group_index]:
                    same_degree_colours = [colour for _, _, colour
                                           in incident[group_index]]
                pixels = _random_alternate(
                    rng,
                    BODY_COLOURS[(group_index + alternate) % len(BODY_COLOURS)],
                    profile["pin_colours"],
                    spectrum[alternate],
                    pin_sequence=same_degree_colours,
                )
                pixels = _inverse_store(pixels, fixed_rotation)
            else:
                raise AssertionError("singleton winner must be alternate zero")
            group.append(pixels)
            group_rotations.append(
                fixed_rotation if stack_size > 1 else 90 * rng.randint(1, 3)
            )
        alternatives.append(group)
        rotations.append(group_rotations)

    minimum_travel = {1: 5, 2: 16, 3: 12, 4: 18, 5: 49, 6: 35}[difficulty]
    starts = _scatter_groups(
        rng, alternatives, rotations, targets, winning, minimum_travel,
    )
    if starts is None:
        return None
    pieces = []
    for group_index, group in enumerate(alternatives):
        x, y = starts[group_index]
        for alternate, pixels in enumerate(group):
            pieces.append({
                "name": f"cn04_g{int(seed)}_{attempt}_{group_index}_{alternate}",
                "group": group_index,
                "alternate": alternate,
                "pixels": pixels,
                "x": x,
                "y": y,
                "rotation": rotations[group_index][alternate],
                "layer": alternate,
                "visible": alternate == 0,
            })
    foreground = _initial_foreground_colours(pieces, profile["grey_masking"])
    background_choices = [
        colour for colour in BACKGROUND_COLOURS if colour not in foreground
    ]
    if not background_choices:
        return None
    spec = {
        "format": FORMAT,
        "source": "generated_only",
        "source_id": SOURCE_ID,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "split_version": SPLIT_VERSION,
        "seed": int(seed),
        "effective_seed": int(seed),
        "generation_attempt": int(attempt),
        "difficulty": int(difficulty),
        "training_context_index": int(difficulty - 1),
        "verification_level_index": int(difficulty - 1),
        "grid_size": [20, 20],
        "background": rng.choice(background_choices),
        "grey_masking": profile["grey_masking"],
        "max_steps": profile["max_steps"],
        "pieces": pieces,
        "constructed_target": [
            [targets[index][0], targets[index][1],
             (rotations[index][winning[index]] if stack_sizes[index] > 1 else 0),
             winning[index]]
            for index in range(group_count)
        ],
        "reference_calibration": (
            "aggregate measurements of one shipped level per tier; explicit "
            "engineering tolerances; no official pixels, geometry, or routes"
        ),
        "relation_grammar": "connected-random-multigraph-with-simultaneous-exact-pairs",
    }
    spec.update(structural_metrics(spec))
    return spec


def build_level(spec):
    """Rebuild one native ARCEngine level entirely from JSON data."""
    sprites = []
    for index, item in enumerate(spec["pieces"]):
        sprites.append(Sprite(
            pixels=[[int(value) for value in row] for row in item["pixels"]],
            name=str(item.get("name", f"cn04_generated_{index}")),
            x=int(item["x"]),
            y=int(item["y"]),
            rotation=int(item.get("rotation", 0)),
            layer=int(item.get("layer", index)),
            visible=bool(item.get("visible", True)),
            collidable=True,
            tags=[names.CLICK_TAG],
        ))
    data = {
        names.KEY_BACKGROUND: int(spec.get("background", names.GREY)),
        names.KEY_MAX_STEPS: int(spec["max_steps"]),
        names.KEY_GREY_MASKING: bool(spec.get("grey_masking", False)),
    }
    grid_size = tuple(int(value) for value in spec.get("grid_size", names.GRID_SIZE))
    return Level(sprites=sprites, grid_size=grid_size, data=data,
                 name=str(spec.get("name", "generated-cn04")))


def _append_action(layout, state, actions, action):
    nxt, _, checks_win = transition(layout, state, action)
    actions.append(action)
    return nxt, checks_win


def _construct_solution(spec):
    env = Env([build_level(spec)])
    env.reset()
    layout = extract(env)
    state = layout.start
    actions = []
    targets = spec["constructed_target"]
    for group_index, target in enumerate(targets):
        if state[2] != group_index:
            click = layout.click_for(state, group_index)
            if click is None:
                return None
            state, _ = _append_action(
                layout, state, actions, (names.ACTION_CLICK, click[0], click[1])
            )
        desired = int(target[3])
        guard = 0
        while state[1][group_index] != desired and guard < 20:
            state, _ = _append_action(
                layout, state, actions, (names.ACTION_ROTATE, None, None)
            )
            guard += 1
        if state[1][group_index] != desired:
            return None
        target_rotation = int(target[2]) // 90
        guard = 0
        while state[0][group_index][2] != target_rotation and guard < 4:
            state, _ = _append_action(
                layout, state, actions, (names.ACTION_ROTATE, None, None)
            )
            guard += 1
        if state[0][group_index][2] != target_rotation:
            return None
        tx, ty = int(target[0]), int(target[1])
        while state[0][group_index][0] < tx:
            state, _ = _append_action(layout, state, actions,
                                      (names.ACTION_RIGHT, None, None))
        while state[0][group_index][0] > tx:
            state, _ = _append_action(layout, state, actions,
                                      (names.ACTION_LEFT, None, None))
        while state[0][group_index][1] < ty:
            state, _ = _append_action(layout, state, actions,
                                      (names.ACTION_DOWN, None, None))
        while state[0][group_index][1] > ty:
            state, _ = _append_action(layout, state, actions,
                                      (names.ACTION_UP, None, None))
        if layout.complete(state) and group_index != len(targets) - 1:
            return None
    return actions if layout.complete(state) else None


def _normalized_cells(pixels):
    array = np.asarray(pixels, dtype=int)
    cells = []
    for y, x in np.argwhere(array >= 0):
        value = int(array[y, x])
        kind = value if value in (0, 3, 8, 13) else 1
        cells.append((int(x), int(y), kind))
    return cells


def _transform_point(x, y, variant):
    swap, sx, sy = variant
    return (sx * (y if swap else x), sy * (x if swap else y))


def geometry_hashes(spec):
    """Hash the actual initial spatial state, independent of its certificate."""
    variants = []
    raw = []
    canonical = {
        order: (group_index, alternate_index)
        for group_index, group in enumerate(native_semantic_groups(spec))
        for alternate_index, (order, _) in enumerate(group)
    }
    for order, piece in enumerate(spec["pieces"]):
        group, alternate = canonical[order]
        rotation = int(piece.get("rotation", 0))
        rendered = _rendered(piece["pixels"], rotation)
        for x, y, kind in _normalized_cells(rendered.tolist()):
            raw.append((group, alternate, x + int(piece["x"]),
                        y + int(piece["y"]), kind))
    for variant in ((swap, sx, sy) for swap in (False, True)
                    for sx in (-1, 1) for sy in (-1, 1)):
        transformed = [
            (group, alternate, *_transform_point(x, y, variant), kind)
            for group, alternate, x, y, kind in raw
        ]
        min_x = min(item[2] for item in transformed)
        min_y = min(item[3] for item in transformed)
        variants.append(sorted(
            (group, alternate, x - min_x, y - min_y, kind)
            for group, alternate, x, y, kind in transformed
        ))
    raw_payload = {
        "grid_size": [int(value) for value in spec["grid_size"]],
        "cells": sorted(raw),
    }
    raw_hash = hashlib.sha256(json.dumps(
        raw_payload, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    d4_hash = hashlib.sha256(json.dumps(min(variants), separators=(",", ":")).encode()).hexdigest()
    return raw_hash, d4_hash


def geometry_split(d4_hash):
    return SPLITS[int(d4_hash[:8], 16) % len(SPLITS)]


def gameplay_hash(spec):
    """Identity of the playable initial state, excluding private witness data."""
    pieces = []
    canonical = {
        order: (group_index, alternate_index)
        for group_index, group in enumerate(native_semantic_groups(spec))
        for alternate_index, (order, _) in enumerate(group)
    }
    for order, piece in enumerate(spec["pieces"]):
        pixels = np.asarray(piece["pixels"], dtype=int)
        if pixels.ndim != 2:
            raise ValueError("piece pixels must be a rectangular 2D array")
        pieces.append({
            "order": order,
            "group": canonical[order][0],
            "alternate": canonical[order][1],
            "pattern": {
                "shape": [int(pixels.shape[1]), int(pixels.shape[0])],
                "cells": _normalized_cells(pixels.tolist()),
            },
            "initial_transform": [
                int(piece["x"]), int(piece["y"]), int(piece.get("rotation", 0)),
            ],
            "layer": int(piece.get("layer", order)),
            "visible": bool(piece.get("visible", True)),
        })
    payload = {
        "source_id": SOURCE_ID,
        "difficulty": spec["difficulty"],
        "grid_size": [int(value) for value in spec["grid_size"]],
        "pieces": pieces,
        "max_steps": spec["max_steps"],
        "grey_masking": spec["grey_masking"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _sprite_set_hash_from_rows(rows):
    sprites = []
    for row in rows:
        cells = _normalized_cells(row)
        variants = []
        for variant in ((swap, sx, sy) for swap in (False, True)
                        for sx in (-1, 1) for sy in (-1, 1)):
            points = [(*_transform_point(x, y, variant), kind) for x, y, kind in cells]
            min_x, min_y = min(x for x, _, _ in points), min(y for _, y, _ in points)
            variants.append(sorted((x - min_x, y - min_y, kind) for x, y, kind in points))
        sprites.append(min(variants))
    return hashlib.sha256(json.dumps(sorted(sprites), separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def _official_sprite_set_hashes():
    return frozenset(_sprite_set_hash_from_rows([sprite.pixels.tolist()
                                                 for sprite in level.get_sprites()])
                     for level in official_levels())


def _candidate_sprite_set_hash(spec):
    return _sprite_set_hash_from_rows([piece["pixels"] for piece in spec["pieces"]])


def _solution_constraints(layout, actions):
    state = layout.start
    for action in actions:
        state, _, _ = transition(layout, state, tuple(action))
    if not layout.complete(state):
        raise ValueError("solution does not end in an exact pin assembly")
    matches = {}
    winning_pin_counts = [0] * len(layout.groups)
    for group_index, (x, y, rotation) in enumerate(state[0]):
        piece = layout.piece(state, group_index)
        for dx, dy, colour in piece.pins(rotation):
            winning_pin_counts[group_index] += 1
            matches.setdefault((x + dx, y + dy, colour), []).append(group_index)
    if not matches or any(len(groups) != 2 or groups[0] == groups[1]
                          for groups in matches.values()):
        raise ValueError("solution relations are not exact two-piece matches")
    edges = tuple(sorted(
        (min(groups), max(groups), colour, x, y)
        for (x, y, colour), groups in matches.items()
    ))
    reached = {0}
    while True:
        changed = reached | {
            right for left, right, _, _, _ in edges if left in reached
        } | {
            left for left, right, _, _, _ in edges if right in reached
        }
        if changed == reached:
            break
        reached = changed
    topology = sorted((left, right, colour) for left, right, colour, _, _ in edges)
    parallel_counts = tuple(sorted(Counter(
        (left, right) for left, right, _, _, _ in edges
    ).values(), reverse=True))
    variants = []
    for variant in ((swap, sx, sy) for swap in (False, True)
                    for sx in (-1, 1) for sy in (-1, 1)):
        transformed = [
            (left, right, colour, *_transform_point(x, y, variant))
            for left, right, colour, x, y in edges
        ]
        min_x = min(item[3] for item in transformed)
        min_y = min(item[4] for item in transformed)
        variants.append(sorted(
            (left, right, colour, x - min_x, y - min_y)
            for left, right, colour, x, y in transformed
        ))
    return {
        "winning_pin_counts": winning_pin_counts,
        "winning_alternates": list(state[1]),
        "relation_edges": len(edges),
        "parallel_relation_counts": list(parallel_counts),
        "connected": len(reached) == len(layout.groups),
        "topology_sha256": hashlib.sha256(json.dumps(
            topology, separators=(",", ":"),
        ).encode()).hexdigest(),
        "relation_geometry_sha256": hashlib.sha256(json.dumps(
            min(variants), separators=(",", ":"),
        ).encode()).hexdigest(),
    }


def certify(spec, *, requested_split=None):
    """Return a fully evidenced row and rejection reason."""
    if profile_errors(spec, require_proof=False):
        return None, "profile_structure"
    constructive_actions = _construct_solution(spec)
    if constructive_actions is None:
        return None, "constructive_witness"
    raw_hash, d4_hash = geometry_hashes(spec)
    partition = geometry_split(d4_hash)
    if requested_split is not None and partition != requested_split:
        return None, "geometry_split"
    spec.update({
        "split": partition,
        "geometry_sha256": raw_hash,
        "geometry_d4_sha256": d4_hash,
        "geometry_split": partition,
        "official_sprite_set_sha256": _candidate_sprite_set_hash(spec),
    })
    if spec["official_sprite_set_sha256"] in _official_sprite_set_hashes():
        return None, "official_copy"

    difficulty = spec["difficulty"]
    levels = [build_level(spec) for _ in range(difficulty)]
    env = Env(levels)
    # Cn04.on_set_level destructively remaps displayed pin pixels. Calling
    # set_level(0) again on a fresh instance is therefore not idempotent; tier
    # one is already in its intended context after construction.
    if difficulty > 1:
        env.set_level(difficulty - 1)
    if env.level_index != difficulty - 1:
        return None, "context_index"
    # Certify the same private-metadata-free route that the shared collector
    # will request.  The constructive target above is only a bounded draft
    # feasibility check; it is never exposed as the teacher certificate.
    teacher = search(env, limit=PROFILES[difficulty]["search_work"])
    if not teacher.solved:
        return None, "teacher_search_truncated" if teacher.truncated else "teacher_search"
    actions = teacher.actions
    low, high = PROFILES[difficulty]["witness_actions"]
    if not low <= len(actions) <= high:
        return None, "reference_action_length"
    spec.update({
        "solution": [list(action) for action in actions],
        "solution_length": len(actions),
    })
    layout = extract(env)
    state = layout.start
    before_score = env.levels_completed
    for index, action in enumerate(actions):
        state, _, _ = transition(layout, state, action)
        observation = env.perform(*action)
        if index < len(actions) - 1:
            if observation.finished or env.levels_completed != before_score:
                return None, "early_completion"
            if extract(env).start != state:
                return None, "transition_mismatch"
    if not observation.won or env.levels_completed != before_score + 1:
        return None, "native_replay"
    mechanics = trace(layout, actions)
    try:
        constraints = _solution_constraints(layout, actions)
    except ValueError:
        return None, "constraint_evidence"
    usable_budget = spec["max_steps"] - 1
    remaining = usable_budget - len(actions)
    spec.update({
        "engine_verified": True,
        "context_engine_verified": True,
        "search_truncated": False,
        "search_limit": PROFILES[difficulty]["search_work"],
        "reachable_states": teacher.expanded,
        "solution_mechanics": mechanics,
        "solution_constraints": constraints,
        "native_budget": {
            "max_steps": spec["max_steps"],
            "usable_actions": usable_budget,
            "witness_actions": len(actions),
            "remaining_actions": remaining,
            "loss_boundary_preserved": True,
        },
        "proof": {
            "backend": "bounded-private-metadata-free-teacher+exact-native-differential",
            "route_kind": "floor-independent-enumerated-assembly-teacher-witness",
            "optimal_actions": None,
            "search_truncated": False,
            "search_limit": PROFILES[difficulty]["search_work"],
            "context_index": difficulty - 1,
            "verification_level_index": difficulty - 1,
            "engine_win": True,
            "levels_completed": 1,
            "native_transition_match": True,
            "mechanics_version": MECHANICS_VERSION,
            "generator_version": GENERATOR_VERSION,
            "expanded": teacher.expanded,
            "teacher_reason": teacher.reason,
            "bounded_assignment_caps": teacher.assignment_caps,
            "assignment_work_limit": GENERATED_ASSIGNMENT_LIMIT,
        },
        "official_copy": False,
    })
    spec["gameplay_sha256"] = gameplay_hash(spec)
    errors = profile_errors(spec)
    if errors:
        return None, "profile_proof: " + "; ".join(errors)
    return _jsonable(spec), None


def generate(seed, difficulty, *, split, attempts=120, record_rejection=None):
    """Generate one tier in its canonical train/validation/test partition."""
    if type(seed) is not int or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be an integer in 1..6")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    if type(attempts) is not int or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    exclusions = Counter()
    for attempt in range(1, attempts + 1):
        draft = _draft(seed, difficulty, attempt)
        if draft is None:
            reason = "invalid_geometry"
            accepted = None
        else:
            accepted, reason = certify(draft, requested_split=split)
        if accepted is not None:
            accepted["generation_exclusions"] = dict(exclusions)
            accepted["proof"]["split"] = split
            accepted["proof"]["geometry_d4_sha256"] = accepted["geometry_d4_sha256"]
            generate.last_rejections = dict(exclusions)
            return accepted
        exclusions[reason] += 1
        if record_rejection is not None:
            record_rejection({
                "seed": seed,
                "difficulty": difficulty,
                "attempt": attempt,
                "split": split,
                "reason": reason,
                "generator_version": GENERATOR_VERSION,
                "mechanics_version": MECHANICS_VERSION,
            })
    generate.last_rejections = dict(exclusions)
    return None


generate.last_rejections = {}


def _child_seed(game_seed, ordinal, difficulty):
    if type(game_seed) is not int:
        raise ValueError("game seed must be an integer")
    payload = f"{SOURCE_ID}:{MECHANICS_VERSION}:{game_seed}:{ordinal}:{difficulty}"
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")


def build_game(specs):
    """Validate, replay, and build exactly one ordered six-tier native episode."""
    try:
        specs = list(specs)
    except TypeError as error:
        raise ValueError("full CN04 game specs must be an iterable of mappings") from error
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("full CN04 games require exactly six specs")
    if not all(isinstance(spec, Mapping) for spec in specs):
        raise ValueError("full CN04 game specs must be mappings")
    seeds = set()
    split = specs[0].get("split") if specs else None
    levels = []
    identities = {"geometry_d4_sha256": set(), "gameplay_sha256": set()}
    for index, (difficulty, spec) in enumerate(zip(DIFFICULTIES, specs)):
        if spec.get("difficulty") != difficulty:
            raise ValueError("full CN04 game difficulties must be ordered 1..6")
        if spec.get("training_context_index") != index or spec.get("verification_level_index") != index:
            raise ValueError("spec context/index does not match its native position")
        if spec.get("split") != split:
            raise ValueError("all full-game specs must use the same split")
        child_seed = spec.get("seed")
        if type(child_seed) is not int:
            raise ValueError("full-game child seeds must be integers")
        if child_seed in seeds:
            raise ValueError("full-game child seeds must be distinct")
        seeds.add(child_seed)
        for field, seen in identities.items():
            identity = spec.get(field)
            if not isinstance(identity, str) or len(identity) != 64:
                raise ValueError(f"full-game specs require valid {field}")
            if identity in seen:
                raise ValueError(f"full-game {field} identities must be distinct")
            seen.add(identity)
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"invalid tier {difficulty}: {'; '.join(errors)}")
        levels.append(build_level(spec))

    env = Env(levels)
    env.reset()
    observation = None
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            raise ValueError(f"native full-game context did not reach tier {index + 1}")
        before = env.levels_completed
        for action_index, action in enumerate(spec["solution"]):
            observation = env.perform(*action)
            if action_index < len(spec["solution"]) - 1 and (
                observation.finished or env.levels_completed != before
            ):
                raise ValueError(f"stored witness completed tier {index + 1} early")
        if observation is None or env.levels_completed != before + 1:
            raise ValueError(f"stored witness failed tier {index + 1} in the full game")
    if not observation.won or env.levels_completed != len(DIFFICULTIES):
        raise ValueError("native full-game replay did not finish all six tiers")
    return levels


def generate_game(seed, *, split, difficulties=None, attempts=320):
    """Generate an increasing-difficulty sequence with stable BLAKE2 child seeds."""
    if type(seed) is not int:
        raise ValueError("game seed must be an integer")
    if difficulties is None:
        selected = DIFFICULTIES
    else:
        if isinstance(difficulties, (str, bytes)):
            raise ValueError("difficulties must be a sequence")
        try:
            selected = tuple(difficulties)
        except TypeError as error:
            raise ValueError("difficulties must be a sequence") from error
    if not selected or any(type(value) is not int or value not in DIFFICULTIES
                           for value in selected):
        raise ValueError("difficulties must be a nonempty sequence drawn from 1..6")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("difficulties must be unique and increasing")
    specs = []
    for ordinal, difficulty in enumerate(selected):
        child = _child_seed(seed, ordinal, difficulty)
        spec = generate(child, difficulty, split=split, attempts=attempts)
        if spec is None:
            return None
        spec["game_seed"] = seed
        spec["game_ordinal"] = ordinal
        spec["game_size"] = len(selected)
        specs.append(spec)
    if selected != DIFFICULTIES:
        return specs

    build_game(specs)
    for index, spec in enumerate(specs):
        spec["proof"]["full_game_replay"] = True
        spec["proof"]["full_game_context_index"] = index
    return _jsonable(specs)


def validate_full_standard(spec, curriculum_entry):
    """Fail-closed family validator used by collection and ``build_game``."""
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    if not isinstance(curriculum_entry, Mapping):
        return ["curriculum entry must be a mapping"]
    schema_errors = primitive_schema_errors(spec)
    if schema_errors:
        return schema_errors
    try:
        errors = list(profile_errors(spec))
    except Exception as error:  # malformed external JSON must fail closed
        errors = [f"malformed profile data: {error}"]
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        errors.append("difficulty must be an integer in 1..6")
        return errors
    expected = PROFILES.get(difficulty)
    if not _exact_tree_equal(
        curriculum_entry, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
    ):
        errors.append("curriculum entry differs from the published six-tier contract")
    for key, value in (
        ("format", FORMAT),
        ("source", "generated_only"),
        ("source_id", SOURCE_ID),
        ("generator_version", GENERATOR_VERSION),
        ("mechanics_version", MECHANICS_VERSION),
        ("difficulty_version", DIFFICULTY_VERSION),
        ("quality_profile_version", QUALITY_PROFILE_VERSION),
        ("geometry_version", GEOMETRY_VERSION),
        ("split_version", SPLIT_VERSION),
    ):
        if type(spec.get(key)) is not type(value) or spec.get(key) != value:
            errors.append(f"{key} mismatch")
    if spec.get("split") not in SPLITS or spec.get("geometry_split") != spec.get("split"):
        errors.append("explicit split and canonical geometry split must agree")
    for key in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256"):
        value = spec.get(key)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"missing {key}")
    try:
        metrics = structural_metrics(spec)
        for key, recomputed in metrics.items():
            stored = spec.get(key)
            if key in ("stack_sizes", "visible_pin_colours"):
                stored = tuple(stored) if isinstance(stored, (list, tuple)) else stored
            if not _exact_tree_equal(stored, recomputed):
                errors.append(f"stored {key} differs from recomputed structure")
        raw_hash, d4_hash = geometry_hashes(spec)
        if spec.get("geometry_sha256") != raw_hash:
            errors.append("raw geometry identity differs from recomputation")
        if spec.get("geometry_d4_sha256") != d4_hash:
            errors.append("D4 geometry identity differs from recomputation")
        partition = geometry_split(d4_hash)
        if partition != spec.get("split"):
            errors.append("recomputed geometry belongs to a different split")
        sprite_hash = _candidate_sprite_set_hash(spec)
        if spec.get("official_sprite_set_sha256") != sprite_hash:
            errors.append("sprite-set identity differs from recomputation")
        if sprite_hash in _official_sprite_set_hashes() or spec.get("official_copy") is not False:
            errors.append("official-copy rejection evidence is missing or contradicted")
        if spec.get("gameplay_sha256") != gameplay_hash(spec):
            errors.append("gameplay identity differs from recomputation")
        foreground = _initial_foreground_colours(
            spec["pieces"], bool(spec.get("grey_masking", False)),
        )
        if spec.get("background") not in BACKGROUND_COLOURS:
            errors.append("background colour is outside the calibrated palette")
        elif spec.get("background") in foreground:
            errors.append("initial sprite foreground is not readable against the background")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"identity/structure recomputation failed: {error}")
    exclusions = spec.get("generation_exclusions")
    if not isinstance(exclusions, dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in (exclusions.items() if isinstance(exclusions, dict) else ())
    ):
        errors.append("bounded rejection counters are missing")
    for key in ("seed", "effective_seed"):
        if type(spec.get(key)) is not int:
            errors.append(f"{key} must be an integer")
    if spec.get("effective_seed") != spec.get("seed"):
        errors.append("effective seed differs from the generated seed")
    if type(spec.get("generation_attempt")) is not int or spec.get("generation_attempt", 0) < 1:
        errors.append("generation attempt must be a positive integer")
    if type(spec.get("training_context_index")) is not int or spec.get("training_context_index") != difficulty - 1:
        errors.append("training context differs from official tier context")
    if type(spec.get("verification_level_index")) is not int or spec.get("verification_level_index") != difficulty - 1:
        errors.append("verification context differs from official tier context")
    for key in ("engine_verified", "context_engine_verified"):
        if spec.get(key) is not True:
            errors.append(f"{key} must be true")
    if spec.get("search_truncated") is not False:
        errors.append("top-level search_truncated must be false")
    if type(spec.get("search_limit")) is not int or spec.get("search_limit") != expected["search_work"]:
        errors.append("top-level search limit differs from the tier work bound")
    if type(spec.get("reachable_states")) is not int or spec.get("reachable_states", -1) < 0:
        errors.append("reachable_states must be a nonnegative integer")
    proof = spec.get("proof", {})
    if not isinstance(proof, Mapping):
        errors.append("proof must be a mapping")
        proof = {}
    expected_proof = {
        "backend": "bounded-private-metadata-free-teacher+exact-native-differential",
        "route_kind": "floor-independent-enumerated-assembly-teacher-witness",
        "optimal_actions": None,
        "search_truncated": False,
        "search_limit": expected["search_work"],
        "context_index": difficulty - 1,
        "verification_level_index": difficulty - 1,
        "engine_win": True,
        "levels_completed": 1,
        "native_transition_match": True,
        "mechanics_version": MECHANICS_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split": spec.get("split"),
        "geometry_d4_sha256": spec.get("geometry_d4_sha256"),
        "expanded": spec.get("reachable_states"),
        "teacher_reason": proof.get("teacher_reason"),
        "bounded_assignment_caps": proof.get("bounded_assignment_caps"),
        "assignment_work_limit": GENERATED_ASSIGNMENT_LIMIT,
    }
    if any(key not in proof
           or type(proof[key]) is not type(value)
           or proof[key] != value
           for key, value in expected_proof.items()):
        errors.append("native contextual replay proof is incomplete")
    if not isinstance(proof.get("teacher_reason"), str) or not proof.get("teacher_reason"):
        errors.append("teacher reason must be a nonempty string")
    if (type(proof.get("bounded_assignment_caps")) is not int
            or proof.get("bounded_assignment_caps", -1) < 0):
        errors.append("bounded assignment-cap count must be a nonnegative integer")
    if "full_game_replay" in proof and proof.get("full_game_replay") is not True:
        errors.append("full-game replay flag must be true when present")
    if "full_game_context_index" in proof and (
        type(proof.get("full_game_context_index")) is not int
        or proof.get("full_game_context_index") != difficulty - 1
    ):
        errors.append("full-game proof context differs from the native tier")
    budget = spec.get("native_budget", {})
    solution = spec.get("solution")
    if not isinstance(solution, list) or not solution:
        errors.append("solution must be a nonempty action list")
        solution = []
    if type(spec.get("solution_length")) is not int or spec.get("solution_length") != len(solution):
        errors.append("solution length differs from the stored action sequence")
    usable = expected["max_steps"] - 1
    remaining = usable - len(solution)
    recomputed_budget = {
        "max_steps": expected["max_steps"],
        "usable_actions": usable,
        "witness_actions": len(solution),
        "remaining_actions": remaining,
        "loss_boundary_preserved": True,
    }
    if (
        not isinstance(budget, dict)
        or any(type(budget.get(key)) is not type(value) for key, value in recomputed_budget.items())
        or budget != recomputed_budget
        or remaining < 0
    ):
        errors.append("native budget evidence is incomplete")

    try:
        actions = []
        for action in solution:
            if not isinstance(action, (list, tuple)) or len(action) != 3:
                raise ValueError("each action must be a triple")
            action_id, x, y = action
            if type(action_id) is not int or action_id not in names.ACTION_IDS:
                raise ValueError("unknown action id")
            if action_id == names.ACTION_CLICK:
                if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
                    raise ValueError("click coordinates must be display integers")
            elif x is not None or y is not None:
                raise ValueError("non-click actions must have null coordinates")
            actions.append((action_id, x, y))

        levels = [build_level(spec) for _ in range(difficulty)]
        env = Env(levels)
        if difficulty > 1:
            env.set_level(difficulty - 1)
        if env.level_index != difficulty - 1:
            raise ValueError("native context index mismatch")
        layout = extract(env)
        state = layout.start
        before_score = env.levels_completed
        for index, action in enumerate(actions):
            state, _, _ = transition(layout, state, action)
            observation = env.perform(*action)
            if index < len(actions) - 1:
                if observation.finished or env.levels_completed != before_score:
                    raise ValueError("route completes before its final action")
                if extract(env).start != state:
                    raise ValueError("symbolic/native transition mismatch")
        if not actions or not observation.won or env.levels_completed != before_score + 1:
            raise ValueError("route does not win in the intended native context")
        mechanics = trace(layout, actions)
        if not _exact_tree_equal(spec.get("solution_mechanics"), mechanics):
            errors.append("stored solution mechanics differ from route simulation")
        constraints = _solution_constraints(layout, actions)
        if not _exact_tree_equal(spec.get("solution_constraints"), constraints):
            errors.append("stored solution constraints differ from route simulation")

        # Re-run the bounded public-state teacher from a fresh native context.
        # This makes the claimed work and bounded-candidate diagnostics
        # evidence rather than trusted metadata.
        teacher_levels = [build_level(spec) for _ in range(difficulty)]
        teacher_env = Env(teacher_levels)
        if difficulty > 1:
            teacher_env.set_level(difficulty - 1)
        teacher = search(teacher_env, limit=expected["search_work"])
        if (
            not teacher.solved
            or teacher.truncated
            or teacher.unsupported
            or teacher.actions != actions
            or teacher.expanded != spec.get("reachable_states")
            or teacher.reason != proof.get("teacher_reason")
            or teacher.assignment_caps != proof.get("bounded_assignment_caps")
        ):
            errors.append("bounded teacher route/work diagnostics differ from recomputation")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        errors.append(f"native witness recomputation failed: {error}")
    return errors


def load_level(spec):
    if spec.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} spec")
    return build_level(spec)
