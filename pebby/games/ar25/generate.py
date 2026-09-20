"""Full eight-tier AR25 procedural generation and native verification.

Default generation follows the shipped progression. Rotation and axis-limited
reflection have zero incidence in the eight official levels, so they remain
explicit engine-extension parameters and never alter default profiles.
"""

from __future__ import annotations

from collections import Counter, deque
from hashlib import blake2b, sha256
import json
import random
from typing import Mapping, Sequence

from arcengine import GameState, Level, Sprite

from . import names
from .env import Env, official_levels, replay, upstream
from .plan import DEFAULT_LIMIT, _display_point, search


FORMAT = "pebby.ar25.level.v2"
GENERATOR_VERSION = 2
DIFFICULTIES = tuple(range(1, 9))
MAX_ATTEMPTS = 64
ALTERNATIVE_CONFIGURATION_LIMIT = 25_000
SPLITS = ("train", "validation", "test")
GENERATION_MODES = ("official_profile", "engine_extension")
QUALITY_PROFILE_VERSION = "ar25-official-eight-v2"
MECHANICS_INVENTORY_VERSION = "ar25-native-reflection-v1"

# One shipped reference exists per tier. These are explicit engineering
# tolerances, not population confidence intervals. Witness lengths below are
# replayed constructive routes and are not claimed optimal.
#
# ``shape_components`` and ``goal_components`` are 4-connected component counts
# measured on the official levels: official art is solid polyomino work, not
# diagonal chains, and official goal sets are a few solid regions. Tier 6's
# second shape is the one official multi-piece sprite (four pieces). Official
# tiers 4 and 8 place the second shape's target orthogonally touching the first
# (``targets_touch``), which is what merges their goal regions to 2 and 4.
REFERENCE_PROFILES = {
    1: {"steps": 64, "goals": 5, "goal_bbox_range": ((3, 3), (3, 3)), "goal_components": (1, 1), "shape_cells": (5,), "shape_sizes": ((3, 3),), "shape_components": (1,), "targets_touch": False, "mirrors": ("vertical",), "fixed": 1, "official_witness": 15, "witness_range": (8, 30)},
    2: {"steps": 64, "goals": 8, "goal_bbox_range": ((5, 5), (4, 4)), "goal_components": (1, 1), "shape_cells": (8,), "shape_sizes": ((5, 4),), "shape_components": (1,), "targets_touch": False, "mirrors": ("vertical",), "fixed": 0, "official_witness": 11, "witness_range": (6, 32)},
    3: {"steps": 128, "goals": 26, "goal_bbox_range": ((8, 16), (13, 21)), "goal_components": (4, 4), "shape_cells": (7, 6), "shape_sizes": ((4, 4), (4, 2)), "shape_components": (1, 1), "targets_touch": False, "mirrors": ("horizontal",), "fixed": 0, "official_witness": 40, "witness_range": (18, 70)},
    4: {"steps": 128, "goals": 24, "goal_bbox_range": ((4, 10), (11, 17)), "goal_components": (2, 2), "shape_cells": (7, 6), "shape_sizes": ((5, 2), (1, 6)), "shape_components": (1, 1), "targets_touch": True, "mirrors": ("horizontal",), "fixed": 0, "official_witness": 22, "witness_range": (12, 60)},
    5: {"steps": 128, "goals": 23, "goal_bbox_range": ((7, 15), (7, 15)), "goal_components": (3, 3), "shape_cells": (11,), "shape_sizes": ((5, 5),), "shape_components": (1,), "targets_touch": False, "mirrors": ("horizontal", "vertical"), "fixed": 0, "official_witness": 28, "witness_range": (14, 70)},
    6: {"steps": 320, "goals": 52, "goal_bbox_range": ((9, 17), (13, 21)), "goal_components": (20, 20), "shape_cells": (5, 8), "shape_sizes": ((4, 2), (5, 5)), "shape_components": (1, 4), "targets_touch": False, "mirrors": ("horizontal", "vertical"), "fixed": 0, "official_witness": 53, "witness_range": (25, 110)},
    7: {"steps": 320, "goals": 42, "goal_bbox_range": ((9, 17), (9, 17)), "goal_components": (24, 24), "shape_cells": (9, 14), "shape_sizes": ((3, 7), (5, 4)), "shape_components": (1, 1), "targets_touch": False, "mirrors": ("horizontal", "vertical"), "fixed": 0, "official_witness": 37, "witness_range": (18, 95)},
    8: {"steps": 320, "goals": 60, "goal_bbox_range": ((13, 21), (13, 21)), "goal_components": (4, 4), "shape_cells": (8, 7), "shape_sizes": ((5, 3), (3, 5)), "shape_components": (1, 1), "targets_touch": True, "mirrors": ("horizontal", "vertical"), "fixed": 0, "official_witness": 47, "witness_range": (22, 115)},
}

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": "ar25-0c556536",
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": tuple(
        {"difficulty": d, "context_index": d - 1, "search_work": 10_000}
        for d in DIFFICULTIES
    ),
    "evidence": {
        "official_tier_characterization": ".scratch/multigame-resume/full-standard/ar25.md#official-tier-characterization",
        "solution_mechanics": ".scratch/multigame-resume/full-standard/ar25.md#mechanic-participation",
        "native_budget": ".scratch/multigame-resume/full-standard/ar25.md#native-budgets",
        "context_engine_replay": ".scratch/multigame-resume/full-standard/ar25.md#native-replay",
        "novelty_split": ".scratch/multigame-resume/full-standard/ar25.md#identity-and-splits",
        "bounded_rejections": ".scratch/multigame-resume/full-standard/ar25.md#bounded-generation",
        "root_acceptance": ".scratch/multigame-resume/full-standard/ar25.md#root-acceptance",
    },
    "caveats": (
        "Each tier has one official reference; tolerances are engineering bounds, not confidence intervals.",
        "Constructive native-replayed witnesses use private teacher targets and are not public-state target-discovery or shortest-route proofs.",
        "The 25,000-state public shortcut audit counts dequeued configurations; truncation leaves shorter public routes unknown and the calibrated action floor is not a proof of intrinsic difficulty.",
        "Rotation and axis restrictions have zero official incidence and remain explicit extensions.",
    ),
}


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _point(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element coordinate")
    return _integer(value[0], f"{label}.x"), _integer(value[1], f"{label}.y")


def _occupied(mask):
    return {(x, y) for y, row in enumerate(mask) for x, value in enumerate(row) if value == "#"}


def _validate_mask(mask, label):
    if not isinstance(mask, (list, tuple)) or not mask or not all(isinstance(row, str) and row for row in mask):
        raise ValueError(f"{label} must contain nonempty strings")
    width = len(mask[0])
    if any(len(row) != width or set(row) - {"#", "."} for row in mask):
        raise ValueError(f"{label} must be rectangular and use only '#' and '.'")
    if not _occupied(mask):
        raise ValueError(f"{label} must contain an occupied pixel")
    return width, len(mask)


_ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(cells):
    """Count 4-connected components; official art and targets are polyominoes."""
    remaining, count = set(cells), 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for dx, dy in _ORTHOGONAL:
                neighbour = (x + dx, y + dy)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    return count


def _touches(cells, others):
    return any((x + dx, y + dy) in others for x, y in cells for dx, dy in _ORTHOGONAL)


def _spaced_seeds(rng, cells, count):
    """Pick ``count`` mutually non-adjacent seed cells, or None if impossible."""
    seeds = set()
    for cell in rng.sample(sorted(cells), len(cells)):
        if not _touches((cell,), seeds):
            seeds.add(cell)
        if len(seeds) == count:
            return sorted(seeds)
    return None


def _grow_regions(rng, allowed, regions, count):
    """Grow 4-connected regions inside ``allowed`` until ``count`` cells in total.

    Regions never become orthogonally adjacent, so the union has exactly
    ``len(regions)`` components. Returns None when growth is stuck.
    """
    regions = [set(region) for region in regions]
    owner = {cell: index for index, region in enumerate(regions) for cell in region}
    if sum(map(len, regions)) != len(owner) or len(owner) > count:
        return None

    def separated(cell, index):
        return all(owner.get((cell[0] + dx, cell[1] + dy), index) == index for dx, dy in _ORTHOGONAL)

    if any(not separated(cell, index) for cell, index in owner.items()):
        return None
    while len(owner) < count:
        options = sorted({
            (index, (x + dx, y + dy))
            for index, region in enumerate(regions)
            for x, y in region
            for dx, dy in _ORTHOGONAL
            if (x + dx, y + dy) in allowed
            if (x + dx, y + dy) not in owner
            if separated((x + dx, y + dy), index)
        })
        if not options:
            return None
        index, cell = rng.choice(options)
        regions[index].add(cell)
        owner[cell] = index
    return set(owner)


def _spanning_path(rng, width, height):
    """Join one anchor per edge with orthogonal steps so art fills its extent."""
    anchors = [
        (0, rng.randrange(height)),
        (width - 1, rng.randrange(height)),
        (rng.randrange(width), 0),
        (rng.randrange(width), height - 1),
    ]
    rng.shuffle(anchors)
    cells = {anchors.pop()}
    for tx, ty in anchors:
        x, y = min(cells, key=lambda cell: abs(cell[0] - tx) + abs(cell[1] - ty))
        while (x, y) != (tx, ty):
            steps = [(1 if tx > x else -1, 0)] if x != tx else []
            if y != ty:
                steps.append((0, 1 if ty > y else -1))
            dx, dy = rng.choice(steps)
            x, y = x + dx, y + dy
            cells.add((x, y))
    return cells


def _random_mask(rng, count, size, components=1):
    """Create polyomino art with the official tier's extent and piece count."""
    width, height = size
    if count < max(width, height) or count > width * height:
        raise ValueError("shape cell count cannot span its calibrated extent")
    if not 1 <= components <= count:
        raise ValueError("shape component count must be within 1..cells")
    board = {(x, y) for x in range(width) for y in range(height)}
    for _ in range(1_000):
        if components == 1:
            regions = [_spanning_path(rng, width, height)]
        else:
            seeds = _spaced_seeds(rng, board, components)
            if seeds is None:
                continue
            regions = [{cell} for cell in seeds]
        cells = _grow_regions(rng, board, regions, count)
        if cells is None or _bbox_size(cells) != (width, height) or _components(cells) != components:
            continue
        return tuple(
            "".join("#" if (x, y) in cells else "." for x in range(width))
            for y in range(height)
        )
    raise ValueError("failed to construct a calibrated polyomino shape mask")


def _square_mask(mask):
    size = max(len(mask), len(mask[0]))
    return tuple(row.ljust(size, ".") for row in mask) + tuple("." * size for _ in range(size - len(mask)))


def _coverage_paths(position, occupied, mirrors, reflection="both"):
    """Match native visited-cell BFS, including expansion from depth 12."""
    result, queue = {}, deque()
    for dx, dy in sorted(occupied):
        cell = (position[0] + dx, position[1] + dy)
        result[cell] = ()
        queue.append((cell, (), 0))
    while queue:
        (x, y), path, depth = queue.popleft()
        if depth > 12:
            continue
        for orientation, coordinate in mirrors:
            if reflection != "both" and reflection != orientation:
                continue
            cell = (2 * coordinate - x, y) if orientation == "vertical" else (x, 2 * coordinate - y)
            if cell in result:
                continue
            next_path = path + (orientation,)
            result[cell] = next_path
            queue.append((cell, next_path, depth + 1))
    return {cell: path for cell, path in result.items() if 0 <= cell[0] < names.GRID and 0 <= cell[1] < names.GRID}


def _target_geometry(rng, masks, mirror_targets, reflections, touching=False):
    """Place targets; later shapes touch earlier images only when the tier does."""
    positions, path_maps, used = [], [], set()
    multiplier = 2 ** len({orientation for orientation, _ in mirror_targets})
    for mask, reflection in zip(masks, reflections):
        occupied = _occupied(mask)
        expected = len(occupied) * (multiplier if reflection == "both" else 2)
        width, height = len(mask[0]), len(mask)
        candidates = [(x, y) for y in range(names.GRID - height + 1) for x in range(names.GRID - width + 1)]
        rng.shuffle(candidates)
        chosen = None
        for position in candidates:
            paths = _coverage_paths(position, occupied, mirror_targets, reflection)
            if len(paths) != expected or set(paths).intersection(used) or not any(paths.values()):
                continue
            if used and _touches(paths, used) != touching:
                continue
            chosen = position, paths
            break
        if chosen is None:
            return None
        position, paths = chosen
        positions.append(position)
        path_maps.append(paths)
        used.update(paths)
    return positions, path_maps


def _goal_requirements_met(goals, path_maps):
    """Every shape contributes and every available reflection depth/axis appears."""
    if any(not goals.intersection(paths) for paths in path_maps):
        return False
    chosen = [paths[goal] for paths in path_maps for goal in goals if goal in paths]
    available = [path for paths in path_maps for path in paths.values()]
    if max(map(len, chosen)) < min(2, max(map(len, available))):
        return False
    return {axis for path in available for axis in path} <= {axis for path in chosen for axis in path}


def _choose_goals(rng, path_maps, count, mirror_targets, component_range):
    """Grow solid 4-connected goal regions over the reflection union.

    Official goal sets are a few polyomino regions of the reflected images,
    not scattered orbit cells; the region count is pinned per tier.
    """
    union = set().union(*(set(paths) for paths in path_maps))
    if len(union) < count:
        return None
    if count == len(union):
        return sorted(union)
    direct_sets = [
        {cell for cell, path in paths.items() if not path}
        for paths in path_maps
    ]
    # Official tiers 1-2 display the reflected silhouette as the target.
    if len(mirror_targets) == 1 and count == sum(map(len, direct_sets)):
        goals = set()
        for paths, direct in zip(path_maps, direct_sets):
            for cell in direct:
                reflected = [
                    member
                    for member in _coverage_paths(cell, {(0, 0)}, mirror_targets)
                    if member in paths and member != cell
                ]
                goals.add(reflected[0])
        return sorted(goals)

    low, high = component_range
    for _ in range(50):
        seeds = _spaced_seeds(rng, union, rng.randint(low, high))
        if seeds is None:
            continue
        goals = _grow_regions(rng, union, [{seed} for seed in seeds], count)
        if goals is not None and _goal_requirements_met(goals, path_maps):
            return sorted(goals)
    return None


def _displaced_coordinate(rng, target, distance=None):
    distances = [distance] if distance is not None else list(range(2, 7))
    choices = [target + sign * delta for delta in distances for sign in (-1, 1) if 0 <= target + sign * delta < names.GRID]
    return rng.choice(choices)


def _bbox_size(points):
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _in_bbox_range(points, profile):
    width, height = _bbox_size(points)
    width_range, height_range = profile["goal_bbox_range"]
    return width_range[0] <= width <= width_range[1] and height_range[0] <= height <= height_range[1]


def _goal_components_in_range(points, profile):
    low, high = profile["goal_components"]
    return low <= _components(tuple(map(tuple, points))) <= high


def _initial_positions(rng, masks, targets, initial_mirrors, reflections, goals, rotation):
    positions, used = [], set()
    for index, (mask, target) in enumerate(zip(masks, targets)):
        width, height = len(mask[0]), len(mask)
        candidates = [(x, y) for y in range(names.GRID - height + 1) for x in range(names.GRID - width + 1) if abs(x - target[0]) + abs(y - target[1]) >= 3]
        if index == 0 and rotation == "vertical":
            candidates = [p for p in candidates if abs(p[0] - target[0]) == 4 and abs(p[1] - target[1]) >= 1]
        if index == 0 and rotation == "horizontal":
            candidates = [p for p in candidates if abs(p[1] - target[1]) == 4 and abs(p[0] - target[0]) >= 1]
        rng.shuffle(candidates)
        occupied = _occupied(mask)
        chosen = None
        for position in candidates:
            cells = {(position[0] + x, position[1] + y) for x, y in occupied}
            paths = _coverage_paths(position, occupied, initial_mirrors, reflections[index])
            has_reflection = any(path for path in paths.values())
            has_recursive = any(len(path) >= 2 for path in paths.values())
            needs_recursive = len(initial_mirrors) > 1 and reflections[index] == "both"
            if (
                not cells.intersection(used)
                and not set(goals) <= set(paths)
                and has_reflection
                and (not needs_recursive or has_recursive)
            ):
                chosen = position
                used.update(cells)
                break
        if chosen is None:
            return None
        positions.append(chosen)
    return positions


def _seed_for_attempt(seed, difficulty, attempt):
    raw = f"ar25-v2:{int(seed)}:{difficulty}:{attempt}".encode()
    return int.from_bytes(blake2b(raw, digest_size=8).digest(), "big")


def _draft(seed, difficulty, attempt, *, reflection_restriction=None, rotation=None):
    profile = REFERENCE_PROFILES[difficulty]
    rng = random.Random(_seed_for_attempt(seed, difficulty, attempt))
    extension = reflection_restriction is not None or rotation is not None
    if reflection_restriction not in (None, "horizontal", "vertical"):
        raise ValueError("reflection_restriction must be horizontal, vertical, or None")
    if rotation not in (None, "horizontal", "vertical"):
        raise ValueError("rotation must be horizontal, vertical, or None")
    if reflection_restriction is not None and difficulty < 5:
        raise ValueError("reflection restrictions require difficulty >= 5")
    if rotation is not None and rotation not in profile["mirrors"]:
        raise ValueError("rotation requires a matching mirror orientation")

    masks = [
        _random_mask(rng, count, size, components)
        for count, size, components in zip(
            profile["shape_cells"], profile["shape_sizes"], profile["shape_components"],
        )
    ]
    if rotation is not None:
        masks[0] = _square_mask(masks[0])
    target_mirrors = [(orientation, rng.randint(8, 12)) for orientation in profile["mirrors"]]
    reflections = ["both"] * len(masks)
    if reflection_restriction is not None:
        reflections[0] = reflection_restriction
    geometry = _target_geometry(rng, masks, target_mirrors, reflections, profile["targets_touch"])
    if geometry is None:
        return None
    targets, path_maps = geometry
    union = set().union(*(set(paths) for paths in path_maps))
    goal_count = profile["goals"] if not extension else min(profile["goals"], len(union))
    for _ in range(8):
        goals = _choose_goals(rng, path_maps, goal_count, target_mirrors, profile["goal_components"])
        if goals is None:
            return None
        if extension or (_in_bbox_range(goals, profile) and _goal_components_in_range(goals, profile)):
            break
    else:
        return None

    mirrors, initial_mirrors = [], []
    for index, (orientation, target) in enumerate(target_mirrors):
        fixed = bool(profile["fixed"] and index == 0)
        rotation_distance = 4 if rotation == orientation and not fixed else None
        initial = target if fixed else _displaced_coordinate(rng, target, rotation_distance)
        mirrors.append({"orientation": orientation, "initial_coordinate": initial, "target_coordinate": target, "fixed": fixed})
        initial_mirrors.append((orientation, initial))
    initials = _initial_positions(rng, masks, targets, initial_mirrors, reflections, goals, rotation)
    if initials is None:
        return None
    shapes = []
    for index, (mask, initial, target, reflection) in enumerate(zip(masks, initials, targets, reflections)):
        shapes.append({
            "mask": list(mask), "color": 5,
            "initial_position": list(initial), "target_position": list(target),
            "reflection": reflection, "rotation": rotation if index == 0 else None,
        })
    selection_mode = "click" if difficulty % 2 == 0 else "cycle"
    return {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "requested_seed": int(seed), "effective_seed": _seed_for_attempt(seed, difficulty, attempt),
        "attempt": attempt, "difficulty": difficulty, "context_index": difficulty - 1,
        "grid_size": names.GRID, "split": None,
        "generation_mode": "engine_extension" if extension else "official_profile",
        "selection_mode": selection_mode, "shapes": shapes, "mirrors": mirrors,
        "goals": [list(cell) for cell in goals], "steps": profile["steps"],
        "construction": {
            "target_shape_positions": [list(point) for point in targets],
            "target_mirrors": [[orientation, coordinate] for orientation, coordinate in target_mirrors],
            "recursive_reflection_cutoff": "enqueue reflections from depth 12; depth 13 nodes do not expand",
        },
        "mechanics": {
            "movable_shapes": len(shapes), "mirrors": len(mirrors),
            "movable_mirrors": sum(not mirror["fixed"] for mirror in mirrors),
            "fixed_mirrors": sum(mirror["fixed"] for mirror in mirrors),
            "reflection_orientations": sorted(profile["mirrors"]),
            "recursive_reflections": len(mirrors) > 1, "selection_mode": selection_mode,
            "rotation_extension": rotation, "restriction_extension": reflection_restriction,
            "official_incidence": {"rotation_extension": 0, "restriction_extension": 0},
        },
    }


def _mirror_sprite(spec):
    orientation, coordinate = spec["orientation"], spec["initial_coordinate"]
    tags = [names.TAG_MIRROR, names.TAG_FIXED if spec["fixed"] else "sys_click"]
    if orientation == "vertical":
        tags.append(names.TAG_VERTICAL_MIRROR)
        return Sprite(pixels=[[10] for _ in range(41)], name="generated-ar25-v", tags=tags, layer=-5).set_position(coordinate, -10)
    tags.append(names.TAG_HORIZONTAL_MIRROR)
    return Sprite(pixels=[[10] * 41], name="generated-ar25-h", tags=tags, layer=-5).set_position(-10, coordinate)


def build_level(spec):
    if not isinstance(spec, Mapping) or spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if _integer(spec.get("context_index"), "context_index") != difficulty - 1:
        raise ValueError("context_index must equal difficulty - 1")
    if _integer(spec.get("grid_size"), "grid_size") != names.GRID:
        raise ValueError("AR25 generated grids must be 21x21")
    shape_specs, mirror_specs = spec.get("shapes"), spec.get("mirrors")
    if not isinstance(shape_specs, (list, tuple)) or not shape_specs:
        raise ValueError("shapes must be a nonempty sequence")
    if not isinstance(mirror_specs, (list, tuple)) or not mirror_specs:
        raise ValueError("mirrors must be a nonempty sequence")
    shapes = []
    for index, row in enumerate(shape_specs):
        if not isinstance(row, Mapping):
            raise ValueError(f"shapes[{index}] must be a mapping")
        width, height = _validate_mask(row.get("mask"), f"shapes[{index}].mask")
        color = _integer(row.get("color"), f"shapes[{index}].color")
        if not 0 <= color <= 15:
            raise ValueError("shape color must be an ARC palette value")
        initial = _point(row.get("initial_position"), f"shapes[{index}].initial_position")
        target = _point(row.get("target_position"), f"shapes[{index}].target_position")
        for label, point in (("initial", initial), ("target", target)):
            if not (0 <= point[0] <= names.GRID - width and 0 <= point[1] <= names.GRID - height):
                raise ValueError(f"shape {label} position is outside the board")
        reflection, rotation = row.get("reflection", "both"), row.get("rotation")
        if reflection not in ("both", "horizontal", "vertical"):
            raise ValueError("shape reflection must be both, horizontal, or vertical")
        if rotation not in (None, "horizontal", "vertical"):
            raise ValueError("shape rotation must be horizontal, vertical, or None")
        tags = [names.TAG_MOVABLE, "sys_click"]
        if reflection == "horizontal": tags.append(names.TAG_REFLECT_HORIZONTAL_ONLY)
        if reflection == "vertical": tags.append(names.TAG_REFLECT_VERTICAL_ONLY)
        if rotation == "horizontal": tags.append(names.TAG_ROTATE_HORIZONTAL)
        if rotation == "vertical": tags.append(names.TAG_ROTATE_VERTICAL)
        pixels = [[color if value == "#" else names.TRANSPARENT for value in line] for line in row["mask"]]
        shapes.append(Sprite(pixels=pixels, name=f"generated-ar25-shape-{index}", tags=tags).set_position(*initial))
    orientations, mirrors, target_mirrors = set(), [], []
    for index, row in enumerate(mirror_specs):
        if not isinstance(row, Mapping):
            raise ValueError(f"mirrors[{index}] must be a mapping")
        orientation = row.get("orientation")
        if orientation not in ("vertical", "horizontal") or orientation in orientations:
            raise ValueError("mirrors must have unique vertical/horizontal orientations")
        orientations.add(orientation)
        initial = _integer(row.get("initial_coordinate"), "mirror initial coordinate")
        target = _integer(row.get("target_coordinate"), "mirror target coordinate")
        if not 0 <= initial < names.GRID or not 0 <= target < names.GRID:
            raise ValueError("mirror coordinate is outside the board")
        if not isinstance(row.get("fixed"), bool) or row["fixed"] and initial != target:
            raise ValueError("fixed mirror metadata is invalid")
        mirrors.append(_mirror_sprite(row)); target_mirrors.append([orientation, target])
    goals = [_point(value, f"goals[{index}]") for index, value in enumerate(spec.get("goals", ()))]
    if not goals or len(set(goals)) != len(goals) or any(not (0 <= x < names.GRID and 0 <= y < names.GRID) for x, y in goals):
        raise ValueError("goals must be unique in-board cells")
    steps = _integer(spec.get("steps"), "steps")
    if steps < 1 or spec.get("selection_mode") not in ("cycle", "click"):
        raise ValueError("steps/selection_mode metadata is invalid")
    descriptor = {
        "kind": names.GENERATED_KIND, "difficulty": difficulty, "context_index": difficulty - 1,
        "selection_mode": spec["selection_mode"],
        "targets": {"mirrors": target_mirrors, "shapes": [list(_point(row["target_position"], "target")) for row in shape_specs]},
    }
    goal_sprites = [upstream().sprites[names.SPRITE_GOAL].clone().set_position(x, y) for x, y in goals]
    return Level(
        sprites=[*goal_sprites, *mirrors, *shapes], grid_size=(names.GRID, names.GRID),
        data={names.KEY_STEPS: steps, names.KEY_GENERATED: descriptor},
        name=f"generated-ar25-d{difficulty}-s{spec.get('effective_seed', 0)}",
    )


def _transforms(x, y):
    edge = names.GRID - 1
    return (
        (x, y), (edge - x, y), (x, edge - y), (edge - x, edge - y),
        (y, x), (edge - y, x), (y, edge - x), (edge - y, edge - x),
    )


def _canonical_geometry(goals, shapes, mirrors):
    variants = []
    for transform_index in range(8):
        payload = {
            "goals": sorted(_transforms(x, y)[transform_index] for x, y in goals),
            "shapes": sorted(
                tuple(sorted(_transforms(x, y)[transform_index] for x, y in cells))
                for cells in shapes
            ),
            "mirrors": sorted(
                (bool(fixed), tuple(sorted(_transforms(x, y)[transform_index] for x, y in cells)))
                for fixed, cells in mirrors
            ),
        }
        variants.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return sha256(min(variants).encode()).hexdigest()


def geometry_identity(spec):
    goals = [_point(value, "goal") for value in spec["goals"]]
    shapes = []
    for shape in spec["shapes"]:
        px, py = _point(shape["initial_position"], "shape initial position")
        shapes.append({(px + x, py + y) for x, y in _occupied(shape["mask"])})
    mirrors = []
    for mirror in spec["mirrors"]:
        coordinate = int(mirror["initial_coordinate"])
        cells = (
            {(coordinate, y) for y in range(names.GRID)}
            if mirror["orientation"] == "vertical"
            else {(x, coordinate) for x in range(names.GRID)}
        )
        mirrors.append((mirror["fixed"], cells))
    return _canonical_geometry(goals, shapes, mirrors)


def _gameplay_payload(spec):
    """Return only public, executable puzzle semantics.

    Constructive targets and selection order are private teacher inputs. They
    must not make two identical public puzzles look semantically different.
    """
    shapes = [
        {
            "mask": list(shape["mask"]),
            "color": shape["color"],
            "initial_position": list(shape["initial_position"]),
            "reflection": shape.get("reflection", "both"),
            "rotation": shape.get("rotation"),
        }
        for shape in spec["shapes"]
    ]
    mirrors = [
        {
            "orientation": mirror["orientation"],
            "initial_coordinate": mirror["initial_coordinate"],
            "fixed": mirror["fixed"],
        }
        for mirror in spec["mirrors"]
    ]
    return {
        "format": spec["format"], "difficulty": spec["difficulty"],
        "context_index": spec["context_index"], "grid_size": spec["grid_size"],
        "shapes": shapes, "mirrors": mirrors,
        "goals": sorted(spec["goals"]), "steps": spec["steps"],
    }


def _hash_payload(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gameplay_identity(spec):
    return _hash_payload(_gameplay_payload(spec))


def canonical_spec_identity(spec):
    """Bind public semantics to the separate private teacher certificate."""
    return _hash_payload({
        "gameplay_sha256": gameplay_identity(spec),
        "teacher": {
            "selection_mode": spec.get("selection_mode"),
            "shape_targets": [shape.get("target_position") for shape in spec["shapes"]],
            "mirror_targets": [mirror.get("target_coordinate") for mirror in spec["mirrors"]],
            "construction": spec.get("construction"),
        },
        "certificate": {
            "solution": spec.get("solution"),
            "context_solution": spec.get("context_solution"),
            "solution_mechanics": spec.get("solution_mechanics"),
            "proof": spec.get("proof"),
            "engine_verified": spec.get("engine_verified"),
            "generation_mode": spec.get("generation_mode"),
            "mechanics": spec.get("mechanics"),
            "alternative_solution_audit": spec.get("alternative_solution_audit"),
        },
    })


def split_for_identity(identity):
    bucket = int(identity[:8], 16) % 5
    return "train" if bucket < 3 else "validation" if bucket == 3 else "test"


_OFFICIAL_IDENTITIES = None


def _official_identities():
    global _OFFICIAL_IDENTITIES
    if _OFFICIAL_IDENTITIES is not None:
        return _OFFICIAL_IDENTITIES
    identities = set()
    for index in range(len(official_levels())):
        env = Env(); env.reset(); env.set_level(index)
        goals = {(int(goal.x), int(goal.y)) for goal in env.goals()}
        shapes = [
            {(int(shape.x) + x, int(shape.y) + y) for y in range(shape.height) for x in range(shape.width) if int(shape.pixels[y, x]) != names.TRANSPARENT}
            for shape in env.movables()
        ]
        mirrors = []
        for mirror in env.mirrors():
            cells = (
                {(int(mirror.x), y) for y in range(names.GRID)}
                if names.TAG_VERTICAL_MIRROR in mirror.tags
                else {(x, int(mirror.y)) for x in range(names.GRID)}
            )
            mirrors.append((names.TAG_FIXED in mirror.tags, cells))
        identities.add(_canonical_geometry(goals, shapes, mirrors))
    _OFFICIAL_IDENTITIES = frozenset(identities)
    return _OFFICIAL_IDENTITIES


def _context_env(spec):
    context_index = int(spec["context_index"])
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context_index + 1)])
    env.reset(); env.set_level(context_index)
    return env


def _native_snapshot_matches(spec):
    env = Env([build_level(spec)])
    env.reset()
    mirrors = [(row["orientation"], int(row["initial_coordinate"])) for row in spec["mirrors"]]
    symbolic = set()
    for row in spec["shapes"]:
        symbolic.update(
            _coverage_paths(
                _point(row["initial_position"], "shape initial"),
                _occupied(row["mask"]), mirrors, row.get("reflection", "both"),
            )
        )
    native = env.game.naxbskjmlg()
    native_cells = {
        (x, y)
        for y in range(native.shape[0])
        for x in range(native.shape[1])
        if int(native[y, x]) >= 0
    }
    return symbolic == native_cells


def _native_route_for_public_tokens(spec, tokens):
    """Materialize abstract public selections and replay a candidate natively."""
    env = _context_env(spec)
    start_score = env.levels_completed
    route = []
    for token in tokens:
        if token[0] == "click":
            target_index = token[1]
            selectables = list(env.game.ayyvxqrhnzw)
            if not 0 <= target_index < len(selectables):
                return None
            action = _display_point(env, selectables[target_index])
        else:
            action = token
        route.append(list(action))
        observation = env.perform(*action)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            return route
        if observation.state == GameState.GAME_OVER:
            return None
    return None


def _alternative_solution_audit(spec, *, work_limit=ALTERNATIVE_CONFIGURATION_LIMIT):
    """Bounded public-state configuration search below the calibrated floor.

    This deliberately ignores every private target/proof field. Symbolic public
    configurations only nominate candidates; every shortcut is replayed in the
    native engine before it can reject a generated row. Work counts dequeued
    configurations; the unsatisfied-spawn admission check consumes no BFS work.
    """
    low = REFERENCE_PROFILES[spec["difficulty"]]["witness_range"][0]
    max_actions = low - 1
    base = {
        "method": "bounded-public-configuration-bfs-plus-native-replay",
        "work_limit": work_limit,
        "max_actions": max_actions,
    }
    env = _context_env(spec)
    if env.game.vplrhaovhr():
        return {
            **base, "status": "initially_covered", "work": 0,
            "truncated": False, "shortcut_actions": 0, "route": [],
        }
    if spec.get("generation_mode") != "official_profile":
        return {
            **base, "status": "not_applicable", "work": 0,
            "truncated": False, "shortcut_actions": None, "route": None,
        }

    native_mirrors, native_shapes = env.mirrors(), env.movables()
    selectables = list(env.game.ayyvxqrhnzw)
    objects = []
    for sprite in selectables:
        mirror_index = next((i for i, value in enumerate(native_mirrors) if value is sprite), None)
        if mirror_index is not None:
            objects.append(("mirror", mirror_index))
            continue
        shape_index = next((i for i, value in enumerate(native_shapes) if value is sprite), None)
        if shape_index is None:
            raise ValueError("native selectable is neither a mirror nor a shape")
        objects.append(("shape", shape_index))
    selected_index = next(
        (i for i, value in enumerate(selectables) if value is env.selected()), None,
    )
    if selected_index is None:
        raise ValueError("native AR25 state has no selected public object")

    mirror_orientations = []
    mirror_coordinates = []
    for mirror in native_mirrors:
        if names.TAG_VERTICAL_MIRROR in mirror.tags:
            mirror_orientations.append("vertical")
            mirror_coordinates.append(int(mirror.x))
        elif names.TAG_HORIZONTAL_MIRROR in mirror.tags:
            mirror_orientations.append("horizontal")
            mirror_coordinates.append(int(mirror.y))
        else:
            raise ValueError("native public audit found an unknown mirror")
    shape_positions = tuple((int(shape.x), int(shape.y)) for shape in native_shapes)
    shape_sizes = tuple((shape.width, shape.height) for shape in native_shapes)
    shape_masks = tuple(
        {
            (x, y)
            for y in range(shape.height)
            for x in range(shape.width)
            if int(shape.pixels[y, x]) != names.TRANSPARENT
        }
        for shape in native_shapes
    )
    goals = {(int(goal.x), int(goal.y)) for goal in env.goals()}
    start = (tuple(mirror_coordinates), shape_positions, selected_index)
    queue = deque([(start, ())])
    visited = {start}
    work = 0

    def wins(state):
        coordinates, positions, _ = state
        mirrors = tuple(zip(mirror_orientations, coordinates))
        coverage = set()
        for position, occupied in zip(positions, shape_masks):
            coverage.update(_coverage_paths(position, occupied, mirrors, "both"))
        return goals <= coverage

    while queue and work < work_limit:
        state, path = queue.popleft()
        work += 1
        if len(path) >= max_actions:
            continue
        coordinates, positions, selected = state
        kind, object_index = objects[selected]
        transitions = []
        if kind == "mirror":
            coordinate = coordinates[object_index]
            orientation = mirror_orientations[object_index]
            if orientation == "vertical":
                movements = ((names.ACTION_LEFT, -1), (names.ACTION_RIGHT, 1))
            else:
                movements = ((names.ACTION_UP, -1), (names.ACTION_DOWN, 1))
            for action_id, delta in movements:
                changed = coordinate + delta
                if 0 <= changed < names.GRID:
                    updated = list(coordinates); updated[object_index] = changed
                    transitions.append(((tuple(updated), positions, selected), (action_id, None, None), True))
        else:
            x, y = positions[object_index]
            width, height = shape_sizes[object_index]
            movements = (
                (names.ACTION_RIGHT, 1, 0), (names.ACTION_DOWN, 0, 1),
                (names.ACTION_LEFT, -1, 0), (names.ACTION_UP, 0, -1),
            )
            for action_id, dx, dy in movements:
                changed = (x + dx, y + dy)
                if 0 <= changed[0] <= names.GRID - width and 0 <= changed[1] <= names.GRID - height:
                    updated = list(positions); updated[object_index] = changed
                    transitions.append(((coordinates, tuple(updated), selected), (action_id, None, None), True))

        next_selected = (selected + 1) % len(objects)
        transitions.append(((coordinates, positions, next_selected), (names.ACTION_CYCLE, None, None), False))
        for target in range(len(objects)):
            if target != selected and target != next_selected:
                transitions.append(((coordinates, positions, target), ("click", target), False))

        for next_state, token, moved in transitions:
            if next_state in visited:
                continue
            visited.add(next_state)
            next_path = path + (token,)
            if moved and wins(next_state):
                route = _native_route_for_public_tokens(spec, next_path)
                if route is not None and len(route) <= max_actions:
                    return {
                        **base, "status": "shortcut", "work": work,
                        "truncated": False, "shortcut_actions": len(route), "route": route,
                    }
            queue.append((next_state, next_path))

    return {
        **base, "status": "no_shortcut_found", "work": work,
        "truncated": bool(queue), "shortcut_actions": None, "route": None,
    }


def _winning_native_path_maps(env):
    """Measure reflection paths from the actual first-winning native state."""
    mirrors = []
    mirror_coordinates = []
    for mirror in env.mirrors():
        if names.TAG_VERTICAL_MIRROR in mirror.tags:
            orientation, coordinate = "vertical", int(mirror.x)
        elif names.TAG_HORIZONTAL_MIRROR in mirror.tags:
            orientation, coordinate = "horizontal", int(mirror.y)
        else:
            raise ValueError("native winning state contains an unknown mirror orientation")
        mirrors.append((orientation, coordinate))
        mirror_coordinates.append([orientation, coordinate])

    positions, path_maps = [], []
    for shape in env.movables():
        position = (int(shape.x), int(shape.y))
        occupied = {
            (x, y)
            for y in range(shape.height)
            for x in range(shape.width)
            if int(shape.pixels[y, x]) != names.TRANSPARENT
        }
        if names.TAG_REFLECT_HORIZONTAL_ONLY in shape.tags:
            reflection = "horizontal"
        elif names.TAG_REFLECT_VERTICAL_ONLY in shape.tags:
            reflection = "vertical"
        else:
            reflection = "both"
        positions.append(list(position))
        path_maps.append(_coverage_paths(position, occupied, mirrors, reflection))
    return positions, mirror_coordinates, path_maps


def _native_evidence(spec):
    env = _context_env(spec)
    start_score, initial_budget = env.levels_completed, env.native_steps_left
    moved_shapes, moved_mirrors = set(), set()
    rotations = cycle_selections = click_selections = movement_attempts = movement_actions = 0
    completed_at = None
    actions = [tuple(action) for action in spec["solution"]]
    for action_index, action in enumerate(actions):
        action_id, x, y = action
        mirrors, shapes, selected = env.mirrors(), env.movables(), env.selected()
        before_masks = [tuple(map(tuple, shape.pixels.tolist())) for shape in shapes]
        before_shape_positions = [(int(shape.x), int(shape.y)) for shape in shapes]
        before_mirror_positions = [(int(mirror.x), int(mirror.y)) for mirror in mirrors]
        selected_shape = shapes.index(selected) if selected in shapes else None
        selected_mirror = mirrors.index(selected) if selected in mirrors else None
        observation = env.perform(action_id, x, y)
        if action_id == names.ACTION_CYCLE: cycle_selections += 1
        elif action_id == names.ACTION_CLICK: click_selections += 1
        elif action_id in (names.ACTION_UP, names.ACTION_DOWN, names.ACTION_LEFT, names.ACTION_RIGHT):
            movement_attempts += 1
            after_shapes, after_mirrors = env.movables(), env.mirrors()
            moved = False
            if (
                selected_shape is not None
                and selected_shape < len(after_shapes)
                and before_shape_positions[selected_shape]
                != (int(after_shapes[selected_shape].x), int(after_shapes[selected_shape].y))
            ):
                moved_shapes.add(selected_shape)
                moved = True
            if (
                selected_mirror is not None
                and selected_mirror < len(after_mirrors)
                and before_mirror_positions[selected_mirror]
                != (int(after_mirrors[selected_mirror].x), int(after_mirrors[selected_mirror].y))
            ):
                moved_mirrors.add(selected_mirror)
                moved = True
            movement_actions += int(moved)
        if env.levels_completed == start_score and observation.state == GameState.NOT_FINISHED:
            after_masks = [tuple(map(tuple, shape.pixels.tolist())) for shape in env.movables()]
            rotations += sum(before != after for before, after in zip(before_masks, after_masks))
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            completed_at = action_index + 1
            break
        if observation.state == GameState.GAME_OVER: break
    if completed_at is None:
        raise ValueError("stored route does not complete in the native context")
    if completed_at != len(actions):
        raise ValueError("stored route contains actions after native completion")

    winning_shapes, winning_mirrors, path_maps = _winning_native_path_maps(env)
    goals = {(int(goal.x), int(goal.y)) for goal in env.goals()}
    contributing = 0
    for index, paths in enumerate(path_maps):
        others = set().union(*(set(row) for j, row in enumerate(path_maps) if j != index))
        if any(goal in paths and goal not in others for goal in goals): contributing += 1
    reflected = {goal for paths in path_maps for goal, path in paths.items() if path and goal in goals}
    recursive = {goal for paths in path_maps for goal, path in paths.items() if len(path) >= 2 and goal in goals}
    orientations = sorted({orientation for paths in path_maps for goal, path in paths.items() if goal in goals for orientation in path})
    return {
        "won": True, "context_index": spec["context_index"], "actions": len(actions),
        "movement_attempts": movement_attempts, "movement_actions": movement_actions,
        "cycle_selections": cycle_selections,
        "click_selections": click_selections, "moved_shape_count": len(moved_shapes),
        "moved_mirror_count": len(moved_mirrors), "contributing_shape_count": contributing,
        "winning_shape_positions": winning_shapes,
        "winning_mirror_coordinates": winning_mirrors,
        "reflected_goal_count": len(reflected), "recursive_goal_count": len(recursive),
        "reflection_orientations_used": orientations,
        "rotations_observed_before_completion": rotations,
        "first_win_on_final_action": completed_at == len(actions),
        "native_budget": initial_budget,
        "native_budget_remaining": env.native_steps_left,
        "native_budget_slack": env.native_steps_left,
        "action_count_budget_estimate": initial_budget - len(actions),
    }


def _profile_errors(spec, evidence=None):
    errors, profile = [], REFERENCE_PROFILES[spec["difficulty"]]
    mode = spec.get("generation_mode")
    default = mode != "engine_extension"
    rotations = [row.get("rotation") for row in spec["shapes"] if row.get("rotation") is not None]
    restrictions = [
        row.get("reflection", "both")
        for row in spec["shapes"]
        if row.get("reflection", "both") != "both"
    ]
    if mode not in GENERATION_MODES:
        errors.append(f"generation_mode must be one of {GENERATION_MODES}")
    expected_mode = "engine_extension" if rotations or restrictions else "official_profile"
    if mode in GENERATION_MODES and mode != expected_mode:
        errors.append("generation_mode does not match declared shape extensions")
    if len(rotations) > 1 or any(row.get("rotation") is not None for row in spec["shapes"][1:]):
        errors.append("rotation extension must affect only the first shape")
    if len(restrictions) > 1 or any(row.get("reflection", "both") != "both" for row in spec["shapes"][1:]):
        errors.append("reflection restriction extension must affect only the first shape")
    rotation = rotations[0] if len(rotations) == 1 else None
    restriction = restrictions[0] if len(restrictions) == 1 else None
    if rotation is not None and rotation not in profile["mirrors"]:
        errors.append("rotation extension lacks a matching mirror orientation")
    if restriction is not None and spec["difficulty"] < 5:
        errors.append("reflection restriction extension requires difficulty >= 5")

    expected_mechanics = {
        "movable_shapes": len(spec["shapes"]),
        "mirrors": len(spec["mirrors"]),
        "movable_mirrors": sum(not row["fixed"] for row in spec["mirrors"]),
        "fixed_mirrors": sum(row["fixed"] for row in spec["mirrors"]),
        "reflection_orientations": sorted(row["orientation"] for row in spec["mirrors"]),
        "recursive_reflections": len(spec["mirrors"]) > 1,
        "selection_mode": spec["selection_mode"],
        "rotation_extension": rotation,
        "restriction_extension": restriction,
        "official_incidence": {"rotation_extension": 0, "restriction_extension": 0},
    }
    if spec.get("mechanics") != expected_mechanics:
        errors.append("mechanics summary does not match executable spec")
    if spec["steps"] != profile["steps"]: errors.append("native step budget differs from the shipped tier")
    if len(spec["shapes"]) != len(profile["shape_cells"]): errors.append("shape count differs from the shipped tier")
    if tuple(row["orientation"] for row in spec["mirrors"]) != profile["mirrors"]: errors.append("mirror composition differs from the shipped tier")
    if sum(bool(row["fixed"]) for row in spec["mirrors"]) != profile["fixed"]: errors.append("fixed mirror count differs from the shipped tier")
    if tuple(len(_occupied(row["mask"])) for row in spec["shapes"]) != profile["shape_cells"]: errors.append("shape density differs from the calibrated tier")
    expected_sizes = list(profile["shape_sizes"])
    if rotation is not None:
        size = max(expected_sizes[0])
        expected_sizes[0] = (size, size)
    if tuple((len(row["mask"][0]), len(row["mask"])) for row in spec["shapes"]) != tuple(expected_sizes): errors.append("shape extent differs from the calibrated mode")
    if tuple(_components(_occupied(row["mask"])) for row in spec["shapes"]) != profile["shape_components"]: errors.append("shape 4-connected piece count differs from the official tier")
    if any(row.get("color") != 5 for row in spec["shapes"]): errors.append("shape contrast differs from the official presentation")
    if default:
        if len(spec["goals"]) != profile["goals"]: errors.append("goal count differs from the shipped tier")
        if not _in_bbox_range(spec["goals"], profile): errors.append("goal extent differs from the calibrated tier")
        if not _goal_components_in_range(spec["goals"], profile): errors.append("goal 4-connected region count differs from the official tier")
        if any(row.get("rotation") is not None for row in spec["shapes"]): errors.append("default profile forces zero-incidence rotation")
        if any(row.get("reflection", "both") != "both" for row in spec["shapes"]): errors.append("default profile forces zero-incidence restriction")
    elif not 1 <= len(spec["goals"]) <= profile["goals"]:
        errors.append("extension goal count is outside its explicit contract")
    if evidence is not None:
        low, high = profile["witness_range"]
        if default and not low <= evidence["actions"] <= high: errors.append("action length is outside calibrated tolerance")
        if evidence.get("first_win_on_final_action") is not True: errors.append("winning route contains post-completion padding")
        if evidence["moved_shape_count"] != len(spec["shapes"]): errors.append("winning route does not move every shape")
        movable = sum(not row["fixed"] for row in spec["mirrors"])
        if evidence["moved_mirror_count"] != movable: errors.append("winning route does not move every movable mirror")
        if evidence["contributing_shape_count"] != len(spec["shapes"]): errors.append("not every shape contributes an exclusive goal")
        if evidence["reflected_goal_count"] < 1: errors.append("winning target does not require reflection")
        if default and len(spec["mirrors"]) > 1 and evidence["recursive_goal_count"] < 1: errors.append("winning target does not exercise recursive reflection")
        expected_orientations = sorted({row["reflection"] for row in spec["shapes"] if row.get("reflection") != "both"} or profile["mirrors"])
        if not set(expected_orientations) <= set(evidence["reflection_orientations_used"]): errors.append("winning goals miss a required reflection orientation")
        selectable = len(spec["shapes"]) + movable
        if spec["selection_mode"] == "click" and selectable > 1 and evidence["click_selections"] < 1: errors.append("route misses display-space selection")
        if spec["selection_mode"] == "cycle" and selectable > 1 and evidence["cycle_selections"] < 1: errors.append("route misses cyclic selection")
        if any(row.get("rotation") is not None for row in spec["shapes"]) and evidence["rotations_observed_before_completion"] < 1: errors.append("rotation extension was not observed")
    return errors


def _store_report(function, report):
    """Publish a detached JSON-ready diagnostic for the most recent call."""
    function.last_report = json.loads(json.dumps(report, sort_keys=True))


def generate(seed, difficulty=1, attempts=MAX_ATTEMPTS, limit=DEFAULT_LIMIT, *, split="train", reflection_restriction=None, rotation=None):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0: raise ValueError("seed must be a nonnegative integer")
    if difficulty not in DIFFICULTIES or isinstance(difficulty, bool): raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS: raise ValueError(f"split must be one of {SPLITS}")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= MAX_ATTEMPTS: raise ValueError(f"attempts must be within 1..{MAX_ATTEMPTS}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0: raise ValueError("limit must be nonnegative")
    report = {
        "format": "pebby.ar25.generation-report.v1",
        "seed": int(seed), "difficulty": int(difficulty), "split": split,
        "caps": {"attempts": attempts, "search_limit": limit},
    }
    if limit == 0:
        _store_report(generate, {
            **report, "status": "exhausted", "attempts_used": 0,
            "search_work": 0, "rejections": {},
            "alternative_configuration_work": 0, "shortcut_rejections": [],
            "terminal_cause": "search_limit_zero",
            "terminal_detail": "no candidate search can run with a zero expansion limit",
        })
        return None
    rejections = Counter()
    search_work = 0
    alternative_work = 0
    shortcut_rejections = []
    last_rejection = None
    for attempt in range(attempts):
        effective_seed = _seed_for_attempt(seed, difficulty, attempt)
        candidate = _draft(seed, difficulty, attempt, reflection_restriction=reflection_restriction, rotation=rotation)
        if candidate is None:
            rejections["construction"] += 1
            last_rejection = {"cause": "construction", "attempt": attempt, "effective_seed": effective_seed}
            continue
        candidate["geometry_sha256"] = geometry_identity(candidate)
        candidate["geometry_d4_sha256"] = candidate["geometry_sha256"]
        candidate["gameplay_sha256"] = gameplay_identity(candidate)
        candidate["split"] = split_for_identity(candidate["geometry_d4_sha256"])
        candidate["geometry_partition"] = candidate["split"]
        if candidate["split"] != split:
            rejections["split_partition"] += 1
            last_rejection = {
                "cause": "split_partition", "attempt": attempt,
                "effective_seed": effective_seed, "actual_split": candidate["split"],
            }
            continue
        if candidate["geometry_d4_sha256"] in _official_identities():
            rejections["official_copy"] += 1
            last_rejection = {"cause": "official_copy", "attempt": attempt, "effective_seed": effective_seed}
            continue
        alternative_audit = _alternative_solution_audit(candidate)
        alternative_work += alternative_audit["work"]
        if alternative_audit["status"] == "initially_covered":
            rejections["initially_covered"] += 1
            last_rejection = {
                "cause": "initially_covered", "attempt": attempt,
                "effective_seed": effective_seed,
                "detail": "initial public configuration already covers all goals",
            }
            continue
        if alternative_audit["shortcut_actions"] is not None:
            rejections["below_floor_shortcut"] += 1
            shortcut_rejection = {
                "attempt": attempt, "effective_seed": effective_seed,
                "actions": alternative_audit["shortcut_actions"],
                "route": alternative_audit["route"], "work": alternative_audit["work"],
            }
            shortcut_rejections.append(shortcut_rejection)
            last_rejection = {"cause": "below_floor_shortcut", **shortcut_rejection}
            continue
        candidate["alternative_solution_audit"] = alternative_audit
        result = search(_context_env(candidate), limit=limit)
        search_work += result.expanded
        if result.actions is None:
            rejections["native_route"] += 1
            last_rejection = {
                "cause": "native_route", "attempt": attempt,
                "effective_seed": effective_seed, "search_work": result.expanded,
                "search_reason": result.reason, "truncated": result.truncated,
                "unsupported": result.unsupported,
            }
            continue
        candidate["solution"] = [list(action) for action in result.actions]
        candidate["context_solution"] = [list(action) for action in result.actions]
        candidate["solution_length"] = len(result.actions)
        try: evidence = _native_evidence(candidate)
        except ValueError as exc:
            rejections["native_replay"] += 1
            last_rejection = {
                "cause": "native_replay", "attempt": attempt,
                "effective_seed": effective_seed, "detail": str(exc),
            }
            continue
        profile_errors = _profile_errors(candidate, evidence)
        if profile_errors:
            rejections["profile"] += 1
            last_rejection = {
                "cause": "profile", "attempt": attempt,
                "effective_seed": effective_seed, "errors": profile_errors,
            }
            continue
        candidate["solution_mechanics"] = evidence
        candidate["proof"] = {
            "method": "private-target-constructive-teacher-plus-native-context-replay",
            "context_index": difficulty - 1, "engine_win": True,
            "witness_optimality_claimed": False, "search_limit": limit,
            "search_work": result.expanded, "native_budget": candidate["steps"],
        }
        candidate["engine_verified"] = True
        candidate["rejections"] = dict(sorted(rejections.items()))
        candidate["attempts_used"] = attempt + 1
        candidate["canonical_spec_identity"] = canonical_spec_identity(candidate)
        _store_report(generate, {
            **report, "status": "accepted", "attempts_used": attempt + 1,
            "search_work": search_work, "rejections": dict(sorted(rejections.items())),
            "alternative_configuration_work": alternative_work,
            "shortcut_rejections": shortcut_rejections,
            "terminal_cause": None, "last_rejection": last_rejection,
            "accepted_attempt": attempt, "effective_seed": effective_seed,
        })
        return candidate
    terminal_cause = last_rejection["cause"] if last_rejection else "attempt_cap_exhausted"
    _store_report(generate, {
        **report, "status": "exhausted", "attempts_used": attempts,
        "search_work": search_work, "rejections": dict(sorted(rejections.items())),
        "alternative_configuration_work": alternative_work,
        "shortcut_rejections": shortcut_rejections,
        "terminal_cause": terminal_cause, "terminal_detail": last_rejection,
    })
    return None


def _child_seed(game_seed, ordinal, difficulty):
    raw = f"ar25-0c556536:game-v1:{int(game_seed)}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(blake2b(raw, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split="train", difficulties=None, attempts=MAX_ATTEMPTS, limit=DEFAULT_LIMIT):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0: raise ValueError("seed must be nonnegative")
    if split not in SPLITS: raise ValueError(f"split must be one of {SPLITS}")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= MAX_ATTEMPTS: raise ValueError(f"attempts must be within 1..{MAX_ATTEMPTS}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0: raise ValueError("limit must be nonnegative")
    if difficulties is None:
        sequence = DIFFICULTIES
    else:
        try:
            sequence = tuple(difficulties)
        except TypeError as exc:
            raise ValueError("difficulties must be strictly increasing unique official integer tiers") from exc
        if (
            not sequence
            or any(isinstance(d, bool) or not isinstance(d, int) or d not in DIFFICULTIES for d in sequence)
            or any(left >= right for left, right in zip(sequence, sequence[1:]))
        ):
            raise ValueError("difficulties must be strictly increasing unique official integer tiers")
    specs, child_reports = [], []
    for ordinal, difficulty in enumerate(sequence):
        child_seed = _child_seed(seed, ordinal, difficulty)
        spec = generate(child_seed, difficulty, attempts=attempts, limit=limit, split=split)
        child_report = json.loads(json.dumps(generate.last_report, sort_keys=True))
        child_reports.append(child_report)
        if spec is None:
            _store_report(generate_game, {
                "format": "pebby.ar25.game-generation-report.v1",
                "status": "exhausted", "seed": int(seed), "split": split,
                "difficulties": list(sequence),
                "caps": {"attempts_per_tier": attempts, "search_limit": limit},
                "completed_tiers": len(specs),
                "search_work": sum(row["search_work"] for row in child_reports),
                "child_reports": child_reports,
                "terminal_child": {
                    "ordinal": ordinal, "difficulty": difficulty,
                    "seed": child_seed, "report": child_report,
                },
            })
            return None
        spec["game_seed"], spec["game_ordinal"] = int(seed), ordinal
        specs.append(spec)
    _store_report(generate_game, {
        "format": "pebby.ar25.game-generation-report.v1",
        "status": "accepted", "seed": int(seed), "split": split,
        "difficulties": list(sequence),
        "caps": {"attempts_per_tier": attempts, "search_limit": limit},
        "completed_tiers": len(specs),
        "search_work": sum(row["search_work"] for row in child_reports),
        "child_reports": child_reports, "terminal_child": None,
    })
    return specs


def build_game(specs):
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)): raise ValueError("specs must be a sequence")
    if len(specs) != len(DIFFICULTIES): raise ValueError("full AR25 games require exactly eight levels")
    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise ValueError(f"specs[{index}] must be a mapping")
        for field in ("split", "geometry_d4_sha256", "gameplay_sha256"):
            try:
                hash(spec.get(field))
            except TypeError as exc:
                raise ValueError(f"specs[{index}].{field} must be hashable") from exc
    if len({spec.get("split") for spec in specs}) != 1: raise ValueError("full game levels must use one split")
    geometries = [spec.get("geometry_d4_sha256") for spec in specs]
    gameplays = [spec.get("gameplay_sha256") for spec in specs]
    if len(set(geometries)) != len(geometries) or len(set(gameplays)) != len(gameplays):
        raise ValueError("full game contains duplicate geometry or gameplay identity")
    for index, (difficulty, spec) in enumerate(zip(DIFFICULTIES, specs)):
        if not isinstance(spec, Mapping) or spec.get("difficulty") != difficulty or spec.get("context_index") != index: raise ValueError("full game must be ordered tiers 1..8 in contexts 0..7")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors: raise ValueError("invalid full-standard AR25 spec: " + "; ".join(errors))
    levels = [build_level(spec) for spec in specs]
    probe = Env([level.clone() for level in levels])
    probe.reset()
    for index, spec in enumerate(specs):
        if probe.level_index != index:
            raise ValueError("native sequential replay entered the wrong level context")
        completed, _ = replay(probe, [tuple(action) for action in spec["solution"]])
        if not completed or probe.levels_completed != index + 1:
            raise ValueError("stored route failed native sequential whole-game replay")
    if probe.state != GameState.WIN:
        raise ValueError("native sequential whole-game replay did not win")
    return levels


def _validate_solution_action(action, index):
    label = f"solution[{index}]"
    if not isinstance(action, (list, tuple)) or len(action) != 3:
        raise ValueError(f"{label} must be an action triple")
    action_id = _integer(action[0], f"{label} action id")
    if action_id not in names.AVAILABLE_ACTIONS:
        raise ValueError(f"{label} action id must be one of {names.AVAILABLE_ACTIONS}")
    coordinates = []
    for name, value in (("x", action[1]), ("y", action[2])):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{label} action {name} must be an integer or None")
        coordinates.append(value)
    x, y = coordinates
    if action_id == names.ACTION_CLICK:
        if x is None or y is None:
            raise ValueError(f"{label} ACTION6 requires integer x and y")
        if not (0 <= x < names.DISPLAY and 0 <= y < names.DISPLAY):
            raise ValueError(f"{label} click coordinates must be within the 64x64 display")
    elif x is not None or y is not None:
        raise ValueError(f"{label} non-click actions require null coordinates")
    return action_id, x, y


def validate_full_standard(spec, curriculum_entry):
    errors = []
    if not isinstance(spec, Mapping): return ["spec must be a mapping"]
    try:
        difficulty = _integer(spec.get("difficulty"), "difficulty")
        context_index = _integer(spec.get("context_index"), "context_index")
        if not isinstance(curriculum_entry, Mapping): errors.append("curriculum entry must be a mapping")
        else:
            curriculum_difficulty = _integer(curriculum_entry.get("difficulty"), "curriculum difficulty")
            curriculum_context = _integer(curriculum_entry.get("context_index"), "curriculum context_index")
            curriculum_work = _integer(curriculum_entry.get("search_work"), "curriculum search_work")
            if difficulty != curriculum_difficulty: errors.append("difficulty does not match curriculum")
            if context_index != curriculum_context: errors.append("context does not match curriculum")
            if not 1 <= curriculum_work <= 32_000_000: errors.append("curriculum search_work is out of range")
        if spec.get("format") != FORMAT: errors.append("unexpected spec format")
        if _integer(spec.get("generator_version"), "generator_version") != GENERATOR_VERSION: errors.append("unexpected generator version")
        if spec.get("quality_profile_version") != QUALITY_PROFILE_VERSION: errors.append("unexpected quality profile version")
        if spec.get("mechanics_inventory_version") != MECHANICS_INVENTORY_VERSION: errors.append("unexpected mechanics inventory version")
        if spec.get("engine_verified") is not True: errors.append("engine_verified must be true")
        if spec.get("split") not in SPLITS: errors.append("invalid generation split")
        if spec.get("generation_mode") not in GENERATION_MODES: errors.append("invalid generation_mode")
        build_level(spec)
        expected_geometry, expected_gameplay = geometry_identity(spec), gameplay_identity(spec)
        expected_canonical = canonical_spec_identity(spec)
        if spec.get("geometry_sha256") != expected_geometry: errors.append("geometry_sha256 mismatch")
        if spec.get("geometry_d4_sha256") != expected_geometry: errors.append("geometry_d4_sha256 mismatch")
        if spec.get("gameplay_sha256") != expected_gameplay: errors.append("gameplay_sha256 mismatch")
        if spec.get("canonical_spec_identity") != expected_canonical: errors.append("canonical spec identity mismatch")
        if spec.get("split") != split_for_identity(expected_geometry): errors.append("geometry belongs to another split")
        if spec.get("geometry_partition") != spec.get("split"): errors.append("geometry_partition differs from split")
        if expected_geometry in _official_identities(): errors.append("geometry duplicates an official level under D4")
        if not _native_snapshot_matches(spec): errors.append("symbolic/native reflection snapshot mismatch")
        expected_alternative_audit = _alternative_solution_audit(spec)
        if spec.get("alternative_solution_audit") != expected_alternative_audit:
            errors.append("alternative public-configuration audit mismatch")
        if expected_alternative_audit["status"] == "initially_covered":
            errors.append("initial public configuration already covers all goals")
        elif expected_alternative_audit["shortcut_actions"] is not None:
            errors.append("public native shortcut falls below the calibrated action floor")
        requested_seed = _integer(spec.get("requested_seed"), "requested_seed")
        attempt = _integer(spec.get("attempt"), "attempt")
        attempts_used = _integer(spec.get("attempts_used"), "attempts_used")
        if requested_seed < 0 or not 0 <= attempt < MAX_ATTEMPTS or attempts_used != attempt + 1:
            errors.append("attempt metadata is out of bounds")
        if spec.get("effective_seed") != _seed_for_attempt(requested_seed, difficulty, attempt):
            errors.append("effective_seed mismatch")
        rejections = spec.get("rejections")
        if not isinstance(rejections, Mapping):
            errors.append("rejections must be a mapping")
        elif any(
            not isinstance(key, str) or not key or isinstance(value, bool)
            or not isinstance(value, int) or value < 0
            for key, value in rejections.items()
        ) or sum(rejections.values()) > attempt:
            errors.append("rejection counters are invalid")
        solution = spec.get("solution")
        if not isinstance(solution, (list, tuple)) or not solution: errors.append("solution must be nonempty")
        else:
            for index, action in enumerate(solution):
                _validate_solution_action(action, index)
            if spec.get("context_solution") != solution: errors.append("context_solution differs from native route")
            if _integer(spec.get("solution_length"), "solution_length") != len(solution): errors.append("solution_length mismatch")
        proof = spec.get("proof")
        if not isinstance(proof, Mapping): errors.append("proof must be a mapping")
        else:
            if _integer(proof.get("context_index"), "proof context_index") != context_index: errors.append("proof context mismatch")
            work = _integer(proof.get("search_work"), "proof search_work")
            if not 1 <= work <= 32_000_000: errors.append("proof search_work is out of range")
            if proof.get("witness_optimality_claimed") is not False: errors.append("constructive proof must not claim optimality")
            expected_limit = curriculum_work if isinstance(curriculum_entry, Mapping) else None
            _integer(proof.get("search_limit"), "proof search_limit")
            if proof.get("search_limit") != expected_limit: errors.append("proof search_limit differs from curriculum")
            if proof.get("method") != "private-target-constructive-teacher-plus-native-context-replay": errors.append("proof method mismatch")
            if proof.get("engine_win") is not True: errors.append("proof engine_win mismatch")
            if _integer(proof.get("native_budget"), "proof native_budget") != spec.get("steps"): errors.append("proof native budget mismatch")
        if solution:
            expected_result = search(
                _context_env(spec),
                limit=curriculum_work if isinstance(curriculum_entry, Mapping) else DEFAULT_LIMIT,
            )
            expected_solution = (
                [list(action) for action in expected_result.actions]
                if expected_result.actions is not None else None
            )
            if solution != expected_solution: errors.append("solution differs from canonical constructive route")
            if isinstance(proof, Mapping) and proof.get("search_work") != expected_result.expanded:
                errors.append("proof search_work mismatch")
            recomputed = _native_evidence(spec)
            if spec.get("solution_mechanics") != recomputed: errors.append("solution mechanic evidence mismatch")
            errors.extend(_profile_errors(spec, recomputed))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(str(exc))
    return list(dict.fromkeys(errors))
