"""Full six-tier procedural CD82 generator with native replay certificates."""

from collections import Counter
from functools import lru_cache
import hashlib
import json
import random

from arcengine import Level, Sprite

from . import names
from .env import Env, upstream
from .generation_quality import (
    IDENTITY_VERSION,
    SPLITS,
    canonical_identities,
    geometry_split,
    raw_geometry_identity,
)
from .layout import atomise, build, extract
from .plan import search
from .reference_profiles import (
    DIFFICULTY_VERSION,
    DIFFICULTIES,
    MECHANICS_INVENTORY_VERSION,
    PROFILES,
    QUALITY_PROFILE_VERSION,
    profile_errors,
)

FORMAT = "pebby-cd82-generated-v2"
GENERATOR_VERSION = 2
ALL_COLOURS = (0, 15, 12, 11, 14, 8, 9)   # the seven swatch colours upstream uses

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": "cd82-fb555c5d",
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": [
        {"difficulty": difficulty, "context_index": difficulty - 1,
         "search_work": PROFILES[difficulty]["search_work"]}
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "docs/generator-evidence/cd82.md#official-reference-characterization",
        "solution_mechanics": "docs/generator-evidence/cd82.md#procedural-generation-and-proof",
        "native_budget": "docs/generator-evidence/cd82.md#authoritative-source-and-complete-mechanics",
        "context_engine_replay": "docs/generator-evidence/cd82.md#root-acceptance-2026-09-18",
        "novelty_split": "docs/generator-evidence/cd82.md#identity-split-and-finite-support",
        "bounded_rejections": "docs/generator-evidence/cd82.md#bounded-quality-audit",
    },
    "caveats": [
        "each tier is calibrated from one official reference and the final generated census is modest",
        "tier 1 has exactly one accepted D4 target class per split and raw orientation capacities 6/6/4 for train/validation/test",
        "later-tier optimality uses the package A* plus native replay and was not independently re-searched by a second solver",
    ],
}


_RECIPES = {
    # Later recipes match the number and composition of native operations in
    # an official optimal solution. Tiers 1/2 sample their complete closures.
    3: {"half": 1, "triangle": 2, "cap": 1},
    4: {"half": 2, "triangle": 1, "cap": 1},
    5: {"half": 1, "triangle": 2, "cap": 1},
    6: {"half": 1, "triangle": 1, "cap": 2},
}


def _diagonal_fill(grid):
    """Fill ignored diagonals for a coherent displayed target."""
    for index in range(names.CANVAS_SIZE):
        for row, col in ((index, index), (index, 9 - index)):
            neighbours = [grid[rr][cc]
                          for rr, cc in ((row, col - 1), (row, col + 1),
                                         (row - 1, col), (row + 1, col))
                          if 0 <= rr < 10 and 0 <= cc < 10
                          and rr != cc and rr + cc != 9]
            if neighbours:
                counts = Counter(neighbours)
                grid[row][col] = max(
                    counts, key=lambda value: (counts[value], -neighbours.index(value)))
    return grid


@lru_cache(maxsize=4)
def _reachable_atoms(colours):
    """All states reachable from blank using native non-indicator paints."""
    operations, atom_cells = atomise(False)
    start = (0,) * len(atom_cells)
    reached = {start}
    frontier = [start]
    while frontier:
        state = frontier.pop()
        for _, _, region in operations:
            for colour in colours:
                changed = list(state)
                for atom in region:
                    changed[atom] = colour
                changed = tuple(changed)
                if changed not in reached:
                    reached.add(changed)
                    frontier.append(changed)
    return tuple(sorted(reached - {start}))


def _tier1_reachable_atoms():
    return _reachable_atoms((0, names.START_COLOR))


def _draft(rng, difficulty):
    profile = PROFILES[difficulty]
    palette = list(ALL_COLOURS[:profile["palette_count"]])
    rng.shuffle(palette)
    indicator = profile["indicator"]
    ops, atom_cells = atomise(indicator)
    atoms = [0] * len(atom_cells)
    halves = [op for op in ops if op[1] == "paint" and op[0] in names.EDGE_DIALS]
    triangles = [op for op in ops if op[1] == "paint" and op[0] in names.CORNER_DIALS]
    caps = [op for op in ops if op[1] == "cap"]
    if difficulty == 1:
        # The 122-state closure is derived from the rules above, not from any
        # official target or route. Uniform sampling covers action-relevant D4
        # orientations that short random programs reach very unevenly.
        atoms = list(rng.choice(_tier1_reachable_atoms()))
        sequence = []
        colours = []
    elif difficulty == 2:
        atoms = list(rng.choice(_reachable_atoms((0, names.START_COLOR, 12))))
        sequence = []
        colours = []
    else:
        recipe = _RECIPES[difficulty]
        sequence = (rng.sample(halves, recipe["half"])
                    + rng.sample(triangles, recipe["triangle"])
                    + rng.sample(caps, recipe["cap"]))
    if difficulty > 2:
        rng.shuffle(sequence)
        colours = rng.sample([colour for colour in palette if colour != 0], len(sequence))
    for (_, _, region), colour in zip(sequence, colours):
        for a in region:
            atoms[a] = colour
    if all(a == 0 for a in atoms):
        return None
    grid = [[0] * names.CANVAS_SIZE for _ in range(names.CANVAS_SIZE)]
    for atom, cells in enumerate(atom_cells):
        for r, c in cells:
            grid[r][c] = atoms[atom]
    spec = {
        "format": FORMAT,
        "game": "cd82",
        "generator_version": GENERATOR_VERSION,
        "difficulty": difficulty,
        "difficulty_version": DIFFICULTY_VERSION,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "reference_level": difficulty,
        "context_index": difficulty - 1,
        "palette": palette,
        "indicator": indicator,
        "target": _diagonal_fill(grid),
        "source": "generated_only",
    }
    return None if profile_errors(spec, require_proof=False) else spec


def swatch_positions(count):
    # Upstream uses x=21..57 for the seven-swatch tiers and the centered
    # arithmetic sequence for the two/three-swatch tiers (cd82.py:257-361).
    start = 21 if count == 7 else 41 - 3 * count
    return [(start + names.SWATCH_PITCH * i, names.SWATCH_Y) for i in range(count)]


def build_level(spec):
    """An ARCEngine Level for `spec`, laid out like the shipped levels."""
    sprites = upstream().sprites
    target = [[int(v) for v in row] for row in spec["target"]]
    if len(target) != 10 or any(len(row) != 10 for row in target):
        raise ValueError("target must be 10x10")
    level_sprites = [
        Sprite(pixels=target, name=names.TARGET_PREFIX + "gen", visible=True, collidable=True)
        .set_position(*names.TARGET_POS),
        sprites[names.SPRITE_LAYOUT].clone(),
        sprites[names.SPRITE_BASKET_HORIZONTAL].clone().set_position(25, 24).set_rotation(180),
    ]
    if spec["indicator"]:
        level_sprites.insert(0, sprites[names.SPRITE_INDICATOR].clone()
                             .set_position(*names.INDICATOR_HOME).set_rotation(180))
    positions = swatch_positions(len(spec["palette"]))
    cursor_at = positions[0]
    for colour, (x, y) in zip(spec["palette"], positions):
        swatch = sprites[names.SPRITE_SWATCH].clone().set_position(x, y)
        if colour != 0:
            swatch.color_remap(0, int(colour))
        level_sprites.append(swatch)
        if colour == names.START_COLOR:
            cursor_at = (x, y)
    level_sprites.append(sprites[names.SPRITE_CANVAS].clone().set_position(*names.CANVAS_POS))
    level_sprites.append(sprites[names.SPRITE_CURSOR].clone()
                         .set_position(cursor_at[0], cursor_at[1] + names.CURSOR_DY))
    return Level(sprites=level_sprites, grid_size=(names.FRAME_SIZE, names.FRAME_SIZE))


def layout_for(spec):
    """The starting Layout of `spec` without touching the engine."""
    swatches = [(c, *names.swatch_click(x, y))
                for c, (x, y) in zip(spec["palette"], swatch_positions(len(spec["palette"])))]
    blank = [[0] * 10 for _ in range(10)]
    return build(dial=names.START_DIAL, color=names.START_COLOR, canvas=blank, target=spec["target"],
                 swatches=swatches, has_indicator=spec["indicator"])


@lru_cache(maxsize=512)
def _cached_generated_search(palette, indicator, target, limit):
    """Memoize immutable exact results; native replay is intentionally fresh."""
    spec = {"palette": list(palette), "indicator": indicator,
            "target": [list(row) for row in target]}
    return search(layout_for(spec), limit)


def solution_mechanics(spec, actions):
    """Describe mechanics used by a concrete action-optimal certificate."""
    swatches = spec.get("swatches")
    if swatches is None:
        swatches = layout_for(spec).palette
    palette = {(x, y): colour for colour, x, y in swatches}
    dial = names.START_DIAL
    colour = names.START_COLOR
    counts = Counter({
        "dial_moves": 0,
        "effective_dial_moves": 0,
        "swatch_clicks": 0,
        "region_paints": 0,
        "half_paints": 0,
        "triangle_paints": 0,
        "cap_paints": 0,
    })
    paint_dials = []
    cap_dials = []
    paint_colours = []
    for action, x, y in actions:
        if action in (1, 2, 3, 4):
            moved = names.move_dial(dial, action)
            counts["dial_moves"] += 1
            counts["effective_dial_moves"] += moved != dial
            dial = moved
        elif action == names.ACTION_PAINT:
            counts["region_paints"] += 1
            key = "half_paints" if dial in names.EDGE_DIALS else "triangle_paints"
            counts[key] += 1
            paint_dials.append(dial)
            paint_colours.append(colour)
        elif action == names.ACTION_CLICK and (x, y) in palette:
            colour = palette[(x, y)]
            counts["swatch_clicks"] += 1
        elif (action == names.ACTION_CLICK and spec["indicator"] and dial in names.EDGE_DIALS
              and (x, y) == names.indicator_click(dial)):
            counts["cap_paints"] += 1
            cap_dials.append(dial)
            paint_colours.append(colour)
        else:
            counts["unrecognised_actions"] += 1
    result = dict(counts)
    result.update(
        distinct_paint_dials=len(set(paint_dials + cap_dials)),
        distinct_paint_colours=len(set(paint_colours)),
        paint_dials=paint_dials,
        cap_dials=cap_dials,
    )
    return result


@lru_cache(maxsize=1)
def _official_geometry_identities():
    env = Env()
    env.reset()
    identities = set()
    for index in range(len(DIFFICULTIES)):
        if index:
            env.set_level(index)
        spec = {
            "target": env.target().tolist(),
            "indicator": env.has_indicator(),
            "palette": [colour for colour, _, _ in env.swatches()],
        }
        identities.add(canonical_identities(spec)[1])
    return frozenset(identities)


@lru_cache(maxsize=1)
def tier1_split_support():
    """Exact D4 and orientation-preserving support after every tier-1 gate."""
    _, atom_cells = atomise(False)
    accepted = {split: {"d4": set(), "raw": set()} for split in SPLITS}
    for atoms in _tier1_reachable_atoms():
        grid = [[0] * names.CANVAS_SIZE for _ in range(names.CANVAS_SIZE)]
        for atom, cells in enumerate(atom_cells):
            for row, col in cells:
                grid[row][col] = atoms[atom]
        spec = {
            "format": FORMAT,
            "game": "cd82",
            "generator_version": GENERATOR_VERSION,
            "difficulty": 1,
            "difficulty_version": DIFFICULTY_VERSION,
            "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
            "quality_profile_version": QUALITY_PROFILE_VERSION,
            "reference_level": 1,
            "context_index": 0,
            "palette": [0, 15],
            "indicator": False,
            "target": _diagonal_fill(grid),
            "source": "generated_only",
        }
        if profile_errors(spec, require_proof=False):
            continue
        _, geometry_d4 = canonical_identities(spec)
        if geometry_d4 in _official_geometry_identities():
            continue
        split = geometry_split(geometry_d4)
        result = _cached_generated_search(
            tuple(spec["palette"]), False,
            tuple(tuple(row) for row in spec["target"]), PROFILES[1]["search_work"])
        if (not result.solved or result.truncated
                or not PROFILES[1]["actions"][0] <= result.length <= PROFILES[1]["actions"][1]):
            continue
        mechanics = solution_mechanics(spec, result.actions)
        if any(mechanics.get(key, 0) < minimum
               for key, minimum in PROFILES[1]["minimum_mechanics"].items()):
            continue
        accepted[split]["d4"].add(geometry_d4)
        accepted[split]["raw"].add(raw_geometry_identity(spec))
    return {
        split: {
            "d4_classes": len(accepted[split]["d4"]),
            "raw_capacity": len(accepted[split]["raw"]),
        }
        for split in SPLITS
    }


@lru_cache(maxsize=1)
def tier1_split_capacities():
    """Exact orientation-preserving geometry capacity after every tier-1 gate."""
    return {split: row["raw_capacity"] for split, row in tier1_split_support().items()}


def _verify(spec, limit=None):
    """Return ``(enriched_spec, rejection_reason)`` after exact/native proof."""
    profile = PROFILES[spec["difficulty"]]
    structural = profile_errors(spec, require_proof=False)
    if structural:
        return None, "profile_structure: " + "; ".join(structural)
    limit = profile["search_work"] if limit is None else min(limit, profile["search_work"])
    expected = layout_for(spec)
    result = _cached_generated_search(
        tuple(spec["palette"]), bool(spec["indicator"]),
        tuple(tuple(row) for row in spec["target"]), limit)
    if result.truncated:
        return None, "search_truncated"
    if not result.solved or not result.actions:
        return None, "unsolvable_or_already_solved"
    if not profile["actions"][0] <= len(result.actions) <= profile["actions"][1]:
        return None, "reference_action_length"
    mechanics = solution_mechanics(spec, result.actions)
    for mechanic, minimum in profile["minimum_mechanics"].items():
        if mechanics.get(mechanic, 0) < minimum:
            return None, "solution_mechanics_" + mechanic

    # Six levels ensure the native engine sees this row at its declared index;
    # the whole-curriculum test separately proves unforced sequential play.
    env = Env([build_level(spec) for _ in DIFFICULTIES])
    env.reset()
    context = spec["difficulty"] - 1
    if context:
        env.set_level(context)
    try:
        layout = extract(env)
    except ValueError:
        return None, "native_layout_inexact"
    if layout.start != expected.start or layout.palette != expected.palette:
        return None, "native_layout_mismatch"
    observation = None
    for action_id, x, y in result.actions:
        if action_id not in env.available_actions:
            return None, "native_action_unavailable"
        observation = env.perform(action_id, x, y)
    level_advanced = bool(observation and (
        observation.won if context == len(DIFFICULTIES) - 1 else env.level_index == context + 1))
    if not level_advanced or env.levels_completed != 1:
        return None, "native_replay_did_not_advance"
    spec = dict(spec)
    spec["solution"] = [[a, x, y] for a, x, y in result.actions]
    spec["solution_length"] = len(result.actions)
    spec["optimal_actions"] = len(result.actions)
    spec["search_expanded"] = result.expanded
    spec["search_limit"] = limit
    spec["search_truncated"] = False
    spec["engine_budget"] = names.BUDGET
    spec["usable_actions"] = names.MAX_ACTIONS
    spec["max_actions"] = profile["max_actions"]
    spec["solution_mechanics"] = mechanics
    spec["training_context_index"] = context
    spec["verification_level_index"] = context
    spec["context_engine_verified"] = True
    spec["engine_verified"] = True
    spec["proof"] = {
        "format": spec["format"],
        "generator_version": spec["generator_version"],
        "difficulty_version": spec["difficulty_version"],
        "mechanics_inventory_version": spec["mechanics_inventory_version"],
        "quality_profile_version": spec["quality_profile_version"],
        "seed": spec["seed"],
        "difficulty": spec["difficulty"],
        "split": spec["split"],
        "gameplay_sha256": spec["gameplay_sha256"],
        "geometry_sha256": spec["geometry_sha256"],
        "geometry_d4_sha256": spec["geometry_d4_sha256"],
        "geometry_version": spec["geometry_version"],
        "context_index": context,
        "level_count": len(DIFFICULTIES),
        "native_budget": names.BUDGET,
        "usable_actions": names.MAX_ACTIONS,
        "optimal_actions": len(result.actions),
        "search_limit": limit,
        "search_expanded": result.expanded,
        "search_truncated": False,
        "level_advanced": True,
        "levels_completed": env.levels_completed,
        "actual_display_coordinates": True,
    }
    errors = profile_errors(spec)
    return (None, "profile_proof: " + "; ".join(errors)) if errors else (spec, None)


def verify(spec, limit=200_000):
    """Compatibility wrapper: return an enriched spec, or ``None``."""
    return _verify(spec, limit)[0]


def validate_full_standard(spec, curriculum_entry):
    """Return errors when ``spec`` does not satisfy a shared curriculum row."""
    if not isinstance(spec, dict):
        return ["spec must be a mapping"]
    errors = []
    if not isinstance(spec.get("proof"), dict):
        errors.append("proof must be a mapping")
    if not isinstance(spec.get("solution_mechanics"), dict):
        errors.append("solution_mechanics must be a mapping")
    try:
        errors.extend(profile_errors(spec))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        errors.append(f"malformed full-standard spec: {error}")
    if not isinstance(curriculum_entry, dict):
        return errors + ["curriculum entry must be a mapping"]
    difficulty = spec.get("difficulty")
    context = spec.get("context_index")
    if type(difficulty) is not int:
        errors.append("spec difficulty must be an integer")
    if type(context) is not int:
        errors.append("spec context must be an integer")
    curriculum_difficulty = curriculum_entry.get("difficulty")
    curriculum_context = curriculum_entry.get("context_index")
    if type(curriculum_difficulty) is not int:
        errors.append("curriculum difficulty must be an integer")
    elif curriculum_difficulty != difficulty:
        errors.append("curriculum difficulty differs")
    if type(curriculum_context) is not int:
        errors.append("curriculum context must be an integer")
    elif curriculum_context != spec.get("context_index"):
        errors.append("curriculum context differs")
    work = curriculum_entry.get("search_work")
    if type(work) is not int or work <= 0:
        errors.append("curriculum search_work must be positive")
    elif type(difficulty) is int and difficulty in PROFILES and work != PROFILES[difficulty]["search_work"]:
        errors.append("curriculum search_work differs from declared tier")
    search_limit = spec.get("search_limit")
    search_expanded = spec.get("search_expanded")
    if type(search_limit) is not int or search_limit < 1:
        errors.append("search_limit must be a positive integer")
    if type(search_expanded) is not int or search_expanded < 0:
        errors.append("search_expanded must be a nonnegative integer")
    if (type(work) is int and work > 0
            and type(search_limit) is int and type(search_expanded) is int
            and (search_limit > work or search_expanded > work)):
        errors.append("certificate exceeds curriculum search_work")
    try:
        _, geometry_d4 = canonical_identities(spec)
        if geometry_d4 in _official_geometry_identities():
            errors.append("target is canonically equivalent to an official level")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        errors.append(f"cannot recompute official-copy identity: {error}")
    errors.extend(_witness_replay_errors(spec))
    return errors


def _witness_replay_errors(spec):
    """Recompute route mechanics and native outcome without repeating search."""
    errors = []
    solution = spec.get("solution")
    if not isinstance(solution, list):
        return ["solution must be a list"]
    actions = []
    for index, item in enumerate(solution):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            errors.append(f"solution action {index} must have three fields")
            continue
        action, x, y = item
        if isinstance(action, bool) or not isinstance(action, int) or action not in names.AVAILABLE_ACTIONS:
            errors.append(f"solution action {index} has invalid action id")
            continue
        if action == names.ACTION_CLICK:
            if (isinstance(x, bool) or not isinstance(x, int)
                    or isinstance(y, bool) or not isinstance(y, int)):
                errors.append(f"solution click {index} needs integer display coordinates")
                continue
        elif x is not None or y is not None:
            errors.append(f"non-click solution action {index} must not carry coordinates")
            continue
        actions.append((action, x, y))
    if errors:
        return errors
    try:
        measured = solution_mechanics(spec, actions)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        return [f"cannot recompute solution mechanics: {error}"]
    if measured != spec.get("solution_mechanics"):
        errors.append("stored solution mechanics differ from route")
    if measured.get("unrecognised_actions", 0):
        errors.append("route contains a click with no native meaning")
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return errors + ["cannot replay route outside difficulty 1..6"]
    context = difficulty - 1
    try:
        env = Env([build_level(spec) for _ in DIFFICULTIES])
        env.reset()
        if context:
            env.set_level(context)
        if env.level_index != context or env.budget != names.BUDGET or env.actions_used != 0:
            errors.append("native replay did not start in the declared fresh context/budget")
            return errors
        observation = None
        for index, (action, x, y) in enumerate(actions):
            if env.level_index != context or env.levels_completed != 0:
                errors.append("route completed before its final stored action")
                return errors
            observation = env.perform(action, x, y)
            if index < len(actions) - 1 and (observation.won or env.level_index != context):
                errors.append("route completed before its final stored action")
                return errors
        advanced = observation is not None and (
            observation.won if context == len(DIFFICULTIES) - 1 else env.level_index == context + 1)
        if not advanced or env.levels_completed != 1:
            errors.append("stored route does not win its declared native context")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        errors.append(f"native witness replay failed: {error}")
    return errors


def generate_report(seed, difficulty=1, attempts=120, limit=200_000, *, split=None,
                    record_rejection=None):
    """Bounded deterministic generation with explicit rejection diagnostics."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(difficulty) is not int or difficulty not in PROFILES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    requested_split = "train" if split is None else split
    if requested_split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    rng = random.Random(
        f"cd82:{GENERATOR_VERSION}:{QUALITY_PROFILE_VERSION}:{seed}:{difficulty}:{requested_split}")
    rejections = Counter()
    for attempt in range(1, attempts + 1):
        spec = _draft(rng, difficulty)
        if spec is None:
            reason = "profile_structure"
        else:
            gameplay, geometry_d4 = canonical_identities(spec)
            geometry = raw_geometry_identity(spec)
            partition = geometry_split(geometry_d4)
            if partition != requested_split:
                reason = "geometry_split"
            elif geometry_d4 in _official_geometry_identities():
                reason = "official_copy"
            else:
                spec.update(
                    seed=seed,
                    generation_attempt=attempt,
                    split=requested_split,
                    gameplay_sha256=gameplay,
                    geometry_sha256=geometry,
                    geometry_d4_sha256=geometry_d4,
                    geometry_version=IDENTITY_VERSION,
                    official_copy=False,
                )
                accepted, reason = _verify(spec, limit)
                if accepted is not None:
                    accepted["generation_exclusions"] = dict(sorted(rejections.items()))
                    report = {
                        "spec": accepted,
                        "attempts": attempt,
                        "accepted": 1,
                        "rejections": dict(sorted(rejections.items())),
                    }
                    generate_report.last_report = report
                    return report
        rejections[reason] += 1
        if record_rejection:
            record_rejection({"seed": seed, "difficulty": difficulty, "attempt": attempt,
                              "split": requested_split, "reason": reason})
    report = {"spec": None, "attempts": attempts, "accepted": 0,
              "rejections": dict(sorted(rejections.items()))}
    generate_report.last_report = report
    return report


generate_report.last_report = None


def generate(seed, difficulty=1, attempts=120, limit=200_000, *, split=None,
             record_rejection=None):
    """A verified, JSON-serialisable level spec for `seed`, or None.

    Deterministic for ``(seed, difficulty, split)``.  A splitless call retains
    the historical API and defaults to the training partition.
    """
    report = generate_report(seed, difficulty, attempts, limit, split=split,
                             record_rejection=record_rejection)
    generate.last_report = report
    return report["spec"]


generate.last_report = None


def _child_seed(game_seed, ordinal, difficulty):
    payload = json.dumps(
        [FULL_STANDARD_CONTRACT["source_id"], game_seed, ordinal, difficulty],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8,
                                           person=b"pebby-cd82-v2").digest(), "big")


def build_game(specs):
    """Validate and build exactly one ordered native six-level curriculum."""
    if not isinstance(specs, (list, tuple)) or len(specs) != len(DIFFICULTIES):
        raise ValueError(f"full CD82 game requires exactly {len(DIFFICULTIES)} specs")
    split = specs[0].get("split") if isinstance(specs[0], dict) else None
    seen = {field: set() for field in
            ("seed", "gameplay_sha256", "geometry_sha256")}
    for index, (spec, entry) in enumerate(zip(specs, FULL_STANDARD_CONTRACT["curriculum"])):
        if not isinstance(spec, dict):
            raise ValueError(f"spec {index} must be a mapping")
        if spec.get("difficulty") != index + 1 or spec.get("context_index") != index:
            raise ValueError(f"spec {index} is out of full-curriculum order/context")
        if spec.get("split") != split:
            raise ValueError("full game mixes data splits")
        for field, values in seen.items():
            value = spec.get(field)
            try:
                if value in values:
                    raise ValueError(f"full game contains duplicate {field}")
                values.add(value)
            except TypeError:
                raise ValueError(f"spec {index} has invalid {field}") from None
        errors = validate_full_standard(spec, entry)
        if errors:
            raise ValueError(f"spec {index} fails full standard: {'; '.join(errors)}")
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    env.reset()
    for index, spec in enumerate(specs):
        if env.level_index != index:
            raise ValueError("native game shifted level context before replay")
        for step, action in enumerate(spec["solution"]):
            observation = env.perform(*action)
            if step < len(spec["solution"]) - 1 and env.level_index != index:
                raise ValueError("native game advanced before stored route ended")
        if env.levels_completed != index + 1:
            raise ValueError(f"native game failed to complete tier {index + 1}")
    if not observation.won or env.levels_completed != len(DIFFICULTIES):
        raise ValueError("native full game did not reach WIN")
    return levels


def generate_game(seed, *, split, difficulties=None, attempts=400, limit=200_000,
                  record_rejection=None):
    """Generate ordered rows using stable BLAKE2-derived per-tier seeds."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    requested = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if (not requested
            or any(type(difficulty) is not int or difficulty not in DIFFICULTIES
                   for difficulty in requested)
            or any(left >= right for left, right in zip(requested, requested[1:]))):
        raise ValueError("difficulties must be a strictly increasing sequence from 1..6")
    specs = []
    tier_reports = []
    for ordinal, difficulty in enumerate(requested):
        child_seed = _child_seed(seed, ordinal, difficulty)
        report = generate_report(child_seed, difficulty, attempts, limit, split=split,
                                 record_rejection=record_rejection)
        tier_reports.append({**{key: value for key, value in report.items() if key != "spec"},
                             "difficulty": difficulty, "child_seed": child_seed})
        if report["spec"] is None:
            generate_game.last_report = {
                "accepted": 0,
                "failed_difficulty": difficulty,
                "tiers": tier_reports,
            }
            return None
        specs.append(report["spec"])

    if requested != DIFFICULTIES:
        generate_game.last_report = {"accepted": 1, "full_game": False, "tiers": tier_reports}
        return specs

    try:
        build_game(specs)
    except ValueError as error:
        generate_game.last_report = {
            "accepted": 0, "reason": f"sequential_native_replay_failed: {error}",
            "tiers": tier_reports,
        }
        return None

    episode_sha256 = hashlib.sha256(json.dumps(
        [spec["gameplay_sha256"] for spec in specs], separators=(",", ":")
    ).encode()).hexdigest()
    enriched = []
    for index, spec in enumerate(specs):
        row = dict(spec)
        row["proof"] = dict(spec["proof"])
        row.update(
            episode_engine_verified=True,
            episode_index=index,
            episode_levels_completed=len(DIFFICULTIES),
            episode_sha256=episode_sha256,
        )
        row["proof"].update(
            episode_engine_verified=True,
            episode_index=index,
            episode_levels_completed=len(DIFFICULTIES),
            episode_sha256=episode_sha256,
            forced_transitions=0,
        )
        enriched.append(row)
    generate_game.last_report = {
        "accepted": 1,
        "full_game": True,
        "levels_completed": len(DIFFICULTIES),
        "episode_sha256": episode_sha256,
        "tiers": tier_reports,
    }
    return enriched


generate_game.last_report = None


def generate_episode(seed, attempts=400, limit=200_000, *, split=None,
                     record_rejection=None):
    """Compatibility name for full ``generate_game``."""
    requested_split = "train" if split is None else split
    result = generate_game(seed, split=requested_split, attempts=attempts, limit=limit,
                           record_rejection=record_rejection)
    generate_episode.last_report = generate_game.last_report
    return result


generate_episode.last_report = None
