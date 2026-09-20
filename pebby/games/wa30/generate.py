"""Full nine-tier procedural WA30 generator.

Drafts use newly sampled geometry and assignments. Admission requires a
bounded exact-state positive witness, replay in the real engine at the intended
native context, reference-profile checks, mechanic-use evidence, and an
independent D4 geometry split. Search cutoffs are rejection reasons, never
proofs of impossibility.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
import hashlib
import json
from numbers import Integral
import random

from arcengine import Level, Sprite

from . import names
from .env import Env, official_levels, replay, upstream
from .layout import extract
from .plan import DEAD, SearchResult, State, complete, search, state_of, transition
from .quality import (
    GAMEPLAY_VERSION,
    GEOMETRY_VERSION,
    SPLITS,
    action_hash,
    gameplay_hash,
    geometry_hash,
    geometry_partition,
)
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    MECHANIC_INVENTORY,
    PROFILES,
    in_range,
)


FORMAT = "pebby.wa30.level.v2"
GENERATOR_VERSION = 2
SOURCE_ID = "wa30-ee6fef47"
MECHANICS_VERSION = "wa30-full-native-v1"
QUALITY_VERSION = "wa30-reference-quality-v1"
DEFAULT_ATTEMPTS = 128
DEFAULT_NODE_LIMIT = 2_200_000
MAX_ATTEMPTS = 10_000
APPROVED_PLANNER_BACKENDS = frozenset({
    "native-constructive-policy",
    "exact-transition-beam",
    "exact-transition-a-star",
})
MEASURED_WORK_SCHEMA = hasattr(SearchResult, "work")
_LAST_LEVEL_GENERATION_REPORT = ContextVar("wa30_last_level_generation_report", default=None)
_LAST_GAME_GENERATION_REPORT = ContextVar("wa30_last_game_generation_report", default=None)


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": DIFFICULTY_VERSION,
    "curriculum": [
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": PROFILES[difficulty]["search_limit"],
        }
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "wa30.md#official-reference-characterization",
        "solution_mechanics": "wa30-native-route-events-v1",
        "native_budget": "wa30-native-step-counter-v1",
        "context_engine_replay": "wa30-final-root-integration.json",
        "novelty_split": GEOMETRY_VERSION,
        "bounded_rejections": "default-128-attempt-generation",
        "independent_review": "native-astra-wa30-final/review.md",
    },
    "caveats": [
        "one shipped level per tier yields engineering tolerances, not population confidence intervals",
        "positive witnesses are bounded and nonoptimal; rejection cap hits remain unknown, not unsolvability proofs",
        "the finite tier-8 grammar has ten D4 identity classes and does not hold out graph-isomorphic topology families",
    ],
}


def goal_sprite(cells_wide, cells_high):
    """A native-looking goal region covering whole 4-pixel cells."""
    width, height = names.CELL * cells_wide, names.CELL * cells_high
    prototype = names.SPRITE_GOALS.get((width, height))
    if prototype is not None:
        return upstream().sprites[prototype].clone()
    pixels = [
        [
            names.GOAL_BORDER_COLOR
            if x in (0, width - 1) or y in (0, height - 1)
            else names.GOAL_FILL_COLOR
            for x in range(width)
        ]
        for y in range(height)
    ]
    return Sprite(
        pixels=pixels,
        name=f"pebby_goal_{cells_wide}x{cells_high}",
        visible=True,
        collidable=False,
        tags=[names.TAG_GOAL],
    )


def bad_sprite(cells_wide, cells_high):
    width, height = names.CELL * cells_wide, names.CELL * cells_high
    prototype = names.SPRITE_BAD.get((width, height))
    if prototype is not None:
        return upstream().sprites[prototype].clone()
    return Sprite(
        pixels=[[names.BAD_FILL_COLOR] * width for _ in range(height)],
        name=f"pebby_bad_{cells_wide}x{cells_high}",
        visible=True,
        collidable=False,
        tags=[names.TAG_BAD],
    )


def build_level(spec):
    """Build a real ARCEngine level using the vendored sprite vocabulary."""
    protos = upstream().sprites
    sprites = []
    for col, row in spec.get("walls", ()):
        sprites.append(protos[names.SPRITE_WALL].clone().set_position(*names.cell_to_pixel(col, row)))
    for col, row in spec.get("fences", ()):
        sprites.append(protos[names.SPRITE_FENCE].clone().set_position(*names.cell_to_pixel(col, row)))
    goals = spec.get("goals")
    if goals is None and "goal" in spec:  # v1 compatibility
        goals = [spec["goal"]]
    for col, row, width, height in goals or ():
        sprites.append(goal_sprite(width, height).set_position(*names.cell_to_pixel(col, row)))
    for col, row, width, height in spec.get("bad_regions", ()):
        sprites.append(bad_sprite(width, height).set_position(*names.cell_to_pixel(col, row)))
    for col, row in spec.get("helpers", ()):
        sprites.append(protos[names.SPRITE_HELPER].clone().set_position(*names.cell_to_pixel(col, row)))
    for col, row in spec.get("thieves", ()):
        sprites.append(protos[names.SPRITE_THIEF].clone().set_position(*names.cell_to_pixel(col, row)))
    for col, row in spec["boxes"]:
        sprites.append(protos[names.SPRITE_BOX].clone().set_position(*names.cell_to_pixel(col, row)))
    col, row = spec["player"]
    sprites.append(protos[names.SPRITE_PLAYER].clone().set_position(*names.cell_to_pixel(col, row)))
    return Level(
        sprites=sprites,
        grid_size=(names.FRAME_SIZE, names.FRAME_SIZE),
        data={
            names.KEY_STEP_COUNTER: int(spec["budget"]),
            "PebbyGenerated": spec.get("format") == FORMAT,
        },
    )


def _region_cells(regions):
    cells = set()
    for col, row, width, height in regions:
        cells.update((col + dx, row + dy) for dx in range(width) for dy in range(height))
    return cells


def _static_component(spec, *, include_walls=True, include_fences=True):
    """Static native-origin component containing the generated player.

    Movable actors and boxes are intentionally ignored, matching the topology
    measurement used for the official reference tiers.  Ordinary wall and
    fence sprite origins are the only blocked lattice cells.
    """
    blocked = set(map(tuple, spec.get("walls", ()))) if include_walls else set()
    if include_fences:
        blocked |= set(map(tuple, spec.get("fences", ())))
    start = tuple(spec["player"])
    if start in blocked:
        return set()
    seen = {start}
    queue = deque([start])
    while queue:
        col, row = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cell = (col + dx, row + dy)
            if (0 <= cell[0] < names.GRID_COLS and 0 <= cell[1] < names.GRID_ROWS
                    and cell not in blocked and cell not in seen):
                seen.add(cell)
                queue.append(cell)
    return seen


def _nearest_goal_distance(origin, goals, blocked):
    if origin in blocked:
        return None
    seen = {origin}
    queue = deque([(origin, 0)])
    while queue:
        cell, distance = queue.popleft()
        if cell in goals:
            return distance
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dx, cell[1] + dy)
            if (0 <= nxt[0] < names.GRID_COLS and 0 <= nxt[1] < names.GRID_ROWS
                    and nxt not in blocked and nxt not in seen):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def _route_detour(spec, *, omit):
    goals = _region_cells(spec.get("goals", ()))
    walls = set(map(tuple, spec.get("walls", ())))
    fences = set(map(tuple, spec.get("fences", ())))
    blocked = walls | fences
    relaxed = blocked - (walls if omit == "walls" else fences)
    origins = [tuple(spec["player"])]
    for key in ("boxes", "helpers", "thieves"):
        origins.extend(map(tuple, spec.get(key, ())))
    detour = 0
    for origin in origins:
        constrained = _nearest_goal_distance(origin, goals, blocked)
        direct = _nearest_goal_distance(origin, goals, relaxed)
        if direct is None:
            continue
        if constrained is None:
            detour += names.GRID_COLS * names.GRID_ROWS
        else:
            detour += max(0, constrained - direct)
    return detour


def structural_metrics(spec):
    goals = _region_cells(spec.get("goals", ()))
    bad = _region_cells(spec.get("bad_regions", ()))
    component = _static_component(spec)
    without_walls = _static_component(spec, include_walls=False)
    without_fences = _static_component(spec, include_fences=False)
    occupied = (
        set(map(tuple, spec.get("walls", ())))
        | set(map(tuple, spec.get("fences", ())))
        | set(map(tuple, spec.get("boxes", ())))
        | set(map(tuple, spec.get("helpers", ())))
        | set(map(tuple, spec.get("thieves", ())))
        | {tuple(spec["player"])}
        | goals
        | bad
    )
    return {
        "box_count": len(spec.get("boxes", ())),
        "helper_count": len(spec.get("helpers", ())),
        "thief_count": len(spec.get("thieves", ())),
        "wall_count": len(set(map(tuple, spec.get("walls", ())))),
        "fence_count": len(set(map(tuple, spec.get("fences", ())))),
        "goal_cells": len(goals),
        "bad_cells": len(bad),
        "player_component_cells": len(component),
        "player_component_goal_cells": len(component & goals),
        "helpers_in_player_component": sum(tuple(cell) in component for cell in spec.get("helpers", ())),
        "thieves_in_player_component": sum(tuple(cell) in component for cell in spec.get("thieves", ())),
        "boxes_in_player_component": sum(tuple(cell) in component for cell in spec.get("boxes", ())),
        "boxes_on_fences": sum(
            tuple(cell) in set(map(tuple, spec.get("fences", ())))
            for cell in spec.get("boxes", ())
        ),
        "wall_removed_component_gain": max(0, len(without_walls) - len(component)),
        "fence_removed_component_gain": max(0, len(without_fences) - len(component)),
        "wall_route_detour": _route_detour(spec, omit="walls"),
        "fence_route_detour": _route_detour(spec, omit="fences"),
        "visual_cell_density": len(occupied) / (names.GRID_COLS * names.GRID_ROWS),
    }


def profile_errors(spec, *, require_proof=True):
    errors = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty must be an integer in 1..9"]
    profile = PROFILES[difficulty]
    if spec.get("difficulty_version") != DIFFICULTY_VERSION:
        errors.append("missing calibrated difficulty version")
    try:
        metrics = structural_metrics(spec)
        for metric, profile_key in (
            ("box_count", "boxes"), ("helper_count", "helpers"),
            ("thief_count", "thieves"), ("wall_count", "walls"),
            ("fence_count", "fences"), ("goal_cells", "goal_cells"),
            ("bad_cells", "bad_cells"),
        ):
            if not in_range(metrics[metric], profile[profile_key]):
                errors.append(f"{metric} outside reference tolerance")
        if spec.get("budget") != profile["budget"]:
            errors.append("native budget differs from reference tier")
        walls = list(map(tuple, spec.get("walls", ())))
        actor_cells = []
        for key in ("boxes", "helpers", "thieves"):
            actor_cells.extend(map(tuple, spec.get(key, ())))
        actor_cells.append(tuple(spec["player"]))
        all_cells = walls + actor_cells
        if len(all_cells) != len(set(all_cells)):
            errors.append("actors, boxes, and walls must have distinct origins")
        if any(not (0 <= x < 16 and 0 <= y < 16) for x, y in walls):
            errors.append("wall origin outside native lattice")
        if any(not (0 <= x < 16 and 0 <= y < 15) for x, y in actor_cells):
            errors.append("sprite origin outside visible non-HUD lattice")
        goals = _region_cells(spec.get("goals", ()))
        bad = _region_cells(spec.get("bad_regions", ()))
        if any(tuple(box) in goals or tuple(box) in bad for box in spec.get("boxes", ())):
            errors.append("boxes must start outside goal and bad regions")
        if goals & bad:
            errors.append("goal and bad regions overlap")
        topology = profile.get("topology", {})
        for metric, bounds in topology.get("ranges", {}).items():
            if not in_range(metrics[metric], bounds):
                errors.append(f"{metric} outside reference topology tolerance")
        for metric, minimum in topology.get("minimums", {}).items():
            if metrics[metric] < minimum:
                errors.append(f"{metric} lacks required native topology effect")
        if require_proof:
            length = spec.get("solution_length", -1)
            if not in_range(length, profile["actions"]):
                errors.append("positive witness length outside reference tolerance")
            if length > spec.get("budget", -1):
                errors.append("positive witness exceeds native budget")
            for key in ("context_index", "training_context_index", "verification_level_index"):
                if spec.get(key) != difficulty - 1:
                    errors.append(f"{key} differs from official tier context")
            mechanics = spec.get("solution_mechanics", {})
            if mechanics.get("manual_grabs", 0) < 1:
                errors.append("winning route does not exercise manual grab")
            if profile["helpers"][0] and mechanics.get("helper_grabs", 0) < 1:
                errors.append("winning route does not exercise helper acquisition")
            if profile["helpers"][0] and mechanics.get("helper_deliveries", 0) < 1:
                errors.append("winning route does not exercise helper delivery")
            if profile["fences"][0] and mechanics.get("fence_box_moves", 0) < 1:
                errors.append("winning route does not exercise box/fence semantics")
            if difficulty in (3, 4):
                required_handoffs = metrics["boxes_in_player_component"]
                if mechanics.get("manual_fence_entries", 0) < required_handoffs:
                    errors.append("winning route does not transport each player-side box to a fence")
                if mechanics.get("player_helper_handoffs", 0) < required_handoffs:
                    errors.append("winning route lacks distinct player-to-helper fence handoffs")
                if mechanics.get("player_helper_deliveries", 0) < required_handoffs:
                    errors.append("winning route lacks helper delivery of the handoff boxes")
            if profile["thieves"][0]:
                if mechanics.get("thief_grabs", 0) < 1 or mechanics.get("thief_bad_deliveries", 0) < 1:
                    errors.append("winning route does not exercise thief bad-region delivery")
                if mechanics.get("thieves_destroyed", 0) < 1:
                    errors.append("winning route does not counter a thief")
                if difficulty in (6, 7) and mechanics.get("player_steals", 0) < 1:
                    errors.append("winning route lacks a native player holder-steal witness")
                if difficulty == 8 and mechanics.get("thief_steals_from_helper", 0) < 1:
                    errors.append("winning route lacks the helper-to-thief holder-steal witness")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"malformed reference spec: {exc}")
    return errors


def _rectangles(count, col, row, max_width=3):
    width = min(max_width, count)
    height, remainder = divmod(count, width)
    regions = []
    if height:
        regions.append([col, row, width, height])
    if remainder:
        regions.append([col, row + height, remainder, 1])
    return regions


def _wall_cells(rng, count, reserved, difficulty, *, corridor_top=None):
    """Build count-exact native-origin barriers, never decorative padding."""
    if count == 0:
        return []
    if difficulty == 7:
        top = corridor_top
        bottom = top + 6
        candidates = (
            [(col, top) for col in range(16)]
            + [(col, bottom) for col in range(16)]
            + [(4 + rng.randrange(2), top + 2), (9 + rng.randrange(2), top + 4)]
        )
    elif difficulty == 4:
        top = corridor_top
        candidates = (
            [(col, top) for col in range(11, 16)]
            + [(col, top + 8) for col in range(8, 16)]
        )
    elif difficulty == 5:
        gaps = set(rng.sample(range(2, 14), 2))
        candidates = (
            [(9, row) for row in range(16) if row not in gaps]
            + [(8, row) for row in range(4, 10)]
        )
    elif difficulty in (6, 8):
        if difficulty == 6:
            gaps = {rng.randrange(3, 13)}
        else:
            gate = rng.choice((4, 10))
            gaps = {gate, gate + 1}
        first_column = 8 + rng.randrange(2) if difficulty == 6 else 9
        candidates = [
            (col, row)
            for col in (first_column, 11)
            for row in range(16)
            if row not in gaps
        ]
    elif difficulty == 9:
        gaps = {9, 10}
        candidates = (
            [(col, row) for col in (7, 9, 10) for row in range(16) if row not in gaps]
            + [(col, 13) for col in (*range(7), 8)]
        )
    else:
        candidates = []
    cells = []
    for cell in candidates:
        if cell not in reserved and cell not in cells:
            cells.append(cell)
    return sorted(cells) if len(cells) == count else []


def _draft(rng, difficulty):
    profile = PROFILES[difficulty]
    box_count = profile["boxes"][0]
    helper_count = profile["helpers"][0]
    thief_count = profile["thieves"][0]
    goal_count = profile["goal_cells"][0]
    bad_count = profile["bad_cells"][0]

    corridor_top = rng.randint(3, 5) if difficulty == 4 else None
    goal_column = 12 + rng.randrange(3) if difficulty == 4 else 14
    if difficulty == 7:
        corridor_top = rng.choice((3, 4, 5))
        middle = corridor_top + 3
        goals = [[1 + rng.randrange(4), middle, 1, 2]]
        bad_regions = [[11 + rng.randrange(3), middle, 1, 2]]
    else:
        goal_row = rng.choice((5, 6)) if difficulty == 6 else rng.randint(1, max(1, 12 - (goal_count + 2) // 3))
        if difficulty == 1:
            goals = [[12, goal_row, 1, goal_count]]
        elif difficulty == 8:
            goals = [
                region
                for row in (0, 3, 6, 9, 12)
                for region in ([12, row, 1, 1], [14, row, 2, 1])
            ] + [[12, 15, 1, 1], [14, 15, 1, 1]]
        elif difficulty in (2, 3, 4, 5, 9):
            goal_row = corridor_top + 1 if difficulty == 4 else rng.randint(1, 15 - goal_count)
            goals = [[goal_column, goal_row, 1, goal_count]]
        else:
            goals = _rectangles(goal_count, 12, goal_row, max_width=3)
        if difficulty == 6:
            bad_regions = [[13, 9 + rng.randrange(3), 2, 2]]
        else:
            bad_col = 4 if difficulty == 9 else 6
            bad_regions = _rectangles(bad_count, bad_col, 10 - min(3, bad_count // 3), max_width=3) if bad_count else []
    goal_cells = _region_cells(goals)
    bad_cells = _region_cells(bad_regions)

    fences = []
    cage = None
    if difficulty == 3:
        divider = rng.choice((7, 8, 9))
        fences = [(divider, row) for row in range(16)]
    elif difficulty == 4:
        width, height = 6, 7
        left = rng.randint(1, 4)
        top = corridor_top
        right, bottom = left + width - 1, top + height - 1
        cage = (left, top, right, bottom)
        fences = (
            [(col, top) for col in range(left, right + 1)]
            + [(col, bottom) for col in range(left, right + 1)]
            + [(left, row) for row in range(top + 1, bottom)]
            + [(right, row) for row in range(top + 1, bottom)]
        )
    elif difficulty == 9:
        fence_col = 12
        fence_top = rng.randint(3, 8)
        fences = [(fence_col, row) for row in range(fence_top, fence_top + 6)]

    walls = _wall_cells(
        rng, profile["walls"][0], goal_cells | bad_cells | set(fences), difficulty,
        corridor_top=fence_top if difficulty == 9 else corridor_top,
    )
    if len(walls) != profile["walls"][0]:
        return None

    boxes = []
    helpers = []
    thieves = []
    used = set(walls) | goal_cells | bad_cells

    if difficulty == 1:
        box_rows = list(range(max(1, goal_row - 2), min(14, goal_row + 4)))
        candidates = [(col, row) for row in box_rows for col in range(7, 11) if (col, row) not in used]
        rng.shuffle(candidates)
        boxes = candidates[:box_count]
    elif difficulty == 2:
        shift = rng.randrange(2)
        boxes = [
            (9, 2 + shift), (9, 4 + shift), (9, 6 + shift),
            (10, 8 + shift), (10, 10 + shift),
        ]
        rng.shuffle(boxes)
    elif difficulty == 3:
        rows = list(range(2, 14))
        rng.shuffle(rows)
        inside_rows = sorted(rows[:3])
        fence_rows = sorted(rows[3:5])
        boxes = (
            [(divider - 2, row) for row in inside_rows]
            + [(divider, row) for row in fence_rows]
        )
        helpers = [(divider + 1, fence_rows[0])]
    elif difficulty == 4:
        left, top, right, bottom = cage
        # Keep the caged transport leg comparable to the shipped 84-action
        # witness: five boxes begin one cell from the right fence, while the
        # sixth begins two cells from it.  Translation, the sixth row, ordered
        # assignment, goal column, and exterior actors still vary.
        interior = [(4, row) for row in range(1, 6)]
        interior.append((3, rng.choice((2, 3, 4))))
        boxes = [(left + dx, top + dy) for dx, dy in interior]
        boxes.append((right + 2, top + 2))
        helpers = [
            (right + 1, top + 2),
            (right + 1, top + 4),
            (right + 1, bottom),
        ]
        rng.shuffle(boxes)
    elif difficulty == 7:
        middle = corridor_top + 3
        contested = (7 + rng.randrange(3), middle)
        boxes = [contested, (5 + rng.randrange(3), middle + 1)]
        thieves = [(contested[0], middle - 1)]
    elif difficulty == 5:
        shift = rng.randrange(2)
        boxes = [
            (10, 2 + shift), (10, 4 + shift), (10, 6 + shift),
            (11, 8 + shift), (11, 10 + shift), (11, 12 + shift),
        ]
        rng.shuffle(boxes)
        helpers = [(5, 3 + shift)]
    elif difficulty == 8:
        bad_left = min(x for x, _ in bad_cells)
        bad_choices = sorted(cell for cell in bad_cells if cell[0] == bad_left)
        for bad in (bad_choices[0], bad_choices[-1]):
            box = (bad[0] - 1, bad[1])
            boxes.append(box)
            thieves.append((box[0] - 1, box[1]))
        right_boxes = (
            [(13, row) for row in range(1, 14, 2)]
            + [(14, row) for row in (1, 2, 4)]
        )
        rng.shuffle(right_boxes)
        boxes.extend(right_boxes)
        boxes.append((14, 5))
    else:
        # Robot tiers stage a native acquisition beside a bad region.  With
        # helpers and thieves together, helper order first attaches the box
        # and thief order then steals that same held box.
        if thief_count:
            bad_left = min(x for x, _ in bad_cells)
            bad_choices = sorted(cell for cell in bad_cells if cell[0] == bad_left)
            for index in range(thief_count):
                choice = round(index * (len(bad_choices) - 1) / max(1, thief_count - 1))
                bad = bad_choices[choice]
                if helper_count:
                    box = (bad[0] - 1, bad[1])
                    thief = (box[0] - 1, box[1])
                else:
                    box = (bad[0] - 1, bad[1])
                    thief = (box[0], box[1] - 1)
                boxes.append(box)
                thieves.append(thief)

        if difficulty == 9:
            fence_boxes = [cell for cell in fences if cell not in boxes]
            rng.shuffle(fence_boxes)
            boxes.extend(fence_boxes)

    used |= set(boxes) | set(helpers) | set(thieves)
    if difficulty not in (1, 3, 4, 7):
        player_anchor = (boxes[0][0], boxes[0][1] + 1) if thieves else None
        columns = range(2, 8)
        candidates = [
            (col, row) for row in range(1, 15) for col in columns
            if (col, row) not in used and (col, row) not in fences
            and (col, row) != player_anchor
        ]
        rng.shuffle(candidates)
        while len(boxes) < box_count and candidates:
            cell = candidates.pop()
            boxes.append(cell)
            used.add(cell)

    if len(boxes) != box_count:
        return None

    if helper_count and difficulty not in (3, 4, 5):
        for index in range(helper_count):
            if difficulty == 9:
                target = next(box for box in boxes if box in set(fences))
                options = ([(target[0] + 1, target[1])]
                           if index == 0 else [(13, row) for row in range(1, 15)])
            elif difficulty == 8 and index == 1:
                target = boxes[-1]
                options = [(target[0] + 1, target[1])]
            elif thieves and index == 0:
                target = boxes[0]
                options = [(target[0], target[1] - 1)]
            elif fences and index == 0:
                target = next((box for box in boxes if box in set(fences)), boxes[0])
                options = [(target[0] + 1, target[1]), (target[0] - 1, target[1])]
            else:
                target = boxes[(thief_count + index) % len(boxes)]
                options = [(target[0] - 1, target[1]), (target[0] + 1, target[1]),
                           (target[0], target[1] - 1), (target[0], target[1] + 1)]
                options += [
                    (col, row) for row in range(1, 15) for col in range(1, 15)
                ]
            cell = next((value for value in options if value not in used and value not in set(fences)
                         and 0 <= value[0] < 16 and 0 < value[1] < 15), None)
            if cell is None:
                return None
            helpers.append(cell)
            used.add(cell)

    if difficulty == 3:
        inside_boxes = [box for box in boxes if box[0] < divider]
        player_options = (
            [(divider - 4, row) for _, row in inside_boxes]
            + [(col, row) for col in range(1, divider - 1) for row in range(2, 14)]
        )
    elif difficulty == 4:
        left, top, right, bottom = cage
        player_options = [
            (col, row)
            for col in range(left + 1, right)
            for row in range(top + 1, bottom)
        ]
    elif difficulty == 5:
        player_options = [(7, row) for row in range(3, 13)]
    elif thieves:
        contested = boxes[0]
        player_options = [
            (contested[0], contested[1] + 1),
            (contested[0] + 1, contested[1]),
            (contested[0], contested[1] + 2),
        ]
        if difficulty == 8:
            # Additional causal approach lanes change the player's native
            # route around the staged thief without perturbing actor order or
            # the helper-to-thief transfer. Admission still proves both.
            player_options.extend((
                (contested[0] - 1, contested[1] + 1),
                (contested[0] - 2, contested[1] + 1),
                (contested[0] - 2, contested[1] + 2),
            ))
    else:
        manual = boxes[-1]
        player_options = [(manual[0], manual[1] + 1), (manual[0] + 1, manual[1])]
    rng.shuffle(player_options)
    blocked = used | set(walls) | set(fences)
    player = next((cell for cell in player_options if cell not in blocked
                   and (difficulty not in (3, 4)
                        or (cell[0], cell[1] - 1) not in set(boxes))
                   and 0 < cell[0] < 15 and 0 < cell[1] < 15), None)
    if player is None:
        return None
    spec = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "source_id": SOURCE_ID,
        "mechanics_version": MECHANICS_VERSION,
        "quality_version": QUALITY_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "gameplay_version": GAMEPLAY_VERSION,
        "difficulty": difficulty,
        "reference_level": difficulty,
        "reference_witness_actions": profile["reference_witness_actions"],
        "reference_calibration": "aggregate measurements from one shipped tier; explicit tolerances, not population intervals",
        "cols": names.GRID_COLS,
        "rows": names.GRID_ROWS,
        "walls": walls,
        "fences": sorted(fences),
        "goals": goals,
        "bad_regions": bad_regions,
        "boxes": [list(cell) for cell in boxes],
        "helpers": [list(cell) for cell in helpers],
        "thieves": [list(cell) for cell in thieves],
        "player": list(player),
        "budget": profile["budget"],
        "source": "generated_only",
    }
    spec.update(structural_metrics(spec))
    return None if profile_errors(spec, require_proof=False) else spec


def _context_env(level, context_index):
    env = Env([level.clone() for _ in range(context_index + 1)])
    env.set_level(context_index)
    return env


_OFFICIAL_FRAME_HASHES = None
_OFFICIAL_SEMANTIC_HASHES = None


def _frame_sha256(level):
    frame = Env([level]).render()
    return hashlib.sha256(bytes(pixel for row in frame for pixel in row)).hexdigest()


def _official_frame_hashes():
    global _OFFICIAL_FRAME_HASHES
    if _OFFICIAL_FRAME_HASHES is None:
        _OFFICIAL_FRAME_HASHES = frozenset(_frame_sha256(level) for level in official_levels())
    return _OFFICIAL_FRAME_HASHES


def _official_semantic_hashes():
    """Native initial-state identities for all nine shipped levels."""
    global _OFFICIAL_SEMANTIC_HASHES
    if _OFFICIAL_SEMANTIC_HASHES is None:
        env = Env()
        values = set()
        for context in range(len(DIFFICULTIES)):
            env.set_level(context)
            layout = extract(env)
            value = {
                "budget": layout.max_steps,
                "walls": [list(cell) for cell in sorted(layout.walls)],
                "fences": [list(cell) for cell in sorted(layout.fences)],
                "goals": [[x, y, 1, 1] for x, y in sorted(layout.goals)],
                "bad_regions": [[x, y, 1, 1] for x, y in sorted(layout.bad)],
                "boxes": [list(cell) for cell in layout.boxes],
                "helpers": [list(cell) for cell in layout.helpers],
                "thieves": [list(cell) for cell in layout.thieves],
                "player": list(layout.player),
                "player_rotation": layout.rotation,
            }
            values.add(gameplay_hash(value))
        _OFFICIAL_SEMANTIC_HASHES = frozenset(values)
    return _OFFICIAL_SEMANTIC_HASHES


def is_official_semantic_copy(spec):
    """True when a row duplicates shipped native semantics independent of art."""
    return gameplay_hash(spec) in _official_semantic_hashes()


def last_generation_report():
    report = _LAST_LEVEL_GENERATION_REPORT.get()
    return None if report is None else json.loads(json.dumps(report))


def last_game_generation_report():
    report = _LAST_GAME_GENERATION_REPORT.get()
    return None if report is None else json.loads(json.dumps(report))


def _compact_destroyed(state: State) -> State:
    keep = [index for index, thief in enumerate(state.thieves) if thief != DEAD]
    head = 1 + len(state.helpers)
    compact = State(
        state.player,
        state.rotation,
        state.boxes,
        state.helpers,
        tuple(state.thieves[index] for index in keep),
        tuple(state.holds[:head]) + tuple(state.holds[head + index] for index in keep),
        state.helper_targets,
        state.thief_targets,
    )
    if not compact.helpers and not compact.thieves:
        held_cell = compact.boxes[compact.holds[0]] if compact.holds[0] >= 0 else None
        boxes = tuple(sorted(compact.boxes))
        compact = compact._replace(
            boxes=boxes,
            holds=(-1 if held_cell is None else boxes.index(held_cell),),
        )
    return compact


def _route_certificate(spec, solution):
    context = spec["difficulty"] - 1
    env = _context_env(build_level(spec), context)
    layout = extract(env)
    state = state_of(layout)
    events = Counter()
    initial_component = _static_component(spec)
    initial_player_boxes = {
        index for index, box in enumerate(state.boxes) if box in initial_component
    }
    manual_fence_boxes = set()
    player_helper_handoffs = set()
    player_helper_deliveries = set()
    result = None
    minimum_steps = env.steps_left()
    for offset, step in enumerate(solution):
        action = int(step[0] if isinstance(step, (list, tuple)) else step)
        before = state
        state = transition(layout, state, action, events)
        manual_box = before.holds[0]
        if (manual_box >= 0
                and before.boxes[manual_box] not in layout.fences
                and state.boxes[manual_box] in layout.fences):
            manual_fence_boxes.add(manual_box)
            events["manual_fence_entries"] += 1
        helper_slice = slice(1, 1 + len(state.helpers))
        before_helper_boxes = {box for box in before.holds[helper_slice] if box >= 0}
        after_helper_boxes = {box for box in state.holds[helper_slice] if box >= 0}
        player_helper_handoffs.update(
            (after_helper_boxes - before_helper_boxes)
            & manual_fence_boxes
            & initial_player_boxes
        )
        player_helper_deliveries.update(
            box for box in (before_helper_boxes - after_helper_boxes)
            if box in player_helper_handoffs and state.boxes[box] in layout.goals
        )
        events["fence_box_moves"] += sum(
            old != new and (old in layout.fences or new in layout.fences)
            for old, new in zip(before.boxes, state.boxes)
        )
        result = env.perform(action)
        minimum_steps = min(minimum_steps, env.steps_left())
        if offset < len(solution) - 1:
            if result.finished:
                raise ValueError("stored route terminates before its final action")
            actual = state_of(extract(env))
            if _compact_destroyed(state) != actual:
                raise ValueError("symbolic/native transition mismatch during route replay")
    if not complete(layout, state):
        raise ValueError("stored route does not complete symbolic WA30 state")
    if result is None or not result.won or env.levels_completed != 1:
        raise ValueError("stored route does not win its native context")
    events["player_helper_handoffs"] = len(player_helper_handoffs)
    events["player_helper_deliveries"] = len(player_helper_deliveries)
    mechanics = dict(events)
    mechanics.update(
        won=True,
        manual=bool(events["manual_grabs"]),
        helper=bool(events["helper_grabs"] and events["helper_deliveries"]),
        thief=bool(events["thief_grabs"] and events["thief_bad_deliveries"]),
        fence=bool(events["fence_box_moves"]),
    )
    return {
        "solution_mechanics": mechanics,
        "minimum_steps_left": minimum_steps,
        "final_steps_left": env.steps_left(),
        "engine_win": True,
        "levels_completed": 1,
    }


def _no_player_control(spec):
    """Bounded native control proving the fence tier is not autonomous.

    ACTION5 leaves the generated tier-3/4 player stationary and, by draft
    construction, initially faces no box. Repeating it therefore advances the
    autonomous phases without transporting a box. The official controls reach
    GAME_OVER under the same policy; generated rows must do likewise.
    """
    if spec["difficulty"] not in (3, 4):
        return None
    env = _context_env(build_level(spec), spec["difficulty"] - 1)
    initial_player = extract(env).player
    moved = False
    held_box = False
    observation = None
    actions_replayed = 0
    for actions_replayed in range(1, spec["budget"] + 1):
        observation = env.perform(names.ACTION_GRAB)
        layout = extract(env)
        moved = moved or layout.player != initial_player
        held_box = held_box or layout.holds[0] >= 0
        if observation.finished:
            break
    won = bool(observation and observation.won)
    return {
        "policy": "repeat_action_5_from_start",
        "actions_replayed": actions_replayed,
        "game_over": bool(observation and observation.finished and not observation.won),
        "won": won,
        "levels_completed": env.levels_completed,
        "player_moved": moved,
        "player_held_box": held_box,
    }


def _topology_counterfactual(spec, solution):
    """Replay a route natively after removing the defining static barrier."""
    difficulty = spec["difficulty"]
    field = "fences" if difficulty in (3, 4) else "walls" if difficulty == 7 else None
    if field is None:
        return None
    altered = {**spec, field: []}
    context = difficulty - 1
    baseline = _context_env(build_level(spec), context)
    ablated = _context_env(build_level(altered), context)
    first_divergence = None
    baseline_result = ablated_result = None
    actions_replayed = 0
    for actions_replayed, step in enumerate(solution, 1):
        action = step[0]
        baseline_result = baseline.perform(action)
        ablated_result = ablated.perform(action)
        if (first_divergence is None
                and (baseline_result.finished, baseline_result.won)
                != (ablated_result.finished, ablated_result.won)):
            first_divergence = actions_replayed
        if not baseline_result.finished and not ablated_result.finished:
            if state_of(extract(baseline)) != state_of(extract(ablated)) and first_divergence is None:
                first_divergence = actions_replayed
        if baseline_result.finished or ablated_result.finished:
            break
    return {
        "ablation": f"remove_all_{field}_origins",
        "first_state_divergence_action": first_divergence,
        "actions_replayed": actions_replayed,
        "baseline_won": bool(baseline_result and baseline_result.won),
        "ablated_won": bool(ablated_result and ablated_result.won),
    }


def certify(spec, *, node_limit=None):
    difficulty = spec["difficulty"]
    profile = PROFILES[difficulty]
    limit = min(profile["search_limit"], DEFAULT_NODE_LIMIT if node_limit is None else int(node_limit))
    if profile_errors(spec, require_proof=False):
        return None, "profile_structure"
    env = _context_env(build_level(spec), difficulty - 1)
    result = search(env, limit=limit, budget=spec["budget"])
    if result.work > limit:
        return None, "search_work_accounting"
    if not result.solved:
        return None, "search_truncated" if result.truncated else "proven_unsolvable"
    solution = [list(step) for step in result.actions]
    if not in_range(len(solution), profile["actions"]):
        return None, "reference_action_length"
    try:
        route = _route_certificate(spec, solution)
    except ValueError:
        return None, "native_replay_mismatch"
    topology_counterfactual = _topology_counterfactual(spec, solution)
    if topology_counterfactual is not None:
        if (not topology_counterfactual["baseline_won"]
                or topology_counterfactual["first_state_divergence_action"] is None):
            return None, "topology_ablation_inert"
    no_player_control = _no_player_control(spec)
    if no_player_control is not None and (
            not no_player_control["game_over"]
            or no_player_control["won"]
            or no_player_control["levels_completed"] != 0
            or no_player_control["player_moved"]
            or no_player_control["player_held_box"]):
        return None, "no_player_control_completed_or_interacted"
    if is_official_semantic_copy(spec):
        return None, "official_semantic_copy"
    start_frame_sha256 = _frame_sha256(build_level(spec))
    if start_frame_sha256 in _official_frame_hashes():
        return None, "official_copy"
    context = difficulty - 1
    proof = {
        "seed": spec["seed"],
        "difficulty": difficulty,
        "context_index": context,
        "context_engine_verified": True,
        "search_truncated": False,
        "route_kind": "positive_witness",
        "optimality_claimed": False,
        "witness_actions": len(solution),
        "engine_win": True,
        "levels_completed": 1,
        "search_limit": limit,
        "search_work_limit": limit,
        "search_work": result.work,
        "search_work_unit": "exact_transition_evaluations",
        "expanded_states": result.expanded,
        "planner_backend": result.backend,
        "native_admission_replay_actions": len(solution),
        "native_topology_counterfactual_actions": (
            0 if topology_counterfactual is None else topology_counterfactual["actions_replayed"]
        ),
        "topology_counterfactual": topology_counterfactual,
        "boxes_in_player_component": spec["boxes_in_player_component"],
        "boxes_on_fences": spec["boxes_on_fences"],
        "native_no_player_control_actions": (
            0 if no_player_control is None else no_player_control["actions_replayed"]
        ),
        "no_player_control": no_player_control,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "split": spec["split"],
        "geometry_version": GEOMETRY_VERSION,
        "geometry_split": spec["geometry_split"],
        "start_frame_sha256": start_frame_sha256,
    }
    final = {
        **spec,
        "solution": solution,
        "context_solution": solution,
        "solution_length": len(solution),
        "positive_witness_actions": len(solution),
        "optimality_claimed": False,
        "context_index": context,
        "training_context_index": context,
        "verification_level_index": context,
        "context_engine_verified": True,
        "engine_verified": True,
        "search_truncated": False,
        "search_limit": limit,
        "search_work_limit": limit,
        "search_work": result.work,
        "search_work_unit": "exact_transition_evaluations",
        "expanded_states": result.expanded,
        "planner_backend": proof["planner_backend"],
        "native_admission_replay_actions": len(solution),
        "native_topology_counterfactual_actions": proof["native_topology_counterfactual_actions"],
        "topology_counterfactual": topology_counterfactual,
        "native_no_player_control_actions": proof["native_no_player_control_actions"],
        "no_player_control": no_player_control,
        "official_copy": False,
        "start_frame_sha256": start_frame_sha256,
        **route,
        "proof": proof,
    }
    final["action_sha256"] = action_hash(solution)
    final["gameplay_sha256"] = gameplay_hash(final)
    errors = profile_errors(final)
    return (None, "mechanic_use" if errors else None) if errors else (json.loads(json.dumps(final)), None)


def generate(seed, difficulty, attempts=DEFAULT_ATTEMPTS, node_limit=DEFAULT_NODE_LIMIT, *, split="train"):
    """Generate one calibrated tier, or return ``None`` after bounded rejection."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(difficulty, bool) or not isinstance(difficulty, Integral) or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    if (isinstance(attempts, bool) or not isinstance(attempts, Integral)
            or not 1 <= attempts <= MAX_ATTEMPTS):
        raise ValueError(f"attempts must be in 1..{MAX_ATTEMPTS}")
    if isinstance(node_limit, bool) or not isinstance(node_limit, Integral) or not 0 < node_limit <= 32_000_000:
        raise ValueError("node_limit must be in 1..32000000")
    seed, difficulty = int(seed), int(difficulty)
    rng = random.Random(f"{MECHANICS_VERSION}:{seed}:{difficulty}")
    exclusions = Counter()
    rejected = []

    def reject(attempt, reason):
        exclusions[reason] += 1
        rejected.append({
            "seed": seed,
            "difficulty": difficulty,
            "attempt": attempt,
            "split": split,
            "reason": reason,
            "generator_version": GENERATOR_VERSION,
            "mechanics_version": MECHANICS_VERSION,
        })

    for attempt in range(1, int(attempts) + 1):
        spec = _draft(rng, difficulty)
        if spec is None:
            reject(attempt, "invalid_geometry")
            continue
        geometry_d4, partition = geometry_partition(spec)
        if partition != split:
            reject(attempt, "geometry_split")
            continue
        spec.update(
            seed=seed,
            requested_seed=seed,
            effective_seed=seed,
            attempt=attempt,
            generation_attempt=attempt,
            split=split,
            effective_split=split,
            geometry_sha256=geometry_hash(spec),
            geometry_d4_sha256=geometry_d4,
            geometry_split=partition,
        )
        accepted, reason = certify(spec, node_limit=node_limit)
        if accepted is not None:
            accepted["generation_exclusions"] = dict(sorted(exclusions.items()))
            accepted["generation_diagnostics"] = {
                "attempts_allowed": int(attempts),
                "attempts_used": attempt,
                "search_work_bound": min(int(node_limit), PROFILES[difficulty]["search_limit"]),
                "rejections": dict(sorted(exclusions.items())),
                "rejected_candidates": rejected,
                "terminal_reason": "accepted_native_positive_witness",
            }
            accepted["proof"]["rejection_count"] = sum(exclusions.values())
            _LAST_LEVEL_GENERATION_REPORT.set({
                "scope": "single_level",
                "status": "accepted",
                "seed": seed,
                "difficulty": difficulty,
                "split": split,
                **accepted["generation_diagnostics"],
            })
            return accepted
        reject(attempt, reason)
    _LAST_LEVEL_GENERATION_REPORT.set({
        "scope": "single_level",
        "status": "failed",
        "seed": seed,
        "difficulty": difficulty,
        "split": split,
        "attempts_allowed": int(attempts),
        "attempts_used": int(attempts),
        "search_work_bound": min(int(node_limit), PROFILES[difficulty]["search_limit"]),
        "rejections": dict(sorted(exclusions.items())),
        "rejected_candidates": rejected,
        "terminal_reason": "bounded_attempts_exhausted",
    })
    return None


def _game_level_seed(game_seed, level_index, difficulty):
    if isinstance(game_seed, bool) or not isinstance(game_seed, Integral) or game_seed < 0:
        raise ValueError("game seed must be a nonnegative integer")
    material = f"wa30-native-full-v1:{int(game_seed)}:{level_index}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split="train", difficulties=None, attempts=DEFAULT_ATTEMPTS, node_limit=DEFAULT_NODE_LIMIT):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("game seed must be a nonnegative integer")
    seed = int(seed)
    try:
        selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    except TypeError as exc:
        raise ValueError("game difficulties must be an iterable") from exc
    if not selected or any(type(value) is not int or value not in DIFFICULTIES for value in selected):
        raise ValueError(f"game difficulties must be drawn from {DIFFICULTIES}")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("game difficulties must be strictly increasing and distinct")
    full_game = selected == DIFFICULTIES
    sequence_kind = "full-official-context" if full_game else "explicit-smoke-subset"
    report_scope = "whole_game" if full_game else "explicit_smoke_subset"
    specs = []
    child_reports = []
    for level_index, difficulty in enumerate(selected):
        child = _game_level_seed(seed, level_index, difficulty)
        spec = generate(child, difficulty, attempts=attempts, node_limit=node_limit, split=split)
        child_reports.append(last_generation_report())
        if spec is None:
            _LAST_GAME_GENERATION_REPORT.set({
                "scope": report_scope, "status": "failed", "game_seed": seed,
                "split": split, "requested_difficulties": list(selected),
                "completed_difficulties": [row["difficulty"] for row in specs],
                "failed_difficulty": difficulty, "failed_child_seed": child,
                "child_reports": child_reports, "terminal_reason": "level_generation_failed",
            })
            return None
        spec.update(
            game_seed=seed, game_level_index=level_index,
            sequence_kind=sequence_kind,
        )
        specs.append(spec)
    if full_game:
        try:
            env = Env(build_game(specs))
            for index, spec in enumerate(specs):
                before = env.levels_completed
                if not replay(env, spec["solution"]):
                    raise ValueError(f"tier {index + 1} did not win sequentially")
                spec["whole_game_context_replay"] = {
                    "context_index": index,
                    "score_before": before,
                    "score_after": env.levels_completed,
                    "first_win_action": len(spec["solution"]),
                    "native_budget": spec["budget"],
                }
            if not env.finished or env.levels_completed != len(DIFFICULTIES):
                raise ValueError("whole generated game did not reach final WIN")
        except ValueError as exc:
            _LAST_GAME_GENERATION_REPORT.set({
                "scope": "whole_game", "status": "failed", "game_seed": seed,
                "split": split, "requested_difficulties": list(selected),
                "completed_difficulties": [], "child_reports": child_reports,
                "terminal_reason": f"sequential_native_replay_failed: {exc}",
            })
            return None
    _LAST_GAME_GENERATION_REPORT.set({
        "scope": report_scope, "status": "accepted", "game_seed": seed,
        "split": split, "requested_difficulties": list(selected),
        "completed_difficulties": [row["difficulty"] for row in specs],
        "child_reports": child_reports,
        "terminal_reason": (
            "accepted_full_sequential_native_replay" if full_game
            else "accepted_explicit_smoke_subset_without_whole_game_claim"
        ),
    })
    return specs


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


_MECHANIC_BOOLEAN_FIELDS = frozenset(("won", "manual", "helper", "thief", "fence"))

_OWNER_STRUCTURAL_COUNT_MIRRORS = (
    "boxes_in_player_component",
    "boxes_on_fences",
)
_OWNER_NATIVE_CONTROL_FIELDS = (
    "native_no_player_control_actions",
    "no_player_control",
)
_NO_PLAYER_CONTROL_FIELDS = frozenset((
    "policy",
    "actions_replayed",
    "game_over",
    "won",
    "levels_completed",
    "player_moved",
    "player_held_box",
))


def _owner_evidence_errors(spec, difficulty):
    """Require exact topology-owner evidence and typed proof mirrors."""
    errors = []
    proof = spec.get("proof")
    proof = proof if type(proof) is dict else {}

    for field in _OWNER_STRUCTURAL_COUNT_MIRRORS:
        value = spec.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"{field} must be a nonnegative exact integer")
        if not _typed_equal(proof.get(field), value):
            errors.append(f"proof.{field} does not mirror top-level evidence")

    for field in _OWNER_NATIVE_CONTROL_FIELDS:
        if field not in spec:
            errors.append(f"{field} is required at top level")
        if field not in proof:
            errors.append(f"proof.{field} is required")
        if not _typed_equal(proof.get(field), spec.get(field)):
            errors.append(f"proof.{field} does not mirror top-level evidence")

    actions = spec.get("native_no_player_control_actions")
    control = spec.get("no_player_control")
    if difficulty not in (3, 4):
        if type(actions) is not int or actions != 0 or control is not None:
            errors.append("unexpected native no-player control evidence")
        return errors

    if (type(actions) is not int or actions < 1 or type(control) is not dict
            or set(control) != _NO_PLAYER_CONTROL_FIELDS
            or control.get("policy") != "repeat_action_5_from_start"
            or type(control.get("actions_replayed")) is not int
            or control.get("actions_replayed", 0) < 1
            or not _typed_equal(control.get("actions_replayed"), actions)
            or type(control.get("levels_completed")) is not int
            or control.get("levels_completed", -1) < 0
            or any(type(control.get(field)) is not bool for field in (
                "game_over", "won", "player_moved", "player_held_box"
            ))):
        errors.append("native no-player control evidence is malformed")
    return errors


def _sha256_field(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


_DRAFT_PROVENANCE_FIELDS = (
    "format", "generator_version", "source_id", "mechanics_version",
    "quality_version", "difficulty_version", "geometry_version",
    "gameplay_version", "difficulty", "reference_level",
    "reference_witness_actions", "reference_calibration", "cols", "rows",
    "walls", "fences", "goals", "bad_regions", "boxes", "helpers",
    "thieves", "player", "budget", "source", "box_count", "helper_count",
    "thief_count", "wall_count", "fence_count", "goal_cells", "bad_cells",
    "player_component_cells", "player_component_goal_cells",
    "helpers_in_player_component", "thieves_in_player_component",
    "boxes_in_player_component", "boxes_on_fences",
    "wall_removed_component_gain", "fence_removed_component_gain",
    "wall_route_detour", "fence_route_detour", "visual_cell_density",
)


def _draft_origin_errors(spec, seed, difficulty, attempt):
    """Bind accepted core puzzle content to its deterministic draft stream."""
    rng = random.Random(f"{MECHANICS_VERSION}:{seed}:{difficulty}")
    candidate = None
    for _ in range(attempt):
        candidate = _draft(rng, difficulty)
    if candidate is None:
        return ["seed/tier/attempt reproduces an invalid draft"]
    candidate = json.loads(json.dumps(candidate))
    return [
        f"seed/tier/attempt does not reproduce draft field {field}"
        for field in _DRAFT_PROVENANCE_FIELDS
        if not _typed_equal(spec.get(field), candidate.get(field))
    ]


def _strict_schema_errors(spec, curriculum_entry, difficulty):
    """Cheap JSON/provenance checks that run before native/profile consumers."""
    errors = []
    profile = PROFILES[difficulty]

    def exact_int(field, *, minimum=None, maximum=None):
        value = spec.get(field)
        if (type(value) is not int
                or minimum is not None and value < minimum
                or maximum is not None and value > maximum):
            errors.append(f"{field} must be an exact bounded integer")
            return None
        return value

    seed = exact_int("seed", minimum=0)
    requested_seed = exact_int("requested_seed", minimum=0)
    effective_seed = exact_int("effective_seed", minimum=0)
    if seed is not None and (requested_seed != seed or effective_seed != seed):
        errors.append("requested/effective seed provenance does not match seed")
    attempt = exact_int("attempt", minimum=1, maximum=MAX_ATTEMPTS)
    generation_attempt = exact_int("generation_attempt", minimum=1, maximum=MAX_ATTEMPTS)
    if attempt is not None and generation_attempt != attempt:
        errors.append("attempt and generation_attempt differ")

    for field in ("difficulty", "reference_level", "budget", "context_index",
                  "training_context_index", "verification_level_index",
                  "solution_length", "positive_witness_actions", "search_limit",
                  "expanded_states", "minimum_steps_left", "final_steps_left",
                  "levels_completed"):
        exact_int(field, minimum=0)
    search_limit = spec.get("search_limit")
    expanded = spec.get("expanded_states")
    if (type(search_limit) is int and type(expanded) is int
            and not 1 <= expanded <= search_limit <= profile["search_limit"]):
        errors.append("expanded/search work is outside the declared curriculum bound")
    if type(spec.get("planner_backend")) is not str or spec.get("planner_backend") not in APPROVED_PLANNER_BACKENDS:
        errors.append("planner_backend is not an approved measured backend")
    measured_fields = (
        "search_work", "search_work_limit", "search_work_unit",
        "native_admission_replay_actions",
    )
    if MEASURED_WORK_SCHEMA and not all(field in spec for field in measured_fields):
        errors.append("measured planner-work evidence is incomplete")
    if any(field in spec for field in measured_fields):
        work = spec.get("search_work")
        work_limit = spec.get("search_work_limit")
        if (type(work) is not int or type(work_limit) is not int
                or not 1 <= work <= work_limit <= profile["search_limit"]):
            errors.append("measured search_work/search_work_limit is malformed or over cap")
        if (spec.get("search_work_unit") != "exact_transition_evaluations"
                or spec.get("search_limit") != work_limit
                or spec.get("expanded_states") != work):
            errors.append("measured planner-work aliases/unit do not match")
        replay_actions = spec.get("native_admission_replay_actions")
        if type(replay_actions) is not int or replay_actions != spec.get("solution_length"):
            errors.append("native_admission_replay_actions must match the certified route")
    counterfactual_actions = spec.get("native_topology_counterfactual_actions")
    counterfactual = spec.get("topology_counterfactual")
    if difficulty in (3, 4, 7):
        if (type(counterfactual_actions) is not int or counterfactual_actions < 1
                or type(counterfactual) is not dict
                or set(counterfactual) != {
                    "ablation", "first_state_divergence_action", "actions_replayed",
                    "baseline_won", "ablated_won",
                }
                or type(counterfactual.get("ablation")) is not str
                or type(counterfactual.get("first_state_divergence_action")) is not int
                or counterfactual.get("first_state_divergence_action", 0) < 1
                or type(counterfactual.get("actions_replayed")) is not int
                or counterfactual.get("actions_replayed", 0) < 1
                or not _typed_equal(
                    counterfactual.get("actions_replayed"), counterfactual_actions
                )
                or counterfactual.get("first_state_divergence_action", 0)
                > counterfactual.get("actions_replayed", 0)
                or type(counterfactual.get("baseline_won")) is not bool
                or counterfactual.get("baseline_won") is not True
                or type(counterfactual.get("ablated_won")) is not bool):
            errors.append("native topology counterfactual evidence is malformed")
    elif (type(counterfactual_actions) is not int
          or counterfactual_actions != 0 or counterfactual is not None):
        errors.append("unexpected topology counterfactual evidence")

    errors.extend(_owner_evidence_errors(spec, difficulty))

    for field in ("context_engine_verified", "engine_verified", "engine_win"):
        if spec.get(field) is not True:
            errors.append(f"{field} must be the boolean true")
    for field in ("search_truncated", "optimality_claimed", "official_copy"):
        expected = field == "official_copy" and False
        if spec.get(field) is not expected:
            errors.append(f"{field} must be the boolean false")

    for field in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256",
                  "action_sha256", "start_frame_sha256"):
        if not _sha256_field(spec.get(field)):
            errors.append(f"{field} must be a lowercase SHA-256 digest")

    def cell_list(field):
        value = spec.get(field)
        if type(value) is not list:
            errors.append(f"{field} must be a JSON list")
            return
        for cell in value:
            if (type(cell) is not list or len(cell) != 2
                    or any(type(item) is not int for item in cell)
                    or not 0 <= cell[0] < names.GRID_COLS
                    or not 0 <= cell[1] < names.GRID_ROWS):
                errors.append(f"{field} entries must be exact two-integer JSON lists")
                return

    for field in ("walls", "fences", "boxes", "helpers", "thieves"):
        cell_list(field)
    player = spec.get("player")
    if (type(player) is not list or len(player) != 2
            or any(type(item) is not int for item in player)
            or not 0 <= player[0] < names.GRID_COLS
            or not 0 <= player[1] < names.GRID_ROWS):
        errors.append("player must be an exact two-integer JSON list")
    if "player_rotation" in spec and spec.get("player_rotation") != 0:
        errors.append("player_rotation is unsupported by the native builder")
    for field in ("goals", "bad_regions"):
        regions = spec.get(field)
        if type(regions) is not list:
            errors.append(f"{field} must be a JSON list")
            continue
        for region in regions:
            if (type(region) is not list or len(region) != 4
                    or any(type(item) is not int for item in region)
                    or region[2] <= 0 or region[3] <= 0
                    or region[0] < 0 or region[1] < 0
                    or region[0] + region[2] > names.GRID_COLS
                    or region[1] + region[3] > names.GRID_ROWS):
                errors.append(f"{field} entries must be positive exact four-integer JSON lists")
                break

    solution = spec.get("solution")
    context_solution = spec.get("context_solution")
    if type(solution) is not list or not solution or not _typed_equal(solution, context_solution):
        errors.append("solution/context_solution must be identical nonempty JSON lists")
    else:
        for step in solution:
            if (type(step) is not list or len(step) != 3 or type(step[0]) is not int
                    or step[0] not in names.ACTION_IDS
                    or step[1] is not None or step[2] is not None):
                errors.append("actions must be exact [action_id, null, null] JSON triples")
                break
        if (type(spec.get("solution_length")) is int
                and (len(solution) != spec.get("solution_length")
                     or len(solution) != spec.get("positive_witness_actions"))):
            errors.append("positive witness lengths do not match route")
        if type(spec.get("budget")) is int and len(solution) > spec["budget"]:
            errors.append("positive witness exceeds native budget")

    mechanics = spec.get("solution_mechanics")
    if type(mechanics) is not dict:
        errors.append("solution_mechanics must be a mapping")
    else:
        for key, value in mechanics.items():
            if type(key) is not str:
                errors.append("solution mechanic keys must be strings")
                break
            if key in _MECHANIC_BOOLEAN_FIELDS:
                if type(value) is not bool:
                    errors.append(f"solution_mechanics.{key} must be a boolean")
                    break
            elif type(value) is not int or value < 0:
                errors.append(
                    f"solution_mechanics.{key} must be a nonnegative exact integer"
                )
                break
        for key in _MECHANIC_BOOLEAN_FIELDS:
            if type(mechanics.get(key)) is not bool:
                errors.append(f"solution_mechanics.{key} must be a boolean")

    exclusions = spec.get("generation_exclusions")
    valid_exclusions = type(exclusions) is dict and all(
        type(key) is str and bool(key) and type(value) is int and value >= 0
        for key, value in exclusions.items()
    )
    if not valid_exclusions:
        errors.append("bounded rejection counters are missing or malformed")
        exclusions = {}
    diagnostics = spec.get("generation_diagnostics")
    if type(diagnostics) is not dict:
        errors.append("bounded generation diagnostics are missing or malformed")
        diagnostics = {}
    attempts_allowed = diagnostics.get("attempts_allowed")
    attempts_used = diagnostics.get("attempts_used")
    if (type(attempts_allowed) is not int or type(attempts_used) is not int
            or not 1 <= attempts_used <= attempts_allowed <= MAX_ATTEMPTS):
        errors.append("bounded attempt diagnostics are malformed")
    if attempt is not None and attempts_used != attempt:
        errors.append("accepted attempt differs from generation diagnostics")
    if (type(diagnostics.get("search_work_bound")) is not int
            or not _typed_equal(diagnostics.get("search_work_bound"), search_limit)):
        errors.append("generation search-work bound differs from proof")
    rejection_copy = diagnostics.get("rejections")
    if not _typed_equal(rejection_copy, exclusions):
        errors.append("generation rejection diagnostics differ from exclusions")
    receipts = diagnostics.get("rejected_candidates")
    rejection_total = sum(exclusions.values())
    if type(receipts) is not list or len(receipts) != rejection_total:
        errors.append("rejected-child receipts do not match rejection counts")
        receipts = []
    if attempt is not None and rejection_total != attempt - 1:
        errors.append("rejection count does not match accepted generation attempt")
    receipt_reasons = Counter()
    for ordinal, receipt in enumerate(receipts, 1):
        if type(receipt) is not dict:
            errors.append("rejected-child receipt must be a mapping")
            continue
        expected_receipt = {
            "seed": seed, "difficulty": difficulty, "attempt": ordinal,
            "split": spec.get("split"), "reason": receipt.get("reason"),
            "generator_version": GENERATOR_VERSION,
            "mechanics_version": MECHANICS_VERSION,
        }
        if (type(receipt.get("reason")) is not str or not receipt.get("reason")
                or not _typed_equal(receipt, expected_receipt)):
            errors.append("rejected-child receipt provenance is malformed")
        else:
            receipt_reasons[receipt["reason"]] += 1
    if dict(sorted(receipt_reasons.items())) != dict(sorted(exclusions.items())):
        errors.append("rejected-child receipt reasons differ from rejection counters")
    if diagnostics.get("terminal_reason") != "accepted_native_positive_witness":
        errors.append("generation terminal reason is not the accepted positive-witness state")

    for field in ("split", "effective_split", "geometry_split"):
        if type(spec.get(field)) is not str or spec.get(field) not in SPLITS:
            errors.append(f"{field} must be a valid split string")
    if not (spec.get("split") == spec.get("effective_split") == spec.get("geometry_split")):
        errors.append("requested/effective/geometry split provenance differs")

    proof = spec.get("proof")
    if type(proof) is not dict:
        errors.append("nested proof is missing or malformed")
    else:
        for field in ("seed", "difficulty", "context_index", "witness_actions",
                      "levels_completed", "search_limit", "expanded_states",
                      "generator_version", "rejection_count"):
            if type(proof.get(field)) is not int:
                errors.append(f"proof.{field} must be an exact integer")
        for field, expected in (("context_engine_verified", True),
                                ("search_truncated", False),
                                ("optimality_claimed", False),
                                ("engine_win", True)):
            if proof.get(field) is not expected:
                errors.append(f"proof.{field} must be the boolean {expected}")
        if (type(proof.get("planner_backend")) is not str
                or proof.get("planner_backend") not in APPROVED_PLANNER_BACKENDS):
            errors.append("proof.planner_backend is not approved")
        if type(proof.get("route_kind")) is not str or proof.get("route_kind") != "positive_witness":
            errors.append("proof.route_kind is invalid")
        if type(proof.get("start_frame_sha256")) is not str:
            errors.append("proof.start_frame_sha256 must be a string")
        if proof.get("rejection_count") != rejection_total:
            errors.append("proof.rejection_count differs from bounded rejections")
        for field in measured_fields:
            if field in spec and not _typed_equal(proof.get(field), spec.get(field)):
                errors.append(f"proof.{field} does not mirror top-level evidence")
        for field in ("native_topology_counterfactual_actions", "topology_counterfactual"):
            if not _typed_equal(proof.get(field), spec.get(field)):
                errors.append(f"proof.{field} does not mirror top-level evidence")

    game_fields = ("game_seed", "game_level_index", "sequence_kind")
    present = [field in spec for field in game_fields]
    if any(present) and not all(present):
        errors.append("enriched game provenance fields must be all present or all absent")
    elif all(present):
        game_seed = spec.get("game_seed")
        level_index = spec.get("game_level_index")
        if (type(game_seed) is not int or game_seed < 0 or type(level_index) is not int
                or not 0 <= level_index < len(DIFFICULTIES)):
            errors.append("game seed/index provenance is malformed")
        elif seed != _game_level_seed(game_seed, level_index, difficulty):
            errors.append("child seed does not derive from game seed/index/tier")
        if spec.get("sequence_kind") not in ("full-official-context", "explicit-smoke-subset"):
            errors.append("sequence_kind is invalid")
    if "whole_game_context_replay" in spec:
        replay = spec.get("whole_game_context_replay")
        if (type(replay) is not dict or type(spec.get("game_level_index")) is not int
                or not _typed_equal(replay, {
                    "context_index": spec["game_level_index"],
                    "score_before": spec["game_level_index"],
                    "score_after": spec["game_level_index"] + 1,
                    "first_win_action": spec.get("solution_length"),
                    "native_budget": spec.get("budget"),
                })):
            errors.append("whole-game context replay metadata is malformed")

    return errors


def validate_full_standard(spec, curriculum_entry):
    """Fail-closed validation with identity, profile, and route recomputation."""
    errors = []
    if type(spec) is not dict:
        return ["generated spec must be an object"]
    if type(curriculum_entry) is not dict:
        return ["curriculum entry must be an object"]
    difficulty = curriculum_entry.get("difficulty")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty must be in 1..9"]
    profile = PROFILES[difficulty]
    if (type(curriculum_entry.get("context_index")) is not int
            or curriculum_entry.get("context_index") != difficulty - 1):
        errors.append("curriculum context does not match tier")
    if (type(curriculum_entry.get("search_work")) is not int
            or curriculum_entry.get("search_work") != profile["search_limit"]):
        errors.append("curriculum search work differs from calibrated tier cap")
    schema_errors = _strict_schema_errors(spec, curriculum_entry, difficulty)
    if schema_errors:
        return errors + schema_errors
    try:
        errors.extend(_draft_origin_errors(
            spec, spec["seed"], difficulty, spec["generation_attempt"]
        ))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        errors.append(f"deterministic draft provenance failed: {exc}")
    errors.extend(profile_errors(spec))
    expected = {
        "format": FORMAT,
        "source_id": SOURCE_ID,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_version": QUALITY_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "gameplay_version": GAMEPLAY_VERSION,
        "difficulty": difficulty,
        "source": "generated_only",
        "context_index": difficulty - 1,
        "training_context_index": difficulty - 1,
        "verification_level_index": difficulty - 1,
        "engine_verified": True,
        "context_engine_verified": True,
        "engine_win": True,
        "levels_completed": 1,
        "search_truncated": False,
        "optimality_claimed": False,
    }
    for key, value in expected.items():
        if not _typed_equal(spec.get(key), value):
            errors.append(f"{key} is missing or inconsistent")
    solution = spec.get("solution")
    if not isinstance(solution, list) or solution != spec.get("context_solution"):
        errors.append("solution/context_solution mismatch")
    elif spec.get("solution_length") != len(solution) or spec.get("positive_witness_actions") != len(solution):
        errors.append("positive witness lengths do not match route")
    try:
        exact_geometry = geometry_hash(spec)
        geometry_d4, partition = geometry_partition(spec)
        if exact_geometry != spec.get("geometry_sha256"):
            errors.append("exact geometry identity differs from recomputation")
        if geometry_d4 != spec.get("geometry_d4_sha256"):
            errors.append("D4 geometry identity differs from recomputation")
        if partition != spec.get("geometry_split") or partition != spec.get("split"):
            errors.append("geometry partition differs from requested split")
        if gameplay_hash(spec) != spec.get("gameplay_sha256"):
            errors.append("gameplay identity differs from recomputation")
        if action_hash(solution) != spec.get("action_sha256"):
            errors.append("action-sequence identity differs from recomputation")
        frame_sha256 = _frame_sha256(build_level(spec))
        if frame_sha256 != spec.get("start_frame_sha256"):
            errors.append("initial frame identity differs from recomputation")
        if (frame_sha256 in _official_frame_hashes()
                or is_official_semantic_copy(spec)
                or spec.get("official_copy") is not False):
            errors.append("official-copy exclusion failed")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"identity recomputation failed: {exc}")
    proof = spec.get("proof")
    if not isinstance(proof, dict):
        errors.append("nested proof is missing")
    else:
        mirrors = {
            "seed": spec.get("seed"), "difficulty": difficulty,
            "context_index": difficulty - 1, "context_engine_verified": True,
            "search_truncated": False, "route_kind": "positive_witness",
            "optimality_claimed": False, "witness_actions": spec.get("solution_length"),
            "engine_win": True, "levels_completed": 1,
            "search_limit": spec.get("search_limit"),
            "expanded_states": spec.get("expanded_states"),
            "planner_backend": spec.get("planner_backend"),
            "native_topology_counterfactual_actions": spec.get("native_topology_counterfactual_actions"),
            "topology_counterfactual": spec.get("topology_counterfactual"),
            "generator_version": GENERATOR_VERSION, "mechanics_version": MECHANICS_VERSION,
            "split": spec.get("split"), "geometry_version": GEOMETRY_VERSION,
            "geometry_split": spec.get("geometry_split"),
            "start_frame_sha256": spec.get("start_frame_sha256"),
        }
        for key, value in mirrors.items():
            if not _typed_equal(proof.get(key), value):
                errors.append(f"proof.{key} does not mirror top-level evidence")
    exclusions = spec.get("generation_exclusions")
    if not isinstance(exclusions, dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in (exclusions.items() if isinstance(exclusions, dict) else ())
    ):
        errors.append("bounded rejection counters are missing or malformed")
    diagnostics = spec.get("generation_diagnostics")
    if not isinstance(diagnostics, Mapping):
        errors.append("bounded generation diagnostics are missing")
    else:
        if (type(diagnostics.get("attempts_allowed")) is not int
                or type(diagnostics.get("attempts_used")) is not int
                or not 1 <= diagnostics.get("attempts_used", 0) <= diagnostics.get("attempts_allowed", 0)):
            errors.append("bounded attempt diagnostics are malformed")
        if not _typed_equal(
            diagnostics.get("search_work_bound"), spec.get("search_limit")
        ):
            errors.append("generation search-work bound differs from proof")
        if dict(diagnostics.get("rejections", {})) != dict(exclusions or {}):
            errors.append("generation rejection diagnostics differ from exclusions")
        receipts = diagnostics.get("rejected_candidates")
        if not isinstance(receipts, list) or len(receipts) != sum((exclusions or {}).values()):
            errors.append("rejected-child receipts do not match rejection counts")
    search_limit = spec.get("search_limit")
    if type(search_limit) is not int or not 1 <= search_limit <= profile["search_limit"]:
        errors.append("stored search limit exceeds curriculum bound")
    if isinstance(solution, list):
        try:
            recomputed = _route_certificate(spec, solution)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"winning route replay failed: {type(exc).__name__}: {exc}")
        else:
            for key, value in recomputed.items():
                if not _typed_equal(spec.get(key), value):
                    errors.append(f"{key} differs from recomputed route evidence")
            topology = _topology_counterfactual(spec, solution)
            if not _typed_equal(topology, spec.get("topology_counterfactual")):
                errors.append("topology counterfactual differs from native replay")
            replayed = 0 if topology is None else topology["actions_replayed"]
            if not _typed_equal(
                replayed, spec.get("native_topology_counterfactual_actions")
            ):
                errors.append("topology counterfactual action count differs from native replay")
            no_player_control = _no_player_control(spec)
            if not _typed_equal(no_player_control, spec.get("no_player_control")):
                errors.append("no-player control differs from native replay")
            control_actions = 0 if no_player_control is None else no_player_control["actions_replayed"]
            if not _typed_equal(control_actions, spec.get("native_no_player_control_actions")):
                errors.append("no-player control action count differs from native replay")
    return errors


def build_game(specs):
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or len(specs) != len(DIFFICULTIES):
        raise ValueError("WA30 build_game requires exactly nine ordered specs")
    levels = []
    split = None
    enriched = [isinstance(spec, Mapping) and "game_seed" in spec for spec in specs]
    if any(enriched) and not all(enriched):
        raise ValueError("full game cannot mix standalone and game-enriched specs")
    game_seed = specs[0].get("game_seed") if all(enriched) else None
    geometries = set()
    gameplays = set()
    for index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if not isinstance(spec, Mapping) or spec.get("difficulty") != difficulty:
            raise ValueError("game specs must use difficulties 1..9 in order")
        if spec.get("game_level_index") is not None and spec.get("game_level_index") != index:
            raise ValueError("game level index is shifted")
        if all(enriched) and (spec.get("game_seed") != game_seed
                              or spec.get("sequence_kind") != "full-official-context"):
            raise ValueError("full game has mixed seed provenance or smoke rows")
        split = spec.get("split") if split is None else split
        if spec.get("split") != split:
            raise ValueError("game specs must all use the same split")
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError(f"spec {index} violates full standard: " + "; ".join(errors))
        geometry = spec["geometry_d4_sha256"]
        gameplay = spec["gameplay_sha256"]
        if geometry in geometries or gameplay in gameplays:
            raise ValueError("duplicate geometry or gameplay identity in generated game")
        geometries.add(geometry)
        gameplays.add(gameplay)
        levels.append(build_level(spec))
    return levels


def load_level(spec):
    if spec.get("format") not in (FORMAT, "pebby.wa30.level.v1"):
        raise ValueError(f"not a supported WA30 spec: {spec.get('format')}")
    return build_level(spec)
