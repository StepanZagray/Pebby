"""Full-mechanic, reference-calibrated procedural SK48 generation.

Only aggregate measurements of the eight shipped levels calibrate this
module. Candidate geometry, assignments, and routes are newly constructed;
official levels and action sequences are never generator inputs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import random

from arcengine import GameState, Level

from . import names
from .env import Env, official_levels, replay, upstream
from .model import extract_model, search as model_search, solved as model_solved


FORMAT = "pebby.sk48.level.v2"
GENERATOR_VERSION = 2
MECHANICS_VERSION = "sk48-full-mechanics-v1"
DIFFICULTY_VERSION = "sk48-official-eight-v1"
QUALITY_VERSION = "sk48-reference-quality-v1"
GEOMETRY_VERSION = "sk48-d4-three-way-v3"
SOURCE_ID = "sk48-d8078629"
SOURCE_KIND = "generated_only"
DIFFICULTIES = tuple(range(1, 9))
SPLITS = ("train", "validation", "test")
DEFAULT_ATTEMPTS = 32
DEFAULT_NODE_LIMIT = 250_000


# One official level per tier: tolerances are engineering bands around those
# references, not population confidence intervals. Constructive route lengths
# were measured by this package and replayed natively; no optimality is claimed.
_REFERENCE = (
    # heads, pairs, pads, rails, blockers, boundary cells, replayed route,
    # route range, non-background pixels, rendered palette size
    (2, 1, 6, 4, 0, 5, 14, (8, 26), 1684, 10),
    (2, 1, 8, 6, 0, 7, 30, (18, 48), 2572, 11),
    (3, 1, 8, 6, 0, 7, 33, (14, 52), 2578, 12),
    (5, 2, 8, 6, 0, 7, 29, (18, 50), 2524, 13),
    (2, 1, 9, 6, 1, 7, 34, (16, 54), 2532, 9),
    (4, 2, 12, 12, 1, 7, 37, (16, 60), 2632, 10),
    (4, 2, 11, 12, 0, 7, 36, (16, 60), 2668, 11),
    (4, 2, 8, 4, 0, 7, 28, (12, 48), 2572, 12),
)
PROFILES = {
    tier: {
        "reference_level": tier,
        "context_index": tier - 1,
        "heads": values[0],
        "pairs": values[1],
        "pads": values[2],
        "rails": values[3],
        "blockers": values[4],
        "boundary_cells": values[5],
        "reference_route_actions": values[6],
        "route_actions": values[7],
        "reference_non_background_pixels": values[8],
        "reference_palette_size": values[9],
        "search_work": DEFAULT_NODE_LIMIT,
    }
    for tier, values in enumerate(_REFERENCE, 1)
}


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "source_id": SOURCE_ID,
    "status": "ready",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": QUALITY_VERSION,
    "curriculum": tuple(
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": PROFILES[difficulty]["search_work"],
        }
        for difficulty in DIFFICULTIES
    ),
    "evidence": {
        "official_tier_characterization": "sk48.md#official-tier-characterization",
        "solution_mechanics": "sk48.md#route-mechanic-certificates",
        "native_budget": "sk48.md#native-budget-and-context-replay",
        "context_engine_replay": "sk48.md#native-budget-and-context-replay",
        "novelty_split": "sk48.md#identity-splits-and-novelty",
        "bounded_rejections": "sk48.md#bounded-quality-audit",
    },
    "caveats": (
        "Profile ranges are tolerances around one official level per tier.",
        "Positive routes use exact stable-state transitions and native replay; there is no global optimality or arbitrary-prefix recovery proof.",
        "Mechanic counterfactuals establish necessity for the accepted route, not for every possible solution.",
        "The grammar is finite and calibrated from one official reference per tier; the earlier 64-row audit is historical after late corrections.",
        "All corrected tier-6/7 below-floor probes reached the 50,000-state cap, so their negative outcomes remain unknown.",
        "Root admitted readiness after independent R1/R2 closure and strict per-split validation; the separately completed split runs did not directly compare one combined identity set.",
    ),
}


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _component(prototype, x, y, rotation=0, length=1):
    return {
        "prototype": prototype,
        "position": [int(x), int(y)],
        "rotation": int(rotation),
        "length": int(length),
    }


def _add_reference(components, pads, prototype, x, colors):
    components.append(_component(prototype, x, 56, 0, len(colors) + 1))
    for index, color in enumerate(colors, 1):
        pads.append({"position": [x + index * names.CELL, 56], "color": color})


def _rails_vertical(rails, x, rows):
    for y in rows:
        rails.append({"position": [x + 2, y + 2], "rotation": 0})


def _rails_horizontal(rails, y, columns):
    for x in columns:
        rails.append({"position": [x + 2, y + 2], "rotation": 90})


def _single_pair_draft(rng, difficulty):
    profile = PROFILES[difficulty]
    cells = profile["boundary_cells"]
    boundary_x = rng.choice((9, 11, 13)) if cells == 7 else rng.choice((15, 17, 19))
    boundary_y = rng.choice((6, 8))
    boundary = {
        "prototype": names.BOUNDARY_SMALL if cells == 5 else names.BOUNDARY_LARGE,
        "position": [boundary_x, boundary_y],
        "cells": cells,
    }
    components, pads, rails, blockers = [], [], [], []
    color_count = 3 if difficulty in (1, 5) else 4
    colors = rng.sample(list(names.COLORS), color_count)
    rows = [boundary_y + index * names.CELL for index in range(cells)]
    goal_y = rng.choice(rows[1:-1])
    shift = rng.choice((-1, 1))
    start_y = goal_y + shift * names.CELL
    head_x = boundary_x - names.CELL
    components.append(_component(names.HEAD_BLUE, head_x, start_y, 0, 2))
    bottom_x = rng.choice(tuple(range(2, 54 - color_count * names.CELL, names.CELL)))
    _add_reference(components, pads, names.HEAD_BLUE, bottom_x, colors)
    _rails_vertical(
        rails, head_x,
        [boundary_y + index * names.CELL for index in range(cells - 1)],
    )

    candidates = [boundary_x + index * names.CELL for index in range(1, cells)]
    rng.shuffle(candidates)
    target_xs = sorted(candidates[:color_count])
    target_colors = list(colors)
    if difficulty != 5:
        while target_colors == colors:
            rng.shuffle(target_colors)
    if difficulty == 5:
        # Duplicate colours plus a central blocker reproduce the reference's
        # pin/re-route pressure without copying its coordinates or sequence.
        reference_colors = [colors[0], colors[1], colors[0]]
        colors = list(reference_colors)
        pads[:] = []
        components.pop()  # replace the first reference head with length four
        _add_reference(components, pads, names.HEAD_BLUE, bottom_x, colors)
        lane_cells = [boundary_x + index * names.CELL for index in range(cells)]
        blocker_x = rng.choice(lane_cells[2:5])
        target_xs = [x for x in lane_cells if x != blocker_x]
        target_colors = [colors[0]] * 3 + [colors[1]] * 3
        rng.shuffle(target_colors)
        blockers.append({"position": [blocker_x, goal_y]})
    for x, color in zip(target_xs, target_colors):
        pads.append({"position": [x, goal_y], "color": color})

    if difficulty == 3:
        # A perpendicular, non-clickable chain crosses the editable target
        # field. Its pad is shared with a target cell so collision semantics
        # participate in the route.
        aux_x = rng.choice(target_xs)
        aux_y = boundary_y - names.CELL
        aux_length = (goal_y - aux_y) // names.CELL + 1
        components.append(_component(names.HEAD_BROWN, aux_x, aux_y, 90, aux_length))

    return boundary, components, pads, rails, blockers


def _auxiliary_crossbar_draft(rng):
    """Two non-clickable goal pairs controlled by one movable crossbar."""
    boundary_x, boundary_y = rng.choice(((9, 8), (11, 6), (13, 8)))
    boundary = {"prototype": names.BOUNDARY_LARGE,
                "position": [boundary_x, boundary_y], "cells": 7}
    components, pads, rails, blockers = [], [], [], []
    controller_y = boundary_y + 6 * names.CELL
    controller_x = boundary_x - names.CELL
    components.append(_component(names.HEAD_BLUE, controller_x, controller_y, 0, 5))
    _rails_vertical(rails, controller_x,
                    [boundary_y + index * names.CELL for index in range(6)])

    columns = rng.sample(
        [boundary_x + index * names.CELL for index in range(1, 6)], 2
    )
    prototypes = (names.HEAD_ORANGE, names.HEAD_BROWN)
    palette = rng.sample(list(names.COLORS), 4)
    for index, (prototype, column) in enumerate(zip(prototypes, columns)):
        colors = palette[index * 2:index * 2 + 2]
        components.append(_component(prototype, column, boundary_y - names.CELL,
                                     90, rng.choice((3, 4))))
        bottom_x = 5 + index * 30
        _add_reference(components, pads, prototype, bottom_x, colors)
        for offset, color in enumerate(colors):
            pads.append({
                "position": [column + rng.choice((-1, 1)) * names.CELL,
                             controller_y - offset * names.CELL],
                "color": color,
            })
    return boundary, components, pads, rails, blockers


def _two_pair_draft(rng, difficulty):
    boundary_x, boundary_y = rng.choice(((9, 8), (11, 8), (13, 6)))
    boundary = {"prototype": names.BOUNDARY_LARGE,
                "position": [boundary_x, boundary_y], "cells": 7}
    components, pads, rails, blockers = [], [], [], []
    count = 3 if difficulty in (6, 7) else 2
    blue = rng.sample(list(names.COLORS), count)
    white = rng.sample(list(names.COLORS), count)

    horizontal_y = boundary_y + rng.choice((4, 5)) * names.CELL
    if difficulty in (6, 7):
        horizontal_goal = boundary_y + rng.choice((1, 2)) * names.CELL
    else:
        horizontal_goal = horizontal_y - names.CELL
    hx = boundary_x - names.CELL
    components.append(_component(names.HEAD_BLUE, hx, horizontal_y, 0, 2))
    if difficulty in (6, 7):
        _rails_vertical(rails, hx,
                        [boundary_y + index * names.CELL for index in range(6)])
    else:
        _rails_vertical(rails, hx, [horizontal_goal, horizontal_y])
    _add_reference(components, pads, names.HEAD_BLUE, 3, blue)

    vertical_x = boundary_x + rng.choice((4, 5)) * names.CELL
    if difficulty in (6, 7):
        vertical_goal = boundary_x + rng.choice((1, 2)) * names.CELL
    else:
        vertical_goal = vertical_x + rng.choice((-1, 1)) * names.CELL
    vy = boundary_y - names.CELL
    components.append(_component(names.HEAD_WHITE, vertical_x, vy, 90, 2))
    if difficulty in (6, 7):
        _rails_horizontal(rails, vy,
                          [boundary_x + index * names.CELL for index in range(6)])
    else:
        _rails_horizontal(rails, vy, [min(vertical_x, vertical_goal),
                                      max(vertical_x, vertical_goal)])
    _add_reference(components, pads, names.HEAD_WHITE, 33, white)

    available_x = [boundary_x + index * names.CELL for index in range(1, 7)]
    available_y = [boundary_y + index * names.CELL for index in range(1, 7)]
    if difficulty == 6:
        horizontal_xs = sorted(
            rng.sample(available_x[:-2], count - 1) + [available_x[-2]]
        )
    else:
        horizontal_xs = sorted(rng.sample(available_x, count))
    vertical_ys = sorted(rng.sample(available_y, count))
    if difficulty in (7, 8):
        horizontal_xs[0] = vertical_goal
        horizontal_xs = sorted(set(horizontal_xs))
        while len(horizontal_xs) < count:
            horizontal_xs.append(next(x for x in available_x if x not in horizontal_xs))
            horizontal_xs.sort()
        vertical_ys[0] = horizontal_goal
        vertical_ys = sorted(set(vertical_ys))
        while len(vertical_ys) < count:
            vertical_ys.append(next(y for y in available_y if y not in vertical_ys))
            vertical_ys.sort()
        shared_index = vertical_ys.index(horizontal_goal)
        white[shared_index] = blue[horizontal_xs.index(vertical_goal)]
        pads[count + shared_index]["color"] = white[shared_index]
    for x, color in zip(horizontal_xs, blue):
        pads.append({"position": [x, horizontal_goal], "color": color})
    for y, color in zip(vertical_ys, white):
        if [vertical_goal, y] not in [pad["position"] for pad in pads]:
            pads.append({"position": [vertical_goal, y], "color": color})

    if difficulty == 6:
        blockers.append({"position": [available_x[-1], horizontal_goal]})
    return boundary, components, pads, rails, blockers


def _draft(seed, difficulty, attempt):
    material = f"{MECHANICS_VERSION}:{seed}:{difficulty}:{attempt}".encode()
    rng = random.Random(int.from_bytes(hashlib.blake2b(material, digest_size=16).digest(), "big"))
    if difficulty in (1, 2, 3, 5):
        boundary, components, pads, rails, blockers = _single_pair_draft(rng, difficulty)
    elif difficulty == 4:
        boundary, components, pads, rails, blockers = _auxiliary_crossbar_draft(rng)
    else:
        boundary, components, pads, rails, blockers = _two_pair_draft(rng, difficulty)
    return {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
        "quality_version": QUALITY_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "source": SOURCE_KIND,
        "source_id": SOURCE_ID,
        "seed": int(seed),
        "effective_seed": int(seed),
        "attempt": int(attempt),
        "difficulty": int(difficulty),
        "context_index": int(difficulty - 1),
        "training_context_index": int(difficulty - 1),
        "boundary": boundary,
        "heads": components,
        "pads": pads,
        "rails": rails,
        "blockers": blockers,
        "native_move_budget": names.MOVE_BUDGET,
    }


def _component_schema_errors(spec):
    """Validate the exact JSON component schema before any dereference/hash."""
    errors = []
    boundary = spec.get("boundary")
    if not isinstance(boundary, Mapping):
        errors.append("boundary must be a mapping")
    else:
        prototype = boundary.get("prototype")
        expected_cells = {
            names.BOUNDARY_SMALL: 5,
            names.BOUNDARY_LARGE: 7,
        }.get(prototype)
        if expected_cells is None:
            errors.append("boundary prototype is invalid")
        if type(boundary.get("cells")) is not int:
            errors.append("boundary cells must be an exact integer")
        elif expected_cells is not None and boundary.get("cells") != expected_cells:
            errors.append("boundary cells do not match the native prototype")
        position = boundary.get("position")
        if (type(position) is not list or len(position) != 2
                or any(type(value) is not int for value in position)):
            errors.append("boundary position must be two exact integers")

    definitions = (
        ("heads", {"prototype", "position", "rotation", "length"}),
        ("pads", {"position", "color"}),
        ("rails", {"position", "rotation"}),
        ("blockers", {"position"}),
    )
    for field, required in definitions:
        values = spec.get(field)
        if type(values) is not list:
            errors.append(f"{field} must be a JSON list")
            continue
        if field == "heads" and not values:
            errors.append("heads must be nonempty")
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                errors.append(f"{field}[{index}] must be a mapping")
                continue
            if not required.issubset(value):
                errors.append(f"{field}[{index}] is missing required fields")
                continue
            position = value.get("position")
            if (type(position) is not list or len(position) != 2
                    or any(type(item) is not int for item in position)):
                errors.append(f"{field}[{index}] position must be two exact integers")
            if field == "heads":
                if value.get("prototype") not in names.HEAD_PROTOTYPES:
                    errors.append(f"heads[{index}] prototype is invalid")
                if (type(value.get("rotation")) is not int
                        or value.get("rotation") not in names.DIRECTION):
                    errors.append(f"heads[{index}] rotation is invalid")
                if (type(value.get("length")) is not int
                        or not 1 <= value.get("length", 0) <= 8):
                    errors.append(f"heads[{index}] length is invalid")
            elif field == "pads":
                if (type(value.get("color")) is not int
                        or value.get("color") not in names.COLORS):
                    errors.append(f"pads[{index}] color is invalid")
            elif field == "rails":
                if (type(value.get("rotation")) is not int
                        or value.get("rotation") not in (0, 90)):
                    errors.append(f"rails[{index}] rotation is invalid")
    if errors:
        return tuple(dict.fromkeys(errors))

    heads = spec["heads"]
    pads = spec["pads"]
    if heads[0]["position"][1] >= names.HUD_ROW:
        errors.append("heads[0] must be an initial board head")

    board_heads = [
        head for head in heads if head["position"][1] < names.HUD_ROW
    ]
    footer_heads = [
        head for head in heads if head["position"][1] >= names.HUD_ROW
    ]
    footer_pads = [
        pad for pad in pads if pad["position"][1] >= names.HUD_ROW
    ]
    if not footer_heads:
        errors.append("at least one footer reference head is required")

    footer_prototypes = Counter(head["prototype"] for head in footer_heads)
    if any(count != 1 for count in footer_prototypes.values()):
        errors.append("footer head prototypes must be unique")
    for prototype in footer_prototypes:
        matching = sum(head["prototype"] == prototype for head in board_heads)
        if matching != 1:
            errors.append("each footer reference must pair with exactly one board head")

    pad_positions = Counter(tuple(pad["position"]) for pad in footer_pads)
    owners = Counter()
    for head in footer_heads:
        dx, dy = names.DIRECTION[head["rotation"]]
        positions = [
            (
                head["position"][0] + offset * dx * names.CELL,
                head["position"][1] + offset * dy * names.CELL,
            )
            for offset in range(head["length"])
        ]
        if pad_positions[positions[0]]:
            errors.append("footer reference head must not contain a color pad")
        if any(pad_positions[position] != 1 for position in positions[1:]):
            errors.append(
                "footer reference requires exactly one pad at every non-head segment"
            )
        for position in positions:
            if pad_positions[position]:
                owners[position] += 1
    if any(owners[position] != 1 for position in pad_positions):
        errors.append("footer pads must belong to exactly one reference chain")
    return tuple(dict.fromkeys(errors))


def build_level(spec):
    """Rebuild a generated level from JSON-round-trippable components."""
    if not isinstance(spec, Mapping):
        raise ValueError("generated SK48 spec must be a mapping")
    schema_errors = _component_schema_errors(spec)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))
    if spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    prototypes = upstream().sprites
    boundary = spec.get("boundary", {})
    boundary_prototype = boundary.get("prototype")
    if boundary_prototype not in (names.BOUNDARY_SMALL, names.BOUNDARY_LARGE):
        raise ValueError("unknown boundary prototype")
    bx, by = boundary.get("position", (None, None))
    bx, by = _integer(bx, "boundary x"), _integer(by, "boundary y")
    sprites = []
    occupied_segments = set()
    for index, raw in enumerate(spec.get("heads", ())):
        prototype = raw.get("prototype")
        if prototype not in names.HEAD_PROTOTYPES:
            raise ValueError(f"heads[{index}] has unknown prototype")
        x, y = raw.get("position", (None, None))
        x, y = _integer(x, "head x"), _integer(y, "head y")
        rotation = _integer(raw.get("rotation", 0), "head rotation")
        length = _integer(raw.get("length"), "head length")
        if rotation not in names.DIRECTION or length < 1 or length > 8:
            raise ValueError("invalid head rotation or length")
        sprites.append(
            prototypes[prototype].clone().set_position(x, y).set_rotation(rotation)
        )
        dx, dy = names.DIRECTION[rotation]
        for segment_index in range(length):
            position = (x + segment_index * dx * names.CELL,
                        y + segment_index * dy * names.CELL, rotation)
            if position in occupied_segments:
                raise ValueError("same-orientation segments overlap")
            occupied_segments.add(position)
            sprites.append(
                prototypes[names.SEGMENT].clone()
                .set_position(position[0], position[1]).set_rotation(rotation)
            )
    for raw in spec.get("pads", ()):
        x, y = raw.get("position", (None, None))
        color = _integer(raw.get("color"), "pad color")
        if color not in names.COLORS:
            raise ValueError("pad color is outside the official palette")
        sprites.append(
            prototypes[names.COLOR_PAD].clone()
            .set_position(_integer(x, "pad x"), _integer(y, "pad y"))
            .color_remap(None, color)
        )
    for raw in spec.get("rails", ()):
        x, y = raw.get("position", (None, None))
        rotation = _integer(raw.get("rotation", 0), "rail rotation")
        if rotation not in (0, 90):
            raise ValueError("rail rotation must be 0 or 90")
        sprites.append(
            prototypes[names.RAIL].clone()
            .set_position(_integer(x, "rail x"), _integer(y, "rail y"))
            .set_rotation(rotation)
        )
    for raw in spec.get("blockers", ()):
        x, y = raw.get("position", (None, None))
        sprites.append(
            prototypes[names.BLOCKER].clone()
            .set_position(_integer(x, "blocker x"), _integer(y, "blocker y"))
        )
    sprites.extend((
        prototypes[boundary_prototype].clone().set_position(bx, by).set_scale(6),
        prototypes[names.FOOTER].clone().set_position(0, 54).set_scale(64),
        prototypes[names.DIVIDER].clone().set_position(0, names.HUD_ROW),
    ))
    return Level(
        sprites=sprites,
        grid_size=(names.FRAME_SIZE, names.FRAME_SIZE),
        data={"grouped_pauses": False, "lit_extension": True},
        name=f"generated-sk48-d{difficulty}-s{spec.get('seed', 0)}",
    )


def _transform_point(x, y, swap, sx, sy):
    tx, ty = (y, x) if swap else (x, y)
    return sx * tx, sy * ty


def _native_boundary_cells(spec):
    """Return the cell extent actually selected by the native prototype."""
    try:
        return {
            names.BOUNDARY_SMALL: 5,
            names.BOUNDARY_LARGE: 7,
        }[spec["boundary"]["prototype"]]
    except (KeyError, TypeError) as error:
        raise ValueError("boundary prototype is invalid") from error


def _require_component_schema(spec):
    if not isinstance(spec, Mapping):
        raise ValueError("generated SK48 spec must be a mapping")
    errors = _component_schema_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))


def _reference_records(spec):
    """Extract target order/pairing while discarding footer presentation."""
    footer_heads = [
        head for head in spec["heads"]
        if head["position"][1] >= names.HUD_ROW
    ]
    footer_pads = [
        pad for pad in spec["pads"]
        if pad["position"][1] >= names.HUD_ROW
    ]
    prototype_counts = Counter(head["prototype"] for head in footer_heads)
    if any(count != 1 for count in prototype_counts.values()):
        raise ValueError("footer head prototypes must be unique")
    pad_positions = [tuple(pad["position"]) for pad in footer_pads]
    if len(set(pad_positions)) != len(pad_positions):
        raise ValueError("footer pad positions must be unique")

    used_pads = set()
    records = []
    for head in footer_heads:
        dx, dy = names.DIRECTION[head["rotation"]]
        offsets = []
        colors = []
        for offset in range(head["length"]):
            position = (
                head["position"][0] + offset * dx * names.CELL,
                head["position"][1] + offset * dy * names.CELL,
            )
            for pad_index, pad in enumerate(footer_pads):
                if tuple(pad["position"]) == position:
                    if pad_index in used_pads:
                        raise ValueError("footer pad belongs to multiple references")
                    used_pads.add(pad_index)
                    offsets.append(offset)
                    colors.append(pad["color"])
        records.append({
            "prototype": head["prototype"],
            "clickable": head["prototype"] in names.CLICKABLE_HEADS,
            "length": head["length"],
            "pad_offsets": tuple(offsets),
            "colors": tuple(colors),
        })
    if len(used_pads) != len(footer_pads):
        raise ValueError("footer pads must lie on exactly one reference chain")
    return records


def _exact_geometry_payload(spec):
    """Preserve render placement while deriving boundary size natively."""
    reverse_direction = {value: key for key, value in names.DIRECTION.items()}
    heads = []
    for head in spec["heads"]:
        dx, dy = names.DIRECTION[head["rotation"]]
        heads.append((*head["position"], reverse_direction[(dx, dy)], head["length"]))
    boundary_x, boundary_y = spec["boundary"]["position"]
    cells = _native_boundary_cells(spec)
    extent = (cells - 1) * names.CELL
    return {
        "boundary_cells": cells,
        "boundary": sorted(
            (boundary_x + dx, boundary_y + dy)
            for dx in (0, extent) for dy in (0, extent)
        ),
        "heads": sorted(heads),
        "pads": sorted(tuple(pad["position"]) for pad in spec["pads"]),
        "rails": sorted(
            (*rail["position"], "v" if rail["rotation"] == 0 else "h")
            for rail in spec["rails"]
        ),
        "blockers": sorted(tuple(blocker["position"])
                           for blocker in spec["blockers"]),
    }


def _board_variant(spec, swap, sx, sy):
    """Canonical board geometry plus footer target relations for one D4 view."""
    references = _reference_records(spec)
    reference_prototypes = {record["prototype"] for record in references}
    reverse_direction = {value: key for key, value in names.DIRECTION.items()}
    transformed_heads = []
    for index, head in enumerate(spec["heads"]):
        if head["position"][1] >= names.HUD_ROW:
            continue
        x, y = _transform_point(*head["position"], swap, sx, sy)
        dx, dy = _transform_point(*names.DIRECTION[head["rotation"]], swap, sx, sy)
        transformed_heads.append((x, y, reverse_direction[(dx, dy)], index, head))
    transformed_pads = [
        (*_transform_point(*pad["position"], swap, sx, sy), pad)
        for pad in spec["pads"] if pad["position"][1] < names.HUD_ROW
    ]
    transformed_rails = []
    for rail in spec["rails"]:
        x, y = _transform_point(*rail["position"], swap, sx, sy)
        axis = (0, 1) if rail["rotation"] == 0 else (1, 0)
        axis = _transform_point(*axis, swap, sx, sy)
        transformed_rails.append((x, y, "h" if axis[0] else "v"))
    transformed_blockers = [
        _transform_point(*blocker["position"], swap, sx, sy)
        for blocker in spec["blockers"]
    ]
    boundary_x, boundary_y = spec["boundary"]["position"]
    cells = _native_boundary_cells(spec)
    extent = (cells - 1) * names.CELL
    transformed_boundary = [
        _transform_point(boundary_x + dx, boundary_y + dy, swap, sx, sy)
        for dx in (0, extent) for dy in (0, extent)
    ]
    positions = (
        [(x, y) for x, y, _, _, _ in transformed_heads]
        + [(x, y) for x, y, _ in transformed_pads]
        + [(x, y) for x, y, _ in transformed_rails]
        + transformed_blockers
        + transformed_boundary
    )
    left = min(x for x, _ in positions)
    top = min(y for _, y in positions)

    pair_classes = {}
    auxiliary_counts = Counter()
    heads = []
    for x, y, rotation, index, head in sorted(
        transformed_heads, key=lambda row: row[:3]
    ):
        clickable = head["prototype"] in names.CLICKABLE_HEADS
        if head["prototype"] in reference_prototypes:
            class_key = (clickable, head["prototype"])
            if class_key not in pair_classes:
                pair_classes[class_key] = sum(
                    key[0] == clickable for key in pair_classes
                )
            role = ("pair", pair_classes[class_key])
        else:
            role = ("auxiliary", auxiliary_counts[clickable])
            auxiliary_counts[clickable] += 1
        heads.append((x - left, y - top, rotation, head["length"],
                      clickable, role, index == 0))

    normalized_references = []
    for record in references:
        class_key = (record["clickable"], record["prototype"])
        if class_key not in pair_classes:
            raise ValueError("footer reference has no matching board head")
        normalized_references.append({
            "clickable": record["clickable"],
            "pair_class": pair_classes[class_key],
            "length": record["length"],
            "pad_offsets": record["pad_offsets"],
            "colors": record["colors"],
        })
    normalized_references.sort(
        key=lambda record: (record["clickable"], record["pair_class"])
    )

    return {
        "boundary_cells": cells,
        "boundary": sorted((x - left, y - top)
                           for x, y in transformed_boundary),
        "heads": heads,
        "pads": sorted((x - left, y - top, pad["color"])
                       for x, y, pad in transformed_pads),
        "rails": sorted((x - left, y - top, axis)
                        for x, y, axis in transformed_rails),
        "blockers": sorted((x - left, y - top)
                           for x, y in transformed_blockers),
        "references": normalized_references,
    }


def exact_geometry_identity(spec):
    _require_component_schema(spec)
    payload = _exact_geometry_payload(spec)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _geometry_variant(spec, swap, sx, sy):
    board = _board_variant(spec, swap, sx, sy)
    return {
        "boundary_cells": board["boundary_cells"],
        "boundary": board["boundary"],
        "heads": board["heads"],
        "pads": [(x, y) for x, y, _ in board["pads"]],
        "rails": board["rails"],
        "blockers": board["blockers"],
        "references": [
            (
                record["clickable"], record["pair_class"], record["length"],
                record["pad_offsets"],
            )
            for record in board["references"]
        ],
    }


def geometry_identity(spec):
    _require_component_schema(spec)
    variants = [
        _geometry_variant(spec, swap, sx, sy)
        for swap in (False, True) for sx in (-1, 1) for sy in (-1, 1)
    ]
    payload = min(
        json.dumps(variant, sort_keys=True, separators=(",", ":"))
        for variant in variants
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _official_layout_variant(spec, swap, sx, sy):
    """Canonical physical puzzle layout, independent of initial selection."""
    variant = _geometry_variant(spec, swap, sx, sy)
    variant["heads"] = [head[:-1] for head in variant["heads"]]
    return variant


def official_layout_identity(spec):
    """D4/translation layout key used only for official-copy exclusion.

    The regular geometry/gameplay identities retain the native initial board
    selection.  This second key prevents a copied official layout from becoming
    admissible merely by selecting another board controller first.
    """
    _require_component_schema(spec)
    variants = [
        _official_layout_variant(spec, swap, sx, sy)
        for swap in (False, True) for sx in (-1, 1) for sy in (-1, 1)
    ]
    payload = min(
        json.dumps(variant, sort_keys=True, separators=(",", ":"))
        for variant in variants
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _semantic_variant(spec, swap, sx, sy):
    board = _board_variant(spec, swap, sx, sy)

    color_classes = {}
    pads = []
    for x, y, color in board["pads"]:
        color_class = color_classes.setdefault(color, len(color_classes))
        pads.append((x, y, color_class))
    references = []
    for record in board["references"]:
        colors = tuple(
            color_classes.setdefault(color, len(color_classes))
            for color in record["colors"]
        )
        references.append((
            record["clickable"], record["pair_class"], record["length"],
            record["pad_offsets"], colors,
        ))
    return {
        "geometry": geometry_identity(spec),
        "boundary_cells": board["boundary_cells"],
        "heads": board["heads"],
        "pads": pads,
        "rails": board["rails"],
        "blockers": board["blockers"],
        "references": references,
        "native_move_budget": spec["native_move_budget"],
        "difficulty": spec["difficulty"],
    }


def gameplay_identity(spec):
    _require_component_schema(spec)
    variants = [
        _semantic_variant(spec, swap, sx, sy)
        for swap in (False, True) for sx in (-1, 1) for sy in (-1, 1)
    ]
    payload = min(
        json.dumps(variant, sort_keys=True, separators=(",", ":"))
        for variant in variants
    )
    return hashlib.sha256(
        payload.encode()
    ).hexdigest()


_OFFICIAL_GEOMETRIES = None


def _official_geometry_identities():
    """Selection-independent canonical layout set for shipped levels."""
    global _OFFICIAL_GEOMETRIES
    if _OFFICIAL_GEOMETRIES is None:
        identities = set()
        for difficulty, level in enumerate(official_levels(), 1):
            env = Env([level])
            boundary = env.level.get_sprites_by_tag(names.TAG_BOUNDARY)[0]
            synthetic = {
                "difficulty": difficulty,
                "native_move_budget": names.MOVE_BUDGET,
                "boundary": {
                    "prototype": boundary.name,
                    "position": [int(boundary.x), int(boundary.y)],
                    "cells": int(boundary.pixels.shape[0]),
                },
                "heads": [
                    {
                        "prototype": head.name,
                        "position": [int(head.x), int(head.y)],
                        "rotation": int(head.rotation),
                        "length": len(env.lines()[head]),
                    }
                    for head in env.heads()
                ],
                "pads": [
                    {
                        "position": [int(pad.x), int(pad.y)],
                        "color": int(pad.pixels[1, 1]),
                    }
                    for pad in env.color_pads()
                ],
                "rails": [
                    {
                        "position": [int(rail.x), int(rail.y)],
                        "rotation": int(rail.rotation),
                    }
                    for rail in env.level.get_sprites_by_tag(names.TAG_RAIL)
                ],
                "blockers": [
                    {"position": [int(blocker.x), int(blocker.y)]}
                    for blocker in env.level.get_sprites_by_tag(names.TAG_BLOCKER)
                ],
            }
            identities.add(official_layout_identity(synthetic))
        _OFFICIAL_GEOMETRIES = frozenset(identities)
    return _OFFICIAL_GEOMETRIES


def split_for_identity(identity):
    return SPLITS[int(identity[:8], 16) % len(SPLITS)]


def structural_metrics(spec):
    env = Env([build_level(spec)])
    frame = env.render()
    colors = {cell for row in frame for cell in row}
    return {
        "heads": len(env.heads()),
        "pairs": len(env.pairs()),
        "pads": len(env.color_pads()),
        "rails": len(spec["rails"]),
        "blockers": len(spec["blockers"]),
        "boundary_cells": 5 if spec["boundary"]["prototype"] == names.BOUNDARY_SMALL else 7,
        "clickable_heads": sum(names.TAG_CLICK in head.tags for head in env.heads()),
        "auxiliary_heads": sum(names.TAG_CLICK not in head.tags for head in env.heads()),
        "orientations": len({head["rotation"] % 180 for head in spec["heads"]}),
        "rendered_non_background": sum(
            cell != upstream().BACKGROUND_COLOR for row in frame for cell in row
        ),
        "rendered_palette_size": len(colors),
    }


def profile_errors(spec, *, require_route=True):
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return ["difficulty must be an integer in 1..8"]
    profile = PROFILES[difficulty]
    errors = []
    try:
        metrics = structural_metrics(spec)
        tolerances = {
            "heads": (profile["heads"] - 1, profile["heads"] + 1),
            "pairs": (profile["pairs"], profile["pairs"]),
            "pads": (profile["pads"] - 2, profile["pads"] + 2),
            "rails": (1, max(profile["rails"], 12)),
            "blockers": (profile["blockers"], profile["blockers"]),
            "boundary_cells": (profile["boundary_cells"], profile["boundary_cells"]),
        }
        for key, (low, high) in tolerances.items():
            if not low <= metrics[key] <= high:
                errors.append(f"{key} outside tier-{difficulty} reference tolerance")
        reference_pixels = profile["reference_non_background_pixels"]
        if not (
            int(reference_pixels * 0.75)
            <= metrics["rendered_non_background"]
            <= int(reference_pixels * 1.25)
        ):
            errors.append("rendered density outside tier reference tolerance")
        reference_palette = profile["reference_palette_size"]
        if not (
            max(7, reference_palette - 3)
            <= metrics["rendered_palette_size"]
            <= min(16, reference_palette + 3)
        ):
            errors.append("rendered palette outside tier reference tolerance")
        if difficulty == 3 and metrics["auxiliary_heads"] < 1:
            errors.append("tier 3 requires a non-clickable auxiliary chain")
        if difficulty == 4 and (metrics["auxiliary_heads"] < 4 or metrics["pairs"] != 2):
            errors.append("tier 4 requires two non-clickable auxiliary pairs")
        if difficulty >= 6 and metrics["clickable_heads"] < 4:
            errors.append("late tiers require two clickable head pairs")
        if difficulty >= 6 and metrics["orientations"] < 2:
            errors.append("late tiers require mixed horizontal/vertical chains")
        if spec.get("native_move_budget") != names.MOVE_BUDGET:
            errors.append("native move budget differs from the fixed official budget")
        if require_route:
            length = spec.get("solution_length")
            low, high = profile["route_actions"]
            if type(length) is not int or not low <= length <= high:
                errors.append("solution length outside reference-calibrated range")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(f"malformed generated spec: {error}")
    return errors


def _native_route_evidence(level, actions, context_index):
    levels = [level.clone() for _ in range(context_index + 1)]
    env = Env(levels)
    env.set_level(context_index)
    before_score = env.levels_completed
    frames = []
    clicks = lateral = extensions = retractions = pad_moves = 0
    blocker_pins = auxiliary_interactions = crossing_interactions = 0
    executed_actions = []
    max_history = len(getattr(env.game, names.ATTR_HISTORY))
    previous_model = extract_model(env)
    previous = previous_model.start
    for action_id, x, y in actions:
        executed_actions.append((action_id, x, y))
        selected = previous.selected
        rotation = previous_model.rotations[selected]
        if action_id == names.ACTION_CLICK:
            clicks += 1
        elif action_id in names.MOVE_ACTIONS:
            dx, dy = names.ACTION_DELTAS[action_id]
            base = names.DIRECTION[rotation]
            if (dx, dy) == base:
                extensions += 1
            elif (dx, dy) == (-base[0], -base[1]):
                retractions += 1
            else:
                lateral += 1
            step_x, step_y = dx * names.CELL, dy * names.CELL
            selected_cells = {(x, y) for x, y, _ in previous.lines[selected]}
            for pad_x, pad_y, _ in previous.pads:
                touches_selected = (
                    (pad_x, pad_y) in selected_cells
                    or (pad_x - step_x, pad_y - step_y) in selected_cells
                )
                if touches_selected and any(
                    left <= pad_x + step_x < left + width
                    and top <= pad_y + step_y < top + height
                    for left, top, width, height in previous_model.blockers
                ):
                    blocker_pins += 1
        orientations_by_cell = {}
        for line in previous.lines:
            for segment_x, segment_y, segment_rotation in line:
                orientations_by_cell.setdefault((segment_x, segment_y), set()).add(
                    segment_rotation % 180
                )
        crossing_interactions += sum(
            len(orientations_by_cell.get((pad_x, pad_y), ())) > 1
            for pad_x, pad_y, _ in previous.pads
        )
        observation = env.perform(action_id, x, y)
        frames.append(len(observation.frames))
        if env.levels_completed > before_score or observation.state == GameState.WIN:
            break
        current_model = extract_model(env)
        current = current_model.start
        auxiliary_cells = {
            (x, y)
            for index, line in enumerate(previous.lines)
            if not previous_model.clickable[index]
            for x, y, _ in line
        }
        for left, right in zip(previous.pads, current.pads):
            if left[:2] == right[:2]:
                continue
            pad_moves += 1
            if left[:2] in auxiliary_cells or right[:2] in auxiliary_cells:
                auxiliary_interactions += 1
        previous_model, previous = current_model, current
        max_history = max(max_history, len(getattr(env.game, names.ATTR_HISTORY)))
    won = env.levels_completed > before_score or env.state == GameState.WIN
    return {
        "won": bool(won),
        "supplied_action_count": len(actions),
        "executed_action_count": len(executed_actions),
        "first_completion_action": len(executed_actions) if won else None,
        "context_index": context_index,
        "movement_actions": sum(
            action[0] in names.MOVE_ACTIONS for action in executed_actions
        ),
        "click_switches": clicks,
        "lateral_moves": lateral,
        "extensions": extensions,
        "retractions": retractions,
        "pad_displacements": pad_moves,
        "blocker_pin_events": blocker_pins,
        "auxiliary_interactions": auxiliary_interactions,
        "crossing_interactions": crossing_interactions,
        "pause_actions": sum(count > 2 for count in frames[:-1]),
        "win_animation_frames": frames[-1] if frames else 0,
        "minimum_moves_left": names.MOVE_BUDGET
        - sum(action[0] in names.MOVE_ACTIONS for action in executed_actions),
        "history_depth": max_history,
    }


def _mechanic_errors(spec, evidence):
    difficulty = spec["difficulty"]
    errors = []
    if not evidence["won"]:
        errors.append("native contextual replay did not win")
    if (evidence["first_completion_action"]
            != evidence["supplied_action_count"]):
        errors.append("winning route has actions after first native completion")
    if evidence["minimum_moves_left"] < 8:
        errors.append("verified route has less than eight native moves of slack")
    if evidence["extensions"] < 1 or evidence["lateral_moves"] < 1:
        errors.append("route did not exercise extension and rail movement")
    if evidence["pause_actions"] < 1:
        errors.append("route did not exercise native pause/intersection animation")
    if difficulty >= 6 and evidence["click_switches"] < 1:
        errors.append("late-tier route did not switch clickable pairs")
    if difficulty in (3, 4) and evidence["auxiliary_interactions"] < 1:
        errors.append("route did not interact with a non-clickable auxiliary chain")
    if (difficulty in (3, 4)
            and evidence.get("counterfactual_auxiliary_shortened_wins") is not False):
        errors.append("auxiliary-chain counterfactual did not break the route")
    if difficulty in (5, 6) and evidence["blocker_pin_events"] < 1:
        errors.append("route did not use a fixed blocker to pin a colour pad")
    if (difficulty in (5, 6)
            and evidence.get("counterfactual_blocker_removed_wins") is not False):
        errors.append("blocker-removal counterfactual did not break the route")
    if difficulty >= 7 and evidence["crossing_interactions"] < 1:
        errors.append("late-tier route did not exercise a mixed-orientation crossing")
    if (difficulty >= 7
            and evidence.get("counterfactual_shared_pad_removed_wins") is not False):
        errors.append("shared-crossing counterfactual did not break the route")
    return errors


def _counterfactual_evidence(spec, actions):
    """Replay the same route after removing the claimed causal mechanism."""
    result = {
        "counterfactual_auxiliary_shortened_wins": None,
        "counterfactual_blocker_removed_wins": None,
        "counterfactual_shared_pad_removed_wins": None,
    }
    difficulty = spec["difficulty"]
    if difficulty in (3, 4):
        changed = deepcopy(spec)
        for head in changed["heads"]:
            if (head["prototype"] not in names.CLICKABLE_HEADS
                    and head["position"][1] < names.HUD_ROW):
                head["length"] = 1
        result["counterfactual_auxiliary_shortened_wins"] = replay(
            Env([build_level(changed)]), actions,
        )
    if difficulty in (5, 6):
        changed = deepcopy(spec)
        changed["blockers"] = []
        result["counterfactual_blocker_removed_wins"] = replay(
            Env([build_level(changed)]), actions,
        )
    if difficulty >= 7:
        changed = deepcopy(spec)
        top_pads = [pad for pad in changed["pads"]
                    if pad["position"][1] < names.HUD_ROW]
        rows = Counter(pad["position"][1] for pad in top_pads)
        columns = Counter(pad["position"][0] for pad in top_pads)
        crossing = next(
            pad for pad in top_pads
            if rows[pad["position"][1]] > 1 and columns[pad["position"][0]] > 1
        )
        changed["pads"].remove(crossing)
        result["counterfactual_shared_pad_removed_wins"] = replay(
            Env([build_level(changed)]), actions,
        )
    return result


def _accepted_spec(spec, split, node_limit):
    schema_errors = _component_schema_errors(spec)
    if schema_errors:
        return None, "schema:" + schema_errors[0]
    geometry = geometry_identity(spec)
    if official_layout_identity(spec) in _official_geometry_identities():
        return None, "official_geometry_copy"
    if split_for_identity(geometry) != split:
        return None, "geometry_split_mismatch"
    errors = profile_errors(spec, require_route=False)
    if errors:
        return None, "profile:" + errors[0]
    level = build_level(spec)
    model = extract_model(Env([level]))
    if model_solved(model, model.start):
        return None, "already_solved"
    guided_model = model
    if spec["difficulty"] in (6, 7):
        # Full late-tier rail fields match the originals but create many
        # irrelevant detours.  A deterministic constructive guide keeps only
        # one legal lateral edge per editable head.  Its route is then replayed
        # against the complete native rail field below.
        top_pads = [pad for pad in spec["pads"] if pad["position"][1] < names.HUD_ROW]
        row_counts = Counter(pad["position"][1] for pad in top_pads)
        column_counts = Counter(pad["position"][0] for pad in top_pads)
        horizontal_goal = row_counts.most_common(1)[0][0]
        vertical_goal = column_counts.most_common(1)[0][0]
        hx, hy = model.start.heads[0]
        vx, vy = model.start.heads[2]
        wanted = {
            (hx + 2, y + 2)
            for y in range(min(hy, horizontal_goal), max(hy, horizontal_goal), names.CELL)
        }
        wanted.update(
            (x + 2, vy + 2)
            for x in range(min(vx, vertical_goal), max(vx, vertical_goal), names.CELL)
        )
        selected_rails = [rect for rect in model.rails if rect[:2] in wanted]
        if len(selected_rails) == len(wanted):
            guided_model = replace(model, rails=tuple(selected_rails))
    work_limit = min(node_limit, 100_000) if spec["difficulty"] in (3, 5) else node_limit
    actions, expanded, truncated = model_search(
        guided_model, node_limit=work_limit, action_limit=names.MOVE_BUDGET
    )
    if actions is None:
        return None, "search_truncated" if truncated else "no_positive_witness"
    route_floor = PROFILES[spec["difficulty"]]["route_actions"][0]
    shortcut_limit = min(node_limit, 50_000)
    shortcut_actions, shortcut_expanded, shortcut_truncated = model_search(
        model,
        node_limit=shortcut_limit,
        action_limit=route_floor - 1,
    )
    if shortcut_actions is not None:
        return None, "shortcut_below_route_floor"
    evidence = _native_route_evidence(level, actions, spec["training_context_index"])
    evidence.update(_counterfactual_evidence(spec, actions))
    mechanic_errors = _mechanic_errors(spec, evidence)
    if mechanic_errors:
        return None, "mechanic_use:" + mechanic_errors[0]
    candidate = dict(spec)
    candidate.update({
        "split": split,
        "geometry_identity": geometry,
        "geometry_sha256": exact_geometry_identity(spec),
        "geometry_d4_sha256": geometry,
        "gameplay_identity": gameplay_identity(spec),
        "gameplay_sha256": gameplay_identity(spec),
        "official_copy": False,
        "solution": [list(action) for action in actions],
        "solution_length": len(actions),
        "solution_mechanics": evidence,
        "engine_verified": True,
        "verification_level_index": spec["training_context_index"],
        "proof": {
            "kind": "exact-stable-model-plus-contextual-native-replay",
            "shortest_route_claimed": False,
            "search_truncated": False,
            "expanded": expanded,
            "search_work": work_limit,
            "action_count": len(actions),
            "shortcut_action_limit": route_floor - 1,
            "shortcut_search_work": shortcut_limit,
            "shortcut_expanded": shortcut_expanded,
            "shortcut_search_truncated": shortcut_truncated,
            "shortcut_node_cap_reached": shortcut_expanded >= shortcut_limit,
        },
    })
    errors = profile_errors(candidate)
    if errors:
        return None, "profile:" + errors[0]
    return candidate, None


def generate(seed, difficulty=1, *, split, attempts=DEFAULT_ATTEMPTS,
             node_limit=DEFAULT_NODE_LIMIT):
    """Generate one verified tier in an explicit canonical split."""
    seed = _integer(seed, "seed")
    difficulty = _integer(difficulty, "difficulty")
    attempts = _integer(attempts, "attempts")
    node_limit = _integer(node_limit, "node_limit")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if attempts < 1 or node_limit < 1 or node_limit > 32_000_000:
        raise ValueError("attempts/search work outside bounded contract")
    rejections = Counter()
    for attempt in range(attempts):
        try:
            spec = _draft(seed, difficulty, attempt)
            accepted, reason = _accepted_spec(spec, split, node_limit)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            accepted, reason = None, f"invalid_geometry:{type(error).__name__}"
        if accepted is not None:
            accepted["rejections_before_acceptance"] = dict(sorted(rejections.items()))
            accepted["generation_attempts"] = attempt + 1
            generate.last_rejections = dict(sorted(rejections.items()))
            return accepted
        rejections[reason] += 1
    generate.last_rejections = dict(sorted(rejections.items()))
    return None


generate.last_rejections = {}


def _child_seed(game_seed, ordinal, difficulty):
    material = f"{SOURCE_ID}:{game_seed}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")


def generate_game(seed, *, split, difficulties=None, attempts=DEFAULT_ATTEMPTS,
                  node_limit=DEFAULT_NODE_LIMIT):
    """Generate the ordered eight-tier game, or an explicit ergonomic subset."""
    seed = _integer(seed, "seed")
    generate_game.last_failure = None
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not selected or any(type(value) is not int or value not in DIFFICULTIES
                           for value in selected):
        raise ValueError("difficulties must be a nonempty sequence drawn from 1..8")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("difficulties must be unique and increasing")
    specs = []
    for ordinal, difficulty in enumerate(selected):
        child_seed = _child_seed(seed, ordinal, difficulty)
        spec = generate(
            child_seed, difficulty,
            split=split, attempts=attempts, node_limit=node_limit,
        )
        if spec is None:
            generate_game.last_failure = {
                "difficulty": difficulty,
                "context_index": difficulty - 1,
                "ordinal": ordinal,
                "child_seed": child_seed,
                "split": split,
                "attempts": attempts,
                "node_limit": node_limit,
                "rejections": dict(generate.last_rejections),
            }
            return None
        spec["game_seed"] = seed
        spec["game_ordinal"] = ordinal
        specs.append(spec)
    return specs


generate_game.last_failure = None


def build_game(specs):
    """Validate, build, and independently replay one exact-eight curriculum."""
    if specs is None or isinstance(specs, (str, bytes, bytearray, Mapping)):
        raise ValueError("full SK48 game specs must be a sequence of mappings")
    try:
        specs = list(specs)
    except TypeError as error:
        raise ValueError(
            "full SK48 game specs must be a sequence of mappings"
        ) from error
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("full SK48 game must contain exactly eight levels")
    if not all(isinstance(spec, Mapping) for spec in specs):
        raise ValueError("every full-game spec must be a mapping")
    split = specs[0].get("split")
    if type(split) is not str or split not in SPLITS:
        raise ValueError("full game split must be a canonical string")
    if any(type(spec.get("split")) is not str or spec.get("split") != split
           for spec in specs):
        raise ValueError("full game specs must use one split")
    identities = set()
    for index, spec in enumerate(specs):
        if spec.get("difficulty") != index + 1:
            raise ValueError("full game difficulties must be 1..8 in order")
        if spec.get("training_context_index") != index:
            raise ValueError("full game context indices must be 0..7 in order")
        identity = spec.get("gameplay_sha256")
        if (type(identity) is not str or len(identity) != 64
                or identity in identities):
            raise ValueError("full game gameplay identities must be nonempty and unique")
        identities.add(identity)
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        if errors:
            raise ValueError("invalid full-standard spec: " + "; ".join(errors))
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            raise ValueError("native full-game context did not advance sequentially")
        actions = spec["solution"]
        completed_at = None
        for action_index, (action_id, x, y) in enumerate(actions, 1):
            observation = env.perform(action_id, x, y)
            if env.levels_completed > index or observation.state == GameState.WIN:
                completed_at = action_index
                break
            if observation.state == GameState.GAME_OVER:
                break
        if completed_at is None:
            raise ValueError(f"tier {index + 1} did not complete in native full game")
        if completed_at != len(actions):
            raise ValueError(f"tier {index + 1} route has a post-win suffix")
        if env.levels_completed != index + 1:
            raise ValueError("native full-game completion count is inconsistent")
    if env.state != GameState.WIN:
        raise ValueError("native full game did not reach final WIN")
    return levels


def validate_full_standard(spec, curriculum_entry):
    """Recompute identity, profile, route evidence, and contextual native win."""
    if not isinstance(spec, Mapping):
        return ("full-standard spec must be a mapping",)
    if not isinstance(curriculum_entry, Mapping):
        return ("curriculum entry must be a mapping",)
    errors = []
    try:
        schema_errors = _component_schema_errors(spec)
        if schema_errors:
            return schema_errors
        for key in ("difficulty", "context_index", "search_work"):
            if type(curriculum_entry.get(key)) is not int:
                errors.append(f"curriculum {key} must be an exact integer")
        difficulty = spec.get("difficulty")
        if type(difficulty) is not int or difficulty not in DIFFICULTIES:
            errors.append("difficulty must be an exact integer in 1..8")
        if curriculum_entry.get("difficulty") != difficulty:
            errors.append("difficulty does not match curriculum entry")
        expected_context = curriculum_entry.get("context_index")
        for key in (
            "seed", "effective_seed", "attempt", "context_index",
            "training_context_index", "verification_level_index",
            "native_move_budget", "solution_length", "generation_attempts",
        ):
            if type(spec.get(key)) is not int:
                errors.append(f"{key} must be an exact integer")
        if spec.get("effective_seed") != spec.get("seed"):
            errors.append("effective seed does not match level seed")
        for key in ("context_index", "training_context_index",
                    "verification_level_index"):
            if spec.get(key) != expected_context:
                errors.append(f"{key} does not match curriculum entry")
        if spec.get("attempt", -1) < 0 or spec.get("generation_attempts", 0) < 1:
            errors.append("generation attempt metadata is out of range")
        if spec.get("native_move_budget") != names.MOVE_BUDGET:
            errors.append("native move budget metadata mismatch")
        if spec.get("engine_verified") is not True:
            errors.append("engine verification flag must be exact true")
        if spec.get("official_copy") is not False:
            errors.append("official-copy gate must be exact false")
        if "game_seed" in spec and type(spec.get("game_seed")) is not int:
            errors.append("game_seed must be an exact integer")
        if "game_ordinal" in spec and type(spec.get("game_ordinal")) is not int:
            errors.append("game_ordinal must be an exact integer")
        for key, expected in (
            ("format", FORMAT), ("generator_version", GENERATOR_VERSION),
            ("mechanics_version", MECHANICS_VERSION),
            ("difficulty_version", DIFFICULTY_VERSION),
            ("quality_version", QUALITY_VERSION),
            ("geometry_version", GEOMETRY_VERSION),
            ("source", SOURCE_KIND), ("source_id", SOURCE_ID),
        ):
            if type(spec.get(key)) is not type(expected) or spec.get(key) != expected:
                errors.append(f"{key} mismatch")
        rejections = spec.get("rejections_before_acceptance")
        if (type(rejections) is not dict
                or any(type(key) is not str or type(value) is not int or value < 0
                       for key, value in rejections.items())):
            errors.append("rejection counters must be string/exact-integer mapping")
        errors.extend(profile_errors(spec))
        geometry = geometry_identity(spec)
        exact_geometry = exact_geometry_identity(spec)
        gameplay = gameplay_identity(spec)
        if spec.get("geometry_identity") != geometry:
            errors.append("geometry identity mismatch")
        if spec.get("geometry_sha256") != exact_geometry:
            errors.append("exact geometry SHA-256 mismatch")
        if spec.get("geometry_d4_sha256") != geometry:
            errors.append("D4 geometry SHA-256 mismatch")
        if spec.get("gameplay_identity") != gameplay:
            errors.append("gameplay identity mismatch")
        if spec.get("gameplay_sha256") != gameplay:
            errors.append("gameplay SHA-256 mismatch")
        for key in (
            "geometry_identity", "geometry_sha256", "geometry_d4_sha256",
            "gameplay_identity", "gameplay_sha256",
        ):
            value = spec.get(key)
            if (type(value) is not str or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                errors.append(f"{key} must be a lowercase SHA-256 digest")
        if official_layout_identity(spec) in _official_geometry_identities():
            errors.append("generated geometry duplicates an official level")
        split = spec.get("split")
        if (type(split) is not str or split not in SPLITS
                or split_for_identity(geometry) != split):
            errors.append("canonical split mismatch")
        solution = spec.get("solution")
        route_valid = type(solution) is list and bool(solution)
        if route_valid:
            for action in solution:
                if type(action) is not list or len(action) != 3:
                    route_valid = False
                    break
                action_id, x, y = action
                if type(action_id) is not int or action_id not in names.ACTION_IDS:
                    route_valid = False
                    break
                if action_id == names.ACTION_CLICK:
                    if (type(x) is not int or type(y) is not int
                            or not 0 <= x < names.FRAME_SIZE
                            or not 0 <= y < names.FRAME_SIZE):
                        route_valid = False
                        break
                elif x is not None or y is not None:
                    route_valid = False
                    break
        if not route_valid:
            errors.append("solution must use exact JSON display-action triples")
            actions = ()
        else:
            actions = tuple(tuple(action) for action in solution)
        if len(actions) != spec.get("solution_length"):
            errors.append("solution length metadata mismatch")
        if route_valid and type(expected_context) is int:
            level = build_level(spec)
            evidence = _native_route_evidence(level, actions, expected_context)
            evidence.update(_counterfactual_evidence(spec, actions))
            errors.extend(_mechanic_errors(spec, evidence))
            stored_evidence = spec.get("solution_mechanics")
            if not isinstance(stored_evidence, Mapping):
                errors.append("solution mechanic evidence must be a mapping")
            else:
                for key, expected in evidence.items():
                    observed = stored_evidence.get(key)
                    if type(observed) is not type(expected) or observed != expected:
                        errors.append("solution mechanic evidence mismatch")
                        break
        proof = spec.get("proof")
        if not isinstance(proof, Mapping):
            errors.append("proof must be a mapping")
            proof = {}
        if (type(proof.get("kind")) is not str
                or proof.get("kind")
                != "exact-stable-model-plus-contextual-native-replay"
                or proof.get("search_truncated") is not False
                or proof.get("shortest_route_claimed") is not False):
            errors.append("proof metadata mismatch")
        for key in ("shortcut_search_truncated", "shortcut_node_cap_reached"):
            if type(proof.get(key)) is not bool:
                errors.append(f"proof {key} must be an exact boolean")
        for key in (
            "expanded", "search_work", "action_count", "shortcut_action_limit",
            "shortcut_search_work", "shortcut_expanded",
        ):
            if type(proof.get(key)) is not int or proof.get(key, -1) < 0:
                errors.append(f"proof {key} must be a nonnegative exact integer")
        if (proof.get("action_count") != len(actions)
                or proof.get("expanded", 0) > proof.get("search_work", -1)
                or proof.get("search_work", 0) < 1
                or proof.get("shortcut_action_limit")
                != PROFILES.get(difficulty, {}).get("route_actions", (1,))[0] - 1
                or proof.get("shortcut_search_work", 0) < 1
                or proof.get("shortcut_search_work", 0) > 50_000
                or proof.get("shortcut_expanded", 0)
                > proof.get("shortcut_search_work", 0)
                or (proof.get("shortcut_node_cap_reached") is True
                    and proof.get("shortcut_expanded")
                    != proof.get("shortcut_search_work"))
                or (proof.get("shortcut_node_cap_reached") is False
                    and proof.get("shortcut_expanded", 0)
                    >= proof.get("shortcut_search_work", 0))):
            errors.append("proof counter metadata mismatch")
    except (IndexError, KeyError, StopIteration, TypeError, ValueError) as error:
        errors.append(f"malformed full-standard spec: {error}")
    return tuple(dict.fromkeys(errors))
