"""Reference-calibrated contracts for all six shipped FT09 tiers.

There is one official level per tier, so the ranges below are explicit design
tolerances around scarce examples, not population confidence intervals.  The
source geometry and level data are in ``third_party/arc3_games/ft09.py``
lines 2038-2267; click, stencil, constraint, budget, and tutorial-flash
semantics are implemented at lines 2272-2520.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from . import names

DIFFICULTY_VERSION = "ft09-reference-v1"
MECHANICS_INVENTORY_VERSION = "ft09-official-mechanics-v1"
DIFFICULTIES = tuple(range(1, 7))

# Measured by exact, untruncated symbolic search and replay in the corresponding
# native level context.  These are facts about the six shipped levels, not
# targets copied into generated layouts.
REFERENCES = {
    1: dict(name="THR", cells=8, constraints=1, palette_size=2, ordinary=8, special=0,
            budget=32, actions=4, active_rule_edges=8, match_edges=4, differ_edges=4,
            bbox_slots=(3, 3), slot_density=1.0),
    2: dict(name="hxv", cells=13, constraints=2, palette_size=2, ordinary=13, special=0,
            budget=32, actions=7, active_rule_edges=16, match_edges=9, differ_edges=7,
            bbox_slots=(3, 5), slot_density=1.0),
    3: dict(name="Fmh", cells=23, constraints=4, palette_size=2, ordinary=23, special=0,
            budget=96, actions=14, active_rule_edges=32, match_edges=18, differ_edges=14,
            bbox_slots=(5, 7), slot_density=27 / 35),
    4: dict(name="oea", cells=18, constraints=3, palette_size=3, ordinary=18, special=0,
            budget=96, actions=16, active_rule_edges=24, match_edges=8, differ_edges=16,
            bbox_slots=(5, 5), slot_density=21 / 25),
    5: dict(name="INW", cells=30, constraints=8, palette_size=2, ordinary=27, special=3,
            budget=128, actions=21, active_rule_edges=43, match_edges=23, differ_edges=20,
            bbox_slots=(7, 7), slot_density=38 / 49),
    6: dict(name="DFx", cells=22, constraints=4, palette_size=2, ordinary=0, special=22,
            budget=128, actions=13, active_rule_edges=24, match_edges=14, differ_edges=10,
            bbox_slots=(7, 6), slot_density=26 / 42),
}

# The official action counts are 4, 7, 14, 16, 21, 13.  Bounds preserve those
# distinct lesson scales without pretending six samples estimate a population.
PROFILES = {
    1: dict(cells=(7, 9), constraints=(1, 1), palette_size=2, budget=32,
            actions=(3, 6), rule_edges=(7, 9), bbox_short=(3, 4), bbox_long=(3, 4),
            density=(0.55, 1.0), match_edges=(2, 6), differ_edges=(2, 6),
            special=(0, 0), context_index=0, search_work=100_000),
    2: dict(cells=(11, 15), constraints=(2, 2), palette_size=2, budget=32,
            actions=(5, 9), rule_edges=(14, 18), bbox_short=(3, 5), bbox_long=(4, 6),
            density=(0.55, 1.0), match_edges=(5, 12), differ_edges=(4, 11),
            special=(0, 0), context_index=1, search_work=250_000),
    3: dict(cells=(20, 26), constraints=(4, 4), palette_size=2, budget=96,
            actions=(11, 17), rule_edges=(28, 36), bbox_short=(4, 6), bbox_long=(6, 7),
            density=(0.55, 0.95), match_edges=(10, 24), differ_edges=(8, 22),
            special=(0, 0), context_index=2, search_work=500_000),
    4: dict(cells=(16, 21), constraints=(3, 3), palette_size=3, budget=96,
            actions=(13, 20), rule_edges=(21, 27), bbox_short=(4, 6), bbox_long=(4, 6),
            density=(0.55, 0.95), match_edges=(5, 12), differ_edges=(12, 20),
            special=(0, 0), context_index=3, search_work=750_000),
    5: dict(cells=(27, 33), constraints=(7, 9), palette_size=2, budget=128,
            actions=(17, 26), rule_edges=(38, 48), bbox_short=(6, 7), bbox_long=(6, 7),
            density=(0.62, 0.90), match_edges=(14, 30), differ_edges=(13, 28),
            special=(2, 4), context_index=4, search_work=1_000_000),
    6: dict(cells=(20, 25), constraints=(3, 5), palette_size=2, budget=128,
            actions=(10, 17), rule_edges=(21, 28), bbox_short=(5, 7), bbox_long=(6, 7),
            density=(0.50, 0.82), match_edges=(7, 20), differ_edges=(6, 17),
            special=(20, 25), context_index=5, search_work=1_000_000),
}


def _between(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def structural_metrics(spec: Mapping) -> dict:
    """Return role, geometry, visual-density, and rule-composition metrics."""
    cells = list(spec.get("cells", ()))
    constraints = list(spec.get("constraints", ()))
    cell_at = {(int(cell["x"]), int(cell["y"])): cell for cell in cells}
    ordinary = sum("stencil" not in cell for cell in cells)
    special = len(cells) - ordinary
    active = match = differ = 0
    for rule in constraints:
        x, y = int(rule["x"]), int(rule["y"])
        mask = rule["mask"]
        for (row, col), (dx, dy) in names.NEIGHBOUR_OFFSETS.items():
            if (x + dx, y + dy) in cell_at:
                active += 1
                if int(mask[row][col]) == names.MATCH_FLAG:
                    match += 1
                else:
                    differ += 1

    anchors = list(cell_at) + [(int(r["x"]), int(r["y"])) for r in constraints]
    if anchors:
        width = (max(x for x, _ in anchors) - min(x for x, _ in anchors)) // names.PITCH + 1
        height = (max(y for _, y in anchors) - min(y for _, y in anchors)) // names.PITCH + 1
    else:
        width = height = 0
    short, long = sorted((width, height))
    area = width * height
    coupled_special = 0
    special_patterns = set()
    for cell in cells:
        if "stencil" not in cell:
            continue
        stencil = tuple(tuple(int(v) for v in row) for row in cell["stencil"])
        special_patterns.add(stencil)
        x, y = int(cell["x"]), int(cell["y"])
        if any(
            int(stencil[row][col])
            and (row, col) != (1, 1)
            and (x + dx, y + dy) in cell_at
            for (row, col), (dx, dy) in names.STENCIL_OFFSETS.items()
        ):
            coupled_special += 1
    return {
        "cell_count": len(cells),
        "ordinary_cell_count": ordinary,
        "special_cell_count": special,
        "constraint_count": len(constraints),
        "active_rule_edges": active,
        "match_rule_edges": match,
        "differ_rule_edges": differ,
        "bbox_slots": [width, height],
        "bbox_short_slots": short,
        "bbox_long_slots": long,
        "slot_density": (len(cells) + len(constraints)) / area if area else 0.0,
        "object_pixel_density": 9 * (len(cells) + len(constraints)) / (names.GRID ** 2),
        "coupled_special_cells": coupled_special,
        "special_stencil_patterns": len(special_patterns),
    }


def profile_errors(spec: Mapping, *, require_proof: bool = True) -> list[str]:
    """Explain every reference-profile or full-mechanics contract violation."""
    errors: list[str] = []
    difficulty = spec.get("difficulty")
    if type(difficulty) is not int or difficulty not in PROFILES:
        return [f"difficulty must be one of {DIFFICULTIES}"]
    profile = PROFILES[difficulty]
    metrics = structural_metrics(spec)
    palette = spec.get("palette", ())
    if spec.get("grid") != names.GRID:
        errors.append("grid must be the native 32x32")
    if len(palette) != profile["palette_size"]:
        errors.append("palette size differs from the official tier")
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in palette)
            or len(set(palette)) != len(palette) or {names.STENCIL_MARKER, 4} & set(palette)):
        errors.append("palette must contain distinct native colours excluding background and stencil marker")
    if spec.get("budget") != profile["budget"]:
        errors.append("budget differs from the native official tier")
    stencil = tuple(tuple(int(v) for v in row) for row in spec.get("stencil", ()))
    if stencil != names.IDENTITY_STENCIL:
        errors.append("official FT09 uses the centre-only global stencil in every tier")
    anchors = [(cell["x"], cell["y"]) for cell in spec.get("cells", ())]
    anchors += [(rule["x"], rule["y"]) for rule in spec.get("constraints", ())]
    if (any(isinstance(value, bool) or not isinstance(value, int)
            for anchor in anchors for value in anchor)
            or any(not 0 <= value <= names.GRID - names.CELL_SIZE
                   for anchor in anchors for value in anchor)
            or len(set(anchors)) != len(anchors)
            or len({x % names.PITCH for x, _ in anchors}) != 1
            or len({y % names.PITCH for _, y in anchors}) != 1):
        errors.append("gameplay objects must occupy unique in-bounds slots on one native pitch lattice")
    if any(rule.get("colour") not in palette for rule in spec.get("constraints", ())):
        errors.append("every rule centre must use a colour from the ordered palette")
    if any(len(cell.get("stencil", ())) != 3
           or any(len(row) != 3 or any(value not in (0, 1) for value in row)
                  for row in cell.get("stencil", ()))
           for cell in spec.get("cells", ()) if "stencil" in cell):
        errors.append("every special-cell stencil must be a binary 3x3 mask")
    if any(len(rule.get("mask", ())) != 3
           or any(len(row) != 3 for row in rule.get("mask", ()))
           for rule in spec.get("constraints", ())):
        errors.append("every rule mask must be 3x3")
    checks = (
        ("cell_count", profile["cells"], "cell count"),
        ("constraint_count", profile["constraints"], "constraint count"),
        ("active_rule_edges", profile["rule_edges"], "active rule-edge count"),
        ("match_rule_edges", profile["match_edges"], "match rule-edge count"),
        ("differ_rule_edges", profile["differ_edges"], "differ rule-edge count"),
        ("bbox_short_slots", profile["bbox_short"], "short board span"),
        ("bbox_long_slots", profile["bbox_long"], "long board span"),
        ("slot_density", profile["density"], "gameplay-object density"),
        ("special_cell_count", profile["special"], "special-cell count"),
    )
    for field, bounds, label in checks:
        if not _between(metrics[field], bounds):
            errors.append(f"{label} {metrics[field]} outside explicit tolerance {bounds}")

    mechanics = spec.get("solution_mechanics", {})
    actions = spec.get("optimal_actions", spec.get("solution_length"))
    if actions is not None and not _between(actions, profile["actions"]):
        errors.append(f"verified actions {actions} outside explicit tolerance {profile['actions']}")
    if difficulty == 1 and not spec.get("tutorial_hint"):
        errors.append("tier 1 must exercise the native tutorial hint/flash")
    if difficulty > 1 and spec.get("tutorial_hint"):
        errors.append("tutorial hint is native only to context index 0")
    if difficulty == 4:
        if mechanics.get("third_colour_actions", 0) < 1:
            errors.append("tier 4 solution does not exercise the third palette colour")
        if not _between(mechanics.get("third_colour_cells_final", 0), (3, 7)):
            errors.append("tier 4 third-colour cell use differs too far from the reference count 5")
        if not _between(mechanics.get("repeat_actions", 0), (3, 7)):
            errors.append("tier 4 solution does not exercise repeated cell clicks")
    if difficulty == 5:
        if not (metrics["ordinary_cell_count"] and metrics["special_cell_count"]):
            errors.append("tier 5 must mix ordinary and special cells")
        cross = ((0, 1, 0), (1, 1, 1), (0, 1, 0))
        stencils = [tuple(tuple(int(v) for v in row) for row in cell["stencil"])
                    for cell in spec.get("cells", ()) if "stencil" in cell]
        if any(stencil != cross for stencil in stencils):
            errors.append("tier 5 special cells must use the official cross-stencil signature")
        if mechanics.get("distinct_special_cells_clicked", 0) != metrics["special_cell_count"]:
            errors.append("tier 5 solution does not use every installed cross-stencil cell")
    if difficulty == 6:
        if metrics["ordinary_cell_count"] or metrics["special_cell_count"] != metrics["cell_count"]:
            errors.append("tier 6 must be an all-special board")
        stencils = [tuple(tuple(int(v) for v in row) for row in cell["stencil"])
                    for cell in spec.get("cells", ()) if "stencil" in cell]
        cardinal = {(0, 1), (1, 0), (1, 2), (2, 1)}
        if any(stencil[1][1] != 1 or sum(map(sum, stencil)) != 2
               or not ({(r, c) for r in range(3) for c in range(3) if stencil[r][c]}
                       - {(1, 1)}) <= cardinal
               for stencil in stencils):
            errors.append("tier 6 special cells must use a centre-plus-cardinal directional stencil")
        if len(set(stencils)) != 1:
            errors.append("tier 6 must use one coherent directional stencil field")
        if metrics["coupled_special_cells"] < 3:
            errors.append("tier 6 lacks enough live directional stencil interactions")
        if mechanics.get("distinct_coupled_special_cells_clicked", 0) < 5:
            errors.append("tier 6 solution does not use enough live directional stencils")

    if require_proof:
        proof = spec.get("proof", {})
        if not spec.get("engine_verified") or not spec.get("context_engine_verified"):
            errors.append("missing real-engine positive certificate")
        if spec.get("context_index") != profile["context_index"]:
            errors.append("wrong native level context")
        if spec.get("search_truncated") is not False or proof.get("search_truncated") is not False:
            errors.append("search cutoff cannot certify the level")
        solution = spec.get("solution")
        if not isinstance(solution, Sequence) or not solution:
            errors.append("missing nonempty verified teacher solution")
        elif actions != len(solution):
            errors.append("solution length and measured optimal actions disagree")
        for field in ("geometry_sha256", "gameplay_sha256"):
            value = spec.get(field)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"missing canonical {field}")
    return errors
