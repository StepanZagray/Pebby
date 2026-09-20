"""Reference-calibrated quality measurements for full TU93 generation.

The shipped game contains one level at each of nine tiers.  Consequently the
tolerances below are explicit engineering bands around scarce references, not
statistical confidence intervals.  Every accepted generated spec is measured
again from its extracted action-boundary layout and winning witness.
"""

from __future__ import annotations

import hashlib
import json

from . import names
from .layout import Layout
from .plan import transition_with_events


MECHANICS_INVENTORY_VERSION = "tu93-complete-mechanics-v1"
QUALITY_PROFILE_VERSION = "tu93-nine-official-reference-v1"

# Values were measured from the nine vendored level definitions with the exact
# symbolic BFS. ``density`` is the fraction of the maze bitmap occupied by a
# 3x3 node or passage block; transparent lattice positions are excluded.
_REFERENCES = (
    # shape, active nodes, edges, cycles, dead ends, junctions, density,
    # H/P/T, native budget, shortest action count
    ((6, 6), 31, 32, 2, 4, 5, 0.520661, (0, 0, 0), 50, 18),
    ((7, 3), 12, 13, 2, 2, 4, 0.384615, (1, 0, 0), 50, 10),
    ((7, 5), 23, 28, 6, 2, 10, 0.435897, (3, 0, 0), 35, 19),
    ((5, 5), 16, 18, 3, 2, 6, 0.419753, (1, 1, 0), 20, 17),
    ((8, 6), 31, 33, 3, 4, 6, 0.387879, (1, 4, 0), 50, 29),
    ((7, 7), 31, 36, 6, 6, 12, 0.396450, (6, 2, 0), 60, 28),
    ((7, 5), 22, 24, 3, 2, 6, 0.393162, (1, 0, 1), 30, 14),
    ((6, 6), 15, 15, 1, 3, 3, 0.247934, (1, 0, 1), 50, 21),
    ((8, 7), 33, 37, 5, 5, 10, 0.358974, (2, 3, 1), 50, 29),
)

_SOLUTION_BANDS = (
    (13, 25),
    (7, 16),
    (14, 27),
    (13, 19),
    (21, 40),
    (21, 42),
    (10, 22),
    (15, 31),
    (21, 42),
)

_REFERENCE_GRID = (39, 45, 45, 51, 49, 45, 45, 42, 51)
_REFERENCE_ORIGIN = (
    (3, 3), (3, 12), (3, 9), (15, 10), (2, 7),
    (3, 3), (3, 9), (3, 3), (3, 4),
)

# Route-level evidence requirements. Tier 5 intentionally has no hunter-use
# quota: its official shortest witness neither eats its hunter nor exposes a
# one-step fatal hunter alternative. The actor is still present and simulated.
_USE_REQUIREMENTS = (
    {},
    {"hunter_use": 1},
    {"eaten_hunters": 2, "hunter_use": 2},
    {"hunter_use": 1, "patroller_moves": 1, "patroller_bounces": 1,
     "patroller_use": 1},
    {"patroller_moves": 1, "patroller_bounces": 1, "patroller_use": 1},
    {"eaten_hunters": 3, "hunter_use": 3, "patroller_moves": 1,
     "patroller_bounces": 1, "patroller_use": 1},
    {"hunter_use": 1, "tail_arms": 1, "tail_moves": 1},
    {"hunter_use": 1, "tail_arms": 1, "tail_moves": 1},
    {"hunter_use": 1, "patroller_moves": 1, "patroller_bounces": 1,
     "patroller_use": 1, "tail_arms": 1, "tail_moves": 1, "eaten_actors": 1},
)


def _band(value, fraction=0.25, floor=2):
    delta = max(floor, round(value * fraction))
    return max(0, value - delta), value + delta


REFERENCE_PROFILES = {}
for _difficulty, _row in enumerate(_REFERENCES, 1):
    (_shape, _active, _edges, _cycles, _dead, _junctions, _density,
     _actors, _budget, _solution) = _row
    _short, _long = sorted(_shape)
    REFERENCE_PROFILES[_difficulty] = {
        "difficulty": _difficulty,
        "context_index": _difficulty - 1,
        "reference": {
            "shape": list(_shape),
            "grid_size": [_REFERENCE_GRID[_difficulty - 1]] * 2,
            "maze_origin": list(_REFERENCE_ORIGIN[_difficulty - 1]),
            "active_nodes": _active,
            "edges": _edges,
            "cycle_rank": _cycles,
            "dead_ends": _dead,
            "junctions": _junctions,
            "maze_density": _density,
            "hunters": _actors[0],
            "patrollers": _actors[1],
            "tails": _actors[2],
            "native_budget": _budget,
            "shortest_symbolic_actions": _solution,
        },
        "tolerance": {
            "shape_short": (max(2, _short - 1), min(9, _short + 1)),
            "shape_long": (max(2, _long - 1), min(9, _long + 1)),
            "grid_size": (
                _REFERENCE_GRID[_difficulty - 1],
                _REFERENCE_GRID[_difficulty - 1] + 8,
            ),
            "active_nodes": _band(_active, 0.25, 3),
            "cycle_rank": (max(0, _cycles - 2), _cycles + 2),
            "dead_ends": _band(_dead, 0.50, 2),
            "junctions": _band(_junctions, 0.40, 3),
            "maze_density": (max(0.10, _density - 0.12), min(0.80, _density + 0.12)),
            "solution_length": _SOLUTION_BANDS[_difficulty - 1],
        },
        "actors": {
            "hunters": _actors[0],
            "patrollers": _actors[1],
            "tails": _actors[2],
        },
        "native_budget": _budget,
        "required_use": dict(_USE_REQUIREMENTS[_difficulty - 1]),
        "calibration_basis": "one shipped reference level; explicit engineering tolerance",
    }


def _active_nodes(layout: Layout):
    nodes = set()
    for a, b in layout.edges:
        nodes.add(a)
        nodes.add(b)
    return nodes


def structural_metrics(layout: Layout):
    """Return JSON-compatible topology, visual-density, actor and budget data."""
    active = _active_nodes(layout)
    degree = {cell: 0 for cell in active}
    adjacency = {cell: [] for cell in active}
    for a, b in layout.edges:
        degree[a] += 1
        degree[b] += 1
        adjacency[a].append(b)
        adjacency[b].append(a)
    components = 0
    unseen = set(active)
    while unseen:
        components += 1
        queue = [unseen.pop()]
        while queue:
            cell = queue.pop()
            for nxt in adjacency[cell]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
    cols = max((cell[0] for cell in layout.nodes), default=-1) + 1
    rows = max((cell[1] for cell in layout.nodes), default=-1) + 1
    pixel_width = names.CELL * (cols - 1) + names.BLOCK
    pixel_height = names.CELL * (rows - 1) + names.BLOCK
    occupied_pixels = names.BLOCK ** 2 * (len(active) + len(layout.edges))
    density = occupied_pixels / (pixel_width * pixel_height) if pixel_width and pixel_height else 0.0
    return {
        "cols": cols,
        "rows": rows,
        "grid_width": int(layout.grid_size[0]),
        "grid_height": int(layout.grid_size[1]),
        "origin_x": int(layout.origin[0]),
        "origin_y": int(layout.origin[1]),
        "active_nodes": len(active),
        "edges": len(layout.edges),
        "components": components,
        "cycle_rank": len(layout.edges) - len(active) + components,
        "dead_ends": sum(value == 1 for value in degree.values()),
        "junctions": sum(value >= 3 for value in degree.values()),
        "maze_density": round(density, 6),
        "hunters": len(layout.hunters),
        "patrollers": len(layout.patrollers),
        "tails": len(layout.tails),
        "exits": len(layout.exits),
        "controls": len(names.ACTION_IDS),
        "native_budget": layout.max_steps,
        "remaining_budget": layout.steps_left,
    }


def witness_metrics(layout: Layout, actions):
    """Measure mechanic events along a winning route and its fatal alternatives."""
    totals = {
        "accepted_moves": 0,
        "eaten_hunters": 0,
        "eaten_patrollers": 0,
        "eaten_tails": 0,
        "eaten_actors": 0,
        "patroller_moves": 0,
        "patroller_bounces": 0,
        "tail_arms": 0,
        "tail_moves": 0,
        "fatal_hunter_alternatives": 0,
        "fatal_patroller_alternatives": 0,
        "fatal_tail_alternatives": 0,
        "hunter_use": 0,
        "patroller_use": 0,
        "won": False,
        "dead": False,
        "action_count": 0,
    }
    state = layout.key()
    for raw_step in actions:
        action = int(raw_step[0] if isinstance(raw_step, (list, tuple)) else raw_step)
        for alternative in names.ACTION_IDS:
            alternative_outcome, alternative_events = transition_with_events(
                layout, state, alternative
            )
            if alternative_outcome is not None and alternative_outcome[2]:
                totals["fatal_hunter_alternatives"] += alternative_events["fatal_hunter"]
                totals["fatal_patroller_alternatives"] += alternative_events["fatal_patroller"]
                totals["fatal_tail_alternatives"] += alternative_events["fatal_tail"]
        outcome, events = transition_with_events(layout, state, action)
        if outcome is None:
            totals["action_count"] += 1
            continue
        state, won, dead = outcome
        totals["action_count"] += 1
        for key in (
            "accepted_move", "eaten_hunters", "eaten_patrollers", "eaten_tails",
            "patroller_moves", "patroller_bounces", "tail_arms", "tail_moves",
        ):
            target = "accepted_moves" if key == "accepted_move" else key
            totals[target] += int(events[key])
        totals["won"] = bool(won)
        totals["dead"] = bool(dead)
        if won or dead:
            break
    totals["eaten_actors"] = (
        totals["eaten_hunters"] + totals["eaten_patrollers"] + totals["eaten_tails"]
    )
    totals["hunter_use"] = (
        totals["eaten_hunters"] + totals["fatal_hunter_alternatives"]
    )
    totals["patroller_use"] = (
        totals["eaten_patrollers"] + totals["fatal_patroller_alternatives"]
    )
    return totals


def measure(layout: Layout, actions):
    result = structural_metrics(layout)
    result["solution_length"] = len(actions)
    result["witness"] = witness_metrics(layout, actions)
    return result


def _outside(value, bounds):
    return value < bounds[0] or value > bounds[1]


def validate_profile(difficulty, metrics):
    """Return concrete deviations from the calibrated tier profile."""
    profile = REFERENCE_PROFILES[int(difficulty)]
    tolerance = profile["tolerance"]
    errors = []
    short, long = sorted((int(metrics["cols"]), int(metrics["rows"])))
    checks = (
        ("shape_short", short),
        ("shape_long", long),
        ("grid_width", int(metrics["grid_width"])),
        ("grid_height", int(metrics["grid_height"])),
        ("active_nodes", int(metrics["active_nodes"])),
        ("cycle_rank", int(metrics["cycle_rank"])),
        ("dead_ends", int(metrics["dead_ends"])),
        ("junctions", int(metrics["junctions"])),
        ("maze_density", float(metrics["maze_density"])),
        ("solution_length", int(metrics["solution_length"])),
    )
    for field, value in checks:
        bounds = tolerance["grid_size"] if field.startswith("grid_") else tolerance[field]
        if _outside(value, bounds):
            errors.append(f"{field}={value!r} outside {tuple(bounds)!r}")
    if int(metrics["components"]) != 1:
        errors.append(f"components={metrics['components']!r}, expected 1")
    for field, expected in profile["actors"].items():
        if int(metrics[field]) != expected:
            errors.append(f"{field}={metrics[field]!r}, expected {expected}")
    if int(metrics["exits"]) != 1 or int(metrics["controls"]) != 4:
        errors.append(
            f"object/control counts exits={metrics['exits']!r}, controls={metrics['controls']!r}; expected 1/4"
        )
    if int(metrics["native_budget"]) != profile["native_budget"]:
        errors.append(
            f"native_budget={metrics['native_budget']!r}, expected {profile['native_budget']}"
        )
    witness = metrics["witness"]
    if not witness.get("won") or witness.get("dead"):
        errors.append("stored witness does not reach a live symbolic win")
    if int(witness.get("action_count", -1)) != int(metrics["solution_length"]):
        errors.append("stored witness contains actions after its terminal transition")
    if int(witness.get("accepted_moves", -1)) != int(metrics["solution_length"]):
        errors.append("stored witness contains blocked/padding actions")
    for event, minimum in profile["required_use"].items():
        if int(witness.get(event, 0)) < minimum:
            errors.append(f"witness {event}={witness.get(event, 0)!r}, expected >= {minimum}")
    return errors


def _transforms(cols, rows):
    # Each entry maps a point and gives the transformed rectangle dimensions.
    return (
        (lambda x, y: (x, y), cols, rows),
        (lambda x, y: (rows - 1 - y, x), rows, cols),
        (lambda x, y: (cols - 1 - x, rows - 1 - y), cols, rows),
        (lambda x, y: (y, cols - 1 - x), rows, cols),
        (lambda x, y: (cols - 1 - x, y), cols, rows),
        (lambda x, y: (rows - 1 - y, cols - 1 - x), rows, cols),
        (lambda x, y: (x, rows - 1 - y), cols, rows),
        (lambda x, y: (y, x), rows, cols),
    )


def _rotation_after(transform, rotation):
    dx, dy = names.ROTATION_DELTA[int(rotation)]
    x0, y0 = transform(0, 0)
    x1, y1 = transform(dx, dy)
    transformed_delta = x1 - x0, y1 - y0
    return next(rot for rot, delta in names.ROTATION_DELTA.items() if delta == transformed_delta)


def _identity_variants(spec, include_gameplay):
    cols, rows = int(spec["cols"]), int(spec["rows"])
    edges = [
        (tuple(int(v) for v in pair[0]), tuple(int(v) for v in pair[1]))
        for pair in spec["edges"]
    ]
    variants = []
    for transform, transformed_cols, transformed_rows in _transforms(cols, rows):
        transformed_edges = []
        for a, b in edges:
            ta, tb = transform(*a), transform(*b)
            transformed_edges.append(sorted((ta, tb)))
        payload = {
            "shape": [transformed_cols, transformed_rows],
            "edges": sorted(transformed_edges),
        }
        if include_gameplay:
            payload.update(
                head=transform(*spec["head"]),
                exit=transform(*spec["exit"]),
                head_rotation=_rotation_after(transform, int(spec.get("head_rotation", 90))),
                budget=int(spec["budget"]),
                difficulty=spec.get("difficulty"),
                context_index=spec.get("context_index"),
            )
            for field in ("hunters", "patrollers", "tails"):
                payload[field] = sorted(
                    (
                        transform(*actor["cell"]),
                        _rotation_after(transform, int(actor["rotation"])),
                    )
                    for actor in spec.get(field, ())
                )
        variants.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return variants


def geometry_identities(spec):
    """Return raw geometry, D4 geometry, and canonical gameplay SHA-256 ids."""
    raw_geometry = _identity_variants(spec, False)[0]
    d4_geometry = min(_identity_variants(spec, False))
    gameplay = min(_identity_variants(spec, True))
    return tuple(
        hashlib.sha256(payload.encode()).hexdigest()
        for payload in (raw_geometry, d4_geometry, gameplay)
    )


def canonical_identities(spec):
    """Backward-compatible pair of D4 geometry and canonical gameplay ids."""
    _, geometry_d4, gameplay = geometry_identities(spec)
    return geometry_d4, gameplay


def layout_as_spec(layout: Layout):
    """Create the identity-bearing portion of a spec from an extracted layout."""
    cols = max(cell[0] for cell in layout.nodes) + 1
    rows = max(cell[1] for cell in layout.nodes) + 1
    return {
        "cols": cols,
        "rows": rows,
        "edges": [[list(a), list(b)] for a, b in sorted(layout.edges)],
        "head": list(layout.head),
        "head_rotation": layout.head_rotation,
        "exit": list(sorted(layout.exits)[0]),
        "hunters": [{"cell": list(cell), "rotation": rotation}
                    for cell, rotation in layout.hunters],
        "patrollers": [{"cell": list(cell), "rotation": rotation}
                       for cell, rotation in layout.patrollers],
        "tails": [{"cell": list(cell), "rotation": rotation}
                  for cell, rotation, _ in layout.tails],
        "budget": layout.max_steps,
    }
