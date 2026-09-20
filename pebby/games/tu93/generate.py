"""Full, reference-calibrated TU93 generation and certification.

The default curriculum has one tier for each of the nine shipped native
contexts. Drafts use new sparse procedural mazes and actor assignments; no
official bitmap or route is copied. Acceptance requires calibrated structure,
an exact bounded symbolic witness, solution-level mechanic-use evidence, and a
winning replay in the unmodified engine at the intended level index.
"""

from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
from numbers import Integral
import random

from arcengine import Level, Sprite

from . import names
from .env import UPSTREAM, Env, official_levels, replay, upstream
from .layout import edge, extract
from .plan import search
from .quality import (
    MECHANICS_INVENTORY_VERSION,
    QUALITY_PROFILE_VERSION,
    REFERENCE_PROFILES,
    canonical_identities,
    geometry_identities,
    layout_as_spec,
    measure,
    structural_metrics,
    validate_profile,
)


SOURCE_SHA256 = "80e41888f9f7b1a0c03e02c0aff3814e0fd68eb5b35ef22bb3649c87fc60a23f"
if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("vendored TU93 source bytes do not match the calibrated generator")
SOURCE_ID = "tu93-0768757b"
FORMAT = "pebby.tu93.full-level.v2"
GENERATOR_VERSION = 2
DIFFICULTIES = tuple(range(1, 10))
SPLITS = ("train", "validation", "test")
SPLIT_BUCKETS = {"train": (0, 80), "validation": (80, 90), "test": (90, 100)}
SEARCH_LIMIT = {
    1: 50_000, 2: 75_000, 3: 150_000, 4: 250_000, 5: 750_000,
    6: 2_000_000, 7: 400_000, 8: 600_000, 9: 3_000_000,
}
DRAFTS_PER_SEED = 320
SPLIT_REDRAWS_PER_ATTEMPT = 32

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": [
        {"difficulty": d, "context_index": d - 1, "search_work": SEARCH_LIMIT[d]}
        for d in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "tu93-nine-official-reference-v1",
        "solution_mechanics": "tu93-action-boundary-event-certificate-v1",
        "native_budget": "tu93-native-step-budget-parity-v1",
        "context_engine_replay": "tu93-native-context-and-sequential-replay-v1",
        "novelty_split": "tu93-d4-geometry-partition-v1",
        "bounded_rejections": "tu93-bounded-rejection-report-v1",
    },
    "caveats": [
        "Each tier has one shipped reference; tolerances are explicit engineering bands, not population estimates.",
        "Tier 5 preserves the official shortest witness's hunter noninteraction and does not invent a hunter-use quota.",
        "Symbolic BFS proves shortestness in the exact action-boundary model; native replay proves a positive win, not independent native optimality.",
        "Mid-animation snapshots are unsupported; live replanning is defined at settled action boundaries.",
        "Admission relies on bounded calibration around one shipped reference per tier; it is not a population estimate.",
    ],
}


def _cell(value, field, cols, rows):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be [column, row]")
    cell = int(value[0]), int(value[1])
    if not (0 <= cell[0] < cols and 0 <= cell[1] < rows):
        raise ValueError(f"{field} {cell} is outside {cols}x{rows}")
    return cell


def _rotation(value):
    value = int(value)
    if value not in names.ROTATION_DELTA:
        raise ValueError(f"rotation must be one of {tuple(names.ROTATION_DELTA)}")
    return value


def _maze_pixels(cols, rows, edges, visible_nodes):
    height = names.CELL * (rows - 1) + names.BLOCK
    width = names.CELL * (cols - 1) + names.BLOCK
    pixels = [[names.EMPTY for _ in range(width)] for _ in range(height)]
    for col, row in visible_nodes:
        x, y = names.CELL * col, names.CELL * row
        for py in range(y, y + names.BLOCK):
            for px in range(x, x + names.BLOCK):
                pixels[py][px] = names.NODE
    for a, b in edges:
        (ac, ar), (bc, br) = a, b
        if abs(ac - bc) + abs(ar - br) != 1:
            raise ValueError(f"maze edge {a!r}-{b!r} is not between adjacent nodes")
        x = names.CELL * min(ac, bc) + (names.BLOCK if ac != bc else 0)
        y = names.CELL * min(ar, br) + (names.BLOCK if ar != br else 0)
        for py in range(y, y + names.BLOCK):
            for px in range(x, x + names.BLOCK):
                pixels[py][px] = names.PASSAGE
    return pixels


def build_level(spec):
    """Build an ``arcengine.Level`` from a JSON-compatible TU93 spec."""
    cols, rows = int(spec["cols"]), int(spec["rows"])
    if not (2 <= cols <= 9 and 2 <= rows <= 9):
        raise ValueError("cols and rows must each be in 2..9")
    origin = _cell(spec.get("origin", [3, 3]), "origin", 64, 64)
    raw_edges = []
    for index, pair in enumerate(spec["edges"]):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"edges[{index}] must contain two cells")
        a = _cell(pair[0], f"edges[{index}][0]", cols, rows)
        b = _cell(pair[1], f"edges[{index}][1]", cols, rows)
        raw_edges.append(edge(a, b))
    edges = tuple(sorted(set(raw_edges)))
    if len(edges) != len(raw_edges):
        raise ValueError("edges must not contain duplicates")
    endpoints = {cell for pair in edges for cell in pair}
    raw_visible = spec.get("visible_nodes")
    if raw_visible is None:
        visible_nodes = {(col, row) for row in range(rows) for col in range(cols)}
    else:
        visible_nodes = {
            _cell(value, f"visible_nodes[{index}]", cols, rows)
            for index, value in enumerate(raw_visible)
        }
        if len(visible_nodes) != len(raw_visible):
            raise ValueError("visible_nodes must not contain duplicates")
    if not endpoints <= visible_nodes:
        raise ValueError("every edge endpoint must be a visible node")
    head = _cell(spec["head"], "head", cols, rows)
    exit_cell = _cell(spec["exit"], "exit", cols, rows)
    if head not in endpoints or exit_cell not in endpoints:
        raise ValueError("head and exit must lie on the connected maze graph")
    budget = int(spec["budget"])
    if budget <= 0:
        raise ValueError("budget must be positive")

    ox, oy = origin
    maze = Sprite(
        pixels=_maze_pixels(cols, rows, edges, visible_nodes),
        name="pebby_tu93_maze", visible=True, collidable=True,
        tags=[names.TAG_MAZE], layer=-1,
    ).set_position(ox, oy)
    protos = upstream().sprites

    def positioned(prototype, cell, rotation=None):
        x, y = names.cell_to_pixel(origin, cell)
        sprite = protos[prototype].clone().set_position(x, y)
        return sprite if rotation is None else sprite.set_rotation(rotation)

    sprites = [
        maze,
        positioned(names.SPRITE_EXIT, exit_cell),
        positioned(names.SPRITE_HEAD, head, _rotation(spec.get("head_rotation", 90))),
    ]
    occupied = {head, exit_cell}
    for field, prototype in (
        ("hunters", names.SPRITE_HUNTER),
        ("patrollers", names.SPRITE_PATROLLER),
        ("tails", names.SPRITE_TAIL),
    ):
        for index, actor in enumerate(spec.get(field, ())):
            cell = _cell(actor["cell"], f"{field}[{index}].cell", cols, rows)
            if cell not in endpoints:
                raise ValueError(f"{field}[{index}] must lie on the connected maze graph")
            if cell in occupied:
                raise ValueError(f"{field}[{index}] overlaps another initial actor")
            occupied.add(cell)
            sprites.append(positioned(prototype, cell, _rotation(actor["rotation"])))

    maze_width = names.CELL * (cols - 1) + names.BLOCK
    maze_height = names.CELL * (rows - 1) + names.BLOCK
    raw_grid_size = spec.get("grid_size")
    if raw_grid_size is None:
        grid_width = max(ox + maze_width + 3, 12)
        grid_height = max(oy + maze_height + 3, 12)
    else:
        if not isinstance(raw_grid_size, (list, tuple)) or len(raw_grid_size) != 2:
            raise ValueError("grid_size must contain width and height")
        grid_width, grid_height = (int(raw_grid_size[0]), int(raw_grid_size[1]))
    if (not 12 <= grid_width <= names.FRAME_SIZE
            or not 12 <= grid_height <= names.FRAME_SIZE):
        raise ValueError("maze does not fit the 64x64 frame")
    if ox + maze_width > grid_width or oy + maze_height > grid_height:
        raise ValueError("maze origin and dimensions exceed grid_size")
    return Level(
        sprites=sprites, grid_size=(grid_width, grid_height),
        data={names.KEY_STEP_COUNTER: budget},
        name=f"TU93 generated d{spec.get('difficulty', '?')}",
    )


def _neighbors(cell, cols, rows):
    col, row = cell
    for dc, dr in names.ACTION_DELTA.values():
        nxt = col + dc, row + dr
        if 0 <= nxt[0] < cols and 0 <= nxt[1] < rows:
            yield nxt


def _connected_cells(rng, cols, rows, count):
    """Grow a connected sparse cell set spanning the requested rectangle."""
    count = max(count, cols + rows - 1)
    if count > cols * rows:
        return None
    sx, sy = rng.choice((0, cols - 1)), rng.choice((0, rows - 1))
    tx, ty = cols - 1 - sx, rows - 1 - sy
    cell = sx, sy
    cells = {cell}
    steps = ([(1 if tx > sx else -1, 0)] * abs(tx - sx)
             + [(0, 1 if ty > sy else -1)] * abs(ty - sy))
    rng.shuffle(steps)
    for dc, dr in steps:
        cell = cell[0] + dc, cell[1] + dr
        cells.add(cell)
    while len(cells) < count:
        frontier = [
            nxt for here in cells for nxt in _neighbors(here, cols, rows)
            if nxt not in cells
        ]
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    return cells


def _spanning_edges(rng, cells, cols, rows):
    start = rng.choice(sorted(cells))
    visited, stack, result = {start}, [start], set()
    while stack:
        here = stack[-1]
        choices = [
            nxt for nxt in _neighbors(here, cols, rows)
            if nxt in cells and nxt not in visited
        ]
        if not choices:
            stack.pop()
            continue
        nxt = rng.choice(choices)
        result.add(edge(here, nxt))
        visited.add(nxt)
        stack.append(nxt)
    return result if len(visited) == len(cells) else None


def _adjacency(edges):
    result = {}
    for a, b in edges:
        result.setdefault(a, []).append(b)
        result.setdefault(b, []).append(a)
    return result


def _shortest_path(start, goal, edges):
    adjacency = _adjacency(edges)
    queue, parent = deque([start]), {start: None}
    while queue:
        here = queue.popleft()
        if here == goal:
            break
        for nxt in adjacency.get(here, ()):
            if nxt not in parent:
                parent[nxt] = here
                queue.append(nxt)
    if goal not in parent:
        return None
    result, here = [], goal
    while here is not None:
        result.append(here)
        here = parent[here]
    return list(reversed(result))


def _endpoint_pair(rng, cells, edges):
    adjacency = _adjacency(edges)
    endpoints = [cell for cell in cells if len(adjacency[cell]) == 1] or list(cells)
    ranked = []
    for start in endpoints:
        distance, queue = {start: 0}, deque([start])
        while queue:
            here = queue.popleft()
            for nxt in adjacency[here]:
                if nxt not in distance:
                    distance[nxt] = distance[here] + 1
                    queue.append(nxt)
        for goal in endpoints:
            if start < goal:
                ranked.append((distance[goal], start, goal))
    if not ranked:
        return None
    best = max(item[0] for item in ranked)
    _, start, goal = rng.choice([item for item in ranked if item[0] >= max(1, best - 2)])
    return (start, goal) if rng.randrange(2) else (goal, start)


def _actor(cell, rotation):
    return {"cell": list(cell), "rotation": int(rotation)}


def _path_rotation(a, b):
    delta = b[0] - a[0], b[1] - a[1]
    return next(rotation for rotation, value in names.ROTATION_DELTA.items() if value == delta)


def _place_actors(rng, difficulty, cells, edges, head, exit_cell, path):
    counts = REFERENCE_PROFILES[difficulty]["actors"]
    rotations = tuple(names.ROTATION_DELTA)
    occupied = {head, exit_cell}
    path_slots = list(path[1:-1])
    rng.shuffle(path_slots)
    free = list(cells - occupied)
    rng.shuffle(free)
    hunters = []
    for _ in range(counts["hunters"]):
        if difficulty != 5 and path_slots:
            cell = path_slots.pop()
        else:
            off_path = [candidate for candidate in free if candidate not in path]
            cell = rng.choice(off_path or free)
        if cell in occupied:
            return None
        occupied.add(cell)
        free = [candidate for candidate in free if candidate != cell]
        if cell in path and path.index(cell) > 0:
            incoming = _path_rotation(cell, path[path.index(cell) - 1])
            choices = [rotation for rotation in rotations if rotation != incoming]
        else:
            choices = list(rotations)
        hunters.append(_actor(cell, rng.choice(choices)))

    patrollers = []
    adjacency = _adjacency(edges)
    for _ in range(counts["patrollers"]):
        candidates = [candidate for candidate in free if candidate not in occupied]
        if not candidates:
            return None
        off_path = [candidate for candidate in candidates if candidate not in path]
        cell = rng.choice(off_path or candidates)
        occupied.add(cell)
        free = [candidate for candidate in free if candidate != cell]
        outgoing = [_path_rotation(cell, nxt) for nxt in adjacency[cell]]
        patrollers.append(_actor(cell, rng.choice(outgoing)))

    tails = []
    for _ in range(counts["tails"]):
        candidates = []
        for route_index, target in enumerate(path[1:-1], 1):
            for rotation, (dx, dy) in names.ROTATION_DELTA.items():
                cell = target[0] - 2 * dx, target[1] - 2 * dy
                if cell in cells and cell not in occupied and cell not in path:
                    candidates.append((abs(route_index - len(path) // 2), cell, rotation))
        if candidates:
            best = min(candidate[0] for candidate in candidates)
            _, cell, rotation = rng.choice([item for item in candidates if item[0] <= best + 2])
        else:
            remaining = [candidate for candidate in free if candidate not in occupied]
            if not remaining:
                return None
            cell, rotation = rng.choice(remaining), rng.choice(rotations)
        occupied.add(cell)
        free = [candidate for candidate in free if candidate != cell]
        tails.append(_actor(cell, rotation))
    return hunters, patrollers, tails


def _draft(rng, difficulty):
    profile = REFERENCE_PROFILES[difficulty]
    tolerance = profile["tolerance"]
    short = rng.randint(*tolerance["shape_short"])
    long = rng.randint(max(short, tolerance["shape_long"][0]), tolerance["shape_long"][1])
    cols, rows = (short, long) if rng.randrange(2) else (long, short)
    lower = max(tolerance["active_nodes"][0], cols + rows - 1)
    upper = min(tolerance["active_nodes"][1], cols * rows)
    if lower > upper:
        return None
    cells = _connected_cells(rng, cols, rows, rng.randint(lower, upper))
    if cells is None:
        return None
    edges = _spanning_edges(rng, cells, cols, rows)
    if edges is None:
        return None
    possible_extra = [
        edge(cell, nxt) for cell in cells for nxt in _neighbors(cell, cols, rows)
        if nxt in cells and edge(cell, nxt) not in edges and cell < nxt
    ]
    rng.shuffle(possible_extra)
    cycle_low, cycle_high = tolerance["cycle_rank"]
    cycle_high = min(cycle_high, len(possible_extra))
    if cycle_low > cycle_high:
        return None
    edges.update(possible_extra[:rng.randint(cycle_low, cycle_high)])
    endpoints = _endpoint_pair(rng, cells, edges)
    if endpoints is None:
        return None
    head, exit_cell = endpoints
    path = _shortest_path(head, exit_cell, edges)
    actors = _place_actors(rng, difficulty, cells, edges, head, exit_cell, path)
    if actors is None:
        return None
    hunters, patrollers, tails = actors
    maze_width = names.CELL * (cols - 1) + names.BLOCK
    maze_height = names.CELL * (rows - 1) + names.BLOCK
    reference_grid = profile["reference"]["grid_size"][0]
    grid_size = max(reference_grid, maze_width + 6, maze_height + 6)
    origin = [(grid_size - maze_width) // 2, (grid_size - maze_height) // 2]
    return {
        "format": FORMAT, "generator_version": GENERATOR_VERSION, "game": "tu93",
        "source_id": SOURCE_ID,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "difficulty": difficulty, "context_index": difficulty - 1,
        "cols": cols, "rows": rows, "origin": origin,
        "grid_size": [grid_size, grid_size],
        "visible_nodes": [list(cell) for cell in sorted(cells)],
        "edges": [[list(a), list(b)] for a, b in sorted(edges)],
        "head": list(head), "head_rotation": rng.choice(tuple(names.ROTATION_DELTA)),
        "exit": list(exit_cell), "hunters": hunters, "patrollers": patrollers,
        "tails": tails, "budget": profile["native_budget"],
    }


def _jsonable(value):
    return json.loads(json.dumps(value, separators=(",", ":")))


def _integer(value, field):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _split_bucket(geometry_identity):
    return int(geometry_identity[:16], 16) % 100


def _split_accepts(split, bucket):
    lower, upper = SPLIT_BUCKETS[split]
    return lower <= bucket < upper


_official_geometry_cache = None


def _official_geometry_ids():
    global _official_geometry_cache
    if _official_geometry_cache is None:
        env, identities = Env(), set()
        for index in range(len(official_levels())):
            env.set_level(index)
            geometry, _ = canonical_identities(layout_as_spec(extract(env)))
            identities.add(geometry)
        _official_geometry_cache = frozenset(identities)
    return _official_geometry_cache


def _native_context_replay(spec, actions, context_index):
    env = Env([build_level(spec) for _ in range(context_index + 1)])
    env.reset()
    if context_index:
        env.set_level(context_index)
    if env.level_index != context_index:
        return None
    start_score = env.levels_completed
    completed, observation = replay(env, actions)
    if not completed or observation is None:
        return None
    return {
        "context_index": context_index, "level_count": context_index + 1,
        "start_score": start_score, "end_score": env.levels_completed,
        "score_delta": env.levels_completed - start_score,
        "end_level_index": env.level_index, "engine_state": observation.state.name,
        "engine_verified": True,
    }


def _structural_errors(difficulty, layout):
    metrics = structural_metrics(layout)
    placeholder = dict(metrics)
    placeholder["solution_length"] = REFERENCE_PROFILES[difficulty]["reference"][
        "shortest_symbolic_actions"
    ]
    placeholder["witness"] = {"won": True, "dead": False}
    return [
        error for error in validate_profile(difficulty, placeholder)
        if not error.startswith("witness ") and not error.startswith("stored witness")
    ]


def certify(spec, limit=None, *, split=None, context_index=None):
    """Return ``(verified_spec, result)`` after full proof and native replay."""
    errors, candidate = [], _jsonable(spec)
    try:
        difficulty = _integer(candidate.get("difficulty"), "difficulty")
    except ValueError as exc:
        certify.last_errors = [f"difficulty: {exc}"]
        return None, None
    if difficulty not in DIFFICULTIES:
        certify.last_errors = ["difficulty: outside the complete TU93 curriculum"]
        return None, None
    expected_context = difficulty - 1
    try:
        requested_context = (
            expected_context if context_index is None
            else _integer(context_index, "context_index")
        )
    except ValueError as exc:
        certify.last_errors = [f"context: {exc}"]
        return None, None
    if requested_context != expected_context:
        errors.append(f"context: expected {expected_context}, got {requested_context}")
    candidate_split = candidate.get("split", split if split is not None else "train")
    if candidate_split not in SPLITS:
        errors.append(f"split: expected one of {SPLITS}, got {candidate_split!r}")
    if split is not None and candidate_split != split:
        errors.append(f"split: spec has {candidate_split!r}, caller requested {split!r}")
    try:
        search_limit = (
            SEARCH_LIMIT[difficulty] if limit is None else _integer(limit, "limit")
        )
    except ValueError as exc:
        errors.append(f"search_work: {exc}")
    if search_limit <= 0:
        errors.append("search_work: must be positive")
    if errors:
        certify.last_errors = errors
        return None, None
    candidate.update(
        format=FORMAT, generator_version=GENERATOR_VERSION, game="tu93",
        source_id=SOURCE_ID, mechanics_inventory_version=MECHANICS_INVENTORY_VERSION,
        quality_profile_version=QUALITY_PROFILE_VERSION, context_index=expected_context,
        split=candidate_split,
    )
    try:
        env = Env([build_level(candidate)])
        env.reset()
        layout = extract(env)
    except (KeyError, TypeError, ValueError) as exc:
        certify.last_errors = [f"build: {exc}"]
        return None, None
    if not layout.exact:
        certify.last_errors = ["layout: " + "; ".join(layout.unsupported)]
        return None, None
    geometry_sha256, geometry_d4_sha256, gameplay_sha256 = geometry_identities(candidate)
    bucket = _split_bucket(geometry_d4_sha256)
    endpoints = {tuple(cell) for pair in candidate["edges"] for cell in pair}
    visible_nodes = {tuple(cell) for cell in candidate.get("visible_nodes", ())}
    if visible_nodes != endpoints:
        errors.append("profile_structure: visible_nodes must exactly equal active graph nodes")
    if not _split_accepts(candidate_split, bucket):
        errors.append(f"split_partition: geometry bucket {bucket} is not {candidate_split}")
    if geometry_d4_sha256 in _official_geometry_ids():
        errors.append("official_copy: canonical geometry matches a shipped level")
    errors.extend(f"profile_structure: {error}" for error in _structural_errors(difficulty, layout))
    if errors:
        certify.last_errors = errors
        return None, None

    result = search(layout, limit=search_limit)
    if not result.solved:
        status = "search_inconclusive" if result.truncated or not result.exact else "search_unsolved"
        certify.last_errors = [f"{status}: {result.reason}"]
        return None, result
    if result.truncated or not result.exact:
        certify.last_errors = [f"search_inconclusive: {result.reason}"]
        return None, result
    actions = [[action, x, y] for action, x, y in result.actions]
    quality = measure(layout, actions)
    profile_errors = validate_profile(difficulty, quality)
    if profile_errors:
        certify.last_errors = [f"profile_witness: {error}" for error in profile_errors]
        return None, result
    native = _native_context_replay(candidate, actions, expected_context)
    if native is None:
        certify.last_errors = ["native_replay: witness did not win at intended context"]
        return None, result
    final = dict(candidate)
    final.update(
        solution=actions, solution_length=len(actions), optimal_actions=len(actions),
        symbolic_shortest=True, search_limit=search_limit,
        search_expanded=result.expanded, search_generated=result.generated,
        search_truncated=False, planner_exact=True, engine_verified=True,
        vendored_source_sha256=SOURCE_SHA256,
        geometry_sha256=geometry_sha256,
        geometry_d4_sha256=geometry_d4_sha256,
        gameplay_sha256=gameplay_sha256,
        geometry_split=candidate_split,
        split_partition_bucket=bucket, quality=quality,
        mechanics={
            "hunters": len(layout.hunters), "patrollers": len(layout.patrollers),
            "tails": len(layout.tails), "witness": quality["witness"],
        },
        proof={
            "format": "tu93-positive-certificate-v2",
            "model": "exact-action-boundary-bfs", "action_count": len(actions),
            "symbolic_shortest": True, "search_work": search_limit,
            "search_expanded": result.expanded, "search_generated": result.generated,
            "search_truncated": False, "planner_exact": True,
            "native_budget": layout.max_steps,
            "remaining_budget_at_start": layout.steps_left,
            "geometry_split": candidate_split,
            "context_engine_replay": native,
        },
    )
    certify.last_errors = []
    return _jsonable(final), result


certify.last_errors = []


def _rejection_code(errors):
    return "draft" if not errors else errors[0].split(":", 1)[0]


def generate(seed, difficulty=1, attempts=DRAFTS_PER_SEED, limit=None, *,
             split="train", node_limit=None):
    """Generate one deterministic full-standard level, or return ``None``."""
    difficulty = _integer(difficulty, "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    attempts = _integer(attempts, "attempts")
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if limit is not None and node_limit is not None:
        raise ValueError("pass at most one of limit and node_limit")
    search_work = SEARCH_LIMIT[difficulty]
    if node_limit is not None:
        search_work = _integer(node_limit, "node_limit")
    elif limit is not None:
        search_work = _integer(limit, "limit")
    if search_work <= 0 or search_work > 32_000_000:
        raise ValueError("search work must be in 1..32,000,000")
    seed = _integer(seed, "seed")
    rng = random.Random(f"{SOURCE_ID}:{GENERATOR_VERSION}:{split}:{seed}:{difficulty}")
    rejection_counts = Counter()
    report = {
        "format": "tu93-generation-report-v1", "seed": seed,
        "difficulty": difficulty, "split": split, "attempt_limit": attempts,
        "search_work": search_work, "attempted": 0, "accepted": False,
        "drafts_drawn": 0, "rejections": {},
    }
    for attempt_index in range(attempts):
        report["attempted"] += 1
        draft = None
        for _ in range(SPLIT_REDRAWS_PER_ATTEMPT):
            report["drafts_drawn"] += 1
            proposal = _draft(rng, difficulty)
            if proposal is None:
                rejection_counts["draft"] += 1
                continue
            _, geometry_d4, _ = geometry_identities(proposal)
            if not _split_accepts(split, _split_bucket(geometry_d4)):
                rejection_counts["split_partition"] += 1
                continue
            draft = proposal
            break
        if draft is None:
            rejection_counts["split_redraw_exhausted"] += 1
            continue
        draft.update(seed=seed, split=split, attempt_index=attempt_index)
        verified, _ = certify(draft, limit=search_work, split=split)
        if verified is None:
            rejection_counts[_rejection_code(certify.last_errors)] += 1
            continue
        report.update(
            accepted=True, accepted_attempt_index=attempt_index,
            rejections=dict(sorted(rejection_counts.items())),
        )
        verified["generation_diagnostics"] = _jsonable(report)
        generate.last_report = _jsonable(report)
        return _jsonable(verified)
    report["rejections"] = dict(sorted(rejection_counts.items()))
    generate.last_report = _jsonable(report)
    return None


generate.last_report = None


def _child_seed(parent_seed, position, difficulty):
    payload = f"{SOURCE_ID}\0{int(parent_seed)}\0{int(position)}\0{int(difficulty)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _difficulty_sequence(difficulties):
    if difficulties is None:
        sequence = DIFFICULTIES
    else:
        if isinstance(difficulties, (str, bytes)):
            raise ValueError("difficulties must be a sequence of integers")
        try:
            raw_sequence = tuple(difficulties)
        except TypeError as exc:
            raise ValueError("difficulties must be a sequence of integers") from exc
        sequence = tuple(_integer(value, "difficulty") for value in raw_sequence)
    if not sequence:
        raise ValueError("difficulties must not be empty")
    if any(value <= 0 or value not in DIFFICULTIES for value in sequence):
        raise ValueError(f"difficulties must be drawn from {DIFFICULTIES}")
    if tuple(sorted(set(sequence))) != sequence:
        raise ValueError("difficulties must be positive, distinct, and strictly increasing")
    return sequence


def _validated_full_game(specs):
    """Validate, build, and replay one complete unforced native episode."""
    specs = list(specs)
    if len(specs) != len(DIFFICULTIES):
        raise ValueError(f"full TU93 games require exactly {len(DIFFICULTIES)} specs")
    if not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("full TU93 game specs must be mappings")
    try:
        difficulties = tuple(
            _integer(spec.get("difficulty"), "difficulty") for spec in specs
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("full TU93 game specs require integer difficulties") from exc
    if difficulties != DIFFICULTIES:
        raise ValueError(f"full TU93 game difficulties must be exactly {DIFFICULTIES}")
    splits = {spec.get("split") for spec in specs}
    if len(splits) != 1 or next(iter(splits)) not in SPLITS:
        raise ValueError("full TU93 game specs must have one consistent valid split")
    for index, spec in enumerate(specs):
        if _integer(spec.get("context_index"), "context_index") != index:
            raise ValueError(f"spec {index} has the wrong native context index")
    for field in ("geometry_d4_sha256", "gameplay_sha256"):
        identities = [spec.get(field) for spec in specs]
        if any(not isinstance(value, str) or len(value) != 64 for value in identities):
            raise ValueError(f"every full-game spec must have a valid {field}")
        if len(set(identities)) != len(identities):
            raise ValueError(f"full TU93 game contains duplicate {field} identities")
    for index, (spec, entry) in enumerate(
        zip(specs, FULL_STANDARD_CONTRACT["curriculum"])
    ):
        validation_errors = validate_full_standard(spec, entry)
        if validation_errors:
            raise ValueError(
                f"spec {index} failed full-standard validation: "
                + "; ".join(validation_errors)
            )
    levels = [build_level(spec) for spec in specs]
    env = Env(levels)
    env.reset()
    replay_rows = []
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            raise ValueError(f"native full-game context did not reach tier {index + 1}")
        completed, observation = replay(env, spec["solution"])
        if not completed or observation is None or env.levels_completed != index + 1:
            raise ValueError(f"native full-game replay failed at tier {index + 1}")
        replay_rows.append({
            "game_position": index,
            "end_score": env.levels_completed,
            "end_level_index": env.level_index,
            "engine_state": observation.state.name,
        })
    if env.levels_completed != len(DIFFICULTIES) or env.state.name != "WIN":
        raise ValueError("native full-game replay did not finish all nine tiers")
    return levels, replay_rows


def build_game(specs):
    """Validate, replay, and build a complete nine-level native game."""
    levels, _ = _validated_full_game(specs)
    return levels


def generate_game(seed, *, split="train", difficulties=None,
                  attempts=DRAFTS_PER_SEED, node_limit=None):
    """Generate an increasing-difficulty sequence, or ``None`` on bounded failure."""
    sequence = _difficulty_sequence(difficulties)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    parent_seed = _integer(seed, "seed")
    specs, reports = [], []
    for position, difficulty in enumerate(sequence):
        child_seed = _child_seed(parent_seed, position, difficulty)
        spec = generate(
            child_seed, difficulty, attempts=attempts, split=split,
            node_limit=node_limit,
        )
        reports.append(_jsonable(generate.last_report))
        if spec is None:
            generate_game.last_report = {
                "accepted": False, "parent_game_seed": parent_seed, "split": split,
                "failed_position": position, "failed_difficulty": difficulty,
                "tier_reports": reports,
            }
            return None
        spec.update(
            parent_game_seed=parent_seed, game_position=position,
            requested_tier=difficulty,
        )
        specs.append(spec)
    if sequence == DIFFICULTIES:
        try:
            _, replay_rows = _validated_full_game(specs)
        except ValueError as exc:
            generate_game.last_report = {
                "accepted": False, "reason": str(exc), "tier_reports": reports,
            }
            return None
        for spec, replay_row in zip(specs, replay_rows):
            spec["proof"]["full_game_replay"] = {
                "parent_game_seed": parent_seed,
                **replay_row,
            }
    generate_game.last_report = {
        "accepted": True, "parent_game_seed": parent_seed, "split": split,
        "difficulties": list(sequence), "tier_reports": reports,
    }
    return _jsonable(specs)


generate_game.last_report = None


def validate_full_standard(spec, curriculum_entry):
    """Recompute the complete full-standard certificate; return all failures."""
    errors = []
    if not isinstance(spec, dict) or not isinstance(curriculum_entry, dict):
        return ["spec and curriculum entry must be mappings"]
    try:
        difficulty = _integer(spec.get("difficulty"), "difficulty")
        entry_difficulty = _integer(
            curriculum_entry.get("difficulty"), "curriculum difficulty"
        )
        entry_context = _integer(
            curriculum_entry.get("context_index"), "curriculum context_index"
        )
        search_work = _integer(
            curriculum_entry.get("search_work"), "curriculum search_work"
        )
    except ValueError:
        return ["difficulty, context index, and search work must be integers"]
    if entry_difficulty != difficulty:
        errors.append("curriculum difficulty differs from spec difficulty")
    expected_entry = next(
        (entry for entry in FULL_STANDARD_CONTRACT["curriculum"]
         if entry["difficulty"] == difficulty), None,
    )
    if expected_entry is None:
        return [f"difficulty {difficulty} is outside the full curriculum"]
    if entry_context != expected_entry["context_index"]:
        errors.append("curriculum context differs from the canonical tier context")
    if curriculum_entry != expected_entry:
        errors.append("curriculum entry does not match the canonical tier entry")
    split = spec.get("split")
    expected_scalars = {
        "format": FORMAT, "generator_version": GENERATOR_VERSION, "game": "tu93",
        "source_id": SOURCE_ID,
        "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "context_index": expected_entry["context_index"],
        "vendored_source_sha256": SOURCE_SHA256,
        "geometry_split": split,
    }
    for field, expected in expected_scalars.items():
        if spec.get(field) != expected:
            errors.append(f"{field}={spec.get(field)!r}, expected {expected!r}")
    if split not in SPLITS:
        errors.append(f"split={split!r}, expected one of {SPLITS}")
    if search_work <= 0 or search_work > 32_000_000:
        errors.append("curriculum search_work must be in 1..32,000,000")
    try:
        env = Env([build_level(spec)])
        env.reset()
        layout = extract(env)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"build failed: {exc}")
        return errors
    if not layout.exact:
        errors.append("extracted layout is unsupported: " + "; ".join(layout.unsupported))
        return errors
    geometry_sha256, geometry_d4_sha256, gameplay_sha256 = geometry_identities(spec)
    endpoints = {tuple(cell) for pair in spec["edges"] for cell in pair}
    visible_nodes = {tuple(cell) for cell in spec.get("visible_nodes", ())}
    if visible_nodes != endpoints:
        errors.append("visible_nodes must exactly equal active graph nodes")
    if spec.get("geometry_sha256") != geometry_sha256:
        errors.append("geometry_sha256 does not match raw coordinate geometry")
    if spec.get("geometry_d4_sha256") != geometry_d4_sha256:
        errors.append("geometry_d4_sha256 does not match canonical D4 geometry")
    if spec.get("gameplay_sha256") != gameplay_sha256:
        errors.append("gameplay_sha256 does not match canonical gameplay")
    bucket = _split_bucket(geometry_d4_sha256)
    if spec.get("split_partition_bucket") != bucket:
        errors.append("split_partition_bucket does not match canonical geometry")
    if split in SPLITS and not _split_accepts(split, bucket):
        errors.append("canonical geometry belongs to a different split partition")
    if spec.get("geometry_split") != split:
        errors.append("geometry_split differs from the requested split")
    if geometry_d4_sha256 in _official_geometry_ids():
        errors.append("canonical geometry duplicates a shipped level")
    raw_actions = spec.get("solution")
    if not isinstance(raw_actions, list) or not raw_actions:
        errors.append("solution must be a nonempty list of action triples")
        return errors
    actions = []
    for index, step in enumerate(raw_actions):
        if (not isinstance(step, (list, tuple)) or len(step) != 3
                or step[0] not in names.ACTION_IDS or list(step[1:]) != [None, None]):
            errors.append(f"solution[{index}] is not a legal TU93 action triple")
            return errors
        actions.append([int(step[0]), None, None])
    quality = measure(layout, actions)
    if spec.get("quality") != quality:
        errors.append("stored quality metrics do not match recomputed metrics")
    errors.extend(validate_profile(difficulty, quality))
    proof = spec.get("proof", {})
    if not isinstance(proof, dict):
        errors.append("proof must be a mapping")
        proof = {}
    expected_proof = {
        "action_count": len(actions), "symbolic_shortest": True,
        "search_work": search_work, "search_truncated": False,
        "planner_exact": True, "native_budget": layout.max_steps,
        "remaining_budget_at_start": layout.steps_left,
        "geometry_split": split,
    }
    for field, expected in expected_proof.items():
        if proof.get(field) != expected:
            errors.append(f"proof.{field}={proof.get(field)!r}, expected {expected!r}")
    for field in ("search_expanded", "search_generated"):
        if not isinstance(proof.get(field), int) or proof[field] <= 0:
            errors.append(f"proof.{field} must be a positive integer")
        if proof.get(field) != spec.get(field):
            errors.append(f"proof.{field} disagrees with top-level {field}")
    if spec.get("solution_length") != len(actions) or spec.get("optimal_actions") != len(actions):
        errors.append("stored solution-length metadata disagrees with the route")
    if spec.get("search_limit") != search_work or spec.get("symbolic_shortest") is not True:
        errors.append("top-level search-work/shortest metadata is inconsistent")
    expected_mechanics = {
        "hunters": len(layout.hunters), "patrollers": len(layout.patrollers),
        "tails": len(layout.tails), "witness": quality["witness"],
    }
    if spec.get("mechanics") != expected_mechanics:
        errors.append("stored mechanics do not match recomputed route mechanics")
    native = _native_context_replay(
        spec, actions, int(expected_entry["context_index"])
    )
    if native is None:
        errors.append("real engine did not replay the witness at its native context")
    elif proof.get("context_engine_replay") != native:
        errors.append("stored context replay metadata does not match real-engine replay")
    if spec.get("engine_verified") is not True:
        errors.append("engine_verified must be true")
    if spec.get("planner_exact") is not True or spec.get("search_truncated") is not False:
        errors.append("top-level planner completeness metadata is inconsistent")
    return errors
