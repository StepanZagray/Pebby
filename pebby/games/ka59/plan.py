"""Bounded exact search over KA59's complete shipped native semantics."""

from dataclasses import dataclass
from collections import deque
import hashlib
import heapq
from itertools import count, permutations

import numpy as np
from arcengine import GameState

from . import names
from .layout import Layout, extract


DEFAULT_NODE_LIMIT = 250_000
HEURISTIC_WEIGHT = 3
Action = tuple[int, int | None, int | None]
_CLICK_CACHE = {}
_DISTANCE_CACHE = {}
_PIXEL_DISTANCE_CACHE = {}


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...] | None
    truncated: bool
    unsupported: bool
    exact: bool
    expanded: int
    generated: int
    reason: str
    node_limit: int

    @property
    def solved(self):
        return self.actions is not None


def _sprites(env, tag):
    return env.level.get_sprites_by_tag(tag)


def _sprite_positions(env, tag, *, pixels=False):
    result = []
    for sprite in _sprites(env, tag):
        row = [
            sprite.name,
            int(sprite.x),
            int(sprite.y),
            int(sprite.width),
            int(sprite.height),
            int(sprite.rotation),
        ]
        if pixels:
            row.append(np.asarray(sprite.pixels).tobytes())
        result.append(tuple(row))
    return tuple(result)


def _state_key(env):
    boxes = env.boxes()
    return (
        _sprite_positions(env, names.TAG_BOX),
        boxes.index(env.selected()),
        _sprite_positions(env, names.TAG_PLAYER),
        _sprite_positions(env, names.TAG_EXPLOSIVE, pixels=True),
        _sprite_positions(env, names.TAG_ENEMY),
        env.steps_left,
    )


def _semantic_key(env):
    key = _state_key(env)
    return key[:-1]


def _selection_clicks(env):
    """Retain one verified native display click per visible box."""
    boxes = env.boxes()
    selected = boxes.index(env.selected())
    geometry = (
        tuple(
            (box.name, int(box.x), int(box.y), int(box.width), int(box.height))
            for box in boxes
        ),
        tuple(env.level.grid_size),
        selected,
    )
    cached = _CLICK_CACHE.get(geometry)
    if cached is not None:
        yield from cached
        return
    representatives = {}
    camera = env.game.camera
    scale = min(
        int(names.FRAME_SIZE / camera.width),
        int(names.FRAME_SIZE / camera.height),
    )
    x_padding = int((names.FRAME_SIZE - camera.width * scale) / 2)
    y_padding = int((names.FRAME_SIZE - camera.height * scale) / 2)
    for index, box in enumerate(boxes):
        if index == selected:
            continue
        for grid_y in range(box.y, box.y + box.height):
            for grid_x in range(box.x, box.x + box.width):
                x = (grid_x - camera.x) * scale + x_padding
                y = (grid_y - camera.y) * scale + y_padding
                if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
                    continue
                point = camera.display_to_grid(x, y)
                if point is not None and env.level.get_sprite_at(
                    *point, names.TAG_BOX
                ) is box:
                    representatives[index] = (x, y)
                    break
            if index in representatives:
                break
    result = tuple(sorted(representatives.items()))
    _CLICK_CACHE[geometry] = result
    yield from result


def _static_signature(env):
    return tuple(
        (
            sprite.name,
            int(sprite.x),
            int(sprite.y),
            int(sprite.width),
            int(sprite.height),
            int(sprite.rotation),
            np.asarray(sprite.pixels).tobytes(),
        )
        for tag in (names.TAG_WALL, names.TAG_BOUNDARY)
        for sprite in _sprites(env, tag)
    )


def _distance_maps(env, prototype, targets):
    """Static-wall distances for one object mask and compatible goal family."""
    targets = tuple(targets)
    key = (
        _static_signature(env),
        prototype.name,
        np.asarray(prototype.pixels).tobytes(),
        targets,
    )
    cached = _DISTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    colliders = tuple(
        sprite
        for tag in (names.TAG_WALL, names.TAG_BOUNDARY)
        for sprite in _sprites(env, tag)
    )
    relevant = [
        (int(sprite.x), int(sprite.y))
        for tag in (names.TAG_BOX, names.TAG_PLAYER)
        for sprite in _sprites(env, tag)
        if sprite.name == prototype.name
    ] + list(targets)
    left = min([point[0] for point in relevant] + [sprite.x for sprite in colliders]) - 3
    top = min([point[1] for point in relevant] + [sprite.y for sprite in colliders]) - 3
    right = max(
        [point[0] for point in relevant]
        + [sprite.x + sprite.width - prototype.width for sprite in colliders]
    ) + 3
    bottom = max(
        [point[1] for point in relevant]
        + [sprite.y + sprite.height - prototype.height for sprite in colliders]
    ) + 3
    x_residue, y_residue = targets[0][0] % 3, targets[0][1] % 3
    left += (x_residue - left) % 3
    top += (y_residue - top) % 3
    valid = set()
    probe = prototype.clone()
    for y in range(top, bottom + 1, names.GRID_STEP):
        for x in range(left, right + 1, names.GRID_STEP):
            probe.set_position(x, y)
            if not any(probe.collides_with(collider) for collider in colliders):
                valid.add((x, y))
    result = []
    for goal in targets:
        distances = {goal: 0} if goal in valid else {}
        queue = deque(distances)
        while queue:
            x, y = queue.popleft()
            distance = distances[(x, y)] + 1
            for dx, dy in ((0, -3), (0, 3), (-3, 0), (3, 0)):
                nxt = (x + dx, y + dy)
                if nxt in valid and nxt not in distances:
                    distances[nxt] = distance
                    queue.append(nxt)
        result.append(distances)
    result = tuple(result)
    _DISTANCE_CACHE[key] = result
    return result


def _pixel_distance_maps(env, prototype, targets):
    """Per-pixel static distances used to guide off-lattice recursive pushes."""
    targets = tuple(targets)
    key = (
        _static_signature(env),
        prototype.name,
        np.asarray(prototype.pixels).tobytes(),
        targets,
    )
    cached = _PIXEL_DISTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    colliders = tuple(
        sprite
        for tag in (names.TAG_WALL, names.TAG_BOUNDARY)
        for sprite in _sprites(env, tag)
    )
    left = min(sprite.x for sprite in colliders) - 3
    top = min(sprite.y for sprite in colliders) - 3
    right = max(sprite.x + sprite.width - prototype.width for sprite in colliders) + 3
    bottom = max(sprite.y + sprite.height - prototype.height for sprite in colliders) + 3
    valid = set()
    probe = prototype.clone()
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            probe.set_position(x, y)
            if not any(probe.collides_with(collider) for collider in colliders):
                valid.add((x, y))
    result = []
    for goal in targets:
        distances = {goal: 0} if goal in valid else {}
        queue = deque(distances)
        while queue:
            x, y = queue.popleft()
            distance = distances[(x, y)] + 1
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxt = (x + dx, y + dy)
                if nxt in valid and nxt not in distances:
                    distances[nxt] = distance
                    queue.append(nxt)
        result.append(distances)
    result = tuple(result)
    _PIXEL_DISTANCE_CACHE[key] = result
    return result


def _assignment_distance(env, objects, targets):
    if not objects:
        return 0
    if len(objects) < len(targets):
        return 1 << 20
    if not targets:
        return 0
    maps = _distance_maps(env, objects[0], targets)
    best = 1 << 30
    for chosen in permutations(objects, len(targets)):
        best = min(
            best,
            sum(
                distances.get(
                    (obj.x, obj.y),
                    (abs(obj.x - target[0]) + abs(obj.y - target[1]))
                    // names.GRID_STEP,
                )
                for obj, target, distances in zip(chosen, targets, maps)
            ),
        )
    return best


def _heuristic(env):
    """Obstacle-relaxed matching score for box and player target families."""
    total = 0
    targets_by_size = {}
    for target in env.targets():
        size = (target.width - 2, target.height - 2)
        targets_by_size.setdefault(size, []).append((target.x + 1, target.y + 1))
    boxes_by_size = {}
    for box in env.boxes():
        boxes_by_size.setdefault((box.width, box.height), []).append(box)
    for size, objects in boxes_by_size.items():
        total += _assignment_distance(env, objects, targets_by_size.get(size, ()))

    player_targets = {}
    for target in _sprites(env, names.TAG_PLAYER_TARGET):
        size = (target.width - 2, target.height - 2)
        player_targets.setdefault(size, []).append((target.x + 1, target.y + 1))
    players = {}
    for player in _sprites(env, names.TAG_PLAYER):
        players.setdefault((player.width, player.height), []).append(player)
    for size, objects in players.items():
        total += _assignment_distance(env, objects, player_targets.get(size, ()))
    return total


def _beam_score(env):
    """Push-aware guidance; only positive witnesses are accepted from the beam."""
    if len(env.boxes()) < 4:
        matched_boxes = sum(
            any(
                target.x + 1 == box.x
                and target.y + 1 == box.y
                and target.width - 2 == box.width
                and target.height - 2 == box.height
                for target in env.targets()
            )
            for box in env.boxes()
        )
        return _heuristic(env) - 25 * matched_boxes
    total = 0
    matched = 0
    targets_by_size = {}
    for target in env.targets():
        targets_by_size.setdefault(
            (target.width - 2, target.height - 2), []
        ).append((target.x + 1, target.y + 1))
    for box in env.boxes():
        goals = targets_by_size[(box.width, box.height)]
        maps = _pixel_distance_maps(env, box, goals)
        distance = min(
            (mapping.get((box.x, box.y), 300) for mapping in maps),
            default=300,
        )
        total += (distance + 2) // 3
        matched += (box.x, box.y) in goals
    player_goals = {}
    for target in _sprites(env, names.TAG_PLAYER_TARGET):
        player_goals.setdefault(
            (target.width - 2, target.height - 2), []
        ).append((target.x + 1, target.y + 1))
    for player in _sprites(env, names.TAG_PLAYER):
        goals = player_goals.get((player.width, player.height), ())
        if goals:
            total += min(
                (abs(player.x - x) + abs(player.y - y) + 2) // 3
                for x, y in goals
            )
            matched += (player.x, player.y) in goals
    return total - 60 * matched


def _stable_tie(key):
    digest = hashlib.blake2b(repr(key).encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _beam_search(layout, *, limit, node_limit, width):
    """Bounded positive-witness search for coupled push/explosion levels.

    Exhausting this pruned beam is always reported as truncation, never as a
    proof of impossibility.
    """
    start = layout.snapshot.clone()
    max_depth = min(start.steps_left, limit if limit is not None else start.steps_left)
    layer = [(start, ())]
    seen = {_semantic_key(start)}
    expanded = generated = 0
    for depth in range(max_depth):
        scored = []
        for env, path in layer:
            if expanded >= node_limit:
                return SearchResult(
                    None, True, False, True, expanded, generated,
                    f"configuration expansion limit {node_limit} reached in bounded beam",
                    node_limit,
                )
            expanded += 1
            actions = [(action, None, None) for action in names.MOVE_ACTIONS]
            actions.extend(
                (names.ACTION_CLICK, x, y) for _, (x, y) in _selection_clicks(env)
            )
            before = _semantic_key(env)
            for action in actions:
                successor = env.clone()
                score = successor.levels_completed
                observation = successor.perform(*action)
                generated += 1
                if successor.levels_completed > score or observation.state == GameState.WIN:
                    return SearchResult(
                        path + (action,), False, False, True, expanded, generated,
                        "bounded native beam found a real-engine completion",
                        node_limit,
                    )
                if observation.state == GameState.GAME_OVER:
                    continue
                key = _semantic_key(successor)
                if key == before or key in seen:
                    continue
                seen.add(key)
                scored.append(
                    (_beam_score(successor), _stable_tie(key), successor, path + (action,))
                )
        if not scored:
            return SearchResult(
                None, True, False, True, expanded, generated,
                "bounded beam frontier exhausted without an impossibility proof",
                node_limit,
            )
        scored.sort(key=lambda item: (item[0], item[1]))
        layer = [(env, path) for _, _, env, path in scored[:width]]
    return SearchResult(
        None, True, False, True, expanded, generated,
        f"action limit {max_depth} reached in bounded beam", node_limit,
    )


def _unwind(parent, state, tail):
    actions = [tail]
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return tuple(actions)


def search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Search cloned real-engine transitions with explicit independent bounds."""
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer or None")
    if isinstance(node_limit, bool) or not isinstance(node_limit, int) or node_limit < 1:
        raise ValueError("node_limit must be a positive integer")
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(
            None, False, True, False, 0, 0,
            "; ".join(layout.unsupported) or "unsupported live state", node_limit,
        )

    probe = layout.snapshot
    targets_by_size = {}
    for target in probe.targets():
        targets_by_size.setdefault(
            (target.width - 2, target.height - 2), []
        ).append((target.x + 1, target.y + 1))
    off_lattice_push_required = any(
        all(
            (box.x, box.y) not in distances
            for distances in _distance_maps(
                probe, box, targets_by_size[(box.width, box.height)]
            )
        )
        for box in probe.boxes()
    )
    has_explosives = bool(_sprites(probe, names.TAG_EXPLOSIVE))
    coupled_beam_case = layout.box_count >= 4 or (
        has_explosives and layout.box_count >= 2
    )
    if off_lattice_push_required and coupled_beam_case:
        return _beam_search(
            layout,
            limit=limit,
            node_limit=node_limit,
            width=900 if layout.box_count >= 4 else 300,
        )

    start_env = layout.snapshot.clone()
    start = _state_key(start_env)
    parent = {start: None}
    best = {start: 0}
    semantic_best = {_semantic_key(start_env): 0}
    serial = count()
    frontier = [
        (HEURISTIC_WEIGHT * _heuristic(start_env), 0, next(serial), start, start_env)
    ]
    expanded = generated = 0
    action_cutoff = False

    while frontier:
        _, cost, _, state, env = heapq.heappop(frontier)
        if cost != best.get(state):
            continue
        if expanded >= node_limit:
            return SearchResult(
                None, True, False, True, expanded, generated,
                f"configuration expansion limit {node_limit} reached", node_limit,
            )
        expanded += 1
        if limit is not None and cost >= limit:
            action_cutoff = True
            continue

        successors = []
        before_semantic = _semantic_key(env)
        for action_id in names.MOVE_ACTIONS:
            moved = env.clone()
            score = moved.levels_completed
            observation = moved.perform(action_id)
            if moved.levels_completed > score or observation.state == GameState.WIN:
                actions = _unwind(parent, state, (action_id, None, None))
                return SearchResult(
                    actions, False, False, True, expanded, generated,
                    "real-engine transition completed the level", node_limit,
                )
            if observation.state == GameState.GAME_OVER:
                continue
            if _semantic_key(moved) != before_semantic:
                successors.append(((action_id, None, None), moved))

        for _, (x, y) in _selection_clicks(env):
            selected = env.clone()
            score = selected.levels_completed
            observation = selected.perform(names.ACTION_CLICK, x, y)
            if selected.levels_completed > score or observation.state == GameState.WIN:
                actions = _unwind(parent, state, (names.ACTION_CLICK, x, y))
                return SearchResult(
                    actions, False, False, True, expanded, generated,
                    "real-engine transition completed the level", node_limit,
                )
            if observation.state != GameState.GAME_OVER:
                successors.append(((names.ACTION_CLICK, x, y), selected))

        next_cost = cost + 1
        for action, successor in successors:
            generated += 1
            semantic = _semantic_key(successor)
            if next_cost >= semantic_best.get(semantic, 1 << 30):
                continue
            nxt = _state_key(successor)
            if next_cost >= best.get(nxt, 1 << 30):
                continue
            semantic_best[semantic] = next_cost
            best[nxt] = next_cost
            parent[nxt] = (state, action)
            heapq.heappush(
                frontier,
                (
                    next_cost + HEURISTIC_WEIGHT * _heuristic(successor),
                    next_cost,
                    next(serial),
                    nxt,
                    successor,
                ),
            )

    if action_cutoff:
        return SearchResult(
            None, True, False, True, expanded, generated,
            f"action limit {limit} reached before the exact graph was exhausted", node_limit,
        )
    return SearchResult(
        None, False, False, True, expanded, generated,
        "no solution exists in the exact native state graph within the remaining budget", node_limit,
    )


def solve(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    result = search(env_or_layout, limit=limit, node_limit=node_limit)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.reason = result.reason
    return list(result.actions) if result.actions is not None else None


solve.result = None
solve.truncated = False
solve.unsupported = False
solve.reason = ""
