"""Bounded full-mechanics teacher for RE86.

RE86 movables never collide with one another: the only dynamic interactions
are between the selected movable and static obstacles/dyes. The teacher uses
that native invariant to enumerate each movable's exact mutable-pixel state
independently, joins their target signatures, and replays the combined route
in the original native context. A positive result is an actual-engine
certificate. A cutoff is unknown, never a proof of impossibility, and no
shortest-route claim is made.
"""

from collections import deque
from dataclasses import dataclass
import itertools
from numbers import Integral

import numpy as np
from arcengine import GameState, Level, Sprite

from . import names
from .env import Env, upstream


DEFAULT_NODE_LIMIT = 200_000
Action = tuple[int, int | None, int | None]


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
    optimal: bool = False

    @property
    def solved(self):
        return self.actions is not None


@dataclass(frozen=True)
class _Candidate:
    sprite: object
    path: tuple[Action, ...]
    selected_signature: tuple[int, ...]
    stable_signature: tuple[int, ...]


def _positive_integer(value, label, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        suffix = " or None" if optional else ""
        raise ValueError(f"{label} must be a positive integer{suffix}")
    return int(value)


def normalize_unselected(sprite, module=None):
    """Apply native ACTION5 centre normalization to one movable clone."""
    module = upstream() if module is None else module
    row, col = sprite.height // 2, sprite.width // 2
    if names.TAG_FLEXIBLE in sprite.tags:
        sprite.pixels[row, col] = names.TRANSPARENT
        return sprite
    color = int(getattr(module, "euqngakkse")(sprite))
    if names.TAG_FIXED_CENTER in sprite.tags:
        sprite.pixels[row, col] = color
        return sprite
    full_col = next(
        (value for value in range(sprite.width)
         if np.all(sprite.pixels[:, value] != names.TRANSPARENT)),
        sprite.width - 1,
    )
    full_row = next(
        (value for value in range(sprite.height)
         if np.all(sprite.pixels[value, :] != names.TRANSPARENT)),
        sprite.height - 1,
    )
    sprite.pixels[row, col] = (
        color if row == full_row or col == full_col else names.TRANSPARENT
    )
    return sprite


def _sprite_fingerprint(sprite):
    return (
        int(sprite.x), int(sprite.y), int(sprite.rotation),
        int(sprite.height), int(sprite.width), sprite.pixels.tobytes(),
    )


def _value_at(sprite, row, col):
    local_row, local_col = row - sprite.y, col - sprite.x
    if 0 <= local_row < sprite.height and 0 <= local_col < sprite.width:
        return int(sprite.pixels[local_row, local_col])
    return names.TRANSPARENT


def _signature(sprite, requirements, *, selected, module):
    probe = sprite.clone()
    if not selected:
        normalize_unselected(probe, module)
    return tuple(_value_at(probe, row, col) for row, col, _ in requirements)


def _search_level(env, index):
    """One selected movable plus exact static interactors and a false target."""
    module = upstream()
    movable = env.movables()[index].clone()
    normalize_unselected(movable, module)
    movable.pixels[movable.height // 2, movable.width // 2] = names.SELECTED_CENTER
    interactors = [
        sprite.clone() for sprite in env.level.get_sprites()
        if names.TAG_OBSTACLE in sprite.tags or names.TAG_DYE in sprite.tags
    ]
    sentinel = np.full(
        (names.FRAME_SIZE, names.FRAME_SIZE), names.TRANSPARENT, dtype=np.int8
    )
    sentinel[0, 0] = 15
    target = Sprite(
        pixels=sentinel,
        name="re86-search-sentinel",
        visible=False,
        collidable=True,
        tags=[names.TAG_TARGET],
        layer=-2,
    )
    level = Level(
        sprites=[*interactors, movable, target],
        grid_size=(names.FRAME_SIZE, names.FRAME_SIZE),
        data={names.KEY_STEP_COUNTER: max(1, env.steps_left)},
        name="re86-factored-search",
    )
    return Env([level])


def _static_sprite(sprite):
    return (int(sprite.x), int(sprite.y), np.asarray(sprite.render()).copy(),
            int(sprite.pixels[1, 1]) if sprite.height > 1 and sprite.width > 1 else None)


def _collides(x, y, pixels, static):
    other_x, other_y, other, _ = static
    height, width = pixels.shape
    other_height, other_width = other.shape
    x0, x1 = max(x, other_x), min(x + width, other_x + other_width)
    y0, y1 = max(y, other_y), min(y + height, other_y + other_height)
    if x0 >= x1 or y0 >= y1:
        return False
    left = pixels[y0 - y:y1 - y, x0 - x:x1 - x]
    right = other[y0 - other_y:y1 - other_y,
                  x0 - other_x:x1 - other_x]
    return bool(np.any((left != names.TRANSPARENT)
                       & (right != names.TRANSPARENT)))


def _array_main_color(pixels):
    values = pixels[(pixels != names.SELECTED_CENTER)
                    & (pixels != names.TRANSPARENT)]
    return int(values[0])


def _normalize_pixels(pixels, *, flexible, fixed_center):
    pixels = pixels.copy()
    row, col = pixels.shape[0] // 2, pixels.shape[1] // 2
    if flexible:
        pixels[row, col] = names.TRANSPARENT
        return pixels
    color = _array_main_color(pixels)
    if fixed_center:
        pixels[row, col] = color
        return pixels
    full_col = next(
        (value for value in range(pixels.shape[1])
         if np.all(pixels[:, value] != names.TRANSPARENT)),
        pixels.shape[1] - 1,
    )
    full_row = next(
        (value for value in range(pixels.shape[0])
         if np.all(pixels[value, :] != names.TRANSPARENT)),
        pixels.shape[0] - 1,
    )
    pixels[row, col] = (
        color if row == full_row or col == full_col else names.TRANSPARENT
    )
    return pixels


def _state(sprite):
    pixels = np.asarray(sprite.pixels, dtype=np.int8)
    return (int(sprite.x), int(sprite.y), pixels.shape[0], pixels.shape[1],
            pixels.tobytes())


def _state_array(state):
    return np.frombuffer(state[4], dtype=np.int8).reshape(state[2], state[3]).copy()


def _sprite_from_state(template, state):
    sprite = template.clone().set_position(state[0], state[1])
    sprite.pixels = _state_array(state)
    return sprite


def _pure_move(state, action_id, *, flexible, fixed_center, obstacles, dyes):
    """Literal action-boundary translation of upstream ucpbzrcoui/step."""
    dx, dy = names.ACTION_DELTAS[action_id]
    old_x, old_y = state[0], state[1]
    x, y = old_x + dx, old_y + dy
    pixels = _state_array(state)
    height, width = pixels.shape
    center_x, center_y = x + width // 2, y + height // 2
    if not (0 <= center_x < names.FRAME_SIZE and 0 <= center_y < names.FRAME_SIZE):
        return state
    for obstacle in obstacles:
        if not _collides(x, y, pixels, obstacle):
            continue
        obstacle_x, obstacle_y, obstacle_pixels, _ = obstacle
        if flexible:
            if dx:
                if width <= 6:
                    return state
                pixels[height // 2, width // 2] = names.TRANSPARENT
                color = _array_main_color(pixels)
                offset = height // 2 - (height + 3) // 2
                pixels = np.full((height + 3, width - 3), names.TRANSPARENT,
                                 dtype=np.int8)
                pixels[:, (0, -1)] = color
                pixels[(0, -1), :] = color
                pixels[pixels.shape[0] // 2, pixels.shape[1] // 2] = 0
                y += offset
                y += round(y / 3) * 3 - y
                if dx < 0:
                    x -= dx
                    y -= dy
            elif dy:
                if height <= 6:
                    return state
                pixels[height // 2, width // 2] = names.TRANSPARENT
                color = _array_main_color(pixels)
                offset = width // 2 - (width + 3) // 2
                pixels = np.full((height - 3, width + 3), names.TRANSPARENT,
                                 dtype=np.int8)
                pixels[:, (0, -1)] = color
                pixels[(0, -1), :] = color
                pixels[pixels.shape[0] // 2, pixels.shape[1] // 2] = 0
                x += offset
                x += round(x / 3) * 3 - x
                if dy < 0:
                    x -= dx
                    y -= dy
            return (x, y, pixels.shape[0], pixels.shape[1], pixels.tobytes())

        color = _array_main_color(pixels)
        full_col = next(
            (value for value in range(width)
             if np.all(pixels[:, value] != names.TRANSPARENT)), width - 1
        )
        full_row = next(
            (value for value in range(height)
             if np.all(pixels[value, :] != names.TRANSPARENT)), height - 1
        )
        horizontal_hit = obstacle_x <= x + full_col < obstacle_x + obstacle_pixels.shape[1]
        vertical_hit = obstacle_y <= y + full_row < obstacle_y + obstacle_pixels.shape[0]
        center_row, center_col = height // 2, width // 2
        if dx:
            backward = -3 if dx > 0 else 3
            forward = 3 if dx > 0 else -3
            can_back = ((full_col > 0 if dx > 0 else full_col < width - 2)
                        and 0 <= full_col + backward < width)
            can_forward = ((full_col < width - 2 if dx > 0 else full_col > 0)
                           and 0 <= full_col + forward < width)
            if horizontal_hit and vertical_hit:
                x, y = old_x, old_y
            elif horizontal_hit:
                if can_back:
                    pixels[:, full_col] = names.TRANSPARENT
                    pixels[:, full_col + backward] = color
                    pixels[full_row, full_col] = color
                    pixels[center_row, center_col] = 0
                else:
                    x, y = old_x, old_y
            elif vertical_hit:
                x, y = old_x, old_y
                if can_forward:
                    pixels[:, full_col] = names.TRANSPARENT
                    pixels[:, full_col + forward] = color
                    pixels[full_row, full_col] = color
                    pixels[center_row, center_col] = 0
        elif dy:
            backward = -3 if dy > 0 else 3
            forward = 3 if dy > 0 else -3
            can_back = ((full_row > 0 if dy > 0 else full_row < height - 2)
                        and 0 <= full_row + backward < height)
            can_forward = ((full_row < height - 2 if dy > 0 else full_row > 0)
                           and 0 <= full_row + forward < height)
            if vertical_hit and horizontal_hit:
                x, y = old_x, old_y
            elif vertical_hit:
                if can_back:
                    pixels[full_row, :] = names.TRANSPARENT
                    pixels[full_row + backward, :] = color
                    pixels[full_row, full_col] = color
                    pixels[center_row, center_col] = 0
                else:
                    x, y = old_x, old_y
            elif horizontal_hit:
                x, y = old_x, old_y
                if can_forward:
                    pixels[full_row, :] = names.TRANSPARENT
                    pixels[full_row + forward, :] = color
                    pixels[full_row, full_col] = color
                    pixels[center_row, center_col] = 0

    pixels = _normalize_pixels(
        pixels, flexible=flexible, fixed_center=fixed_center
    )
    for dye in dyes:
        dye_color = int(dye[3])
        if not _collides(x, y, pixels, dye) or _array_main_color(pixels) == dye_color:
            continue
        dye_x, dye_y, dye_pixels, _ = dye
        for row in range(pixels.shape[0]):
            for col in range(pixels.shape[1]):
                global_x, global_y = x + col, y + row
                if (dye_x <= global_x < dye_x + dye_pixels.shape[1]
                        and dye_y <= global_y < dye_y + dye_pixels.shape[0]
                        and pixels[row, col] != names.TRANSPARENT):
                    pixels[row, col] = dye_color
        # Every shipped/generated movable is one 8-connected occupied
        # component; native animation therefore ends at this same stable mask.
        pixels[pixels != names.TRANSPARENT] = dye_color
        break
    pixels[pixels.shape[0] // 2, pixels.shape[1] // 2] = 0
    return (x, y, pixels.shape[0], pixels.shape[1], pixels.tobytes())


def _reconstruct(records, state):
    actions = []
    while records[state][0] is not None:
        previous, action, _ = records[state]
        actions.append((action, None, None))
        state = previous
    actions.reverse()
    return tuple(actions)


def _enumerate_shape(env, index, requirements, cap):
    module = upstream()
    template = env.movables()[index].clone()
    normalize_unselected(template, module)
    template.pixels[template.height // 2, template.width // 2] = 0
    root = _state(template)
    queue = deque([root])
    records = {root: (None, None, 0)}
    signatures = {}
    expanded = 0
    obstacles = tuple(_static_sprite(sprite) for sprite in
                      env.level.get_sprites_by_tag(names.TAG_OBSTACLE))
    dyes = tuple(_static_sprite(sprite) for sprite in
                 env.level.get_sprites_by_tag(names.TAG_DYE))
    flexible = names.TAG_FLEXIBLE in template.tags
    fixed_center = names.TAG_FIXED_CENTER in template.tags
    while queue and expanded < cap:
        state = queue.popleft()
        expanded += 1
        sprite = _sprite_from_state(template, state)
        selected = _signature(sprite, requirements, selected=True, module=module)
        stable = _signature(sprite, requirements, selected=False, module=module)
        key = (selected, stable)
        if key not in signatures:
            signatures[key] = _Candidate(
                sprite, _reconstruct(records, state), selected, stable
            )
        depth = records[state][2]
        if depth >= env.steps_left:
            continue
        for action_id in names.MOVE_ACTIONS:
            child = _pure_move(
                state, action_id, flexible=flexible,
                fixed_center=fixed_center, obstacles=obstacles, dyes=dyes,
            )
            if child in records:
                continue
            records[child] = (state, action_id, depth + 1)
            queue.append(child)
    return tuple(signatures.values()), expanded, len(records), bool(queue)


def _requirements(env):
    targets = env.targets()
    if len(targets) != 1:
        return None, "full search requires exactly one target sprite"
    target = targets[0]
    if (target.x, target.y, target.width, target.height) != (
            0, 0, names.FRAME_SIZE, names.FRAME_SIZE):
        return None, "target must be an unshifted 64x64 native mask"
    result = []
    mask = ((target.pixels != names.TRANSPARENT)
            & (target.pixels != names.TARGET_GUIDE))
    for row, col in np.argwhere(mask):
        result.append((int(row), int(col), int(target.pixels[row, col])))
    if not result:
        return None, "target has no coloured requirements"
    return tuple(result), None


def _combination_matches(candidates, active, requirements, module):
    sprites = []
    for index, candidate in enumerate(candidates):
        sprite = candidate.sprite.clone()
        if index != active:
            normalize_unselected(sprite, module)
        sprites.append(sprite)
    for row, col, wanted in requirements:
        actual = names.TRANSPARENT
        for sprite in sprites:
            value = _value_at(sprite, row, col)
            if value != names.TRANSPARENT:
                actual = value
        if actual != wanted:
            return False
    return True


def _switches(current, target, count):
    return ((names.ACTION_NEXT, None, None),) * ((target - current) % count)


def _verified_prefix(env, actions):
    probe = env.clone()
    before = probe.levels_completed
    prefix = []
    for action in actions:
        prefix.append(action)
        observation = probe.perform(*action)
        if probe.levels_completed > before or observation.state == GameState.WIN:
            return tuple(prefix)
        if observation.state == GameState.GAME_OVER:
            return None
    return None


def search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT, budget=None):
    """Return one bounded full-mechanics native witness when found."""
    limit = _positive_integer(limit, "limit", optional=True)
    budget = _positive_integer(budget, "budget", optional=True)
    node_limit = _positive_integer(node_limit, "node_limit")
    env = getattr(env_or_layout, "snapshot", env_or_layout)
    if not isinstance(env, Env):
        raise TypeError("search expects an Env or extracted Layout")
    if env.state != GameState.NOT_FINISHED:
        result = SearchResult(None, False, True, False, 0, 0,
                              f"terminal state {env.state.value}", node_limit)
        search.result = result
        return result
    if not env.stable():
        result = SearchResult(None, False, True, False, 0, 0,
                              "native dye animation is still pending", node_limit)
        search.result = result
        return result
    requirements, error = _requirements(env)
    if error:
        result = SearchResult(None, False, True, False, 0, 0, error, node_limit)
        search.result = result
        return result
    movables = env.movables()
    active = env.selected_index()
    if not movables or active is None:
        result = SearchResult(
            None, False, True, False, 0, 0,
            "full search requires movables and exactly one selected centre",
            node_limit,
        )
        search.result = result
        return result
    cap = max(1, node_limit // len(movables))
    per_shape = []
    expanded = generated = 0
    any_truncated = False
    for index in range(len(movables)):
        candidates, work, states, truncated = _enumerate_shape(
            env, index, requirements, cap
        )
        expanded += work
        generated += states
        any_truncated = any_truncated or truncated
        per_shape.append(candidates)

    module = upstream()
    join_work = 0
    action_cutoff = False
    action_cap = min(
        value for value in (limit, budget, env.steps_left) if value is not None
    )
    for final_active in range(len(movables)):
        useful = []
        for index, candidates in enumerate(per_shape):
            ranked = []
            for candidate in candidates:
                signature = (candidate.selected_signature
                             if index == final_active
                             else candidate.stable_signature)
                matches = sum(
                    value == wanted
                    for value, (_, _, wanted) in zip(signature, requirements)
                )
                if matches:
                    ranked.append((-matches, len(candidate.path), candidate))
            ranked.sort(key=lambda item: (item[0], item[1]))
            useful.append(tuple(item[2] for item in ranked))
        if any(not values for values in useful):
            continue
        for combination in itertools.product(*useful):
            join_work += 1
            if expanded + join_work > node_limit:
                any_truncated = True
                break
            if not _combination_matches(
                    combination, final_active, requirements, module):
                continue
            others = tuple(index for index in range(len(movables))
                           if index != final_active)
            for prefix in itertools.permutations(others):
                order = prefix + (final_active,)
                route = []
                selected = active
                for index in order:
                    route.extend(_switches(selected, index, len(movables)))
                    selected = index
                    route.extend(combination[index].path)
                if not route:
                    continue
                if len(route) > action_cap:
                    action_cutoff = True
                    continue
                verified = _verified_prefix(env, route)
                if verified is None:
                    continue
                result = SearchResult(
                    verified, False, False, True, expanded + join_work,
                    generated,
                    "positive factored state search; real-engine replay verified; "
                    "route is not claimed optimal",
                    node_limit,
                    False,
                )
                search.result = result
                return result
        if expanded + join_work > node_limit:
            break
    truncated = bool(
        any_truncated or action_cutoff or expanded + join_work >= node_limit
    )
    reason = (
        "search cutoff reached; solvability remains unknown"
        if truncated
        else "complete factored state space has no joined target assignment"
    )
    result = SearchResult(None, truncated, False, not truncated,
                          expanded + join_work, generated, reason, node_limit)
    search.result = result
    return result


search.result = None


def solve(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT, budget=None):
    """Compatibility wrapper returning action triples or ``None``."""
    result = search(env_or_layout, limit=limit, node_limit=node_limit, budget=budget)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    return list(result.actions) if result.actions is not None else None


solve.result = None
solve.truncated = False
solve.unsupported = False
solve.exact = False


def solution_mechanics(env, actions):
    """Replay and return route-derived mechanic participation facts."""
    probe = env.clone()
    before_score = probe.levels_completed
    facts = {
        "move_actions": 0,
        "selection_actions": 0,
        "distinct_selected": [],
        "dye_events": 0,
        "dye_animation_frames": 0,
        "resize_events": 0,
        "deformation_events": 0,
        "blocked_moves": 0,
        "fixed_center_selections": 0,
        "flexible_selections": 0,
        "target_constrained_selection_centers": 0,
        "same_color_shape_pairs": 0,
        "ambiguous_target_assignment": False,
    }
    selected_seen = set()
    requirements = _requirements(probe)[0] or ()
    initial_colors = [int(getattr(probe.module, "euqngakkse")(sprite))
                      for sprite in probe.movables()]
    facts["same_color_shape_pairs"] = sum(
        left == right for index, left in enumerate(initial_colors)
        for right in initial_colors[index + 1:]
    )
    facts["ambiguous_target_assignment"] = facts["same_color_shape_pairs"] > 0
    for action_id, x, y in actions:
        movables = probe.movables()
        selected = probe.selected_index()
        if selected is not None:
            selected_seen.add(selected)
        before = [
            (_sprite_fingerprint(sprite),
             int(getattr(probe.module, "euqngakkse")(sprite)))
            for sprite in movables
        ]
        before_positions = [(sprite.x, sprite.y) for sprite in movables]
        if action_id == names.ACTION_NEXT and selected is not None:
            stable = normalize_unselected(movables[selected].clone(), probe.module)
            center = (stable.y + stable.height // 2,
                      stable.x + stable.width // 2)
            value = int(stable.pixels[stable.height // 2, stable.width // 2])
            if value > 0 and any(
                    (row, col, wanted) == (*center, value)
                    for row, col, wanted in requirements):
                facts["target_constrained_selection_centers"] += 1
        observation = probe.perform(action_id, x, y)
        after = probe.movables() if probe.levels_completed == before_score else []
        if action_id == names.ACTION_NEXT:
            facts["selection_actions"] += 1
            if after:
                current = probe.selected_index()
                if current is not None and names.TAG_FIXED_CENTER in after[current].tags:
                    facts["fixed_center_selections"] += 1
                if current is not None and names.TAG_FLEXIBLE in after[current].tags:
                    facts["flexible_selections"] += 1
        else:
            facts["move_actions"] += 1
        if after and selected is not None and action_id in names.MOVE_ACTIONS:
            sprite = after[selected]
            old_fp, old_color = before[selected]
            new_color = int(getattr(probe.module, "euqngakkse")(sprite))
            if new_color != old_color:
                facts["dye_events"] += 1
                facts["dye_animation_frames"] += max(1, len(observation.frames) - 1)
            old_height, old_width = old_fp[3], old_fp[4]
            if (sprite.height, sprite.width) != (old_height, old_width):
                facts["resize_events"] += 1
            elif sprite.pixels.tobytes() != old_fp[5] and new_color == old_color:
                facts["deformation_events"] += 1
            if (sprite.x, sprite.y) == before_positions[selected]:
                facts["blocked_moves"] += 1
        if probe.levels_completed > before_score or observation.state == GameState.WIN:
            break
    facts["distinct_selected"] = sorted(selected_seen)
    facts["engine_win"] = probe.levels_completed > before_score
    facts["actions_replayed"] = (
        facts["move_actions"] + facts["selection_actions"]
    )
    return facts
