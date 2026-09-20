"""Bounded full-mechanics native teacher for DC22.

Generated rows carry a constructive route, but it is never trusted blindly:
the teacher matches the live native state to a replayed route prefix and
replays the proposed suffix on a clone. Arbitrary/off-route states use a
bounded native A* fallback over every visible click control. Neither backend
claims shortest-route optimality.
"""

from dataclasses import dataclass
import heapq
from itertools import count

from arcengine import GameState

from . import names
from .layout import Layout, extract


DEFAULT_NODE_LIMIT = 100_000
Action = tuple[int, int | None, int | None]
_CLICK_CACHE = {}


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...] | None
    status: str
    truncated: bool
    unsupported: bool
    exact: bool
    expanded: int
    generated: int
    reason: str
    node_limit: int
    optimal: bool = False
    backend: str = "native-a-star"

    @property
    def solved(self):
        return self.actions is not None


def _sprite_state(env):
    return tuple(
        (
            sprite.name,
            int(sprite.x),
            int(sprite.y),
            int(sprite.interaction.value),
            bool(sprite.is_visible),
            bool(sprite.is_collidable),
            tuple(sprite.tags),
        )
        for sprite in env.level.get_sprites()
        if (
            sprite.name == names.SPRITE_PLAYER
            or sprite.tags
            or sprite.interaction.name != "TANGIBLE"
        )
    )


def _semantic_key(env):
    game = env.game
    return (
        _sprite_state(env),
        int(getattr(game, "sjixewahg", 0)),
        int(getattr(game, "uxtzlxsiq", 0)),
        str(getattr(game, "svxnnbpjl", "none")),
        str(getattr(game, "fvwekbbhj", "")),
        int(getattr(game, "ozarnpwde", 0)),
        int(getattr(game, "bbobkhxob", 0)),
    )


def _state_key(env):
    return _semantic_key(env) + (int(env.steps_left),)


def _button_clicks(env):
    """One full-display coordinate for every currently visible sys-click sprite."""
    buttons = tuple(
        (
            sprite.name,
            int(sprite.x),
            int(sprite.y),
            int(sprite.interaction.value),
            bool(sprite.is_visible),
            tuple(sprite.tags),
        )
        for sprite in env.level.get_sprites_by_tag(names.TAG_CLICK)
    )
    geometry = (tuple(env.level.grid_size), buttons)
    cached = _CLICK_CACHE.get(geometry)
    if cached is not None:
        return cached
    hit_method = getattr(env.game, names.METHOD_HIT_VISIBLE)
    representatives = {}
    for y in range(names.FRAME_SIZE):
        for x in range(names.FRAME_SIZE):
            point = env.game.camera.display_to_grid(x, y)
            if point is None:
                continue
            hit = hit_method(*point, names.TAG_CLICK)
            if hit is None:
                continue
            signature = (hit.name, int(hit.x), int(hit.y), tuple(hit.tags))
            representatives.setdefault(signature, (x, y))
    result = tuple(representatives[key] for key in sorted(representatives))
    _CLICK_CACHE[geometry] = result
    return result


def _heuristic(env):
    return (
        abs(int(env.player.x) - int(env.goal.x))
        + abs(int(env.player.y) - int(env.goal.y))
    ) // names.MOVE_STEP


def _unwind(parent, state, tail):
    actions = [tail]
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return tuple(actions)


def _positive_int(value, label, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{label} must be a positive integer{suffix}")
    return int(value)


def _replay_wins(env, actions):
    clone = env.clone()
    before = clone.levels_completed
    for action in actions:
        observation = clone.perform(*action)
        if clone.levels_completed > before or observation.state == GameState.WIN:
            return True
        if observation.state == GameState.GAME_OVER:
            return False
    return False


def _constructive_suffix(env, action_bound):
    """Return a verified suffix when the live state is on a generated route."""
    spec = env.level.get_data(names.KEY_GENERATED_SPEC)
    if not isinstance(spec, dict):
        return None
    raw = spec.get("solution") or spec.get("constructed_solution")
    if not isinstance(raw, list):
        return None
    try:
        route = tuple((int(a), x, y) for a, x, y in raw)
    except (TypeError, ValueError):
        return None
    from .generate import _context_env, build_level

    context = spec.get("context_index")
    if isinstance(context, bool) or not isinstance(context, int) or context < 0:
        return None
    pristine = _context_env(build_level(spec), context)
    target = _state_key(env)
    for prefix in range(len(route) + 1):
        if _state_key(pristine) == target:
            suffix = route[prefix:]
            if (action_bound is None or len(suffix) <= action_bound) and _replay_wins(env, suffix):
                return suffix
        if prefix < len(route):
            observation = pristine.perform(*route[prefix])
            if observation.state == GameState.GAME_OVER or pristine.levels_completed:
                break
    return None


def search(env_or_layout, limit=None, budget=None, node_limit=DEFAULT_NODE_LIMIT):
    """Search the actual live native state with explicit action/work bounds."""
    limit = _positive_int(limit, "limit", allow_none=True)
    budget = _positive_int(budget, "budget", allow_none=True)
    node_limit = _positive_int(node_limit, "node_limit")
    bounds = [value for value in (limit, budget) if value is not None]
    action_bound = min(bounds) if bounds else None
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(
            None, "unsupported", False, True, False, 0, 0,
            "; ".join(layout.unsupported) or "unsupported live state", node_limit,
            False, "native-a-star",
        )

    start_env = layout.snapshot.clone()
    suffix = _constructive_suffix(start_env, action_bound)
    if suffix is not None:
        return SearchResult(
            suffix, "solved", False, False, True, 0, 0,
            f"native-replayed {len(suffix)}-action suffix of the procedural constructive witness; no search nodes expanded",
            node_limit, False, "constructive-native-replay",
        )

    # Official layouts do not carry generated hints. Their coupled surface and
    # crusher phases are far smaller in the compact settled state model than as
    # deep-copied engine objects. A route is accepted only after native replay.
    from .symbolic import official_search
    actions, symbolic_expanded, symbolic_generated, symbolic_truncated = official_search(
        start_env, node_limit,
    )
    if actions is not None and (action_bound is None or len(actions) <= action_bound):
        if _replay_wins(start_env, actions):
            return SearchResult(
                actions, "solved", False, False, True,
                symbolic_expanded, symbolic_generated,
                "compact settled-state witness replayed in the native engine; optimality not claimed",
                node_limit, False, "symbolic-plus-native-replay",
            )
    if symbolic_truncated:
        return SearchResult(
            None, "truncated", True, False, True,
            symbolic_expanded, symbolic_generated,
            f"compact configuration expansion limit {node_limit} reached",
            node_limit, False, "symbolic-plus-native-replay",
        )

    start = _state_key(start_env)
    parent = {start: None}
    best_cost = {start: 0}
    pareto = {_semantic_key(start_env): [(start_env.steps_left, 0)]}
    serial = count()
    # A weighted distance ordering is intentional: this backend seeks a bounded
    # positive witness and never labels the first route optimal.
    frontier = [(10 * _heuristic(start_env), 0, next(serial), start, start_env)]
    expanded = generated = 0
    action_cutoff = False
    while frontier:
        _, cost, _, state, env = heapq.heappop(frontier)
        if cost != best_cost.get(state):
            continue
        if expanded >= node_limit:
            return SearchResult(
                None, "truncated", True, False, True, expanded, generated,
                f"native configuration expansion limit {node_limit} reached",
                node_limit, False, "native-a-star",
            )
        expanded += 1
        if action_bound is not None and cost >= action_bound:
            action_cutoff = True
            continue
        actions = [(action_id, None, None) for action_id in names.MOVE_ACTIONS]
        actions.extend((names.ACTION_CLICK, x, y) for x, y in _button_clicks(env))
        before_semantic = _semantic_key(env)
        next_cost = cost + 1
        for action in actions:
            successor = env.clone()
            score = successor.levels_completed
            observation = successor.perform(*action)
            if successor.levels_completed > score or observation.state == GameState.WIN:
                witness = _unwind(parent, state, action)
                if _replay_wins(start_env, witness):
                    return SearchResult(
                        witness, "solved", False, False, True, expanded, generated,
                        "bounded native-transition witness; optimality not claimed",
                        node_limit, False, "native-a-star",
                    )
            if observation.state == GameState.GAME_OVER or _semantic_key(successor) == before_semantic:
                continue
            generated += 1
            semantic = _semantic_key(successor)
            profiles = pareto.setdefault(semantic, [])
            if any(steps >= successor.steps_left and prior_cost <= next_cost for steps, prior_cost in profiles):
                continue
            nxt = _state_key(successor)
            if next_cost >= best_cost.get(nxt, 1 << 30):
                continue
            profiles[:] = [
                (steps, prior_cost) for steps, prior_cost in profiles
                if not (successor.steps_left >= steps and next_cost <= prior_cost)
            ]
            profiles.append((successor.steps_left, next_cost))
            best_cost[nxt] = next_cost
            parent[nxt] = (state, action)
            heapq.heappush(
                frontier,
                (next_cost + 10 * _heuristic(successor), next_cost, next(serial), nxt, successor),
            )
    if action_cutoff:
        return SearchResult(
            None, "truncated", True, False, True, expanded, generated,
            f"action limit {action_bound} reached before exhaustive native search",
            node_limit, False, "native-a-star",
        )
    return SearchResult(
        None, "unsolved", False, False, True, expanded, generated,
        "native state space exhausted within the remaining step budget",
        node_limit, False, "native-a-star",
    )


def solve(env_or_layout, limit=None, budget=None, node_limit=DEFAULT_NODE_LIMIT):
    result = search(env_or_layout, limit=limit, budget=budget, node_limit=node_limit)
    solve.result = result
    solve.status = result.status
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    solve.reason = result.reason
    return list(result.actions) if result.actions is not None else None


solve.result = None
solve.status = "not-run"
solve.truncated = False
solve.unsupported = False
solve.exact = False
solve.reason = ""
