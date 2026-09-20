"""Bounded exact compact-state search for native VC33 click puzzles."""

from dataclasses import dataclass
import heapq
from itertools import count
import math

from . import names
from .layout import Layout, extract


DEFAULT_NODE_LIMIT = 100_000
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

    @property
    def solved(self):
        return self.actions is not None


def _selection_clicks(env, cache=None):
    """Classify display pixels into each click effect plus one no-op class."""
    cache = {} if cache is None else cache
    # Clickable sprites never move in a completed VC33 action.  Swap-bar
    # animations run to completion inside ``perform`` and return the bar to its
    # original cell; support/load changes therefore cannot alter the display
    # representatives.  Key only the click geometry instead of the entire
    # puzzle state so the 64x64 classification is performed once per search.
    semantic = (
        tuple(env.level.grid_size),
        tuple(
            (
                sprite.name,
                int(sprite.x),
                int(sprite.y),
                int(sprite.width),
                int(sprite.height),
                int(sprite.layer),
                bool(sprite.is_visible),
            )
            for sprite in env.level.get_sprites()
            if names.TAG_CLICK in sprite.tags
        ),
    )
    cached = cache.get(semantic)
    if cached is not None:
        yield from cached
        return
    sprite_indices = {id(sprite): index
                      for index, sprite in enumerate(env.level.get_sprites())}
    representatives = {}
    for y in range(names.FRAME_SIZE):
        for x in range(names.FRAME_SIZE):
            point = env.game.camera.display_to_grid(x, y)
            if point is None:
                continue
            hit = env.level.get_sprite_at(*point)
            index = (
                sprite_indices[id(hit)]
                if hit is not None and names.TAG_CLICK in hit.tags
                else -1
            )
            representatives.setdefault(index, (x, y))
    result = tuple(sorted(representatives.items()))
    cache[semantic] = result
    yield from result


@dataclass(frozen=True)
class _Button:
    donor: int
    receiver: int
    action: Action


@dataclass(frozen=True)
class _Swap:
    left: tuple[int, ...]
    right: tuple[int, ...]
    edge: int
    action: Action


@dataclass(frozen=True)
class _CompactModel:
    """Exact stable-action model of VC33's support/load state.

    Static walls, floors, targets and click geometry never change.  A completed
    native action can only resize two supports, move their carried loads, or
    exchange the loads of two equal-height supports.  The obfuscated engine's
    animation is presentation of that atomic exchange; positive certificates
    are still replayed by the real engine.
    """

    gravity: int
    positive: bool
    supports: tuple[object, ...]
    anchors: tuple[int, ...]
    empty_limits: tuple[int, ...]
    occupied_limits: tuple[int, ...]
    load_sizes: tuple[int, ...]
    goals: tuple[tuple[tuple[int, int], ...], ...]
    buttons: tuple[_Button, ...]
    swaps: tuple[_Swap, ...]

    def load_primary(self, edge, load):
        return edge - self.load_sizes[load] if self.positive else edge

    def solved(self, state):
        edges, assignments = state
        return all(
            (assignments[load], self.load_primary(edges[assignments[load]], load))
            in self.goals[load]
            for load in range(len(assignments))
        )

    def heuristic(self, state):
        edges, assignments = state
        lower_bound = 0
        magnitude = abs(self.gravity)
        for load, support in enumerate(assignments):
            primary = self.load_primary(edges[support], load)
            distance = min(
                (abs(primary - goal_primary) for _, goal_primary in self.goals[load]),
                default=0,
            )
            lower_bound = max(lower_bound, math.ceil(distance / magnitude))
        return lower_bound

    def transition_button(self, state, button):
        edges, assignments = state
        donor_edge = edges[button.donor]
        receiver_edge = edges[button.receiver]
        donor_thickness = (
            self.anchors[button.donor] - donor_edge
            if self.positive
            else donor_edge - self.anchors[button.donor]
        )
        occupied = button.receiver in assignments
        limit = (
            self.occupied_limits[button.receiver]
            if occupied
            else self.empty_limits[button.receiver]
        )
        receiver_can_grow = (
            receiver_edge > limit if self.positive else receiver_edge < limit
        )
        if donor_thickness <= 0 or not receiver_can_grow:
            return None
        changed = list(edges)
        changed[button.donor] += self.gravity
        changed[button.receiver] -= self.gravity
        return tuple(changed), assignments

    def transition_swap(self, state, swap):
        edges, assignments = state
        left = next((support for support in swap.left
                     if edges[support] == swap.edge), None)
        right = next((support for support in swap.right
                      if edges[support] == swap.edge), None)
        if left is None or right is None:
            return None
        changed = tuple(
            right if support == left
            else left if support == right
            else support
            for support in assignments
        )
        if changed == assignments:
            return None
        return edges, changed


def _primary_size(env, sprite):
    return int(env.game.pjfzvvjgud(sprite))


def _compact_model(env):
    """Extract the complete stable VC33 transition system from a native state."""
    game = env.game
    gravity = int(env.gravity[0] or env.gravity[1])
    positive = game.qhmwbtpcsk()
    sprites = env.level.get_sprites()
    supports = tuple(env.level.get_sprites_by_tag(names.TAG_SUPPORT))
    support_index = {id(sprite): index for index, sprite in enumerate(supports)}
    loads = tuple(env.level.get_sprites_by_tag(names.TAG_LOAD))
    walls = tuple(env.level.get_sprites_by_tag(names.TAG_WALL))
    targets = tuple(env.level.get_sprites_by_tag(names.TAG_TARGET))

    edges = tuple(int(game.hpakcxndwy(sprite)) for sprite in supports)
    anchors = tuple(
        int(game.xitrlzpbgu(sprite) + _primary_size(env, sprite))
        if positive else int(game.xitrlzpbgu(sprite))
        for sprite in supports
    )
    assignments = []
    for load in loads:
        resting = [index for index, support in enumerate(supports)
                   if game.bcpuwqzpxw(load, support)]
        if len(resting) != 1:
            raise ValueError("each load must rest on exactly one support")
        assignments.append(resting[0])

    goals = []
    for load in loads:
        color = int(load.pixels[-1, -1])
        accepted = set()
        for target in targets:
            if color not in target.pixels:
                continue
            target_walls = [wall for wall in walls if wall.collides_with(target)]
            if not target_walls:
                continue
            wall = target_walls[0]
            for index, support in enumerate(supports):
                if wall in game.rcbyiqlbza(support):
                    accepted.add((index, int(game.xitrlzpbgu(target))))
        if not accepted:
            raise ValueError("a load has no reachable same-colour target/support pair")
        goals.append(tuple(sorted(accepted)))

    empty_limits = []
    occupied_limits = []
    for support in supports:
        floors = game.kectayqmfn(support)
        if floors:
            floor_edges = [int(game.zfcrfmorna(floor)) for floor in floors]
            empty_limits.append(max(floor_edges))
            occupied_offset = 6 if env.gravity[0] == -3 else 4
            occupied_limits.append(max(edge - occupied_offset for edge in floor_edges))
        else:
            adjacent = game.rcbyiqlbza(support)
            if not adjacent:
                raise ValueError("each support needs a wall or floor transfer boundary")
            boundary = (
                max(int(game.hpakcxndwy(wall)) for wall in adjacent)
                if positive else
                min(int(game.hpakcxndwy(wall)) for wall in adjacent)
            )
            empty_limits.append(boundary)
            occupied_limits.append(boundary)

    representatives = dict(_selection_clicks(env))
    buttons = []
    for button, pair in getattr(game, names.ATTR_BUTTON_PAIRS).items():
        sprite_i = sprites.index(button)
        if sprite_i not in representatives:
            raise ValueError("a balance button has no display click representative")
        buttons.append(_Button(
            support_index[id(pair[0])],
            support_index[id(pair[1])],
            (names.ACTION_CLICK, *representatives[sprite_i]),
        ))

    swaps = []
    for bar in env.level.get_sprites_by_tag(names.TAG_SWAP):
        sprite_i = sprites.index(bar)
        if sprite_i not in representatives:
            raise ValueError("a swap bar has no display click representative")
        before = [
            index for index, support in enumerate(supports)
            if game.mzqwlqlkrv(support) + game.vmjlfzxesj(support)
            == game.mzqwlqlkrv(bar)
        ]
        after = [
            index for index, support in enumerate(supports)
            if game.mzqwlqlkrv(support)
            == game.mzqwlqlkrv(bar) + game.vmjlfzxesj(bar)
        ]
        if not before or not after:
            raise ValueError("each swap bar must bridge supports on both sides")
        swaps.append(_Swap(
            tuple(before), tuple(after), int(game.zfcrfmorna(bar)),
            (names.ACTION_CLICK, *representatives[sprite_i]),
        ))

    return (
        _CompactModel(
            gravity=gravity,
            positive=positive,
            supports=supports,
            anchors=anchors,
            empty_limits=tuple(empty_limits),
            occupied_limits=tuple(occupied_limits),
            load_sizes=tuple(_primary_size(env, load) for load in loads),
            goals=tuple(goals),
            buttons=tuple(buttons),
            swaps=tuple(swaps),
        ),
        (edges, tuple(assignments)),
    )


def search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Search with independent action and configuration-expansion bounds."""
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

    start_env = layout.snapshot.clone()
    try:
        model, start = _compact_model(start_env)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return SearchResult(
            None, False, True, False, 0, 0,
            f"unsupported compact state: {exc}", node_limit,
        )
    parent = {start: None}
    best = {start: 0}
    serial = count()
    frontier = [(model.heuristic(start), 0, next(serial), start)]
    expanded = generated = 0
    action_cutoff = False

    while frontier:
        _, cost, _, state = heapq.heappop(frontier)
        if cost != best.get(state):
            continue
        if expanded >= node_limit:
            return SearchResult(
                None, True, False, True, expanded, generated,
                f"configuration expansion limit {node_limit} reached", node_limit,
            )
        expanded += 1
        if model.solved(state):
            actions = []
            cursor = state
            while parent[cursor] is not None:
                cursor, action = parent[cursor]
                actions.append(action)
            actions.reverse()
            return SearchResult(
                tuple(actions), False, False, True, expanded, generated,
                "compact exact state reached the native win predicate", node_limit,
            )
        if limit is not None and cost >= limit:
            action_cutoff = True
            continue

        next_cost = cost + 1
        transitions = (
            ((button.action, model.transition_button(state, button))
             for button in model.buttons),
            ((swap.action, model.transition_swap(state, swap))
             for swap in model.swaps),
        )
        for family in transitions:
            for action, successor in family:
                generated += 1
                if successor is None or next_cost >= best.get(successor, 1 << 30):
                    continue
                best[successor] = next_cost
                parent[successor] = (state, action)
                heapq.heappush(
                    frontier,
                    (next_cost + model.heuristic(successor), next_cost,
                     next(serial), successor),
                )

    if action_cutoff:
        return SearchResult(
            None, True, False, True, expanded, generated,
            f"action limit {limit} reached before the exact graph was exhausted", node_limit,
        )
    return SearchResult(
        None, False, False, True, expanded, generated,
        "no solution exists within the remaining native click budget", node_limit,
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
