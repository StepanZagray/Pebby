"""Exact logical transition model and bounded teacher for WA30.

The native game is deterministic. A player action is followed, in sprite
order, by every helper and then every live thief. Robot target sets are
intentionally cached and refreshed at different moments; those caches and box
holder identities are consequently part of the search state. The transition
below mirrors ``third_party/arc3_games/wa30.py:934-1248`` without cloning an
engine for every edge. Positive plans are still replayed by :mod:`env`.

For manual-only levels A* uses an admissible lower bound and returns a shortest
route. Autonomous actors can move several boxes during one player action, so
their heuristic is deliberately only an ordering hint and no optimality claim
is made. A cutoff is reported as ``truncated`` and is never called impossible.
"""

from __future__ import annotations

from collections import Counter, deque
import heapq
from typing import NamedTuple

from . import names
from .layout import Layout, extract


DEAD = (-1, -1)
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # native BFS order


class Unplannable(Exception):
    """The layout cannot be represented exactly."""


class State(NamedTuple):
    player: tuple[int, int]
    rotation: int
    boxes: tuple[tuple[int, int], ...]       # native sprite order
    helpers: tuple[tuple[int, int], ...]     # native sprite order
    thieves: tuple[tuple[int, int], ...]     # DEAD after player destroys one
    holds: tuple[int, ...]                   # player, helpers, thieves -> box index/-1
    helper_targets: tuple[tuple[int, int], ...]
    thief_targets: tuple[tuple[int, int], ...]


class SearchResult:
    """A bounded planner result.

    ``work`` counts every call to :func:`transition` made while planning.  It
    deliberately excludes the independent native admission replay performed
    after a witness is found.  ``expanded`` remains as a compatibility alias
    for older audit code, but is no longer described as a state count.
    """

    __slots__ = ("actions", "truncated", "work", "reason", "backend")

    def __init__(self, actions, truncated, work, reason, backend):
        self.actions = actions
        self.truncated = truncated
        self.work = work
        self.reason = reason
        self.backend = backend

    @property
    def expanded(self):
        return self.work

    @property
    def solved(self):
        return self.actions is not None


def state_of(layout: Layout) -> State:
    state = State(
        layout.player,
        layout.rotation,
        tuple(layout.boxes),
        tuple(layout.helpers),
        tuple(layout.thieves),
        tuple(layout.holds),
        tuple(layout.helper_targets),
        tuple(layout.thief_targets),
    )
    return _canonical_manual(state) if not state.helpers and not state.thieves else state


def _canonical_manual(state: State) -> State:
    """Boxes are interchangeable when no autonomous actor can select by order."""
    if state.helpers or state.thieves:
        return state
    held_cell = state.boxes[state.holds[0]] if state.holds and state.holds[0] >= 0 else None
    boxes = tuple(sorted(state.boxes))
    holds = (-1 if held_cell is None else boxes.index(held_cell),)
    return state._replace(boxes=boxes, holds=holds)


def _holder_of(state: State) -> list[int]:
    inverse = [-1] * len(state.boxes)
    for actor, box in enumerate(state.holds):
        if box >= 0:
            inverse[box] = actor
    return inverse


def _occupancy(layout: Layout, state: State) -> set[tuple[int, int]]:
    return set(layout.walls) | {state.player, *state.boxes, *state.helpers} | {
        thief for thief in state.thieves if thief != DEAD
    }


def _free(layout: Layout, state: State, cell: tuple[int, int]) -> bool:
    return layout.in_bounds(cell) and cell not in _occupancy(layout, state) and cell not in layout.fences


def _actor_position(state: State, actor: int) -> tuple[int, int]:
    if actor == 0:
        return state.player
    actor -= 1
    if actor < len(state.helpers):
        return state.helpers[actor]
    return state.thieves[actor - len(state.helpers)]


def _replace_actor(state: State, actor: int, cell: tuple[int, int]) -> State:
    if actor == 0:
        return state._replace(player=cell)
    actor -= 1
    if actor < len(state.helpers):
        values = list(state.helpers)
        values[actor] = cell
        return state._replace(helpers=tuple(values))
    values = list(state.thieves)
    values[actor - len(state.helpers)] = cell
    return state._replace(thieves=tuple(values))


def _pair_can_move(layout: Layout, state: State, actor: int, target: tuple[int, int]) -> bool:
    actor_pos = _actor_position(state, actor)
    box_index = state.holds[actor]
    box_pos = state.boxes[box_index]
    offset = (box_pos[0] - actor_pos[0], box_pos[1] - actor_pos[1])
    box_target = (target[0] + offset[0], target[1] + offset[1])
    occupied = _occupancy(layout, state)
    actor_ok = (
        layout.in_bounds(target)
        and (target not in occupied or target == box_pos)
        and target not in layout.fences
    )
    box_ok = layout.in_bounds(box_target) and (box_target not in occupied or box_target == actor_pos)
    return actor_ok and box_ok


def _targets(layout: Layout, state: State, *, helpers: bool) -> tuple[tuple[int, int], ...]:
    holder = _holder_of(state)
    targets = set()
    first_thief_actor = 1 + len(state.helpers)
    for index, box in enumerate(state.boxes):
        if helpers:
            eligible = holder[index] < 0 and box not in layout.goals
        else:
            eligible = not (holder[index] >= first_thief_actor) and box not in layout.bad
        if eligible:
            for dx, dy in NEIGHBORS:
                targets.add((box[0] + dx, box[1] + dy))
    return tuple(sorted(targets))


def _refresh_targets(layout: Layout, state: State) -> State:
    return state._replace(
        helper_targets=_targets(layout, state, helpers=True),
        thief_targets=_targets(layout, state, helpers=False),
    )


def _detach(layout: Layout, state: State, actor: int) -> State:
    if state.holds[actor] < 0:
        return state
    holds = list(state.holds)
    holds[actor] = -1
    return _refresh_targets(layout, state._replace(holds=tuple(holds)))


def _attach(layout: Layout, state: State, actor: int, box: int) -> tuple[State, bool]:
    holder = _holder_of(state)[box]
    stolen = holder >= 0 and holder != actor
    if holder >= 0:
        state = _detach(layout, state, holder)
    holds = list(state.holds)
    holds[actor] = box
    return _refresh_targets(layout, state._replace(holds=tuple(holds))), stolen


def _move_actor(layout: Layout, state: State, actor: int, target: tuple[int, int]) -> State:
    box = state.holds[actor]
    if box >= 0:
        if not _pair_can_move(layout, state, actor, target):
            return state
        old_actor = _actor_position(state, actor)
        old_box = state.boxes[box]
        offset = (old_box[0] - old_actor[0], old_box[1] - old_actor[1])
        boxes = list(state.boxes)
        boxes[box] = (target[0] + offset[0], target[1] + offset[1])
        moved = _replace_actor(state._replace(boxes=tuple(boxes)), actor, target)
        # Native pair movement only refreshes thief targets.
        return moved._replace(thief_targets=_targets(layout, moved, helpers=False))
    if _free(layout, state, target):
        return _replace_actor(state, actor, target)
    return state


def _path(layout: Layout, state: State, actor: int, targets, *, carrying: bool) -> list[tuple[int, int]] | None:
    start = _actor_position(state, actor)
    seen = {start}
    queue = deque([[start]])
    targets = set(targets)
    while queue:
        path = queue.popleft()
        cell = path[-1]
        if cell in targets:
            return path
        for dx, dy in NEIGHBORS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen:
                continue
            # The native BFS queries the live obstacle set for every candidate;
            # it does not move a hypothetical actor while exploring the path.
            # This is unusual (and lets a search path revisit the actor's
            # origin only through the explicit pair exceptions), but is part of
            # the shipped semantics.
            allowed = _pair_can_move(layout, state, actor, nxt) if carrying else _free(layout, state, nxt)
            if allowed:
                seen.add(nxt)
                queue.append(path + [nxt])
    return None


def _robots_act(layout: Layout, state: State, *, helpers: bool, events: dict | None) -> State:
    positions = state.helpers if helpers else state.thieves
    offset = 1 if helpers else 1 + len(state.helpers)
    for number in range(len(positions)):
        actor = offset + number
        if _actor_position(state, actor) == DEAD:
            continue
        box = state.holds[actor]
        if box >= 0:
            destination = layout.goals if helpers else layout.bad
            if state.boxes[box] in destination:
                state = _detach(layout, state, actor)
                if events is not None:
                    key = "helper_deliveries" if helpers else "thief_bad_deliveries"
                    events[key] = events.get(key, 0) + 1
                continue
            actor_pos = _actor_position(state, actor)
            box_pos = state.boxes[box]
            dx, dy = box_pos[0] - actor_pos[0], box_pos[1] - actor_pos[1]
            target_cells = [(x - dx, y - dy) for x, y in destination]
            path = _path(layout, state, actor, target_cells, carrying=True)
            if path and len(path) > 1:
                before_box = state.boxes[box]
                state = _move_actor(layout, state, actor, path[1])
                if events is not None and state.boxes[box] != before_box:
                    key = "helper_box_moves" if helpers else "thief_box_moves"
                    events[key] = events.get(key, 0) + 1
            continue

        holder = _holder_of(state)
        for box, box_pos in enumerate(state.boxes):
            eligible = (holder[box] < 0 and box_pos not in layout.goals) if helpers else (
                not (holder[box] >= 1 + len(state.helpers)) and box_pos not in layout.bad
            )
            actor_pos = _actor_position(state, actor)
            if eligible and abs(actor_pos[0] - box_pos[0]) + abs(actor_pos[1] - box_pos[1]) == 1:
                previous_holder = holder[box]
                state, stolen = _attach(layout, state, actor, box)
                if events is not None:
                    key = "helper_grabs" if helpers else "thief_grabs"
                    events[key] = events.get(key, 0) + 1
                    if stolen:
                        events["robot_steals"] = events.get("robot_steals", 0) + 1
                        key = "helper_steals" if helpers else "thief_steals"
                        events[key] = events.get(key, 0) + 1
                        if not helpers and 1 <= previous_holder < 1 + len(state.helpers):
                            events["thief_steals_from_helper"] = events.get("thief_steals_from_helper", 0) + 1
                return state  # native returns from the whole actor phase
        targets = state.helper_targets if helpers else state.thief_targets
        path = _path(layout, state, actor, targets, carrying=False)
        if path and len(path) > 1:
            state = _move_actor(layout, state, actor, path[1])
            if events is not None:
                key = "helper_moves" if helpers else "thief_moves"
                events[key] = events.get(key, 0) + 1
    return state


def transition(layout: Layout, state: State, action: int, events: dict | None = None) -> State:
    """Apply one player action plus the ordered helper/thief phases."""
    if action not in names.ACTION_IDS:
        raise ValueError(f"action must be one of {names.ACTION_IDS}")
    if action == names.ACTION_GRAB:
        if state.holds[0] >= 0:
            state = _detach(layout, state, 0)
            if events is not None:
                events["manual_drops"] = events.get("manual_drops", 0) + 1
        else:
            fx, fy = names.ROTATION_FACING[state.rotation]
            ahead = (state.player[0] + fx, state.player[1] + fy)
            for box, cell in enumerate(state.boxes):
                if cell == ahead:
                    state, stolen = _attach(layout, state, 0, box)
                    if events is not None:
                        events["manual_grabs"] = events.get("manual_grabs", 0) + 1
                        if stolen:
                            events["player_steals"] = events.get("player_steals", 0) + 1
                    break
            for number, cell in enumerate(state.thieves):
                if cell == ahead:
                    actor = 1 + len(state.helpers) + number
                    state = _detach(layout, state, actor)
                    thieves = list(state.thieves)
                    thieves[number] = DEAD
                    state = state._replace(thieves=tuple(thieves))
                    if events is not None:
                        events["thieves_destroyed"] = events.get("thieves_destroyed", 0) + 1
                    break
    else:
        dx, dy = names.ACTION_DELTAS[action]
        if state.holds[0] < 0:
            state = state._replace(rotation=names.DIRECTION_ROTATION[(dx, dy)])
        before = state
        state = _move_actor(layout, state, 0, (state.player[0] + dx, state.player[1] + dy))
        if events is not None and state.player != before.player:
            events["player_moves"] = events.get("player_moves", 0) + 1
            held = before.holds[0]
            if held >= 0:
                events["manual_box_moves"] = events.get("manual_box_moves", 0) + 1
                if state.boxes[held] in layout.fences:
                    events["fence_box_entries"] = events.get("fence_box_entries", 0) + 1
    state = _robots_act(layout, state, helpers=True, events=events)
    state = _robots_act(layout, state, helpers=False, events=events)
    return _canonical_manual(state)


def simulate(layout, state, action):
    """Compatibility alias for one exact transition."""
    if not isinstance(state, State):
        player, rotation, held, boxes = state
        holds = (-1,) if held is None else (
            tuple(boxes).index((player[0] + held[0], player[1] + held[1])),
        )
        state = State(player, rotation, tuple(boxes), (), (), holds, (), ())
        nxt = transition(layout, state, action)
        held_box = nxt.holds[0]
        offset = None if held_box < 0 else (
            nxt.boxes[held_box][0] - nxt.player[0], nxt.boxes[held_box][1] - nxt.player[1]
        )
        return nxt.player, nxt.rotation, offset, tuple(sorted(nxt.boxes))
    return transition(layout, state, action)


def complete(layout: Layout, state: State) -> bool:
    return all(box in layout.goals for box in state.boxes) and all(box < 0 for box in state.holds)


def _heuristic(layout: Layout, state: State) -> int:
    goals = layout.goals
    distances = [
        min((abs(x - gx) + abs(y - gy) for gx, gy in goals), default=0)
        for x, y in state.boxes
    ]
    total = sum(distances)
    total += sum(box >= 0 for box in state.holds)
    total += 2 * sum(box in layout.bad for box in state.boxes)
    if not layout.helpers and not layout.thieves and any(distances) and state.holds[0] < 0:
        total += min(
            abs(state.player[0] - x) + abs(state.player[1] - y)
            for x, y in state.boxes
        )
    return total


def _autonomous_score(layout: Layout, state: State):
    """Ordering score for the non-exhaustive autonomous beam.

    Live thieves dominate because they can undo delivery indefinitely. Goal
    occupancy then dominates distance. The remaining tuple is a stable tie
    break, so identical inputs are reproducible across processes.
    """
    off_goal = [box for box in state.boxes if box not in layout.goals]
    open_goals = set(layout.goals) - set(state.boxes)
    distance = sum(
        min((abs(x - gx) + abs(y - gy) for gx, gy in open_goals), default=0)
        for x, y in off_goal
    )
    bad = sum(box in layout.bad for box in state.boxes)
    live = [thief for thief in state.thieves if thief != DEAD]
    player_to_thief = min(
        (abs(state.player[0] - x) + abs(state.player[1] - y) for x, y in live),
        default=0,
    )
    player_to_box = min(
        (abs(state.player[0] - x) + abs(state.player[1] - y) for x, y in off_goal),
        default=0,
    )
    primary = (
        len(live) * 1000
        + player_to_thief * 10
        + bad * 200
        + len(off_goal) * 100
        + distance * 3
        + player_to_box
        - (20 if state.holds[0] >= 0 else 0)
    )
    return primary, state.player, state.rotation, state.boxes, state.holds


def _actions_to(parent, state):
    actions = []
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return [(action, None, None) for action in actions]


_DELTA_ACTION = {delta: action for action, delta in names.ACTION_DELTAS.items()}


def _step_from_path(path):
    if not path or len(path) < 2:
        return None
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    return _DELTA_ACTION[(dx, dy)]


def _constructive_search(layout: Layout, start: State, limit: int, budget: int) -> SearchResult:
    """Fast receding-horizon teacher for the procedural robot layouts.

    It first lets thieves demonstrate acquisition/delivery, then counters live
    thieves, and finally carries the nearest available off-goal box while
    helpers continue in parallel. Every choice is recomputed after the exact
    ordered robot phase, so this is a real policy witness rather than a stored
    route.
    """
    state = start
    route = []
    seen = Counter()
    work = 0
    reposition = None
    # Tiers 3/4 use a complete divider or cage. Tier 9's six-cell partial
    # fence remains on the established mixed-actor policy.
    closed_handoff = bool(layout.helpers and len(layout.fences) >= 16)
    for tick in range(budget):
        if complete(layout, state):
            return SearchResult(
                [(action, None, None) for action in route], False, work,
                "solved constructively", "native-constructive-policy",
            )
        seen[state] += 1
        action = None
        live_thieves = [cell for cell in state.thieves if cell != DEAD]

        # A real holder-steal witness: on the first encounter with a box held
        # by a robot, take it when already facing it.  The ordered robot phase
        # may immediately steal it back; that is still the native holder
        # transfer, and later ticks proceed to the normal thief counter-route.
        holder = _holder_of(state)
        if tick < 3 and state.holds[0] < 0 and seen[state] == 1:
            for index, owner in enumerate(holder):
                box = state.boxes[index]
                delta = (box[0] - state.player[0], box[1] - state.player[1])
                if owner >= 1 and delta in names.DIRECTION_ROTATION:
                    desired = names.DIRECTION_ROTATION[delta]
                    action = names.ACTION_GRAB if state.rotation == desired else _DELTA_ACTION[delta]
                    break

        # Give the shipped thief policy three ticks to attach, carry, and drop
        # into a nearby bad region before the player counters it.
        if action is None and live_thieves and tick >= 3:
            targets = {
                (x + dx, y + dy)
                for x, y in live_thieves
                for dx, dy in NEIGHBORS
            }
            path = _path(layout, state, 0, targets, carrying=False)
            action = _step_from_path(path)
            if action is None and state.player in targets:
                thief = min(live_thieves, key=lambda cell: abs(state.player[0] - cell[0]) + abs(state.player[1] - cell[1]))
                delta = (thief[0] - state.player[0], thief[1] - state.player[1])
                desired = names.DIRECTION_ROTATION[delta]
                action = names.ACTION_GRAB if state.rotation == desired else _DELTA_ACTION[delta]

        held = state.holds[0]
        if action is None and held >= 0:
            reposition = None
            if state.boxes[held] in layout.goals:
                action = names.ACTION_GRAB
            else:
                actor = state.player
                box = state.boxes[held]
                dx, dy = box[0] - actor[0], box[1] - actor[1]
                targets = [(x - dx, y - dy) for x, y in layout.goals]
                action = _step_from_path(_path(layout, state, 0, targets, carrying=True))
                # Closed fence tiers require a box handoff: actors cannot
                # cross, but a carried box may enter a fence cell.  If no goal
                # route exists, move the box to the nearest reachable fence
                # origin and release it for a helper on the other side.
                if action is None and closed_handoff:
                    if box in layout.fences:
                        action = names.ACTION_GRAB
                    else:
                        # A closed handoff tier can have several equally near
                        # fence cells, but they are not equally useful to the
                        # helper. Score exact reachable player paths together
                        # with the remaining fence-to-goal distance so the
                        # player transports toward a meaningful transfer point
                        # rather than repeatedly choosing the nearest corner.
                        handoffs = []
                        for fence in layout.fences:
                            target = (fence[0] - dx, fence[1] - dy)
                            path = _path(layout, state, 0, [target], carrying=True)
                            if path:
                                delivery = min(
                                    abs(fence[0] - gx) + abs(fence[1] - gy)
                                    for gx, gy in layout.goals
                                )
                                handoffs.append((len(path) + 3 * delivery, len(path), fence, path))
                        if handoffs:
                            _, _, _, path = min(handoffs)
                            action = _step_from_path(path)
                if action is None:
                    # This actor/box orientation cannot reach any destination
                    # (typically at a narrow gate).  Release and approach from
                    # a different side instead of consuming the rest of the
                    # native budget in a deterministic wait loop.
                    reposition = (held, state.player)
                    action = names.ACTION_GRAB

        if action is None and held < 0 and reposition is not None and not live_thieves:
            index, old_side = reposition
            holder = _holder_of(state)
            if holder[index] >= 0 or state.boxes[index] in layout.goals:
                reposition = None
            else:
                box = state.boxes[index]
                targets = {
                    (box[0] + dx, box[1] + dy)
                    for dx, dy in NEIGHBORS
                    if (box[0] + dx, box[1] + dy) != old_side
                }
                path = _path(layout, state, 0, targets, carrying=False)
                action = _step_from_path(path)
                if action is None and state.player in targets:
                    delta = (box[0] - state.player[0], box[1] - state.player[1])
                    desired = names.DIRECTION_ROTATION[delta]
                    action = names.ACTION_GRAB if state.rotation == desired else _DELTA_ACTION[delta]

        if action is None and held < 0 and not live_thieves:
            holder = _holder_of(state)
            options = []
            for index, box in enumerate(state.boxes):
                if box in layout.goals or holder[index] >= 0:
                    continue
                if not closed_handoff:
                    targets = {(box[0] + dx, box[1] + dy) for dx, dy in NEIGHBORS}
                    path = _path(layout, state, 0, targets, carrying=False)
                    if path:
                        options.append((len(path), len(path), index, path))
                    continue
                approaches = []
                for ndx, ndy in NEIGHBORS:
                    target = (box[0] + ndx, box[1] + ndy)
                    path = _path(layout, state, 0, [target], carrying=False)
                    if not path:
                        continue
                    score = len(path)
                    offset = (box[0] - target[0], box[1] - target[1])
                    transfers = []
                    for fence in layout.fences:
                        actor_end = (fence[0] - offset[0], fence[1] - offset[1])
                        if (not layout.in_bounds(actor_end)
                                or actor_end in layout.walls
                                or actor_end in layout.fences):
                            continue
                        carry = abs(target[0] - actor_end[0]) + abs(target[1] - actor_end[1])
                        delivery = min(
                            abs(fence[0] - gx) + abs(fence[1] - gy)
                            for gx, gy in layout.goals
                        )
                        transfers.append(carry + 3 * delivery)
                    if transfers:
                        score += min(transfers)
                    approaches.append((score, len(path), target, path))
                if approaches:
                    score, path_length, _, path = min(approaches)
                    options.append((score, path_length, index, path))
            if options:
                _, _, index, path = min(options)
                action = _step_from_path(path)
                if action is None:
                    box = state.boxes[index]
                    delta = (box[0] - state.player[0], box[1] - state.player[1])
                    desired = names.DIRECTION_ROTATION[delta]
                    action = names.ACTION_GRAB if state.rotation == desired else _DELTA_ACTION[delta]

            if action is None and any(owner >= 1 for owner in holder):
                # Do not body-block a carrying robot's native BFS.  Moving
                # away is an intentional policy step, not a no-op route pad.
                robots = [*state.helpers, *(cell for cell in state.thieves if cell != DEAD)]
                moves = []
                for candidate in names.MOVE_ACTIONS:
                    dx, dy = names.ACTION_DELTAS[candidate]
                    target = (state.player[0] + dx, state.player[1] + dy)
                    if _free(layout, state, target):
                        clearance = min(
                            (abs(target[0] - x) + abs(target[1] - y) for x, y in robots),
                            default=0,
                        )
                        moves.append((clearance, candidate))
                if moves:
                    action = max(moves)[1]

        if action is None:
            # Prefer an actual blocked wait so only the robots advance.
            for candidate in names.MOVE_ACTIONS:
                dx, dy = names.ACTION_DELTAS[candidate]
                target = (state.player[0] + dx, state.player[1] + dy)
                if state.holds[0] < 0 and not _free(layout, state, target):
                    action = candidate
                    break
            action = names.ACTION_GRAB if action is None else action

        if work >= limit:
            return SearchResult(
                None, True, work, "constructive transition-work limit reached",
                "native-constructive-policy",
            )
        nxt = transition(layout, state, action)
        work += 1
        # Escape deterministic local loops by trying the other exact actions;
        # this remains bounded and does not pad a successful route.
        if seen[nxt] > 1:
            alternatives = []
            for candidate in names.ACTION_IDS:
                if work >= limit:
                    return SearchResult(
                        None, True, work,
                        "constructive transition-work limit reached during loop escape",
                        "native-constructive-policy",
                    )
                alternatives.append(transition(layout, state, candidate))
                work += 1
            ranked = sorted(
                zip(names.ACTION_IDS, alternatives),
                key=lambda pair: (_autonomous_score(layout, pair[1]), seen[pair[1]]),
            )
            action, nxt = ranked[0]
        route.append(action)
        state = nxt
    if complete(layout, state):
        return SearchResult(
            [(action, None, None) for action in route], False, work,
            "solved constructively", "native-constructive-policy",
        )
    remaining = sum(box not in layout.goals for box in state.boxes)
    live = sum(thief != DEAD for thief in state.thieves)
    return SearchResult(
        None,
        True,
        work,
        f"constructive policy exhausted native budget (remaining_boxes={remaining}, live_thieves={live}, "
        f"player={state.player}, boxes={state.boxes}, helpers={state.helpers}, thieves={state.thieves}, "
        f"holds={state.holds})",
        "native-constructive-policy",
    )


def _beam_search(layout: Layout, start: State, limit: int, budget: int) -> SearchResult:
    """Bounded positive-witness search for deterministic autonomous levels.

    Beam pruning means failure is always ``truncated``/unknown. A returned
    route is exact because every edge uses :func:`transition` and callers
    replay it in the native engine.
    """
    # Width is independent from branching count; the separate ``expanded``
    # ceiling remains the hard work bound. A 4k beam is needed for the shipped
    # thief maze even though it finds its witness well below the cap.
    width = max(256, min(4_000, limit // max(1, budget * 2)))
    current = {start: ()}
    seen = {start}
    expanded = 0
    for _depth in range(budget):
        candidates = {}
        for state, path in current.items():
            for action in names.ACTION_IDS:
                if expanded >= limit:
                    return SearchResult(
                        None, True, expanded, "autonomous beam transition-work limit reached",
                        "exact-transition-beam",
                    )
                nxt = transition(layout, state, action)
                expanded += 1
                if nxt in seen:
                    continue
                route = path + (action,)
                if complete(layout, nxt):
                    return SearchResult(
                        [(value, None, None) for value in route], False, expanded,
                        "solved", "exact-transition-beam",
                    )
                seen.add(nxt)
                candidates[nxt] = route
        if not candidates:
            return SearchResult(
                None, True, expanded, "autonomous beam exhausted after pruning",
                "exact-transition-beam",
            )
        ordered = sorted(candidates, key=lambda item: _autonomous_score(layout, item))[:width]
        current = {item: candidates[item] for item in ordered}
    return SearchResult(
        None, True, expanded, "native action budget exhausted after beam pruning",
        "exact-transition-beam",
    )


def search(env_or_layout, limit=200_000, budget=None):
    """Bounded exact-state teacher; positive routes are not claimed minima."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        raise Unplannable("layout is not aligned to WA30's 4-pixel lattice")
    backend = "native-constructive-policy" if layout.helpers or layout.thieves else "exact-transition-a-star"
    if not layout.goals or len(layout.goals) < len(layout.boxes):
        return SearchResult(None, False, 0, "fewer goal cells than boxes", backend)
    budget = layout.steps_left if budget is None else budget
    start = state_of(layout)
    if complete(layout, start):
        return SearchResult([], False, 0, "already complete", backend)
    if layout.helpers or layout.thieves:
        constructed = _constructive_search(layout, start, limit, budget)
        if constructed.solved:
            return constructed
        if layout.generated:
            return constructed
        remaining = limit - constructed.work
        if remaining <= 0:
            return constructed
        beam = _beam_search(layout, start, remaining, budget)
        return SearchResult(
            beam.actions, beam.truncated, constructed.work + beam.work,
            f"after constructive phase ({constructed.work} work): {beam.reason}",
            beam.backend,
        )
    best = {start: 0}
    parent = {start: None}
    counter = 0
    frontier = [(_heuristic(layout, start), 0, counter, start)]
    expanded = 0
    while frontier:
        _, depth, _, state = heapq.heappop(frontier)
        if depth != best.get(state):
            continue
        if complete(layout, state):
            return SearchResult(
                _actions_to(parent, state), False, expanded, "solved",
                "exact-transition-a-star",
            )
        if depth >= budget:
            continue
        for action in names.ACTION_IDS:
            if expanded >= limit:
                return SearchResult(
                    None, True, expanded, "A* transition-work limit reached",
                    "exact-transition-a-star",
                )
            nxt = transition(layout, state, action)
            expanded += 1
            cost = depth + 1
            if cost < best.get(nxt, budget + 1):
                best[nxt] = cost
                parent[nxt] = (state, action)
                counter += 1
                weight = 1 if not layout.helpers and not layout.thieves else 3
                heapq.heappush(frontier, (cost + weight * _heuristic(layout, nxt), cost, counter, nxt))
    return SearchResult(
        None, False, expanded, "search space exhausted within budget",
        "exact-transition-a-star",
    )


def solve(env_or_layout, limit=200_000, budget=None):
    result = search(env_or_layout, limit=limit, budget=budget)
    solve.result = result
    solve.truncated = result.truncated
    return result.actions


solve.result = None
solve.truncated = False
