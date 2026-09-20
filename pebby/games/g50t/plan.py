"""Bounded full-mechanics teacher for G50T.

The search is deliberately positive-certificate oriented. It represents all
shipped mechanics, but keeps a bounded number of historical walks for each
live configuration. A found plan is exact after native replay; exhaustion is
reported as inconclusive rather than as impossibility.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, replace
import heapq

from . import names
from .layout import EnemyState, Ghost, Layout, World, extract


Action = tuple[int, None, None]
DEFAULT_NODE_LIMIT = 500_000
HISTORIES_PER_CONFIGURATION = 6
MAX_REWIND_CANDIDATES = 400


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

    @property
    def status(self):
        if self.solved:
            return "solved"
        if self.unsupported:
            return "unsupported"
        if self.truncated:
            return "truncated"
        return "unsolved"


def _move(cell, action):
    dx, dy = names.ACTION_DELTA[action]
    return cell[0] + dx * names.GRID_STEP, cell[1] + dy * names.GRID_STEP


def _actors(world):
    values = [("p", 0, world.player, not world.dead)]
    values.extend(("g", i, ghost.position, True) for i, ghost in enumerate(world.ghosts))
    values.extend(("e", i, enemy.position, not enemy.dead)
                  for i, enemy in enumerate(world.enemies))
    return values


def _blocked(layout, doors):
    return frozenset(door.opened if active else door.closed
                     for door, active in zip(layout.doors, doors))


def _circuit(layout, circuit, value, doors, positions, pad_sets, *, rewind=False):
    doors = list(doors)
    for index, door in enumerate(layout.doors):
        if door.circuit != circuit:
            continue
        if door.toggle:
            if value:
                doors[index] = not doors[index]
        else:
            doors[index] = value
    if (value and not rewind) or (rewind and not value):
        for index, link in enumerate(layout.teleports):
            if link.circuit != circuit:
                continue
            # A link's animation completes after actor movement. Native pad
            # membership can lag the scheduled move, but the completed swap
            # observes every actor geometrically on a pad (not merely the set
            # populated earlier in the arrival loop).
            left = {actor for actor, position in positions.items()
                    if position == link.pads[0]}
            right = {actor for actor, position in positions.items()
                     if position == link.pads[1]}
            for actor in tuple(left):
                positions[actor] = link.pads[1]
            for actor in tuple(right):
                positions[actor] = link.pads[0]
            pad_sets[index] = (set(right), set(left))
    return tuple(doors)


def _occupancy(layout, world):
    switch_sets = [set() for _ in layout.switches]
    pad_sets = [(set(), set()) for _ in layout.teleports]
    for kind, index, position, alive in _actors(world):
        if not alive:
            continue
        actor = (kind, index)
        for number, (cell, _) in enumerate(layout.switches):
            if position == cell:
                switch_sets[number].add(actor)
        for number, link in enumerate(layout.teleports):
            if position == link.pads[0]:
                pad_sets[number][0].add(actor)
            elif position == link.pads[1]:
                pad_sets[number][1].add(actor)
    return switch_sets, pad_sets


def _leave_inputs(layout, actor, position, switch_sets, pad_sets, doors, positions,
                  *, rewind=False):
    for number, (cell, circuit) in enumerate(layout.switches):
        if position != cell or actor not in switch_sets[number]:
            continue
        was_active = bool(switch_sets[number])
        switch_sets[number].remove(actor)
        if was_active and not switch_sets[number]:
            doors = _circuit(layout, circuit, False, doors, positions, pad_sets,
                             rewind=rewind)
    for number, link in enumerate(layout.teleports):
        for side in (0, 1):
            if position == link.pads[side]:
                pad_sets[number][side].discard(actor)
    return doors


def _enter_inputs(layout, actor, switch_sets, pad_sets, doors, positions,
                  *, rewind=False):
    position = positions[actor]
    for number, (cell, circuit) in enumerate(layout.switches):
        if position != cell:
            continue
        was_active = bool(switch_sets[number])
        switch_sets[number].add(actor)
        if not was_active:
            doors = _circuit(layout, circuit, True, doors, positions, pad_sets,
                             rewind=rewind)
        break
    for number, link in enumerate(layout.teleports):
        if position == link.pads[0]:
            pad_sets[number][0].add(actor)
            break
        if position == link.pads[1]:
            pad_sets[number][1].add(actor)
            break
    return doors


def _enemy_move(layout, number, enemy, blocked):
    if enemy.dead:
        return enemy, None
    orientation = enemy.orientation
    for _ in range(4):
        choices = (orientation, (orientation - 1) % 4,
                   (orientation + 1) % 4, (orientation + 2) % 4)
        action = None
        chosen = orientation
        for direction in choices:
            candidate_action = (names.ACTION_DOWN, names.ACTION_RIGHT,
                                names.ACTION_UP, names.ACTION_LEFT)[direction]
            if _move(enemy.position, candidate_action) in layout.enemies[number].path:
                action = candidate_action
                chosen = direction
                break
        if action is None:
            return replace(enemy, orientation=orientation), None
        orientation = chosen
        candidate = _move(enemy.position, action)
        if candidate in layout.allowed and candidate not in blocked:
            return EnemyState(candidate, orientation, False,
                              enemy.history + (action,)), action
        orientation = (orientation + 2) % 4
    return replace(enemy, orientation=orientation), None


def _inverse_timeline(layout, world):
    """Run the native action-5 inverse history, including circuit side effects."""
    recorded_player_history = world.history
    current = world
    first_departure = True
    while current.history:
        player_history = current.history[:-1]
        history_index = len(player_history)
        # Native starts the first inverse moves before raising its rewind-phase
        # flag.  It raises the flag before those actors arrive (when another
        # inverse tick remains), keeps it raised for intermediate ticks, then
        # clears it immediately after scheduling the final departures and
        # before their arrivals.  Circuit, toggle and teleport callbacks see
        # the phase at their own departure/arrival boundary.
        departure_rewind = not first_departure
        arrival_rewind = bool(player_history)
        old_blocked = _blocked(layout, current.doors)
        switch_sets, pad_sets = _occupancy(layout, current)
        positions = {
            (kind, index): position
            for kind, index, position, alive in _actors(current) if alive
        }
        doors = current.doors
        enemies = list(current.enemies)
        moving = []

        # Native inverse departure order is enemies, existing ghosts, player.
        # Enemy histories are compact: once the player index falls below their
        # length, the latest successful enemy step is popped and inverted.
        for index, enemy in enumerate(current.enemies):
            if history_index >= len(enemy.history):
                continue
            action = enemy.history[-1]
            enemies[index] = replace(enemy, dead=False, history=enemy.history[:-1])
            positions[("e", index)] = enemy.position
            moving.append(("e", index, action))
        for index, ghost in enumerate(current.ghosts):
            if history_index < len(ghost.path):
                moving.append(("g", index, ghost.path[history_index]))
        moving.append(("p", 0, current.history[-1]))

        moved = set()
        for kind, index, forward_action in moving:
            actor = (kind, index)
            if actor not in positions:
                continue
            dx, dy = names.ACTION_DELTA[forward_action]
            inverse_action = next(
                action for action, delta in names.ACTION_DELTA.items()
                if delta == (-dx, -dy)
            )
            candidate = _move(positions[actor], inverse_action)
            if candidate not in layout.allowed or candidate in old_blocked:
                continue
            doors = _leave_inputs(
                layout, actor, positions[actor], switch_sets, pad_sets, doors,
                positions, rewind=departure_rewind,
            )
            # An intermediate/final rewind-triggered teleport can move the
            # actor during departure; native movement then applies the inverse
            # delta at that new pad.  The first departure is forward-phase, so
            # a switch release cannot trigger that inverse teleport.
            positions[actor] = _move(positions[actor], inverse_action)
            moved.add(actor)

        for actor in ([('p', 0)]
                      + [('e', i) for i in range(len(enemies))]
                      + [('g', i) for i in range(len(current.ghosts))]):
            if actor in moved:
                doors = _enter_inputs(
                    layout, actor, switch_sets, pad_sets, doors, positions,
                    rewind=arrival_rewind,
                )

        blocked_after = _blocked(layout, doors)
        dead_player = positions[("p", 0)] in blocked_after
        for index, enemy in enumerate(enemies):
            position = positions.get(("e", index), enemy.position)
            enemies[index] = replace(
                enemy, position=position,
                dead=enemy.dead or position in blocked_after,
            )
        ghosts = tuple(
            replace(ghost, position=positions.get(("g", index), ghost.position))
            for index, ghost in enumerate(current.ghosts)
        )
        current = World(
            positions[("p", 0)], player_history, ghosts, tuple(enemies),
            doors, current.stage, dead_player,
        )
        first_departure = False

    stage = current.stage + 1
    if stage == layout.stage_count:
        ghosts = ()
        stage = 0
        events = {"rewind"}
    else:
        ghosts = current.ghosts + (Ghost(layout.start, recorded_player_history),)
        events = {"rewind", "ghost_created"}
    reset = World(
        layout.start, (), ghosts,
        tuple(replace(enemy, history=()) for enemy in current.enemies),
        current.doors, stage, False,
    )
    return reset, layout.start == layout.goal, events


def transition(layout, world, action):
    """Apply one stable native action; return ``(world, won, events)`` or None."""
    if action == names.ACTION_REWIND:
        if not world.history:
            return None
        return _inverse_timeline(layout, world)
    if action not in names.MOVE_ACTIONS or world.dead:
        return None

    old_blocked = _blocked(layout, world.doors)
    player_next = _move(world.player, action)
    if player_next not in layout.allowed or player_next in old_blocked:
        return None

    positions = {(kind, index): position for kind, index, position, alive in _actors(world)
                 if alive}
    switch_sets, pad_sets = _occupancy(layout, world)
    doors = world.doors
    moving = [("p", 0, player_next, action)]
    history_index = len(world.history)
    for index, ghost in enumerate(world.ghosts):
        if history_index >= len(ghost.path):
            continue
        ghost_action = ghost.path[history_index]
        candidate = _move(ghost.position, ghost_action)
        if candidate in layout.allowed and candidate not in old_blocked:
            moving.append(("g", index, candidate, ghost_action))

    enemy_values = list(world.enemies)
    for index, enemy in enumerate(world.enemies):
        updated, used = _enemy_move(layout, index, enemy, old_blocked)
        enemy_values[index] = updated
        if used is not None:
            moving.append(("e", index, updated.position, used))

    # Native departure order is player, ghosts, enemies. Door sprites finish
    # moving only after all candidates have been admitted against old_blocked.
    order = {"p": 0, "g": 1, "e": 2}
    for kind, index, candidate, _ in sorted(moving, key=lambda x: (order[x[0]], x[1])):
        actor = (kind, index)
        doors = _leave_inputs(layout, actor, positions[actor], switch_sets,
                              pad_sets, doors, positions)
        positions[actor] = candidate

    events = {"move"}
    if any(before != after for before, after in zip(world.doors, doors)):
        events.add("door_changed")
    blocked_after = _blocked(layout, doors)
    dead_player = positions[("p", 0)] in blocked_after
    for index, enemy in enumerate(enemy_values):
        actor = ("e", index)
        if enemy.dead or actor not in positions:
            continue
        if positions[actor] in blocked_after:
            enemy_values[index] = replace(enemy, dead=True)
            events.add("enemy_crushed")

    # Native arrival order is player, enemies, ghosts. A teleport circuit can
    # move actors already registered on its pads before later actors arrive.
    arrival = [("p", 0)]
    arrival += [("e", i) for i, enemy in enumerate(enemy_values) if not enemy.dead]
    arrival += [("g", i) for i in range(len(world.ghosts))]
    for actor in arrival:
        if actor not in positions:
            continue
        before_positions = dict(positions)
        before_doors = doors
        doors = _enter_inputs(layout, actor, switch_sets, pad_sets, doors, positions)
        if positions != before_positions:
            events.add("teleport")
        if doors != before_doors:
            events.add("door_changed")
        blocked_now = _blocked(layout, doors)
        if positions.get(("p", 0)) in blocked_now:
            dead_player = True
        for index, enemy in enumerate(enemy_values):
            key = ("e", index)
            if not enemy.dead and positions.get(key) in blocked_now:
                enemy_values[index] = replace(enemy, dead=True)
                events.add("enemy_crushed")

    player_position = positions[("p", 0)]
    for index, enemy in enumerate(enemy_values):
        if not enemy.dead and positions.get(("e", index)) == player_position:
            dead_player = True
            events.add("enemy_contact")
    ghosts = tuple(Ghost(positions.get(("g", index), ghost.position), ghost.path)
                   for index, ghost in enumerate(world.ghosts))
    enemies = tuple(replace(enemy, position=positions.get(("e", index), enemy.position))
                    for index, enemy in enumerate(enemy_values))
    next_world = World(player_position, world.history + (action,), ghosts,
                       enemies, doors, world.stage, dead_player)
    won = not dead_player and player_position == layout.goal
    return next_world, won, events


def _distances(layout, doors, start):
    blocked = _blocked(layout, doors)
    if start not in layout.allowed or start in blocked:
        return {}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        for action in names.MOVE_ACTIONS:
            nxt = _move(here, action)
            if nxt in layout.allowed and nxt not in blocked and nxt not in distances:
                distances[nxt] = distances[here] + 1
                queue.append(nxt)
    return distances


def _priority(layout, world, depth, serial):
    distances = _distances(layout, world.doors, world.player)
    if layout.goal in distances:
        target = distances[layout.goal]
    else:
        reachable_switches = [distances[cell] for cell, _ in layout.switches
                              if cell in distances and cell != world.player]
        target = (min(reachable_switches) if reachable_switches else 30) + 12
    occupied = sum(any(position == cell and alive
                       for _, _, position, alive in _actors(world))
                   for cell, _ in layout.switches)
    score = depth + target - 5 * world.stage - 2 * occupied
    return score, depth, serial


def _core(world):
    # A replay route is future program state, not incidental history. Keep
    # endpoint/length classes distinct so a short early rewind cannot evict a
    # later route that parks a ghost on a pressure switch.
    return (world.player,
            tuple((g.position, len(g.path), g.path[-1] if g.path else None)
                  for g in world.ghosts),
            tuple((e.position, e.orientation, e.dead) for e in world.enemies),
            world.doors, world.stage, len(world.history))


def _unwind(parent, state, final_action):
    actions = [final_action]
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return tuple((action, None, None) for action in actions)


def _motion_key(world):
    """Dominance key inside one stage.

    The concrete history is retained on ``world`` for a later replay ghost,
    while equal-length walks reaching the same complete live configuration are
    interchangeable for finding a positive continuation in the current stage.
    """
    return (world.player, len(world.history),
            tuple(g.position for g in world.ghosts),
            tuple((e.position, e.orientation, e.dead) for e in world.enemies),
            world.doors, world.stage)


def _stage_search(layout, start, cap, counter):
    """Find a direct win and useful rewind endpoints in one timeline slot."""
    interests = {cell for cell, _ in layout.switches}
    interests.update(pad for link in layout.teleports for pad in link.pads)
    queue = deque([(start, ())])
    seen = {_motion_key(start)}
    candidates = []
    candidate_keys = set()
    while queue:
        world, route = queue.popleft()
        if len(route) >= cap:
            continue
        if counter[0] >= counter[2]:
            return None, candidates, True
        counter[0] += 1
        for action in names.MOVE_ACTIONS:
            outcome = transition(layout, world, action)
            if outcome is None:
                continue
            nxt, won, _ = outcome
            if nxt.dead:
                continue
            counter[1] += 1
            next_route = route + (action,)
            if won:
                return next_route, candidates, False
            key = _motion_key(nxt)
            if key in seen:
                continue
            seen.add(key)
            queue.append((nxt, next_route))
            if nxt.player in interests:
                candidate_key = (nxt.player, len(nxt.history),
                                 tuple(g.position for g in nxt.ghosts),
                                 tuple((e.position, e.orientation, e.dead)
                                       for e in nxt.enemies), nxt.doors)
                if candidate_key not in candidate_keys:
                    candidate_keys.add(candidate_key)
                    candidates.append((nxt, next_route))
                    if len(candidates) >= MAX_REWIND_CANDIDATES:
                        return None, candidates, False
    return None, candidates, False


def _macro_search(layout, start, cap, node_limit):
    """Search timeline slots by enumerating useful ghost-route endpoints."""
    counter = [0, 0, node_limit]  # expanded, generated, bound
    macro_seen = set()

    def visit(world, remaining, prefix):
        macro_key = (world.stage, tuple((g.path, g.position) for g in world.ghosts),
                     tuple((e.position, e.orientation, e.dead) for e in world.enemies),
                     world.doors)
        if macro_key in macro_seen:
            return None, False
        macro_seen.add(macro_key)
        direct, candidates, stopped = _stage_search(layout, world, remaining, counter)
        if direct is not None:
            return prefix + direct, False
        if stopped or world.stage + 1 >= layout.stage_count:
            return None, stopped
        # Short routes to ordinary switches are usually the useful ghost
        # programs. Pads remain candidates for levels whose teleport timing
        # requires parking an actor before another route triggers the link.
        switch_circuits = {cell: circuit for cell, circuit in layout.switches}
        ordinary = {door.circuit for door in layout.doors if not door.toggle}
        teleporting = {link.circuit for link in layout.teleports}

        def candidate_rank(item):
            candidate, route = item
            circuit = switch_circuits.get(candidate.player)
            occupied = candidate.player in {ghost.position for ghost in candidate.ghosts}
            if circuit in ordinary:
                mechanic = 0
            elif circuit in teleporting:
                mechanic = 1
            elif circuit is not None:
                mechanic = 2
            else:
                mechanic = 3
            return occupied, mechanic, len(route), candidate.player

        candidates.sort(key=candidate_rank)
        truncated = False
        for candidate, route in candidates:
            if candidate.player in {ghost.position for ghost in candidate.ghosts}:
                continue
            if len(route) + 1 >= remaining:
                continue
            rewound = transition(layout, candidate, names.ACTION_REWIND)
            if rewound is None:
                continue
            result, child_stopped = visit(
                rewound[0], remaining - len(route) - 1,
                prefix + route + (names.ACTION_REWIND,),
            )
            truncated |= child_stopped
            if result is not None:
                return result, truncated
            if counter[0] > node_limit:
                return None, True
        return None, truncated

    result, truncated = visit(start, cap, ())
    return result, counter[0], counter[1], truncated


def search(env_or_layout, limit=None, budget=None, node_limit=DEFAULT_NODE_LIMIT):
    for value, label in ((limit, "limit"), (budget, "budget")):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{label} must be a non-negative integer or None")
    if isinstance(node_limit, bool) or not isinstance(node_limit, int) or node_limit < 1:
        raise ValueError("node_limit must be a positive integer")
    source_env = None if isinstance(env_or_layout, Layout) else env_or_layout
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(None, False, True, False, 0, 0,
                            "; ".join(layout.unsupported) or "unsupported live state",
                            node_limit)
    caps = [layout.steps_left]
    if limit is not None:
        caps.append(limit)
    if budget is not None:
        caps.append(budget)
    cap = min(caps)
    start = layout.world
    macro, macro_expanded, macro_generated, macro_stopped = _macro_search(
        layout, start, cap, node_limit,
    )
    if macro is not None:
        result = tuple((action, None, None) for action in macro)
        if source_env is not None:
            try:
                from .env import replay
                verified = replay(source_env.clone(), result)
            except Exception as exc:
                return SearchResult(None, True, False, False, macro_expanded,
                                    macro_generated,
                                    f"native witness replay raised {type(exc).__name__}: {exc}",
                                    node_limit)
            if verified:
                return SearchResult(result, False, False, True, macro_expanded,
                                    macro_generated,
                                    "full-mechanics macro plan replayed in the native engine",
                                    node_limit)
        else:
            return SearchResult(result, False, False, True, macro_expanded,
                                macro_generated, "full-mechanics macro plan found",
                                node_limit)
    if macro_stopped or macro_expanded >= node_limit:
        return SearchResult(None, True, False, False, macro_expanded,
                            macro_generated,
                            f"configuration expansion limit {node_limit} reached",
                            node_limit)

    remaining_limit = max(1, node_limit - macro_expanded)
    queue = []
    serial = 0
    heapq.heappush(queue, (*_priority(layout, start, 0, serial), start, 0))
    parent = {start: None}
    histories = defaultdict(int)
    histories[_core(start)] = 1
    expanded = generated = 0
    while queue and expanded < remaining_limit:
        *_, world, depth = heapq.heappop(queue)
        expanded += 1
        if depth >= cap:
            continue
        actions = list(names.MOVE_ACTIONS)
        if world.history:
            actions.append(names.ACTION_REWIND)
        for action in actions:
            outcome = transition(layout, world, action)
            if outcome is None:
                continue
            nxt, won, _ = outcome
            if nxt.dead:
                continue
            generated += 1
            if won:
                result = _unwind(parent, world, action)
                if source_env is not None:
                    try:
                        from .env import replay
                        verified = replay(source_env.clone(), result)
                    except Exception as exc:
                        return SearchResult(None, True, False, False,
                                            macro_expanded + expanded,
                                            macro_generated + generated,
                                            f"native witness replay raised {type(exc).__name__}: {exc}",
                                            node_limit)
                    if not verified:
                        continue
                return SearchResult(result, False, False, True,
                                    macro_expanded + expanded,
                                    macro_generated + generated,
                                    "full-mechanics plan replayed in the native engine"
                                    if source_env is not None else
                                    "full-mechanics symbolic plan found",
                                    node_limit)
            if nxt in parent:
                continue
            core = _core(nxt)
            if histories[core] >= HISTORIES_PER_CONFIGURATION:
                continue
            histories[core] += 1
            parent[nxt] = (world, action)
            serial += 1
            heapq.heappush(queue, (*_priority(layout, nxt, depth + 1, serial),
                                   nxt, depth + 1))
    expanded += macro_expanded
    generated += macro_generated
    reason = (f"configuration expansion limit {node_limit} reached" if expanded >= node_limit
              else f"bounded historical search found no witness within action cap {cap}")
    return SearchResult(None, True, False, False, expanded, generated, reason, node_limit)


def solve(env_or_layout, limit=None, budget=None, node_limit=DEFAULT_NODE_LIMIT):
    result = search(env_or_layout, limit=limit, budget=budget, node_limit=node_limit)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    solve.reason = result.reason
    return list(result.actions) if result.actions is not None else None


solve.result = None
solve.truncated = False
solve.unsupported = False
solve.exact = False
solve.reason = ""
