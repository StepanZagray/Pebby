"""Bounded complete-program planner for every shipped TN36 mechanic.

The inner dynamic program exhausts opcode values per instruction while merging
identical world states. A small outer Dijkstra search handles failed runs whose
checkpoint or gate phase changes. Cutoffs remain explicit unknowns.
"""

from dataclasses import dataclass
import heapq

from . import names
from .layout import Gate, Layout, Rect, Transform, extract


DEFAULT_NODE_LIMIT = 400_000
Action = tuple[int, int | None, int | None]
World = tuple[Transform, Transform, bool, tuple[bool, ...]]


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
    program_runs: int = 0

    @property
    def solved(self):
        return self.actions is not None


@dataclass(frozen=True)
class Execution:
    world: World
    program: tuple[int, ...]
    toggle_cost: int
    won: bool
    events: tuple[str, ...]
    satisfied: int = 0


# Every event the boundary model can emit, plus the selector click event that
# only the public route (not a program run) can produce.
PROGRAM_EVENTS = frozenset({
    "translation", "collision_rollback", "scale", "rotation", "recolor", "noop",
    "gate_destroy", "platform_contact", "gate_toggle", "platform_checkpoint",
    "win", "gate_toggle_after_failure", "failed_run_reset",
})
SELECTOR_EVENT = "preset_selection"


def apply_code(state: Transform, code: int) -> Transform:
    """Apply an opcode without collision context (legacy/public helper)."""
    x, y, rotation, scale, color = state
    kind, value = names.OPCODE_EFFECTS.get(int(code), ("noop", 0))
    if kind == "dx":
        x += value
    elif kind == "dy":
        y += value
    elif kind == "rotation":
        rotation = (rotation + value) % 360
    elif kind == "scale":
        scale = max(1, scale + value)
    elif kind == "color":
        color = value
    return (x, y, rotation, scale, color)


def execute(initial: Transform, program):
    state = initial
    for code in program:
        state = apply_code(state, code)
    return state


def _validate_bound(value, name, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        suffix = " or None" if optional else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return int(value)


def _overlap(actor: Transform, rect: Rect) -> bool:
    x, y, _, scale, _ = actor
    side = 4 * scale
    return (
        x < rect.x + rect.width and rect.x < x + side
        and y < rect.y + rect.height and rect.y < y + side
    )


def _gate_rect(gate: Gate, visible: bool) -> Rect:
    return gate.barrier if visible else gate.body


def _step(layout: Layout, world: World, code: int, instruction: int, last: int):
    actor, checkpoint, alive, gate_phase = world
    events = []
    kind, _ = names.OPCODE_EFFECTS.get(int(code), ("noop", 0))
    proposed = actor
    if alive and kind in ("dx", "dy"):
        proposed = apply_code(actor, code)
        if any(_overlap(proposed, wall) for wall in layout.walls):
            proposed = actor
            events.append("collision_rollback")
        else:
            events.append("translation")
    elif alive and kind == "scale":
        proposed = apply_code(actor, code)
        if any(_overlap(proposed, wall) for wall in layout.walls):
            proposed = actor
            events.append("collision_rollback")
        else:
            events.append("scale")
    elif kind == "rotation":
        proposed = apply_code(actor, code)
        events.append("rotation")
    elif kind == "color":
        proposed = apply_code(actor, code)
        events.append("recolor")
    elif code == 0:
        events.append("noop")
    actor = proposed
    if alive and kind in ("dx", "dy", "scale"):
        if any(
            _overlap(actor, _gate_rect(gate, visible))
            for gate, visible in zip(layout.gates, gate_phase)
        ):
            alive = False
            events.append("gate_destroy")
        if any(actor[3] == 1 and _overlap(actor, platform) for platform in layout.platforms):
            events.append("platform_contact")
    if instruction % 3 == 2 and instruction < last:
        gate_phase = tuple(not value for value in gate_phase)
        if gate_phase:
            events.append("gate_toggle")
        if alive and any(
            visible and _overlap(actor, gate.barrier)
            for gate, visible in zip(layout.gates, gate_phase)
        ):
            alive = False
            events.append("gate_destroy")
    return (actor, checkpoint, alive, gate_phase), tuple(events)


def _finish(layout: Layout, world: World):
    actor, checkpoint, alive, gate_phase = world
    events = []
    for platform in layout.platforms:
        if alive and actor[3] == 1 and _overlap(actor, platform):
            checkpoint = actor
            events.append("platform_checkpoint")
            break
    won = alive and actor == layout.target
    if won:
        events.append("win")
        return (actor, checkpoint, alive, gate_phase), True, tuple(events)
    if gate_phase:
        gate_phase = tuple(not value for value in gate_phase)
        events.append("gate_toggle_after_failure")
    if layout.reset_after_run:
        actor, alive = checkpoint, True
        events.append("failed_run_reset")
    return (actor, checkpoint, alive, gate_phase), False, tuple(events)


def execute_program(layout: Layout, world: World, program):
    """Execute one program in the complete action-boundary model."""
    _, checkpoint, _, gate_phase = world
    # Native run control always restores the saved start/checkpoint before its
    # first instruction. reset_after_run controls only a failed run's finish.
    world = (checkpoint, checkpoint, True, gate_phase)
    events = []
    last = len(program) - 1
    for index, code in enumerate(program):
        world, emitted = _step(layout, world, int(code), index, last)
        events.extend(emitted)
    world, won, emitted = _finish(layout, world)
    events.extend(emitted)
    return world, won, tuple(events)


def program_end_state(layout: Layout, world: World, program):
    """Return state/events immediately before end-of-program win/reset logic."""
    _, checkpoint, _, gate_phase = world
    world = (checkpoint, checkpoint, True, gate_phase)
    events = []
    last = len(program) - 1
    for index, code in enumerate(program):
        world, emitted = _step(layout, world, int(code), index, last)
        events.extend(emitted)
    return world, tuple(events)


def _toggle_actions(layout, before, program):
    actions = []
    for current, wanted, clicks in zip(before, program, layout.bit_clicks):
        difference = current ^ wanted
        for bit, (x, y) in enumerate(clicks):
            if difference & (1 << bit):
                actions.append((names.ACTION_CLICK, x, y))
    actions.append((names.ACTION_CLICK, *layout.run_click))
    return tuple(actions)


def program_actions(layout, before, program):
    """Encode one desired program and its run click as public action triples."""
    if len(before) != len(program) or len(program) != len(layout.bit_clicks):
        raise ValueError("program length does not match the live instruction grid")
    return _toggle_actions(layout, tuple(before), tuple(program))


def _satisfy(mask, emitted, tracked):
    for event in emitted:
        bit = tracked.get(event)
        if bit is not None:
            mask |= bit
    return mask


def _executions(layout, world, current_program, counter, node_limit, tracked, mask):
    # (world, satisfied required events) -> (toggle cost, program, events)
    _, checkpoint, _, gate_phase = world
    run_start = (checkpoint, checkpoint, True, gate_phase)
    frontier = {(run_start, mask): (0, (), ())}
    last = len(layout.bit_clicks) - 1
    for slot, clicks in enumerate(layout.bit_clicks):
        next_frontier = {}
        for (state, satisfied), (cost, program, events) in frontier.items():
            counter[0] += 1
            if counter[0] > node_limit:
                return None
            for code in range(1 << len(clicks)):
                counter[1] += 1
                after, emitted = _step(layout, state, code, slot, last)
                candidate = (
                    cost + (current_program[slot] ^ code).bit_count(),
                    program + (code,), events + emitted,
                )
                key = (after, _satisfy(satisfied, emitted, tracked))
                previous = next_frontier.get(key)
                if previous is None or candidate[:2] < previous[:2]:
                    next_frontier[key] = candidate
        frontier = next_frontier
    # Reset/checkpoint handling can collapse thousands of terminal transforms
    # to the same persistent world. Keep the cheapest edit for each such world
    # and satisfied-event set; this is a constructive multi-run reduction, not
    # a shortest-route proof.
    finished = {}
    for (state, satisfied), (cost, program, events) in frontier.items():
        after, won, emitted = _finish(layout, state)
        satisfied = _satisfy(satisfied, emitted, tracked)
        candidate = Execution(after, program, cost, won, events + emitted, satisfied)
        key = (after, won, satisfied)
        previous = finished.get(key)
        if previous is None or (candidate.toggle_cost, candidate.program) < (previous.toggle_cost, previous.program):
            finished[key] = candidate
    return tuple(finished.values())


def _effects(program):
    return {
        names.OPCODE_EFFECTS.get(int(code), ("noop", 0))
        for code in program
    } - {("noop", 0)}


def selector_index(layout, programs):
    """Choose the preset demonstrating an effect the executed programs use.

    Exact opcode effect first, then the same effect family, then the first
    selector. The choice is a pure function of the layout and route so an
    independent recomputation reproduces it.
    """
    used = set()
    for program in programs:
        used |= _effects(program)
    kinds = {kind for kind, _ in used}
    for index, preset in enumerate(layout.presets):
        if _effects(preset.program) & used:
            return index
    for index, preset in enumerate(layout.presets):
        if {kind for kind, _ in _effects(preset.program)} & kinds:
            return index
    return 0


def _required(required_events):
    if required_events is None:
        return ()
    if isinstance(required_events, str) or not isinstance(required_events, (tuple, list, set, frozenset)):
        raise ValueError("required_events must be a collection of event names")
    required = tuple(dict.fromkeys(required_events))
    unknown = [event for event in required if event not in PROGRAM_EVENTS and event != SELECTOR_EVENT]
    if unknown:
        raise ValueError(f"unknown required events: {unknown}")
    return required


def search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT, budget=None, *,
           required_events=()):
    """Return a bounded exact minimum-click program/run route.

    ``required_events`` names mechanic events the returned route must
    exercise. Program events are tracked as a satisfied-set inside the
    world-state graph, so the result is the cheapest route among those that
    exercise every one of them. ``preset_selection`` is not a program event:
    it is satisfied by prepending one selector click (which costs one timer
    click and leaves the goal panel untouched) chosen by ``selector_index``.
    """
    limit = _validate_bound(limit, "limit", optional=True)
    node_limit = _validate_bound(node_limit, "node_limit")
    budget = _validate_bound(budget, "budget", optional=True)
    required = _required(required_events)
    needs_selector = SELECTOR_EVENT in required
    tracked = {event: 1 << index for index, event in enumerate(
        event for event in required if event != SELECTOR_EVENT)}
    full_mask = sum(tracked.values())
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(None, False, True, False, 0, 0,
                            "; ".join(layout.unsupported) or "unsupported live state", node_limit)
    if needs_selector and not layout.presets:
        return SearchResult(None, False, False, True, 0, 0,
                            "preset selection is required but the level has no selectors", node_limit)
    available = layout.clicks_left if budget is None else min(layout.clicks_left, budget)
    external_cap = available if limit is None else min(available, limit)
    if needs_selector:
        external_cap -= 1
    if external_cap < 1:
        return SearchResult(None, False, False, True, 0, 0,
                            "native scrolling timer has expired", node_limit)
    if layout.current == layout.target and layout.actor_alive and not required:
        x, y = layout.bit_clicks[0][0]
        return SearchResult(((names.ACTION_CLICK, x, y),), False, False, True,
                            0, 0, "live actor already matches target", node_limit)

    start_world = (
        layout.current, layout.initial, layout.actor_alive,
        tuple(gate.visible for gate in layout.gates),
    )
    start_key = (start_world, layout.program, 0)
    queue = [(0, 0, start_key, (), ())]
    best = {start_key: 0}
    counter = [0, 0]
    while queue:
        cost, runs, (world, current_program, mask), prior, programs = heapq.heappop(queue)
        if best.get((world, current_program, mask)) != cost:
            continue
        executions = _executions(layout, world, current_program, counter, node_limit, tracked, mask)
        if executions is None:
            return SearchResult(None, True, False, True, min(counter[0], node_limit), counter[1],
                                f"configuration expansion limit {node_limit} reached", node_limit, runs)
        winning = []
        for execution in executions:
            actions = _toggle_actions(layout, current_program, execution.program)
            total = cost + len(actions)
            if total > external_cap:
                continue
            combined = prior + actions
            if execution.won:
                if execution.satisfied == full_mask:
                    winning.append((total, execution.program, combined, programs + (execution.program,)))
                continue
            if execution.world == world and execution.satisfied == mask:
                # An intermediate run with no persistent world effect cannot
                # improve a later edit sequence (Hamming distance obeys the
                # triangle inequality), and only burns a timer click.
                continue
            key = (execution.world, execution.program, execution.satisfied)
            if total < best.get(key, 1 << 60):
                best[key] = total
                heapq.heappush(queue, (total, runs + 1, key, combined, programs + (execution.program,)))
        if winning:
            _, _, combined, route_programs = min(winning)
            if needs_selector:
                x, y = layout.presets[selector_index(layout, route_programs)].click
                combined = ((names.ACTION_CLICK, x, y),) + combined
            complete = runs == 0
            qualifier = "minimum one-run" if complete else "constructive multi-run"
            if required:
                qualifier += f" route exercising {', '.join(required)};"
            return SearchResult(combined, False, False, complete, counter[0], counter[1],
                                f"{qualifier} route found across {runs + 1} program run(s)",
                                node_limit, runs + 1)
    reason = "winning state is unreachable after exhausting the exact bounded program/world graph"
    if limit is not None and limit < available:
        return SearchResult(None, True, False, True, counter[0], counter[1],
                            f"no win within action limit {limit}; larger routes remain unknown", node_limit)
    return SearchResult(None, False, False, True, counter[0], counter[1], reason, node_limit)


def solve(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT, budget=None, *,
          required_events=()):
    result = search(env_or_layout, limit=limit, node_limit=node_limit, budget=budget,
                    required_events=required_events)
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
