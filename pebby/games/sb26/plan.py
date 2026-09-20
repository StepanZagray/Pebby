"""Bounded exact teacher for SB26's complete recursive frame grammar.

The only combinatorial choice is the assignment of movable tiles to non-fixed
frame slots. At most nine tiles move in the shipped game, so an exhaustive
multiset assignment search is small and avoids cloning animation-heavy engine
states. Every published positive result is also replayed by the real engine.
"""

from collections import Counter
from dataclasses import dataclass
from numbers import Integral

from . import names
from .layout import Layout, extract


Action = tuple[int, int | None, int | None]


@dataclass(frozen=True)
class TraversalResult:
    won: bool
    reason: str
    regular_visits: int
    link_traversals: int
    distinct_link_tiles: int
    distinct_frames_entered: int
    repeated_frame_entries: int
    cycle_reentries: int
    cycle_guard_rejections: int
    maximum_depth: int
    fixed_regular_visits: int
    movable_link_visits: int
    emitted: tuple[int, ...]

    def metadata(self):
        return {
            "won": self.won,
            "regular_visits": self.regular_visits,
            "link_traversals": self.link_traversals,
            "distinct_link_tiles": self.distinct_link_tiles,
            "distinct_frames_entered": self.distinct_frames_entered,
            "repeated_frame_entries": self.repeated_frame_entries,
            "cycle_reentries": self.cycle_reentries,
            "cycle_guard_rejections": self.cycle_guard_rejections,
            "maximum_depth": self.maximum_depth,
            "fixed_regular_visits": self.fixed_regular_visits,
            "movable_link_visits": self.movable_link_visits,
            "emitted_colours": list(self.emitted),
        }


@dataclass(frozen=True)
class SearchResult:
    actions: list[Action] | None
    truncated: bool
    unsupported: bool
    exact: bool
    expanded: int
    reason: str
    energy_cost: int | None = None
    traversal: TraversalResult | None = None

    @property
    def solved(self):
        return self.actions is not None

    @property
    def length(self):
        return None if self.actions is None else len(self.actions)

    def metadata(self):
        return {
            "solved": self.solved,
            "truncated": self.truncated,
            "unsupported": self.unsupported,
            "exact": self.exact,
            "expanded": self.expanded,
            "reason": self.reason,
            "solution_length": self.length,
            "energy_cost": self.energy_cost,
        }


def _cap(value, field, default):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _assignments(signatures):
    """Yield unique multiset permutations without retaining a seen set."""
    counts = Counter(signatures)
    ordered = sorted(counts)
    output = []

    def visit():
        if len(output) == len(signatures):
            yield tuple(output)
            return
        for signature in ordered:
            if not counts[signature]:
                continue
            counts[signature] -= 1
            output.append(signature)
            yield from visit()
            output.pop()
            counts[signature] += 1

    yield from visit()


def assignment_from_layout(layout):
    """Return a complete slot assignment when every native slot is occupied."""
    assignment = {}
    for frame_index, frame in enumerate(layout.frames):
        for slot_index, tile in enumerate(frame.occupants):
            if tile is None:
                return None
            assignment[(frame_index, slot_index)] = tile
    return assignment


def traverse(layout, assignment):
    """Execute the native recursive grammar up to success or a proven failure."""
    if not layout.frames:
        return TraversalResult(False, "no root frame", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ())
    colour_to_frame = {frame.colour: index for index, frame in enumerate(layout.frames)}
    stack = [(0, 0)]
    emitted = []
    visited_states = set()
    entered = [0]
    link_cells = set()
    link_traversals = 0
    repeat_entries = 0
    cycle_reentries = 0
    guard_rejections = 0
    maximum_depth = 1
    fixed_regular = 0
    movable_links = 0
    reason = "traversal ended before all goals"

    while len(emitted) < len(layout.goals):
        state = tuple(stack), len(emitted)
        if state in visited_states:
            reason = "recursive traversal repeats without emitting a goal colour"
            break
        visited_states.add(state)
        frame_index, slot_index = stack[-1]
        frame = layout.frames[frame_index]
        if slot_index >= frame.arity:
            if len(stack) == 1:
                break
            stack.pop()
            parent_frame, parent_slot = stack[-1]
            stack[-1] = parent_frame, parent_slot + 1
            continue
        tile = assignment.get((frame_index, slot_index))
        if tile is None:
            reason = f"empty slot {frame_index}:{slot_index}"
            break
        if tile.kind == "regular":
            wanted = layout.goals[len(emitted)]
            if tile.colour != wanted:
                reason = f"colour {tile.colour} does not match goal {wanted}"
                break
            emitted.append(tile.colour)
            if not tile.movable:
                fixed_regular += 1
            stack[-1] = frame_index, slot_index + 1
            continue
        if tile.kind != "link" or tile.colour not in colour_to_frame:
            reason = f"invalid link at {frame_index}:{slot_index}"
            break
        # This is the engine's explicit first-cell recursion guard. Other
        # cycles are legal when the goal prefix completes before divergence.
        if (
            slot_index == 0
            and (frame_index, slot_index) in stack[:-1]
            and len(stack) > 1
            and stack[-2][1] == 0
        ):
            guard_rejections += 1
            reason = "native first-slot link-cycle guard rejects traversal"
            break
        target = colour_to_frame[tile.colour]
        link_traversals += 1
        link_cells.add((frame_index, slot_index))
        if tile.movable:
            movable_links += 1
        if target in entered:
            repeat_entries += 1
        if target in (entry[0] for entry in stack):
            cycle_reentries += 1
        entered.append(target)
        stack.append((target, 0))
        maximum_depth = max(maximum_depth, len(stack))

    won = len(emitted) == len(layout.goals)
    if won:
        reason = "goal prefix matched"
    return TraversalResult(
        won=won,
        reason=reason,
        regular_visits=len(emitted),
        link_traversals=link_traversals,
        distinct_link_tiles=len(link_cells),
        distinct_frames_entered=len(set(entered)),
        repeated_frame_entries=repeat_entries,
        cycle_reentries=cycle_reentries,
        cycle_guard_rejections=guard_rejections,
        maximum_depth=maximum_depth,
        fixed_regular_visits=fixed_regular,
        movable_link_visits=movable_links,
        emitted=tuple(emitted),
    )


def _plan_moves(layout, assignment):
    """Construct clicks that realize a satisfying signature assignment."""
    movable = [{"tile": tile, "position": tile.position} for tile in layout.movable_tiles]
    fixed_destinations = set()
    actions = []
    moves = 0
    if layout.selected is not None:
        actions.append((names.ACTION_CLICK, *names.click_at(layout.selected)))

    for frame_index, frame in enumerate(layout.frames):
        for slot_index, destination in enumerate(frame.slots):
            target = assignment[(frame_index, slot_index)]
            if not target.movable:
                fixed_destinations.add(destination)
                continue
            occupant = next(
                (index for index, entry in enumerate(movable) if entry["position"] == destination),
                None,
            )
            if occupant is not None and movable[occupant]["tile"].signature == target.signature:
                fixed_destinations.add(destination)
                continue
            candidates = [
                index
                for index, entry in enumerate(movable)
                if entry["tile"].signature == target.signature
                and entry["position"] not in fixed_destinations
            ]
            if not candidates:
                raise AssertionError("satisfying assignment has no movable source tile")
            source = next(
                (index for index in candidates if movable[index]["position"] not in layout.slots),
                candidates[0],
            )
            source_position = movable[source]["position"]
            actions.extend(
                (
                    (names.ACTION_CLICK, *names.click_at(source_position)),
                    (names.ACTION_CLICK, *names.click_at(destination)),
                )
            )
            moves += 1
            movable[source]["position"] = destination
            if occupant is not None:
                movable[occupant]["position"] = source_position
            fixed_destinations.add(destination)
    actions.append((names.ACTION_SUBMIT, None, None))
    return actions, moves


def search(env_or_layout, limit=512, budget=None, node_limit=500_000):
    """Find a winning placement under explicit action and assignment bounds.

    ``exact`` means the symbolic transition/traversal model covers this live
    state. It is deliberately not an optimality claim. A node cutoff is
    reported as truncated/unknown, never as unsolvable.
    """
    action_cap = _cap(limit, "limit", 512)
    work_cap = _cap(node_limit, "node_limit", 500_000)
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(
            None, False, True, False, 0,
            "unsupported layout: " + "; ".join(layout.unsupported),
        )

    fixed = {}
    open_slots = []
    for frame_index, frame in enumerate(layout.frames):
        for slot_index, occupant in enumerate(frame.occupants):
            if occupant is not None and not occupant.movable:
                fixed[(frame_index, slot_index)] = occupant
            else:
                open_slots.append((frame_index, slot_index))
    if len(open_slots) != len(layout.movable_tiles):
        return SearchResult(
            None, False, True, False, 0,
            f"unsupported tile/slot count: {len(layout.movable_tiles)} movable tiles for {len(open_slots)} open slots",
        )

    expanded = 0
    last_traversal = None
    signatures = tuple(tile.signature for tile in layout.movable_tiles)
    for permutation in _assignments(signatures):
        if expanded >= work_cap:
            return SearchResult(
                None, True, False, True, expanded,
                "node limit reached while assigning recursive frame slots",
                traversal=last_traversal,
            )
        expanded += 1
        assignment = dict(fixed)
        pools = {}
        for tile in layout.movable_tiles:
            pools.setdefault(tile.signature, []).append(tile)
        for slot, signature in zip(open_slots, permutation):
            assignment[slot] = pools[signature].pop()
        traversal = traverse(layout, assignment)
        last_traversal = traversal
        if not traversal.won:
            continue
        actions, moves = _plan_moves(layout, assignment)
        energy_cap = layout.energy
        if budget is not None:
            energy_cap = min(energy_cap, _cap(budget, "budget", energy_cap))
        energy_cost = moves + 1
        if moves >= energy_cap:
            return SearchResult(
                None, True, False, True, expanded,
                f"native energy bound {energy_cap} is insufficient for {moves} placements and submit",
                energy_cost,
                traversal,
            )
        if len(actions) > action_cap:
            return SearchResult(
                None, True, False, True, expanded,
                f"action limit {action_cap} is below the constructive route length {len(actions)}",
                energy_cost,
                traversal,
            )
        return SearchResult(
            actions, False, False, True, expanded, "solved constructive witness", energy_cost, traversal
        )
    return SearchResult(
        None, False, False, True, expanded,
        "exhaustive assignment search proved no winning placement",
        traversal=last_traversal,
    )


def solve(env_or_layout, limit=512, budget=None, node_limit=500_000):
    result = search(env_or_layout, limit=limit, budget=budget, node_limit=node_limit)
    solve.result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    solve.reason = result.reason
    return result.actions


solve.result = None
solve.truncated = False
solve.unsupported = False
solve.exact = None
solve.reason = "not run"
