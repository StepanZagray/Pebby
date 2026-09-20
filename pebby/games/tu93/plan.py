"""Exact bounded planner for TU93.

The search expands one logical state per accepted action.  It models all three
enemy types in the same order as upstream ``Tu93.step``: the head moves and
eats occupants, hunters lunge, patrollers and armed tails move, patrollers
bounce, and tails update their delayed rotation queue.  Blocked actions are
not expanded because they only consume budget and leave every sprite fixed.
"""

from collections import deque
from dataclasses import dataclass

from . import names
from .layout import Layout, extract


State = tuple[
    tuple[int, int],
    int,
    tuple[tuple[tuple[int, int], int], ...],
    tuple[tuple[tuple[int, int], int], ...],
    tuple[tuple[tuple[int, int], int, tuple[int, ...] | None], ...],
]


@dataclass(frozen=True)
class SearchResult:
    actions: list[tuple[int, None, None]] | None
    truncated: bool
    expanded: int
    generated: int
    reason: str
    exact: bool = True

    @property
    def solved(self):
        return self.actions is not None

    def metadata(self):
        return {
            "solved": self.solved,
            "truncated": self.truncated,
            "expanded": self.expanded,
            "generated": self.generated,
            "reason": self.reason,
            "exact": self.exact,
            "solution_length": None if self.actions is None else len(self.actions),
        }


def _advance(cell, rotation, distance=1):
    dx, dy = names.ROTATION_DELTA[rotation]
    return cell[0] + distance * dx, cell[1] + distance * dy


def _facing_at(actor, target, distance):
    cell, rotation = actor
    return _advance(cell, rotation, distance) == target


def _open_rotation(layout, cell, rotation):
    action = names.ROTATION_ACTION[rotation]
    return layout.open(cell, action)


def transition_with_events(layout: Layout, state: State, action: int):
    """Apply one action and return ``(outcome, event_counts)``.

    ``outcome`` has the same shape as :func:`transition`.  The event counts are
    derived while applying the rules, rather than inferred from a rendered
    frame, and are used by full-standard generation to prove that a winning
    witness actually exercises its required mechanics.

    ``None`` safely prunes blocked actions: upstream charges a step but performs
    no enemy phase, so repeating the same state with less budget cannot help.
    """
    events = {
        "accepted_move": 0,
        "eaten_hunters": 0,
        "eaten_patrollers": 0,
        "eaten_tails": 0,
        "patroller_moves": 0,
        "patroller_bounces": 0,
        "tail_arms": 0,
        "tail_moves": 0,
        "fatal_hunter": 0,
        "fatal_patroller": 0,
        "fatal_tail": 0,
    }
    head, _, hunters, patrollers, tails = state
    if action not in names.ACTION_IDS:
        raise ValueError(f"action must be one of {names.ACTION_IDS}")
    if not layout.open(head, action):
        return None, events
    rotation = names.ACTION_ROTATION[action]
    head = _advance(head, rotation)
    events["accepted_move"] = 1

    # The head consumes every entity sharing its destination before enemies
    # start their phase (the engine may animate this over frames, but the
    # action-level outcome is removal).
    events["eaten_hunters"] = sum(actor[0] == head for actor in hunters)
    events["eaten_patrollers"] = sum(actor[0] == head for actor in patrollers)
    events["eaten_tails"] = sum(tail[0] == head for tail in tails)
    hunters = tuple(actor for actor in hunters if actor[0] != head)
    patrollers = tuple(actor for actor in patrollers if actor[0] != head)
    tails = tuple(tail for tail in tails if tail[0] != head)

    # Existing armed tails receive this move before their current move starts.
    queued_tails = []
    for cell, tail_rotation, queue in tails:
        queued_tails.append(
            (cell, tail_rotation, None if queue is None else queue + (rotation,))
        )
    tails = tuple(queued_tails)

    # A lunge is inevitably fatal at this action boundary.  Hunters never turn
    # or otherwise move, so a non-lunging one stays fixed.
    if any(_facing_at(hunter, head, 1) for hunter in hunters):
        events["fatal_hunter"] = 1
        return ((head, rotation, hunters, patrollers, tails), False, True), events

    moved_patrollers = tuple(
        (_advance(cell, enemy_rotation), enemy_rotation)
        for cell, enemy_rotation in patrollers
    )
    events["patroller_moves"] = len(moved_patrollers)
    if any(cell == head for cell, _ in moved_patrollers):
        events["fatal_patroller"] = 1
        return ((head, rotation, hunters, moved_patrollers, tails), False, True), events

    moved_tails = []
    for cell, tail_rotation, queue in tails:
        moved = _advance(cell, tail_rotation) if queue is not None else cell
        events["tail_moves"] += queue is not None
        moved_tails.append((moved, tail_rotation, queue))
    if any(cell == head for cell, _, queue in moved_tails if queue is not None):
        events["fatal_tail"] = 1
        return (
            (head, rotation, hunters, moved_patrollers, tuple(moved_tails)),
            False,
            True,
        ), events

    # Patrollers reverse after landing when the next passage is closed.
    bounced = []
    for cell, enemy_rotation in moved_patrollers:
        if not _open_rotation(layout, cell, enemy_rotation):
            enemy_rotation = names.OPPOSITE[enemy_rotation]
            events["patroller_bounces"] += 1
        bounced.append((cell, enemy_rotation))

    # Dormant tails arm only after all moving entities settle.  Every active
    # queue then pops its front rotation into the sprite, including a newly
    # armed [r, r] queue.
    advanced_tails = []
    for cell, tail_rotation, queue in moved_tails:
        if queue is None and _facing_at((cell, tail_rotation), head, 2):
            queue = (tail_rotation, tail_rotation)
            events["tail_arms"] += 1
        if queue is not None and queue:
            tail_rotation, queue = queue[0], queue[1:]
        advanced_tails.append((cell, tail_rotation, queue))

    nxt = (
        head,
        rotation,
        hunters,
        tuple(bounced),
        tuple(advanced_tails),
    )
    return (nxt, head in layout.exits, False), events


def transition(layout: Layout, state: State, action: int):
    """Apply one action. Return ``(state, won, dead)`` or ``None`` if blocked."""
    outcome, _ = transition_with_events(layout, state, action)
    return outcome


def _unwind(parent, state):
    actions = []
    while parent[state] is not None:
        state, action = parent[state]
        actions.append((action, None, None))
    actions.reverse()
    return actions


def search(env_or_layout, limit=200_000, budget=None):
    """Breadth-first exact search bounded by expansions and remaining steps.

    ``None`` with ``truncated=False`` proves no solution exists within the game
    budget, but only for an exact/supported layout.  Unsupported snapshots use
    ``truncated=True`` and ``exact=False`` so callers cannot mistake them for a
    proof of unsolvability.
    """
    if limit is None or int(limit) < 0:
        raise ValueError("limit must be a non-negative node cap")
    limit = int(limit)
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(
            None,
            True,
            0,
            0,
            "unsupported layout: " + "; ".join(layout.unsupported),
            exact=False,
        )
    if layout.head is None or not layout.exits:
        return SearchResult(None, True, 0, 0, "unsupported layout: missing head or exit", exact=False)
    remaining = layout.steps_left if budget is None else min(int(budget), layout.steps_left)
    if remaining <= 0:
        return SearchResult(None, False, 0, 1, "no actions remain in the step budget")

    start: State = layout.key()
    frontier = deque([(start, 0)])
    parent = {start: None}
    expanded = 0
    generated = 1
    while frontier:
        state, depth = frontier.popleft()
        if expanded >= limit:
            return SearchResult(None, True, expanded, generated, "expansion limit reached")
        expanded += 1
        if depth >= remaining:
            continue
        for action in names.ACTION_IDS:
            outcome = transition(layout, state, action)
            if outcome is None:
                continue
            nxt, won, dead = outcome
            if dead:
                continue
            if won:
                # Reconstruct through the current state and append the action
                # that actually triggers Tu93's post-entity win check.  This
                # matters when a level starts on its exit and a cycle returns
                # to the already-seen start state: parent[start] must remain
                # None, but the winning move must not disappear.
                actions = _unwind(parent, state)
                actions.append((action, None, None))
                return SearchResult(
                    actions, False, expanded, generated + (nxt not in parent), "solved"
                )
            if nxt not in parent:
                parent[nxt] = (state, action)
                generated += 1
                frontier.append((nxt, depth + 1))
    return SearchResult(None, False, expanded, generated, "search space exhausted within budget")


def solve(env_or_layout, limit=200_000):
    """Return action triples completing the current level, or ``None``.

    ``solve.result`` preserves the detailed independent result and
    ``solve.truncated`` distinguishes an inconclusive cap/unsupported state
    from an exact no-solution proof.
    """
    result = search(env_or_layout, limit=limit)
    solve.result = result
    solve.truncated = result.truncated
    return result.actions


solve.result = None
solve.truncated = False
