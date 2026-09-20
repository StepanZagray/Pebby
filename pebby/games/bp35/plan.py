"""Native-replayed witnesses for official and generated BP35 levels.

Official routes cover all nine shipped levels. Generated routes are stored in
the private level descriptor after construction and native certification. A
search result is returned only after the proposed route wins from a clone of
the supplied live state. Bounded failures from a perturbed state remain
explicitly unsupported; they are never treated as impossibility proofs.
"""

from collections import deque
from dataclasses import dataclass

from . import names
from .env import replay
from .layout import Layout, extract


DEFAULT_LIMIT = 10_000
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
    limit: int

    @property
    def solved(self):
        return self.actions is not None


# Display coordinates are the exact camera-relative cells exercised against
# the immutable engine.  All lie in the public 64x64 click range.
OFFICIAL_LEVEL1: tuple[Action, ...] = (
    (4, None, None),
    (4, None, None),
    (4, None, None),
    (4, None, None),
    (6, 45, 33),
    (3, None, None),
    (3, None, None),
    (6, 27, 39),
    (3, None, None),
    (6, 27, 33),
    (6, 27, 33),
    (4, None, None),
    (6, 33, 33),
    (3, None, None),
    (3, None, None),
)


OFFICIAL_ROUTES: dict[int, tuple[Action, ...]] = {
    1: OFFICIAL_LEVEL1,
    2: ((6,21,33),(6,27,39),(4,None,None),(6,33,39),(4,None,None),(6,33,33),(6,39,39),(4,None,None),(6,33,39),(3,None,None),(6,15,33),(6,15,39),(6,21,39),(6,27,39),(3,None,None),(3,None,None),(3,None,None),(4,None,None),(4,None,None),(4,None,None),(6,33,33),(6,33,33),(6,21,33),(3,None,None),(3,None,None),(6,21,33),(6,21,33),(6,27,39),(4,None,None),(6,33,39),(4,None,None),(6,39,39),(6,45,39),(6,51,33),(6,51,39),(4,None,None),(4,None,None),(4,None,None),(6,51,33),(6,51,33),(6,45,39),(3,None,None),(3,None,None),(3,None,None),(6,33,33)),
    3: ((4,None,None),(6,33,39),(4,None,None),(4,None,None),(6,39,33),(6,21,33),(6,27,33),(6,33,33),(3,None,None),(3,None,None),(3,None,None),(3,None,None),(4,None,None),(4,None,None),(6,33,33),(6,33,39),(4,None,None),(6,39,33),(6,39,39),(4,None,None),(4,None,None),(6,21,33),(6,27,33),(6,33,33),(6,39,33),(3,None,None),(3,None,None),(3,None,None),(3,None,None),(4,None,None),(6,33,39),(4,None,None),(4,None,None),(4,None,None)),
    4: ((4,None,None),(4,None,None),(6,33,3),(3,None,None),(3,None,None),(6,21,35),(3,None,None),(4,None,None),(4,None,None),(6,21,41),(4,None,None),(6,33,57),(4,None,None),(6,45,35),(6,45,35),(3,None,None),(3,None,None),(3,None,None),(6,27,41)),
    5: ((4,None,None),(4,None,None),(4,None,None),(6,51,39),(6,45,35),(4,None,None),(6,51,41),(4,None,None),(3,None,None),(6,51,59),(3,None,None),(3,None,None),(6,21,33),(6,27,33),(3,None,None),(3,None,None),(3,None,None),(4,None,None),(4,None,None),(4,None,None),(6,39,33),(4,None,None),(4,None,None),(4,None,None),(4,None,None),(6,57,33),(6,51,39),(3,None,None),(6,45,39),(3,None,None),(3,None,None),(3,None,None)),
    6: ((4,None,None),(4,None,None),(4,None,None),(4,None,None),(4,None,None),(6,39,33),(6,27,29),(3,None,None),(3,None,None),(4,None,None),(4,None,None),(3,None,None),(3,None,None),(6,33,33),(3,None,None),(3,None,None),(4,None,None),(4,None,None),(4,None,None),(6,51,3),(3,None,None),(3,None,None),(6,27,59),(3,None,None),(6,39,35),(4,None,None),(4,None,None),(3,None,None),(3,None,None),(4,None,None),(6,39,53),(4,None,None),(6,45,35),(4,None,None),(4,None,None),(3,None,None),(3,None,None),(3,None,None),(3,None,None),(3,None,None),(3,None,None)),
    7: ((6,3,39),(6,39,17),(4,None,None),(4,None,None),(4,None,None),(6,3,29),(4,None,None),(4,None,None),(3,None,None),(3,None,None),(6,3,39),(3,None,None),(3,None,None),(6,27,17),(6,3,29),(3,None,None),(3,None,None),(4,None,None),(6,3,39),(6,27,17),(4,None,None),(6,3,29),(6,33,39),(4,None,None),(6,33,51),(6,3,33),(6,39,29),(4,None,None),(6,39,17),(6,3,23),(6,45,15),(4,None,None),(6,45,45),(6,3,33),(4,None,None),(4,None,None),(3,None,None),(3,None,None),(6,3,29),(3,None,None),(3,None,None),(3,None,None),(3,None,None),(6,3,39)),
    8: ((4,None,None),(6,15,21),(6,21,21),(6,27,21),(4,None,None),(6,33,33),(6,39,39),(4,None,None),(6,45,39),(4,None,None),(6,51,39),(4,None,None),(6,51,33),(6,51,33),(6,51,33),(6,51,33),(6,45,39),(3,None,None),(3,None,None),(6,21,33),(3,None,None),(6,21,9),(3,None,None),(3,None,None),(6,27,27),(4,None,None),(4,None,None),(6,33,33),(6,33,33),(6,33,33),(6,27,45),(6,33,33),(6,33,33),(6,33,33),(6,33,33),(6,33,33),(6,33,33),(6,33,33),(6,33,33),(6,33,3),(6,33,35),(6,39,29),(4,None,None),(6,45,29),(4,None,None),(6,51,29),(4,None,None),(6,51,35),(4,None,None)),
    # External demonstrated positive witness from the public ARC3.Games
    # getTopSequences record for bp35-0a0ad940 level 9, retrieved 2026-09-18.
    # It is reference verification only, not generator input or an optimality
    # claim. search() always certifies it again against the pinned local engine.
    9: ((6,38,16),(4,None,None),(4,None,None),(6,33,33),(6,33,33),(6,33,33),(6,27,39),(3,None,None),(6,27,33),(6,27,26),(6,28,33),(6,27,33),(6,27,33),(6,28,33),(6,28,33),(6,28,33),(6,27,33),(6,34,39),(4,None,None),(6,39,40),(4,None,None),(6,3,39),(6,40,35),(6,40,36),(6,46,30),(4,None,None),(6,51,30),(4,None,None),(6,57,29),(4,None,None),(6,56,24),(6,3,41),(6,56,33),(6,57,32),(6,57,32),(6,57,32),(6,57,32),(6,57,33),(6,52,39),(3,None,None),(6,46,39),(3,None,None),(6,39,40),(3,None,None),(6,33,40),(3,None,None),(6,27,39),(3,None,None),(6,28,33),(6,44,3),(6,38,3),(6,31,3),(6,25,3),(6,20,3),(3,None,None),(3,None,None),(6,15,34),(6,9,39),(3,None,None),(3,None,None),(6,10,3),(6,3,35),(6,15,3),(6,3,35),(6,21,3),(6,4,34),(6,27,3),(6,3,35),(6,33,3),(4,None,None),(4,None,None)),
}

OFFICIAL_REFERENCE_PROVENANCE = {
    9: {
        "kind": "external_demonstrated_positive_witness",
        "source": "https://arc3.games/api/",
        "request": "action=getTopSequences&gameId=bp35-0a0ad940&level=9&order=ASC&limit=3",
        "retrieved": "2026-09-18",
        "native_source_id": "bp35-0a0ad940",
        "vendored_source_sha256": "e9aecb52c629c3e742276c2db04b81f91555051d8617b6cbc58bac410824867f",
        "locally_replayed": True,
        "optimality_claimed": False,
        "usage": "official reference verification only; excluded from generation and training",
    }
}


def _history_depth_env(env):
    history = getattr(env.world, names.ATTR_HISTORY, None)
    return len(getattr(history, names.ATTR_HISTORY_STACK, ())) if history is not None else 0


def _positive_replay(env, actions):
    probe = env.clone()
    completed, _ = replay(probe, actions)
    return completed


def _stored_generated_search(env, descriptor, limit, action_budget):
    raw = descriptor.get("solution")
    if not isinstance(raw, list):
        return None
    stored = []
    for triple in raw:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return SearchResult(None, False, True, False, 0, 0, "stored generated route is malformed", limit)
        action, x, y = triple
        if type(action) is not int or action not in names.AVAILABLE_ACTIONS:
            return SearchResult(None, False, True, False, 0, 0, "stored generated route is malformed", limit)
        if action == names.ACTION_CLICK:
            if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
                return SearchResult(None, False, True, False, 0, 0, "stored generated route is malformed", limit)
        elif x is not None or y is not None:
            return SearchResult(None, False, True, False, 0, 0, "stored generated route is malformed", limit)
        stored.append((action, x, y))
    stored = tuple(stored)
    if action_budget == 0:
        return SearchResult(
            None, False, False, True, 0, 0,
            "unfinished level cannot complete within a zero-action budget", limit,
        )
    if limit == 0:
        pristine = env.action_count == 0 and _history_depth_env(env) == 0
        return SearchResult(
            None, True, not pristine, pristine, 0, 1,
            "generated recovery candidate limit reached", limit,
        )
    candidates = []
    for suffix in range(len(stored) + 1):
        candidates.append(stored[suffix:])
    history = min(_history_depth_env(env), action_budget)
    for undos in range(1, history + 1):
        prefix = ((names.ACTION_UNDO, None, None),) * undos
        candidates.append(prefix + stored)
    expanded = 0
    for actions in candidates:
        if expanded >= limit:
            return SearchResult(None, True, True, False, expanded, len(candidates), "generated recovery candidate limit reached", limit)
        expanded += 1
        if len(actions) <= action_budget and _positive_replay(env, actions):
            return SearchResult(actions, False, False, True, expanded, len(candidates), "real-engine replay verified stored generated route or recovery suffix", limit)
    return SearchResult(None, False, True, False, expanded, len(candidates), "no stored-route suffix or bounded UNDO recovery completed this live state", limit)


def _transition(layout, position, dx):
    """Apply one horizontal move in the proved ordinary-platform subset."""
    x, y = position
    target = (x + dx, y)
    if not (0 <= target[0] < layout.width):
        return position, False, True
    if target == layout.gem:
        return target, True, True
    if target in layout.walls:
        return position, False, True
    # Remote hazards are passable in BP35 and the generated safety gap proves
    # they cannot enter the region reached by this search.
    landing = target
    for _ in range(layout.height + 1):
        below = (landing[0], landing[1] - 1)
        if below == layout.gem:
            return below, True, True
        if below in layout.walls:
            return landing, False, True
        if below[1] < 0:
            return position, False, False
        landing = below
    return position, False, False


def _inconclusive(layout):
    return not layout.generated_pristine


def _generated_search(layout, limit, action_budget):
    start = layout.player
    queue = deque([(start, ())])
    best = {start: 0}
    expanded = 0
    generated = 1
    depth_cutoff = False
    while queue:
        position, path = queue.popleft()
        if expanded >= limit:
            live = _inconclusive(layout)
            return SearchResult(
                None, True, live, not live, expanded, generated,
                f"configuration expansion limit {limit} reached"
                + ("; live-history negative result is inconclusive" if live else ""),
                limit,
            )
        expanded += 1
        if len(path) >= action_budget:
            depth_cutoff = True
            continue
        for action_id, dx in ((names.ACTION_LEFT, -1), (names.ACTION_RIGHT, 1)):
            next_position, won, safe = _transition(layout, position, dx)
            if not safe:
                return SearchResult(
                    None, False, True, False, expanded, generated,
                    "generated map admits an unbounded gravity fall", limit,
                )
            if next_position == position and not won:
                continue
            actions = path + ((action_id, None, None),)
            if won:
                probe = layout.snapshot.clone()
                completed, _ = replay(probe, actions)
                if completed:
                    return SearchResult(
                        actions, False, False, True, expanded, generated,
                        "real-engine replay verified generated witness", limit,
                    )
                return SearchResult(
                    None, False, True, False, expanded, generated,
                    "symbolic witness disagreed with the real engine", limit,
                )
            cost = len(actions)
            if cost < best.get(next_position, action_budget + 1):
                best[next_position] = cost
                generated += 1
                queue.append((next_position, actions))
    reason = (
        f"no solution within action budget {action_budget}"
        if depth_cutoff
        else "supported finite position graph exhausted"
    )
    live = _inconclusive(layout)
    if live:
        reason += "; live-history UNDO branches were not searched, so failure is inconclusive"
    return SearchResult(None, False, live, not live, expanded, generated, reason, limit)


def search(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    """Return an exact real-engine witness or a classified failure.

    ``limit`` bounds expanded symbolic configurations. ``budget`` optionally
    lowers the native action cap, matching the shared collector's signature.
    Positive generated witnesses are exact because they are replayed from the
    supplied live snapshot.  A failed live-history search is inconclusive.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if budget is not None and (
        not isinstance(budget, int) or isinstance(budget, bool) or budget < 0
    ):
        raise ValueError("budget must be a nonnegative integer or None")
    if not isinstance(env_or_layout, Layout):
        descriptor = env_or_layout.generated_descriptor
        action_budget = env_or_layout.steps_left if budget is None else min(env_or_layout.steps_left, budget)
        if isinstance(descriptor, dict) and descriptor.get("kind") == names.GENERATED_KIND:
            generated = _stored_generated_search(env_or_layout, descriptor, limit, action_budget)
            if generated is not None:
                return generated
        if (
            descriptor is None
            and env_or_layout.level_index + 1 in OFFICIAL_ROUTES
            and env_or_layout.action_count == 0
            and _history_depth_env(env_or_layout) == 0
        ):
            if limit == 0:
                return SearchResult(
                    None, True, False, True, 0, 1,
                    "official witness lookup exceeds configuration limit 0", limit,
                )
            route = OFFICIAL_ROUTES[env_or_layout.level_index + 1]
            if len(route) > action_budget:
                return SearchResult(None, True, False, True, 0, 1, "pinned official witness exceeds the requested budget", limit)
            if _positive_replay(env_or_layout, route):
                return SearchResult(route, False, False, True, 1, 1, "real-engine replay verified pinned official witness", limit)
            return SearchResult(None, False, True, False, 1, 1, "pinned official witness disagreed with the engine", limit)
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if not layout.exact:
        return SearchResult(
            None, False, True, False, 0, 0,
            "; ".join(layout.unsupported) or "unsupported BP35 state", limit,
        )
    if limit == 0:
        live = layout.generated_exact and not layout.generated_pristine
        return SearchResult(
            None, True, live, not live, 0, 1,
            "configuration expansion limit 0 reached"
            + ("; live-history negative result is inconclusive" if live else ""),
            limit,
        )
    action_budget = layout.action_budget
    if budget is not None:
        action_budget = min(action_budget, budget)

    if layout.official1_initial:
        if action_budget < len(OFFICIAL_LEVEL1):
            return SearchResult(
                None, True, False, True, 0, 1,
                "official witness exceeds the requested action budget; shorter paths were not searched",
                limit,
            )
        probe = layout.snapshot.clone()
        completed, _ = replay(probe, OFFICIAL_LEVEL1)
        if not completed:
            return SearchResult(
                None, False, True, False, 1, 1,
                "pinned official witness no longer agrees with the real engine", limit,
            )
        return SearchResult(
            OFFICIAL_LEVEL1, False, False, True, 1, 1,
            "real-engine replay verified official level 1 witness", limit,
        )
    return _generated_search(layout, limit, action_budget)


def solve(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    result = search(env_or_layout, limit=limit, budget=budget)
    solve.last_result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    return list(result.actions) if result.actions is not None else None


solve.last_result = None
solve.truncated = False
solve.unsupported = False
solve.exact = True
