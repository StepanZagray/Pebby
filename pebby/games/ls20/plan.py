"""Bounded exact LS20 planning from the current live engine state.

The native :class:`pebby.ls20.plan.Oracle` builds a policy for every state
reachable from a pristine level.  This adapter deliberately builds that policy
from a reset clone of the *current engine level*, then looks up the actual live
state.  That preserves consumed refills, solved goals, remaining budget and
moving-cycler phase instead of accidentally restarting a mid-level query.
"""

from dataclasses import dataclass

from arcengine import GameState

from pebby.ls20 import names
from pebby.ls20.layout import Layout, extract
from pebby.ls20.plan import Oracle, Unplannable, advance


DEFAULT_NODE_LIMIT = 600_000


@dataclass
class Result:
    actions: list | None
    truncated: bool
    expanded: int
    reason: str
    unsupported: bool = False

    @property
    def length(self):
        return None if self.actions is None else len(self.actions)


def _oracle_and_state(env_or_layout, node_limit):
    if isinstance(env_or_layout, Layout):
        oracle = Oracle(env_or_layout, limit=node_limit)
        return oracle, oracle.start, None

    if not hasattr(env_or_layout, "clone") or not hasattr(env_or_layout, "game"):
        raise TypeError("live LS20 search requires pebby.games.ls20.env.Env or a Layout")
    if env_or_layout.state != GameState.NOT_FINISHED:
        return None, None, f"cannot search terminal state {env_or_layout.state.value}"

    pristine = env_or_layout.clone()
    # level_reset retains the real engine index while restoring its clean level
    # and on_set_level state. That index controls LS20's first-level match hint.
    pristine.game.level_reset()
    layout = extract(pristine)
    oracle = Oracle(layout, limit=node_limit)
    return oracle, oracle.state_of(env_or_layout), None


def search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Find a live-state plan with explicit action and search-work bounds.

    ``limit`` caps returned actions (default: no cap beyond the modelled engine
    budget), while ``node_limit`` caps Oracle's reachable-state search. Every
    returned action has the shared ``(id, x, y)`` shape.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        return Result(None, False, 0, "action limit must be a positive integer")
    if isinstance(node_limit, bool) or not isinstance(node_limit, int) or node_limit < 1:
        return Result(None, True, 0, "node limit must be a positive integer")

    try:
        oracle, state, error = _oracle_and_state(env_or_layout, node_limit)
    except (Unplannable, ValueError) as exc:
        return Result(None, False, 0, str(exc), unsupported=True)
    if error is not None:
        return Result(None, False, 0, error)

    expanded = oracle._reachable
    distance = oracle.distance_for(state)
    if distance is None:
        reason = "live state has no proven path to completion"
        if oracle.truncated:
            reason += " before the node limit"
        return Result(None, oracle.truncated, expanded, reason)

    action_limit = distance if limit is None else limit
    if distance > action_limit:
        return Result(None, oracle.truncated, expanded, "no solution within the action limit")

    actions = []
    current = state
    while current[4] != oracle.full_mask:
        action_index = oracle.action_for(current)
        if action_index is None:
            return Result(None, oracle.truncated, expanded, "live policy has no next action")
        action_id = names.ACTION_IDS[action_index]
        actions.append((action_id, None, None))
        current = advance(oracle.layout, current, action_index, oracle.refills)
        if current is None:
            return Result(None, oracle.truncated, expanded, "live policy produced an invalid transition")
        if len(actions) > action_limit:
            return Result(None, oracle.truncated, expanded, "no solution within the action limit")

    if not actions:
        return Result(None, oracle.truncated, expanded, "level is already logically complete")
    return Result(actions, oracle.truncated, expanded, "solved")


def solve(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Return executable action triples or ``None``; retain result metadata."""
    result = search(env_or_layout, limit=limit, node_limit=node_limit)
    solve.result = result
    solve.truncated = result.truncated
    return result.actions


solve.result = None
solve.truncated = False
