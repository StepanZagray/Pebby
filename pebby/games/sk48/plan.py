"""Bounded stable-state SK48 search with mandatory native positive replay."""

from dataclasses import dataclass

from arcengine import GameState

from . import names
from .layout import Layout, extract, unsupported_conditions
from .model import beam_search as model_beam_search
from .model import extract_model, search as model_search


DEFAULT_NODE_LIMIT = 30_000
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


def _state_key(env):
    lines = env.lines()
    heads = list(lines)
    selected = heads.index(env.selected())
    line_state = tuple(
        (
            int(head.x), int(head.y), int(head.rotation),
            tuple((int(segment.x), int(segment.y), int(segment.rotation))
                  for segment in lines[head]),
        )
        for head in heads
    )
    colors = tuple(
        (int(pad.x), int(pad.y), int(pad.pixels[1, 1]))
        for pad in env.color_pads()
    )
    head_index = {id(head): index for index, head in enumerate(heads)}
    pads = env.color_pads()
    pad_index = {id(pad): index for index, pad in enumerate(pads)}
    history = []
    for snapshot in getattr(env.game, names.ATTR_HISTORY):
        saved = []
        for sprite, x, y, length in snapshot:
            identity = id(sprite)
            if identity in head_index:
                kind_index = ("head", head_index[identity])
            elif identity in pad_index:
                kind_index = ("pad", pad_index[identity])
            else:  # Layout validation makes this unreachable for supported levels.
                kind_index = ("unknown", identity)
            saved.append((*kind_index, int(x), int(y), int(length)))
        history.append(tuple(saved))
    return line_state, colors, selected, env.moves_left, tuple(history)


def search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Search the live level with separate action and configuration bounds."""
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

    # The compact model mirrors one complete stable native action and was
    # parity-checked across every official mechanic.  It avoids retaining a
    # deep-copied render engine at every frontier node.  A positive route is
    # still replayed below in a fresh native clone before it becomes evidence.
    native = layout.snapshot.clone()
    model = extract_model(native)
    action_limit = limit if limit is not None else names.MOVE_BUDGET
    if model.blockers and node_limit >= 500_000:
        actions, expanded, truncated = model_beam_search(
            model, node_limit=node_limit, action_limit=action_limit,
        )
    else:
        actions, expanded, truncated = model_search(
            model, node_limit=node_limit, action_limit=action_limit,
        )
    if actions is not None:
        replay_env = layout.snapshot.clone()
        before = replay_env.levels_completed
        won = False
        for action_id, x, y in actions:
            observation = replay_env.perform(action_id, x, y)
            if replay_env.levels_completed > before or observation.state == GameState.WIN:
                won = True
                break
            if observation.state == GameState.GAME_OVER:
                break
        if won:
            return SearchResult(
                tuple(actions), False, False, True, expanded, expanded,
                "stable-state route completed a fresh native replay", node_limit,
            )
        return SearchResult(
            None, False, True, False, expanded, expanded,
            "stable-state route diverged during mandatory native replay", node_limit,
        )
    return SearchResult(
        None, bool(truncated), False, False, expanded, expanded,
        (f"bounded stable-state search reached work limit {node_limit}"
         if truncated else
         "no positive witness found; undo/history-complete impossibility was not proved"),
        node_limit,
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
