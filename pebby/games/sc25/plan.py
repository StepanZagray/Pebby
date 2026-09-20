"""Exact bounded symbolic planner for SC25.

The search works on the logical state extracted by :mod:`layout`; it does not
advance a copied game as an oracle.  Movement, scale changes, teleport cursors,
fire targets and block families, pickups, the tutorial demo, facing, and the
level's unusual budget accounting are all explicit transitions.

Grid input is represented by the shortest safe toggle sequence from the live
3x3 bitmap to each available spell. The tiny 512-mask graph is searched
exactly, stopping at any spell that would auto-cast. This keeps normal search
compact while supporting arbitrary non-casting live prefixes. ``None`` is a
proof of exhaustion only when ``truncated`` and ``unsupported`` are both false.
"""

from collections import deque
from dataclasses import dataclass

from . import names
from .layout import (
    KIND_BLOCK,
    KIND_BLOCK_ALT,
    KIND_TARGET,
    KIND_TARGET_ALT,
    Layout,
    Unplannable,
    extract,
)


DEFAULT_LIMIT = 200_000


@dataclass
class Result:
    actions: list[tuple[int, int | None, int | None]] | None = None
    expanded: int = 0
    truncated: bool = False
    unsupported: bool = False
    reason: str = ""

    @property
    def solved(self):
        return self.actions is not None


def _core(state):
    """State without budget used; lower used dominates for an identical core."""
    return state[:7] + state[8:]


def _pickup(layout, state, width, height):
    x, y, scale, facing, pad_i, small_i, removed, used, demo, grid = state
    item = layout.pickup_at(x, y, width, height, removed)
    if item is not None:
        removed |= 1 << item.bit
        used = max(0, used - names.PICKUP_REFUND)
    return (x, y, scale, facing, pad_i, small_i, removed, used, demo, grid)


def _move(layout, state, action):
    x, y, scale, _, pad_i, small_i, removed, used, demo, grid = state
    facing = names.MOVE_FACING[action]
    dx, dy = names.MOVE_DELTAS[action]
    step = 2 if scale == 1 else 4

    nx, ny = x + dx * step, y + dy * step
    blocked, door = layout.blocked(nx, ny, scale, removed)
    if blocked:
        nx, ny = x, y
        # A scale-2 player retries a colliding four-pixel move at two pixels.
        # Only the retry's collisions are subsequently considered by upstream.
        if scale == 2:
            hx, hy = x + dx * 2, y + dy * 2
            half_blocked, half_door = layout.blocked(hx, hy, scale, removed)
            door = half_door
            if not half_blocked:
                nx, ny = hx, hy
                door = False
    else:
        door = False

    if door:
        return None, True

    used += 1
    if layout.budget is not None and used > layout.budget:
        return None, False
    nxt = (nx, ny, scale, facing, pad_i, small_i, removed, used, demo, grid)
    # Movement's pickup overlap accidentally uses the unscaled 2x2 raw player
    # dimensions in upstream (sc25.py:2721-2729).
    return _pickup(layout, nxt, 2, 2), False


def _spell_actions(spell):
    return tuple((6, *names.cell_click(row, col)) for row, col in names.PATTERNS[spell])


_SPELL_MASKS = {
    spell: sum(1 << (row * 3 + col) for row, col in cells)
    for spell, cells in names.PATTERNS.items()
}


def _entry_actions(layout, grid, target_spell):
    """Shortest toggles reaching a spell without auto-casting another spell."""
    target = _SPELL_MASKS[target_spell]
    if grid == target:
        return ()
    spell_masks = {_SPELL_MASKS[spell] for spell in layout.spells}
    frontier = deque([grid])
    parent = {grid: None}
    while frontier:
        mask = frontier.popleft()
        for bit in range(9):
            nxt = mask ^ (1 << bit)
            if nxt in parent:
                continue
            parent[nxt] = (mask, bit)
            if nxt == target:
                bits = []
                cursor = nxt
                while parent[cursor] is not None:
                    cursor, changed = parent[cursor]
                    bits.append(changed)
                bits.reverse()
                return tuple(
                    (6, *names.cell_click(changed // 3, changed % 3))
                    for changed in bits
                )
            if nxt not in spell_masks:
                frontier.append(nxt)
    return None


def _cast(layout, state, spell):
    x, y, scale, facing, pad_i, small_i, removed, used, demo, grid = state
    actions = _entry_actions(layout, grid, spell)
    if actions is None:
        return None

    # Each cell toggle is charged before the cast animation. The animation's
    # completion charges one further unit, after a growth pickup refund.
    used_after_clicks = used + len(actions)
    if layout.budget is not None and used_after_clicks > layout.budget:
        return None
    used = used_after_clicks

    if spell == names.SPELL_TELEPORT:
        pads = layout.small_pads if scale == 1 else layout.pads
        index = small_i if scale == 1 else pad_i
        if not pads:
            return None  # upstream never starts a completing teleport animation
        x, y = pads[index]
        index = (index + 1) % len(pads)
        if scale == 1:
            small_i = index
        else:
            pad_i = index

    elif spell == names.SPELL_GROW:
        old = (x, y, scale)
        if scale == 2:
            scale = 1
        else:
            position = None
            for ox in range(-2, 1):
                for oy in range(-2, 1):
                    if not layout.grow_blocked(x + ox, y + oy, removed):
                        position = (x + ox, y + oy)
                        break
                if position is not None:
                    break
            if position is None:
                return None  # blocked cast only spends budget
            x, y = position
            scale = 2
        if (x, y, scale) == old:
            return None
        grown = (x, y, scale, facing, pad_i, small_i, removed, used, demo, 0)
        grown = _pickup(layout, grown, 2 * scale, 2 * scale)
        x, y, scale, facing, pad_i, small_i, removed, used, demo, _ = grown

    elif spell == names.SPELL_FIRE:
        hit = layout.fire_hit(x, y, scale, facing, removed)
        if hit is None or hit.kind not in (KIND_TARGET, KIND_TARGET_ALT):
            return None  # a miss cannot improve any logical state
        removed |= 1 << hit.bit
        block_kind = KIND_BLOCK if hit.kind == KIND_TARGET else KIND_BLOCK_ALT
        for item in layout.removables:
            if item.kind == block_kind:
                removed |= 1 << item.bit
    else:  # guarded by the caller, retained as an explicit unsupported seam
        return None

    used += 1
    if layout.budget is not None and used > layout.budget:
        return None
    return (x, y, scale, facing, pad_i, small_i, removed, used, demo, 0)


def _unwind(parent, state):
    chunks = []
    while parent[state] is not None:
        previous, actions = parent[state]
        chunks.append(actions)
        state = previous
    chunks.reverse()
    return [action for chunk in chunks for action in chunk]


def search(env_or_layout, limit=DEFAULT_LIMIT):
    """Search the current level, returning a :class:`Result`.

    ``truncated`` means the work limit stopped a still-live frontier.
    ``unsupported`` means the live state could not be represented exactly.
    Otherwise an actions value of ``None`` means the finite budget/state graph
    was exhausted without reaching the door.
    """
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    try:
        layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    except (Unplannable, AttributeError, ValueError) as exc:
        return Result(unsupported=True, reason=str(exc))

    unknown = [spell for spell in getattr(layout, "raw_spells", layout.spells)
               if spell not in names.PATTERNS]
    if unknown:
        return Result(unsupported=True, reason=f"unknown spell mechanics: {unknown}")
    if not layout.door_present:
        return Result(reason="level has no exit door")
    if layout.budget is not None and layout.start[7] > layout.budget:
        return Result(reason="action budget is already exceeded")

    start = layout.start
    if layout.budget is None:
        start = start[:7] + (0,) + start[8:]
    frontier = deque([start])
    parent = {start: None}
    best_used = {_core(start): start[7]}
    expanded = 0

    def enqueue(previous, nxt, actions):
        if nxt is None:
            return
        if layout.budget is None:
            nxt = nxt[:7] + (0,) + nxt[8:]
        core = _core(nxt)
        if nxt[7] >= best_used.get(core, 1 << 60):
            return
        best_used[core] = nxt[7]
        parent[nxt] = (previous, actions)
        frontier.append(nxt)

    while frontier:
        if expanded >= limit:
            return Result(expanded=expanded, truncated=True, reason="node expansion limit reached")
        state = frontier.popleft()
        expanded += 1

        if state[8]:
            nxt = state[:8] + (False, state[9])
            enqueue(state, nxt, ((1, None, None),))
            continue

        for action in names.MOVE_ACTIONS:
            nxt, won = _move(layout, state, action)
            edge = ((action, None, None),)
            if won:
                return Result(actions=_unwind(parent, state) + list(edge), expanded=expanded)
            if nxt is not None and nxt != state:
                enqueue(state, nxt, edge)

        for spell in layout.spells:
            actions = _entry_actions(layout, state[9], spell)
            if actions is None:
                continue
            nxt = _cast(layout, state, spell)
            if nxt is not None and nxt != state:
                enqueue(state, nxt, actions)

    return Result(expanded=expanded, reason="exhausted reachable states within the action budget")


def solve(env_or_layout, limit=DEFAULT_LIMIT):
    """Return action triples completing this level, or ``None``.

    Inspect ``solve.truncated``, ``solve.unsupported``, and ``solve.result`` to
    distinguish a work cutoff or unsupported live state from proved exhaustion.
    """
    result = search(env_or_layout, limit)
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.result = result
    return result.actions


solve.truncated = False
solve.unsupported = False
solve.result = None
