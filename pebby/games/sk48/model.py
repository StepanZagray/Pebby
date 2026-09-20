"""Fast stable-state model of SK48, checked by native replay before use.

The vendored game animates a six-pixel lattice move over two engine ticks.
``Env.perform`` drains those ticks, so the public action semantics are a
stable-state transition.  This module mirrors that transition without render
objects.  It is a route-finding accelerator only: accepted certificates are
always replayed in a fresh unmodified native game.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count

from . import names


@dataclass(frozen=True)
class State:
    heads: tuple[tuple[int, int], ...]
    lines: tuple[tuple[tuple[int, int, int], ...], ...]
    pads: tuple[tuple[int, int, int], ...]
    selected: int


@dataclass(frozen=True)
class Model:
    rotations: tuple[int, ...]
    clickable: tuple[bool, ...]
    pairs: tuple[tuple[int, int], ...]
    rails: tuple[tuple[int, int, int, int], ...]
    blockers: tuple[tuple[int, int, int, int], ...]
    boundary: tuple[int, int, int, int]
    start: State


def _rectangles(sprites):
    return tuple(
        (int(sprite.x), int(sprite.y), int(sprite.width), int(sprite.height))
        for sprite in sprites
    )


def extract_model(env):
    """Extract all static geometry and mutable stable state from ``Env``."""
    heads = env.heads()
    indices = {id(head): index for index, head in enumerate(heads)}
    boundaries = env.level.get_sprites_by_tag(names.TAG_BOUNDARY)
    if len(boundaries) != 1:
        raise ValueError("SK48 model requires exactly one movement boundary")
    boundary = boundaries[0]
    # Native qzvlbxkjgk intentionally uses prototype pixels * CELL.
    right = int(boundary.x + boundary.pixels.shape[1] * names.CELL)
    bottom = int(boundary.y + boundary.pixels.shape[0] * names.CELL)
    state = State(
        heads=tuple((int(head.x), int(head.y)) for head in heads),
        lines=tuple(
            tuple((int(segment.x), int(segment.y), int(segment.rotation))
                  for segment in env.lines()[head])
            for head in heads
        ),
        pads=tuple(
            (int(pad.x), int(pad.y), int(pad.pixels[1, 1]))
            for pad in env.color_pads()
        ),
        selected=heads.index(env.selected()),
    )
    return Model(
        rotations=tuple(int(head.rotation) for head in heads),
        clickable=tuple(names.TAG_CLICK in head.tags for head in heads),
        pairs=tuple((indices[id(top)], indices[id(bottom)])
                    for top, bottom in env.pairs().items()),
        rails=_rectangles(env.level.get_sprites_by_tag(names.TAG_RAIL)),
        blockers=_rectangles(env.level.get_sprites_by_tag(names.TAG_BLOCKER)),
        boundary=(int(boundary.x), int(boundary.y), right, bottom),
        start=state,
    )


def _contains(rect, x, y):
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def _blocked(model, x, y):
    left, top, right, bottom = model.boundary
    if x < left or y < top or x + names.CELL > right or y + names.CELL > bottom:
        return True
    return any(_contains(rect, x, y) for rect in model.blockers)


def _segment_at(lines, x, y, horizontal, exclude=None):
    for line_index, line in enumerate(lines):
        for segment_index, segment in enumerate(line):
            if exclude == (line_index, segment_index):
                continue
            sx, sy, rotation = segment
            if sx == x and sy == y and (rotation in (0, 180)) == horizontal:
                return line_index, segment_index
    return None


def _pad_at(pads, x, y):
    return next((index for index, pad in enumerate(pads)
                 if pad[0] == x and pad[1] == y), None)


def transition(model, state, action):
    """Return the next stable state for a movement action, or the same state."""
    if action not in names.MOVE_ACTIONS:
        raise ValueError("model.transition only accepts movement actions")
    move_x, move_y = names.ACTION_DELTAS[action]
    dx, dy = move_x * names.CELL, move_y * names.CELL
    selected = state.selected
    rotation = model.rotations[selected]
    base_x, base_y = names.DIRECTION[rotation]
    heads = [list(position) for position in state.heads]
    lines = [[list(segment) for segment in line] for line in state.lines]
    pads = [list(pad) for pad in state.pads]
    moving = set()

    def blocked_entity(kind, identity, mx, my):
        if kind == "segment":
            line_index, segment_index = identity
            x, y, rotation_value = lines[line_index][segment_index]
        else:
            x, y, _ = pads[identity]
            rotation_value = None
        if _blocked(model, x + mx * names.CELL, y + my * names.CELL):
            if not (
                kind == "segment"
                and (names.DIRECTION[rotation_value] == (-mx, -my)
                     or _blocked(model, x, y))
            ):
                return True
        return False

    def push(kind, identity, mx, my, source_segment=None):
        marker = (kind, identity)
        if marker in moving:
            return True
        if blocked_entity(kind, identity, mx, my):
            return False
        local_dx, local_dy = mx * names.CELL, my * names.CELL
        if kind == "segment":
            line_index, segment_index = identity
            x, y, segment_rotation = lines[line_index][segment_index]
            for offset_x, offset_y in ((0, 0), (local_dx, local_dy)):
                pad_index = _pad_at(pads, x + offset_x, y + offset_y)
                if pad_index is None:
                    continue
                if push("pad", pad_index, mx, my, identity):
                    moving.add(("pad", pad_index))
                elif ((segment_rotation in (90, 270)) != (mx == 0)):
                    return False
                # The remaining native branch records a visual pause.  It has
                # no additional stable-state mutation.
        else:
            x, y, _ = pads[identity]
            horizontal_move = mx != 0
            for offset_x, offset_y in ((0, 0), (local_dx, local_dy)):
                segment = _segment_at(
                    lines, x + offset_x, y + offset_y,
                    not horizontal_move, exclude=source_segment,
                )
                if segment is not None:
                    segment_rotation = lines[segment[0]][segment[1]][2]
                    if ((segment_rotation in (90, 270)) != (mx == 0)):
                        return False
            destination = _pad_at(pads, x + local_dx, y + local_dy)
            if destination is not None and not push("pad", destination, mx, my):
                return False
        moving.add(marker)
        return True

    line = lines[selected]
    if (move_x, move_y) == (base_x, base_y):
        tail = line[-1]
        if _blocked(model, tail[0] + dx, tail[1] + dy):
            return state
        for segment_index in range(len(line)):
            push("segment", (selected, segment_index), move_x, move_y)
        # A new segment fills the selected head's old cell.
        line.insert(0, [heads[selected][0], heads[selected][1], rotation])
        # Existing indices shifted by one after insertion.
        moving = {
            (kind, (line_index, segment_index + 1)
             if kind == "segment" and line_index == selected
             else identity)
            for kind, identity in moving
            for line_index, segment_index in ([identity] if kind == "segment" else [(-1, -1)])
        }
    elif (move_x, move_y) == (-base_x, -base_y):
        if len(line) == 1:
            return state
        line.pop(0)
        for segment_index in range(len(line)):
            push("segment", (selected, segment_index), move_x, move_y)
    else:
        hx, hy = heads[selected]
        probe_x, probe_y = hx + 2 + dx // 2, hy + 2 + dy // 2
        if not any(_contains(rect, probe_x, probe_y) for rect in model.rails):
            return state
        for segment_index in range(len(line)):
            if not push("segment", (selected, segment_index), move_x, move_y):
                return state
        moving.add(("head", selected))

    for kind, identity in moving:
        if kind == "head":
            heads[identity][0] += dx
            heads[identity][1] += dy
        elif kind == "pad":
            pads[identity][0] += dx
            pads[identity][1] += dy
        else:
            line_index, segment_index = identity
            lines[line_index][segment_index][0] += dx
            lines[line_index][segment_index][1] += dy
    return State(
        tuple(tuple(position) for position in heads),
        tuple(tuple(tuple(segment) for segment in line) for line in lines),
        tuple(tuple(pad) for pad in pads),
        selected,
    )


def select(model, state, index):
    if index == state.selected or not model.clickable[index]:
        return state
    # Native clicks select the editable/top member of a paired colour, even
    # when the reference/bottom member was clicked.
    target = next((top for top, bottom in model.pairs if index in (top, bottom)), None)
    if target is None or target == state.selected:
        return state
    return State(state.heads, state.lines, state.pads, target)


def visited_colors(state, line_index):
    by_position = {(x, y): color for x, y, color in state.pads}
    return tuple(by_position[(x, y)] for x, y, _ in state.lines[line_index]
                 if (x, y) in by_position)


def target_colors(state, line_index):
    """Reference colors compared by native's fixed indicator count.

    Native creates one indicator for every reference segment after its head,
    then compares only that many visited reference colors.  Legitimate generated
    references place one pad on each such segment, but extraction can also see
    native states with a pad on the reference head or another surplus color.
    """
    indicator_count = max(0, len(state.lines[line_index]) - 1)
    return visited_colors(state, line_index)[:indicator_count]


def solved(model, state):
    for top, bottom in model.pairs:
        actual = visited_colors(state, top)
        target = target_colors(state, bottom)
        if len(actual) < len(target) or actual[:len(target)] != target:
            return False
    return True


def mismatch(model, state):
    value = 0
    for top, bottom in model.pairs:
        actual, target = visited_colors(state, top), target_colors(state, bottom)
        value += max(0, len(target) - len(actual))
        value += sum(left != right for left, right in zip(actual[:len(target)], target))
    return value


def heuristic(model, state):
    value = 100 * mismatch(model, state)
    for top, bottom in model.pairs:
        actual, target = visited_colors(state, top), target_colors(state, bottom)
        first = next((i for i, pair in enumerate(zip(actual, target))
                      if pair[0] != pair[1]), min(len(actual), len(target)))
        if first >= len(target):
            continue
        wanted = target[first]
        candidates = [(x, y) for x, y, color in state.pads if color == wanted]
        if candidates:
            value += min((abs(sx - px) + abs(sy - py)) // names.CELL
                         for sx, sy, _ in state.lines[top]
                         for px, py in candidates)
    return value


def search(model, *, node_limit=200_000, action_limit=196):
    """Guided finite search; returns ``(actions, expanded, truncated)``."""
    start = model.start
    if solved(model, start):
        return (), 0, False
    frontier = [(heuristic(model, start), 0, 0, start)]
    serial = count(1)
    best = {start: 0}
    parent = {start: None}
    expanded = 0
    action_cutoff = False
    while frontier:
        _, cost, _, state = heapq.heappop(frontier)
        if cost != best.get(state):
            continue
        if expanded >= node_limit:
            return None, expanded, True
        expanded += 1
        if cost >= action_limit:
            action_cutoff = True
            continue
        successors = []
        for action in names.MOVE_ACTIONS:
            nxt = transition(model, state, action)
            if nxt != state:
                successors.append(((action, None, None), nxt))
        for index, clickable in enumerate(model.clickable):
            if clickable:
                nxt = select(model, state, index)
                if nxt != state:
                    x, y = state.heads[index]
                    successors.append(((names.ACTION_CLICK, x + 2, y + 2), nxt))
        for action, nxt in successors:
            next_cost = cost + 1
            if next_cost >= best.get(nxt, 1 << 30):
                continue
            best[nxt] = next_cost
            parent[nxt] = (state, action)
            if solved(model, nxt):
                route = [action]
                cursor = state
                while parent[cursor] is not None:
                    cursor, previous = parent[cursor]
                    route.append(previous)
                route.reverse()
                return tuple(route), expanded, False
            heapq.heappush(
                frontier,
                (heuristic(model, nxt) + next_cost, next_cost, next(serial), nxt),
            )
    return None, expanded, action_cutoff


def beam_search(model, *, node_limit=2_000_000, action_limit=196, width=20_000):
    """Deterministic bounded fallback for broad blocker/coupling plateaus."""
    start = model.start
    if solved(model, start):
        return (), 0, False
    layer = {start}
    seen = {start}
    parent = {start: None}
    expanded = 0
    for _depth in range(1, action_limit + 1):
        if expanded + len(layer) > node_limit:
            return None, expanded, True
        expanded += len(layer)
        candidates = {}
        for state in layer:
            successors = []
            for action in names.MOVE_ACTIONS:
                nxt = transition(model, state, action)
                if nxt != state:
                    successors.append(((action, None, None), nxt))
            for index, clickable in enumerate(model.clickable):
                if clickable:
                    nxt = select(model, state, index)
                    if nxt != state:
                        x, y = state.heads[index]
                        successors.append(((names.ACTION_CLICK, x + 2, y + 2), nxt))
            for action, nxt in successors:
                if nxt in seen or nxt in candidates:
                    continue
                candidates[nxt] = (state, action)
                if solved(model, nxt):
                    route = [action]
                    cursor = state
                    while parent[cursor] is not None:
                        cursor, previous = parent[cursor]
                        route.append(previous)
                    route.reverse()
                    return tuple(route), expanded, False
        if not candidates:
            return None, expanded, False
        ranked = sorted(candidates, key=lambda state: (heuristic(model, state), hash(state)))
        layer = set(ranked[:width])
        seen.update(layer)
        for state in layer:
            parent[state] = candidates[state]
    return None, expanded, True
