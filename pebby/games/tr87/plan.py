"""Symbolic planner and full-mechanics teachers for TR87.

State = (editable digits, cursor). Plain levels edit one target tile per group;
`alter_rules` levels edit whole rule sides. The search is a uniform-cost
best-first search (BFS order when the heuristic is zero) with an admissible,
consistent heuristic, so the first solution found has minimal length. Upstream
only runs the win test after ACTION1/ACTION2 (tr87.py:1012), so the goal is a
separate terminal node entered by a cycle action; a level whose target already
matches still needs two actions.

`search` is the bounded shared entry: it is exact for target-edit layouts and
dispatches live editable-rule/tree environments to replayable constructive
teachers. Those certificates do not claim shortest routes. Any path stopped by
its node limit returns `truncated=True`. `solve` collapses the result to a list
or None.
"""

from dataclasses import dataclass
from functools import lru_cache
import heapq

from . import names
from .layout import Layout, extract, solved, translate

WIN = "WIN"
DEFAULT_NODE_LIMIT = 400_000


@dataclass
class Result:
    actions: list | None      # [(action_id, None, None)] or None
    truncated: bool
    expanded: int
    reason: str

    @property
    def length(self):
        return None if self.actions is None else len(self.actions)


def _cover_walk(n, start, points):
    """Fewest cursor moves from `start` visiting every position in `points` on an n-cycle."""
    offsets = sorted({(p - start) % n for p in points} - {0})
    if not offsets:
        return 0
    best = min(offsets[-1], n - offsets[0])
    for j in range(len(offsets) - 1):
        a, b = offsets[j], offsets[j + 1]
        best = min(best, 2 * a + (n - b), 2 * (n - b) + a)
    return best


class _Model:
    def __init__(self, layout):
        self.layout = layout
        self.n = layout.group_count
        if layout.alter_rules:
            self.base = tuple(tuple(layout.group_symbols(i)) for i in range(self.n))
            self.demanded = None
        else:
            self.base = None
            self.demanded = translate(layout)
        self.families = tuple(sym[0] for sym in layout.target)

    def initial(self):
        if self.layout.alter_rules:
            digits = tuple(0 for _ in range(self.n))          # per-side offsets
        else:
            digits = tuple(int(sym[1]) for sym in self.layout.target)
        return (digits, self.layout.cursor)

    def impossible(self):
        """A reason the level can never be solved, or None."""
        if self.layout.alter_rules:
            return None
        if self.demanded is None:
            return "source row has no parse under the rules"
        if len(self.demanded) > len(self.layout.target):
            return "translation longer than the target row"
        for sym, fam in zip(self.demanded, self.families):
            if sym[0] != fam:
                return f"target family {fam} can never show {sym}"
        return None

    def rules_for(self, digits):
        rules = []
        for r in range(len(self.layout.rules)):
            lhs = tuple(names.cycle(s, digits[2 * r]) for s in self.base[2 * r])
            rhs = tuple(names.cycle(s, digits[2 * r + 1]) for s in self.base[2 * r + 1])
            rules.append((lhs, rhs))
        return tuple(rules)

    def is_solved(self, digits):
        if self.layout.alter_rules:
            return solved(self.layout, rules=self.rules_for(digits))
        target = tuple(f + str(d) for f, d in zip(self.families, digits))
        return solved(self.layout, target=target)

    def heuristic(self, state):
        if self.layout.alter_rules:
            return 0
        digits, cursor = state
        wanted = [int(sym[1]) for sym in self.demanded]
        total, pending = 0, []
        for i, d in enumerate(wanted):
            gap = names.digit_distance(digits[i], d)
            total += gap
            if gap and i != cursor:
                pending.append(i)
        return total + _cover_walk(self.n, cursor, pending)

    def successors(self, state):
        digits, cursor = state
        for action, delta in names.SELECT_DELTA.items():
            yield action, (digits, (cursor + delta) % self.n)
        for action, delta in names.CYCLE_DELTA.items():
            new = list(digits)
            if self.layout.alter_rules:
                new[cursor] = (new[cursor] + delta) % names.SYMBOL_COUNT
            else:
                new[cursor] = (new[cursor] + delta - 1) % names.SYMBOL_COUNT + 1
            new = tuple(new)
            yield action, (WIN if self.is_solved(new) else (new, cursor))


def search(layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Bounded shared search; target-edit results are shortest, teachers need not be."""
    if not isinstance(layout, Layout):
        live = layout
        layout = extract(live)
        if layout.alter_rules:
            return teacher_search(live, limit=limit, node_limit=node_limit)
    limit = layout.budget if limit is None else min(limit, layout.budget)
    model = _Model(layout)
    reason = model.impossible()
    if reason:
        return Result(None, False, 0, reason)
    start = model.initial()
    best = {start: 0}
    parent = {start: None}
    counter = 0
    frontier = [(model.heuristic(start), 0, counter, start)]
    expanded = 0
    while frontier:
        f, g, _, state = heapq.heappop(frontier)
        if g != best.get(state):
            continue
        if state == WIN:
            actions = []
            while parent[state] is not None:
                state, action = parent[state]
                actions.append((action, None, None))
            actions.reverse()
            return Result(actions, False, expanded, "solved")
        if g >= limit:
            continue
        expanded += 1
        if expanded > node_limit:
            return Result(None, True, expanded, "node limit reached")
        for action, nxt in model.successors(state):
            ng = g + 1
            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                parent[nxt] = (state, action)
                h = 0 if nxt == WIN else model.heuristic(nxt)
                counter += 1
                heapq.heappush(frontier, (ng + h, ng, counter, nxt))
    return Result(None, False, expanded, "no solution within the action limit")


def _uniform_delta(current, wanted):
    if len(current) != len(wanted):
        return None
    deltas = {(int(b[1]) - int(a[1])) % names.SYMBOL_COUNT for a, b in zip(current, wanted)
              if a[0] == b[0]}
    if len(deltas) != 1 or any(a[0] != b[0] for a, b in zip(current, wanted)):
        return None
    return deltas.pop()


def _shift(sequence, delta):
    return tuple(names.cycle(symbol, delta) for symbol in sequence)


class _RouteLimit(Exception):
    pass


def _route_offsets(offsets, start=0, node_limit=DEFAULT_NODE_LIMIT):
    """Shortest cursor tour applying a fixed offset to every nonzero group."""
    points = tuple(i for i, delta in enumerate(offsets) if delta)
    count = len(offsets)
    expanded = 0

    @lru_cache(maxsize=None)
    def visit(cursor, remaining):
        nonlocal expanded
        expanded += 1
        if expanded > node_limit:
            raise _RouteLimit
        if not remaining:
            return 0, ()
        best = None
        for k, point in enumerate(remaining):
            forward, backward = (point - cursor) % count, (cursor - point) % count
            navigation = (names.ACTION_SELECT_NEXT,) * forward if forward <= backward \
                else (names.ACTION_SELECT_PREV,) * backward
            delta = offsets[point]
            cycles = (names.ACTION_CYCLE_UP,) * delta if delta <= 3 \
                else (names.ACTION_CYCLE_DOWN,) * (7 - delta)
            rest = remaining[:k] + remaining[k + 1:]
            cost, tail = visit(point, rest)
            candidate = (len(navigation) + len(cycles) + cost, navigation + cycles + tail)
            if best is None or candidate < best:
                best = candidate
        return best

    try:
        actions = [(action, None, None) for action in visit(start, points)[1]]
    except _RouteLimit:
        return None, expanded, True
    return actions, expanded, False


def _clean_rule_offsets(env, layout):
    """Offsets restoring clean rule sides, if that clean grammar solves now."""
    teacher = env.level.get_data(names.KEY_TEACHER_RULES)
    if teacher is not None:
        wanted_rules = tuple((tuple(lhs), tuple(rhs)) for lhs, rhs in teacher)
        offsets = []
        for current_rule, wanted_rule in zip(layout.rules, wanted_rules):
            for current, wanted in zip(current_rule, wanted_rule):
                delta = _uniform_delta(current, wanted)
                if delta is None:
                    return None
                offsets.append(delta)
        return tuple(offsets) if solved(layout, rules=wanted_rules) else None
    clean = env.game._clean_levels[env.level_index]
    by_position = {(sprite.x, sprite.y): names.symbol(sprite.name)
                   for sprite in clean.get_sprites_by_tag(names.TAG_TILE)}
    wanted_rules, offsets = [], []
    for lhs, rhs in env.rule_sprites():
        wanted_rule = []
        for side in (lhs, rhs):
            current = tuple(names.symbol(sprite.name) for sprite in side)
            try:
                wanted = tuple(by_position[(sprite.x, sprite.y)] for sprite in side)
            except KeyError:
                return None
            delta = _uniform_delta(current, wanted)
            if delta is None:
                return None
            offsets.append(delta)
            wanted_rule.append(wanted)
        wanted_rules.append(tuple(wanted_rule))
    return tuple(offsets) if solved(layout, rules=tuple(wanted_rules)) else None


def _direct_alter_offsets(layout, node_limit):
    """All direct-translation rule assignments, with exact greedy validation."""
    rules = layout.rules
    solutions = set()
    expanded = 0
    truncated = False

    def visit(source_position, target_position, assignments):
        nonlocal expanded, truncated
        if truncated:
            return
        expanded += 1
        if expanded > node_limit:
            truncated = True
            return
        if source_position == len(layout.source):
            offsets = tuple(assignments.get(i, 0) for i in range(2 * len(rules)))
            shifted = tuple((_shift(lhs, offsets[2 * i]), _shift(rhs, offsets[2 * i + 1]))
                            for i, (lhs, rhs) in enumerate(rules))
            if solved(layout, rules=shifted):
                solutions.add(offsets)
            return
        for i, (lhs, rhs) in enumerate(rules):
            source_end = source_position + len(lhs)
            target_end = target_position + len(rhs)
            if source_end > len(layout.source) or target_end > len(layout.target):
                continue
            left_delta = _uniform_delta(lhs, layout.source[source_position:source_end])
            right_delta = _uniform_delta(rhs, layout.target[target_position:target_end])
            if left_delta is None or right_delta is None:
                continue
            if assignments.get(2 * i, left_delta) != left_delta:
                continue
            if assignments.get(2 * i + 1, right_delta) != right_delta:
                continue
            changed = dict(assignments)
            changed[2 * i], changed[2 * i + 1] = left_delta, right_delta
            visit(source_end, target_end, changed)

    visit(0, 0, {})
    return solutions, expanded, truncated


def teacher_search(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Return a replayable teacher for every official mechanic/tier.

    Direct target-edit tiers retain the exact A* planner. Editable-rule tiers
    use either a clean-state constructive certificate (including tree mode) or
    an exhaustive grammar assignment for direct translation. The latter proves
    its final assignments against the engine-faithful greedy mirror.
    """
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    action_limit = layout.budget if limit is None else min(limit, layout.budget)
    if node_limit < 1:
        return Result(None, True, 0, "node limit reached")
    if not layout.alter_rules:
        return search(layout, limit=action_limit, node_limit=node_limit)
    if not isinstance(env_or_layout, Layout):
        offsets = _clean_rule_offsets(env_or_layout, layout)
        if offsets is not None:
            actions, expanded, truncated = _route_offsets(
                offsets, layout.cursor, node_limit=node_limit)
            if truncated:
                return Result(None, True, expanded, "node limit reached")
            if len(actions) > action_limit:
                return Result(None, False, expanded, "constructive witness exceeds action limit")
            return Result(actions, False, expanded, "constructive rule certificate")
    if not layout.tree_translation and not layout.double_translation:
        solutions, expanded, truncated = _direct_alter_offsets(layout, node_limit)
        if truncated:
            return Result(None, True, expanded, "node limit reached")
        if solutions:
            routes = []
            work = expanded
            for offsets in sorted(solutions):
                actions, route_work, truncated = _route_offsets(
                    offsets, layout.cursor, node_limit=max(0, node_limit - work))
                work += route_work
                if truncated:
                    return Result(None, True, work, "node limit reached")
                routes.append(actions)
            actions = min(routes, key=len)
            if len(actions) > action_limit:
                return Result(None, False, work, "constructive witness exceeds action limit")
            return Result(actions, False, work, "exhaustive direct-rule certificate")
        return Result(None, False, expanded, "no satisfying direct-rule assignment")
    return Result(None, False, 0, "tree/double teacher requires a live environment")


def solve(env_or_layout, limit=None, node_limit=DEFAULT_NODE_LIMIT):
    """Return actions or None; expose the last outcome via `result` and `truncated`.

    `limit` bounds action count, `node_limit` bounds search work. Use `search`
    directly when concurrent callers need independent result metadata.
    """
    result = search(env_or_layout, limit, node_limit)
    solve.result = result
    solve.truncated = result.truncated
    return result.actions


solve.result = None
solve.truncated = False
