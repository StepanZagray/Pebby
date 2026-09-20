"""An exact FT09 solver.

Clicks commute: the final colour of a cell is its start plus the number of
times each stencil covering it was clicked, modulo the palette size. So a
level is solved by a click-count vector c in Z_k^n with

    initial + A c  (mod k)  in  allowed,   1 <= sum(c) <= budget,

and the click order is irrelevant except that the level ends at the first
click after which every constraint holds (ft09.py:2389-2392). Two exact
methods are used:

* breadth-first search over colour states when k^n is small; this yields a
  shortest solution and proves there is none if the search exhausts;
* for prime k, Gaussian elimination mod k for every admissible target
  assignment of the constrained cells, then enumeration of the affine
  solution space to find the fewest clicks. If the enumeration would exceed
  `limit` candidates it stops, and the result is flagged `truncated` whenever
  the answer could have been changed by the unexplored part.

`limit` bounds the work in either method. A result with `actions is None` and
`truncated False` is a proof that the level cannot be completed within budget.
"""

from collections import deque
from dataclasses import dataclass
import itertools

from .layout import Layout, extract


@dataclass
class Solution:
    actions: list | None      # (action_id, x, y) tuples, or None
    truncated: bool           # True if the search was cut off before it could settle the question
    clicks: tuple = ()        # cell indices clicked, in order
    explored: int = 0         # states or candidates examined

    def __bool__(self):
        return self.actions is not None

    def __len__(self):
        return 0 if self.actions is None else len(self.actions)


def _layout(env_or_layout):
    if isinstance(env_or_layout, Layout):
        return env_or_layout
    return extract(env_or_layout)


def _is_prime(k):
    return k >= 2 and all(k % d for d in range(2, int(k ** 0.5) + 1))


def _first_winning_prefix(layout, order):
    """Truncate a click order at the first click that completes the level."""
    state = layout.initial
    for n, index in enumerate(order, 1):
        state = layout.click(state, index)
        if layout.satisfied(state):
            return tuple(order[:n])
    return None


# --- breadth-first search ------------------------------------------------------

def _bfs(layout, limit):
    start = layout.initial
    parent = {start: None}
    depths = {start: 0}
    queue = deque([start])
    explored = 0
    while queue:
        state = queue.popleft()
        if depths[state] >= layout.budget:
            continue
        if explored >= limit:
            return Solution(None, True, (), explored)
        explored += 1
        for index in range(layout.size):
            nxt = layout.click(state, index)
            # Check before deduplication: an initially satisfied board can require
            # a nonempty cycle back to its initial state to trigger completion.
            if layout.satisfied(nxt):
                order = [index]
                cur = state
                while parent[cur] is not None:
                    cur, i = parent[cur]
                    order.append(i)
                order.reverse()
                return Solution(layout.clicks_to_actions(order), False, tuple(order), explored)
            if nxt not in parent:
                parent[nxt] = (state, index)
                depths[nxt] = depths[state] + 1
                queue.append(nxt)
    return Solution(None, False, (), explored)


# --- linear algebra mod prime k ------------------------------------------------

def _solve_mod_p(rows, rhs, n, p):
    """Solve rows * c = rhs over Z_p. Returns (particular, nullspace basis) or None."""
    m = len(rows)
    a = [list(r) + [b % p] for r, b in zip(rows, rhs)]
    pivots = []
    r = 0
    for col in range(n):
        piv = next((i for i in range(r, m) if a[i][col] % p), None)
        if piv is None:
            continue
        a[r], a[piv] = a[piv], a[r]
        inv = pow(a[r][col], -1, p)
        a[r] = [(v * inv) % p for v in a[r]]
        for i in range(m):
            if i != r and a[i][col]:
                f = a[i][col]
                a[i] = [(vi - f * vr) % p for vi, vr in zip(a[i], a[r])]
        pivots.append(col)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if a[i][n] % p:
            return None
    particular = [0] * n
    for i, col in enumerate(pivots):
        particular[col] = a[i][n]
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = [0] * n
        v[f] = 1
        for i, col in enumerate(pivots):
            v[col] = (-a[i][f]) % p
        basis.append(v)
    return particular, basis


def _algebra(layout, limit):
    k, n = layout.colours, layout.size
    # A[j][i] = number of times a click on i advances cell j.
    matrix = [[0] * n for _ in range(n)]
    for i, hit in enumerate(layout.affects):
        for j in hit:
            matrix[j][i] += 1
    constrained = [j for j, allowed in enumerate(layout.allowed) if allowed is not None]
    choices = [sorted(layout.allowed[j]) for j in constrained]
    rows = [matrix[j] for j in constrained]

    best = None
    explored = 0
    truncated = False
    for target in itertools.product(*choices):
        if explored >= limit:
            truncated = True
            break
        rhs = [(t - layout.initial[j]) % k for t, j in zip(target, constrained)]
        solved = _solve_mod_p(rows, rhs, n, k)
        if solved is None:
            explored += 1
            continue
        particular, basis = solved
        combos = itertools.product(range(k), repeat=len(basis))
        for coeffs in combos:
            if explored >= limit:
                truncated = True
                break
            explored += 1
            c = list(particular)
            for coef, vec in zip(coeffs, basis):
                if coef:
                    c = [(ci + coef * vi) % k for ci, vi in zip(c, vec)]
            weight = sum(c)
            if weight >= 1 and (best is None or weight < best[0]):
                best = (weight, c)
        if truncated:
            break

    if best is None:
        return Solution(None, truncated, (), explored)
    weight, counts = best
    order = [i for i in range(n) for _ in range(counts[i])]
    order = _first_winning_prefix(layout, order)
    if order is None or len(order) > layout.budget:
        return Solution(None, truncated, (), explored)
    # A solution found by a cut-off enumeration is still a proof of solvability,
    # though not necessarily the shortest one; report the cut-off honestly.
    return Solution(layout.clicks_to_actions(order), truncated, order, explored)


def search(env_or_layout, limit=2_000_000):
    """Exact solution of the current level.

    Returns a `Solution`; `.actions` is a list of (6, x, y) clicks completing
    the level within budget, or None. `.truncated` is True only when the search
    hit `limit` before exhaustive search or shortest-path proof.
    """
    layout = _layout(env_or_layout)
    if any(allowed is not None and not allowed for allowed in layout.allowed):
        return Solution(None, False)          # a constraint no palette colour can satisfy
    if layout.size == 0 or layout.budget < 1:
        return Solution(None, False)
    if limit < 1:
        return Solution(None, True)
    k, n = layout.colours, layout.size
    if layout.satisfied(layout.initial) or k ** n <= limit:
        return _bfs(layout, limit)
    if _is_prime(k):
        return _algebra(layout, limit)
    return Solution(None, True)


def verify(layout, actions):
    """Symbolically replay `actions` and report whether they complete the level in budget."""
    by_click = {click: i for i, click in enumerate(layout.clicks)}
    order = []
    for action_id, x, y in actions:
        if action_id != 6 or (x, y) not in by_click:
            return False
        order.append(by_click[(x, y)])
    prefix = _first_winning_prefix(layout, order)
    return prefix is not None and len(prefix) == len(order) <= layout.budget


def solve(env_or_layout, limit=2_000_000):
    """Return executable action tuples or None; inspect solve.truncated or search()."""
    result = search(env_or_layout, limit)
    solve.truncated = result.truncated
    return result.actions


solve.truncated = False
