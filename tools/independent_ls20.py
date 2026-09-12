"""Independent LS20 model + BFS, written from third_party/ls20/ls20.py directly.

Does NOT import pebby.ls20.plan / fastplan / layout. Geometry is read from the
bank row's own spec fields, not from the project's `extract`.

Rules transcribed from upstream:
  Ls20.step()            ls20.py:1912-2014
  Ls20.txnfzvzetn()      ls20.py:1871-1910   (apply cell effects)
  Ls20.pbznecvnfr()      ls20.py:2042-2060   (win check)
  Ls20.vqfjzzkhid()      ls20.py:2022-2038   (match hint; level_index==0 only)
  hbuhvkxlhc.mfyzdfvxsm  ls20.py:1509-1512   (budget consume)
  twkzhcfelv.ullzqnksoj  ls20.py:1596-1608   (launcher reach)
  dboxixicic.npdjlrkhsg  ls20.py:1696-1712   (patroller choose-step)
"""
from collections import deque
from math import lcm
import time

ORIGINAL_TRANSCRIPTION_SHA256 = "ba50d7604aba5938d273360cb303c7b2701e935d24a998b8c22f1ffa47b79e2c"

# upstream action ids 1..4 -> (dx, dy); ls20.py:1943-1953 (ACTION1 = y-1, etc.)
DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
ACTIONS = (1, 2, 3, 4)
SIZES = {"shape": 6, "color": 4, "rotation": 4}   # ls20.py:1771-1778
IDX = {"shape": 0, "color": 1, "rotation": 2}
# dboxixicic.nakogfhyus, ls20.py:1754-1760: dir 0=+y, 1=+x, 2=-y, 3=-x
PDELTA = ((0, 1), (1, 0), (0, -1), (-1, 0))


class Level:
    def __init__(self, row):
        _validate_supported(row)
        self.walls = {tuple(c) for c in row["walls"]}
        self.goal_triple = {tuple(g["cell"]): tuple(g["triple"]) for g in row["goals"]}
        self.goal_cells = [tuple(g["cell"]) for g in row["goals"]]
        self.n_goals = len(self.goal_cells)
        self.full = (1 << self.n_goals) - 1
        self.goal_index = {c: i for i, c in enumerate(self.goal_cells)}
        self.refills = tuple(sorted(tuple(c) for c in row["refills"]))
        self.refill_slot = {c: i for i, c in enumerate(self.refills)}
        self.max_steps = row["step_counter"]
        self.cost = row["step_cost"]
        self.match_hint = (row["training_context_index"] == 0)   # vqfjzzkhid: level_index>0 -> False
        self.start = tuple(row["start"])
        self.start_triple = tuple(row["start_triple"])

        rail_sets = [frozenset(tuple(c) for c in e["cells"]) for e in row["rails"]]
        static = {}
        self.patrollers = []          # list of dict(kind, seq_cells, tail, period)
        for e in row["cyclers"]:
            cell, kind = tuple(e["cell"]), e["kind"]
            rail = next((rs for rs in rail_sets if cell in rs), None)
            if rail is None:
                static[cell] = kind
            else:
                self.patrollers.append(self._walk(rail, cell, kind))
        self.cyclers = static

        # launchers: reach from ullzqnksoj, blockers frozen at load = walls + goal pads
        blockers = self.walls | set(self.goal_cells)
        self.launchers = []
        for e in row["launchers"]:
            cell, (dx, dy) = tuple(e["cell"]), tuple(e["delta"])
            dist = 0
            for k in range(1, 12):
                if (cell[0] + dx * k, cell[1] + dy * k) in blockers:
                    dist = max(0, k - 1)
                    break
            # bounding-box overlap: the pad mounts 1px back, so it also touches
            # the cell behind it (always a wall in these banks)
            trig = {cell, (cell[0] - dx, cell[1] - dy)}
            self.launchers.append((trig, (dx, dy), dist))

        self.p_tail = max((p["tail"] for p in self.patrollers), default=0)
        self.p_period = lcm(*(p["period"] for p in self.patrollers)) if self.patrollers else 1

    @staticmethod
    def _walk(rail, cell, kind):
        seq, seen = [], {}
        state = (cell, 0)                       # _dir starts at 0 (ls20.py:1681)
        while state not in seen:
            seen[state] = len(seq)
            seq.append(state)
            c, d = state
            nxt = None
            for cand in (d, (d - 1) % 4, (d + 1) % 4, (d + 2) % 4):
                ddx, ddy = PDELTA[cand]
                t = (c[0] + ddx, c[1] + ddy)
                if t in rail:
                    nxt = (t, cand)
                    break
            state = nxt if nxt is not None else (c, d)   # boxed in: nothing moves
        tail = seen[state]
        return {"kind": kind, "cells": tuple(s[0] for s in seq),
                "tail": tail, "period": len(seq) - tail}

    def tick0(self):
        return tuple(0 for _ in self.patrollers)

    def tick_next(self, ticks):
        out = []
        for p, t in zip(self.patrollers, ticks):
            n = t + 1
            out.append(n if n < p["tail"] + p["period"] else p["tail"])
        return tuple(out)

    def moving_at(self, ticks):
        return {p["cells"][t]: p["kind"] for p, t in zip(self.patrollers, ticks)}

    def free(self, cell):
        return cell not in self.walls and 0 <= cell[0] < 12 and 0 <= cell[1] < 12

    def start_state(self):
        return (self.start, self.start_triple[0], self.start_triple[1],
                self.start_triple[2], 0, 0, self.max_steps, self.tick0())


def cycle(kind, sh, co, ro):
    if kind == "shape":
        return (sh + 1) % 6, co, ro
    if kind == "color":
        return sh, (co + 1) % 4, ro
    return sh, co, (ro + 1) % 4


def step(L, st, action):
    """One perform_action. Returns (next_state, outcome)."""
    cell, sh, co, ro, goals, taken, steps, ticks = st
    dx, dy = DELTAS[action]
    target = (cell[0] + dx, cell[1] + dy)
    nticks = L.tick_next(ticks)                     # every patroller advances first
    moving = L.moving_at(nticks)

    blocked = refilled = cycled = False
    if not L.free(target):
        blocked = True                              # wall: `break`, no other effect
    else:
        gi = L.goal_index.get(target)
        if gi is not None and not (goals >> gi) & 1 and (sh, co, ro) != L.goal_triple[target]:
            blocked = True                          # rejecting pad: flash, no break
        if target in L.refill_slot and not (taken >> L.refill_slot[target]) & 1:
            taken |= 1 << L.refill_slot[target]
            steps = L.max_steps
            refilled = True
        kind = L.cyclers.get(target) or moving.get(target)
        if kind is not None:
            sh, co, ro = cycle(kind, sh, co, ro)
            cycled = True

    if blocked:
        pos, ticks_out = cell, ticks                # patrollers rewound
    else:
        pos, ticks_out = target, nticks

    # a rejecting pad sets the reject flash; step() then returns before charging
    rejecting = blocked and L.free(target)
    if rejecting:
        return (pos, sh, co, ro, goals, taken, steps, ticks_out), "rejected"
    # level-0 only: a cycle that reveals a match also flashes and returns free
    if L.match_hint and cycled and any(
            (sh, co, ro) == L.goal_triple[c] for i, c in enumerate(L.goal_cells)
            if not (goals >> i) & 1):
        return (pos, sh, co, ro, goals, taken, steps, ticks_out), "hint"

    if not refilled:
        steps -= L.cost
    exhausted = steps < 0

    if not exhausted:
        for trig, (ldx, ldy), dist in L.launchers:
            if pos in trig and dist > 0:
                pos = (pos[0] + ldx * dist, pos[1] + ldy * dist)
                if pos in L.refill_slot and not (taken >> L.refill_slot[pos]) & 1:
                    taken |= 1 << L.refill_slot[pos]
                    steps = L.max_steps
                lk = L.cyclers.get(pos) or L.moving_at(ticks_out).get(pos)
                if lk is not None:
                    sh, co, ro = cycle(lk, sh, co, ro)
                return (pos, sh, co, ro, goals, taken, steps, ticks_out), "launched"

    gi = L.goal_index.get(pos)
    if gi is not None and not (goals >> gi) & 1 and (sh, co, ro) == L.goal_triple[pos]:
        goals |= 1 << gi
    out = (pos, sh, co, ro, goals, taken, steps, ticks_out)
    if goals == L.full:
        return out, "won"
    if exhausted:
        return out, "died"
    return out, "moved"



def _validate_supported(row):
    """Fail closed outside the independently transcribed generated-level domain.

    Wall-mounted launchers cannot share a goal or a pickup. This also makes
    remaining budget monotone: extra budget cannot prevent a winning action by
    enabling a launcher on an otherwise exhausted goal-pad move.
    """
    walls = {tuple(c) for c in row['walls']}
    goals = [tuple(g['cell']) for g in row['goals']]
    pickups = [tuple(c) for c in row['refills']]
    cyclers = [tuple(c['cell']) for c in row['cyclers']]
    specials = goals + pickups + cyclers
    if len(set(specials)) != len(specials) or set(specials) & walls:
        raise ValueError('independent model requires disjoint pickups/goals/cyclers and walls')
    rails = [set(map(tuple, rail['cells'])) for rail in row.get('rails', [])]
    walked = set()
    for rail in rails:
        if len(rail & set(cyclers)) > 1:
            raise ValueError('independent model requires at most one cycler on each rail')
        if rail & (walked | walls | set(goals + pickups)):
            raise ValueError('independent model requires mutually disjoint unobstructed rails')
        walked |= rail
    pads = set()
    for pad in row.get('launchers', []):
        cell = tuple(pad['cell'])
        dx, dy = pad['delta']
        if ((dx, dy) not in DELTAS.values() or (cell[0]-dx, cell[1]-dy) not in walls
                or cell in walls | set(specials) | walked | pads):
            raise ValueError('independent model supports only disjoint wall-mounted launchers')
        pads.add(cell)
    if row['training_context_index'] == 0 and pads:
        raise ValueError('context-zero launcher animation hints are outside this model')


def bfs(level, *, limit=2_000_000, seconds=120, budget_dominance=True):
    """Independent bounded breadth-first shortest path, stopping at the first win.

    FIFO expansion proves minimal action count. With dominance enabled, a state
    reached at the same or earlier depth with >= remaining budget dominates the
    newly discovered state. Cell/triple/goals/refills/every patroller clock must
    all agree. We never drop an already queued earlier state when a later state
    improves its budget. Disjoint wall-mounted launchers make budget monotone.
    Exhaustion/limit results explicitly provide no minimum proof.
    """
    if limit < 1 or seconds <= 0:
        raise ValueError('positive state/time bounds required')
    started = time.monotonic()
    start = level.start_state()
    def key(state):
        return state[:6] + (state[7],)
    seen = {key(start): start[6]} if budget_dominance else {start}
    queue = deque([(start, 0)])
    expanded, discovered = 0, 1
    def result(distance, complete, reason=None):
        return dict(minimum_actions=distance, complete=complete, reason=reason,
                    expanded_states=expanded, discovered_states=discovered,
                    unique_keys=len(seen), seconds=time.monotonic()-started,
                    budget_dominance=budget_dominance, state_limit=limit)
    while queue:
        if expanded % 1024 == 0 and time.monotonic()-started >= seconds:
            return result(None, False, 'time_limit')
        state, depth = queue.popleft()
        expanded += 1
        for action in ACTIONS:
            successor, outcome = step(level, state, action)
            if outcome in ('rejected', 'died'):
                continue
            if outcome == 'won':
                return result(depth+1, True)
            if budget_dominance:
                identity = key(successor)
                if seen.get(identity, -1) >= successor[6]:
                    continue
            else:
                identity = successor
                if identity in seen:
                    continue
            if discovered >= limit:
                return result(None, False, 'state_limit')
            if budget_dominance:
                seen[identity] = successor[6]
            else:
                seen.add(identity)
            queue.append((successor, depth+1))
            discovered += 1
    return result(None, True, 'unsolvable')
