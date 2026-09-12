"""Rail-riding cyclers, the one LS20 tile whose position has memory.

A cycler sprite that overlaps an invisible "rail" sprite is wrapped in upstream's
patroller class (`dboxixicic`, ls20.py:1674-1762) and walks that rail one whole
5px cell per advance. Upstream advances *every* patroller before it resolves the
player's move and rewinds every one of them when that move turns out to be
blocked (ls20.py:1953-1961), so a patroller's position is a pure function of one
number: how many player moves have been accepted so far.

That number is unbounded, but the walk is not. `choose_step` is deterministic and
a rail has finitely many cells, so the sequence of (position, direction) states
is eventually periodic. This module runs the walk offline once per patroller and
folds it into a single lookup table, which is why the planner only has to carry a
small tick counter instead of a per-patroller position.

The fold keeps a *tail*, because `_dir` starts at 0 whatever the rail looks like
and that is rarely the heading the walk settles into: four of the five shipped
patrollers never return to their opening state, so tick 0 sits outside the cycle.
Detecting the repeat on (position, direction) rather than on position alone is
what catches that -- level 5's cycler is back on its starting cell after four
advances but heading the other way, and level 6's ring cycler likewise.
"""

from math import lcm

from . import names

# Upstream `nakogfhyus`, ls20.py:1754-1760. Index is `_dir`, which starts at 0.
DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))

# Upstream `_cell`, ls20.py:1676 -- patrollers move a whole player-width at once.
STEP = names.CELL


def _walkable(rail, x, y):
    """Upstream `iiosonyanc`, ls20.py:1739-1752.

    Two conditions: the point is inside the rail sprite's bounding box, and the
    rail's own pixel there is not the transparent -1. The L6 ring rail is a
    hollow square, so the second test is what keeps its cycler on the border.
    """
    dx, dy = x - rail.x, y - rail.y
    if not (0 <= dx < rail.width and 0 <= dy < rail.height):
        return False
    return int(rail.pixels[dy, dx]) >= 0


def _choose_step(rail, x, y, direction):
    """Upstream `npdjlrkhsg`, ls20.py:1696-1712. Returns (x, y, direction).

    Straight on first, then *left*, then right, then back the way it came. The
    left-before-right preference is what makes the ring rail circle one way
    rather than the other, so the order matters.
    """
    for candidate in (direction, (direction - 1) % 4, (direction + 1) % 4, (direction + 2) % 4):
        dx, dy = DELTAS[candidate]
        nx, ny = x + dx * STEP, y + dy * STEP
        if _walkable(rail, nx, ny):
            return nx, ny, candidate
    return x, y, direction  # boxed in: upstream's `step` returns False and nothing moves


def _walk(rail, x, y):
    """Every state the patroller will ever be in, as (states, tail, period).

    `states[t]` is (x, y, direction) after `t` advances, for t < tail + period;
    after that the walk repeats `states[tail:]` forever.
    """
    states, seen = [], {}
    state = (x, y, 0)  # upstream `_dir` starts at 0 (ls20.py:1681) and resets to 0
    while state not in seen:
        seen[state] = len(states)
        states.append(state)
        state = _choose_step(rail, *state)
    tail = seen[state]
    return tuple(states), tail, len(states) - tail


def _pairs(game):
    """Reproduce upstream's pairing loop, ls20.py:1851-1856.

    Every (rail, cycler) pair that overlaps becomes its own patroller, so one
    rail with two cyclers on it would yield two. Iterating the cycler tags in
    `names.CYCLER_TAGS` order keeps our list in upstream's list order, which is
    the order cell effects are applied in.
    """
    level = game.current_level
    found = []
    for rail in level.get_sprites_by_tag(names.TAG_PATROL_RAIL):
        for tag, kind in names.CYCLER_TAGS.items():
            for sprite in level.get_sprites_by_tag(tag):
                if rail.collides_with(sprite, ignoreMode=True):
                    found.append((rail, sprite, kind))
    return found


def _origins(game):
    """Where each patrolling sprite was placed, keyed by sprite identity.

    Upstream remembers this (ls20.py:1678-1679) and snaps back to it when a life
    is lost, so the walk has to be replayed from there rather than from wherever
    the sprite happens to be standing when a layout is extracted mid-level.
    `_sprite`, `_start_x` and `_start_y` escaped the obfuscator; the rail
    reference did not, which is why the pairing above is reproduced rather than
    read off `game`.
    """
    return {id(patroller._sprite): (patroller._start_x, patroller._start_y)
            for patroller in getattr(game, names.ATTR_PATROLLERS, ())}


def patrollers(game):
    """Every rail-riding cycler on the level `game` is currently playing.

    Each entry carries its `kind` ("shape"/"color"/"rotation"), the lattice cell
    it occupies at each tick, and the (position, direction) states behind them,
    which `phase_of` needs to read a tick back off a running game.
    """
    origins = _origins(game)
    found = []
    for rail, sprite, kind in _pairs(game):
        x, y = origins.get(id(sprite), (sprite.x, sprite.y))
        states, tail, period = _walk(rail, x, y)
        found.append({"kind": kind, "states": states, "tail": tail, "period": period,
                      "cells": tuple(names.pixel_to_cell(sx, sy) for sx, sy, _ in states),
                      "start_cell": names.pixel_to_cell(x, y)})
    return found


def riders(game):
    """Identities of the cycler sprites that ride a rail rather than sitting still.

    A caller walking the sprite list cannot tell the two apart from tags alone --
    a patroller is an ordinary cycler that happens to overlap a rail -- so it has
    to ask here. The ids are only meaningful for the duration of one sprite walk.
    """
    return frozenset(id(sprite) for _, sprite, _ in _pairs(game))


def schedule(found):
    """Fold the patrollers onto one shared clock. Returns (tail, period, ticks).

    They all advance on the same events, so a single counter drives every one of
    them. `ticks[t]` maps cell -> kind for tick t, and `t` runs over
    `range(tail + period)`; `advance` wraps back to `tail`, not to 0.
    """
    tail = max((p["tail"] for p in found), default=0)
    period = lcm(*(p["period"] for p in found)) if found else 1
    ticks = []
    for tick in range(tail + period):
        ticks.append({patroller["cells"][_fold(patroller, tick)]: patroller["kind"]
                      for patroller in found})
    return tail, period, tuple(ticks)


def _fold(patroller, tick):
    """`tick` on the shared clock, as an index into this patroller's own walk."""
    own_tail, own_period = patroller["tail"], patroller["period"]
    if tick < own_tail:
        return tick
    return own_tail + (tick - own_tail) % own_period


def advance(tick, tail, period):
    """The next tick on the shared clock, staying inside `range(tail + period)`."""
    return tick + 1 if tick + 1 < tail + period else tail


def phase_of(found, tail, period, observed):
    """Recover the tick of a running game from its patroller sprites.

    `observed` is one (x, y, direction) per patroller, in `patrollers` order.
    Matching on direction as well as position is what makes this unambiguous: a
    cycler bouncing along a straight rail visits the same cell twice per lap.
    Returns None when nothing matches, which can only happen if the game was
    reset out from under the layout.
    """
    for tick in range(tail + period):
        if all(patroller["states"][_fold(patroller, tick)] == state
               for patroller, state in zip(found, observed)):
            return tick
    return None


def live_states(game):
    """The (x, y, direction) each patroller is in right now, in `patrollers` order.

    `_dir` is not obfuscated, and reading it is the only way to tell the two
    halves of a bounce apart from the outside.
    """
    return [(patroller._sprite.x, patroller._sprite.y, patroller._dir)
            for patroller in getattr(game, names.ATTR_PATROLLERS, ())]
