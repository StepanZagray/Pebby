"""Read a level's logical structure out of a running LS20 game.

The planner reasons over cells, not sprites. This module is the only place that
knows how upstream's sprite soup maps onto the 12x12 lattice, and it works
identically for the seven shipped levels and for generated ones.
"""

from . import names, rails as rail_walks


class Layout:
    """Static description of one level, plus the state the player starts in.

    Rail-riding cyclers move, so they are not in `cyclers`; they live in
    `patrollers` and are looked up through `cycler_at`, which takes the tick they
    should be read at. Every patroller advances on the same event, so one shared
    clock drives them all: `tick_span` ticks, of which the first `tick_tail` run
    once and the remaining `tick_period` repeat forever (see `rails`).

    `exact` says the planner models every mechanic this level uses, so a plan for
    it is a completability proof. `extract` refuses to build an inexact Layout.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    @property
    def exact(self):
        # Walls, goals, refills, launchers, static cyclers and now rail-riding
        # cyclers are all modelled, and `extract` rejects any level that overlaps
        # two of them on one cell, which is the only case whose outcome would
        # depend on upstream's sprite list order rather than on its rules.
        return True

    def free(self, cell):
        col, row = cell
        return (0 <= col < self.cols and 0 <= row < self.rows and cell not in self.walls)

    def next_tick(self, tick):
        """The patroller clock after one accepted player move."""
        return rail_walks.advance(tick, self.tick_tail, self.tick_period)

    def cycler_at(self, cell, tick):
        """Which cycler stands on `cell` at `tick`, or None.

        Static tiles are checked first only for speed: `extract` guarantees a
        patroller never shares a cell with anything else, so at most one hits.
        """
        kind = self.cyclers.get(cell)
        return kind if kind is not None else self.moving_cyclers[tick].get(cell)

    def describe(self):
        return (f"{self.cols}x{self.rows} cells | walls {len(self.walls)} | "
                f"goals {len(self.goals)} | cyclers {len(self.cyclers)} | "
                f"refills {len(self.refills)} | launchers {len(self.launchers)} | "
                f"rails {len(self.rails)} | patrollers {len(self.patrollers)} "
                f"(tick {self.tick_tail}+{self.tick_period}) | "
                f"budget {self.max_steps}/{self.step_cost} "
                f"= {self.max_steps // self.step_cost} moves | exact={self.exact}")


def _cell_of(sprite):
    """Which lattice cell a sprite sits in.

    Upstream tests membership with `sprites_in_rect(x, y, 5, 5)`, i.e. a sprite
    belongs to the cell whose 5x5 rect contains its top-left corner. Floor
    division reproduces that for centred 3x3 pickups and 1px-offset launchers
    alike.
    """
    return names.pixel_to_cell(sprite.x, sprite.y)


def _launcher_direction(sprite):
    # Upstream derives the push direction from the sprite name suffix (ls20.py:1576-1583).
    suffix = sprite.name.rsplit("_", 1)[-1]
    return {"t": (0, -1), "b": (0, 1), "l": (-1, 0), "r": (1, 0)}.get(suffix, (0, 0))


def _overlaps(cell, x, y):
    """Does the 5x5 player box at `cell` overlap a 5x5 sprite box at (x, y)?

    Launcher pads sit one pixel off the lattice, so they overlap two adjacent
    cells and fire from either of them (upstream tests a plain bounding box in
    `try_launch`, ls20.py:1654).
    """
    px, py = names.cell_to_pixel(*cell)
    return px < x + names.CELL and x < px + names.CELL and py < y + names.CELL and y < py + names.CELL


def _launchers(sprites, blockers):
    """Resolve each pad to the cells it fires from and how far it throws.

    Upstream corrects the pad's one-pixel mounting offset by stepping once in
    the launch direction before scanning (ls20.py:1598), then walks whole cells
    until it meets a wall or goal pad and stops one cell short.
    """
    resolved = []
    for sprite in sprites:
        dx, dy = _launcher_direction(sprite)
        if (dx, dy) == (0, 0):
            continue
        origin = (sprite.x + dx, sprite.y + dy)
        distance = 0
        for step in range(1, 12):
            probe = (origin[0] + dx * names.CELL * step, origin[1] + dy * names.CELL * step)
            if probe in blockers:
                distance = max(0, step - 1)
                break
        triggers = tuple(cell for cell in
                         (names.pixel_to_cell(*origin), names.pixel_to_cell(sprite.x, sprite.y))
                         if _overlaps(cell, sprite.x, sprite.y))
        resolved.append({"cell": names.pixel_to_cell(*origin), "delta": (dx, dy),
                         "distance": distance, "triggers": triggers})
    return resolved


def extract(env):
    """Build a Layout from the level `env` is currently on."""
    game = env.game
    walls, cyclers, refills, rails = set(), {}, set(), []
    launcher_sprites, blockers = [], set()
    goal_pads = []
    # A rail-riding cycler is an ordinary cycler sprite that overlaps a rail, so
    # it has to be identified before the sprite walk or it would be recorded as a
    # static tile sitting on whatever cell it happened to be standing on.
    riders = rail_walks.riders(game)
    patrollers = rail_walks.patrollers(game)
    tick_tail, tick_period, moving_cyclers = rail_walks.schedule(patrollers)
    for sprite in game.current_level.get_sprites():
        tags = sprite.tags or ()
        cell = _cell_of(sprite)
        if id(sprite) in riders:
            continue
        if names.TAG_WALL in tags:
            walls.add(cell)
            blockers.add((sprite.x, sprite.y))
        elif names.TAG_GOAL_PAD in tags:
            goal_pads.append(sprite)
            blockers.add((sprite.x, sprite.y))
        elif names.TAG_STEP_REFILL in tags:
            refills.add(cell)
        elif names.TAG_LAUNCHER in tags:
            launcher_sprites.append(sprite)
        elif names.TAG_PATROL_RAIL in tags:
            rails.append(cell)
        else:
            for tag, kind in names.CYCLER_TAGS.items():
                if tag in tags:
                    cyclers[cell] = kind

    # Goal order must match upstream's own `get_sprites_by_tag` order, because
    # the required triples are indexed by that order.
    # Upstream freezes the blocker set at level load, so a cleared goal pad still
    # stops a launch (ls20.py:1857-1861). Resolve launchers against that snapshot.
    launchers = _launchers(launcher_sprites, blockers)

    ordered = game.current_level.get_sprites_by_tag(names.TAG_GOAL_PAD)
    goals = [(_cell_of(pad), triple) for pad, triple in zip(ordered, env.goal_triples())]
    goal_at = {cell: index for index, (cell, _) in enumerate(goals)}

    # The planner assumes one special per cell, which upstream's levels respect.
    # Overlaps would make effect order depend on sprite list order, so refuse.
    # A patroller has to be clear of the others at *every* tick of its walk, not
    # just where it starts; on the shipped levels the rails keep to empty lanes.
    specials = list(cyclers) + list(refills) + [cell for cell, _ in goals]
    walked = set().union(*moving_cyclers) if patrollers else set()
    if (len(specials) != len(set(specials)) or set(specials) & walls
            or walked & (set(specials) | walls)
            or any(len(tick) != len(patrollers) for tick in moving_cyclers)):
        raise ValueError("level places two interacting tiles on one cell")

    hud = getattr(game, names.ATTR_STEP_HUD)
    return Layout(
        cols=names.GRID_COLS, rows=names.GRID_ROWS, walls=walls, cyclers=cyclers,
        refills=refills, launchers=launchers, rails=rails, goals=goals, goal_at=goal_at,
        patrollers=patrollers, moving_cyclers=moving_cyclers,
        tick_tail=tick_tail, tick_period=tick_period, tick_span=tick_tail + tick_period,
        start_cell=env.player_cell(), start_triple=env.triple(),
        max_steps=getattr(hud, names.HUD_MAX_STEPS),
        step_cost=getattr(hud, names.HUD_STEP_COST),
        # Upstream's "your triple now matches" hint fires only on level 1, and it
        # returns from step() before the budget is charged (ls20.py:1970-1972),
        # making such a move free. Model it or level 1 plans come out wrong.
        match_hint=(env.level_index == 0),
        fog=env.fog(), level_index=env.level_index)
