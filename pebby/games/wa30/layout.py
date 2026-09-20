"""Read a level's logical structure out of a running WA30 game.

The planner reasons over 16x16 cells, not sprites. This module is the only
place that knows how upstream's sprite soup maps onto the lattice, and it works
identically for the shipped levels and for generated ones.
"""

from . import names


class Layout:
    """Static description of one level plus the dynamic state it is in.

    Coordinates are cells: pixel (x, y) -> (x // 4, y // 4). Everything upstream
    positions is a multiple of 4 in the shipped levels; `extract` refuses a level
    where an actor or box is misaligned, because the planner's lattice would not
    match the engine's then.

    The full planner models helpers, thieves, holder identity, destroyed
    thieves, and the two deliberately stale target caches used by the native
    robot pathfinders. ``exact`` therefore describes alignment validity rather
    than excluding autonomous actors.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    @property
    def exact(self):
        return self.aligned

    def in_bounds(self, cell):
        col, row = cell
        return 0 <= col < self.cols and 0 <= row < self.rows

    def actor_free(self, cell, boxes):
        """Where an unencumbered actor may step (upstream `kblzhbvysd`, wa30.py:993)."""
        return (self.in_bounds(cell) and cell not in self.walls and cell not in self.fences
                and cell not in boxes and cell != self.player)

    def describe(self):
        return (f"{self.cols}x{self.rows} cells | walls {len(self.walls)} | fences {len(self.fences)} | "
                f"goal cells {len(self.goals)} | boxes {len(self.boxes)} | helpers {len(self.helpers)} | "
                f"thieves {len(self.thieves)} | budget {self.max_steps} (left {self.steps_left}) | "
                f"exact={self.exact}")

    def key(self):
        """The planner's state key for the current dynamic state."""
        return (
            self.player,
            self.rotation,
            self.boxes,
            self.helpers,
            self.thieves,
            self.holds,
            self.helper_targets,
            self.thief_targets,
        )


def _cell(sprite):
    if sprite.x % names.CELL or sprite.y % names.CELL:
        raise ValueError(f"sprite {sprite.name} at ({sprite.x}, {sprite.y}) is not on the 4px lattice")
    return names.pixel_to_cell(sprite.x, sprite.y)


def _covered_cells(sprite):
    """Cells whose origin pixel lies inside the sprite (upstream tests the origin pixel)."""
    cells = set()
    for col in range(names.GRID_COLS):
        for row in range(names.GRID_ROWS):
            x, y = names.cell_to_pixel(col, row)
            if sprite.x <= x < sprite.x + sprite.width and sprite.y <= y < sprite.y + sprite.height:
                cells.add((col, row))
    return cells


def extract(env):
    """Build a Layout from the env's current level and position."""
    game = env.game
    level = game.current_level
    walls, fences, goals, bad = set(), set(), set(), set()
    boxes, helpers, thieves = [], [], []
    player = None
    for sprite in level.get_sprites():
        tags = set(sprite.tags)
        if names.TAG_PLAYER in tags:
            player = sprite
        elif names.TAG_BOX in tags:
            boxes.append(sprite)
        elif names.TAG_HELPER in tags:
            helpers.append(sprite)
        elif names.TAG_THIEF in tags:
            thieves.append(sprite)
        elif names.TAG_FENCE in tags:
            fences.add(_cell(sprite))
        elif names.TAG_GOAL in tags:
            goals |= _covered_cells(sprite)
        elif names.TAG_BAD in tags:
            bad |= _covered_cells(sprite)
        elif sprite.is_collidable:
            # Upstream registers only the origin pixel of a collidable sprite as
            # an obstacle (wa30.py:906-908), so a 64x20 slab blocks one cell.
            walls.add(_cell(sprite))
    if player is None:
        raise ValueError("level has no player sprite")

    held_by = getattr(game, names.ATTR_HELD_BY)
    box_cells = tuple(_cell(box) for box in boxes)
    player_cell = _cell(player)
    held = None
    if player in held_by:
        box = held_by[player]
        held = (box.x // names.CELL - player_cell[0], box.y // names.CELL - player_cell[1])

    actors = [player, *helpers, *thieves]
    box_index = {box: index for index, box in enumerate(boxes)}
    holds = tuple(box_index[held_by[actor]] if actor in held_by else -1 for actor in actors)

    hud = getattr(game, names.ATTR_STEP_HUD)
    return Layout(
        cols=names.GRID_COLS, rows=names.GRID_ROWS,
        walls=frozenset(walls), fences=frozenset(fences), goals=frozenset(goals), bad=frozenset(bad),
        player=player_cell, rotation=int(player.rotation), held=held, boxes=box_cells,
        helpers=tuple(_cell(helper) for helper in helpers),
        thieves=tuple(_cell(thief) for thief in thieves), holds=holds,
        helper_targets=tuple(sorted(names.pixel_to_cell(x, y) for x, y in getattr(game, names.ATTR_HELPER_TARGETS))),
        thief_targets=tuple(sorted(names.pixel_to_cell(x, y) for x, y in getattr(game, names.ATTR_THIEF_TARGETS))),
        aligned=True, generated=bool(level.get_data("PebbyGenerated")),
        max_steps=getattr(hud, names.HUD_MAX_STEPS), steps_left=getattr(hud, names.HUD_CURRENT_STEPS),
    )
