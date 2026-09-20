"""Compact symbolic state of the current FT09 level.

`extract(env)` reads the live level once and produces everything an exact
solver needs: the cells, their starting palette indices, which cells each
click advances, and for each constrained cell the set of palette indices it
may end on. Every lookup goes through the engine's own `get_sprite_at` with
the same tag order the game uses (ft09.py:2380-2384, 2422-2425), so the model
cannot disagree with the game about which sprite a stencil offset or a
constraint border hits.
"""

from dataclasses import dataclass, field

from . import names


def _cell_at(level, x, y):
    """The cell the game would find at grid (x, y): ordinary first, then special."""
    sprite = level.get_sprite_at(x, y, names.TAG_CELL)
    if sprite is None:
        sprite = level.get_sprite_at(x, y, names.TAG_SPECIAL_CELL)
    return sprite


@dataclass(frozen=True)
class Layout:
    palette: tuple            # colour values in cycle order
    cells: tuple              # (x, y, special) per cell, engine order
    initial: tuple            # palette index per cell at level start
    affects: tuple            # per cell: tuple of cell indices advanced by one click (with multiplicity)
    allowed: tuple            # per cell: frozenset of permitted final palette indices, or None if unconstrained
    budget: int               # remaining clicks (winning click included)
    clicks: tuple             # per cell: (x, y) display coordinates that hit it
    stencil: tuple = names.IDENTITY_STENCIL
    constraints: tuple = field(default=())   # (x, y, centre colour, 3x3 mask) for reference / rebuilding

    @property
    def size(self):
        return len(self.cells)

    @property
    def colours(self):
        return len(self.palette)

    def click(self, state, index):
        """State after clicking cell `index`. States are tuples of palette indices."""
        state = list(state)
        k = self.colours
        for target in self.affects[index]:
            state[target] = (state[target] + 1) % k
        return tuple(state)

    def satisfied(self, state):
        return all(allowed is None or state[i] in allowed for i, allowed in enumerate(self.allowed))

    def action(self, index):
        x, y = self.clicks[index]
        return (names.ACTION_CLICK, x, y)

    def clicks_to_actions(self, indices):
        return [self.action(i) for i in indices]


def extract(env):
    level = env.level
    game = env.game
    palette = tuple(int(c) for c in env.palette())
    stencil = env.stencil()
    cells = env.cells()
    index_of = {id(sprite): i for i, sprite in enumerate(cells)}

    # Starting colours: after on_set_level every cell is palette[0] (ft09.py:2340-2347),
    # but read them anyway so a clone mid-level extracts correctly.
    initial = []
    for sprite in cells:
        colour = int(sprite.pixels[1][1])
        if colour not in palette:
            raise ValueError(f"cell colour {colour} is not in palette {palette}")
        initial.append(palette.index(colour))

    affects, clicks, kinds = [], [], []
    special = set(id(s) for s in getattr(game, names.ATTR_SPECIAL_CELLS))
    for sprite in cells:
        is_special = id(sprite) in special
        if is_special:  # ft09.py:2362-2368: own colour-6 pixels plus the centre
            mask = [[1 if int(sprite.pixels[r][c]) == names.STENCIL_MARKER else 0 for c in range(3)]
                    for r in range(3)]
            mask[1][1] = 1
        else:
            mask = [list(row) for row in stencil]
        hit = []
        # Upstream iterates i (column) outer, j (row) inner (ft09.py:2377-2379); order is irrelevant
        # because increments commute, but keep it anyway.
        for col in range(3):
            for row in range(3):
                if mask[row][col] == 1:
                    dx, dy = names.STENCIL_OFFSETS[(row, col)]
                    target = _cell_at(level, sprite.x + dx, sprite.y + dy)
                    if target is not None:
                        hit.append(index_of[id(target)])
        affects.append(tuple(hit))
        kinds.append((int(sprite.x), int(sprite.y), is_special))
        # A click must resolve, through the camera, to this very sprite.
        x, y = names.cell_click(sprite.x, sprite.y)
        grid = game.camera.display_to_grid(x, y)
        if grid is None or _cell_at(level, *grid) is not sprite:
            raise ValueError(f"cell at {(sprite.x, sprite.y)} is not clickable at {(x, y)}")
        clicks.append((x, y))

    allowed = [None] * len(cells)
    constraints = []
    for sprite in env.constraints():
        centre = int(sprite.pixels[1][1])
        mask = tuple(tuple(int(sprite.pixels[r][c]) for c in range(3)) for r in range(3))
        constraints.append((int(sprite.x), int(sprite.y), centre, mask))
        for (row, col), (dx, dy) in names.NEIGHBOUR_OFFSETS.items():
            target = _cell_at(level, sprite.x + dx, sprite.y + dy)
            if target is None:
                continue
            must_match = mask[row][col] == names.MATCH_FLAG
            ok = frozenset(i for i, colour in enumerate(palette) if (colour == centre) == must_match)
            i = index_of[id(target)]
            allowed[i] = ok if allowed[i] is None else allowed[i] & ok

    budget, _ = env.budget()
    return Layout(palette=palette, cells=tuple(kinds), initial=tuple(initial), affects=tuple(affects),
                  allowed=tuple(allowed), budget=int(budget), clicks=tuple(clicks), stencil=stencil,
                  constraints=tuple(constraints))
