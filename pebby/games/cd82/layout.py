"""Read a level's logical structure out of a running CD82 game.

The planner reasons over *atoms*, not pixels. The win check ignores both
diagonals, and every paint operation (eight ACTION5 regions plus, when the
level ships the indicator, four cap strips) colours whole unions of atoms,
where an atom is a maximal set of off-diagonal cells that every operation
treats identically. Without the indicator there are 8 atoms of 10 cells; with
it, 16. Because the canvas starts uniform and operations only ever paint whole
atoms, the canvas is always uniform per atom and the search state is just one
colour per atom.
"""

import numpy as np

from . import names


class Layout:
    """Static description of one level plus the state the player is in.

    Fields:
      ops          tuple of (dial, kind, atoms) where kind is "paint" (ACTION5)
                   or "cap" (click on the indicator); atoms is a tuple of ints
      atom_cells   tuple of tuple of (row, col) per atom
      palette      tuple of (colour, click_x, click_y) swatches
      start        (dial, colour, atom colours tuple)
      target       tuple of colour per atom, or None if the target is not
                   uniform per atom (then no sequence of paints can match it)
      actions_left usable actions before the budget loses
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    @property
    def exact(self):
        # Every mechanic the game has (dial moves, swatch clicks, region and cap
        # paints, the off-diagonal win check, the per-level budget) is modelled.
        return True

    @property
    def solvable_shape(self):
        return self.target is not None

    def canvas_from_atoms(self, colours, fill=None):
        """A 10x10 grid with `colours` per atom; diagonal cells get `fill` (or 0)."""
        grid = np.full((names.CANVAS_SIZE, names.CANVAS_SIZE), 0 if fill is None else fill, dtype=np.int8)
        for atom, cells in enumerate(self.atom_cells):
            for r, c in cells:
                grid[r, c] = colours[atom]
        return grid


def operations(has_indicator):
    """[(dial, kind, mask)] for every paint the level supports."""
    ops = [(dial, "paint", names.region_mask(dial)) for dial in range(names.DIAL_COUNT)]
    if has_indicator:
        ops += [(dial, "cap", names.cap_mask(dial)) for dial in names.EDGE_DIALS]
    return ops


def atomise(has_indicator):
    """(ops with atom tuples, atom_cells) for a level with or without the indicator."""
    raw = operations(has_indicator)
    compare = names.compare_mask()
    signatures = {}
    for r in range(names.CANVAS_SIZE):
        for c in range(names.CANVAS_SIZE):
            if not compare[r, c]:
                continue
            key = tuple(bool(mask[r, c]) for _, _, mask in raw)
            signatures.setdefault(key, []).append((r, c))
    keys = sorted(signatures, key=lambda k: signatures[k][0])
    atom_cells = tuple(tuple(signatures[k]) for k in keys)
    ops = tuple((dial, kind, tuple(i for i, k in enumerate(keys) if k[j]))
                for j, (dial, kind, _) in enumerate(raw))
    return ops, atom_cells


def atom_colours(grid, atom_cells):
    """Colour per atom, or None if some atom is not uniform in `grid`."""
    grid = np.asarray(grid)
    out = []
    for cells in atom_cells:
        values = {int(grid[r, c]) for r, c in cells}
        if len(values) != 1:
            return None
        out.append(values.pop())
    return tuple(out)


def build(*, dial, color, canvas, target, swatches, has_indicator, actions_used=0, level_index=0):
    ops, atom_cells = atomise(has_indicator)
    start_atoms = atom_colours(canvas, atom_cells)
    if start_atoms is None:
        raise ValueError("canvas is not uniform per atom; the game cannot have produced it")
    palette = tuple((int(c), int(x), int(y)) for c, x, y in swatches)
    return Layout(ops=ops, atom_cells=atom_cells, palette=palette,
                  has_indicator=bool(has_indicator),
                  start=(int(dial), int(color), start_atoms),
                  target=atom_colours(target, atom_cells),
                  target_grid=np.asarray(target).tolist(),
                  actions_left=names.MAX_ACTIONS - int(actions_used),
                  level_index=int(level_index))


def extract(env):
    """The Layout of `env`'s current level in its current state."""
    return build(dial=env.dial(), color=env.color(), canvas=env.canvas(), target=env.target(),
                 swatches=env.swatches(), has_indicator=env.has_indicator(),
                 actions_used=env.actions_used, level_index=env.level_index)
