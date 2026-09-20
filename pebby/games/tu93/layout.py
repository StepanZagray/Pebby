"""Symbolic TU93 layouts extracted from a live upstream game.

Coordinates in this module are maze nodes, not pixels.  Node ``(c, r)`` is at
``maze_origin + (6*c, 6*r)``.  The maze bitmap is the source of truth for
edges, exactly as :class:`Tu93` treats a value-2 block halfway between nodes as
an open passage.
"""

from dataclasses import dataclass

from . import names


Cell = tuple[int, int]
Actor = tuple[Cell, int]
Tail = tuple[Cell, int, tuple[int, ...] | None]


@dataclass(frozen=True)
class Layout:
    origin: tuple[int, int]
    grid_size: tuple[int, int]
    nodes: frozenset[Cell]
    edges: frozenset[tuple[Cell, Cell]]
    exits: frozenset[Cell]
    head: Cell | None
    head_rotation: int
    hunters: tuple[Actor, ...]
    patrollers: tuple[Actor, ...]
    tails: tuple[Tail, ...]
    max_steps: int
    steps_left: int
    unsupported: tuple[str, ...] = ()

    @property
    def exact(self):
        return not self.unsupported

    def open(self, cell: Cell, action: int) -> bool:
        dx, dy = names.ACTION_DELTA[action]
        target = (cell[0] + dx, cell[1] + dy)
        return _edge(cell, target) in self.edges

    def key(self):
        return (self.head, self.head_rotation, self.hunters, self.patrollers, self.tails)

    def describe(self):
        return (
            f"{len(self.nodes)} nodes/{len(self.edges)} edges | exits {len(self.exits)} | "
            f"hunters {len(self.hunters)} | patrollers {len(self.patrollers)} | "
            f"tails {len(self.tails)} | budget {self.max_steps} (left {self.steps_left}) | "
            f"exact={self.exact}"
        )


def _edge(a: Cell, b: Cell):
    return (a, b) if a <= b else (b, a)


def _cell(origin, sprite):
    cell = names.pixel_to_cell(origin, sprite.x, sprite.y)
    if cell is None:
        raise ValueError(
            f"sprite {sprite.name} at ({sprite.x}, {sprite.y}) is off the maze lattice "
            f"with origin {origin}"
        )
    return cell


def _maze_graph(maze):
    pixels = maze.pixels
    height, width = pixels.shape
    # NODE=0 is cosmetic.  Upstream never tests it; it only tests whether the
    # midpoint three pixels from the current node is PASSAGE=2.  Include every
    # lattice intersection inside the bitmap and derive edges solely from that
    # exact midpoint predicate.  Isolated transparent intersections are inert.
    nodes = {
        (x // names.CELL, y // names.CELL)
        for y in range(0, height, names.CELL)
        for x in range(0, width, names.CELL)
    }
    edges = set()
    # Scan the exact midpoint coordinates the four action branches inspect.
    # A passage on the last partial 6px band may legally lead one lattice node
    # beyond the bitmap, so add both endpoints rather than requiring cosmetic
    # endpoint pixels to exist.
    for y in range(0, height, names.CELL):
        for x in range(names.BLOCK, width, names.CELL):
            if int(pixels[y, x]) == names.PASSAGE:
                a = ((x - names.BLOCK) // names.CELL, y // names.CELL)
                b = (a[0] + 1, a[1])
                nodes.update((a, b))
                edges.add(_edge(a, b))
    for y in range(names.BLOCK, height, names.CELL):
        for x in range(0, width, names.CELL):
            if int(pixels[y, x]) == names.PASSAGE:
                a = (x // names.CELL, (y - names.BLOCK) // names.CELL)
                b = (a[0], a[1] + 1)
                nodes.update((a, b))
                edges.add(_edge(a, b))
    return frozenset(nodes), frozenset(edges)


def extract(env):
    """Return an exact logical snapshot at an action boundary.

    Unknown/malformed mechanics are recorded in ``Layout.unsupported``.  The
    planner treats such a layout as inconclusive rather than claiming it has no
    solution.
    """
    maze = env.maze()
    origin = (maze.x, maze.y)
    nodes, edges = _maze_graph(maze)
    unsupported = []
    if env.phase() != 0:
        unsupported.append("game is between action boundaries")

    heads = env.heads()
    if len(heads) != 1:
        unsupported.append(f"expected one live head, found {len(heads)}")
    head = _cell(origin, heads[0]) if len(heads) == 1 else None
    head_rotation = int(heads[0].rotation) if len(heads) == 1 else 0

    def actors(tag):
        result = []
        for sprite in env.sprites_by_tag(tag):
            if sprite.pixels.shape != (3, 3):
                unsupported.append(f"animated {tag} sprite is not at an action boundary")
                continue
            result.append((_cell(origin, sprite), int(sprite.rotation)))
        return tuple(result)

    hunters = actors(names.TAG_HUNTER)
    patrollers = actors(names.TAG_PATROLLER)
    queues = getattr(env.game, names.ATTR_TAIL_QUEUES)
    tails = []
    for sprite in env.sprites_by_tag(names.TAG_TAIL):
        if sprite.pixels.shape != (3, 3):
            unsupported.append("animated tail is not at an action boundary")
            continue
        queue = tuple(int(r) for r in queues[sprite]) if sprite in queues else None
        active = int(sprite.pixels[0, 1]) == names.ACTIVE_MARK
        if active != (queue is not None):
            unsupported.append("tail active marker and rotation queue disagree")
        tails.append((_cell(origin, sprite), int(sprite.rotation), queue))

    exits = frozenset(_cell(origin, sprite) for sprite in env.sprites_by_tag(names.TAG_EXIT))
    if not exits:
        unsupported.append("level has no exit")
    return Layout(
        origin=origin,
        grid_size=tuple(int(value) for value in env.level.grid_size),
        nodes=nodes,
        edges=edges,
        exits=exits,
        head=head,
        head_rotation=head_rotation,
        hunters=hunters,
        patrollers=patrollers,
        tails=tuple(tails),
        max_steps=int(env.max_steps()),
        steps_left=int(env.steps_left()),
        unsupported=tuple(unsupported),
    )


def edge(a, b):
    """Canonical undirected edge helper used by generation and tests."""
    return _edge(tuple(a), tuple(b))
