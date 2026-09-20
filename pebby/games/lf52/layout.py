"""Exact logical snapshots for LF52's complete ten-level ruleset."""

from dataclasses import dataclass

from arcengine import GameState

from . import names


Cell = tuple[int, int]
PegEntity = tuple[Cell, str]


@dataclass(frozen=True)
class Layout:
    """Immutable source-semantics snapshot used by the bounded planner.

    ``ordinary_cells`` and ``moving_cells`` are deliberately separate: native
    landing legality depends on the exact entity stack, not merely on whether a
    coordinate looks like a hole.  Rails remain fixed while moving holes carry
    any peg or blocker colocated with them.
    """

    ordinary_cells: frozenset[Cell]
    moving_cells: frozenset[Cell]
    rails: frozenset[Cell]
    peg_entities: tuple[PegEntity, ...]
    obstacles: frozenset[Cell]
    selected: Cell | None
    selected_kind: str | None
    origin: tuple[int, int]
    tile_size: tuple[int, int]
    logical_level: int
    action_count: int
    actions_left: int
    history_depth: int
    initial_blue_count: int
    reset_prompt: bool
    pending_auto_undo: bool
    generated_kind: str | None
    unsupported: tuple[str, ...] = ()

    @property
    def exact(self):
        return not self.unsupported

    @property
    def cells(self):
        return self.ordinary_cells | self.moving_cells

    @property
    def pegs(self):
        """Compatibility view containing occupied peg coordinates."""
        return frozenset(cell for cell, _ in self.peg_entities)

    @property
    def peg_map(self):
        return dict(self.peg_entities)

    @property
    def key(self):
        return (
            self.peg_entities,
            self.moving_cells,
            self.obstacles,
            self.selected,
            self.selected_kind,
            self.origin,
            self.reset_prompt,
        )

    @property
    def target_nonblue(self):
        return 2 if self.logical_level in (6, 7) else 1

    def click(self, cell, origin=None):
        origin = self.origin if origin is None else origin
        return names.cell_click(origin, cell, self.tile_size[0])

    def visible(self, cell, origin=None):
        origin = self.origin if origin is None else origin
        left = origin[0] + cell[0] * self.tile_size[0]
        top = origin[1] + cell[1] * self.tile_size[1]
        return (
            0 <= left
            and left + self.tile_size[0] <= names.DISPLAY
            and 0 <= top
            and top + self.tile_size[1] <= names.DISPLAY
        )


def _entity_cells(entity):
    return {tuple(map(int, cell)) for cell in getattr(entity, names.PROP_CELLS)}


def _entity_cell(entity):
    return tuple(map(int, getattr(entity, names.PROP_GRID_POSITION)))


def extract(env):
    """Return a complete logical snapshot without mutating ``env``."""
    grid = env.grid
    entities = list(getattr(grid, names.METHOD_GRID_ENTITIES))
    ordinary = set()
    moving = set()
    rails = set()
    pegs = []
    obstacles = set()
    unsupported = []
    unexpected = set()

    for entity in entities:
        entity_name = str(entity.name)
        occupied = _entity_cells(entity)
        if entity_name == names.HOLE:
            ordinary.update(occupied)
        elif entity_name == names.MOVING_HOLE:
            moving.update(occupied)
        elif entity_name.startswith(names.RAIL_PREFIX):
            rails.update(occupied)
        elif entity_name in names.PEG_KINDS:
            pegs.append((_entity_cell(entity), entity_name))
        elif entity_name.startswith(names.OBSTACLE):
            obstacles.update(occupied)
        else:
            unexpected.add(entity_name)

    if unexpected:
        unsupported.append("unmodelled logical entities: " + ", ".join(sorted(unexpected)))
    if env.state != GameState.NOT_FINISHED:
        unsupported.append(f"game state is {env.state.value}, not an active level")
    if not ordinary and not moving:
        unsupported.append("board has no landing holes")
    if not pegs:
        unsupported.append("board has no pegs")
    if any(cell not in ordinary and cell not in moving for cell, _ in pegs):
        unsupported.append("a peg is not backed by a logical hole")
    if any(cell not in ordinary and cell not in moving for cell in obstacles):
        unsupported.append("an obstacle is not backed by a logical hole")
    if names.ACTION_CLICK not in env.available_actions:
        unsupported.append("the native click action is unavailable")

    world = env.world
    selected_parent = getattr(getattr(world, names.ATTR_SELECTED_TOKEN), "qoifrofmiu")
    selected = None
    selected_kind = None
    if selected_parent is not None:
        selected = _entity_cell(selected_parent)
        selected_kind = str(selected_parent.name)
        if (selected, selected_kind) not in pegs:
            unsupported.append("selection points at a missing peg")

    descriptor = env.generated_descriptor
    kind = descriptor.get("kind") if isinstance(descriptor, dict) else None
    if kind not in (None, names.GENERATED_KIND, names.FULL_GENERATED_KIND):
        unsupported.append(f"unknown generated layout kind {kind!r}")

    origin = tuple(map(int, grid.cdpcbbnfdp))
    tile_size = tuple(map(int, grid.tile_size))
    if tile_size != (names.TILE, names.TILE):
        unsupported.append(f"tile size {tile_size!r} is not {(names.TILE, names.TILE)!r}")

    return Layout(
        ordinary_cells=frozenset(ordinary),
        moving_cells=frozenset(moving),
        rails=frozenset(rails),
        peg_entities=tuple(sorted(pegs)),
        obstacles=frozenset(obstacles),
        selected=selected,
        selected_kind=selected_kind,
        origin=origin,
        tile_size=tile_size,
        logical_level=env.logical_level,
        action_count=env.action_count,
        actions_left=env.steps_left,
        history_depth=env.history_depth,
        initial_blue_count=int(getattr(world, "lzoqlpcwzpu", 0)),
        reset_prompt=bool(getattr(world, names.ATTR_RESET_PROMPT, False)),
        pending_auto_undo=bool(getattr(world, "yxhdgwykzi", False)),
        generated_kind=kind,
        unsupported=tuple(unsupported),
    )
