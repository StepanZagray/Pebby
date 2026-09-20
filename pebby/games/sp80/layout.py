"""Logical action-boundary snapshots for SP80.

The planner explores editing states but asks the real engine to execute every
move and flow.  Keeping an independent engine snapshot here makes that search
exact even for SP80's pixel collision and water branching rules.
"""

from dataclasses import dataclass, field

from . import names


@dataclass(frozen=True)
class Piece:
    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    grid_size: tuple[int, int]
    rotation_k: int
    pieces: tuple[Piece, ...]
    selected: int | None
    cups: tuple[tuple[int, int, int, int], ...]
    sources: tuple[tuple[int, int], ...]
    sinks: tuple[tuple[int, int, int, int], ...]
    steps_left: int
    failed_flows: int
    unsupported: tuple[str, ...] = ()
    snapshot: object = field(default=None, compare=False, repr=False)

    @property
    def exact(self):
        return not self.unsupported and self.snapshot is not None

    def key(self):
        return (
            tuple((piece.x, piece.y) for piece in self.pieces),
            self.selected,
            self.failed_flows,
        )

    def describe(self):
        return (
            f"{self.grid_size[0]}x{self.grid_size[1]} | pieces {len(self.pieces)} | "
            f"sources {len(self.sources)} | cups {len(self.cups)} | sinks {len(self.sinks)} | "
            f"steps {self.steps_left} | failures {self.failed_flows} | exact={self.exact}"
        )


def _selected_index(movables, selected):
    return next((index for index, sprite in enumerate(movables) if sprite is selected), None)


def extract(env):
    """Capture the current editing state and an independent real-engine snapshot."""
    movables = env.movables()
    unsupported = []
    if env.mode != "change":
        unsupported.append(f"snapshot is in {env.mode!r} mode rather than an action boundary")
    if not movables:
        unsupported.append("level has no movable pipe or deflector")
    if not env.cups():
        unsupported.append("level has no cups")
    sources = env.sprites_by_tag(names.TAG_SOURCE)
    water = env.sprites_by_tag(names.TAG_WATER)
    if not sources and not water:
        unsupported.append("level has no water source or initial flow head")
    pieces = tuple(
        Piece(
            index=index,
            name=sprite.name,
            x=int(sprite.x),
            y=int(sprite.y),
            width=int(sprite.width),
            height=int(sprite.height),
            tags=tuple(sprite.tags),
        )
        for index, sprite in enumerate(movables)
    )
    return Layout(
        grid_size=env.grid_size,
        rotation_k=env.rotation_k,
        pieces=pieces,
        selected=_selected_index(movables, env.selected()),
        cups=tuple((int(s.x), int(s.y), int(s.width), int(s.height)) for s in env.cups()),
        sources=tuple((int(s.x), int(s.y)) for s in sources),
        sinks=tuple(
            (int(s.x), int(s.y), int(s.width), int(s.height))
            for s in env.sprites_by_tag(names.TAG_SINK)
        ),
        steps_left=env.steps_left,
        failed_flows=env.failed_flows,
        unsupported=tuple(unsupported),
        snapshot=env.clone(),
    )


def from_snapshot(env):
    """Alias used by tests and diagnostics to stress snapshot semantics."""
    return extract(env)
