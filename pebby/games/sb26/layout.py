"""Exact action-boundary extraction for the full SB26 frame grammar."""

from dataclasses import dataclass

from arcengine import GameState

from . import names


Position = tuple[int, int]


@dataclass(frozen=True)
class Tile:
    """A regular colour tile or a recursive frame link."""

    kind: str
    colour: int
    position: Position
    movable: bool

    @property
    def signature(self):
        return self.kind, self.colour


@dataclass(frozen=True)
class Frame:
    """One native frame in the engine's traversal order."""

    colour: int
    position: Position
    slots: tuple[Position, ...]
    occupants: tuple[Tile | None, ...]

    @property
    def arity(self):
        return len(self.slots)


@dataclass(frozen=True)
class Layout:
    """Complete static grammar plus the live placement/history state."""

    frames: tuple[Frame, ...]
    goals: tuple[int, ...]
    tiles: tuple[Tile, ...]
    selected: Position | None
    energy: int
    initial_energy: int
    history_depth: int
    level_index: int
    state: str
    connector_count: int = 0
    unsupported: tuple[str, ...] = ()

    @property
    def exact(self):
        return not self.unsupported

    @property
    def slots(self):
        return tuple(position for frame in self.frames for position in frame.slots)

    @property
    def occupants(self):
        return tuple(tile for frame in self.frames for tile in frame.occupants)

    @property
    def fixed_tiles(self):
        return tuple(tile for tile in self.tiles if not tile.movable)

    @property
    def movable_tiles(self):
        return tuple(tile for tile in self.tiles if tile.movable)

    def key(self):
        return (
            tuple(
                (
                    frame.colour,
                    frame.position,
                    tuple(None if tile is None else (tile.kind, tile.colour, tile.position, tile.movable)
                          for tile in frame.occupants),
                )
                for frame in self.frames
            ),
            tuple((tile.kind, tile.colour, tile.position, tile.movable) for tile in self.tiles),
            self.goals,
            self.selected,
            self.energy,
            self.history_depth,
            self.connector_count,
        )


def _state_name(state):
    return str(getattr(state, "value", state)).upper()


def _animations_at_rest(game):
    checks = (
        (names.ATTR_SWAP_ANIMATION, -1),
        (names.ATTR_DELAY, 0),
        (names.ATTR_FLASH_ANIMATION, -1),
        (names.ATTR_MOVEMENT_FRAME, -1),
        (names.ATTR_TILE_FILL_ANIMATION, -1),
        (names.ATTR_RESET_ANIMATION, -1),
        (names.ATTR_WIN_ANIMATION, -1),
        (names.ATTR_FAILURE_FLASH, -1),
    )
    return all(getattr(game, field) == expected for field, expected in checks) and not (
        getattr(game, names.ATTR_COLOUR_ANIMATIONS) or getattr(game, names.ATTR_MOVEMENTS)
    )


def extract(env):
    """Extract every shipped mechanic without advancing the real engine."""
    game = env.game
    reasons = []
    state = _state_name(env.state)
    if env.state in (GameState.WIN, GameState.GAME_OVER):
        reasons.append(f"terminal engine state {state}")
    if not _animations_at_rest(game):
        reasons.append("game is between action boundaries")

    native_frames = tuple(getattr(game, names.ATTR_FRAMES))
    if not native_frames:
        reasons.append("level has no frames")

    tiles = []
    by_position = {}
    for sprite in tuple(getattr(game, names.ATTR_TILES)):
        if sprite.name == names.SPRITE_TILE:
            kind = "regular"
        elif sprite.name == names.SPRITE_LINK:
            kind = "link"
        else:
            reasons.append(f"unknown tile prototype {sprite.name!r}")
            continue
        position = int(sprite.x), int(sprite.y)
        tile = Tile(
            kind=kind,
            colour=names.tile_colour(sprite),
            position=position,
            movable=names.TAG_CLICK in sprite.tags,
        )
        if position in by_position:
            reasons.append(f"multiple tiles occupy {position}")
        by_position[position] = tile
        tiles.append(tile)

    frames = []
    frame_colours = []
    all_slots = set()
    for sprite in native_frames:
        arity = names.ARITY_FOR_FRAME.get(sprite.name)
        if arity is None:
            reasons.append(f"unknown frame prototype {sprite.name!r}")
            continue
        colour = int(sprite.pixels[0, 0])
        frame_colours.append(colour)
        slots = names.frame_cells(sprite, arity)
        if any(position in all_slots for position in slots):
            reasons.append("frame slots overlap")
        all_slots.update(slots)
        frames.append(
            Frame(
                colour=colour,
                position=(int(sprite.x), int(sprite.y)),
                slots=slots,
                occupants=tuple(by_position.get(position) for position in slots),
            )
        )
    if len(set(frame_colours)) != len(frame_colours):
        reasons.append("duplicate frame colours make native link targets ambiguous")

    for tile in tiles:
        if tile.position not in all_slots and tile.position[1] < names.TRAY_MIN_Y:
            reasons.append(f"tile at {tile.position} is neither framed nor in the tray")
        if tile.kind == "link" and tile.colour not in frame_colours:
            reasons.append(f"link colour {tile.colour} has no matching frame")

    goals = tuple(names.goal_colour(goal) for goal in getattr(game, names.ATTR_GOALS))
    if not goals:
        reasons.append("level has no ordered goals")

    selected_sprite = getattr(game, names.ATTR_SELECTION)
    selected = None
    if selected_sprite is not None:
        selected = int(selected_sprite.x), int(selected_sprite.y)
        if selected not in by_position or not by_position[selected].movable:
            reasons.append("selected tile is absent or fixed")

    unsupported = tuple(dict.fromkeys(reasons))
    return Layout(
        frames=tuple(frames),
        goals=goals,
        tiles=tuple(tiles),
        selected=selected,
        energy=env.energy,
        initial_energy=env.initial_energy,
        history_depth=env.history_depth,
        level_index=env.level_index,
        state=state,
        connector_count=len(env.level.get_sprites_by_name(names.SPRITE_CONNECTOR)),
        unsupported=unsupported,
    )
