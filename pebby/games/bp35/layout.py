"""Symbolic extraction for the verified BP35 planning subsets."""

from dataclasses import dataclass

from arcengine import GameState

from . import names


@dataclass(frozen=True)
class Layout:
    snapshot: object
    player: tuple[int, int]
    gem: tuple[int, int] | None
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    hazards: frozenset[tuple[int, int]]
    entity_state: tuple[tuple, ...]
    generated_kind: str | None
    generated_exact: bool
    generated_pristine: bool
    official1_initial: bool
    action_count: int
    history_depth: int
    move_count: int
    facing_right: bool
    action_budget: int
    unsupported: tuple[str, ...]

    @property
    def exact(self):
        return self.generated_exact or self.official1_initial

    @property
    def key(self):
        """A diagnostic key; the planner does not merge custom-history states."""
        return (
            self.player,
            self.gem,
            self.entity_state,
            self.action_count,
            self.history_depth,
            self.move_count,
            self.facing_right,
        )


def _history_depth(world):
    history = getattr(world, names.ATTR_HISTORY, None)
    if history is None:
        return 0
    return len(getattr(history, names.ATTR_HISTORY_STACK, ()))


def extract(env):
    """Read the custom BP35 grid without changing the real engine."""
    world = env.world
    grid = getattr(world, names.ATTR_GRID)
    entities = list(getattr(grid, names.ATTR_ENTITIES))
    player_entity = getattr(world, names.ATTR_PLAYER)
    player = tuple(map(int, player_entity.qumspquyus))
    gems = [entity for entity in entities if entity.name == names.GEM]
    gem = tuple(map(int, gems[0].qumspquyus)) if len(gems) == 1 else None
    walls = frozenset(
        tuple(map(int, cell))
        for entity in entities
        if entity.name == names.WALL
        for cell in entity.uafphpbluk
    )
    hazards = frozenset(
        tuple(map(int, cell))
        for entity in entities
        if entity.name in (names.HAZARD_A, names.HAZARD_B)
        for cell in entity.uafphpbluk
    )
    entity_state = tuple(
        sorted(
            (
                str(entity.name),
                int(entity.grid_x),
                int(entity.grid_y),
                tuple(sorted(tuple(map(int, cell)) for cell in entity.hrlzbohbpn)),
                str(entity.flrpnczugo),
                bool(entity.collidable),
            )
            for entity in entities
        )
    )
    descriptor = env.generated_descriptor
    kind = descriptor.get("kind") if isinstance(descriptor, dict) else None
    action_count = env.action_count
    history_depth = _history_depth(world)
    move_count = int(getattr(world, names.ATTR_MOVE_COUNT))
    facing_right = bool(getattr(world, names.ATTR_FACING_RIGHT))
    entity_names = {entity.name for entity in entities}
    ordinary_names = {
        names.WALL,
        names.PLAYER_RIGHT,
        names.GEM,
        names.HAZARD_A,
        names.HAZARD_B,
    }
    hazard_gap = min((y for _, y in hazards), default=-1) - player[1]
    width, height = map(int, grid.grid_size)
    enclosed_play_region = (
        all((x, 0) in walls for x in range(width))
        and all((0, y) in walls and (width - 1, y) in walls for y in range(player[1] + 1))
    )
    # The live route finder emits only movement actions.  Starting with an odd
    # movement count can move the hazards on the next action; starting even
    # moves them on the second.  This bound is deliberately based on every
    # remaining native action, even though the returned route is usually much
    # shorter.  Falling only lowers the player and increases the separation.
    remaining_hazard_descents = (env.steps_left + (move_count & 1)) // 2
    generated_exact = (
        kind == names.GENERATED_KIND
        and env.state == GameState.NOT_FINISHED
        and bool(getattr(world, names.ATTR_GRAVITY_DOWN))
        and not bool(getattr(world, names.ATTR_WORLD_WIN))
        and not bool(getattr(world, names.ATTR_WORLD_LOSS))
        and len(gems) == 1
        and entity_names <= ordinary_names
        and names.HAZARD_A in entity_names
        and names.HAZARD_B in entity_names
        and enclosed_play_region
        # A hazard moves at most once per two movement actions.  Clicks can
        # change parity but also consume budget.  The returned witness never
        # uses undo, so the current parity, positions and remaining budget are
        # sufficient.  The extra two rows keep the band outside the route.
        and hazard_gap > remaining_hazard_descents + 2
    )
    generated_pristine = (
        generated_exact
        and action_count == 0
        and history_depth == 0
        and move_count == 0
        and facing_right
    )

    destructibles = frozenset(
        tuple(map(int, cell))
        for entity in entities
        if entity.name == names.DESTRUCTIBLE
        for cell in entity.uafphpbluk
    )
    official1_initial = (
        descriptor is None
        and env.level_index == 0
        and env.logical_level == 1
        and env.state == GameState.NOT_FINISHED
        and action_count == 0
        and history_depth == 0
        and player == (3, 23)
        and gem == (3, 7)
        and {(7, 19), (4, 16), (4, 15), (4, 12), (5, 9)} <= destructibles
    )

    unsupported = []
    if not (generated_exact or official1_initial):
        if kind == names.GENERATED_KIND:
            unsupported.append("generated map is outside the ordinary-platform safety proof")
        else:
            unsupported.append(
                "interactive click, spike, gravity-switch, growth and general undo layouts are not modelled"
            )
    return Layout(
        snapshot=env.clone(),
        player=player,
        gem=gem,
        width=width,
        height=height,
        walls=walls,
        hazards=hazards,
        entity_state=entity_state,
        generated_kind=kind,
        generated_exact=generated_exact,
        generated_pristine=generated_pristine,
        official1_initial=official1_initial,
        action_count=action_count,
        history_depth=history_depth,
        move_count=move_count,
        facing_right=facing_right,
        action_budget=env.steps_left,
        unsupported=tuple(unsupported),
    )
