"""Exact, readable snapshots of the complete shipped G50T ruleset.

The vendored game uses obfuscated wrapper names.  This module is the single
translation boundary: it turns those live wrappers into immutable cells,
circuits, doors, teleport links, enemies and replay routes.  The planner never
looks at an official level definition or at sprite names after extraction.
"""

from dataclasses import dataclass

from arcengine import GameState

from . import names


Cell = tuple[int, int]


@dataclass(frozen=True)
class Door:
    closed: Cell
    opened: Cell
    toggle: bool
    circuit: int


@dataclass(frozen=True)
class Teleport:
    pads: tuple[Cell, Cell]
    circuit: int


@dataclass(frozen=True)
class Enemy:
    start: Cell
    path: frozenset[Cell]


@dataclass(frozen=True)
class Ghost:
    position: Cell
    path: tuple[int, ...]


@dataclass(frozen=True)
class EnemyState:
    position: Cell
    orientation: int
    dead: bool
    history: tuple[int, ...]


@dataclass(frozen=True)
class World:
    player: Cell
    history: tuple[int, ...]
    ghosts: tuple[Ghost, ...]
    enemies: tuple[EnemyState, ...]
    doors: tuple[bool, ...]
    stage: int
    dead: bool = False


@dataclass(frozen=True)
class Layout:
    exact: bool
    unsupported: tuple[str, ...]
    level_index: int
    allowed: frozenset[Cell]
    start: Cell
    goal: Cell
    switches: tuple[tuple[Cell, int], ...]
    doors: tuple[Door, ...]
    teleports: tuple[Teleport, ...]
    enemies: tuple[Enemy, ...]
    stage_count: int
    world: World
    steps_used: int
    steps_left: int

    # Compatibility accessors retained for callers which inspected the old
    # tier-one layout directly.
    @property
    def player(self):
        return self.world.player

    @property
    def history(self):
        return tuple(names.ACTION_DELTA[action] for action in self.world.history)

    @property
    def ghosts(self):
        return self.world.ghosts

    @property
    def stage(self):
        return self.world.stage

    @property
    def switch(self):
        return self.switches[0][0] if self.switches else (0, 0)

    @property
    def door_closed(self):
        return self.doors[0].closed if self.doors else (0, 0)

    @property
    def door_open(self):
        return self.doors[0].opened if self.doors else (0, 0)

    @property
    def door_is_open(self):
        return self.world.doors[0] if self.world.doors else False


def _action_of(delta):
    delta = tuple(map(int, delta))
    for action, expected in names.ACTION_DELTA.items():
        if delta == expected:
            return action
    raise ValueError(f"non-cardinal native movement {delta!r}")


def _actor_key(controller, actor):
    if actor is getattr(controller, names.CTRL_PLAYER):
        return (0, 0)
    ghosts = list(getattr(controller, names.CTRL_GHOSTS))
    if actor in ghosts:
        return (1, ghosts.index(actor))
    enemies = list(getattr(controller, names.CTRL_ENEMIES))
    if actor in enemies:
        return (2, enemies.index(actor))
    return (9, id(actor))


def extract(env):
    module = env.module
    controller = env.controller
    reasons = []
    if env.state != GameState.NOT_FINISHED:
        reasons.append(f"terminal state {env.state.value}")
    if not env.stable():
        reasons.append("a native movement or rewind animation is pending")
    if env.available_actions != names.ACTION_IDS:
        reasons.append(f"unexpected native action set {env.available_actions}")

    level = env.level
    players = level.get_sprites_by_tag(names.TAG_PLAYER)
    goals = level.get_sprites_by_tag(names.TAG_GOAL)
    boundaries = level.get_sprites_by_tag(names.TAG_BOUNDARY)
    checkpoints = list(getattr(controller, names.CTRL_CHECKPOINTS))
    if not players:
        reasons.append("live-player sprite is missing")
    if len(goals) != 1:
        reasons.append(f"expected one goal, found {len(goals)}")
    if len(boundaries) != 1:
        reasons.append(f"expected one movement boundary, found {len(boundaries)}")
    if len(checkpoints) not in (2, 3):
        reasons.append(f"expected two or three timeline slots, found {len(checkpoints)}")
    if reasons and not all((players, goals, boundaries, checkpoints)):
        empty = World((0, 0), (), (), (), (), 0)
        return Layout(False, tuple(dict.fromkeys(reasons)), env.level_index,
                      frozenset(), (0, 0), (0, 0), (), (), (), (),
                      len(checkpoints), empty, env.steps_used, env.steps_left)

    player = getattr(controller, names.CTRL_PLAYER)
    boundary = getattr(controller, names.CTRL_BOUNDARY)
    start = (int(getattr(controller, names.CTRL_START_X)),
             int(getattr(controller, names.CTRL_START_Y)))
    goal_wrapper = getattr(controller, names.CTRL_GOAL)
    goal = (int(goal_wrapper.x) + 1, int(goal_wrapper.y) + 1)
    mod_x, mod_y = start[0] % names.GRID_STEP, start[1] % names.GRID_STEP
    allowed = set()
    for y in range(mod_y, names.FRAME_SIZE - player.height + 1, names.GRID_STEP):
        for x in range(mod_x, names.FRAME_SIZE - player.width + 1, names.GRID_STEP):
            if boundary.esidlbhbhw(x + player.width // 2, y + player.height // 2) >= 0:
                allowed.add((x, y))

    native_switches = [value for value in getattr(controller, names.CTRL_INPUTS)
                       if isinstance(value, getattr(module, names.CLASS_SWITCH))]
    native_pads = [value for value in getattr(controller, names.CTRL_INPUTS)
                   if isinstance(value, getattr(module, names.CLASS_TELEPORT_PAD))]
    native_doors = list(getattr(controller, names.CTRL_DOORS))
    circuits = []
    for switch in native_switches:
        circuit = getattr(switch, names.SWITCH_OUTPUT)
        if circuit is None or not isinstance(circuit, getattr(module, names.CLASS_CIRCUIT)):
            reasons.append("switch is not connected to a native circuit")
        elif circuit not in circuits:
            circuits.append(circuit)
    circuit_index = {id(value): index for index, value in enumerate(circuits)}

    switches = []
    for switch in native_switches:
        circuit = getattr(switch, names.SWITCH_OUTPUT)
        if circuit is not None and id(circuit) in circuit_index:
            switches.append(((int(switch.x), int(switch.y)), circuit_index[id(circuit)]))

    doors = []
    door_states = []
    for door in native_doors:
        if not isinstance(door, getattr(module, names.CLASS_DOOR)):
            reasons.append("unknown door wrapper type")
            continue
        owners = [index for index, circuit in enumerate(circuits)
                  if door in getattr(circuit, names.CIRCUIT_OUTPUTS)]
        if len(owners) != 1:
            reasons.append("every door must have exactly one circuit")
            continue
        direction = tuple(map(int, getattr(door, names.DOOR_DIRECTION)()))
        if direction not in names.ACTION_DELTA.values():
            reasons.append("door has a non-cardinal displacement")
            continue
        active = bool(getattr(door, names.DOOR_ACTIVE))
        current = (int(door.x), int(door.y))
        closed = ((current[0] - direction[0] * names.GRID_STEP,
                   current[1] - direction[1] * names.GRID_STEP) if active else current)
        opened = (closed[0] + direction[0] * names.GRID_STEP,
                  closed[1] + direction[1] * names.GRID_STEP)
        doors.append(Door(closed, opened, bool(getattr(door, names.DOOR_TOGGLE)), owners[0]))
        door_states.append(active)

    links = []
    for index, circuit in enumerate(circuits):
        for output in getattr(circuit, names.CIRCUIT_OUTPUTS):
            if not isinstance(output, getattr(module, names.CLASS_TELEPORT_LINK)):
                continue
            left = getattr(output, "jvabfgsorb")
            right = getattr(output, "xixykigflg")
            if left is None or right is None:
                reasons.append("teleport link does not contain exactly two pads")
                continue
            links.append(Teleport(((int(left.x), int(left.y)),
                                   (int(right.x), int(right.y))), index))
    if len(native_pads) != 2 * len(links):
        reasons.append("every teleport pad must belong to one paired link")

    enemies = []
    enemy_states = []
    native_enemies = list(getattr(controller, names.CTRL_ENEMIES).items())
    clean_level = env.game._clean_levels[env.level_index]
    clean_enemies = clean_level.get_sprites_by_tag(names.TAG_ENEMY)
    if len(clean_enemies) != len(native_enemies):
        reasons.append("live and immutable enemy counts disagree")
    for enemy_index, (enemy, history) in enumerate(native_enemies):
        guide = getattr(enemy, "baygqyisjz")
        if guide is None:
            reasons.append("enemy is missing its native guide path")
            path = frozenset()
        else:
            path = frozenset(cell for cell in allowed
                             if guide.qenvjwzlxy(*cell) and guide.esidlbhbhw(*cell) >= 0)
        action_history = tuple(_action_of(move) for move in history)
        # Read the immutable clean-level coordinate.  Subtracting movement
        # history from the live coordinate is wrong after a circuit-triggered
        # teleport because the native history records only cardinal steps.
        if enemy_index < len(clean_enemies):
            origin = (int(clean_enemies[enemy_index].x),
                      int(clean_enemies[enemy_index].y))
        else:
            origin = (int(enemy.x), int(enemy.y))
        enemies.append(Enemy(origin, path))
        enemy_states.append(EnemyState(
            (int(enemy.x), int(enemy.y)), int(getattr(enemy, "wzgvpxcawd")),
            bool(getattr(enemy, "pddqxjztas")),
            action_history,
        ))

    history = tuple(_action_of(move) for move in getattr(controller, names.CTRL_HISTORY))
    ghosts = []
    for actor, path in sorted(getattr(controller, names.CTRL_GHOSTS).items(),
                              key=lambda item: _actor_key(controller, item[0])):
        ghosts.append(Ghost((int(actor.x), int(actor.y)),
                            tuple(_action_of(move) for move in path)))
    stage = int(getattr(controller, names.CTRL_STAGE))
    if not 0 <= stage < len(checkpoints):
        reasons.append("timeline stage is outside the native checkpoint range")
    if len(ghosts) != stage:
        reasons.append("timeline stage and replay-ghost count disagree")
    if env.steps_left <= 0:
        reasons.append("native countdown budget is exhausted")

    all_cells = [start, goal, (int(player.x), int(player.y))]
    all_cells += [cell for cell, _ in switches]
    all_cells += [door.closed for door in doors]
    all_cells += [pad for link in links for pad in link.pads]
    all_cells += [enemy.start for enemy in enemies]
    if any((x - start[0]) % names.GRID_STEP or (y - start[1]) % names.GRID_STEP
           for x, y in all_cells):
        reasons.append("actors do not share the native six-pixel movement lattice")
    required = [start, goal, *[x[0] for x in switches], *[x.closed for x in doors],
                *[p for link in links for p in link.pads], *[x.start for x in enemies]]
    if any(cell not in allowed for cell in required):
        reasons.append("a required actor is outside the walkable boundary")

    world = World(
        (int(player.x), int(player.y)), history, tuple(ghosts), tuple(enemy_states),
        tuple(door_states), stage, bool(getattr(player, "pddqxjztas")),
    )
    return Layout(not reasons, tuple(dict.fromkeys(reasons)), env.level_index,
                  frozenset(allowed), start, goal, tuple(switches), tuple(doors),
                  tuple(links), tuple(enemies), len(checkpoints), world,
                  env.steps_used, env.steps_left)
