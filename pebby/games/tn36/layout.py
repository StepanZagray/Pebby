"""Extract the complete settled-action TN36 planning state.

TN36 is a visual-programming game. Clicks edit a bit-encoded program and a
run click executes the whole program as one native action. Later levels add
collision rollback, scale checkpoints, and phase-changing gates.
"""

from dataclasses import dataclass

from . import names


Transform = tuple[int, int, int, int, int]
Point = tuple[int, int]


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Gate:
    body: Rect
    barrier: Rect
    visible: bool


@dataclass(frozen=True)
class Preset:
    click: Point
    program: tuple[int, ...]
    position: Point
    rotation: int
    scale: int
    reset: bool


@dataclass(frozen=True)
class Layout:
    initial: Transform
    current: Transform
    target: Transform
    actor_alive: bool
    program: tuple[int, ...]
    bit_clicks: tuple[tuple[Point, ...], ...]
    run_click: Point | None
    clicks_left: int
    level_index: int
    reset_after_run: bool
    walls: tuple[Rect, ...]
    platforms: tuple[Rect, ...]
    gates: tuple[Gate, ...]
    presets: tuple[Preset, ...]
    exact: bool
    unsupported: tuple[str, ...]

    @property
    def bit_widths(self):
        return tuple(len(clicks) for clicks in self.bit_clicks)


def _transform(actor):
    return (
        int(actor.x), int(actor.y), int(actor.rotation) % 360,
        int(actor.scale), int(actor.sjmtdfxdrc),
    )


def _center(sprite):
    return (int(sprite.x + sprite.width // 2), int(sprite.y + sprite.height // 2))


def _rect(sprite):
    return Rect(int(sprite.x), int(sprite.y), int(sprite.width), int(sprite.height))


def extract(env):
    """Describe a settled live level, naming rather than guessing omissions."""
    controller = env.controller
    panel = getattr(controller, names.ATTR_GOAL_PANEL)
    actor = getattr(panel, names.ATTR_ACTOR)
    target = getattr(panel, names.ATTR_TARGET)
    program = getattr(panel, names.ATTR_PROGRAM)
    run = getattr(panel, names.ATTR_RUN)
    reasons = []

    if getattr(controller, names.ATTR_ACTIVE):
        reasons.append("an instruction sequence is mid-execution")
    if bool(getattr(program, names.ATTR_PROGRAM_LOCKED)):
        reasons.append("the goal-side program is locked")
    if target is None:
        reasons.append("the goal panel has no target")
    if run is None:
        reasons.append("the goal panel has no run control")

    groups = tuple(getattr(program, names.ATTR_PROGRAM_GROUPS))
    bit_clicks = []
    values = []
    if not groups:
        reasons.append("the editable program has no instruction cells")
    for index, group in enumerate(groups):
        bits = tuple(getattr(group, names.ATTR_BITS))
        if not 1 <= len(bits) <= 6:
            reasons.append(f"instruction {index} has unsupported bit width {len(bits)}")
        bit_clicks.append(tuple(_center(bit) for bit in bits))
        values.append(sum(
            1 << bit_index
            for bit_index, bit in enumerate(bits)
            if bool(getattr(bit, names.ATTR_BIT_ON))
        ))

    gates = tuple(
        Gate(_rect(value.olbuwgbgyz), _rect(value.axbjgpzkyi), bool(value.is_visible))
        for value in getattr(panel, names.ATTR_GATES)
    )
    selectors = tuple(getattr(controller, names.ATTR_SELECTORS))
    programs = env.level.get_data("Programs") or []
    positions = env.level.get_data("Positions") or []
    rotations = env.level.get_data("Rotations") or []
    scales = env.level.get_data("scvkkws") or []
    resets = env.level.get_data("Reset") or []
    presets = []
    if selectors:
        lengths = {len(selectors), len(programs), len(positions), len(rotations), len(scales)}
        if len(lengths) != 1:
            reasons.append("preset selectors and preset metadata have different lengths")
        else:
            for index, selector in enumerate(selectors):
                presets.append(Preset(
                    _center(selector), tuple(int(value) for value in programs[index]),
                    tuple(int(value) for value in positions[index]),
                    int(rotations[index]) % 360, int(scales[index]),
                    bool(resets[index]) if index < len(resets) else True,
                ))

    initial = (
        int(getattr(panel, names.ATTR_INITIAL_X)),
        int(getattr(panel, names.ATTR_INITIAL_Y)),
        int(getattr(panel, names.ATTR_INITIAL_ROTATION)) % 360,
        int(getattr(panel, names.ATTR_INITIAL_SCALE)),
        int(getattr(panel, names.ATTR_INITIAL_COLOR)),
    )
    target_state = initial if target is None else _transform(target)
    return Layout(
        initial=initial,
        current=_transform(actor),
        target=target_state,
        actor_alive=bool(actor.brvmvgfchj),
        program=tuple(values),
        bit_clicks=tuple(bit_clicks),
        run_click=None if run is None else _center(run),
        clicks_left=int(env.clicks_left),
        level_index=int(env.level_index),
        reset_after_run=bool(program.kviwnrvuri),
        walls=tuple(_rect(value.axbjgpzkyi) for value in getattr(panel, names.ATTR_WALLS)),
        platforms=tuple(_rect(value.axbjgpzkyi) for value in getattr(panel, names.ATTR_PLATFORMS)),
        gates=gates,
        presets=tuple(presets),
        exact=not reasons,
        unsupported=tuple(reasons),
    )
