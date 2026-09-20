"""Compact symbolic view of a live S5I5 level.

`extract(env)` reads the current level out of the real game: every rod's
colour, base rotation, length and position, the pins and targets, and the
controls with one canonical click per distinct effect (rail extend, rail
retract, button rotate). The planner keys its search on `Layout.key()`, which
is the complete mutable state of a level between actions: rods and pins are
the only sprites the game ever moves, and it moves them by whole sprites.
"""

from dataclasses import dataclass, field

from . import names


@dataclass(frozen=True)
class Rod:
    index: int          # position in the level's sprite list
    name: str
    color: int
    x: int
    y: int
    width: int
    height: int
    rotation: int       # 0/90/180/270 as upstream reads it from the cap line
    length: int         # upstream "index": units of 3 pixels along the long axis
    controlled: bool    # some rail or button acts on this colour
    children: tuple     # names of sprites moving rigidly with this rod


@dataclass(frozen=True)
class Control:
    kind: str           # "extend" | "retract" | "rotate"
    name: str
    colors: tuple       # every rod colour dispatched by this native control
    click: tuple        # (x, y) display pixel that upstream resolves to this control

    @property
    def color(self):
        """Compatibility alias for single-colour callers."""
        return self.colors[0]


@dataclass(frozen=True)
class Layout:
    budget: int
    steps_left: int
    rods: tuple
    pins: tuple         # (x, y) per pin sprite, in sprite order
    targets: tuple      # (x, y) per target sprite
    controls: tuple     # Control, deterministic order
    env: object = field(compare=False, hash=False, repr=False, default=None)

    @property
    def actions(self):
        """Every distinct click worth trying, as (action_id, x, y)."""
        return tuple((names.ACTION_CLICK, c.click[0], c.click[1]) for c in self.controls)

    def key(self):
        return state_key(self.env)

    @property
    def won(self):
        pins = set(self.pins)
        return all(t in pins for t in self.targets)


def rod_rotation(game, sprite):
    return int(getattr(game, names.METHOD_ROTATION_OF)(sprite))


def rod_length(sprite):
    # upstream s5i5.py:2231-2234
    if sprite.height > sprite.width:
        return sprite.height // names.ROD_THICKNESS
    return sprite.width // names.ROD_THICKNESS


def _resolve(level, tag, sprite, ideal, predicate):
    """The pixel closest to `ideal` that upstream's get_sprite_at maps to `sprite`
    and that satisfies `predicate(offset_x, offset_y)`. None if no such pixel."""
    best = None
    for oy in range(sprite.height):
        for ox in range(sprite.width):
            if not predicate(ox, oy):
                continue
            x, y = sprite.x + ox, sprite.y + oy
            if not (0 <= x < names.FRAME and 0 <= y < names.FRAME):
                continue
            if level.get_sprite_at(x, y, tag) is not sprite:
                continue
            d = (x - ideal[0]) ** 2 + (y - ideal[1]) ** 2
            if best is None or d < best[0]:
                best = (d, (x, y))
    return None if best is None else best[1]


def rail_controls(level, rail):
    """Extend and retract clicks for one rail, mirroring s5i5.py:2219-2240."""
    horizontal = rail.width > rail.height
    half = (rail.width if horizontal else rail.height) // 2
    colors = tuple(sorted({int(v) for v in rail.pixels.flatten().tolist()
                           if v >= 0 and v not in names.RAIL_CHROME_COLORS}))
    color = colors[0] if colors else -1
    if horizontal:
        cy = rail.y + rail.height // 2
        ideal_ext = (rail.x + half + (rail.width - half) // 2, cy)
        ideal_ret = (rail.x + half // 2, cy)
        ext = _resolve(level, names.TAG_RAIL, rail, ideal_ext, lambda ox, oy: ox > half)
        ret = _resolve(level, names.TAG_RAIL, rail, ideal_ret, lambda ox, oy: ox < half)
    else:
        cx = rail.x + rail.width // 2
        ideal_ext = (cx, rail.y + half + (rail.height - half) // 2)
        ideal_ret = (cx, rail.y + half // 2)
        ext = _resolve(level, names.TAG_RAIL, rail, ideal_ext, lambda ox, oy: oy > half)
        ret = _resolve(level, names.TAG_RAIL, rail, ideal_ret, lambda ox, oy: oy < half)
    out = []
    if ext is not None:
        out.append(Control("extend", rail.name, colors, ext))
    if ret is not None:
        out.append(Control("retract", rail.name, colors, ret))
    return out, colors


def button_control(level, button):
    """The rotate click for one button, mirroring s5i5.py:2195-2197. The colour
    upstream reads is pixels[h//2, h//2]; the click must land on the sprite."""
    h = button.height
    color = int(button.pixels[h // 2, h // 2])
    ideal = (button.x + button.width // 2, button.y + h // 2)
    click = _resolve(level, names.TAG_BUTTON, button, ideal, lambda ox, oy: True)
    if click is None:
        return None, color
    return Control("rotate", button.name, (color,), click), color


def state_key(env):
    """The complete between-action state of the current level: geometry of
    every rod and pin. Everything else in the level is immutable."""
    level = env.game.current_level
    parts = []
    for s in level.get_sprites_by_tag(names.TAG_ROD):
        parts.append((s.x, s.y, s.pixels.shape, s.pixels.tobytes()))
    for p in level.get_sprites_by_tag(names.TAG_PIN):
        parts.append((p.x, p.y))
    return tuple(parts)


def extract(env):
    game = env.game
    level = game.current_level
    sprites = level.get_sprites()
    children_map = getattr(game, names.ATTR_CHILDREN)

    controls = []
    acted_colors = set()
    for rail in level.get_sprites_by_tag(names.TAG_RAIL):
        ctrls, colors = rail_controls(level, rail)
        controls.extend(ctrls)
        if ctrls:
            acted_colors.update(colors)
    for button in level.get_sprites_by_tag(names.TAG_BUTTON):
        ctrl, color = button_control(level, button)
        if ctrl is not None:
            controls.append(ctrl)
            acted_colors.add(color)

    rods = []
    for i, s in enumerate(sprites):
        if names.TAG_ROD not in s.tags:
            continue
        color = int(s.pixels[1, 1]) if s.width > 1 and s.height > 1 else -1
        kids = tuple(sorted(c.name for c in children_map.get(s, ())))
        rods.append(Rod(i, s.name, color, s.x, s.y, s.width, s.height, rod_rotation(game, s),
                        rod_length(s), color in acted_colors, kids))
    pins = tuple((p.x, p.y) for p in level.get_sprites_by_tag(names.TAG_PIN))
    targets = tuple((t.x, t.y) for t in level.get_sprites_by_tag(names.TAG_TARGET))
    return Layout(budget=env.max_steps(), steps_left=env.steps_left(), rods=tuple(rods),
                  pins=pins, targets=targets, controls=tuple(controls), env=env)
