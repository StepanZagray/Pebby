"""Read a level's logical structure out of a running SC25 game.

The planner reasons over a handful of integers -- player pixel position, scale,
facing, teleport cursors, which removable sprites are gone, toggles spent -- not
over sprites. This module is the only place that knows how upstream's sprite list
maps onto that state, and it works identically for shipped and generated levels.

Collision is reproduced exactly: the player is an opaque square, so a move
collides iff any opaque pixel of a collidable sprite lies under it
(`Sprite.collides_with`, PIXEL_PERFECT). Fireball and growth checks use upstream's
own, slightly different, bounding-box / raw-pixel tests (sc25.py:1974-2045 and
2497-2537), so those get separate masks.
"""

import numpy as np

from . import names

PAD = 16          # sprites may sit partly outside the 64x64 frame (duvwsv-4 is at (-2,-5))
SIZE = 64 + 2 * PAD

KIND_TARGET, KIND_TARGET_ALT, KIND_BLOCK, KIND_BLOCK_ALT, KIND_PICKUP = range(5)
REMOVABLE_KINDS = {
    names.SPRITE_TARGET: KIND_TARGET, names.SPRITE_TARGET_ALT: KIND_TARGET_ALT,
    names.SPRITE_BLOCK: KIND_BLOCK, names.SPRITE_BLOCK_ALT: KIND_BLOCK_ALT,
    names.SPRITE_PICKUP: KIND_PICKUP,
}
FIRE_BBOX_NAMES = {names.SPRITE_BLOCK, names.SPRITE_BLOCK_ALT, names.SPRITE_RING_OBSTACLE,
                   names.SPRITE_TARGET, names.SPRITE_TARGET_ALT}
GROW_BLOCKER_NAMES = {names.SPRITE_RING_OBSTACLE, names.SPRITE_BLOCK, names.SPRITE_BLOCK_ALT}


class Unplannable(Exception):
    """The game is in a state this module does not model (e.g. mid-animation)."""


def _blank():
    return np.zeros((SIZE, SIZE), dtype=bool)


def _paint(mask, x, y, pixels):
    """OR the opaque pixels of `pixels` placed at (x, y) into the padded mask."""
    h, w = pixels.shape
    x0, y0 = x + PAD, y + PAD
    xs, ys = max(0, -x0), max(0, -y0)
    xe, ye = min(w, SIZE - x0), min(h, SIZE - y0)
    if xe <= xs or ye <= ys:
        return
    mask[y0 + ys:y0 + ye, x0 + xs:x0 + xe] |= pixels[ys:ye, xs:xe] != -1


def _window(mask, size):
    """window[y, x] is True iff the size x size square at padded (x, y) touches a True pixel."""
    out = np.zeros_like(mask)
    for dy in range(size):
        for dx in range(size):
            out[:SIZE - dy, :SIZE - dx] |= mask[dy:, dx:]
    return out


class Removable:
    __slots__ = ("bit", "kind", "name", "x", "y", "width", "height", "raw_w", "raw_h",
                 "move_block", "grow_block", "collidable")

    def __init__(self, bit, kind, sprite):
        self.bit, self.kind, self.name = bit, kind, sprite.name
        self.x, self.y = sprite.x, sprite.y
        rendered = sprite.render()
        self.height, self.width = rendered.shape
        self.raw_h, self.raw_w = sprite.pixels.shape
        self.collidable = sprite.is_collidable
        mask = _blank()
        if self.collidable:
            _paint(mask, sprite.x, sprite.y, rendered)
        self.move_block = {1: _window(mask, 2), 2: _window(mask, 4)}
        raw = _blank()
        if self.collidable and sprite.name in GROW_BLOCKER_NAMES:
            _paint(raw, sprite.x, sprite.y, np.asarray(sprite.pixels))
        self.grow_block = _window(raw, 4)


class Layout:
    """Static description of one level plus the state the player starts in.

    Coordinates are frame pixels. `start` is the planner's state tuple:
    (x, y, scale, facing, pad_index, small_pad_index, removed_bits, used,
    demo, grid_bits). ``grid_bits`` is row-major over the 3x3 spell grid.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    # -- queries used by the planner ------------------------------------------

    def present(self, removed, bit):
        return not (removed >> bit) & 1

    def blocked(self, x, y, scale, removed):
        """(collides, touches_door) for the player square at frame (x, y)."""
        px, py = x + PAD, y + PAD
        if not (0 <= px < SIZE and 0 <= py < SIZE):
            return False, False
        hit = bool(self.static_block[scale][py, px])
        door = bool(self.door_block[scale][py, px])
        if not hit:
            for item in self.removables:
                if item.collidable and self.present(removed, item.bit) and item.move_block[scale][py, px]:
                    hit = True
                    break
        return hit or door, door

    def grow_blocked(self, x, y, removed):
        """Upstream `qbdokwllfg` for a 4x4 square at (x, y): out of frame or on a blocker pixel."""
        if x < 0 or y < 0 or x + 4 > 64 or y + 4 > 64:
            return True
        px, py = x + PAD, y + PAD
        if self.grow_static_block[py, px]:
            return True
        return any(self.present(removed, item.bit) and item.grow_block[py, px]
                   for item in self.removables if item.kind in (KIND_BLOCK, KIND_BLOCK_ALT))

    def pickup_at(self, x, y, w, h, removed):
        """First present pickup (in upstream list order) overlapping the box, or None."""
        for item in self.pickups:
            if self.present(removed, item.bit) and x < item.x + item.raw_w and x + w > item.x \
                    and y < item.y + item.raw_h and y + h > item.y:
                return item
        return None

    def fire_hit(self, x, y, scale, facing, removed):
        """The sprite a fireball cast now would hit (upstream `okbritiujy`), or None."""
        fdx, fdy = names.FACING_DELTAS[facing]
        size = 2 * scale
        sx = x + (size - 1 if fdx > 0 else 0)
        sy = y + (size - 1 if fdy > 0 else 0)
        for i in range(1, 64):
            px, py = sx + fdx * i, sy + fdy * i
            if not (0 <= px < 64 and 0 <= py < 64):
                return None
            for entry in self.fire_order:
                kind, item = entry
                if kind == "bbox":
                    if item.bit is not None and not self.present(removed, item.bit):
                        continue
                    if item.x <= px < item.x + item.width and item.y <= py < item.y + item.height:
                        return item
                else:  # a wall bitmap, raw pixels
                    if self.wall_raw[py + PAD, px + PAD]:
                        return WALL
        return None

    def describe(self):
        return (f"player ({self.start[0]},{self.start[1]}) scale {self.start[2]} | spells {self.spells} | "
                f"budget {self.budget} | removables {len(self.removables)} | pads {len(self.pads)}"
                f"+{len(self.small_pads)} | door {self.door_present} | demo {self.start[8]}")


class _Static:
    """Stand-in for a permanent fireball blocker (crzdcq) or the wall bitmap."""
    __slots__ = ("bit", "kind", "name", "x", "y", "width", "height")

    def __init__(self, sprite):
        self.bit, self.kind, self.name = None, None, sprite.name
        self.x, self.y = sprite.x, sprite.y
        self.height, self.width = sprite.render().shape


WALL = _Static.__new__(_Static)
WALL.bit, WALL.kind, WALL.name, WALL.x, WALL.y, WALL.width, WALL.height = None, None, "wall", 0, 0, 0, 0


def extract(env):
    """Build a Layout from the current level of a live `Env`. Refuses mid-animation states."""
    game = env.game
    for attr in (names.ATTR_CAST_ANIM, names.ATTR_TELEPORT_ANIM, names.ATTR_GROW_ANIM,
                 names.ATTR_FIRE_ANIM, names.ATTR_FLASH_ANIM, names.ATTR_DEMO_ANIM):
        if getattr(game, attr).get("acyylh"):
            raise Unplannable(f"animation {attr} in progress")
    if getattr(game, names.ATTR_WALKING_OUT):
        raise Unplannable("door walk-out in progress")
    grid = getattr(game, names.ATTR_GRID)
    grid_bits = sum(
        1 << (row * 3 + col)
        for row in range(3)
        for col in range(3)
        if grid[row][col]
    )
    player = getattr(game, names.ATTR_PLAYER)
    if player is None:
        raise Unplannable("level has no player")
    if player.scale not in (1, 2):
        raise Unplannable(f"player scale {player.scale} is not 1 or 2")

    level_sprites = list(getattr(game, names.ATTR_LEVEL_SPRITES))
    current = game.current_level.get_sprites()

    static = _blank()
    door = _blank()
    wall_raw = _blank()
    grow_static = _blank()
    removables, pickups, fire_order = [], [], []
    door_present = False
    for sprite in level_sprites:
        name = sprite.name
        if sprite is player:
            continue
        if name == names.SPRITE_DOOR:
            door_present = True
            if sprite.is_collidable:
                _paint(door, sprite.x, sprite.y, sprite.render())
            continue
        kind = REMOVABLE_KINDS.get(name)
        if kind is not None:
            item = Removable(len(removables), kind, sprite)
            removables.append(item)
            if kind == KIND_PICKUP:
                pickups.append(item)
            else:
                fire_order.append(("bbox", item))
            continue
        if sprite.is_collidable:
            _paint(static, sprite.x, sprite.y, sprite.render())
        if name.startswith(names.SPRITE_WALL_PREFIX):
            _paint(wall_raw, sprite.x, sprite.y, np.asarray(sprite.pixels))
            fire_order.append(("wall", None))
            if sprite.is_collidable:
                _paint(grow_static, sprite.x, sprite.y, np.asarray(sprite.pixels))
        elif name == names.SPRITE_RING_OBSTACLE:
            fire_order.append(("bbox", _Static(sprite)))
            if sprite.is_collidable:
                _paint(grow_static, sprite.x, sprite.y, np.asarray(sprite.pixels))
    # Sprites added after on_set_level (only animation sprites, which we refused above)
    # would be missed; make sure nothing collidable is unaccounted for.
    extra = [s for s in current if s not in level_sprites and s is not player and s.is_collidable]
    if extra:
        raise Unplannable(f"unexpected live sprites: {[s.name for s in extra]}")
    # Upstream re-tests wall bitmaps once per wall sprite. A single merged mask at
    # the first wall's position in the scan order is equivalent: walls are never
    # removed and only the first object at each ray point matters. Some generated
    # or diagnostic levels have no wall, so do not assume that one exists.
    compact_order = []
    wall_seen = False
    for entry in fire_order:
        if entry[0] == "wall":
            if wall_seen:
                continue
            wall_seen = True
        compact_order.append(entry)
    fire_order = compact_order

    pads = [(s.x, s.y) for s in getattr(game, names.ATTR_TELEPORT_PADS)]
    small_pads = [(s.x, s.y) for s in getattr(game, names.ATTR_TELEPORT_PADS_SMALL)]
    raw_spells = list(getattr(game, names.ATTR_SPELLS))
    spells = [s for s in raw_spells if s in names.PATTERNS]
    selected = getattr(game, names.ATTR_SELECTED)
    demo = bool(getattr(game, names.ATTR_DEMO_PENDING)) and selected in names.PATTERNS

    icon_clicks = {}
    for sprite in current:
        if sprite.name.startswith(names.SPRITE_ICON_PREFIX) and sprite.name not in (
                names.SPRITE_ICON_FRAME, names.SPRITE_GRID_PANEL, names.SPRITE_GRID_CELL):
            spell = sprite.name[len(names.SPRITE_ICON_PREFIX):]
            rendered = sprite.render()
            ys, xs = np.nonzero(rendered != -1)
            if len(xs):
                icon_clicks.setdefault(spell, (sprite.x + int(xs[0]), sprite.y + int(ys[0])))

    start = (player.x, player.y, player.scale, getattr(game, names.ATTR_FACING),
             getattr(game, names.ATTR_TELEPORT_INDEX), getattr(game, names.ATTR_TELEPORT_INDEX_SMALL),
             0, getattr(game, names.ATTR_USED), demo, grid_bits)
    budget = getattr(game, names.ATTR_BUDGET)
    return Layout(
        static_block={1: _window(static, 2), 2: _window(static, 4)},
        door_block={1: _window(door, 2), 2: _window(door, 4)},
        door_present=door_present,
        wall_raw=wall_raw,
        grow_static_block=_window(grow_static, 4),
        removables=removables, pickups=pickups, fire_order=fire_order,
        pads=pads, small_pads=small_pads, spells=spells, raw_spells=raw_spells, selected=selected,
        icon_clicks=icon_clicks, budget=budget, start=start,
        level_index=env.level_index,
    )
