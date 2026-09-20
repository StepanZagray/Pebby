"""Compact exact-state search for shipped DC22 layouts.

This mirrors the stable, settled effects in the vendored engine and is used to
obtain bounded official positive witnesses. Every returned route is replayed in
the real engine by :mod:`pebby.games.dc22.plan`; a symbolic mismatch is never
accepted as a certificate.
"""

from collections import deque
import heapq
from itertools import count

from . import names
from .env import upstream


_COLOR_PREFIXES = ("tewfutpibpar", "tewfutrefgps", "tewfutyefmyf", "tewfutblrmbx")
_DIRECTIONS = {
    "up": (0, 1),
    "dowlja": (0, -1),
    "lersnf": (-1, 0),
    "riidpd": (1, 0),
}


def _pixels(sprite):
    rendered = sprite.render()
    return frozenset(
        (x, y)
        for y in range(sprite.height)
        for x in range(sprite.width)
        if rendered[y, x] >= 0
    )


def _overlap(pixels, position, x, y):
    px, py = position
    return any((x + dx - px, y + dy - py) in pixels for dx in (0, 1) for dy in (0, 1))


class OfficialModel:
    """Finite settled state model for one official native level."""

    def __init__(self, env):
        self.env = env
        self.module = upstream()
        self.goal = (int(env.goal.x), int(env.goal.y))
        self.grid_size = tuple(env.level.grid_size)
        self.slots = []
        # Every raw toggle sprite is an independent slot. Level 6 deliberately
        # stacks multiple same-prefix phases at one coordinate, so grouping the
        # processed clones by (prefix, position) would collapse real state.
        clean_level = env.game._clean_levels[env.level_index]
        raw_toggles = clean_level.get_sprites_by_tag(names.TAG_TOGGLE)
        phases = []
        colors = []
        for sprite in raw_toggles:
            if not sprite.name[-1].isdigit():
                continue
            prefix = sprite.name[:-1]
            variants = []
            for phase in range(1, 9):
                if f"{prefix}{phase}" not in self.module.sprites:
                    break
                variants.append(phase)
            if not variants:
                continue
            active = int(sprite.name[-1])
            tags = set(sprite.tags)
            if env.level_index == 5 and int(sprite.x) == 18 and int(sprite.y) == 48 and names.TAG_BRIDGE in tags:
                tags.update((names.TAG_BRIDGE_COLOR_CYCLE, "d"))
            self.slots.append({
                "prefix": prefix,
                "position": (int(sprite.x), int(sprite.y)),
                "variants": tuple(variants),
                "tags": frozenset(tags),
                "special_color": names.TAG_BRIDGE_COLOR_CYCLE in tags,
            })
            phases.append(active)
            color_index = next((i for i, value in enumerate(_COLOR_PREFIXES) if prefix == value), -1)
            colors.append(color_index)

        self.keys = []
        for sprite in env.level.get_sprites_by_tag(names.TAG_GATE_KEY):
            color = next((tag for tag in sprite.tags if len(tag) == 1), None)
            self.keys.append({
                "color": color,
                "position": (int(sprite.x), int(sprite.y)),
                "pixels": _pixels(sprite),
            })

        self.sensors = []
        for sprite in env.level.get_sprites_by_tag(names.TAG_PRESSURE):
            color = next((tag for tag in sprite.tags if len(tag) == 1), None)
            self.sensors.append({
                "color": color,
                "position": (int(sprite.x), int(sprite.y)),
                "pixels": _pixels(sprite),
            })
        self.sensor_colors = {sensor["color"] for sensor in self.sensors}
        self.key_colors = {key["color"] for key in self.keys}

        self.static_support = set()
        self.static_blocked = set()
        dynamic_tags = {
            names.TAG_TOGGLE,
            names.TAG_PLAYER,
            names.TAG_GATE_KEY,
            names.TAG_CRUSHER,
            names.TAG_BRIDGE_OBJECT,
            names.TAG_FALL_BLOCKER,
        }
        for sprite in env.level.get_sprites():
            if any(tag in sprite.tags for tag in dynamic_tags) or "ignore" in sprite.tags:
                continue
            for dx, dy in _pixels(sprite):
                point = (int(sprite.x) + dx, int(sprite.y) + dy)
                if sprite.interaction.name == "INTANGIBLE":
                    self.static_support.add(point)
                if sprite.is_collidable:
                    self.static_blocked.add(point)

        crushers = env.level.get_sprites_by_tag(names.TAG_CRUSHER)
        self.crusher = crushers[0] if crushers else None
        self.crusher_base = (
            (int(self.crusher.x), int(self.crusher.y)) if self.crusher else (0, 0)
        )
        self.track_mode = bool(self.crusher and self.crusher.name.startswith("brixtocrzsjq"))
        self.track_pixels = set()
        for sprite in env.level.get_sprites_by_tag(names.TAG_FALL_BLOCKER):
            self.track_pixels.update(
                (int(sprite.x) + dx, int(sprite.y) + dy)
                for dx, dy in _pixels(sprite)
            )
        objects = env.level.get_sprites_by_tag(names.TAG_BRIDGE_OBJECT)
        self.object = objects[0] if objects else None
        self.object_start = (
            (int(self.object.x), int(self.object.y)) if self.object else (0, 0)
        )
        self.object_pixels = _pixels(self.object) if self.object else frozenset()

        self.controls = []
        scale, x_offset, y_offset = env.game.camera._calculate_scale_and_offset()
        for sprite in env.level.get_sprites_by_tag(names.TAG_CLICK):
            visible = _pixels(sprite)
            if not visible:
                continue
            center = ((sprite.width - 1) / 2, (sprite.height - 1) / 2)
            dx, dy = min(visible, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))
            color = next((tag for tag in sprite.tags if len(tag) == 1), None)
            operation = next((tag for tag in (*_DIRECTIONS, "grawwq") if tag in sprite.tags), None)
            self.controls.append({
                "name": sprite.name,
                "tags": frozenset(sprite.tags),
                "color": color,
                "operation": operation,
                "initial_visible": bool(sprite.is_visible),
                "action": (
                    names.ACTION_CLICK,
                    int(x_offset + (int(sprite.x) + dx) * scale),
                    int(y_offset + (int(sprite.y) + dy) * scale),
                ),
            })

        # (x, y, phases, key_mask, crusher_x, crusher_y, attached_slot, colors)
        self.start = (
            int(env.player.x),
            int(env.player.y),
            tuple(phases),
            (1 << len(self.keys)) - 1,
            int(getattr(env.game, "sjixewahg", 0)),
            int(getattr(env.game, "uxtzlxsiq", 0)),
            -1,
            tuple(colors),
        )

    def _slot_sprite(self, state, index):
        phase = state[2][index]
        slot = self.slots[index]
        prefix = slot["prefix"]
        color = state[7][index]
        if color >= 0:
            prefix = _COLOR_PREFIXES[color]
        return self.module.sprites[f"{prefix}{phase}"]

    def _slot_tags(self, state, index):
        return set(self._slot_sprite(state, index).tags) | set(self.slots[index]["tags"])

    def _slot_position(self, state, index):
        if state[6] == index:
            crusher_x = self.crusher_base[0] + state[4] * 4
            crusher_y = self.crusher_base[1] - state[5] * 4
            sprite = self._slot_sprite(state, index)
            return (
                crusher_x + self.crusher.width // 2 - sprite.width // 2,
                crusher_y + self.crusher.height // 2 - sprite.height // 2,
            )
        return self.slots[index]["position"]

    def _object_position(self, state):
        if state[6] == -2:
            crusher_x = self.crusher_base[0] + state[4] * 4
            crusher_y = self.crusher_base[1] - state[5] * 4
            offset = (9, 4) if not self.track_mode else (
                self.crusher.width // 2, self.crusher.height // 2,
            )
            return (
                crusher_x + offset[0] - self.object.width // 2,
                crusher_y + offset[1] - self.object.height // 2,
            )
        return self.object_start

    def _remaining_key_support(self, state, x, y):
        return any(
            state[3] & (1 << index)
            and (x - key["position"][0], y - key["position"][1]) in key["pixels"]
            for index, key in enumerate(self.keys)
        )

    def support(self, state, x, y):
        if (x, y) in self.static_support or self._remaining_key_support(state, x, y):
            return True
        if self.object is not None:
            ox, oy = self._object_position(state)
            if (x - ox, y - oy) in self.object_pixels:
                return True
        for index, slot in enumerate(self.slots):
            sprite = self._slot_sprite(state, index)
            tags = self._slot_tags(state, index)
            if "omvz" not in tags and "buezna" not in tags:
                continue
            sx, sy = self._slot_position(state, index)
            if (x - sx, y - sy) in _pixels(sprite):
                return True
        return False

    def blocked(self, state, x, y):
        player_pixels = {(x + dx, y + dy) for dx in (0, 1) for dy in (0, 1)}
        if player_pixels & self.static_blocked:
            return True
        for index in range(len(self.slots)):
            sprite = self._slot_sprite(state, index)
            tags = self._slot_tags(state, index)
            if "omvz" in tags or "inzejtible" in tags or "buezna" in tags:
                continue
            sx, sy = self._slot_position(state, index)
            occupied = {(sx + dx, sy + dy) for dx, dy in _pixels(sprite)}
            if player_pixels & occupied:
                return True
        if self.crusher is not None and not self.track_mode:
            prefix = self.crusher.name.rsplit("-", 1)[0]
            sprite = self.module.sprites[f"{prefix}-{'2' if state[6] != -1 else '1'}"]
            cx = self.crusher_base[0] + state[4] * 4
            cy = self.crusher_base[1] - state[5] * 4
            occupied = {(cx + dx, cy + dy) for dx, dy in _pixels(sprite)}
            if player_pixels & occupied:
                return True
        return False

    def _collect_key(self, state):
        x, y = state[:2]
        player_pixels = {(x + dx, y + dy) for dx in (0, 1) for dy in (0, 1)}
        for index, key in enumerate(self.keys):
            occupied = {
                (key["position"][0] + dx, key["position"][1] + dy)
                for dx, dy in key["pixels"]
            }
            if state[3] & (1 << index) and player_pixels & occupied:
                values = list(state)
                values[3] &= ~(1 << index)
                return tuple(values)
        return state

    def _sensor_active(self, state, color):
        x, y = state[:2]
        return any(
            sensor["color"] == color
            and _overlap(sensor["pixels"], sensor["position"], x, y)
            for sensor in self.sensors
        )

    def _key_unlocked(self, state, color):
        relevant = [i for i, key in enumerate(self.keys) if key["color"] == color]
        return bool(relevant) and all(not state[3] & (1 << index) for index in relevant)

    def _control_available(self, state, control):
        color = control["color"]
        if color in self.sensor_colors:
            return self._sensor_active(state, color)
        if color in self.key_colors and not control["initial_visible"]:
            return self._key_unlocked(state, color)
        return control["initial_visible"]

    def _cycle_color(self, state, color, special):
        values = list(state)
        phases, colors = list(state[2]), list(state[7])
        x, y = state[:2]
        bridges = []
        for index, slot in enumerate(self.slots):
            sprite = self._slot_sprite(state, index)
            tags = self._slot_tags(state, index)
            if color in tags and names.TAG_BRIDGE in tags and not special:
                bridges.append((index, sprite, self._slot_position(state, index)))
        own = next(((index, sprite) for index, sprite, pos in bridges if pos == (x, y)), None)
        if own:
            own_index, own_sprite = own
            slot = self.slots[own_index]
            variants = slot["variants"]
            next_phase = variants[(variants.index(state[2][own_index]) + 1) % len(variants)]
            own_prefix = own_sprite.name[:-1]
            next_name = f"{own_prefix}{next_phase}"
            target = next((pos for index, sprite, pos in bridges if index != own_index and sprite.name == next_name), None)
            if target is None:
                target = next((pos for index, sprite, pos in bridges if index != own_index and sprite.name == own_sprite.name), None)
            if target is not None:
                values[0], values[1] = target
        for index, slot in enumerate(self.slots):
            sprite = self._slot_sprite(state, index)
            if color not in self._slot_tags(state, index):
                continue
            if special and slot["special_color"] and colors[index] >= 0:
                colors[index] = (colors[index] + 1) % len(_COLOR_PREFIXES)
                continue
            if special and slot["special_color"]:
                continue
            variants = slot["variants"]
            phases[index] = variants[(variants.index(phases[index]) + 1) % len(variants)]
        values[2], values[7] = tuple(phases), tuple(colors)
        result = tuple(values)
        return result if self.support(result, result[0], result[1]) else None

    def _crusher_action(self, state, operation):
        if self.crusher is None:
            return state
        values = list(state)
        ix, iy = state[4], state[5]
        if operation == "grawwq":
            if state[6] != -1:
                return state
            if self.object is not None:
                ox, oy = self._object_position(state)
                center = (ox + self.object.width // 2, oy + self.object.height // 2)
                anchor = (
                    self.crusher_base[0] + ix * 4 + (self.crusher.width // 2 if self.track_mode else 9),
                    self.crusher_base[1] - iy * 4 + (self.crusher.height // 2 if self.track_mode else 4),
                )
                if center == anchor:
                    values[6] = -2
                    return tuple(values)
            if self.track_mode:
                anchor = (self.crusher_base[0] + ix * 4 + self.crusher.width // 2,
                          self.crusher_base[1] - iy * 4 + self.crusher.height // 2)
                for index, slot in enumerate(self.slots):
                    sprite = self._slot_sprite(state, index)
                    if not sprite.name.startswith("brixto") or len(slot["variants"]) != 2:
                        continue
                    sx, sy = self._slot_position(state, index)
                    if (sx + sprite.width // 2, sy + sprite.height // 2) == anchor:
                        values[6] = index
                        return tuple(values)
            return state

        dx, dy = _DIRECTIONS[operation]
        nx, ny = ix + dx, iy + dy
        valid = False
        if self.track_mode:
            cx = self.crusher_base[0] + nx * 4 + self.crusher.width // 2
            cy = self.crusher_base[1] - ny * 4 + self.crusher.height // 2
            valid = (cx, cy) in self.track_pixels
        elif operation == "up":
            valid = ix == 0 and iy < 3
        elif operation == "dowlja":
            valid = ix == 0 and iy > 0
        elif operation == "lersnf":
            valid = iy in (0, 3) and ix > 0
        elif operation == "riidpd":
            valid = iy in (0, 3) and ix < 3
        if not valid:
            return state
        values[4], values[5] = nx, ny
        result = tuple(values)
        return result if self.support(result, result[0], result[1]) else None

    def successors(self, state):
        x, y = state[:2]
        for action, (dx, dy) in names.MOVE_DELTAS.items():
            nx, ny = x + dx, y + dy
            if not self.blocked(state, nx, ny) and self.support(state, nx, ny):
                values = list(state)
                values[0], values[1] = nx, ny
                yield self._collect_key(tuple(values)), (action, None, None)
        for control in self.controls:
            if not self._control_available(state, control):
                continue
            result = state
            if names.TAG_BUTTON in control["tags"] and control["color"] is not None:
                result = self._cycle_color(
                    result,
                    control["color"],
                    names.TAG_BRIDGE_COLOR_BUTTON in control["tags"],
                )
            if result is not None and control["operation"] is not None:
                result = self._crusher_action(result, control["operation"])
            if result is not None and result != state:
                yield result, control["action"]


def official_search(env, node_limit):
    """Return ``(actions, expanded, generated, truncated)`` from a pristine level."""
    model = OfficialModel(env)
    weighted = env.level_index == 5
    serial = count()
    if weighted:
        def priority(state, cost):
            distance = abs(state[0] - model.goal[0]) + abs(state[1] - model.goal[1])
            return cost + 5 * (distance // 2)

        queue = [(priority(model.start, 0), 0, next(serial), model.start)]
        best = {model.start: 0}
    else:
        queue = deque([model.start])
    parent = {model.start: None}
    expanded = generated = 0
    goal = None
    while queue:
        if expanded >= node_limit:
            return None, expanded, generated, True
        if weighted:
            _, cost, _, state = heapq.heappop(queue)
            if best.get(state) != cost:
                continue
        else:
            state = queue.popleft()
            cost = 0
        expanded += 1
        if state[:2] == model.goal:
            goal = state
            break
        for successor, action in model.successors(state):
            generated += 1
            if weighted:
                next_cost = cost + 1
                if next_cost >= best.get(successor, 1 << 30):
                    continue
                best[successor] = next_cost
                parent[successor] = (state, action)
                heapq.heappush(
                    queue,
                    (priority(successor, next_cost), next_cost, next(serial), successor),
                )
            elif successor not in parent:
                parent[successor] = (state, action)
                queue.append(successor)
    if goal is None:
        return None, expanded, generated, False
    actions = []
    while parent[goal] is not None:
        goal, action = parent[goal]
        actions.append(action)
    actions.reverse()
    return tuple(actions), expanded, generated, False
