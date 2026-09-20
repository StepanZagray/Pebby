"""A bounded S5I5 planner with native-replayed positive certificates.

Early contexts use breadth-first search over a compact transition model; late
contexts use bounded best-first guidance. Every positive route is replayed on
a fresh vendored engine clone before it is returned. ``exact`` therefore means
an exact native-positive transition certificate, while ``optimal`` separately
records whether breadth-first layers establish shortestness. A truncated
negative result makes no impossibility claim.
"""

from collections import deque
import heapq
import itertools

import numpy as np

from arcengine import ActionInput, GameAction, GameState

from . import names
from .layout import Layout, extract

DEFAULT_MAX_NODES = 60_000


class Solution(list):
    """The shortest click sequence found, as (action_id, x, y) triples."""

    truncated = False
    nodes = 0


class SearchResult:
    __slots__ = ("actions", "truncated", "nodes", "depth_reached", "exact", "optimal", "reason")

    def __init__(self, actions, truncated, nodes, depth_reached, exact=None,
                 optimal=None, reason=""):
        self.actions = actions
        self.truncated = truncated
        self.nodes = nodes
        self.depth_reached = depth_reached
        self.exact = exact
        self.optimal = optimal
        self.reason = reason

    @property
    def explored(self):
        return self.nodes


class _Engine:
    """Applies clicks to one private game copy and snapshots/restores its state
    without cloning the game per node."""

    def __init__(self, env):
        self.env = env.clone()
        self.game = self.env.game
        level = self.game.current_level
        self.rods = level.get_sprites_by_tag(names.TAG_ROD)
        self.pins = level.get_sprites_by_tag(names.TAG_PIN)
        self.hud = getattr(self.game, names.ATTR_STEP_HUD)
        self.score0 = self.game._score
        self.level0 = self.game.level_index
        self._pixel_tokens = {}
        self._pixel_arrays = []

    def _intern_pixels(self, pixels):
        identity = (pixels.shape, pixels.tobytes())
        token = self._pixel_tokens.get(identity)
        if token is None:
            token = len(self._pixel_arrays)
            self._pixel_tokens[identity] = token
            self._pixel_arrays.append(pixels.copy())
        return token, self._pixel_arrays[token]

    def snapshot(self):
        rods = []
        rod_key = []
        for sprite in self.rods:
            token, pixels = self._intern_pixels(sprite.pixels)
            rods.append((sprite._x, sprite._y, pixels))
            rod_key.append((sprite._x, sprite._y, token))
        rods = tuple(rods)
        pins = tuple((p._x, p._y) for p in self.pins)
        key = (tuple(rod_key), pins)
        return key, (rods, pins)

    def restore(self, data):
        rods, pins = data
        for s, (x, y, a) in zip(self.rods, rods):
            s._x, s._y = x, y
            s.pixels = a
        for p, (x, y) in zip(self.pins, pins):
            p._x, p._y = x, y

    def click(self, x, y):
        """One click, exactly as perform_action would run it minus rendering.
        Returns True if the level was completed by this click."""
        game = self.game
        assert game.level_index == self.level0
        game._state = GameState.NOT_FINISHED
        setattr(self.hud, names.HUD_CURRENT_STEPS, 1 << 30)   # budget is the search depth
        setattr(game, names.ATTR_BACKUP, dict())
        game._action = ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        game._action_complete = False
        while not game.is_action_complete():
            if game._next_level:
                game._really_set_next_level()
            else:
                game.step()
        return game._score > self.score0


def _layout_and_env(env_or_layout):
    if isinstance(env_or_layout, Layout):
        if env_or_layout.env is None:
            raise ValueError("this Layout was extracted without a live env")
        return env_or_layout, env_or_layout.env
    return extract(env_or_layout), env_or_layout


def search(env_or_layout, limit=None, max_nodes=DEFAULT_MAX_NODES):
    """Exhaustive BFS up to ``limit`` clicks.

    With no explicit limit the game's remaining click budget is searched in
    full.  Reaching that budget is therefore an exhaustive negative result,
    not truncation.  A smaller caller-supplied limit *is* truncation when a
    frontier remains, as is exhausting ``max_nodes``.
    """
    layout, env = _layout_and_env(env_or_layout)
    if env.state in (GameState.WIN, GameState.GAME_OVER):
        return SearchResult(None, False, 0, 0, True, None, "terminal state")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative")

    # S5I5 evaluates its win predicate after consuming a click.  A freshly
    # loaded level whose pins already match its targets therefore needs one
    # harmless click to complete.  The upstream step also checks victory
    # before checking the exhausted counter, so this remains true for a level
    # constructed with a zero StepCounter.
    natural_limit = max(layout.steps_left, 1 if layout.won else 0)
    explicit_cutoff = limit is not None and limit < natural_limit
    limit = natural_limit if limit is None else min(limit, natural_limit)
    actions = layout.actions
    if layout.won:
        # A control can move an already aligned pin off its target. Include a
        # neutral border click first, and let the real engine check completion.
        actions = ((names.ACTION_CLICK, 0, 0),) + actions
    if limit <= 0 or not actions:
        truncated = explicit_cutoff and bool(actions)
        return SearchResult(None, truncated, 0, 0, not truncated, None,
                            "empty bounded frontier" if truncated else "exhausted")

    # The last two official contexts have much larger rotation/attachment
    # products. A bounded best-first pass supplies a replayable positive
    # witness without pretending it is shortest. Earlier contexts use compact
    # breadth-first search with the same transition model and native replay.
    if not layout.won:
        if env.level_index >= 6:
            return _search_fast_witness(layout, env, limit, max_nodes)
        return _search_fast_bfs(layout, env, limit, max_nodes, explicit_cutoff)

    engine = _Engine(env)
    start_key, start_data = engine.snapshot()
    parents = {start_key: None}
    store = {start_key: start_data}
    queue = deque([(start_key, 0)])
    truncated = False
    nodes = 0
    deepest = 0
    while queue:
        key, depth = queue.popleft()
        if depth >= limit:
            # A frontier at the real budget cannot lead to a legal solution;
            # only a caller-imposed shallower cutoff leaves relevant work.
            truncated = truncated or explicit_cutoff
            continue
        if nodes >= max_nodes:
            truncated = True
            break
        nodes += 1
        data = store[key]
        for action in actions:
            engine.restore(data)
            won = engine.click(action[1], action[2])
            if won:
                path = [action]
                k = key
                while parents[k] is not None:
                    k, a = parents[k]
                    path.append(a)
                path.reverse()
                sol = Solution(path)
                sol.nodes = nodes
                return SearchResult(sol, False, nodes, depth + 1, True, True,
                                    "shortest path by exhaustive BFS layers")
            nkey, ndata = engine.snapshot()
            if nkey in parents:
                continue
            parents[nkey] = (key, action)
            store[nkey] = ndata
            queue.append((nkey, depth + 1))
            deepest = max(deepest, depth + 1)
    return SearchResult(None, truncated, nodes, deepest, not truncated,
                        not truncated,
                        "search work exhausted" if truncated else "state space exhausted")


def _pin_distance(pins, targets):
    """Small deterministic guidance score; never presented as an admissible bound."""
    if len(pins) < len(targets):
        return 1 << 20
    best = 1 << 30
    for chosen in itertools.permutations(pins, len(targets)):
        distance = sum(abs(px - tx) + abs(py - ty)
                       for (px, py), (tx, ty) in zip(chosen, targets))
        best = min(best, distance)
    return best // names.ROD_THICKNESS


class _FastEngine:
    """Compact exact bar/tree transitions for bounded late-tier guidance.

    S5I5's mechanically moving sprites are solid three-pixel bars. Static
    obstacle masks retain their exact opaque pixels. Positive routes from this
    model are always replayed from the live native state before publication.
    """

    _ROTATED = {0: 270, 270: 180, 180: 90, 90: 0}

    def __init__(self, env, layout):
        game = env.game
        all_rods = list(game.current_level.get_sprites_by_tag(names.TAG_ROD))
        all_pins = list(game.current_level.get_sprites_by_tag(names.TAG_PIN))
        acted_colors = {color for control in layout.controls for color in control.colors}
        controlled = {
            rod for rod in all_rods
            if rod.width > 1 and rod.height > 1 and int(rod.pixels[1, 1]) in acted_colors
        }
        movable = set(controlled)
        stack = list(controlled)
        while stack:
            parent = stack.pop()
            for child in getattr(game, names.ATTR_CHILDREN).get(parent, ()):
                if child in all_rods and child not in movable:
                    movable.add(child)
                    stack.append(child)
        self.native_rods = [rod for rod in all_rods if rod in movable]
        local = {rod: index for index, rod in enumerate(self.native_rods)}
        pin_local = {pin: index for index, pin in enumerate(all_pins)}
        self.colors = tuple(int(rod.pixels[1, 1]) for rod in self.native_rods)
        self.solid = tuple(rod in controlled for rod in self.native_rods)
        self.masks = []
        rotation_method = getattr(game, names.METHOD_ROTATION_OF)
        for rod in self.native_rods:
            initial_rotation = int(rotation_method(rod))
            pixels = rod.pixels
            variants = {}
            for turns in range(4):
                rotation = (initial_rotation - 90 * turns) % 360
                variants[rotation] = frozenset(
                    (x, y) for y, row in enumerate(pixels.tolist())
                    for x, value in enumerate(row) if value >= 0
                )
                pixels = np.rot90(pixels)
            self.masks.append(variants)
        self.children = [[] for _ in self.native_rods]
        self.pin_children = [[] for _ in self.native_rods]
        self.parent = {}
        native_children = getattr(game, names.ATTR_CHILDREN)
        for parent, parent_index in local.items():
            for child in native_children.get(parent, ()):
                if child in local:
                    child_index = local[child]
                    self.children[parent_index].append(child_index)
                    self.parent.setdefault(child_index, parent_index)
                elif child in pin_local:
                    self.pin_children[parent_index].append(pin_local[child])
        self.static_pixels = set()
        for rod in all_rods:
            if rod in movable:
                continue
            for y, row in enumerate(rod.pixels.tolist()):
                for x, value in enumerate(row):
                    if value >= 0:
                        self.static_pixels.add((rod.x + x, rod.y + y))
        self.actions = tuple(layout.actions)
        controls = {(names.ACTION_CLICK, control.click[0], control.click[1]): control
                    for control in layout.controls}
        self.action_effects = tuple((controls[action].kind, controls[action].colors)
                                    for action in self.actions)
        self.targets = tuple(layout.targets)
        rods = tuple((rod.x, rod.y, int(rotation_method(rod)),
                      rod.height // names.ROD_THICKNESS if rod.height > rod.width
                      else rod.width // names.ROD_THICKNESS)
                     for rod in self.native_rods)
        pins = tuple((pin.x, pin.y) for pin in all_pins)
        pending = None
        native_backup = getattr(game, names.ATTR_BACKUP)
        if native_backup:
            restored_rods = list(rods)
            restored_pins = list(pins)
            for sprite, clone in native_backup.items():
                if sprite in local:
                    restored_rods[local[sprite]] = (
                        clone.x, clone.y, int(rotation_method(clone)),
                        clone.height // names.ROD_THICKNESS
                        if clone.height > clone.width
                        else clone.width // names.ROD_THICKNESS,
                    )
                elif sprite in pin_local:
                    restored_pins[pin_local[sprite]] = (clone.x, clone.y)
            pending = (tuple(restored_rods), tuple(restored_pins))
        self.start = (rods, pins, pending)

    @staticmethod
    def _size(rod):
        _, _, rotation, length = rod
        return ((names.ROD_THICKNESS, length * names.ROD_THICKNESS)
                if rotation in (0, 180)
                else (length * names.ROD_THICKNESS, names.ROD_THICKNESS))

    def _move_tree(self, rods, pins, index, dx, dy):
        x, y, rotation, length = rods[index]
        rods[index] = [x + dx, y + dy, rotation, length]
        for child in self.children[index]:
            self._move_tree(rods, pins, child, dx, dy)
        for pin in self.pin_children[index]:
            pins[pin][0] += dx
            pins[pin][1] += dy

    def _rotate_child(self, rods, pins, index, pivot_x, pivot_y):
        x, y, rotation, length = rods[index]
        width, _ = self._size(rods[index])
        dx, dy = pivot_x - x, pivot_y - y
        rods[index] = [pivot_x - dy, pivot_y + dx - (width - names.ROD_THICKNESS),
                       self._ROTATED[rotation], length]
        for child in self.children[index]:
            self._rotate_child(rods, pins, child, pivot_x, pivot_y)
        for pin in self.pin_children[index]:
            px, py = pins[pin]
            pins[pin] = [pivot_x - (pivot_y - py), pivot_y + (pivot_x - px)]

    def _rotate_rod(self, rods, pins, index):
        x, y, rotation, length = rods[index]
        width, height = self._size(rods[index])
        if rotation == 0:
            pivot_x, pivot_y = x, y + height - names.ROD_THICKNESS
            move_x, move_y = -height + names.ROD_THICKNESS, height - names.ROD_THICKNESS
        elif rotation == 90:
            pivot_x, pivot_y = x, y
            move_x, move_y = 0, -width + names.ROD_THICKNESS
        elif rotation == 180:
            pivot_x, pivot_y = x, y
            move_x = move_y = 0
        else:
            pivot_x, pivot_y = x + width - names.ROD_THICKNESS, y
            move_x, move_y = width - names.ROD_THICKNESS, 0
        for child in self.children[index]:
            self._rotate_child(rods, pins, child, pivot_x, pivot_y)
        for pin in self.pin_children[index]:
            px, py = pins[pin]
            pins[pin] = [pivot_x - (pivot_y - py), pivot_y + (pivot_x - px)]
        rods[index] = [x + move_x, y + move_y, self._ROTATED[rotation], length]

    def _resize(self, rods, pins, index, delta):
        x, y, rotation, length = rods[index]
        next_length = max(1, length + delta)
        old_width, old_height = self._size(rods[index])
        next_width = (names.ROD_THICKNESS if rotation in (0, 180)
                      else next_length * names.ROD_THICKNESS)
        next_height = (next_length * names.ROD_THICKNESS if rotation in (0, 180)
                       else names.ROD_THICKNESS)
        dx = dy = 0
        if rotation == 0:
            dy = -(next_height - old_height)
            rods[index] = [x, y + dy, rotation, next_length]
        elif rotation == 90:
            dx = next_width - old_width
            rods[index] = [x, y, rotation, next_length]
        elif rotation == 180:
            dy = next_height - old_height
            rods[index] = [x, y, rotation, next_length]
        else:
            dx = -(next_width - old_width)
            rods[index] = [x + dx, y, rotation, next_length]
        for child in self.children[index]:
            self._move_tree(rods, pins, child, dx, dy)
        for pin in self.pin_children[index]:
            pins[pin][0] += dx
            pins[pin][1] += dy

    def _snapshot_tree(self, rods, pins, index, rod_backup, pin_backup):
        rod_backup[index] = tuple(rods[index])
        for child in self.children[index]:
            self._snapshot_tree(rods, pins, child, rod_backup, pin_backup)
        for pin in self.pin_children[index]:
            pin_backup[pin] = tuple(pins[pin])

    def _collides(self, rods):
        occupancies = []
        for index, rod in enumerate(rods):
            x, y, _, _ = rod
            width, height = self._size(rod)
            if self.solid[index]:
                occupied = {(x + dx, y + dy) for dy in range(height) for dx in range(width)}
            else:
                occupied = {(x + dx, y + dy) for dx, dy in self.masks[index][rod[2]]}
            if occupied & self.static_pixels:
                return True
            if any(occupied & previous for previous in occupancies):
                return True
            occupancies.append(occupied)
        return False

    def step(self, state, action_index):
        original_rods, original_pins, pending = state
        if pending is not None:
            restored_rods, restored_pins = pending
            result = (restored_rods, restored_pins, None)
            return result, all(target in restored_pins for target in self.targets)
        rods = [list(rod) for rod in original_rods]
        pins = [list(pin) for pin in original_pins]
        rod_backup = {}
        pin_backup = {}
        kind, colors = self.action_effects[action_index]
        for index, own_color in enumerate(self.colors):
            if own_color not in colors:
                continue
            self._snapshot_tree(rods, pins, index, rod_backup, pin_backup)
            if kind == "rotate":
                parent = self.parent.get(index)
                if (parent is not None
                        and abs(rods[index][2] - 90 - rods[parent][2]) == 180):
                    self._rotate_rod(rods, pins, index)
                self._rotate_rod(rods, pins, index)
            else:
                self._resize(rods, pins, index, 1 if kind == "extend" else -1)
        if self._collides(rods):
            restored_rods = [tuple(rod) for rod in rods]
            restored_pins = [tuple(pin) for pin in pins]
            for index, backup in rod_backup.items():
                restored_rods[index] = backup
            for index, backup in pin_backup.items():
                restored_pins[index] = backup
            # ``Env.perform`` advances both native animation frames: the
            # collided frame and then the backup restoration. Descendants
            # directly dispatched later overwrite their parent's earlier
            # backup, so restoration can intentionally be partial.
            return (tuple(restored_rods), tuple(restored_pins), None), False
        result = (tuple(tuple(rod) for rod in rods), tuple(tuple(pin) for pin in pins), None)
        won = all(target in result[1] for target in self.targets)
        return result, won


def _native_positive(env, actions):
    probe = env.clone()
    score = probe.levels_completed
    for action in actions:
        observation = probe.perform(action[0], action[1], action[2])
        if observation.state == GameState.GAME_OVER:
            return False
        if probe.levels_completed > score:
            return True
    return False


def _search_fast_bfs(layout, env, limit, max_nodes, explicit_cutoff):
    """Shortest-path BFS over compact bar/tree state, with native positives."""
    engine = _FastEngine(env, layout)
    start = engine.start
    parents = {start: None}
    queue = deque([(start, 0)])
    nodes = 0
    deepest = 0
    model_mismatch = False
    while queue:
        state, depth = queue.popleft()
        if depth >= limit:
            continue
        if nodes >= max_nodes:
            return SearchResult(
                None, True, nodes, deepest, False, False,
                "compact BFS work exhausted; impossibility not proved",
            )
        nodes += 1
        for action_index, action in enumerate(engine.actions):
            next_state, won = engine.step(state, action_index)
            if next_state == state:
                continue
            if won:
                path = [action]
                cursor = state
                while parents[cursor] is not None:
                    cursor, previous = parents[cursor]
                    path.append(previous)
                path.reverse()
                if _native_positive(env, path):
                    solution = Solution(path)
                    solution.nodes = nodes
                    return SearchResult(
                        solution, False, nodes, depth + 1, True, True,
                        "shortest compact BFS witness passed fresh native replay",
                    )
                model_mismatch = True
            if next_state in parents:
                continue
            parents[next_state] = (state, action)
            queue.append((next_state, depth + 1))
            deepest = max(deepest, depth + 1)
    truncated = explicit_cutoff or model_mismatch
    return SearchResult(
        None, truncated, nodes, deepest, not truncated, not truncated,
        ("compact/native transition mismatch; negative result withheld"
         if model_mismatch else
         "caller action cutoff reached; impossibility not proved"
         if explicit_cutoff else "compact state space exhausted"),
    )


def _search_fast_witness(layout, env, limit, max_nodes):
    """Bounded guided symbolic search with mandatory fresh native replay."""
    engine = _FastEngine(env, layout)
    start = engine.start
    parents = {start: None}
    depths = {start: 0}
    serial = itertools.count()
    start_h = _pin_distance(start[1], layout.targets)
    queue = [(4 * start_h, start_h, 0, next(serial), start)]
    nodes = 0
    deepest = 0
    while queue and nodes < max_nodes:
        _, _, depth, _, state = heapq.heappop(queue)
        if depth != depths.get(state) or depth >= limit:
            continue
        nodes += 1
        for action_index, action in enumerate(engine.actions):
            next_state, won = engine.step(state, action_index)
            if next_state == state:
                continue
            next_depth = depth + 1
            if won:
                path = [action]
                cursor = state
                while parents[cursor] is not None:
                    cursor, previous = parents[cursor]
                    path.append(previous)
                path.reverse()
                if _native_positive(env, path):
                    solution = Solution(path)
                    solution.nodes = nodes
                    return SearchResult(
                        solution, False, nodes, next_depth, True, False,
                        "guided bar/tree witness passed fresh native replay; not shortest",
                    )
            if next_depth >= depths.get(next_state, 1 << 30):
                continue
            depths[next_state] = next_depth
            parents[next_state] = (state, action)
            heuristic = _pin_distance(next_state[1], layout.targets)
            heapq.heappush(queue, (4 * heuristic + next_depth, heuristic,
                                  next_depth, next(serial), next_state))
            deepest = max(deepest, next_depth)
    return SearchResult(
        None, True, nodes, deepest, False, False,
        "bounded guided model exhausted; impossibility not proved",
    )


def _search_witness(layout, env, limit, max_nodes):
    """Bounded best-first native search for a positive late-tier witness.

    This is deliberately labelled non-exact. Failure is only a cutoff, never
    evidence of impossibility; a returned route is still checked by the same
    native transition loop used by BFS and by caller replay tests.
    """
    engine = _Engine(env)
    start_key, start_data = engine.snapshot()
    parents = {start_key: None}
    depths = {start_key: 0}
    store = {start_key: start_data}
    serial = itertools.count()
    start_h = _pin_distance(start_key[1], layout.targets)
    queue = [(4 * start_h, start_h, 0, next(serial), start_key)]
    nodes = 0
    deepest = 0
    while queue and nodes < max_nodes:
        _, _, depth, _, key = heapq.heappop(queue)
        if depth != depths.get(key) or depth >= limit:
            continue
        nodes += 1
        data = store[key]
        for action in layout.actions:
            engine.restore(data)
            won = engine.click(action[1], action[2])
            if won:
                path = [action]
                cursor = key
                while parents[cursor] is not None:
                    cursor, previous = parents[cursor]
                    path.append(previous)
                path.reverse()
                solution = Solution(path)
                solution.nodes = nodes
                return SearchResult(solution, False, nodes, depth + 1, True, False,
                                    "bounded best-first native positive witness; not shortest")
            next_key, next_data = engine.snapshot()
            next_depth = depth + 1
            if next_depth >= depths.get(next_key, 1 << 30):
                continue
            depths[next_key] = next_depth
            parents[next_key] = (key, action)
            store[next_key] = next_data
            heuristic = _pin_distance(next_key[1], layout.targets)
            # The light depth term prevents arbitrarily long greedy detours;
            # the heuristic weight focuses work on exact pin placement.
            heapq.heappush(queue, (4 * heuristic + next_depth, heuristic,
                                  next_depth, next(serial), next_key))
            deepest = max(deepest, next_depth)
    return SearchResult(None, True, nodes, deepest, False, False,
                        "bounded best-first search exhausted; impossibility not proved")


def solve(env_or_layout, limit=None, max_nodes=DEFAULT_MAX_NODES):
    """A native-replayed completing sequence, or ``None``.

    ``solve.last`` retains the full :class:`SearchResult`; callers must inspect
    ``optimal`` rather than infer shortestness from a successful return.
    """
    result = search(env_or_layout, limit, max_nodes)
    solve.last = result
    solve.truncated = result.truncated
    solve.exact = result.exact
    solve.optimal = result.optimal
    if result.actions is None:
        return None
    result.actions.truncated = result.truncated
    return result.actions


solve.last = None
solve.truncated = False
solve.exact = None
solve.optimal = None
