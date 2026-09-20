"""Run LF52 through the immutable vendored engine.

LF52 keeps its real logical boards in a module-global dictionary separate from
the placeholder ``Level`` list.  Every construction, reset, explicit level
change, and action (which may advance a level) therefore installs an Env's
private boards under one re-entrant lock.  Official operations take the same
lock, so generated and official environments can safely interleave.
"""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "lf52.py"

_module = None
_module_lock = threading.RLock()


def upstream():
    """Import the pinned source once without modifying it."""
    global _module
    with _module_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_lf52", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Return fresh clones of the ten shipped placeholder levels."""
    with _module_lock:
        return [level.clone() for level in upstream().levels]


def _grid_from_descriptor(module, descriptor):
    rows = list(descriptor["rows_top_down"])
    # The compact generated alphabet covers every entity stack used by the
    # ten shipped levels.  Rail artwork is cosmetic to the logical engine;
    # the movement rule only tests the common ``kraubslpehi`` prefix.
    legend = {
        "x": [names.PEG, names.HOLE],
        "r": [names.PEG_RED, names.HOLE],
        "b": [names.PEG_BLUE, names.HOLE],
        "g": [names.PEG_GRAY, names.HOLE],
        ".": [names.HOLE],
        "p": [names.OBSTACLE, names.HOLE],
        "-": [names.RAIL_PREFIX],
        "|": [f"{names.RAIL_PREFIX}-up"],
        ",": [names.MOVING_HOLE, names.RAIL_PREFIX],
        ";": [names.MOVING_HOLE, f"{names.RAIL_PREFIX}-up"],
        "X": [names.PEG, names.MOVING_HOLE, names.RAIL_PREFIX],
        "R": [names.PEG_RED, names.MOVING_HOLE, names.RAIL_PREFIX],
        "B": [names.PEG_BLUE, names.MOVING_HOLE, names.RAIL_PREFIX],
        "G": [names.PEG_GRAY, names.MOVING_HOLE, names.RAIL_PREFIX],
        "P": [names.OBSTACLE, names.MOVING_HOLE, names.RAIL_PREFIX],
    }
    custom = descriptor.get("legend")
    if custom is not None:
        if not isinstance(custom, dict):
            raise ValueError("generated LF52 legend must be an object")
        legend.update({str(key): list(value) for key, value in custom.items()})
    return getattr(module, names.CLASS_LAYOUT)(
        rows,
        legend,
        {"tile_size": (names.TILE, names.TILE)},
        {"image_groups": [[names.HOLE]]},
    )


@contextmanager
def _installed(levels=None, grids=None):
    module = upstream()
    with _module_lock:
        old_levels = module.levels
        old_grids = {}
        try:
            if levels is not None:
                module.levels = list(levels)
            for key, grid in (grids or {}).items():
                old_grids[key] = module.kciatvszkc.get(key)
                module.kciatvszkc[key] = grid
            yield module
        finally:
            module.levels = old_levels
            for key, grid in old_grids.items():
                if grid is None:
                    module.kciatvszkc.pop(key, None)
                else:
                    module.kciatvszkc[key] = grid


def _frame_list(frame):
    return frame.tolist() if hasattr(frame, "tolist") else copy.deepcopy(frame)


class Observation:
    __slots__ = ("frames", "state", "levels_completed", "win_levels", "available_actions")

    def __init__(self, raw):
        self.frames = [_frame_list(frame) for frame in raw.frame]
        self.state = raw.state
        self.levels_completed = raw.levels_completed
        self.win_levels = raw.win_levels
        self.available_actions = tuple(raw.available_actions)

    @property
    def frame(self):
        return self.frames[-1] if self.frames else None

    @property
    def finished(self):
        return self.state in (GameState.WIN, GameState.GAME_OVER)

    @property
    def won(self):
        return self.state == GameState.WIN


class Env:
    """One independent real LF52 episode."""

    def __init__(self, levels=None, _game=None, _descriptors=None, _last_frame=None):
        self.module = upstream()
        if _game is not None:
            self.game = _game
            self._descriptors = copy.deepcopy(tuple(_descriptors or ()))
            self._grids = {
                f"grid{index + 1}": _grid_from_descriptor(self.module, descriptor)
                for index, descriptor in enumerate(self._descriptors)
                if descriptor is not None
            }
            self._last_frame = copy.deepcopy(_last_frame)
            self.level_count = len(self.game._levels)
            return

        level_list = None if levels is None else list(levels)
        if level_list is None:
            # ``module.levels`` is temporarily replaced while a generated Env
            # is being constructed.  Read the official list under the same
            # lock as installation so this constructor can never observe that
            # temporary value.
            with _module_lock:
                self._descriptors = tuple(None for _ in self.module.levels)
        else:
            self._descriptors = tuple(
                level.get_data(names.LEVEL_LAYOUT_DATA) for level in level_list
            )
        self._grids = {
            f"grid{index + 1}": _grid_from_descriptor(self.module, descriptor)
            for index, descriptor in enumerate(self._descriptors)
            if descriptor is not None
        }
        with _installed(level_list, self._grids) as module:
            self.game = module.Lf52()
            frame = self.game.camera.render(self.game.current_level.get_sprites())
        self._last_frame = _frame_list(frame)
        self.level_count = len(self.game._levels)

    def _scope(self):
        return _installed(grids=self._grids)

    def reset(self):
        """Reset the whole episode to its first level."""
        with self._scope():
            self.game.full_reset()
            frame = self.game.camera.render(self.game.current_level.get_sprites())
        self._last_frame = _frame_list(frame)
        return self.render()

    def set_level(self, index):
        with self._scope():
            self.game.set_level(int(index))
            frame = self.game.camera.render(self.game.current_level.get_sprites())
        self._last_frame = _frame_list(frame)

    def perform(self, action_id, x=None, y=None):
        """Execute a native action, using full 64x64 display coordinates."""
        action_id = int(action_id)
        if action_id != names.ACTION_RESET and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 requires display coordinates x and y")
            x, y = int(x), int(y)
            if not (0 <= x < names.DISPLAY and 0 <= y < names.DISPLAY):
                raise ValueError("click coordinates must be within the 64x64 display")
            data = {"x": x, "y": y}
        with self._scope():
            raw = self.game.perform_action(
                ActionInput(id=GameAction.from_id(action_id), data=data)
            )
        observation = Observation(raw)
        if observation.frame is not None:
            self._last_frame = copy.deepcopy(observation.frame)
        return observation

    def render(self):
        """Return the latest real 64x64 palette frame without consuming frames."""
        return copy.deepcopy(self._last_frame)

    def clone(self):
        """Deep-copy engine state, selection, undo history, and future boards."""
        return Env(
            _game=copy.deepcopy(self.game),
            _descriptors=self._descriptors,
            _last_frame=self._last_frame,
        )

    @property
    def state(self):
        return self.game._state

    @property
    def level_index(self):
        return self.game.level_index

    @property
    def levels_completed(self):
        return self.game._score

    @property
    def available_actions(self):
        return tuple(self.game._available_actions)

    @property
    def world(self):
        return getattr(self.game, names.ATTR_WORLD)

    @property
    def grid(self):
        return getattr(self.world, names.ATTR_GRID)

    @property
    def logical_level(self):
        return int(getattr(self.world, names.ATTR_LOGICAL_LEVEL))

    @property
    def action_count(self):
        return int(getattr(self.world, names.ATTR_ACTION_COUNT))

    @property
    def steps_left(self):
        return max(0, names.native_action_limit(self.logical_level) - self.action_count)

    @property
    def history_depth(self):
        manager = getattr(self.world, names.ATTR_UNDO_MANAGER, None)
        return len(getattr(manager, names.ATTR_UNDO_STACK, ())) if manager else 0

    @property
    def generated_descriptor(self):
        index = self.level_index
        if 0 <= index < len(self._descriptors):
            return copy.deepcopy(self._descriptors[index])
        return None


def replay(env, actions):
    """Replay triples until the current level completes or the episode loses."""
    start_score = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation
