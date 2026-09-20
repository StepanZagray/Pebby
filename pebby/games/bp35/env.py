"""Real-engine wrapper for the immutable vendored BP35 implementation.

BP35's ARCEngine levels are placeholders: ``on_set_level`` loads a separate
module-global logical grid.  Generated games therefore install their logical
grids under a lock for every operation that can construct or advance a level.
The globals are restored before returning, so separate environments do not
leak generated maps into one another or into official play.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "bp35.py"

_module = None
_module_lock = threading.RLock()


def upstream():
    """Import the byte-for-byte vendored module once."""
    global _module
    with _module_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_bp35", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Return fresh clones of the nine public ARCEngine placeholder levels."""
    return [level.clone() for level in upstream().levels]


def _grid_from_descriptor(module, descriptor):
    return module.qipeamczaw(
        list(descriptor["rows_bottom_up"]),
        copy.deepcopy(descriptor.get("legend", names.LEGEND)),
        {"vxruyoesvkf": (6, 6)},
        {"jibupgvgfzf": copy.deepcopy(descriptor.get("groups", names.GROUPS))},
    )


@contextmanager
def _installed(levels=None, grids=None):
    module = upstream()
    # Official operations also take the lock: otherwise they could observe a
    # generated grid while another environment has temporarily installed it.
    with _module_lock:
        old_levels = module.levels
        old_grids = {}
        try:
            if levels is not None:
                module.levels = list(levels)
            for key, value in (grids or {}).items():
                old_grids[key] = module.tjdtolkmxo.get(key)
                module.tjdtolkmxo[key] = value
            yield module
        finally:
            module.levels = old_levels
            for key, value in old_grids.items():
                if value is None:
                    module.tjdtolkmxo.pop(key, None)
                else:
                    module.tjdtolkmxo[key] = value


class Observation:
    __slots__ = ("frames", "state", "levels_completed", "win_levels", "available_actions")

    def __init__(self, frame_data):
        self.frames = frame_data.frame
        self.state = frame_data.state
        self.levels_completed = frame_data.levels_completed
        self.win_levels = frame_data.win_levels
        self.available_actions = tuple(frame_data.available_actions)

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
    """One independent BP35 game, including its custom undo history."""

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
            self._descriptors = tuple(None for _ in self.module.levels)
            grids = {}
        else:
            self._descriptors = tuple(level.get_data(names.LEVEL_GRID_DATA) for level in level_list)
            grids = {
                f"grid{index + 1}": _grid_from_descriptor(self.module, descriptor)
                for index, descriptor in enumerate(self._descriptors)
                if descriptor is not None
            }
        self._grids = grids
        with _installed(level_list, grids) as module:
            self.game = module.Bp35()
            frame = self.game.camera.render(self.game.current_level.get_sprites())
        self._last_frame = frame.tolist()
        self.level_count = len(self.game._levels)

    def _scope(self):
        return _installed(grids=self._grids)

    def reset(self):
        with self._scope():
            self.game.full_reset()
            frame = self.game.camera.render(self.game.current_level.get_sprites())
        self._last_frame = frame.tolist()
        return self.render()

    def set_level(self, index):
        with self._scope():
            self.game.set_level(int(index))
            frame = self.game.camera.render(self.game.current_level.get_sprites())
        self._last_frame = frame.tolist()

    def perform(self, action_id, x=None, y=None):
        action_id = int(action_id)
        if action_id != 0 and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y")
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
            self._last_frame = (
                observation.frame.tolist()
                if hasattr(observation.frame, "tolist")
                else copy.deepcopy(observation.frame)
            )
        return observation

    def render(self):
        """Return the last real public frame without consuming queued animation."""
        return copy.deepcopy(self._last_frame)

    def clone(self):
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
    def action_count(self):
        return int(getattr(self.game, names.ATTR_ACTION_COUNT))

    @property
    def logical_level(self):
        return int(getattr(self.world, names.ATTR_LOGICAL_LEVEL))

    @property
    def steps_left(self):
        return max(0, names.native_action_limit(self.logical_level) - self.action_count)

    @property
    def generated_descriptor(self):
        index = self.level_index
        if 0 <= index < len(self._descriptors):
            return copy.deepcopy(self._descriptors[index])
        return None


def replay(env, actions):
    """Execute triples until the starting level completes or the game loses."""
    start_score = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation
