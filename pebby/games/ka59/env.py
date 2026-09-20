"""Drive KA59 through its unmodified vendored ARCEngine implementation."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "ka59.py"

_import_lock = threading.RLock()
_module = None


def upstream():
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_ka59", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    return [level.clone() for level in upstream().levels]


@contextmanager
def _installed_levels(levels):
    module = upstream()
    with _import_lock:
        original = module.levels
        module.levels = list(levels)
        try:
            yield module
        finally:
            module.levels = original


class Observation:
    __slots__ = ("frames", "state", "levels_completed", "win_levels",
                 "available_actions")

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
    def won(self):
        return self.state == GameState.WIN


class Env:
    """Stateful facade over one native ``Ka59`` game."""

    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is None:
            selected = official_levels() if levels is None else list(levels)
            if not selected:
                raise ValueError("KA59 needs at least one level")
            with _installed_levels(selected) as module:
                self.game = module.Ka59()
            self.reset()
        else:
            self.game = _game
        self.level_count = len(self.game._levels)

    def reset(self):
        self.game.full_reset()
        return self.render()

    def set_level(self, index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("level index must be an integer")
        if not 0 <= index < self.level_count:
            raise ValueError("level index is outside the installed native sequence")
        self.game.set_level(index)
        return self.render()

    def perform(self, action_id, x=None, y=None):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id must be an integer")
        if action_id != names.ACTION_RESET and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in (x, y)):
                raise ValueError("ACTION6 requires integer x and y")
            if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
                raise ValueError("click coordinates must be display pixels in 0..63")
            data = {"x": x, "y": y}
        elif x is not None or y is not None:
            raise ValueError("non-click actions do not take coordinates")
        frame_data = self.game.perform_action(
            ActionInput(id=GameAction.from_id(action_id), data=data)
        )
        return Observation(frame_data)

    def render(self):
        return self.game.camera.render(self.level.get_sprites()).tolist()

    def clone(self):
        return Env(_game=copy.deepcopy(self.game))

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
    def level(self):
        return self.game.current_level

    @property
    def steps_left(self):
        return int(getattr(self.game, names.ATTR_STEP_COUNTER).current_steps)

    @property
    def max_steps(self):
        return int(getattr(self.game, names.ATTR_STEP_COUNTER).koyyeuyzyr)

    def boxes(self):
        return list(self.level.get_sprites_by_tag(names.TAG_BOX))

    def targets(self):
        return list(self.level.get_sprites_by_tag(names.TAG_TARGET))

    def selected(self):
        return getattr(self.game, names.ATTR_SELECTED)

    def stable(self):
        return (
            not getattr(self.game, names.ATTR_PENDING_PUSH)
            and not getattr(self.game, names.ATTR_PENDING_EXPLOSION)
        )


def replay(env, actions):
    """Replay action triples and report whether the current level completes."""
    before = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if observation.state == GameState.GAME_OVER:
            return False
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True
    return bool(observation and env.levels_completed > before)
