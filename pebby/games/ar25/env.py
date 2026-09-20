"""Stateful wrapper around the immutable vendored AR25 engine."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "ar25.py"
_module = None
_module_lock = threading.RLock()


def upstream():
    global _module
    with _module_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_ar25", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    return [level.clone() for level in upstream().levels]


@contextmanager
def _installed_levels(levels=None):
    module = upstream()
    # Official construction takes the lock too, so it cannot observe another
    # environment's temporary generated level list.
    with _module_lock:
        original = module.levels
        try:
            if levels is not None:
                module.levels = list(levels)
            yield module
        finally:
            module.levels = original


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
    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is None:
            with _installed_levels(levels) as module:
                self.game = module.Ar25()
        else:
            self.game = _game
        self.level_count = len(self.game._levels)

    def reset(self):
        with _installed_levels():
            self.game.full_reset()
        return self.render()

    def set_level(self, index):
        with _installed_levels():
            self.game.set_level(int(index))

    def perform(self, action_id, x=None, y=None):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id must be an integer")
        if action_id != 0 and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y")
            if isinstance(x, bool) or not isinstance(x, int):
                raise ValueError("click x must be an integer")
            if isinstance(y, bool) or not isinstance(y, int):
                raise ValueError("click y must be an integer")
            if not (0 <= x < names.DISPLAY and 0 <= y < names.DISPLAY):
                raise ValueError("click coordinates must be within the 64x64 display")
            data = {"x": x, "y": y}
        elif x is not None or y is not None:
            raise ValueError("non-click actions require null coordinates")
        with _installed_levels():
            raw = self.game.perform_action(
                ActionInput(id=GameAction.from_id(action_id), data=data)
            )
        return Observation(raw)

    def render(self):
        with _installed_levels():
            return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

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
    def native_steps_left(self):
        interface = getattr(self.game, names.ATTR_STEP_INTERFACE)
        return int(interface.current_steps)

    @property
    def history_depth(self):
        return len(getattr(self.game, names.ATTR_HISTORY))

    @property
    def steps_left(self):
        # Useful plans can consume all remaining movements plus any existing
        # free undos. This is the action cap advertised to the shared collector.
        return self.native_steps_left + self.history_depth

    @property
    def generated_descriptor(self):
        return self.game.current_level.get_data(names.KEY_GENERATED)

    def movables(self):
        return list(getattr(self.game, names.ATTR_MOVABLES))

    def goals(self):
        return list(getattr(self.game, names.ATTR_GOALS))

    def mirrors(self):
        return list(getattr(self.game, names.ATTR_MIRRORS))

    def selected(self):
        return getattr(self.game, names.ATTR_SELECTED)


def replay(env, actions):
    start_score = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation
