"""Stateful adapter over the unmodified vendored G50T engine."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "g50t.py"

_module_lock = threading.RLock()
_module = None


def upstream():
    """Load the pinned source once without modifying it."""
    global _module
    with _module_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_g50t", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Return independent clones of all seven shipped levels."""
    with _module_lock:
        return [level.clone() for level in upstream().levels]


def prototype(name):
    with _module_lock:
        return upstream().sprites[name].clone()


@contextmanager
def _installed_levels(levels):
    """Install constructor input while holding the module-global lock.

    ``G50t.__init__`` reads the module-level ``levels`` value.  The lock spans
    construction and the initial full reset so concurrent generated and
    official environments cannot observe each other's temporary level list.
    """
    with _module_lock:
        module = upstream()
        original = module.levels
        module.levels = list(levels)
        try:
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
    def won(self):
        return self.state == GameState.WIN


class Env:
    """Thin public facade over one native ``G50t`` instance."""

    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is None:
            selected = official_levels() if levels is None else [level.clone() for level in levels]
            if not selected:
                raise ValueError("G50T needs at least one level")
            with _installed_levels(selected) as module:
                self.game = module.G50t()
                self.game.full_reset()
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
            raise ValueError("level index is outside this native game")
        self.game.set_level(index)
        return self.render()

    def perform(self, action_id, x=None, y=None):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id must be an integer")
        if action_id != names.ACTION_RESET and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
                raise ValueError("ACTION6 requires integer x and y")
            if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
                raise ValueError("click coordinates must be display pixels in 0..63")
            data = {"x": x, "y": y}
        elif x is not None or y is not None:
            raise ValueError("reset and non-click actions do not take coordinates")
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
    def controller(self):
        return getattr(self.game, names.ATTR_CONTROLLER)

    @property
    def steps_used(self):
        return int(getattr(self.game, names.ATTR_ACTION_COUNTER))

    @property
    def max_steps(self):
        timer = getattr(self.game, names.ATTR_TIMER)
        return 2 * (int(timer.width) + 1) - 1

    @property
    def steps_left(self):
        return max(0, self.max_steps - self.steps_used)

    def stable(self):
        controller = self.controller
        return not (
            getattr(controller, names.CTRL_ANIMATIONS)
            or getattr(controller, names.CTRL_REWINDING)
        )


def replay(env, actions):
    """Replay action triples and report completion of the current level."""
    before = env.levels_completed
    for step in actions:
        action_id, x, y = step
        observation = env.perform(action_id, x, y)
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True
        if observation.state == GameState.GAME_OVER:
            return False
    return False
