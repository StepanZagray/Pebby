"""Thin, stateful access to the unmodified vendored RE86 engine."""

from contextlib import contextmanager
import copy
import importlib.util
from numbers import Integral
from pathlib import Path
import sys
import threading

import numpy as np
from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "re86.py"

_module_lock = threading.RLock()
_module = None


def upstream():
    """Import the pinned source once, without changing the vendored file."""
    global _module
    with _module_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location(
                "pebby_vendored_re86", UPSTREAM
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Return fresh clones of all eight shipped levels."""
    with _module_lock:
        return [level.clone() for level in upstream().levels]


@contextmanager
def _installed_levels(levels):
    """Temporarily install levels while holding the module-global lock.

    ``ARCBaseGame`` clones the module list during construction.  Holding the
    lock across the complete constructor prevents concurrent official and
    generated ``Env`` construction from observing each other's level lists.
    """
    with _module_lock:
        module = upstream()
        original = module.levels
        module.levels = list(levels)
        try:
            yield module
        finally:
            module.levels = original


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


class Observation:
    """One native transition with JSON-like integer frames."""

    __slots__ = (
        "frames",
        "state",
        "levels_completed",
        "win_levels",
        "available_actions",
    )

    def __init__(self, frame_data):
        self.frames = [np.asarray(frame).astype(int).tolist()
                       for frame in frame_data.frame]
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
    """A cloneable RE86 environment using native rules and native budgets."""

    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is None:
            selected = official_levels() if levels is None else list(levels)
            if not selected:
                raise ValueError("RE86 needs at least one level")
            with _installed_levels(selected) as module:
                self.game = module.Re86()
            self.reset()
        else:
            self.game = _game
        self.level_count = len(self.game._levels)

    def reset(self):
        """Reset the whole sequential game to level zero."""
        self.game.full_reset()
        return self.render()

    def set_level(self, index):
        """Enter an installed native context without changing its semantics."""
        index = _integer(index, "index")
        if not 0 <= index < self.level_count:
            raise ValueError("level index is outside the installed episode")
        self.game.set_level(index)
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """Perform one native action.

        RE86 exposes ACTION1..ACTION5.  The generic ACTION6 coordinate contract
        is still enforced if a future compatible source exposes it: coordinates
        are full 64x64 display pixels, never grid coordinates.
        """
        action_id = _integer(action_id, "action_id")
        if action_id == names.ACTION_RESET:
            if x is not None or y is not None:
                raise ValueError("RESET does not take coordinates")
            data = {}
        else:
            if action_id not in self.available_actions:
                raise ValueError(
                    f"action must be 0 or one of {self.available_actions}"
                )
            if action_id == names.ACTION_CLICK:
                x = _integer(x, "x")
                y = _integer(y, "y")
                if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
                    raise ValueError("click coordinates must be display pixels in 0..63")
                data = {"x": x, "y": y}
            else:
                if x is not None or y is not None:
                    raise ValueError("non-click actions do not take coordinates")
                data = {}
        frame_data = self.game.perform_action(
            ActionInput(id=GameAction.from_id(action_id), data=data)
        )
        return Observation(frame_data)

    def render(self):
        """Return the current public 64x64 palette frame as Python integers."""
        frame = self.game.camera.render(self.level.get_sprites())
        return np.asarray(frame).astype(int).tolist()

    def clone(self):
        """Return a fully independent mid-game copy."""
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
    def actions_used(self):
        return int(self.game._action_count)

    @property
    def steps_left(self):
        counter = getattr(self.game, names.ATTR_STEP_COUNTER)
        return int(counter.current_steps)

    @property
    def max_steps(self):
        counter = getattr(self.game, names.ATTR_STEP_COUNTER)
        return int(counter.ytjxfhcefb)

    def movables(self):
        return list(self.level.get_sprites_by_tag(names.TAG_MOVABLE))

    def targets(self):
        return list(self.level.get_sprites_by_tag(names.TAG_TARGET))

    def selected_index(self):
        for index, sprite in enumerate(self.movables()):
            if int(sprite.pixels[sprite.height // 2, sprite.width // 2]) == 0:
                return index
        return None

    def stable(self):
        return (
            getattr(self.game, names.ATTR_PENDING_DYE) is None
            and getattr(self.game, names.ATTR_DYE_OBJECT) is None
        )


def replay(env, actions):
    """Replay triples and report whether the current native level completes."""
    before = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True
        if observation.state == GameState.GAME_OVER:
            return False
    return bool(observation and env.levels_completed > before)
