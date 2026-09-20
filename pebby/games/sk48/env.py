"""Drive SK48 through its unmodified vendored ARCEngine implementation."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "sk48.py"

_import_lock = threading.RLock()
_module = None


def upstream():
    """Import the vendored game once without modifying its source."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_sk48", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Return fresh clones of all eight shipped levels."""
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
    """Stateful facade over one real ``Sk48`` instance."""

    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is None:
            selected = official_levels() if levels is None else list(levels)
            if not selected:
                raise ValueError("SK48 needs at least one level")
            with _installed_levels(selected) as module:
                self.game = module.Sk48()
            self.reset()
        else:
            self.game = _game
        self.level_count = len(self.game._levels)

    def reset(self):
        self.game.full_reset()
        return self.render()

    def set_level(self, index):
        self.game.set_level(int(index))

    def perform(self, action_id, x=None, y=None):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id must be an integer")
        if action_id != names.ACTION_RESET and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
                raise ValueError("click coordinates must be integers")
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
    def level(self):
        return self.game.current_level

    @property
    def moves_left(self):
        return int(getattr(self.game, names.ATTR_MOVES_LEFT))

    def heads(self):
        return list(getattr(self.game, names.ATTR_LINES))

    def lines(self):
        return getattr(self.game, names.ATTR_LINES)

    def pairs(self):
        return getattr(self.game, names.ATTR_PAIRS)

    def selected(self):
        return getattr(self.game, names.ATTR_SELECTED)

    def color_pads(self):
        return list(getattr(self.game, names.ATTR_COLOR_PADS))

    def visited_colors(self):
        return getattr(self.game, names.ATTR_VISITED_COLORS)

    def stable(self):
        return (
            not getattr(self.game, names.ATTR_PENDING_MOVES)
            and not getattr(self.game, names.ATTR_PENDING_PAUSES)
            and getattr(self.game, names.ATTR_WIN_ANIMATION) < 0
        )


def replay(env, actions):
    """Replay action triples; return whether the current level completed."""
    before = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if observation.state == GameState.GAME_OVER:
            return False
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True
    return bool(observation and env.levels_completed > before)
