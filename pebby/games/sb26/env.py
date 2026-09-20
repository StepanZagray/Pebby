"""Thin SB26 adapter over the unmodified vendored real engine."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "sb26.py"

# Construction reads a module-global ``levels`` list.  One re-entrant lock
# covers import, official cloning, temporary installation, construction and
# reset, so concurrent adapters can never observe another adapter's levels.
_module_lock = threading.RLock()
_module = None


def upstream():
    """Import the pinned vendored module once without modifying its source."""
    global _module
    with _module_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_sb26", UPSTREAM)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load SB26 source at {UPSTREAM}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
        return _module


def official_levels():
    """Return fresh clones of all eight upstream levels."""
    with _module_lock:
        return [level.clone() for level in upstream().levels]


def prototype(name):
    """Return a fresh clone of one upstream sprite prototype."""
    with _module_lock:
        return upstream().sprites[name].clone()


@contextmanager
def _installed_levels(levels):
    """Temporarily install levels while the real constructor clones them."""
    with _module_lock:
        module = upstream()
        if levels is None:
            yield module
            return
        original = module.levels
        module.levels = list(levels)
        try:
            yield module
        finally:
            module.levels = original


class Observation:
    """Public fields returned by one real-engine action."""

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
    """Stateful adapter for official or caller-supplied SB26 levels.

    Public actions are triples ``(id, x, y)``.  RESET is 0, the native legal
    actions are 5/6/7, and ACTION6 consumes unscaled display coordinates in
    the inclusive range 0..63 because SB26's camera is exactly 64x64.
    """

    def __init__(self, levels=None, _game=None):
        if _game is not None:
            self.game = _game
        else:
            with _installed_levels(levels) as module:
                self.game = module.Sb26()
        self.level_count = len(self.game._levels)

    def reset(self):
        """Fully reset to level zero and return the current 64x64 frame."""
        with _module_lock:
            self.game.full_reset()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """Perform RESET or one native action and return an ``Observation``."""
        action_id = int(action_id)
        if action_id != 0 and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 (RESET) or one of {self.available_actions}")
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 requires display coordinates x and y")
            x, y = int(x), int(y)
            if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
                raise ValueError("ACTION6 display coordinates must be in 0..63")
            data = {"x": x, "y": y}
        else:
            if x is not None or y is not None:
                raise ValueError("non-click actions do not accept coordinates")
            data = {}
        action = ActionInput(id=GameAction.from_id(action_id), data=data)
        return Observation(self.game.perform_action(action))

    def render(self):
        """Return the public 64x64 frame as palette integers."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def clone(self):
        """Return an independent copy of the complete live episode."""
        duplicate = Env.__new__(Env)
        duplicate.game = copy.deepcopy(self.game)
        duplicate.level_count = self.level_count
        return duplicate

    def set_level(self, index):
        with _module_lock:
            self.game.set_level(int(index))

    @property
    def available_actions(self):
        return tuple(int(action) for action in self.game._available_actions)

    @property
    def state(self):
        return self.game._state

    @property
    def level_index(self):
        return int(self.game.level_index)

    @property
    def levels_completed(self):
        return int(self.game._score)

    @property
    def level(self):
        return self.game.current_level

    @property
    def energy(self):
        return int(getattr(self.game, names.ATTR_ENERGY))

    @property
    def initial_energy(self):
        return int(getattr(self.game, names.ATTR_INITIAL_ENERGY))

    @property
    def history_depth(self):
        return len(getattr(self.game, names.ATTR_HISTORY))


def replay(env, actions):
    """Replay from the current state and report whether its level completed."""
    start_score = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation
