"""Run SP80 through the unmodified vendored ARCEngine implementation."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "sp80.py"

_import_lock = threading.RLock()
_module = None


def upstream():
    """Import the byte-for-byte vendored game once."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_sp80", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Fresh clones of the six levels shipped with SP80."""
    return [level.clone() for level in upstream().levels]


@contextmanager
def _installed_levels(levels):
    module = upstream()
    with _import_lock:
        if levels is None:
            # Official construction must share the same lock: another thread
            # may temporarily have generated levels installed in the module.
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
    """A small stateful wrapper around one upstream ``Sp80`` instance."""

    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is None:
            with _installed_levels(levels) as module:
                self.game = module.Sp80()
        else:
            self.game = _game
        self.level_count = len(self.game._levels)

    def reset(self):
        self.game.full_reset()
        return self.render()

    def set_level(self, index):
        self.game.set_level(int(index))

    def perform(self, action_id, x=None, y=None):
        """Execute RESET or an advertised action; clicks require legal display pixels."""
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
        return Observation(
            self.game.perform_action(ActionInput(id=GameAction.from_id(action_id), data=data))
        )

    def render(self):
        """Render the current 64x64 public frame without advancing the game."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def clone(self):
        """Return an independent deep snapshot, including sprite-keyed sets."""
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
    def grid_size(self):
        return tuple(self.level.grid_size)

    @property
    def steps_left(self):
        return int(getattr(self.game, names.ATTR_STEPS_LEFT))

    @property
    def failed_flows(self):
        return int(getattr(self.game, names.ATTR_FAILED_FLOWS))

    @property
    def rotation_k(self):
        return int(getattr(self.game, names.ATTR_ROTATION_K))

    @property
    def mode(self):
        return str(getattr(self.game, names.ATTR_MODE))

    def movables(self):
        return list(getattr(self.game, names.METHOD_MOVABLES)())

    def cups(self):
        return list(getattr(self.game, names.METHOD_CUPS)())

    def selected(self):
        return getattr(self.game, names.ATTR_SELECTED)

    def sprites_by_tag(self, tag):
        return list(self.level.get_sprites_by_tag(tag))


def replay(env, actions):
    """Replay action triples until the starting level completes or the game loses."""
    start_score = env.levels_completed
    observation = None
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation
