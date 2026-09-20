"""Thread-safe facade over DC22's unmodified vendored engine."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "arc3_games" / "dc22.py"

# Dc22.on_set_level reads its module-global ``levels`` even after construction.
# One re-entrant lock therefore covers import, construction, reset, and every
# action that might advance a level.  This prevents two Env instances in the
# same process from observing each other's generated level list.
_module_lock = threading.RLock()
_module = None


def upstream():
    global _module
    with _module_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location(
                "pebby_vendored_dc22", UPSTREAM
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Return fresh clones of all six shipped levels."""
    with _module_lock:
        return [level.clone() for level in upstream().levels]


@contextmanager
def _installed_levels(levels):
    module = upstream()
    with _module_lock:
        original = module.levels
        module.levels = levels
        try:
            yield module
        finally:
            module.levels = original


def _public_frame(frame):
    values = frame.tolist() if hasattr(frame, "tolist") else frame
    return [[int(pixel) for pixel in row] for row in values]


class Observation:
    __slots__ = (
        "frames", "state", "levels_completed", "win_levels",
        "available_actions",
    )

    def __init__(self, frame_data):
        self.frames = [_public_frame(frame) for frame in frame_data.frame]
        self.state = frame_data.state
        self.levels_completed = int(frame_data.levels_completed)
        self.win_levels = int(frame_data.win_levels)
        self.available_actions = tuple(int(action) for action in frame_data.available_actions)

    @property
    def frame(self):
        return self.frames[-1] if self.frames else None

    @property
    def won(self):
        return self.state == GameState.WIN


class Env:
    """A stateful DC22 game with native sequential-level semantics."""

    def __init__(self, levels=None, _game=None, _installed=None):
        self.module = upstream()
        if _game is None:
            selected = official_levels() if levels is None else [level.clone() for level in levels]
            if not selected:
                raise ValueError("DC22 needs at least one level")
            # Preserve the exact list while Dc22 constructs and calls set_level.
            self._installed = selected
            with _installed_levels(self._installed) as module:
                self.game = module.Dc22()
                self.game.full_reset()
        else:
            self.game = _game
            self._installed = _installed
        self.level_count = len(self.game._levels)

    def reset(self):
        with _installed_levels(self._installed):
            self.game.full_reset()
        return self.render()

    def set_level(self, index):
        """Select a native context while keeping this environment's level list installed."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("level index must be an integer")
        if not 0 <= index < self.level_count:
            raise ValueError("level index is outside the installed game")
        with _installed_levels(self._installed):
            self.game.set_level(index)

    def perform(self, action_id, x=None, y=None):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id must be an integer")
        if action_id != names.ACTION_RESET and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in (x, y)):
                raise ValueError("ACTION6 requires integer display coordinates")
            if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
                raise ValueError("click coordinates must be display pixels in 0..63")
            data = {"x": x, "y": y}
        elif x is not None or y is not None:
            raise ValueError("non-click actions do not take coordinates")
        action = ActionInput(id=GameAction.from_id(action_id), data=data)
        with _installed_levels(self._installed):
            return Observation(self.game.perform_action(action))

    def render(self):
        return _public_frame(self.game.camera.render(self.level.get_sprites()))

    def clone(self):
        return Env(
            _game=copy.deepcopy(self.game),
            _installed=[level.clone() for level in self._installed],
        )

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
    def available_actions(self):
        return tuple(int(action) for action in self.game._available_actions)

    @property
    def level(self):
        return self.game.current_level

    @property
    def steps_left(self):
        hud = getattr(self.game, names.ATTR_HUD)
        return int(getattr(hud, names.HUD_CURRENT_STEPS))

    @property
    def max_steps(self):
        hud = getattr(self.game, names.ATTR_HUD)
        return int(getattr(hud, names.HUD_MAX_STEPS))

    @property
    def player(self):
        return getattr(self.game, names.ATTR_PLAYER)

    @property
    def goal(self):
        return getattr(self.game, names.ATTR_GOAL)

    def stable(self):
        return not any(
            bool(getattr(self.game, attr))
            for attr in (
                names.ATTR_FALLING,
                names.ATTR_CRUSHER_MOVING,
                names.ATTR_CRUSHER_ANIMATING,
            )
        )


def replay(env, actions):
    """Replay action triples and report completion of the current level."""
    before = env.levels_completed
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True
        if observation.state == GameState.GAME_OVER:
            return False
    return False
