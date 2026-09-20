"""Drive the real S5I5 game.

Pebby does not re-implement S5I5's rules. The verbatim upstream module at
``third_party/arc3_games/s5i5.py`` is loaded and played, so generated levels
and the eight shipped ones run under identical logic. Generated levels are
injected by swapping the module-level ``levels`` list while the game is
constructed; ``ARCBaseGame`` clones them, so nothing global stays mutated.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "s5i5.py"

_import_lock = threading.RLock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_s5i5", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Fresh clones of the eight shipped levels."""
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
    """One `perform_action` result, in the shape an ARC-AGI-3 agent receives."""

    __slots__ = ("frames", "state", "levels_completed", "win_levels", "available_actions")

    def __init__(self, frame_data):
        self.frames = frame_data.frame
        self.state = frame_data.state
        self.levels_completed = frame_data.levels_completed
        self.win_levels = frame_data.win_levels
        self.available_actions = list(frame_data.available_actions)

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
    """A stateful wrapper around one `S5i5` instance.

    `levels=None` plays the eight shipped levels; otherwise pass ARCEngine
    `Level` objects. The game is reset on construction.
    """

    def __init__(self, levels=None):
        with _installed_levels(official_levels() if levels is None else levels) as module:
            self.game = module.S5i5()
        self.level_count = len(self.game._levels)
        self.reset()

    # -- playing --------------------------------------------------------------

    def reset(self):
        self.game.full_reset()
        return self.render()

    @property
    def available_actions(self):
        return list(self.game._available_actions)

    def perform(self, action_id, x=None, y=None):
        """Action ids: 0 = RESET, 6 = click at display pixel (x, y). Only ids the
        game advertises in `available_actions` (plus RESET) are ever sent."""
        if action_id != names.ACTION_RESET and action_id not in self.available_actions:
            raise ValueError(f"action {action_id} is not in available_actions {self.available_actions}")
        data = {}
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 needs x and y")
            if not (0 <= x < names.FRAME and 0 <= y < names.FRAME):
                raise ValueError("click coordinates must be within the 64x64 frame")
            data = {"x": int(x), "y": int(y)}
        return Observation(self.game.perform_action(ActionInput(id=GameAction.from_id(action_id), data=data)))

    def click(self, x, y):
        return self.perform(names.ACTION_CLICK, x, y)

    def render(self):
        """The current 64x64 frame (ints) without advancing the game."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def set_level(self, index):
        self.game.set_level(index)

    def clone(self):
        """An independent copy of the live game state."""
        other = Env.__new__(Env)
        other.game = copy.deepcopy(self.game)
        other.level_count = self.level_count
        return other

    # -- logical state --------------------------------------------------------

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
    def level(self):
        return self.game.current_level

    def steps_left(self):
        return getattr(getattr(self.game, names.ATTR_STEP_HUD), names.HUD_CURRENT_STEPS)

    def max_steps(self):
        return getattr(getattr(self.game, names.ATTR_STEP_HUD), names.HUD_MAX_STEPS)

    def rods(self):
        return self.level.get_sprites_by_tag(names.TAG_ROD)

    def pins(self):
        return self.level.get_sprites_by_tag(names.TAG_PIN)

    def rails(self):
        return self.level.get_sprites_by_tag(names.TAG_RAIL)

    def buttons(self):
        return self.level.get_sprites_by_tag(names.TAG_BUTTON)

    def targets(self):
        return self.level.get_sprites_by_tag(names.TAG_TARGET)

    def is_won_position(self):
        """Upstream's own win predicate on the current sprites (no click)."""
        return bool(getattr(self.game, names.METHOD_IS_WON)())


def replay(levels, actions, level_index=0):
    """Play `actions` ((id, x, y) triples) on a fresh Env and report whether the
    level at `level_index` was completed by exactly that sequence.

    Returns (completed, env). `completed` is True only if `levels_completed`
    advanced past `level_index` and the game never reported GAME_OVER first.
    """
    env = Env(levels)
    if level_index:
        env.set_level(level_index)
    before = env.levels_completed
    for action_id, x, y in actions:
        obs = env.perform(action_id, x, y)
        if obs.state == GameState.GAME_OVER:
            return False, env
        if env.levels_completed > before:
            return True, env
    return env.levels_completed > before, env
