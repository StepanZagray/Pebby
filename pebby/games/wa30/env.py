"""Drive the real WA30 game.

Pebby does not re-implement WA30's rules. This module loads the verbatim
upstream module from ``third_party/arc3_games/wa30.py`` and plays it, so
generated levels and the nine shipped levels run under identical logic.
Generated levels are injected by swapping the module-level ``levels`` list for
the duration of construction; ``ARCBaseGame`` keeps its own reference, so
nothing global stays mutated.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "wa30.py"

_import_lock = threading.Lock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_wa30", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Fresh clones of the nine shipped levels, in order."""
    return [level.clone() for level in upstream().levels]


@contextmanager
def _installed_levels(levels):
    module = upstream()
    with _import_lock:
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
    """One `perform_action` result, in the shape an ARC-AGI-3 agent receives."""

    __slots__ = ("frames", "state", "levels_completed", "win_levels", "available_actions")

    def __init__(self, frame_data):
        self.frames = frame_data.frame
        self.state = frame_data.state
        self.levels_completed = frame_data.levels_completed
        self.win_levels = frame_data.win_levels
        self.available_actions = frame_data.available_actions

    @property
    def frame(self):
        """The current 64x64 grid of colour indices. Empty only after a terminal action."""
        return self.frames[-1] if self.frames else None

    @property
    def finished(self):
        return self.state in (GameState.WIN, GameState.GAME_OVER)

    @property
    def won(self):
        return self.state == GameState.WIN


class Env:
    """A thin, stateful wrapper around one `Wa30` instance.

    `levels=None` plays the nine shipped levels. Pass a list of ARCEngine
    `Level` objects to play generated ones instead.
    """

    available_actions = names.ACTION_IDS

    def __init__(self, levels=None):
        with _installed_levels(levels) as module:
            self.game = module.Wa30()
        self.module = upstream()
        self.level_count = len(self.game._levels)
        self.reset()

    # -- playing --------------------------------------------------------------

    def reset(self):
        """Full reset to level 0. RESET on a fresh game is a full reset upstream."""
        self.game.full_reset()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """`action_id` is 1..5 or 0 for RESET. WA30 has no click action, so x/y must be None."""
        if action_id not in (0,) + names.ACTION_IDS:
            raise ValueError(f"action must be 0 (RESET) or one of {names.ACTION_IDS}")
        if x is not None or y is not None:
            raise ValueError("WA30 has no click action; x and y must be None")
        return Observation(self.game.perform_action(ActionInput(id=GameAction.from_id(action_id))))

    def render(self):
        """The current frame without advancing the game."""
        frame = self.game.camera.render(self.game.current_level.get_sprites())
        return frame.tolist()

    def clone(self):
        """An independent copy of the whole game, including the current level state.

        The cached upstream module is intentionally shared and immutable during
        play.  Python modules cannot be pickled, so deep-copy the mutable engine
        object rather than the wrapper itself.
        """
        duplicate = object.__new__(type(self))
        duplicate.game = copy.deepcopy(self.game)
        duplicate.module = self.module
        duplicate.level_count = self.level_count
        return duplicate

    def set_level(self, index):
        self.game.set_level(index)

    # -- logical state --------------------------------------------------------

    @property
    def level_index(self):
        return self.game.level_index

    @property
    def state(self):
        return self.game._state

    @property
    def levels_completed(self):
        return self.game._score

    @property
    def finished(self):
        return self.state in (GameState.WIN, GameState.GAME_OVER)

    def sprites(self):
        return self.game.current_level.get_sprites()

    def by_tag(self, tag):
        return self.game.current_level.get_sprites_by_tag(tag)

    def player(self):
        return self.by_tag(names.TAG_PLAYER)[0]

    def steps_left(self):
        return getattr(getattr(self.game, names.ATTR_STEP_HUD), names.HUD_CURRENT_STEPS)

    def max_steps(self):
        return getattr(getattr(self.game, names.ATTR_STEP_HUD), names.HUD_MAX_STEPS)


def replay(env, actions):
    """Play `actions` from the env's current position; True iff they complete the level.

    Completion means `levels_completed` grew by one (or the game reports WIN on
    the final level) no later than the last action, without the game being
    lost on the way. Actions past completion are not played.
    """
    before = env.levels_completed
    for step in actions:
        action_id = step[0] if isinstance(step, (tuple, list)) else step
        obs = env.perform(action_id)
        if obs.levels_completed > before or obs.won:
            return True
        if obs.finished:
            return False
    return False
