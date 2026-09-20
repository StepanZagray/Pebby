"""Drive the real TU93 game.

Pebby does not re-implement TU93's rules. This module loads the verbatim
upstream module from ``third_party/arc3_games/tu93.py`` and plays it, so
generated levels and the nine shipped levels run under identical logic.
Generated levels are injected by swapping the module-level ``levels`` list for
the duration of construction; ``ARCBaseGame`` clones them, so nothing global
stays mutated.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "tu93.py"

_import_lock = threading.Lock()
_levels_lock = threading.RLock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_tu93", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """The nine shipped ``Level`` objects, as upstream defines them."""
    with _levels_lock:
        return list(upstream().levels)


def prototype(name):
    """A fresh clone of one upstream sprite prototype."""
    return upstream().sprites[name].clone()


@contextmanager
def _installed_levels(levels):
    # The vendored constructor reads a module-level level list. Keep the lock
    # for the complete temporary swap and construction so concurrent official
    # and generated Env instances cannot observe one another's levels.
    with _levels_lock:
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
    """One ``perform_action`` result, in the shape an ARC-AGI-3 agent receives."""

    __slots__ = ("frames", "state", "levels_completed", "win_levels", "available_actions")

    def __init__(self, frame_data):
        self.frames = frame_data.frame
        self.state = frame_data.state
        self.levels_completed = frame_data.levels_completed
        self.win_levels = frame_data.win_levels
        self.available_actions = frame_data.available_actions

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
    """A thin, stateful wrapper around one ``Tu93`` instance.

    ``levels=None`` plays the nine shipped levels. Pass a list of ARCEngine
    ``Level`` objects to play generated ones instead.
    """

    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is not None:
            self.game = _game
        else:
            with _installed_levels(levels) as module:
                self.game = module.Tu93()
        self.level_count = len(self.game._levels)

    # -- playing --------------------------------------------------------------

    def reset(self):
        """Full reset to level 0 and return the first frame."""
        self.game.full_reset()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """Play one action. ``action_id`` is 0 (RESET) or one of ``available_actions``."""
        if action_id != 0 and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 (RESET) or one of {self.available_actions}")
        action = GameAction.from_id(action_id)
        data = {}
        if action.is_complex():
            data = {"x": 0 if x is None else int(x), "y": 0 if y is None else int(y)}
        return Observation(self.game.perform_action(ActionInput(id=action, data=data)))

    def render(self):
        """The current 64x64 frame without advancing the game."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def set_level(self, index):
        self.game.set_level(index)

    def clone(self):
        """An independent copy of the whole game, including sprite identities used as dict keys."""
        return Env(_game=copy.deepcopy(self.game))

    # -- logical state --------------------------------------------------------

    @property
    def available_actions(self):
        return tuple(self.game._available_actions)

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
    def level(self):
        return self.game.current_level

    def sprites_by_tag(self, tag):
        return self.level.get_sprites_by_tag(tag)

    def maze(self):
        mazes = self.sprites_by_tag(names.TAG_MAZE)
        if len(mazes) != 1:
            raise ValueError(f"expected one maze sprite, found {len(mazes)}")
        return mazes[0]

    def heads(self):
        return self.sprites_by_tag(names.TAG_HEAD)

    def steps_left(self):
        return getattr(getattr(self.game, names.ATTR_HUD), names.HUD_CURRENT_STEPS)

    def max_steps(self):
        return getattr(getattr(self.game, names.ATTR_HUD), names.HUD_MAX_STEPS)

    def phase(self):
        return getattr(self.game, names.ATTR_PHASE)


def replay(env, actions):
    """Play ``actions`` on ``env`` from its current state.

    Returns (completed, observation): ``completed`` is True when the level the
    env started on was reported complete by the engine (score advanced or WIN).
    """
    start_score = env.levels_completed
    observation = None
    for step in actions:
        action_id, x, y = step if isinstance(step, (tuple, list)) else (step, None, None)
        observation = env.perform(action_id, x, y)
        if observation.levels_completed > start_score or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation
