"""Drive the real R11L game.

Nothing here re-implements R11L's rules. The verbatim upstream module
``third_party/arc3_games/r11l.py`` is loaded once and played, so generated
levels and the six shipped levels run under identical logic. Generated levels
are injected by swapping the module-level ``levels`` list for the duration of
construction; ``ARCBaseGame`` clones the list, so nothing global stays mutated.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "r11l.py"

_import_lock = threading.Lock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_r11l", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Fresh clones of the six shipped levels."""
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
        return self.frames[-1] if self.frames else None

    @property
    def finished(self):
        return self.state in (GameState.WIN, GameState.GAME_OVER)

    @property
    def won(self):
        return self.state == GameState.WIN


class Env:
    """A thin, stateful wrapper around one upstream `R11l` instance.

    `levels=None` plays the six shipped levels; pass a list of ARCEngine
    `Level` objects to play generated ones. `max_actions` optionally overrides
    the per-level action budget (upstream hardcodes 60; the game's own budget
    check and HUD read the overridden value).
    """

    def __init__(self, levels=None, max_actions=None):
        with _installed_levels(levels) as module:
            self.game = module.R11l()
        self.module = upstream()
        self.max_actions = names.MAX_ACTIONS if max_actions is None else int(max_actions)
        if self.max_actions != names.MAX_ACTIONS:
            self._apply_budget()
        self.level_count = len(self.game._levels)

    def _apply_budget(self):
        setattr(self.game, names.ATTR_MAX_ACTIONS, self.max_actions)
        hud = getattr(self.game, names.ATTR_STEP_HUD)
        setattr(hud, names.HUD_MAX_STEPS, self.max_actions)
        setattr(hud, names.HUD_CURRENT_STEPS, self.max_actions - self.game._action_count)

    # -- playing --------------------------------------------------------------

    def reset(self):
        """Full reset to level 0 (RESET on a fresh game is a full reset upstream)."""
        self.game.full_reset()
        if self.max_actions != names.MAX_ACTIONS:
            self._apply_budget()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """`action_id` is 6 (click at display pixel x, y in 0..63) or 0 for RESET."""
        if action_id == 0:
            return Observation(self.game.perform_action(ActionInput(id=GameAction.RESET)))
        if action_id not in self.available_actions:
            raise ValueError(f"action must be 0 (RESET) or one of {self.available_actions}")
        if x is None or y is None:
            raise ValueError("a click needs x and y")
        if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
            raise ValueError("click coordinates must be display pixels in 0..63")
        action = ActionInput(id=GameAction.from_id(action_id), data={"x": int(x), "y": int(y)})
        return Observation(self.game.perform_action(action))

    def render(self):
        """The current 64x64 frame (list of lists of colour indices) without advancing."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def set_level(self, index):
        self.game.set_level(index)

    def clone(self):
        """An independent copy of the whole game, including sprite state."""
        other = Env.__new__(Env)
        other.game = copy.deepcopy(self.game)
        other.module = self.module
        other.max_actions = self.max_actions
        other.level_count = self.level_count
        return other

    # -- logical state ----------------------------------------------------------

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
    def actions_taken(self):
        return self.game._action_count

    @property
    def actions_left(self):
        """Clicks that can still be issued without losing (the budget's last action loses)."""
        return self.max_actions - 1 - self.game._action_count

    def hazards_hit(self):
        return getattr(self.game, names.ATTR_HAZARD_COUNT)

    def selected(self):
        return getattr(self.game, names.ATTR_SELECTED)

    def fragments(self):
        return list(getattr(self.game, names.ATTR_FRAGMENTS))

    def groups(self):
        return getattr(self.game, names.ATTR_GROUPS)

    def walls(self):
        return list(getattr(self.game, names.ATTR_WALLS))

    def hazards(self):
        return [s for s in self.game.current_level.get_sprites() if s.name.startswith(names.PREFIX_HAZARD)]

    def pickups(self):
        return list(getattr(self.game, names.ATTR_PICKUPS))

    def absorbing_cores(self):
        return dict(getattr(self.game, names.ATTR_ABSORBED))


def replay(env, actions):
    """Play `actions` (tuples of (action_id, x, y)) on `env`.

    Returns True if the current level was completed during the replay (score
    rose, or the game was won) without the game ending in GAME_OVER first.
    """
    before = env.levels_completed
    for action in actions:
        action_id, x, y = action
        observation = env.perform(action_id, x, y)
        if observation.state == GameState.GAME_OVER:
            return False
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True
    return env.levels_completed > before
