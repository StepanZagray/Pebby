"""Drive the real SC25 game.

Pebby does not re-implement SC25's rules. This module loads the verbatim upstream
module from ``third_party/arc3_games/sc25.py`` and plays it, so generated levels
and the six shipped levels run under identical logic. Generated levels are
injected by swapping the module-level ``levels`` list for the duration of
construction; ``ARCBaseGame`` clones them, so nothing global stays mutated.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "sc25.py"

_import_lock = threading.Lock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_sc25", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """The six shipped `Level` objects (the module's own instances; do not mutate)."""
    return list(upstream().levels)


@contextmanager
def _installed_levels(levels):
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
    """A thin, stateful wrapper around one `Sc25` instance.

    `levels=None` plays the six shipped levels. Pass a list of ARCEngine `Level`
    objects to play generated ones instead. The constructor already performs the
    engine's initial `set_level(0)`; call `reset()` to start a fresh episode.
    """

    def __init__(self, levels=None):
        with _installed_levels(levels) as module:
            self.game = module.Sc25()
        self.module = upstream()
        self.level_count = len(self.game._levels)

    # -- playing --------------------------------------------------------------

    def reset(self):
        """Full reset to level 0 through the engine's own RESET action."""
        self.game.perform_action(ActionInput(id=GameAction.RESET))
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """`action_id` is 1..4 (up/down/left/right), 6 (click at display x,y) or 0 (RESET)."""
        if action_id not in (0,) + names.ACTION_IDS:
            raise ValueError(f"action must be 0 (RESET) or one of {names.ACTION_IDS}")
        if action_id == 6:
            if x is None or y is None:
                raise ValueError("ACTION6 needs display coordinates x, y in 0..63")
            action = ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        else:
            action = ActionInput(id=GameAction.from_id(action_id))
        return Observation(self.game.perform_action(action))

    def render(self):
        """The current 64x64 frame (list of lists of colour ints) without advancing the game."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def clone(self):
        """An independent copy of the whole game, animations and all."""
        twin = Env.__new__(Env)
        twin.game = copy.deepcopy(self.game)
        twin.module = self.module
        twin.level_count = self.level_count
        return twin

    def set_level(self, index):
        self.game.set_level(index)

    # -- logical state ----------------------------------------------------------

    @property
    def available_actions(self):
        return list(self.game._available_actions)

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
    def player(self):
        return getattr(self.game, names.ATTR_PLAYER)

    def budget(self):
        return getattr(self.game, names.ATTR_BUDGET)

    def used(self):
        return getattr(self.game, names.ATTR_USED)

    def spells(self):
        return list(getattr(self.game, names.ATTR_SPELLS))

    def snapshot(self):
        player = self.player
        return {"level": self.level_index, "x": player.x, "y": player.y, "scale": player.scale,
                "facing": getattr(self.game, names.ATTR_FACING), "used": self.used(),
                "budget": self.budget(), "state": self.state.value,
                "levels_completed": self.levels_completed}
