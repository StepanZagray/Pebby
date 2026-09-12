"""Drive the real LS20 game.

Pebby does not re-implement LS20's rules. This module loads the verbatim
upstream module from ``third_party/ls20/ls20.py`` and plays it, so generated
levels and the seven shipped levels run under identical logic. Generated levels
are injected by swapping the module-level ``levels`` list for the duration of
construction; ``ARCBaseGame`` clones them, so nothing global stays mutated.
"""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "ls20" / "ls20.py"

_import_lock = threading.Lock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_ls20", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


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
        """The current 64x64 grid of colour indices. Empty only after a terminal action."""
        return self.frames[-1] if self.frames else None

    @property
    def finished(self):
        return self.state in (GameState.WIN, GameState.GAME_OVER)

    @property
    def won(self):
        return self.state == GameState.WIN


class Ls20Env:
    """A thin, stateful wrapper around one `Ls20` instance.

    `levels=None` plays the seven shipped levels. Pass a list of ARCEngine
    `Level` objects to play generated ones instead.
    """

    def __init__(self, levels=None):
        with _installed_levels(levels) as module:
            self.game = module.Ls20()
        self.module = upstream()
        self.level_count = len(self.game._levels)

    # -- playing --------------------------------------------------------------

    def reset(self):
        """Full reset to level 0. RESET on a fresh game is a full reset upstream."""
        self.game.full_reset()
        return self.render()

    def perform(self, action):
        """`action` is 1..4 (up/down/left/right) or 0 for RESET."""
        if action not in (0,) + names.ACTION_IDS:
            raise ValueError(f"action must be 0 (RESET) or one of {names.ACTION_IDS}")
        return Observation(self.game.perform_action(ActionInput(id=GameAction.from_id(action))))

    def render(self):
        """The current frame without advancing the game."""
        frame = self.game.camera.render(self.game.current_level.get_sprites())
        return frame.tolist()

    def set_level(self, index):
        self.game.set_level(index)

    # -- logical state, for planning and verification -------------------------

    @property
    def level_index(self):
        return self.game.level_index

    @property
    def state(self):
        return self.game._state

    @property
    def levels_completed(self):
        return self.game._score

    def player_cell(self):
        player = getattr(self.game, names.ATTR_PLAYER)
        return names.pixel_to_cell(player.x, player.y)

    def triple(self):
        """(shape index, colour index, rotation index) currently carried."""
        return (getattr(self.game, names.ATTR_SHAPE_INDEX),
                getattr(self.game, names.ATTR_COLOR_INDEX),
                getattr(self.game, names.ATTR_ROTATION_INDEX))

    def goal_triples(self):
        return list(zip(getattr(self.game, names.ATTR_GOAL_SHAPE_INDEX),
                        getattr(self.game, names.ATTR_GOAL_COLOR_INDEX),
                        getattr(self.game, names.ATTR_GOAL_ROTATION_INDEX)))

    def goals_solved(self):
        return list(getattr(self.game, names.ATTR_GOAL_SOLVED))

    def lives(self):
        return getattr(self.game, names.ATTR_LIVES)

    def steps_left(self):
        hud = getattr(self.game, names.ATTR_STEP_HUD)
        return getattr(hud, names.HUD_CURRENT_STEPS)

    def step_cost(self):
        hud = getattr(self.game, names.ATTR_STEP_HUD)
        return getattr(hud, names.HUD_STEP_COST)

    def fog(self):
        return bool(getattr(self.game, names.ATTR_FOG))

    def snapshot(self):
        """Everything the planner reasons about, as a hashable-ish dict."""
        return {"level": self.level_index, "cell": self.player_cell(), "triple": self.triple(),
                "goals_solved": tuple(self.goals_solved()), "steps_left": self.steps_left(),
                "lives": self.lives(), "state": self.state.value}


class Ls20Scenario(Ls20Env):
    """Play one supplied level with its intended first/later-level rules.

    LS20 gives matching-cycle hints only at engine index zero. Prefix copies
    are never played; they let the unmodified engine execute the selected index.
    The visible episode still contains exactly one level, and score starts at 0.
    """

    def __init__(self, level, context_index=0):
        if isinstance(context_index, bool) or not isinstance(context_index, int) or context_index < 0:
            raise ValueError("context_index must be a nonnegative integer")
        self.context_index = context_index
        super().__init__([level] * (context_index + 1))
        self.level_count = 1
        self.reset()

    def reset(self):
        self.game.full_reset()
        self.game.set_level(self.context_index)
        return self.render()

    def set_level(self, index):
        if index != 0:
            raise IndexError("a scenario contains one played level")
        self.game.set_level(self.context_index)
