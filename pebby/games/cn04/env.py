"""Drive the real CN04 game.

Pebby does not re-implement CN04's rules for play. This module loads the
verbatim upstream module from ``third_party/arc3_games/cn04.py`` (byte-identical
to the file ``arc_agi`` downloads) and plays it, so generated levels and the six
shipped levels run under identical logic. Generated levels are injected by
swapping the module-level ``levels`` list for the duration of construction;
``ARCBaseGame`` clones them, so nothing global stays mutated.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "cn04.py"

_import_lock = threading.Lock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_cn04", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """The six shipped ``Level`` objects (fresh clones)."""
    return [level.clone() for level in upstream().levels]


@contextmanager
def _installed_levels(levels):
    module = upstream()
    if levels is None:
        yield module
        return
    with _import_lock:
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
    """A thin, stateful wrapper around one ``Cn04`` instance.

    ``levels=None`` plays the six shipped levels; pass a list of ARCEngine
    ``Level`` objects to play generated ones instead.
    """

    def __init__(self, levels=None):
        with _installed_levels(levels) as module:
            self.game = getattr(module, names.CLASS_GAME)()
        self.module = upstream()
        self.level_count = len(self.game._levels)

    # -- playing ------------------------------------------------------------------

    def reset(self):
        """Full reset to level 0 (RESET on a fresh game is a full reset upstream)."""
        self.game.full_reset()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """Apply one action. ``action_id`` is 0 (RESET) or 1..7; 6 needs display x, y in 0..63."""
        action_id = int(action_id)
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 needs display coordinates x, y")
            action = ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        elif action_id in (0, 1, 2, 3, 4, 5, 7):
            action = ActionInput(id=GameAction.from_id(action_id))
        else:
            raise ValueError(f"unknown action id {action_id}")
        return Observation(self.game.perform_action(action))

    def render(self):
        """The current 64x64 frame (list of lists of ints) without advancing the game."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def clone(self):
        """An independent copy of the whole game, including selection and match state."""
        other = Env.__new__(Env)
        other.game = copy.deepcopy(self.game)
        other.module = self.module
        other.level_count = self.level_count
        return other

    def set_level(self, index):
        # Upstream's on_set_level remaps pin 13 to 8 in-place on the active
        # clone. Re-entering an already fresh level would therefore make setup
        # non-idempotent and corrupt the pristine-pixel table used by the rules.
        # A level at action zero is already in the requested native context.
        index = int(index)
        if index == self.level_index and self.actions_used() == 0:
            return
        self.game.set_level(index)

    # -- observable status ---------------------------------------------------------

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
        return list(self.game._available_actions)

    # -- logical state, for planning and verification ------------------------------

    @property
    def grid_size(self):
        return tuple(self.game.current_level.grid_size)

    def sprites(self):
        """All sprites of the current level in level order (the click-resolution order)."""
        return self.game.current_level.get_sprites()

    def selected(self):
        return getattr(self.game, names.ATTR_SELECTED)

    def stacks(self):
        """Sprite -> list of alternates (the upstream stack map, shared lists)."""
        return getattr(self.game, names.ATTR_STACKS)

    def original_pixels(self, sprite):
        """Pristine pixels with pins still coloured 8 / 13, as a numpy array."""
        return getattr(self.game, names.ATTR_ORIGINAL_PIXELS)[sprite.name]

    def cycle_forward(self):
        return bool(getattr(self.game, names.ATTR_CYCLE_FORWARD))

    def grey_masking(self):
        return bool(getattr(self.game, names.ATTR_GREY_MASKING))

    def max_steps(self):
        return int(getattr(self.game, names.ATTR_MAX_STEPS))

    def actions_used(self):
        return int(self.game._action_count)

    def steps_left(self):
        """Actions that can still be applied without losing on this level."""
        return self.max_steps() - 1 - self.actions_used()

    def level_won_pending(self):
        return bool(getattr(self.game, names.ATTR_LEVEL_WON))

    def is_solved(self):
        """Upstream's own completion test on the current arrangement."""
        return bool(getattr(self.game, names.METHOD_IS_SOLVED)())


def replay(env, actions):
    """True iff ``actions`` complete the current level on their final step."""
    before = env.levels_completed
    for index, (action_id, x, y) in enumerate(actions):
        observation = env.perform(action_id, x, y)
        if observation.state == GameState.GAME_OVER:
            return False
        if observation.levels_completed > before or observation.won:
            return index == len(actions) - 1
    return False


def completes_level(levels, actions):
    """True when replaying ``actions`` on a fresh game of ``levels`` completes level 0.

    Completion is read from the engine's own ``levels_completed`` counter, and the
    run must not end earlier than the last action (no early win, no loss).
    """
    env = Env(levels)
    env.reset()
    return replay(env, actions)
