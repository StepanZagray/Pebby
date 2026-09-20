"""Drive the real CD82 game.

Pebby does not re-implement CD82's rules. This module loads the verbatim
upstream module from ``third_party/arc3_games/cd82.py`` and plays it, so
generated levels and the six shipped levels run under identical logic.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "cd82.py"

_import_lock = threading.Lock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_cd82", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """The six shipped levels, as fresh clones."""
    return [level.clone() for level in upstream().levels]


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
    """A thin, stateful wrapper around one `Cd82` instance.

    `levels=None` plays the six shipped levels. Pass a list of ARCEngine `Level`
    objects to play generated ones instead. Actions are `(action_id, x, y)`:
    1-4 move the dial, 5 paints, 6 clicks display pixel (x, y).
    """

    def __init__(self, levels=None):
        with _installed_levels(levels) as module:
            self.game = module.Cd82()
        self.level_count = len(self.game._levels)

    # -- playing --------------------------------------------------------------

    def reset(self):
        """Full reset to level 0."""
        self.game.full_reset()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        if action_id == 0:
            return Observation(self.game.perform_action(ActionInput(id=GameAction.RESET)))
        if action_id not in names.AVAILABLE_ACTIONS:
            raise ValueError(f"action must be 0 (RESET) or one of {names.AVAILABLE_ACTIONS}")
        if action_id == names.ACTION_CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 needs display coordinates x, y in 0..63")
            data = {"x": int(x), "y": int(y)}
        else:
            data = {}
        action = ActionInput(id=GameAction.from_id(action_id), data=data)
        return Observation(self.game.perform_action(action))

    def perform_all(self, actions):
        """Replay a list of (action_id, x, y); returns the last observation."""
        observation = None
        for action_id, x, y in actions:
            observation = self.perform(action_id, x, y)
        return observation

    def render(self):
        """The current 64x64 frame (list of lists of ints) without advancing the game."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def clone(self):
        """An independent copy of the whole game, mid-episode state included."""
        other = Env.__new__(Env)
        other.game = copy.deepcopy(self.game)
        other.level_count = self.level_count
        return other

    def set_level(self, index):
        self.game.set_level(index)

    # -- logical state, for planning and verification -------------------------

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
    def actions_used(self):
        return self.game._action_count

    @property
    def budget(self):
        return getattr(self.game, names.ATTR_BUDGET)

    def dial(self):
        return getattr(self.game, names.ATTR_DIAL)

    def color(self):
        return getattr(self.game, names.ATTR_COLOR)

    def has_indicator(self):
        return bool(getattr(self.game, names.ATTR_HAS_INDICATOR))

    def _sprite(self, name):
        found = self.game.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def canvas(self):
        return self._sprite(names.SPRITE_CANVAS).pixels.copy()

    def target(self):
        for sprite in self.game.current_level.get_sprites():
            if sprite.name.startswith(names.TARGET_PREFIX):
                return sprite.pixels.copy()
        return None

    def swatches(self):
        """[(colour, click_x, click_y)] in sprite order."""
        result = []
        for sprite in self.game.current_level.get_sprites():
            if sprite.name.startswith(names.SPRITE_SWATCH):
                x, y = names.swatch_click(sprite.x, sprite.y)
                result.append((int(sprite.pixels[2, 2]), x, y))
        return result

    def valid_clicks(self):
        """The game's own list of meaningful ACTION6 targets (swatches + indicator)."""
        game = self.game
        clicks = getattr(game, names.METHOD_SWATCH_CLICKS)() + getattr(game, names.METHOD_INDICATOR_CLICKS)()
        return [(int(a.data["x"]), int(a.data["y"])) for a in clicks]

    def snapshot(self):
        return {"level": self.level_index, "dial": self.dial(), "color": self.color(),
                "actions_used": self.actions_used, "state": self.state.value,
                "levels_completed": self.levels_completed}
