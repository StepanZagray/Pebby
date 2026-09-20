"""Drive the real TR87 game.

Pebby does not re-implement TR87's rules. This module loads the verbatim
upstream module from ``third_party/arc3_games/tr87.py`` and plays it, so
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "tr87.py"

_import_lock = threading.Lock()
_levels_lock = threading.RLock()
_module = None


def upstream():
    """Import the vendored game once. The file itself is never modified."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_tr87", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """The six shipped `Level` objects (clean prototypes; clone before mutating)."""
    return list(upstream().levels)


@contextmanager
def _installed_levels(levels):
    module = upstream()
    with _levels_lock:
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
    """A thin, stateful wrapper around one `Tr87` instance.

    `levels=None` plays the six shipped levels. Pass a list of ARCEngine `Level`
    objects to play generated ones instead. Upstream indexes its seed and budget
    tables by level index (tr87.py:912, 941, 967), so at most six levels fit.
    Full-generator solutions are certified at their declared native context.
    Ad-hoc levels without such a certificate should be solved from live state
    after each transition because the fixed scramble seed depends on index.
    """

    available_actions = names.ACTION_IDS

    def __init__(self, levels=None):
        if levels is not None and not 1 <= len(levels) <= len(names.BUDGET_BY_LEVEL_INDEX):
            raise ValueError(f"TR87 supports 1..{len(names.BUDGET_BY_LEVEL_INDEX)} levels per game")
        with _installed_levels(levels) as module:
            self.game = module.Tr87()
        self.module = upstream()
        self.level_count = len(self.game._levels)

    # -- playing --------------------------------------------------------------

    def reset(self):
        """Full reset to level 0 (RESET on a fresh game is a full reset upstream)."""
        self.game.full_reset()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        """`action_id` is 1..4 or 0 for RESET. TR87 has no click action; x/y are ignored."""
        if action_id not in (0,) + names.ACTION_IDS:
            raise ValueError(f"action must be 0 (RESET) or one of {names.ACTION_IDS}")
        return Observation(self.game.perform_action(ActionInput(id=GameAction.from_id(action_id))))

    def render(self):
        """The current 64x64 frame without advancing the game."""
        return self.game.camera.render(self.game.current_level.get_sprites()).tolist()

    def set_level(self, index):
        self.game.set_level(index)

    def clone(self):
        """An independent copy at the same logical state."""
        twin = copy.copy(self)
        twin.game = copy.deepcopy(self.game)
        return twin

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
    def level(self):
        return self.game.current_level

    def flag(self, key):
        return bool(self.level.get_data(key))

    def all_tiles(self):
        return list(getattr(self.game, names.ATTR_ALL_TILES))

    def source_row(self):
        return [names.symbol(s.name) for s in getattr(self.game, names.ATTR_SOURCE_ROW)]

    def target_row(self):
        return [names.symbol(s.name) for s in getattr(self.game, names.ATTR_TARGET_ROW)]

    def rule_sprites(self):
        """[(lhs sprites, rhs sprites)] in upstream order."""
        return [(list(lhs), list(rhs)) for lhs, rhs in getattr(self.game, names.ATTR_RULES)]

    def rules(self):
        return [([names.symbol(s.name) for s in lhs], [names.symbol(s.name) for s in rhs])
                for lhs, rhs in self.rule_sprites()]

    def cursor(self):
        return getattr(self.game, names.ATTR_CURSOR_INDEX)

    def budget_left(self):
        return getattr(self.game, names.ATTR_BUDGET_LEFT)

    def budget_max(self):
        return getattr(self.game, names.ATTR_BUDGET_MAX)

    def animating(self):
        return getattr(self.game, names.ATTR_ANIMATION_STEP) >= 0

    def marker_at(self, sprite):
        """Name of the invisible double-translation marker under a tile's origin, if any."""
        marker = self.level.get_sprite_at(sprite.x, sprite.y, names.TAG_MARKER)
        return marker.name if marker else None

    def snapshot(self):
        return {"level": self.level_index, "source": self.source_row(), "target": self.target_row(),
                "rules": self.rules(), "cursor": self.cursor(), "budget_left": self.budget_left(),
                "state": self.state.value, "levels_completed": self.levels_completed}


def replay(env, actions):
    """Perform `actions` ([(id, x, y)] or [id]) on `env`; return the last observation."""
    observation = None
    for action in actions:
        action_id = action[0] if isinstance(action, (tuple, list)) else action
        observation = env.perform(action_id)
        if observation.finished:
            break
    return observation
