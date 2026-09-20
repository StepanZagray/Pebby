"""Thread-safe wrapper around the real vendored TN36 engine."""

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import threading

from arcengine import ActionInput, GameAction, GameState

from . import names


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / names.SOURCE_FILE
_LOCK = threading.RLock()
_MODULE = None


def upstream():
    """Load the pinned upstream module once without modifying the vendor file."""
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            spec = importlib.util.spec_from_file_location("pebby_vendored_tn36", UPSTREAM)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _MODULE = module
    return _MODULE


@contextmanager
def _installed_levels(levels):
    """Serialize every construction while the module-global level list is visible."""
    with _LOCK:
        module = upstream()
        original = module.levels
        if levels is not None:
            module.levels = list(levels)
        try:
            yield module
        finally:
            module.levels = original


def official_levels():
    """Return fresh clones of all seven shipped levels."""
    with _LOCK:
        return [level.clone() for level in upstream().levels]


class Observation:
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
    """Stateful TN36 environment using display-coordinate action triples."""

    def __init__(self, levels=None):
        with _installed_levels(levels) as module:
            self.game = module.Tn36()
        self.level_count = len(self.game._levels)

    def reset(self):
        self.game.full_reset()
        return self.render()

    def perform(self, action_id, x=None, y=None):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id must be an integer")
        if action_id == names.ACTION_RESET:
            if x is not None or y is not None:
                raise ValueError("RESET coordinates must be None")
            return Observation(self.game.perform_action(ActionInput(id=GameAction.RESET)))
        if action_id not in names.ACTION_IDS:
            raise ValueError(f"action must be 0 (RESET) or one of {names.ACTION_IDS}")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
            raise ValueError("ACTION6 needs integer display coordinates x, y")
        if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
            raise ValueError("ACTION6 display coordinates must be in 0..63")
        action = ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
        return Observation(self.game.perform_action(action))

    def render(self):
        # Camera rendering translates sprite objects in-place in this engine.
        # Render clones so observation requests cannot corrupt the live click
        # coordinate system or invalidate a later teacher action.
        sprites = [sprite.clone() for sprite in self.game.current_level.get_sprites()]
        return self.game.camera.render(sprites).tolist()

    def clone(self):
        other = Env.__new__(Env)
        other.game = copy.deepcopy(self.game)
        # The vendored panel stores opcode callbacks as lambdas closed over the
        # original controller. deepcopy cannot rebind function closures, so a
        # raw copied game would edit its own bits but execute the source actor.
        # Rebuilding the controller also resets selection, editable history,
        # saved checkpoints and reset flags. Preserve the complete deep-copied
        # settled state and replace only the two callback dictionaries.
        for panel in (other.controller.mvqheosngn, other.controller.bzirenxmrg):
            actor = panel.htntnzkbzu

            def invoke(code, *, panel=panel, actor=actor):
                kind, value = names.OPCODE_EFFECTS[code]
                if code == 0:
                    actor.mlejdghzfo()
                elif kind == "dx":
                    panel.cwtesiybfx(value, 0)
                elif kind == "dy":
                    panel.cwtesiybfx(0, value)
                elif kind == "rotation":
                    actor.rotate(value)
                elif kind == "scale":
                    panel.adjust_scale(value)
                elif kind == "color":
                    actor.knfgrcbayu(value)

            panel.okllwtboml = {
                code: (lambda code=code, invoke=invoke: invoke(code))
                for code in names.OPCODE_EFFECTS
            }
        other.level_count = self.level_count
        return other

    def set_level(self, index):
        self.game.set_level(index)

    @property
    def available_actions(self):
        return tuple(self.game._available_actions)

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
    def actions_used(self):
        return self.game._action_count

    @property
    def level(self):
        return self.game.current_level

    @property
    def controller(self):
        return getattr(self.game, names.ATTR_CONTROLLER)

    @property
    def clicks_left(self):
        """Exact external clicks before the native scrolling timer loses."""
        timer = self.level.get_sprites_by_tag(names.TAG_TIMER)[0]
        background = self.level.get_sprites_by_tag(names.TAG_TIMER_BACKGROUND)[0]
        distance = max(0, int(timer.x + timer.width - background.x))
        # Official levels 6-7 move the timer only every second click.
        if self.level_index >= 5:
            counter = int(getattr(self.game.lmkazecqdh, "lmkazecqdh"))
            return max(0, 2 * distance - (counter % 2))
        return distance


def replay(env, actions):
    """Replay until this level completes; return whether it completed."""
    start = env.levels_completed
    for action_id, x, y in actions:
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start or observation.state == GameState.WIN:
            return True
        if observation.state == GameState.GAME_OVER:
            return False
    return False
