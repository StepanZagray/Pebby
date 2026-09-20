"""Run LP85 through its unmodified vendored ARCEngine implementation.

LP85 keeps both its level list and its numbered movement maps in module
globals.  ``Lp85.__init__`` compiles the maps, while ``on_set_level`` reads the
global level count again.  Every construction, reset, and action is therefore
performed under one re-entrant lock with the owning game's clean levels and
generated maps temporarily installed.  This makes sequential generated games
correct without leaving process-global state behind.
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
UPSTREAM = ROOT / "third_party" / "arc3_games" / "lp85.py"

_import_lock = threading.RLock()
_module = None


def upstream():
    """Import the pinned source once, without modifying it."""
    global _module
    with _import_lock:
        if _module is None:
            spec = importlib.util.spec_from_file_location(
                "pebby_vendored_lp85", UPSTREAM
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _module = module
    return _module


def official_levels():
    """Return independent clones of all eight shipped levels."""
    with _import_lock:
        return [level.clone() for level in upstream().levels]


def _level_maps(levels):
    """Collect generated movement maps embedded in ``levels``."""
    found = {}
    for level in levels:
        level_name = level.get_data(names.KEY_LEVEL_NAME)
        raw_map = level.get_data(names.KEY_GENERATED_MAP)
        if raw_map is None:
            continue
        if not isinstance(level_name, str) or not level_name:
            raise ValueError("a generated LP85 map needs a non-empty level_name")
        if level_name in found and found[level_name] != raw_map:
            raise ValueError(f"conflicting generated LP85 maps named {level_name!r}")
        found[level_name] = copy.deepcopy(raw_map)
    return found


@contextmanager
def _installed_configuration(levels):
    """Temporarily install a complete level/map configuration."""
    module = upstream()
    selected = list(levels)
    generated_maps = _level_maps(selected)
    with _import_lock:
        original_levels = module.levels
        original_maps = getattr(module, names.GLOBAL_MAPS)
        merged_maps = dict(original_maps)
        for level_name, raw_map in generated_maps.items():
            if level_name in original_maps and original_maps[level_name] != raw_map:
                raise ValueError(
                    f"generated LP85 map {level_name!r} collides with an official map"
                )
            merged_maps[level_name] = raw_map
        module.levels = selected
        setattr(module, names.GLOBAL_MAPS, merged_maps)
        try:
            yield module
        finally:
            module.levels = original_levels
            setattr(module, names.GLOBAL_MAPS, original_maps)


class Observation:
    """Public fields returned by one native engine action."""

    __slots__ = (
        "frames",
        "state",
        "levels_completed",
        "win_levels",
        "available_actions",
    )

    def __init__(self, frame_data):
        self.frames = frame_data.frame
        self.state = frame_data.state
        self.levels_completed = frame_data.levels_completed
        self.win_levels = frame_data.win_levels
        self.available_actions = tuple(frame_data.available_actions)

    @property
    def frame(self):
        return self.frames[-1] if self.frames else None

    @property
    def won(self):
        return self.state == GameState.WIN


class Env:
    """Stateful facade over one real ``Lp85`` game."""

    def __init__(self, levels=None, _game=None):
        self.module = upstream()
        if _game is None:
            selected = official_levels() if levels is None else list(levels)
            if not selected:
                raise ValueError("LP85 needs at least one level")
            with _installed_configuration(selected) as module:
                self.game = module.Lp85()
                self.game.full_reset()
        else:
            self.game = _game
        self.level_count = len(self.game._levels)

    def _configuration(self):
        # _clean_levels are never mutated by gameplay and retain generated map
        # metadata through Level.clone().
        return self.game._clean_levels

    def reset(self):
        with _installed_configuration(self._configuration()):
            self.game.full_reset()
        return self.render()

    def set_level(self, index):
        """Select a native level while preserving its generated map context."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("level index must be an integer")
        with _installed_configuration(self._configuration()):
            self.game.set_level(index)
        return self.render()

    def perform(self, action_id, x=None, y=None):
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ValueError("action_id must be an integer")
        if action_id != names.ACTION_RESET and action_id not in self.available_actions:
            raise ValueError(f"action must be 0 or one of {self.available_actions}")
        if action_id == names.ACTION_CLICK:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (x, y)
            ):
                raise ValueError("ACTION6 requires integer x and y")
            if not (0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE):
                raise ValueError("click coordinates must be display pixels in 0..63")
            data = {"x": x, "y": y}
        else:
            if x is not None or y is not None:
                raise ValueError("RESET does not take coordinates")
            data = {}
        with _installed_configuration(self._configuration()):
            frame_data = self.game.perform_action(
                ActionInput(id=GameAction.from_id(action_id), data=data)
            )
        return Observation(frame_data)

    def render(self):
        return self.game.camera.render(self.level.get_sprites()).tolist()

    def clone(self):
        with _import_lock:
            return Env(_game=copy.deepcopy(self.game))

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
        return tuple(self.game._available_actions)

    @property
    def level(self):
        return self.game.current_level

    @property
    def steps_left(self):
        hud = getattr(self.game, names.ATTR_STEP_COUNTER)
        return int(getattr(hud, names.HUD_CURRENT_STEPS))

    @property
    def max_steps(self):
        hud = getattr(self.game, names.ATTR_STEP_COUNTER)
        return int(getattr(hud, names.HUD_MAX_STEPS))

    @property
    def level_name(self):
        return self.level.get_data(names.KEY_LEVEL_NAME)

    def compiled_maps(self):
        return getattr(self.game, names.ATTR_COMPILED_MAPS).get(self.level_name, {})


def replay(env, actions):
    """Replay action triples from the live state and report level completion."""
    before = env.levels_completed
    observation = None
    for raw in actions:
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            raise ValueError("each LP85 action must be a triple")
        observation = env.perform(*raw)
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True
        if observation.state == GameState.GAME_OVER:
            return False
    return bool(observation and env.levels_completed > before)
