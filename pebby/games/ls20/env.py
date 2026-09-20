"""Shared-pipeline facade over the existing real-engine LS20 wrapper."""

import copy
import threading

from pebby.ls20.env import Ls20Env, Observation, upstream


_levels_lock = threading.RLock()


def official_levels():
    """Return fresh clones of LS20's seven shipped levels."""
    return [level.clone() for level in upstream().levels]


class Env(Ls20Env):
    """Play shipped or generated LS20 levels through the unmodified engine.

    The constructor accepts a whole ordered game.  Level indices are real
    engine indices: after a win, LS20 advances to the next supplied level and
    increments ``levels_completed`` exactly once.
    """

    def __init__(self, levels=None):
        selected = official_levels() if levels is None else list(levels)
        if not selected:
            raise ValueError("LS20 needs at least one level")
        # The legacy wrapper installs levels by temporarily replacing the
        # upstream module global. Keep construction atomic for this facade.
        with _levels_lock:
            super().__init__(selected)
        self.level_count = len(selected)
        self.reset()

    @property
    def available_actions(self):
        return tuple(self.game._available_actions)

    @property
    def level(self):
        return self.game.current_level

    def perform(self, action_id, x=None, y=None):
        """Perform RESET (0) or one of the four advertised move actions."""
        if x is not None or y is not None:
            raise ValueError("LS20 actions do not take coordinates")
        if action_id != 0 and action_id not in self.available_actions:
            raise ValueError(
                f"action {action_id} is not in available_actions {self.available_actions}"
            )
        return super().perform(action_id)

    def clone(self):
        """Return an independent copy at the same live engine state."""
        twin = copy.copy(self)
        twin.game = copy.deepcopy(self.game)
        return twin


def replay(env, actions):
    """Replay action triples or ids and return the last observation."""
    observation = None
    for action in actions:
        if isinstance(action, (tuple, list)):
            if len(action) != 3:
                raise ValueError("an action must be an id or an (id, x, y) triple")
            observation = env.perform(*action)
        else:
            observation = env.perform(action)
        if observation.finished:
            break
    return observation
