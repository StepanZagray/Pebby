"""Full-mechanics live snapshots for DC22's native transition teacher."""

from dataclasses import dataclass

from arcengine import GameState

from . import names


@dataclass(frozen=True)
class Layout:
    snapshot: object
    exact: bool
    unsupported: tuple[str, ...]
    level_index: int
    steps_left: int
    player: tuple[int, int]
    goal: tuple[int, int]


def unsupported_conditions(env):
    reasons = []
    if env.state != GameState.NOT_FINISHED:
        reasons.append(f"terminal state {env.state.value}")
    if not env.stable():
        reasons.append("a native fall/crusher animation is still pending")
    if len(env.level.get_sprites_by_tag(names.TAG_PLAYER)) != 1:
        reasons.append("exact search requires exactly one player")
    if len(env.level.get_sprites_by_tag(names.TAG_GOAL)) != 1:
        reasons.append("exact search requires exactly one goal")
    if env.steps_left <= 0:
        reasons.append("native action budget is exhausted")
    return tuple(dict.fromkeys(reasons))


def extract(env):
    reasons = unsupported_conditions(env)
    return Layout(
        snapshot=env.clone(),
        exact=not reasons,
        unsupported=reasons,
        level_index=env.level_index,
        steps_left=env.steps_left,
        player=(int(env.player.x), int(env.player.y)),
        goal=(int(env.goal.x), int(env.goal.y)),
    )
