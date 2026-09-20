"""Exact live-state declaration for every mechanic shipped by KA59.

The planner deliberately searches cloned native engine states. This module
only decides whether a state is well formed enough for that search; it does
not replace native push, pursuit, timer, or explosion semantics.
"""

from dataclasses import dataclass
from collections import Counter

from arcengine import GameState

from . import names


@dataclass(frozen=True)
class Layout:
    snapshot: object
    exact: bool
    unsupported: tuple[str, ...]
    level_index: int
    steps_left: int
    box_count: int


def unsupported_conditions(env):
    reasons = []
    if env.state != GameState.NOT_FINISHED:
        reasons.append(f"terminal state {env.state.value}")
    if not env.stable():
        reasons.append("a native push/explosion animation is still pending")
    boxes = env.boxes()
    targets = env.targets()
    if not boxes:
        reasons.append("level has no clickable movable objects")
    if boxes and env.selected() not in boxes:
        reasons.append("selected object is not in the clickable object set")
    if any(box.name not in names.BOX_PROTOTYPES for box in boxes):
        reasons.append("unknown movable-object prototype")
    if any(target.name not in names.TARGET_PROTOTYPES for target in targets):
        reasons.append("unknown ordinary target prototype")
    compatible = sorted((box.width, box.height) for box in boxes)
    wanted = sorted((target.width - 2, target.height - 2) for target in targets)
    if compatible != wanted:
        reasons.append("movable objects and size-compatible targets do not match")
    players = env.level.get_sprites_by_tag(names.TAG_PLAYER)
    player_targets = env.level.get_sprites_by_tag(names.TAG_PLAYER_TARGET)
    compatible_players = Counter((sprite.width, sprite.height) for sprite in players)
    wanted_players = Counter(
        (sprite.width - 2, sprite.height - 2) for sprite in player_targets
    )
    if any(compatible_players[size] < count for size, count in wanted_players.items()):
        reasons.append("player targets lack enough size-compatible players")
    enemies = env.level.get_sprites_by_tag(names.TAG_ENEMY)
    if enemies and not players:
        reasons.append("pursuing enemies require a player")
    if len(env.level.get_sprites_by_tag(names.TAG_BOUNDARY)) != 1:
        reasons.append("exact search requires one static boundary")
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
        box_count=len(env.boxes()),
    )
