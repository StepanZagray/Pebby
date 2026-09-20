"""Settled native-state extraction for complete SU15 planning."""

from dataclasses import dataclass

from arcengine import GameState

from . import names


@dataclass(frozen=True)
class Layout:
    snapshot: object
    native_steps_left: int
    action_budget: int
    history_depth: int
    fruit_count: int
    enemy_count: int
    target_count: int
    generated: bool
    animation_pending: bool
    exact_positive_scope: bool
    unsupported: tuple[str, ...]
    # Compatibility observations retained for old callers when meaningful.
    fruit_position: tuple[int, int]
    fruit_size: tuple[int, int]
    fruit_tier: int
    target_bounds: tuple[int, int, int, int]
    undo_positions: tuple[tuple[int, int], ...]


def extract(env):
    fruits, enemies, targets = env.fruits(), env.enemies(), env.targets()
    reasons = []
    if env.state not in (GameState.NOT_PLAYED, GameState.NOT_FINISHED):
        reasons.append("level is already terminal")
    animation = bool(getattr(env.game, names.ATTR_ANIMATING))
    if animation:
        reasons.append("an engine animation is pending")
    descriptor = env.generated_descriptor
    generated = isinstance(descriptor, dict) and descriptor.get("kind") == names.GENERATED_KIND
    if fruits:
        fruit = fruits[0]
        fruit_position = int(fruit.x), int(fruit.y)
        fruit_size = int(fruit.width), int(fruit.height)
        fruit_tier = int(env.game.kqywaxhmsb[fruit])
    else:
        fruit_position, fruit_size, fruit_tier = (0, 0), (0, 0), -1
    if targets:
        target = targets[0]
        target_bounds = (int(target.x), int(target.y),
                         int(target.x + target.width - 1),
                         int(target.y + target.height - 1))
    else:
        target_bounds = (0, 0, -1, -1)
    undo_positions = []
    if len(fruits) == 1 and not enemies:
        for snapshot in reversed(getattr(env.game, names.ATTR_HISTORY)):
            entries = [entry for entry in snapshot if entry[0] == names.TAG_FRUIT]
            if len(entries) != 1:
                break
            undo_positions.append((int(entries[0][2]), int(entries[0][3])))
    return Layout(
        snapshot=env.clone(), native_steps_left=env.native_steps_left,
        action_budget=env.steps_left, history_depth=env.history_depth,
        fruit_count=len(fruits), enemy_count=len(enemies), target_count=len(targets),
        generated=generated, animation_pending=animation,
        exact_positive_scope=not reasons, unsupported=tuple(reasons),
        fruit_position=fruit_position, fruit_size=fruit_size, fruit_tier=fruit_tier,
        target_bounds=target_bounds, undo_positions=tuple(undo_positions),
    )
