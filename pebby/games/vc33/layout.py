"""VC33 native snapshot and explicit structural support declaration."""

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
    clickable_count: int


def unsupported_conditions(env):
    reasons = []
    if env.state != GameState.NOT_FINISHED:
        reasons.append(f"terminal state {env.state.value}")
    if not env.stable():
        reasons.append("a native swap animation is still pending")
    gravity = env.gravity
    if len(gravity) != 2 or not any(gravity) or all(gravity):
        reasons.append("gravity must act on exactly one axis")
    if env.steps_left <= 0:
        reasons.append("native click budget is exhausted")
    if not env.clickable():
        reasons.append("level has no clickable buttons or swap bars")
    if not env.loads() or not env.targets():
        reasons.append("level needs colored loads and target markers")
    walls = env.level.get_sprites_by_tag(names.TAG_WALL)
    supports = env.level.get_sprites_by_tag(names.TAG_SUPPORT)
    if not walls:
        reasons.append("level needs target-bearing walls")
    if not supports:
        reasons.append("level needs gravity supports")
    if supports and any(
        not any(env.game.bcpuwqzpxw(load, support) for support in supports)
        for load in env.loads()
    ):
        reasons.append("every colored load must rest on a native support")
    if walls and any(
        not any(wall.collides_with(target) for wall in walls)
        for target in env.targets()
    ):
        reasons.append("every target marker must overlap a target-bearing wall")
    marker_colors = {int(value) for target in env.targets() for value in target.pixels.flat}
    if any(int(load.pixels[-1, -1]) not in marker_colors for load in env.loads()):
        reasons.append("every colored load needs a matching target marker")
    pairs = getattr(env.game, names.ATTR_BUTTON_PAIRS)
    buttons = env.level.get_sprites_by_tag(names.TAG_BUTTON)
    if any(button not in pairs for button in buttons):
        reasons.append("a balance button has no native support pair")
    return tuple(dict.fromkeys(reasons))


def extract(env):
    reasons = unsupported_conditions(env)
    return Layout(
        snapshot=env.clone(),
        exact=not reasons,
        unsupported=reasons,
        level_index=env.level_index,
        steps_left=env.steps_left,
        clickable_count=len(env.clickable()),
    )
