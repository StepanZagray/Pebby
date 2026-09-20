"""Live LP85 engine snapshots and exactness declarations."""

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
    marker_count: int
    control_count: int


def _button_sprites(env):
    return [
        sprite
        for sprite in env.level._sprites
        if sprite.tags and sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX)
    ]


def unsupported_conditions(env):
    """Return conditions outside the exact native-transition search model.

    The search executes real engine clones, including overlapping controls and
    shared or moving cycles.  Its only structural requirement is that the
    current level name has compiled movement maps.  Work limits may still make
    large official layouts inconclusive; that is ``truncated``, not a coverage
    claim.
    """
    reasons = []
    if env.state != GameState.NOT_FINISHED:
        reasons.append(f"terminal state {env.state.value}")
    if env.steps_left <= 0:
        reasons.append("native click budget is exhausted")
    compiled = env.compiled_maps()
    if not isinstance(env.level_name, str) or not env.level_name:
        reasons.append("level has no movement-map name")
    elif not compiled:
        reasons.append(f"no compiled movement maps for {env.level_name!r}")
    else:
        for button in _button_sprites(env):
            parts = button.tags[0].split("_")
            if len(parts) == 3 and parts[1] not in compiled:
                reasons.append(
                    f"button group {parts[1]!r} has no compiled movement map"
                )
    if env.available_actions != names.ACTION_IDS:
        reasons.append(
            f"unexpected native action set {env.available_actions}; expected {names.ACTION_IDS}"
        )
    return tuple(dict.fromkeys(reasons))


def extract(env):
    reasons = unsupported_conditions(env)
    markers = (
        len(env.level.get_sprites_by_tag(names.TAG_TARGET_MARKER))
        + len(env.level.get_sprites_by_tag(names.TAG_ALT_TARGET_MARKER))
    )
    return Layout(
        snapshot=env.clone(),
        exact=not reasons,
        unsupported=reasons,
        level_index=env.level_index,
        steps_left=env.steps_left,
        marker_count=markers,
        control_count=len(_button_sprites(env)),
    )
