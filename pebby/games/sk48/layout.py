"""Stable symbolic fingerprint plus real-engine snapshot for SK48 search."""

from dataclasses import dataclass

from arcengine import GameState

from . import names


@dataclass(frozen=True)
class Layout:
    snapshot: object
    exact: bool
    unsupported: tuple[str, ...]
    level_index: int
    moves_left: int
    pair_count: int


def unsupported_conditions(env):
    """Return reasons a live state cannot be represented by the compact model.

    Fixed blockers, crossings, auxiliary heads, and pause-producing pad
    interactions are represented. Positive routes are replayed from this
    native snapshot before they become evidence.
    """
    unsupported = []
    if env.state != GameState.NOT_FINISHED:
        unsupported.append(f"terminal state {env.state.value}")
    if not env.stable():
        unsupported.append("an SK48 animation is still pending")
    heads = env.heads()
    if not heads:
        unsupported.append("level has no chains")
    if not env.pairs():
        unsupported.append("level has no editable/reference chain pairs")
    if env.selected() not in heads:
        unsupported.append("selected chain is not present in the live chain set")
    click_targets = set(env.level.get_sprites_by_tag(names.TAG_CLICK))
    if not click_targets.issubset(set(heads)):
        unsupported.append("a clickable sprite is not a chain head")
    for index, left in enumerate(heads):
        if not (0 <= left.x and 0 <= left.y
                and left.x + left.width <= names.FRAME_SIZE
                and left.y + left.height <= names.FRAME_SIZE):
            unsupported.append("chain heads must remain fully inside the display")
        for right in heads[index + 1:]:
            separated = (
                left.x + left.width <= right.x or right.x + right.width <= left.x
                or left.y + left.height <= right.y or right.y + right.height <= left.y
            )
            if not separated:
                unsupported.append("chain head bounds must remain disjoint")
    for head, segments in env.lines().items():
        seen = set()
        for segment in segments:
            identity = id(segment)
            if identity in seen:
                unsupported.append("one chain contains the same segment object twice")
                break
            seen.add(identity)

    if len(env.level.get_sprites_by_tag(names.TAG_BOUNDARY)) != 1:
        unsupported.append("exact search requires one static movement boundary")
    return tuple(dict.fromkeys(unsupported))


def extract(env):
    """Capture the current live state and declare any unsupported condition."""
    unsupported = unsupported_conditions(env)
    return Layout(
        snapshot=env.clone(),
        exact=not unsupported,
        unsupported=unsupported,
        level_index=env.level_index,
        moves_left=env.moves_left,
        pair_count=len(env.pairs()),
    )
