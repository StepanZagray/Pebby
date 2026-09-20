"""Bounded constructive SU15 teacher over the complete settled native state.

Positive routes are always checked on a private real-engine clone.  The
constructor is deliberately not an exhaustive pixel search, so every negative
result is unknown/cut off rather than a proof of impossibility.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from arcengine import GameState

from . import names
from .layout import Layout, extract


DEFAULT_LIMIT = 50_000
Action = tuple[int, int | None, int | None]


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...] | None
    truncated: bool
    unsupported: bool
    exact: bool
    work: int
    reason: str
    limit: int

    @property
    def solved(self):
        return self.actions is not None


class _WorkMeter:
    """One unit per native transition, consumed before ``perform``."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.cutoff = False

    @property
    def remaining(self):
        return self.limit - self.used

    def perform(self, env, action):
        if self.used >= self.limit:
            self.cutoff = True
            return None
        self.used += 1
        return env.perform(*action)


def _center(sprite):
    return int(sprite.x + sprite.width // 2), int(sprite.y + sprite.height // 2)


def _enemy_kind(game, sprite):
    return int(game.dfqhmningy(game.kcuphgwar[sprite]))


def _requirements(game):
    raw = game.dsqlbvwaj
    rows = raw if isinstance(raw[0], (list, tuple)) else [raw]
    result = Counter()
    reverse = {value: key for key, value in names.ENEMY_REQUIREMENT_KEYS.items()}
    for key, count in rows:
        if isinstance(key, str) and key in reverse:
            result[("enemy", reverse[key])] += int(count)
        else:
            result[("fruit", int(key))] += int(count)
    return result


def _advanced(env, start_score):
    return env.levels_completed > start_score or env.state == GameState.WIN


def _verified(env, actions, meter):
    probe = env.clone()
    start_score = probe.levels_completed
    for index, action in enumerate(actions):
        observation = meter.perform(probe, action)
        if observation is None:
            return False
        if _advanced(probe, start_score):
            return index == len(actions) - 1
        if observation.state == GameState.GAME_OVER:
            return False
    return False


def _descriptor_suffix(env, meter):
    descriptor = env.generated_descriptor
    raw = descriptor.get("solution") if isinstance(descriptor, dict) else None
    if not isinstance(raw, list) or not raw:
        return None
    actions = []
    for value in raw:
        if (not isinstance(value, (list, tuple)) or len(value) != 3
                or type(value[0]) is not int):
            return None
        actions.append((int(value[0]), value[1], value[2]))
    # Prefix recovery: only a suffix that independently wins from the exact
    # current native state is admitted.  No stored boolean is trusted.
    for offset in range(len(actions) - 1, -1, -1):
        suffix = tuple(actions[offset:])
        if _verified(env, suffix, meter):
            return suffix
        if meter.cutoff:
            return None
    return None


class _Constructor:
    def __init__(self, env, meter, budget):
        self.env = env.clone()
        self.meter = meter
        self.budget = budget
        self.start_score = self.env.levels_completed
        self.actions = []

    def _room(self):
        return (self.meter.remaining > 0
                and len(self.actions) < self.budget
                and self.env.state != GameState.GAME_OVER
                and not _advanced(self.env, self.start_score))

    def click(self, point):
        if not self._room():
            if self.meter.remaining <= 0:
                self.meter.cutoff = True
            return False
        x = max(0, min(63, int(round(point[0]))))
        y = max(names.PLAY_MIN_Y, min(names.PLAY_MAX_Y, int(round(point[1]))))
        action = (names.ACTION_CLICK, x, y)
        observation = self.meter.perform(self.env, action)
        if observation is None:
            return False
        self.actions.append(action)
        return self.env.state != GameState.GAME_OVER

    def _objects(self, family, tier):
        if family == "fruit":
            return [sprite for sprite in self.env.fruits()
                    if int(self.env.game.kqywaxhmsb[sprite]) == tier]
        return [sprite for sprite in self.env.enemies()
                if _enemy_kind(self.env.game, sprite) == tier]

    @staticmethod
    def _masses(env):
        fruit = sum(1 << int(env.game.kqywaxhmsb[s]) for s in env.fruits())
        enemy = sum(1 << (_enemy_kind(env.game, s) - 1) for s in env.enemies())
        return fruit, enemy

    def _park_threat_preserving_mass(self):
        if not self.env.enemies() or not self.env.fruits():
            return False
        before = self._masses(self.env)
        before_fruits = tuple(sorted((int(self.env.game.kqywaxhmsb[s]), *_center(s))
                                     for s in self.env.fruits()))
        ranked = []
        for enemy in self.env.enemies():
            origin = _center(enemy)
            for corner in ((1, 11), (62, 11), (1, 61), (62, 61)):
                distance = math.dist(origin, corner)
                if not distance:
                    continue
                point = (origin[0] + (corner[0] - origin[0]) * 6 / distance,
                         origin[1] + (corner[1] - origin[1]) * 6 / distance)
                click = (max(0, min(63, int(round(point[0])))),
                         max(names.PLAY_MIN_Y, min(names.PLAY_MAX_Y, int(round(point[1])))))
                probe = self.env.clone()
                observation = self.meter.perform(
                    probe, (names.ACTION_CLICK, click[0], click[1]))
                if observation is None:
                    return False
                after_fruits = tuple(sorted((int(probe.game.kqywaxhmsb[s]), *_center(s))
                                            for s in probe.fruits()))
                if (observation.state == GameState.GAME_OVER or self._masses(probe) != before
                        or after_fruits != before_fruits):
                    continue
                separation = min(math.dist(_center(e), _center(f))
                                 for e in probe.enemies() for f in probe.fruits())
                ranked.append((-separation, click))
        if not ranked:
            return False
        _, click = min(ranked)
        return self.click(click)

    def merge_to_counts(self, family, required):
        maximum = 8 if family == "fruit" else 3
        for tier in range(maximum):
            if family == "fruit" and required:
                maximum_required = max(required)
                required_mass = sum((1 << value) * count for value, count in required.items())
                usable_above = sum(
                    1 << int(self.env.game.kqywaxhmsb[s])
                    for s in self.env.fruits()
                    if tier < int(self.env.game.kqywaxhmsb[s]) <= maximum_required
                )
                # Extra lower-tier fruit need not be merged when the higher
                # inventory already exactly supplies all requested value.
                # Avoiding that needless work is essential in official tier 5,
                # where pursuers pressure four irrelevant tier-0 distractors.
                if usable_above >= required_mass:
                    continue
            stalled = 0
            parks = 0
            while self._room():
                objects = self._objects(family, tier)
                keep = required.get(tier, 0)
                if len(objects) < keep + 2:
                    break
                if family == "fruit" and self.env.enemies() and not any(
                    int(self.env.game.kqywaxhmsb[s]) > max(required, default=8)
                    for s in self.env.fruits()
                ):
                    separation = min(math.dist(_center(enemy), _center(fruit))
                                     for enemy in self.env.enemies() for fruit in self.env.fruits())
                    if separation < 28 and parks < 4 and self._park_threat_preserving_mass():
                        parks += 1
                        continue
                before_fruit_mass = sum(1 << int(self.env.game.kqywaxhmsb[s])
                                        for s in self.env.fruits())
                before_enemy_mass = sum(1 << (_enemy_kind(self.env.game, s) - 1)
                                        for s in self.env.enemies())
                required_fruit_mass = sum((1 << value) * count
                                          for value, count in required.items()) if family == "fruit" else 0
                maximum_required = max(required, default=8)
                before_relevant_mass = (sum(
                    1 << int(self.env.game.kqywaxhmsb[s])
                    for s in self.env.fruits()
                    if tier <= int(self.env.game.kqywaxhmsb[s]) <= maximum_required
                ) if family == "fruit" else 0)
                before_high_excess = (sum(
                    (1 << int(self.env.game.kqywaxhmsb[s])) - (1 << maximum_required)
                    for s in self.env.fruits()
                    if int(self.env.game.kqywaxhmsb[s]) > maximum_required
                ) if family == "fruit" else 0)
                allow_high_degrade = (family == "fruit" and any(
                    int(self.env.game.kqywaxhmsb[s]) > maximum_required
                    for s in self.env.fruits()))
                candidate_points = []
                for index, first in enumerate(objects):
                    for second in objects[index + 1:]:
                        a, b = _center(first), _center(second)
                        distance = math.dist(a, b)
                        if distance == 0:
                            candidate_points.append(a)
                            continue
                        for origin, goal in ((a, b), (b, a)):
                            step = min(6.0, distance)
                            candidate_points.append((
                                origin[0] + (goal[0] - origin[0]) * step / distance,
                                origin[1] + (goal[1] - origin[1]) * step / distance,
                            ))
                        if distance <= 20:
                            candidate_points.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
                ranked = []
                seen = set()
                for point in candidate_points:
                    click = (max(0, min(63, int(round(point[0])))),
                             max(names.PLAY_MIN_Y, min(names.PLAY_MAX_Y, int(round(point[1])))))
                    if click in seen:
                        continue
                    seen.add(click)
                    probe = self.env.clone()
                    before_steps = probe.native_steps_left
                    observation = self.meter.perform(
                        probe, (names.ACTION_CLICK, click[0], click[1]))
                    if observation is None:
                        return False
                    if observation.state == GameState.GAME_OVER or before_steps - probe.native_steps_left > 1:
                        continue
                    fruit_mass = sum(1 << int(probe.game.kqywaxhmsb[s]) for s in probe.fruits())
                    enemy_mass = sum(1 << (_enemy_kind(probe.game, s) - 1) for s in probe.enemies())
                    after_relevant_mass = (sum(
                        1 << int(probe.game.kqywaxhmsb[s])
                        for s in probe.fruits()
                        if tier <= int(probe.game.kqywaxhmsb[s]) <= maximum_required
                    ) if family == "fruit" else 0)
                    after_high_excess = (sum(
                        (1 << int(probe.game.kqywaxhmsb[s])) - (1 << maximum_required)
                        for s in probe.fruits()
                        if int(probe.game.kqywaxhmsb[s]) > maximum_required
                    ) if family == "fruit" else 0)
                    mass_loss = before_fruit_mass - fruit_mass
                    allowed_high_loss = before_high_excess - after_high_excess
                    # Merge moves must preserve both value systems.  Pursuer
                    # degradation belongs to the explicit later phase.
                    fruit_changed_illegally = family == "fruit" and (
                        after_relevant_mass < before_relevant_mass
                        or (fruit_mass != before_fruit_mass
                            and after_relevant_mass == before_relevant_mass
                            and mass_loss > allowed_high_loss
                            and tier == 0)
                    )
                    if fruit_changed_illegally or enemy_mass != before_enemy_mass:
                        continue
                    after_objects = ([s for s in probe.fruits()
                                      if int(probe.game.kqywaxhmsb[s]) == tier]
                                     if family == "fruit" else
                                     [s for s in probe.enemies()
                                      if _enemy_kind(probe.game, s) == tier])
                    progress = len(objects) - len(after_objects)
                    if len(after_objects) >= 2:
                        nearest = min(math.dist(_center(a), _center(b))
                                      for i, a in enumerate(after_objects)
                                      for b in after_objects[i + 1:])
                    else:
                        nearest = 0
                    ranked.append((-progress, nearest, click))
                if not ranked:
                    return False
                _, _, click = min(ranked)
                before = len(objects)
                if not self.click(click):
                    return False
                after = len(self._objects(family, tier))
                stalled = stalled + 1 if after >= before else 0
                if stalled >= 16:
                    return False
        return True

    def degrade_high_fruit(self, required):
        if not required or not self.env.enemies():
            return True
        maximum = max(required)
        stalled = 0
        required_mass = sum((1 << tier) * count for tier, count in required.items())
        while self._room():
            high = [sprite for sprite in self.env.fruits()
                    if int(self.env.game.kqywaxhmsb[sprite]) > maximum]
            if not high:
                return True
            before = sorted(int(self.env.game.kqywaxhmsb[s]) for s in high)
            protected = {}
            for (family, tier), count in self.requirements.items():
                available = len(self._objects(family, tier))
                if available >= count:
                    protected[(family, tier)] = count
            candidate_points = [(1, 11), (62, 11), (1, 61), (62, 61)]
            for fruit in high:
                a = _center(fruit)
                for enemy in self.env.enemies():
                    b = _center(enemy)
                    distance = math.dist(a, b)
                    if distance:
                        candidate_points.append((a[0] + (b[0] - a[0]) * min(6, distance) / distance,
                                                 a[1] + (b[1] - a[1]) * min(6, distance) / distance))
            ranked = []
            for point in candidate_points:
                click = (max(0, min(63, int(round(point[0])))),
                         max(names.PLAY_MIN_Y, min(names.PLAY_MAX_Y, int(round(point[1])))))
                probe = self.env.clone()
                observation = self.meter.perform(
                    probe, (names.ACTION_CLICK, click[0], click[1]))
                if observation is None:
                    return False
                if observation.state == GameState.GAME_OVER:
                    continue
                if any(len([s for s in (probe.fruits() if family == "fruit" else probe.enemies())
                                if (int(probe.game.kqywaxhmsb[s]) if family == "fruit"
                                    else _enemy_kind(probe.game, s)) == tier]) < count
                       for (family, tier), count in protected.items()):
                    continue
                if self._masses(probe)[0] < required_mass:
                    continue
                high_after = [s for s in probe.fruits()
                              if int(probe.game.kqywaxhmsb[s]) > maximum]
                excess = sum((1 << int(probe.game.kqywaxhmsb[s])) - (1 << maximum)
                             for s in high_after)
                distance_after = (min(math.dist(_center(f), _center(e))
                                      for f in high_after for e in probe.enemies())
                                  if high_after and probe.enemies() else 0)
                ranked.append((excess, distance_after, click))
            if not ranked:
                return False
            _, _, click = min(ranked)
            if not self.click(click):
                return False
            after = sorted(int(self.env.game.kqywaxhmsb[s]) for s in self.env.fruits()
                           if int(self.env.game.kqywaxhmsb[s]) > maximum)
            stalled = stalled + 1 if after == before else 0
            if stalled >= 18:
                return False
        return not any(int(self.env.game.kqywaxhmsb[s]) > maximum for s in self.env.fruits())

    @staticmethod
    def _inside(point, target):
        return (target.x <= point[0] < target.x + target.width
                and target.y <= point[1] < target.y + target.height)

    def _has_required_inventory(self, env):
        for (family, tier), count in self.requirements.items():
            if family == "fruit":
                available = sum(int(env.game.kqywaxhmsb[s]) == tier for s in env.fruits())
            else:
                available = sum(_enemy_kind(env.game, s) == tier for s in env.enemies())
            if available < count:
                return False
        return True

    def _candidate_move(self, family, tier, current, target):
        """Choose a native click by one-step lookahead, useful for type-1 overshoot."""
        cx, cy = current
        tx, ty = _center(target)
        distance = math.dist((cx, cy), (tx, ty))
        directions = []
        if distance:
            directions.append(((tx - cx) / distance, (ty - cy) / distance))
        directions.extend((math.cos(angle), math.sin(angle))
                          for angle in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4,
                                        math.pi, 5 * math.pi / 4, 3 * math.pi / 2,
                                        7 * math.pi / 4))
        best = None
        for dx, dy in directions:
            point = (cx + dx * 6, cy + dy * 6)
            x = max(0, min(63, int(round(point[0]))))
            y = max(names.PLAY_MIN_Y, min(names.PLAY_MAX_Y, int(round(point[1]))))
            probe = self.env.clone()
            observation = self.meter.perform(probe, (names.ACTION_CLICK, x, y))
            if observation is None:
                return None
            if observation.state == GameState.GAME_OVER:
                continue
            if _advanced(probe, self.start_score):
                return names.ACTION_CLICK, x, y
            if not _advanced(probe, self.start_score) and not self._has_required_inventory(probe):
                continue
            candidates = ([s for s in probe.fruits() if int(probe.game.kqywaxhmsb[s]) == tier]
                          if family == "fruit" else
                          [s for s in probe.enemies() if _enemy_kind(probe.game, s) == tier])
            if not candidates:
                continue
            new_point = min((_center(s) for s in candidates), key=lambda p: math.dist(p, current))
            target_distance = 0 if self._inside(new_point, target) else math.dist(new_point, _center(target))
            candidate = (target_distance, (names.ACTION_CLICK, x, y))
            if best is None or candidate < best:
                best = candidate
        return None if best is None else best[1]

    def _park_pursuer(self):
        """Move one threat away while preserving every already-built type."""
        if not self.env.enemies() or not self.env.fruits():
            return False
        fruits = self.env.fruits()
        threats = sorted(self.env.enemies(), key=lambda enemy: min(
            math.dist(_center(enemy), _center(fruit)) for fruit in fruits))
        ranked = []
        for enemy in threats:
            origin = _center(enemy)
            for corner in ((1, 11), (62, 11), (1, 61), (62, 61)):
                distance = math.dist(origin, corner)
                if not distance:
                    continue
                point = (origin[0] + (corner[0] - origin[0]) * 6 / distance,
                         origin[1] + (corner[1] - origin[1]) * 6 / distance)
                click = (max(0, min(63, int(round(point[0])))),
                         max(names.PLAY_MIN_Y, min(names.PLAY_MAX_Y, int(round(point[1])))))
                probe = self.env.clone()
                observation = self.meter.perform(
                    probe, (names.ACTION_CLICK, click[0], click[1]))
                if observation is None:
                    return False
                if observation.state == GameState.GAME_OVER or not self._has_required_inventory(probe):
                    continue
                separation = min(math.dist(_center(e), _center(f))
                                 for e in probe.enemies() for f in probe.fruits())
                ranked.append((-separation, click))
        if not ranked:
            return False
        _, click = min(ranked)
        return self.click(click)

    def place_requirements(self, requirements, *, enemies_first=False):
        if _advanced(self.env, self.start_score):
            return True
        targets = list(self.env.targets())
        occupied = []
        rows = []
        for (family, tier), count in sorted(requirements.items(), key=lambda item: (
            (item[0][0] != "enemy") if enemies_first else (item[0][0] == "enemy"),
            -item[0][1],
        )):
            rows.extend((family, tier) for _ in range(count))
        for family, tier in rows:
            while self._room():
                objects = self._objects(family, tier)
                available = [s for s in objects if not any(
                    self._inside(_center(s), target) for target in occupied)]
                if not available:
                    return False
                pairs = [(math.dist(_center(sprite), _center(target)), sprite, target)
                         for sprite in available for target in targets if target not in occupied]
                if not pairs:
                    return False
                _, sprite, target = min(pairs, key=lambda value: value[0])
                current = _center(sprite)
                if self._inside(current, target):
                    occupied.append(target)
                    break
                action = self._candidate_move(family, tier, current, target)
                if action is None:
                    if not self._park_pursuer():
                        return False
                    continue
                observation = self.meter.perform(self.env, action)
                if observation is None:
                    return False
                self.actions.append(action)
                if _advanced(self.env, self.start_score):
                    return True
                if self.env.state == GameState.GAME_OVER:
                    return False
            else:
                return False
        # Completion is checked only after a click.  If every object was
        # already centered, a harmless far click triggers the native predicate.
        if not _advanced(self.env, self.start_score) and self._room():
            self.click((0, names.PLAY_MIN_Y))
        return _advanced(self.env, self.start_score)

    def run(self):
        requirements = _requirements(self.env.game)
        self.requirements = requirements
        fruit_required = Counter({tier: count for (family, tier), count in requirements.items()
                                  if family == "fruit"})
        enemy_required = Counter({tier: count for (family, tier), count in requirements.items()
                                  if family == "enemy"})
        # Resolve ordinary merge trees before a pursuer can knock their small
        # leaves apart.  High fruit remains untouched by this ascending pass.
        if not self.merge_to_counts("fruit", fruit_required):
            return None
        if _advanced(self.env, self.start_score):
            return (tuple(self.actions)
                    if _verified(self.env_from_start, self.actions, self.meter) else None)
        # Tier 9 creates its required class-3 pursuer and then uses that same
        # actor causally to degrade the high ordinary fruit.
        if enemy_required and not self.merge_to_counts("enemy", enemy_required):
            return None
        if _advanced(self.env, self.start_score):
            return (tuple(self.actions)
                    if _verified(self.env_from_start, self.actions, self.meter) else None)
        if not self.degrade_high_fruit(fruit_required):
            return None
        if _advanced(self.env, self.start_score):
            return (tuple(self.actions)
                    if _verified(self.env_from_start, self.actions, self.meter) else None)
        checkpoint, prefix = self.env.clone(), list(self.actions)
        if not self.place_requirements(requirements, enemies_first=False):
            self.env, self.actions = checkpoint, prefix
            if not self.place_requirements(requirements, enemies_first=True):
                return None
        return (tuple(self.actions)
                if _verified(self.env_from_start, self.actions, self.meter) else None)

    @property
    def env_from_start(self):
        # The constructor mutates only its private clone.  Rewind through an
        # equally private copy by undoing all click snapshots; native energy is
        # not restored, so retain an immutable start supplied by caller instead.
        return self._start

    def set_start(self, start):
        self._start = start.clone()
        return self


def search(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 0):
        raise ValueError("budget must be a nonnegative integer or None")
    layout = env_or_layout if isinstance(env_or_layout, Layout) else extract(env_or_layout)
    if layout.animation_pending:
        return SearchResult(None, False, True, False, 0,
                            "mid-animation SU15 snapshots are unsupported", limit)
    env = layout.snapshot
    action_budget = layout.action_budget if budget is None else min(layout.action_budget, budget)
    meter = _WorkMeter(limit)
    suffix = _descriptor_suffix(env, meter)
    if suffix is not None and len(suffix) <= action_budget:
        return SearchResult(suffix, False, False, True, meter.used,
                            "stored generated certificate suffix revalidated on the live native state", limit)
    if meter.cutoff:
        return SearchResult(None, True, False, False, meter.used,
                            f"certificate suffix work limit {limit} reached", limit)
    constructor = _Constructor(env, meter, action_budget).set_start(env)
    actions = constructor.run()
    if actions is not None:
        return SearchResult(actions, False, False, True, meter.used,
                            "constructive full-mechanics route verified in the real engine", limit)
    truncated = meter.cutoff or meter.used >= limit or len(constructor.actions) >= action_budget
    return SearchResult(None, truncated, False, False, meter.used,
                        "bounded constructive search found no verified route; impossibility is not claimed", limit)


def solve(env_or_layout, limit=DEFAULT_LIMIT, budget=None):
    result = search(env_or_layout, limit=limit, budget=budget)
    solve.last_result = result
    solve.truncated = result.truncated
    solve.unsupported = result.unsupported
    solve.exact = result.exact
    return list(result.actions) if result.actions is not None else None


solve.last_result = None
solve.truncated = False
solve.unsupported = False
solve.exact = False
