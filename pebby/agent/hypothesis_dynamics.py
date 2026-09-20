"""Explicit-inference predictor for the LS20 rule-variant experiment.

One GAME fixes an action permutation (``variant_id``, indexing ``PERMUTATIONS
= tuple(itertools.permutations(range(4)))``; identity is index 0) that maps
agent actions 0..3 to engine actions 0..3 (up/down/left/right), plus seven
sequential levels played with competition reset semantics (see
``pebby.variants``). LS20's own rules never change within a game.

This module's arm knows the LS20 rules but not the permutation. It keeps the
set of permutations consistent with every observed transition so far and
predicts the next transition under the surviving hypotheses. It never reads
``engine_action``, ``action_map`` or ``variant_id`` -- :func:`step_fields_at`
is the only supported way to turn one row of a game's arrays into the dict a
predictor sees, and it simply does not expose those three columns.

Two rule models share one interface, ``predict_direction(step_fields,
direction) -> dict | None`` (``None`` means abstain: the rule model has no
opinion for that candidate engine direction, and the step must not be used to
eliminate that hypothesis on any field):

* :class:`LocalRuleModel` -- a hand-written, APPROXIMATE local rule using only
  the public per-step fields (neighbour tile classes and the carried glyph).
  It has no signal for goal-mismatch blocking (the goal triple is not in the
  arrays) or for life loss, so it always predicts "moved" on a goal neighbour
  and always predicts ``life_lost=False``; both are documented simplifications
  rather than modelled mechanics. ``ELIMINATE_FIELDS`` excludes ``life_lost``
  for exactly this reason: an unmodelled approximation must never be allowed
  to evict the true hypothesis from the surviving set. It also does not model
  teleports (life loss / level change snap the player to a spawn cell, not to
  the neighbour it was facing), so :class:`HypothesisPredictor` additionally
  suppresses movement/shape/color/rotation elimination on any step flagged
  ``level_changed``, ``reset`` or ``life_lost`` for rule models that do not
  set ``MODELS_TELEPORTS = True`` -- exactly the steps that
  ``pebby.agent.variant_metrics.transition_mask`` also excludes from scoring,
  and for the same reason.
* :class:`EngineRuleModel` -- exact. It reconstructs the game's own engine
  (identity-labelled: ``pebby.variants.VariantGame(specs, variant_id=0)``, so
  raw actions 0..3 go straight through with no permutation) from
  ``tier_seeds`` via ``pebby.variants.game_specs`` and the reference bank at
  ``data/ls20-reference-unequal-v1/train.jsonl``. At every step it clones that
  reference game once per *distinct* candidate direction among the surviving
  hypotheses, steps the clone with ``VariantGame.step`` (the same method that
  produced the recorded data, so life-loss/level-reset/level-change handling
  is exact, not reimplemented here) and compares the resulting factored state
  to the recorded after-state. It never replays the logged ``engine_action``
  column. Because it is exact, ``MODELS_TELEPORTS = True`` and it keeps
  ``life_lost`` (and movement/shape/color/rotation, which its clone also gets
  right across a teleport) in ``ELIMINATE_FIELDS`` unconditionally.

:class:`HypothesisPredictor` drives either rule model: ``reset_game`` starts a
new game with all 24 permutations alive, ``predict`` returns a prior-weighted
plurality prediction (each surviving hypothesis votes, independently per
field, weighted by ``prior``; a uniform prior is the default), and ``observe``
eliminates hypotheses whose rule-model prediction disagrees with the recorded
after-state on the applicable fields. The prior only ever changes which value
wins a tied plurality vote; it never participates in elimination.

``score_predictions`` / ``aggregate`` / ``identity_baseline_predictions``
delegate to ``pebby.agent.variant_metrics`` when that module is importable
(it is, as of this module's introduction) and fall back to a small local
equivalent, documented inline, otherwise.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np

from ..ls20 import names
from .world_data import clone_env

# --- neighbour tile classes (matches pebby.variants.TILE_CLASSES exactly) ---
NEI_FREE = 0
NEI_WALL = 1
NEI_CYCLER_SHAPE = 2
NEI_CYCLER_COLOR = 3
NEI_CYCLER_ROTATION = 4
NEI_REFILL = 5
NEI_LAUNCHER = 6
NEI_GOAL = 7
NEI_RAIL = 8
NEI_OUTSIDE = 9

NEIGHBOUR_CLASS_NAMES = (
    "free", "wall", "cycler_shape", "cycler_color", "cycler_rotation",
    "refill", "launcher", "goal", "rail", "outside",
)

# --- the permutation space ----------------------------------------------------
PERMUTATIONS = tuple(itertools.permutations(range(4)))
assert len(PERMUTATIONS) == 24 and PERMUTATIONS[0] == (0, 1, 2, 3)
IDENTITY_VARIANT_ID = 0

DEFAULT_BANK_PATH = Path("data/ls20-reference-unequal-v1/train.jsonl")

# The public per-step fields a predictor is allowed to see. Deliberately
# excludes engine_action, action_map and variant_id.
PUBLIC_STEP_FIELDS = (
    "level_index", "step_in_level",
    "before_player_x", "before_player_y", "before_shape", "before_color",
    "before_rotation", "before_steps_left", "before_lives", "before_goals_mask",
    "neighbours", "agent_action",
    "after_player_x", "after_player_y", "after_shape", "after_color",
    "after_rotation", "after_steps_left", "after_lives", "after_goals_mask",
    "life_lost", "level_changed", "reset", "won", "finished",
)


def _has(arrays, key):
    try:
        return key in arrays
    except TypeError:
        return hasattr(arrays, key)


def step_fields_at(arrays, t):
    """One step's public fields, as a plain dict. Never exposes the three label columns."""
    return {field: arrays[field][t] for field in PUBLIC_STEP_FIELDS}


def _sign(value):
    return (value > 0) - (value < 0)


def movement_class(before_x, before_y, after_x, after_y):
    """0 stayed, else 1..4 for the engine direction (up/down/left/right) moved to.

    Unit-agnostic: only the sign of the displacement is used, so it works
    whether the coordinates are cell or pixel units, as long as both sides use
    the same units (the data contract uses cell units, matching
    ``pebby.agent.variant_metrics.movement_class``, which requires an exact
    unit-cell delta; a sign-based match agrees with it on every unit step and
    is a superset-safe generalisation).
    """
    dx = _sign(int(after_x) - int(before_x))
    dy = _sign(int(after_y) - int(before_y))
    if dx == 0 and dy == 0:
        return 0
    for direction, (ddx, ddy) in enumerate(names.ACTION_DELTAS):
        if _sign(ddx) == dx and _sign(ddy) == dy:
            return direction + 1
    return 0  # not a single-axis unit step; treat conservatively as "stayed"


def _observed(step_fields):
    return {
        "movement": movement_class(step_fields["before_player_x"], step_fields["before_player_y"],
                                    step_fields["after_player_x"], step_fields["after_player_y"]),
        "shape": int(step_fields["after_shape"]),
        "color": int(step_fields["after_color"]),
        "rotation": int(step_fields["after_rotation"]),
        "life_lost": bool(step_fields["life_lost"]),
    }


def _is_teleport(step_fields):
    """A life loss or level change snaps the player to a spawn cell, matching
    ``pebby.agent.variant_metrics.transition_mask``'s exclusion exactly."""
    return bool(step_fields.get("level_changed")) or bool(step_fields.get("reset")) \
        or bool(step_fields.get("life_lost"))


# --- rule model A: local, approximate ----------------------------------------

class LocalRuleModel:
    """Hand-written LS20 local rule from the public per-step fields alone.

    See the module docstring for the documented approximations (goal cells,
    life loss, teleports). Launcher and rail neighbours abstain entirely.
    """

    ELIMINATE_FIELDS = ("movement", "shape", "color", "rotation")
    MODELS_TELEPORTS = False
    ABSTAIN_CLASSES = (NEI_LAUNCHER, NEI_RAIL)
    BLOCK_CLASSES = (NEI_WALL, NEI_OUTSIDE)

    def predict_direction(self, step_fields, direction):
        tile = int(step_fields["neighbours"][direction])
        if tile in self.ABSTAIN_CLASSES:
            return None
        moved = tile not in self.BLOCK_CLASSES  # approximate: goal treated as always-moves
        shape = int(step_fields["before_shape"])
        color = int(step_fields["before_color"])
        rotation = int(step_fields["before_rotation"])
        if tile == NEI_CYCLER_SHAPE:
            shape = (shape + 1) % names.SHAPE_COUNT
        elif tile == NEI_CYCLER_COLOR:
            color = (color + 1) % names.COLOR_COUNT
        elif tile == NEI_CYCLER_ROTATION:
            rotation = (rotation + 1) % names.ROTATION_COUNT
        return {
            "movement": direction + 1 if moved else 0,
            "shape": shape, "color": color, "rotation": rotation,
            "life_lost": False,  # unmodelled; excluded from ELIMINATE_FIELDS
        }


# --- rule model B: the real engine, exact -------------------------------------

_BANK_SPECS_CACHE = {}


def _bank_specs(bank_path):
    key = str(bank_path)
    cached = _BANK_SPECS_CACHE.get(key)
    if cached is not None:
        return cached
    from ..ls20.bank import load as load_bank
    specs = load_bank(bank_path)
    _BANK_SPECS_CACHE[key] = specs
    return specs


def _build_reference_game(tier_seeds, tiers, bank_path):
    """The identity-labelled reference ``VariantGame`` for one game.

    Uses the shared ``pebby.variants`` contract: ``game_specs`` looks each
    tier's seed up in the reference bank, ``VariantGame`` plays them with the
    exact competition reset semantics the recorded data was collected under
    (see ``tools/collect_variant_games.py``, which is the same call with the
    true, unknown ``variant_id``).
    """
    from .. import variants as _variants
    bank_specs = _bank_specs(bank_path)
    specs = _variants.game_specs(bank_specs, tier_seeds, tiers=tiers)
    return _variants.VariantGame(specs, _variants.IDENTITY_VARIANT)


def _clone_variant_game(game):
    """Cheap clone: share the read-only level/spec/layout-cache references
    (``VariantGame.step`` never mutates them), deep-copy only the live engine."""
    clone = copy.copy(game)
    clone.env = clone_env(game.env)
    return clone


class EngineRuleModel:
    """Exact LS20 rules, using the real engine (identity-labelled) as the rule model.

    Maintains one reference ``VariantGame`` per game. ``predict_direction``
    clones it, applies the candidate raw action and reads off the resulting
    factored state; results are cached per (level_index, step_in_level,
    direction) so that a step's ``predict`` and ``observe`` calls, which each
    try the same candidate directions, never repeat the clone+step work.
    ``commit`` (called by :class:`HypothesisPredictor` once elimination for a
    step is decided) adopts the matching clone as the new reference state, so
    the reference game advances purely from ``agent_action`` plus its own
    rule predictions -- never from the logged ``engine_action`` column.
    """

    ELIMINATE_FIELDS = ("movement", "shape", "color", "rotation", "life_lost")
    MODELS_TELEPORTS = True

    def __init__(self, bank_path=None):
        self.bank_path = Path(bank_path) if bank_path else DEFAULT_BANK_PATH
        self._game = None
        self._cache_token = None
        self._cache = {}
        self.engine_steps = 0  # VariantGame.step() calls, for throughput reporting

    def reset_game(self, tier_seeds, tiers=None):
        if tier_seeds is None:
            raise ValueError("EngineRuleModel.reset_game requires tier_seeds")
        tiers = [int(t) for t in tiers] if tiers is not None else None
        self._game = _build_reference_game([int(s) for s in tier_seeds], tiers, self.bank_path)
        self._cache_token = None
        self._cache = {}

    def _refresh_cache(self, step_fields):
        token = (int(step_fields["level_index"]), int(step_fields["step_in_level"]))
        if token != self._cache_token:
            self._cache_token = token
            self._cache = {}

    def predict_direction(self, step_fields, direction):
        self._refresh_cache(step_fields)
        cached = self._cache.get(direction)
        if cached is not None:
            return cached[0]
        clone = _clone_variant_game(self._game)
        state_after, info = clone.step(direction)
        self.engine_steps += 1
        pred = {
            "movement": movement_class(step_fields["before_player_x"], step_fields["before_player_y"],
                                        state_after["player_x"], state_after["player_y"]),
            "shape": state_after["shape"], "color": state_after["color"], "rotation": state_after["rotation"],
            "life_lost": bool(info["life_lost"]),
        }
        self._cache[direction] = (pred, clone)
        return pred

    def commit(self, step_fields, confirmed_direction):
        if confirmed_direction is None or confirmed_direction not in self._cache:
            return
        self._game = self._cache[confirmed_direction][1]


# --- the hypothesis-tracking predictor ----------------------------------------

class HypothesisPredictor:
    """Prior-weighted plurality prediction over the surviving permutations."""

    NUM_VARIANTS = len(PERMUTATIONS)

    def __init__(self, rule_model, prior=None):
        self.rule_model = rule_model
        if prior is None:
            self.prior = np.full(self.NUM_VARIANTS, 1.0 / self.NUM_VARIANTS)
        else:
            prior = np.asarray(prior, dtype=float)
            if prior.shape != (self.NUM_VARIANTS,):
                raise ValueError(f"prior must have {self.NUM_VARIANTS} entries, got {prior.shape}")
            total = float(prior.sum())
            if total <= 0:
                raise ValueError("prior must have positive total mass")
            self.prior = prior / total
        self._eliminate_fields = tuple(getattr(
            rule_model, "ELIMINATE_FIELDS", ("movement", "shape", "color", "rotation", "life_lost")))
        self._models_teleports = bool(getattr(rule_model, "MODELS_TELEPORTS", False))
        self.surviving = set(range(self.NUM_VARIANTS))
        self.steps_until_unique = None
        self._step_index = 0

    def reset_game(self, tier_seeds=None, tiers=None):
        self.surviving = set(range(self.NUM_VARIANTS))
        self.steps_until_unique = None
        self._step_index = 0
        reset_hook = getattr(self.rule_model, "reset_game", None)
        if reset_hook is not None:
            reset_hook(tier_seeds, tiers=tiers)

    def predict(self, step_fields):
        agent_action = int(step_fields["agent_action"])
        tallies = {"movement": {}, "shape": {}, "color": {}, "rotation": {}, "life_lost": {}}
        informative_weight = 0.0
        for h in sorted(self.surviving):
            direction = PERMUTATIONS[h][agent_action]
            pred = self.rule_model.predict_direction(step_fields, direction)
            if pred is None:
                continue
            weight = float(self.prior[h])
            informative_weight += weight
            for field, votes in tallies.items():
                votes[pred[field]] = votes.get(pred[field], 0.0) + weight
        abstained = informative_weight <= 0.0
        if abstained:
            result = {
                "movement": 0,
                "shape": int(step_fields["before_shape"]),
                "color": int(step_fields["before_color"]),
                "rotation": int(step_fields["before_rotation"]),
                "life_lost": False,
            }
        else:
            result = {field: max(votes.items(), key=lambda kv: kv[1])[0] for field, votes in tallies.items()}
        result["surviving"] = len(self.surviving)
        result["abstained"] = abstained
        return result

    def observe(self, step_fields):
        agent_action = int(step_fields["agent_action"])
        eliminate_fields = self._eliminate_fields
        if _is_teleport(step_fields) and not self._models_teleports:
            # The recorded after-state is a spawn teleport, not this rule
            # model's ordinary movement outcome; only life_lost (already
            # excluded from ELIMINATE_FIELDS for such a rule model) would be
            # meaningful, so this step carries no elimination signal at all.
            eliminate_fields = tuple(f for f in eliminate_fields if f == "life_lost")
        if not eliminate_fields:
            self._step_index += 1
            return {"abstained": True, "surviving": len(self.surviving)}
        observed = _observed(step_fields)
        cache = {}
        survivors = set()
        confirmed_direction = None
        any_informative = False
        for h in sorted(self.surviving):
            direction = PERMUTATIONS[h][agent_action]
            if direction not in cache:
                cache[direction] = self.rule_model.predict_direction(step_fields, direction)
            pred = cache[direction]
            if pred is None:
                survivors.add(h)  # abstain: this step cannot eliminate h
                continue
            any_informative = True
            if all(pred[field] == observed[field] for field in eliminate_fields):
                survivors.add(h)
                if confirmed_direction is None:
                    confirmed_direction = direction
        if not survivors:
            # Should be unreachable when the true hypothesis is still alive and
            # the rule model is exact; guard rather than crash on a surprise.
            survivors = set(self.surviving)
        self.surviving = survivors
        commit = getattr(self.rule_model, "commit", None)
        if commit is not None:
            commit(step_fields, confirmed_direction)
        self._step_index += 1
        if self.steps_until_unique is None and len(self.surviving) == 1:
            self.steps_until_unique = self._step_index
        return {"abstained": not any_informative, "surviving": len(self.surviving)}


def run_game(predictor, arrays):
    """Drive predict+observe across one game's arrays. Returns (pred_arrays, diagnostics)."""
    tier_seeds = arrays["tier_seeds"] if _has(arrays, "tier_seeds") else None
    tiers = arrays["tiers"] if _has(arrays, "tiers") else None
    predictor.reset_game(tier_seeds=tier_seeds, tiers=tiers)
    steps = len(arrays["agent_action"])
    movement, shape, color, rotation, life_lost = [], [], [], [], []
    surviving, abstained = [], []
    for t in range(steps):
        fields = step_fields_at(arrays, t)
        pred = predictor.predict(fields)
        movement.append(pred["movement"])
        shape.append(pred["shape"])
        color.append(pred["color"])
        rotation.append(pred["rotation"])
        life_lost.append(pred["life_lost"])
        surviving.append(pred["surviving"])
        abstained.append(pred["abstained"])
        predictor.observe(fields)
    pred_arrays = {
        "movement": np.asarray(movement, dtype=np.int64),
        "shape": np.asarray(shape, dtype=np.int64),
        "color": np.asarray(color, dtype=np.int64),
        "rotation": np.asarray(rotation, dtype=np.int64),
        "life_lost": np.asarray(life_lost, dtype=np.int64),
        "surviving": np.asarray(surviving, dtype=np.int64),
        "abstained": np.asarray(abstained, dtype=bool),
    }
    diagnostics = {
        "steps": steps,
        "steps_until_unique": predictor.steps_until_unique,
        "final_surviving": len(predictor.surviving),
        "survivor_history": surviving,
        "abstention_rate": float(np.mean(abstained)) if steps else 0.0,
    }
    return pred_arrays, diagnostics


# --- scoring: the shared contract, or a documented local fallback ------------

try:
    from .variant_metrics import (  # noqa: F401
        score_predictions as _shared_score_predictions,
        aggregate as _shared_aggregate,
        identity_baseline_predictions as _shared_identity_baseline_predictions,
        transition_mask as _shared_transition_mask,
    )
except ImportError:
    _shared_score_predictions = None
    _shared_aggregate = None
    _shared_identity_baseline_predictions = None
    _shared_transition_mask = None


def _fallback_transition_mask(arrays):
    if _shared_transition_mask is not None:
        return _shared_transition_mask(arrays)
    level_changed = np.asarray(arrays["level_changed"]).astype(bool)
    reset = np.asarray(arrays["reset"]).astype(bool)
    life_lost = np.asarray(arrays["life_lost"]).astype(bool)
    return ~(level_changed | reset | life_lost)


def _score_predictions_fallback(pred, arrays):
    """Only used when pebby.agent.variant_metrics is not importable.

    Not a stand-in for that module's real schema -- just enough to exercise
    this module (tests, the CLI tool) before it lands. Mirrors its
    steps-are-a-teleport exclusion for movement/shape/color/rotation.
    """
    steps = len(arrays["agent_action"])
    observed_movement = np.asarray([
        movement_class(arrays["before_player_x"][t], arrays["before_player_y"][t],
                        arrays["after_player_x"][t], arrays["after_player_y"][t])
        for t in range(steps)
    ])
    transition = _fallback_transition_mask(arrays)
    abstained = np.asarray(pred.get("abstained", np.zeros(steps, dtype=bool)))
    scored = transition & ~abstained
    life_scored = ~abstained

    def _acc(key, target, mask):
        if mask.sum() == 0:
            return None
        values = np.asarray(pred[key])
        return float((values[mask] == target[mask]).mean())

    return {
        "steps": int(steps),
        "scored_steps": int(scored.sum()),
        "abstention_rate": float(abstained.mean()) if steps else 0.0,
        "movement_accuracy": _acc("movement", observed_movement, scored),
        "shape_accuracy": _acc("shape", np.asarray(arrays["after_shape"]), scored),
        "color_accuracy": _acc("color", np.asarray(arrays["after_color"]), scored),
        "rotation_accuracy": _acc("rotation", np.asarray(arrays["after_rotation"]), scored),
        "life_lost_accuracy": _acc("life_lost", np.asarray(arrays["life_lost"]).astype(np.int64), life_scored),
    }


def _aggregate_fallback(reports):
    reports = [r for r in reports if r]
    if not reports:
        return {}
    total_steps = sum(r["steps"] for r in reports)
    total_scored = sum(r["scored_steps"] for r in reports)

    def _weighted(key):
        pairs = [(r[key], r["scored_steps"]) for r in reports if r.get(key) is not None]
        weight = sum(w for _, w in pairs)
        return (sum(v * w for v, w in pairs) / weight) if weight else None

    return {
        "games": len(reports),
        "steps": total_steps,
        "scored_steps": total_scored,
        "abstention_rate": (sum(r["abstention_rate"] * r["steps"] for r in reports) / total_steps)
                           if total_steps else 0.0,
        "movement_accuracy": _weighted("movement_accuracy"),
        "shape_accuracy": _weighted("shape_accuracy"),
        "color_accuracy": _weighted("color_accuracy"),
        "rotation_accuracy": _weighted("rotation_accuracy"),
        "life_lost_accuracy": _weighted("life_lost_accuracy"),
    }


def _identity_baseline_predictions_fallback(arrays, rule_model):
    """Guess as though variant_id were always 0: direction = agent_action, no elimination."""
    steps = len(arrays["agent_action"])
    movement, shape, color, rotation, life_lost, abstained = [], [], [], [], [], []
    for t in range(steps):
        fields = step_fields_at(arrays, t)
        pred = rule_model.predict_direction(fields, int(fields["agent_action"]))
        if pred is None:
            movement.append(0)
            shape.append(int(fields["before_shape"]))
            color.append(int(fields["before_color"]))
            rotation.append(int(fields["before_rotation"]))
            life_lost.append(False)
            abstained.append(True)
        else:
            movement.append(pred["movement"])
            shape.append(pred["shape"])
            color.append(pred["color"])
            rotation.append(pred["rotation"])
            life_lost.append(pred["life_lost"])
            abstained.append(False)
    return {
        "movement": np.asarray(movement, dtype=np.int64),
        "shape": np.asarray(shape, dtype=np.int64),
        "color": np.asarray(color, dtype=np.int64),
        "rotation": np.asarray(rotation, dtype=np.int64),
        "life_lost": np.asarray(life_lost, dtype=np.int64),
        "abstained": np.asarray(abstained, dtype=bool),
    }


def score_predictions(pred, arrays):
    if _shared_score_predictions is not None:
        return _shared_score_predictions(pred, arrays)
    return _score_predictions_fallback(pred, arrays)


def aggregate(reports):
    if _shared_aggregate is not None:
        return _shared_aggregate(reports)
    return _aggregate_fallback(reports)


def identity_baseline_predictions(arrays, rule_model=None):
    if _shared_identity_baseline_predictions is not None:
        return _shared_identity_baseline_predictions(arrays)
    return _identity_baseline_predictions_fallback(arrays, rule_model or LocalRuleModel())
