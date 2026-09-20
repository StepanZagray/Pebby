"""Shared metrics for the LS20 rule-variant (action-permutation) transition-prediction experiment.

Every predictor of the "next factored state" -- the in-context transformer, the memoryless
control, the hand-written identity baseline, or anything else -- is scored through
:func:`score_predictions` so that all reports have exactly the same shape and can be pooled
with :func:`aggregate`.

Data contract (one NPZ per game, arrays of length ``T``):
``level_index, step_in_level, before_*, neighbours [T, 4], agent_action, engine_action,
after_*, life_lost, level_changed, reset, won, finished`` plus scalar labels
``game_id, variant_id, action_map, tier_seeds, truncated, levels_completed``.
Only public arrays are read here; ``engine_action``, ``action_map`` and ``variant_id`` are
labels and are never consulted by any function in this module.

Targets (the five factored fields predicted for every step)
-----------------------------------------------------------
``movement``   5-way: 0 stayed, 1 moved to the engine-up cell, 2 down, 3 left, 4 right,
               computed from (after_x - before_x, after_y - before_y).
``shape``      after_shape, 6-way.      ``color``  after_color, 4-way.
``rotation``   after_rotation, 4-way.   ``life_lost``  2-way.

Steps flagged ``level_changed``, ``reset`` or ``life_lost`` are *not* scored for
movement/shape/color/rotation: their after-state is the start of the next level or the restored
level (a life loss puts the player back on the level start), a teleport rather than the outcome
of the action.  ``life_lost`` itself is scored on every step.  A step's *joint* prediction
is correct when every scored field on that step is correct.
"""
from __future__ import annotations

import numpy as np

FIELDS = ("movement", "shape", "color", "rotation", "life_lost")
FIELD_SIZES = {"movement": 5, "shape": 6, "color": 4, "rotation": 4, "life_lost": 2}
# (dx, dy) for movement classes 0..4: stayed, up, down, left, right -- engine direction order.
MOVEMENT_DELTAS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
GLYPH_DOMAINS = {"shape": 6, "color": 4, "rotation": 4}

# Neighbour tile classes in ``neighbours[:, direction]`` (direction order up, down, left, right).
(TILE_FREE, TILE_WALL, TILE_CYCLER_SHAPE, TILE_CYCLER_COLOR, TILE_CYCLER_ROTATION, TILE_REFILL,
 TILE_LAUNCHER, TILE_GOAL, TILE_RAIL, TILE_OUTSIDE) = range(10)
BLOCKING_TILES = (TILE_WALL, TILE_OUTSIDE)
CYCLER_FOR_FIELD = {"shape": TILE_CYCLER_SHAPE, "color": TILE_CYCLER_COLOR,
                    "rotation": TILE_CYCLER_ROTATION}

STABLE_RUN = 20          # consecutive correct scored steps that define "stable"
FINE_BIN = 10            # step-index bins of 10 up to FINE_LIMIT ...
FINE_LIMIT = 200
COARSE_BIN = 50          # ... then bins of 50


def as_int(arrays, key):
    return np.asarray(arrays[key]).astype(np.int64)


def as_bool(arrays, key):
    return np.asarray(arrays[key]).astype(bool)


def movement_class(before_x, before_y, after_x, after_y):
    """5-way movement class from before/after coordinates; non-unit jumps map to 0 (stayed)."""
    dx = np.asarray(after_x).astype(np.int64) - np.asarray(before_x).astype(np.int64)
    dy = np.asarray(after_y).astype(np.int64) - np.asarray(before_y).astype(np.int64)
    classes = np.zeros(dx.shape, dtype=np.int64)
    for label, (ddx, ddy) in enumerate(MOVEMENT_DELTAS[1:], start=1):
        classes[(dx == ddx) & (dy == ddy)] = label
    return classes


def targets_from_arrays(arrays):
    """The five per-step target arrays (int64, length T) built from public fields only."""
    return {
        "movement": movement_class(arrays["before_player_x"], arrays["before_player_y"],
                                   arrays["after_player_x"], arrays["after_player_y"]),
        "shape": as_int(arrays, "after_shape"),
        "color": as_int(arrays, "after_color"),
        "rotation": as_int(arrays, "after_rotation"),
        "life_lost": as_bool(arrays, "life_lost").astype(np.int64),
    }


def transition_mask(arrays):
    """True on steps whose after-state is the action's outcome: no level change, reset or life
    loss (all three restore or replace the level, so the after position is a teleport)."""
    return ~(as_bool(arrays, "level_changed") | as_bool(arrays, "reset") | as_bool(arrays, "life_lost"))


def target_masks(arrays):
    """Per-field boolean masks (length T): which steps are scored for each field."""
    transition = transition_mask(arrays)
    return {"movement": transition, "shape": transition, "color": transition,
            "rotation": transition, "life_lost": np.ones_like(transition)}


def bin_edges(steps):
    """[lo, hi) step-index bins: width 10 up to 200, then width 50; a prefix of a fixed scheme,
    so bin ``i`` means the same range in every game and reports can be pooled by index."""
    edges = []
    lo = 0
    while lo < steps:
        width = FINE_BIN if lo < FINE_LIMIT else COARSE_BIN
        edges.append((lo, lo + width))
        lo += width
    return edges


def _cell(correct, count):
    return {"correct": int(correct), "count": int(count),
            "accuracy": (float(correct) / float(count)) if count else None}


def steps_to_stable(correct, scored=None, run=STABLE_RUN):
    """Index of the first step opening a run of ``run`` consecutive correct *scored* steps.

    Unscored steps (``scored`` False) neither break nor extend the run.  Returns ``None`` when
    no such run exists.  The value is the step index (0-based) of the first step in the run,
    i.e. from this step on the predictor made ``run`` correct predictions in a row.
    """
    correct = np.asarray(correct, dtype=bool)
    scored = np.ones_like(correct) if scored is None else np.asarray(scored, dtype=bool)
    indices = np.flatnonzero(scored)
    if len(indices) < run:
        return None
    hits = correct[indices]
    streak = 0
    for position, hit in enumerate(hits):
        streak = streak + 1 if hit else 0
        if streak >= run:
            return int(indices[position - run + 1])
    return None


def _curve(correct, scored, edges):
    cells = []
    for lo, hi in edges:
        window = scored[lo:hi]
        cells.append(_cell(np.count_nonzero(correct[lo:hi] & window), np.count_nonzero(window)))
    return cells


def score_predictions(pred, arrays):
    """Score integer predictions per field against a game's arrays.

    ``pred`` maps each name in :data:`FIELDS` to an int array of length T.  Returns::

        {"steps": T, "scored_steps": n_transition_steps,
         "fields": {field: {"correct", "count", "accuracy"}},
         "joint": {"correct", "count", "accuracy"},
         "movement_accuracy": float | None,
         "curve": {"bins": [[lo, hi], ...],
                   "joint": [{"correct", "count", "accuracy"}, ...],
                   "movement": [{"correct", "count", "accuracy"}, ...]},
         "steps_to_stable": int | None,            # joint, 20 consecutive correct scored steps
         "movement_steps_to_stable": int | None}   # movement field only
    """
    targets = targets_from_arrays(arrays)
    masks = target_masks(arrays)
    steps = len(targets["movement"])
    joint_correct = np.ones(steps, dtype=bool)
    joint_scored = np.zeros(steps, dtype=bool)
    fields = {}
    per_field_correct = {}
    for field in FIELDS:
        guess = np.asarray(pred[field]).astype(np.int64).reshape(-1)
        if guess.shape[0] != steps:
            raise ValueError(f"prediction for {field!r} has length {guess.shape[0]}, expected {steps}")
        hit = guess == targets[field]
        per_field_correct[field] = hit
        scored = masks[field]
        fields[field] = _cell(np.count_nonzero(hit & scored), np.count_nonzero(scored))
        joint_correct &= hit | ~scored
        joint_scored |= scored
    edges = bin_edges(steps)
    movement_hit, movement_scored = per_field_correct["movement"], masks["movement"]
    return {
        "steps": int(steps),
        "scored_steps": int(np.count_nonzero(masks["movement"])),
        "fields": fields,
        "joint": _cell(np.count_nonzero(joint_correct & joint_scored), np.count_nonzero(joint_scored)),
        "movement_accuracy": fields["movement"]["accuracy"],
        "curve": {"bins": [[int(lo), int(hi)] for lo, hi in edges],
                  "joint": _curve(joint_correct, joint_scored, edges),
                  "movement": _curve(movement_hit, movement_scored, edges)},
        "steps_to_stable": steps_to_stable(joint_correct, joint_scored),
        "movement_steps_to_stable": steps_to_stable(movement_hit, movement_scored),
    }


def _summarise_stable(values):
    reached = [v for v in values if v is not None]
    return {"values": values, "games": len(values), "reached": len(reached),
            "fraction_reached": (len(reached) / len(values)) if values else None,
            "median": (float(np.median(reached)) if reached else None),
            "mean": (float(np.mean(reached)) if reached else None)}


def aggregate(reports):
    """Pool per-game reports from :func:`score_predictions`.

    Accuracies are pooled (sum correct / sum count) and also averaged per game; curves are
    pooled per bin index with the number of games contributing to each bin.
    """
    reports = list(reports)
    out = {"games": len(reports), "fields": {}, "curve": {"bins": [], "joint": [], "movement": []}}
    for field in FIELDS:
        correct = sum(r["fields"][field]["correct"] for r in reports)
        count = sum(r["fields"][field]["count"] for r in reports)
        per_game = [r["fields"][field]["accuracy"] for r in reports
                    if r["fields"][field]["accuracy"] is not None]
        out["fields"][field] = {**_cell(correct, count),
                                "mean_game_accuracy": float(np.mean(per_game)) if per_game else None}
    joint_correct = sum(r["joint"]["correct"] for r in reports)
    joint_count = sum(r["joint"]["count"] for r in reports)
    joint_games = [r["joint"]["accuracy"] for r in reports if r["joint"]["accuracy"] is not None]
    out["joint"] = {**_cell(joint_correct, joint_count),
                    "mean_game_accuracy": float(np.mean(joint_games)) if joint_games else None}
    out["movement_accuracy"] = out["fields"]["movement"]["accuracy"]
    longest = max((len(r["curve"]["bins"]) for r in reports), default=0)
    if longest:
        out["curve"]["bins"] = bin_edges(max(r["steps"] for r in reports))
        out["curve"]["bins"] = [[int(lo), int(hi)] for lo, hi in out["curve"]["bins"]]
    for key in ("joint", "movement"):
        for index in range(longest):
            cells = [r["curve"][key][index] for r in reports if index < len(r["curve"][key])]
            contributing = [c for c in cells if c["count"]]
            out["curve"][key].append({**_cell(sum(c["correct"] for c in cells),
                                              sum(c["count"] for c in cells)),
                                      "games": len(contributing)})
    out["steps_to_stable"] = _summarise_stable([r["steps_to_stable"] for r in reports])
    out["movement_steps_to_stable"] = _summarise_stable(
        [r["movement_steps_to_stable"] for r in reports])
    out["mean_steps"] = float(np.mean([r["steps"] for r in reports])) if reports else None
    return out


def identity_baseline_predictions(arrays):
    """The "memorised standard controls" baseline: a hand rule, not an engine.

    It assumes the action map is the identity (agent action i == engine direction i) and
    applies the plain LS20 movement rule: the player moves into the neighbouring cell in that
    direction unless the cell is a wall or outside the grid, in which case it stays.  Glyph
    fields are predicted unchanged unless that neighbour is the matching cycler tile, in which
    case the field advances by one modulo its domain.  ``life_lost`` is always predicted 0 (no
    step budget rule is modelled).  On non-identity variants its movement accuracy shows how
    much a predictor that memorised the standard controls would get right by luck.
    """
    action = as_int(arrays, "agent_action")
    neighbours = np.asarray(arrays["neighbours"]).astype(np.int64)
    steps = action.shape[0]
    facing = neighbours[np.arange(steps), np.clip(action, 0, 3)]
    blocked = np.isin(facing, BLOCKING_TILES)
    pred = {"movement": np.where(blocked, 0, action + 1).astype(np.int64),
            "life_lost": np.zeros(steps, dtype=np.int64)}
    for field, domain in GLYPH_DOMAINS.items():
        before = as_int(arrays, f"before_{field}")
        advance = facing == CYCLER_FOR_FIELD[field]
        pred[field] = np.where(advance, (before + 1) % domain, before).astype(np.int64)
    return pred
