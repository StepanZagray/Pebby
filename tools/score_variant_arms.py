"""Score every LS20 rule-variant transition predictor on the same set of games.

Five arms are compared: the three trained checkpoints from
``tools/train_incontext_dynamics.py`` (``transformer``, ``memoryless``,
``transformer-no-prev`` -- names and paths come from ``--models``), the
hand-written ``identity_baseline_predictions`` control from
``pebby.agent.variant_metrics``, and the explicit-inference
``HypothesisPredictor`` driven by ``LocalRuleModel`` with a uniform prior
(:mod:`pebby.agent.hypothesis_dynamics`).

Why this script exists: in ``data/variant-games-v1/all`` 90% of recorded
actions are oracle actions that move toward the goal, so a step's outcome is
largely predictable from the board without knowing the hidden action-to-
direction permutation at all -- a predictor with no idea what an action does
can still score well by exploiting that shortcut (this is why the
memoryless control, which cannot infer the mapping in-context, still landed
around 0.80 held-out movement accuracy). ``--only-random-steps`` restricts
every accuracy figure to steps whose ``action_source`` was 0 (a uniformly
random action), which is the honest test of "does this arm know what the
action does" rather than "can it guess the board's incentive".

Every field accuracy, joint accuracy and step-bin curve in this script's
output uses ONE mask per run: ``transition_mask`` (excludes steps whose
after-state is a teleport -- level change, reset or life loss, see
``pebby.agent.variant_metrics``) intersected with ``action_source == 0`` when
``--only-random-steps`` is given. This differs from
``variant_metrics.score_predictions``, which always scores ``life_lost`` on
every step regardless of transition_mask; here ``life_lost`` is restricted
like every other field so that a run with ``--only-random-steps`` is scoring
"only random steps" uniformly across all five fields.

Independent of that flag, every arm's report also includes an in-context
learning curve: accuracy on RANDOM steps (transition_mask & action_source==0,
computed regardless of ``--only-random-steps``) within the first 10 steps of
each game versus after step 30. A predictor that is actually inferring the
permutation from the game's own early steps should improve from "early" to
"late"; a predictor with no such mechanism (the identity baseline, or a
memoryless control) should not.

Example::

    PYTHONPATH=. uv run python tools/score_variant_arms.py \\
        --data-dir data/variant-games-v1/all --variants 4 9 13 18 21 23 \\
        --only-random-steps \\
        --models transformer=artifacts/variant-inference-v1/transformer/best.pt \\
                 memoryless=artifacts/variant-inference-v1/memoryless/best.pt \\
                 transformer-no-prev=artifacts/variant-inference-v1/transformer-no-prev/best.pt \\
        --out artifacts/variant-inference-v1/score-random-steps-mixed.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pebby.agent import incontext_dynamics as icd
from pebby.agent.hypothesis_dynamics import HypothesisPredictor, LocalRuleModel, run_game
from pebby.agent.variant_metrics import (FIELDS, _cell, _curve, bin_edges, identity_baseline_predictions,
                                         steps_to_stable, targets_from_arrays, transition_mask)

NUM_VARIANTS = 24
EARLY_LIMIT = 10   # steps [0, EARLY_LIMIT) of a game
LATE_START = 30    # steps [LATE_START, inf) of a game


# ------------------------------------------------------------------ data
def find_games(data_dir, variants):
    """Every ``games/*.npz`` whose scalar ``variant_id`` is in ``variants``."""
    games_dir = Path(data_dir) / "games"
    wanted = set(variants)
    selected = []
    for path in sorted(games_dir.glob("*.npz")):
        with np.load(path) as data:
            variant_id = int(data["variant_id"])
        if variant_id in wanted:
            selected.append((path, variant_id))
    return selected


def load_arrays(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def random_step_mask(arrays):
    """``transition_mask`` steps whose action was drawn uniformly at random
    (``action_source == 0``). Falls back to no random steps if the game has
    no ``action_source`` column (not expected for this experiment's data)."""
    mask = transition_mask(arrays)
    if "action_source" in arrays:
        mask = mask & (np.asarray(arrays["action_source"]).astype(np.int64) == 0)
    else:
        mask = np.zeros_like(mask)
    return mask


def scoring_mask(arrays, only_random_steps):
    mask = transition_mask(arrays)
    if only_random_steps:
        mask = mask & (np.asarray(arrays["action_source"]).astype(np.int64) == 0)
    return mask


# ------------------------------------------------------------------ scoring
def _field_hits(pred, arrays):
    targets = targets_from_arrays(arrays)
    hits = {}
    for field in FIELDS:
        guess = np.asarray(pred[field]).astype(np.int64).reshape(-1)
        if guess.shape[0] != targets[field].shape[0]:
            raise ValueError(f"prediction for {field!r} has length {guess.shape[0]}, "
                              f"expected {targets[field].shape[0]}")
        hits[field] = guess == targets[field]
    return hits


def _joint_hit(hits, mask):
    """A step is joint-correct if every field is correct, restricted to ``mask``
    uniformly across fields (unlike ``variant_metrics.score_predictions``, which
    always scores ``life_lost`` on every step -- see module docstring)."""
    joint = np.ones_like(mask)
    for field in FIELDS:
        joint &= hits[field] | ~mask
    return joint


def _summarise_stable(values):
    reached = [v for v in values if v is not None]
    return {"games": len(values), "reached": len(reached),
            "fraction_reached": (len(reached) / len(values)) if values else None,
            "median": (float(np.median(reached)) if reached else None),
            "mean": (float(np.mean(reached)) if reached else None)}


def score_game_arm(pred, arrays, mask, random_mask):
    """One arm's report for one game: field/joint accuracy and curve restricted
    to ``mask``, plus early-vs-late random-step accuracy restricted to
    ``random_mask`` regardless of ``mask`` (the in-context learning curve)."""
    hits = _field_hits(pred, arrays)
    joint = _joint_hit(hits, mask)
    steps = len(mask)
    edges = bin_edges(steps)
    fields = {f: _cell(int(np.count_nonzero(hits[f] & mask)), int(np.count_nonzero(mask))) for f in FIELDS}
    joint_cell = _cell(int(np.count_nonzero(joint & mask)), int(np.count_nonzero(mask)))
    idx = np.arange(steps)
    early = random_mask & (idx < EARLY_LIMIT)
    late = random_mask & (idx >= LATE_START)
    return {
        "steps": int(steps),
        "scored_steps": int(np.count_nonzero(mask)),
        "fields": fields,
        "joint": joint_cell,
        "movement_accuracy": fields["movement"]["accuracy"],
        "curve": {"bins": [[int(lo), int(hi)] for lo, hi in edges],
                  "joint": _curve(joint, mask, edges),
                  "movement": _curve(hits["movement"], mask, edges)},
        "steps_to_stable": steps_to_stable(joint, mask),
        "movement_steps_to_stable": steps_to_stable(hits["movement"], mask),
        "early_random": {
            "movement": _cell(int(np.count_nonzero(hits["movement"] & early)), int(np.count_nonzero(early))),
            "joint": _cell(int(np.count_nonzero(joint & early)), int(np.count_nonzero(early))),
        },
        "late_random": {
            "movement": _cell(int(np.count_nonzero(hits["movement"] & late)), int(np.count_nonzero(late))),
            "joint": _cell(int(np.count_nonzero(joint & late)), int(np.count_nonzero(late))),
        },
    }


def aggregate_arm(game_reports):
    """Pool per-game :func:`score_game_arm` reports for one arm across games."""
    games = len(game_reports)
    fields = {}
    for f in FIELDS:
        correct = sum(r["fields"][f]["correct"] for r in game_reports)
        count = sum(r["fields"][f]["count"] for r in game_reports)
        fields[f] = _cell(correct, count)
    joint_correct = sum(r["joint"]["correct"] for r in game_reports)
    joint_count = sum(r["joint"]["count"] for r in game_reports)
    longest = max((len(r["curve"]["bins"]) for r in game_reports), default=0)
    bins = []
    if longest:
        bins = [[int(lo), int(hi)] for lo, hi in bin_edges(max(r["steps"] for r in game_reports))]
    curve = {"bins": bins, "joint": [], "movement": []}
    for key in ("joint", "movement"):
        for index in range(longest):
            cells = [r["curve"][key][index] for r in game_reports if index < len(r["curve"][key])]
            curve[key].append(_cell(sum(c["correct"] for c in cells), sum(c["count"] for c in cells)))
    out = {
        "games": games,
        "fields": fields,
        "joint": _cell(joint_correct, joint_count),
        "movement_accuracy": fields["movement"]["accuracy"],
        "curve": curve,
        "steps_to_stable": _summarise_stable([r["steps_to_stable"] for r in game_reports]),
        "movement_steps_to_stable": _summarise_stable([r["movement_steps_to_stable"] for r in game_reports]),
        "mean_steps": float(np.mean([r["steps"] for r in game_reports])) if game_reports else None,
    }
    for window in ("early_random", "late_random"):
        out[window] = {}
        for metric in ("movement", "joint"):
            correct = sum(r[window][metric]["correct"] for r in game_reports)
            count = sum(r[window][metric]["count"] for r in game_reports)
            out[window][metric] = _cell(correct, count)
    return out


# ------------------------------------------------------------------ arms
def predict_checkpoint(model, arrays, device):
    features = icd.featurize(arrays)
    return icd.predict_game(model, features, device=device)


def predict_hypothesis(arrays):
    predictor = HypothesisPredictor(LocalRuleModel(), prior=None)
    pred_arrays, _diagnostics = run_game(predictor, arrays)
    return pred_arrays


def run_arms(games, model_paths, device, only_random_steps):
    """Score every arm on every game. Returns {arm_name: {"per_game": [...], "aggregate": {...},
    "elapsed": float}}."""
    checkpoints = {}
    for name, path in model_paths.items():
        model, _payload = icd.load_checkpoint(path, device=device)
        checkpoints[name] = model

    arm_names = list(model_paths.keys()) + ["identity", "hypothesis-local"]
    per_game = {name: [] for name in arm_names}
    elapsed = {name: 0.0 for name in arm_names}
    scored_steps_total = {name: 0 for name in arm_names}

    for path, variant_id in games:
        arrays = load_arrays(path)
        mask = scoring_mask(arrays, only_random_steps)
        random_mask = random_step_mask(arrays)

        for name, model in checkpoints.items():
            started = time.perf_counter()
            pred = predict_checkpoint(model, arrays, device)
            elapsed[name] += time.perf_counter() - started
            report = score_game_arm(pred, arrays, mask, random_mask)
            report["game_id"] = path.stem
            report["variant_id"] = variant_id
            per_game[name].append(report)
            scored_steps_total[name] += report["scored_steps"]

        started = time.perf_counter()
        pred = identity_baseline_predictions(arrays)
        elapsed["identity"] += time.perf_counter() - started
        report = score_game_arm(pred, arrays, mask, random_mask)
        report["game_id"] = path.stem
        report["variant_id"] = variant_id
        per_game["identity"].append(report)
        scored_steps_total["identity"] += report["scored_steps"]

        started = time.perf_counter()
        pred = predict_hypothesis(arrays)
        elapsed["hypothesis-local"] += time.perf_counter() - started
        report = score_game_arm(pred, arrays, mask, random_mask)
        report["game_id"] = path.stem
        report["variant_id"] = variant_id
        per_game["hypothesis-local"].append(report)
        scored_steps_total["hypothesis-local"] += report["scored_steps"]

    out = {}
    for name in arm_names:
        out[name] = {"aggregate": aggregate_arm(per_game[name]), "per_game": per_game[name],
                     "elapsed_seconds": elapsed[name], "scored_steps": scored_steps_total[name]}
    return out


# ------------------------------------------------------------------ reporting
def _fmt_pct(value):
    return f"{value:6.1%}" if value is not None else "     -"


def _fmt_int(value):
    return f"{value:6d}" if value is not None else "     -"


def print_table(args, results):
    print(f"\nvariant-arm report: data_dir={args.data_dir} variants={args.variants} "
          f"only_random_steps={args.only_random_steps}")
    header = f"  {'arm':<20}{'movement':>10}{'joint':>10}{'scored':>10}{'early<10':>10}{'late>=30':>10}{'stable_med':>12}"
    print(header)
    for name, arm in results.items():
        agg = arm["aggregate"]
        early = agg["early_random"]["movement"]["accuracy"]
        late = agg["late_random"]["movement"]["accuracy"]
        stable = agg["movement_steps_to_stable"]["median"]
        print(f"  {name:<20}{_fmt_pct(agg['movement_accuracy']):>10}{_fmt_pct(agg['joint']['accuracy']):>10}"
              f"{_fmt_int(arm['scored_steps']):>10}{_fmt_pct(early):>10}{_fmt_pct(late):>10}"
              f"{(f'{stable:8.1f}' if stable is not None else '       -'):>12}")
    print()
    print("  per-field accuracy (movement / shape / color / rotation / life_lost):")
    for name, arm in results.items():
        agg = arm["aggregate"]
        cells = "  ".join(_fmt_pct(agg["fields"][f]["accuracy"]) for f in FIELDS)
        print(f"  {name:<20}{cells}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--variants", type=int, nargs="+", required=True)
    parser.add_argument("--only-random-steps", action="store_true",
                        help="restrict every accuracy/curve figure to transition_mask & action_source==0")
    parser.add_argument("--models", nargs="+", required=True, metavar="name=path",
                        help="e.g. transformer=artifacts/.../best.pt memoryless=artifacts/.../best.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU thread count")
    args = parser.parse_args(argv)
    if any(not 0 <= v < NUM_VARIANTS for v in args.variants):
        parser.error(f"variants must be in 0..{NUM_VARIANTS - 1}")
    model_paths = {}
    for item in args.models:
        if "=" not in item:
            parser.error(f"--models entries must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        model_paths[name] = Path(path)
    args.model_paths = model_paths
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    games = find_games(args.data_dir, args.variants)
    if not games:
        raise SystemExit(f"no games under {args.data_dir}/games matched variants {args.variants}")

    results = run_arms(games, args.model_paths, args.device, args.only_random_steps)
    print_table(args, results)

    payload = {
        "data_dir": str(args.data_dir), "variants": args.variants,
        "only_random_steps": args.only_random_steps, "models": {k: str(v) for k, v in args.model_paths.items()},
        "games": len(games), "arms": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
