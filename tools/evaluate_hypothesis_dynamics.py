"""Evaluate the explicit-inference predictor on collected LS20 rule-variant games.

Runs :class:`pebby.agent.hypothesis_dynamics.HypothesisPredictor` (either rule
model) over every game under ``--data-dir/games`` whose ``variant_id`` is in
``--variants``, scores it with ``pebby.agent.variant_metrics.score_predictions``
(falling back to ``hypothesis_dynamics``'s local equivalent if that module is
absent), and reports, on top of the usual field accuracies:

* surviving-hypothesis count by step bin (how fast the hypothesis set shrinks);
* steps until a single hypothesis survives (per game, then summarised);
* abstention rate (only ever nonzero for ``--rule local``: the engine rule
  model never abstains);
* the identity-baseline score for the same games (``--prior`` never applies to
  this baseline; it is the "assume variant 0 always" hand rule, exactly the
  comparison this predictor needs to beat on non-identity variants).

``--prior manifest`` builds a 24-entry prior from the empirical variant
frequency across every game recorded in ``--data-dir/manifest.json`` (as
written by ``tools/collect_variant_games.py``), independent of which
``--variants`` are actually being evaluated here -- this is what lets a run
demonstrate that a prior fit on one distribution is close to useless when
evaluated on held-out permutations it under-weights.

Example::

    PYTHONPATH=. uv run python tools/evaluate_hypothesis_dynamics.py \
        --data-dir /tmp/variant-pilot --variants 0 5 --rule local --out report.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from pebby.agent.hypothesis_dynamics import (
    EngineRuleModel,
    HypothesisPredictor,
    LocalRuleModel,
    aggregate,
    identity_baseline_predictions,
    run_game,
    score_predictions,
)

RULE_MODELS = {"local": LocalRuleModel, "engine": EngineRuleModel}
NUM_VARIANTS = 24
# [lo, hi) step-index bins for the surviving-hypothesis-count report.
STEP_BIN_EDGES = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 10 ** 9))


def _step_bin(step_index):
    for lo, hi in STEP_BIN_EDGES:
        if lo <= step_index < hi:
            return (lo, hi)
    return STEP_BIN_EDGES[-1]


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


def empirical_prior_from_manifest(data_dir):
    """24-entry variant frequency from every game in ``manifest.json``, normalised."""
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"--prior manifest needs {manifest_path}, which does not exist")
    manifest = json.loads(manifest_path.read_text())
    games = manifest.get("games")
    if not games:
        raise SystemExit(f"{manifest_path} has no 'games' list to build an empirical prior from")
    counts = Counter(int(game["variant_id"]) for game in games)
    prior = np.ones(NUM_VARIANTS)  # Laplace smoothing: an unseen variant is not zero-probability.
    for variant_id, count in counts.items():
        prior[variant_id] += count
    return prior / prior.sum()


def evaluate_one(path, variant_id, rule, prior):
    arrays = load_arrays(path)
    rule_model = RULE_MODELS[rule]()
    predictor = HypothesisPredictor(rule_model, prior=prior)
    started = time.perf_counter()
    pred_arrays, diagnostics = run_game(predictor, arrays)
    elapsed = time.perf_counter() - started
    report = score_predictions(pred_arrays, arrays)
    baseline_pred = identity_baseline_predictions(arrays)
    baseline_report = score_predictions(baseline_pred, arrays)
    engine_steps = getattr(rule_model, "engine_steps", 0)
    return dict(
        game_id=path.stem, variant_id=variant_id, report=report, baseline_report=baseline_report,
        diagnostics=diagnostics, elapsed=elapsed, engine_steps=engine_steps,
        true_variant_survives=variant_id in predictor.surviving,
        final_surviving=len(predictor.surviving),
    )


def summarise(results):
    steps_until_unique = [r["diagnostics"]["steps_until_unique"] for r in results]
    reached = [v for v in steps_until_unique if v is not None]
    abstention_rates = [r["diagnostics"]["abstention_rate"] for r in results]
    survivor_bins = defaultdict(list)
    for r in results:
        for step_index, count in enumerate(r["diagnostics"]["survivor_history"]):
            survivor_bins[_step_bin(step_index)].append(count)
    survivor_by_bin = {
        f"[{lo},{hi})": {"mean_surviving": float(np.mean(counts)), "n_steps": len(counts)}
        for (lo, hi), counts in sorted(survivor_bins.items())
    }
    total_engine_steps = sum(r["engine_steps"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)
    return {
        "games": len(results),
        "true_variant_recovered_rate": float(np.mean([r["true_variant_survives"] for r in results])),
        "final_surviving_mean": float(np.mean([r["final_surviving"] for r in results])),
        "steps_until_unique": {
            "games": len(steps_until_unique), "reached": len(reached),
            "fraction_reached": (len(reached) / len(steps_until_unique)) if steps_until_unique else None,
            "mean": (float(np.mean(reached)) if reached else None),
            "median": (float(np.median(reached)) if reached else None),
        },
        "abstention_rate_mean": float(np.mean(abstention_rates)) if abstention_rates else 0.0,
        "surviving_by_step_bin": survivor_by_bin,
        "engine_steps_per_second": (total_engine_steps / total_elapsed)
                                   if total_elapsed > 0 and total_engine_steps > 0 else None,
        "aggregate": aggregate([r["report"] for r in results]),
        "identity_baseline_aggregate": aggregate([r["baseline_report"] for r in results]),
    }


def _field_accuracy(aggregated, field):
    """Read one field's accuracy from either the real variant_metrics.aggregate
    schema (``aggregated["fields"][field]["accuracy"]``) or this module's flat
    fallback schema (``aggregated[f"{field}_accuracy"]``)."""
    fields = aggregated.get("fields")
    if fields is not None:
        return fields.get(field, {}).get("accuracy")
    return aggregated.get(f"{field}_accuracy")


def _fmt_pct(value):
    return f"{value:5.1%}" if value is not None else "    -"


def print_table(args, summary):
    agg = summary["aggregate"]
    baseline = summary["identity_baseline_aggregate"]
    print(f"\nhypothesis-dynamics report: variants={args.variants} rule={args.rule} prior={args.prior}")
    print(f"  games evaluated              {summary['games']}")
    print(f"  true variant recovered       {summary['true_variant_recovered_rate']:.1%}")
    print(f"  final surviving (mean)       {summary['final_surviving_mean']:.2f} / {NUM_VARIANTS}")
    stu = summary["steps_until_unique"]
    if stu["fraction_reached"] is not None:
        print(f"  steps until unique           reached {stu['fraction_reached']:.1%}"
              f"  mean {stu['mean']:.1f}  median {stu['median']:.1f}")
    else:
        print("  steps until unique           reached 0.0%")
    print(f"  abstention rate (mean)       {summary['abstention_rate_mean']:.1%}")
    if summary["engine_steps_per_second"] is not None:
        print(f"  engine clone+step() rate     {summary['engine_steps_per_second']:.1f} steps/sec")
    print()
    print(f"  {'field':<10}{'predictor':>12}{'identity baseline':>20}")
    for field in ("movement", "shape", "color", "rotation", "life_lost"):
        print(f"  {field:<10}{_fmt_pct(_field_accuracy(agg, field)):>12}{_fmt_pct(_field_accuracy(baseline, field)):>20}")
    print()
    print("  surviving hypotheses by step bin (mean count, n_steps):")
    for bin_label, cell in summary["surviving_by_step_bin"].items():
        print(f"    {bin_label:<12} mean={cell['mean_surviving']:5.2f}  n={cell['n_steps']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--variants", type=int, nargs="+", required=True)
    parser.add_argument("--rule", choices=("local", "engine"), default="local")
    parser.add_argument("--prior", choices=("none", "manifest"), default="none")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if any(not 0 <= v < NUM_VARIANTS for v in args.variants):
        parser.error(f"variants must be in 0..{NUM_VARIANTS - 1}")
    return args


def main(argv=None):
    args = parse_args(argv)
    games = find_games(args.data_dir, args.variants)
    if not games:
        raise SystemExit(f"no games under {args.data_dir}/games matched variants {args.variants}")
    prior = empirical_prior_from_manifest(args.data_dir) if args.prior == "manifest" else None

    results = [evaluate_one(path, variant_id, args.rule, prior) for path, variant_id in games]
    summary = summarise(results)
    print_table(args, summary)

    if args.out:
        payload = {
            "data_dir": str(args.data_dir), "variants": args.variants, "rule": args.rule, "prior": args.prior,
            "summary": summary,
            "games": [{"game_id": r["game_id"], "variant_id": r["variant_id"],
                       "true_variant_survives": r["true_variant_survives"],
                       "final_surviving": r["final_surviving"],
                       "steps_until_unique": r["diagnostics"]["steps_until_unique"],
                       "movement_accuracy": r["report"].get("movement_accuracy")} for r in results],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
