"""Compare a frozen world-policy prior with short latent-MPC beams.

This tool is deliberately generated-only.  It never imports the shipped level
bank, never asks the planner for an Oracle, and passes contextual optima only as
post-rollout reporting metadata.  The controller itself sees public frame
history, the learned predictor, and the learned value heads.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

# Support both ``python -m tools.evaluate_latent_mpc`` and direct execution
# from the repository root, matching the other bounded diagnostic tools.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby.agent import evaluate as ev
from pebby.agent.latent_mpc import MPCConfig, choose_action
from pebby.agent.model import load_checkpoint
from pebby.ls20.env import Ls20Scenario


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_indices(specs, per_difficulty=4, seed=42):
    """Select distinct generated bank rows, balanced over difficulties."""
    if per_difficulty < 1:
        raise ValueError("per_difficulty must be positive")
    rng = np.random.default_rng(seed)
    groups = [np.array([i for i, spec in enumerate(specs)
                        if spec.get("difficulty") == difficulty], dtype=np.int64)
              for difficulty in range(1, 6)]
    if any(len(group) < per_difficulty for group in groups):
        raise ValueError("bank lacks the requested generated difficulty coverage")
    chosen = np.concatenate([rng.choice(group, per_difficulty, replace=False)
                             for group in groups])
    return chosen.tolist()


class Controller:
    """Adapter exposing MPC decisions through the existing policy-history seam."""

    def __init__(self, model, config):
        self.model = model
        self._config = model.config()
        self.config_spec = config
        self.decisions = []

    def config(self):
        return self._config

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, frames, history_valid=None, previous_actions=None):
        decision = choose_action(self.model, frames, history_valid, previous_actions,
                                 self.config_spec)
        self.decisions.append(decision)
        return decision.action_scores[None]


def aggregate(runs, decisions, config):
    total = len(runs)
    return {
        "levels": total,
        "completed": sum(run["completed"] for run in runs),
        "completion_rate": sum(run["completed"] for run in runs) / total,
        "goals_cleared": sum(run["goals_cleared"] for run in runs),
        "goals_total": sum(run["goals_total"] for run in runs),
        "game_over": sum(run["ending"] == "game_over" for run in runs),
        "capped": sum(run["ending"] == "capped" for run in runs),
        "stuck": sum(run["ending"] == "stuck" for run in runs),
        "stalled_actions": sum(run["stalls"] for run in runs),
        "mean_actions": sum(run["actions"] for run in runs) / total,
        "mean_actions_vs_optimal": sum(run["actions_vs_optimal"] for run in runs)
        / total,
        "controller_decisions": len(decisions),
        "prior_action_rank_mean": sum(d.root_action_rank for d in decisions)
        / len(decisions),
        "proposed_override_fraction": sum(d.overrode_prior for d in decisions)
        / len(decisions),
        "horizon": config.horizon,
        "beam_width": config.beam_width,
        "prior_only": config.prior_only,
        "config": {key: getattr(config, key) for key in config.__dataclass_fields__},
    }


def run_variant(model, levels, optima, specs, selected, config, max_actions):
    runs = []
    decisions = []
    for index in selected:
        controller = Controller(model, config)
        level = levels[index]
        context = specs[index]["training_context_index"]
        env = Ls20Scenario(level, context)
        run = ev.rollout(controller, env, max_actions, torch.device("cpu"),
                         optima[index], on_stall="repeat")
        for key in ("seed", "difficulty", "training_context_index", "context_optimal_actions"):
            run[key] = specs[index][key]
        runs.append(run)
        decisions.extend(controller.decisions)
    return {"summary": aggregate(runs, decisions, config), "runs": runs,
            "decisions": [{"action": d.action, "prior_action": d.prior_action,
                           "root_action_rank": d.root_action_rank,
                           "overrode_prior": d.overrode_prior,
                           "best_score": d.best_score,
                           "beam_nodes": d.beam_nodes,
                           "expanded_depth": d.expanded_depth}
                          for d in decisions]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints/ls20-world-onpolicy-round2-b1024.epoch1.pt")
    parser.add_argument("--bank", type=Path,
                        default=ROOT / "data/ls20-verified-validation-monitor.jsonl")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/world-latent-mpc-20.json")
    parser.add_argument("--per-difficulty", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--variants", nargs="+",
                        choices=("prior", "horizon2_beam4", "horizon4_beam4"),
                        help="variants to run; default runs prior, horizon2, and horizon4")
    args = parser.parse_args(argv)
    if args.max_actions < 1:
        raise ValueError("max-actions must be positive")
    torch.set_num_threads(1)
    checkpoint_sha = digest(args.checkpoint)
    bank_sha = digest(args.bank)
    levels, optima, specs = ev.bank_levels(args.bank)
    selected = balanced_indices(specs, args.per_difficulty, args.selection_seed)
    selected_specs = [specs[index] for index in selected]
    model, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    model.eval()

    variants = {
        "prior": MPCConfig(prior_only=True),
        "horizon2_beam4": MPCConfig(horizon=2, beam_width=4),
        "horizon4_beam4": MPCConfig(horizon=4, beam_width=4),
    }
    if args.variants is not None:
        variants = {name: variants[name] for name in args.variants}
    results = {}
    for name, config in variants.items():
        results[name] = run_variant(model, levels, optima, specs, selected,
                                    config, args.max_actions)
        print(name, json.dumps(results[name]["summary"], sort_keys=True), flush=True)

    report = {
        "format": "pebby.latent-mpc-diagnostic.v1",
        "status": "complete",
        "source": "generated_only",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "parameters": model.parameter_count(),
        "bank": str(args.bank),
        "bank_sha256": bank_sha,
        "levels_total": len(specs),
        "levels_evaluated": len(selected),
        "selection_seed": args.selection_seed,
        "selected_indices": selected,
        "selected_specs": selected_specs,
        "max_actions": args.max_actions,
        "strict_protocol": "on_stall=repeat; deterministic argmax",
        "oracle_calls": 0,
        "inference_inputs": ["public frame history", "previous public actions",
                             "learned latent predictor", "learned value heads",
                             "learned current-policy prior"],
        "results": results,
        "limitations": [
            "Fixed generic score weights are diagnostic and were not tuned on this 20-level report.",
            "The model has deterministic latent dynamics and no epistemic uncertainty; short horizons limit hallucinated-win exploitation.",
            "Cached contextual optima are reporting fields only and are never passed to the controller.",
            "No official levels, layouts, routes, or planner calls are used.",
            "Completion is an engine rollout result; MPC does not receive hidden engine state.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
