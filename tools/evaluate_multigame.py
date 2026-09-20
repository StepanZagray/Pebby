#!/usr/bin/env python3
"""Run the frozen public policy on official games under the phased protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby.agent.multigame_evaluation import (  # noqa: E402
    OfficialEvaluationConfig,
    evaluate_official,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("training-families", "heldout"), required=True)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-actions-per-level", type=int, default=256)
    parser.add_argument("--max-game-actions", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    phase = "training_families" if args.phase == "training-families" else "heldout"
    report = evaluate_official(
        args.checkpoint,
        args.out,
        phase=phase,
        config=OfficialEvaluationConfig(
            max_actions_per_level=args.max_actions_per_level,
            max_game_actions=args.max_game_actions,
            device=args.device,
        ),
        sources=args.sources,
        smoke=args.smoke,
        training_report=args.training_report,
    )
    print(json.dumps({
        "report": str(args.out),
        "phase": report["phase"],
        "games_won": report["games_won"],
        "levels_completed": report["levels_completed"],
        "actions": report["actions"],
        "failures": report["failures"],
    }, indent=2))
    return report


def cli(argv=None) -> int:
    try:
        report = main(argv)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"official evaluation rejected: {exc}", file=sys.stderr)
        return 2
    return 0 if report["phase_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(cli())
