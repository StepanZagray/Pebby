#!/usr/bin/env python3
"""Prepare bounded split-safe whole-game train/validation collections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby import multigame as M  # noqa: E402
from pebby.multigame_dataset import (  # noqa: E402
    DEFAULT_GAME_SECONDS,
    DEFAULT_RECOVERY_SECONDS,
    DatasetPreparationConfig,
    prepare_multigame_dataset,
)
from pebby.multigame_variants import VariantOptions  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-master-seed", type=int, required=True)
    parser.add_argument("--validation-master-seed", type=int, required=True)
    parser.add_argument(
        "--games", nargs="+", default=None,
        help="explicit smoke source subset; omit to require all 24 training families",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="required label for any explicit --games subset",
    )
    parser.add_argument("--completed-teacher-games-per-family", type=int, default=1)
    parser.add_argument("--mixed-games-per-family", type=int, default=1)
    parser.add_argument("--mixed-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--difficulties", type=int, nargs="+", default=None,
        help="explicit reduced smoke curriculum; full preparation uses each family contract",
    )
    parser.add_argument("--max-actions-per-level", type=int, default=512)
    parser.add_argument(
        "--max-search-work", type=int, default=None,
        help="search-work ceiling (default: 32M full, 2M smoke)",
    )
    parser.add_argument("--outer-generation-attempts", type=int, default=8)
    parser.add_argument("--generator-attempts", type=int, default=50)
    parser.add_argument("--max-candidate-attempts", type=int, default=8)
    parser.add_argument("--max-game-steps", type=int, default=4096)
    parser.add_argument(
        "--click-region-probe-limit", type=int, default=M.CLICK_REGION_PROBE_LIMIT,
        help="verified candidate pixels per teacher click (0 disables region labels)",
    )
    parser.add_argument("--variants", action="store_true")
    parser.add_argument(
        "--variant-components", nargs="+", choices=("controls", "spatial", "palette"),
        default=("controls", "spatial", "palette"),
    )
    parser.add_argument("--variant-mix", type=float, default=1.0)
    parser.add_argument("--variant-seed", type=int, default=0)
    parser.add_argument(
        "--recovery-seconds", type=float, default=DEFAULT_RECOVERY_SECONDS,
        help="wall-clock cap on each live teacher recovery search; 0 disables the cap",
    )
    parser.add_argument(
        "--game-seconds", type=float, default=DEFAULT_GAME_SECONDS,
        help="wall-clock cap on one whole rollout; 0 disables the cap",
    )
    parser.add_argument(
        "--perturbation", choices=("random", "learner"), default="random",
        help="what executes at a mixed-cohort perturbation point before teacher recovery",
    )
    parser.add_argument(
        "--learner-checkpoint", type=Path, default=None,
        help="multigame training checkpoint driving --perturbation learner",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.recovery_seconds < 0 or args.game_seconds < 0:
        parser.error("--recovery-seconds and --game-seconds must be non-negative")
    if args.perturbation == "learner" and args.learner_checkpoint is None:
        parser.error("--perturbation learner requires --learner-checkpoint")
    if args.perturbation != "learner" and args.learner_checkpoint is not None:
        parser.error("--learner-checkpoint requires --perturbation learner")
    if args.games is None and args.smoke:
        parser.error("--smoke requires explicit --games")
    if args.games is not None and not args.smoke:
        parser.error("--games requires --smoke")
    if args.games is None and args.difficulties is not None:
        parser.error("--difficulties is smoke-only; full preparation uses family curricula")
    return args


def main(argv=None):
    args = parse_args(argv)
    components = set(args.variant_components)
    variants = VariantOptions(
        enabled=args.variants,
        mix_probability=args.variant_mix,
        seed=args.variant_seed,
        controls="controls" in components,
        spatial="spatial" in components,
        palette="palette" in components,
    )
    config = DatasetPreparationConfig(
        output_root=args.output_root,
        train_master_seed=args.train_master_seed,
        validation_master_seed=args.validation_master_seed,
        games=None if args.games is None else tuple(args.games),
        smoke=args.smoke,
        completed_teacher_games_per_family=args.completed_teacher_games_per_family,
        mixed_games_per_family=args.mixed_games_per_family,
        mixed_epsilon=args.mixed_epsilon,
        difficulties=(None if args.difficulties is None else tuple(args.difficulties)),
        limits=M.SearchLimits(
            args.max_actions_per_level,
            args.max_search_work if args.max_search_work is not None else (
                2_000_000 if args.smoke else M.MAX_FULL_SEARCH_WORK
            ),
        ),
        outer_generation_attempts=args.outer_generation_attempts,
        generator_attempts=args.generator_attempts,
        max_candidate_attempts=args.max_candidate_attempts,
        max_game_steps=args.max_game_steps,
        variants=variants,
        recovery_seconds=args.recovery_seconds or None,
        game_seconds=args.game_seconds or None,
        perturbation=args.perturbation,
        learner_checkpoint=args.learner_checkpoint,
        click_region_probe_limit=args.click_region_probe_limit,
    )
    progress = (lambda message: None) if args.quiet else (
        lambda message: print(message, flush=True)
    )
    result = prepare_multigame_dataset(config, progress=progress)
    print(json.dumps({
        "complete": result.complete,
        "scope": result.summary["scope"],
        "preparation": str(result.summary_path),
        "manifests": result.summary["training_cli_arguments"],
        "strict_audit_error": result.summary["strict_audit_error"],
    }, indent=2))
    return result


def cli(argv=None) -> int:
    try:
        result = main(argv)
    except (M.PreflightError, OSError, TypeError, ValueError) as exc:
        print(f"dataset preparation rejected: {exc}", file=sys.stderr)
        return 2
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(cli())
