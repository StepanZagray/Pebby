#!/usr/bin/env python3
"""Collect bounded whole games from generated game packages.

Omitting ``--games`` is the full protocol and preflights all 24 training
families before creating the output directory.  Passing ``--games`` opts into
a clearly labelled smoke subset.  The held-out m0r0 family is rejected in both
modes.

Example bounded smoke::

    uv run python tools/collect_multigame_games.py \
      --games cd82 ft09 tr87 --games-per-source 1 \
      --difficulties 1 2 --out-dir /tmp/pebby-multigame-smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby.multigame import (  # noqa: E402
    PreflightError,
    SearchLimits,
    RolloutOptions,
    VariantOptions,
    collect_generated_game,
    curriculum_for,
    new_manifest,
    preflight,
    save_collected_game,
    save_manifest,
    update_manifest_summary,
)
from pebby.multigame_dataset import (  # noqa: E402
    DEFAULT_GAME_SECONDS,
    DEFAULT_RECOVERY_SECONDS,
    build_perturber,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--games", nargs="+", default=None, metavar="GAME",
        help="explicit smoke subset by slug/source id; omit to require all 24 training sources",
    )
    parser.add_argument("--games-per-source", type=int, default=1)
    parser.add_argument(
        "--difficulties", type=int, nargs="+", default=None,
        help="explicit reduced smoke curriculum; full collection uses each family contract",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions-per-level", type=int, default=512)
    parser.add_argument(
        "--max-search-work", type=int, default=None,
        help="search-work ceiling (default: 32M full, 2M smoke)",
    )
    parser.add_argument("--max-generation-attempts", type=int, default=8,
                        help="deterministic outer seed attempts per requested level")
    parser.add_argument("--generator-attempts", type=int, default=50,
                        help="inner attempts when a generator exposes an attempts parameter")
    parser.add_argument(
        "--variants", action="store_true",
        help="explicitly enable fixed per-game public control/spatial/palette variants",
    )
    parser.add_argument(
        "--variant-components", nargs="+", choices=("controls", "spatial", "palette"),
        default=("controls", "spatial", "palette"),
        help="variant components to sample when --variants is enabled",
    )
    parser.add_argument(
        "--variant-mix", type=float, default=1.0,
        help="probability each whole game receives a sampled variant (otherwise identity)",
    )
    parser.add_argument(
        "--variant-seed", type=int, default=None,
        help="private variant mixture seed; defaults to --seed",
    )
    parser.add_argument(
        "--random-action-probability", type=float, default=0.0,
        help="per-step probability of a uniform legal random action before exact re-planning",
    )
    parser.add_argument(
        "--max-game-steps", type=int, default=4096,
        help="hard whole-game executed-transition cap",
    )
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
        help="what executes at a perturbation point before exact teacher recovery",
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
    if args.games_per_source < 1:
        parser.error("--games-per-source must be positive")
    if args.games is None and args.difficulties is not None:
        parser.error("--difficulties is smoke-only; full collection uses family curricula")
    if args.games is not None and args.difficulties is None:
        args.difficulties = [1, 2, 3]
    if args.difficulties is not None and (
        not args.difficulties or any(value < 1 for value in args.difficulties)
    ):
        parser.error("--difficulties must contain positive integers")
    if args.difficulties is not None and len(set(args.difficulties)) != len(args.difficulties):
        parser.error("--difficulties must be distinct")
    if any(value < 1 for value in (
        args.max_actions_per_level,
        args.max_search_work if args.max_search_work is not None else 1,
        args.max_generation_attempts,
        args.generator_attempts,
        args.max_game_steps,
    )):
        parser.error("all action, search, and generation bounds must be positive")
    if not 0.0 <= args.variant_mix <= 1.0:
        parser.error("--variant-mix must be in [0,1]")
    if not 0.0 <= args.random_action_probability <= 1.0:
        parser.error("--random-action-probability must be in [0,1]")
    if args.variants and not args.variant_components:
        parser.error("--variants requires at least one --variant-components value")
    return args


def main(argv=None):
    args = parse_args(argv)
    log = (lambda *values, **kwargs: None) if args.quiet else (
        lambda *values, **kwargs: print(*values, **kwargs, flush=True)
    )

    # This is intentionally before mkdir: a default run missing any one of the
    # 24 packages must reject without leaving an output that looks collectable.
    explicit_subset = args.games is not None
    modules = preflight(args.games, require_full_standard=not explicit_subset)
    limits = SearchLimits(
        args.max_actions_per_level,
        args.max_search_work if args.max_search_work is not None else (
            2_000_000 if explicit_subset else 32_000_000
        ),
    )
    curricula = {
        package.source.source_id: curriculum_for(
            package,
            args.difficulties,
            require_full_standard=not explicit_subset,
            smoke_search_work=limits.max_search_work,
        )
        for package in modules
    }
    if any(
        entry.search_work > limits.max_search_work
        for curriculum in curricula.values()
        for entry in curriculum
    ):
        raise ValueError(
            "configured --max-search-work is below a required family tier cap"
        )
    contracts = (
        None if explicit_subset else {
            package.source.source_id: package.full_standard for package in modules
        }
    )
    components = set(args.variant_components)
    variants = VariantOptions(
        enabled=args.variants,
        mix_probability=args.variant_mix,
        seed=args.seed if args.variant_seed is None else args.variant_seed,
        controls="controls" in components,
        spatial="spatial" in components,
        palette="palette" in components,
    )
    rollout = RolloutOptions(
        random_action_probability=args.random_action_probability,
        max_game_steps=args.max_game_steps,
        recovery_seconds=args.recovery_seconds or None,
        game_seconds=args.game_seconds or None,
        perturbation=args.perturbation,
        learner_checkpoint=(
            None if args.learner_checkpoint is None else str(args.learner_checkpoint)
        ),
    )
    perturber = build_perturber(rollout)
    manifest = new_manifest(
        sources=[item.source for item in modules],
        explicit_subset=explicit_subset,
        seed=args.seed,
        games_per_source=args.games_per_source,
        difficulties=args.difficulties,
        limits=limits,
        variants=variants,
        rollout=rollout,
        curricula_by_source=curricula,
        full_standard_contracts=contracts,
    )
    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(
            f"output directory {out_dir} is not empty; resume/overwrite semantics are not implemented"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    save_manifest(manifest_path, manifest)
    log(
        f"scope={manifest['scope']} sources={len(modules)} games={manifest['games_requested']} "
        f"curriculum=family-specific rollout={manifest['rollout_mode']} "
        f"variants={manifest['whole_game_rule_variants']}"
    )

    for package_index, package in enumerate(modules):
        for within_source in range(args.games_per_source):
            game_index = package_index * args.games_per_source + within_source
            key = f"{package.source.slug}-{within_source:06d}"
            curriculum = curricula[package.source.source_id]
            difficulties = tuple(entry.difficulty for entry in curriculum)
            collected = collect_generated_game(
                package,
                master_seed=args.seed,
                game_index=game_index,
                difficulties=difficulties,
                curriculum=curriculum,
                split="train" if not explicit_subset else None,
                require_full_standard=not explicit_subset,
                limits=limits,
                outer_generation_attempts=args.max_generation_attempts,
                generator_attempts=args.generator_attempts,
                variants=variants,
                rollout=rollout,
                perturber=perturber,
            )
            record = save_collected_game(out_dir, key, collected)
            manifest["records"].append(record)
            update_manifest_summary(manifest)
            save_manifest(manifest_path, manifest)
            log(
                f"{key}: {record['status']} levels={record.get('levels_completed', 0)}/"
                f"{len(curriculum)} steps={record.get('steps', 0)} "
                f"recovery={record.get('live_recovery_teacher_actions', 0)} "
                f"recovery_timeouts={record.get('recovery_timeouts', 0)}"
                f"{' game_timeout' if record.get('game_timeout') else ''}"
            )

    update_manifest_summary(manifest)
    save_manifest(manifest_path, manifest)
    log(
        f"wrote {manifest_path}: won={manifest['games_won']}/{manifest['games_requested']} "
        f"levels={manifest['levels_completed']}/{manifest['levels_requested']} steps={manifest['steps']} "
        f"teacher={manifest['teacher_steps']} random={manifest['random_steps']} "
        f"learner={manifest['learner_steps']} recovery={manifest['live_recovery_steps']} "
        f"recovery_timeouts={manifest['recovery_timeouts']} "
        f"game_timeouts={manifest['game_timeouts']} failures={manifest['rollout_failures']}"
    )
    return manifest


def cli(argv=None) -> int:
    try:
        manifest = main(argv)
    except (PreflightError, ValueError) as exc:
        print(f"collection rejected: {exc}", file=sys.stderr)
        return 2
    if manifest["rollout_mode"].startswith("mixed_teacher_"):
        hard_failures = sum(
            manifest["status_counts"].get(status, 0)
            for status in ("generation_failed", "engine_init_failed")
        )
        return 0 if manifest["games_recorded"] == manifest["games_requested"] and not hard_failures else 1
    return 0 if manifest["games_won"] == manifest["games_requested"] else 1


if __name__ == "__main__":
    raise SystemExit(cli())
