#!/usr/bin/env python3
"""Train the manifest-audited whole-game public visual policy.

The default command requires full 24-family train and validation manifests.
Use ``--smoke`` explicitly for bounded subsets; smoke checkpoints cannot unlock
the official held-out phase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby import multigame as M  # noqa: E402
from pebby.agent.multigame_model import (  # noqa: E402
    ARCHITECTURES,
    LossWeights,
    MultiGameModelConfig,
)
from pebby.agent.multigame_training import (  # noqa: E402
    HISTORY_MODES,
    UPDATE_MODES,
    LoadTimeVariantOptions,
    TrainingConfig,
    StopRequest,
    graceful_training_signals,
    load_training_checkpoint,
    audit_manifest_pair,
    click_region_census,
    train_multigame,
)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, nargs="+")
    parser.add_argument("--validation-manifest", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="resume latest.pt with its saved config and manifests")
    parser.add_argument("--initialize-from", type=Path, help="weights-only initialization of a NEW run; resets optimizer, RNG, selection and logs")
    parser.add_argument("--checkpoint-every-games", type=int, default=25, help="also commit recoverable checkpoints every N whole games")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--auxiliary-transitions-per-chunk", type=int, default=8)
    parser.add_argument("--metric-transitions-per-game", type=int, default=32)
    parser.add_argument("--closed-loop-interval", type=int, default=1)
    parser.add_argument("--closed-loop-games", type=int, default=24)
    parser.add_argument(
        "--closed-loop-train-games", type=int, default=len(M.TRAIN_SOURCE_IDS),
        help="diagnostic panel of TRAINING games played closed-loop each interval (0 disables); "
             "defaults to one game per intended family; never used for checkpoint selection",
    )
    parser.add_argument(
        "--history-mode", choices=HISTORY_MODES, default="full",
        help="previous-action history fed to the policy during training: 'full' keeps the "
             "executed triple, 'none' feeds BOS at every step of every game",
    )
    parser.add_argument(
        "--history-dropout", type=float, default=0.0,
        help="probability in [0,1] that a whole training game is fed BOS history for an epoch "
             "(sampled once per game per epoch, carried across its chunks); requires "
             "--history-mode full",
    )
    parser.add_argument(
        "--history-free", action="store_true",
        help="diagnostic: also score offline validation and play the generated closed-loop "
             "panels with history removed at inference; logged under *_history_free keys and "
             "never used for checkpoint selection",
    )
    parser.add_argument(
        "--update-mode", choices=UPDATE_MODES, default="chunk",
        help="'chunk': one clipped optimizer step per TBPTT chunk (current); 'game': per-term "
             "loss sums over all chunks divided by the game's target counts, one step per game",
    )
    parser.add_argument(
        "--frame-weight", type=float, default=LossWeights().next_frame,
        help="weight of the auxiliary next-frame loss; with --event-weight 0 the auxiliary "
             "heads are not computed",
    )
    parser.add_argument(
        "--event-weight", type=float, default=LossWeights().events,
        help="weight of the auxiliary event loss",
    )
    parser.add_argument("--changed-pixel-weight", type=float, default=1.0,
                        help="relative next-frame CE weight on changed pixels (default1 preserves ordinary CE)")
    parser.add_argument("--event-positive-weight", type=float, default=1.0,
                        help="relative BCE weight on positive events (default1 preserves ordinary BCE)")
    parser.add_argument(
        "--require-click-regions", action="store_true",
        help="fail fast when no training click row carries a stored click region",
    )
    parser.add_argument("--min-click-region-coverage", type=float, default=0.0,
                        help="require this fraction of training click targets to have stored regions (0..1)")
    parser.add_argument("--validation-max-actions-per-level", type=int, default=256)
    parser.add_argument("--validation-max-game-actions", type=int, default=2048)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--cpu-test-model", action="store_true",
        help="use the small wiring-test architecture; checkpoint remains smoke-quality evidence",
    )
    parser.add_argument(
        "--architecture", choices=ARCHITECTURES, default="v2",
        help="model network layout (default: v2): 'v1' is the original globally pooled action path with the "
             "query/bias click head, 'v2' keeps the 8x8 feature grid on the action path and "
             "decodes clicks with a contextual convolutional head",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=MultiGameModelConfig().hidden_dim,
        help="model hidden dimension; ignored under --cpu-test-model, which always keeps its "
             "own small wiring-test dimension",
    )
    parser.add_argument(
        "--canonical", action="store_true",
        help="canonical baseline: invert every stored public variant at load so all train and "
             "validation games are in raw engine space and the closed-loop panel plays the "
             "identity variant; load-time variants stay off unless --load-variants is given",
    )
    parser.add_argument(
        "--load-variants", action="store_true",
        help="compose a fresh whole-game public variant onto every training game each epoch "
             "(with --canonical this composes onto the canonical raw game)",
    )
    parser.add_argument(
        "--load-variant-mix", type=float, default=1.0,
        help="probability in [0,1] that a game gets a non-identity load-time variant",
    )
    parser.add_argument(
        "--load-variant-seed", type=int, default=None,
        help="seed for load-time variant sampling (defaults to --seed)",
    )
    parser.add_argument(
        "--load-variant-components", nargs="+", choices=("controls", "spatial", "palette"),
        default=("controls", "spatial", "palette"),
        help="which bijection components load-time variants may change",
    )
    parser.add_argument(
        "--load-variant-validation", action="store_true",
        help="also transform offline validation metrics (closed-loop panel is never transformed)",
    )
    args = parser.parse_args(argv)
    if args.resume is not None:
        allowed = {"--resume", "--epochs", "--device", "--checkpoint-every-games", "--out-dir"}
        supplied = {value.split("=", 1)[0] for value in argv if value.startswith("--")}
        if supplied - allowed:
            parser.error("--resume restores saved config/manifests; only --epochs, --device, --checkpoint-every-games and --out-dir may accompany it")
        saved = load_training_checkpoint(args.resume)
        args.out_dir = args.out_dir or args.resume.absolute().parent
        args.train_manifest = saved["dataset_audit"]["train"]["manifests"]
        args.validation_manifest = saved["dataset_audit"]["validation"]["manifests"]
        args.smoke = saved["smoke"]
        overrides = {}
        for option, field in (("--epochs", "epochs"), ("--device", "device"),
                              ("--checkpoint-every-games", "checkpoint_every_games")):
            if option in supplied:
                overrides[field] = getattr(args, field)
        args.saved_config = replace(TrainingConfig.from_dict(saved["training_config"]), **overrides)
        return args
    if args.train_manifest is None or args.validation_manifest is None or args.out_dir is None:
        parser.error("new runs require --train-manifest, --validation-manifest and --out-dir")
    if args.cpu_test_model and not args.smoke:
        parser.error("--cpu-test-model requires --smoke")
    if not 0.0 <= args.history_dropout <= 1.0:
        parser.error("--history-dropout must be in [0,1]")
    if args.history_mode == "none" and args.history_dropout != 0.0:
        parser.error("--history-dropout requires --history-mode full")
    if args.frame_weight < 0 or args.event_weight < 0:
        parser.error("--frame-weight/--event-weight must be non-negative")
    if not args.load_variants and (
        args.load_variant_seed is not None or args.load_variant_validation
        or args.load_variant_mix != 1.0
        or tuple(args.load_variant_components) != ("controls", "spatial", "palette")
    ):
        parser.error("--load-variant-* options require --load-variants")
    return args


def load_variant_options(args) -> LoadTimeVariantOptions:
    components = set(args.load_variant_components)
    return LoadTimeVariantOptions(
        enabled=bool(args.load_variants),
        mix_probability=float(args.load_variant_mix),
        seed=int(args.seed if args.load_variant_seed is None else args.load_variant_seed),
        controls="controls" in components,
        spatial="spatial" in components,
        palette="palette" in components,
        augment_validation=bool(args.load_variant_validation),
    )


def main(argv=None):
    args = parse_args(argv)
    # Audit before constructing a model or creating an output that might be
    # mistaken for a valid run.
    bundle = audit_manifest_pair(
        args.train_manifest, args.validation_manifest, smoke=args.smoke,
    )
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and args.resume is None:
        raise ValueError(
            f"output directory {args.out_dir} is not empty; pass --resume for an existing run"
        )
    config = args.saved_config if args.resume is not None else TrainingConfig(
        epochs=args.epochs,
        checkpoint_every_games=args.checkpoint_every_games,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        chunk_steps=args.chunk_steps,
        auxiliary_transitions_per_chunk=args.auxiliary_transitions_per_chunk,
        metric_transitions_per_game=args.metric_transitions_per_game,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        device=args.device,
        closed_loop_interval=args.closed_loop_interval,
        closed_loop_games=args.closed_loop_games,
        closed_loop_train_games=args.closed_loop_train_games,
        validation_max_actions_per_level=args.validation_max_actions_per_level,
        validation_max_game_actions=args.validation_max_game_actions,
        canonical_inputs=bool(args.canonical),
        model=(
            MultiGameModelConfig.cpu_test(architecture=args.architecture)
            if args.cpu_test_model
            else MultiGameModelConfig(architecture=args.architecture, hidden_dim=args.hidden_dim)
        ),
        loss=LossWeights(next_frame=float(args.frame_weight), events=float(args.event_weight),
                         changed_pixel_weight=args.changed_pixel_weight,
                         event_positive_weight=args.event_positive_weight),
        load_variants=load_variant_options(args),
        history_mode=args.history_mode,
        history_dropout=float(args.history_dropout),
        history_free_diagnostic=bool(args.history_free),
        update_mode=args.update_mode,
        require_click_regions=bool(args.require_click_regions),
        min_click_region_coverage=args.min_click_region_coverage,
    )
    if args.initialize_from is not None:
        initial = load_training_checkpoint(args.initialize_from)
        config = replace(config, model=MultiGameModelConfig(**initial["model_config"]))
    stop = StopRequest()
    with graceful_training_signals(stop):
        result = train_multigame(bundle, args.out_dir, config, resume=args.resume,
                                 initialize_from=args.initialize_from, stop_request=stop)
    (args.out_dir / "dataset-audit.json").write_text(
        json.dumps(bundle.summary(), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "scope": result.scope,
        "stopped": result.stopped,
        "best_available": result.best_checkpoint.is_file(),
        "best_checkpoint": str(result.best_checkpoint),
        "latest_checkpoint": str(result.latest_checkpoint),
        "training_log": str(result.logs_path),
        "canonical_inputs": config.canonical_inputs,
        "model_config": config.model.to_dict(),
        "closed_loop_train_games": config.closed_loop_train_games,
        "load_time_variants": config.load_variants.to_dict(),
        "update_mode": config.update_mode,
        "loss_weights": asdict(config.loss),
        "auxiliaries_enabled": config.auxiliaries_enabled,
        "history_mode": config.history_mode,
        "history_dropout": config.history_dropout,
        "history_free_diagnostic": config.history_free_diagnostic,
        "click_region_census": {
            "train": click_region_census(bundle.train),
            "validation": click_region_census(bundle.validation),
        },
        "scientific_limit": (
            "a recurrent imitation policy plus one-step prediction is not evidence of "
            "multi-step rule inference or search"
        ),
    }, indent=2))
    return result


def cli(argv=None) -> int:
    try:
        main(argv)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"training rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
