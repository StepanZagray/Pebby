"""Train and evaluate the in-context transition predictor on LS20 rule-variant games.

Example::

    PYTHONPATH=. uv run python tools/train_incontext_dynamics.py \
        --data-dir /tmp/variant-pilot --train-variants 0 1 2 ... --test-variants 3 7 ... \
        --out-dir runs/incontext --updates 3000 --batch-size 16 --eval-every 250

Games are ``DIR/games/<game_id>.npz`` (or ``DIR/*.npz``).  ``variant_id`` is used for ONE
thing: assigning each game to the train / held-out split.  It is never an input to the model.
10% of the train-variant games (at least one) are kept aside as the *held-in* evaluation set
(seen permutations, unseen games); the test-variant games are the *held-out* set (unseen
permutations).  ``report.json`` in ``--out-dir`` is rewritten after every evaluation with the
loss curve, per-split accuracy curves by step bin, steps-to-stable distributions and the
identity-baseline ("memorised standard controls") scores for the same games.  ``best.pt`` is
the checkpoint with the best held-out movement accuracy, ``final.pt`` the last one.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from pebby.agent import incontext_dynamics as icd
from pebby.agent import variant_metrics as vm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-variants", type=int, nargs="*", default=None,
                        help="variant ids to train on (default: every variant not in --test-variants)")
    parser.add_argument("--test-variants", type=int, nargs="*", default=[],
                        help="held-out variant ids (never trained on)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--start-bias", type=float, default=0.7,
                        help="probability that a training window starts at the game start")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--holdout-fraction", type=float, default=0.1,
                        help="fraction of train-variant games kept as the held-in eval set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memoryless", action="store_true",
                        help="train the MemorylessBaseline control instead of the transformer")
    parser.add_argument("--history", type=int, default=None,
                        help="0 is an alias for --memoryless; any other value selects the transformer")
    parser.add_argument("--no-previous-outcome", action="store_true",
                        help="drop the previous-step outcome features from the transformer tokens")
    parser.add_argument("--movement-weight", type=float, default=2.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.history == 0:
        args.memoryless = True
    return args


def resolve_device(requested):
    if requested == "cpu" or (requested == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    return torch.device("cuda")


def game_files(data_dir: Path):
    games = data_dir / "games"
    folder = games if games.is_dir() else data_dir
    return sorted(folder.glob("*.npz"))


def load_games(data_dir: Path):
    games = []
    for path in game_files(data_dir):
        arrays = icd.load_game(path)
        game_id = str(arrays["game_id"]) if "game_id" in arrays else path.stem
        variant = int(arrays["variant_id"])  # label: used for the split only
        games.append({"game_id": game_id, "variant_id": variant, "path": str(path), "arrays": arrays,
                      "steps": int(np.asarray(arrays["agent_action"]).shape[0])})
    if not games:
        raise SystemExit(f"no .npz games found under {data_dir}")
    return games


def split_games(games, train_variants, test_variants, holdout_fraction, seed):
    test_set = set(test_variants)
    if train_variants is None:
        train_variants = sorted({g["variant_id"] for g in games} - test_set)
    train_set = set(train_variants)
    if train_set & test_set:
        raise SystemExit(f"variants in both splits: {sorted(train_set & test_set)}")
    train_pool = [g for g in games if g["variant_id"] in train_set]
    held_out = [g for g in games if g["variant_id"] in test_set]
    if not train_pool:
        raise SystemExit("no games for the train variants")
    rng = random.Random(seed)
    order = list(range(len(train_pool)))
    rng.shuffle(order)
    held_in_count = min(len(train_pool) - 1, max(1, round(holdout_fraction * len(train_pool)))) \
        if len(train_pool) > 1 else 0
    held_in = [train_pool[i] for i in sorted(order[:held_in_count])]
    train = [train_pool[i] for i in sorted(order[held_in_count:])]
    return {"train": train, "held_in": held_in, "held_out": held_out,
            "train_variants": sorted(train_set), "test_variants": sorted(test_set)}


def split_summary(split):
    return {name: {"games": [g["game_id"] for g in split[name]],
                   "variants": sorted({g["variant_id"] for g in split[name]}),
                   "steps": [g["steps"] for g in split[name]]}
            for name in ("train", "held_in", "held_out")}


def evaluate_split(model, games, device, max_steps):
    reports = [icd.evaluate_game(model, g["arrays"], device=device, max_steps=max_steps,
                                 features=g["features"]) for g in games]
    summary = vm.aggregate(reports)
    summary["per_game"] = [{"game_id": g["game_id"], "variant_id": g["variant_id"],
                            "movement_accuracy": r["movement_accuracy"],
                            "joint_accuracy": r["joint"]["accuracy"],
                            "steps_to_stable": r["steps_to_stable"],
                            "movement_steps_to_stable": r["movement_steps_to_stable"]}
                           for g, r in zip(games, reports)]
    return summary


def baseline_split(games):
    return vm.aggregate(vm.score_predictions(vm.identity_baseline_predictions(g["arrays"]), g["arrays"])
                        for g in games)


def lr_lambda(warmup, total):
    def schedule(update):
        if update < warmup:
            return (update + 1) / max(1, warmup)
        progress = (update - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return schedule


def write_report(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(path)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    games = load_games(args.data_dir)
    split = split_games(games, args.train_variants, args.test_variants, args.holdout_fraction, args.seed)
    for name in ("train", "held_in", "held_out"):
        for g in split[name]:
            g["features"] = icd.featurize(g["arrays"])
    longest = max(g["steps"] for g in games)
    max_steps = args.max_steps
    model = icd.build_model(args.memoryless, d_model=args.d_model, layers=args.layers, heads=args.heads,
                            max_steps=max_steps, include_previous=not args.no_previous_outcome).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args.warmup, args.updates))
    weights = {"movement": args.movement_weight}
    # Evaluation splits in priority order; with no held-in/held-out games (e.g. a single-game
    # pilot) fall back to scoring the training games themselves and say so in the report.
    eval_splits = [name for name in ("held_in", "held_out") if split[name]] or ["train"]
    baselines = {name: baseline_split(split[name]) for name in eval_splits}
    report = {
        "format": icd.FORMAT,
        "config": {**vars(args), "data_dir": str(args.data_dir), "out_dir": str(args.out_dir),
                   "device": str(device), "model": type(model).__name__,
                   "parameters": model.parameter_count(), "longest_game": longest},
        "splits": split_summary(split),
        "loss": [],
        "evaluations": [],
        "identity_baseline": baselines,
        "best": None,
        "eval_splits": eval_splits,
        "evaluated_on_training_games": eval_splits == ["train"],
        "finished": False,
    }
    if not args.quiet:
        print(f"{type(model).__name__}: {model.parameter_count():,} parameters on {device}; "
              f"train {len(split['train'])} games / held-in {len(split['held_in'])} / "
              f"held-out {len(split['held_out'])}; longest game {longest} steps (window {max_steps})")
        for name, summary in baselines.items():
            print(f"  identity baseline {name}: movement {summary['movement_accuracy']:.3f} "
                  f"joint {summary['joint']['accuracy']:.3f}")
    report_path = args.out_dir / "report.json"
    write_report(report_path, report)

    def run_eval(update):
        entry = {"update": update, "time": time.time() - started}
        for name in eval_splits:
            entry[name] = evaluate_split(model, split[name], device, max_steps)
        report["evaluations"].append(entry)
        key = eval_splits[-1]   # held_out if present, else held_in, else train
        score = entry[key]["movement_accuracy"]
        if score is not None and (report["best"] is None or score > report["best"]["movement_accuracy"]):
            report["best"] = {"update": update, "split": key, "movement_accuracy": score,
                              "joint_accuracy": entry[key]["joint"]["accuracy"]}
            icd.save_checkpoint(model, args.out_dir / "best.pt", update=update, score=score)
        if not args.quiet:
            bits = []
            for name in eval_splits:
                if name in entry:
                    e = entry[name]
                    bits.append(f"{name}: movement {e['movement_accuracy']:.3f} joint {e['joint']['accuracy']:.3f} "
                                f"stable {e['steps_to_stable']['reached']}/{e['steps_to_stable']['games']}"
                                f" (median {e['steps_to_stable']['median']})")
            print(f"[eval @ {update}] " + " | ".join(bits))
        write_report(report_path, report)

    started = time.time()
    train_games = split["train"]
    running = []
    model.train()
    for update in range(1, args.updates + 1):
        picks = rng.integers(0, len(train_games), size=args.batch_size)
        windows = []
        for index in picks:
            features = train_games[int(index)]["features"]
            start, end = icd.sample_window(int(features["action"].shape[0]), max_steps, rng, args.start_bias)
            windows.append(icd.slice_features(features, start, end))
        batch = icd.batch_to(icd.collate(windows), device)
        logits = model(batch)
        loss, parts = icd.prediction_loss(logits, batch, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running.append(float(loss.detach()))
        if update % 10 == 0 or update == args.updates:
            report["loss"].append({"update": update, "loss": float(np.mean(running)),
                                   "fields": parts, "lr": scheduler.get_last_lr()[0]})
            if not args.quiet and (update % 100 == 0 or update == args.updates):
                print(f"[{update}] loss {np.mean(running):.4f} movement {parts['movement']:.4f}")
            running = []
        if update % args.eval_every == 0 or update == args.updates:
            run_eval(update)
            model.train()
    icd.save_checkpoint(model, args.out_dir / "final.pt", update=args.updates)
    report["finished"] = True
    report["seconds"] = time.time() - started
    write_report(report_path, report)
    return report


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
