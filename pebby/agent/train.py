"""Fit the LS20 policy on oracle-labelled shards, on the GPU.

Behaviour cloning with two wrinkles that decide whether the reported numbers
mean anything.

*Validation is a different set of LEVELS, never a slice of samples.* Samples
from one level are near duplicates of each other -- same walls, same pads,
player one cell over -- so splitting by sample leaks the answer and reports a
validation accuracy that says nothing about an unseen level. Prefer
``--validation-shards`` built from the validation bank, whose seeds start at
1,000,000 and cannot collide with training seeds; without it the shards are cut
by level seed, which is the same idea done in-place.

*Compare against the majority-action prior, every time.* An earlier ARC-AGI-3
model on this machine watched its loss fall while its held-out cross-entropy
(1.66739) stayed WORSE than the empirical prior of the training labels
(1.65815): it had learned nothing, and the loss curve hid it
(tofy-py ``docs/ACTION_RECALL_RESULTS.md``). So every run prints the constant
predictor's cross-entropy next to the model's, and says plainly when the model
fails to beat it.

The kept checkpoint is the best validation cross-entropy, not the last epoch.
Measured on this task: validation CE bottomed at epoch 3 and had roughly doubled
by epoch 20 while training CE kept falling, so saving the final weights would
have shipped a measurably worse policy than the run actually found.

*The target is the optimal action SET, not one member of it.* Measured over 981
on-path states: 70% have exactly one optimal action, but 22% have two, 6% three
and 2% all four, averaging 1.397. Cross-entropy against a single arbitrary
member docks the model for playing correctly at nearly a third of all states,
and hedging between tied actions is the correct response to that loss -- which
is then exactly what loses under argmax. When a shard carries the optimal-set
bitmask, the target puts equal mass on every optimal action instead. Two
accuracies are reported: `accuracy` still matches the oracle's single pick, so
it stays comparable with older runs and is capped near 83.6% for a perfect
player, while `set_accuracy` asks the question that decides completion -- did
the policy choose SOME optimal action.

Loss is action cross-entropy and nothing else. The auxiliary
actions-to-completion head was removed after measured failures of auxiliary
heads on this workload; ``to_go`` is still carried in the shards for analysis
and curriculum, and is deliberately unused here.

Float32 in eager mode, with TF32 disabled. bf16 autocast was measured to buy no
speedup on this class of workload, and ``torch.compile``/triton are unverified
on this box, so none of them is worth the risk of a silent numerical change.
And remember what the accuracy below is: a proxy. The number that counts is
closed-loop completion from ``pebby.agent.evaluate``, because where the oracle
has equal-length routes, matching its exact action was never the objective.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..ls20 import names
from .model import build_policy, load_checkpoint, save_checkpoint

ACTION_COUNT = len(names.ACTION_IDS)


def split_by_seed(seeds, fraction=.15, rng_seed=0):
    """Partition SEEDS (not samples) into train/validation; returns two masks."""
    unique = np.unique(seeds)
    if len(unique) < 2:
        return np.ones(len(seeds), dtype=bool), np.zeros(len(seeds), dtype=bool), unique.tolist(), []
    shuffled = np.random.default_rng(rng_seed).permutation(unique)
    # At least one level each side, so a small run still reports a real number.
    held = min(max(1, round(fraction * len(unique))), len(unique) - 1)
    validation_seeds, train_seeds = shuffled[:held], shuffled[held:]
    validation = np.isin(seeds, validation_seeds)
    return ~validation, validation, sorted(int(s) for s in train_seeds), sorted(int(s) for s in validation_seeds)


def label_prior(actions):
    """Empirical action frequencies of a label set, as a probability vector."""
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=ACTION_COUNT).astype(np.float64)
    return counts / max(1., counts.sum())


def optimal_targets(masks):
    """Bitmask per state -> [N, 4] target with equal mass on every optimal action."""
    bits = ((np.asarray(masks, dtype=np.uint8)[:, None] >> np.arange(ACTION_COUNT)) & 1).astype(np.float32)
    return bits / np.maximum(1., bits.sum(1, keepdims=True))


def prior_scores(prior, actions, masks=None):
    """What a constant predictor that ignores the frame would score.

    This is the bar. A model whose cross-entropy sits above it has learned
    nothing about the frame, however pretty its loss curve looks. Scored against
    the same soft target the model sees, so the two stay comparable.
    """
    actions = np.asarray(actions, dtype=np.int64)
    if not len(actions):
        return {"cross_entropy": None, "accuracy": None, "set_accuracy": None}
    target = optimal_targets(masks if masks is not None else (1 << actions).astype(np.uint8))
    guess = int(prior.argmax())
    return {"cross_entropy": float(-(target * np.log(np.clip(prior, 1e-12, None))).sum(1).mean()),
            "accuracy": float((actions == guess).mean()),
            "set_accuracy": float((target[:, guess] > 0).mean())}


def disjoint_seeds(train_shard, validation_shard):
    """Refuse to report a validation number that shares a level with training."""
    shared = np.intersect1d(np.unique(train_shard["seeds"]), np.unique(validation_shard["seeds"]))
    if len(shared):
        raise ValueError(f"{len(shared)} level seed(s) appear in both the training and validation "
                         f"shards, e.g. {shared[:5].tolist()}; validation would be meaningless")


def make_dataset(shard, mask):
    """Frames stay uint8 in host memory; only the batch is widened on the GPU.

    The optimal-set bitmask rides along as a third column. A shard without one
    (format v1) gets a mask holding only the recorded label, which makes the
    soft target identical to plain cross-entropy -- old shards still train, they
    just do not get the correction.
    """
    actions = shard["actions"][mask].astype(np.int64)
    optimal = shard["optimal"][mask] if shard.get("optimal") is not None else (1 << actions).astype(np.uint8)
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(shard["frames"][mask])),
                         torch.from_numpy(actions),
                         torch.from_numpy(optimal.astype(np.uint8)))


def schedule(optimizer, total_steps, warmup=.03):
    """Linear warmup into cosine decay, stepped per batch."""
    warmup_steps = max(1, int(warmup * total_steps))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def bits_of(masks, device):
    """uint8 bitmask -> float [B, 4], one column per action index."""
    powers = torch.arange(ACTION_COUNT, device=device)
    return ((masks.to(device, non_blocking=True).long()[:, None] >> powers) & 1).float()


def run_epoch(model, loader, device, optimizer=None, scheduler=None, *,
              train_min_loops=1, loop_loss="all", depth_generator=None):
    """One pass. With no optimizer this is the validation pass."""
    model.train(optimizer is not None)
    total_loss, objective_loss, correct, in_set, samples = 0., 0., 0, 0, 0
    depth_counts, depth_loss = {}, {}
    clipped_steps, optimizer_steps = 0, 0
    for frames, actions, masks in loader:
        frames = frames.to(device, non_blocking=True).long()
        actions = actions.to(device, non_blocking=True)
        bits = bits_of(masks, device)
        target = bits / bits.sum(1, keepdim=True).clamp_min(1.)
        with torch.set_grad_enabled(optimizer is not None):
            if optimizer is not None and model.config().get("architecture") == "looped":
                depth = int(torch.randint(train_min_loops, model.loops + 1, (),
                                          generator=depth_generator))
                depth_counts[str(depth)] = depth_counts.get(str(depth), 0) + 1
                exits = model(frames, loops=depth, return_all=loop_loss == "all")
                logits = exits[-1] if loop_loss == "all" else exits
                objective = -(target * F.log_softmax(exits, dim=-1)).sum(-1).mean()
            else:
                logits = model(frames)
                objective = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            # Soft cross-entropy against the optimal set. With a single optimal
            # action this is exactly F.cross_entropy, so nothing regresses.
            loss = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            clipped_steps += int(norm > 1.)
            optimizer_steps += 1
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if optimizer is not None and model.config().get("architecture") == "looped":
            depth_loss[str(depth)] = depth_loss.get(str(depth), 0.) + loss.item()
        chosen = logits.argmax(1)
        total_loss += len(frames) * loss.item()
        objective_loss += len(frames) * objective.item()
        correct += int((chosen == actions).sum())
        in_set += int(bits.gather(1, chosen[:, None]).sum())
        samples += len(frames)
    divisor = max(1, samples)
    # Cross-entropy is reported in the same units as the prior, so the two are
    # directly comparable; that comparison is the point of reporting either.
    return {"cross_entropy": total_loss / divisor, "accuracy": correct / divisor,
            "set_accuracy": in_set / divisor, "samples": samples,
            "objective_cross_entropy": objective_loss / divisor,
            "depth_batches": depth_counts,
            "depth_final_cross_entropy": {key: depth_loss[key] / count
                                          for key, count in depth_counts.items()},
            "gradient_clip_fraction": clipped_steps / max(1, optimizer_steps)}


def resolve_device(requested):
    """Never pretend: say which device is really in use and why."""
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")
        print("CUDA unavailable -- training on CPU, which is only sane for a smoke test", flush=True)
        return torch.device("cpu")
    return torch.device("cuda")


def training_sets(shard, held, args):
    """Training and validation sets, plus the seeds each side owns.

    `held` is a separately generated shard -- the validation bank -- when one was
    given. Otherwise the single shard is cut by level seed.
    """
    def labels(source, mask):
        optimal = source["optimal"][mask] if source.get("optimal") is not None else None
        return source["actions"][mask], optimal

    if held is not None:
        disjoint_seeds(shard, held)
        whole, all_held = np.ones(len(shard["actions"]), bool), np.ones(len(held["actions"]), bool)
        return (make_dataset(shard, whole), make_dataset(held, all_held),
                labels(shard, whole), labels(held, all_held),
                sorted(int(s) for s in np.unique(shard["seeds"])),
                sorted(int(s) for s in np.unique(held["seeds"])))
    train_mask, validation_mask, train_seeds, validation_seeds = split_by_seed(
        shard["seeds"], args.validation_fraction, args.seed)
    return (make_dataset(shard, train_mask), make_dataset(shard, validation_mask),
            labels(shard, train_mask), labels(shard, validation_mask),
            train_seeds, validation_seeds)


def train(shard, held, args, device):
    train_set, validation_set, train_labels, validation_labels, train_seeds, validation_seeds = (
        training_sets(shard, held, args))
    # The prior comes from the TRAINING labels only: a baseline fitted to the
    # held-out labels would be cheating in the baseline's favour.
    prior = (optimal_targets(train_labels[1]).mean(0).astype(np.float64)
             if train_labels[1] is not None else label_prior(train_labels[0]))
    baseline = {"prior": prior.tolist(), "train": prior_scores(prior, *train_labels),
                "validation": prior_scores(prior, *validation_labels),
                "labelled_with_optimal_set": train_labels[1] is not None}
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=args.loader_workers, pin_memory=pin,
                        persistent_workers=args.loader_workers > 0)
    validation_loader = (DataLoader(validation_set, batch_size=args.batch_size, pin_memory=pin,
                                    num_workers=args.loader_workers,
                                    persistent_workers=args.loader_workers > 0)
                         if len(validation_set) else None)
    config = {"architecture": args.architecture, "channels": args.channels,
              "blocks": args.blocks, "hidden": args.hidden, "reduce_channels": args.reduce_channels}
    if args.architecture == "looped":
        config.update(heads=args.heads, expansion=args.expansion, loops=args.loops)
    else:
        config.update(broadcast_hud=args.broadcast_hud, condition_channels=args.condition_channels)
    model = build_policy(config).to(device)
    depth_generator = torch.Generator().manual_seed(args.seed + 1)
    parameters = model.parameters()
    if args.architecture == "looped":
        # Biases and normalization scales are not weight matrices.
        parameters = [{"params": [p for p in model.parameters() if p.ndim >= 2]},
                      {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.}]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = schedule(optimizer, args.epochs * max(1, len(loader)))

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"{device} ({name}) | float32, TF32 off, eager | {model.parameter_count():,} parameters", flush=True)
    print(f"{len(train_set):,} train samples over {len(train_seeds)} level seeds | "
          f"{len(validation_set):,} validation samples over {len(validation_seeds)} held-out seeds", flush=True)
    print(f"Majority-action prior {np.round(prior, 3).tolist()} | train CE {baseline['train']['cross_entropy']:.5f}"
          + (f" | validation CE {baseline['validation']['cross_entropy']:.5f}"
             if baseline["validation"]["cross_entropy"] is not None else " | no validation labels"), flush=True)
    if validation_loader is None:
        print("No held-out seeds: validation is not reported. Generate more levels.", flush=True)
    if device.type == "cuda":
        assert next(model.parameters()).is_cuda, "model is not on CUDA"  # Proof, not a claim.

    if args.architecture == "looped":
        print(f"Shared transformer: {args.blocks} blocks x {args.loops} loops at validation; "
              f"train depth uniform {args.train_min_loops}..{args.loops}, "
              f"{args.loop_loss}-exit action loss, full BPTT", flush=True)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        stats = run_epoch(model, loader, device, optimizer, scheduler,
                          train_min_loops=args.train_min_loops, loop_loss=args.loop_loss,
                          depth_generator=depth_generator)
        entry = {"epoch": epoch, "train": stats, "lr": scheduler.get_last_lr()[0]}
        line = (f"Epoch {epoch}/{args.epochs} | train CE {stats['cross_entropy']:.5f} "
                f"(prior {baseline['train']['cross_entropy']:.5f}) | train acc {stats['accuracy']:.4f} "
                f"set {stats['set_accuracy']:.4f}")
        if validation_loader is not None:
            with torch.inference_mode():
                entry["validation"] = run_epoch(model, validation_loader, device)
            line += (f" | val CE {entry['validation']['cross_entropy']:.5f} "
                     f"(prior {baseline['validation']['cross_entropy']:.5f}) | "
                     f"val acc {entry['validation']['accuracy']:.4f} "
                     f"set {entry['validation']['set_accuracy']:.4f}")
        # Keep the epoch that generalised, not the one that memorised hardest.
        # Held on the CPU so the spare copy never competes for GPU memory.
        # MEASURED CAVEAT: on this task the checkpoint with the best validation
        # cross-entropy did NOT complete more levels than a later, overfitted one
        # (9/100 vs 15/100 on 100 unseen levels -- a 1.3-sigma difference, i.e.
        # indistinguishable at that sample size). Cross-entropy is a proxy and it
        # is not the objective, so the criterion is a flag, not a law.
        scored = entry.get("validation", entry["train"])
        score = (-scored[args.select_on] if args.select_on in ("accuracy", "set_accuracy")
                 else scored["cross_entropy"])
        if args.select_on == "last":
            score = -epoch
        if best is None or score < best["score"]:
            best = {"epoch": epoch, "score": score, "criterion": args.select_on,
                    "cross_entropy": scored["cross_entropy"], "accuracy": scored["accuracy"],
                    "selected_on": "validation" if "validation" in entry else "train",
                    "weights": copy.deepcopy({k: v.detach().to("cpu") for k, v in model.state_dict().items()})}
            line += " *"
        history.append(entry)
        print(line, flush=True)
        if device.type == "cuda" and epoch == 1:
            print(f"Peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    model.load_state_dict(best.pop("weights"))
    best.pop("score")
    print(f"Keeping epoch {best['epoch']} by {best['criterion']} "
          f"({best['selected_on']} CE {best['cross_entropy']:.5f}, acc {best['accuracy']:.4f}), "
          f"not epoch {args.epochs}" if best["epoch"] != args.epochs else
          f"Best epoch was the last one ({best['epoch']})", flush=True)
    return model, history, baseline, best, train_seeds, validation_seeds


def verdict(history, baseline, best=None):
    """State plainly whether the KEPT model beat the frame-blind baseline."""
    chosen = history[best["epoch"] - 1] if best else history[-1]
    split = "validation" if "validation" in chosen else "train"
    model_ce = chosen[split]["cross_entropy"]
    prior_ce = baseline[split]["cross_entropy"]
    if prior_ce is None:
        return {"split": split, "beats_prior": None,
                "message": "No baseline available: the prior has no labels to score."}
    beats = model_ce < prior_ce
    # Accuracy is reported alongside because the two can disagree: a model can be
    # right more often than the prior while being worse calibrated than it, and
    # that combination is exactly what a loss curve alone would hide.
    model_accuracy, prior_accuracy = chosen[split]["accuracy"], baseline[split].get("accuracy")
    return {"split": split, "epoch": chosen.get("epoch"), "model_cross_entropy": model_ce,
            "prior_cross_entropy": prior_ce, "model_accuracy": model_accuracy,
            "prior_accuracy": prior_accuracy, "beats_prior": beats,
            "message": (f"{split} cross-entropy {model_ce:.5f} "
                        f"{'beats' if beats else 'does NOT beat'} the majority-action prior "
                        f"{prior_ce:.5f}" +
                        (f" (accuracy {model_accuracy:.4f} vs {prior_accuracy:.4f})"
                         if prior_accuracy is not None else "") +
                        ("" if beats else " -- the loss curve is not evidence of learning here; "
                         "judge this run on completion rate alone"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", type=Path, nargs="+", required=True,
                        help="npz shards written by pebby.agent.data")
    parser.add_argument("--validation-shards", type=Path, nargs="+",
                        help="Shards from the validation bank; without these the training "
                             "shards are cut by level seed instead")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--validation-fraction", type=float, default=.15,
                        help="Fraction of LEVEL SEEDS held out, never a fraction of samples")
    parser.add_argument("--architecture", choices=("looped", "cnn"), default="looped")
    parser.add_argument("--channels", type=int, help="Default: looped 64, CNN 48")
    parser.add_argument("--blocks", type=int, help="Physical blocks; default: looped 2, CNN 4")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--expansion", type=int, default=4, help="Transformer MLP width multiplier")
    parser.add_argument("--loops", type=int, default=4, help="Maximum training and fixed validation depth")
    parser.add_argument("--train-min-loops", type=int, default=1,
                        help="Sample a uniform depth per batch from this value through --loops")
    parser.add_argument("--loop-loss", choices=("all", "final"), default="final",
                        help="Mean action CE over exits, or only the sampled final exit")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--reduce-channels", type=int, default=8)
    parser.add_argument("--broadcast-hud", action="store_true",
                        help="Tile the HUD features over the 12x12 lattice before the residual "
                             "trunk, so matching the carried triple to a pad icon is a local op")
    parser.add_argument("--condition-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select-on", choices=("cross_entropy", "accuracy", "set_accuracy", "last"),
                        default="cross_entropy",
                        help="Which epoch's weights to keep. Cross-entropy is the default but it is "
                             "only a proxy: measured here, the best-CE epoch did not complete more "
                             "levels than a later overfitted one")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--checkpoint-out", type=Path, help="Defaults to a separate path per architecture")
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    if args.channels is None:
        args.channels = 64 if args.architecture == "looped" else 48
    if args.blocks is None:
        args.blocks = 2 if args.architecture == "looped" else 4
    if args.checkpoint_out is None:
        args.checkpoint_out = Path("checkpoints/ls20-looped-policy.pt" if args.architecture == "looped"
                                   else "checkpoints/ls20-policy.pt")
    if min(args.heads, args.expansion, args.loops, args.train_min_loops) < 1:
        parser.error("heads, expansion, loops and train-min-loops must be positive")
    if args.train_min_loops > args.loops:
        parser.error("train-min-loops must be <= loops")
    if args.architecture == "looped" and args.channels % args.heads:
        parser.error("channels must be divisible by heads")
    if args.architecture == "looped" and args.broadcast_hud:
        parser.error("broadcast-hud is CNN-only; the looped model attends to HUD tokens")
    if any(value < 1 for value in (args.epochs, args.batch_size, args.channels, args.blocks,
                                   args.hidden, args.reduce_channels)):
        parser.error("epochs, batch-size, channels, blocks, hidden and reduce-channels must be positive")
    if not 0 < args.lr < 1 or args.loader_workers < 0:
        parser.error("lr must be in (0, 1) and loader-workers must not be negative")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be strictly between 0 and 1")
    missing = [str(path) for path in args.shards + (args.validation_shards or []) if not path.exists()]
    if missing:
        parser.error(f"missing shards: {', '.join(missing)}")

    torch.manual_seed(args.seed)
    # TF32 silently rounds float32 matmuls to 10 mantissa bits. Off, so a number
    # measured here is the number this code computes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    # Imported here, not at module scope: reading shards is the only thing the
    # trainer needs from the data pipeline, and only once the arguments are valid.
    from . import data as shard_data

    shard = shard_data.load_shards(args.shards)
    held = shard_data.load_shards(args.validation_shards) if args.validation_shards else None
    if not len(shard["actions"]):
        parser.error("the shards contain no samples")
    if held is not None and not len(held["actions"]):
        parser.error("the validation shards contain no samples")
    generator_versions = sorted({str(meta.get("generator_version"))
                                 for meta in shard["meta"] + (held["meta"] if held else [])})
    label_sources = [{"format": meta.get("format"), "samples": meta.get("samples"),
                      "mask_fallbacks": meta.get("mask_fallbacks"),
                      "multi_optimal_samples": meta.get("multi_optimal_samples")}
                     for meta in shard["meta"]]
    if any(meta.get("format") != shard_data.DATA_FORMAT or meta.get("mask_fallbacks", 0)
           for meta in shard["meta"] + (held["meta"] if held else [])):
        print("Label caveat: legacy or fallback labels are single-action targets; "
              "they do not establish the full optimal action set.", flush=True)
    try:
        model, history, baseline, best, train_seeds, validation_seeds = train(shard, held, args, device)
    except ValueError as error:
        parser.error(str(error))
    decision = verdict(history, baseline, best)

    checkpoint = save_checkpoint(
        args.checkpoint_out, model.to("cpu"),
        data_format=shard["meta"][0].get("format"),
        generator_version=generator_versions[0] if len(generator_versions) == 1 else generator_versions,
        train_seeds=train_seeds, validation_seeds=validation_seeds, epochs=args.epochs,
        best_epoch=best["epoch"], best_cross_entropy=best["cross_entropy"],
        selected_on=best["selected_on"], select_on=args.select_on, batch_size=args.batch_size, lr=args.lr,
        samples=int(len(shard["actions"])), device=str(device), baseline=baseline,
        beats_prior=decision["beats_prior"],
        label_sources=label_sources,
        shards=[str(path) for path in args.shards],
        validation_shards=[str(path) for path in (args.validation_shards or [])],
        loop_training=({"min_loops": args.train_min_loops, "max_loops": args.loops,
                        "sampling": "uniform_per_batch", "loss": args.loop_loss,
                        "backpropagation": "full", "validation_loops": args.loops}
                       if args.architecture == "looped" else None),
        trained=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    # Reload immediately: a checkpoint that cannot be rebuilt is not a checkpoint.
    load_checkpoint(args.checkpoint_out)

    report = {"format": checkpoint["format"], "config": checkpoint["config"],
              "loop_training": checkpoint["loop_training"], "history": history, "baseline": baseline, "verdict": decision,
              "best": best,
              "train_seeds": train_seeds, "validation_seeds": validation_seeds,
              "checkpoint": str(args.checkpoint_out), "device": str(device),
              "parameters": checkpoint["parameters"]}
    path = args.report_out or args.checkpoint_out.with_suffix(".training.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.checkpoint_out} (epoch {best['epoch']} of {args.epochs})\nReport {path}", flush=True)
    print(decision["message"], flush=True)
    print("Imitation accuracy is a proxy; run pebby.agent.evaluate for the completion rate.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
