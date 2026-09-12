"""CPU-only attribute probes on frozen generated-state representations.

Teacher player coordinates select diagnostic features only. No engine, oracle,
official data, policy changes, or encoder updates are used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pebby.agent.world_model import CELLS, HUD_BOTTOM, HUD_TOP, _planes, load_world_checkpoint


SIZES = (6, 4, 4)
ATTRIBUTES = ("shape", "color", "rotation")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def selected_rows(path, count, seed):
    """Decompress one shard array at a time; retain one random row per level."""
    rng = np.random.default_rng(seed)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"].item()))
        if meta.get("source") != "generated_only":
            raise ValueError("probe input must be generated-only")
        seeds = archive["seeds"]
        distinct = np.unique(seeds)
        chosen = rng.permutation(distinct)[:count]
        indices = np.asarray([rng.choice(np.flatnonzero(seeds == s)) for s in chosen])
        arrays = {key: archive[key][indices] for key in (
            "frames", "history_valid", "previous_actions", "player_cell", "current_triple")}
    arrays["seeds"] = chosen
    return arrays


def features(model, arrays, batch_size):
    results = {name: [] for name in ("raw_player_cell", "refined_player_cell", "latent", "raw_hud",
                                    "refined_hud", "reduced_hud")}
    if model.cfg.glyph_recall:
        results["glyph_logits"] = []
    with torch.inference_mode():
        for start in range(0, len(arrays["seeds"]), batch_size):
            batch = {k: torch.as_tensor(v[start:start + batch_size])
                     for k, v in arrays.items() if k != "seeds"}
            frames, valid, actions = model._prepare(
                batch["frames"], batch["history_valid"], batch["previous_actions"])
            b, h = frames.shape[:2]
            tokens = model.frame_tokens(frames.flatten(0, 1)).view(b, h, model.tokens, -1)
            # With glyph_recall the current public crop is classified, as in encode().
            glyph = model.glyph_logits(frames[:, -1]) if model.cfg.glyph_recall else None
            encoded = model.assemble(tokens, valid, actions, glyph_logits=glyph)
            if glyph is not None:
                results["glyph_logits"].append(glyph.clone())
            player = batch["player_cell"].long()
            index = player[:, 1] * 12 + player[:, 0]
            row = torch.arange(b)
            results["raw_player_cell"].append(tokens[row, -1, index].clone())
            results["refined_player_cell"].append(encoded["cells"][row, index].clone())
            results["latent"].append(encoded["latent"].clone())
            results["raw_hud"].append(model.hud(
                _planes(frames[:, -1, HUD_TOP:HUD_BOTTOM, :])).flatten(1).clone())
            hud_state = encoded["state"][:, CELLS:]
            results["refined_hud"].append(hud_state.flatten(1).clone())
            results["reduced_hud"].append(model.reduce(hud_state).flatten(1).clone())
    return {key: torch.cat(value).clone() for key, value in results.items()}


def accuracies(logits, labels):
    return {name: float((scores.argmax(-1) == labels[:, index]).float().mean())
            for index, (name, scores) in enumerate(zip(ATTRIBUTES, logits.split(SIZES, -1)))}


def fit_probe(train, validation, train_labels, validation_labels, kind, seed, steps):
    torch.manual_seed(seed)
    # All preprocessing is fit on training features; validation never selects epochs.
    mean, scale = train.mean(0), train.std(0).clamp_min(1e-5)
    train, validation = (train - mean) / scale, (validation - mean) / scale
    head = (nn.Linear(train.shape[1], sum(SIZES)) if kind == "linear" else
            nn.Sequential(nn.Linear(train.shape[1], 64), nn.GELU(), nn.Linear(64, sum(SIZES))))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.0001)
    for _ in range(steps):
        scores = head(train).split(SIZES, -1)
        loss = sum(F.cross_entropy(s, train_labels[:, i]) for i, s in enumerate(scores)) / 3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return {"train_accuracy": accuracies(head(train), train_labels),
                "validation_accuracy": accuracies(head(validation), validation_labels),
                "final_training_loss": float(loss.detach()),
                "parameters": sum(p.numel() for p in head.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ls20-world-curriculum-b1024.pt")
    parser.add_argument("--train", default="data/ls20-world-mixedpath-train-part-00000.npz")
    parser.add_argument("--validation", default="data/ls20-world-mixedpath-validation-part-00000.npz")
    parser.add_argument("--out", default="artifacts/world-frozen-attribute-probes.json")
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("five-minute cap")))
    signal.alarm(300)
    output = Path(args.out)
    report = {"status": "running", "pid": os.getpid(), "device": "cpu", "threads": 1,
              "seed": args.seed, "teacher_player_selection": "diagnostic only; never policy input",
              "frozen_encoder": True, "probe_steps": args.steps,
              "probe_optimizer": {"type": "AdamW", "lr": .01, "weight_decay": .0001},
              "selection": "one random row per distinct level; fixed seed; no validation selection",
              "hashes": {p: digest(p) for p in (args.checkpoint, args.train, args.validation,
                         __file__, "pebby/agent/world_model.py")}, "results": {}}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)

    persist()
    print("PID", os.getpid(), flush=True)
    try:
        train = selected_rows(args.train, args.train_count, args.seed)
        validation = selected_rows(args.validation, args.validation_count, args.seed + 1)
        assert not set(train["seeds"]) & set(validation["seeds"])
        report["counts"] = {"train": len(train["seeds"]), "validation": len(validation["seeds"])}
        report["seeds"] = {"train": train["seeds"].tolist(), "validation": validation["seeds"].tolist()}
        labels = torch.as_tensor(train["current_triple"]).long()
        val_labels = torch.as_tensor(validation["current_triple"]).long()
        report["majority"] = {}
        for index, (name, size) in enumerate(zip(ATTRIBUTES, SIZES)):
            counts = torch.bincount(labels[:, index], minlength=size)
            majority = int(counts.argmax())
            report["majority"][name] = {"class": majority, "train_counts": counts.tolist(),
                "validation_counts": torch.bincount(val_labels[:, index], minlength=size).tolist(),
                "train_accuracy": float((labels[:, index] == majority).float().mean()),
                "validation_accuracy": float((val_labels[:, index] == majority).float().mean())}
        model, checkpoint = load_world_checkpoint(args.checkpoint, "cpu")
        model.requires_grad_(False)
        report["checkpoint_epoch"] = checkpoint.get("best_epoch", checkpoint.get("epoch"))
        report["encoder_loops"] = model.loops
        report["hud_pipeline"] = {
            "raw_stem": [model.cfg.hud_channels, model.cfg.hud_tokens],
            "projected_tokens": [model.cfg.hud_tokens, model.cfg.channels],
            "refined_tokens": [model.cfg.hud_tokens, model.cfg.channels],
            "reduced_tokens": [model.cfg.hud_tokens, model.cfg.reduce],
            "global_projector_input": [(CELLS + model.cfg.hud_tokens) * model.cfg.reduce],
            "global_latent": [model.cfg.latent],
            "flow": "HUD stem -> projection plus position -> causal temporal attention -> shared refinement loops -> per-token reduce -> flatten all cell and HUD tokens -> global projector",
        }
        train_features = features(model, train, args.batch_size)
        del train
        validation_features = features(model, validation, args.batch_size)
        del validation, model
        report["feature_shapes"] = {key: {"train": list(value.shape),
            "validation": list(validation_features[key].shape)} for key, value in train_features.items()}
        persist()
        print("extracted", report["counts"], "seconds", round(report["elapsed_seconds"], 1), flush=True)
        for name, feature in train_features.items():
            report["results"][name] = {}
            for kind in ("linear", "mlp64"):
                result = fit_probe(feature, validation_features[name], labels, val_labels,
                                   kind, args.seed, args.steps)
                report["results"][name][kind] = result
                persist()
                print(name, kind, result, flush=True)
        report["status"] = "complete"
    except TimeoutError:
        report["status"] = "incomplete_time_cap"
    finally:
        signal.alarm(0)
        persist()
    print("DONE", report["status"], report["elapsed_seconds"], "seconds", report["peak_rss_mib"], "MiB", flush=True)


if __name__ == "__main__":
    main()
