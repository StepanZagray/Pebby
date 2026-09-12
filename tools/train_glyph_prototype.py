"""Pretrain a standalone glyph MLP on audited generated-only level shards."""

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


SIZES = (6, 4, 4)
FIELDS = ("shape", "color", "rotation")
FORMAT = "pebby.glyph-prototype.v1"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as source:
        meta = json.loads(str(source["meta"].item()))
        if meta.get("format") != "pebby.ls20-glyphs.v1" or meta.get("source") != "generated_only":
            raise ValueError("expected generated-only glyph data")
        if meta.get("exact_crop_contradictions") != 0:
            raise ValueError("glyph labels have not passed the exact-crop identifiability audit")
        return {key: source[key] for key in ("glyphs", "triples", "seeds")}


def encode(glyphs):
    """Input order: row, then column, then 16 palette indicators."""
    return F.one_hot(torch.as_tensor(glyphs).long(), num_classes=16).float().flatten(1)


def evaluate(model, data):
    """Exact all-row metrics using unique glyph forwards plus inverse mapping."""
    packed = np.ascontiguousarray(data["glyphs"]).reshape(-1, 36).view("V36").reshape(-1)
    patterns, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
    with torch.inference_mode():
        scores = model(encode(data["glyphs"][first])).split(SIZES, -1)
        predicted = np.stack([value.argmax(-1).numpy() for value in scores], axis=1)[inverse]
    matches = predicted == data["triples"]
    return {"rows": len(packed), "unique_patterns_evaluated": len(patterns),
            "accuracy": {field: float(matches[:, i].mean()) for i, field in enumerate(FIELDS)},
            "joint_accuracy": float(matches.all(1).mean()),
            "evaluation": "all rows, weighted exactly through inverse mapping of unique glyphs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/ls20-glyph-train.npz"))
    parser.add_argument("--validation", type=Path, default=Path("data/ls20-glyph-validation.npz"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/world-glyph-data-audit.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ls20-glyph-prototype.pt"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/world-glyph-pretraining-prototype.json"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()

    def timeout(*_):
        raise TimeoutError("three-minute CPU prototype cap reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    report = {"format": FORMAT, "status": "running", "pid": os.getpid(), "device": "cpu",
              "threads": 1, "seed": args.seed, "steps_requested": args.steps,
              "steps_completed": 0, "batch_size": args.batch_size,
              "architecture": ["Linear(576,64)", "GELU", "Linear(64,14)"],
              "input": "uint8[N,6,6] -> one_hot16[N,6,6,16] -> float32[N,576]; row,column,palette flatten; no standardization",
              "output_fields": dict(zip(FIELDS, SIZES)),
              "optimizer": {"name": "AdamW", "lr": .003, "weight_decay": .01},
              "sampler": "uniform distinct level indices without replacement, then uniform row within each level",
              "validation_used_for_training_or_selection": False,
              "hashes": {str(path): digest(path) for path in (args.train, args.validation, args.audit, Path(__file__))},
              "training_log": []}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    persist()
    print("PID", os.getpid(), flush=True)
    try:
        audit = json.loads(args.audit.read_text())
        if audit.get("status") != "complete" or audit["census"]["contradictory_pattern_count"]:
            raise ValueError("complete contradiction-free glyph audit required")
        for split, path in (("train", args.train), ("validation", args.validation)):
            if audit["outputs"][split]["sha256"] != report["hashes"][str(path)]:
                raise ValueError("glyph dataset differs from audited bytes")
        train, validation = load(args.train), load(args.validation)
        seeds, counts = np.unique(train["seeds"], return_counts=True)
        validation_seeds = np.unique(validation["seeds"])
        if len(seeds) != 10296 or len(validation_seeds) != 1998 or np.intersect1d(seeds, validation_seeds).size:
            raise ValueError("expected all 10296 training and 1998 disjoint validation levels")
        if args.batch_size > len(seeds):
            raise ValueError("cannot draw a batch of distinct levels")
        order = np.argsort(train["seeds"], kind="stable")
        offsets = np.r_[0, counts.cumsum()[:-1]]
        seen = np.zeros(len(seeds), dtype=bool)
        report["eligible_train_seeds"] = seeds.tolist()
        report["validation_seeds"] = validation_seeds.tolist()
        report["source_rows"] = {"train": len(train["seeds"]), "validation": len(validation["seeds"])}
        model = nn.Sequential(nn.Linear(576, 64), nn.GELU(), nn.Linear(64, 14))
        report["parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
        minimum_distinct = args.batch_size
        for step in range(args.steps):
            groups = rng.choice(len(seeds), size=args.batch_size, replace=False)
            rows = order[offsets[groups] + rng.integers(0, counts[groups])]
            distinct = len(np.unique(train["seeds"][rows]))
            if distinct != args.batch_size:
                raise AssertionError("batch contains repeated level seeds")
            minimum_distinct = min(minimum_distinct, distinct)
            seen[groups] = True
            scores = model(encode(train["glyphs"][rows])).split(SIZES, -1)
            labels = torch.from_numpy(train["triples"][rows]).long()
            loss = sum(F.cross_entropy(score, labels[:, i]) for i, score in enumerate(scores)) / 3
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            report["steps_completed"] = step + 1
            if (step + 1) % 50 == 0:
                entry = {"step": step + 1, "training_batch_loss": float(loss.detach()),
                         "unique_levels_observed": int(seen.sum())}
                report["training_log"].append(entry)
                persist()
                print(entry, flush=True)
        model.eval()
        report["observed_train_seed_count"] = int(seen.sum())
        report["observed_train_seeds"] = seeds[seen].tolist()
        report["minimum_distinct_levels_per_batch"] = minimum_distinct
        report["train"] = evaluate(model, train)
        report["validation"] = evaluate(model, validation)
        report["status"] = "complete"
        persist()
        checkpoint = {"format": FORMAT, "architecture": report["architecture"],
                      "input": report["input"], "output_fields": report["output_fields"],
                      "weights": model.state_dict(), "provenance": report.copy()}
        temporary = args.checkpoint.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(args.checkpoint)
        report["checkpoint"] = str(args.checkpoint)
        report["checkpoint_sha256"] = digest(args.checkpoint)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        persist()
    print("DONE", report["train"], report["validation"], "seconds", report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
