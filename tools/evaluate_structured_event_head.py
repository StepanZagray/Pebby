"""Evaluate the frozen structured-transition event head on generated caches.

The current field and FP32 imagined successors are loaded through the existing
provenance guards.  The transition model is not recomputed: only its event
head is applied to summaries of those two cached fields and each action.
"""

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from tools.cache_structured_policy_successors import load_imagined_cache
from tools.train_structured_policy import (
    build_training_policy,
    check_policy_encoder,
    digest,
    load_policy_cache,
)


EVENTS = ("lost_life", "terminal", "won")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _safe(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _rank_auc(probability, target):
    """Rank AUC with average ranks for tied probabilities."""
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    positive = int(target.sum())
    negative = int(len(target) - positive)
    if not positive or not negative:
        return None
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(len(target), dtype=np.float64)
    first = 0
    while first < len(target):
        last = first + 1
        while last < len(target) and sorted_probability[last] == sorted_probability[first]:
            last += 1
        ranks[order[first:last]] = (first + 1 + last) / 2.0
        first = last
    positive_rank_sum = float(ranks[target.astype(bool)].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _average_precision(probability, target):
    """Average precision over unique score thresholds, without threshold fitting."""
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    positive = int(target.sum())
    if not positive or positive == len(target):
        return None
    order = np.argsort(-probability, kind="mergesort")
    scores = probability[order]
    labels = target[order]
    true_positive = 0
    average_precision = 0.0
    first = 0
    while first < len(labels):
        last = first + 1
        while last < len(labels) and scores[last] == scores[first]:
            last += 1
        group_positive = int(labels[first:last].sum())
        true_positive += group_positive
        if group_positive:
            average_precision += (true_positive / last) * (group_positive / positive)
        first = last
    return average_precision


def _calibration(probability, target, bins=10):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    entries = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = ((probability >= low) & (probability < high if index + 1 < bins else probability <= high))
        count = int(selected.sum())
        entries.append({
            "bin": index,
            "lower_inclusive": low,
            "upper_inclusive": high if index + 1 == bins else None,
            "count": count,
            "mean_probability": _safe(probability[selected].mean()) if count else None,
            "positive_fraction": _safe(target[selected].mean()) if count else None,
        })
    return entries


def _event_metrics(probability, target):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    predicted = probability >= 0.5
    positive = target.astype(bool)
    negative = ~positive
    true_positive = int((predicted & positive).sum())
    false_positive = int((predicted & negative).sum())
    true_negative = int((~predicted & negative).sum())
    false_negative = int((~predicted & positive).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "examples": int(len(target)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "positive_rate": float(positive.mean()),
        "threshold": 0.5,
        "predicted_positive": int(predicted.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": _safe(true_positive / precision_denominator) if precision_denominator else None,
        "recall": _safe(true_positive / recall_denominator) if recall_denominator else None,
        "specificity": _safe(true_negative / int(negative.sum())) if negative.any() else None,
        "accuracy": float((predicted == positive).mean()),
        "f1": _safe(2 * true_positive / f1_denominator) if f1_denominator else None,
        "average_precision": _safe(_average_precision(probability, target)),
        "roc_auc": _safe(_rank_auc(probability, target)),
        "bce": float(np.mean(-(target * np.log(np.clip(probability, 1e-7, 1.0))
                                + (1 - target) * np.log(np.clip(1 - probability, 1e-7, 1.0))))),
        "brier": float(np.mean((probability - target) ** 2)),
        "calibration_bins": _calibration(probability, target),
    }


@torch.inference_mode()
def _evaluate_split(policy, source, imagined_root, split, chunk_size):
    data, manifest = load_policy_cache(source, split)
    check_policy_encoder(manifest["field_encoder"], policy, True)
    imagined, fingerprints = load_imagined_cache(imagined_root, split, source, data, manifest, policy)
    n = len(data["seeds"])
    probabilities = {name: [] for name in EVENTS}
    targets = {name: [] for name in EVENTS}
    for first in range(0, n, chunk_size):
        last = min(first + chunk_size, n)
        current = torch.as_tensor(np.array(data["fields"][first:last], copy=True), dtype=torch.float32)
        predicted = torch.as_tensor(np.array(imagined[first:last], copy=True), dtype=torch.float32)
        batch = last - first
        current_summary = policy.dynamics.readout.summary(current)[:, None, :].expand(-1, 4, -1).reshape(-1, 96)
        predicted_summary = policy.dynamics.readout.summary(predicted.reshape(-1, 148, 96))
        actions = torch.arange(4, dtype=torch.long).repeat(batch)
        logits = policy.dynamics.event_head(torch.cat((current_summary, predicted_summary,
                                                        policy.dynamics.action_embedding(actions)), -1))
        probability = logits.sigmoid().cpu().numpy().reshape(batch, 4, 3)
        for index, name in enumerate(EVENTS):
            probabilities[name].append(probability[..., index].reshape(-1))
            targets[name].append(np.asarray(data[name][first:last], dtype=np.int8).reshape(-1))
    metrics = {}
    for name in EVENTS:
        probability = np.concatenate(probabilities[name])
        target = np.concatenate(targets[name])
        metrics[name] = _event_metrics(probability, target)
    terminal_failure = np.logical_and(np.asarray(data["terminal"], dtype=bool),
                                      ~np.asarray(data["won"], dtype=bool)).reshape(-1)
    return {
        "levels": n,
        "branches": n * 4,
        "source_root": str(Path(source).resolve()),
        "source_manifest_sha256": digest(Path(source) / "manifest.json"),
        "cache_fingerprints": fingerprints,
        "full_source_and_imagined_arrays_hash_checked_once": True,
        "event_positive_counts": {name: int(np.asarray(data[name], dtype=bool).sum()) for name in EVENTS},
        "terminal_failure_count": int(terminal_failure.sum()),
        "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ls20-factored-local-h4-400.pt"))
    parser.add_argument("--source-cache", type=Path, default=Path("data/structured-field-16384"))
    parser.add_argument("--imagined-cache", type=Path, default=Path("data/structured-policy-imagined-local-h4-400"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/structured-local-h4-400-event-head-evaluation.json"))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error(f"refusing existing report: {args.report}")
    if not 1 <= args.chunk_size <= 1024 or not 1 <= args.seconds <= 120:
        parser.error("chunk-size must be1..1024 and seconds must be1..120")
    torch.set_num_threads(1)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("event-head deadline")))
    signal.alarm(args.seconds)
    report = {
        "status": "running", "pid": os.getpid(), "device": "cpu", "torch_threads": 1,
        "checkpoint": str(args.checkpoint.resolve()), "source_cache": str(args.source_cache.resolve()),
        "imagined_cache": str(args.imagined_cache.resolve()), "chunk_size": args.chunk_size,
        "official_inputs_used": False, "dynamics_recomputed": False, "splits": {},
    }
    try:
        source_hash_before = digest(__file__)
        checkpoint_hash_before = digest(args.checkpoint)
        policy, _, factored = build_training_policy(
            str(args.checkpoint.parents[0] / "ls20-world-cell-recall-b1024.pt"),
            str(args.checkpoint.parents[0] / "ls20-cell-visibility-initial-200.pt"),
            str(args.checkpoint), {"mode": "successors"})
        policy.eval().requires_grad_(False)
        if not factored or policy.training or policy.dynamics.training or any(p.requires_grad for p in policy.parameters()):
            raise ValueError("event evaluation requires a frozen factored policy")
        report.update({
            "checkpoint_sha256": checkpoint_hash_before,
            "parameters": policy.dynamics.parameter_count(),
            "config": policy.dynamics.config(),
            "source_hashes": dict(policy.sources["code_hashes"]),
            "evaluator_sha256_before": source_hash_before,
        })
        for split in ("train", "validation"):
            report["splits"][split] = _evaluate_split(
                policy, args.source_cache / split, args.imagined_cache, split, args.chunk_size)
            report["elapsed_seconds"] = time.monotonic() - started
            _atomic_json(args.report, report)
        if digest(__file__) != source_hash_before or digest(args.checkpoint) != checkpoint_hash_before:
            raise ValueError("source or checkpoint changed during event evaluation")
        report.update(status="complete", source_unchanged=True, checkpoint_unchanged=True,
                      evaluator_sha256_after=digest(__file__),
                      caveats=[
                          "Metrics are generated-cache event-head evidence, not closed-loop gameplay results.",
                          "Terminal-failure targets are reported separately; zero such targets cannot establish third-life or game-over prediction.",
                          "The fixed 0.5 threshold was not fitted on either split.",
                          "Cached FP32 imagined fields were consumed; the H4 dynamics transition was not recomputed.",
                      ])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        _atomic_json(args.report, report)
        print(json.dumps({"status": report["status"], "pid": os.getpid(),
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
