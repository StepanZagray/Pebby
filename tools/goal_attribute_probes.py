"""Probe frozen world-model features at generated goal cells.

Goal coordinates and triples are used only to select diagnostic crops and labels.
They are never passed to the world model or to the probe as input.  This is a
generated-only CPU diagnostic; it does not load the official game levels.
"""

from __future__ import annotations

from pebby.ls20.provenance import difficulty_stages, difficulty_version, validate_difficulty

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pebby.agent.world_model import CELLS, CELL, X_ORIGIN, Y_ORIGIN, load_world_checkpoint
from pebby.ls20.generate import build_level
from pebby.ls20.env import Ls20Scenario


SIZES = (6, 4, 4)
ATTRIBUTES = ("shape", "color", "rotation")
GOAL_CROP_SIZE = 3
GLYPH_HIDDEN = 64
GLYPH_INPUTS = 6 * 6 * 16
GLYPH_CLASSES = sum(SIZES)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_specs(path):
    with Path(path).open() as stream:
        return {int((spec := json.loads(line))["seed"]): spec for line in stream}


def stratified_seeds(specs, count, seed):
    """Choose distinct verified levels with deterministic equal difficulty quotas."""
    if count <= 0:
        raise ValueError("level count must be positive")
    versions = {difficulty_version(spec) for spec in specs.values()}
    if len(versions) > 1:
        raise ValueError("mixed legacy and calibrated difficulty versions")
    stages = difficulty_stages({"difficulty_version": next(iter(versions), None)})
    groups = {difficulty: [] for difficulty in stages}
    for value, spec in specs.items():
        difficulty = validate_difficulty(spec)
        if difficulty not in groups:
            raise ValueError(f"seed {value} has invalid difficulty {difficulty}")
        groups[difficulty].append(int(value))
    if any(not values for values in groups.values()):
        raise ValueError(f"verified bank must contain every difficulty 1..{len(stages)}")
    base, remainder = divmod(count, len(stages))
    quotas = {difficulty: base + int(difficulty <= remainder) for difficulty in groups}
    for difficulty, quota in quotas.items():
        if len(groups[difficulty]) < quota:
            raise ValueError(
                f"difficulty {difficulty} has only {len(groups[difficulty])} levels; "
                f"need {quota} for {count}-level stratified sample")
    rng = np.random.default_rng(seed)
    selected = []
    for difficulty in stages:
        values = np.asarray(groups[difficulty], dtype=np.int64)
        selected.extend(int(value) for value in rng.permutation(values)[:quotas[difficulty]])
    return selected


def canonical_goal_patterns(glyph_data):
    """Recover exact 3x3 goal patterns from generated carried-glyph templates."""
    patterns = {}
    with np.load(glyph_data, allow_pickle=False) as archive:
        glyphs, triples = archive["glyphs"], archive["triples"]
        meta = json.loads(str(archive["meta"].item()))
    if meta.get("source") != "generated_only":
        raise ValueError("glyph template data must be generated-only")
    for glyph, triple in zip(glyphs, triples):
        key = tuple(int(value) for value in triple)
        pattern = glyph[::2, ::2].copy()
        prior = patterns.get(key)
        if prior is not None and not np.array_equal(prior, pattern):
            raise ValueError(f"contradictory canonical glyph pattern for {key}")
        patterns[key] = pattern
    if len(patterns) != 96:
        raise ValueError(f"expected 96 canonical glyph patterns, got {len(patterns)}")
    return patterns


def load_glyph_classifier(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint["weights"]
    classifier = nn.Sequential(nn.Linear(GLYPH_INPUTS, GLYPH_HIDDEN), nn.GELU(),
                               nn.Linear(GLYPH_HIDDEN, GLYPH_CLASSES))
    classifier.load_state_dict(weights, strict=True)
    classifier.eval()
    return classifier, checkpoint


def initial_level_samples(specs, seeds, canonical):
    """Render one initial public frame and visible goal crops per generated level."""
    frames, labels, records = [], [], []
    exclusions = {"partial_fog": 0, "fully_fogged": 0, "noncanonical": 0}
    for seed in seeds:
        spec = specs.get(seed)
        if spec is None:
            raise ValueError(f"seed {seed} missing from verified bank")
        level = build_level(spec)
        env = Ls20Scenario(level, int(spec["training_context_index"]))
        frame = np.asarray(env.reset(), dtype=np.uint8)
        goals = spec["goals"]
        for goal_index, goal in enumerate(goals):
            col, row = (int(goal["cell"][0]), int(goal["cell"][1]))
            left = X_ORIGIN + CELL * col + 1
            top = Y_ORIGIN + CELL * row + 1
            crop = frame[top:top + GOAL_CROP_SIZE, left:left + GOAL_CROP_SIZE]
            triple = tuple(int(value) for value in goal["triple"])
            expected = canonical[triple]
            differing = int(np.count_nonzero(crop != expected))
            record = {"seed": seed, "difficulty": int(spec["difficulty"]),
                      "context_index": int(spec["training_context_index"]),
                      "goal_index": goal_index, "goal_cell": [col, row],
                      "goal_triple": list(triple), "fog": bool(spec.get("fog", False)),
                      "differing_native_pixels": differing}
            if differing:
                key = "fully_fogged" if differing == 9 else "partial_fog"
                exclusions[key] += 1
                record["excluded"] = key
                records.append(record)
                continue
            frames.append(frame)
            labels.append(triple)
            record["excluded"] = None
            records.append(record)
    return np.asarray(frames, dtype=np.uint8), np.asarray(labels, dtype=np.int64), records, exclusions


def feature_batches(model, frames, goal_records, batch_size):
    """Encode padded initial histories and select only the known goal cells."""
    valid_records = [record for record in goal_records if record["excluded"] is None]
    if len(valid_records) != len(frames):
        raise ValueError("valid goal records and frames disagree")
    by_frame = []
    cursor = 0
    for record in goal_records:
        if record["excluded"] is None:
            by_frame.append((cursor, record))
            cursor += 1
    outputs = {"raw_goal": [], "refined_goal": []}
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            current = torch.as_tensor(frames[start:start + batch_size])
            batch = current[:, None].expand(-1, model.cfg.history, -1, -1).contiguous()
            valid = torch.zeros((len(current), model.cfg.history), dtype=torch.bool)
            valid[:, -1] = True
            actions = torch.full_like(valid, -1, dtype=torch.long)
            prepared, valid, actions = model._prepare(batch, valid, actions)
            tokens = model.frame_tokens(prepared.flatten(0, 1)).view(
                len(current), model.cfg.history, model.tokens, -1)
            encoded = model.assemble(tokens, valid, actions)
            selected = by_frame[start:start + len(current)]
            indices = torch.tensor([record["goal_cell"][1] * 12 + record["goal_cell"][0]
                                    for _, record in selected], dtype=torch.long)
            rows = torch.arange(len(current))
            outputs["raw_goal"].append(tokens[rows, -1, indices].cpu())
            outputs["refined_goal"].append(encoded["cells"][rows, indices].cpu())
    return {key: torch.cat(value, dim=0) for key, value in outputs.items()}


def glyph_features(classifier, frames, records):
    features = []
    cursor = 0
    for record in records:
        if record["excluded"] is not None:
            continue
        col, row = record["goal_cell"]
        left = X_ORIGIN + CELL * col + 1
        top = Y_ORIGIN + CELL * row + 1
        crop = torch.as_tensor(frames[cursor, top:top + 3, left:left + 3])
        upsampled = crop.repeat_interleave(2, 0).repeat_interleave(2, 1)
        with torch.inference_mode():
            features.append(classifier(F.one_hot(upsampled.long(), 16).flatten().float()))
        cursor += 1
    return torch.stack(features)


def visible_level_groups(records):
    """Return one index group per selected level, containing visible goal records."""
    groups = {}
    cursor = 0
    for record in records:
        if record["excluded"] is None:
            groups.setdefault(int(record["seed"]), []).append(cursor)
            cursor += 1
    return list(groups.values())


def probe_metrics(scores, labels):
    fields = scores.split(SIZES, -1)
    hits = torch.stack([part.argmax(-1) == labels[:, index]
                        for index, part in enumerate(fields)], dim=-1)
    return {"shape": float(hits[:, 0].float().mean()),
            "color": float(hits[:, 1].float().mean()),
            "rotation": float(hits[:, 2].float().mean()),
            "joint": float(hits.all(-1).float().mean())}


def fit_probe(train, validation, train_labels, validation_labels, kind, seed, steps,
              batch_size, train_level_groups):
    if batch_size <= 0 or batch_size & (batch_size - 1):
        raise ValueError("probe batch size must be a positive power of two")
    if len(train_level_groups) < batch_size:
        raise ValueError(f"need {batch_size} visible train levels, got {len(train_level_groups)}")
    torch.manual_seed(seed)
    mean = train.mean(0)
    scale = train.std(0).clamp_min(1e-5)
    train = (train - mean) / scale
    validation = (validation - mean) / scale
    head = (nn.Linear(train.shape[1], sum(SIZES)) if kind == "linear" else
            nn.Sequential(nn.Linear(train.shape[1], 64), nn.GELU(),
                          nn.Linear(64, sum(SIZES))))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed + 1000)
    for _ in range(steps):
        level_indices = torch.randperm(len(train_level_groups), generator=generator)[:batch_size]
        indices = torch.tensor([
            train_level_groups[int(level)][int(torch.randint(
                len(train_level_groups[int(level)]), (), generator=generator))]
            for level in level_indices
        ], dtype=torch.long)
        if len(torch.unique(level_indices)) != batch_size:
            raise AssertionError("probe batch reused a selected level")
        scores = head(train[indices])
        parts = scores.split(SIZES, -1)
        loss = sum(F.cross_entropy(part, train_labels[indices, index])
                   for index, part in enumerate(parts)) / 3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return {"train_accuracy": probe_metrics(head(train), train_labels),
                "validation_accuracy": probe_metrics(head(validation), validation_labels),
                "final_training_loss": float(loss),
                "parameters": sum(parameter.numel() for parameter in head.parameters()),
                "batch_size": batch_size, "distinct_levels_per_batch": batch_size,
                "sampling": "one random visible goal per distinct selected train level",
                "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ls20-world-mixedpath-b1024.epoch1.pt")
    parser.add_argument("--glyph-checkpoint", default="checkpoints/ls20-glyph-prototype.pt")
    parser.add_argument("--train-bank", default="data/ls20-verified-train.jsonl")
    parser.add_argument("--validation-bank", default="data/ls20-verified-validation.jsonl")
    parser.add_argument("--glyph-data", default="data/ls20-glyph-train.npz")
    parser.add_argument("--out", default="artifacts/world-goal-attribute-probes.json")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("five-minute cap")))
    signal.alarm(300)
    output = Path(args.out)
    paths = [args.checkpoint, args.glyph_checkpoint, args.train_bank, args.validation_bank,
             args.glyph_data, __file__,
             "pebby/agent/world_model.py", "pebby/ls20/generate.py", "pebby/ls20/env.py"]
    report = {"status": "running", "pid": os.getpid(), "device": "cpu", "threads": 1,
              "seed": args.seed, "probe_steps": args.steps,
              "feature_batch_size": args.feature_batch_size,
              "probe_batch_size": args.probe_batch_size,
              "selection": "full verified banks; equal difficulty quotas; true goal coordinates select diagnostic crops only",
              "goal_visibility_criterion": "exact native 3x3 crop equals canonical triple pattern; all differing pixels excluded",
              "official_data_or_routes_used": False,
              "code_snapshot": "artifacts/world-mixedpath-final-code",
              "probe_protocol": {
                  "standardization": "train feature mean and std only; reused unchanged on validation",
                  "feature_batch_size": args.feature_batch_size,
                  "probe_batch_size": args.probe_batch_size,
                  "steps": args.steps,
                  "train_sampling": "each step samples distinct selected train levels without replacement; one random visible goal per selected level",
                  "heads": "linear and MLP with hidden width 64",
              },
              "limitations": [
                  "Generated initial reset frames only; this probes representation and does not establish closed-loop control or unseen-level completion.",
                  "Goal coordinates and triples select diagnostic crops and labels only; they are never model or probe inputs.",
                  "Glyph logits use a known goal location and a classifier trained on 96 generated templates, so they are a perception upper-bound rather than a controller result.",
                  "Only exact canonical native 3x3 goal crops are retained; partial and fully fogged crops are excluded and counts are reported.",
                  "The selected levels are stratified by difficulty from the full verified generated banks; they are disjoint train and validation banks.",
              ],
              "hashes": {path: digest(path) for path in paths}, "results": {}}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)

    persist()
    print("PID", os.getpid(), flush=True)
    try:
        train_specs, validation_specs = read_specs(args.train_bank), read_specs(args.validation_bank)
        train_seeds = stratified_seeds(train_specs, args.train_count, args.seed)
        validation_seeds = stratified_seeds(validation_specs, args.validation_count, args.seed + 1)
        if set(train_seeds) & set(validation_seeds):
            raise ValueError("train and validation diagnostic seeds overlap")
        canonical = canonical_goal_patterns(args.glyph_data)
        train_frames, train_labels, train_records, train_excluded = initial_level_samples(
            train_specs, train_seeds, canonical)
        validation_frames, validation_labels, validation_records, validation_excluded = initial_level_samples(
            validation_specs, validation_seeds, canonical)
        train_visible_groups = visible_level_groups(train_records)
        validation_visible_groups = visible_level_groups(validation_records)
        if len(train_visible_groups) < args.probe_batch_size:
            raise ValueError(
                f"only {len(train_visible_groups)} selected train levels have visible goals; "
                f"need at least {args.probe_batch_size}")

        def level_counts(seeds, specs):
            return {str(difficulty): sum(int(specs[seed]["difficulty"] == difficulty)
                                         for seed in seeds)
                    for difficulty in range(1, 6)}

        def visible_counts(records):
            return {str(difficulty): len({record["seed"] for record in records
                                          if record["excluded"] is None and
                                          record["difficulty"] == difficulty})
                    for difficulty in range(1, 6)}

        def exclusion_counts(records):
            return {str(difficulty): sum(1 for record in records
                                         if record["excluded"] is not None and
                                         record["difficulty"] == difficulty)
                    for difficulty in range(1, 6)}

        report["samples"] = {
            "train_levels": len(train_seeds), "validation_levels": len(validation_seeds),
            "train_goals_total": len(train_records), "validation_goals_total": len(validation_records),
            "train_goals_visible": len(train_labels), "validation_goals_visible": len(validation_labels),
            "train_exclusions": train_excluded, "validation_exclusions": validation_excluded,
            "train_level_counts_by_difficulty": level_counts(train_seeds, train_specs),
            "validation_level_counts_by_difficulty": level_counts(validation_seeds, validation_specs),
            "train_visible_levels": len(train_visible_groups),
            "validation_visible_levels": len(validation_visible_groups),
            "train_visible_level_counts_by_difficulty": visible_counts(train_records),
            "validation_visible_level_counts_by_difficulty": visible_counts(validation_records),
            "train_exclusions_by_difficulty": exclusion_counts(train_records),
            "validation_exclusions_by_difficulty": exclusion_counts(validation_records),
            "minimum_visible_train_levels_required": args.probe_batch_size,
            "train_seeds": train_seeds, "validation_seeds": validation_seeds,
        }
        model, checkpoint = load_world_checkpoint(args.checkpoint, "cpu")
        model.requires_grad_(False)
        report["checkpoint"] = {"parameters": checkpoint.get("parameters"),
                                "epoch": checkpoint.get("epoch", checkpoint.get("best_epoch")),
                                "architecture": model.config().get("architecture"),
                                "loops": model.loops}
        glyph_classifier, glyph_checkpoint = load_glyph_classifier(args.glyph_checkpoint)
        report["glyph_classifier"] = {"parameters": sum(p.numel() for p in glyph_classifier.parameters()),
                                      "format": glyph_checkpoint.get("format"),
                                      "known_templates": 96}
        train_features = feature_batches(model, train_frames, train_records, args.feature_batch_size)
        validation_features = feature_batches(model, validation_frames, validation_records, args.feature_batch_size)
        train_features["glyph_logits"] = glyph_features(glyph_classifier, train_frames, train_records)
        validation_features["glyph_logits"] = glyph_features(glyph_classifier, validation_frames, validation_records)
        report["feature_shapes"] = {key: {"train": list(value.shape), "validation": list(validation_features[key].shape)}
                                     for key, value in train_features.items()}
        report["majority"] = {}
        train_labels_t, validation_labels_t = torch.as_tensor(train_labels), torch.as_tensor(validation_labels)
        for index, (name, size) in enumerate(zip(ATTRIBUTES, SIZES)):
            train_counts = torch.bincount(train_labels_t[:, index], minlength=size)
            validation_counts = torch.bincount(validation_labels_t[:, index], minlength=size)
            majority = int(train_counts.argmax())
            report["majority"][name] = {
                "train_counts": train_counts.tolist(), "validation_counts": validation_counts.tolist(),
                "train_accuracy": float((train_labels_t[:, index] == majority).float().mean()),
                "validation_accuracy": float((validation_labels_t[:, index] == majority).float().mean()),
            }
        del model, glyph_classifier
        persist()
        for name, feature in train_features.items():
            report["results"][name] = {}
            for kind in ("linear", "mlp64"):
                report["results"][name][kind] = fit_probe(
                    feature, validation_features[name], train_labels_t, validation_labels_t,
                    kind, args.seed, args.steps, args.probe_batch_size, train_visible_groups)
                persist()
                print(name, kind, report["results"][name][kind], flush=True)
        report["status"] = "complete"
    except TimeoutError:
        report["status"] = "incomplete_time_cap"
    finally:
        signal.alarm(0)
        persist()
    print("DONE", report["status"], report["elapsed_seconds"], "seconds", report["peak_rss_mib"], "MiB", flush=True)


if __name__ == "__main__":
    main()
