#!/usr/bin/env python3
"""Score public appearance immediately around real generated life resets.

This is a small companion to ``audit_cell_appearance_dynamics.py``.  It uses
only ``env.perform`` on ten fixed generated validation levels, stopping at the
first observed life loss on each level.  Engine state supplies diagnostic
labels and masks; the frozen decoder sees only freshly rendered frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pebby.agent.cell_appearance import CellAppearance, FORMAT
from tools.audit_cell_appearance_dynamics import (
    ACTION_IDS,
    capture_state,
    digest,
    export_public_frames,
    infer,
    is_terminal,
    load_specs,
    make_env,
    snapshot,
    summarize,
    transition,
)


def select_probe_seeds(seeds: list[int], specs: dict[int, dict]) -> list[int]:
    by_difficulty = defaultdict(list)
    for seed in sorted(seeds):
        by_difficulty[int(specs[seed]["difficulty"])].append(seed)
    selected = [seed for difficulty in sorted(by_difficulty) for seed in by_difficulty[difficulty][:2]]
    if len(selected) != 10:
        raise ValueError(f"expected two levels for each of five difficulties, got {selected}")
    return selected


def replay_reset(specs: dict[int, dict], seeds: list[int]):
    rng = np.random.default_rng(20260912)
    samples = []
    levels = []
    for seed in seeds:
        spec = specs[seed]
        env = make_env(spec)
        initial_lives = int(env.lives())
        actions = 0
        pre = None
        post = None
        life_transition = None
        while actions < 500 and not is_terminal(env):
            action = int(rng.choice(np.asarray(ACTION_IDS, dtype=np.int64)))
            before = snapshot(env, spec)
            pre = capture_state(env, spec, "reset_pre_action", actions, action)
            env.perform(action)
            actions += 1
            trans = transition(before, env)
            if trans["life_lost"]:
                life_transition = trans
                post = capture_state(env, spec, "reset_post_action", actions, action, trans)
                break
        if post is None or not life_transition or not life_transition["life_lost"]:
            raise RuntimeError(f"no life loss within 500 actions for reset probe seed {seed}")
        samples.extend((pre, post))
        levels.append({
            "seed": seed,
            "difficulty": int(spec["difficulty"]),
            "actions_to_first_life_loss": actions,
            "initial_lives": initial_lives,
            "lives_before": int(pre["lives"]),
            "lives_after": int(post["lives"]),
            "player_before": list(pre["player_cell"]),
            "player_after": list(post["player_cell"]),
            "reset_to_start": bool(tuple(post["player_cell"]) == tuple(spec["start"])),
            "fog_before": bool(pre["fog"]),
            "fog_after": bool(post["fog"]),
            "support7_before": int(pre["support7"].sum()),
            "support7_after": int(post["support7"].sum()),
            "role_surface_changed_cells": int(np.count_nonzero(pre["roles"] != post["roles"])),
            "object_surface_changed_cells": int(np.count_nonzero(
                (pre["roles"] & 127) != (post["roles"] & 127))),
            "frame_changed_pixels": int(np.count_nonzero(pre["frame"] != post["frame"])),
            "goal_status_before": [item["status"] for item in pre["goal_status"]],
            "goal_status_after": [item["status"] for item in post["goal_status"]],
        })
    return samples, levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cell-appearance-2k-400.pt"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/ls20-visible-cell-labels.npz"))
    parser.add_argument("--validation-bank", type=Path, default=Path("data/ls20-verified-validation.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cell-appearance-reset-audit.json"))
    parser.add_argument("--public-export", type=Path,
                        default=Path("data/cell-appearance-reset-public-frames.npz"))
    args = parser.parse_args()
    if args.out.exists() or args.public_export.exists():
        raise FileExistsError("reset audit output already exists; choose a new path or remove the owned artifact")
    torch.set_num_threads(1)
    print("PID", os.getpid(), flush=True)

    with np.load(args.validation_data, allow_pickle=False) as archive:
        seeds = archive["seeds"][archive["split"] == "validation"].astype(np.int64).tolist()
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError(f"expected fixed100 validation seeds, got {len(seeds)}")
    if any(seed < 1_000_000 or seed >= 2_000_000 for seed in seeds):
        raise ValueError("reset probe requires the verified validation seed range")
    specs = load_specs(args.validation_bank, set(map(int, seeds)))
    probe_seeds = select_probe_seeds(list(map(int, seeds)), specs)
    paths = [args.checkpoint, args.validation_data, args.validation_bank, Path(__file__),
             Path("tools/audit_cell_appearance_dynamics.py"), Path("pebby/agent/cell_appearance.py"),
             Path("pebby/ls20/env.py"), Path("pebby/ls20/generate.py"), Path("pebby/ls20/names.py"),
             Path("third_party/ls20/ls20.py")]
    hashes_before = {str(path): digest(path) for path in paths}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("unsupported appearance checkpoint")
    architecture_hash = digest(Path("pebby/agent/cell_appearance.py"))
    if checkpoint.get("source_hashes", {}).get("pebby/agent/cell_appearance.py") != architecture_hash:
        raise ValueError("checkpoint architecture provenance mismatch")
    model = CellAppearance()
    model.load_state_dict(checkpoint["weights"])
    model.eval()

    samples, levels = replay_reset(specs, probe_seeds)
    predictions = infer(model, samples)
    categories = {
        "all_reset_frames": list(range(len(samples))),
        "pre_reset": [i for i, sample in enumerate(samples) if sample["collection"] == "reset_pre_action"],
        "post_reset": [i for i, sample in enumerate(samples) if sample["collection"] == "reset_post_action"],
        "fog_frames": [i for i, sample in enumerate(samples) if sample["fog"]],
        "player_overlap_frames": [i for i, sample in enumerate(samples) if sample["player_overlap"]],
    }
    metrics = {name: summarize(samples, predictions, indices) for name, indices in categories.items()}
    if any(digest(Path(path)) != value for path, value in hashes_before.items()):
        raise ValueError("reset audit source changed during run")
    public_export_sha256 = export_public_frames(samples, args.public_export)
    report = {
        "status": "complete",
        "format": "pebby.cell-appearance-reset-audit.v1",
        "training_performed": False,
        "device": "cpu",
        "cpu_threads": 1,
        "checkpoint_parameters": model.parameter_count(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashes_before[str(args.checkpoint)],
        "source_hashes": hashes_before,
        "fixed_validation": {
            "levels": 100,
            "seed_count": len(seeds),
            "selection_source": str(args.validation_data),
            "engine_bank": str(args.validation_bank),
            "probe_seeds": probe_seeds,
            "probe_seed": 20260912,
        },
        "sample_counts": {"pre_reset": len(categories["pre_reset"]),
                          "post_reset": len(categories["post_reset"]),
                          "total": len(samples)},
        "metrics_by_state_category": metrics,
        "levels": levels,
        "reset_effects": {
            "all_reset_to_start": all(level["reset_to_start"] for level in levels),
            "support7_before_total": sum(level["support7_before"] for level in levels),
            "support7_after_total": sum(level["support7_after"] for level in levels),
            "frame_changed_pixels_total": sum(level["frame_changed_pixels"] for level in levels),
            "role_surface_changed_cells_total": sum(level["role_surface_changed_cells"] for level in levels),
            "object_surface_changed_cells_total": sum(level["object_surface_changed_cells"] for level in levels),
            "fog_before_levels": sum(level["fog_before"] for level in levels),
            "fog_after_levels": sum(level["fog_after"] for level in levels),
        },
        "public_export": {
            "path": str(args.public_export),
            "sha256": public_export_sha256,
            "rows": len(samples),
            "schema": "pebby.cell-appearance-dynamic-public.v1",
            "contains_engine_labels": False,
        },
        "limitations": [
            "Generated validation only; no official layouts, frames, routes, or labels.",
            "Ten stratified levels and one deterministic random walk per level are a reset-appearance probe, not policy evaluation.",
            "Frames before and after the first actual life loss are scored; no synthetic engine state was injected.",
            "Engine labels and current-player/fog masks are diagnostic only and are never model inputs.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"levels": len(levels), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
