"""Train-only positive-slope Platt calibration for frozen event logits.

The field encoder and H4 dynamics remain frozen.  Current fields and FP32
imagined fields come from the provenance-checked caches; only the six Platt
parameters see event labels during the optional CPU calibration fit.  The
validation split is loaded and scored only after all fixed training updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from pebby.agent.event_calibration import (
    EVENT_NAMES,
    FORMAT,
    PositiveSlopePlatt,
    calibration_bce,
)
from pebby.agent.structured_factored_policy import state_digest
from tools.cache_structured_policy_successors import load_imagined_cache
from tools.evaluate_structured_event_head import _event_metrics
from tools.train_structured_policy import (
    build_training_policy,
    check_policy_encoder,
    digest,
    load_policy_cache,
)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _cache_event_logits(policy, source, imagined_root, split, chunk_size, level_limit=None):
    """Load/check a split, then produce cached-current/cached-successor logits."""
    data, manifest = load_policy_cache(source, split)
    check_policy_encoder(manifest["field_encoder"], policy, True)
    imagined, fingerprints = load_imagined_cache(imagined_root, split, source, data, manifest, policy)
    source_fingerprints = {
        str((Path(source) / "manifest.json").resolve()): digest(Path(source) / "manifest.json")
    }
    for name, info in manifest["arrays"].items():
        source_fingerprints[str((Path(source) / (name + ".npy")).resolve())] = info["sha256"]
    n = len(data["seeds"])
    if level_limit is None:
        level_limit = n
    if not 1 <= level_limit <= n:
        raise ValueError(f"{split} level limit must be1..{n}")
    logits_parts, label_parts = [], []
    with torch.inference_mode():
        for first in range(0, level_limit, chunk_size):
            last = min(first + chunk_size, level_limit)
            current = torch.as_tensor(np.array(data["fields"][first:last], copy=True), dtype=torch.float32)
            predicted = torch.as_tensor(np.array(imagined[first:last], copy=True), dtype=torch.float32)
            batch = last - first
            current_summary = policy.dynamics.readout.summary(current)[:, None, :].expand(-1, 4, -1).reshape(-1, 96)
            predicted_summary = policy.dynamics.readout.summary(predicted.reshape(-1, 148, 96))
            actions = torch.arange(4, dtype=torch.long).repeat(batch)
            event_logits = policy.dynamics.event_head(torch.cat((
                current_summary, predicted_summary,
                policy.dynamics.action_embedding(actions)), -1))
            logits_parts.append(event_logits.cpu().float().numpy().reshape(batch, 4, 3))
            label_parts.append(np.stack([np.asarray(data[name][first:last], dtype=np.float32)
                                         for name in EVENT_NAMES], axis=-1))
    return {
        "logits": np.concatenate(logits_parts, axis=0),
        "labels": np.concatenate(label_parts, axis=0),
        "levels": level_limit,
        "branches": level_limit * 4,
        "source_root": str(Path(source).resolve()),
        "source_manifest_sha256": digest(Path(source) / "manifest.json"),
        "cache_fingerprints": fingerprints,
        "source_cache_fingerprints": source_fingerprints,
        "full_source_and_imagined_arrays_hash_checked_once": True,
        "positive_counts": {name: int(np.asarray(data[name][:level_limit], dtype=bool).sum())
                            for name in EVENT_NAMES},
        "terminal_failure_count": int(np.logical_and(
            np.asarray(data["terminal"][:level_limit], dtype=bool),
            ~np.asarray(data["won"][:level_limit], dtype=bool)).sum()),
        "seed_range": [int(np.asarray(data["seeds"][:level_limit]).min()),
                       int(np.asarray(data["seeds"][:level_limit]).max())],
    }


def _metrics(logits, labels):
    result = {}
    for index, name in enumerate(EVENT_NAMES):
        result[name] = _event_metrics(torch.sigmoid(torch.as_tensor(logits[..., index])).numpy().reshape(-1),
                                      labels[..., index].astype(np.int8).reshape(-1))
    return result


def _coherent_metrics(logits, labels, calibrator):
    """Report conditional-win and normalized joint outcome metrics."""
    raw = torch.as_tensor(logits, dtype=torch.float32)
    target = np.asarray(labels, dtype=np.int8)
    with torch.inference_mode():
        probability = calibrator.coherent_probabilities(raw)
        calibrated = calibrator.probabilities(raw)
    loss = probability["loss"].numpy()
    conditional_win = probability["conditional_win"].numpy()
    joint_win = probability["win"].numpy()
    no_loss = target[..., 0] < 1
    return {
        "lost_life": _event_metrics(loss.reshape(-1), target[..., 0].reshape(-1)),
        "terminal_diagnostic": _event_metrics(
            calibrated[..., 1].numpy().reshape(-1), target[..., 1].reshape(-1)),
        "won_conditional_no_life_loss": _event_metrics(
            conditional_win[no_loss].reshape(-1), target[..., 2][no_loss].reshape(-1)),
        "won_joint_unconditional": _event_metrics(
            joint_win.reshape(-1), target[..., 2].reshape(-1)),
        "coherent_probability_checks": {
            "max_sum_error": float(np.max(np.abs(loss + joint_win + probability["continue"].numpy() - 1.))),
            "min_loss": float(loss.min()), "max_loss": float(loss.max()),
            "min_conditional_win": float(conditional_win.min()),
            "max_conditional_win": float(conditional_win.max()),
            "conditional_win_examples": int(no_loss.sum()),
            "conditional_win_positive": int(target[..., 2][no_loss].sum()),
        },
    }


def _binding(policy, checkpoint, source_cache, imagined_cache, split_data, tool_path):
    return {
        "format": FORMAT,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "source_cache": str(Path(source_cache).resolve()),
        "imagined_cache": str(Path(imagined_cache).resolve()),
        "splits": {split: {key: value for key, value in split_data[split].items()
                            if key in ("source_root", "source_manifest_sha256", "cache_fingerprints",
                                       "source_cache_fingerprints", "levels", "branches", "positive_counts", "terminal_failure_count",
                                       "seed_range")}
                   for split in ("train", "validation")},
        "source_hashes": split_data["source_hashes"],
    }


def _save_checkpoint(path, calibrator, binding, fit):
    payload = {
        "format": FORMAT,
        "parameters": 6,
        "event_names": list(EVENT_NAMES),
        "config": {"initial_slope": 1.0, "positive_slope": True,
                   "conditional_won": True,
                   "terminal_semantics": "diagnostic_all_branches",
                   "coherent_outcomes": ["loss", "win", "continue"]},
        "source_binding": binding,
        "fit": fit,
        "state_dict": {key: value.detach().cpu() for key, value in calibrator.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _assert_hashes_unchanged(hashes):
    changed = [path for path, expected in hashes.items()
               if digest(Path(path)) != expected]
    if changed:
        raise ValueError(f"source/cache changed during calibration: {changed[:3]}")
    return True


def _expected_cache_hashes(source_cache, imagined_cache):
    """Snapshot provenance metadata without loading validation examples."""
    hashes = {}
    paths = [Path(source_cache) / split / 'manifest.json' for split in ('train', 'validation')]
    paths += [Path(imagined_cache) / 'manifest.json']
    paths += [Path(imagined_cache) / split / 'manifest.json' for split in ('train', 'validation')]
    for path in paths:
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        for name, info in manifest.get('arrays', {}).items():
            hashes[str((path.parent / (name + '.npy')).resolve())] = info['sha256']
    return hashes


def fit_train_only(train, *, updates=200, batch_levels=1024, lr=0.02, seed=20260912):
    if updates < 1 or batch_levels < 1 or batch_levels > train["levels"]:
        raise ValueError("invalid fixed update/batch schedule")
    if not 0 < lr:
        raise ValueError("learning rate must be positive")
    if batch_levels & (batch_levels - 1) or batch_levels > 1024:
        raise ValueError("batch_levels must be a power of two no greater than1024")
    if not math.isfinite(float(lr)) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    torch.manual_seed(seed)
    calibrator = PositiveSlopePlatt().train()
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=lr, weight_decay=0.)
    logits = torch.as_tensor(train["logits"], dtype=torch.float32)
    labels = torch.as_tensor(train["labels"], dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    curve = []
    for update in range(updates):
        levels = torch.randperm(train["levels"], generator=generator)[:batch_levels]
        batch_logits = logits[levels].reshape(-1, 3)
        batch_labels = labels[levels].reshape(-1, 3)
        # Fit loss on every life-loss target and on win targets only when the
        # actual branch did not lose a life. Terminal is diagnostic-only in
        # the current data but remains an all-branch calibration parameter.
        batch_mask = torch.ones_like(batch_labels, dtype=torch.bool)
        batch_mask[:, 2] = batch_labels[:, 0] < 0.5
        optimizer.zero_grad(set_to_none=True)
        loss = calibration_bce(calibrator, batch_logits, batch_labels, batch_mask)
        loss.backward()
        optimizer.step()
        if update == 0 or (update + 1) % max(1, updates // 10) == 0 or update + 1 == updates:
            curve.append({"update": update + 1, "bce": float(loss.detach()),
                          "slopes": calibrator.slope().detach().tolist(),
                          "intercepts": calibrator.intercept.detach().tolist()})
    calibrator.eval().requires_grad_(False)
    return calibrator, {
        "updates": updates, "batch_levels": batch_levels, "branches_per_update": batch_levels * 4,
        "optimizer": "Adam", "learning_rate": lr, "weight_decay": 0., "seed": seed,
        "loss": "unweighted BCE; lost_life and terminal all branches, won conditional on no actual life loss; TRAIN only",
        "sampling": "distinct level rows without replacement inside each update; all four branches per level",
        "curve": curve,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ls20-factored-local-h4-400.pt"))
    parser.add_argument("--source-cache", type=Path, default=Path("data/structured-field-16384"))
    parser.add_argument("--imagined-cache", type=Path, default=Path("data/structured-policy-imagined-local-h4-400"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/structured-event-calibration.json"))
    parser.add_argument("--out-checkpoint", type=Path)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-levels", type=int, default=1024)
    parser.add_argument("--train-levels", type=int)
    parser.add_argument("--validation-levels", type=int)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args(argv)
    if args.report.exists() or (args.out_checkpoint is not None and args.out_checkpoint.exists()):
        parser.error("refusing to overwrite calibration outputs")
    if not 1 <= args.updates <= 200 or not 1 <= args.batch_levels <= 1024 \
            or args.batch_levels & (args.batch_levels - 1):
        parser.error("updates must be1..200 and batch-levels must be a power of two <=1024")
    if args.train_levels is not None and args.train_levels < args.batch_levels:
        parser.error("train-levels must contain at least one complete batch")
    if args.chunk_size <= 0:
        parser.error("chunk-size must be positive")
    if not math.isfinite(float(args.lr)) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    torch.set_num_threads(1)
    started = time.monotonic()
    checkpoint_sha = digest(args.checkpoint)
    policy, _, factored = build_training_policy(
        str(args.checkpoint.parents[0] / "ls20-world-cell-recall-b1024.pt"),
        str(args.checkpoint.parents[0] / "ls20-cell-visibility-initial-200.pt"),
        str(args.checkpoint), {"mode": "successors"})
    policy.eval().requires_grad_(False)
    if not factored or any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("calibration requires a frozen factored checkpoint")
    report = {"status": "running", "pid": os.getpid(), "device": "cpu", "torch_threads": 1,
              "official_inputs_used": False, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": checkpoint_sha, "parameters": policy.dynamics.parameter_count(),
              "updates_requested": args.updates, "batch_levels": args.batch_levels, "lr": args.lr,
              "validation_used_for_fit_or_selection": False, "splits": {}}
    try:
        source_hashes = {}
        for path, sha in policy.sources["code_hashes"].items():
            source_hashes[str(Path(path).resolve())] = sha
        for record in policy.sources["artifacts"].values():
            source_hashes[str(Path(record["path"]).resolve())] = record["sha256"]
        source_hashes[str(Path(args.checkpoint).resolve())] = checkpoint_sha
        for path in (Path(__file__), Path("pebby/agent/event_calibration.py"),
                     Path("tools/evaluate_structured_event_head.py"),
                     Path("tools/train_structured_policy.py"),
                     Path("tools/cache_structured_policy_successors.py")):
            source_hashes[str(path.resolve())] = digest(path)
        source_hashes.update(_expected_cache_hashes(args.source_cache, args.imagined_cache))
        model_state_hashes = {
            "encoder": state_digest(policy.encoder.state_dict()),
            "dynamics": state_digest(policy.dynamics.state_dict()),
        }
        report["source_snapshot_captured_before_cache_processing"] = True
        train = _cache_event_logits(policy, args.source_cache / "train", args.imagined_cache, "train",
                                    args.chunk_size, args.train_levels)
        report['validation_examples_loaded_after_fit'] = True
        identity = PositiveSlopePlatt().eval().requires_grad_(False)
        report["metrics_before"] = {"train": {
            "raw_event_logits": _metrics(train["logits"], train["labels"]),
            "coherent_unfitted": _coherent_metrics(train["logits"], train["labels"], identity),
        }}
        calibrator, fit = fit_train_only(train, updates=args.updates, batch_levels=args.batch_levels,
                                          lr=args.lr, seed=args.seed)
        report["fit"] = fit
        report["metrics_after"] = {"train": _coherent_metrics(
            train["logits"], train["labels"], calibrator)}
        validation = _cache_event_logits(policy, args.source_cache / "validation", args.imagined_cache,
                                         "validation", args.chunk_size, args.validation_levels)
        split_metadata = {"train": train, "validation": validation}
        report["splits"] = {split: {key: value for key, value in data.items()
                                    if key != "logits" and key != "labels"}
                            for split, data in split_metadata.items()}
        for data in split_metadata.values():
            checked = dict(data['cache_fingerprints']) | data['source_cache_fingerprints']
            for path, sha in checked.items():
                path = str(Path(path).resolve())
                if path not in source_hashes or source_hashes[path] != sha:
                    raise ValueError('cache provenance changed since initial snapshot')
        report["metrics_before"]["validation"] = {
            "raw_event_logits": _metrics(validation["logits"], validation["labels"]),
            "coherent_unfitted": _coherent_metrics(
                validation["logits"], validation["labels"], PositiveSlopePlatt()),
        }
        report["metrics_after"]["validation"] = _coherent_metrics(
            validation["logits"], validation["labels"], calibrator)
        _assert_hashes_unchanged(source_hashes)
        current_state_hashes = {
            "encoder": state_digest(policy.encoder.state_dict()),
            "dynamics": state_digest(policy.dynamics.state_dict()),
        }
        if current_state_hashes != model_state_hashes:
            raise ValueError("frozen encoder/dynamics weights changed during calibration")
        binding = _binding(policy, args.checkpoint, args.source_cache, args.imagined_cache,
                           {"train": train, "validation": validation, "source_hashes": source_hashes},
                           Path(__file__))
        report.update({"source_binding": binding,
                       "source_unchanged": True,
                       "frozen_model_state_unchanged": True,
                       "calibration": {"format": FORMAT, "parameters": 6,
                                       "conditional_won": True,
                                       "terminal_semantics": "diagnostic_all_branches",
                                       "coherent_outcomes": ["loss", "win", "continue"],
                                       "slopes": calibrator.slope().tolist(),
                                       "intercepts": calibrator.intercept.tolist(),
                                       "ranking_preserved_by_positive_slopes": bool((calibrator.slope() > 0).all())},
                       "limitations": [
                           "TRAIN-only unweighted BCE fit; validation is scored after the fixed updates and never selects parameters.",
                           "TRAIN and validation have different event priors; this calibration does not correct a future deployment prior shift.",
                           "Both source splits contain zero terminal-failure branches, so no third-life/game-over calibration evidence exists.",
                           "Terminal is an independent diagnostic head; coherent loss/win/continue excludes terminal because no terminal-failure targets exist.",
                           "Calibration changes event probabilities only; the event head, fields, and policy are frozen.",
                       ]})
        if args.out_checkpoint is not None:
            _save_checkpoint(args.out_checkpoint, calibrator, binding, fit)
            report["out_checkpoint"] = str(args.out_checkpoint.resolve())
            report["out_checkpoint_sha256"] = digest(args.out_checkpoint)
        report.update(status="complete", elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        _atomic_json(args.report, report)
        print(json.dumps({"status": report["status"], "pid": os.getpid(),
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
