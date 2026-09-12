"""CPU-only generated evidence runner for a factored H4 checkpoint.

The runner intentionally requires an explicit command-line readiness token.  A
checkpoint appearing on disk is not sufficient to start evaluation.  It loads
one split at a time, scores transition/readout metrics and changed-vs-copy
glyph metrics, and binds every cache/checkpoint/code hash used by the report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.structured_global_glyph import GLOBAL_GLYPH_FORMAT, GlobalGlyphTransition
from pebby.agent.structured_local_glyph import LOCAL_GLYPH_FORMAT, LocalGlobalGlyphTransition
from pebby.agent.structured_transition import StructuredTransition
from tools.structured_glyph_diagnostics import GlyphDiagnosticAccumulator, LOGIT_KEYS
from tools.structured_sequence_metrics import evaluate
from tools.train_structured_mixed_sequences import load_exploratory_cache
from tools.train_structured_sequences import load_cache
from tools.train_structured_transition import atomic_json, digest


READY_TOKEN = "ROOT_EXPLICIT_CHECKPOINT_READY"
BASE_FORMAT = "pebby.structured-transition.v1"
HORIZONS = 4
REQUIRED_ARRAYS = (
    "fields", "next_fields", "actions", "triple", "next_player_cell",
    "next_triple", "next_steps", "next_lives", "lost_life", "terminal", "won",
    "seeds", "difficulties",
)
MODEL_SOURCES = (
    "pebby/agent/structured_transition.py",
    "pebby/agent/structured_global_glyph.py",
    "pebby/agent/structured_local_glyph.py",
    "pebby/agent/structured_sequence_objective.py",
    "tools/structured_sequence_metrics.py",
    "tools/structured_glyph_diagnostics.py",
    "tools/train_structured_sequences.py",
    "tools/train_structured_mixed_sequences.py",
)


def _load_model(raw):
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    fmt = saved.get("format")
    if fmt == LOCAL_GLYPH_FORMAT:
        model = LocalGlobalGlyphTransition(saved["config"])
    elif fmt == GLOBAL_GLYPH_FORMAT:
        model = GlobalGlyphTransition(saved["config"])
    elif fmt == BASE_FORMAT:
        model = StructuredTransition(saved["config"])
    else:
        raise ValueError(f"unsupported factored checkpoint format: {fmt!r}")
    model.load_state_dict(saved["weights"], strict=True)
    model.eval()
    if saved.get("parameters") != model.parameter_count():
        raise ValueError("checkpoint parameter count does not match loaded model")
    if saved.get("official_inputs_used") is not False:
        raise ValueError("checkpoint provenance permits official inputs")
    return model, saved


def _cache_hashes(path, manifest):
    path = Path(path)
    result = {str(path / "manifest.json"): digest(path / "manifest.json")}
    for name, info in manifest["arrays"].items():
        source = path / f"{name}.npy"
        if digest(source) != info["sha256"]:
            raise ValueError(f"cache array changed or manifest is stale: {source}")
        result[str(source)] = info["sha256"]
    return result


def _selected_data(data, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) < 1:
        raise ValueError("selection must be a nonempty row vector")
    # Keep only arrays consumed by the scorer and conditional glyph utility;
    # this bounds memory to the selected split rather than the whole bank.
    return {name: np.array(data[name][rows], copy=True) for name in REQUIRED_ARRAYS}


def _stratified_rows(data, count, seed):
    if count != 1024:
        raise ValueError("this evidence protocol requires exactly 1024 exploratory rows")
    difficulties = np.asarray(data["difficulties"])
    rng = np.random.default_rng(seed)
    quotas = np.full(5, count // 5, dtype=np.int64)
    quotas[:count % 5] += 1
    selected = []
    for difficulty, quota in enumerate(quotas, 1):
        pool = np.flatnonzero(difficulties == difficulty)
        if len(pool) < quota:
            raise ValueError(f"exploratory difficulty {difficulty} has only {len(pool)} rows")
        selected.append(np.sort(rng.choice(pool, size=int(quota), replace=False)))
    rows = np.concatenate(selected)
    if len(rows) != count or len(np.unique(rows)) != count:
        raise RuntimeError("stratified exploratory selection is not unique")
    return rows


def _field_glyph_readout(fields, mode):
    if mode == "token0":
        probabilities = fields[:, 0, 70:84].float()
    elif mode == "board_mean":
        probabilities = fields[:, :144, 70:84].float().mean(1)
    else:
        raise ValueError(f"unknown field glyph view: {mode}")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("field carried probabilities are nonfinite")
    if bool((probabilities < -5e-3).any()) or bool((probabilities > 1.005).any()):
        raise ValueError("field carried probabilities outside [0,1]")
    groups = probabilities.split((6, 4, 4), -1)
    if any(bool((group.sum(-1) - 1.).abs().max() > 5e-2) for group in groups):
        raise ValueError("field carried groups are not normalized probabilities")
    logits = probabilities.clamp_min(1e-8).log()
    shape, color, rotation = logits.split((6, 4, 4), -1)
    return dict(zip(LOGIT_KEYS, (shape, color, rotation)))


def _conditional_glyph(model, data, batch_size):
    names = (
        "predicted_readout", "actual_target_readout", "predicted_field_token0",
        "predicted_field_board_mean", "actual_target_field_token0",
        "actual_target_field_board_mean", "initial_copy_field_token0",
        "initial_copy_field_board_mean",
    )
    accumulators = {name: [GlyphDiagnosticAccumulator() for _ in range(HORIZONS)]
                   for name in names}
    clamp_counts = {name: [0] * HORIZONS for name in names
                    if "field" in name}
    with torch.inference_mode():
        for begin in range(0, len(data["seeds"]), batch_size):
            end = min(begin + batch_size, len(data["seeds"]))
            field = torch.as_tensor(np.array(data["fields"][begin:end], copy=True)).float()
            target_field = torch.as_tensor(
                np.array(data["next_fields"][begin:end], copy=True)).float()
            actions = torch.as_tensor(np.array(data["actions"][begin:end], copy=True)).long()
            output = model.rollout(field, actions)
            predicted_field = output["fields"]
            triples = torch.as_tensor(np.array(data["next_triple"][begin:end], copy=True)).long()
            current_triple = torch.as_tensor(
                np.array(data["triple"][begin:end], copy=True)).long()
            previous_actual = torch.cat((current_triple[:, None], triples[:, :-1]), 1)
            for horizon in range(HORIZONS):
                predicted_readout = {
                    key: value[:, horizon] for key, value in output["readout"].items()
                }
                actual_readout = model.readout(target_field[:, horizon])
                views = {
                    "predicted_readout": predicted_readout,
                    "actual_target_readout": actual_readout,
                    "predicted_field_token0": _field_glyph_readout(
                        predicted_field[:, horizon], "token0"),
                    "predicted_field_board_mean": _field_glyph_readout(
                        predicted_field[:, horizon], "board_mean"),
                    "actual_target_field_token0": _field_glyph_readout(
                        target_field[:, horizon], "token0"),
                    "actual_target_field_board_mean": _field_glyph_readout(
                        target_field[:, horizon], "board_mean"),
                    "initial_copy_field_token0": _field_glyph_readout(field, "token0"),
                    "initial_copy_field_board_mean": _field_glyph_readout(field, "board_mean"),
                }
                for name, readout in views.items():
                    accumulators[name][horizon].update(
                        readout, previous_actual[:, horizon], triples[:, horizon])
                for name in clamp_counts:
                    source = views[name]
                    # Each grouped log-probability view applies the same
                    # diagnostic floor; count rows protected by it.
                    clamp_counts[name][horizon] += sum(
                        int((value.exp() < 1e-8).sum())
                        for value in source.values())
    return {
        "sources": {name: [acc.summary() for acc in values]
                    for name, values in accumulators.items()},
        "clamp_counts_by_horizon": clamp_counts,
        "notes": [
            "Changed/unchanged masks compare each target to the previous ACTUAL triple: current at H1, then actual H1..H3.",
            "Field views are diagnostic grouped probabilities; their CE-like log conversion is only for argmax scoring.",
            "No public-role partition is reported: learned role argmax is not an engine-mechanics label and may assign low-floor cells to launcher.",
        ],
    }


def _split_report(model, data, saved, batch_size):
    scale = saved["feature_scale"]
    metrics = evaluate(model, data, scale, device="cpu", batch_size=batch_size)
    target_events = {name: int(np.asarray(data[name], dtype=bool).sum())
                     for name in ("lost_life", "terminal", "won")}
    target_events["terminal_failure"] = int(
        (np.asarray(data["terminal"], dtype=bool)
         & ~np.asarray(data["won"], dtype=bool)).sum())
    return metrics | {
        "target_event_counts": target_events,
        "conditional_glyph": _conditional_glyph(model, data, batch_size),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exploratory-cache", type=Path, required=True)
    parser.add_argument("--closing-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exploratory-seed", type=int, default=20260912)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--ready-ack", required=True,
                        help=f"must equal {READY_TOKEN!r}; checkpoint existence is not readiness")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.ready_ack != READY_TOKEN:
        raise SystemExit("refusing evaluation without the explicit root checkpoint-ready acknowledgement")
    if args.report.exists():
        raise FileExistsError(args.report)
    if not 1 <= args.batch_size <= 128 or not 1 <= args.seconds <= 180:
        raise SystemExit("batch size must be 1..128 and timeout must be 1..180 seconds")
    torch.set_num_threads(1)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("factored H4 evidence timeout")))
    signal.alarm(args.seconds)
    report = {
        "status": "running", "pid": os.getpid(), "device": "cpu", "torch_threads": 1,
        "checkpoint": str(args.checkpoint), "official_inputs_used": False,
        "selection_protocol": {
            "exploratory_rows": 1024, "exploratory_seed": args.exploratory_seed,
            "exploratory_difficulty_quotas": [205, 205, 205, 205, 204],
            "closing_rows": 2000, "validation_rows": 128,
        },
        "limitations": [
            "This is generated-only evidence, not official-level evaluation.",
            "The selected caches contain zero reset and terminal-failure rows; those outcomes are unmeasured.",
            "Conditional role partitioning is omitted because learned public role argmax is not engine ground truth.",
            "Targets and actual next fields are metric-only; model inputs are current public fields and actions.",
        ],
    }
    atomic_json(args.report, report)
    try:
        raw = args.checkpoint.read_bytes()
        checkpoint_sha = hashlib.sha256(raw).hexdigest()
        model, saved = _load_model(raw)
        caches = {}
        manifests = {}
        for name, path, split in (
                ("exploratory", args.exploratory_cache, "exploratory"),
                ("closing", args.closing_cache, "train"),
                ("validation", args.validation_cache, "validation")):
            if name == "exploratory":
                caches[name], manifests[name] = load_exploratory_cache(path)
            else:
                caches[name], manifests[name] = load_cache(path, split)
            if manifests[name]["field_encoder"] != manifests["exploratory"]["field_encoder"]:
                raise ValueError("evaluation cache field encoders differ")
        if len(caches["exploratory"]["seeds"]) < 1024:
            raise ValueError("exploratory cache has fewer than 1024 rows")
        if len(caches["closing"]["seeds"]) != 2000:
            raise ValueError("closing cache must contain exactly 2000 rows")
        if len(caches["validation"]["seeds"]) != 128:
            raise ValueError("validation cache must contain exactly 128 rows")
        if np.intersect1d(caches["exploratory"]["seeds"], caches["validation"]["seeds"]).size:
            raise ValueError("exploratory/validation seed leakage")
        if np.intersect1d(caches["closing"]["seeds"], caches["validation"]["seeds"]).size:
            raise ValueError("closing/validation seed leakage")
        for manifest in manifests.values():
            if manifest.get("source") != "generated_only":
                raise ValueError("evaluation cache is not generated-only")
        rows = {
            "exploratory": _stratified_rows(caches["exploratory"], 1024,
                                             args.exploratory_seed),
            "closing": np.arange(2000, dtype=np.int64),
            "validation": np.arange(128, dtype=np.int64),
        }
        selected = {name: _selected_data(caches[name], chosen)
                    for name, chosen in rows.items()}
        source_hashes = {str(args.checkpoint): checkpoint_sha}
        for name, path in (("exploratory", args.exploratory_cache),
                           ("closing", args.closing_cache),
                           ("validation", args.validation_cache)):
            source_hashes.update(_cache_hashes(path, manifests[name]))
        for source in MODEL_SOURCES + (__file__,):
            source_hashes[str(source)] = digest(source)
        declared = saved.get("sources", {})
        checked_declared = {}
        for source, expected in declared.items():
            source_path = Path(source)
            if not source_path.exists():
                raise ValueError(f"checkpoint declared missing source: {source}")
            actual = digest(source_path)
            if actual != expected:
                raise ValueError(f"checkpoint declared source changed: {source}")
            checked_declared[source] = actual
        report.update(
            checkpoint_format=saved["format"], parameters=model.parameter_count(),
            checkpoint_sha256=checkpoint_sha, cache_field_encoder=manifests["exploratory"]["field_encoder"],
            source_hashes=source_hashes, checkpoint_declared_sources=checked_declared,
            selected_seeds={name: [int(value) for value in caches[name]["seeds"][chosen]]
                            for name, chosen in rows.items()},
            selected_difficulty_counts={name: np.bincount(
                caches[name]["difficulties"][chosen], minlength=6)[1:6].astype(int).tolist()
                for name, chosen in rows.items()},
        )
        atomic_json(args.report, report)
        report["splits"] = {}
        for name in ("validation", "exploratory", "closing"):
            report["splits"][name] = _split_report(model, selected[name], saved, args.batch_size)
            atomic_json(args.report, report)
            del selected[name]
            gc.collect()
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
            raise ValueError("checkpoint changed during evaluation")
        if any(digest(Path(path)) != expected for path, expected in source_hashes.items()
               if Path(path).exists()):
            raise ValueError("evaluation source changed during scoring")
        report.update(status="complete", source_unchanged=True,
                      elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.report, report)


if __name__ == "__main__":
    main()
