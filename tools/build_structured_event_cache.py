"""Encode the generated event-transition archive with the frozen H8 field encoder.

The event collector has a deliberately different contract from the ordinary
world-training NPZ: it keeps repeated pre-loss rows, permits ``optimal == 0``
for a doomed current state, and names its source columns ``seed`` and
``next_distance``.  This adapter therefore has its own cache format and never
passes the archive through ``world_train.load_dataset`` or the one-row-per-level
cache loader.

The four successor slots are action alternatives (0, 1, 2, 3), not four
chronological actions.  The cache is suitable for one-step/event training and
diagnostics.  It must not be presented to the H4 chronological objective as a
sequence without a separate trajectory collector.
"""

from __future__ import annotations

from pebby.ls20.provenance import generated_context, validate_difficulty, row_contexts, cache_difficulty_metadata

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from tools.build_structured_field_cache import digest, encode_fields


FORMAT = "pebby.structured-event-field-cache.v1"
SOURCE_FORMAT = "pebby.ls20-structured-event-transitions-fast.v1"
ACTION_COUNT = 4
HISTORY = 8


SOURCE_REQUIRED = (
    "frames", "history_valid", "previous_actions", "next_frames",
    "terminal", "won", "lost_life", "optimal", "seed", "context_index",
    "split_id", "player_cell", "next_player_cell", "current_triple",
    "next_triple", "current_steps", "next_steps", "current_lives",
    "next_lives", "current_reachable", "current_distance", "next_distance",
    "next_reachable", "next_optimal", "branch_events", "selected_action",
    "actual_lost_life", "actual_terminal", "actual_won", "actual_event",
)


def _json_meta(value):
    try:
        if isinstance(value, np.ndarray):
            value = value.item()
        return json.loads(str(value))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("event archive meta is not JSON") from exc


def _same_shape(data, name, shape):
    if name not in data or tuple(data[name].shape) != tuple(shape):
        got = None if name not in data else tuple(data[name].shape)
        raise ValueError(f"{name} must have shape {shape}, got {got}")


def _bool_array(value, name, shape):
    if value.dtype != np.bool_ or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must be boolean {shape}")


def validate_event_arrays(data: dict[str, np.ndarray]) -> None:
    """Validate the collector's row/branch contract without game access."""
    missing = [name for name in SOURCE_REQUIRED if name not in data]
    if missing:
        raise ValueError(f"event archive missing arrays: {missing}")
    n = len(data["frames"])
    if n < 1:
        raise ValueError("event archive contains no retained rows")
    _same_shape(data, "frames", (n, HISTORY, 64, 64))
    _same_shape(data, "history_valid", (n, HISTORY))
    _same_shape(data, "previous_actions", (n, HISTORY))
    _same_shape(data, "next_frames", (n, ACTION_COUNT, 64, 64))
    for name in ("seed", "context_index", "split_id", "current_steps", "current_lives",
                 "current_distance", "optimal", "selected_action", "actual_lost_life",
                 "actual_terminal", "actual_won", "actual_event"):
        _same_shape(data, name, (n,))
    for name in ("terminal", "won", "lost_life", "branch_events", "next_steps", "next_lives",
                 "next_distance", "next_reachable", "next_optimal"):
        _same_shape(data, name, (n, ACTION_COUNT))
    _same_shape(data, "player_cell", (n, 2))
    _same_shape(data, "next_player_cell", (n, ACTION_COUNT, 2))
    _same_shape(data, "current_triple", (n, 3))
    _same_shape(data, "next_triple", (n, ACTION_COUNT, 3))
    _same_shape(data, "next_distance", (n, ACTION_COUNT))
    _same_shape(data, "next_reachable", (n, ACTION_COUNT))
    for name in ("terminal", "won", "lost_life", "next_reachable"):
        _bool_array(data[name], name, (n, ACTION_COUNT))
    _bool_array(data["history_valid"], "history_valid", (n, HISTORY))
    for name in ("current_reachable", "actual_lost_life", "actual_terminal", "actual_won"):
        _bool_array(data[name], name, (n,))
    if not np.issubdtype(data["previous_actions"].dtype, np.integer) or data["previous_actions"].dtype == np.bool_:
        raise ValueError("previous_actions must be integer")
    for name in ("optimal", "next_optimal", "branch_events", "selected_action", "actual_event",
                 "context_index", "split_id"):
        if (not np.issubdtype(data[name].dtype, np.integer)
                or data[name].dtype == np.bool_):
            raise ValueError(f"{name} must be integer")
    for name in ("seed", "player_cell", "next_player_cell", "current_triple", "next_triple",
                 "current_steps", "next_steps", "current_lives", "next_lives", "current_distance",
                 "next_distance"):
        if not np.issubdtype(data[name].dtype, np.integer):
            raise ValueError(f"{name} must be integer")
    if data["frames"].dtype != np.uint8 or data["next_frames"].dtype != np.uint8:
        raise ValueError("public frames must be uint8 palette arrays")
    valid = data["history_valid"]
    previous = data["previous_actions"]
    if not bool(valid[:, -1].all()):
        raise ValueError("every current history slot must be valid")
    if np.any(valid[:, :-1] & ~valid[:, 1:]):
        raise ValueError("history validity must be a contiguous suffix")
    if np.any((previous < -1) | (previous > 3)) or np.any(previous[~valid] != -1):
        raise ValueError("history actions must be -1 in padding and 0..3 otherwise")
    if np.any((data["selected_action"] < 0) | (data["selected_action"] >= ACTION_COUNT)):
        raise ValueError("selected_action outside 0..3")
    if np.any((data["split_id"] < 0) | (data["split_id"] > 1)):
        raise ValueError("split_id must be 0 (train) or 1 (validation)")
    if np.any((data["optimal"] < 0) | (data["optimal"] > 15)):
        raise ValueError("optimal must be a 4-bit mask")
    if np.any((data["next_optimal"] < 0) | (data["next_optimal"] > 15)):
        raise ValueError("next_optimal must be a 4-bit mask")
    if np.any((data["branch_events"] < 0) | (data["branch_events"] > 3)):
        raise ValueError("branch_events must use event codes 0..3")
    if np.any(data["terminal"] & (data["next_optimal"] != 0)):
        raise ValueError("terminal successors cannot carry policy masks")
    if np.any(data["won"] & ~data["terminal"]):
        raise ValueError("won must imply terminal")
    if np.any(data["current_reachable"] != (data["current_distance"] > 0)):
        raise ValueError("current reachable/distance mismatch")
    if np.any((~data["current_reachable"]) & (data["optimal"] != 0)):
        raise ValueError("doomed current rows must have optimal mask zero")
    if np.any(data["current_reachable"] & (data["optimal"] == 0)):
        raise ValueError("reachable current rows need a nonempty optimal mask")
    if np.any((data["current_steps"] < -3) | (data["current_steps"] > 42)):
        raise ValueError("current steps outside observed -3..42 domain")
    if np.any((data["next_steps"] < -3) | (data["next_steps"] > 42)):
        raise ValueError("successor steps outside observed -3..42 domain")
    if (np.any((data["current_lives"] < 0) | (data["current_lives"] > 3))
            or np.any((data["next_lives"] < 0) | (data["next_lives"] > 3))):
        raise ValueError("lives outside 0..3")
    if np.any(data["lost_life"] != (data["next_lives"] < data["current_lives"][:, None])):
        raise ValueError("lost_life does not match the actual life decrement")
    if np.any(data["lost_life"] & (data["next_lives"] != data["current_lives"][:, None] - 1)):
        raise ValueError("a life-loss branch must decrement lives exactly once")
    if np.any((~data["lost_life"]) & (data["next_lives"] != data["current_lives"][:, None])):
        raise ValueError("a non-loss branch changed lives")
    if np.any(data["next_reachable"] != (data["next_distance"] > 0)):
        raise ValueError("successor reachable/distance mismatch")
    if np.any((~data["next_reachable"]) & (data["next_optimal"] != 0)):
        raise ValueError("unreachable successors must have zero policy masks")
    if np.any(data["next_reachable"] & (data["next_optimal"] == 0)):
        raise ValueError("reachable successors need a nonempty policy mask")
    if np.any((data["next_distance"] < -1)):
        raise ValueError("successor distance below -1")
    if np.any(data["won"] & data["lost_life"]):
        raise ValueError("winning branches cannot lose a life")
    terminal_failure = data["terminal"] & ~data["won"]
    if np.any(terminal_failure & ~data["lost_life"]):
        raise ValueError("terminal failures must be third-life loss branches")
    if np.any(terminal_failure & ((data["current_lives"][:, None] != 1)
                                  | (data["next_lives"] != 0))):
        raise ValueError("terminal failures must transition the last life from 1 to 0")
    if np.any(data["won"] & (data["next_distance"] != 0)):
        raise ValueError("winning successors must have distance zero")
    if np.any(terminal_failure & (data["next_distance"] != -1)):
        raise ValueError("terminal failures must have distance -1")
    expected_events = np.where(
        data["won"], 3,
        np.where(data["lost_life"] & data["terminal"], 2,
                 np.where(data["lost_life"], 1, 0)),
    ).astype(data["branch_events"].dtype, copy=False)
    if not np.array_equal(data["branch_events"], expected_events):
        raise ValueError("branch_events do not match terminal/win/loss labels")
    selected = data["selected_action"].astype(np.int64, copy=False)
    rows = np.arange(n)
    for source, actual in (("lost_life", "actual_lost_life"),
                           ("terminal", "actual_terminal"),
                           ("won", "actual_won")):
        if not np.array_equal(data[actual], data[source][rows, selected]):
            raise ValueError(f"{actual} is not aligned with selected_action")
    if not np.array_equal(data["actual_event"], data["branch_events"][rows, selected]):
        raise ValueError("actual_event is not aligned with selected_action")


def _level_index(meta):
    levels = meta.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("event metadata lacks per-level proofs")
    result = {}
    for level in levels:
        if not isinstance(level, dict):
            raise ValueError("malformed level proof")
        seed = int(level.get("seed", -1))
        if seed in result:
            raise ValueError(f"duplicate level proof for seed {seed}")
        if level.get("context_engine_verified") is not True:
            raise ValueError(f"seed {seed} lacks contextual engine verification")
        if level.get("context_index") != generated_context(level):
            raise ValueError(f"seed {seed} has wrong context index")
        if not isinstance(level.get("difficulty"), (int, np.integer)):
            raise ValueError(f"seed {seed} lacks difficulty")
        result[seed] = level
    return result


def load_event_source(source, report, split):
    """Load one split from a completed combined event archive."""
    source, report = Path(source), Path(report)
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    source_hash = digest(source)
    report_hash = digest(report)
    report_data = json.loads(report.read_text())
    if report_data.get("status") != "complete":
        raise ValueError("event collector report is not complete")
    if report_data.get("source_unchanged") is not True or report_data.get("code_unchanged") is not True:
        raise ValueError("event collector report lacks unchanged-source/code proof")
    output = report_data.get("output", {})
    if output.get("sha256") != source_hash:
        raise ValueError("event report output digest does not match source archive")
    report_sources = report_data.get("sources")
    if not isinstance(report_sources, dict) or set(report_sources) != {"train", "validation"}:
        raise ValueError("event report lacks both source split proofs")
    for source_split, info in report_sources.items():
        if not isinstance(info, dict) or info.get("sha256_before") != info.get("sha256_after"):
            raise ValueError(f"{source_split} source changed during collection")
        source_path = Path(info.get("path", ""))
        if not source_path.exists() or digest(source_path) != info["sha256_before"]:
            raise ValueError(f"{source_split} source proof no longer matches its input")
    code_hashes = report_data.get("source_code_hashes")
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise ValueError("event report lacks source code hashes")
    for code_path, expected in code_hashes.items():
        path = Path(code_path)
        if not path.exists() or digest(path) != expected:
            raise ValueError(f"event collector code proof changed: {code_path}")
    with np.load(source, allow_pickle=False) as archive:
        meta = _json_meta(archive["meta"]) if "meta" in archive.files else None
        if not isinstance(meta, dict) or meta.get("format") != SOURCE_FORMAT:
            raise ValueError("unsupported event source format")
        if meta.get("source") != "generated_only" or meta.get("oracle_search") != "complete_only":
            raise ValueError("event source needs generated-only complete-oracle provenance")
        if meta.get("history") != HISTORY or meta.get("alternatives_per_state") != ACTION_COUNT:
            raise ValueError("event source history/action contract mismatch")
        if meta.get("official_inputs_used") is not False:
            raise ValueError("official inputs are forbidden")
        data = {name: np.array(archive[name], copy=True) for name in SOURCE_REQUIRED}
        stored_meta = meta
    validate_event_arrays(data)
    if digest(source) != source_hash or digest(report) != report_hash:
        raise ValueError("event archive/report changed while being loaded")
    expected_split = 0 if split == "train" else 1
    all_seeds = data["seed"]
    train_seeds = set(map(int, all_seeds[data["split_id"] == 0]))
    validation_seeds = set(map(int, all_seeds[data["split_id"] == 1]))
    if train_seeds & validation_seeds:
        raise ValueError("event source has train/validation seed overlap")
    rows = np.flatnonzero(data["split_id"] == expected_split)
    if len(rows) == 0:
        raise ValueError(f"event source has no {split} rows")
    seeds = data["seed"][rows]
    low, high = (0, 1_000_000) if split == "train" else (1_000_000, 2_000_000)
    if not bool(((seeds >= low) & (seeds < high)).all()):
        raise ValueError(f"{split} seed namespace mismatch")
    proofs = _level_index(stored_meta)
    if not set(map(int, seeds)) <= set(proofs):
        raise ValueError("event rows lack per-level contextual proof")
    selected = report_data.get("sources", {}).get(split, {}).get("selected", [])
    if not set(map(int, seeds)) <= set(map(int, selected)):
        raise ValueError("event rows are outside the collector's selected split levels")
    if np.any(data["context_index"][rows] != row_contexts(seeds, proofs.values())):
        raise ValueError("row context index mismatch")
    difficulties = np.asarray([int(proofs[int(seed)]["difficulty"]) for seed in seeds], dtype=np.int8)
    for proof in proofs.values():
        validate_difficulty(proof)
    return {name: value[rows] for name, value in data.items()} | {
        "seeds": seeds.astype(np.int64, copy=True),
        "difficulties": difficulties,
        "source_rows": rows.astype(np.int64, copy=True),
        "context_index": data["context_index"][rows].astype(np.int8, copy=True),
        "meta": stored_meta,
        "source_sha256": source_hash,
        "report_sha256": digest(report),
    }


def _cache_arrays(data, fields, next_fields):
    """Map source names to the stable event-cache names."""
    n = len(data["seeds"])
    arrays = {
        "fields": fields, "next_fields": next_fields,
        "seeds": data["seeds"], "difficulties": data["difficulties"],
        "source_rows": data["source_rows"], "context_index": data["context_index"],
        "branch_actions": np.broadcast_to(np.arange(4, dtype=np.int8), (n, 4)).copy(),
        "player_cell": data["player_cell"], "next_player_cell": data["next_player_cell"],
        "triple": data["current_triple"], "next_triple": data["next_triple"],
        "steps": data["current_steps"], "next_steps": data["next_steps"],
        "lives": data["current_lives"], "next_lives": data["next_lives"],
        "optimal": data["optimal"], "next_optimal": data["next_optimal"],
        "distances": data["next_distance"], "next_distance": data["next_distance"],
        "current_distance": data["current_distance"],
        "current_reachable": data["current_reachable"], "next_reachable": data["next_reachable"],
        "lost_life": data["lost_life"], "terminal": data["terminal"], "won": data["won"],
        "branch_events": data["branch_events"],
        "selected_action": data["selected_action"], "actual_event": data["actual_event"],
        "actual_lost_life": data["actual_lost_life"], "actual_terminal": data["actual_terminal"],
        "actual_won": data["actual_won"],
    }
    return arrays


def _metadata_file_hashes(metadata):
    """Return source files named by frozen encoder provenance."""
    found = {}

    def visit(value):
        if isinstance(value, dict):
            expected = value.get("sha256")
            for path_key in ("path", "checkpoint"):
                path = value.get(path_key)
                if isinstance(path, str) and isinstance(expected, str):
                    found[path] = expected
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(metadata)
    return found


def build_split(source, report, out, world, visibility, split, *, batch_size=8,
                max_encoder_batch=128, device="cpu"):
    """Encode all retained rows for ``split`` into a separate event cache."""
    import torch
    from pebby.agent.structured_field import load_structured_field_encoder

    if Path(out).exists():
        raise FileExistsError(out)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(max_encoder_batch) is not int or not 1 <= max_encoder_batch <= 1024:
        raise ValueError("max_encoder_batch must be 1..1024")
    data = load_event_source(source, report, split)
    source_before, report_before = data["source_sha256"], data["report_sha256"]
    builder_paths = [Path(__file__), Path("tools/build_structured_field_cache.py"),
                     Path("pebby/agent/structured_field.py"), Path("pebby/agent/world_model.py"),
                     Path("pebby/agent/world_readout.py"), Path("pebby/agent/world_rollout.py"),
                     Path("pebby/agent/world_grounding.py"), Path("pebby/agent/cell_appearance.py"),
                     Path("pebby/agent/cell_appearance_dense.py"), Path("pebby/agent/glyph_model.py"),
                     Path("pebby/agent/cell_visibility.py")]
    builder_hashes = {str(path): digest(path) for path in builder_paths if path.exists()}
    checkpoint_paths = {"world": Path(world), "visibility": Path(visibility)}
    checkpoint_hashes = {}
    for label, path in checkpoint_paths.items():
        if not path.exists():
            raise ValueError(f"missing {label} checkpoint: {path}")
        checkpoint_hashes[str(path)] = digest(path)
    assembler = load_structured_field_encoder(world, visibility, device=device).eval()
    if any(parameter.requires_grad for parameter in assembler.parameters()):
        raise ValueError("field encoder must be frozen")
    provenance = assembler.metadata()
    encoder_hashes = dict(builder_hashes)
    encoder_hashes.update(checkpoint_hashes)
    encoder_hashes.update(_metadata_file_hashes(provenance))
    for path, expected in encoder_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"frozen encoder source changed before encoding: {path}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    current_parts, future_parts, quant_error = [], [], 0.0
    for begin in range(0, len(data["seeds"]), batch_size):
        end = min(begin + batch_size, len(data["seeds"]))
        batch = {name: data[name][begin:end]
                 for name in ("frames", "history_valid", "previous_actions", "next_frames", "lost_life")}
        current, future, error = encode_fields(assembler, batch, max_encoder_batch=max_encoder_batch)
        current_parts.append(current); future_parts.append(future); quant_error = max(quant_error, error)
    current = np.concatenate(current_parts, axis=0)
    future = np.concatenate(future_parts, axis=0)
    arrays = _cache_arrays(data, current, future)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        inventory = {}
        for name, value in arrays.items():
            path = temporary / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            inventory[name] = {"shape": list(loaded.shape), "dtype": str(loaded.dtype),
                               "sha256": digest(path)}
        manifest = {
            **cache_difficulty_metadata(data["meta"], data["seeds"]),
            "format": FORMAT, "status": "complete", "source": "generated_only", "split": split,
            "rows": len(data["seeds"]), "levels": int(len(np.unique(data["seeds"]))),
            "history": HISTORY, "alternatives_per_state": ACTION_COUNT,
            "source_archive": str(Path(source).resolve()), "source_sha256": data["source_sha256"],
            "collector_report": str(Path(report).resolve()), "collector_report_sha256": data["report_sha256"],
            "field_encoder": provenance, "builder_code_hashes": builder_hashes,
            "checkpoint_hashes": checkpoint_hashes,
            "encoder_source_hashes": encoder_hashes, "arrays": inventory,
            "event_coverage": {
                "branches": int(data["lost_life"].size),
                "lost_life": int(data["lost_life"].sum()),
                "terminal_failure": int((data["terminal"] & ~data["won"]).sum()),
                "won": int(data["won"].sum()),
                "doomed_current_rows": int((~data["current_reachable"]).sum()),
            },
            "history_contract": (
                "current fields use stored causal H8; each branch target appends its actual frame. "
                "Only a lost-life branch uses the post-loss frame as the sole valid reset slot "
                "and -1 actions (including third-life GAME_OVER); a terminal WIN appends normally. "
                "Four slots are action alternatives, never chronology."),
            "training_contract": (
                "repeated seeds and optimal==0 are intentional; load with this event-cache format, "
                "not world_train.load_dataset or chronological H4 loaders."),
            "no_future_inputs_to_current_field": True,
            "float16_max_absolute_quantization_error": quant_error,
            "encoding_device": device,
            "encoding_precision": {"compute_dtype": "float32", "autocast": False,
                                   "tf32": False, "cache_dtype": "float16",
                                   "batch_size": batch_size, "max_encoder_batch": max_encoder_batch},
            "encoding_seconds": time.monotonic() - started,
        }
        if digest(source) != source_before or digest(report) != report_before:
            raise ValueError("event source/report changed during encoding")
        for path, expected in encoder_hashes.items():
            if digest(path) != expected:
                raise ValueError(f"frozen encoder source changed during encoding: {path}")
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-encoder-batch", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if not 1 <= args.max_encoder_batch <= 1024:
        parser.error("max-encoder-batch must be 1..1024")
    import torch
    torch.set_num_threads(1)
    manifest = build_split(args.source, args.report, args.out, args.world, args.visibility,
                           args.split, batch_size=args.batch_size,
                           max_encoder_batch=args.max_encoder_batch, device=args.device)
    print(json.dumps({"status": "complete", "out": str(args.out),
                      "rows": manifest["rows"], "event_coverage": manifest["event_coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
