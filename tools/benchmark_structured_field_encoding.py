#!/usr/bin/env python3
"""Bounded CPU/CUDA benchmark for frozen structured-field cache encoding.

The benchmark reads one row per distinct level from the immutable 2,048-level
field-cache source selection.  Each timed unit encodes the current public H8
history and all four actual branch/reset histories using the same helper as the
cache builder.  It never trains, writes cache arrays, or changes shared code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import torch

from pebby.agent.structured_field import load_structured_field_encoder
from pebby.agent.world_train import load_dataset
from tools.build_structured_field_cache import actual_histories


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ls20-world-combined-train.npz"
CACHE = ROOT / "data/structured-field-pilot-2048/train"
WORLD = ROOT / "checkpoints/ls20-world-cell-recall-b1024.pt"
VISIBILITY = ROOT / "checkpoints/ls20-cell-visibility-initial-200.pt"
OUTPUT = ROOT / "artifacts/structured-field-encoding-benchmark.json"
PARITY_OUTPUT = ROOT / "artifacts/structured-field-encoding-parity-corrected.json"
PILOT_LEVELS = 2048
SELECTED_LEVELS = 128
HISTORY_KEYS = ("frames", "history_valid", "previous_actions", "next_frames", "lost_life")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def selected_rows(data):
    source_rows = np.load(CACHE / "source_rows.npy", mmap_mode="r", allow_pickle=False)
    cached_seeds = np.load(CACHE / "seeds.npy", mmap_mode="r", allow_pickle=False)
    if len(source_rows) != PILOT_LEVELS or len(cached_seeds) != PILOT_LEVELS:
        raise ValueError("immutable 2048 cache selection has an unexpected row count")
    if len(np.unique(source_rows)) != PILOT_LEVELS or len(np.unique(cached_seeds)) != PILOT_LEVELS:
        raise ValueError("immutable cache selection must contain distinct rows and levels")
    if np.any(source_rows < 0) or np.any(source_rows >= len(data["frames"])):
        raise ValueError("immutable cache source_rows exceed the current source archive")
    if not np.array_equal(np.asarray(data["seeds"])[source_rows], np.asarray(cached_seeds)):
        raise ValueError("immutable cache source_rows/seeds no longer match the source archive")
    positions = np.linspace(0, PILOT_LEVELS - 1, SELECTED_LEVELS, dtype=np.int64)
    rows = np.asarray(source_rows[positions], dtype=np.int64)
    seeds = np.asarray(cached_seeds[positions], dtype=np.int64)
    if len(np.unique(rows)) != SELECTED_LEVELS or len(np.unique(seeds)) != SELECTED_LEVELS:
        raise ValueError("benchmark selection lost distinct level identity")
    return rows, seeds, positions


def materialize(data, rows):
    started = time.perf_counter()
    batch = {key: np.array(data[key][rows], copy=True) for key in HISTORY_KEYS}
    histories, validity, actions = actual_histories(batch)
    elapsed = time.perf_counter() - started
    return batch, (histories, validity, actions), elapsed


def make_inputs(batch, history_triplet, device: torch.device, resident: bool):
    histories, validity, actions = history_triplet
    def convert(value):
        value = torch.as_tensor(value) if isinstance(value, np.ndarray) else value
        return value.to(device=device) if resident else value
    current = tuple(convert(batch[key]) for key in ("frames", "history_valid", "previous_actions"))
    target = (convert(histories.flatten(0, 1)), convert(validity.flatten(0, 1)),
              convert(actions.flatten(0, 1)))
    return current, target


def encode_pair(encoder, inputs):
    current, target = inputs
    with torch.inference_mode():
        field = encoder(*current)
        future = encoder(*target)
    return field, future.reshape(len(current[0]), 4, 148, 96)


def benchmark_one(encoder, batch, history_triplet, device: torch.device, resident: bool,
                  repeats: int, warmup: int):
    inputs = make_inputs(batch, history_triplet, device, resident)
    for _ in range(warmup):
        encode_pair(encoder, inputs)
    synchronize(device)
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = encode_pair(encoder, inputs)
        synchronize(device)
        samples.append(time.perf_counter() - started)
    return {
        "device": str(device), "resident_inputs": resident,
        "batch": int(len(batch["frames"])), "target_batch": int(len(batch["frames"]) * 4),
        "warmup": warmup, "repeats": repeats,
        "seconds": samples, "median_seconds": float(np.median(samples)),
        "min_seconds": float(min(samples)), "max_seconds": float(max(samples)),
        "levels_per_second": float(len(batch["frames"]) / np.median(samples)),
        "output_shape": [list(output[0].shape), list(output[1].shape)],
        "output_max": float(max(output[0].abs().max().item(), output[1].abs().max().item())),
    }, output


def numerical_agreement(cpu, gpu):
    """Compare float outputs and their independent semantic probability groups.

    Appearance and carried glyph channels concatenate independent Bernoulli or
    categorical heads.  A single argmax over the concatenation is meaningless,
    so each role bit and categorical head is scored separately.
    """
    cpu_current, cpu_future = cpu
    gpu_current, gpu_future = gpu
    fields = [("current", cpu_current, gpu_current), ("next", cpu_future, gpu_future)]
    result = {}
    for name, left, right in fields:
        delta = (left.float().cpu() - right.float().cpu()).abs()
        result[f"{name}_max_abs_error"] = float(delta.max().item())
        result[f"{name}_mean_abs_error"] = float(delta.mean().item())
        board_left = left[..., :144, :].float().cpu()
        board_right = right[..., :144, :].float().cpu()
        roles_equal = ((board_left[..., 48:56] >= .5) ==
                       (board_right[..., 48:56] >= .5)).all(-1)
        result[f"{name}_board_role_bit_threshold_equal_fraction"] = float(roles_equal.float().mean().item())
        visibility_equal = ((board_left[..., 84] >= .5) ==
                            (board_right[..., 84] >= .5))
        result[f"{name}_board_visibility_threshold_equal_fraction"] = float(
            visibility_equal.float().mean().item())

        appearance_hits = []
        for label, start, stop in (("shape", 56, 62), ("color", 62, 66), ("rotation", 66, 70)):
            left_index = board_left[..., start:stop].argmax(-1)
            right_index = board_right[..., start:stop].argmax(-1)
            hit = left_index == right_index
            appearance_hits.append(hit)
            result[f"{name}_board_goal_{label}_argmax_equal_fraction"] = float(hit.float().mean().item())
        result[f"{name}_board_goal_joint_argmax_equal_fraction"] = float(
            torch.stack(appearance_hits, -1).all(-1).float().mean().item())

        # The carried glyph is broadcast to every field token.  Compare one
        # board token per frame (cell 0), avoiding token-count weighting.
        left_glyph = left[..., 0, 70:84].float().cpu()
        right_glyph = right[..., 0, 70:84].float().cpu()
        glyph_hits = []
        for label, start, stop in (("shape", 0, 6), ("color", 6, 10), ("rotation", 10, 14)):
            left_index = left_glyph[..., start:stop].argmax(-1)
            right_index = right_glyph[..., start:stop].argmax(-1)
            hit = left_index == right_index
            glyph_hits.append(hit)
            result[f"{name}_carried_{label}_argmax_equal_fraction"] = float(hit.float().mean().item())
        result[f"{name}_carried_joint_argmax_equal_fraction"] = float(
            torch.stack(glyph_hits, -1).all(-1).float().mean().item())
    result["semantic_notes"] = [
        "Role and visibility fractions threshold independent sigmoid channels at 0.5 on board cells only.",
        "Goal categorical fractions compare shape 6, color 4, and rotation 4 heads separately and jointly over all board cells; no gold visibility mask is applied.",
        "Carried glyph categorical fractions compare one broadcast board token per frame, separately and jointly.",
    ]
    return result


def parity_only() -> int:
    """Run only the corrected CPU/CUDA B=8 parity check."""
    torch.set_num_threads(1)
    started = time.monotonic()
    report = {
        "status": "running", "pid": os.getpid(), "cpu_threads": 1,
        "official_inputs_used": False, "source": str(SOURCE), "cache": str(CACHE),
        "supersedes_metric_in": str(OUTPUT),
        "original_report_sha256": digest(OUTPUT),
        "original_benchmark_code_sha256": "4cba6e4e7911b531cf48a236ea23214a03affa99f94f873daab4c95ba309deaf",
    }
    error = None
    try:
        source_sha_before = digest(SOURCE)
        cache_manifest_sha = digest(CACHE / "manifest.json")
        cache_manifest = json.loads((CACHE / "manifest.json").read_text())
        if cache_manifest.get("source_sha256") != source_sha_before:
            raise ValueError("immutable 2048 cache manifest does not bind current source bytes")
        code_paths = [Path("pebby/agent/structured_field.py"), Path(__file__)]
        code_before = {str(path): digest(ROOT / path if not path.is_absolute() else path)
                       for path in code_paths}
        checkpoint_before = {str(path): digest(path) for path in (WORLD, VISIBILITY)}
        data = load_dataset(SOURCE, history=8, cache_dir=ROOT / "data/world-array-cache")
        rows, seeds, positions = selected_rows(data)
        batch, history_triplet, materialize_seconds = materialize(data, rows[:8])
        report.update({
            "source_sha256_before": source_sha_before,
            "cache_manifest_sha256": cache_manifest_sha,
            "cache_source_rows_sha256": digest(CACHE / "source_rows.npy"),
            "cache_seeds_sha256": digest(CACHE / "seeds.npy"),
            "selected_levels": 8,
            "selected_source_rows": [int(value) for value in rows[:8]],
            "selected_seeds": [int(value) for value in seeds[:8]],
            "selected_cache_positions": [int(value) for value in positions[:8]],
            "selected_seed_sha256": hashlib.sha256(seeds[:8].tobytes()).hexdigest(),
            "batch_materialize_seconds": materialize_seconds,
            "model": {"world": str(WORLD), "visibility": str(VISIBILITY),
                      "world_sha256": checkpoint_before[str(WORLD)],
                      "visibility_sha256": checkpoint_before[str(VISIBILITY)]},
            "code_hashes_before": code_before,
        })
        cpu_encoder = load_structured_field_encoder(WORLD, VISIBILITY, device="cpu").eval()
        _, cpu_output = benchmark_one(cpu_encoder, batch, history_triplet, torch.device("cpu"),
                                      True, repeats=1, warmup=0)
        del cpu_encoder
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for corrected parity")
        device = torch.device("cuda")
        cuda_encoder = load_structured_field_encoder(WORLD, VISIBILITY, device=device).eval()
        _, cuda_output = benchmark_one(cuda_encoder, batch, history_triplet, device,
                                       True, repeats=1, warmup=0)
        report["cuda_memory"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
        report["output_shapes"] = {
            "cpu": [list(cpu_output[0].shape), list(cpu_output[1].shape)],
            "cuda": [list(cuda_output[0].shape), list(cuda_output[1].shape)],
        }
        report["corrected_semantic_agreement"] = numerical_agreement(cpu_output, cuda_output)
        report["source_sha256_after"] = digest(SOURCE)
        report["code_hashes_after"] = {str(path): digest(ROOT / path if not path.is_absolute() else path)
                                        for path in code_paths}
        report["checkpoint_hashes_after"] = {str(path): digest(path) for path in (WORLD, VISIBILITY)}
        report["source_unchanged"] = report["source_sha256_after"] == source_sha_before
        report["code_unchanged"] = report["code_hashes_after"] == code_before
        report["checkpoints_unchanged"] = report["checkpoint_hashes_after"] == checkpoint_before
        if not (report["source_unchanged"] and report["code_unchanged"] and report["checkpoints_unchanged"]):
            raise RuntimeError("source/code/checkpoint bytes changed during parity check")
        report["status"] = "complete"
    except Exception as exc:
        error = exc
        report["status"] = "failed_closed"
        report["error"] = repr(exc)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        PARITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = PARITY_OUTPUT.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(PARITY_OUTPUT)
        print(json.dumps({key: report.get(key) for key in
                          ("status", "elapsed_seconds", "error")}, indent=2), flush=True)
    return 1 if error is not None or report["status"] != "complete" else 0


def main() -> int:
    torch.set_num_threads(1)
    started = time.monotonic()
    report = {"status": "running", "pid": os.getpid(), "cpu_threads": 1,
              "official_inputs_used": False, "source": str(SOURCE), "cache": str(CACHE)}
    seconds = 110

    def timeout(_signum, _frame):
        raise TimeoutError(f"benchmark exceeded {seconds}-second bound")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(seconds)
    error = None
    try:
        source_sha_before = digest(SOURCE)
        cache_manifest = json.loads((CACHE / "manifest.json").read_text())
        cache_manifest_sha = digest(CACHE / "manifest.json")
        if cache_manifest.get("source_sha256") != source_sha_before:
            raise ValueError("immutable 2048 cache manifest does not bind current source bytes")
        code_paths = [Path("pebby/agent/structured_field.py"), Path("tools/build_structured_field_cache.py")]
        code_before = {str(path): digest(path) for path in code_paths}
        checkpoint_before = {str(path): digest(path) for path in (WORLD, VISIBILITY)}

        load_started = time.perf_counter()
        data = load_dataset(SOURCE, history=8, cache_dir=ROOT / "data/world-array-cache")
        load_seconds = time.perf_counter() - load_started
        rows, seeds, positions = selected_rows(data)
        batch, history_triplet, materialize_seconds = materialize(data, rows[:128])
        report.update({
            "source_sha256_before": source_sha_before,
            "cache_manifest_sha256": cache_manifest_sha,
            "cache_source_rows_sha256": digest(CACHE / "source_rows.npy"),
            "cache_seeds_sha256": digest(CACHE / "seeds.npy"),
            "selected_levels": SELECTED_LEVELS,
            "selected_seed_min": int(seeds.min()), "selected_seed_max": int(seeds.max()),
            "selected_seed_sha256": hashlib.sha256(seeds.tobytes()).hexdigest(),
            "selected_cache_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "source_load_seconds": load_seconds,
            "batch_materialize_seconds": materialize_seconds,
            "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
            "batch_dtypes": {key: str(value.dtype) for key, value in batch.items()},
            "model": {"world": str(WORLD), "visibility": str(VISIBILITY),
                      "world_sha256_before": checkpoint_before[str(WORLD)],
                      "visibility_sha256_before": checkpoint_before[str(VISIBILITY)]},
        })

        cpu_load_started = time.perf_counter()
        cpu_encoder = load_structured_field_encoder(WORLD, VISIBILITY, device="cpu").eval()
        cpu_load_seconds = time.perf_counter() - cpu_load_started
        report["cpu_encoder_load_seconds"] = cpu_load_seconds
        cpu_results = {}
        cpu_outputs = {}
        for size, repeats in ((8, 3), (32, 3)):
            item = {key: value[:size] for key, value in batch.items()}
            h = tuple(value[:size] for value in history_triplet)
            result, output = benchmark_one(cpu_encoder, item, h, torch.device("cpu"), True,
                                           repeats, warmup=1)
            cpu_results[f"batch{size}"] = result
            cpu_outputs[f"batch{size}"] = output
        report["cpu"] = cpu_results

        cuda_available = torch.cuda.is_available()
        report["cuda_available"] = bool(cuda_available)
        if cuda_available:
            device = torch.device("cuda")
            free_before, total_memory = torch.cuda.mem_get_info(device)
            cuda_load_started = time.perf_counter()
            cuda_encoder = load_structured_field_encoder(WORLD, VISIBILITY, device=device).eval()
            synchronize(device)
            cuda_load_seconds = time.perf_counter() - cuda_load_started
            report["cuda_encoder_load_seconds"] = cuda_load_seconds
            cuda_results = {}
            cuda_outputs = {}
            for size in (32, 64, 128):
                item = {key: value[:size] for key, value in batch.items()}
                h = tuple(value[:size] for value in history_triplet)
                result, output = benchmark_one(cuda_encoder, item, h, device, True, repeats=5, warmup=2)
                cuda_results[f"batch{size}_resident"] = result
                cuda_outputs[f"batch{size}_resident"] = output
                result, _ = benchmark_one(cuda_encoder, item, h, device, False, repeats=3, warmup=1)
                cuda_results[f"batch{size}_host_inputs"] = result
            free_after, _ = torch.cuda.mem_get_info(device)
            report["cuda_memory"] = {
                "total_bytes": int(total_memory), "free_before_bytes": int(free_before),
                "free_after_bytes": int(free_after),
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            report["cuda"] = cuda_results
            # Compare the same B=8 inputs separately; B=32/64/128 are timing-only.
            item = {key: value[:8] for key, value in batch.items()}
            h = tuple(value[:8] for value in history_triplet)
            _, cuda_b8 = benchmark_one(cuda_encoder, item, h, device, True, repeats=1, warmup=1)
            report["cpu_cuda_numerical_agreement"] = numerical_agreement(cpu_outputs["batch8"], cuda_b8)
        else:
            report["cuda"] = None
            report["cpu_cuda_numerical_agreement"] = None

        field_bytes = 148 * 96 * 2
        per_level_bytes = field_bytes * 5
        gpu_timing = report.get("cuda", {}).get("batch128_resident") if report.get("cuda") else None
        cpu_timing = cpu_results["batch32"]
        report["storage_estimate_16000_levels"] = {
            "current_field_bytes_per_level": field_bytes,
            "five_fields_bytes_per_level": per_level_bytes,
            "fields_and_next_fields_bytes": int(per_level_bytes * 16_000),
            "fields_and_next_fields_gib": per_level_bytes * 16_000 / (1024 ** 3),
            "small_labels_excluded": True,
        }
        report["throughput_estimate_16000_levels"] = {
            "gpu_batch128_resident": None if gpu_timing is None else {
                "median_levels_per_second": gpu_timing["levels_per_second"],
                "encoder_seconds": 16_000 / gpu_timing["levels_per_second"],
                "plus_materialization_seconds_at_measured_batch128":
                    16_000 / 128 * materialize_seconds,
            },
            "cpu_batch32": {
                "median_levels_per_second": cpu_timing["levels_per_second"],
                "encoder_seconds": 16_000 / cpu_timing["levels_per_second"],
            },
            "loader_note": "Extrapolates repeated selected-row materialization; source mmap/cache warmup is reported separately.",
        }
        report["rss_peak_mib"] = rss_mib()
        report["source_sha256_after"] = digest(SOURCE)
        report["code_hashes_before"] = code_before
        report["checkpoint_hashes_after"] = {str(path): digest(path) for path in (WORLD, VISIBILITY)}
        report["code_unchanged"] = all(digest(Path(path)) == value for path, value in code_before.items())
        report["checkpoints_unchanged"] = all(
            digest(Path(path)) == value for path, value in checkpoint_before.items())
        report["source_unchanged"] = report["source_sha256_after"] == source_sha_before
        if not (report["code_unchanged"] and report["checkpoints_unchanged"] and report["source_unchanged"]):
            raise ValueError("source/code/checkpoint bytes changed during benchmark")
        report["status"] = "complete"
    except TimeoutError as exc:
        error = exc
        report["status"] = "bounded_partial"
        report["error"] = repr(exc)
    except Exception as exc:
        error = exc
        report["status"] = "failed_closed"
        report["error"] = repr(exc)
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = rss_mib()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(OUTPUT)
        print(json.dumps({key: report.get(key) for key in
                          ("status", "elapsed_seconds", "peak_rss_mib", "cuda_available", "error")}, indent=2),
              flush=True)
    return 1 if error is not None or report["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(parity_only() if "--parity-only" in sys.argv[1:] else main())
