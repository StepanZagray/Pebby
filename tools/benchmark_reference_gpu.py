"""Benchmark one real ``world_train.run_epoch`` update on a generated shard.

This is a performance probe, not a training entry point.  It constructs fresh
weights, uses one fixed CPU batch, discards the updated weights, and never
writes a checkpoint.  The report records the exact source/data hashes and the
full loss path used by :func:`pebby.agent.world_train.run_epoch`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import statistics
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pebby.agent.world_model import DEFAULT_WEIGHTS, WorldPolicy, parameter_groups
from pebby.agent.world_train import as_tensors, load_dataset, run_epoch
from pebby.agent.world_runtime import configure_execution
from tools.preflight_reference_world import fresh_config


DEFAULT_DATA = PROJECT_ROOT / "data/reference-world-base-v1/train/shard-000000.npz"
DEFAULT_WARMUPS = 2
DEFAULT_TIMED = 3


def file_sha256(path: Path) -> str:
    """Hash a file without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    names = (
        "world_model.py",
        "world_training_objectives.py",
        "world_train.py",
        "world_cache.py",
        "world_grounding.py",
        "world_rollout.py",
        "curriculum_sampling.py",
        "glyph_model.py",
        "world_readout.py",
        "world_runtime.py",
    )
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tools/preflight_reference_world.py",
        *(PROJECT_ROOT / "pebby/agent" / name for name in names),
    ]
    return [path.resolve() for path in paths]


def hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths}


def proc_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks, or ``None`` outside procfs."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        close = text.rfind(")")
        fields_after_name = text[close + 2 :].split()
        return int(fields_after_name[19])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def json_safe(value: Any) -> Any:
    """Convert torch/numpy scalars and nested metric values to JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        return value.detach().cpu().item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def validate_args(args: argparse.Namespace) -> None:
    if type(args.batch_size) is not int or not 1 <= args.batch_size <= 1024:
        raise ValueError("batch-size must be an integer in 1..1024")
    if args.batch_size & (args.batch_size - 1):
        raise ValueError("batch-size must be a power of two")
    if type(args.chunk_size) is not int or args.chunk_size < 1:
        raise ValueError("chunk-size must be a positive integer")
    if args.warmup_updates < 2 or args.timed_updates < 3:
        raise ValueError("at least two warmup and three timed updates are required")
    if args.optimizer not in ("default", "fused"):
        raise ValueError("optimizer must be default or fused")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the benchmark is intended for a CUDA run")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 is unavailable")
    if args.profiler and args.device != "cuda":
        raise ValueError("--profiler requires --device cuda")
    if not 1 <= args.deadline_seconds <= 600:
        raise ValueError("deadline-seconds must be in 1..600")
    if args.min_mem_available_gib < 0:
        raise ValueError("min-mem-available-gib must be nonnegative")


def memory_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


def gpu_processes() -> dict[str, Any]:
    """Return a short NVIDIA compute-process snapshot for the report/guard."""
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "error": repr(error), "processes": []}
    if result.returncode != 0:
        return {"status": "error", "returncode": result.returncode,
                "stderr": result.stderr[-500:], "processes": []}
    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0]:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        processes.append({"pid": pid, "process_name": fields[1] if len(fields) > 1 else "",
                          "used_memory_mib": float(fields[2]) if len(fields) > 2 else None})
    return {"status": "ok", "processes": processes}


def check_resources(args: argparse.Namespace, report: dict[str, Any]) -> None:
    available = memory_available_bytes()
    report["host_memory_before"] = {
        "mem_available_bytes": available,
        "mem_available_gib": (available / 2 ** 30 if available is not None else None),
        "minimum_gib": args.min_mem_available_gib,
    }
    if available is None and args.min_mem_available_gib > 0:
        raise RuntimeError("host MemAvailable could not be read")
    if available is not None and available < args.min_mem_available_gib * 2 ** 30:
        raise RuntimeError("host MemAvailable is below the benchmark safety floor")
    if args.device == "cuda":
        snapshot = gpu_processes()
        report["gpu_processes_before"] = snapshot
        foreign = [item for item in snapshot.get("processes", []) if item["pid"] != os.getpid()]
        if (snapshot.get("status") != "ok" or foreign) and not args.allow_nonidle:
            raise RuntimeError("GPU is not provably idle; use --allow-nonidle only for an explicit diagnostic")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="one generated transition shard (default: first reference TRAIN shard)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new directory for the report and optional profiler/parity files")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup-updates", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--timed-updates", type=int, default=DEFAULT_TIMED)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--optimizer", choices=("default", "fused"), default="default")
    parser.add_argument('--compile-core', action='store_true')
    parser.add_argument('--temporal-backend', choices=('auto', 'math', 'cudnn', 'flash'), default='auto')
    parser.add_argument("--checkpoint-loops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profiler", action="store_true",
                        help="profile one warmup update and save a Chrome trace/operator table")
    parser.add_argument("--save-parity-arrays", action="store_true",
                        help="save final FP32 parameters/gradients, names, and shapes as a diagnostic NPZ")
    parser.add_argument("--allow-oom-fallback", action="store_true",
                        help="after a requested-batch CUDA OOM, retry halved powers of two")
    parser.add_argument("--deadline-seconds", type=int, default=600,
                        help="hard wall-clock budget for the benchmark (maximum 600 seconds)")
    parser.add_argument("--min-mem-available-gib", type=float, default=6.0,
                        help="fail closed when host MemAvailable is below this value")
    parser.add_argument("--allow-nonidle", action="store_true",
                        help="allow other NVIDIA compute processes (unsafe for timing comparisons)")
    args = parser.parse_args(argv)
    validate_args(args)
    args.data = args.data.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.out_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    return args


def fixed_batch(data: dict[str, Any], requested: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Convert exactly the first ``requested`` shard rows to CPU tensors."""
    total = len(data["frames"])
    if total < requested:
        raise ValueError(f"shard has {total} rows, fewer than requested batch {requested}")
    tensors = as_tensors(data)
    selected = {name: tensor[:requested].contiguous() for name, tensor in tensors.items()}
    seeds = np.asarray(data["seeds"][:requested])
    description = {
        "source_rows": int(total),
        "rows_used": int(requested),
        "row_indices": {"first": 0, "last_exclusive": int(requested), "order": "ascending"},
        "row_indices_sha256": hashlib.sha256(np.arange(requested, dtype=np.int64).tobytes()).hexdigest(),
        "distinct_seed_count": int(np.unique(seeds).size),
        "seed_sha256": hashlib.sha256(np.ascontiguousarray(seeds).tobytes()).hexdigest(),
        "production_capacity_claim": False,
    }
    return selected, description


def make_optimizer(model: torch.nn.Module, optimizer_name: str) -> torch.optim.Optimizer:
    groups = parameter_groups(model, 0.05)
    kwargs: dict[str, Any] = {"lr": 0.0003}
    if optimizer_name == "fused":
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def prepare_model(cfg: Any, args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = WorldPolicy(cfg).to(device).train()
    model.checkpoint_loops = args.checkpoint_loops
    model.checkpoint_encoder = args.checkpoint_encoder
    model.encoder_chunk_size = args.chunk_size
    configure_execution(model, compile_core=args.compile_core, temporal_backend=args.temporal_backend)
    optimizer = make_optimizer(model, args.optimizer)
    return model, optimizer


def timing_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  tensors: dict[str, torch.Tensor], device: torch.device,
                  weights: dict[str, float], batch_size: int, generator: torch.Generator) -> tuple[dict[str, Any], float, float | None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    start = time.perf_counter()
    stats = run_epoch(model, tensors, device, weights, batch_size,
                      optimizer=optimizer, generator=generator, precision="bf16",
                      drop_last=True)
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
        cuda_ms = float(start_event.elapsed_time(end_event))
    else:
        cuda_ms = None
    wall = time.perf_counter() - start
    return stats, wall, cuda_ms


def save_parity(path: Path, model: torch.nn.Module, step_count: int) -> dict[str, Any]:
    names: list[str] = []
    shapes: list[list[int]] = []
    parameters: list[np.ndarray] = []
    gradients: list[np.ndarray] = []
    gradient_present: list[bool] = []
    for name, parameter in model.named_parameters():
        names.append(name)
        shapes.append(list(parameter.shape))
        parameters.append(parameter.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy().copy())
        if parameter.grad is None:
            gradient_present.append(False)
            gradients.append(np.zeros(parameter.numel(), dtype=np.float32))
        else:
            gradient_present.append(True)
            gradients.append(parameter.grad.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy().copy())
    parameter_vector = np.concatenate(parameters).astype(np.float32, copy=False)
    gradient_vector = np.concatenate(gradients).astype(np.float32, copy=False)
    npz_path = path.with_suffix(".npz")
    temporary = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp")
    # Passing an open file prevents numpy from appending a second ``.npz``
    # suffix to the transaction path.
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, parameters=parameter_vector, gradients=gradient_vector,
                            gradient_present=np.asarray(gradient_present, dtype=np.bool_))
    os.replace(temporary, npz_path)
    metadata = {
        "format": "pebby.reference-gpu-benchmark-parity.v1",
        "step_count": int(step_count),
        "parameter_count": int(parameter_vector.size),
        "gradient_count": int(gradient_vector.size),
        "gradient_present": gradient_present,
        "gradient_present_names": [name for name, present in zip(names, gradient_present) if present],
        "names": names,
        "shapes": shapes,
        "parameter_dtype": "float32",
        "gradient_dtype": "float32",
        "npz": str(npz_path),
        "sha256": file_sha256(npz_path),
        "checkpoint": False,
        "optimizer_state": False,
    }
    atomic_write(path.with_suffix(".json"), metadata)
    return metadata


def profile_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   tensors: dict[str, torch.Tensor], device: torch.device,
                   weights: dict[str, float], batch_size: int, generator: torch.Generator,
                   out_dir: Path) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    trace_path = out_dir / "profiler.trace.json"
    table_path = out_dir / "profiler.table.txt"
    with torch.profiler.profile(activities=activities, record_shapes=True,
                                profile_memory=True, with_stack=False) as profiler:
        stats, wall, cuda_ms = timing_update(model, optimizer, tensors, device, weights,
                                             batch_size, generator)
        profiler.step()
    profiler.export_chrome_trace(str(trace_path))
    table_path.write_text(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=80))
    return {"stats": stats, "wall_seconds": wall, "cuda_ms": cuda_ms,
            "trace": str(trace_path), "table": str(table_path)}


def run_candidate(args: argparse.Namespace, data: dict[str, Any], batch: dict[str, torch.Tensor],
                  cfg: Any, weights: dict[str, float], device: torch.device, batch_size: int,
                  report: dict[str, Any]) -> dict[str, Any]:
    model = optimizer = None
    steps: list[dict[str, Any]] = []
    try:
        model, optimizer = prepare_model(cfg, args, device)
        report["parameters"] = int(model.parameter_count())
        report["model_config"] = model.config()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        generator = torch.Generator().manual_seed(args.seed)
        total_updates = 0
        for index in range(args.warmup_updates):
            if time.monotonic() - report["started_monotonic"] > args.deadline_seconds:
                raise TimeoutError("benchmark deadline reached before warmup update")
            report["benchmark_progress"] = {"phase": "warmup", "index": index,
                                             "completed_updates": total_updates}
            atomic_write(args.out_dir / "report.json", report)
            stats, wall, cuda_ms = timing_update(model, optimizer, batch, device, weights,
                                                 batch_size, generator)
            step = {"stats": stats, "wall_seconds": wall, "cuda_ms": cuda_ms}
            step.update(index=index, phase="warmup")
            steps.append(json_safe(step))
            total_updates += 1
            report["benchmark_progress"] = {"phase": "warmup", "index": index,
                                             "completed_updates": total_updates,
                                             "last_wall_seconds": wall, "last_cuda_ms": cuda_ms}
            atomic_write(args.out_dir / "report.json", report)
        timed_start = time.perf_counter()
        for index in range(args.timed_updates):
            if time.monotonic() - report["started_monotonic"] > args.deadline_seconds:
                raise TimeoutError("benchmark deadline reached before timed update")
            report["benchmark_progress"] = {"phase": "timed", "index": index,
                                             "completed_updates": total_updates}
            atomic_write(args.out_dir / "report.json", report)
            stats, wall, cuda_ms = timing_update(model, optimizer, batch, device, weights,
                                                 batch_size, generator)
            steps.append(json_safe({"index": index, "phase": "timed", "stats": stats,
                                    "wall_seconds": wall, "cuda_ms": cuda_ms}))
            total_updates += 1
            report["benchmark_progress"] = {"phase": "timed", "index": index,
                                             "completed_updates": total_updates,
                                             "last_wall_seconds": wall, "last_cuda_ms": cuda_ms}
            atomic_write(args.out_dir / "report.json", report)
        timed_total_wall = time.perf_counter() - timed_start
        timed = [step for step in steps if step["phase"] == "timed"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = peak_reserved = None
        # Save parity before optional profiling.  The profile update is a real
        # extra optimizer step, so it must not alter the measured steps or the
        # parameter/gradient vectors used to compare two benchmark arms.
        parity = save_parity(args.out_dir / "parity", model, total_updates) if args.save_parity_arrays else None
        profile = None
        if args.profiler:
            try:
                profile = profile_update(model, optimizer, batch, device, weights, batch_size,
                                         generator, args.out_dir)
                profile['status'] = 'complete'
            except torch.cuda.OutOfMemoryError as error:
                # Profiling retains extra metadata. Its OOM must not invalidate
                # measured real updates or trigger a smaller capacity claim.
                profile = {'status': 'cuda_oom', 'error': str(error)}
        return {
            "status": "complete",
            "requested_batch_size": int(args.batch_size),
            "batch_size": int(batch_size),
            "fallback_used": batch_size != args.batch_size,
            "updates": steps,
            "timed_update_count": len(timed),
            "timed_total_wall_seconds": timed_total_wall,
            "timed_wall_seconds": [step["wall_seconds"] for step in timed],
            "timed_cuda_ms": [step["cuda_ms"] for step in timed],
            "mean_timed_wall_seconds": statistics.mean(step["wall_seconds"] for step in timed),
            "median_timed_wall_seconds": statistics.median(step["wall_seconds"] for step in timed),
            "mean_timed_cuda_ms": statistics.mean(step["cuda_ms"] for step in timed)
            if device.type == "cuda" else None,
            "peak_memory_bytes": {"allocated": peak_allocated, "reserved": peak_reserved},
            "parity": parity,
            "profile": profile,
        }
    finally:
        del optimizer
        del model
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_files = source_paths()
    source_before = hashes(source_files)
    data_before = file_sha256(args.data)
    report: dict[str, Any] = {
        "format": "pebby.reference-gpu-benchmark.v1",
        "status": "running",
        "pid": os.getpid(),
        "start_ticks": proc_start_ticks(os.getpid()),
        "argv": sys.argv if argv is None else [str(Path(__file__))] + list(argv),
        "started_monotonic": time.monotonic(),
        "platform": {"python": sys.version, "torch": torch.__version__,
                      "platform": platform.platform(), "hostname": platform.node()},
        "device_requested": args.device,
        "allocator_environment": {key: os.environ.get(key) for key in
                                  ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')},
        "data": str(args.data),
        "data_sha256_before": data_before,
        "sources_before": source_before,
        "settings": {"batch_size": args.batch_size, "chunk_size": args.chunk_size,
                      "warmup_updates": args.warmup_updates, "timed_updates": args.timed_updates,
                      "seed": args.seed, "precision": "bf16", "tf32": False,
                      "checkpoint_loops": args.checkpoint_loops,
                      "checkpoint_encoder": args.checkpoint_encoder,
                      "optimizer": args.optimizer, "profiler": args.profiler,
                      "compile_core": args.compile_core, "temporal_backend": args.temporal_backend,
                      "save_parity_arrays": args.save_parity_arrays,
                      "allow_oom_fallback": args.allow_oom_fallback},
        "checkpoint_saved": False,
    }
    report_path = args.out_dir / "report.json"
    atomic_write(report_path, report)
    device = torch.device(args.device)
    def deadline(*_):
        raise TimeoutError('benchmark wall-clock deadline exceeded')
    old_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        check_resources(args, report)
        report["stages"] = {"load": "running", "benchmark": "pending", "cleanup": "pending"}
        atomic_write(report_path, report)
        cfg = fresh_config()
        data = load_dataset(args.data, cfg.history, None)
        if data.get("meta", {}).get("source") != "generated_only":
            raise ValueError("benchmark data must declare generated_only provenance")
        if data.get("meta", {}).get("oracle_search") != "complete_only":
            raise ValueError("benchmark data must declare complete_only oracle provenance")
        distances = np.asarray(data["distances"])
        if distances.dtype.kind not in "iu" or distances.ndim != 2 or distances.shape[1] != 4:
            raise ValueError("benchmark distances must be integer [N,4]")
        finite_distances = distances[distances >= 0]
        maximum_distance = int(finite_distances.max()) if finite_distances.size else -1
        if maximum_distance >= cfg.max_distance:
            raise ValueError(f"fresh config max_distance={cfg.max_distance} cannot represent shard distance {maximum_distance}")
        report["distance_support"] = {
            "maximum_finite_distance": maximum_distance,
            "config_max_distance": int(cfg.max_distance),
            "unreachable_branches": int((distances < 0).sum()),
        }
        batch, batch_description = fixed_batch(data, args.batch_size)
        report["batch"] = batch_description
        report["metadata"] = data.get("meta", {})
        report["config"] = dict(cfg.__dict__)
        weights = {**DEFAULT_WEIGHTS, "successor_policy": 1.0}
        report["weights"] = weights
        report["optimizer"] = {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.05,
                                "fused": args.optimizer == "fused"}
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            report["cuda_device"] = {
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory_bytes": int(properties.total_memory),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        report["stages"]["load"] = "complete"
        atomic_write(report_path, report)
        attempts: list[dict[str, Any]] = []
        sizes = [args.batch_size]
        if args.allow_oom_fallback:
            sizes.extend(args.batch_size >> shift for shift in range(1, args.batch_size.bit_length()))
        for size in sizes:
            if size != args.batch_size:
                batch = {name: value[:size].contiguous() for name, value in batch.items()}
            try:
                candidate = run_candidate(args, data, batch, cfg, weights, device, size, report)
                attempts.append(candidate)
                report["candidate"] = candidate
                break
            except torch.cuda.OutOfMemoryError as error:
                attempts.append({"status": "cuda_oom", "batch_size": size,
                                 "error": repr(error)})
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if not args.allow_oom_fallback:
                    raise
        else:
            raise RuntimeError("no requested or fallback batch size completed")
        report["attempts"] = attempts
        report["stages"]["benchmark"] = "complete"
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_write(report_path, report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        report["stages"] = {**report.get("stages", {}), "cleanup": "complete"}
        report["data_sha256_after"] = file_sha256(args.data)
        report["sources_after"] = hashes(source_files)
        report["source_files_unchanged"] = report["sources_before"] == report["sources_after"]
        report["data_unchanged"] = report.get("data_sha256_before") == report["data_sha256_after"]
        if not report["source_files_unchanged"] or not report["data_unchanged"]:
            report["status"] = "failed"
            report["error"] = {
                "type": "ProvenanceDrift",
                "message": "source or input data bytes changed during the benchmark",
            }
        report["finished_monotonic"] = time.monotonic()
        atomic_write(report_path, report)
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if report.get("status") == "failed" and report.get("error", {}).get("type") == "ProvenanceDrift":
        raise RuntimeError(report["error"]["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
