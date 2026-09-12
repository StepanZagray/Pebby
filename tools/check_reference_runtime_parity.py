"""Compare eager SDPA execution with math SDPA plus compiled core.

This diagnostic loads the final parameter vector from a completed performance
probe, runs one full generated B=1024 loss/backward pass per arm, and writes
only numerical comparison evidence.  It never steps an optimizer or saves a
model checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from pebby.agent.world_model import WorldPolicy
from pebby.agent.world_train import as_tensors, load_dataset
from pebby.agent.world_training_objectives import world_losses
from pebby.agent.world_runtime import configure_execution


DEFAULT_DIR = ROOT / "artifacts/reference-gpu-no-inner-checkpoint-v1"
SOURCE_NAMES = (
    "world_model.py", "world_training_objectives.py", "world_train.py",
    "world_runtime.py", "world_grounding.py", "world_rollout.py",
    "world_readout.py", "glyph_model.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proc_start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return int(text[text.rfind(")") + 2 :].split()[19])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def memory_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def gpu_compute_processes() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"could not prove GPU idleness: {error!r}") from error
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr[-500:]}")
    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0]:
            try:
                processes.append({"pid": int(fields[0]),
                                  "name": fields[1] if len(fields) > 1 else "",
                                  "used_memory_mib": float(fields[2]) if len(fields) > 2 else None})
            except ValueError:
                raise RuntimeError(f"unparseable nvidia-smi row: {line!r}")
    return processes


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--parity-json", type=Path)
    parser.add_argument("--parity-npz", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--deadline-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.batch_size != 1024 or args.chunk_size != 128:
        raise ValueError("this diagnostic is pinned to B=1024 and encoder chunk 128")
    if args.seed != 42:
        raise ValueError("this diagnostic is pinned to seed 42")
    if not 1 <= args.deadline_seconds <= 300:
        raise ValueError("deadline-seconds must be in 1..300")
    args.benchmark_dir = args.benchmark_dir.expanduser().resolve()
    args.report = (args.report or args.benchmark_dir / "report.json").expanduser().resolve()
    args.parity_json = (args.parity_json or args.benchmark_dir / "parity.json").expanduser().resolve()
    args.parity_npz = (args.parity_npz or args.benchmark_dir / "parity.npz").expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.out.exists():
        raise FileExistsError(args.out)
    for path in (args.report, args.parity_json, args.parity_npz):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    return args


def source_files() -> list[Path]:
    return [Path(__file__).resolve(), *(ROOT / "pebby/agent" / name for name in SOURCE_NAMES)]


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    report = json.loads(args.report.read_text())
    parity_meta = json.loads(args.parity_json.read_text())
    if report.get("status") != "complete" or not report.get("data_unchanged") or not report.get('source_files_unchanged'):
        raise ValueError("completed unchanged benchmark report required")
    if report.get("candidate", {}).get("status") != "complete":
        raise ValueError("benchmark candidate is incomplete")
    settings = report.get("settings", {})
    if (settings.get("batch_size") != 1024 or settings.get("chunk_size") != 128
            or settings.get("checkpoint_encoder") is not True
            or settings.get("checkpoint_loops") is not False
            or settings.get("precision") != "bf16"
            or settings.get("tf32") is not False):
        raise ValueError("benchmark settings do not match the pinned parity contract")
    if parity_meta.get("format") != "pebby.reference-gpu-benchmark-parity.v1":
        raise ValueError("unsupported parity artifact format")
    if parity_meta.get("checkpoint") or parity_meta.get("optimizer_state"):
        raise ValueError("parity artifact must not contain a checkpoint or optimizer state")
    if int(report.get("batch", {}).get("rows_used", -1)) != args.batch_size:
        raise ValueError("benchmark artifact does not contain the required B=1024 rows")
    if int(report.get("settings", {}).get("seed", -1)) != args.seed:
        raise ValueError("benchmark and diagnostic seeds differ")
    with np.load(args.parity_npz, allow_pickle=False) as archive:
        parameters = np.array(archive["parameters"], dtype=np.float32, copy=True)
        if "gradient_present" in archive:
            gradient_present = np.array(archive["gradient_present"], dtype=np.bool_, copy=True)
        else:
            gradient_present = None
    if sha256(args.parity_npz) != parity_meta.get("sha256"):
        raise ValueError("parity NPZ hash does not match parity metadata")
    if int(parameters.size) != int(parity_meta.get("parameter_count", -1)):
        raise ValueError("parity parameter count mismatch")
    if gradient_present is not None and gradient_present.size != len(parity_meta["names"]):
        raise ValueError("parity gradient presence shape mismatch")
    data_path = Path(report["data"]).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    return report, {"meta": parity_meta, "parameters": parameters,
                    "gradient_present": gradient_present}, data_path


def restore_parameters(model: torch.nn.Module, parity: dict[str, Any], device: torch.device) -> None:
    metadata = parity["meta"]
    expected_names = list(metadata["names"])
    expected_shapes = [tuple(shape) for shape in metadata["shapes"]]
    actual = list(model.named_parameters())
    if [name for name, _ in actual] != expected_names:
        raise ValueError("model parameter names differ from the benchmark state")
    offset = 0
    with torch.no_grad():
        for (name, parameter), shape in zip(actual, expected_shapes, strict=True):
            if tuple(parameter.shape) != shape:
                raise ValueError(f"parameter shape changed for {name}")
            count = parameter.numel()
            values = parity["parameters"][offset:offset + count].reshape(shape)
            parameter.copy_(torch.from_numpy(values).to(device=device, dtype=parameter.dtype))
            offset += count
    if offset != parity["parameters"].size or offset != int(metadata["parameter_count"]):
        raise ValueError("parameter vector was not consumed exactly")


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def run_arm(model: torch.nn.Module, cpu_batch: dict[str, torch.Tensor], device: torch.device,
            weights: dict[str, float], cuda_rng_state: torch.Tensor) -> dict[str, Any]:
    torch.cuda.set_rng_state(cuda_rng_state, device=device)
    model.zero_grad(set_to_none=True)
    batch = {name: value.to(device=device, non_blocking=True) for name, value in cpu_batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = world_losses(model, batch, weights)
        total = output["total"]
    total.backward()
    torch.cuda.synchronize(device)
    logits = output["logits"].detach().float()
    return {"total": scalar(total),
            "losses": {name: scalar(value) for name, value in output["losses"].items()},
            "logits": logits.cpu().numpy().copy(),
            "model": model}


def compare_logits(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    a = torch.from_numpy(left.astype(np.float32, copy=False))
    b = torch.from_numpy(right.astype(np.float32, copy=False))
    delta = (b - a).abs()
    pa, pb = a.softmax(-1), b.softmax(-1)
    kl = (pa * (pa.clamp_min(1e-30).log() - pb.clamp_min(1e-30).log())).sum(-1)
    top_a = a.topk(2, dim=-1).values
    top_b = b.topk(2, dim=-1).values
    disagreement = a.argmax(-1) != b.argmax(-1)
    return {
        "shape": list(left.shape),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "softmax_kl_auto_to_math_compile": float(kl.mean()),
        "argmax_disagreement_fraction": float(disagreement.float().mean()),
        "argmax_disagreement_count": int(disagreement.sum()),
        "auto_tie_gap_mean": float((top_a[:, 0] - top_a[:, 1]).mean()),
        "auto_tie_gap_min": float((top_a[:, 0] - top_a[:, 1]).min()),
        "math_compile_tie_gap_mean": float((top_b[:, 0] - top_b[:, 1]).mean()),
        "math_compile_tie_gap_min": float((top_b[:, 0] - top_b[:, 1]).min()),
        "disagreement_auto_tie_gap_mean": float((top_a[:, 0] - top_a[:, 1])[disagreement].mean())
        if bool(disagreement.any()) else None,
    }


def compare_gradients(auto: torch.nn.Module, compiled: torch.nn.Module) -> dict[str, Any]:
    left, right, per_parameter = [], [], []
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
            auto.named_parameters(), compiled.named_parameters(), strict=True):
        if name_a != name_b or parameter_a.shape != parameter_b.shape:
            raise ValueError(f"parameter mismatch at {name_a!r}/{name_b!r}")
        grad_a = None if parameter_a.grad is None else parameter_a.grad.detach().float().cpu().numpy().reshape(-1)
        grad_b = None if parameter_b.grad is None else parameter_b.grad.detach().float().cpu().numpy().reshape(-1)
        present_a, present_b = grad_a is not None, grad_b is not None
        if present_a != present_b:
            raise ValueError(f"gradient presence differs for {name_a}")
        if not present_a:
            per_parameter.append({"name": name_a, "present": False})
            continue
        left.append(grad_a.astype(np.float64, copy=False))
        right.append(grad_b.astype(np.float64, copy=False))
        delta = grad_b.astype(np.float64) - grad_a.astype(np.float64)
        per_parameter.append({"name": name_a, "present": True,
                              "numel": int(delta.size),
                              "auto_norm": float(np.linalg.norm(grad_a)),
                              "math_compile_norm": float(np.linalg.norm(grad_b)),
                              "max_abs": float(np.abs(delta).max()),
                              "l2": float(np.linalg.norm(delta)),
                              "relative_l2": float(np.linalg.norm(delta) /
                                                   max(np.linalg.norm(grad_a), 1e-30))})
    a = np.concatenate(left) if left else np.empty(0, dtype=np.float64)
    b = np.concatenate(right) if right else np.empty(0, dtype=np.float64)
    delta = b - a
    return {
        "parameter_count": len(per_parameter),
        "gradient_present_count": sum(item["present"] for item in per_parameter),
        "global": {"auto_norm": float(np.linalg.norm(a)),
                   "math_compile_norm": float(np.linalg.norm(b)),
                   "max_abs": float(np.abs(delta).max()) if delta.size else 0.,
                   "l2": float(np.linalg.norm(delta)),
                   "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(a), 1e-30)),
                   "cosine": float(np.dot(a, b) /
                                   max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))},
        "top_drift": sorted((item for item in per_parameter if item["present"]),
                             key=lambda item: item["max_abs"], reverse=True)[:20],
        "per_parameter": per_parameter,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    available = memory_available()
    if available < 6 * 2 ** 30:
        raise RuntimeError("MemAvailable is below the 6 GiB safety floor")
    gpu_process_snapshot = gpu_compute_processes()
    processes = [item for item in gpu_process_snapshot if item["pid"] != os.getpid()]
    if processes:
        raise RuntimeError(f"GPU is not idle: {processes}")
    started = time.monotonic()
    source_list = source_files()
    artifact_list = [args.report, args.parity_json, args.parity_npz]
    source_before = {str(path): sha256(path) for path in source_list}
    artifact_before = {str(path): sha256(path) for path in artifact_list}
    report: dict[str, Any] = {"format": "pebby.reference-runtime-parity.v1", "status": "running",
        "pid": os.getpid(), "start_ticks": proc_start_ticks(os.getpid()), "argv": sys.argv,
        "source_hashes_before": source_before, "artifact_hashes_before": artifact_before,
        "host_mem_available_before": available, "gpu_processes_before": gpu_process_snapshot,
        "settings": {"batch_size": 1024, "chunk_size": 128, "history": 8,
                      "seed": args.seed, "precision": "bf16", "tf32": False,
                      "auto": {"compile_core": False, "temporal_backend": "auto"},
                      "math_compile": {"compile_core": True, "temporal_backend": "math"}}}
    atomic_write(args.out, report)
    auto_model = compiled_model = None
    auto = compiled = None
    def deadline(*_):
        raise TimeoutError('runtime parity deadline exceeded')
    old_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        report_data, parity, data_path = load_inputs(args)
        data_before = sha256(data_path)
        if data_before != report_data.get("data_sha256_before"):
            raise ValueError("benchmark report data hash does not match the current data file")
        config = report_data["config"]
        weights = report_data["weights"]
        data = load_dataset(data_path, config["history"], None)
        tensors = as_tensors(data)
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(args.batch_size, generator=generator)
        cpu_batch = {name: tensor[:args.batch_size][order].contiguous()
                     for name, tensor in tensors.items()}
        device = torch.device("cuda")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        auto_model = WorldPolicy(config).to(device).train()
        auto_model.checkpoint_encoder = True
        auto_model.checkpoint_loops = False
        auto_model.encoder_chunk_size = 128
        restore_parameters(auto_model, parity, device)
        compiled_model = WorldPolicy(config).to(device).train()
        compiled_model.checkpoint_encoder = True
        compiled_model.checkpoint_loops = False
        compiled_model.encoder_chunk_size = 128
        restore_parameters(compiled_model, parity, device)
        configure_execution(compiled_model, compile_core=True, temporal_backend="math")
        cuda_rng_state = torch.cuda.get_rng_state(device)
        if time.monotonic() - started > args.deadline_seconds:
            raise TimeoutError("deadline reached before execution")
        auto = run_arm(auto_model, cpu_batch, device, weights, cuda_rng_state)
        if time.monotonic() - started > args.deadline_seconds:
            raise TimeoutError("deadline reached after auto execution")
        compiled = run_arm(compiled_model, cpu_batch, device, weights, cuda_rng_state)
        report.update({"status": "complete", "data": str(data_path), "data_sha256_before": data_before,
                       "data_rows": args.batch_size, "config": config, "weights": weights,
                       "parameter_count": int(parity["meta"]["parameter_count"]),
                       "losses": {"auto": {"total": auto["total"], **auto["losses"]},
                                  "math_compile": {"total": compiled["total"], **compiled["losses"]}},
                       "logits": compare_logits(auto["logits"], compiled["logits"]),
                       "gradients": compare_gradients(auto_model, compiled_model),
                       "notes": ["No optimizer step; both arms use identical loaded parameters and one full loss/backward.",
                                 "SIGReg CUDA RNG state was reset identically before each arm.",
                                 "CPU row permutation was generated once with seed 42 and reused."]})
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        auto = compiled = None
        del auto_model, compiled_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        report["source_hashes_after"] = {str(path): sha256(path) for path in source_list}
        report["artifact_hashes_after"] = {str(path): sha256(path) for path in artifact_list}
        report["data_sha256_after"] = sha256(data_path) if "data_path" in locals() else None
        report["source_unchanged"] = report["source_hashes_before"] == report["source_hashes_after"]
        report["artifacts_unchanged"] = report["artifact_hashes_before"] == report["artifact_hashes_after"]
        report["data_unchanged"] = report.get("data_sha256_before") == report["data_sha256_after"]
        report["elapsed_seconds"] = time.monotonic() - started
        if not report["source_unchanged"] or not report["artifacts_unchanged"] or not report["data_unchanged"]:
            report.update(status="failed", error={"type": "ProvenanceDrift",
                           "message": "source, parameter artifact, or input data changed"})
        atomic_write(args.out, report)
    if report["status"] != "complete":
        raise RuntimeError(report["error"]["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
