"""Cache frozen current public H8 encodings for a new one-step outcome model.

Only main() runs extraction. The protected base parent is never trained; no
successor pixels, positional additions, or old projector weights are published.
"""

import argparse
from datetime import datetime, timedelta
import gc
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/reference-world-base-v1"
SELECTION = ROOT / "data/reference-position-inputs-v1"
PARENT = ROOT / "checkpoints/ls20-reference-base-v1.pt"
PARENT_SHA = "6db7f40d4008ff809e9b3a8b05d6a9a18801b343f02e52a9f585eece5e31cff9"
PUBLIC = ("frames", "history_valid", "previous_actions")
SCHEMA = {"raw": ("float32", (160, 64)), "state": ("float32", (160, 64)),
          "glyph": ("float32", (14,)), "seeds": ("int64", ()), "rows": ("int64", ()),
          "optimal": ("uint8", ()), "next_optimal": ("uint8", (4,)),
          "next_player_cell": ("int16", (4, 2)), "next_triple": ("int16", (4, 3)),
          "next_steps": ("int16", (4,)), "next_lives": ("int16", (4,)),
          "distances": ("int16", (4,)), "lost_life": ("bool", (4,)),
          "terminal": ("bool", (4,)), "won": ("bool", (4,)),
          "player_cell": ("int16", (2,)), "current_triple": ("int16", (3,)),
          "current_steps": ("int16", ()), "current_lives": ("int16", ())}
LABELS = tuple(key for key in SCHEMA if key not in ("raw", "state", "glyph", "rows"))
COUNTS = {"train": (80000, 10000), "validation": (4000, 500)}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stat(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def process_start_ticks(pid=None):
    fields = Path(f"/proc/{os.getpid() if pid is None else pid}/stat").read_text().rpartition(")")[2].split()
    return int(fields[19])


def require_no_foreign_cuda():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=10)
    pids = sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})
    foreign = [pid for pid in pids if pid != os.getpid()]
    if foreign:
        raise RuntimeError(f"foreign CUDA compute processes are active: {foreign}")
    return pids


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Bindings:
    """Hash each input once; reject metadata changes during hashing or extraction."""

    def __init__(self):
        self.hashes, self.before = {}, {}

    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        if path not in self.hashes:
            before = stat(path)
            value = sha(path)
            if stat(path) != before:
                raise ValueError(f"source changed during hashing: {path}")
            self.hashes[path], self.before[path] = value, before
        if expected is not None and self.hashes[path] != expected:
            raise ValueError(f"source SHA256 mismatch: {path}")
        return self.hashes[path]

    def verify(self):
        after = {path: stat(path) for path in self.before}
        if after != self.before:
            raise ValueError("source file stats changed during cache extraction")
        return after


def release(arrays, *, close=False, flush=False):
    for value in arrays.values():
        backing = getattr(value, "_mmap", None)
        if backing is not None and not backing.closed:
            if flush:
                value.flush()
            backing.madvise(mmap.MADV_DONTNEED)
            if close:
                backing.close()


def guard(report, started, deadline):
    available = next(int(s.split()[1]) * 1024 for s in Path("/proc/meminfo").read_text().splitlines()
                     if s.startswith("MemAvailable:"))
    report["minimum_memavailable_bytes"] = min(available, report.get("minimum_memavailable_bytes", available))
    if available < 7 * 2**30:
        raise MemoryError("below 7 GiB abort threshold protecting the 6 GiB reserve")
    if time.monotonic() - started >= deadline:
        raise TimeoutError("cache preparation exceeded its extraction deadline")


def open_array(path, info, bindings):
    bindings.add(path, info["sha256"])
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != info["shape"] or value.dtype.str != info["dtype"]:
        value._mmap.close()
        raise ValueError(f"array schema mismatch: {path}")
    return value


def select_current(position, source, expected_rows, expected_levels, fixed):
    """Use the published source0/current selection, retaining its exact row order."""
    indices = np.flatnonzero((position["source"] == 0) & (position["branch"] == -1))
    rows = np.asarray(position["rows"][indices], dtype=np.int64)
    if len(rows) != expected_rows or len(np.unique(rows)) != expected_rows:
        raise ValueError("current selection has wrong count or duplicate source rows")
    if np.any(rows < 0) or np.any(rows >= len(source["seeds"])) or not np.array_equal(rows, fixed):
        raise ValueError("current selection differs from published selected source indices")
    seeds = np.asarray(source["seeds"][rows], dtype=np.int64)
    unique, counts = np.unique(seeds, return_counts=True)
    if len(unique) != expected_levels or not np.all(counts == 8):
        raise ValueError("selection must contain eight roots from every required level")
    if not np.array_equal(unique, np.unique(source["seeds"])):
        raise ValueError("selection omits a source level")
    if not np.array_equal(seeds, position["seeds"][indices]):
        raise ValueError("selected source and cached seeds disagree")
    return rows, indices, unique


def encode_current(model, arrays, rows, device):
    import torch
    public = [torch.from_numpy(np.array(arrays[key][rows], copy=True)).to(device) for key in PUBLIC]
    if tuple(public[0].shape[1:]) != (8, 64, 64):
        raise ValueError("current encoder inputs must be public H8 observations")
    encoding = model.encode(*public)
    for key in ("raw", "state", "glyph"):
        value = encoding[key]
        if (tuple(value.shape) != (len(rows), *SCHEMA[key][1]) or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid current encoding: {key}")
    return encoding


def original_inputs(model, encoding):
    result = model.projector_inputs(encoding["state"], encoding["raw"][:, 144:].flatten(1), encoding["glyph"])
    if result.shape[-1] != 1742:
        raise ValueError("protected parent must expose its original 1742 projector inputs")
    return result


def verify_parity(model, arrays, rows, position, position_indices, device, cached=None):
    """Check old math cache separately from native auto/cuDNN-TF32 behavior."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    chosen = np.unique(np.linspace(0, len(rows) - 1, min(8, len(rows)), dtype=np.int64))
    selected = rows[chosen]
    native = encode_current(model, arrays, selected, device)
    native_projector = original_inputs(model, native)
    expected = torch.from_numpy(np.array(position["inputs"][position_indices[chosen], :1742], copy=True)).to(device)
    old_chunk, old_cudnn = model.encoder_chunk_size, torch.backends.cudnn.allow_tf32
    try:
        model.encoder_chunk_size = 128
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            old = encode_current(model, arrays, selected, device)
            old_projector = original_inputs(model, old)
        torch.testing.assert_close(old_projector, expected, atol=3e-5, rtol=1e-4)
    finally:
        model.encoder_chunk_size = old_chunk
        torch.backends.cudnn.allow_tf32 = old_cudnn
    torch.testing.assert_close(native_projector, expected, atol=3e-3, rtol=3e-3)
    result = dict(source_rows=selected.tolist(), position_cache_rows=position_indices[chosen].tolist(),
                  old_math_vs_original_max_abs=float((old_projector - expected).abs().max()),
                  native_auto_vs_original_max_abs=float((native_projector - expected).abs().max()),
                  old_cache_tolerance=dict(atol=3e-5, rtol=1e-4),
                  native_backend_tolerance=dict(atol=3e-3, rtol=3e-3))
    if cached is not None:
        differences = {}
        for key in ("raw", "state", "glyph"):
            stored = torch.from_numpy(np.array(cached[key][chosen], copy=True)).to(device)
            torch.testing.assert_close(stored, native[key], atol=3e-4, rtol=3e-4)
            differences[key] = float((stored - native[key]).abs().max())
        rebuilt = {key: torch.from_numpy(np.array(cached[key][chosen], copy=True)).to(device)
                   for key in ("raw", "state", "glyph")}
        torch.testing.assert_close(original_inputs(model, rebuilt), native_projector, atol=3e-4, rtol=3e-4)
        result["direct_encode_reconstruction"] = differences
    return result


def required_bytes(rows):
    return sum(rows * np.dtype(dtype).itemsize * int(np.prod(tail, dtype=int)) + 128
               for dtype, tail in SCHEMA.values())


def publish_directory(staging, output):
    """Linux atomic directory publication that cannot replace an existing target."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def validate_published(path):
    """Read-only validator; hashes each output once and returns bindings for a loader.

    The returned ``validated_output_stats`` can be checked again after training.
    This does not write the manifest, map features writable, or create a cache.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("parent_sha256") != PARENT_SHA
            or not manifest.get("sources_unchanged") or not manifest.get("validation_disjoint")
            or manifest.get("public_inputs") != list(PUBLIC)
            or manifest.get("current_source_id") != 0 or manifest.get("current_branch") != -1):
        raise ValueError("not a verified current-only protected-parent outcome cache")
    guard_bindings, levels = Bindings(), {}
    guard_bindings.add(path / "manifest.json")
    for split, (count, level_count) in COUNTS.items():
        if set(manifest["arrays"][split]) != set(SCHEMA):
            raise ValueError("published cache array names differ from outcome schema")
        seeds = rows = None
        try:
            for key, (dtype, tail) in SCHEMA.items():
                info = manifest["arrays"][split][key]
                if info["shape"] != [count, *tail] or info["dtype"] != np.dtype(dtype).str:
                    raise ValueError(f"published array schema mismatch: {split}/{key}")
                array = open_array(path / split / f"{key}.npy", info, guard_bindings)
                if key == "seeds":
                    seeds = np.array(array, copy=True)
                if key == "rows":
                    rows = np.array(array, copy=True)
                release({key: array}, close=True)
            unique, counts = np.unique(seeds, return_counts=True)
            if len(unique) != level_count or not np.all(counts == 8):
                raise ValueError("published seed counts do not cover all levels eight times")
            if len(np.unique(rows)) != count or np.any(rows < 0):
                raise ValueError("published source rows are duplicated or negative")
            if hashlib.sha256(rows.tobytes()).hexdigest() != manifest["selection"][split]["source_rows_sha256"]:
                raise ValueError("published source row order changed")
            levels[split] = unique
        finally:
            del seeds, rows
    if np.intersect1d(levels["train"], levels["validation"]).size:
        raise ValueError("published TRAIN and validation seeds overlap")
    manifest["validated_output_stats"] = guard_bindings.verify()
    manifest["validated_output_hashes"] = guard_bindings.hashes
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/reference-outcome-inputs-v1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.deadline_seconds <= 0:
        parser.error("batch size and deadline must be positive")
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bytes_needed = sum(required_bytes(n) for n, _ in COUNTS.values())
    free = shutil.disk_usage(output.parent).free
    if free < bytes_needed + 2**30:
        raise OSError("insufficient disk space for cache plus 1 GiB margin")
    started, started_local = time.monotonic(), datetime.now().astimezone()
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    staging.mkdir()
    report = dict(status="validating", pid=os.getpid(), start_ticks=process_start_ticks(), started_local=started_local.isoformat(),
                  deadline_local=(started_local + timedelta(seconds=args.deadline_seconds)).isoformat(),
                  output=str(output), staging=str(staging), storage_bytes_required=bytes_needed,
                  initial_disk_free_bytes=free, parent_path=str(PARENT), parent_sha256=PARENT_SHA,
                  public_inputs=list(PUBLIC), official_frames_or_routes_used=False,
                  current_source_id=0, current_branch=-1, policy_optimizer_used=False,
                  frozen_parent=True, positional_extra_used=False, actual_successor_features_used=False,
                  settings=dict(device=args.device, batch_size=args.batch_size, precision="float32",
                                execution="native_eager", temporal_backend="auto", matmul_tf32=False,
                                cudnn_tf32=True, encoder_chunk_size=0, reserve_gib=6,
                                abort_memavailable_gib=7, deadline_seconds=args.deadline_seconds), progress={})
    bindings, sources, positions, outputs, selection = Bindings(), {}, {}, {}, {}
    write_json(staging / "progress.json", report)
    print(json.dumps(dict(pid=os.getpid(), started_local=report["started_local"],
                          deadline_local=report["deadline_local"], status="validating")), flush=True)
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def deadline_handler(*_):
        raise TimeoutError(f"{args.deadline_seconds}-second bounded cache deadline expired")
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.alarm(args.deadline_seconds)
    model = None
    try:
        guard(report, started, args.deadline_seconds)
        bindings.add(PARENT, PARENT_SHA)
        bindings.add(SELECTION / "manifest.json")
        prior = json.loads((SELECTION / "manifest.json").read_text())
        if (prior["status"] != "complete" or prior["parent_sha256"] != PARENT_SHA
                or not prior["sources_unchanged"] or not prior["public_input_only"]):
            raise ValueError("position selection is not a verified protected-parent cache")
        bindings.add(Path(__file__))
        for name in ("world_model.py", "world_runtime.py", "glyph_model.py", "world_readout.py", "model.py", "looped.py"):
            path = ROOT / "pebby/agent" / name
            bindings.add(path, prior["source_bindings"].get(str(path)))
        bindings.add(ROOT / "pebby/ls20/names.py")
        for path in (DATA / "manifest.json", DATA / "build-report.json"):
            bindings.add(path, prior["source_bindings"][str(path)])
        build = json.loads((DATA / "build-report.json").read_text())
        if build["status"] != "complete" or build["official_frames_or_routes_used"] or not build["sources_unchanged"]:
            raise ValueError("base data is not verified generated-only data")
        report["upstream_calibration"] = prior.get("upstream_bank_calibration")
        for split, (count, level_count) in COUNTS.items():
            guard(report, started, args.deadline_seconds)
            name = "base" if split == "train" else "validation"
            source_hash = prior["published_npz_sha256"][name]
            bindings.add(DATA / f"{split}.npz", source_hash)
            cache = DATA / "array-cache" / (source_hash + "-58c1c61b602f42df")
            bindings.add(cache / "manifest.json", prior["source_bindings"][str(cache / "manifest.json")])
            manifest = json.loads((cache / "manifest.json").read_text())
            if manifest["source_sha256"] != source_hash:
                raise ValueError("source cache and published NPZ disagree")
            sources[split] = {}
            for key in (*PUBLIC, *LABELS, "meta"):
                guard(report, started, args.deadline_seconds)
                sources[split][key] = open_array(cache / f"{key}.npy", manifest["arrays"][key], bindings)
            if json.loads(str(sources[split]["meta"].item()))["source"] != "generated_only":
                raise ValueError("source observations are not generated-only")
            positions[split] = {key: open_array(SELECTION / split / f"{key}.npy", prior["outputs"][split][key], bindings)
                                for key in ("source", "branch", "rows", "seeds", "inputs")}
            fixed_path = SELECTION / prior["selection"][name]["path"]
            bindings.add(fixed_path, prior["selection"][name]["sha256"])
            fixed = np.load(fixed_path, allow_pickle=False)
            selection[split] = select_current(positions[split], sources[split], count, level_count, fixed)
            release(sources[split]); release(positions[split])
        if np.intersect1d(selection["train"][2], selection["validation"][2]).size:
            raise ValueError("TRAIN and validation seeds overlap")
        report["selection"] = {split: dict(rows=len(value[0]), levels=len(value[2]), roots_per_level=8,
                                                   source_rows_sha256=hashlib.sha256(value[0].tobytes()).hexdigest())
                               for split, value in selection.items()}
        report["validation_disjoint"] = True
        report["source_sha256"], report["source_stats_before"] = bindings.hashes, bindings.before
        write_json(staging / "progress.json", report)
        print(json.dumps(dict(pid=os.getpid(), started_local=report["started_local"],
                              deadline_local=report["deadline_local"], status="sources_verified")), flush=True)
        import torch
        from pebby.agent.world_model import load_world_checkpoint
        from pebby.agent.world_runtime import configure_execution
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("highest")
        guard(report, started, args.deadline_seconds)
        bindings.verify()
        report["cuda_compute_pids_before_load"] = require_no_foreign_cuda() if args.device == "cuda" else []
        model, checkpoint = load_world_checkpoint(PARENT, args.device)
        guard(report, started, args.deadline_seconds)
        model.eval().requires_grad_(False)
        model.encoder_chunk_size = 0
        report["runtime"] = configure_execution(model, compile_core=False, temporal_backend="auto")
        report["parent_config"] = model.config()
        report["parent_config_sha256"] = hashlib.sha256(json.dumps(checkpoint["config"], sort_keys=True).encode()).hexdigest()
        if model.config() != checkpoint["config"]:
            raise ValueError("parent config changed while loading")
        report["parent_parameters"] = model.parameter_count()
        report["parity"] = {}
        with torch.inference_mode():
            for split, (rows, position_indices, _) in selection.items():
                guard(report, started, args.deadline_seconds)
                report["parity"][split] = {"before": verify_parity(model, sources[split], rows, positions[split],
                                                                    position_indices, args.device)}
                directory = staging / split
                directory.mkdir()
                outputs[split] = {key: np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+", dtype=dtype,
                                                                shape=(len(rows), *tail)) for key, (dtype, tail) in SCHEMA.items()}
                split_started = time.monotonic()
                for begin in range(0, len(rows), args.batch_size):
                    guard(report, started, args.deadline_seconds)
                    chosen = rows[begin:begin + args.batch_size]
                    encoding = encode_current(model, sources[split], chosen, args.device)
                    payload = {key: encoding[key].cpu().numpy() for key in ("raw", "state", "glyph")}
                    payload.update({key: np.array(sources[split][key][chosen], copy=True) for key in LABELS})
                    payload["rows"] = chosen
                    for key, value in payload.items():
                        if not np.can_cast(value.dtype, np.dtype(SCHEMA[key][0]), casting="safe"):
                            raise ValueError(f"unsafe label conversion: {key}")
                        outputs[split][key][begin:begin + len(chosen)] = value
                    del encoding, payload
                    release(sources[split])
                    if begin % (args.batch_size * 16) == 0 or begin + len(chosen) == len(rows):
                        release(outputs[split], flush=True)
                        done = begin + len(chosen)
                        seconds_left = (time.monotonic() - split_started) * (len(rows) - done) / done
                        report["progress"][split] = dict(rows=done, total=len(rows),
                            updated_local=datetime.now().astimezone().isoformat(),
                            eta_local=(datetime.now().astimezone() + timedelta(seconds=seconds_left)).isoformat())
                        print(json.dumps(dict(pid=os.getpid(), split=split, **report["progress"][split])), flush=True)
                        write_json(staging / "progress.json", report)
                report["parity"][split]["after"] = verify_parity(model, sources[split], rows, positions[split],
                                                                   position_indices, args.device, outputs[split])
                if any(p.grad is not None or p.requires_grad for p in model.parameters()):
                    raise ValueError("frozen encoder acquired gradients")
                release(outputs[split], close=True, flush=True)
                release(sources[split], close=True); release(positions[split], close=True)
        report["arrays"] = {}
        for split, (count, _) in COUNTS.items():
            report["arrays"][split] = {}
            for key, (dtype, tail) in SCHEMA.items():
                guard(report, started, args.deadline_seconds)
                path = staging / split / f"{key}.npy"
                report["arrays"][split][key] = dict(shape=[count, *tail], dtype=np.dtype(dtype).str,
                                                    sha256=sha(path), size_bytes=path.stat().st_size)
                path.chmod(0o444)
        report["source_stats_after"] = bindings.verify()
        report.update(status="complete", sources_unchanged=True, all_features_finite=True,
                      policy_gradients_absent=True, elapsed_seconds=time.monotonic() - started,
                      finished_local=datetime.now().astimezone().isoformat())
        write_json(staging / "manifest.json", report)
        (staging / "progress.json").unlink()
        (staging / "manifest.json").chmod(0o444)
        publish_directory(staging, output)
        print(json.dumps(dict(status="published", pid=os.getpid(), output=str(output))), flush=True)
        return report
    except BaseException as error:
        report.update(status="failed", error=repr(error), partial_unpublished=True)
        if staging.exists():
            write_json(staging / "progress.json", report)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        for groups in (sources, positions, outputs):
            for arrays in groups.values():
                release(arrays, close=True)
        del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
