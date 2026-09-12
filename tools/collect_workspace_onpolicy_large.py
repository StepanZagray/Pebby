"""Collect a bounded generated TRAIN on-policy level bank for a workspace checkpoint.

This sibling collector keeps the established world-transition row contract while
using the generic public checkpoint loader.  It deliberately reuses the checked
``collect_level`` and disk-backed ``RowStore`` from ``collect_onpolicy_world``:
all policy decisions see only the public H8 history, while the complete oracle
and four real branches are used afterward for labels and mechanical checks.

The default bank is 512 stratified levels from each disjoint TRAIN source
(1024 distinct levels total). It is preparation for a later cache/training
run; this module does not start collection during import or tests.
"""

from __future__ import annotations

from pebby.ls20.provenance import generated_context, validate_difficulty

import argparse
from collections import Counter
import json
import multiprocessing
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import torch

from pebby.agent import world_data as wd
from pebby.agent.model import load_checkpoint
from pebby.agent.world_train import load_dataset, require_verified_data, require_winning_coverage
from pebby.ls20.generate import FORMAT as LEVEL_FORMAT, GENERATOR_VERSION
from tools.collect_onpolicy_world import RowStore, collect_level as _collect_level, select as _select
from tools.goal_attribute_probes import digest


CHECKPOINT = Path("checkpoints/ls20-structured-workspace-comparison-600-evolving.pt")
EXPECTED_CHECKPOINT_SHA256 = "6a5d7716fca4403048434a8100e26770dc26fb9d19ce66600fb3eccca7d22e0c"
DEFAULT_OLD_BANK = Path("data/ls20-verified-train.jsonl")
DEFAULT_EXTENDED_BANK = Path("data/extended-bank-v1/train.jsonl")
DEFAULT_COMBINED_TRAIN = Path("data/ls20-world-combined-train.npz")
DEFAULT_ELIGIBLE_SEEDS = Path("data/structured-field-16384/train/seeds.npy")
DEFAULT_ARRAY_CACHE = Path("data/world-array-cache")
DEFAULT_OUT = Path("data/ls20-world-workspace-onpolicy-large-train.npz")
DEFAULT_REPORT = Path("artifacts/world-workspace-onpolicy-large.json")
MAX_ACTIONS = 48
HISTORY = 8
ROW_CAPACITY_PER_LEVEL = MAX_ACTIONS + 2

_WORKER_POLICY = None


def _json_lines(path: Path):
    try:
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid generated bank: {path}") from error
    if not values:
        raise ValueError(f"generated bank is empty: {path}")
    return values


def _validate_bank(path: Path, values):
    seeds = []
    for spec in values:
        if (not isinstance(spec, dict) or spec.get("format") != LEVEL_FORMAT
                or spec.get("generator_version") != GENERATOR_VERSION):
            raise ValueError(f"{path} is not a current generated TRAIN bank")
        seed = spec.get("seed")
        if type(seed) is not int or not 0 <= seed < 1_000_000:
            raise ValueError(f"{path} contains a non-TRAIN seed")
        validate_difficulty(spec)
        if spec.get("official_inputs_used") is True:
            raise ValueError(f"{path} is marked as using official inputs")
        if (spec.get("context_engine_verified") is not True
                or spec.get("search_truncated") is True
                or spec.get("training_context_index", generated_context(spec)) != generated_context(spec)):
            raise ValueError(f"{path} contains an unverified or mismatched contextual proof")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path} contains duplicate seeds")
    return set(seeds)


def _cached_seed_array(combined_train: Path, cache_root: Path):
    """Find the existing mmap seed array bound to the combined NPZ.

    The compressed NPZ is intentionally not decompressed just to obtain its
    seed column.  A cache is accepted only when its manifest source digest,
    seed metadata, and seed-array digest all agree.
    """
    source_sha = digest(combined_train)
    candidates = []
    for manifest_path in sorted(cache_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("source_sha256") != source_sha:
            continue
        info = manifest.get("arrays", {}).get("seeds")
        seed_path = manifest_path.parent / "seeds.npy"
        if not isinstance(info, dict) or not seed_path.exists():
            continue
        if digest(seed_path) != info.get("sha256"):
            raise ValueError(f"combined seed cache changed: {seed_path}")
        seeds = np.load(seed_path, mmap_mode="r", allow_pickle=False)
        if seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError(f"combined seed cache has invalid shape/dtype: {seed_path}")
        expected_shape = tuple(info.get("shape", ()))
        expected_dtype = info.get("dtype")
        try:
            dtype_matches = np.dtype(expected_dtype) == seeds.dtype
        except (TypeError, ValueError):
            dtype_matches = False
        if expected_shape != tuple(seeds.shape) or not dtype_matches:
            raise ValueError(f"combined seed cache metadata mismatch: {seed_path}")
        candidates.append((manifest_path, seed_path, manifest, seeds))
    if len(candidates) != 1:
        raise ValueError(f"need exactly one valid mmap seed cache for {combined_train}; found {len(candidates)}")
    return source_sha, candidates[0]


def _load_allowed_seeds(combined_train: Path, eligible_seeds: Path, array_cache: Path):
    combined_sha, (manifest_path, seed_path, manifest, combined_seeds) = _cached_seed_array(
        combined_train, array_cache)
    eligible_sha = digest(eligible_seeds)
    eligible = np.load(eligible_seeds, mmap_mode="r", allow_pickle=False)
    if eligible.ndim != 1 or not np.issubdtype(eligible.dtype, np.integer):
        raise ValueError("eligible-seeds must be a one-dimensional integer .npy")
    combined_set = set(map(int, np.unique(combined_seeds)))
    eligible_set = set(map(int, np.unique(eligible)))
    allowed = combined_set & eligible_set
    if not allowed or any(seed < 0 or seed >= 1_000_000 for seed in allowed):
        raise ValueError("eligible combined TRAIN seed intersection is empty or non-TRAIN")
    return allowed, {
        "combined_train": {"path": str(combined_train), "sha256": combined_sha,
                           "levels": len(combined_set)},
        "combined_seed_cache": {"manifest": str(manifest_path),
                                 "seeds": str(seed_path),
                                 "sha256": digest(seed_path),
                                 "rows": int(len(combined_seeds)),
                                 "manifest_source_sha256": manifest.get("source_sha256")},
        "eligible_seeds": {"path": str(eligible_seeds), "sha256": eligible_sha,
                           "levels": len(eligible_set)},
        "intersection_levels": len(allowed),
    }


def select_sources(old_bank: Path, extended_bank: Path, combined_train: Path,
                   eligible_seeds: Path = DEFAULT_ELIGIBLE_SEEDS,
                   array_cache: Path = DEFAULT_ARRAY_CACHE, per_source: int = 512,
                   selection_seed: int = 20260912):
    """Select deterministic equal-difficulty source subsets from TRAIN only."""
    if type(per_source) is not int or not 1 <= per_source <= 512:
        raise ValueError("per_source must be in 1..512 for the large collector")
    old_values, extended_values = _json_lines(old_bank), _json_lines(extended_bank)
    old_seeds = _validate_bank(old_bank, old_values)
    extended_seeds = _validate_bank(extended_bank, extended_values)
    if old_seeds & extended_seeds:
        raise ValueError("original and extended source banks overlap")
    allowed, membership = _load_allowed_seeds(combined_train, eligible_seeds, array_cache)
    old = _select(old_bank, per_source, selection_seed, allowed)
    extended = _select(extended_bank, per_source, selection_seed + 1, allowed)
    specs = old + extended
    selected = [int(spec["seed"]) for spec in specs]
    if len(selected) != 2 * per_source or len(set(selected)) != len(selected):
        raise ValueError(f"source selection did not produce distinct {per_source}/source TRAIN levels")
    if not set(selected) <= allowed:
        raise ValueError("selected source seed is absent from combined/eligible TRAIN membership")
    source_records = []
    for label, path, selected_specs, all_seeds in (
            ("original", old_bank, old, old_seeds),
            ("extended", extended_bank, extended, extended_seeds)):
        source_records.append({"name": label, "path": str(path), "sha256": digest(path),
                               "levels": len(all_seeds),
                               "selected_seeds": [int(spec["seed"]) for spec in selected_specs],
                               "selection_seed": selection_seed if label == "original" else selection_seed + 1,
                               "selected_difficulties": Counter(int(spec["difficulty"]) for spec in selected_specs)})
    return specs, source_records, membership


def _bound_code_paths():
    return [Path(__file__), Path("tools/collect_onpolicy_world.py"),
            Path("tools/validate_extended_collector.py"), Path("tools/goal_attribute_probes.py"),
            Path("pebby/agent/world_data.py"), Path("pebby/agent/model.py"),
            Path("pebby/agent/world_train.py"), Path("pebby/agent/history.py"),
            Path("pebby/agent/structured_workspace_controller.py"),
            Path("pebby/agent/structured_workspace_policy.py"),
            Path("pebby/agent/structured_factored_policy.py"),
            Path("pebby/agent/structured_policy.py"), Path("pebby/agent/structured_field.py"),
            Path("pebby/ls20/env.py"), Path("pebby/ls20/layout.py"), Path("pebby/ls20/plan.py"),
            Path("pebby/ls20/fastplan.py"), Path("pebby/ls20/_fastplan.c"), Path("pebby/ls20/rails.py"),
            Path("pebby/ls20/generate.py"), Path("pebby/ls20/curriculum.py"),
            Path("pebby/ls20/extended_curriculum.py"), Path("pebby/ls20/names.py"),
            Path("third_party/ls20/ls20.py")]


def _source_hashes(paths):
    return {str(Path(path)): digest(path) for path in paths if Path(path).exists()}


def _metadata_file_hashes(value):
    """Extract file bindings nested in a checkpoint's source metadata."""
    found = {}
    if isinstance(value, dict):
        for key in ("path", "checkpoint", "bank", "proof"):
            path = value.get(key)
            sha = value.get("sha256") or value.get(key + "_sha256")
            if isinstance(path, str) and isinstance(sha, str) and len(sha) == 64:
                found[path] = sha
        for key in ("code_hashes", "source_hashes", "checkpoint_hashes", "artifacts"):
            nested = value.get(key)
            if isinstance(nested, dict):
                for path, sha in nested.items():
                    if isinstance(path, str) and isinstance(sha, str) and len(sha) == 64:
                        found[path] = sha
        for child in value.values():
            found.update(_metadata_file_hashes(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_metadata_file_hashes(child))
    return found


def _bind_metadata_sources(source_hashes, policy):
    for path, expected in _metadata_file_hashes(getattr(policy, "sources", {})).items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f"workspace policy source changed or is missing: {path}")
        source_hashes[path] = expected


def _load_workspace_policy(checkpoint: Path, expected_sha: str):
    actual = digest(checkpoint)
    if actual != expected_sha:
        raise ValueError(f"workspace checkpoint SHA mismatch: expected {expected_sha}, got {actual}")
    policy, info = load_checkpoint(checkpoint, "cpu")
    config = policy.config()
    if (info.get("format") != "pebby.structured-workspace-readout.v1"
            or config.get("architecture") != "structured"
            or config.get("mode") != "successors"
            or config.get("history") != HISTORY):
        raise ValueError("checkpoint is not a generated H8 workspace successor policy")
    policy.eval().requires_grad_(False)
    if digest(checkpoint) != expected_sha:
        raise ValueError("workspace checkpoint changed during load")
    return policy, {"format": info.get("format"), "config": config,
                    "parameters": int(info.get("parameters", policy.parameter_count())),
                    "actor_checkpoint": info.get("actor_checkpoint"),
                    "actor_sha256": info.get("actor_sha256"),
                    "checkpoint_sha256": expected_sha,
                    "sources": getattr(policy, "sources", {})}


def _worker_init(checkpoint, expected_sha):
    global _WORKER_POLICY
    torch.set_num_threads(1)
    _WORKER_POLICY, _ = _load_workspace_policy(Path(checkpoint), expected_sha)


def _worker_collect(spec):
    rows, proof, on_policy_count = _collect_level(spec, _WORKER_POLICY, max_actions=MAX_ACTIONS)
    proof["worker_pid"] = os.getpid()
    proof["worker_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return rows, proof, on_policy_count


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_collection(*, checkpoint=CHECKPOINT, old_bank=DEFAULT_OLD_BANK,
                   extended_bank=DEFAULT_EXTENDED_BANK, combined_train=DEFAULT_COMBINED_TRAIN,
                   eligible_seeds=DEFAULT_ELIGIBLE_SEEDS, array_cache=DEFAULT_ARRAY_CACHE,
                   out=DEFAULT_OUT, report=DEFAULT_REPORT, per_source=512, workers=1,
                   selection_seed=20260912, deadline_seconds=1800):
    """Run the large bank collector; kept separate so tests never launch it."""
    checkpoint, old_bank, extended_bank = map(Path, (checkpoint, old_bank, extended_bank))
    combined_train, eligible_seeds, array_cache = map(Path, (combined_train, eligible_seeds, array_cache))
    out, report = Path(out), Path(report)
    if out.exists() or report.exists():
        raise ValueError("refusing to overwrite existing workspace pilot outputs")
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    if type(deadline_seconds) is not int or not 30 <= deadline_seconds <= 1800:
        raise ValueError("deadline_seconds must be 30..1800")
    started = time.monotonic()
    torch.set_num_threads(1)
    print("PID", os.getpid(), flush=True)
    code_paths = _bound_code_paths()
    source_hashes = _source_hashes([*code_paths, checkpoint, old_bank, extended_bank,
                                    combined_train, eligible_seeds])
    specs, source_records, membership = select_sources(
        old_bank, extended_bank, combined_train, eligible_seeds, array_cache,
        per_source, selection_seed)
    for cache_path in (membership["combined_seed_cache"]["manifest"],
                       membership["combined_seed_cache"]["seeds"]):
        source_hashes[str(cache_path)] = digest(cache_path)
    policy, checkpoint_info = _load_workspace_policy(checkpoint, EXPECTED_CHECKPOINT_SHA256)
    _bind_metadata_sources(source_hashes, policy)
    policy_hash = source_hashes[str(checkpoint)]
    actor_path = checkpoint_info.get("actor_checkpoint")
    actor_hash = checkpoint_info.get("actor_sha256")
    if actor_path and actor_hash:
        actor_path = Path(actor_path)
        if digest(actor_path) != actor_hash:
            raise ValueError("workspace actor checkpoint changed or has the wrong SHA")
        source_hashes[str(actor_path)] = actor_hash
    report_data = {
        "status": "running", "pid": os.getpid(), "workers": workers, "torch_threads": 1,
        "device": "cpu", "history": HISTORY, "max_policy_actions_per_level": MAX_ACTIONS,
        "source": "generated_only", "official_inputs_used": False,
        "policy_checkpoint": str(checkpoint), "policy_sha256": policy_hash,
        "source_hashes": source_hashes,
        "policy": checkpoint_info, "sources": source_records,
        "membership": membership, "levels": [], "on_policy_rows": 0,
        "limits": "Policy-visited rows, route-proportional expert anchors, and actual exhaustion failures; "
                   "all four real branches checked for every retained row; not exhaustive.",
        "selection_contract": f"{per_source} deterministic equal-difficulty levels from each disjoint "
                              "TRAIN bank, intersected with combined TRAIN seeds and eligible cached seeds.",
    }

    def persist():
        report_data["elapsed_seconds"] = time.monotonic() - started
        report_data["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        _atomic_json(report, report_data)

    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.TemporaryDirectory(prefix="workspace-onpolicy-", dir=out.parent)
    pool = None
    temporary_output = out.with_suffix(".tmp.npz")
    worker_pids = []
    try:
        persist()
        store = RowStore(scratch.name, len(specs) * ROW_CAPACITY_PER_LEVEL)
        on_policy_indices, auxiliary_indices = [], []
        source_by_seed = {int(spec["seed"]): source_records[0]["name"]
                          for spec in specs[:per_source]}
        source_by_seed.update({int(spec["seed"]): source_records[1]["name"]
                               for spec in specs[per_source:]})
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(workers, initializer=_worker_init,
                            initargs=(str(checkpoint), EXPECTED_CHECKPOINT_SHA256))
        worker_pids = [process.pid for process in pool._pool]
        report_data["worker_pids"] = worker_pids
        persist()
        for offset in range(0, len(specs), workers):
            remaining = deadline_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("workspace on-policy collection deadline exceeded")
            results = pool.map_async(_worker_collect, specs[offset:offset + workers]).get(timeout=remaining)
            for rows, proof, on_count in results:
                if not rows or not 1 <= on_count <= MAX_ACTIONS or len(rows) != on_count + proof.get("expert_samples", 0) + proof.get("failure_samples", 0):
                    raise ValueError(f"invalid retained rows for seed {proof.get('seed')}")
                if proof.get("context_engine_verified") is not True or proof.get("search_truncated") is not False:
                    raise ValueError(f"incomplete context proof for seed {proof.get('seed')}")
                start = store.count
                on_policy_indices.extend(range(start, start + on_count))
                auxiliary_indices.extend(range(start + on_count, start + len(rows)))
                store.append(rows)
                proof["source_bank"] = source_by_seed[int(proof["seed"])]
                proof["row_start"] = start
                proof["row_count"] = len(rows)
                proof["on_policy_row_count"] = on_count
                report_data["levels"].append(proof)
                report_data["rows"] = store.count
                report_data["on_policy_rows"] = len(on_policy_indices)
                persist()
                completed = len(report_data["levels"])
                if completed % 8 == 0 or completed == len(specs):
                    print(f"{completed}/{len(specs)} levels rows={store.count}", flush=True)
        pool.close(); pool.join(); pool = None
        report_data["worker_pids_exited"] = all(not Path(f"/proc/{pid}").exists() for pid in worker_pids)
        if not report_data["worker_pids_exited"]:
            raise RuntimeError("worker cleanup incomplete")
        arrays = store.arrays()
        arrays["meta"] = {
            "format": wd.FORMAT, "source": "generated_only", "oracle_search": "complete_only",
            "history": HISTORY, "alternatives_per_state": 4, "samples": int(store.count),
            "seeds": sorted(int(spec["seed"]) for spec in specs), "accepted_levels": len(specs),
            "win_covered_levels": len(specs), "coverage": "workspace_on_policy_with_expert_anchors",
            "collection_policy": "model_greedy", "max_policy_actions": MAX_ACTIONS,
            "on_policy_rows": on_policy_indices, "auxiliary_rows": auxiliary_indices, "levels": report_data["levels"],
            "behavior_checkpoint": {"path": str(checkpoint), "sha256": policy_hash,
                                    "parameters": checkpoint_info["parameters"],
                                    "config": checkpoint_info["config"]},
            "source_banks": source_records, "combined_train_membership": membership,
            "source_hashes": source_hashes,
        }
        require_verified_data(arrays)
        require_winning_coverage(arrays, "workspace-onpolicy")
        if len(np.unique(arrays["seeds"])) != len(specs):
            raise ValueError("retained rows do not cover exactly the selected levels")
        wd.save(temporary_output, arrays)
        loaded = load_dataset(temporary_output)
        require_verified_data(loaded)
        require_winning_coverage(loaded, "workspace-onpolicy-reload")
        if set(map(int, loaded["seeds"])) != {int(spec["seed"]) for spec in specs}:
            raise ValueError("reloaded output seed coverage mismatch")
        if any(digest(path) != expected for path, expected in source_hashes.items()):
            raise ValueError("bound source changed during workspace collection")
        temporary_output.replace(out)
        report_data.update(status="complete", output=str(out), output_sha256=digest(out),
                           rows=int(store.count), on_policy_rows=len(on_policy_indices),
                           expert_rows=sum(level["expert_samples"] for level in report_data["levels"]),
                           failure_rows=sum(level.get("failure_samples", 0) for level in report_data["levels"]),
                           source_unchanged=True, checkpoint_unchanged=digest(checkpoint) == policy_hash,
                           worker_pids_exited=True)
        persist()
        return report_data
    except BaseException as error:
        report_data.update(status="failed", error=repr(error))
        persist()
        raise
    finally:
        if pool is not None:
            pool.terminate(); pool.join()
        remaining = [pid for pid in worker_pids if Path(f"/proc/{pid}").exists()]
        report_data["worker_pids_exited"] = not remaining
        if temporary_output.exists():
            temporary_output.unlink()
        scratch.cleanup()
        if report_data.get("status") == "running":
            report_data["status"] = "failed"
        report_data["cleanup_verified"] = not remaining
        persist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--old-bank", type=Path, default=DEFAULT_OLD_BANK)
    parser.add_argument("--extended-bank", type=Path, default=DEFAULT_EXTENDED_BANK)
    parser.add_argument("--combined-train", type=Path, default=DEFAULT_COMBINED_TRAIN)
    parser.add_argument("--eligible-seeds", type=Path, default=DEFAULT_ELIGIBLE_SEEDS)
    parser.add_argument("--array-cache", type=Path, default=DEFAULT_ARRAY_CACHE)
    parser.add_argument("--per-source", type=int, default=512)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--selection-seed", type=int, default=20260912)
    parser.add_argument("--deadline-seconds", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        run_collection(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
