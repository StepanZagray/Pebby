"""Measure exact public-input aliases in world-transition archives.

The key is the canonical public input consumed by ``WorldPolicy``:
``frames`` history with temporally masked slots zeroed, ``history_valid`` and
``previous_actions``.  Dataset labels, successors, seeds and engine state are
deliberately excluded from the key.
This reports an observed duplicate-input lower bound, not a full partially
observable Bayes floor.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time
import zipfile

import numpy as np


KEY_FIELDS = ("frames", "history_valid", "previous_actions")
ARRAY_SPECS = {
    "frames": ((8, 64, 64), np.uint8),
    "history_valid": ((8,), np.bool_),
    "previous_actions": ((8,), np.int64),
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise ValueError("truncated NPY member")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


class NpyRows:
    """Read one compressed NPY member in bounded row chunks."""

    def __init__(self, stream, name: str, expected_tail, expected_dtype):
        self.stream = stream
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0,
                   (3, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"{name}: unsupported NPY version {version}")
        shape, fortran, dtype = readers[version](stream)
        if fortran or dtype.hasobject or tuple(shape[1:]) != tuple(expected_tail):
            raise ValueError(f"{name}: unexpected NPY schema {shape}, {dtype}")
        if dtype != np.dtype(expected_dtype):
            raise ValueError(f"{name}: expected dtype {expected_dtype}, got {dtype}")
        self.count = int(shape[0])
        self.tail = tuple(shape[1:])
        self.dtype = dtype
        self.row_bytes = int(np.prod(self.tail, dtype=np.int64) or 1) * dtype.itemsize
        self.read_rows_count = 0

    def rows(self, count: int):
        if self.read_rows_count + count > self.count:
            raise ValueError("NPY row count exceeds declared shape")
        block = read_exact(self.stream, count * self.row_bytes)
        values = np.frombuffer(block, dtype=self.dtype)
        values = values.reshape((count,) + self.tail)
        self.read_rows_count += count
        return values

    def finish(self):
        if self.read_rows_count != self.count:
            raise ValueError("NPY member ended before declared row count")
        if self.stream.read(1):
            raise ValueError("NPY member has trailing bytes")


def read_small_arrays(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        missing = {"optimal", "seeds"} - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        optimal = np.asarray(archive["optimal"])
        seeds = np.asarray(archive["seeds"])
        if optimal.ndim != 1 or seeds.ndim != 1 or len(optimal) != len(seeds):
            raise ValueError(f"{path}: optimal and seeds must be aligned vectors")
        if not np.issubdtype(optimal.dtype, np.integer) or np.any((optimal < 1) | (optimal > 15)):
            raise ValueError(f"{path}: optimal must be nonempty integer 4-bit masks")
        return optimal.astype(np.uint8, copy=False), seeds.astype(np.int64, copy=False)


def empty_group(seed: int, mask: int):
    return {"count": 0, "first_seed": int(seed), "seed_values": None,
            "masks": Counter()}


def key_digest(frames, valid, previous):
    """Hash the model-visible public input, ignoring masked frame pixels."""
    canonical_frames = np.array(frames, copy=True, order="C")
    canonical_frames[~np.asarray(valid, dtype=bool)] = 0
    result = hashlib.sha256()
    result.update(canonical_frames.tobytes(order="C"))
    result.update(np.asarray(valid, dtype=bool).tobytes(order="C"))
    result.update(np.asarray(previous, dtype=np.int64).tobytes(order="C"))
    return result.digest()


def add_row(groups, key, seed: int, mask: int):
    group = groups.get(key)
    if group is None:
        group = groups[key] = empty_group(seed, mask)
    elif group["seed_values"] is None and seed != group["first_seed"]:
        group["seed_values"] = {group["first_seed"], int(seed)}
    elif group["seed_values"] is not None:
        group["seed_values"].add(int(seed))
    group["count"] += 1
    group["masks"][int(mask)] += 1


def group_stats(groups):
    total_rows = sum(group["count"] for group in groups.values())
    duplicate_groups = [group for group in groups.values() if group["count"] > 1]
    same_level = [group for group in duplicate_groups if group["seed_values"] is None]
    cross_level = [group for group in duplicate_groups if group["seed_values"] is not None]
    conflicts = [group for group in groups.values() if len(group["masks"]) > 1]

    disjoint = []
    for group in conflicts:
        masks = list(group["masks"])
        if any(left & right == 0 for index, left in enumerate(masks)
               for right in masks[index + 1:]):
            disjoint.append(group)

    def rows(items):
        return sum(group["count"] for group in items)

    weighted_entropy = 0.0
    unweighted_entropy = 0.0
    baseline_log_k = 0.0
    baseline_group_log_k = 0.0
    for group in groups.values():
        count = group["count"]
        probabilities = np.zeros(4, dtype=np.float64)
        group_log_k = 0.0
        for mask, occurrences in group["masks"].items():
            actions = [action for action in range(4) if mask & (1 << action)]
            probability = 1.0 / len(actions)
            probabilities[actions] += occurrences * probability / count
            baseline_log_k += occurrences * np.log(len(actions))
            group_log_k += occurrences * np.log(len(actions)) / count
        entropy = float(-sum(value * np.log(value) for value in probabilities if value > 0))
        weighted_entropy += count * entropy
        unweighted_entropy += entropy
        baseline_group_log_k += group_log_k

    group_count = len(groups)
    mean_floor = weighted_entropy / total_rows if total_rows else None
    mean_log_k = baseline_log_k / total_rows if total_rows else None
    return {
        "rows": total_rows,
        "unique_public_inputs": group_count,
        "duplicate_public_input_groups": len(duplicate_groups),
        "duplicate_rows_beyond_first": rows(duplicate_groups) - len(duplicate_groups),
        "same_level_duplicate_groups": len(same_level),
        "same_level_duplicate_rows": rows(same_level),
        "cross_level_duplicate_groups": len(cross_level),
        "cross_level_duplicate_rows": rows(cross_level),
        "conflicting_label_groups": len(conflicts),
        "conflicting_label_rows": rows(conflicts),
        "disjoint_optimal_set_groups": len(disjoint),
        "disjoint_optimal_set_rows": rows(disjoint),
        "observed_softtarget_crossentropy_floor": mean_floor,
        "observed_softtarget_crossentropy_floor_unweighted_groups": (
            unweighted_entropy / group_count if group_count else None),
        "baseline_mean_log_k": mean_log_k,
        "baseline_mean_log_k_unweighted_groups": (
            baseline_group_log_k / group_count if group_count else None),
        "floor_minus_baseline": (mean_floor - mean_log_k
                                  if mean_floor is not None and mean_log_k is not None else None),
        "natural_log_units": True,
    }


def analyze(path: Path, chunk_rows: int):
    optimal, seeds = read_small_arrays(path)
    groups = {}
    with zipfile.ZipFile(path) as archive:
        streams = {}
        readers = {}
        try:
            for name, (tail, dtype) in ARRAY_SPECS.items():
                member = name + ".npy"
                if member not in archive.namelist():
                    raise ValueError(f"{path}: missing {name}")
                streams[name] = archive.open(member)
                readers[name] = NpyRows(streams[name], name, tail, dtype)
            counts = {reader.count for reader in readers.values()}
            if counts != {len(optimal)}:
                raise ValueError(f"{path}: key arrays have incompatible row counts {counts}")
            for start in range(0, len(optimal), chunk_rows):
                count = min(chunk_rows, len(optimal) - start)
                blocks = {name: readers[name].rows(count) for name in KEY_FIELDS}
                for index in range(count):
                    add_row(groups, key_digest(blocks["frames"][index],
                                               blocks["history_valid"][index],
                                               blocks["previous_actions"][index]),
                            int(seeds[start + index]), int(optimal[start + index]))
            for reader in readers.values():
                reader.finish()
        finally:
            for stream in streams.values():
                stream.close()
    result = group_stats(groups)
    result["file"] = str(path)
    result["sha256"] = digest(path)
    result["chunk_rows"] = chunk_rows
    result["key_hash"] = ("sha256(canonical_frames_row_bytes + history_valid_row_bytes + "
                         "previous_actions_row_bytes); canonical_frames[~history_valid] = 0")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=64)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    started = time.monotonic()
    result = {
        "status": "running",
        "pid": __import__("os").getpid(),
        "semantics": {
            "key_fields": list(KEY_FIELDS),
            "history": "all H=8 public frames are hashed after zeroing slots masked by history_valid",
            "actions": "previous_actions int64 are included byte-for-byte; valid slots are 0..3 and padding is -1",
            "model_contract": "current slot must be valid; history is left-padded with a contiguous valid suffix",
            "excluded_from_key": ["next_frames", "terminal", "won", "optimal", "distances", "seeds", "context_index", "engine state"],
            "interpretation": "observed exact duplicate-input lower bound only; not a full partially-observable Bayes floor",
        },
        "splits": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".tmp").write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".tmp").replace(args.out)
    try:
        result["splits"]["train"] = analyze(args.train, args.chunk_rows)
        result["splits"]["validation"] = analyze(args.validation, args.chunk_rows)
        result["status"] = "complete"
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.out.with_suffix(".tmp").write_text(json.dumps(result, indent=2) + "\n")
        args.out.with_suffix(".tmp").replace(args.out)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "peak_rss_mib": result["peak_rss_mib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
