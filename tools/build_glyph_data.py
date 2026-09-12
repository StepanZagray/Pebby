"""Extract and audit generated current/successor HUD glyph supervision.

Reads one large frame array at a time from individual transition shards.
No engine, oracle, model, or official level data is loaded.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np


FORMAT = "pebby.ls20-glyphs.v1"
CROP = {"rows": [55, 61], "columns": [3, 9], "bounds": "half-open"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def extract_shard(path):
    """Copy the small crop before reading the next full frame array."""
    with np.load(path, allow_pickle=False) as source:
        meta = json.loads(str(source["meta"].item()))
        if meta.get("source") != "generated_only" or meta.get("oracle_search") != "complete_only":
            raise ValueError(f"{path}: expected generated-only complete-oracle data")
        seeds = source["seeds"].astype(np.int32, copy=False)
        current = source["current_triple"]
        successor = source["next_triple"]
        count = len(seeds)
        if current.shape != (count, 3) or successor.shape != (count, 4, 3):
            raise ValueError(f"{path}: triple axes do not match current/all-four-successor contract")
        triples = np.concatenate((current[:, None], successor), axis=1).reshape(-1, 3)
        if np.any(triples < 0) or np.any(triples >= np.array([6, 4, 4])):
            raise ValueError(f"{path}: invalid triple class")
        # The indexing expression owns no reference to the decompressed full
        # array once copy() returns. Only the two small crop arrays coexist.
        current_crop = source["frames"][:, -1, 55:61, 3:9].copy()
        successor_crop = source["next_frames"][:, :, 55:61, 3:9].copy()
        if current_crop.shape != (count, 6, 6) or successor_crop.shape != (count, 4, 6, 6):
            raise ValueError(f"{path}: unexpected glyph shape")
        glyphs = np.concatenate((current_crop[:, None], successor_crop), axis=1).reshape(-1, 6, 6)
        if np.any(glyphs < 0) or np.any(glyphs > 15):
            raise ValueError(f"{path}: glyph pixels outside public palette")
    return {"glyphs": glyphs.astype(np.uint8, copy=False),
            "triples": triples.astype(np.uint8),
            "seeds": np.repeat(seeds, 5),
            "views": np.tile(np.arange(5, dtype=np.uint8), count)}, count


def build_split(paths, split, expected_levels, progress):
    pieces, provenance, seen = [], [], set()
    offset = 0
    for path in paths:
        before = digest(path)
        arrays, states = extract_shard(path)
        if digest(path) != before:
            raise ValueError(f"{path}: source changed during extraction")
        distinct = set(map(int, np.unique(arrays["seeds"])))
        if seen & distinct:
            raise ValueError(f"{path}: level seeds repeated across source shards")
        seen.update(distinct)
        provenance.append({"path": str(path), "sha256": before, "states": states,
                           "glyph_rows": len(arrays["seeds"]), "distinct_levels": len(distinct),
                           "output_row_start": offset, "output_row_stop": offset + len(arrays["seeds"])})
        offset += len(arrays["seeds"])
        pieces.append(arrays)
        progress(split, len(provenance), len(paths), offset, len(seen))
    if len(seen) != expected_levels:
        raise ValueError(f"{split}: expected {expected_levels} levels, found {len(seen)}")
    combined = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
    combined["meta"] = {"format": FORMAT, "source": "generated_only", "split": split,
                        "crop": CROP, "views": {"0": "current", "1..4": "actual successors in action order"},
                        "row_order": "source-shard order; each source state contributes views 0,1,2,3,4",
                        "shards": provenance, "distinct_levels": len(seen), "glyph_rows": offset}
    return combined, seen


def pattern_census(splits):
    glyphs = np.concatenate([arrays["glyphs"] for arrays in splits.values()])
    triples = np.concatenate([arrays["triples"] for arrays in splits.values()])
    packed = np.ascontiguousarray(glyphs).reshape(-1, 36).view("V36").reshape(-1)
    patterns, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    label_codes = (triples[:, 0].astype(np.int64) * 4 + triples[:, 1]) * 4 + triples[:, 2]
    pairs, frequencies = np.unique(inverse * 96 + label_codes, return_counts=True)
    inventory = [{"pattern_id": index, "glyph_hex": bytes(pattern).hex(),
                  "glyph": np.frombuffer(bytes(pattern), dtype=np.uint8).reshape(6, 6).tolist(),
                  "frequency": int(count), "labels": []}
                 for index, (pattern, count) in enumerate(zip(patterns, counts))]
    for pair, frequency in zip(pairs, frequencies):
        pattern, code = divmod(int(pair), 96)
        inventory[pattern]["labels"].append({"triple": [code // 16, (code // 4) % 4, code % 4],
                                            "frequency": int(frequency)})
    contradictions = [item for item in inventory if len(item["labels"]) > 1]
    offsets, offset = {}, 0
    for split, arrays in splits.items():
        offsets[split] = offset
        offset += len(arrays["glyphs"])
    for item in contradictions:
        item["affected"] = {}
        for split, arrays in splits.items():
            start = offsets[split]
            mask = inverse[start:start + len(arrays["glyphs"])] == item["pattern_id"]
            selected = np.flatnonzero(mask)
            grouped = Counter((int(arrays["seeds"][i]), int(arrays["views"][i]),
                               tuple(map(int, arrays["triples"][i]))) for i in selected)
            item["affected"][split] = {
                "frequency": len(selected),
                "seeds": sorted({key[0] for key in grouped}),
                "views": sorted({key[1] for key in grouped}),
                "seed_view_label_counts": [{"seed": seed, "view": view, "triple": list(triple), "count": count}
                                           for (seed, view, triple), count in sorted(grouped.items())],
            }
    return {"unique_patterns": len(patterns), "distinct_triples": int(len(np.unique(label_codes))),
            "patterns": inventory, "contradictory_patterns": contradictions,
            "contradictory_pattern_count": len(contradictions),
            "rows_in_contradictory_patterns": sum(item["frequency"] for item in contradictions),
            "identifiable_from_exact_crop_on_audited_rows": not contradictions}


def save_npz(path, arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{key: value for key, value in arrays.items() if key != "meta"},
                            meta=np.array(json.dumps(arrays["meta"])))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/world-glyph-data-audit.json"))
    parser.add_argument("--train-levels", type=int, default=10296)
    parser.add_argument("--validation-levels", type=int, default=1998)
    args = parser.parse_args()
    started = time.monotonic()
    audit = {"status": "running", "format": FORMAT, "pid": os.getpid(), "source": "generated_only",
             "crop": CROP, "cpu_workers": 1, "script_sha256": digest(__file__), "progress": {}}

    def persist():
        audit["elapsed_seconds"] = time.monotonic() - started
        audit["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        save_json(args.audit, audit)

    def progress(split, completed, total, rows, levels):
        audit["progress"][split] = {"shards_done": completed, "shards_total": total,
                                      "glyph_rows": rows, "distinct_levels": levels}
        persist()
        print(split, f"{completed}/{total}", rows, "glyphs", levels, "levels", flush=True)

    persist()
    print("PID", os.getpid(), flush=True)
    try:
        splits, seed_sets = {}, {}
        for split, expected in (("train", args.train_levels), ("validation", args.validation_levels)):
            paths = sorted(args.data_dir.glob(f"ls20-world-mixedpath-{split}-part-*.npz"))
            if not paths:
                raise ValueError(f"no {split} shards")
            splits[split], seed_sets[split] = build_split(paths, split, expected, progress)
        overlap = seed_sets["train"] & seed_sets["validation"]
        if overlap:
            raise ValueError(f"train/validation seed collision: {sorted(overlap)}")
        audit["train_validation_disjoint"] = True
        audit["census"] = pattern_census(splits)
        audit["outputs"] = {}
        for split, arrays in splits.items():
            path = args.data_dir / f"ls20-glyph-{split}.npz"
            arrays["meta"]["exact_crop_contradictions"] = audit["census"]["contradictory_pattern_count"]
            save_npz(path, arrays)
            audit["outputs"][split] = {"path": str(path), "sha256": digest(path),
                "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for key, value in arrays.items() if key != "meta"},
                "distinct_levels": len(seed_sets[split]), "seeds": sorted(seed_sets[split]),
                "source_shards": arrays["meta"]["shards"]}
        audit["status"] = "complete"
    except Exception as error:
        audit["status"] = "failed"
        audit["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        persist()
    print("DONE", audit["status"], "contradictions", audit["census"]["contradictory_pattern_count"],
          "seconds", audit["elapsed_seconds"], "peak MiB", audit["peak_rss_mib"], flush=True)


if __name__ == "__main__":
    main()
