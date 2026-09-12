"""Wait for and attest the old plus extended world-data merge inputs.

This is deliberately a preparation step.  It reads only small NPZ members
(``meta``, seeds, context indices, and terminal/won masks); transition image
arrays are never decompressed.  A complete output is published only after the
generator and collector reports, their immutable bank snapshots, and every
shard's small proof arrays agree.
"""

from __future__ import annotations

from pebby.ls20.provenance import metadata_difficulty_version

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = "train"
VALIDATION = "validation"
SPLITS = (TRAIN, VALIDATION)
TARGETS = {TRAIN: 10_000, VALIDATION: 2_000}
EXTENDED_PARTS = {TRAIN: 20, VALIDATION: 4}
OLD_NPZ = {
    TRAIN: Path("data/ls20-world-mixedpath-train.npz"),
    VALIDATION: Path("data/ls20-world-mixedpath-validation.npz"),
}
OLD_LABELS = {
    TRAIN: Path("data/ls20-world-successor-train-labels.npz"),
    VALIDATION: Path("data/ls20-world-successor-validation-labels.npz"),
}
OLD_BANK = {
    TRAIN: Path("data/ls20-verified-train.jsonl"),
    VALIDATION: Path("data/ls20-verified-validation.jsonl"),
}
OLD_AUDIT = Path("artifacts/world-full-context-validity.json")
GENERATION_REPORT = Path("artifacts/world-extended-bank-v1.json")
COLLECTION_REPORT = Path("artifacts/world-extended-collection.json")
EXTENDED_BANK = {
    TRAIN: Path("data/extended-bank-v1/train.jsonl"),
    VALIDATION: Path("data/extended-bank-v1/validation.jsonl"),
}
OUTPUT_VALIDITY = Path("artifacts/world-combined-context-validity.json")
OUTPUT_INPUTS = Path("artifacts/world-combined-merge-inputs.json")


class PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{path}: cannot read JSON ({error})") from error
    if not isinstance(value, dict):
        raise PreparationError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise PreparationError(f"{path}:{line_number}: empty JSONL row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PreparationError(f"{path}:{line_number}: expected an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{path}: cannot read JSONL ({error})") from error
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def report_status(path: Path) -> str:
    report = read_json(path)
    status = report.get("status")
    if status in {"failed", "error", "aborted", "timeout"}:
        raise PreparationError(f"{path}: producer failed with status {status!r}: {report.get('error')}")
    if status not in {"running", "complete"}:
        raise PreparationError(f"{path}: unexpected producer status {status!r}")
    return status


def wait_for_reports(generation: Path, collection: Path, deadline: float, poll: float) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    while True:
        generation_status = report_status(generation)
        collection_status = report_status(collection)
        print(f"watch generation={generation_status} collection={collection_status}", flush=True)
        if generation_status == collection_status == "complete":
            return read_json(generation), read_json(collection)
        if time.monotonic() - started >= deadline:
            raise PreparationError(f"producer reports did not both complete within {deadline:g}s")
        time.sleep(min(poll, max(0.1, deadline - (time.monotonic() - started))))


def gameplay_hash(spec: dict[str, Any]) -> str:
    # Keep this local so attestations do not import the model or world trainer.
    import hashlib as _hashlib

    gameplay = {
        "walls": sorted(spec["walls"]),
        "start": spec["start"],
        "start_triple": spec["start_triple"],
        "goals": sorted(spec["goals"], key=lambda x: x["cell"]),
        "cyclers": sorted(spec["cyclers"], key=lambda x: (x["cell"], x["kind"])),
        "rails": sorted([sorted(r["cells"]) for r in spec.get("rails", [])]),
        "launchers": sorted(spec.get("launchers", []), key=lambda x: x["cell"]),
        "refills": sorted(spec["refills"]),
        "step_counter": spec["step_counter"],
        "step_cost": spec["step_cost"],
        "fog": spec["fog"],
    }
    return _hashlib.sha256(json.dumps(gameplay, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if not path.is_file():
            raise PreparationError(f"missing input: {path}")
        result[str(path)] = sha256(path)
    return result


def npz_small(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read only proof-sized members, never frames/next_frames or other images."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"meta", "seeds", "context_index", "terminal", "won"}
            missing = required.difference(archive.files)
            if missing:
                raise PreparationError(f"{path}: missing small proof arrays {sorted(missing)}")
            meta_value = archive["meta"]
            if meta_value.shape != ():
                raise PreparationError(f"{path}: meta is not scalar JSON")
            meta = json.loads(str(meta_value.item()))
            if not isinstance(meta, dict):
                raise PreparationError(f"{path}: meta is not an object")
            arrays = {name: np.asarray(archive[name]) for name in required - {"meta"}}
            if "next_optimal" in archive.files:
                arrays["next_optimal"] = np.asarray(archive["next_optimal"])
            return meta, arrays
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        if isinstance(error, PreparationError):
            raise
        raise PreparationError(f"{path}: cannot read proof members ({error})") from error


def validate_small_shard(path: Path, report_entry: dict[str, Any], bank_rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_hash = report_entry.get("npz_sha256")
    actual_hash = sha256(path)
    if expected_hash != actual_hash:
        raise PreparationError(f"{path}: hash differs from collector report")
    snapshot = ROOT / report_entry["snapshot"]
    if not snapshot.is_file() or sha256(snapshot) != report_entry.get("snapshot_sha256"):
        raise PreparationError(f"{path}: immutable bank snapshot hash mismatch")
    snapshot_rows = read_jsonl(snapshot)
    if len(snapshot_rows) != int(report_entry.get("levels", -1)):
        raise PreparationError(f"{path}: snapshot level count mismatch")
    meta, arrays = npz_small(path)
    if meta.get("format") != "pebby.ls20-world-transitions.v1" or meta.get("source") != "generated_only" or meta.get("oracle_search") != "complete_only":
        raise PreparationError(f"{path}: unsupported provenance in meta")
    levels = meta.get("levels")
    if not isinstance(levels, list) or len(levels) != len(snapshot_rows):
        raise PreparationError(f"{path}: meta level proof count mismatch")
    seeds = np.asarray(arrays["seeds"])
    contexts = np.asarray(arrays["context_index"])
    terminal = np.asarray(arrays["terminal"])
    won = np.asarray(arrays["won"])
    if seeds.ndim != 1:
        raise PreparationError(f"{path}: seeds must be a one-dimensional row array")
    if contexts.shape != seeds.shape or terminal.shape != (len(seeds), 4) or won.shape != (len(seeds), 4):
        raise PreparationError(f"{path}: malformed small proof array shapes")
    if terminal.dtype != np.bool_ or won.dtype != np.bool_ or not np.all((~won) | terminal):
        raise PreparationError(f"{path}: won must be boolean terminal coverage")
    if not np.all(contexts == seeds % 7):
        raise PreparationError(f"{path}: per-row context index mismatch")
    meta_seeds = [int(row.get("seed", -1)) for row in levels]
    snapshot_seeds = [int(row.get("seed", -1)) for row in snapshot_rows]
    row_seed_order = list(dict.fromkeys(map(int, seeds)))
    if meta_seeds != snapshot_seeds or row_seed_order != snapshot_seeds:
        raise PreparationError(f"{path}: NPZ, proof, and snapshot seed order differs")
    if snapshot_rows != bank_rows[bank_index_for_snapshot(bank_rows, snapshot_seeds[0]):bank_index_for_snapshot(bank_rows, snapshot_seeds[0]) + len(snapshot_rows)]:
        # The bank is checked again below; this guards against a stale or
        # cross-split snapshot even when seeds happen to look plausible.
        raise PreparationError(f"{path}: snapshot rows are not the final bank rows")
    if meta.get("win_covered_levels") != len(levels):
        raise PreparationError(f"{path}: meta does not claim winning coverage for every level")
    if int(meta.get("samples", len(seeds))) != len(seeds):
        raise PreparationError(f"{path}: meta sample count mismatch")
    proof_rows: list[dict[str, Any]] = []
    for level, bank in zip(levels, snapshot_rows):
        seed = int(level.get("seed", -1))
        for source, record in (("meta", level), ("bank", bank)):
            if record.get("search_truncated", False) is not False:
                raise PreparationError(f"{path} seed {seed}: truncated proof")
            if source == "meta" and record.get("context_engine_verified") is not True:
                raise PreparationError(f"{path} seed {seed}: missing engine context proof")
            if source == "meta" and record.get("context_index") != seed % 7:
                raise PreparationError(f"{path} seed {seed}: wrong proof context")
        if bank.get("context_engine_verified") is not True or bank.get("training_context_index") != seed % 7 or bank.get("verification_lives") != 3:
            raise PreparationError(f"{path} seed {seed}: bank context proof incomplete")
        if not np.any(won[np.flatnonzero(seeds == seed)]):
            raise PreparationError(f"{path} seed {seed}: no winning successor row")
        proof = dict(bank)
        proof.update({"npz_proof": dict(level), "npz_path": str(path), "split": split,
                      "source": "extended_shard", "accepted": True,
                      "context_validity": True, "engine_win": True,
                      "replay_lives": 3, "levels_completed": 1,
                      "context_index": seed % 7, "status": "verified"})
        proof_rows.append(proof)
    return proof_rows, {"path": str(path), "sha256": actual_hash, "levels": len(levels), "rows": len(seeds), "snapshot": str(snapshot)}


def bank_index_for_snapshot(bank_rows: list[dict[str, Any]], first_seed: int) -> int:
    for index, row in enumerate(bank_rows):
        if int(row.get("seed", -1)) == first_seed:
            return index
    raise PreparationError(f"snapshot first seed {first_seed} absent from final bank")


def require_frozen_legacy_bank(rows):
    if metadata_difficulty_version({'levels': rows}):
        raise PreparationError('this frozen old-plus-extended aggregate requires legacy five-tier banks; '
                               'prepare calibrated shards separately with tools/merge_world_data.py '
                               '--inputs SHARD... --validity AUDIT --out OUTPUT --split train|validation')


def validate_bank(rows: list[dict[str, Any]], split: str) -> tuple[set[int], set[str]]:
    require_frozen_legacy_bank(rows)
    if len(rows) != TARGETS[split]:
        raise PreparationError(f"{split} extended bank has {len(rows)} rows; need {TARGETS[split]}")
    seeds: set[int] = set()
    hashes: set[str] = set()
    lower = 20_000 if split == TRAIN else 1_020_000
    # Rejected draft attempts consume seed numbers.  The namespace is bounded
    # by the split's million-wide allocation, rather than by target count.
    upper = 1_000_000 if split == TRAIN else 2_000_000
    for index, row in enumerate(rows):
        seed = row.get("seed")
        if type(seed) is not int or seed in seeds or not lower <= seed < upper:
            raise PreparationError(f"{split} bank row {index}: invalid/duplicate namespace seed")
        if row.get("difficulty") != index % 5 + 1:
            raise PreparationError(f"{split} bank seed {seed}: difficulty sequence mismatch")
        if row.get("training_context_index") != seed % 7 or row.get("context_engine_verified") is not True or row.get("verification_lives") != 3 or row.get("search_truncated") is not False:
            raise PreparationError(f"{split} bank seed {seed}: contextual generation proof incomplete")
        computed = gameplay_hash(row)
        if row.get("gameplay_sha256") != computed or computed in hashes:
            raise PreparationError(f"{split} bank seed {seed}: gameplay hash mismatch/duplicate")
        seeds.add(seed)
        hashes.add(computed)
    return seeds, hashes


def old_level_sets() -> tuple[dict[str, set[int]], dict[str, set[str]], dict[str, dict[int, dict[str, Any]]]]:
    audit = read_json(OLD_AUDIT)
    if audit.get("status") != "complete":
        raise PreparationError("frozen old context audit is not complete")
    audit_by_split = {split: {} for split in SPLITS}
    for record in audit.get("levels", []):
        if record.get("split") in audit_by_split:
            audit_by_split[record["split"]][int(record["seed"])] = record
    old_seeds: dict[str, set[int]] = {}
    old_hashes: dict[str, set[str]] = {}
    for split in SPLITS:
        bank_rows = read_jsonl(OLD_BANK[split])
        require_frozen_legacy_bank(bank_rows)
        bank_seeds = {int(row["seed"]) for row in bank_rows}
        bank_hashes = {gameplay_hash(row) for row in bank_rows}
        with np.load(OLD_NPZ[split], allow_pickle=False) as archive:
            seeds = set(map(int, np.unique(archive["seeds"])))
        if not seeds.issubset(audit_by_split[split]):
            raise PreparationError(f"{split} old NPZ has seeds without frozen context proof")
        if seeds != bank_seeds:
            raise PreparationError(f"{split} old aggregate and verified bank seed sets differ")
        old_seeds[split] = seeds
        old_hashes[split] = set()
        for seed in seeds:
            record = audit_by_split[split][seed]
            if record.get("status") != "verified" or record.get("engine_win") is not True or record.get("replay_lives") != 3 or record.get("levels_completed") != 1 or record.get("search_truncated") is not False or record.get("context_index") != seed % 7:
                raise PreparationError(f"old {split} seed {seed}: frozen proof incomplete")
            old_hashes[split].add(record["gameplay_sha256"])
        if old_hashes[split] != bank_hashes:
            raise PreparationError(f"{split} old aggregate gameplay proofs differ from verified bank")
        if len(old_seeds[split]) != (10296 if split == TRAIN else 1998):
            raise PreparationError(f"old {split} aggregate level count changed")
    return old_seeds, old_hashes, audit_by_split


def build_reports(generation: dict[str, Any], collection: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if generation.get("status") != "complete" or collection.get("status") != "complete":
        raise PreparationError("producer reports changed away from complete")
    if generation.get("requested_total") != TARGETS or collection.get("target_levels") != TARGETS:
        raise PreparationError("producer target levels are not exactly 10000/2000")
    banks = {split: read_jsonl(EXTENDED_BANK[split]) for split in SPLITS}
    bank_sets = {split: validate_bank(banks[split], split) for split in SPLITS}
    old_seeds, old_hashes, audit_by_split = old_level_sets()
    new_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    shards_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_new: set[int] = set()
    seen_new_hashes: set[str] = set()
    for split in SPLITS:
        entries = [entry for entry in collection.get("shards", []) if entry.get("split") == split]
        if len(entries) != EXTENDED_PARTS[split] or sorted(int(x.get("part", -1)) for x in entries) != list(range(EXTENDED_PARTS[split])):
            raise PreparationError(f"{split}: collector shard list is incomplete")
        entries.sort(key=lambda x: int(x["part"]))
        expected_offset = 0
        for entry in entries:
            path = ROOT / entry["npz"]
            snapshot_rows = read_jsonl(ROOT / entry["snapshot"])
            if snapshot_rows != banks[split][expected_offset:expected_offset + len(snapshot_rows)]:
                raise PreparationError(f"{path}: snapshot does not match final generated bank at offset {expected_offset}")
            proof_rows, info = validate_small_shard(path, entry, banks[split], split)
            seeds = {int(row["seed"]) for row in proof_rows}
            hashes = {str(row["gameplay_sha256"]) for row in proof_rows}
            if seen_new.intersection(seeds) or seen_new_hashes.intersection(hashes):
                raise PreparationError(f"{path}: duplicate seed/gameplay hash across extended shards")
            seen_new.update(seeds); seen_new_hashes.update(hashes)
            new_rows[split].extend(proof_rows)
            shards_by_split[split].append(info)
            expected_offset += len(snapshot_rows)
        if expected_offset != TARGETS[split] or {int(row["seed"]) for row in new_rows[split]} != bank_sets[split][0]:
            raise PreparationError(f"{split}: shards do not cover exactly the final bank")
    if old_seeds[TRAIN] & old_seeds[VALIDATION] or (old_seeds[TRAIN] | old_seeds[VALIDATION]) & seen_new:
        raise PreparationError("old and extended seed namespaces overlap")
    if (old_hashes[TRAIN] | old_hashes[VALIDATION]) & seen_new_hashes:
        raise PreparationError("old and extended gameplay hashes overlap")
    # Also corroborate the generator's own existing-bank hashes when present.
    for split in SPLITS:
        expected = generation.get("existing_banks", {}).get(str(OLD_BANK[split]))
        if expected is not None and expected != sha256(OLD_BANK[split]):
            raise PreparationError(f"generator existing-bank hash changed for {split}")

    combined_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for seed in sorted(old_seeds[split]):
            original = dict(audit_by_split[split][seed])
            original.update({"source": "old_aggregate", "accepted": True, "context_validity": True,
                             "npz_path": str(OLD_NPZ[split]), "split": split})
            original["proof"] = {"context_engine_verified": True, "search_truncated": False,
                                  "context_index": int(original["context_index"]), "split": split}
            combined_rows.append(original)
        combined_rows.extend(sorted(new_rows[split], key=lambda row: int(row["seed"])))
    source_paths = [OLD_AUDIT, GENERATION_REPORT, COLLECTION_REPORT]
    source_paths += [OLD_NPZ[s] for s in SPLITS] + [OLD_LABELS[s] for s in SPLITS]
    source_paths += [OLD_BANK[s] for s in SPLITS] + [EXTENDED_BANK[s] for s in SPLITS]
    source_paths += [Path(info["path"]) for split in SPLITS for info in shards_by_split[split]]
    source_paths += [Path(info["snapshot"]) for split in SPLITS for info in shards_by_split[split]]
    hashes = require_hashes(source_paths)
    validity = {
        "format": "pebby.ls20-combined-context-validity.v1",
        "status": "complete",
        "rows": combined_rows,
        "summary": {split: {"old_levels": len(old_seeds[split]), "extended_levels": TARGETS[split],
                             "levels": len(old_seeds[split]) + TARGETS[split],
                             "extended_shards": shards_by_split[split]} for split in SPLITS},
        "source_sha256": hashes,
        "original_validity": {"path": str(OLD_AUDIT), "sha256": hashes[str(OLD_AUDIT)]},
        "generation_report": {"path": str(GENERATION_REPORT), "sha256": hashes[str(GENERATION_REPORT)]},
        "collection_report": {"path": str(COLLECTION_REPORT), "sha256": hashes[str(COLLECTION_REPORT)]},
        "provenance": {"source": "generated_only", "oracle_search": "complete_only",
                        "official_gameplay_inputs_used": False, "all_old_proofs_frozen": True},
    }
    inputs = {
        "format": "pebby.ls20-combined-merge-inputs.v1",
        "status": "complete",
        "inputs": {split: [str(OLD_NPZ[split])] + [info["path"] for info in shards_by_split[split]] for split in SPLITS},
        "successor_sidecars": {str(OLD_NPZ[split]): str(OLD_LABELS[split]) for split in SPLITS},
        "min_levels": {split: len(old_seeds[split]) + TARGETS[split] for split in SPLITS},
        "source_sha256": hashes,
        "validity": str(OUTPUT_VALIDITY),
        "generation_report": str(GENERATION_REPORT),
        "collection_report": str(COLLECTION_REPORT),
    }
    return validity, inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--deadline-seconds", type=float, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if args.deadline_seconds <= 0 or not 1 <= args.poll_seconds <= 30:
        parser.error("deadline must be positive and poll-seconds must be 1..30")
    print(f"PID {os.getpid()} waiting for generation={GENERATION_REPORT} collection={COLLECTION_REPORT}", flush=True)
    try:
        generation, collection = wait_for_reports(GENERATION_REPORT, COLLECTION_REPORT, args.deadline_seconds, args.poll_seconds)
        validity, inputs = build_reports(generation, collection)
        # Publish validity first; the input manifest points at it and is only
        # published after the validity file has reached its final hash.
        atomic_json(OUTPUT_VALIDITY, validity)
        inputs["validity_sha256"] = sha256(OUTPUT_VALIDITY)
        atomic_json(OUTPUT_INPUTS, inputs)
        print(f"complete validity={OUTPUT_VALIDITY} inputs={OUTPUT_INPUTS} train={validity['summary']['train']['levels']} validation={validity['summary']['validation']['levels']}", flush=True)
        return 0
    except PreparationError as error:
        print(f"FAILED CLOSED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
