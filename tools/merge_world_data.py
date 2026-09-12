"""Merge verified generated world transition shards with bounded memory.

The input files are compressed NPZ files, so NumPy must decompress an array
when it is read.  This module nevertheless keeps the working set bounded: it
streams bounded row chunks, writes selected rows to an on-disk memmap,
and only then compresses the completed memmaps into the final NPZ.
Input members must use numeric C-order NPY arrays. Optional successor-label
sidecars are validated against the complete original source before filtering;
this lets older image archives merge with newer shards containing those labels.

Validity files use the following small JSON contract::

    {"status": "complete", "rows": [
        {"split": "train", "seed": 10, "difficulty": 1,
         "context_validity": true, "engine_win": true,
         "replay_lives": 3, "levels_completed": 1,
         "context_index": 3, "search_truncated": false}
    ]}

For compatibility with earlier producers, ``levels`` may also be a mapping
from seed to proof records, and a split may be nested under ``train`` or
``validation``.  An accepted record still needs an untruncated contextual
engine proof in the form consumed by ``world_train.require_verified_data``.
"""

from __future__ import annotations

from pebby.ls20.provenance import generated_context, difficulty_version

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from typing import Any, Iterable, Mapping

import numpy as np


FORMAT = "pebby.ls20-world-transitions.v1"
PROVENANCE = "generated_only"
ORACLE_SEARCH = "complete_only"
SPLITS = ("train", "validation")
_HASH_CHUNK = 1024 * 1024
ROW_CHUNK_BYTES = 16 * 1024 * 1024


class MergeError(ValueError):
    """Raised when a shard or validity audit violates the merge contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _json_meta(archive: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
    if "meta" not in archive.files:
        raise MergeError(f"{path}: missing JSON meta array")
    try:
        value = archive["meta"]
        if value.shape != ():
            raise TypeError(f"meta has shape {value.shape}")
        meta = json.loads(str(value.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise MergeError(f"{path}: malformed meta JSON ({error})") from error
    if not isinstance(meta, dict):
        raise MergeError(f"{path}: meta must be a JSON object")
    return meta


def _int_seed(value: Any, where: str) -> int:
    # bool is an integer subclass but cannot identify an LS20 level.
    if isinstance(value, bool):
        raise MergeError(f"{where}: seed must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MergeError(f"{where}: seed must be an integer") from error
    if isinstance(value, float) and value != result:
        raise MergeError(f"{where}: seed must be an integer")
    return result


def _split_check(value: Any, split: str, where: str) -> None:
    if value is not None and value != split:
        raise MergeError(f"{where}: proof belongs to split {value!r}, not {split!r}")


def _record_from_validity(raw: Any, seed_hint: Any = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MergeError("validity level records must be JSON objects")
    record = dict(raw)
    if "seed" not in record:
        if seed_hint is None:
            raise MergeError("validity level record lacks seed")
        record["seed"] = seed_hint
    seed = _int_seed(record["seed"], "validity record")
    record["seed"] = seed
    proof = record.get("proof")
    if proof is None:
        # world_data writes proof fields directly in meta.levels.  Keep those
        # fields as the nested proof as well as flattening them in the output.
        excluded = {"seed", "difficulty", "accepted", "split"}
        proof = {key: value for key, value in record.items() if key not in excluded}
    if not isinstance(proof, Mapping):
        raise MergeError(f"validity seed {seed}: proof must be an object")
    record["proof"] = dict(proof)
    # The world-data producer records the proof fields directly, while the
    # validity audit may summarize them as status/engine_win/lives/completed.
    # Normalize both forms to the contract consumed by world_train.
    proof = record["proof"]
    summary_fields = {"context_validity", "engine_win", "replay_lives", "levels_completed"}
    has_summary = bool(summary_fields.intersection(record))
    if has_summary:
        # These values are deliberately not defaulted: accepting a row with a
        # missing replay result would turn an incomplete audit into training
        # data.  Invalid rows are filtered; if none remain merge() fails.
        summary_ok = (record.get("context_validity") is True
                      and record.get("engine_win") is True
                      and record.get("replay_lives") == 3
                      and record.get("levels_completed") == 1
                      and record.get("context_index") == generated_context(record)
                      and record.get("search_truncated") is False)
    else:
        summary_ok = True
    if "context_engine_verified" not in proof and summary_ok and has_summary:
        proof["context_engine_verified"] = True
    if "search_truncated" not in proof and "search_truncated" in record:
        proof["search_truncated"] = record["search_truncated"]
    if "context_index" not in proof and record.get("context_index") is not None:
        proof["context_index"] = record["context_index"]
    if "accepted" in raw:
        record["accepted"] = bool(record["accepted"])
    else:
        record["accepted"] = (proof.get("context_engine_verified") is True
                               and proof.get("search_truncated") is False
                               and proof.get("context_index") == generated_context(record)
                               and "excluded" not in proof
                               and summary_ok
                               and record.get("context_validity", True) is not False)
    # Explicit audit failures never become accepted merely because another
    # field says ``accepted``.  A complete status is required when status is
    # supplied, and the summarized replay proof must be successful when those
    # fields are present.
    if record.get("context_validity") is False:
        record["accepted"] = False
    if has_summary and not summary_ok:
        record["accepted"] = False
    if record.get("status") is not None and record["status"] not in {
            "verified", "accepted", "complete", "completed", "ok"}:
        record["accepted"] = False
    if any(key in record for key in ("engine_win", "lives", "completed", "levels_completed")):
        record["accepted"] = bool(record["accepted"] and record.get("engine_win", True)
                                   and record.get("lives", 3) == 3
                                   and record.get("completed", record.get("levels_completed", 1)) == 1)
    record["proof"] = proof
    return record


def _validity_records(path: Path, split: str) -> tuple[dict[int, dict[str, Any]], str]:
    if not path.is_file():
        raise MergeError(f"missing validity audit: {path}")
    audit_hash = _sha256(path)
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MergeError(f"{path}: invalid validity JSON ({error})") from error
    if not isinstance(document, Mapping):
        raise MergeError(f"{path}: validity audit must be a JSON object")

    # A complete audit is required before any shard can be promoted.  This
    # prevents a live/partial audit from looking like an empty exclusion list.
    if document.get("status") != "complete":
        raise MergeError(f"{path}: validity audit status must be 'complete'")

    # Permit an audit containing both splits while keeping the selected split
    # explicit.  A nested audit must still state its split when it is present.
    nested = False
    if split in document and isinstance(document[split], Mapping) and "levels" in document[split]:
        document = document[split]
        nested = True
    _split_check(document.get("split"), split, f"{path}")
    if "split" not in document and not nested and "rows" not in document \
            and "levels" not in document:
        raise MergeError(f"{path}: validity audit must declare split {split!r}")

    raw_levels = document.get("levels", document.get("rows", document.get("seeds")))
    if raw_levels is None:
        raw_levels = document.get("proofs")
    if raw_levels is None:
        raise MergeError(f"{path}: validity audit lacks levels")
    if isinstance(raw_levels, Mapping):
        iterable: Iterable[dict[str, Any]] = (
            _record_from_validity(value, seed_hint=key)
            for key, value in raw_levels.items()
        )
    elif isinstance(raw_levels, list):
        iterable = (_record_from_validity(value) for value in raw_levels)
    else:
        raise MergeError(f"{path}: validity levels must be a list or object")

    result: dict[int, dict[str, Any]] = {}
    for record in iterable:
        seed = record["seed"]
        # A rows-style audit contains both splits.  Ignore the other split;
        # a selected record with an explicit proof split is checked below.
        if record.get("split") is not None and record.get("split") != split:
            continue
        if seed in result:
            raise MergeError(f"{path}: duplicate validity proof for seed {seed}")
        _split_check(record.get("split"), split, f"{path} seed {seed}")
        _split_check(record["proof"].get("split"), split, f"{path} seed {seed}")
        if record["accepted"]:
            proof = record["proof"]
            if proof.get("context_engine_verified") is not True:
                raise MergeError(f"{path} seed {seed}: accepted proof is not engine verified")
            if proof.get("search_truncated") is not False:
                raise MergeError(f"{path} seed {seed}: accepted proof is truncated or incomplete")
            if proof.get("context_index") != generated_context(record):
                raise MergeError(f"{path} seed {seed}: context index does not match seed")
        result[seed] = record
    return result, audit_hash


def _source_levels(meta: Mapping[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    levels = meta.get("levels")
    if not isinstance(levels, list):
        raise MergeError(f"{path}: meta.levels must be a list")
    result: dict[int, dict[str, Any]] = {}
    for raw in levels:
        if not isinstance(raw, Mapping) or "seed" not in raw:
            raise MergeError(f"{path}: every meta level needs a seed")
        level = dict(raw)
        seed = _int_seed(level["seed"], f"{path} meta level")
        level["seed"] = seed
        if seed in result and result[seed] != level:
            raise MergeError(f"{path}: inconsistent duplicate meta level {seed}")
        result[seed] = level
    return result


def _validate_provenance(meta: Mapping[str, Any], path: Path, split: str) -> None:
    if meta.get("format") != FORMAT:
        raise MergeError(f"{path}: unsupported format {meta.get('format')!r}")
    if meta.get("source") != PROVENANCE:
        raise MergeError(f"{path}: source must be {PROVENANCE!r}")
    if meta.get("oracle_search") != ORACLE_SEARCH:
        raise MergeError(f"{path}: oracle_search must be {ORACLE_SEARCH!r}")
    _split_check(meta.get("split"), split, f"{path} meta")


def _array_header(stream, where):
    version = np.lib.format.read_magic(stream)
    reader = {(1, 0): np.lib.format.read_array_header_1_0,
              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
    if reader is None:
        raise MergeError(f"{where}: unsupported NPY header version {version}")
    shape, fortran, dtype = reader(stream)
    if not shape or dtype.hasobject or fortran:
        raise MergeError(f"{where}: expected a numeric C-order row array")
    return shape, dtype


def _schema(archive: np.lib.npyio.NpzFile, path: Path) -> tuple[tuple[str, ...], dict[str, tuple[str, tuple[int, ...]]], np.ndarray]:
    if len(set(archive.files)) != len(archive.files):
        raise MergeError(f"{path}: duplicate archive array names")
    # Member ordering has no meaning in a named-array archive.
    names = tuple(sorted(name for name in archive.files if name != "meta"))
    if "seeds" not in names:
        raise MergeError(f"{path}: missing seeds array")
    schema: dict[str, tuple[str, tuple[int, ...]]] = {}
    seed_array: np.ndarray | None = None
    count: int | None = None
    with zipfile.ZipFile(path) as zipped:
        for name in names:
            with zipped.open(name + '.npy') as stream:
                shape, dtype = _array_header(stream, f'{path}:{name}')
                expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if zipped.getinfo(name + '.npy').file_size - stream.tell() != expected_bytes:
                    raise MergeError(f'{path}:{name}: NPY payload size does not match its header')
            if count is None:
                count = int(shape[0])
            elif shape[0] != count:
                raise MergeError(f"{path}: arrays do not have the same row count")
            schema[name] = (dtype.str, tuple(shape[1:]))
            if name == "seeds":
                if not np.issubdtype(dtype, np.integer):
                    raise MergeError(f"{path}: seeds must have an integer dtype")
                seed_array = np.asarray(archive[name])
    assert count is not None and seed_array is not None
    return names, schema, seed_array


def _copy_selected_rows(path, name, destination, offset, indices, schema, source_rows):
    """Copy sorted selected rows with at most one 16 MiB input chunk plus its selection."""
    with zipfile.ZipFile(path) as zipped, zipped.open(name + '.npy') as stream:
        shape, dtype = _array_header(stream, f'{path}:{name}')
        if shape != (source_rows, *schema[1]) or dtype.str != schema[0]:
            raise MergeError(f'{path}:{name}: schema changed during merge')
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        if row_bytes > ROW_CHUNK_BYTES:
            raise MergeError(f'{path}:{name}: one row exceeds the streaming chunk limit')
        chunk_rows = max(1, ROW_CHUNK_BYTES // max(1, row_bytes))
        for start in range(0, source_rows, chunk_rows):
            count = min(chunk_rows, source_rows - start)
            payload = stream.read(count * row_bytes)
            if len(payload) != count * row_bytes:
                raise MergeError(f'{path}:{name}: truncated NPY payload')
            lower, upper = np.searchsorted(indices, (start, start + count))
            if upper > lower:
                chunk = np.frombuffer(payload, dtype=dtype).reshape(count, *shape[1:])
                destination[offset + lower:offset + upper] = chunk[indices[lower:upper] - start]
        if stream.read(1):
            raise MergeError(f'{path}:{name}: trailing NPY payload')


def _successor_sidecar(path, source_path, source_hash, archive, seeds):
    """Attach small action labels only when bound to every exact original row."""
    if 'next_optimal' in archive.files:
        raise MergeError(f'{source_path}: cannot replace embedded successor labels')
    with np.load(path, allow_pickle=False) as labels:
        meta = _json_meta(labels, path)
        masks, label_seeds = labels['next_optimal'], labels['seeds']
    if (meta.get('format') != 'pebby.ls20-successor-labels.v1'
            or meta.get('source') != PROVENANCE or meta.get('source_sha256') != source_hash):
        raise MergeError(f'{path}: successor labels provenance/source hash mismatch')
    if not np.issubdtype(label_seeds.dtype, np.integer) or not np.array_equal(label_seeds, seeds):
        raise MergeError(f'{path}: successor labels seeds must match source rows in exact order')
    for name, expected in (('rows', len(seeds)), ('levels', len(np.unique(seeds)))):
        if type(meta.get(name)) is not int or meta[name] != expected:
            raise MergeError(f'{path}: successor labels {name} must equal {expected}')
    if (masks.shape != (len(seeds), 4) or not np.issubdtype(masks.dtype, np.integer)
            or np.any((masks < 0) | (masks > 15))):
        raise MergeError(f'{path}: successor masks must be integer [N,4] values in 0..15')
    terminal = archive['terminal']
    if terminal.shape != masks.shape or np.any(terminal.astype(bool) & (masks != 0)):
        raise MergeError(f'{path}: terminal successors must have mask zero')
    return masks, {'path': str(path), 'sha256': _sha256(path),
                   'source_sha256': source_hash, 'rows': len(seeds),
                   'levels': len(np.unique(seeds)), 'format': meta['format']}


def _level_for_seed(levels: Mapping[int, dict[str, Any]], seed: int, path: Path) -> dict[str, Any]:
    level = levels.get(seed)
    if level is None:
        raise MergeError(f"{path}: row seed {seed} is absent from meta.levels")
    if level.get("excluded"):
        raise MergeError(f"{path}: row seed {seed} is marked excluded")
    if level.get("search_truncated", False):
        raise MergeError(f"{path}: row seed {seed} has truncated proof")
    # Older generated shards contain the complete-oracle and replay metadata
    # but the retroactive validity audit supplies the contextual engine proof.
    # If a shard does carry that field, a false value is still a hard failure.
    if "context_engine_verified" in level and level["context_engine_verified"] is not True:
        raise MergeError(f"{path}: row seed {seed} lacks engine verification")
    if "context_index" in level and level["context_index"] != generated_context(level):
        raise MergeError(f"{path}: row seed {seed} has the wrong context index")
    return level


def merge(inputs: Iterable[str | Path], out: str | Path, validity: str | Path,
          split: str, min_levels: int = 1, *,
          successor_sidecars: Mapping[str | Path, str | Path] | None = None) -> dict[str, Any]:
    """Merge verified rows and atomically write *out*; return output metadata."""
    if split not in SPLITS:
        raise MergeError(f"split must be one of {SPLITS}, got {split!r}")
    if min_levels < 1:
        raise MergeError("min_levels must be positive")
    input_paths = [Path(value) for value in inputs]
    if not input_paths:
        raise MergeError("at least one input shard is required")
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_inputs = [path.resolve() for path in input_paths]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise MergeError("input shards must be distinct")
    if output_path.resolve() in set(resolved_inputs):
        raise MergeError("output must not replace an input shard")
    sidecars = {Path(key).resolve(): Path(value) for key, value in (successor_sidecars or {}).items()}
    if len(sidecars) != len(successor_sidecars or {}):
        raise MergeError('successor sidecar map repeats a resolved input path')
    if set(sidecars) - set(resolved_inputs):
        raise MergeError('successor sidecar map contains an unknown input shard')
    if output_path.resolve() in {p.resolve() for p in sidecars.values()}:
        raise MergeError('output must not replace a successor sidecar')

    validity_path = Path(validity)
    if output_path.resolve() == validity_path.resolve():
        raise MergeError('output must not replace the validity audit')
    accepted, audit_hash = _validity_records(validity_path, split)
    source_hashes: dict[str, str] = {}
    label_sources: dict[str, dict[str, Any]] = {}
    shard_info: list[dict[str, Any]] = []
    expected_names: tuple[str, ...] | None = None
    expected_schema: dict[str, tuple[str, tuple[int, ...]]] | None = None
    seen_shard_seeds: set[int] = set()
    output_level_sources: dict[int, dict[str, Any]] = {}
    total_rows = 0
    accepted_seeds = {seed for seed, record in accepted.items() if record.get("accepted")}

    for path in input_paths:
        if not path.is_file():
            raise MergeError(f"missing input shard: {path}")
        source_hashes[str(path)] = _sha256(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = _json_meta(archive, path)
                _validate_provenance(meta, path, split)
                levels = _source_levels(meta, path)
                names, schema, seed_array = _schema(archive, path)
                successor_masks = None
                if path.resolve() in sidecars:
                    successor_masks, label_provenance = _successor_sidecar(
                        sidecars[path.resolve()], path, source_hashes[str(path)], archive, seed_array)
                    label_sources[str(path)] = label_provenance
                    names = tuple(sorted((*names, 'next_optimal')))
                    schema['next_optimal'] = (successor_masks.dtype.str, (4,))
        except (OSError, ValueError, KeyError) as error:
            if isinstance(error, MergeError):
                raise
            raise MergeError(f"{path}: unable to read shard ({error})") from error
        if expected_names is None:
            expected_names, expected_schema = names, schema
        elif names != expected_names:
            raise MergeError(f"{path}: array names differ from the first shard")
        assert expected_schema is not None
        if schema != expected_schema:
            raise MergeError(f"{path}: array dtypes or trailing shapes differ from the first shard")

        row_seeds = np.asarray(seed_array)
        unique_seeds = {_int_seed(value, f"{path} seeds") for value in np.unique(row_seeds)}
        overlap = seen_shard_seeds.intersection(unique_seeds)
        if overlap:
            example = sorted(overlap)[:5]
            raise MergeError(f"seed(s) {example} occur in more than one input shard")
        seen_shard_seeds.update(unique_seeds)
        for seed in unique_seeds:
            _level_for_seed(levels, seed, path)
            source_level = levels[seed]
            record = accepted.get(seed)
            if record is not None and record.get("accepted"):
                if difficulty_version(record) != difficulty_version(source_level):
                    raise MergeError(f"{path}: seed {seed} difficulty_version disagrees with validity audit")
                difficulty = record.get("difficulty")
                if difficulty is not None and source_level.get("difficulty") is not None \
                        and int(difficulty) != int(source_level["difficulty"]):
                    raise MergeError(f"{path}: seed {seed} difficulty disagrees with validity audit")
                if seed in output_level_sources and output_level_sources[seed] != source_level:
                    raise MergeError(f"{path}: seed {seed} has inconsistent source level metadata")
                output_level_sources[seed] = source_level
        accepted_indices = np.flatnonzero(np.isin(row_seeds, list(accepted_seeds)))
        if "context_index" in names and len(accepted_indices):
            # The schema archive above is closed before this check.  Reopen
            # the compressed member so the seeds and context arrays are never
            # resident alongside a transition array.
            with np.load(path, allow_pickle=False) as context_archive:
                context_values = np.asarray(context_archive["context_index"])
            if context_values.ndim != 1 or len(context_values) != len(row_seeds):
                raise MergeError(f"{path}: context_index must be a row-shaped array")
            for index in accepted_indices:
                seed = int(row_seeds[index])
                expected_context = accepted[seed]["proof"].get("context_index")
                if expected_context is not None and int(context_values[index]) != int(expected_context):
                    raise MergeError(f"{path}: seed {seed} context_index disagrees with validity audit")
            del context_values
        seed_counts = {int(seed): int(count) for seed, count in zip(
            *np.unique(row_seeds[accepted_indices], return_counts=True))}
        shard_info.append({"path": path, "indices": accepted_indices,
                           "rows": int(len(accepted_indices)), "seed_counts": seed_counts,
                           "source_rows": len(row_seeds), 'successor_masks': successor_masks})
        total_rows += int(len(accepted_indices))

    if expected_names is None or expected_schema is None:
        raise MergeError("no shard schema")
    if total_rows == 0:
        raise MergeError("validity audit accepted no rows")

    # Derive the level set from the row seeds without retaining whole shard
    # arrays.  The accepted row indices above are enough to do this in a
    # second, seed-only pass; this remains bounded by one seeds member.
    selected_seed_set: set[int] = set()
    for info in shard_info:
        path = info["path"]
        with np.load(path, allow_pickle=False) as archive:
            seeds = np.asarray(archive["seeds"])
            selected_seed_set.update(int(value) for value in seeds[info["indices"]])
        del seeds
    selected_seeds = sorted(selected_seed_set)
    if len(selected_seeds) < min_levels:
        raise MergeError(f"only {len(selected_seeds)} accepted levels remain; need {min_levels}")

    temp_dir = Path(tempfile.mkdtemp(prefix=".merge-world-data-", dir=str(output_path.parent)))
    memmap_paths: dict[str, Path] = {}
    memmap_arrays: dict[str, np.memmap] = {}
    try:
        offset = 0
        for name in expected_names:
            dtype_string, tail_shape = expected_schema[name]
            path = temp_dir / f"{len(memmap_paths):04d}-{name}.mmap"
            destination = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.dtype(dtype_string),
                shape=(total_rows, *tail_shape), fortran_order=False)
            offset = 0
            for info in shard_info:
                count = info["rows"]
                if count:
                    if name == 'next_optimal' and info['successor_masks'] is not None:
                        destination[offset:offset + count] = info['successor_masks'][info['indices']]
                    else:
                        _copy_selected_rows(info['path'], name, destination, offset,
                                            info['indices'], expected_schema[name], info['source_rows'])
                    offset += count
            destination.flush()
            del destination
            memmap_paths[name] = path

        # np.savez_compressed consumes the memmaps sequentially and does not
        # materialize the merged arrays.  A file handle prevents NumPy from
        # appending an unwanted suffix to our atomic temporary output.
        for name, path in memmap_paths.items():
            memmap_arrays[name] = np.lib.format.open_memmap(path, mode="r")
        output_meta: dict[str, Any] = {
            "format": FORMAT,
            "source": PROVENANCE,
            "oracle_search": ORACLE_SEARCH,
            "split": split,
            "merged": True,
            "samples": total_rows,
            "seeds": selected_seeds,
            "levels": [],
            "source_sha256": source_hashes,
            "audit_sha256": audit_hash,
            "validity": str(validity_path),
        }
        if label_sources:
            output_meta['successor_label_sources'] = label_sources
        for seed in selected_seeds:
            source_level = copy.deepcopy(output_level_sources[seed])
            record = accepted[seed]
            proof = copy.deepcopy(record["proof"])
            difficulty = record.get("difficulty", source_level.get("difficulty"))
            if difficulty is not None:
                source_level["difficulty"] = int(difficulty)
            source_level["accepted"] = True
            source_level.update(proof)
            source_level["proof"] = proof
            source_level["samples"] = sum(info["seed_counts"].get(seed, 0) for info in shard_info)
            output_meta["levels"].append(source_level)
        temp_output = temp_dir / "output.npz"
        with temp_output.open("wb") as handle:
            np.savez_compressed(handle, **memmap_arrays,
                                meta=np.array(json.dumps(output_meta, sort_keys=True)))
        # Never publish a mixture assembled while a source was being changed.
        for path in input_paths:
            if _sha256(path) != source_hashes[str(path)]:
                raise MergeError(f'{path}: source changed during merge')
        for provenance in label_sources.values():
            if _sha256(Path(provenance['path'])) != provenance['sha256']:
                raise MergeError(f"{provenance['path']}: successor labels changed during merge")
        if _sha256(validity_path) != audit_hash:
            raise MergeError('validity audit changed during merge')
        os.replace(temp_output, output_path)
        return output_meta
    finally:
        memmap_arrays.clear()
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--min-levels", type=int, default=1)
    parser.add_argument('--successor-sidecars', type=Path,
                        help='JSON object mapping exact source NPZ paths to source-hashed label sidecars')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sidecars = None
        if args.successor_sidecars:
            sidecars = json.loads(args.successor_sidecars.read_text())
            if not isinstance(sidecars, dict):
                raise MergeError('successor sidecars file must be a JSON object')
        meta = merge(args.inputs, args.out, args.validity, args.split, args.min_levels,
                     successor_sidecars=sidecars)
    except MergeError as error:
        build_parser().error(str(error))
    print(f"Merged {meta['samples']} rows from {len(args.inputs)} shard(s) into {args.out}")
    return 0


if __name__ == "__main__":
    main()
