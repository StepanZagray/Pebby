"""Manifest-audited training for the causal whole-game visual policy.

This module never opens official levels.  Checkpoint promotion uses a fixed
panel of generated validation games played closed-loop without a teacher.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import importlib
import inspect
import json
from numbers import Integral
import os
from pathlib import Path
import random
import shutil
import signal
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from pebby import multigame as M
from pebby.multigame_variants import (
    D4_NAMES,
    VariantOptions,
    WholeGameVariant,
    sample_whole_game_variant,
    transform_frame,
)
from .multigame_model import (
    ACTION_COUNT,
    CLICK_ACTION,
    COORDINATE_NONE,
    FRAME_SIZE,
    GameSequence,
    LossWeights,
    MultiGameModel,
    MultiGameModelConfig,
    click_region_nll,
    collate_game_sequences,
    compute_multigame_loss,
    load_supervised_game,
)


TRAINING_RECIPE = "whole-game-aux-v2"
TRAINING_FORMAT = "pebby-multigame-training-v1"
AUDIT_FORMAT = "pebby-multigame-dataset-audit-v1"
HISTORY_MODES = ("full", "none")
UPDATE_MODES = ("chunk", "game")
_HASH_LABELS = ("public_npz", "teacher_npz", "generated_specs")
_SPEC_DROP_MARKERS = (
    "seed", "solution", "proof", "plan", "search", "attempt", "engine_verified",
    "metadata", "generator_version", "sha256", "hash", "calibration", "statistic",
    "coverage", "limitation", "note",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _safe_path(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes manifest root: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {candidate}")
    return candidate


def _declared_hash(manifest: Mapping[str, Any], record: Mapping[str, Any], label: str) -> str | None:
    relative = record[label]
    candidates = (
        record.get(f"{label}_sha256"),
        (record.get("file_hashes") or {}).get(label),
        (record.get("file_hashes") or {}).get(relative),
        (manifest.get("file_hashes") or {}).get(relative),
    )
    values = {str(value) for value in candidates if value is not None}
    if len(values) > 1:
        raise ValueError(f"conflicting declared hashes for {relative}")
    return next(iter(values), None)


def _walk_named(value: Any, predicate) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if predicate(str(key).lower()):
                found.append(child)
            found.extend(_walk_named(child, predicate))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_named(child, predicate))
    return found


def _canonical_spec(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_spec(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if not any(marker in str(key).lower() for marker in _SPEC_DROP_MARKERS)
        }
    if isinstance(value, list):
        return [_canonical_spec(child) for child in value]
    return value


def _record_variant(record: Mapping[str, Any]) -> WholeGameVariant:
    metadata = record.get("variant")
    if metadata is None:
        return WholeGameVariant.identity()
    if not isinstance(metadata, Mapping):
        raise ValueError("variant metadata must be an object")
    if not isinstance(metadata.get("selected"), bool):
        raise ValueError("variant selected must be bool")
    if not isinstance(metadata.get("seed"), Integral) or isinstance(metadata.get("seed"), bool):
        raise ValueError("variant seed must be an integer")
    if not isinstance(metadata.get("spatial"), str):
        raise ValueError("variant spatial transform must be a string")
    return WholeGameVariant(
        metadata["selected"],
        int(metadata["seed"]),
        metadata["control_raw_to_public"],
        metadata["spatial"],
        metadata["palette_raw_to_public"],
    )


def _strict_int(value: Any, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _full_standard_manifest(
    manifest: Mapping[str, Any], requested: Sequence[str], name: str,
) -> tuple[dict[str, list[dict[str, int]]], dict[str, str]]:
    metadata = manifest.get("full_standard")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{name} manifest lacks full-standard readiness metadata")
    if (
        metadata.get("format") != M.FULL_STANDARD_FORMAT
        or metadata.get("required") is not True
        or metadata.get("ready") is not True
    ):
        raise ValueError(f"{name} manifest is not an accepted full-standard collection")
    curricula = manifest.get("curricula_by_source")
    contracts = metadata.get("contracts")
    hashes = metadata.get("contract_hashes")
    expected = set(requested)
    if not all(isinstance(value, Mapping) for value in (curricula, contracts, hashes)):
        raise ValueError(f"{name} full-standard curricula/contracts/hashes must be objects")
    if set(curricula) != expected or set(contracts) != expected or set(hashes) != expected:
        raise ValueError(f"{name} full-standard metadata does not cover requested sources exactly")
    normalized: dict[str, list[dict[str, int]]] = {}
    normalized_hashes: dict[str, str] = {}
    for source_id in requested:
        contract = contracts[source_id]
        curriculum = curricula[source_id]
        if (
            not isinstance(contract, Mapping)
            or contract.get("format") != M.FULL_STANDARD_FORMAT
            or contract.get("status") != "ready"
            or contract.get("source_id") != source_id
        ):
            raise ValueError(f"{name} source {source_id} has an invalid full-standard contract")
        if not isinstance(curriculum, list) or not curriculum or contract.get("curriculum") != curriculum:
            raise ValueError(f"{name} source {source_id} curriculum differs from its contract")
        official_level_count = _strict_int(
            contract.get("official_level_count"),
            f"{name}:{source_id}:official_level_count",
        )
        if official_level_count < 1 or len(curriculum) != official_level_count:
            raise ValueError(
                f"{name} source {source_id} curriculum does not match official level count"
            )
        parsed = []
        for index, entry in enumerate(curriculum):
            if not isinstance(entry, Mapping):
                raise ValueError(f"{name} source {source_id} curriculum[{index}] is not an object")
            value = {
                key: _strict_int(entry.get(key), f"{name}:{source_id}:curriculum[{index}]:{key}")
                for key in ("difficulty", "context_index", "search_work")
            }
            if (
                value["difficulty"] != index + 1
                or value["context_index"] != index
                or not 1 <= value["search_work"] <= M.MAX_FULL_SEARCH_WORK
            ):
                raise ValueError(f"{name} source {source_id} curriculum values are out of range")
            parsed.append(value)
        declared = hashes[source_id]
        actual = _digest_json(contract)
        if not isinstance(declared, str) or declared != actual:
            raise ValueError(f"{name} source {source_id} full-standard contract hash mismatch")
        normalized[source_id] = parsed
        normalized_hashes[source_id] = actual
    return normalized, normalized_hashes


@dataclass(frozen=True)
class PuzzleIdentity:
    """Known split identity for generated specs and any observed level starts."""

    effective_seeds: tuple[int, ...]
    fingerprints: tuple[tuple[str, str], ...]


def puzzle_identity(
    specs: Sequence[Mapping[str, Any]],
    *,
    frames: np.ndarray | None = None,
    level_boundaries: np.ndarray | None = None,
    variant: WholeGameVariant | None = None,
    label: str = "generated candidate",
) -> PuzzleIdentity:
    """Compute the same conservative puzzle identity used by dataset auditing.

    Specs always contribute, including generated levels a partial rollout did
    not reach. When a public trajectory is available, its level-start frames
    are mapped back through the private whole-game variant before hashing.
    """
    if not specs or not all(isinstance(spec, Mapping) for spec in specs):
        raise ValueError(f"{label} specs must be a non-empty sequence of objects")
    spec_values = list(specs)
    effective_values = _walk_named(spec_values, lambda key_name: key_name == "effective_seed")
    effective = tuple(
        _strict_int(value, f"{label}:effective_seed") for value in effective_values
    )
    fingerprints: list[tuple[str, str]] = []
    for spec_index, spec in enumerate(spec_values):
        # A D4 identity is the strongest geometry key currently shared by the
        # full generators. Once present, do not also bind a raw-orientation
        # geometry hash as if it were an independent puzzle identity.
        d4_values = _walk_named(spec, lambda key_name: key_name == "geometry_d4_sha256")
        geometry_values = d4_values or _walk_named(
            spec,
            lambda key_name: (
                key_name != "geometry_d4_sha256"
                and "geometry" in key_name
                and ("hash" in key_name or "sha256" in key_name)
            ),
        )
        for value in geometry_values:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} spec {spec_index} has a malformed geometry identity")
            fingerprints.append(("geometry_d4" if d4_values else "geometry", value))
        for kind, names in (
            ("gameplay", {"gameplay_hash", "gameplay_sha256"}),
            ("puzzle", {"puzzle_hash", "puzzle_sha256"}),
        ):
            for value in _walk_named(spec, lambda key_name, names=names: key_name in names):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{label} spec {spec_index} has a malformed {kind} identity"
                    )
                fingerprints.append((kind, value))
    for spec in spec_values:
        canonical = _canonical_spec(spec)
        if canonical not in ({}, []):
            fingerprints.append(("canonical_spec", _digest_json(canonical)))

    if (frames is None) != (level_boundaries is None):
        raise ValueError(f"{label} frames and level_boundaries must be supplied together")
    if frames is not None and level_boundaries is not None:
        frame_values = np.asarray(frames)
        boundary_values = np.asarray(level_boundaries)
        if (
            frame_values.ndim != 3
            or frame_values.shape[1:] != M.FRAME_SHAPE
            or boundary_values.shape != (len(frame_values) - 1,)
        ):
            raise ValueError(f"{label} public trajectory has invalid frame/boundary shapes")
        transform = variant or WholeGameVariant.identity()
        starts = [0, *(np.flatnonzero(boundary_values).astype(int) + 1).tolist()]
        starts = [value for value in starts if value < len(frame_values) - 1]
        for value in starts:
            raw = transform.raw_frame(frame_values[value])
            fingerprints.append(("raw_initial_frame", hashlib.sha256(raw.tobytes()).hexdigest()))
    return PuzzleIdentity(effective, tuple(fingerprints))


@dataclass(frozen=True)
class AuditedGame:
    key: str
    source_id: str
    source: str
    master_seed: int
    game_index: int
    status: str
    partial: bool
    record: dict[str, Any]
    public_path: Path
    teacher_path: Path
    specs_path: Path
    file_hashes: dict[str, str]
    specs: tuple[dict[str, Any], ...]
    sequence: GameSequence
    effective_seeds: tuple[int, ...]
    puzzle_fingerprints: tuple[tuple[str, str], ...]
    # True once ``sequence`` has been pulled back through the inverse of the
    # stored public variant, i.e. it lives in raw engine space.  The closed-loop
    # evaluator then plays the engine through the identity variant so frames
    # and actions match what the policy was trained on.
    canonical: bool = False

    @property
    def stored_variant(self) -> WholeGameVariant:
        return _record_variant(self.record)

    @property
    def play_variant(self) -> WholeGameVariant:
        """Raw->public bijection the closed-loop evaluator must play through."""
        return WholeGameVariant.identity() if self.canonical else self.stored_variant


@dataclass(frozen=True)
class DatasetSide:
    name: str
    manifest_paths: tuple[Path, ...]
    manifest_hashes: dict[str, str]
    source_ids: tuple[str, ...]
    games: tuple[AuditedGame, ...]
    exclusions: tuple[dict[str, Any], ...]
    failed_records: tuple[str, ...]
    dataset_hash: str
    rows: int
    random_rows: int
    teacher_rows: int

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manifests": [str(path) for path in self.manifest_paths],
            "manifest_hashes": dict(self.manifest_hashes),
            "source_ids": list(self.source_ids),
            "games_used": len(self.games),
            "rows_used": self.rows,
            "random_rows": self.random_rows,
            "teacher_rows": self.teacher_rows,
            "partial_games_used": sum(game.partial for game in self.games),
            "canonical_games": sum(game.canonical for game in self.games),
            "stored_variants_inverted": sum(
                game.canonical and not game.stored_variant.is_identity for game in self.games
            ),
            "completed_games_by_family": {
                source_id: sum(
                    game.source_id == source_id and game.status == "won" for game in self.games
                )
                for source_id in self.source_ids
            },
            "excluded": list(self.exclusions),
            "failed_records": list(self.failed_records),
            "dataset_hash": self.dataset_hash,
        }


@dataclass(frozen=True)
class DatasetBundle:
    train: DatasetSide
    validation: DatasetSide
    smoke: bool
    scope: str
    fingerprint_limitations: str
    canonical_inputs: bool = False

    @property
    def hashes(self) -> dict[str, str]:
        return {"train": self.train.dataset_hash, "validation": self.validation.dataset_hash}

    def summary(self) -> dict[str, Any]:
        return {
            "format": AUDIT_FORMAT,
            "scope": self.scope,
            "smoke": self.smoke,
            "canonical_inputs": self.canonical_inputs,
            "train": self.train.summary(),
            "validation": self.validation.summary(),
            "fingerprint_limitations": self.fingerprint_limitations,
        }


def _audit_side(path: str | Path, *, name: str, smoke: bool) -> DatasetSide:
    manifest_path = Path(path).resolve()
    manifest = M.load_manifest(manifest_path)
    requested = tuple(manifest["requested_source_ids"])
    if len(requested) != len(set(requested)):
        raise ValueError(f"{name} manifest repeats requested source IDs")
    if M.HELD_OUT_SOURCE_ID in requested:
        raise ValueError(f"{name} manifest includes held-out m0r0")
    if not smoke:
        if set(requested) != set(M.TRAIN_SOURCE_IDS) or not manifest.get("is_full_experiment_collection"):
            raise ValueError(f"{name} manifest is not a full 24-family experiment collection")
        full_curricula, full_contract_hashes = _full_standard_manifest(
            manifest, requested, name,
        )
    else:
        full_curricula, full_contract_hashes = {}, {}
    root = manifest_path.parent
    exclusions: list[dict[str, Any]] = []
    games: list[AuditedGame] = []
    failed_records: list[str] = []
    dataset_entries: list[dict[str, Any]] = []
    for index, original in enumerate(manifest.get("records", ())):
        record = dict(original)
        source = M.source_for(record.get("source_id", record.get("source", "")))
        if source.held_out:
            raise ValueError(f"{name} record {index} includes held-out m0r0")
        record_key = str(record.get("record", f"record-{index}"))
        key = f"{manifest_path}:{record_key}"
        if record.get("status") != "won":
            failed_records.append(key)
        steps = _strict_int(record.get("steps", 0), f"{key}:steps")
        missing_artifacts = [label for label in _HASH_LABELS if not record.get(label)]
        if steps <= 0 or missing_artifacts:
            exclusions.append({
                "record": key,
                "status": record.get("status"),
                "steps": steps,
                "reason": "zero-step or missing trajectory/spec artifacts",
                "missing": missing_artifacts,
            })
            continue
        paths = {label: _safe_path(root, record[label], f"{key}:{label}") for label in _HASH_LABELS}
        hashes = {label: sha256_file(found) for label, found in paths.items()}
        for label, actual in hashes.items():
            declared = _declared_hash(manifest, record, label)
            if declared is not None and declared != actual:
                raise ValueError(f"{key}:{label} hash mismatch: declared {declared}, actual {actual}")
        sequence = load_supervised_game(paths["public_npz"], paths["teacher_npz"])
        if len(sequence) != steps:
            raise ValueError(f"{key} record steps={steps} but NPZ has {len(sequence)}")
        specs_value = json.loads(paths["generated_specs"].read_text())
        if not isinstance(specs_value, list) or not specs_value or not all(
            isinstance(item, dict) for item in specs_value
        ):
            raise ValueError(f"{key} generated specs must be a non-empty list of objects")
        levels = record.get("levels")
        if not isinstance(levels, list) or len(levels) != len(specs_value):
            raise ValueError(f"{key} generator provenance does not align with generated specs")
        if not smoke:
            curriculum = full_curricula[source.source_id]
            expected_split = "train" if name.startswith("train") else "validation"
            if len(specs_value) != len(curriculum):
                raise ValueError(f"{key} does not contain the complete source curriculum")
            if (
                record.get("full_standard_required") is not True
                or record.get("full_standard_validated") is not True
                or record.get("full_standard_contract_hash")
                != full_contract_hashes[source.source_id]
                or record.get("requested_curriculum") != curriculum
                or record.get("generation_split") != expected_split
            ):
                raise ValueError(f"{key} full-standard record metadata is missing or inconsistent")
        for level_index, (level, spec) in enumerate(zip(levels, specs_value)):
            if not isinstance(level, Mapping):
                raise ValueError(f"{key} level {level_index} provenance must be an object")
            if level.get("generator_version") is None:
                raise ValueError(f"{key} level {level_index} has no generator_version provenance")
            if (spec.get("generator_version") is not None
                    and spec["generator_version"] != level["generator_version"]):
                raise ValueError(f"{key} level {level_index} generator_version mismatch")
            if not smoke:
                entry = full_curricula[source.source_id][level_index]
                if any(level.get(field) != entry[field] for field in entry):
                    raise ValueError(f"{key} level {level_index} curriculum provenance mismatch")
                if (
                    level.get("generation_split") != expected_split
                    or level.get("full_standard_validated") is not True
                    or spec.get("split") != expected_split
                ):
                    raise ValueError(f"{key} level {level_index} full-standard split is inconsistent")
                geometry = spec.get("geometry_d4_sha256", spec.get("geometry_sha256"))
                gameplay = spec.get("gameplay_sha256", spec.get("gameplay_hash"))
                if not all(isinstance(value, str) and value for value in (geometry, gameplay)):
                    raise ValueError(
                        f"{key} level {level_index} lacks geometry/gameplay split identities"
                    )

        with np.load(paths["public_npz"], allow_pickle=False) as public:
            frames = public["frames"]
            boundaries = public["level_boundary"]
            final_terminal = bool(public["terminal"][-1])
            final_won = bool(public["won"][-1])
            final_state = str(public["state"][-1]).upper()
            levels_completed = int(public["levels_completed"][-1])
        if levels_completed > len(specs_value):
            raise ValueError(f"{key} public trace completed more levels than generated specs")
        actual_whole_game_win = (
            final_terminal and final_won and final_state == "WIN"
            and levels_completed == len(specs_value)
        )
        if (record.get("status") == "won") != actual_whole_game_win:
            raise ValueError(
                f"{key} record status contradicts final public state/completed generated levels"
            )
        if record.get("levels_completed") is not None and _strict_int(
            record["levels_completed"], f"{key}:levels_completed",
        ) != levels_completed:
            raise ValueError(f"{key} record/public levels_completed mismatch")
        if record.get("levels_generated") is not None and _strict_int(
            record["levels_generated"], f"{key}:levels_generated",
        ) != len(specs_value):
            raise ValueError(f"{key} record/spec generated level count mismatch")
        if record.get("level_boundaries") is not None and _strict_int(
            record["level_boundaries"], f"{key}:level_boundaries",
        ) != int(boundaries.sum()):
            raise ValueError(f"{key} record/public level boundary count mismatch")
        if record.get("final_state") is not None and str(record["final_state"]).upper() != final_state:
            raise ValueError(f"{key} record/public final state mismatch")
        variant = _record_variant(record)
        identity = puzzle_identity(
            specs_value,
            frames=frames,
            level_boundaries=boundaries,
            variant=variant,
            label=key,
        )
        effective = identity.effective_seeds
        fingerprints = identity.fingerprints

        action_source = sequence.action_source
        assert action_source is not None
        game = AuditedGame(
            key=key,
            source_id=source.source_id,
            source=source.slug,
            master_seed=_strict_int(record.get("master_seed"), f"{key}:master_seed"),
            game_index=_strict_int(record.get("game_index"), f"{key}:game_index"),
            status=str(record["status"]),
            partial=record["status"] != "won",
            record=record,
            public_path=paths["public_npz"],
            teacher_path=paths["teacher_npz"],
            specs_path=paths["generated_specs"],
            file_hashes=hashes,
            specs=tuple(dict(item) for item in specs_value),
            sequence=sequence,
            effective_seeds=effective,
            puzzle_fingerprints=fingerprints,
        )
        games.append(game)
        dataset_entries.append({
            "record": record_key,
            "source_id": source.source_id,
            "master_seed": game.master_seed,
            "game_index": game.game_index,
            "status": game.status,
            "files": hashes,
            "effective_seeds": effective,
            "fingerprints": list(fingerprints),
            # This binds replay-affecting private mappings and generator/search
            # provenance even when the three artifact files are unchanged.
            "record_metadata_hash": _digest_json(record),
        })
    if not games:
        raise ValueError(f"{name} manifest has no usable non-empty games")
    rows = sum(len(game.sequence) for game in games)
    random_rows = sum(int((game.sequence.action_source == M.RANDOM_SOURCE).sum()) for game in games)
    return DatasetSide(
        name=name,
        manifest_paths=(manifest_path,),
        manifest_hashes={str(manifest_path): sha256_file(manifest_path)},
        source_ids=requested,
        games=tuple(games),
        exclusions=tuple(exclusions),
        failed_records=tuple(failed_records),
        dataset_hash=_digest_json(dataset_entries),
        rows=rows,
        random_rows=random_rows,
        teacher_rows=rows - random_rows,
    )


def _manifest_paths(value: str | Path | Sequence[str | Path]) -> tuple[str | Path, ...]:
    if isinstance(value, (str, Path)):
        return (value,)
    paths = tuple(value)
    if not paths:
        raise ValueError("at least one manifest is required for each dataset side")
    return paths


def _merge_sides(parts: Sequence[DatasetSide], *, name: str, smoke: bool) -> DatasetSide:
    if not parts:
        raise ValueError(f"{name} has no manifest parts")
    games = tuple(game for part in parts for game in part.games)
    source_ids = tuple(dict.fromkeys(source for part in parts for source in part.source_ids))
    if not smoke and set(source_ids) != set(M.TRAIN_SOURCE_IDS):
        raise ValueError(f"{name} merged manifests do not cover all 24 intended families")
    if not smoke and name == "train":
        completed = {game.source_id for game in games if game.status == "won"}
        missing = sorted(set(M.TRAIN_SOURCE_IDS) - completed)
        if missing:
            raise ValueError(
                f"{name} has no completed whole-game episode for families: {', '.join(missing)}"
            )
    manifests = tuple(path for part in parts for path in part.manifest_paths)
    manifest_hashes = {
        path: digest for part in parts for path, digest in part.manifest_hashes.items()
    }
    dataset_entries = sorted([
        {
            "part_dataset_hash": part.dataset_hash,
            "manifest_hashes": sorted(part.manifest_hashes.values()),
        }
        for part in parts
    ], key=lambda item: _canonical_json(item))
    return DatasetSide(
        name=name,
        manifest_paths=manifests,
        manifest_hashes=manifest_hashes,
        source_ids=source_ids,
        games=games,
        exclusions=tuple(item for part in parts for item in part.exclusions),
        failed_records=tuple(item for part in parts for item in part.failed_records),
        dataset_hash=_digest_json(dataset_entries),
        rows=sum(part.rows for part in parts),
        random_rows=sum(part.random_rows for part in parts),
        teacher_rows=sum(part.teacher_rows for part in parts),
    )


def _audit_many(
    paths: str | Path | Sequence[str | Path], *, name: str, smoke: bool,
) -> DatasetSide:
    parts = [
        _audit_side(path, name=f"{name}[{index}]", smoke=smoke)
        for index, path in enumerate(_manifest_paths(paths))
    ]
    return _merge_sides(parts, name=name, smoke=smoke)


def _overlap(left: Iterable[Any], right: Iterable[Any]) -> set[Any]:
    return set(left) & set(right)


def audit_manifest_pair(
    train_manifest: str | Path | Sequence[str | Path],
    validation_manifest: str | Path | Sequence[str | Path],
    *,
    smoke: bool = False,
) -> DatasetBundle:
    """Audit generated whole-game train/validation data before model creation."""
    train = _audit_many(train_manifest, name="train", smoke=smoke)
    validation = _audit_many(validation_manifest, name="validation", smoke=smoke)
    checks = {
        "whole-game identities": (
            ((game.source_id, game.master_seed, game.game_index) for game in train.games),
            ((game.source_id, game.master_seed, game.game_index) for game in validation.games),
        ),
        "master seeds": (
            ((game.source_id, game.master_seed) for game in train.games),
            ((game.source_id, game.master_seed) for game in validation.games),
        ),
        "effective seeds": (
            ((game.source_id, seed) for game in train.games for seed in game.effective_seeds),
            ((game.source_id, seed) for game in validation.games for seed in game.effective_seeds),
        ),
        "puzzle fingerprints": (
            ((game.source_id, kind, digest) for game in train.games
             for kind, digest in game.puzzle_fingerprints),
            ((game.source_id, kind, digest) for game in validation.games
             for kind, digest in game.puzzle_fingerprints),
        ),
    }
    for label, (left, right) in checks.items():
        duplicates = _overlap(left, right)
        if duplicates:
            sample = sorted(map(str, duplicates))[:3]
            raise ValueError(f"train/validation overlap in {label}: {sample}")
    return DatasetBundle(
        train=train,
        validation=validation,
        smoke=bool(smoke),
        scope="smoke" if smoke else "full_24_family_experiment",
        fingerprint_limitations=(
            "Identity/raw initial-frame hashes can miss hidden engine state; canonical specs can "
            "over- or under-identify semantic equality. Explicit geometry/gameplay hashes are "
            "included when generators provide them, and every detected overlap is rejected."
        ),
    )


def slice_game_sequence(game: GameSequence, start: int, end: int) -> GameSequence:
    if not (0 <= start < end <= len(game)):
        raise ValueError("invalid whole-game chunk bounds")
    values: dict[str, Any] = {}
    for field in fields(GameSequence):
        value = getattr(game, field.name)
        values[field.name] = None if value is None else value[start:end]
    return GameSequence(**values)


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _d4_composition_table() -> dict[tuple[str, str], str]:
    """Derive the D4 group table from ``transform_frame`` so it cannot drift."""
    probe = np.arange(FRAME_SIZE * FRAME_SIZE, dtype=np.int64).reshape(FRAME_SIZE, FRAME_SIZE)
    images = {name: transform_frame(probe, name) for name in D4_NAMES}
    table: dict[tuple[str, str], str] = {}
    for first in D4_NAMES:
        for second in D4_NAMES:
            combined = transform_frame(images[first], second)
            matches = [name for name in D4_NAMES if np.array_equal(images[name], combined)]
            if len(matches) != 1:
                raise RuntimeError(f"D4 composition {first}->{second} is not unique")
            table[(first, second)] = matches[0]
    return table


_D4_COMPOSE = _d4_composition_table()


def compose_d4(first: str, second: str) -> str:
    """Name of the transform equal to applying ``first`` and then ``second``."""
    try:
        return _D4_COMPOSE[(first, second)]
    except KeyError as exc:
        raise ValueError(f"unknown D4 transform pair {first!r}, {second!r}") from exc


def inverse_d4(name: str) -> str:
    for candidate in D4_NAMES:
        if compose_d4(name, candidate) == "identity":
            return candidate
    raise ValueError(f"unknown D4 transform {name!r}")


def compose_variants(first: WholeGameVariant, second: WholeGameVariant) -> WholeGameVariant:
    """Bijection equal to ``first`` (raw->public) followed by ``second`` on top."""
    controls = tuple(
        second.control_raw_to_public[first.control_raw_to_public[index]]
        for index in range(ACTION_COUNT)
    )
    palette = tuple(
        second.palette_raw_to_public[first.palette_raw_to_public[index]]
        for index in range(len(first.palette_raw_to_public))
    )
    return WholeGameVariant(
        first.selected or second.selected,
        second.seed,
        controls,
        compose_d4(first.spatial, second.spatial),
        palette,
    )


def inverse_variant(variant: WholeGameVariant) -> WholeGameVariant:
    return WholeGameVariant(
        variant.selected,
        variant.seed,
        variant.control_public_to_raw,
        inverse_d4(variant.spatial),
        variant.palette_public_to_raw,
    )


def _click_position_map(name: str) -> np.ndarray:
    """Flat raw pixel index -> flat public pixel index, exactly as frames move."""
    probe = np.arange(FRAME_SIZE * FRAME_SIZE, dtype=np.int64).reshape(FRAME_SIZE, FRAME_SIZE)
    transformed = transform_frame(probe, name)
    positions = np.empty(FRAME_SIZE * FRAME_SIZE, dtype=np.int64)
    positions[transformed.ravel()] = np.arange(FRAME_SIZE * FRAME_SIZE, dtype=np.int64)
    return positions


_FRAME_KEYS = ("frames", "next_frames")
_MASK_KEYS = ("target_click_region",)  # spatial-only: follows the frame D4 map, no palette
_ACTION_PREFIXES = ("", "previous_", "executed_", "target_")
_LEGAL_MASK_KEY = "legal_action_mask"


def _legal_action_id_set(legal_action_ids: Iterable[int]) -> set[int]:
    legal = {_strict_int(value, "legal_action_ids") for value in legal_action_ids}
    if any(not 0 <= value < ACTION_COUNT for value in legal):
        raise ValueError("legal_action_ids must be in 0..7")
    return legal


def _transform_action_triple(
    ids: np.ndarray, x: np.ndarray, y: np.ndarray, *, control: np.ndarray, positions: np.ndarray, name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    id_values = np.asarray(ids).astype(np.int64)
    x_values = np.asarray(x).astype(np.int64)
    y_values = np.asarray(y).astype(np.int64)
    if not (id_values.shape == x_values.shape == y_values.shape):
        raise ValueError(f"{name} action id/x/y arrays must share one shape")
    present = id_values != COORDINATE_NONE
    if np.any(present & ((id_values < 0) | (id_values >= ACTION_COUNT))):
        raise ValueError(f"{name} action IDs must be in 0..7 or -1")
    click = present & (id_values == CLICK_ACTION)
    if np.any(click & (
        (x_values < 0) | (x_values >= FRAME_SIZE) | (y_values < 0) | (y_values >= FRAME_SIZE)
    )):
        raise ValueError(f"{name} click coordinates must be in 0..63")
    new_ids = id_values.copy()
    new_ids[present] = control[id_values[present]]
    new_x = x_values.copy()
    new_y = y_values.copy()
    flat = positions[y_values[click] * FRAME_SIZE + x_values[click]]
    new_y[click] = flat // FRAME_SIZE
    new_x[click] = flat % FRAME_SIZE
    return (
        new_ids.astype(np.asarray(ids).dtype),
        new_x.astype(np.asarray(x).dtype),
        new_y.astype(np.asarray(y).dtype),
    )


def apply_variant_to_game(
    arrays: GameSequence | Mapping[str, np.ndarray | None],
    variant: WholeGameVariant,
    legal_action_ids: Iterable[int],
) -> GameSequence | dict[str, np.ndarray | None]:
    """Push one whole game through ``variant`` as a further public bijection.

    Accepts either the stored public/teacher contract (``frames``, ``action_id``
    /``action_x``/``action_y``, ``legal_action_mask`` ...) or an in-memory
    ``GameSequence``.  Frames move spatially and through the palette; every
    ``*action_id`` follows the control permutation; every ``*action_x``/``_y``
    of a click row follows the same D4 map as the frame, as does the optional
    ``target_click_region`` pixel mask; ``legal_action_mask``
    columns are permuted.  Event flags, ``target_valid``, ``action_source`` and
    any other provenance are returned untouched.  The result has the same kind
    as the input; nothing is mutated in place.
    """
    legal = _legal_action_id_set(legal_action_ids)
    control = np.asarray(variant.control_raw_to_public, dtype=np.int64)
    if {int(control[value]) for value in legal} != legal:
        raise ValueError("variant control permutation does not preserve this game's legal action set")
    as_sequence = isinstance(arrays, GameSequence)
    values: dict[str, Any] = (
        {field.name: getattr(arrays, field.name) for field in fields(GameSequence)}
        if as_sequence else dict(arrays)
    )
    positions = _click_position_map(variant.spatial)
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            out[key] = None
        elif key in _FRAME_KEYS:
            frame = np.asarray(value)
            if frame.dtype != np.uint8:
                raise ValueError(f"{key} must be uint8 palette indices")
            out[key] = variant.public_frame(frame)
        elif key in _MASK_KEYS:
            mask = np.asarray(value)
            if mask.dtype != np.bool_:
                raise ValueError(f"{key} must be a bool pixel mask")
            out[key] = transform_frame(mask, variant.spatial)
        elif key == _LEGAL_MASK_KEY:
            mask = np.asarray(value)
            if mask.dtype != np.bool_ or mask.shape[-1] != ACTION_COUNT:
                raise ValueError("legal_action_mask must be bool [..., 8]")
            permuted = np.zeros_like(mask)
            permuted[..., control] = mask
            out[key] = permuted
        else:
            out[key] = value
    for prefix in _ACTION_PREFIXES:
        id_key, x_key, y_key = f"{prefix}action_id", f"{prefix}action_x", f"{prefix}action_y"
        if values.get(id_key) is None:
            continue
        if values.get(x_key) is None or values.get(y_key) is None:
            raise ValueError(f"{id_key} requires {x_key} and {y_key}")
        out[id_key], out[x_key], out[y_key] = _transform_action_triple(
            values[id_key], values[x_key], values[y_key],
            control=control, positions=positions, name=prefix + "action",
        )
    if as_sequence:
        return GameSequence(**out)
    return out


def game_legal_action_ids(sequence: GameSequence) -> tuple[int, ...]:
    """Union of advertised public action IDs across a whole game."""
    return tuple(int(value) for value in np.flatnonzero(np.asarray(sequence.legal_action_mask).any(axis=0)))


def canonical_sequence(sequence: GameSequence, stored_variant: WholeGameVariant) -> GameSequence:
    """Pull a stored public game back into raw engine space.

    The stored variant is a whole-game bijection, so its inverse is applied over
    the full action-ID domain rather than the game's advertised legal subset:
    undoing a permutation cannot create an illegal action, and a partial game may
    never have advertised every control the permutation touched.
    """
    if stored_variant.is_identity:
        return sequence
    restored = apply_variant_to_game(sequence, inverse_variant(stored_variant), range(ACTION_COUNT))
    assert isinstance(restored, GameSequence)
    return restored


def canonicalize_game(game: AuditedGame) -> AuditedGame:
    if game.canonical:
        return game
    return replace(game, sequence=canonical_sequence(game.sequence, game.stored_variant), canonical=True)


def canonicalize_side(side: DatasetSide) -> DatasetSide:
    return replace(side, games=tuple(canonicalize_game(game) for game in side.games))


def canonicalize_bundle(bundle: DatasetBundle) -> DatasetBundle:
    """Every train and validation sequence in raw engine space (identity variant)."""
    if bundle.canonical_inputs:
        return bundle
    return replace(
        bundle,
        train=canonicalize_side(bundle.train),
        validation=canonicalize_side(bundle.validation),
        canonical_inputs=True,
    )


@dataclass(frozen=True)
class LoadTimeVariantOptions:
    """Whole-game public variants sampled at load time as training augmentation.

    One variant is drawn per game per epoch from (seed, epoch, game index) and
    composed on top of whatever variant the stored game already carries.
    Validation stays untouched unless ``augment_validation`` is set, and even
    then only the offline metrics move; the closed-loop panel plays the real
    engine through the stored variant only.
    """

    enabled: bool = False
    mix_probability: float = 1.0
    seed: int = 0
    controls: bool = True
    spatial: bool = True
    palette: bool = True
    augment_validation: bool = False

    def __post_init__(self) -> None:
        self.variant_options()  # validates ranges/components exactly like VariantOptions

    def variant_options(self) -> VariantOptions:
        return VariantOptions(
            enabled=bool(self.enabled),
            mix_probability=float(self.mix_probability),
            seed=int(self.seed),
            controls=bool(self.controls),
            spatial=bool(self.spatial),
            palette=bool(self.palette),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_time_variant(
    options: LoadTimeVariantOptions,
    *,
    epoch: int,
    game_index: int,
    legal_action_ids: Iterable[int],
) -> WholeGameVariant:
    """The single variant used for every step of one game during one epoch."""
    return sample_whole_game_variant(
        options.variant_options(),
        source_id=f"load-time-epoch:{int(epoch)}",
        game_index=int(game_index),
        legal_action_ids=legal_action_ids,
    )


def augment_sequence_for_epoch(
    sequence: GameSequence,
    options: LoadTimeVariantOptions,
    *,
    epoch: int,
    game_index: int,
) -> tuple[GameSequence, WholeGameVariant]:
    legal = game_legal_action_ids(sequence)
    variant = load_time_variant(options, epoch=epoch, game_index=game_index, legal_action_ids=legal)
    if variant.is_identity:
        return sequence, variant
    transformed = apply_variant_to_game(sequence, variant, legal)
    assert isinstance(transformed, GameSequence)
    return transformed, variant


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    checkpoint_every_games: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    chunk_steps: int = 64
    auxiliary_transitions_per_chunk: int = 8
    metric_transitions_per_game: int = 32
    gradient_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    closed_loop_interval: int = 1
    closed_loop_games: int = 24
    # Diagnostic-only panel of TRAINING games played closed-loop on the same
    # schedule; 0 disables it.  Never used for checkpoint selection.  The
    # default covers every intended family once (the stratified panel picks one
    # game per family) instead of the first eight alphabetically.
    closed_loop_train_games: int = len(M.TRAIN_SOURCE_IDS)
    validation_max_actions_per_level: int = 128
    validation_max_game_actions: int = 512
    # Previous-action history fed to the policy.  ``"full"`` is the stored
    # executed-action triple; ``"none"`` replaces it with BOS at every step of
    # every game.  ``history_dropout`` (only with ``"full"``) is the probability
    # that one whole training game is fed BOS history this epoch; the decision
    # is sampled once per (seed, epoch, game) and carried across its chunks.
    # Targets and executed-action auxiliary labels never change.
    history_mode: str = "full"
    history_dropout: float = 0.0
    # Diagnostic only: additionally play the generated closed-loop panels (and
    # score offline validation) with history removed at inference.  Logged
    # under ``*_history_free`` keys and never used for checkpoint selection.
    history_free_diagnostic: bool = False
    # ``"chunk"``: one clipped optimizer step per TBPTT chunk (chunk-mean
    # losses, so short trailing chunks weigh as much as full ones).
    # ``"game"``: per-term loss sums over every chunk of a game divided by the
    # game's whole target counts, backward through each detached chunk, then one
    # clip + step per game.
    update_mode: str = "chunk"
    # Refuse to train when no training click row carries a stored click region.
    require_click_regions: bool = False
    min_click_region_coverage: float = 0.0
    # When True every stored train/validation game is pulled back through the
    # inverse of its stored public variant at load, so the policy trains and is
    # evaluated in raw engine space.  Load-time variants stay off unless enabled
    # explicitly; when enabled they compose on top of the canonical game.
    canonical_inputs: bool = False
    model: MultiGameModelConfig = MultiGameModelConfig()
    loss: LossWeights = LossWeights()
    load_variants: LoadTimeVariantOptions = LoadTimeVariantOptions()

    def __post_init__(self) -> None:
        # Training configuration is the authoritative provenance source.
        object.__setattr__(self, "model", replace(self.model, history_dropout=float(self.history_dropout)))
        for name in (
            "epochs", "checkpoint_every_games", "chunk_steps", "auxiliary_transitions_per_chunk",
            "metric_transitions_per_game", "closed_loop_interval", "closed_loop_games",
            "validation_max_actions_per_level", "validation_max_game_actions",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.min_click_region_coverage <= 1.0:
            raise ValueError("min_click_region_coverage must be in [0,1]")
        if self.closed_loop_train_games < 0:
            raise ValueError("closed_loop_train_games must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("optimizer rates and gradient_clip are invalid")
        if self.history_mode not in HISTORY_MODES:
            raise ValueError(f"history_mode must be one of {HISTORY_MODES}")
        if not 0.0 <= float(self.history_dropout) <= 1.0:
            raise ValueError("history_dropout must be in [0,1]")
        if self.history_mode == "none" and self.history_dropout != 0.0:
            raise ValueError("history_dropout only applies to history_mode='full'")
        if self.update_mode not in UPDATE_MODES:
            raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
        for name in ("action", "click", "next_frame", "events"):
            if getattr(self.loss, name) < 0:
                raise ValueError(f"loss weight {name} must be non-negative")

    @property
    def auxiliaries_enabled(self) -> bool:
        """False when both auxiliary heads have zero weight: they are then never run."""
        return self.loss.next_frame > 0 or self.loss.events > 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        data = dict(value)
        data["model"] = MultiGameModelConfig(**data.get("model", {}))
        data["loss"] = LossWeights(**data.get("loss", {}))
        data["load_variants"] = LoadTimeVariantOptions(**data.get("load_variants", {}))
        return cls(**data)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def history_keep_for_game(
    config: TrainingConfig, *, epoch: int, game_index: int,
) -> bool:
    """Whether one training game keeps its previous-action history this epoch.

    ``history_mode="none"`` always drops it.  Otherwise the decision is a
    deterministic function of (seed, epoch, game index) so every chunk of the
    game shares it and resumed runs reproduce it.
    """
    if config.history_mode == "none":
        return False
    probability = float(config.history_dropout)
    if probability <= 0.0:
        return True
    if probability >= 1.0:
        return False
    digest = hashlib.sha256(
        f"history-dropout:{int(config.seed)}:{int(epoch)}:{int(game_index)}".encode()
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return draw >= probability


def click_region_census(side: DatasetSide) -> dict[str, int]:
    """How many teacher click rows carry a stored (collector-verified) region.

    Games whose teacher file has no region array fall back to the exact pixel
    at collate time; those rows count as ``click_exact_fallback_rows``.
    ``click_multi_pixel_region_rows`` counts stored regions larger than one
    pixel, the only rows where set-valued supervision differs from exact.
    """
    census = {
        "games": len(side.games),
        "games_with_click_regions": 0,
        "click_rows": 0,
        "click_region_stored_rows": 0,
        "click_exact_fallback_rows": 0,
        "click_multi_pixel_region_rows": 0,
    }
    for game in side.games:
        sequence = game.sequence
        assert sequence.target_action_id is not None and sequence.target_valid is not None
        click_rows = np.asarray(sequence.target_valid).astype(bool) & (
            np.asarray(sequence.target_action_id) == CLICK_ACTION
        )
        count = int(click_rows.sum())
        census["click_rows"] += count
        if sequence.has_click_regions:
            assert sequence.target_click_region is not None
            census["games_with_click_regions"] += 1
            census["click_region_stored_rows"] += count
            sizes = np.asarray(sequence.target_click_region).reshape(len(sequence), -1).sum(-1)
            census["click_multi_pixel_region_rows"] += int((click_rows & (sizes > 1)).sum())
        else:
            census["click_exact_fallback_rows"] += count
    return census


def _sample_indices(length: int, count: int, device: torch.device) -> Tensor:
    chosen = np.sort(np.random.choice(length, size=min(length, count), replace=False))
    return torch.tensor([[0, int(index)] for index in chosen], dtype=torch.long, device=device)



def _game_auxiliary_indices(length: int, config: TrainingConfig, device: torch.device) -> list[Tensor]:
    bounds = _chunk_bounds(length, config.chunk_steps)
    # Preserve the old compute budget, but every game row has equal inclusion probability.
    count = sum(min(end - start, config.auxiliary_transitions_per_chunk) for start, end in bounds)
    chosen = _sample_indices(length, count, device)[:, 1]
    return [
        torch.stack((torch.zeros_like(local), local), dim=1)
        for start, end in bounds
        for local in (chosen[(chosen >= start) & (chosen < end)] - start,)
    ]

def _empty_transition_accumulator() -> dict[str, float | int]:
    return {
        "rows": 0, "pixels": 0, "correct_pixels": 0, "exact_frames": 0,
        "copy_correct_pixels": 0, "changed_pixels": 0, "changed_correct": 0,
        "frame_nll_sum": 0.0, "events": 0, "correct_events": 0,
        **{f"{event}_{count}": 0 for event in ("level_boundary", "terminal", "won")
           for count in ("tp", "fp", "fn", "positive")},
    }


def _transition_summary(values: Mapping[str, float | int]) -> dict[str, Any]:
    rows = int(values["rows"])
    if rows == 0:
        return {"available": False, "rows": 0}
    pixels = int(values["pixels"])
    changed = int(values["changed_pixels"])
    events = int(values["events"])
    return {
        "available": True,
        "rows": rows,
        "frame_cross_entropy": float(values["frame_nll_sum"]) / pixels,
        "pixel_accuracy": int(values["correct_pixels"]) / pixels,
        "exact_frame_accuracy": int(values["exact_frames"]) / rows,
        "copy_current_pixel_accuracy": int(values["copy_correct_pixels"]) / pixels,
        "changed_pixels": changed,
        "changed_pixel_accuracy": (
            int(values["changed_correct"]) / changed if changed else None
        ),
        "copy_current_changed_pixel_accuracy": 0.0 if changed else None,
        "event_accuracy": int(values["correct_events"]) / events if events else None,
        "positive_events": {
            event: {
                "true_positive": int(values[f"{event}_tp"]),
                "false_positive": int(values[f"{event}_fp"]),
                "false_negative": int(values[f"{event}_fn"]),
                "support": int(values[f"{event}_positive"]),
                "precision": _ratio(values[f"{event}_tp"], values[f"{event}_tp"] + values[f"{event}_fp"]),
                "recall": _ratio(values[f"{event}_tp"], values[f"{event}_positive"]),
            } for event in ("level_boundary", "terminal", "won")
        },
    }


def _accumulate_transition_metrics(
    accumulator: dict[str, float | int],
    frame_logits: Tensor,
    event_logits: Tensor,
    current: Tensor,
    target: Tensor,
    event_target: Tensor,
) -> None:
    guess = frame_logits.argmax(1)
    correct = guess == target
    changed = current != target
    accumulator["rows"] += len(target)
    accumulator["pixels"] += target.numel()
    accumulator["correct_pixels"] += int(correct.sum())
    accumulator["exact_frames"] += int(correct.flatten(1).all(1).sum())
    accumulator["copy_correct_pixels"] += int((current == target).sum())
    accumulator["changed_pixels"] += int(changed.sum())
    accumulator["changed_correct"] += int((correct & changed).sum())
    accumulator["frame_nll_sum"] += float(
        F.cross_entropy(frame_logits, target.long(), reduction="sum").detach()
    )
    event_guess = event_logits >= 0
    accumulator["events"] += event_target.numel()
    accumulator["correct_events"] += int((event_guess == event_target).sum())
    for index, event in enumerate(("level_boundary", "terminal", "won")):
        positive, predicted = event_target[:, index].bool(), event_guess[:, index]
        accumulator[f"{event}_tp"] += int((predicted & positive).sum())
        accumulator[f"{event}_fp"] += int((predicted & ~positive).sum())
        accumulator[f"{event}_fn"] += int((~predicted & positive).sum())
        accumulator[f"{event}_positive"] += int(positive.sum())


def _empty_policy_accumulator() -> dict[str, Any]:
    return {
        "actions": 0, "action_correct": 0, "repeat_baseline_correct": 0, "joint_correct": 0,
        "joint_region_correct": 0,
        "switch_targets": 0, "switch_correct": 0, "repeat_targets": 0, "repeat_correct": 0,
        "bos_targets": 0, "bos_correct": 0, "bos_joint_correct": 0,
        "clicks": 0, "click_correct": 0, "click_region_correct": 0, "click_region_size_sum": 0,
        "click_region_labelled": 0, "click_region_stored_rows": 0, "click_exact_fallback_rows": 0,
        "action_nll_sum": 0.0, "click_nll_sum": 0.0,
        # zero-based level index within the game -> counts
        "by_level": {},
    }


def _empty_level_accumulator() -> dict[str, int]:
    return {"actions": 0, "action_correct": 0, "joint_correct": 0, "joint_region_correct": 0}


def _accumulate_policy_metrics(
    accumulator: dict[str, Any],
    *,
    action_guess: Tensor,
    click_x: Tensor,
    click_y: Tensor,
    target_id: Tensor,
    target_x: Tensor,
    target_y: Tensor,
    previous_id: Tensor,
    selected: Tensor,
    action_nll: Tensor | None = None,
    click_nll: Tensor | None = None,
    click_region: Tensor | None = None,
    level_index: Tensor | None = None,
    region_stored: bool = False,
) -> None:
    """Add exact-match policy metrics over the ``selected`` rows.

    ``repeat_baseline`` scores the trivial policy that re-emits the previous
    executed action ID; ``joint`` requires the action ID to match and, for click
    targets, the exact pixel too.  ``click_region`` (bool ``[..., 64, 64]``)
    additionally scores a click as a region hit when the argmax pixel lies
    anywhere inside the engine-equivalent set; on exact-only labels this equals
    the exact hit.  ``joint_region`` is ``joint`` with the region hit in place
    of the exact pixel.

    Rows are also split into *switch* targets (the target differs from the
    previous executed action the policy was fed, i.e. rows where the repeat
    baseline is wrong) and *repeat* targets, and *BOS* rows (no previous
    action: the first step of a game).  ``level_index`` (zero-based, ``[...]``
    long) buckets rows per level within their game.  ``region_stored`` says
    whether this game's click labels came from a collector-verified region
    (as opposed to the exact-pixel fallback) so coverage can be reported.
    Per-row NLL tensors, when given, are summed over the same rows so losses
    can later be averaged per actual target.
    """
    selected = selected.bool()
    is_click = target_id == CLICK_ACTION
    action_ok = action_guess == target_id
    click_ok = (click_x == target_x) & (click_y == target_y)
    repeat = previous_id == target_id
    bos = previous_id == COORDINATE_NONE
    joint_ok = action_ok & (~is_click | click_ok)
    selected_click = selected & is_click
    if click_region is not None:
        region = click_region.bool()
        safe_x = click_x.long().clamp(0, FRAME_SIZE - 1)
        safe_y = click_y.long().clamp(0, FRAME_SIZE - 1)
        region_ok = torch.gather(
            region.flatten(-2), -1, (safe_y * FRAME_SIZE + safe_x).unsqueeze(-1),
        ).squeeze(-1)
        size = region.flatten(-2).sum(-1)
        accumulator["click_region_correct"] += int((region_ok & selected_click).sum())
        accumulator["click_region_size_sum"] += int(size[selected_click].sum())
        accumulator["click_region_labelled"] += int(((size > 1) & selected_click).sum())
    else:
        region_ok = click_ok
        accumulator["click_region_correct"] += int((click_ok & selected_click).sum())
        accumulator["click_region_size_sum"] += int(selected_click.sum())
    joint_region_ok = action_ok & (~is_click | region_ok)
    accumulator["actions"] += int(selected.sum())
    accumulator["action_correct"] += int((action_ok & selected).sum())
    accumulator["repeat_baseline_correct"] += int((repeat & selected).sum())
    accumulator["joint_correct"] += int((joint_ok & selected).sum())
    accumulator["joint_region_correct"] += int((joint_region_ok & selected).sum())
    accumulator["switch_targets"] += int((~repeat & selected).sum())
    accumulator["switch_correct"] += int((action_ok & ~repeat & selected).sum())
    accumulator["repeat_targets"] += int((repeat & selected).sum())
    accumulator["repeat_correct"] += int((action_ok & repeat & selected).sum())
    accumulator["bos_targets"] += int((bos & selected).sum())
    accumulator["bos_correct"] += int((action_ok & bos & selected).sum())
    accumulator["bos_joint_correct"] += int((joint_ok & bos & selected).sum())
    accumulator["clicks"] += int(selected_click.sum())
    accumulator["click_correct"] += int((click_ok & selected_click).sum())
    click_rows = int(selected_click.sum())
    if click_region is not None and region_stored:
        accumulator["click_region_stored_rows"] += click_rows
    else:
        accumulator["click_exact_fallback_rows"] += click_rows
    if level_index is not None:
        by_level = accumulator["by_level"]
        for value in torch.unique(level_index[selected]).tolist():
            rows = selected & (level_index == value)
            bucket = by_level.setdefault(int(value), _empty_level_accumulator())
            bucket["actions"] += int(rows.sum())
            bucket["action_correct"] += int((action_ok & rows).sum())
            bucket["joint_correct"] += int((joint_ok & rows).sum())
            bucket["joint_region_correct"] += int((joint_region_ok & rows).sum())
    if action_nll is not None:
        accumulator["action_nll_sum"] += float(action_nll[selected].sum())
    if click_nll is not None:
        accumulator["click_nll_sum"] += float(click_nll[selected_click].sum())


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return numerator / denominator if denominator else None


def _policy_summary(values: Mapping[str, Any]) -> dict[str, Any]:
    actions = int(values["actions"])
    clicks = int(values["clicks"])
    by_level = {
        str(index): {
            "actions": int(bucket["actions"]),
            "action_accuracy": _ratio(int(bucket["action_correct"]), int(bucket["actions"])),
            "joint_accuracy": _ratio(int(bucket["joint_correct"]), int(bucket["actions"])),
            "joint_region_accuracy": _ratio(
                int(bucket["joint_region_correct"]), int(bucket["actions"]),
            ),
        }
        for index, bucket in sorted(values.get("by_level", {}).items())
    }
    return {
        "actions": actions,
        "action_correct": int(values["action_correct"]),
        "action_accuracy": _ratio(int(values["action_correct"]), actions),
        "repeat_baseline_correct": int(values["repeat_baseline_correct"]),
        "repeat_baseline_accuracy": _ratio(int(values["repeat_baseline_correct"]), actions),
        "joint_correct": int(values["joint_correct"]),
        "joint_accuracy": _ratio(int(values["joint_correct"]), actions),
        # Joint with the region hit instead of the exact pixel for click rows.
        "joint_region_correct": int(values["joint_region_correct"]),
        "joint_region_accuracy": _ratio(int(values["joint_region_correct"]), actions),
        # Switch: target differs from the fed previous action (repeat baseline
        # wrong); repeat: target equals it.  BOS: first step, no previous action.
        "switch_targets": int(values["switch_targets"]),
        "switch_accuracy": _ratio(int(values["switch_correct"]), int(values["switch_targets"])),
        "repeat_targets": int(values["repeat_targets"]),
        "repeat_accuracy": _ratio(int(values["repeat_correct"]), int(values["repeat_targets"])),
        "bos_targets": int(values["bos_targets"]),
        "bos_accuracy": _ratio(int(values["bos_correct"]), int(values["bos_targets"])),
        "bos_joint_accuracy": _ratio(int(values["bos_joint_correct"]), int(values["bos_targets"])),
        "clicks": clicks,
        "click_correct": int(values["click_correct"]),
        "click_accuracy": _ratio(int(values["click_correct"]), clicks),
        # Region hit: argmax anywhere inside the engine-equivalent click set.
        "click_region_correct": int(values["click_region_correct"]),
        "click_region_accuracy": _ratio(int(values["click_region_correct"]), clicks),
        "click_region_mean_size": _ratio(int(values["click_region_size_sum"]), clicks),
        "click_region_labelled": int(values["click_region_labelled"]),
        # Click rows whose label came from a stored collector region versus the
        # exact-pixel fallback used for teacher files without regions.
        "click_region_stored_rows": int(values["click_region_stored_rows"]),
        "click_exact_fallback_rows": int(values["click_exact_fallback_rows"]),
        "by_level_index": by_level,
        "action_loss": _ratio(float(values["action_nll_sum"]), actions),
        "click_loss_per_target": _ratio(float(values["click_nll_sum"]), clicks),
    }


@torch.inference_mode()
def evaluate_generated_offline(
    model: MultiGameModel,
    games: Sequence[AuditedGame],
    *,
    chunk_steps: int,
    transition_limit_per_game: int,
    device: torch.device,
    history_free: bool = False,
    stop_request: StopRequest | None = None,
) -> dict[str, Any]:
    """Offline generated validation diagnostics; sources remain separate.

    Losses are averaged per actual target (every valid action target for the
    action loss, every click target for the click loss) rather than per chunk.
    ``history_free`` feeds BOS in place of every previous-action triple
    (``history_keep=False``) as a diagnostic of history dependence.
    """
    was_training = model.training
    model.eval()
    try:
        return _evaluate_generated_offline(
            model, games, chunk_steps=chunk_steps,
            transition_limit_per_game=transition_limit_per_game, device=device,
            history_free=history_free, stop_request=stop_request,
        )
    finally:
        if was_training:
            model.train()


def _history_keep_kwargs(history_free: bool, *, device: torch.device) -> dict[str, Tensor]:
    """``history_keep`` only when history is removed, so legacy models still run."""
    if not history_free:
        return {}
    return {"history_keep": torch.zeros(1, dtype=torch.bool, device=device)}


def _evaluate_generated_offline(
    model: MultiGameModel,
    games: Sequence[AuditedGame],
    *,
    chunk_steps: int,
    transition_limit_per_game: int,
    device: torch.device,
    history_free: bool,
    stop_request: StopRequest | None = None,
) -> dict[str, Any]:
    overall = _empty_policy_accumulator()
    policy_groups = {"random": _empty_policy_accumulator(), "teacher": _empty_policy_accumulator()}
    by_family: dict[str, dict[str, Any]] = {}
    transitions = {"random": _empty_transition_accumulator(), "teacher": _empty_transition_accumulator()}
    history_kwargs = _history_keep_kwargs(history_free, device=device)
    for game in games:
        family = by_family.setdefault(game.source_id, _empty_policy_accumulator())
        memory = model.initial_memory(1, device=device)
        metric_rows = {}
        for source_value in (0, 1):
            available = np.flatnonzero(np.asarray(game.sequence.action_source) == source_value)
            positions = np.linspace(0, len(available) - 1, min(len(available), transition_limit_per_game), dtype=int)
            metric_rows[source_value] = available[positions]
        region_stored = game.sequence.has_click_regions
        levels_before = 0
        for start in range(0, len(game.sequence), chunk_steps):
            if stop_request is not None:
                stop_request.check()
            chunk = slice_game_sequence(game.sequence, start, min(len(game.sequence), start + chunk_steps))
            batch = _to_device(collate_game_sequences((chunk,)), device)
            encoded = model.encode_history(batch, initial_memory=memory, **history_kwargs)
            memory = encoded.final_memory
            policy = model.policy_from_history(encoded, batch["legal_action_mask"])
            valid = batch["target_valid"] & ~batch["padding_mask"]
            target_id = batch["target_action_id"]
            # Zero-based level index of each row: boundaries crossed before it.
            boundaries = batch["previous_level_boundary"].long()
            level_index = levels_before + boundaries.cumsum(1)
            levels_before += int(boundaries.sum())
            safe_target = target_id.clamp(min=0)
            action_nll = F.cross_entropy(
                policy.action_logits.flatten(0, 1), safe_target.flatten(), reduction="none",
            ).view_as(target_id)
            # Set-valued click NLL (mass inside the equivalent region); rows
            # without a click target get a placeholder region so the tensor is
            # finite everywhere, and are never selected below.
            click_region = batch["target_click_region"].bool()
            placeholder = click_region.clone()
            placeholder[..., 0, 0] |= ~(target_id == CLICK_ACTION)
            click_nll = click_region_nll(policy.click_logits, placeholder)
            action_guess = policy.action_logits.argmax(-1)
            click_x, click_y = model.decode_click(policy.click_logits)
            metric_inputs = dict(
                action_guess=action_guess, click_x=click_x, click_y=click_y,
                target_id=target_id, target_x=batch["target_action_x"], target_y=batch["target_action_y"],
                previous_id=batch["previous_action_id"], action_nll=action_nll, click_nll=click_nll,
                click_region=click_region, level_index=level_index, region_stored=region_stored,
            )
            _accumulate_policy_metrics(overall, selected=valid, **metric_inputs)
            _accumulate_policy_metrics(family, selected=valid, **metric_inputs)
            for source_value, label in ((0, "random"), (1, "teacher")):
                selected = valid & (batch["action_source"] == source_value)
                _accumulate_policy_metrics(policy_groups[label], selected=selected, **metric_inputs)

                rows = metric_rows[source_value]
                local = rows[(rows >= start) & (rows < start + len(chunk))] - start
                if not len(local):
                    continue
                chosen = torch.as_tensor(local, dtype=torch.long, device=device)
                indices = torch.stack((torch.zeros_like(chosen), chosen), dim=1)
                predicted = model.predict_transitions(
                    encoded,
                    batch["executed_action_id"], batch["executed_action_x"], batch["executed_action_y"],
                    indices=indices,
                )
                current = batch["frames"][0, chosen]
                target = batch["next_frames"][0, chosen]
                event_target = torch.stack((
                    batch["next_level_boundary"][0, chosen],
                    batch["next_terminal"][0, chosen],
                    batch["next_won"][0, chosen],
                ), dim=-1).bool()
                _accumulate_transition_metrics(
                    transitions[label], predicted.next_frame_logits, predicted.event_logits,
                    current, target, event_target,
                )
    if stop_request is not None:
        stop_request.check()
    summary = _policy_summary(overall)
    return {
        "history_free": bool(history_free),
        "policy_action_loss": summary["action_loss"],
        # Both losses are averaged per actual target; the click loss counts
        # only rows whose teacher target is a click.
        "policy_click_loss": summary["click_loss_per_target"],
        "policy_click_loss_per_target": summary["click_loss_per_target"],
        "action_targets": summary["actions"],
        "click_targets": summary["clicks"],
        "action_accuracy": summary["action_accuracy"],
        "repeat_baseline_accuracy": summary["repeat_baseline_accuracy"],
        "joint_action_click_accuracy": summary["joint_accuracy"],
        "joint_action_region_accuracy": summary["joint_region_accuracy"],
        "switch_targets": summary["switch_targets"],
        "switch_accuracy": summary["switch_accuracy"],
        "repeat_targets": summary["repeat_targets"],
        "repeat_accuracy": summary["repeat_accuracy"],
        "bos_targets": summary["bos_targets"],
        "bos_accuracy": summary["bos_accuracy"],
        "bos_joint_accuracy": summary["bos_joint_accuracy"],
        "click_accuracy": summary["click_accuracy"],
        "click_region_accuracy": summary["click_region_accuracy"],
        "click_region_mean_size": summary["click_region_mean_size"],
        "click_region_labelled": summary["click_region_labelled"],
        "click_region_stored_rows": summary["click_region_stored_rows"],
        "click_exact_fallback_rows": summary["click_exact_fallback_rows"],
        "by_level_index": summary["by_level_index"],
        "policy": {label: _policy_summary(value) for label, value in policy_groups.items()},
        "by_family": {
            source_id: _policy_summary(value) for source_id, value in sorted(by_family.items())
        },
        "transition": {label: _transition_summary(value) for label, value in transitions.items()},
    }


CLOSED_LOOP_OUTCOMES = ("won", "game_over", "budget_exhausted", "adapter_error")
_BUDGET_FAILURES = frozenset({"per_level_action_budget", "whole_game_action_budget"})


def _build_generated_env(audited: AuditedGame) -> Any:
    """Construct the real engine for one generated whole game (all its levels)."""
    source = M.source_for(audited.source_id)
    env_module = importlib.import_module(f"{source.package}.env")
    generate_module = importlib.import_module(f"{source.package}.generate")
    levels = [generate_module.build_level(spec) for spec in audited.specs]
    return env_module.Env(levels)


def _env_state(env: Any) -> str:
    return str(getattr(getattr(env, "state"), "value", getattr(env, "state"))).upper()


def classify_closed_loop_outcome(
    *, state: str, levels_completed: int, levels_requested: int, failure: str | None,
) -> str:
    """Exactly one of ``CLOSED_LOOP_OUTCOMES`` for a finished closed-loop game.

    ``won`` needs the engine WIN state with every requested level complete;
    ``game_over`` is the engine's own terminal failure; ``budget_exhausted`` is
    an external per-level or whole-game action cap while the game was still
    alive; everything else (engine exceptions, malformed frames, inconsistent
    terminal states) is an ``adapter_error``.
    """
    if state == "WIN" and levels_completed == levels_requested and failure is None:
        return "won"
    if state == "GAME_OVER" and failure is None:
        return "game_over"
    if failure in _BUDGET_FAILURES and state not in {"WIN", "GAME_OVER"}:
        return "budget_exhausted"
    return "adapter_error"


def summarize_closed_loop_games(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = {name: 0 for name in CLOSED_LOOP_OUTCOMES}
    by_family: dict[str, dict[str, int]] = {}
    for item in results:
        outcomes[item["outcome"]] += 1
        family = by_family.setdefault(str(item["source_id"]), {
            "games": 0, "levels_completed": 0, "levels_requested": 0, "games_won": 0,
            **{f"outcome_{name}": 0 for name in CLOSED_LOOP_OUTCOMES},
        })
        family["games"] += 1
        family["levels_completed"] += int(item["levels_completed"])
        family["levels_requested"] += int(item["levels_requested"])
        family["games_won"] += int(bool(item["won"]))
        family[f"outcome_{item['outcome']}"] += 1
    return {
        "games": list(results),
        "games_played": len(results),
        "games_won": sum(bool(item["won"]) for item in results),
        "levels_completed": sum(int(item["levels_completed"]) for item in results),
        "levels_requested": sum(int(item["levels_requested"]) for item in results),
        "actions": sum(int(item["actions"]) for item in results),
        "outcomes": outcomes,
        "by_family": dict(sorted(by_family.items())),
    }


@torch.inference_mode()
def evaluate_generated_closed_loop(
    model: MultiGameModel,
    games: Sequence[AuditedGame],
    *,
    max_games: int,
    max_actions_per_level: int,
    max_game_actions: int,
    device: torch.device,
    history_free: bool = False,
    stop_request: StopRequest | None = None,
) -> dict[str, Any]:
    """Play a fixed generated panel without teacher/planner fallback.

    Each game is played through ``audited.play_variant``: the stored public
    variant for as-stored data, or the identity when the game was canonicalized,
    so the policy always sees the same frame/action space it was trained on.
    ``history_free`` removes previous-action history at inference. Normal
    selection uses it when the training contract is history_mode="none";
    additional history interventions remain separate diagnostics.

    An engine that advertises no public action (or an action ID outside the
    contract) is reported as an ``adapter_error`` for that game without
    executing anything; it never aborts the panel.
    """
    was_training = model.training
    model.eval()
    selected = _source_stratified_panel(games, max_games=max_games)
    try:
        results = []
        for audited in selected:
            if stop_request is not None:
                stop_request.check()
            results.append(_play_generated_game(
                model, audited,
                max_actions_per_level=max_actions_per_level,
                max_game_actions=max_game_actions,
                device=device, history_free=history_free,
            ))
        if stop_request is not None:
            stop_request.check()
    finally:
        if was_training:
            model.train()
    return {
        "panel_records": [game.key for game in selected],
        "canonical_inputs": all(game.canonical for game in selected) if selected else None,
        "history_free": bool(history_free),
        **summarize_closed_loop_games(results),
    }


def _public_legal_mask(env: Any, variant: WholeGameVariant) -> np.ndarray:
    """Validated public legal mask; raises ``ValueError`` on contract breaches.

    Action 0 (RESET) is never offered to the policy.  Any other ID outside
    ``1..7``, a non-integer ID, or a set with no playable action is an adapter
    fault the caller reports instead of letting the model raise mid-game.
    """
    raw_mask = np.zeros(M.ACTION_COUNT, dtype=np.bool_)
    for value in env.available_actions:
        try:
            action_id = int(getattr(value, "value", value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid_legal_action_id:{value!r}") from exc
        if not 0 <= action_id < M.ACTION_COUNT:
            raise ValueError(f"invalid_legal_action_id:{action_id}")
        if action_id:
            raw_mask[action_id] = True
    public_mask = variant.public_legal_mask(raw_mask)
    public_mask[0] = False
    if not public_mask[1:].any():
        raise ValueError("no_legal_public_action")
    return public_mask


def _play_generated_game(
    model: MultiGameModel,
    audited: AuditedGame,
    *,
    max_actions_per_level: int,
    max_game_actions: int,
    device: torch.device,
    history_free: bool,
) -> dict[str, Any]:
    history_kwargs = _history_keep_kwargs(history_free, device=device)
    source = M.source_for(audited.source_id)
    variant = audited.play_variant
    memory = model.initial_memory(1, device=device)
    previous_id = previous_x = previous_y = -1
    previous_boundary = False
    actions = 0
    level_actions = 0
    failure = None
    levels_requested = len(audited.specs)
    env = None
    try:
        env = _build_generated_env(audited)
        env.reset()
    except Exception as exc:
        failure = f"engine_error:{type(exc).__name__}:{exc}"
    while failure is None and actions < max_game_actions:
        state = _env_state(env)
        if state in {"WIN", "GAME_OVER"}:
            break
        if level_actions >= max_actions_per_level:
            failure = "per_level_action_budget"
            break
        # Validate the advertised action set before any inference so a
        # contract breach is reported, not raised from inside the model.
        try:
            public_mask = _public_legal_mask(env, variant)
        except ValueError as exc:
            failure = str(exc)
            break
        try:
            raw_frame = _validated_raw_frame(env.render())
        except ValueError as exc:
            failure = str(exc)
            break
        frame = torch.from_numpy(variant.public_frame(raw_frame))[None].to(device)
        output, memory = model.policy_step(
            frame=frame,
            previous_action_id=torch.tensor([previous_id], device=device),
            previous_action_x=torch.tensor([previous_x], device=device),
            previous_action_y=torch.tensor([previous_y], device=device),
            previous_level_boundary=torch.tensor([previous_boundary], device=device),
            terminal=torch.tensor([False], device=device),
            won=torch.tensor([False], device=device),
            legal_action_mask=torch.from_numpy(public_mask)[None].to(device),
            memory=memory,
            **history_kwargs,
        )
        public_id = int(output.action_logits.argmax(-1).item())
        if public_id == CLICK_ACTION:
            x_value, y_value = model.decode_click(output.click_logits)
            public_x, public_y = int(x_value.item()), int(y_value.item())
        else:
            public_x = public_y = None
        raw_id, raw_x, raw_y = variant.raw_action(public_id, public_x, public_y)
        before = int(env.levels_completed)
        try:
            env.perform(raw_id, raw_x, raw_y)
        except Exception as exc:
            failure = f"engine_error:{type(exc).__name__}:{exc}"
            break
        after = int(env.levels_completed)
        previous_boundary = after > before
        previous_id = public_id
        previous_x = -1 if public_x is None else public_x
        previous_y = -1 if public_y is None else public_y
        actions += 1
        level_actions = 0 if previous_boundary else level_actions + 1
    if env is None:
        state, completed = "UNAVAILABLE", 0
    else:
        state = _env_state(env)
        completed = int(env.levels_completed)
    if actions >= max_game_actions and state not in {"WIN", "GAME_OVER"} and failure is None:
        failure = "whole_game_action_budget"
    outcome = classify_closed_loop_outcome(
        state=state, levels_completed=completed, levels_requested=levels_requested,
        failure=failure,
    )
    return {
        "record": audited.key,
        "source_id": source.source_id,
        "levels_requested": levels_requested,
        "levels_completed": completed,
        "won": outcome == "won",
        "outcome": outcome,
        "final_state": state,
        "actions": actions,
        "failure": failure,
        "variant": "identity" if variant.is_identity else "stored",
    }


def _source_stratified_panel(
    games: Sequence[AuditedGame], *, max_games: int,
) -> list[AuditedGame]:
    """Choose a stable round-robin panel so an alphabetical family cannot dominate."""
    by_source: dict[str, list[AuditedGame]] = {}
    for game in sorted(games, key=lambda item: (item.source_id, item.master_seed, item.game_index, item.key)):
        by_source.setdefault(game.source_id, []).append(game)
    selected: list[AuditedGame] = []
    depth = 0
    while len(selected) < max_games:
        added = False
        for source_id in sorted(by_source):
            if depth < len(by_source[source_id]):
                selected.append(by_source[source_id][depth])
                added = True
                if len(selected) == max_games:
                    break
        if not added:
            break
        depth += 1
    return selected


def _validated_raw_frame(value: Any) -> np.ndarray:
    frame = np.asarray(value)
    if frame.shape != M.FRAME_SHAPE:
        raise ValueError(f"invalid raw frame shape: {frame.shape}")
    if frame.dtype == np.bool_ or not np.issubdtype(frame.dtype, np.integer):
        raise ValueError(f"invalid raw frame dtype: {frame.dtype}; expected non-bool integers")
    if frame.size and (int(frame.min()) < 0 or int(frame.max()) >= 16):
        raise ValueError("invalid raw frame palette outside 0..15")
    return frame.astype(np.uint8, copy=True)


def _training_signature(config: TrainingConfig) -> dict[str, Any]:
    value = config.to_dict()
    value.pop("epochs", None)
    value.pop("device", None)
    value.pop("checkpoint_every_games", None)
    # Diagnostics and pre-flight checks do not change what is learned.
    value.pop("history_free_diagnostic", None)
    value.pop("require_click_regions", None)
    value.pop("min_click_region_coverage", None)
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _commit_checkpoints(output_dir: Path, payload: Mapping[str, Any], *, promoted: bool) -> None:
    """Commit latest and its bound best through one durable pointer replacement.

    Files in a generation are immutable. Readers of latest.pt/best.pt follow the
    same .checkpoint-current directory pointer. A crash before pointer replacement
    exposes the preceding complete pair; a crash after it exposes the new pair.
    Unselected best weights use hard links, avoiding duplicate checkpoint bytes.
    """
    generations = output_dir / ".checkpoint-generations"
    generations.mkdir(exist_ok=True)
    current = output_dir / ".checkpoint-current"
    previous = current.resolve() if current.is_symlink() else None
    generation = generations / uuid.uuid4().hex
    generation.mkdir()
    _atomic_torch_save(generation / "latest.pt", payload)
    if promoted:
        _atomic_torch_save(generation / "best.pt", {**payload, "checkpoint_role": "best_generated_validation"})
    elif previous is not None and (previous / "best.pt").is_file():
        os.link(previous / "best.pt", generation / "best.pt")
    _fsync_directory(generation)
    _fsync_directory(generations)
    # Stable public aliases are installed before the very first commit.
    for name in ("latest.pt", "best.pt"):
        alias = output_dir / name
        if not alias.is_symlink():
            if alias.exists():
                raise ValueError(f"refusing to replace unmanaged checkpoint {alias}")
            alias.symlink_to(Path(".checkpoint-current") / name)
    pending = output_dir / ".checkpoint-current.tmp"
    pending.unlink(missing_ok=True)
    pending.symlink_to(generation.relative_to(output_dir), target_is_directory=True)
    os.replace(pending, current)
    _fsync_directory(output_dir)
    # Keep the preceding committed generation as well. Clean interrupted writes
    # only after the next successful commit, never before a recoverable commit.
    for candidate in generations.iterdir():
        if candidate != generation and candidate != previous and candidate.is_dir():
            shutil.rmtree(candidate)


def _validate_checkpoint_layout(output_dir: Path) -> None:
    """Reject flattened exports or changed aliases before any resumed update."""
    for name in ("latest.pt", "best.pt"):
        alias = output_dir / name
        expected = f".checkpoint-current/{name}"
        if not alias.is_symlink() or os.readlink(alias) != expected:
            raise ValueError(f"resume checkpoint layout requires {name} -> {expected}; preserve the whole run directory or use --initialize-from")
    current = output_dir / ".checkpoint-current"
    generations = output_dir / ".checkpoint-generations"
    if not current.is_symlink() or not generations.is_dir() or generations.is_symlink():
        raise ValueError("resume checkpoint layout requires a committed generation pointer")
    target_text = os.readlink(current)
    target = Path(target_text)
    if (len(target.parts) != 2 or target.parts[0] != ".checkpoint-generations"
            or target.parts[1] in (".", "..") or str(target) != target_text):
        raise ValueError("resume checkpoint generation pointer is invalid")
    generation = output_dir / target
    if not generation.is_dir() or generation.is_symlink():
        raise ValueError("resume committed checkpoint generation is missing or redirected")
    latest = generation / "latest.pt"
    best = generation / "best.pt"
    if not latest.is_file() or latest.is_symlink() or best.is_symlink():
        raise ValueError("resume committed generation must contain regular checkpoint files")


class _TrainingStopped(Exception):
    pass


class StopRequest:
    """Signal handlers only set a flag; all writes happen at safe game boundaries."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def request(self, signum=None, _frame=None) -> None:
        self.requested = True
        self.signal_number = signum

    def check(self) -> None:
        if self.requested:
            raise _TrainingStopped()


@contextmanager
def graceful_training_signals(stop: StopRequest):
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        for number in previous:
            signal.signal(number, stop.request)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _ordered_game_identity(bundle: DatasetBundle) -> dict[str, str]:
    return {
        name: _digest_json([
            {"key": game.key, "hashes": game.file_hashes}
            for game in side.games
        ])
        for name, side in (("train", bundle.train), ("validation", bundle.validation))
    }


def load_training_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != TRAINING_FORMAT:
        raise ValueError(f"unsupported training checkpoint format {payload.get('format')!r}")
    return payload


def model_from_training_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[MultiGameModel, dict[str, Any]]:
    payload = load_training_checkpoint(path, device=device)
    model = MultiGameModel(MultiGameModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


SELECTION_BASIS = (
    "generated_validation_closed_loop: (games_won, levels_completed) then lower offline "
    "action loss; outcome categories are reported but never a tie-break"
)


def selection_score(closed_loop: Mapping[str, Any], offline: Mapping[str, Any]) -> tuple[Any, ...]:
    """Checkpoint ranking key; larger is better.

    Only real closed-loop progress (games won, then levels completed) ranks
    candidates.  The offline action loss is the last resort tie-break.  How a
    game ended (game_over vs. budget exhaustion vs. adapter error) is reported
    for diagnosis but deliberately does not enter the key: a policy that dies
    must not outrank one that stays alive without progress, and vice versa.
    """
    action_loss = offline.get("policy_action_loss")
    action_loss = float("inf") if action_loss is None else float(action_loss)
    return (
        int(closed_loop["games_won"]),
        int(closed_loop["levels_completed"]),
        -action_loss,
    )


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _closed_loop_phrase(panel: Mapping[str, Any] | None) -> str:
    if panel is None:
        return "skipped"
    levels = f"levels {int(panel.get('levels_completed', 0))}"
    requested = panel.get("levels_requested")
    if requested is not None:
        levels += f"/{int(requested)}"
    text = f"{levels} won {int(panel.get('games_won', 0))}/{int(panel.get('games_played', len(panel.get('games', ()))))}"
    outcomes = panel.get("outcomes")
    if outcomes:
        text += " [" + " ".join(
            f"{name}={int(outcomes.get(name, 0))}" for name in CLOSED_LOOP_OUTCOMES if name != "won"
        ) + "]"
    return text


def format_epoch_line(log: Mapping[str, Any]) -> str:
    """One compact human-readable line per epoch."""
    offline = log.get("generated_validation_offline") or {}
    return (
        f"epoch {int(log['epoch'])} | train loss {_fmt(log['train'].get('total'))}"
        f" | offline action acc {_fmt(offline.get('action_accuracy'))}"
        f" vs repeat {_fmt(offline.get('repeat_baseline_accuracy'))}"
        f" joint {_fmt(offline.get('joint_action_click_accuracy'))}"
        f" action loss {_fmt(offline.get('policy_action_loss'), 4)}"
        f" click loss/target {_fmt(offline.get('policy_click_loss_per_target'), 4)}"
        f" switch {_fmt(offline.get('switch_accuracy'))}"
        f" click acc exact {_fmt(offline.get('click_accuracy'))}"
        f" region {_fmt(offline.get('click_region_accuracy'))}"
        f" | hist-drop {_fmt(log['train'].get('history_dropped_fraction'), 2)}"
        f" | val closed-loop {_closed_loop_phrase(log.get('generated_validation_closed_loop'))}"
        f" | train closed-loop {_closed_loop_phrase(log.get('generated_train_closed_loop'))}"
        f"{' | promoted' if log.get('promoted') else ''}"
    )


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint: Path
    latest_checkpoint: Path
    logs_path: Path
    logs: tuple[dict[str, Any], ...]
    scope: str
    stopped: bool = False



def _supports_kwarg(function: Any, name: str) -> bool:
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _history_loss_kwargs(keep: bool, *, device: torch.device) -> dict[str, Tensor]:
    """``history_keep=False`` for a dropped game; nothing when kept (legacy path)."""
    if keep:
        return {}
    return {"history_keep": torch.zeros(1, dtype=torch.bool, device=device)}


def _check_model_capabilities(model: MultiGameModel, config: TrainingConfig) -> None:
    """Fail before training when a requested mode needs model support that is absent."""
    if (config.history_mode != "full" or config.history_dropout > 0) and not _supports_kwarg(
        compute_multigame_loss, "history_keep",
    ):
        raise ValueError(
            "history_mode='none' / history_dropout>0 require a model build whose "
            "compute_multigame_loss accepts history_keep"
        )
    if config.update_mode == "game" and not _supports_kwarg(compute_multigame_loss, "reduction"):
        raise ValueError(
            "update_mode='game' requires a model build whose compute_multigame_loss "
            "accepts reduction='sum'"
        )
    if (config.history_free_diagnostic or config.history_mode == "none") and not (
        _supports_kwarg(model.policy_step, "history_keep")
        and _supports_kwarg(model.encode_history, "history_keep")
    ):
        raise ValueError(
            "history_free_diagnostic / history_mode='none' requires a model whose policy_step/encode_history "
            "accept history_keep"
        )


def _empty_update_stats() -> dict[str, float | int]:
    return {
        "updates": 0, "chunks": 0, "total": 0.0, "action": 0.0, "click": 0.0,
        "frame": 0.0, "events": 0.0, "click_nll_sum": 0.0, "click_targets": 0,
        "click_exact_correct": 0, "click_region_correct": 0,
    }


def _merge_update_stats(into: dict[str, float | int], part: Mapping[str, float | int]) -> None:
    for key, value in part.items():
        into[key] += value


def _chunk_bounds(length: int, chunk_steps: int) -> list[tuple[int, int]]:
    return [(start, min(length, start + chunk_steps)) for start in range(0, length, chunk_steps)]


def train_game_by_chunks(
    model: MultiGameModel,
    optimizer: torch.optim.Optimizer,
    sequence: GameSequence,
    config: TrainingConfig,
    *,
    device: torch.device,
    history_keep: bool = True,
) -> dict[str, float | int]:
    """``update_mode="chunk"``: one clipped optimizer step per TBPTT chunk.

    Losses are the model's per-chunk means, so the returned ``total``/``action``
    /... are sums over chunks of chunk means (divide by ``updates``).
    """
    stats = _empty_update_stats()
    memory = model.initial_memory(1, device=device)
    history_kwargs = _history_loss_kwargs(history_keep, device=device)
    bounds = _chunk_bounds(len(sequence), config.chunk_steps)
    selected = _game_auxiliary_indices(len(sequence), config, device) if config.auxiliaries_enabled else [None] * len(bounds)
    total_selected = sum(len(indices) for indices in selected if indices is not None)
    for (start, end), transition_indices in zip(bounds, selected):
        chunk = slice_game_sequence(sequence, start, end)
        batch = _to_device(collate_game_sequences((chunk,)), device)
        # Chunk updates intentionally retain chunk-mean policy loss. Scale auxiliary
        # means to the game sample mass so tiny tails are not overweighted.
        auxiliary_scale = len(bounds) * len(transition_indices) / total_selected if total_selected else 0.0
        weights = replace(config.loss, next_frame=config.loss.next_frame * auxiliary_scale,
                          events=config.loss.events * auxiliary_scale)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_multigame_loss(
            model,
            batch,
            weights=weights,
            transition_indices=transition_indices,
            initial_memory=memory,
            **history_kwargs,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        assert loss.final_memory is not None
        memory = loss.final_memory.detach()
        for key, value in (
            ("total", loss.total), ("action", loss.action), ("click", loss.click),
            ("frame", loss.next_frame * auxiliary_scale), ("events", loss.events * auxiliary_scale),
        ):
            stats[key] += float(value.detach())
        # ``loss.click`` is a per-target mean inside one chunk; weight it by its
        # click-target count so the epoch figure is per target rather than a
        # mean over chunks that may have no clicks.
        stats["click_nll_sum"] += float(loss.click.detach()) * loss.click_targets
        stats["click_targets"] += loss.click_targets
        stats["click_exact_correct"] += loss.click_exact_correct
        stats["click_region_correct"] += loss.click_region_correct
        stats["chunks"] += 1
        stats["updates"] += 1
    return stats


def _whole_game_auxiliary_count(chunk_count: int, chunk_rows: int, game_rows: int) -> int:
    """Whole-game auxiliary target count from one chunk's count and row total.

    The model counts auxiliary targets at a fixed multiplicity per sampled
    transition row (e.g. pixels per frame or flags per event), so the game
    total is that multiplicity times the pre-sampled row total.
    """
    if chunk_rows == 0:
        return 0
    if chunk_count % chunk_rows:
        raise RuntimeError("auxiliary target count is not a fixed multiple of sampled rows")
    return (chunk_count // chunk_rows) * game_rows


def accumulate_game_gradients(
    model: MultiGameModel,
    sequence: GameSequence,
    config: TrainingConfig,
    *,
    device: torch.device,
    history_keep: bool = True,
) -> dict[str, float | int]:
    """``update_mode="game"``: backward the game-normalised loss over its chunks.

    Every chunk is scored with ``reduction="sum"``; each term is divided by the
    game's whole target count for that term (valid action targets, click
    targets, sampled auxiliary rows) so the gradient equals that of one
    per-target mean over the entire game, independent of how the game was
    partitioned into chunks (modulo the detached memory between chunks).  The
    caller zeroes gradients before and clips/steps after.  Returned losses are
    the game's per-target means (``updates`` is 1).
    """
    assert sequence.target_valid is not None and sequence.target_action_id is not None
    bounds = _chunk_bounds(len(sequence), config.chunk_steps)
    valid = np.asarray(sequence.target_valid).astype(bool)
    action_targets = int(valid.sum())
    click_targets = int((valid & (np.asarray(sequence.target_action_id) == CLICK_ACTION)).sum())
    if config.auxiliaries_enabled:
        transition_indices: list[Tensor | None] = _game_auxiliary_indices(len(sequence), config, device)
        game_rows = sum(len(indices) for indices in transition_indices if indices is not None)
    else:
        transition_indices = [None] * len(bounds)
        game_rows = 0
    history_kwargs = _history_loss_kwargs(history_keep, device=device)
    memory = model.initial_memory(1, device=device)
    stats = _empty_update_stats()
    sums = {"action": 0.0, "click": 0.0, "frame": 0.0, "events": 0.0}
    frame_total = event_total = 0
    frame_seen = event_seen = 0
    for (start, end), indices in zip(bounds, transition_indices):
        chunk = slice_game_sequence(sequence, start, end)
        batch = _to_device(collate_game_sequences((chunk,)), device)
        loss = compute_multigame_loss(
            model,
            batch,
            weights=config.loss,
            transition_indices=indices,
            initial_memory=memory,
            reduction="sum",
            **history_kwargs,
        )
        scaled = (
            config.loss.action * loss.action / max(action_targets, 1)
            + config.loss.click * loss.click / max(click_targets, 1)
        )
        if config.auxiliaries_enabled and indices is not None and len(indices):
            frame_norm = _whole_game_auxiliary_count(loss.frame_targets, len(indices), game_rows)
            event_norm = _whole_game_auxiliary_count(loss.event_targets, len(indices), game_rows)
            if frame_total and (frame_norm, event_norm) != (frame_total, event_total):
                raise RuntimeError("auxiliary target multiplicity changed between chunks")
            frame_total, event_total = frame_norm, event_norm
            frame_seen += int(loss.frame_targets)
            event_seen += int(loss.event_targets)
            scaled = (
                scaled
                + config.loss.next_frame * loss.next_frame / max(frame_norm, 1)
                + config.loss.events * loss.events / max(event_norm, 1)
            )
            sums["frame"] += float(loss.next_frame.detach())
            sums["events"] += float(loss.events.detach())
        scaled.backward()
        assert loss.final_memory is not None
        memory = loss.final_memory.detach()
        sums["action"] += float(loss.action.detach())
        sums["click"] += float(loss.click.detach())
        stats["click_targets"] += loss.click_targets
        stats["click_exact_correct"] += loss.click_exact_correct
        stats["click_region_correct"] += loss.click_region_correct
        stats["chunks"] += 1
    if config.auxiliaries_enabled and (frame_seen, event_seen) != (frame_total, event_total):
        raise RuntimeError("auxiliary target counts do not sum to the whole-game total")
    if stats["click_targets"] != click_targets:
        raise RuntimeError("model click-target count disagrees with the game's click rows")
    stats["action"] = sums["action"] / max(action_targets, 1)
    stats["click"] = sums["click"] / max(click_targets, 1)
    stats["frame"] = sums["frame"] / max(frame_total, 1)
    stats["events"] = sums["events"] / max(event_total, 1)
    stats["total"] = (
        config.loss.action * stats["action"] + config.loss.click * stats["click"]
        + config.loss.next_frame * stats["frame"] + config.loss.events * stats["events"]
    )
    stats["click_nll_sum"] = sums["click"]
    stats["updates"] = 1
    return stats


def train_game(
    model: MultiGameModel,
    optimizer: torch.optim.Optimizer,
    sequence: GameSequence,
    config: TrainingConfig,
    *,
    device: torch.device,
    history_keep: bool = True,
) -> dict[str, float | int]:
    """Apply ``config.update_mode`` to one whole training game."""
    if config.update_mode == "game":
        optimizer.zero_grad(set_to_none=True)
        stats = accumulate_game_gradients(
            model, sequence, config, device=device, history_keep=history_keep,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        return stats
    return train_game_by_chunks(
        model, optimizer, sequence, config, device=device, history_keep=history_keep,
    )


def _format_census(name: str, census: Mapping[str, int]) -> str:
    return (
        f"{name}: stored-region click rows {census['click_region_stored_rows']}"
        f"/{census['click_rows']} (multi-pixel {census['click_multi_pixel_region_rows']},"
        f" games with regions {census['games_with_click_regions']}/{census['games']})"
    )


def train_multigame(
    bundle: DatasetBundle,
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    resume: str | Path | None = None,
    initialize_from: str | Path | None = None,
    stop_request: StopRequest | None = None,
) -> TrainingResult:
    """Train with TBPTT state carry and generated closed-loop selection."""
    if resume is not None and initialize_from is not None:
        raise ValueError("resume and initialize_from are mutually exclusive")
    stop_request = stop_request or StopRequest()
    output_dir = Path(output_dir).resolve()
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    if resume is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"fresh training output directory is not empty: {output_dir}")
    else:
        resume_path = Path(resume).parent.resolve() / Path(resume).name
        if resume_path != latest_path:
            raise ValueError("resume must use latest.pt in the same output directory")
    # Report stored click-region coverage before any model work; the offline
    # metrics later split click rows the same way (stored region vs exact).
    click_census = {
        "train": click_region_census(bundle.train),
        "validation": click_region_census(bundle.validation),
    }
    print(
        "click regions | " + " | ".join(
            _format_census(name, census) for name, census in click_census.items()
        ),
        flush=True,
    )
    if config.require_click_regions and click_census["train"]["click_region_stored_rows"] == 0:
        raise ValueError(
            "require_click_regions: no training click row carries a stored click region "
            f"({click_census['train']['click_rows']} click rows all use the exact-pixel fallback)"
        )
    click_rows = click_census["train"]["click_rows"]
    click_coverage = click_census["train"]["click_region_stored_rows"] / click_rows if click_rows else 1.0
    if click_coverage < config.min_click_region_coverage:
        raise ValueError(
            f"min_click_region_coverage: training stored-region coverage {click_coverage:.6f} "
            f"is below required {config.min_click_region_coverage:.6f}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = MultiGameModel(config.model).to(device)
    _check_model_capabilities(model, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    start_epoch = 0
    global_step = 0
    best_score: tuple[Any, ...] | None = None
    logs: list[dict[str, Any]] = []
    progress = None
    ordered_identity = _ordered_game_identity(bundle)
    initialization = None
    if initialize_from is not None:
        initial = load_training_checkpoint(initialize_from, device=device)
        initialization = {"path": str(Path(initialize_from).resolve()), "sha256": sha256_file(initialize_from)}
        model.load_state_dict(initial["model_state"], strict=True)
    if resume is not None:
        saved = load_training_checkpoint(resume, device=device)
        if saved.get("training_recipe") != TRAINING_RECIPE:
            raise ValueError("resume training recipe is incompatible; use --initialize-from for a weights-only new run")
        _validate_checkpoint_layout(output_dir)
        if saved.get("ordered_game_identity") != ordered_identity:
            raise ValueError("resume ordered game identity changed")
        if saved["dataset_hashes"] != bundle.hashes or saved["scope"] != bundle.scope:
            raise ValueError("resume dataset hashes/scope do not match the audited inputs")
        if saved["training_signature"] != _training_signature(config):
            raise ValueError("resume training configuration changed")
        if saved.get("best_score") is not None:
            if not best_path.is_file():
                raise ValueError("resume output is missing its bound best.pt checkpoint")
            bound_best = load_training_checkpoint(best_path, device="cpu")
            if (
                bound_best.get("dataset_hashes") != saved["dataset_hashes"]
                or bound_best.get("scope") != saved["scope"]
                or bound_best.get("training_signature") != saved["training_signature"]
                or bound_best.get("best_score") != saved.get("best_score")
            ):
                raise ValueError("resume best.pt is not bound to latest.pt")
        elif best_path.exists():
            raise ValueError("resume has best.pt although latest.pt records no selected candidate")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        initialization = saved.get("initialize_from")
        progress = saved.get("epoch_progress")
        start_epoch = int(progress["epoch"]) if progress is not None else int(saved["epoch"]) + 1
        global_step = int(saved["global_step"])
        best_score = tuple(saved["best_score"]) if saved.get("best_score") is not None else None
        logs = list(saved.get("logs", ()))
        _restore_rng(saved["rng_state"])
    if start_epoch >= config.epochs:
        raise ValueError("resume checkpoint already reached requested epochs")

    if config.canonical_inputs:
        bundle = canonicalize_bundle(bundle)
    audit_summary = bundle.summary()
    validation_sources = {game.source_id for game in bundle.validation.games}
    if bundle.scope == "full_24_family_experiment":
        if validation_sources != set(M.TRAIN_SOURCE_IDS):
            raise ValueError("full generated validation has no usable game for every family")
        if config.closed_loop_games < len(M.TRAIN_SOURCE_IDS):
            raise ValueError("full checkpoint selection requires at least one panel game per family")
    def checkpoint(epoch: int, epoch_progress: dict[str, Any] | None, *, promoted: bool = False) -> None:
        payload = {
            "format": TRAINING_FORMAT,
            "training_recipe": TRAINING_RECIPE,
            "scope": bundle.scope,
            "smoke": bundle.smoke,
            "epoch": epoch,
            "epoch_progress": epoch_progress,
            "global_step": global_step,
            "model_config": asdict(model.config),
            "training_config": config.to_dict(),
            "training_signature": _training_signature(config),
            "canonical_inputs": bool(config.canonical_inputs),
            "load_time_variants": config.load_variants.to_dict(),
            "update_mode": config.update_mode,
            "loss_weights": asdict(config.loss),
            "history_mode": config.history_mode,
            "history_dropout": float(config.history_dropout),
            "click_region_census": click_census,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _rng_state(),
            "dataset_hashes": bundle.hashes,
            "ordered_game_identity": ordered_identity,
            "dataset_audit": audit_summary,
            "logs": logs,
            "best_score": best_score,
            "selection_basis": SELECTION_BASIS,
            "official_levels_used_for_selection": False,
            "frozen_for_official_evaluation": epoch_progress is None,
            "initialize_from": initialization,
        }
        _commit_checkpoints(output_dir, payload, promoted=promoted)
        # Derived convenience log; the committed checkpoint is authoritative.
        (output_dir / "training-log.json").write_text(json.dumps(logs, indent=2) + "\n")

    def result(*, stopped: bool = False) -> TrainingResult:
        return TrainingResult(best_path, latest_path, output_dir / "training-log.json",
                              tuple(logs), bundle.scope, stopped=stopped)

    if resume is None:
        checkpoint(-1, {"epoch": 0, "order": None, "cursor": 0,
                        "stats": _empty_update_stats(), "variants_applied": 0, "games_dropped": 0})
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_stats = dict(progress["stats"]) if progress else _empty_update_stats()
        order = np.asarray(progress["order"], dtype=int) if progress and progress["order"] is not None else np.random.permutation(len(bundle.train.games))
        cursor = int(progress["cursor"]) if progress else 0
        train_variants_applied = int(progress["variants_applied"]) if progress else 0
        train_games_dropped = int(progress["games_dropped"]) if progress else 0
        if sorted(order.tolist()) != list(range(len(bundle.train.games))) or not 0 <= cursor <= len(order):
            raise ValueError("resume epoch order/cursor is invalid")
        progress = None

        def game_boundary_checkpoint() -> None:
            checkpoint(epoch, {"epoch": epoch, "order": order.tolist(), "cursor": cursor,
                               "stats": epoch_stats, "variants_applied": train_variants_applied,
                               "games_dropped": train_games_dropped})

        for game_index in order[cursor:]:
            if stop_request.requested:
                game_boundary_checkpoint()
                print("training stopped at game boundary; resume from latest.pt", flush=True)
                return result(stopped=True)
            sequence, game_variant = augment_sequence_for_epoch(
                bundle.train.games[int(game_index)].sequence, config.load_variants,
                epoch=epoch, game_index=int(game_index),
            )
            train_variants_applied += int(not game_variant.is_identity)
            keep = history_keep_for_game(config, epoch=epoch, game_index=int(game_index))
            train_games_dropped += int(not keep)
            stats = train_game(
                model, optimizer, sequence, config, device=device, history_keep=keep,
            )
            _merge_update_stats(epoch_stats, stats)
            global_step += int(stats["updates"])
            cursor += 1
            if stop_request.requested or cursor % config.checkpoint_every_games == 0:
                game_boundary_checkpoint()
            if stop_request.requested:
                print("training stopped at game boundary; resume from latest.pt", flush=True)
                return result(stopped=True)
        if stop_request.requested:
            game_boundary_checkpoint()
            return result(stopped=True)
        # Preserve fitting progress before validation. If validation is interrupted,
        # replay it from the same RNG state without repeating any optimizer update.
        game_boundary_checkpoint()
        evaluation_rng = _rng_state()
        try:
            offline_games: Sequence[AuditedGame] = bundle.validation.games
            validation_variants_applied = 0
            if config.load_variants.enabled and config.load_variants.augment_validation:
                augmented = []
                for validation_index, audited in enumerate(bundle.validation.games):
                    sequence, game_variant = augment_sequence_for_epoch(
                        audited.sequence, config.load_variants,
                        epoch=epoch, game_index=validation_index,
                    )
                    validation_variants_applied += int(not game_variant.is_identity)
                    augmented.append(replace(audited, sequence=sequence))
                offline_games = tuple(augmented)
            offline = evaluate_generated_offline(
                model,
                offline_games,
                chunk_steps=config.chunk_steps,
                transition_limit_per_game=config.metric_transitions_per_game,
                device=device,
                history_free=config.history_mode == "none", stop_request=stop_request,
            )
            offline_history_free = None
            if config.history_free_diagnostic and config.history_mode == "none":
                offline_history_free = offline
            elif config.history_free_diagnostic:
                offline_history_free = evaluate_generated_offline(
                    model,
                    offline_games,
                    chunk_steps=config.chunk_steps,
                    transition_limit_per_game=config.metric_transitions_per_game,
                    device=device,
                    history_free=True, stop_request=stop_request,
                )
            closed_loop = None
            train_closed_loop = None
            closed_loop_history_free = None
            train_closed_loop_history_free = None
            promoted = False
            if (epoch + 1) % config.closed_loop_interval == 0 or epoch + 1 == config.epochs:
                panel_kwargs = dict(
                    max_actions_per_level=config.validation_max_actions_per_level,
                    max_game_actions=config.validation_max_game_actions,
                    device=device, stop_request=stop_request,
                )
                closed_loop = evaluate_generated_closed_loop(
                    model, bundle.validation.games, max_games=config.closed_loop_games,
                    history_free=config.history_mode == "none", **panel_kwargs,
                )
                if config.closed_loop_train_games > 0:
                    # Diagnostic only: can the policy replay games it was trained on?
                    train_closed_loop = evaluate_generated_closed_loop(
                        model, bundle.train.games, max_games=config.closed_loop_train_games,
                        history_free=config.history_mode == "none", **panel_kwargs,
                    )
                if config.history_free_diagnostic and config.history_mode == "none":
                    closed_loop_history_free = closed_loop
                    train_closed_loop_history_free = train_closed_loop
                elif config.history_free_diagnostic:
                    # Diagnostic only: same panels with previous-action history
                    # removed at inference.  Never enters the selection score.
                    closed_loop_history_free = evaluate_generated_closed_loop(
                        model, bundle.validation.games, max_games=config.closed_loop_games,
                        history_free=True, **panel_kwargs,
                    )
                    if config.closed_loop_train_games > 0:
                        train_closed_loop_history_free = evaluate_generated_closed_loop(
                            model, bundle.train.games, max_games=config.closed_loop_train_games,
                            history_free=True, **panel_kwargs,
                        )
                stop_request.check()
                score = selection_score(closed_loop, offline)
                if best_score is None or score > best_score:
                    best_score = score
                    promoted = True
        except _TrainingStopped:
            _restore_rng(evaluation_rng)
            game_boundary_checkpoint()
            print("training stopped during validation; resume replays validation from latest.pt", flush=True)
            return result(stopped=True)
        updates = int(epoch_stats["updates"])
        train_metrics: dict[str, Any] = {
            key: float(epoch_stats[key]) / max(updates, 1)
            for key in ("total", "action", "click", "frame", "events")
        }
        train_metrics["updates"] = updates
        train_metrics["chunks"] = int(epoch_stats["chunks"])
        train_metrics["update_mode"] = config.update_mode
        train_click_targets = int(epoch_stats["click_targets"])
        train_metrics["click_per_target"] = _ratio(
            float(epoch_stats["click_nll_sum"]), train_click_targets,
        )
        train_metrics["click_targets"] = train_click_targets
        train_metrics["click_exact_accuracy"] = _ratio(
            int(epoch_stats["click_exact_correct"]), train_click_targets,
        )
        train_metrics["click_region_accuracy"] = _ratio(
            int(epoch_stats["click_region_correct"]), train_click_targets,
        )
        train_games = len(bundle.train.games)
        train_metrics["history_dropped_games"] = train_games_dropped
        train_metrics["history_dropped_fraction"] = _ratio(train_games_dropped, train_games)
        log = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "generated_validation_offline": offline,
            "generated_validation_closed_loop": closed_loop,
            "generated_train_closed_loop": train_closed_loop,
            # History-free diagnostics: reported only, never used for selection.
            "generated_validation_offline_history_free": offline_history_free,
            "generated_validation_closed_loop_history_free": closed_loop_history_free,
            "generated_train_closed_loop_history_free": train_closed_loop_history_free,
            "promoted": promoted,
            "selection_basis": SELECTION_BASIS,
            "canonical_inputs": bool(config.canonical_inputs),
            "update_mode": config.update_mode,
            "loss_weights": asdict(config.loss),
            "auxiliaries_enabled": config.auxiliaries_enabled,
            "history": {
                "mode": config.history_mode,
                "dropout": float(config.history_dropout),
                "train_games_dropped": train_games_dropped,
                "train_games": train_games,
                "dropped_fraction": _ratio(train_games_dropped, train_games),
                "history_free_diagnostic": bool(config.history_free_diagnostic),
            },
            "click_region_census": click_census,
            "load_time_variants": {
                "options": config.load_variants.to_dict(),
                "train_games_transformed": train_variants_applied,
                "train_games": train_games,
                "validation_offline_games_transformed": validation_variants_applied,
                "validation_closed_loop_transformed": False,
                "closed_loop_variant": "identity" if config.canonical_inputs else "stored",
                "stored_variants_inverted": {
                    "train": audit_summary["train"]["stored_variants_inverted"],
                    "validation": audit_summary["validation"]["stored_variants_inverted"],
                },
            },
        }
        logs.append(log)
        print(format_epoch_line(log), flush=True)
        checkpoint(epoch, None, promoted=promoted)
        if stop_request.requested:
            return result(stopped=True)
    if not best_path.exists():
        raise RuntimeError("no generated closed-loop validation candidate was promoted")
    return result()


__all__ = [
    "AUDIT_FORMAT", "CLOSED_LOOP_OUTCOMES", "HISTORY_MODES", "SELECTION_BASIS",
    "TRAINING_FORMAT", "UPDATE_MODES",
    "AuditedGame", "DatasetBundle", "DatasetSide",
    "LoadTimeVariantOptions", "PuzzleIdentity",
    "TrainingConfig", "TrainingResult", "StopRequest", "graceful_training_signals",
    "TRAINING_RECIPE", "apply_variant_to_game", "audit_manifest_pair",
    "augment_sequence_for_epoch", "canonical_sequence", "canonicalize_bundle",
    "accumulate_game_gradients", "click_region_census",
    "canonicalize_game", "canonicalize_side", "classify_closed_loop_outcome",
    "compose_d4", "compose_variants",
    "evaluate_generated_closed_loop",
    "evaluate_generated_offline", "format_epoch_line", "game_legal_action_ids",
    "history_keep_for_game",
    "inverse_d4", "inverse_variant",
    "load_time_variant", "load_training_checkpoint", "model_from_training_checkpoint",
    "puzzle_identity", "selection_score", "sha256_file", "slice_game_sequence",
    "summarize_closed_loop_games", "train_game", "train_game_by_chunks", "train_multigame",
]
