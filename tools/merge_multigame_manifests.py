#!/usr/bin/env python3
"""Merge independently prepared multigame cohorts into one train/validation pair.

Several dataset preparations (for example ``teacher`` with six games per family
and ``teacher-p1`` .. ``teacher-p12`` with one game per family) can be unioned
into one strict-audit-compatible collection.  The merge:

* unions every train cohort into ``OUT/train`` and every validation cohort into
  ``OUT/validation`` (copying or hard-linking the per-game artifacts and
  renaming games whose ids collide);
* drops exact duplicate games within each side, keeping the first occurrence
  in CLI order.  A duplicate is the same trajectory (a digest of every public
  NPZ array) on the same whole-game level fingerprint; games that merely replay
  the same levels with a different trajectory (for example mixed/recovery games
  collected on a teacher game's levels) are all kept;
* drops validation games whose identity keys (whole-game identity, master seed,
  effective seeds, and every puzzle fingerprint kind the trainer's audit uses:
  geometry/gameplay/puzzle hashes, canonical specs, raw initial frames) collide
  with any retained train game, when ``--drop-validation-collisions`` is set;
* optionally (``--keep-validation FAMILY:N``) keeps up to N validation games of
  a family by sacrificing the train games they collide with instead, choosing
  the validation candidates that cost the fewest train games and never emptying
  a family's train set; every sacrificed train game is recorded;
* records every drop and rename with its reason in ``OUT/merge-report.json``;
* writes ``OUT/train/manifest.json`` and ``OUT/validation/manifest.json`` in the
  schema ``pebby.agent.multigame_training.audit_manifest_pair`` reads, marking
  ``is_full_experiment_collection`` only when every one of the 24 training
  families has at least one game on that side; and
* runs the strict (non-smoke) audit on the result and fails loudly otherwise.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby import multigame as M  # noqa: E402
from pebby.agent.multigame_training import (  # noqa: E402
    _declared_hash,
    _record_variant,
    _safe_path,
    audit_manifest_pair,
    puzzle_identity,
    sha256_file,
)
from pebby.multigame_dataset import _whole_game_fingerprint  # noqa: E402


MERGE_FORMAT = "pebby-multigame-manifest-merge-v1"
_ARTIFACT_LABELS = ("public_npz", "teacher_npz", "generated_specs")
_DUPLICATE_REASON = "duplicate game trajectory within side"
# Manifest blocks that must agree across every merged input because records
# bind to them (contract hashes, curricula) or the audit compares against them.
_MUST_MATCH = ("format", "requested_source_ids", "full_standard", "curricula_by_source")
# Audit overlap labels (see audit_manifest_pair) keyed by identity kind.
_AUDIT_LABELS = {
    "whole_game": "whole-game identities",
    "master_seed": "master seeds",
    "effective_seed": "effective seeds",
}


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class MergeError(ValueError):
    """Raised when inputs cannot be merged into an audit-compatible collection."""


class _Game:
    """One input game with its resolved artifacts and audit identity keys."""

    def __init__(
        self,
        *,
        manifest_index: int,
        manifest_path: Path,
        record: dict[str, Any],
        record_index: int,
        paths: dict[str, Path],
        declared_hashes: dict[str, str | None],
        whole_game: tuple[str, str],
        trajectory_sha256: str,
        identity_keys: frozenset[tuple[str, ...]],
    ) -> None:
        self.manifest_index = manifest_index
        self.manifest_path = manifest_path
        self.record = record
        self.record_index = record_index
        self.paths = paths
        self.declared_hashes = declared_hashes
        self.whole_game = whole_game
        self.trajectory_sha256 = trajectory_sha256
        self.identity_keys = identity_keys
        self.source_id = str(record["source_id"])
        self.slug = str(record["source"])

    @property
    def duplicate_key(self) -> tuple[str, str, str]:
        """Within-side duplicate identity: same levels *and* the same trajectory."""
        return (*self.whole_game, self.trajectory_sha256)

    @property
    def original_key(self) -> str:
        relative = self.record.get("record")
        if isinstance(relative, str) and relative:
            return Path(relative).stem
        return f"{self.slug}-{int(self.record.get('game_index', self.record_index)):06d}"

    @property
    def label(self) -> str:
        return f"{self.manifest_path}:{self.record.get('record', self.original_key)}"

    def describe(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "record": self.record.get("record"),
            "source_id": self.source_id,
            "master_seed": self.record.get("master_seed"),
            "game_index": self.record.get("game_index"),
            "status": self.record.get("status"),
            "whole_game_fingerprint": self.whole_game[1],
            "trajectory_sha256": self.trajectory_sha256,
        }


def _trajectory_sha256(public: Mapping[str, np.ndarray]) -> str:
    """Digest every public NPZ array (name, dtype, shape, bytes) in sorted key order.

    Hashing the arrays rather than the file makes the digest independent of
    NPZ container details while still distinguishing any difference in what
    the agent saw or did.
    """
    digest = hashlib.sha256()
    for name in sorted(public.keys()):
        array = np.ascontiguousarray(public[name])
        for part in (name, str(array.dtype), str(array.shape)):
            digest.update(part.encode())
            digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _identity_keys(record: Mapping[str, Any], identity) -> frozenset[tuple[str, ...]]:
    source_id = str(record["source_id"])
    keys: set[tuple[str, ...]] = {
        ("whole_game", source_id, str(record["master_seed"]), str(record["game_index"])),
        ("master_seed", source_id, str(record["master_seed"])),
    }
    keys.update(("effective_seed", source_id, str(seed)) for seed in identity.effective_seeds)
    keys.update((kind, source_id, digest) for kind, digest in identity.fingerprints)
    return frozenset(keys)


def _audit_label(kind: str) -> str:
    return _AUDIT_LABELS.get(kind, f"puzzle fingerprints[{kind}]")


def _load_side(
    manifest_paths: Sequence[Path], *, side: str,
) -> tuple[list[dict[str, Any]], list[_Game]]:
    manifests: list[dict[str, Any]] = []
    games: list[_Game] = []
    for manifest_index, given in enumerate(manifest_paths):
        manifest_path = Path(given).resolve()
        manifest = M.load_manifest(manifest_path)
        manifest["__path__"] = str(manifest_path)
        manifests.append(manifest)
        root = manifest_path.parent
        for record_index, original in enumerate(manifest.get("records", ())):
            record = copy.deepcopy(dict(original))
            key = f"{side}[{manifest_index}]:{record.get('record', record_index)}"
            for field in ("source_id", "source", "master_seed", "game_index", "status"):
                if record.get(field) is None:
                    raise MergeError(f"{key} lacks required record field {field!r}")
            missing = [label for label in _ARTIFACT_LABELS if not record.get(label)]
            if missing or int(record.get("steps", 0)) <= 0:
                raise MergeError(
                    f"{key} is a zero-step record or lacks artifacts {missing}; "
                    "merge only complete preparations"
                )
            paths = {label: _safe_path(root, record[label], f"{key}:{label}") for label in _ARTIFACT_LABELS}
            declared = {label: _declared_hash(manifest, record, label) for label in _ARTIFACT_LABELS}
            specs = json.loads(paths["generated_specs"].read_text())
            if not isinstance(specs, list) or not specs:
                raise MergeError(f"{key} generated specs must be a non-empty list")
            with np.load(paths["public_npz"], allow_pickle=False) as public:
                frames = public["frames"]
                boundaries = public["level_boundary"]
                trajectory_sha256 = _trajectory_sha256(public)
            # Historical manifests omitted these counters. Derive provenance from
            # the actual aligned rows in our new record; never edit source files.
            with np.load(paths["teacher_npz"], allow_pickle=False) as teacher:
                actual_counts = M.route_source_counts_from_teacher(teacher, int(record["steps"]))
            declared_counts = record.get("route_source_counts")
            if declared_counts and declared_counts != actual_counts:
                raise MergeError(f"{key} route_source_counts disagree with teacher NPZ")
            record["route_source_counts"] = actual_counts
            identity = puzzle_identity(
                specs,
                frames=frames,
                level_boundaries=boundaries,
                variant=_record_variant(record),
                label=key,
            )
            whole_game = _whole_game_fingerprint(
                str(record["source_id"]), specs, allow_canonical_fallback=True,
            )
            games.append(_Game(
                manifest_index=manifest_index,
                manifest_path=manifest_path,
                record=record,
                record_index=record_index,
                paths=paths,
                declared_hashes=declared,
                whole_game=whole_game,
                trajectory_sha256=trajectory_sha256,
                identity_keys=_identity_keys(record, identity),
            ))
    return manifests, games


def _dedupe_side(games: Sequence[_Game]) -> tuple[list[_Game], list[dict[str, Any]]]:
    kept: list[_Game] = []
    dropped: list[dict[str, Any]] = []
    first_seen: dict[tuple[str, str, str], _Game] = {}
    for game in games:
        earlier = first_seen.get(game.duplicate_key)
        if earlier is not None:
            dropped.append({
                "reason": _DUPLICATE_REASON,
                "game": game.describe(),
                "kept_duplicate_of": earlier.describe(),
            })
            continue
        first_seen[game.duplicate_key] = game
        kept.append(game)
    return kept, dropped


_SACRIFICE_REASON = "train game sacrificed to keep a validation game (--keep-validation)"


def parse_keep_validation(values: Sequence[str] | None) -> dict[str, int]:
    """Parse ``FAMILY:N`` items (slug or source id) into ``{source_id: N}``."""
    parsed: dict[str, int] = {}
    for item in values or ():
        family, separator, count = str(item).partition(":")
        if not separator or not count.isdigit():
            raise MergeError(f"--keep-validation expects FAMILY:N, got {item!r}")
        source = M.source_for(family)
        if source.held_out:
            raise MergeError(f"--keep-validation cannot target held-out {source.slug}")
        if source.source_id in parsed:
            raise MergeError(f"--keep-validation repeats family {source.slug}")
        parsed[source.source_id] = int(count)
    return parsed


def _rescue_validation(
    validation: Sequence[_Game], train: Sequence[_Game], keep: Mapping[str, int],
) -> tuple[list[_Game], list[dict[str, Any]], dict[str, Any]]:
    """Sacrifice colliding train games so listed families keep up to N validation games.

    Candidates are rescued greedily by the number of additional train games
    they cost given earlier sacrifices; a rescue is infeasible if it would leave
    the family without any train game.  Returns the reduced train list, the
    sacrifice drop entries, and a per-family report.
    """
    owners: dict[tuple[str, ...], list[_Game]] = {}
    for game in train:
        for key in game.identity_keys:
            owners.setdefault(key, []).append(game)

    def collisions(game: _Game) -> dict[str, _Game]:
        return {
            owner.label: owner for key in game.identity_keys for owner in owners.get(key, ())
        }

    sacrificed: dict[str, tuple[_Game, _Game]] = {}
    report: dict[str, Any] = {}
    for source_id, budget in keep.items():
        family_train = [game for game in train if game.source_id == source_id]
        candidates = [game for game in validation if game.source_id == source_id]
        costs = {game.label: collisions(game) for game in candidates}
        free = [game for game in candidates if not costs[game.label]]
        colliding = [game for game in candidates if costs[game.label]]
        kept = len(free)
        rescued: list[dict[str, Any]] = []
        while kept < budget and colliding:
            best: tuple[int, int, _Game] | None = None
            for index, game in enumerate(colliding):
                extra = [
                    owner for label, owner in costs[game.label].items() if label not in sacrificed
                ]
                remaining = len(family_train) - len(
                    {label for label in sacrificed if sacrificed[label][0].source_id == source_id}
                ) - len(extra)
                if remaining < 1:
                    continue
                if best is None or (len(extra), index) < (best[0], best[1]):
                    best = (len(extra), index, game)
            if best is None:
                break
            _, _, game = best
            extra = [
                owner for label, owner in costs[game.label].items() if label not in sacrificed
            ]
            for owner in extra:
                sacrificed[owner.label] = (owner, game)
            colliding.remove(game)
            kept += 1
            rescued.append({
                "game": game.describe(),
                "train_games_sacrificed": [owner.describe() for owner in extra],
                "cost": len(extra),
            })
        report[source_id] = {
            "requested": budget,
            "collision_free": len(free),
            "candidates_colliding": len(costs) - len(free),
            "rescued": rescued,
            "kept": kept,
            "shortfall": max(0, budget - kept),
        }
    dropped = [
        {
            "reason": _SACRIFICE_REASON,
            "game": owner.describe(),
            "kept_validation_game": rescued_game.describe(),
        }
        for owner, rescued_game in sacrificed.values()
    ]
    remaining = [game for game in train if game.label not in sacrificed]
    return remaining, dropped, report


def _drop_collisions(
    validation: Sequence[_Game], train: Sequence[_Game], *, drop: bool,
) -> tuple[list[_Game], list[dict[str, Any]]]:
    owners: dict[tuple[str, ...], _Game] = {}
    for game in train:
        for key in game.identity_keys:
            owners.setdefault(key, game)
    kept: list[_Game] = []
    dropped: list[dict[str, Any]] = []
    for game in validation:
        overlap = sorted(key for key in game.identity_keys if key in owners)
        if not overlap:
            kept.append(game)
            continue
        colliding_train = {owners[key].label: owners[key].describe() for key in overlap}
        entry = {
            "reason": "validation identity overlaps a train game",
            "game": game.describe(),
            "audit_checks_violated": sorted({_audit_label(key[0]) for key in overlap}),
            "overlap_sample": [list(key) for key in overlap[:5]],
            "overlap_count": len(overlap),
            "colliding_train_games": list(colliding_train.values()),
        }
        if not drop:
            raise MergeError(
                "validation game overlaps train identities; rerun with "
                f"--drop-validation-collisions to drop it: {json.dumps(entry, sort_keys=True)}"
            )
        dropped.append(entry)
    return kept, dropped


def _place_file(source: Path, destination: Path, *, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise MergeError(f"destination already exists: {destination}")
    if mode == "link":
        try:
            os.link(source, destination)
            return "link"
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
                raise
            shutil.copy2(source, destination)
            return "copy(link-fallback)"
    shutil.copy2(source, destination)
    return "copy"


def _assign_key(game: _Game, taken: set[str]) -> tuple[str, bool]:
    base = game.original_key
    if base not in taken:
        taken.add(base)
        return base, False
    suffix = 2
    while f"{base}-{suffix}" in taken:
        suffix += 1
    key = f"{base}-{suffix}"
    taken.add(key)
    return key, True


def _materialize_side(
    games: Sequence[_Game], root: Path, *, mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    renamed: list[dict[str, Any]] = []
    placements: Counter[str] = Counter()
    taken: set[str] = set()
    for game in games:
        key, was_renamed = _assign_key(game, taken)
        relative = {
            "public_npz": f"games/{key}.npz",
            "teacher_npz": f"teacher/{key}.npz",
            "generated_specs": f"teacher/{key}.levels.json",
            "record": f"records/{key}.json",
        }
        record = copy.deepcopy(game.record)
        hashes: dict[str, str] = {}
        for label in _ARTIFACT_LABELS:
            destination = root / relative[label]
            placements[_place_file(game.paths[label], destination, mode=mode)] += 1
            actual = sha256_file(destination)
            declared = game.declared_hashes[label]
            if declared is not None and declared != actual:
                raise MergeError(
                    f"{game.label}:{label} hash mismatch: declared {declared}, actual {actual}"
                )
            hashes[label] = actual
            record[label] = relative[label]
            record.pop(f"{label}_sha256", None)
        record["record"] = relative["record"]
        record["file_hashes"] = hashes
        record["merged_from"] = {
            "manifest": str(game.manifest_path),
            "record": game.record.get("record"),
            "manifest_sha256": sha256_file(game.manifest_path),
        }
        _atomic_json(root / relative["record"], record)
        records.append(record)
        if was_renamed:
            renamed.append({
                "game": game.describe(),
                "original_key": game.original_key,
                "merged_key": key,
            })
    return records, renamed, placements


def _require_matching(manifests: Sequence[dict[str, Any]], side: str) -> None:
    for field in _MUST_MATCH:
        digests = {_json_hash(manifest.get(field)) for manifest in manifests}
        if len(digests) != 1:
            paths = [manifest["__path__"] for manifest in manifests]
            raise MergeError(
                f"{side} manifests disagree on {field!r}; cannot merge {paths} into one "
                "collection"
            )


def _distinct(values: Sequence[Any]) -> list[Any]:
    seen: dict[str, Any] = {}
    for value in values:
        seen.setdefault(_json_hash(value), value)
    return list(seen.values())


def _build_manifest(
    manifests: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    side: str,
    input_report: list[dict[str, Any]],
    dropped: Sequence[dict[str, Any]],
    file_mode: str,
) -> dict[str, Any]:
    _require_matching(manifests, side)
    base = copy.deepcopy(manifests[0])
    base.pop("__path__", None)
    requested = list(base["requested_source_ids"])
    if set(requested) != set(M.TRAIN_SOURCE_IDS) or M.HELD_OUT_SOURCE_ID in requested:
        raise MergeError(f"{side} inputs do not request exactly the 24 training families")
    per_source = Counter(record["source_id"] for record in records)
    full_standard = base.get("full_standard") or {}
    full_standard_ready = bool(
        full_standard.get("required") is True and full_standard.get("ready") is True
        and set(full_standard.get("contract_hashes") or {}) == set(requested)
    )
    complete = all(per_source[source_id] > 0 for source_id in requested)
    is_full = bool(complete and full_standard_ready)

    seeds = _distinct([manifest.get("seed") for manifest in manifests])
    kinds = sorted({str(manifest.get("preparation_kind", "unknown")) for manifest in manifests})
    provenance = _distinct([manifest.get("provenance") for manifest in manifests])
    variant_requests = _distinct([manifest.get("variant_request") for manifest in manifests])

    def _merged_hash(field: str) -> str | None:
        values = [manifest.get(field) for manifest in manifests]
        distinct = sorted({str(value) for value in values if value is not None})
        if not distinct:
            return None
        return distinct[0] if len(distinct) == 1 else _json_hash(distinct)

    base.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "full_24_source_collection" if is_full
            else "incomplete_full_24_source_collection"
        ),
        "is_full_experiment_collection": is_full,
        "requested_source_ids": requested,
        "held_out_source_id": M.HELD_OUT_SOURCE_ID,
        "seed": seeds[0],
        "games_per_source": max(per_source.values(), default=0),
        "games_requested": sum(int(manifest.get("games_requested", 0)) for manifest in manifests),
        "levels_requested": sum(int(manifest.get("levels_requested", 0)) for manifest in manifests),
        "records": list(records),
        "preparation_complete": complete,
        "preparation_cohort": f"merged-{side}",
        "preparation_split": side,
        "preparation_kind": kinds[0] if len(kinds) == 1 else "mixed_kinds",
        "preparation_parameter_hash": _merged_hash("preparation_parameter_hash"),
        "preparation_provenance_hash": _merged_hash("preparation_provenance_hash"),
        "preparation_source_hash": _merged_hash("preparation_source_hash"),
        "provenance": provenance[0],
        "variant_request": variant_requests[0],
        "preparation_report": {
            "name": f"merged-{side}",
            "split": side,
            "kind": "merged",
            "accepted": len(records),
            "rejected_duplicate": sum(
                1 for item in dropped if item["reason"].startswith("duplicate")
            ),
            "rejected_overlap": sum(
                1 for item in dropped if item["reason"].startswith("validation identity")
            ),
            "sacrificed_for_validation": sum(
                1 for item in dropped if item["reason"] == _SACRIFICE_REASON
            ),
            "sources": {
                source_id: {"accepted": per_source[source_id]} for source_id in requested
            },
            "inputs": input_report,
        },
        "merge": {
            "format": MERGE_FORMAT,
            "tool": "tools/merge_multigame_manifests.py",
            "file_mode": file_mode,
            "input_manifests": [manifest["__path__"] for manifest in manifests],
            "input_manifest_hashes": {
                manifest["__path__"]: sha256_file(manifest["__path__"]) for manifest in manifests
            },
            "seeds": seeds,
            "provenance_variants": len(provenance),
            "provenance_hashes": sorted({_json_hash(value) for value in provenance}),
            "variant_requests": variant_requests,
            "preparation_kinds": kinds,
            "families_without_games": sorted(
                source_id for source_id in requested if per_source[source_id] == 0
            ),
            "dropped": list(dropped),
        },
    })
    M.update_manifest_summary(base)
    return base


def _input_report(manifests: Sequence[dict[str, Any]], games: Sequence[_Game], kept: Sequence[_Game]) -> list[dict[str, Any]]:
    loaded = Counter(game.manifest_index for game in games)
    retained = Counter(game.manifest_index for game in kept)
    return [
        {
            "manifest": manifest["__path__"],
            "manifest_sha256": sha256_file(manifest["__path__"]),
            "seed": manifest.get("seed"),
            "preparation_cohort": manifest.get("preparation_cohort"),
            "is_full_experiment_collection": bool(manifest.get("is_full_experiment_collection")),
            "games": loaded[index],
            "kept": retained[index],
        }
        for index, manifest in enumerate(manifests)
    ]


def merge_manifests(
    train_manifests: Sequence[str | Path],
    validation_manifests: Sequence[str | Path],
    out_root: str | Path,
    *,
    drop_validation_collisions: bool = False,
    file_mode: str = "copy",
    keep_validation: Mapping[str, int] | None = None,
    progress=print,
) -> dict[str, Any]:
    """Merge cohorts, write manifests and ``merge-report.json``, and strictly audit."""
    if file_mode not in ("copy", "link"):
        raise MergeError(f"unsupported file mode {file_mode!r}")
    if not train_manifests or not validation_manifests:
        raise MergeError("at least one train and one validation manifest are required")
    out_root = Path(out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise MergeError(f"output root is not empty: {out_root}")
    for path in (*train_manifests, *validation_manifests):
        try:
            Path(path).resolve().relative_to(out_root)
        except ValueError:
            continue
        raise MergeError(f"input manifest {path} lies inside the output root {out_root}")

    progress(f"loading {len(train_manifests)} train manifest(s)")
    train_manifests_loaded, train_games = _load_side([Path(p) for p in train_manifests], side="train")
    progress(f"loading {len(validation_manifests)} validation manifest(s)")
    validation_manifests_loaded, validation_games = _load_side(
        [Path(p) for p in validation_manifests], side="validation",
    )

    train_kept, train_dropped = _dedupe_side(train_games)
    validation_kept, validation_dropped = _dedupe_side(validation_games)
    keep_validation = dict(keep_validation or {})
    train_kept, sacrificed, keep_report = _rescue_validation(
        validation_kept, train_kept, keep_validation,
    )
    train_dropped.extend(sacrificed)
    validation_kept, collision_dropped = _drop_collisions(
        validation_kept, train_kept, drop=drop_validation_collisions,
    )
    validation_dropped.extend(collision_dropped)
    progress(
        f"train: {len(train_games)} loaded, {len(train_kept)} kept, {len(train_dropped)} dropped; "
        f"validation: {len(validation_games)} loaded, {len(validation_kept)} kept, "
        f"{len(validation_dropped)} dropped"
    )

    out_root.mkdir(parents=True, exist_ok=True)
    sides: dict[str, dict[str, Any]] = {}
    manifest_paths: dict[str, Path] = {}
    for side, manifests, games, kept, dropped in (
        ("train", train_manifests_loaded, train_games, train_kept, train_dropped),
        ("validation", validation_manifests_loaded, validation_games, validation_kept,
         validation_dropped),
    ):
        root = out_root / side
        records, renamed, placements = _materialize_side(kept, root, mode=file_mode)
        inputs = _input_report(manifests, games, kept)
        manifest = _build_manifest(
            manifests, records, side=side, input_report=inputs, dropped=dropped,
            file_mode=file_mode,
        )
        manifest_path = M.save_manifest(root / "manifest.json", manifest)
        manifest_paths[side] = manifest_path
        per_source = Counter(record["source_id"] for record in records)
        sides[side] = {
            "manifest": str(manifest_path),
            "inputs": inputs,
            "games_loaded": len(games),
            "games_kept": len(records),
            "games_per_source": {
                source_id: per_source[source_id] for source_id in manifest["requested_source_ids"]
            },
            "families_without_games": manifest["merge"]["families_without_games"],
            "is_full_experiment_collection": manifest["is_full_experiment_collection"],
            "dropped": dropped,
            "dropped_by_reason": dict(Counter(item["reason"] for item in dropped)),
            "renamed": renamed,
            "file_placements": dict(placements),
        }

    report: dict[str, Any] = {
        "format": MERGE_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "train_manifest": [str(Path(p).resolve()) for p in train_manifests],
            "validation_manifest": [str(Path(p).resolve()) for p in validation_manifests],
            "out_root": str(out_root),
            "drop_validation_collisions": bool(drop_validation_collisions),
            "keep_validation": keep_validation,
            "file_mode": file_mode,
        },
        "keep_validation": keep_report,
        "sides": sides,
        "audit": None,
        "audit_error": None,
    }
    report_path = out_root / "merge-report.json"
    _atomic_json(report_path, report)

    progress("running strict audit on merged manifests")
    try:
        bundle = audit_manifest_pair(
            [manifest_paths["train"]], [manifest_paths["validation"]], smoke=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        report["audit_error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(report_path, report)
        raise MergeError(f"merged manifests failed the strict audit: {exc}") from exc
    report["audit"] = bundle.summary()
    _atomic_json(report_path, report)
    progress(f"strict audit passed: {report_path}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--drop-validation-collisions", action="store_true",
        help="drop validation games whose identities overlap train games instead of failing",
    )
    parser.add_argument(
        "--keep-validation", nargs="+", metavar="FAMILY:N", default=None,
        help="keep up to N validation games of FAMILY by sacrificing colliding train games",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy", dest="file_mode", action="store_const", const="copy")
    mode.add_argument("--link", dest="file_mode", action="store_const", const="link",
                      help="hard-link per-game artifacts instead of copying them")
    parser.set_defaults(file_mode="copy")
    return parser.parse_args(argv)


def _print_summary(report: Mapping[str, Any]) -> None:
    sides = report["sides"]
    train, validation = sides["train"], sides["validation"]
    print("family          train  validation")
    for source_id in train["games_per_source"]:
        print(
            f"{source_id:<15} {train['games_per_source'][source_id]:>5}  "
            f"{validation['games_per_source'][source_id]:>10}"
        )
    for side_name, side in sides.items():
        print(
            f"{side_name}: loaded {side['games_loaded']}, kept {side['games_kept']}, "
            f"dropped {side['dropped_by_reason']}, renamed {len(side['renamed'])}, "
            f"full={side['is_full_experiment_collection']}"
        )
        if side["families_without_games"]:
            print(f"{side_name}: families without games: {side['families_without_games']}")
    for source_id, item in report["keep_validation"].items():
        print(
            f"keep-validation {source_id}: requested {item['requested']}, kept {item['kept']} "
            f"({item['collision_free']} collision-free, {len(item['rescued'])} rescued at "
            f"{sum(entry['cost'] for entry in item['rescued'])} train games), "
            f"shortfall {item['shortfall']}"
        )
    audit = report["audit"]
    print(
        f"audit: scope={audit['scope']} train_games={audit['train']['games_used']} "
        f"validation_games={audit['validation']['games_used']}"
    )


def main(argv=None) -> dict[str, Any]:
    args = parse_args(argv)
    report = merge_manifests(
        args.train_manifest,
        args.validation_manifest,
        args.out_root,
        drop_validation_collisions=args.drop_validation_collisions,
        file_mode=args.file_mode,
        keep_validation=parse_keep_validation(args.keep_validation),
    )
    _print_summary(report)
    return report


def cli(argv=None) -> int:
    try:
        main(argv)
    except (OSError, ValueError) as exc:
        print(f"merge rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
