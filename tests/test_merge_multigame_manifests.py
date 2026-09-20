"""Merging independent multigame preparations into one strict-audit collection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pebby import multigame as M
from pebby.agent import multigame_training as T
from pebby.multigame_dataset import DatasetPreparationConfig, prepare_multigame_dataset
from tests.test_multigame_dataset import _modules, _ScriptedCollector
from tools import merge_multigame_manifests as merge_cli
from tools import train_multigame as train_cli


_DUPLICATE_REASON = merge_cli._DUPLICATE_REASON
_COLLISION_REASON = "validation identity overlaps a train game"


class _RemappedCollector(_ScriptedCollector):
    """Reproduce specific games of another preparation to force merge conflicts.

    ``replay`` entries reproduce the other game's levels (specs and initial
    frame) but play them differently, like a mixed/recovery game collected on
    a teacher game's levels: the later frame and the action differ.
    """

    def __init__(
        self,
        remap: dict[tuple[str, int, int], int],
        *,
        replay: dict[tuple[str, int, int], int] | None = None,
    ) -> None:
        super().__init__()
        self.remap = remap
        self.replay = replay or {}

    def __call__(self, package, **kwargs):
        slug = M.source_for(package.source).slug
        key = (slug, kwargs["master_seed"], kwargs["game_index"])
        target = self.remap.get(key)
        if target is not None:
            kwargs = dict(kwargs, master_seed=target)
        replayed = self.replay.get(key)
        if replayed is not None:
            kwargs = dict(kwargs, master_seed=replayed)
        collected = super().__call__(package, **kwargs)
        if replayed is not None:
            # Same initial frame and specs (so every audit identity key matches),
            # but the frame observed after the action differs.
            assert collected.public is not None
            frames = collected.public["frames"]
            frames[1, 8, 9] = (int(frames[1, 8, 9]) + 3) % 16
            frames[1, 20:24, 20:24] = 5
        return collected


def _prepare(root: Path, *, train_seed: int, validation_seed: int, collector) -> Path:
    result = prepare_multigame_dataset(
        DatasetPreparationConfig(
            output_root=root,
            train_master_seed=train_seed,
            validation_master_seed=validation_seed,
            completed_teacher_games_per_family=1,
            mixed_games_per_family=0,
            limits=M.SearchLimits(2, 2),
            outer_generation_attempts=1,
            generator_attempts=1,
            max_candidate_attempts=1,
            max_game_steps=2,
        ),
        modules=[_modules(source, full_standard=True) for source in M.TRAIN_SOURCES],
        collect_fn=collector,
        progress=lambda message: None,
    )
    assert result.complete
    return result.output_root


@pytest.fixture(scope="module")
def preparations(tmp_path_factory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("preparations")
    first = _prepare(
        base / "a", train_seed=11, validation_seed=12, collector=_ScriptedCollector(),
    )
    second = _prepare(
        base / "b",
        train_seed=21,
        validation_seed=22,
        collector=_RemappedCollector({
            # B's cd82 validation game is A's cd82 train game: a cross-side collision.
            ("cd82", 22, 0): 11,
            # B's ar25 train game is A's ar25 train game: an exact within-side duplicate.
            ("ar25", 21, 0): 11,
        }),
    )
    return first, second


@pytest.fixture(scope="module")
def replayed_preparation(tmp_path_factory, preparations) -> Path:
    """A cohort whose ar25 train game replays A's ar25 train levels with another trajectory."""
    return _prepare(
        tmp_path_factory.mktemp("preparations") / "c",
        train_seed=31,
        validation_seed=32,
        collector=_RemappedCollector({}, replay={("ar25", 31, 0): 11}),
    )


def _manifests(preparations: tuple[Path, ...]) -> tuple[list[str], list[str]]:
    train = [str(root / "train-teacher" / "manifest.json") for root in preparations]
    validation = [str(root / "validation-teacher" / "manifest.json") for root in preparations]
    return train, validation


def test_cross_run_collision_fails_without_drop_flag_and_leaves_no_output(tmp_path, preparations):
    train, validation = _manifests(preparations)
    out = tmp_path / "merged"
    with pytest.raises(merge_cli.MergeError, match="--drop-validation-collisions"):
        merge_cli.merge_manifests(train, validation, out)
    assert not out.exists()


def test_merge_drops_duplicates_and_collisions_then_passes_strict_audit_and_trains(
    tmp_path, preparations, monkeypatch,
):
    train, validation = _manifests(preparations)
    out = tmp_path / "merged"
    report = merge_cli.main([
        "--train-manifest", *train,
        "--validation-manifest", *validation,
        "--out-root", str(out),
        "--drop-validation-collisions",
        "--link",
    ])
    written = json.loads((out / "merge-report.json").read_text())
    assert written["sides"] == report["sides"]

    cd82 = M.source_for("cd82").source_id
    ar25 = M.source_for("ar25").source_id
    train_side, validation_side = report["sides"]["train"], report["sides"]["validation"]

    assert train_side["games_loaded"] == 48 and train_side["games_kept"] == 47
    assert train_side["dropped_by_reason"] == {_DUPLICATE_REASON: 1}
    [duplicate] = train_side["dropped"]
    assert duplicate["game"]["source_id"] == ar25
    assert duplicate["game"]["manifest"] == train[1]
    assert duplicate["kept_duplicate_of"]["manifest"] == train[0]

    assert validation_side["games_loaded"] == 48 and validation_side["games_kept"] == 47
    assert validation_side["dropped_by_reason"] == {_COLLISION_REASON: 1}
    [collision] = validation_side["dropped"]
    assert collision["game"]["source_id"] == cd82
    assert collision["game"]["manifest"] == validation[1]
    assert [item["manifest"] for item in collision["colliding_train_games"]] == [train[0]]
    kinds = {key[0] for key in collision["overlap_sample"]}
    assert kinds & {"raw_initial_frame", "gameplay", "geometry_d4", "effective_seed"}
    assert "puzzle fingerprints[raw_initial_frame]" in collision["audit_checks_violated"]

    for side, counts in (
        ("train", {source_id: (1 if source_id == ar25 else 2) for source_id in M.TRAIN_SOURCE_IDS}),
        ("validation", {source_id: (1 if source_id == cd82 else 2) for source_id in M.TRAIN_SOURCE_IDS}),
    ):
        assert report["sides"][side]["games_per_source"] == counts
        assert report["sides"][side]["families_without_games"] == []
        assert report["sides"][side]["is_full_experiment_collection"] is True
        assert len(report["sides"][side]["renamed"]) == 23
        manifest = M.load_manifest(out / side / "manifest.json")
        assert manifest["is_full_experiment_collection"] is True
        assert manifest["scope"] == "full_24_source_collection"
        assert set(manifest["requested_source_ids"]) == set(M.TRAIN_SOURCE_IDS)
        assert manifest["games_recorded"] == len(manifest["records"]) == sum(counts.values())
        assert manifest["merge"]["format"] == merge_cli.MERGE_FORMAT
        assert manifest["merge"]["input_manifests"] == (train if side == "train" else validation)
        keys = [Path(record["record"]).stem for record in manifest["records"]]
        assert len(keys) == len(set(keys))
        for record in manifest["records"]:
            for label in ("public_npz", "teacher_npz", "generated_specs"):
                path = out / side / record[label]
                assert path.is_file()
                assert os.stat(path).st_nlink >= 2  # --link hard-links artifacts
                assert T.sha256_file(path) == record["file_hashes"][label]
            private = json.loads((out / side / record["record"]).read_text())
            assert private == record
            assert Path(private["merged_from"]["manifest"]).is_file()

    assert report["audit_error"] is None
    assert report["audit"]["scope"] == "full_24_family_experiment"
    assert report["audit"]["train"]["games_used"] == 47
    assert report["audit"]["validation"]["games_used"] == 47
    bundle = T.audit_manifest_pair(
        [out / "train" / "manifest.json"], [out / "validation" / "manifest.json"], smoke=False,
    )
    assert bundle.scope == "full_24_family_experiment"

    calls = []

    def fake_closed(model, games, **kwargs):
        calls.append(len(games))
        return {
            "games_won": 0, "levels_completed": 0, "games": [{"failure": None}],
            "panel_records": ["fake"], "actions": 1,
        }

    monkeypatch.setattr(T, "evaluate_generated_closed_loop", fake_closed)
    run = tmp_path / "run"
    result = train_cli.main([
        "--train-manifest", str(out / "train" / "manifest.json"),
        "--validation-manifest", str(out / "validation" / "manifest.json"),
        "--out-dir", str(run), "--smoke", "--cpu-test-model", "--device", "cpu",
        "--epochs", "1", "--chunk-steps", "2", "--auxiliary-transitions-per-chunk", "1",
        "--metric-transitions-per-game", "1", "--closed-loop-games", "1",
        "--closed-loop-train-games", "0",
    ])
    # One closed-loop pass over the merged validation panel; the training-game
    # diagnostic panel is disabled above so it doesn't add a second call.
    assert len(calls) == 1 and calls[0] == 47
    logs = json.loads((run / "training-log.json").read_text())
    assert len(logs) == 1 and logs[-1]["global_step"] >= 1
    assert Path(result.latest_checkpoint).is_file()
    audit = json.loads((run / "dataset-audit.json").read_text())
    assert audit["train"]["games_used"] == 47 and audit["validation"]["games_used"] == 47


def test_same_levels_with_different_trajectory_are_kept_but_identical_games_are_deduped(
    tmp_path, preparations, replayed_preparation,
):
    first, second = preparations
    train, validation = _manifests((first, second, replayed_preparation))
    ar25 = M.source_for("ar25").source_id
    out = tmp_path / "merged"
    report = merge_cli.main([
        "--train-manifest", *train,
        "--validation-manifest", *validation,
        "--out-root", str(out),
        "--drop-validation-collisions",
        "--link",
    ])
    train_side = report["sides"]["train"]

    # C's ar25 train game shares A's ar25 levels (same whole-game fingerprint and
    # every audit identity key) but played a different trajectory: it is kept.
    # B's ar25 train game is byte-for-byte the same trajectory on the same levels:
    # it is still de-duplicated against A's.
    assert train_side["games_loaded"] == 72 and train_side["games_kept"] == 71
    assert train_side["dropped_by_reason"] == {_DUPLICATE_REASON: 1}
    [duplicate] = train_side["dropped"]
    assert duplicate["game"]["manifest"] == train[1]
    assert duplicate["kept_duplicate_of"]["manifest"] == train[0]
    assert duplicate["game"]["trajectory_sha256"] == duplicate["kept_duplicate_of"]["trajectory_sha256"]
    assert train_side["games_per_source"][ar25] == 2

    kept_ar25 = [
        record for record in M.load_manifest(out / "train" / "manifest.json")["records"]
        if record["source_id"] == ar25
    ]
    assert sorted(Path(record["merged_from"]["manifest"]).as_posix() for record in kept_ar25) == sorted(
        [train[0], train[2]]
    )
    a_game, c_game = sorted(kept_ar25, key=lambda record: train.index(record["merged_from"]["manifest"]))
    assert a_game["file_hashes"]["generated_specs"] == c_game["file_hashes"]["generated_specs"]
    assert a_game["file_hashes"]["public_npz"] != c_game["file_hashes"]["public_npz"]
    _, games = merge_cli._load_side([Path(train[0]), Path(train[2])], side="train")
    a_loaded, c_loaded = (game for game in games if game.source_id == ar25)
    assert a_loaded.whole_game == c_loaded.whole_game
    assert a_loaded.identity_keys == c_loaded.identity_keys
    assert a_loaded.trajectory_sha256 != c_loaded.trajectory_sha256
    assert a_loaded.duplicate_key != c_loaded.duplicate_key

    # The cross-side guard is unchanged: the replayed levels are still A's, so a
    # validation game on them would still collide (here nothing else collides
    # beyond B's cd82 game from the shared fixture).
    validation_side = report["sides"]["validation"]
    assert validation_side["games_loaded"] == 72 and validation_side["games_kept"] == 71
    assert validation_side["dropped_by_reason"] == {_COLLISION_REASON: 1}
    assert report["audit_error"] is None
    assert report["audit"]["train"]["games_used"] == 71
    assert report["audit"]["validation"]["games_used"] == 71


def test_keep_validation_sacrifices_cheapest_colliding_train_games(tmp_path, preparations):
    train, validation = _manifests(preparations)
    cd82 = M.source_for("cd82").source_id
    ar25 = M.source_for("ar25").source_id
    with pytest.raises(merge_cli.MergeError, match="FAMILY:N"):
        merge_cli.parse_keep_validation(["cd82"])
    assert merge_cli.parse_keep_validation(["cd82:2", ar25 + ":5"]) == {cd82: 2, ar25: 5}

    out = tmp_path / "merged"
    report = merge_cli.main([
        "--train-manifest", *train,
        "--validation-manifest", *validation,
        "--out-root", str(out),
        "--drop-validation-collisions",
        "--keep-validation", "cd82:2", "ar25:5",
        "--link",
    ])
    train_side, validation_side = report["sides"]["train"], report["sides"]["validation"]

    # B's cd82 validation game is rescued by sacrificing A's colliding cd82 train game;
    # B's own cd82 train game keeps the family populated on the train side.
    cd82_report = report["keep_validation"][cd82]
    assert {key: cd82_report[key] for key in ("requested", "collision_free", "candidates_colliding", "kept", "shortfall")} == {
        "requested": 2, "collision_free": 1, "candidates_colliding": 1, "kept": 2, "shortfall": 0,
    }
    [rescue] = cd82_report["rescued"]
    assert rescue["cost"] == 1
    assert rescue["game"]["manifest"] == validation[1] and rescue["game"]["source_id"] == cd82
    [sacrificed] = rescue["train_games_sacrificed"]
    assert sacrificed["manifest"] == train[0] and sacrificed["source_id"] == cd82
    # ar25 has two collision-free validation games and nothing to rescue: honest shortfall.
    assert report["keep_validation"][ar25] == {
        "requested": 5, "collision_free": 2, "candidates_colliding": 0, "rescued": [],
        "kept": 2, "shortfall": 3,
    }

    assert train_side["dropped_by_reason"] == {
        _DUPLICATE_REASON: 1, merge_cli._SACRIFICE_REASON: 1,
    }
    [sacrifice_entry] = [
        item for item in train_side["dropped"] if item["reason"] == merge_cli._SACRIFICE_REASON
    ]
    assert sacrifice_entry["game"] == sacrificed
    assert sacrifice_entry["kept_validation_game"] == rescue["game"]
    assert validation_side["dropped"] == []
    assert train_side["games_per_source"][cd82] == 1
    assert train_side["games_per_source"][ar25] == 1
    assert validation_side["games_per_source"][cd82] == 2
    assert train_side["games_kept"] == 46 and validation_side["games_kept"] == 48
    for side in ("train", "validation"):
        assert report["sides"][side]["is_full_experiment_collection"] is True
        assert M.load_manifest(out / side / "manifest.json")["is_full_experiment_collection"]
    assert report["audit_error"] is None
    assert report["audit"]["scope"] == "full_24_family_experiment"
    assert report["audit"]["train"]["games_used"] == 46
    assert report["audit"]["validation"]["games_used"] == 48
