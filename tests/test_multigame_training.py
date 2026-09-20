"""Focused audit, TBPTT, metric, and resume tests for whole-game training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pebby import multigame as M
from pebby.agent.multigame_model import (
    ACTION_COUNT,
    FRAME_SIZE,
    LossWeights,
    MultiGameModel,
    MultiGameModelConfig,
    collate_game_sequences,
    compute_multigame_loss,
)
from pebby.agent import multigame_training as T


def _write_manifest(
    root: Path,
    name: str,
    *,
    source_slug: str = "cd82",
    master_seed: int = 1,
    game_index: int = 0,
    effective_seed: int = 101,
    puzzle: str = "a",
    status: str = "won",
    action_sources: tuple[int, ...] = (1, 0),
    declared_hash: str | None = None,
) -> Path:
    folder = root / name
    (folder / "games").mkdir(parents=True)
    (folder / "teacher").mkdir()
    source = M.source_for(source_slug)
    steps = len(action_sources)
    actions = np.asarray(([1, 6] * ((steps + 1) // 2))[:steps], dtype=np.int8)
    x = np.full(steps, -1, dtype=np.int16)
    y = np.full(steps, -1, dtype=np.int16)
    x[actions == M.CLICK_ACTION] = 61
    y[actions == M.CLICK_ACTION] = 43
    frames = np.zeros((steps + 1, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    color = (sum(puzzle.encode()) % 14) + 1
    frames[:, 3:6, 5:8] = color
    for index in range(1, steps + 1):
        frames[index, index, index] = (color + index) % 16
    legal = np.zeros((steps + 1, ACTION_COUNT), dtype=np.bool_)
    legal[:-1, 1:] = True
    won = status == "won"
    boundary = np.zeros(steps, dtype=np.bool_)
    if won:
        boundary[-1] = True
    terminal = np.zeros(steps + 1, dtype=np.bool_)
    terminal[-1] = won
    public_path = folder / "games" / "game.npz"
    np.savez(
        public_path,
        frames=frames,
        legal_action_mask=legal,
        state=np.asarray(["NOT_FINISHED"] * steps + (["WIN"] if won else ["NOT_FINISHED"])),
        level_index=np.zeros(steps + 1, dtype=np.int16),
        levels_completed=np.concatenate((np.zeros(steps, dtype=np.int16), [int(won)])),
        terminal=terminal,
        won=terminal.copy(),
        action_id=actions,
        action_x=x,
        action_y=y,
        level_boundary=boundary,
    )
    teacher_path = folder / "teacher" / "game.npz"
    np.savez(
        teacher_path,
        target_action_id=actions.copy(),
        target_action_x=x.copy(),
        target_action_y=y.copy(),
        source=np.asarray(action_sources, dtype=np.int8),
    )
    specs = [{
        "generator_version": 1,
        "effective_seed": effective_seed,
        "layout": {"puzzle_token": puzzle},
    }]
    specs_path = folder / "teacher" / "game.levels.json"
    specs_path.write_text(json.dumps(specs))
    record = {
        "format": M.FORMAT,
        "source": source.slug,
        "source_id": source.source_id,
        "master_seed": master_seed,
        "game_index": game_index,
        "status": status,
        "steps": steps,
        "public_npz": "games/game.npz",
        "teacher_npz": "teacher/game.npz",
        "generated_specs": "teacher/game.levels.json",
        "record": "records/game.json",
        "levels": [{"generator_version": 1}],
    }
    if declared_hash is not None:
        record["public_npz_sha256"] = declared_hash
    manifest = {
        "format": M.MANIFEST_FORMAT,
        "requested_source_ids": [source.source_id],
        "held_out_source_id": M.HELD_OUT_SOURCE_ID,
        "is_full_experiment_collection": False,
        "records": [record],
    }
    manifest_path = folder / "manifest.json"
    M.save_manifest(manifest_path, manifest)
    return manifest_path


def _pair(tmp_path: Path):
    train = _write_manifest(
        tmp_path, "train", master_seed=11, effective_seed=111, puzzle="train",
    )
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=222, puzzle="validation",
    )
    return train, validation


def test_strict_audit_requires_full_scope_but_smoke_reports_partial_and_random_rows(tmp_path):
    train, validation = _pair(tmp_path)
    with pytest.raises(ValueError, match="full 24-family"):
        T.audit_manifest_pair(train, validation)
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    assert bundle.scope == "smoke"
    assert bundle.train.rows == 2
    assert bundle.train.random_rows == 1
    assert bundle.train.teacher_rows == 1
    assert bundle.train.summary()["completed_games_by_family"][M.source_for("cd82").source_id] == 1


def test_strict_audit_rejects_legacy_all24_flag_without_full_standard_contracts(tmp_path):
    train, validation = _pair(tmp_path)
    for path in (train, validation):
        manifest = json.loads(path.read_text())
        manifest["requested_source_ids"] = list(M.TRAIN_SOURCE_IDS)
        manifest["is_full_experiment_collection"] = True
        M.save_manifest(path, manifest)
    with pytest.raises(ValueError, match="full-standard readiness"):
        T.audit_manifest_pair(train, validation)


def test_audit_rejects_hash_and_every_cross_split_puzzle_identity(tmp_path):
    bad_train = _write_manifest(
        tmp_path, "bad", master_seed=1, effective_seed=1, puzzle="bad", declared_hash="0" * 64,
    )
    validation = _write_manifest(
        tmp_path, "val", master_seed=2, effective_seed=2, puzzle="other",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        T.audit_manifest_pair(bad_train, validation, smoke=True)

    cases = (
        ({"master_seed": 7, "game_index": 0, "effective_seed": 10, "puzzle": "left"},
         {"master_seed": 7, "game_index": 1, "effective_seed": 20, "puzzle": "right"}, "master seeds"),
        ({"master_seed": 8, "effective_seed": 30, "puzzle": "left2"},
         {"master_seed": 9, "effective_seed": 30, "puzzle": "right2"}, "effective seeds"),
        ({"master_seed": 10, "effective_seed": 40, "puzzle": "same"},
         {"master_seed": 11, "effective_seed": 41, "puzzle": "same"}, "puzzle fingerprints"),
    )
    for index, (left, right, message) in enumerate(cases):
        train = _write_manifest(tmp_path, f"left-{index}", **left)
        val = _write_manifest(tmp_path, f"right-{index}", **right)
        with pytest.raises(ValueError, match=message):
            T.audit_manifest_pair(train, val, smoke=True)


def test_puzzle_identity_prefers_d4_geometry_and_keeps_gameplay_identity():
    identity = T.puzzle_identity([{
        "generator_version": 1,
        "geometry_sha256": "1" * 64,
        "geometry_d4_sha256": "2" * 64,
        "gameplay_sha256": "3" * 64,
        "layout": {"cell": [1, 2]},
    }])
    assert ("geometry_d4", "2" * 64) in identity.fingerprints
    assert ("gameplay", "3" * 64) in identity.fingerprints
    assert not any(kind == "geometry" for kind, _ in identity.fingerprints)
    assert not any(digest == "1" * 64 for _, digest in identity.fingerprints)


def test_audit_merges_multiple_roots_and_retains_partial_nonempty_traces(tmp_path):
    train_a = _write_manifest(
        tmp_path, "train-a", source_slug="cd82", master_seed=1,
        effective_seed=101, puzzle="a",
    )
    train_b = _write_manifest(
        tmp_path, "train-b", source_slug="ft09", master_seed=2,
        effective_seed=102, puzzle="b", status="rollout_failed", action_sources=(0,),
    )
    val_a = _write_manifest(
        tmp_path, "val-a", source_slug="cd82", master_seed=3,
        effective_seed=201, puzzle="c",
    )
    val_b = _write_manifest(
        tmp_path, "val-b", source_slug="ft09", master_seed=4,
        effective_seed=202, puzzle="d",
    )
    bundle = T.audit_manifest_pair([train_a, train_b], [val_a, val_b], smoke=True)
    assert len(bundle.train.manifest_paths) == 2
    assert len(bundle.train.games) == 2
    assert sum(game.partial for game in bundle.train.games) == 1
    assert bundle.train.failed_records


def test_completed_family_coverage_cannot_be_forged_by_stale_record_status(tmp_path):
    stale = _write_manifest(
        tmp_path, "stale", master_seed=1, effective_seed=101,
        puzzle="stale", status="rollout_failed", action_sources=(0,),
    )
    manifest = json.loads(stale.read_text())
    manifest["records"][0]["status"] = "won"
    M.save_manifest(stale, manifest)
    validation = _write_manifest(
        tmp_path, "stale-val", master_seed=2, effective_seed=202, puzzle="valid",
    )
    with pytest.raises(ValueError, match="status contradicts"):
        T.audit_manifest_pair(stale, validation, smoke=True)


def test_generated_selection_panel_is_source_stratified_and_raw_frames_are_strict(tmp_path):
    train_a = _write_manifest(
        tmp_path, "panel-a", source_slug="cd82", master_seed=1,
        effective_seed=101, puzzle="a",
    )
    train_b = _write_manifest(
        tmp_path, "panel-b", source_slug="ft09", master_seed=2,
        effective_seed=102, puzzle="b",
    )
    val_a = _write_manifest(
        tmp_path, "panel-c", source_slug="cd82", master_seed=3,
        effective_seed=201, puzzle="c",
    )
    val_b = _write_manifest(
        tmp_path, "panel-d", source_slug="ft09", master_seed=4,
        effective_seed=202, puzzle="d",
    )
    bundle = T.audit_manifest_pair([train_a, train_b], [val_a, val_b], smoke=True)
    # Repeating the alphabetically first source cannot occupy both seats.
    games = [bundle.validation.games[0], bundle.validation.games[0], bundle.validation.games[1]]
    panel = T._source_stratified_panel(games, max_games=2)
    assert {game.source_id for game in panel} == {
        M.source_for("cd82").source_id, M.source_for("ft09").source_id,
    }
    from_list = T._validated_raw_frame([[3] * 64 for _ in range(64)])
    from_int8 = T._validated_raw_frame(np.full((64, 64), 15, dtype=np.int8))
    assert from_list.dtype == from_int8.dtype == np.uint8
    assert int(from_list[0, 0]) == 3 and int(from_int8[0, 0]) == 15
    for invalid in (
        np.zeros((64, 64), dtype=np.float32),
        np.zeros((64, 64), dtype=np.bool_),
        np.full((64, 64), -1, dtype=np.int8),
        np.full((64, 64), 16, dtype=np.int64),
    ):
        with pytest.raises(ValueError, match="dtype|palette"):
            T._validated_raw_frame(invalid)


def test_tbptt_carries_finite_memory_across_level_boundary_and_detaches_between_chunks(tmp_path):
    train, validation = _pair(tmp_path)
    sequence = T.audit_manifest_pair(train, validation, smoke=True).train.games[0].sequence
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()
    first = T.slice_game_sequence(sequence, 0, 1)
    second = T.slice_game_sequence(sequence, 1, 2)
    first_loss = compute_multigame_loss(
        model, collate_game_sequences((first,)), transition_indices=torch.tensor([[0, 0]]),
    )
    first_loss.total.backward()
    assert first_loss.final_memory is not None
    carried = first_loss.final_memory.detach()
    model.zero_grad(set_to_none=True)
    second_loss = compute_multigame_loss(
        model,
        collate_game_sequences((second,)),
        transition_indices=torch.tensor([[0, 0]]),
        initial_memory=carried,
    )
    second_loss.total.backward()
    reset_loss = compute_multigame_loss(
        model,
        collate_game_sequences((second,)),
        transition_indices=torch.tensor([[0, 0]]),
        initial_memory=model.initial_memory(1),
    )
    assert torch.isfinite(second_loss.total)
    assert not torch.equal(second_loss.final_memory, reset_loss.final_memory)


def test_random_transition_metrics_include_copy_and_changed_pixel_baselines():
    accumulator = T._empty_transition_accumulator()
    current = torch.zeros((1, 64, 64), dtype=torch.long)
    target = current.clone()
    target[0, 9, 13] = 4
    logits = torch.full((1, 16, 64, 64), -20.0)
    logits.scatter_(1, target[:, None], 20.0)
    T._accumulate_transition_metrics(
        accumulator,
        logits,
        torch.tensor([[20.0, -20.0, 20.0]]),
        current,
        target,
        torch.tensor([[True, False, True]]),
    )
    report = T._transition_summary(accumulator)
    assert report["available"]
    assert report["pixel_accuracy"] == report["changed_pixel_accuracy"] == 1.0
    assert report["copy_current_pixel_accuracy"] == (4095 / 4096)
    assert report["copy_current_changed_pixel_accuracy"] == 0.0
    assert report["event_accuracy"] == 1.0
    assert T._transition_summary(T._empty_transition_accumulator()) == {
        "available": False, "rows": 0,
    }


def test_resume_restores_state_and_promotion_uses_closed_loop_before_offline_loss(
    tmp_path, monkeypatch,
):
    train, validation = _pair(tmp_path)
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    closed_scores = iter(((0, 0), (1, 1)))

    def fake_closed(*args, **kwargs):
        wins, levels = next(closed_scores)
        return {
            "games_won": wins,
            "levels_completed": levels,
            "games": [{"failure": None}],
            "panel_records": ["fake"],
            "actions": 1,
        }

    losses = iter((0.01, 9.0))
    monkeypatch.setattr(T, "evaluate_generated_closed_loop", fake_closed)
    monkeypatch.setattr(T, "evaluate_generated_offline", lambda *args, **kwargs: {
        "policy_action_loss": next(losses),
        "policy_click_loss": None,
        "policy": {},
        "transition": {},
    })
    out = tmp_path / "run"
    base = dict(
        model=MultiGameModelConfig.cpu_test(),
        loss=LossWeights(next_frame=0.0, events=0.0),
        chunk_steps=1,
        auxiliary_transitions_per_chunk=1,
        metric_transitions_per_game=1,
        closed_loop_games=1,
        closed_loop_train_games=0,
        validation_max_actions_per_level=1,
        validation_max_game_actions=1,
        device="cpu",
    )
    first = T.train_multigame(bundle, out, T.TrainingConfig(epochs=1, **base))
    saved_first = T.load_training_checkpoint(first.latest_checkpoint)
    # Simulate an interruption after an interval-skipped epoch: latest is
    # resumable even though no generated-game candidate has been selected yet.
    interrupted = dict(saved_first)
    interrupted["best_score"] = None
    interrupted["logs"] = [dict(interrupted["logs"][0], promoted=False)]
    torch.save(interrupted, first.latest_checkpoint)
    first.best_checkpoint.resolve().unlink()
    second = T.train_multigame(
        bundle, out, T.TrainingConfig(epochs=2, **base), resume=first.latest_checkpoint,
    )
    latest = T.load_training_checkpoint(second.latest_checkpoint)
    best = T.load_training_checkpoint(second.best_checkpoint)
    assert latest["global_step"] > saved_first["global_step"]
    assert len(latest["logs"]) == 2
    assert best["epoch"] == 1  # worse offline loss, better closed-loop gameplay
    assert latest["official_levels_used_for_selection"] is False
    assert latest["smoke"] and latest["scope"] == "smoke"


def test_fresh_training_rejects_nonempty_output_before_model_work(tmp_path):
    train, validation = _pair(tmp_path)
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "foreign.txt").write_text("do not overwrite")
    with pytest.raises(ValueError, match="not empty"):
        T.train_multigame(
            bundle,
            out,
            T.TrainingConfig(
                model=MultiGameModelConfig.cpu_test(), device="cpu", closed_loop_games=1,
            ),
        )
