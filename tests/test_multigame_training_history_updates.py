"""Diagnostic metrics, closed-loop guards, history dropout and game-mode updates."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pebby import multigame as M
from pebby.agent.multigame_model import (
    ACTION_COUNT,
    CLICK_ACTION,
    FRAME_SIZE,
    LossWeights,
    MultiGameModel,
    MultiGameModelConfig,
    collate_game_sequences,
    compute_multigame_loss,
)
from pebby.agent import multigame_training as T


# --------------------------------------------------------------------------- fixtures


def _write_manifest(
    root: Path,
    name: str,
    *,
    source_slug: str = "cd82",
    master_seed: int = 1,
    game_index: int = 0,
    effective_seed: int = 101,
    puzzle: str = "a",
    actions: tuple[int, ...] = (1, 6),
    action_sources: tuple[int, ...] | None = None,
    levels: int = 1,
    region_radius: int | None = None,
) -> Path:
    """One won game; ``levels`` splits it into equal levels; regions optional."""
    folder = root / name
    (folder / "games").mkdir(parents=True)
    (folder / "teacher").mkdir()
    source = M.source_for(source_slug)
    steps = len(actions)
    if action_sources is None:
        action_sources = tuple(1 if index % 2 == 0 else 0 for index in range(steps))
    assert len(action_sources) == steps
    ids = np.asarray(actions, dtype=np.int8)
    x = np.full(steps, -1, dtype=np.int16)
    y = np.full(steps, -1, dtype=np.int16)
    x[ids == CLICK_ACTION] = 30
    y[ids == CLICK_ACTION] = 20
    frames = np.zeros((steps + 1, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    color = (sum(puzzle.encode()) % 14) + 1
    frames[:, 3:6, 5:8] = color
    for index in range(1, steps + 1):
        frames[index, index, index] = (color + index) % 16
    legal = np.zeros((steps + 1, ACTION_COUNT), dtype=np.bool_)
    legal[:-1, 1:] = True
    boundary = np.zeros(steps, dtype=np.bool_)
    assert steps % levels == 0
    per_level = steps // levels
    for level in range(levels):
        boundary[(level + 1) * per_level - 1] = True
    completed = np.concatenate(([0], np.cumsum(boundary))).astype(np.int16)
    level_index = np.minimum(completed, levels - 1).astype(np.int16)
    terminal = np.zeros(steps + 1, dtype=np.bool_)
    terminal[-1] = True
    np.savez(
        folder / "games" / "game.npz",
        frames=frames,
        legal_action_mask=legal,
        state=np.asarray(["NOT_FINISHED"] * steps + ["WIN"]),
        level_index=level_index,
        levels_completed=completed,
        terminal=terminal,
        won=terminal.copy(),
        action_id=ids,
        action_x=x,
        action_y=y,
        level_boundary=boundary,
    )
    teacher = dict(
        target_action_id=ids.copy(),
        target_action_x=x.copy(),
        target_action_y=y.copy(),
        source=np.asarray(action_sources, dtype=np.int8),
    )
    if region_radius is not None:
        mask = np.zeros((steps, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        for row in np.flatnonzero(ids == CLICK_ACTION):
            mask[row, 20 - region_radius:21 + region_radius, 30 - region_radius:31 + region_radius] = 1
        teacher["click_region_mask"] = mask
        teacher["click_region_size"] = mask.sum(axis=(1, 2)).astype(np.int16)
    np.savez(folder / "teacher" / "game.npz", **teacher)
    specs = [
        {"generator_version": 1, "effective_seed": effective_seed + level,
         "layout": {"puzzle_token": f"{puzzle}-{level}"}}
        for level in range(levels)
    ]
    (folder / "teacher" / "game.levels.json").write_text(json.dumps(specs))
    record = {
        "format": M.FORMAT,
        "source": source.slug,
        "source_id": source.source_id,
        "master_seed": master_seed,
        "game_index": game_index,
        "status": "won",
        "steps": steps,
        "public_npz": "games/game.npz",
        "teacher_npz": "teacher/game.npz",
        "generated_specs": "teacher/game.levels.json",
        "record": "records/game.json",
        "levels": [{"generator_version": 1} for _ in range(levels)],
    }
    manifest = {
        "format": M.MANIFEST_FORMAT,
        "requested_source_ids": [source.source_id],
        "held_out_source_id": M.HELD_OUT_SOURCE_ID,
        "is_full_experiment_collection": False,
        "records": [record],
    }
    M.save_manifest(folder / "manifest.json", manifest)
    return folder / "manifest.json"


def _bundle(tmp_path: Path, **train_kwargs):
    train = _write_manifest(
        tmp_path, "train", master_seed=11, effective_seed=1100, puzzle="train", **train_kwargs,
    )
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=2200, puzzle="validation",
    )
    return T.audit_manifest_pair(train, validation, smoke=True)


def _base_config(**overrides) -> T.TrainingConfig:
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
    base.update(overrides)
    return T.TrainingConfig(**base)


def _model_supports(function, name: str) -> bool:
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _require_loss_kwarg(name: str) -> None:
    if not _model_supports(compute_multigame_loss, name):
        pytest.skip(f"model build does not accept compute_multigame_loss(..., {name}=...) yet")


class FakePolicy(nn.Module):
    """Always picks action 1; records every policy_step call."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def initial_memory(self, batch_size, *, device=None):
        return torch.zeros((batch_size, 1), device=device)

    def policy_step(self, **kwargs):
        self.calls.append(kwargs)
        logits = torch.full((1, M.ACTION_COUNT), float("-inf"))
        logits[:, 1] = 0.0
        return SimpleNamespace(action_logits=logits, click_logits=torch.zeros((1, 64, 64))), kwargs["memory"]

    @staticmethod
    def decode_click(logits):
        return torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)


class FakeEnv:
    def __init__(self, available=(1,)):
        self.available_actions = available
        self.levels_completed = 0
        self.state = "NOT_FINISHED"
        self.perform_calls = []

    def reset(self):
        pass

    def render(self):
        return np.zeros((64, 64), dtype=np.uint8)

    def perform(self, action_id, x=None, y=None):
        self.perform_calls.append((action_id, x, y))
        self.levels_completed += 1
        self.state = "WIN"


# --------------------------------------------------------------------------- item 1: metrics


def test_policy_metrics_split_switch_repeat_bos_level_and_region_coverage_with_known_counts():
    accumulator = T._empty_policy_accumulator()
    target_id = torch.tensor([1, 1, 6, 6])
    previous_id = torch.tensor([-1, 1, 1, 6])       # BOS, repeat, switch, repeat
    action_guess = torch.tensor([1, 2, 6, 6])       # wrong only on the repeat row 1
    target_x = torch.tensor([-1, -1, 5, 7])
    target_y = torch.tensor([-1, -1, 5, 7])
    click_x = torch.tensor([0, 0, 6, 7])            # row 2: region hit but not exact
    click_y = torch.tensor([0, 0, 5, 7])
    region = torch.zeros((4, FRAME_SIZE, FRAME_SIZE), dtype=torch.bool)
    region[2, 5, 5:7] = True
    region[3, 7, 7] = True
    level_index = torch.tensor([0, 0, 1, 1])
    T._accumulate_policy_metrics(
        accumulator, action_guess=action_guess, click_x=click_x, click_y=click_y,
        target_id=target_id, target_x=target_x, target_y=target_y, previous_id=previous_id,
        selected=torch.ones(4, dtype=torch.bool), click_region=region,
        level_index=level_index, region_stored=True,
    )
    summary = T._policy_summary(accumulator)
    assert summary["actions"] == 4 and summary["action_correct"] == 3
    assert summary["switch_targets"] == 2 and summary["switch_accuracy"] == 1.0
    assert summary["repeat_targets"] == 2 and summary["repeat_accuracy"] == 0.5
    assert summary["bos_targets"] == 1 and summary["bos_accuracy"] == 1.0
    assert summary["bos_joint_accuracy"] == 1.0
    assert summary["joint_correct"] == 2 and summary["joint_region_correct"] == 3
    assert summary["clicks"] == 2 and summary["click_correct"] == 1
    assert summary["click_region_correct"] == 2 and summary["click_region_labelled"] == 1
    assert summary["click_region_stored_rows"] == 2
    assert summary["click_exact_fallback_rows"] == 0
    assert summary["by_level_index"] == {
        "0": {"actions": 2, "action_accuracy": 0.5, "joint_accuracy": 0.5,
              "joint_region_accuracy": 0.5},
        "1": {"actions": 2, "action_accuracy": 1.0, "joint_accuracy": 0.5,
              "joint_region_accuracy": 1.0},
    }
    # Exact-fallback labels count towards the other bucket; unselected rows are ignored.
    fallback = T._empty_policy_accumulator()
    T._accumulate_policy_metrics(
        fallback, action_guess=action_guess, click_x=click_x, click_y=click_y,
        target_id=target_id, target_x=target_x, target_y=target_y, previous_id=previous_id,
        selected=torch.tensor([True, True, True, False]), click_region=region,
        region_stored=False,
    )
    assert fallback["click_exact_fallback_rows"] == 1 and fallback["click_region_stored_rows"] == 0
    assert fallback["actions"] == 3 and fallback["by_level"] == {}


def test_offline_evaluation_reports_per_level_and_region_coverage_on_real_games(tmp_path):
    train = _write_manifest(
        tmp_path, "train", master_seed=11, effective_seed=1100, puzzle="train",
        actions=(1, 6, 2, 6), levels=2, region_radius=1,
    )
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=2200, puzzle="validation",
        actions=(1, 6, 2, 6), levels=2,
    )
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()
    report = T.evaluate_generated_offline(
        model, [*bundle.train.games, *bundle.validation.games], chunk_steps=3,
        transition_limit_per_game=1, device=torch.device("cpu"),
    )
    assert model.training  # mode restored
    assert report["history_free"] is False
    assert report["action_targets"] == 8 and report["click_targets"] == 4
    assert report["click_region_stored_rows"] == 2
    assert report["click_exact_fallback_rows"] == 2
    assert report["bos_targets"] == 2
    assert report["switch_targets"] + report["repeat_targets"] == 8
    assert set(report["by_level_index"]) == {"0", "1"}
    assert report["by_level_index"]["0"]["actions"] == 4
    assert report["by_level_index"]["1"]["actions"] == 4
    for value in (report["joint_action_region_accuracy"], report["switch_accuracy"],
                  report["bos_accuracy"]):
        assert value is not None and 0.0 <= value <= 1.0
    assert report["joint_action_region_accuracy"] >= report["joint_action_click_accuracy"]
    family = report["by_family"][bundle.train.games[0].source_id]
    assert family["by_level_index"]["1"]["actions"] == 4
    assert family["click_region_stored_rows"] == 2 and family["click_exact_fallback_rows"] == 2


def test_default_training_panel_covers_every_family_once():
    config = T.TrainingConfig(model=MultiGameModelConfig.cpu_test())
    assert config.closed_loop_train_games == len(M.TRAIN_SOURCE_IDS) == 24
    games = [
        SimpleNamespace(source_id=source_id, master_seed=seed, game_index=0, key=f"{source_id}:{seed}")
        for source_id in M.TRAIN_SOURCE_IDS
        for seed in (2, 1)
    ]
    panel = T._source_stratified_panel(games, max_games=config.closed_loop_train_games)
    assert len(panel) == 24
    assert {game.source_id for game in panel} == set(M.TRAIN_SOURCE_IDS)
    assert all(game.master_seed == 1 for game in panel)
    assert len(T._source_stratified_panel(games, max_games=3)) == 3  # still configurable


@pytest.mark.parametrize("available, expected", (
    ((), "no_legal_public_action"),
    ((0,), "no_legal_public_action"),
    ((9,), "invalid_legal_action_id:9"),
    (("x",), "invalid_legal_action_id:'x'"),
))
def test_closed_loop_reports_bad_legal_sets_as_one_adapter_error_without_acting(
    tmp_path, monkeypatch, available, expected,
):
    bundle = _bundle(tmp_path)
    env = FakeEnv(available=available)
    monkeypatch.setattr(T, "_build_generated_env", lambda audited: env)
    model = FakePolicy().train()
    report = T.evaluate_generated_closed_loop(
        model, bundle.validation.games, max_games=1, max_actions_per_level=4,
        max_game_actions=4, device=torch.device("cpu"),
    )
    assert report["outcomes"]["adapter_error"] == 1 and report["games_played"] == 1
    assert report["games"][0]["failure"] == expected
    assert report["games"][0]["actions"] == 0
    assert env.perform_calls == [] and model.calls == []
    assert model.training  # restored in ``finally``


def test_closed_loop_restores_train_mode_even_when_the_panel_raises(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(T, "_play_generated_game", broken)
    model = FakePolicy().train()
    with pytest.raises(RuntimeError, match="boom"):
        T.evaluate_generated_closed_loop(
            model, bundle.validation.games, max_games=1, max_actions_per_level=1,
            max_game_actions=1, device=torch.device("cpu"),
        )
    assert model.training


def test_closed_loop_history_free_passes_history_keep_false_only_when_requested(
    tmp_path, monkeypatch,
):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(T, "_build_generated_env", lambda audited: FakeEnv())
    model = FakePolicy()
    normal = T.evaluate_generated_closed_loop(
        model, bundle.validation.games, max_games=1, max_actions_per_level=2,
        max_game_actions=2, device=torch.device("cpu"),
    )
    assert normal["history_free"] is False and normal["games_won"] == 1
    assert all("history_keep" not in call for call in model.calls)
    model.calls.clear()
    free = T.evaluate_generated_closed_loop(
        model, bundle.validation.games, max_games=1, max_actions_per_level=2,
        max_game_actions=2, device=torch.device("cpu"), history_free=True,
    )
    assert free["history_free"] is True and free["games_won"] == 1
    assert model.calls and all(
        call["history_keep"].dtype == torch.bool and not bool(call["history_keep"].item())
        for call in model.calls
    )


# --------------------------------------------------------------------------- item 2: history dropout


def test_history_keep_sampling_is_seeded_per_game_and_epoch():
    config = _base_config(history_dropout=0.5, seed=3)
    draws = {
        (epoch, game): T.history_keep_for_game(config, epoch=epoch, game_index=game)
        for epoch in range(4) for game in range(50)
    }
    assert {True, False} <= set(draws.values())
    kept = sum(draws.values()) / len(draws)
    assert 0.3 < kept < 0.7
    per_epoch = [tuple(draws[(epoch, game)] for game in range(50)) for epoch in range(4)]
    assert len(set(per_epoch)) == 4  # differs across epochs
    assert any(len(set(row)) == 2 for row in per_epoch)  # differs across games
    again = _base_config(history_dropout=0.5, seed=3)
    assert all(
        T.history_keep_for_game(again, epoch=epoch, game_index=game) == value
        for (epoch, game), value in draws.items()
    )
    other_seed = _base_config(history_dropout=0.5, seed=4)
    assert any(
        T.history_keep_for_game(other_seed, epoch=epoch, game_index=game) != value
        for (epoch, game), value in draws.items()
    )
    assert T.history_keep_for_game(_base_config(), epoch=0, game_index=0) is True
    assert T.history_keep_for_game(_base_config(history_mode="none"), epoch=0, game_index=0) is False
    with pytest.raises(ValueError, match="history_dropout"):
        _base_config(history_mode="none", history_dropout=0.5)
    with pytest.raises(ValueError, match="history_mode"):
        _base_config(history_mode="half")
    with pytest.raises(ValueError, match="history_dropout"):
        _base_config(history_dropout=1.5)
    restored = T.TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config


def test_trainer_carries_one_history_keep_decision_across_a_games_chunks(tmp_path, monkeypatch):
    train_a = _write_manifest(
        tmp_path, "train-a", master_seed=11, effective_seed=1100, puzzle="a", actions=(1, 6, 2, 6),
    )
    train_b = _write_manifest(
        tmp_path, "train-b", source_slug="ft09", master_seed=12, effective_seed=1200, puzzle="b",
        actions=(1, 6, 2, 6),
    )
    train_c = _write_manifest(
        tmp_path, "train-c", source_slug="ka59", master_seed=13, effective_seed=1300, puzzle="c",
        actions=(1, 6, 2, 6),
    )
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=2200, puzzle="v",
    )
    bundle = T.audit_manifest_pair([train_a, train_b, train_c], validation, smoke=True)
    real = compute_multigame_loss
    real_supports = _model_supports(real, "history_keep")
    calls = []

    def recording(model, batch, *, history_keep=None, **kwargs):
        calls.append({
            "history_keep": None if history_keep is None else bool(history_keep.item()),
            "targets": batch["target_action_id"][0].clone(),
            "executed": batch["executed_action_id"][0].clone(),
            "memory_is_initial": bool((kwargs["initial_memory"] == 0).all()),
        })
        if real_supports and history_keep is not None:
            kwargs["history_keep"] = history_keep
        return real(model, batch, **kwargs)

    monkeypatch.setattr(T, "compute_multigame_loss", recording)
    monkeypatch.setattr(T, "evaluate_generated_closed_loop", lambda *args, **kwargs: {
        "games_won": 0, "levels_completed": 0, "games": [], "panel_records": [], "actions": 0,
    })
    config = _base_config(epochs=3, history_dropout=0.5, seed=3)
    result = T.train_multigame(bundle, tmp_path / "run", config)
    assert len(calls) == 3 * 3 * 4
    # Group the recorded chunks back into games via the initial-memory marker.
    games = []
    for call in calls:
        if call["memory_is_initial"]:
            games.append([])
        games[-1].append(call)
    assert len(games) == 9 and all(len(game) == 4 for game in games)
    decisions = []
    for game in games:
        keeps = {chunk["history_keep"] for chunk in game}
        assert len(keeps) == 1  # one decision carried across every chunk
        decisions.append(keeps.pop())
        targets = torch.cat([chunk["targets"] for chunk in game]).tolist()
        executed = torch.cat([chunk["executed"] for chunk in game]).tolist()
        assert targets == executed == [1, 6, 2, 6]  # labels untouched by dropout
    # Kept games pass no history_keep (legacy path); dropped games pass False.
    assert set(decisions) == {None, False}
    dropped = [log["train"]["history_dropped_games"] for log in result.logs]
    assert sum(dropped) == decisions.count(False)
    assert all(log["history"]["dropout"] == 0.5 for log in result.logs)
    assert all(log["train"]["history_dropped_fraction"] == count / 3 for log, count in zip(result.logs, dropped))
    payload = T.load_training_checkpoint(result.latest_checkpoint)
    assert payload["history_dropout"] == 0.5 and payload["history_mode"] == "full"


def test_history_dropout_without_model_support_fails_before_training(tmp_path, monkeypatch):
    if _model_supports(compute_multigame_loss, "history_keep"):
        pytest.skip("model build already supports history_keep")
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="history_keep"):
        T.train_multigame(bundle, tmp_path / "run", _base_config(history_dropout=0.5))
    assert not (tmp_path / "run" / "latest.pt").exists()


# --------------------------------------------------------------------------- item 4: click regions


def test_click_region_census_and_require_click_regions_gate(tmp_path):
    region_free = _bundle(tmp_path / "free")
    census = T.click_region_census(region_free.train)
    assert census == {
        "games": 1, "games_with_click_regions": 0, "click_rows": 1,
        "click_region_stored_rows": 0, "click_exact_fallback_rows": 1,
        "click_multi_pixel_region_rows": 0,
    }
    with pytest.raises(ValueError, match="require_click_regions"):
        T.train_multigame(
            region_free, tmp_path / "free" / "run", _base_config(require_click_regions=True),
        )
    assert not (tmp_path / "free" / "run").exists()  # failed before any output

    labelled = _bundle(tmp_path / "labelled", actions=(1, 6, 6), region_radius=1)
    census = T.click_region_census(labelled.train)
    assert census["click_region_stored_rows"] == 2 and census["click_multi_pixel_region_rows"] == 2
    assert census["games_with_click_regions"] == 1 and census["click_exact_fallback_rows"] == 0


def test_require_click_regions_cli_fails_fast_on_region_free_manifest(tmp_path, capsys):
    from tools import train_multigame as cli

    train = _write_manifest(tmp_path, "train", master_seed=11, effective_seed=1100, puzzle="train")
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=2200, puzzle="validation",
    )
    out = tmp_path / "run"
    code = cli.cli([
        "--train-manifest", str(train), "--validation-manifest", str(validation),
        "--out-dir", str(out), "--smoke", "--cpu-test-model", "--epochs", "1",
        "--require-click-regions",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "require_click_regions" in captured.err
    assert "click regions | train: stored-region click rows 0/1" in captured.out
    assert not out.exists()


# --------------------------------------------------------------------------- item 5: update modes


class _NoRecurrence(nn.Module):
    """Drop-in for the GRU cell whose output ignores the carried memory."""

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, token, memory):
        return self.proj(token)


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }


def test_game_update_gradients_match_a_single_chunk_pass_without_recurrence(tmp_path):
    _require_loss_kwarg("reduction")
    bundle = _bundle(tmp_path, actions=(1, 6, 2, 6, 3, 6))
    sequence = bundle.train.games[0].sequence
    torch.manual_seed(0)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()
    model.memory = _NoRecurrence(model.config.hidden_dim)
    weights = LossWeights(next_frame=0.0, events=0.0)

    model.zero_grad(set_to_none=True)
    reference = compute_multigame_loss(
        model, collate_game_sequences((sequence,)), weights=weights, transition_indices=None,
    )
    reference.total.backward()
    single = _grads(model)
    assert single

    model.zero_grad(set_to_none=True)
    stats = T.accumulate_game_gradients(
        model, sequence, _base_config(chunk_steps=2, loss=weights), device=torch.device("cpu"),
    )
    chunked = _grads(model)
    assert stats["chunks"] == 3 and stats["updates"] == 1
    assert set(chunked) == set(single)
    for name, value in single.items():
        assert torch.allclose(chunked[name], value, atol=1e-6, rtol=1e-5), name
    assert stats["action"] == pytest.approx(float(reference.action.detach()), rel=1e-5)
    assert stats["click"] == pytest.approx(float(reference.click.detach()), rel=1e-5)
    assert stats["total"] == pytest.approx(float(reference.total.detach()), rel=1e-5)

    # A different partition gives the same gradient too.
    model.zero_grad(set_to_none=True)
    T.accumulate_game_gradients(
        model, sequence, _base_config(chunk_steps=4, loss=weights), device=torch.device("cpu"),
    )
    for name, value in _grads(model).items():
        assert torch.allclose(single[name], value, atol=1e-6, rtol=1e-5), name


def test_game_update_normalises_sparse_clicks_without_nan(tmp_path):
    _require_loss_kwarg("reduction")
    bundle = _bundle(tmp_path, actions=(1, 2, 3, 1))  # no click targets at all
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()
    model.zero_grad(set_to_none=True)
    stats = T.accumulate_game_gradients(
        model, bundle.train.games[0].sequence, _base_config(chunk_steps=3),
        device=torch.device("cpu"),
    )
    assert stats["click_targets"] == 0 and stats["click"] == 0.0 and stats["click_nll_sum"] == 0.0
    assert np.isfinite(stats["total"]) and stats["action"] > 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_game_update_mode_steps_once_per_game_and_is_recorded(tmp_path, monkeypatch):
    _require_loss_kwarg("reduction")
    bundle = _bundle(tmp_path, actions=(1, 6, 2, 6))
    monkeypatch.setattr(T, "evaluate_generated_closed_loop", lambda *args, **kwargs: {
        "games_won": 0, "levels_completed": 0, "games": [], "panel_records": [], "actions": 0,
    })
    steps = []
    original = torch.optim.AdamW.step

    def counting(self, *args, **kwargs):
        steps.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting)
    weights = LossWeights(next_frame=0.5, events=0.2)
    result = T.train_multigame(
        bundle, tmp_path / "run",
        _base_config(epochs=1, update_mode="game", chunk_steps=1, loss=weights),
    )
    log = result.logs[0]
    assert len(steps) == 1 and log["global_step"] == 1
    assert log["train"]["updates"] == 1 and log["train"]["chunks"] == 4
    assert log["update_mode"] == "game" and log["train"]["update_mode"] == "game"
    assert log["loss_weights"] == {
        "action": 1.0, "click": 1.0, "next_frame": 0.5, "events": 0.2,
        "changed_pixel_weight": 1.0, "event_positive_weight": 1.0,
    }
    assert log["auxiliaries_enabled"] is True
    assert log["train"]["click_targets"] == 2
    assert np.isfinite(log["train"]["frame"]) and np.isfinite(log["train"]["events"])
    payload = T.load_training_checkpoint(result.latest_checkpoint)
    assert payload["update_mode"] == "game"
    assert payload["loss_weights"]["next_frame"] == 0.5


def test_zero_auxiliary_weights_skip_the_auxiliary_heads(tmp_path):
    bundle = _bundle(tmp_path)
    sequence = bundle.train.games[0].sequence
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()

    def forbidden(*args, **kwargs):
        raise AssertionError("auxiliary heads must not run when both weights are 0")

    model.predict_transitions = forbidden  # type: ignore[method-assign]
    weights = LossWeights(next_frame=0.0, events=0.0)
    try:
        compute_multigame_loss(model, collate_game_sequences((sequence,)), weights=weights)
    except AssertionError:
        pytest.skip("model build does not skip auxiliary heads at zero weight yet")
    config = _base_config(chunk_steps=1, loss=weights)
    assert config.auxiliaries_enabled is False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = T.train_game(model, optimizer, sequence, config, device=torch.device("cpu"))
    assert stats["updates"] == 2 and stats["frame"] == 0.0 and stats["events"] == 0.0
    if _model_supports(compute_multigame_loss, "reduction"):
        model.zero_grad(set_to_none=True)
        game_stats = T.accumulate_game_gradients(
            model, sequence, config, device=torch.device("cpu"),
        )
        assert game_stats["frame"] == 0.0 and game_stats["events"] == 0.0
    assert T.TrainingConfig(model=MultiGameModelConfig.cpu_test()).auxiliaries_enabled is True
    with pytest.raises(ValueError, match="update_mode"):
        _base_config(update_mode="epoch")
    with pytest.raises(ValueError, match="non-negative"):
        _base_config(loss=LossWeights(next_frame=-1.0))


def test_chunk_mode_default_is_unchanged_and_records_weights(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, actions=(1, 6, 2, 6))
    monkeypatch.setattr(T, "evaluate_generated_closed_loop", lambda *args, **kwargs: {
        "games_won": 0, "levels_completed": 0, "games": [], "panel_records": [], "actions": 0,
    })
    result = T.train_multigame(bundle, tmp_path / "run", _base_config(epochs=1, chunk_steps=3))
    log = result.logs[0]
    assert log["global_step"] == 2 and log["train"]["updates"] == 2 and log["train"]["chunks"] == 2
    assert log["update_mode"] == "chunk" and log["auxiliaries_enabled"] is False
    assert log["history"] == {
        "mode": "full", "dropout": 0.0, "train_games_dropped": 0, "train_games": 1,
        "dropped_fraction": 0.0, "history_free_diagnostic": False,
    }
    assert log["generated_validation_offline_history_free"] is None
    assert log["generated_validation_closed_loop_history_free"] is None
    assert log["click_region_census"]["train"]["click_exact_fallback_rows"] == 2
    assert "hist-drop 0.00" in T.format_epoch_line(log)
    assert "switch" in T.format_epoch_line(log)
    signature = T._training_signature(_base_config(history_free_diagnostic=True, require_click_regions=True))
    assert signature == T._training_signature(_base_config())
    assert signature["update_mode"] == "chunk" and signature["history_dropout"] == 0.0


# --------------------------------------------------------------------------- CLI


def _cli_manifests(tmp_path: Path) -> tuple[Path, Path]:
    train = _write_manifest(
        tmp_path, "train", master_seed=11, effective_seed=1100, puzzle="train", actions=(1, 6, 2, 6),
    )
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=2200, puzzle="validation",
    )
    return train, validation


def test_cli_parses_new_flags_and_rejects_inconsistent_history_options(tmp_path):
    from tools import train_multigame as cli

    common = ["--train-manifest", "a", "--validation-manifest", "b", "--out-dir", str(tmp_path)]
    args = cli.parse_args(common)
    assert args.history_mode == "full" and args.history_dropout == 0.0 and not args.history_free
    assert args.update_mode == "chunk" and not args.require_click_regions
    assert args.frame_weight == 0.25 and args.event_weight == 0.1
    assert args.closed_loop_train_games == len(M.TRAIN_SOURCE_IDS)
    args = cli.parse_args([
        *common, "--history-mode", "none", "--update-mode", "game", "--frame-weight", "0",
        "--event-weight", "0", "--history-free", "--require-click-regions",
    ])
    assert args.history_mode == "none" and args.update_mode == "game"
    assert args.frame_weight == 0.0 and args.event_weight == 0.0
    assert args.history_free and args.require_click_regions
    with pytest.raises(SystemExit):
        cli.parse_args([*common, "--history-mode", "none", "--history-dropout", "0.5"])
    with pytest.raises(SystemExit):
        cli.parse_args([*common, "--history-dropout", "2"])
    with pytest.raises(SystemExit):
        cli.parse_args([*common, "--frame-weight", "-1"])


def test_cli_parses_architecture_and_hidden_dim_flags(tmp_path):
    from tools import train_multigame as cli

    common = ["--train-manifest", "a", "--validation-manifest", "b", "--out-dir", str(tmp_path)]
    args = cli.parse_args(common)
    assert args.architecture == "v1"
    assert args.hidden_dim == MultiGameModelConfig().hidden_dim

    args = cli.parse_args([*common, "--architecture", "v2", "--hidden-dim", "128"])
    assert args.architecture == "v2" and args.hidden_dim == 128
    with pytest.raises(SystemExit):
        cli.parse_args([*common, "--architecture", "bogus"])


def test_cli_smoke_with_architecture_v2_completes(tmp_path, monkeypatch, capsys):
    from tools import train_multigame as cli

    train, validation = _cli_manifests(tmp_path)
    monkeypatch.setattr(T, "_build_generated_env", lambda audited: FakeEnv())
    out = tmp_path / "run"
    code = cli.cli([
        "--train-manifest", str(train), "--validation-manifest", str(validation),
        "--out-dir", str(out), "--smoke", "--cpu-test-model", "--architecture", "v2",
        "--epochs", "1", "--chunk-steps", "3", "--closed-loop-games", "1",
        "--closed-loop-train-games", "1", "--device", "cpu",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    summary = json.loads(captured.out[captured.out.index("{"):])
    assert summary["model_config"]["architecture"] == "v2"
    payload = T.load_training_checkpoint(out / "latest.pt")
    assert payload["model_config"]["architecture"] == "v2"
    assert payload["training_config"]["model"]["architecture"] == "v2"


def test_cli_smoke_with_zero_auxiliaries_and_train_panel(tmp_path, monkeypatch, capsys):
    from tools import train_multigame as cli

    train, validation = _cli_manifests(tmp_path)
    monkeypatch.setattr(T, "_build_generated_env", lambda audited: FakeEnv())
    out = tmp_path / "run"
    code = cli.cli([
        "--train-manifest", str(train), "--validation-manifest", str(validation),
        "--out-dir", str(out), "--smoke", "--cpu-test-model", "--epochs", "1",
        "--chunk-steps", "3", "--closed-loop-games", "1", "--closed-loop-train-games", "1",
        "--frame-weight", "0", "--event-weight", "0", "--device", "cpu",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    summary = json.loads(captured.out[captured.out.index("{"):])
    assert summary["update_mode"] == "chunk" and summary["auxiliaries_enabled"] is False
    assert summary["loss_weights"] == {
        "action": 1.0, "click": 1.0, "next_frame": 0.0, "events": 0.0,
        "changed_pixel_weight": 1.0, "event_positive_weight": 1.0,
    }
    assert summary["click_region_census"]["train"]["click_rows"] == 2
    log = json.loads((out / "training-log.json").read_text())[0]
    assert log["generated_train_closed_loop"]["games_played"] == 1
    assert log["generated_validation_closed_loop"]["games_played"] == 1


def test_cli_smoke_with_history_dropout_and_game_update_mode(tmp_path, monkeypatch, capsys):
    from tools import train_multigame as cli

    for name in ("history_keep", "reduction"):
        _require_loss_kwarg(name)
    train, validation = _cli_manifests(tmp_path)
    monkeypatch.setattr(T, "_build_generated_env", lambda audited: FakeEnv())
    out = tmp_path / "run"
    code = cli.cli([
        "--train-manifest", str(train), "--validation-manifest", str(validation),
        "--out-dir", str(out), "--smoke", "--cpu-test-model", "--epochs", "2",
        "--chunk-steps", "3", "--closed-loop-games", "1", "--closed-loop-train-games", "1",
        "--history-dropout", "0.5", "--update-mode", "game", "--history-mode", "full",
        "--device", "cpu",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    summary = json.loads(captured.out[captured.out.index("{"):])
    assert summary["update_mode"] == "game" and summary["history_dropout"] == 0.5
    logs = json.loads((out / "training-log.json").read_text())
    assert [log["global_step"] for log in logs] == [1, 2]  # one step per game per epoch
    assert all(log["train"]["chunks"] == 2 for log in logs)
    assert all(log["history"]["dropout"] == 0.5 for log in logs)
    assert (out / "best.pt").exists()


def test_cli_history_free_diagnostic_logs_separate_panels(tmp_path, monkeypatch, capsys):
    from tools import train_multigame as cli

    model = MultiGameModel(MultiGameModelConfig.cpu_test())
    if not (_model_supports(model.policy_step, "history_keep")
            and _model_supports(model.encode_history, "history_keep")):
        pytest.skip("model build does not accept history_keep at inference yet")
    train, validation = _cli_manifests(tmp_path)
    monkeypatch.setattr(T, "_build_generated_env", lambda audited: FakeEnv())
    out = tmp_path / "run"
    code = cli.cli([
        "--train-manifest", str(train), "--validation-manifest", str(validation),
        "--out-dir", str(out), "--smoke", "--cpu-test-model", "--epochs", "1",
        "--closed-loop-games", "1", "--closed-loop-train-games", "1", "--history-free",
        "--device", "cpu",
    ])
    assert code == 0, capsys.readouterr().err
    log = json.loads((out / "training-log.json").read_text())[0]
    assert log["generated_validation_offline_history_free"]["history_free"] is True
    assert log["generated_validation_closed_loop_history_free"]["history_free"] is True
    assert log["generated_train_closed_loop_history_free"]["history_free"] is True
    assert log["generated_validation_closed_loop"]["history_free"] is False
    assert log["history"]["history_free_diagnostic"] is True
