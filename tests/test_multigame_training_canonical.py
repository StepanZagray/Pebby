"""Canonical (raw engine space) training, honest selection, and richer metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pebby import multigame as M
from pebby.agent import multigame_training as T
from pebby.agent.multigame_model import (
    CLICK_ACTION,
    FRAME_SIZE,
    GameSequence,
    MultiGameModel,
    MultiGameModelConfig,
    load_supervised_game,
)
from pebby.multigame_variants import WholeGameVariant

from test_multigame_training import _write_manifest


CONTROLS = (0, 3, 1, 2, 5, 4, 6, 7)
PALETTE = tuple((value * 7) % 16 for value in range(16))
STORED_VARIANT = WholeGameVariant(True, 99, CONTROLS, "rot90", PALETTE)


def _write_transformed_manifest(
    root: Path, name: str, variant: WholeGameVariant, **kwargs,
) -> tuple[Path, GameSequence]:
    """Write a synthetic game, then re-store it through ``variant`` like the collector."""
    manifest_path = _write_manifest(root, name, **kwargs)
    folder = manifest_path.parent
    public_path, teacher_path = folder / "games" / "game.npz", folder / "teacher" / "game.npz"
    raw = load_supervised_game(public_path, teacher_path)
    legal = T.game_legal_action_ids(raw)
    public = dict(np.load(public_path, allow_pickle=False))
    teacher = dict(np.load(teacher_path, allow_pickle=False))
    np.savez(public_path, **T.apply_variant_to_game(public, variant, legal))
    np.savez(teacher_path, **T.apply_variant_to_game(teacher, variant, legal))
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["variant"] = {
        "selected": variant.selected,
        "seed": variant.seed,
        "control_raw_to_public": list(variant.control_raw_to_public),
        "spatial": variant.spatial,
        "palette_raw_to_public": list(variant.palette_raw_to_public),
        "identity": variant.is_identity,
    }
    M.save_manifest(manifest_path, manifest)
    return manifest_path, raw


def _assert_sequences_equal(left: GameSequence, right: GameSequence) -> None:
    for name in GameSequence.__dataclass_fields__:
        assert np.array_equal(getattr(left, name), getattr(right, name)), name


def _canonical_pair(tmp_path: Path):
    train, raw_train = _write_transformed_manifest(
        tmp_path, "train", STORED_VARIANT, master_seed=11, effective_seed=111, puzzle="train",
    )
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=222, puzzle="validation",
    )
    return train, raw_train, validation


def test_canonical_loading_inverts_the_stored_variant_exactly(tmp_path):
    train, raw_train, validation = _canonical_pair(tmp_path)
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    stored = bundle.train.games[0]
    assert stored.stored_variant == STORED_VARIANT and not stored.canonical
    assert stored.play_variant == STORED_VARIANT
    # The stored files really are transformed: the loaded sequence is not raw.
    assert not np.array_equal(stored.sequence.frames, raw_train.frames)
    assert not np.array_equal(stored.sequence.target_action_id, raw_train.target_action_id)

    canonical = T.canonicalize_bundle(bundle)
    assert canonical.canonical_inputs and not bundle.canonical_inputs
    game = canonical.train.games[0]
    assert game.canonical and game.play_variant.is_identity
    assert game.stored_variant == STORED_VARIANT  # provenance is kept
    _assert_sequences_equal(game.sequence, raw_train)
    # Validation had no stored variant: canonical is a no-op on its arrays.
    untouched = canonical.validation.games[0]
    assert untouched.canonical and untouched.sequence is bundle.validation.games[0].sequence
    summary = canonical.summary()
    assert summary["canonical_inputs"] is True
    assert summary["train"]["stored_variants_inverted"] == 1
    assert summary["validation"]["stored_variants_inverted"] == 0
    assert summary["validation"]["canonical_games"] == 1
    # Idempotent and hash-preserving: canonical is a training-side view.
    assert T.canonicalize_bundle(canonical) is canonical
    assert canonical.hashes == bundle.hashes


def test_load_time_variants_are_off_by_default_under_canonical(tmp_path):
    from tools import train_multigame as cli

    config = T.TrainingConfig(canonical_inputs=True, model=MultiGameModelConfig.cpu_test())
    assert config.load_variants.enabled is False
    assert T._training_signature(config)["canonical_inputs"] is True
    restored = T.TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    legacy = config.to_dict()
    legacy.pop("canonical_inputs")
    legacy.pop("closed_loop_train_games")
    assert T.TrainingConfig.from_dict(legacy).canonical_inputs is False

    common = ["--train-manifest", "a", "--validation-manifest", "b", "--out-dir", str(tmp_path)]
    args = cli.parse_args([*common, "--canonical"])
    assert args.canonical is True
    assert cli.load_variant_options(args).enabled is False
    assert args.closed_loop_train_games == len(M.TRAIN_SOURCE_IDS) == 24
    explicit = cli.parse_args([*common, "--canonical", "--load-variants", "--closed-loop-train-games", "0"])
    assert cli.load_variant_options(explicit).enabled is True
    assert explicit.closed_loop_train_games == 0
    with pytest.raises(ValueError, match="non-negative"):
        T.TrainingConfig(closed_loop_train_games=-1, model=MultiGameModelConfig.cpu_test())


def _closed(won: int, levels: int, outcome: str) -> dict:
    return {
        "games_won": won, "levels_completed": levels,
        "games": [{"outcome": outcome, "failure": None if outcome == "game_over" else "per_level_action_budget"}],
        "outcomes": {name: int(name == outcome) for name in T.CLOSED_LOOP_OUTCOMES},
    }


def test_selection_prefers_alive_without_progress_and_more_levels():
    # The investigation's scenario: the old key ranked GAME_OVER (failure None)
    # above budget exhaustion (failure set) even though the alive candidate had
    # the better offline action loss.  Outcome category is no longer a key.
    alive = T.selection_score(_closed(0, 0, "budget_exhausted"), {"policy_action_loss": 0.8836})
    dead = T.selection_score(_closed(0, 0, "game_over"), {"policy_action_loss": 0.8936})
    assert alive > dead
    # Dying with more levels beats staying alive with none, regardless of loss.
    dead_progress = T.selection_score(_closed(0, 1, "game_over"), {"policy_action_loss": 5.0})
    alive_none = T.selection_score(_closed(0, 0, "budget_exhausted"), {"policy_action_loss": 0.1})
    assert dead_progress > alive_none
    # Games won outrank levels, levels outrank loss; missing loss is worst.
    assert T.selection_score(_closed(1, 1, "won"), {"policy_action_loss": 9.0}) > dead_progress
    assert T.selection_score(_closed(0, 0, "adapter_error"), {"policy_action_loss": None}) < dead
    # Same progress and loss: the way the game ended does not break the tie.
    assert T.selection_score(_closed(0, 0, "game_over"), {"policy_action_loss": 0.5}) == (
        T.selection_score(_closed(0, 0, "budget_exhausted"), {"policy_action_loss": 0.5})
    )
    assert "outcome" in T.SELECTION_BASIS and "failure" not in T.SELECTION_BASIS.split(";")[0]


class _State:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeEnv:
    """Scripted engine: ``win`` / ``die`` after one action, ``stall`` forever, ``raise``."""

    def __init__(self, script: str, levels: int = 1) -> None:
        self.script = script
        self.levels = levels
        self.state = _State("NOT_PLAYED")
        self.levels_completed = 0
        self.performed: list[tuple[int, int | None, int | None]] = []

    def reset(self) -> None:
        self.state = _State("NOT_FINISHED")

    @property
    def available_actions(self) -> tuple[int, ...]:
        return (1, 2, 3, 4, 5, CLICK_ACTION)

    def render(self) -> np.ndarray:
        frame = np.zeros(M.FRAME_SHAPE, dtype=np.int64)
        frame[0, len(self.performed) % FRAME_SIZE] = 1
        return frame

    def perform(self, action_id, x=None, y=None) -> None:
        self.performed.append((int(action_id), x, y))
        if self.script == "raise":
            raise RuntimeError("scripted engine failure")
        if self.script == "win":
            self.levels_completed = self.levels
            self.state = _State("WIN")
        elif self.script == "die":
            self.state = _State("GAME_OVER")


def _four_game_bundle(tmp_path: Path):
    trains = [
        _write_manifest(tmp_path, f"train-{index}", master_seed=10 + index,
                        effective_seed=100 + index, puzzle=f"train-{index}")
        for index in range(4)
    ]
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=222, puzzle="validation",
    )
    return T.audit_manifest_pair(trains, [validation], smoke=True)


def test_closed_loop_outcomes_are_exclusive_and_cover_every_game(tmp_path, monkeypatch):
    bundle = _four_game_bundle(tmp_path)
    scripts = {}
    envs = {}
    for game, script in zip(
        sorted(bundle.train.games, key=lambda item: item.master_seed), ("win", "die", "stall", "raise"),
    ):
        scripts[game.key] = script

    def build(audited):
        envs[audited.key] = FakeEnv(scripts[audited.key])
        return envs[audited.key]

    monkeypatch.setattr(T, "_build_generated_env", build)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).eval()
    report = T.evaluate_generated_closed_loop(
        model, bundle.train.games, max_games=4, max_actions_per_level=3, max_game_actions=50,
        device=torch.device("cpu"),
    )
    assert report["games_played"] == len(report["games"]) == 4
    by_script = {scripts[item["record"]]: item for item in report["games"]}
    assert by_script["win"]["outcome"] == "won" and by_script["win"]["won"] is True
    assert by_script["die"]["outcome"] == "game_over" and by_script["die"]["failure"] is None
    assert by_script["stall"]["outcome"] == "budget_exhausted"
    assert by_script["stall"]["failure"] == "per_level_action_budget" and by_script["stall"]["actions"] == 3
    assert by_script["raise"]["outcome"] == "adapter_error"
    assert by_script["raise"]["failure"].startswith("engine_error:RuntimeError")
    for item in report["games"]:
        assert item["outcome"] in T.CLOSED_LOOP_OUTCOMES
        assert item["won"] == (item["outcome"] == "won")
        # These records carry no stored variant, so as-stored play is already identity.
        assert item["variant"] == "identity"
    assert report["outcomes"] == {name: 1 for name in T.CLOSED_LOOP_OUTCOMES}
    assert sum(report["outcomes"].values()) == report["games_played"]
    assert report["games_won"] == 1 and report["levels_completed"] == 1
    family = report["by_family"][bundle.train.games[0].source_id]
    assert family["games"] == 4 and family["levels_completed"] == 1 and family["outcome_game_over"] == 1
    assert report["canonical_inputs"] is False
    # Every performed raw action was legal for the fake engine.
    for env in envs.values():
        assert all(1 <= action <= CLICK_ACTION for action, _, _ in env.performed)

    # The classifier itself: one category for every reachable combination.
    cases = {
        ("WIN", 1, 1, None): "won",
        ("WIN", 0, 1, None): "adapter_error",  # inconsistent win
        ("GAME_OVER", 0, 1, None): "game_over",
        ("NOT_FINISHED", 0, 1, "per_level_action_budget"): "budget_exhausted",
        ("NOT_FINISHED", 0, 1, "whole_game_action_budget"): "budget_exhausted",
        ("NOT_FINISHED", 0, 1, "engine_error:RuntimeError:x"): "adapter_error",
        ("NOT_FINISHED", 0, 1, "invalid raw frame shape: (3, 3)"): "adapter_error",
        ("UNAVAILABLE", 0, 1, "engine_error:KeyError:spec"): "adapter_error",
    }
    for (state, completed, requested, failure), expected in cases.items():
        got = T.classify_closed_loop_outcome(
            state=state, levels_completed=completed, levels_requested=requested, failure=failure,
        )
        assert got == expected and got in T.CLOSED_LOOP_OUTCOMES

    # Canonical games play through the identity variant.
    canonical = T.canonicalize_bundle(bundle)
    report = T.evaluate_generated_closed_loop(
        model, canonical.train.games, max_games=1,
        max_actions_per_level=1, max_game_actions=1, device=torch.device("cpu"),
    )
    assert report["canonical_inputs"] is True and report["games"][0]["variant"] == "identity"


def test_stored_variant_game_plays_through_its_variant_unless_canonical(tmp_path, monkeypatch):
    train, _, validation = _canonical_pair(tmp_path)
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    envs = []

    def build(audited):
        envs.append(FakeEnv("stall"))
        return envs[-1]

    monkeypatch.setattr(T, "_build_generated_env", build)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).eval()
    kwargs = dict(max_games=1, max_actions_per_level=2, max_game_actions=2, device=torch.device("cpu"))
    stored = T.evaluate_generated_closed_loop(model, bundle.train.games, **kwargs)
    assert stored["games"][0]["variant"] == "stored" and stored["canonical_inputs"] is False
    canonical = T.evaluate_generated_closed_loop(
        model, T.canonicalize_bundle(bundle).train.games, **kwargs,
    )
    assert canonical["games"][0]["variant"] == "identity" and canonical["canonical_inputs"] is True
    # Both runs performed raw engine actions within the engine's own legal set.
    for env in envs:
        assert env.performed and all(1 <= action <= CLICK_ACTION for action, _, _ in env.performed)


def test_policy_metric_accumulator_matches_hand_computation():
    accumulator = T._empty_policy_accumulator()
    target = torch.tensor([[1, 6, 6, 2, 3, 4]])
    previous = torch.tensor([[-1, 1, 6, 6, 3, 4]])
    guess = torch.tensor([[1, 6, 6, 3, 3, 4]])
    target_x = torch.tensor([[-1, 10, 20, -1, -1, -1]])
    target_y = torch.tensor([[-1, 11, 21, -1, -1, -1]])
    click_x = torch.tensor([[10, 10, 5, 0, 0, 0]])
    click_y = torch.tensor([[11, 11, 21, 0, 0, 0]])
    selected = torch.tensor([[True, True, True, True, True, False]])
    T._accumulate_policy_metrics(
        accumulator, action_guess=guess, click_x=click_x, click_y=click_y,
        target_id=target, target_x=target_x, target_y=target_y, previous_id=previous,
        selected=selected, action_nll=torch.full((1, 6), 0.5), click_nll=torch.full((1, 6), 2.0),
    )
    summary = T._policy_summary(accumulator)
    assert summary["actions"] == 5 and summary["clicks"] == 2
    assert summary["action_accuracy"] == pytest.approx(4 / 5)
    # previous == target on rows 2 (6==6) and 4 (3==3); the unselected row 5 is ignored.
    assert summary["repeat_baseline_accuracy"] == pytest.approx(2 / 5)
    # Row 1 click exact, row 2 click wrong pixel, row 3 action wrong: rows 0,1,4 are joint hits.
    assert summary["joint_accuracy"] == pytest.approx(3 / 5)
    assert summary["click_accuracy"] == pytest.approx(1 / 2)
    assert summary["action_loss"] == pytest.approx(0.5)
    assert summary["click_loss_per_target"] == pytest.approx(2.0)
    empty = T._policy_summary(T._empty_policy_accumulator())
    assert empty["action_accuracy"] is None and empty["click_loss_per_target"] is None


def test_offline_report_has_per_family_and_baseline_fields(tmp_path):
    train, _, validation = _canonical_pair(tmp_path)
    bundle = T.canonicalize_bundle(T.audit_manifest_pair(train, validation, smoke=True))
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).eval()
    games = bundle.train.games + bundle.validation.games
    report = T.evaluate_generated_offline(
        model, games, chunk_steps=1, transition_limit_per_game=1, device=torch.device("cpu"),
    )
    valid_rows = sum(int(game.sequence.target_valid.sum()) for game in games)
    click_rows = sum(
        int((game.sequence.target_valid & (game.sequence.target_action_id == CLICK_ACTION)).sum())
        for game in games
    )
    repeat_hits = sum(
        int((game.sequence.target_valid
             & (game.sequence.previous_action_id == game.sequence.target_action_id)).sum())
        for game in games
    )
    assert report["action_targets"] == valid_rows == 4
    assert report["click_targets"] == click_rows == 2
    assert report["repeat_baseline_accuracy"] == pytest.approx(repeat_hits / valid_rows)
    assert report["policy_click_loss_per_target"] == report["policy_click_loss"]
    assert report["policy_click_loss_per_target"] > 0
    source_id = M.source_for("cd82").source_id
    assert set(report["by_family"]) == {source_id}
    family = report["by_family"][source_id]
    assert family["actions"] == valid_rows and family["clicks"] == click_rows
    assert family["repeat_baseline_accuracy"] == report["repeat_baseline_accuracy"]
    assert family["action_accuracy"] == report["action_accuracy"]
    for key in ("action_accuracy", "joint_action_click_accuracy", "click_accuracy"):
        assert 0.0 <= report[key] <= 1.0
    assert report["joint_action_click_accuracy"] <= report["action_accuracy"]
    groups = report["policy"]
    assert groups["teacher"]["actions"] + groups["random"]["actions"] == valid_rows
    assert groups["teacher"]["action_loss"] is not None


def test_cli_canonical_smoke_run_reports_new_fields(tmp_path, monkeypatch, capsys):
    from tools import train_multigame as cli

    train, _, validation = _canonical_pair(tmp_path)
    built = []

    def build(audited):
        built.append((audited.key, audited.canonical, audited.play_variant.is_identity))
        return FakeEnv("stall")

    monkeypatch.setattr(T, "_build_generated_env", build)
    out = tmp_path / "run"
    result = cli.main([
        "--train-manifest", str(train), "--validation-manifest", str(validation),
        "--out-dir", str(out), "--smoke", "--cpu-test-model", "--canonical", "--device", "cpu",
        "--epochs", "2", "--chunk-steps", "2", "--auxiliary-transitions-per-chunk", "1",
        "--metric-transitions-per-game", "1", "--closed-loop-games", "1",
        "--closed-loop-train-games", "1", "--validation-max-actions-per-level", "2",
        "--validation-max-game-actions", "4",
    ])
    # Two epochs x one two-step chunk = two optimizer updates.
    logs = json.loads((out / "training-log.json").read_text())
    assert len(logs) == 2 and logs[-1]["global_step"] == 2
    # Every closed-loop game (validation panel and train panel) was canonical.
    assert built and all(canonical and identity for _, canonical, identity in built)
    assert len(built) == 4
    for log in logs:
        assert log["canonical_inputs"] is True
        offline = log["generated_validation_offline"]
        for key in (
            "by_family", "repeat_baseline_accuracy", "joint_action_click_accuracy",
            "policy_click_loss_per_target", "action_targets", "click_targets", "action_accuracy",
        ):
            assert key in offline
        closed = log["generated_validation_closed_loop"]
        assert set(closed["outcomes"]) == set(T.CLOSED_LOOP_OUTCOMES)
        assert sum(closed["outcomes"].values()) == closed["games_played"] == 1
        assert closed["outcomes"]["budget_exhausted"] == 1
        assert closed["games"][0]["variant"] == "identity"
        train_panel = log["generated_train_closed_loop"]
        assert train_panel["games_played"] == 1 and "by_family" in train_panel
        assert train_panel["games"][0]["variant"] == "identity"
        variants = log["load_time_variants"]
        assert variants["options"]["enabled"] is False
        assert variants["closed_loop_variant"] == "identity"
        assert variants["stored_variants_inverted"] == {"train": 1, "validation": 0}
        assert log["train"]["click_targets"] == 1 and log["train"]["click_per_target"] is not None
        assert log["selection_basis"] == T.SELECTION_BASIS
    payload = T.load_training_checkpoint(result.latest_checkpoint)
    assert payload["canonical_inputs"] is True
    assert payload["training_config"]["canonical_inputs"] is True
    assert payload["training_signature"]["canonical_inputs"] is True
    assert payload["dataset_audit"]["canonical_inputs"] is True
    assert payload["dataset_audit"]["train"]["stored_variants_inverted"] == 1
    assert payload["load_time_variants"]["enabled"] is False
    assert payload["best_score"] is not None and len(payload["best_score"]) == 3
    captured = capsys.readouterr().out
    assert "offline action acc" in captured and "vs repeat" in captured
    assert "val closed-loop" in captured and "train closed-loop" in captured
    assert "budget_exhausted=1" in captured
    summary = json.loads(captured[captured.index("{"):])
    assert summary["canonical_inputs"] is True and summary["closed_loop_train_games"] == 1
