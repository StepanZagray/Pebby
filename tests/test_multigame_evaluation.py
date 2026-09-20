"""Protocol tests for frozen public-policy official evaluation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pebby import multigame as M
from pebby.agent.multigame_model import MultiGameModel, MultiGameModelConfig
from pebby.agent import multigame_evaluation as E
from pebby.agent import multigame_training as T


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def initial_memory(self, batch_size, *, device=None):
        return torch.zeros((batch_size, 1), device=device)

    def policy_step(self, **kwargs):
        self.calls.append({
            "memory": kwargs["memory"].detach().clone(),
            "previous_boundary": bool(kwargs["previous_level_boundary"].item()),
            "previous_action_id": int(kwargs["previous_action_id"].item()),
        })
        logits = torch.full((1, M.ACTION_COUNT), float("-inf"))
        logits[:, 1] = 0.0
        click = torch.zeros((1, 64, 64))
        return SimpleNamespace(action_logits=logits, click_logits=click), kwargs["memory"] + 1

    @staticmethod
    def decode_click(logits):
        return torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)


class FakeEnv:
    def __init__(
        self, *, levels=2, never_finish=False, frame_dtype=np.uint8, frame_value=0,
        frame_as_list=False, available_actions=(1,),
    ):
        self.level_count = levels
        self._never_finish = never_finish
        self._dtype = frame_dtype
        self._frame_value = frame_value
        self._frame_as_list = frame_as_list
        self.reset_calls = 0
        self.perform_calls = []
        self.levels_completed = 0
        self.state = "NOT_FINISHED"
        self.available_actions = available_actions

    def reset(self):
        self.reset_calls += 1
        self.levels_completed = 0
        self.state = "NOT_FINISHED"

    def render(self):
        frame = np.full((64, 64), self._frame_value, dtype=self._dtype)
        return frame.tolist() if self._frame_as_list else frame

    def perform(self, action_id, x=None, y=None):
        self.perform_calls.append((action_id, x, y))
        if not self._never_finish:
            self.levels_completed += 1
            if self.levels_completed >= self.level_count:
                self.state = "WIN"


def _checkpoint(path: Path, *, smoke: bool, scope: str) -> Path:
    model = MultiGameModel(MultiGameModelConfig.cpu_test())
    payload = {
        "format": T.TRAINING_FORMAT,
        "scope": scope,
        "smoke": smoke,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def _full_gate(path: Path, checkpoint: Path, config: E.OfficialEvaluationConfig, *, unsupported=False):
    results = [
        {
            "source_id": source.source_id,
            "status": "unsupported_adapter" if unsupported else "evaluated",
        }
        for source in M.TRAIN_SOURCES
    ]
    value = {
        "format": E.REPORT_FORMAT,
        "phase": "training_families",
        "phase_complete": True,
        "scope": "full_24_family_experiment",
        "smoke": False,
        "source_ids": list(M.TRAIN_SOURCE_IDS),
        "checkpoint_sha256": T.sha256_file(checkpoint),
        "evaluation_config_hash": config.hash,
        "results": results,
    }
    path.write_text(json.dumps(value))
    return path


def test_policy_memory_crosses_level_boundary_and_only_one_initial_reset_occurs():
    model = FakePolicy()
    env = FakeEnv(levels=2)
    report = E.run_policy_game(
        model, env, source_id=M.source_for("cd82").source_id,
        config=E.OfficialEvaluationConfig(max_actions_per_level=3, max_game_actions=4),
    )
    assert report["whole_game_win"] and report["levels_completed"] == 2
    assert report["actions"] == 2
    assert env.reset_calls == report["initial_resets"] == 1
    assert report["voluntary_resets"] == 0
    assert model.calls[0]["memory"].item() == 0
    assert model.calls[1]["memory"].item() == 1
    assert model.calls[1]["previous_boundary"]
    assert model.calls[1]["previous_action_id"] == 1


def test_policy_stops_at_explicit_action_budget_without_restart_or_fallback():
    model = FakePolicy()
    env = FakeEnv(never_finish=True)
    report = E.run_policy_game(
        model, env, source_id=M.source_for("ft09").source_id,
        config=E.OfficialEvaluationConfig(max_actions_per_level=2, max_game_actions=9),
    )
    assert report["actions"] == 2
    assert report["failures"] == ["per_level_action_budget"]
    assert env.reset_calls == 1
    assert len(env.perform_calls) == 2


def test_checkpoint_history_mode_none_masks_previous_actions(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "none.pt", smoke=True, scope="smoke")
    payload = torch.load(checkpoint, weights_only=False)
    payload["training_config"] = {"history_mode": "none"}
    torch.save(payload, checkpoint)
    original_loader = E.model_from_training_checkpoint
    observed = []

    def loader(*args, **kwargs):
        model, saved = original_loader(*args, **kwargs)
        original_step = model.policy_step

        def step(**inputs):
            observed.append(inputs.get("history_keep"))
            return original_step(**inputs)

        monkeypatch.setattr(model, "policy_step", step)
        return model, saved

    monkeypatch.setattr(E, "model_from_training_checkpoint", loader)
    report = E.evaluate_official(
        checkpoint, tmp_path / "report.json", phase="training_families", smoke=True,
        sources=["cd82"], config=E.OfficialEvaluationConfig(max_game_actions=2),
        env_factory=lambda source: FakeEnv(levels=2),
    )
    assert report["protocol"]["history_mode"] == "none"
    assert len(observed) == 2
    assert all(mask is not None and not mask.any() for mask in observed)


@pytest.mark.parametrize(
    "env",
    (
        FakeEnv(levels=1, frame_as_list=True),
        FakeEnv(levels=1, frame_dtype=np.int8, frame_value=15),
    ),
)
def test_integer_public_frames_are_validated_then_normalized(env):
    report = E.run_policy_game(
        FakePolicy(), env,
        source_id=M.source_for("cd82").source_id,
        config=E.OfficialEvaluationConfig(max_actions_per_level=1, max_game_actions=1),
    )
    assert report["whole_game_win"] and report["actions"] == 1
    assert report["failures"] == []


@pytest.mark.parametrize(
    "env",
    (
        FakeEnv(frame_dtype=np.float32),
        FakeEnv(frame_dtype=np.bool_),
        FakeEnv(frame_dtype=np.int8, frame_value=-1),
        FakeEnv(frame_dtype=np.int64, frame_value=16),
    ),
)
def test_invalid_public_frames_are_rejected_before_lossy_cast(env):
    report = E.run_policy_game(
        FakePolicy(), env,
        source_id=M.source_for("cd82").source_id,
        config=E.OfficialEvaluationConfig(max_actions_per_level=1, max_game_actions=1),
    )
    assert report["actions"] == 0
    assert report["failures"][0].startswith("invalid_public_frame:")


def test_smoke_training_family_report_is_stamped_and_cannot_unlock_holdout(tmp_path):
    checkpoint = _checkpoint(tmp_path / "smoke.pt", smoke=True, scope="smoke")
    report_path = tmp_path / "training.json"
    report = E.evaluate_official(
        checkpoint, report_path, phase="training_families", sources=["cd82"], smoke=True,
        env_factory=lambda source: FakeEnv(levels=1),
        config=E.OfficialEvaluationConfig(max_actions_per_level=2, max_game_actions=2),
    )
    assert report["phase_complete"] and report["smoke"] and report["scope"] == "smoke"
    with pytest.raises(ValueError, match="smoke evaluation"):
        E.evaluate_official(
            checkpoint, tmp_path / "heldout.json", phase="heldout", smoke=True,
            training_report=report_path, env_factory=lambda source: FakeEnv(levels=1),
        )


@pytest.mark.parametrize("corruption", ("unsupported", "duplicate"))
def test_holdout_gate_rejects_unsupported_or_duplicate_training_family_results(
    tmp_path, corruption,
):
    checkpoint = _checkpoint(
        tmp_path / "full.pt", smoke=False, scope="full_24_family_experiment",
    )
    config = E.OfficialEvaluationConfig(max_actions_per_level=2, max_game_actions=2)
    gate = _full_gate(
        tmp_path / "gate.json", checkpoint, config, unsupported=corruption == "unsupported",
    )
    if corruption == "duplicate":
        value = json.loads(gate.read_text())
        value["source_ids"][-1] = value["source_ids"][0]
        value["results"][-1]["source_id"] = value["results"][0]["source_id"]
        gate.write_text(json.dumps(value))
    called = False

    def factory(source):
        nonlocal called
        called = True
        return FakeEnv(levels=1)

    with pytest.raises(ValueError, match="duplicate|unsupported"):
        E.evaluate_official(
            checkpoint,
            tmp_path / "heldout.json",
            phase="heldout",
            config=config,
            training_report=gate,
            env_factory=factory,
        )
    assert not called


def test_valid_full_training_report_unlocks_only_policy_evaluation_of_m0r0(tmp_path):
    checkpoint = _checkpoint(
        tmp_path / "full.pt", smoke=False, scope="full_24_family_experiment",
    )
    config = E.OfficialEvaluationConfig(max_actions_per_level=2, max_game_actions=2)
    gate = _full_gate(tmp_path / "gate.json", checkpoint, config)
    requested = []

    def factory(source):
        requested.append(source.source_id)
        return FakeEnv(levels=1)

    report = E.evaluate_official(
        checkpoint,
        tmp_path / "heldout.json",
        phase="heldout",
        config=config,
        training_report=gate,
        env_factory=factory,
    )
    assert requested == [M.HELD_OUT_SOURCE_ID]
    assert report["phase_complete"] and report["games_won"] == 1
    assert report["protocol"]["teacher_or_planner_fallback"] is False
    assert report["protocol"]["forced_level_advance"] is False


@pytest.mark.parametrize("available, expected", (
    ((), "no_legal_public_action"),
    ((0,), "no_legal_public_action"),
    ((1, 9), "invalid_legal_action_id:9"),
    ((None,), "invalid_legal_action_id:None"),
))
def test_official_runner_reports_bad_legal_sets_without_inference_or_actions(available, expected):
    model = FakePolicy()
    env = FakeEnv(levels=1, available_actions=available)
    report = E.run_policy_game(
        model, env, source_id=M.source_for("cd82").source_id,
        config=E.OfficialEvaluationConfig(max_actions_per_level=3, max_game_actions=3),
    )
    assert report["failures"] == [expected]
    assert report["actions"] == 0 and env.perform_calls == [] and model.calls == []
    assert env.reset_calls == 1 and report["voluntary_resets"] == 0


def test_official_evaluation_marks_bad_legal_sets_as_adapter_failure(tmp_path):
    checkpoint = _checkpoint(tmp_path / "smoke.pt", smoke=True, scope="smoke")
    report = E.evaluate_official(
        checkpoint, tmp_path / "training.json", phase="training_families",
        sources=["cd82"], smoke=True,
        env_factory=lambda source: FakeEnv(levels=1, available_actions=(9,)),
        config=E.OfficialEvaluationConfig(max_actions_per_level=2, max_game_actions=2),
    )
    assert report["results"][0]["status"] == "adapter_failure"
    assert report["results"][0]["failures"] == ["invalid_legal_action_id:9"]
    assert report["phase_complete"] is False and report["actions"] == 0
