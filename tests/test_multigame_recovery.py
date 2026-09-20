"""Bounded live-recovery collection: wall-clock caps, partial retention, labels.

Synthetic families drive the real ``collect_generated_game`` loop; the
sleeping planner stands in for a family whose solver cannot be interrupted by
a work budget (dc22's constructive builder stalled for hours before the caps).
"""

from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pebby import multigame as M
from pebby import multigame_dataset as D
from pebby.agent.multigame_training import audit_manifest_pair
from tools import collect_multigame_games as collect_cli
from tools import prepare_multigame_dataset as prepare_cli


class _ToyEnv:
    """Deterministic engine: action 1 completes the level, action 2 is a no-op."""

    def __init__(self, levels):
        self.levels = list(levels)
        self.reset()

    def reset(self):
        self.state = "NOT_FINISHED"
        self.level_index = 0
        self.levels_completed = 0
        self.actions = 0

    @property
    def available_actions(self):
        return (1, 2)

    def render(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[0, 0] = self.levels_completed
        frame[0, 1] = self.actions % 16
        # Distinct generated seeds must look like distinct puzzles so the
        # split-safety identity checks see a different game per master seed.
        level = self.levels[min(self.level_index, len(self.levels) - 1)]
        seed = int(level.get("seed", 0)) if isinstance(level, dict) else 0
        digest = hashlib.sha256(str(seed).encode()).digest()
        frame[1, :32] = np.frombuffer(digest, dtype=np.uint8) % 16
        return frame

    def perform(self, action_id, x=None, y=None):
        self.actions += 1
        if action_id == 1:
            self.levels_completed += 1
            if self.levels_completed == len(self.levels):
                self.state = "WIN"
            else:
                self.level_index = self.levels_completed

    def clone(self):
        cloned = object.__new__(type(self))
        cloned.__dict__.update(self.__dict__)
        cloned.levels = list(self.levels)
        return cloned


def _plan(actions=((1, None, None),), reason="toy live plan"):
    return SimpleNamespace(
        actions=list(actions), truncated=False, unsupported=False, exact=True,
        expanded=1, reason=reason,
    )


def _sleeping_search(seconds: float, *, only_after_perturbation: bool = True):
    """Planner that blocks in a plain Python loop (no interruption hook)."""
    calls = []

    def search(env, limit):
        calls.append(env.actions)
        if not only_after_perturbation or env.actions > 0:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                sum(range(1000))
        return _plan(reason="toy recovery after sleeping")

    search.calls = calls
    return search


def _smoke_modules(*, search=None, source="cd82"):
    def generate(seed, difficulty):
        return {
            "seed": seed,
            "difficulty": difficulty,
            "generator_version": 1,
            # A semantic identity the smoke canonical-spec fallback keeps, so
            # distinct seeds are distinct puzzles for overlap/duplicate checks.
            "puzzle_token": f"toy:{seed}",
            "coverage": {"toy": True},
            "omitted_mechanics": ["none invented"],
            "generator_notes": "test declaration",
            "limitations": ["test only"],
        }

    return M.GameModules(
        source=M.source_for(source),
        env=SimpleNamespace(Env=_ToyEnv),
        generate=SimpleNamespace(generate=generate, build_level=lambda spec: dict(spec)),
        plan=SimpleNamespace(search=(lambda env, limit: _plan()) if search is None else search),
        solver_adapter="work_limit",
    )


def _full_modules(*, search=None, levels=2):
    stored = [[[1, None, None]] for _ in range(levels)]

    def generate(seed, difficulty, *, split):
        return {
            "seed": seed,
            "difficulty": difficulty,
            "split": split,
            "generator_version": 1,
            "solution": stored[difficulty - 1],
            "solution_length": 1,
        }

    source = M.source_for("cd82")
    curriculum = tuple(M.CurriculumEntry(d, d - 1, 10) for d in range(1, levels + 1))
    contract = M.FullStandardContract(
        source.source_id, "ready", "fixture-mechanics", "fixture-quality", levels,
        curriculum, tuple((key, f"fixture-{key}") for key in M.FULL_STANDARD_EVIDENCE),
        ("synthetic certified-route fixture",),
    )
    generator = SimpleNamespace(
        DIFFICULTIES=tuple(range(1, levels + 1)),
        generate=generate,
        generate_game=lambda seed, *, split, difficulties=None: [],
        build_level=lambda spec: dict(spec),
        build_game=lambda specs: [dict(spec) for spec in specs],
        validate_full_standard=lambda spec, entry: [],
    )
    return M.GameModules(
        source=source,
        env=SimpleNamespace(Env=_ToyEnv),
        generate=generator,
        plan=SimpleNamespace(search=(lambda env, limit: _plan()) if search is None else search),
        solver_adapter="work_limit",
        full_standard=contract,
    )


@pytest.fixture
def perturb_with_noop(monkeypatch):
    def noop(game, variant, rng):
        raw = game.validate_action(M.Action(2))
        return M.Action(*variant.public_action(*raw.as_tuple())), raw

    monkeypatch.setattr(M, "_sample_random_action", noop)


def _collect(modules, *, rollout, perturber=None, full=True, **overrides):
    kwargs = dict(
        master_seed=3, game_index=0,
        difficulties=tuple(range(1, len(modules.full_standard.curriculum) + 1)) if full else (1,),
        split="train" if full else None,
        require_full_standard=full,
        limits=M.SearchLimits(8, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
        rollout=rollout,
    )
    kwargs.update(overrides)
    return M.collect_generated_game(modules, perturber=perturber, **kwargs)


def test_rollout_options_validate_caps_and_perturbation_mode():
    options = M.RolloutOptions(0.2, recovery_seconds=20, game_seconds=300)
    assert options.recovery_seconds == 20.0 and options.game_seconds == 300.0
    assert options.mode == "mixed_teacher_random"
    assert M.RolloutOptions(0.0).mode == "teacher_only"
    assert M.RolloutOptions(0.5, perturbation="learner").mode == "mixed_teacher_learner"
    with pytest.raises(ValueError, match="recovery_seconds"):
        M.RolloutOptions(0.2, recovery_seconds=0)
    with pytest.raises(ValueError, match="game_seconds"):
        M.RolloutOptions(0.2, game_seconds=-1)
    with pytest.raises(ValueError, match="perturbation"):
        M.RolloutOptions(0.2, perturbation="curious")


def test_sleeping_planner_hits_recovery_timeout_and_retains_partial_trace(perturb_with_noop):
    search = _sleeping_search(5.0)
    started = time.monotonic()
    collected = _collect(
        _full_modules(search=search),
        rollout=M.RolloutOptions(1.0, max_game_steps=8, recovery_seconds=0.2, game_seconds=30),
    )
    elapsed = time.monotonic() - started
    record = collected.record
    assert elapsed < 3.0, "the wall-clock cap must interrupt the uninterruptible planner"
    assert record["status"] == "rollout_failed"
    assert record["recovery_timeouts"] == 1 and record["search_timeouts"] == 1
    assert record["game_timeout"] is False
    assert record["timeout"]["kind"] == M.RECOVERY_TIMEOUT_REASON
    assert record["timeout"]["trigger"] == "after_perturbation"
    assert record["timeout"]["cap_seconds"] == 0.2
    assert record["timeout"]["binding_cap"] == "recovery"
    assert 0.2 <= record["timeout"]["elapsed_seconds"] < 3.0
    assert any(M.RECOVERY_TIMEOUT_REASON in error for error in record["errors"])
    # The perturbation transition before the capped search is kept.
    assert record["steps"] == 1
    assert collected.teacher["route_source"].tolist() == [M.RANDOM_ROUTE_SOURCE]
    assert collected.teacher["source"].tolist() == [M.RANDOM_SOURCE]
    assert collected.public["frames"].shape[0] == 2
    assert record["route_source_counts"] == {M.RANDOM_ROUTE_SOURCE: 1}
    timed_out = [item for item in record["searches"] if item["timed_out"]]
    assert len(timed_out) == 1
    assert timed_out[0]["reason"] == M.RECOVERY_TIMEOUT_REASON
    assert timed_out[0]["route_source"] == M.RECOVERY_ROUTE_SOURCE
    assert timed_out[0]["trigger"] == "after_perturbation"
    assert search.calls == [1]
    assert record["recovery_seconds"] == 0.2 and record["game_seconds"] == 30.0


def test_game_seconds_cap_ends_the_rollout_with_partial_trace(perturb_with_noop):
    # Each recovery blocks 0.15s and stays under the per-search cap; the whole
    # game cap trips on the second loop iteration.
    search = _sleeping_search(0.15)
    collected = _collect(
        _full_modules(search=search, levels=3),
        rollout=M.RolloutOptions(1.0, max_game_steps=64, recovery_seconds=1.0, game_seconds=0.2),
    )
    record = collected.record
    assert record["status"] == "rollout_failed"
    assert record["game_timeout"] is True
    assert record["timeout"]["kind"] == M.GAME_TIMEOUT_REASON
    assert record["timeout"]["binding_cap"] == "game"
    assert record["steps"] >= 1
    assert record["recovery_timeouts"] == 0 and record["search_timeouts"] == 1
    assert record["rollout_wall_clock_seconds"] >= 0.2
    assert any(M.GAME_TIMEOUT_REASON in error for error in record["errors"])
    assert search.calls[-1] >= 1  # the capped search started from a perturbed state


def test_ordinary_family_produces_random_and_live_recovery_transitions(perturb_with_noop):
    calls = []

    def search(env, limit):
        calls.append(env.actions)
        return _plan(reason="fast recovery")

    collected = _collect(
        _full_modules(search=search),
        rollout=M.RolloutOptions(0.5, max_game_steps=8, recovery_seconds=20, game_seconds=300),
    )
    record = collected.record
    assert record["status"] == "won", record["errors"]
    routes = collected.teacher["route_source"].tolist()
    assert M.RANDOM_ROUTE_SOURCE in routes and M.RECOVERY_ROUTE_SOURCE in routes
    assert set(routes) <= {
        M.CERTIFIED_ROUTE_SOURCE, M.RANDOM_ROUTE_SOURCE, M.RECOVERY_ROUTE_SOURCE,
    }
    assert record["recovery_timeouts"] == 0 and record["game_timeout"] is False
    assert record["live_recovery_teacher_actions"] == routes.count(M.RECOVERY_ROUTE_SOURCE)
    assert record["random_steps"] == routes.count(M.RANDOM_ROUTE_SOURCE)
    assert record["learner_steps"] == 0
    assert sum(record["route_source_counts"].values()) == record["steps"]
    assert all(item["elapsed_seconds"] is not None for item in record["searches"]
               if item["route_source"] != M.CERTIFIED_ROUTE_SOURCE)
    assert all(item["deadline_seconds"] == 20.0 or item["deadline_seconds"] is None
               for item in record["searches"])
    assert not record["certified_solution_completed"]


def test_smoke_recovery_after_perturbation_is_labelled_live_recovery(perturb_with_noop):
    collected = _collect(
        _smoke_modules(search=lambda env, limit: _plan()),
        rollout=M.RolloutOptions(1.0, max_game_steps=4, recovery_seconds=5),
        full=False,
    )
    assert collected.record["status"] == "rollout_failed"  # every step is a no-op perturbation
    searches = collected.record["searches"]
    assert searches[0]["route_source"] == M.LEGACY_ROUTE_SOURCE
    assert all(item["route_source"] == M.RECOVERY_ROUTE_SOURCE for item in searches[1:])


class _ScriptedLearner(M.Perturber):
    """Perturber that always proposes a public action and records what it saw."""

    def __init__(self, action_id: int):
        self.action_id = action_id
        self.resets = 0
        self.observations = []

    def reset(self):
        self.resets += 1

    def step(self, frame, legal_action_mask, previous_action, previous_level_boundary):
        self.observations.append((
            frame.shape, legal_action_mask.tolist(),
            None if previous_action is None else previous_action.as_tuple(),
            previous_level_boundary,
        ))
        return M.Action(self.action_id)


def test_learner_perturbation_executes_learner_actions_and_labels_them():
    learner = _ScriptedLearner(2)
    collected = _collect(
        _full_modules(),
        rollout=M.RolloutOptions(0.5, max_game_steps=8, perturbation="learner",
                                 learner_checkpoint="ckpt.pt", recovery_seconds=5),
        perturber=learner,
    )
    record = collected.record
    assert record["status"] == "won", record["errors"]
    assert record["perturbation"] == "learner" and record["learner_checkpoint"] == "ckpt.pt"
    assert record["rollout_mode"] == "mixed_teacher_learner"
    routes = collected.teacher["route_source"].tolist()
    assert M.LEARNER_ROUTE_SOURCE in routes and M.RECOVERY_ROUTE_SOURCE in routes
    assert M.RANDOM_ROUTE_SOURCE not in routes
    assert record["learner_steps"] == routes.count(M.LEARNER_ROUTE_SOURCE) > 0
    assert record["random_steps"] == record["learner_steps"]  # non-teacher source rows
    assert record["actual_trajectory_coverage"]["random_actions"] == 0
    assert record["actual_trajectory_coverage"]["learner_actions"] == record["learner_steps"]
    assert learner.resets == 1
    assert len(learner.observations) == record["steps"]  # observed every transition
    assert learner.observations[0][2] is None
    assert learner.observations[0][0] == (64, 64)


def test_learner_perturbation_illegal_proposal_falls_back_to_random(perturb_with_noop):
    learner = _ScriptedLearner(5)  # never legal in the toy engine
    collected = _collect(
        _full_modules(),
        rollout=M.RolloutOptions(1.0, max_game_steps=2, perturbation="learner",
                                 learner_checkpoint="ckpt.pt"),
        perturber=learner,
    )
    record = collected.record
    assert record["learner_illegal_fallbacks"] >= 1
    assert record["learner_steps"] == 0
    assert set(collected.teacher["route_source"].tolist()) <= {
        M.RANDOM_ROUTE_SOURCE, M.RECOVERY_ROUTE_SOURCE,
    }


def test_learner_perturbation_requires_a_perturber():
    with pytest.raises(ValueError, match="perturber"):
        _collect(
            _full_modules(),
            rollout=M.RolloutOptions(0.5, perturbation="learner", learner_checkpoint="x"),
        )


def test_prepare_retains_timed_out_recovery_and_passes_strict_smoke_audit(
    tmp_path, perturb_with_noop,
):
    search = _sleeping_search(5.0)
    config = D.DatasetPreparationConfig(
        output_root=tmp_path / "prepared",
        train_master_seed=100,
        validation_master_seed=200,
        games=("cd82",),
        smoke=True,
        completed_teacher_games_per_family=1,
        mixed_games_per_family=1,
        mixed_epsilon=1.0,
        difficulties=(1,),
        limits=M.SearchLimits(4, 8),
        outer_generation_attempts=1,
        generator_attempts=1,
        max_candidate_attempts=2,
        max_game_steps=4,
        recovery_seconds=0.2,
        game_seconds=30,
    )
    assert config.rollout_options("mixed").recovery_seconds == 0.2
    assert config.rollout_options("teacher").random_action_probability == 0.0
    started = time.monotonic()
    result = D.prepare_multigame_dataset(
        config,
        modules=(_smoke_modules(search=search),),
        progress=lambda message: None,
        perturber_factory=lambda rollout: None,
    )
    assert time.monotonic() - started < 10.0
    assert result.complete, result.summary["strict_audit_error"]
    cohorts = result.summary["cohorts"]
    for name in ("train-mixed", "validation-mixed"):
        report = cohorts[name]
        assert report["accepted"] == 1 and report["retained_partial"] == 1
        assert report["recovery_timeouts"] == 1 and report["game_timeouts"] == 0
        assert report["recovery_seconds"] == 0.2 and report["perturbation"] == "random"
        source_report = report["sources"][M.source_for("cd82").source_id]
        assert source_report["recovery_timeouts"] == 1
        outcome = source_report["attempts"][-1]
        assert outcome["outcome"] == "accepted_partial"
        assert outcome["timeout"]["kind"] == M.RECOVERY_TIMEOUT_REASON
        assert outcome["route_source_counts"] == {M.RANDOM_ROUTE_SOURCE: 1}
        manifest = json.loads((tmp_path / "prepared" / name / "manifest.json").read_text())
        assert manifest["recovery_timeouts"] == 1
        assert manifest["recovery_seconds"] == 0.2 and manifest["game_seconds"] == 30.0
        assert manifest["route_source_counts"] == {M.RANDOM_ROUTE_SOURCE: 1}
        assert manifest["records"][0]["recovery_timeouts"] == 1
    for name in ("train-teacher", "validation-teacher"):
        assert cohorts[name]["recovery_timeouts"] == 0
        assert cohorts[name]["accepted"] == 1 and cohorts[name]["retained_partial"] == 0
    # The trainer's own audit accepts the produced manifests as-is.
    bundle = audit_manifest_pair(
        result.summary["training_cli_arguments"]["train_manifest"],
        result.summary["training_cli_arguments"]["validation_manifest"],
        smoke=True,
    )
    assert bundle.train.random_rows == 1 and bundle.validation.random_rows == 1
    assert len(bundle.train.failed_records) == 1


def test_prepare_timeout_family_does_not_block_other_families(tmp_path, perturb_with_noop):
    slow = _smoke_modules(search=_sleeping_search(5.0), source="cd82")
    fast = _smoke_modules(search=lambda env, limit: _plan(), source="ft09")
    config = D.DatasetPreparationConfig(
        output_root=tmp_path / "prepared",
        train_master_seed=100,
        validation_master_seed=200,
        games=("cd82", "ft09"),
        smoke=True,
        mixed_games_per_family=2,
        mixed_epsilon=1.0,
        difficulties=(1,),
        limits=M.SearchLimits(4, 8),
        outer_generation_attempts=1,
        generator_attempts=1,
        max_candidate_attempts=2,
        max_game_steps=3,
        recovery_seconds=0.1,
        game_seconds=5,
    )
    started = time.monotonic()
    result = D.prepare_multigame_dataset(
        config, modules=(slow, fast), progress=lambda message: None,
        perturber_factory=lambda rollout: None,
    )
    assert time.monotonic() - started < 10.0
    cohorts = result.summary["cohorts"]
    slow_report = cohorts["train-mixed"]["sources"][M.source_for("cd82").source_id]
    fast_report = cohorts["train-mixed"]["sources"][M.source_for("ft09").source_id]
    assert slow_report["accepted"] == 2 and slow_report["recovery_timeouts"] == 2
    assert slow_report["random_steps"] == 2  # one retained perturbation per capped game
    assert fast_report["accepted"] == 2 and fast_report["recovery_timeouts"] == 0
    # Every step is a perturbation at epsilon 1.0, so the fast family reaches
    # the step cap with recovery re-planned after each one and never capped.
    assert fast_report["random_steps"] == 2 * config.max_game_steps
    for outcome in fast_report["attempts"]:
        assert outcome["outcome"] == "accepted_partial" and outcome["timeout"] is None
        assert outcome["route_source_counts"] == {M.RANDOM_ROUTE_SOURCE: config.max_game_steps}


def test_learner_config_and_cli_flags_are_validated(tmp_path):
    with pytest.raises(ValueError, match="learner_checkpoint"):
        D.DatasetPreparationConfig(
            output_root=tmp_path, train_master_seed=1, validation_master_seed=2,
            perturbation="learner",
        )
    with pytest.raises(ValueError, match="recovery_seconds"):
        D.DatasetPreparationConfig(
            output_root=tmp_path, train_master_seed=1, validation_master_seed=2,
            recovery_seconds=0,
        )
    config = D.DatasetPreparationConfig(
        output_root=tmp_path, train_master_seed=1, validation_master_seed=2,
        perturbation="learner", learner_checkpoint=tmp_path / "ckpt.pt",
    )
    payload = config.parameter_payload()
    assert payload["perturbation"] == "learner"
    assert payload["learner_checkpoint"].endswith("ckpt.pt")
    assert payload["recovery_seconds"] == 20.0 and payload["game_seconds"] == 300.0
    assert D.build_perturber(M.RolloutOptions(0.2)) is None
    assert D.build_perturber(M.RolloutOptions(0.0, perturbation="learner")) is None

    args = prepare_cli.parse_args([
        "--output-root", str(tmp_path / "out"), "--train-master-seed", "1",
        "--validation-master-seed", "2", "--recovery-seconds", "7", "--game-seconds", "0",
    ])
    assert args.recovery_seconds == 7.0 and args.game_seconds == 0.0
    with pytest.raises(SystemExit):
        prepare_cli.parse_args([
            "--output-root", str(tmp_path / "out"), "--train-master-seed", "1",
            "--validation-master-seed", "2", "--perturbation", "learner",
        ])
    with pytest.raises(SystemExit):
        collect_cli.parse_args([
            "--out-dir", str(tmp_path / "out"), "--games", "cd82", "--recovery-seconds", "-1",
        ])
    args = collect_cli.parse_args([
        "--out-dir", str(tmp_path / "out"), "--games", "cd82", "--perturbation", "learner",
        "--learner-checkpoint", str(tmp_path / "ckpt.pt"),
    ])
    assert args.perturbation == "learner"
