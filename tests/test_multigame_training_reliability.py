"""Tiny CPU regressions for sampling, interruption recovery and exact continuation."""
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from pebby.agent import multigame_training as T
from pebby.agent.multigame_model import LossWeights, MultiGameModelConfig
from test_multigame_training import _write_manifest


def _bundle(tmp_path):
    manifests = [
        _write_manifest(tmp_path, f"train{i}", master_seed=10 + i,
                        effective_seed=100 + i, puzzle=f"train-{i}", action_sources=(1, 0, 1, 0))
        for i in range(3)
    ]
    validation = _write_manifest(tmp_path, "validation", master_seed=20,
                                  effective_seed=200, puzzle="validation")
    return T.audit_manifest_pair(manifests, validation, smoke=True)


def _config(**kwargs):
    return T.TrainingConfig(**{
        "epochs": 2, "model": MultiGameModelConfig.cpu_test(), "device": "cpu",
        "chunk_steps": 3, "update_mode": "game", "history_dropout": 0.5,
        "auxiliary_transitions_per_chunk": 1, "checkpoint_every_games": 1,
        "loss": LossWeights(next_frame=0.1, events=0.1),
        "closed_loop_games": 1, "closed_loop_train_games": 0,
        **kwargs,
    })


def _mock_evaluation(monkeypatch):
    # Consume RNG as a real stochastic evaluator could, to test full restoration.
    def offline(*args, **kwargs):
        return {"policy_action_loss": random.random() + float(np.random.random()) + float(torch.rand(()))}
    monkeypatch.setattr(T, "evaluate_generated_offline", offline)
    monkeypatch.setattr(T, "evaluate_generated_closed_loop", lambda *args, **kwargs: {
        "games_won": 0, "levels_completed": 0, "games": [], "actions": 0,
    })


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("update_mode", ["game", "chunk"])
def test_stop_resume_is_bit_exact_mid_epoch(tmp_path, monkeypatch, update_mode):
    bundle = _bundle(tmp_path)
    config = _config(update_mode=update_mode)
    _mock_evaluation(monkeypatch)
    uninterrupted = T.train_multigame(bundle, tmp_path / "uninterrupted", config)
    stop = T.StopRequest()
    original = T.train_game
    calls = 0

    def stop_after_second(*args, **kwargs):
        nonlocal calls
        stats = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            stop.request()
        return stats

    monkeypatch.setattr(T, "train_game", stop_after_second)
    interrupted = T.train_multigame(bundle, tmp_path / "resumed", config, stop_request=stop)
    assert interrupted.stopped and not interrupted.best_checkpoint.exists()
    partial = T.load_training_checkpoint(interrupted.latest_checkpoint)
    assert partial["epoch_progress"]["cursor"] == 2
    assert len(partial["epoch_progress"]["order"]) == 3
    monkeypatch.setattr(T, "train_game", original)
    resumed = T.train_multigame(bundle, tmp_path / "resumed", config, resume=interrupted.latest_checkpoint)
    full = T.load_training_checkpoint(uninterrupted.latest_checkpoint)
    recovered = T.load_training_checkpoint(resumed.latest_checkpoint)
    for key in ("model_state", "optimizer_state", "rng_state", "logs", "global_step", "best_score"):
        _assert_tree_equal(full[key], recovered[key])


def test_stop_during_validation_replays_evaluation_not_training(tmp_path, monkeypatch):
    bundle, config = _bundle(tmp_path), _config()
    _mock_evaluation(monkeypatch)
    uninterrupted = T.train_multigame(bundle, tmp_path / "full", config)
    offline = T.evaluate_generated_offline
    stop = T.StopRequest()

    def interrupt_evaluation(*args, **kwargs):
        offline(*args, **kwargs)  # Advance every RNG before interruption.
        stop.request()
        stop.check()

    monkeypatch.setattr(T, "evaluate_generated_offline", interrupt_evaluation)
    partial = T.train_multigame(bundle, tmp_path / "resume", config, stop_request=stop)
    state = T.load_training_checkpoint(partial.latest_checkpoint)
    assert state["epoch_progress"]["cursor"] == 3
    assert state["global_step"] == 3
    monkeypatch.setattr(T, "evaluate_generated_offline", offline)
    resumed = T.train_multigame(bundle, tmp_path / "resume", config, resume=partial.latest_checkpoint)
    for key in ("model_state", "optimizer_state", "rng_state", "logs", "global_step"):
        _assert_tree_equal(T.load_training_checkpoint(uninterrupted.latest_checkpoint)[key],
                           T.load_training_checkpoint(resumed.latest_checkpoint)[key])


@pytest.mark.parametrize("failure", ["latest", "best", "before_pointer", "after_pointer"])
def test_atomic_checkpoint_pair_survives_each_interrupted_promotion(tmp_path, monkeypatch, failure):
    first = {"format": T.TRAINING_FORMAT, "epoch": 0, "best_score": (0, 0, -1.0)}
    second = {"format": T.TRAINING_FORMAT, "epoch": 1, "best_score": (1, 1, -1.0)}
    T._commit_checkpoints(tmp_path, first, promoted=True)
    save, replace_file = T._atomic_torch_save, T.os.replace

    def fail_save(path, payload):
        save(path, payload)
        if path.name == f"{failure}.pt":
            raise InterruptedError("injected interruption")

    def fail_replace(source, destination):
        if Path(destination).name == ".checkpoint-current" and failure == "before_pointer":
            raise InterruptedError("injected interruption")
        replace_file(source, destination)
        if Path(destination).name == ".checkpoint-current" and failure == "after_pointer":
            raise InterruptedError("injected interruption")

    monkeypatch.setattr(T, "_atomic_torch_save", fail_save)
    monkeypatch.setattr(T.os, "replace", fail_replace)
    with pytest.raises(InterruptedError):
        T._commit_checkpoints(tmp_path, second, promoted=True)
    latest = T.load_training_checkpoint(tmp_path / "latest.pt")
    best = T.load_training_checkpoint(tmp_path / "best.pt")
    assert latest["best_score"] == best["best_score"]
    assert latest["epoch"] == (1 if failure == "after_pointer" else 0)
    monkeypatch.setattr(T, "_atomic_torch_save", save)
    monkeypatch.setattr(T.os, "replace", replace_file)
    T._commit_checkpoints(tmp_path, second, promoted=True)
    assert T.load_training_checkpoint(tmp_path / "latest.pt")["epoch"] == 1
    assert len(list((tmp_path / ".checkpoint-generations").iterdir())) == 2


def test_resume_binds_order_and_rejects_old_recipe(tmp_path, monkeypatch):
    bundle, config = _bundle(tmp_path), _config()
    _mock_evaluation(monkeypatch)
    stop = T.StopRequest()
    stop.request()
    partial = T.train_multigame(bundle, tmp_path / "run", config, stop_request=stop)
    reordered = replace(bundle, train=replace(bundle.train, games=tuple(reversed(bundle.train.games))))
    assert reordered.hashes == bundle.hashes
    with pytest.raises(ValueError, match="ordered game identity"):
        T.train_multigame(reordered, tmp_path / "run", config, resume=partial.latest_checkpoint)
    legacy = T.load_training_checkpoint(partial.latest_checkpoint)
    legacy.pop("training_recipe")
    torch.save(legacy, partial.latest_checkpoint)
    with pytest.raises(ValueError, match="initialize-from"):
        T.train_multigame(bundle, tmp_path / "run", config, resume=partial.latest_checkpoint)
    initialized = T.train_multigame(bundle, tmp_path / "new", config,
                                   initialize_from=partial.latest_checkpoint, stop_request=stop)
    source = T.load_training_checkpoint(initialized.latest_checkpoint)["initialize_from"]
    assert source["sha256"] == T.sha256_file(partial.latest_checkpoint)
    continued = T.train_multigame(bundle, tmp_path / "new", config,
                                 resume=initialized.latest_checkpoint)
    assert T.load_training_checkpoint(continued.latest_checkpoint)["initialize_from"] == source


def test_whole_game_sampler_has_uniform_tail_inclusion():
    config = _config(chunk_steps=64, auxiliary_transitions_per_chunk=8)
    np.random.seed(7)
    counts = np.zeros(68, dtype=int)
    for _ in range(2000):
        samples = T._game_auxiliary_indices(68, config, torch.device("cpu"))
        assert sum(map(len, samples)) == 12
        for chunk, indices in enumerate(samples):
            counts[indices[:, 1].numpy() + 64 * chunk] += 1
    # Old sampler selected all four tail rows on every draw, >5x overrepresented.
    assert abs(float(counts[-4:].mean() / counts[:64].mean()) - 1.0) < 0.08


def test_history_provenance_and_saved_config_cli(tmp_path, monkeypatch):
    from tools.train_multigame import parse_args
    bundle, config = _bundle(tmp_path), _config(history_dropout=0.75)
    stop = T.StopRequest()
    stop.request()
    partial = T.train_multigame(bundle, tmp_path / "run", config, stop_request=stop)
    saved = T.load_training_checkpoint(partial.latest_checkpoint)
    assert saved["model_config"]["history_dropout"] == 0.75
    args = parse_args(["--resume", str(partial.latest_checkpoint), "--epochs", "9", "--device", "cpu"])
    assert args.saved_config == replace(config, epochs=9)
    assert args.train_manifest == [str(path) for path in bundle.train.manifest_paths]
    assert args.out_dir == tmp_path / "run"
    with pytest.raises(SystemExit):
        parse_args(["--resume", str(partial.latest_checkpoint), "--frame-weight", "0"])


def test_transition_positive_event_metrics():
    accumulator = T._empty_transition_accumulator()
    current = torch.zeros((3, 64, 64), dtype=torch.long)
    logits = torch.zeros((3, 16, 64, 64))
    T._accumulate_transition_metrics(
        accumulator, logits, torch.tensor([[1., -1., -1.], [1., -1., -1.], [-1., -1., -1.]]),
        current, current, torch.tensor([[True, False, False], [False, False, False], [True, False, False]]),
    )
    report = T._transition_summary(accumulator)["positive_events"]
    assert report["level_boundary"] == {
        "true_positive": 1, "false_positive": 1, "false_negative": 1,
        "support": 2, "precision": 0.5, "recall": 0.5,
    }
    assert report["terminal"]["recall"] is None


@pytest.mark.parametrize("diagnostic", [False, True])
def test_regular_selection_honors_no_history_mode(tmp_path, monkeypatch, diagnostic):
    bundle = _bundle(tmp_path)
    seen = []

    def offline(*args, **kwargs):
        seen.append(("offline", kwargs["history_free"]))
        return {"policy_action_loss": 1.0}

    def closed(*args, **kwargs):
        seen.append(("closed", kwargs["history_free"]))
        return {"games_won": 0, "levels_completed": 0, "games": [], "actions": 0}

    monkeypatch.setattr(T, "evaluate_generated_offline", offline)
    monkeypatch.setattr(T, "evaluate_generated_closed_loop", closed)
    result = T.train_multigame(bundle, tmp_path / "run", _config(
        epochs=1, history_mode="none", history_dropout=0, history_free_diagnostic=diagnostic,
        closed_loop_train_games=1, loss=LossWeights(next_frame=0, events=0),
    ))
    assert seen == [("offline", True), ("closed", True), ("closed", True)]
    for key in ("generated_validation_offline", "generated_validation_closed_loop", "generated_train_closed_loop"):
        assert result.logs[0][key + "_history_free"] == (result.logs[0][key] if diagnostic else None)


def test_offline_transition_metrics_include_late_game_events(tmp_path):
    from pebby.agent.multigame_model import MultiGameModel
    train = _write_manifest(tmp_path, "train", puzzle="train", effective_seed=100,
                            action_sources=(1,) * 40)
    validation = _write_manifest(tmp_path, "val", puzzle="val", effective_seed=200, master_seed=2,
                                 action_sources=(1,) * 40)
    bundle = T.audit_manifest_pair(train, validation, smoke=True)
    model = MultiGameModel(MultiGameModelConfig.cpu_test())
    report = T.evaluate_generated_offline(model, bundle.validation.games, chunk_steps=7,
                                         transition_limit_per_game=2, device=torch.device("cpu"))
    transition = report["transition"]["teacher"]
    assert transition["rows"] == 2
    # Only the last transition has these events; prefix sampling missed them.
    assert transition["positive_events"]["terminal"]["support"] == 1
    assert transition["positive_events"]["won"]["support"] == 1


def test_min_click_region_coverage_rejects_sparse_labels_before_training(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="stored-region coverage"):
        T.train_multigame(bundle, tmp_path / "run", _config(min_click_region_coverage=0.9))
    assert not (tmp_path / "run").exists()
    assert T._training_signature(_config()) == T._training_signature(_config(min_click_region_coverage=1))
    for invalid in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="min_click_region_coverage"):
            _config(min_click_region_coverage=invalid)


@pytest.mark.parametrize("spelling", ["symlink_parent", "dotdot"])
def test_resume_accepts_equivalent_parent_path(tmp_path, spelling):
    bundle, config = _bundle(tmp_path), _config()
    stop = T.StopRequest()
    stop.request()
    partial = T.train_multigame(bundle, tmp_path / "run", config, stop_request=stop)
    if spelling == "symlink_parent":
        parent = tmp_path / "run-alias"
        parent.symlink_to(tmp_path / "run", target_is_directory=True)
        resume = parent / "latest.pt"
    else:
        resume = tmp_path / "run" / ".." / "run" / "latest.pt"
    resumed = T.train_multigame(bundle, resume.parent, config, resume=resume, stop_request=stop)
    assert resumed.stopped and resumed.latest_checkpoint == partial.latest_checkpoint


@pytest.mark.parametrize("damage", ["flat_latest", "absolute_latest_alias", "missing_best_alias", "redirected_pointer"])
def test_resume_rejects_broken_generation_layout_before_training(tmp_path, monkeypatch, damage):
    bundle, config = _bundle(tmp_path), _config()
    stop = T.StopRequest()
    stop.request()
    partial = T.train_multigame(bundle, tmp_path / "run", config, stop_request=stop)
    saved = T.load_training_checkpoint(partial.latest_checkpoint)
    if damage == "flat_latest":
        partial.latest_checkpoint.unlink()
        torch.save(saved, partial.latest_checkpoint)
    elif damage == "absolute_latest_alias":
        resolved = partial.latest_checkpoint.resolve()
        partial.latest_checkpoint.unlink()
        partial.latest_checkpoint.symlink_to(resolved)
    elif damage == "missing_best_alias":
        partial.best_checkpoint.unlink()
    else:
        # This still resolves to a readable checkpoint but escapes the committed directory.
        export = tmp_path / "export"
        export.mkdir()
        torch.save(saved, export / "latest.pt")
        pointer = tmp_path / "run" / ".checkpoint-current"
        pointer.unlink()
        pointer.symlink_to(export, target_is_directory=True)
    monkeypatch.setattr(T, "train_game", lambda *args, **kwargs: pytest.fail("must reject before any update"))
    with pytest.raises(ValueError, match="checkpoint (layout|generation pointer)"):
        T.train_multigame(bundle, tmp_path / "run", config, resume=partial.latest_checkpoint)


def test_no_history_contract_checks_inference_capability(monkeypatch):
    from pebby.agent.multigame_model import MultiGameModel
    model = MultiGameModel(MultiGameModelConfig.cpu_test())
    monkeypatch.setattr(model, "policy_step", lambda frame: None)
    with pytest.raises(ValueError, match="policy_step/encode_history"):
        T._check_model_capabilities(model, _config(history_mode="none", history_dropout=0))
