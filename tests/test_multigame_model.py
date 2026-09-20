"""Causal-boundary tests for the generic whole-game visual model."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pebby.agent.multigame_model import (
    ACTION_COUNT,
    CLICK_ACTION,
    FRAME_SIZE,
    MODEL_INPUT_KEYS,
    MultiGameModel,
    MultiGameModelConfig,
    MultiGameSequenceDataset,
    collate_game_sequences,
    compute_multigame_loss,
    load_multigame_checkpoint,
    load_public_game,
    load_supervised_game,
    save_multigame_checkpoint,
)


def _write_game(
    root: Path,
    name: str,
    *,
    actions: tuple[int, ...] = (1, 6, 7),
    click_xy: tuple[int, int] = (59, 37),
) -> tuple[Path, Path]:
    steps = len(actions)
    frames = np.zeros((steps + 1, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    for index in range(steps + 1):
        frames[index, index: index + 2, (3 * index): (3 * index + 2)] = index + 1
    legal = np.zeros((steps + 1, ACTION_COUNT), dtype=np.bool_)
    legal[:-1, 1:] = True
    x = np.full(steps, -1, dtype=np.int16)
    y = np.full(steps, -1, dtype=np.int16)
    for index, action in enumerate(actions):
        if action == CLICK_ACTION:
            x[index], y[index] = click_xy

    # Put a level boundary before the final decision and another on the final
    # transition.  This exercises previous-action carry across a boundary.
    boundary = np.zeros(steps, dtype=np.bool_)
    if steps > 1:
        boundary[-2:] = True
    else:
        boundary[-1] = True
    completed = np.concatenate((np.array([0]), np.cumsum(boundary))).astype(np.int16)
    terminal = np.zeros(steps + 1, dtype=np.bool_)
    terminal[-1] = True
    won = terminal.copy()
    public = {
        "frames": frames,
        "legal_action_mask": legal,
        "state": np.asarray(["NOT_FINISHED"] * steps + ["WIN"], dtype="<U16"),
        "level_index": completed.copy(),
        "levels_completed": completed,
        "terminal": terminal,
        "won": won,
        "action_id": np.asarray(actions, dtype=np.int8),
        "action_x": x,
        "action_y": y,
        "level_boundary": boundary,
    }
    teacher = {
        "target_action_id": np.asarray(actions, dtype=np.int8),
        "target_action_x": x.copy(),
        "target_action_y": y.copy(),
        "source": np.ones(steps, dtype=np.int8),
        "plan_level": completed[:-1].copy(),
        "plan_offset": np.arange(steps, dtype=np.int16),
        "plan_length": np.full(steps, steps, dtype=np.int16),
    }
    public_path = root / f"{name}.public.npz"
    teacher_path = root / f"{name}.teacher.npz"
    np.savez(public_path, **public)
    np.savez(teacher_path, **teacher)
    return public_path, teacher_path


@pytest.fixture()
def games(tmp_path):
    first = _write_game(tmp_path, "first")
    second = _write_game(tmp_path, "second", actions=(2, 7))
    return (
        load_supervised_game(*first),
        load_supervised_game(*second),
        first,
    )


@pytest.fixture(params=["v1", "v2"])
def model(request):
    torch.manual_seed(4)
    return MultiGameModel(MultiGameModelConfig.cpu_test(architecture=request.param)).eval()


def test_loader_builds_strict_n_plus_one_causal_timeline_and_keeps_teacher_separate(games):
    first, _, paths = games
    public_only = load_public_game(paths[0])
    assert len(first) == 3
    assert not public_only.supervised
    assert first.frames.shape == first.next_frames.shape == (3, 64, 64)
    assert first.previous_action_id.tolist() == [-1, 1, 6]
    assert first.previous_action_x.tolist() == [-1, -1, 59]
    assert first.previous_action_y.tolist() == [-1, -1, 37]
    assert first.previous_level_boundary.tolist() == [False, False, True]
    assert first.next_level_boundary.tolist() == [False, True, True]
    assert first.target_action_id.tolist() == [1, 6, 7]


def test_loader_rejects_misaligned_teacher_illegal_actions_and_private_public_fields(tmp_path):
    public_path, teacher_path = _write_game(tmp_path, "bad")
    with np.load(teacher_path, allow_pickle=False) as payload:
        teacher = {key: payload[key] for key in payload.files}
    teacher["target_action_id"] = teacher["target_action_id"][:-1]
    np.savez(teacher_path, **teacher)
    with pytest.raises(ValueError, match="align"):
        load_supervised_game(public_path, teacher_path)

    public_path, _ = _write_game(tmp_path, "illegal")
    with np.load(public_path, allow_pickle=False) as payload:
        public = {key: payload[key] for key in payload.files}
    public["action_id"][0] = 0
    np.savez(public_path, **public)
    with pytest.raises(ValueError, match="1..7"):
        load_public_game(public_path)

    public_path, _ = _write_game(tmp_path, "leak")
    with np.load(public_path, allow_pickle=False) as payload:
        public = {key: payload[key] for key in payload.files}
    public["target_action_id"] = public["action_id"].copy()
    np.savez(public_path, **public)
    with pytest.raises(ValueError, match="private/teacher"):
        load_public_game(public_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("target_action_id", np.asarray([1.5, 6.0, 7.0], dtype=np.float32)),
        ("target_action_x", np.asarray([False, False, True], dtype=np.bool_)),
    ),
)
def test_loader_rejects_teacher_float_and_bool_actions_before_cast(tmp_path, field, replacement):
    public_path, teacher_path = _write_game(tmp_path, "typed")
    with np.load(teacher_path, allow_pickle=False) as payload:
        teacher = {key: payload[key] for key in payload.files}
    teacher[field] = replacement
    np.savez(teacher_path, **teacher)
    with pytest.raises(ValueError, match="integer dtype"):
        load_supervised_game(public_path, teacher_path)


def test_exact_teacher_rows_are_bound_to_the_executed_public_trace(tmp_path):
    public_path, teacher_path = _write_game(tmp_path, "wrong-teacher")
    with np.load(teacher_path, allow_pickle=False) as payload:
        teacher = {key: payload[key] for key in payload.files}
    teacher["target_action_id"][0] = 2
    np.savez(teacher_path, **teacher)
    with pytest.raises(ValueError, match="do not match"):
        load_supervised_game(public_path, teacher_path)


def test_collation_right_pads_games_without_cross_game_history(games):
    first, second, _ = games
    dataset = MultiGameSequenceDataset((first, second))
    batch = collate_game_sequences((dataset[0], dataset[1]))
    assert batch["padding_mask"].tolist() == [[False, False, False], [False, False, True]]
    assert batch["previous_action_id"][:, 0].tolist() == [-1, -1]
    assert batch["previous_level_boundary"][:, 0].tolist() == [False, False]
    assert set(MODEL_INPUT_KEYS).issubset(batch)
    assert "source" not in batch and "source_id" not in batch and "plan_level" not in batch


def test_policy_is_temporally_causal_and_cannot_read_targets_actions_or_next_frames(games, model):
    batch = collate_game_sequences((games[0],))
    clean = model(batch)

    poisoned = {key: value.clone() for key, value in batch.items()}
    poisoned["target_action_id"].fill_(3)
    poisoned["target_action_x"].fill_(-1)
    poisoned["target_action_y"].fill_(-1)
    poisoned["executed_action_id"].fill_(5)
    poisoned["next_frames"].fill_(15)
    poisoned["source_id"] = torch.full_like(poisoned["target_action_id"], 999)
    after_labels = model(poisoned)
    assert torch.equal(clean.action_logits, after_labels.action_logits)
    assert torch.equal(clean.click_logits, after_labels.click_logits)

    future_changed = {key: value.clone() for key, value in batch.items()}
    future_changed["frames"][:, 2].fill_(15)
    after_future = model(future_changed)
    assert torch.equal(clean.action_logits[:, :2], after_future.action_logits[:, :2])
    assert torch.equal(clean.click_logits[:, :2], after_future.click_logits[:, :2])


def test_illegal_action_ids_are_masked_and_click_readout_has_exact_64x64_destinations(games, model):
    batch = collate_game_sequences((games[0],))
    batch["legal_action_mask"][:, :, :] = False
    batch["legal_action_mask"][:, :, 2] = True
    batch["legal_action_mask"][:, :, 7] = True
    output = model(batch)
    assert torch.isneginf(output.action_logits[..., 0]).all()
    assert torch.isneginf(output.action_logits[..., 1]).all()
    assert torch.isfinite(output.action_logits[..., 2]).all()
    assert torch.isfinite(output.action_logits[..., 7]).all()
    assert output.click_logits.shape == (1, 3, 64, 64)

    exact = torch.full((4, 64, 64), -10.0)
    destinations = ((0, 0), (63, 0), (0, 63), (59, 37))
    for row, (x, y) in enumerate(destinations):
        exact[row, y, x] = 10.0
    x, y = model.decode_click(exact)
    assert list(zip(x.tolist(), y.tolist())) == list(destinations)


def test_level_boundary_retains_online_memory_and_new_game_resets_it(games, model):
    batch = collate_game_sequences((games[0],))

    def step(index, memory):
        return model.policy_step(
            frame=batch["frames"][:, index],
            previous_action_id=batch["previous_action_id"][:, index],
            previous_action_x=batch["previous_action_x"][:, index],
            previous_action_y=batch["previous_action_y"][:, index],
            previous_level_boundary=batch["previous_level_boundary"][:, index],
            terminal=batch["terminal"][:, index],
            won=batch["won"][:, index],
            legal_action_mask=batch["legal_action_mask"][:, index],
            memory=memory,
        )

    new_game = model.initial_memory(1)
    assert torch.count_nonzero(new_game) == 0
    _, after_first = step(0, new_game)
    _, after_second = step(1, after_first)
    assert batch["previous_level_boundary"][0, 2]
    _, carried_across_boundary = step(2, after_second)
    _, reset_at_same_public_state = step(2, model.initial_memory(1))
    assert not torch.equal(carried_across_boundary, reset_at_same_public_state)
    assert torch.count_nonzero(carried_across_boundary) > 0


def test_separate_action_conditioned_predictor_does_not_change_policy(games, model):
    batch = collate_game_sequences((games[0],))
    encoded = model.encode_history(batch)
    policy_before = model.policy_from_history(encoded, batch["legal_action_mask"])
    first = model.predict_transitions(
        encoded,
        batch["executed_action_id"], batch["executed_action_x"], batch["executed_action_y"],
        indices=torch.tensor([[0, 0]]),
    )
    alternate_id = batch["executed_action_id"].clone()
    alternate_id[0, 0] = 7
    second = model.predict_transitions(
        encoded, alternate_id, batch["executed_action_x"], batch["executed_action_y"],
        indices=torch.tensor([[0, 0]]),
    )
    policy_after = model.policy_from_history(encoded, batch["legal_action_mask"])
    assert torch.equal(policy_before.action_logits, policy_after.action_logits)
    assert first.next_frame_logits.shape == (1, 16, 64, 64)
    assert first.event_logits.shape == (1, 3)
    assert not torch.equal(first.next_frame_logits, second.next_frame_logits)


def test_minimal_full_loss_backward_uses_click_only_for_click_targets(games):
    torch.manual_seed(8)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()
    batch = collate_game_sequences((games[0], games[1]))
    loss = compute_multigame_loss(model, batch)
    assert loss.policy_targets == 5
    assert loss.click_targets == 1
    assert loss.auxiliary_targets == 5
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.action_head.weight.grad is not None
    assert model.click_query.weight.grad is not None
    assert model.frame_predictor[-1].weight.grad is not None
    assert model.event_predictor[-1].weight.grad is not None


def test_all_missing_policy_labels_keep_auxiliary_loss_and_backward_finite(tmp_path):
    public_path, teacher_path = _write_game(tmp_path, "random-only")
    with np.load(teacher_path, allow_pickle=False) as payload:
        teacher = {key: payload[key] for key in payload.files}
    teacher["target_action_id"].fill(-1)
    teacher["target_action_x"].fill(-1)
    teacher["target_action_y"].fill(-1)
    teacher["source"].fill(0)
    np.savez(teacher_path, **teacher)
    sequence = load_supervised_game(public_path, teacher_path)
    batch = collate_game_sequences((sequence,))
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).train()
    loss = compute_multigame_loss(model, batch)
    assert loss.policy_targets == loss.click_targets == 0
    assert loss.action.item() == loss.click.item() == 0.0
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.frame_predictor[-1].weight.grad is not None


def test_checkpoint_roundtrip_preserves_config_weights_and_metadata(tmp_path, games):
    torch.manual_seed(2)
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).eval()
    batch = collate_game_sequences((games[0],))
    expected = model(batch)
    path = save_multigame_checkpoint(model, tmp_path / "model.pt", metadata={"epoch": 3})
    restored, metadata = load_multigame_checkpoint(path)
    restored.eval()
    actual = restored(batch)
    assert restored.config == model.config
    assert metadata == {"epoch": 3}
    assert torch.equal(expected.action_logits, actual.action_logits)
    assert torch.equal(expected.click_logits, actual.click_logits)
