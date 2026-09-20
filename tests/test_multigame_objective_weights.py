"""Regression checks for rare-event supervision and sparse auxiliary chunks."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from pebby.agent.multigame_model import (
    LossWeights, MultiGameModel, MultiGameModelConfig, collate_game_sequences,
    compute_multigame_loss, load_supervised_game,
)
from test_multigame_model import _write_game


def test_weighted_auxiliaries_match_pixel_and_event_objectives(tmp_path):
    sequence = load_supervised_game(*_write_game(tmp_path, "weighted"))
    batch = collate_game_sequences([sequence])
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).eval()
    indices = torch.tensor([[0, 0], [0, 2]])
    encoded = model.encode_history(batch)
    prediction = model.predict_transitions(
        encoded, batch["executed_action_id"], batch["executed_action_x"],
        batch["executed_action_y"], indices=indices,
    )
    next_frames = batch["next_frames"][0, [0, 2]].long()
    changed = next_frames != batch["frames"][0, [0, 2]]
    assert changed.any() and (~changed).any()
    expected_frame = (
        F.cross_entropy(prediction.next_frame_logits, next_frames, reduction="none")
        * torch.where(changed, 5.0, 1.0)
    ).sum()
    labels = torch.stack([
        batch[key][0, [0, 2]]
        for key in ("next_level_boundary", "next_terminal", "next_won")
    ], dim=-1).float()
    expected_events = F.binary_cross_entropy_with_logits(
        prediction.event_logits, labels, pos_weight=torch.tensor(3.0), reduction="sum",
    )
    loss = compute_multigame_loss(
        model, batch, transition_indices=indices, reduction="sum",
        weights=LossWeights(changed_pixel_weight=5, event_positive_weight=3),
    )
    torch.testing.assert_close(loss.next_frame, expected_frame)
    torch.testing.assert_close(loss.events, expected_events)
    mean_loss = compute_multigame_loss(
        model, batch, transition_indices=indices,
        weights=LossWeights(changed_pixel_weight=5, event_positive_weight=3),
    )
    torch.testing.assert_close(mean_loss.next_frame, loss.next_frame / loss.frame_targets)
    torch.testing.assert_close(mean_loss.events, loss.events / loss.event_targets)


def test_empty_auxiliary_selection_keeps_policy_gradients_finite(tmp_path):
    batch = collate_game_sequences([
        load_supervised_game(*_write_game(tmp_path, "empty")),
    ])
    model = MultiGameModel(MultiGameModelConfig.cpu_test()).eval()
    loss = compute_multigame_loss(model, batch, transition_indices=torch.empty((0, 2), dtype=torch.long))
    assert loss.auxiliary_targets == loss.frame_targets == loss.event_targets == 0
    assert loss.events.item() == loss.next_frame.item() == 0
    assert torch.isfinite(loss.total)
    loss.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


@pytest.mark.parametrize("field,value", [
    ("events", -1), ("next_frame", float("nan")), ("changed_pixel_weight", 0),
    ("event_positive_weight", float("inf")),
])
def test_invalid_loss_weights_rejected(field, value):
    with pytest.raises(ValueError):
        replace(LossWeights(), **{field: value})
