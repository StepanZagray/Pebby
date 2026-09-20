"""Architecture v2, history masking, loss reductions and v1 compatibility."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pebby.agent.multigame_model import (
    ACTION_COUNT,
    CLICK_ACTION,
    FRAME_SIZE,
    GRID_SIZE,
    GameSequence,
    LossWeights,
    MultiGameModel,
    MultiGameModelConfig,
    collate_game_sequences,
    compute_multigame_loss,
    load_multigame_checkpoint,
    load_supervised_game,
    save_multigame_checkpoint,
)

REPO = Path(__file__).resolve().parents[1]
FROZEN_V1 = REPO / "artifacts/multigame-v1/frozen-canonical-recovery-v1.pt"
FROZEN_GAME = REPO / "artifacts/multigame-smoke-20260918-resume/games/cd82-000000.npz"
FROZEN_TEACHER = REPO / "artifacts/multigame-smoke-20260918-resume/teacher/cd82-000000.npz"
# Written by the pre-change code (see the task log); used when present.
PRE_CHANGE_REFERENCE = Path("/tmp/pebby-v1-reference/reference.pt")

# Compact reference computed by the pre-change code on FROZEN_GAME with FROZEN_V1
# (25 steps, whole game, eval mode).  These pin v1 permanently, independently of
# the /tmp snapshot.
V1_REFERENCE = {
    "steps": 25,
    "finite_action_sum": 16.52920150756836,
    "action_argmax": [1, 6, 4, 4, 4, 5, 5, 2, 5, 5, 5, 5, 5, 5, 6, 3, 5, 6, 2, 3, 3, 6, 3, 1, 5],
    "click_xy": [(6, 6), (43, 4), (43, 4), (37, 4), (45, 45), (31, 4), (49, 4), (31, 4), (9, 5),
                 (30, 36), (10, 5), (40, 4), (32, 20), (34, 4), (46, 4), (33, 3), (50, 39), (46, 4),
                 (46, 4), (41, 45), (32, 57), (34, 4), (52, 4), (32, 41), (13, 39)],
    "click_abs_sum": 1357029.25,
    "click_max": 18.37488555908203,
    "losses": (2.6868197917938232, 1.8155156373977661, 0.7872775197029114,
               0.28141626715660095, 0.13672639429569244),
}


def _public_batch(steps: int = 6, *, seed: int = 0, batch: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, 16, (batch, steps, FRAME_SIZE, FRAME_SIZE), generator=generator, dtype=torch.uint8)
    legal = torch.zeros(batch, steps, ACTION_COUNT, dtype=torch.bool)
    legal[..., 1:] = True
    previous_id = torch.full((batch, steps), -1, dtype=torch.long)
    previous_x = torch.full((batch, steps), -1, dtype=torch.long)
    previous_y = torch.full((batch, steps), -1, dtype=torch.long)
    # A real history: a click at step 0 then directional moves.
    previous_id[:, 1] = CLICK_ACTION
    previous_x[:, 1] = 12
    previous_y[:, 1] = 40
    previous_id[:, 2:] = 3
    return {
        "frames": frames,
        "previous_action_id": previous_id,
        "previous_action_x": previous_x,
        "previous_action_y": previous_y,
        "previous_level_boundary": torch.zeros(batch, steps, dtype=torch.bool),
        "terminal": torch.zeros(batch, steps, dtype=torch.bool),
        "won": torch.zeros(batch, steps, dtype=torch.bool),
        "legal_action_mask": legal,
        "padding_mask": torch.zeros(batch, steps, dtype=torch.bool),
    }


def _marker_game(steps: int = 8, *, seed: int = 3) -> GameSequence:
    """Frames with one 3x3 marker; even steps move by quadrant, odd steps click it."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((steps, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    target_id = np.zeros(steps, dtype=np.int64)
    target_x = np.full(steps, -1, dtype=np.int64)
    target_y = np.full(steps, -1, dtype=np.int64)
    for step in range(steps):
        cx, cy = (int(v) for v in rng.integers(4, 60, size=2))
        frames[step, cy - 1:cy + 2, cx - 1:cx + 2] = 9
        if step % 2 == 0:
            target_id[step] = 1 + int(cx >= 32) + 2 * int(cy >= 32)
        else:
            target_id[step] = CLICK_ACTION
            target_x[step], target_y[step] = cx, cy
    legal = np.zeros((steps, ACTION_COUNT), dtype=np.bool_)
    legal[:, 1:] = True
    previous_id = np.full(steps, -1, dtype=np.int64)
    previous_x = previous_id.copy()
    previous_y = previous_id.copy()
    previous_id[1:], previous_x[1:], previous_y[1:] = target_id[:-1], target_x[:-1], target_y[:-1]
    zeros = lambda: np.zeros(steps, dtype=np.bool_)  # noqa: E731
    return GameSequence(
        frames=frames,
        previous_action_id=previous_id, previous_action_x=previous_x, previous_action_y=previous_y,
        previous_level_boundary=zeros(), terminal=zeros(), won=zeros(), legal_action_mask=legal,
        executed_action_id=target_id.copy(), executed_action_x=target_x.copy(), executed_action_y=target_y.copy(),
        next_frames=np.roll(frames, -1, axis=0), next_level_boundary=zeros(), next_terminal=zeros(), next_won=zeros(),
        target_action_id=target_id, target_action_x=target_x, target_action_y=target_y,
        target_valid=np.ones(steps, dtype=np.bool_), action_source=np.ones(steps, dtype=np.int64),
    )


def _model(architecture: str, *, seed: int = 0, train: bool = False) -> MultiGameModel:
    torch.manual_seed(seed)
    model = MultiGameModel(MultiGameModelConfig.cpu_test(architecture=architecture))
    return model.train() if train else model.eval()


# --- config -----------------------------------------------------------------

def test_config_round_trips_and_old_checkpoint_dicts_default_the_new_keys(tmp_path):
    config = MultiGameModelConfig(architecture="v2", history_dropout=0.5, hidden_dim=64)
    assert MultiGameModelConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["architecture"] == "v2"
    assert config.to_dict()["history_dropout"] == 0.5

    old = {"palette_dim": 24, "spatial_channels": 32, "conv_channels": 96, "hidden_dim": 256,
           "action_dim": 32, "coordinate_dim": 24, "dropout": 0.1}
    restored = MultiGameModelConfig.from_dict(old)
    assert restored.architecture == "v1"
    assert restored.history_dropout == 0.0
    assert restored == MultiGameModelConfig()

    with pytest.raises(ValueError):
        MultiGameModelConfig(architecture="v3")
    with pytest.raises(ValueError):
        MultiGameModelConfig(history_dropout=1.5)

    # Model checkpoints carry the new keys and load them back.
    model = _model("v2")
    path = save_multigame_checkpoint(model, tmp_path / "v2.pt")
    restored_model, _ = load_multigame_checkpoint(path)
    assert restored_model.config == model.config


# --- history_keep -----------------------------------------------------------

@pytest.mark.parametrize("architecture", ["v1", "v2"])
def test_history_keep_masks_the_previous_action_triple_at_every_step_only(architecture):
    model = _model(architecture)
    batch = _public_batch(batch=2)
    original = {key: value.clone() for key, value in batch.items()}
    bos = {key: value.clone() for key, value in batch.items()}
    for key in ("previous_action_id", "previous_action_x", "previous_action_y"):
        bos[key][1].fill_(-1)
    keep = torch.tensor([True, False])

    with torch.no_grad():
        masked = model.encode_history(batch, history_keep=keep)
        reference = model.encode_history(bos)
        unmasked = model.encode_history(batch)

    # Inputs are never mutated; targets never enter encode_history at all.
    for key, value in original.items():
        assert torch.equal(batch[key], value)
    # Row 1 with history dropped equals the row fed BOS at every step by hand.
    assert torch.equal(masked.hidden, reference.hidden)
    assert torch.equal(masked.final_memory, reference.final_memory)
    # Row 0 keeps its history; row 1 genuinely changed relative to no masking.
    assert torch.equal(masked.hidden[0], unmasked.hidden[0])
    assert not torch.equal(masked.hidden[1], unmasked.hidden[1])
    # Frames, flags and legal masks still flow: spatial features are identical.
    assert torch.equal(masked.spatial, unmasked.spatial)

    with pytest.raises(ValueError):
        model.encode_history(batch, history_keep=torch.tensor([True]))


def test_history_keep_reaches_policy_forward_policy_step_and_loss():
    model = _model("v2")
    batch = _public_batch(batch=1)
    keep = torch.tensor([False])
    with torch.no_grad():
        dropped = model(batch, history_keep=keep)
        kept = model(batch)
        stepwise, memory = model.policy_step(
            frame=batch["frames"][:, 1],
            previous_action_id=batch["previous_action_id"][:, 1],
            previous_action_x=batch["previous_action_x"][:, 1],
            previous_action_y=batch["previous_action_y"][:, 1],
            previous_level_boundary=batch["previous_level_boundary"][:, 1],
            terminal=batch["terminal"][:, 1],
            won=batch["won"][:, 1],
            legal_action_mask=batch["legal_action_mask"][:, 1],
            memory=None,
            history_keep=keep,
        )
        bos_step, _ = model.policy_step(
            frame=batch["frames"][:, 1],
            previous_action_id=torch.tensor([-1]),
            previous_action_x=torch.tensor([-1]),
            previous_action_y=torch.tensor([-1]),
            previous_level_boundary=batch["previous_level_boundary"][:, 1],
            terminal=batch["terminal"][:, 1],
            won=batch["won"][:, 1],
            legal_action_mask=batch["legal_action_mask"][:, 1],
        )
    assert not torch.equal(dropped.action_logits[:, 1:], kept.action_logits[:, 1:])
    assert torch.equal(stepwise.action_logits, bos_step.action_logits)
    assert torch.equal(stepwise.click_logits, bos_step.click_logits)
    assert memory.shape == (1, model.config.hidden_dim)

    # Loss: targets and auxiliary executed-action labels are untouched, so the
    # counts match and only the policy terms move.
    game = _marker_game()
    supervised = collate_game_sequences([game])
    original = {key: value.clone() for key, value in supervised.items()}
    with torch.no_grad():
        with_history = compute_multigame_loss(model, supervised)
        without = compute_multigame_loss(model, supervised, history_keep=torch.tensor([False]))
    for key, value in original.items():
        assert torch.equal(supervised[key], value)
    assert without.policy_targets == with_history.policy_targets == len(game)
    assert without.click_targets == with_history.click_targets == len(game) // 2
    assert without.auxiliary_targets == with_history.auxiliary_targets == len(game)
    assert not torch.equal(without.action, with_history.action)


# --- v1 compatibility -------------------------------------------------------

@pytest.mark.skipif(not FROZEN_V1.exists() or not FROZEN_GAME.exists(), reason="frozen v1 artifacts absent")
def test_frozen_v1_checkpoint_loads_and_reproduces_pre_change_logits():
    torch.set_num_threads(1)
    payload = torch.load(FROZEN_V1, map_location="cpu", weights_only=False)
    assert "architecture" not in payload["model_config"]
    model = MultiGameModel(MultiGameModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    assert model.config.architecture == "v1"
    assert model.parameter_count() == 999_068

    game = load_supervised_game(FROZEN_GAME, FROZEN_TEACHER)
    batch = collate_game_sequences([game])
    with torch.no_grad():
        output = model(batch)
        loss = compute_multigame_loss(model, batch)
    action = output.action_logits[0]
    click = output.click_logits[0]
    assert action.shape[0] == V1_REFERENCE["steps"]
    finite = torch.isfinite(action)
    assert abs(action[finite].sum().item() - V1_REFERENCE["finite_action_sum"]) < 1e-4
    assert action.argmax(-1).tolist() == V1_REFERENCE["action_argmax"]
    x, y = MultiGameModel.decode_click(click)
    assert list(zip(x.tolist(), y.tolist())) == V1_REFERENCE["click_xy"]
    assert abs(click.abs().sum().item() - V1_REFERENCE["click_abs_sum"]) < 1.0
    assert abs(click.max().item() - V1_REFERENCE["click_max"]) < 1e-5
    for term, expected in zip((loss.total, loss.action, loss.click, loss.next_frame, loss.events),
                              V1_REFERENCE["losses"]):
        assert abs(term.item() - expected) < 1e-5

    if PRE_CHANGE_REFERENCE.exists():
        reference = torch.load(PRE_CHANGE_REFERENCE, weights_only=False)
        expected_action = reference["action_logits"][0]
        assert torch.equal(torch.isfinite(expected_action), finite)
        assert torch.allclose(action[finite], expected_action[finite], atol=1e-6, rtol=0)
        # Bit-identical when run with the snapshot's thread count; a different
        # thread count changes the conv reduction order by a few float32 ulps
        # on logits of magnitude ~30 (measured max 1.5e-5), hence 1e-4.
        assert torch.allclose(click, reference["click_logits"][0], atol=1e-4, rtol=0)
        assert abs(loss.total.item() - reference["loss_total"].item()) < 1e-6


# --- v2 architecture --------------------------------------------------------

def test_v2_forward_shapes_and_click_map_coordinate_convention():
    model = _model("v2")
    batch = _public_batch(steps=5, batch=2)
    with torch.no_grad():
        encoded = model.encode_history(batch)
        output = model.policy_from_history(encoded, batch["legal_action_mask"])
    assert encoded.context is not None
    assert encoded.context.shape == (2, 5, model.config.conv_channels, GRID_SIZE, GRID_SIZE)
    assert encoded.spatial.shape == (2, 5, model.config.spatial_channels, FRAME_SIZE, FRAME_SIZE)
    assert output.action_logits.shape == (2, 5, ACTION_COUNT)
    assert output.click_logits.shape == (2, 5, FRAME_SIZE, FRAME_SIZE)
    assert torch.isfinite(output.click_logits).all()
    assert torch.isneginf(output.action_logits[..., 0]).all()

    # decode_click reads [.., y, x]: flat argmax -> x = index % 64, y = index // 64.
    x, y = MultiGameModel.decode_click(output.click_logits)
    flat_index = output.click_logits.flatten(-2).argmax(-1)
    assert torch.equal(x, flat_index % FRAME_SIZE)
    assert torch.equal(y, flat_index // FRAME_SIZE)
    rows = torch.arange(2)[:, None].expand(2, 5)
    cols = torch.arange(5)[None, :].expand(2, 5)
    picked = output.click_logits[rows, cols, y, x]
    assert torch.equal(picked, output.click_logits.flatten(-2).max(-1).values)

    # v1 has no context grid and v2 refuses to decode clicks without one.
    v1 = _model("v1")
    with torch.no_grad():
        assert v1.encode_history(batch).context is None


def test_v2_action_logits_respond_to_a_single_frame_edit_where_v1_barely_does():
    batch = _public_batch(steps=6, seed=0)
    edited = {key: value.clone() for key, value in batch.items()}
    patch = edited["frames"][0, 3, 10:13, 20:23]
    edited["frames"][0, 3, 10:13, 20:23] = (patch + 5) % 16
    deltas = {}
    for architecture in ("v1", "v2"):
        model = _model(architecture, seed=0)
        with torch.no_grad():
            before = model(batch).action_logits[0, 3, 1:]
            after = model(edited).action_logits[0, 3, 1:]
        deltas[architecture] = (before - after).abs().max().item()
    assert deltas["v2"] > 5 * deltas["v1"], deltas
    assert deltas["v2"] > 1e-4


def test_v2_parameter_count_exceeds_v1_and_default_configs_build():
    v1 = MultiGameModel(MultiGameModelConfig())
    v2 = MultiGameModel(MultiGameModelConfig(architecture="v2"))
    assert v1.parameter_count() == 999_068
    assert v2.parameter_count() > v1.parameter_count()
    assert not hasattr(v2, "click_query")
    assert not hasattr(v1, "click_decoder")


# --- loss reductions and auxiliaries ------------------------------------------

@pytest.mark.parametrize("architecture", ["v1", "v2"])
def test_sum_reduction_equals_mean_times_counts(architecture):
    model = _model(architecture)
    batch = collate_game_sequences([_marker_game(8), _marker_game(5, seed=9)])
    with torch.no_grad():
        mean = compute_multigame_loss(model, batch)
        total = compute_multigame_loss(model, batch, reduction="sum")
    assert mean.reduction == "mean" and total.reduction == "sum"
    assert total.action_targets == mean.action_targets == mean.policy_targets == 13
    assert total.click_targets == mean.click_targets == 4 + 2
    assert total.auxiliary_targets == 13
    assert total.frame_targets == 13 * FRAME_SIZE * FRAME_SIZE
    assert total.event_targets == 13 * 3
    assert torch.isclose(total.action, mean.action * total.action_targets, rtol=1e-5)
    assert torch.isclose(total.click, mean.click * total.click_targets, rtol=1e-5)
    assert torch.isclose(total.next_frame, mean.next_frame * total.frame_targets, rtol=1e-5)
    assert torch.isclose(total.events, mean.events * total.event_targets, rtol=1e-5)
    assert total.click_exact_correct == mean.click_exact_correct
    assert total.click_region_correct == mean.click_region_correct
    with pytest.raises(ValueError):
        compute_multigame_loss(model, batch, reduction="max")


def test_zero_auxiliary_weights_skip_the_transition_head(monkeypatch):
    model = _model("v2", train=True)
    batch = collate_game_sequences([_marker_game()])

    def forbidden(*args, **kwargs):
        raise AssertionError("predict_transitions must not run when both auxiliary weights are 0")

    monkeypatch.setattr(model, "predict_transitions", forbidden)
    loss = compute_multigame_loss(model, batch, weights=LossWeights(next_frame=0.0, events=0.0))
    assert loss.next_frame.item() == 0.0 and loss.events.item() == 0.0
    assert loss.auxiliary_targets == loss.frame_targets == loss.event_targets == 0
    assert loss.policy_targets == 8 and loss.click_targets == 4
    loss.total.backward()
    assert model.action_head.weight.grad is not None
    assert model.click_decoder[-1].weight.grad is not None
    assert model.frame_predictor[-1].weight.grad is None

    # Either weight alone keeps the head running.
    monkeypatch.undo()
    partial = compute_multigame_loss(model, batch, weights=LossWeights(next_frame=0.0, events=0.1))
    assert partial.auxiliary_targets == 8 and partial.next_frame.item() > 0


def test_tiny_v2_model_overfits_a_directional_and_click_fixture():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        game = _marker_game(8, seed=3)
        batch = collate_game_sequences([game])
        model = _model("v2", seed=0, train=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        weights = LossWeights(next_frame=0.0, events=0.0)
        first = None
        for _ in range(200):
            loss = compute_multigame_loss(model, batch, weights=weights)
            if first is None:
                first = loss.total.item()
            optimizer.zero_grad()
            loss.total.backward()
            optimizer.step()
        final = loss.total.item()
        assert first / final > 5, (first, final)
        model.eval()
        with torch.no_grad():
            output = model(batch)
        predicted = output.action_logits[0].argmax(-1)
        assert predicted.tolist() == game.target_action_id.tolist()
        x, y = MultiGameModel.decode_click(output.click_logits[0])
        clicks = game.target_action_id == CLICK_ACTION
        assert x[clicks].tolist() == game.target_action_x[clicks].tolist()
        assert y[clicks].tolist() == game.target_action_y[clicks].tolist()
    finally:
        torch.set_num_threads(threads)
