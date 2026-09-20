"""Load-time whole-game variant augmentation for multigame training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pebby.agent import multigame_training as T
from pebby.agent.multigame_model import (
    ACTION_COUNT,
    CLICK_ACTION,
    COORDINATE_NONE,
    FRAME_SIZE,
    GameSequence,
    MultiGameModelConfig,
    load_supervised_game,
)
from pebby.multigame_variants import D4_NAMES, WholeGameVariant, transform_frame

from test_multigame_training import _write_manifest


CONTROLS_A = (0, 3, 1, 2, 5, 4, 6, 7)
CONTROLS_B = (0, 2, 4, 1, 3, 7, 6, 5)
PALETTE_A = tuple(reversed(range(16)))
PALETTE_B = tuple((value * 7) % 16 for value in range(16))
LEGAL = (1, 2, 3, 4, 5, 6, 7)


def _variant(controls, spatial, palette, seed=1) -> WholeGameVariant:
    return WholeGameVariant(True, seed, controls, spatial, palette)


def _stored_arrays(steps: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic public+teacher arrays following the stored NPZ contract."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 16, size=(steps + 1, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    action_id = rng.choice(LEGAL, size=steps).astype(np.int8)
    action_id[::3] = CLICK_ACTION
    x = np.full(steps, COORDINATE_NONE, dtype=np.int16)
    y = np.full(steps, COORDINATE_NONE, dtype=np.int16)
    click = action_id == CLICK_ACTION
    x[click] = rng.integers(0, FRAME_SIZE, size=int(click.sum()))
    y[click] = rng.integers(0, FRAME_SIZE, size=int(click.sum()))
    legal = rng.random((steps + 1, ACTION_COUNT)) < 0.6
    legal[:, 0] = False
    legal[np.arange(steps), action_id.astype(np.int64)] = True
    target_id = action_id.astype(np.int16).copy()
    target_id[1::4] = COORDINATE_NONE
    target_x, target_y = x.astype(np.int16).copy(), y.astype(np.int16).copy()
    target_x[target_id == COORDINATE_NONE] = COORDINATE_NONE
    target_y[target_id == COORDINATE_NONE] = COORDINATE_NONE
    return {
        "frames": frames,
        "legal_action_mask": legal,
        "action_id": action_id,
        "action_x": x,
        "action_y": y,
        "level_boundary": np.zeros(steps, dtype=np.bool_),
        "terminal": np.zeros(steps + 1, dtype=np.bool_),
        "target_action_id": target_id,
        "target_action_x": target_x,
        "target_action_y": target_y,
        "source": np.ones(steps, dtype=np.int8),
    }


def _assert_same(left, right) -> None:
    assert set(left) == set(right)
    for key in left:
        if left[key] is None:
            assert right[key] is None
            continue
        assert left[key].dtype == right[key].dtype, key
        assert np.array_equal(left[key], right[key]), key


@pytest.mark.parametrize("spatial", D4_NAMES)
def test_variant_then_inverse_restores_stored_arrays(spatial):
    arrays = _stored_arrays()
    variant = _variant(CONTROLS_A, spatial, PALETTE_A)
    forward = T.apply_variant_to_game(arrays, variant, LEGAL)
    restored = T.apply_variant_to_game(forward, T.inverse_variant(variant), LEGAL)
    _assert_same(restored, arrays)
    if spatial != "identity":
        assert not np.array_equal(forward["frames"], arrays["frames"])
    # Untouched provenance / event fields come back as the very same arrays.
    for key in ("level_boundary", "terminal", "source"):
        assert forward[key] is arrays[key]


def test_variant_on_top_of_stored_variant_equals_composed_bijection():
    arrays = _stored_arrays(seed=3)
    first = _variant(CONTROLS_A, "rot90", PALETTE_A)
    second = _variant(CONTROLS_B, "flip_x", PALETTE_B)
    stacked = T.apply_variant_to_game(T.apply_variant_to_game(arrays, first, LEGAL), second, LEGAL)
    composed = T.compose_variants(first, second)
    assert composed.control_raw_to_public == tuple(CONTROLS_B[CONTROLS_A[i]] for i in range(8))
    assert composed.palette_raw_to_public == tuple(PALETTE_B[PALETTE_A[i]] for i in range(16))
    assert composed.spatial == T.compose_d4("rot90", "flip_x")
    _assert_same(stacked, T.apply_variant_to_game(arrays, composed, LEGAL))


def test_d4_composition_table_matches_frame_transforms():
    probe = np.arange(FRAME_SIZE * FRAME_SIZE).reshape(FRAME_SIZE, FRAME_SIZE)
    assert T.compose_d4("rot90", "rot90") == "rot180"
    assert T.compose_d4("flip_x", "flip_x") == "identity"
    assert T.compose_d4("rot90", "rot270") == "identity"
    for first in D4_NAMES:
        assert T.compose_d4(first, T.inverse_d4(first)) == "identity"
        for second in D4_NAMES:
            expected = transform_frame(transform_frame(probe, first), second)
            assert np.array_equal(transform_frame(probe, T.compose_d4(first, second)), expected)


@pytest.mark.parametrize("spatial", D4_NAMES)
def test_click_coordinates_follow_the_marked_pixel(spatial):
    steps = 4
    arrays = _stored_arrays(steps=steps, seed=5)
    arrays["frames"][:] = 0
    marks = [(61, 43), (0, 0), (63, 63), (7, 20)]
    for step, (mx, my) in enumerate(marks):
        arrays["frames"][step, my, mx] = 9
        arrays["action_id"][step] = CLICK_ACTION
        arrays["action_x"][step], arrays["action_y"][step] = mx, my
    arrays["target_action_id"][:] = arrays["action_id"]
    arrays["target_action_x"][:] = arrays["action_x"]
    arrays["target_action_y"][:] = arrays["action_y"]
    arrays["legal_action_mask"][:, CLICK_ACTION] = True
    variant = _variant(tuple(range(8)), spatial, PALETTE_B)
    out = T.apply_variant_to_game(arrays, variant, LEGAL)
    for step in range(steps):
        px, py = int(out["action_x"][step]), int(out["action_y"][step])
        assert out["frames"][step, py, px] == PALETTE_B[9]
        assert (out["frames"][step] == PALETTE_B[9]).sum() == 1
        assert (int(out["target_action_x"][step]), int(out["target_action_y"][step])) == (px, py)


def test_nonclick_actions_keep_missing_coordinates_and_mask_permutes_with_ids():
    arrays = _stored_arrays(seed=8)
    variant = _variant(CONTROLS_A, "transpose", tuple(range(16)))
    out = T.apply_variant_to_game(arrays, variant, LEGAL)
    ids = arrays["action_id"].astype(np.int64)
    nonclick = ids != CLICK_ACTION
    assert np.all(out["action_x"][nonclick] == COORDINATE_NONE)
    assert np.all(out["action_y"][nonclick] == COORDINATE_NONE)
    assert np.array_equal(out["action_id"], np.asarray(CONTROLS_A)[ids].astype(np.int8))
    assert out["action_id"].dtype == arrays["action_id"].dtype
    missing = arrays["target_action_id"] == COORDINATE_NONE
    assert np.all(out["target_action_id"][missing] == COORDINATE_NONE)
    assert np.all(out["target_action_x"][missing] == COORDINATE_NONE)
    steps = len(ids)
    # Each executed public id remains legal in its aligned permuted mask, and
    # the mask permutation is exactly the public_legal_mask contract per row.
    assert np.all(out["legal_action_mask"][np.arange(steps), out["action_id"].astype(np.int64)])
    for row in range(steps + 1):
        assert np.array_equal(
            out["legal_action_mask"][row], variant.public_legal_mask(arrays["legal_action_mask"][row]),
        )
    assert np.array_equal(out["legal_action_mask"].sum(1), arrays["legal_action_mask"].sum(1))


def test_apply_rejects_variant_that_moves_legal_ids_outside_the_legal_set():
    arrays = _stored_arrays()
    with pytest.raises(ValueError, match="legal action set"):
        T.apply_variant_to_game(arrays, _variant(CONTROLS_A, "identity", tuple(range(16))), (1, 2))


def test_game_sequence_transform_matches_loader_on_transformed_files(tmp_path):
    train = _write_manifest(tmp_path, "seq", master_seed=1, effective_seed=1, puzzle="seq")
    root = train.parent
    sequence = load_supervised_game(root / "games" / "game.npz", root / "teacher" / "game.npz")
    legal = T.game_legal_action_ids(sequence)
    variant = _variant(CONTROLS_B, "rot270", PALETTE_A)
    transformed = T.apply_variant_to_game(sequence, variant, legal)
    assert isinstance(transformed, GameSequence)
    # Transform the stored files with the same variant and reload: previous_*,
    # executed_*, target_*, next_frames and the mask must all agree.
    public = dict(np.load(root / "games" / "game.npz", allow_pickle=False))
    teacher = dict(np.load(root / "teacher" / "game.npz", allow_pickle=False))
    moved_public = T.apply_variant_to_game(public, variant, legal)
    moved_teacher = T.apply_variant_to_game(teacher, variant, legal)
    np.savez(tmp_path / "public.npz", **moved_public)
    np.savez(tmp_path / "teacher.npz", **moved_teacher)
    reloaded = load_supervised_game(tmp_path / "public.npz", tmp_path / "teacher.npz")
    for name in GameSequence.__dataclass_fields__:
        assert np.array_equal(getattr(transformed, name), getattr(reloaded, name)), name


def test_per_epoch_sampling_is_fixed_within_a_game_and_changes_across_epochs():
    options = T.LoadTimeVariantOptions(enabled=True, seed=42)
    same = [T.load_time_variant(options, epoch=0, game_index=3, legal_action_ids=LEGAL) for _ in range(3)]
    assert same[0] == same[1] == same[2] and not same[0].is_identity
    later = T.load_time_variant(options, epoch=1, game_index=3, legal_action_ids=LEGAL)
    assert later != same[0]
    other_seed = T.load_time_variant(
        T.LoadTimeVariantOptions(enabled=True, seed=43), epoch=0, game_index=3, legal_action_ids=LEGAL,
    )
    assert other_seed != same[0]
    # The whole game moves under exactly one bijection: chunking the transformed
    # sequence equals transforming each chunk with the same variant.
    arrays = _stored_arrays(steps=9, seed=11)
    sequence = _sequence_from_arrays(arrays)
    whole, variant = T.augment_sequence_for_epoch(sequence, options, epoch=0, game_index=3)
    assert variant == same[0]
    for start in range(0, len(sequence), 4):
        chunk = T.slice_game_sequence(sequence, start, min(len(sequence), start + 4))
        moved = T.apply_variant_to_game(chunk, variant, T.game_legal_action_ids(sequence))
        expected = T.slice_game_sequence(whole, start, min(len(sequence), start + 4))
        for name in GameSequence.__dataclass_fields__:
            assert np.array_equal(getattr(moved, name), getattr(expected, name)), name


def _sequence_from_arrays(arrays: dict[str, np.ndarray]) -> GameSequence:
    steps = len(arrays["action_id"])
    ids = arrays["action_id"].astype(np.int64)
    x = arrays["action_x"].astype(np.int64)
    y = arrays["action_y"].astype(np.int64)
    prev = lambda value: np.concatenate(([COORDINATE_NONE], value[:-1])).astype(np.int64)  # noqa: E731
    return GameSequence(
        frames=arrays["frames"][:-1],
        previous_action_id=prev(ids),
        previous_action_x=prev(x),
        previous_action_y=prev(y),
        previous_level_boundary=np.zeros(steps, dtype=np.bool_),
        terminal=np.zeros(steps, dtype=np.bool_),
        won=np.zeros(steps, dtype=np.bool_),
        legal_action_mask=arrays["legal_action_mask"][:-1],
        executed_action_id=ids,
        executed_action_x=x,
        executed_action_y=y,
        next_frames=arrays["frames"][1:],
        next_level_boundary=np.zeros(steps, dtype=np.bool_),
        next_terminal=np.zeros(steps, dtype=np.bool_),
        next_won=np.zeros(steps, dtype=np.bool_),
        target_action_id=arrays["target_action_id"].astype(np.int64),
        target_action_x=arrays["target_action_x"].astype(np.int64),
        target_action_y=arrays["target_action_y"].astype(np.int64),
        target_valid=arrays["target_action_id"] != COORDINATE_NONE,
        action_source=arrays["source"].astype(np.int64),
    )


def test_disabled_options_give_identity_and_leave_the_sequence_object_alone():
    arrays = _stored_arrays(steps=5, seed=2)
    sequence = _sequence_from_arrays(arrays)
    options = T.LoadTimeVariantOptions()
    for epoch in range(3):
        moved, variant = T.augment_sequence_for_epoch(sequence, options, epoch=epoch, game_index=epoch)
        assert variant.is_identity and moved is sequence
    with pytest.raises(ValueError, match="at least one component"):
        T.LoadTimeVariantOptions(enabled=True, controls=False, spatial=False, palette=False)
    zero_mix = T.LoadTimeVariantOptions(enabled=True, mix_probability=0.0)
    assert T.load_time_variant(zero_mix, epoch=0, game_index=0, legal_action_ids=LEGAL).is_identity


def test_training_config_round_trips_load_variant_options():
    config = T.TrainingConfig(
        model=MultiGameModelConfig.cpu_test(),
        load_variants=T.LoadTimeVariantOptions(enabled=True, seed=9, palette=False),
    )
    restored = T.TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert T._training_signature(config)["load_variants"]["seed"] == 9
    legacy = dict(config.to_dict())
    legacy.pop("load_variants")
    assert T.TrainingConfig.from_dict(legacy).load_variants == T.LoadTimeVariantOptions()


def test_cli_smoke_run_with_load_variants_completes_and_records_options(tmp_path, monkeypatch):
    from tools import train_multigame as cli

    train = _write_manifest(tmp_path, "train", master_seed=11, effective_seed=111, puzzle="train")
    validation = _write_manifest(
        tmp_path, "validation", master_seed=22, effective_seed=222, puzzle="validation",
    )
    calls = []

    def fake_closed(model, games, **kwargs):
        calls.append(len(games))
        return {
            "games_won": 0, "levels_completed": 0, "games": [{"failure": None}],
            "panel_records": ["fake"], "actions": 1,
        }

    monkeypatch.setattr(T, "evaluate_generated_closed_loop", fake_closed)
    out = tmp_path / "run"
    result = cli.main([
        "--train-manifest", str(train), "--validation-manifest", str(validation),
        "--out-dir", str(out), "--smoke", "--cpu-test-model", "--device", "cpu",
        "--epochs", "2", "--chunk-steps", "2", "--auxiliary-transitions-per-chunk", "1",
        "--metric-transitions-per-game", "1", "--closed-loop-games", "1",
        "--load-variants", "--load-variant-mix", "1.0", "--load-variant-seed", "7",
        "--load-variant-components", "controls", "spatial", "palette",
    ])
    # Validation panel plus the diagnostic training-game panel, every epoch.
    assert calls == [1, 1, 1, 1]
    payload = T.load_training_checkpoint(result.latest_checkpoint)
    expected = {
        "enabled": True, "mix_probability": 1.0, "seed": 7, "controls": True,
        "spatial": True, "palette": True, "augment_validation": False,
    }
    assert payload["load_time_variants"] == expected
    assert payload["training_config"]["load_variants"] == expected
    assert payload["training_signature"]["load_variants"] == expected
    logs = json.loads((out / "training-log.json").read_text())
    assert len(logs) == 2 and logs[-1]["global_step"] == 2
    for log in logs:
        summary = log["load_time_variants"]
        assert summary["options"] == expected
        assert summary["train_games_transformed"] == 1
        assert summary["validation_offline_games_transformed"] == 0
        assert summary["validation_closed_loop_transformed"] is False


def test_cli_rejects_load_variant_options_without_enable_flag(tmp_path):
    from tools import train_multigame as cli

    with pytest.raises(SystemExit):
        cli.parse_args([
            "--train-manifest", "a", "--validation-manifest", "b", "--out-dir", str(tmp_path),
            "--load-variant-seed", "3",
        ])
    args = cli.parse_args([
        "--train-manifest", "a", "--validation-manifest", "b", "--out-dir", str(tmp_path),
        "--seed", "5", "--load-variants", "--load-variant-components", "spatial",
    ])
    options = cli.load_variant_options(args)
    assert options == T.LoadTimeVariantOptions(
        enabled=True, seed=5, controls=False, spatial=True, palette=False,
    )
