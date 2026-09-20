"""Exact bijection tests for fixed whole-game public variants."""

import numpy as np
import pytest

from pebby.multigame_variants import (
    D4_NAMES,
    VariantOptions,
    WholeGameVariant,
    sample_whole_game_variant,
    transform_frame,
    transform_xy,
)


@pytest.mark.parametrize("name", D4_NAMES)
def test_d4_frame_and_every_full_pixel_click_coordinate_agree_and_roundtrip(name):
    frame = np.arange(64 * 64, dtype=np.int32).reshape(64, 64)
    transformed = transform_frame(frame, name)
    variant = WholeGameVariant(
        True, 7, tuple(range(8)), name, tuple(range(16)),
    )
    for y in range(64):
        for x in range(64):
            public_x, public_y = transform_xy(name, x, y)
            assert transformed[public_y, public_x] == frame[y, x]
            public = variant.public_action(6, x, y)
            assert public == (6, public_x, public_y)
            assert variant.raw_action(*public) == (6, x, y)


def test_palette_frame_and_action_control_bijections_are_exact():
    controls = (0, 4, 1, 5, 2, 7, 6, 3)
    palette = tuple(reversed(range(16)))
    variant = WholeGameVariant(True, 11, controls, "rot90", palette)
    frame = (np.arange(64 * 64).reshape(64, 64) % 16).astype(np.uint8)
    assert np.array_equal(variant.raw_frame(variant.public_frame(frame)), frame)
    for raw_id in (0, 1, 2, 3, 4, 5, 7):
        public = variant.public_action(raw_id, None, None)
        assert variant.raw_action(*public) == (raw_id, None, None)


def test_control_permutation_maps_legal_sets_and_leaves_click_fixed():
    variant = WholeGameVariant(
        True,
        5,
        (0, 3, 1, 2, 4, 5, 6, 7),
        "identity",
        tuple(range(16)),
    )
    raw = np.zeros(8, dtype=np.bool_)
    raw[[1, 2, 6]] = True
    public = variant.public_legal_mask(raw)
    assert np.flatnonzero(public).tolist() == [1, 3, 6]
    assert variant.public_action(6, 63, 0) == (6, 63, 0)


def test_sampling_is_deterministic_per_game_and_changes_across_games():
    options = VariantOptions(enabled=True, mix_probability=1.0, seed=123)
    first = sample_whole_game_variant(
        options, source_id="cd82-fb555c5d", game_index=4, legal_action_ids=(1, 2, 3, 4),
    )
    repeated = sample_whole_game_variant(
        options, source_id="cd82-fb555c5d", game_index=4, legal_action_ids=(1, 2, 3, 4),
    )
    other = sample_whole_game_variant(
        options, source_id="cd82-fb555c5d", game_index=5, legal_action_ids=(1, 2, 3, 4),
    )
    assert first == repeated
    assert first.seed != other.seed
    assert first.private_metadata() != other.private_metadata()


def test_identity_is_default_and_component_validation_is_explicit():
    identity = sample_whole_game_variant(
        VariantOptions(), source_id="anything", game_index=0, legal_action_ids=(1, 2, 6),
    )
    assert identity.is_identity and not identity.selected
    with pytest.raises(ValueError, match="at least one component"):
        VariantOptions(enabled=True, controls=False, spatial=False, palette=False)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        VariantOptions(enabled=True, mix_probability=1.1)


def test_variant_normalizes_caller_lists_and_rejects_lossy_mapping_casts():
    controls = list(range(8))
    palette = list(range(16))
    variant = WholeGameVariant(True, 1, controls, "identity", palette)
    controls[1], controls[2] = controls[2], controls[1]
    palette.reverse()
    assert variant.control_raw_to_public == tuple(range(8))
    assert variant.palette_raw_to_public == tuple(range(16))
    with pytest.raises(ValueError, match="entries must be integers"):
        WholeGameVariant(True, 1, (0, 1.0, 2, 3, 4, 5, 6, 7), "identity", tuple(range(16)))
    with pytest.raises(ValueError, match="entries must be integers"):
        WholeGameVariant(True, 1, tuple(range(8)), "identity", (False, *range(1, 16)))
