"""Native causal handoff coverage for WA30's closed fence tiers."""

from copy import deepcopy
import random

import pytest

from pebby.games.wa30.generate import (
    FULL_STANDARD_CONTRACT,
    MECHANICS_VERSION,
    _draft,
    _no_player_control,
    generate,
    structural_metrics,
    validate_full_standard,
)
from pebby.games.wa30.quality import SPLITS, geometry_partition


@pytest.fixture(scope="module")
def handoff_rows():
    rows = {
        difficulty: [generate(seed, difficulty) for seed in range(3)]
        for difficulty in (3, 4)
    }
    assert all(all(tier) for tier in rows.values())
    return rows


def test_closed_fence_tiers_require_real_player_to_helper_transport(handoff_rows):
    expected = {3: (3, 2), 4: (6, 0)}
    for difficulty, tier in handoff_rows.items():
        player_side, on_fence = expected[difficulty]
        for spec in tier:
            metrics = structural_metrics(spec)
            assert metrics["boxes_in_player_component"] == player_side
            assert metrics["boxes_on_fences"] == on_fence
            mechanics = spec["solution_mechanics"]
            assert mechanics["manual_box_moves"] >= player_side
            assert mechanics["manual_fence_entries"] >= player_side
            assert mechanics["player_helper_handoffs"] == player_side
            assert mechanics["player_helper_deliveries"] == player_side
            assert validate_full_standard(
                spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            ) == []


def test_no_player_native_control_loses_without_movement_or_holding(handoff_rows):
    for difficulty, tier in handoff_rows.items():
        for spec in tier:
            control = _no_player_control(spec)
            assert control == spec["no_player_control"]
            assert control == spec["proof"]["no_player_control"]
            assert control["actions_replayed"] == spec["budget"] == 100
            assert control["game_over"] is True
            assert control["won"] is False
            assert control["levels_completed"] == 0
            assert control["player_moved"] is False
            assert control["player_held_box"] is False


def test_handoff_rows_vary_geometry_and_native_solution(handoff_rows):
    for tier in handoff_rows.values():
        assert len({row["geometry_d4_sha256"] for row in tier}) == 3
        assert len({row["action_sha256"] for row in tier}) == 3
        assert len({row["solution_length"] for row in tier}) >= 2


def test_native_control_and_handoff_evidence_are_recomputed(handoff_rows):
    changed = deepcopy(handoff_rows[3][0])
    changed["no_player_control"]["won"] = True
    changed["proof"]["no_player_control"]["won"] = True
    assert validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][2])

    changed = deepcopy(handoff_rows[4][0])
    changed["solution_mechanics"]["player_helper_handoffs"] -= 1
    assert validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][3])


def test_tier8_finite_draft_support_reaches_every_split():
    rng = random.Random(f"{MECHANICS_VERSION}:tier8-split-support:8")
    partitions = {
        geometry_partition(spec)[1]
        for _ in range(256)
        if (spec := _draft(rng, 8)) is not None
    }
    assert partitions == set(SPLITS)


def test_tier8_validation_seed_has_native_causal_positive_witness():
    spec = generate(2555997001157478120, 8, split="validation")
    assert spec is not None
    assert spec["geometry_split"] == "validation"
    assert spec["solution_mechanics"]["thief_steals_from_helper"] >= 1
    assert spec["solution_mechanics"]["thieves_destroyed"] >= 1
    assert validate_full_standard(
        spec, FULL_STANDARD_CONTRACT["curriculum"][7]
    ) == []
