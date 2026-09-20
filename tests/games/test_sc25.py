"""SC25 full-curriculum planner, generator, and native replay checks."""

import copy
from functools import lru_cache
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from arcengine import GameState

from pebby.games.sc25 import names
from pebby.games.sc25.env import Env, official_levels
from pebby.games.sc25.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    SPLITS,
    build_game,
    build_level,
    generate,
    generate_engine_variant,
    generate_game,
    geometry_partition,
    validate_full_standard,
)
from pebby.games.sc25.layout import extract
from pebby.games.sc25.plan import _cast, _spell_actions, search, solve
from pebby.games.sc25.reference_profiles import REFERENCE_PROFILES, profile_errors


def replay(env, actions):
    observation = None
    for action, x, y in actions:
        assert action in env.available_actions
        observation = env.perform(action, x, y)
    return observation


@lru_cache(maxsize=1)
def generated_game_json():
    specs = generate_game(77, split="test")
    assert specs is not None
    return json.dumps(specs)


def generated_game():
    return json.loads(generated_game_json())


def test_authoritative_count_contract_and_reference_profiles_are_exactly_six():
    assert len(official_levels()) == 6
    assert DIFFICULTIES == tuple(range(1, 7))
    assert FULL_STANDARD_CONTRACT["format"] == "pebby-full-generator-contract-v1"
    assert FULL_STANDARD_CONTRACT["source_id"] == "sc25-635fd71a"
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert FULL_STANDARD_CONTRACT["evidence"]["root_acceptance"] == "sc25.md#root-acceptance"
    assert len(FULL_STANDARD_CONTRACT["curriculum"]) == 6
    assert [row["difficulty"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(6))
    assert [REFERENCE_PROFILES[d]["budget"] for d in DIFFICULTIES] == [50, 25, 50, 35, 65, 60]


def test_teacher_solves_and_native_replays_all_official_levels_sequentially():
    env = Env()
    env.reset()
    lengths = []
    for index in range(6):
        assert env.level_index == index
        result = search(env, limit=600_000)
        assert result.solved, result
        assert not result.truncated and not result.unsupported
        lengths.append(len(result.actions))
        observation = replay(env, result.actions)
        assert env.levels_completed == index + 1
    assert observation.won and env.state == GameState.WIN
    assert lengths == [13, 5, 11, 22, 36, 35]


def test_partial_spell_grid_prefix_is_exactly_recovered_and_replayed():
    env = Env()
    env.set_level(4)
    x, y = names.cell_click(2, 2)
    env.perform(6, x, y)
    layout = extract(env)
    assert layout.start[9] == 1 << 8
    result = search(layout, limit=600_000)
    assert result.solved and not result.truncated and not result.unsupported
    observation = replay(env, result.actions)
    assert env.levels_completed == 1
    assert env.level_index == 5
    assert observation.state == GameState.NOT_FINISHED


def test_env_clone_is_independent_and_exposes_complete_interface():
    env = Env()
    frame = env.reset()
    assert len(frame) == len(frame[0]) == 64
    assert set(env.available_actions) == set(names.ACTION_IDS)
    env.perform(1)
    twin = env.clone()
    x, y = names.cell_click(0, 0)
    env.perform(6, x, y)
    assert getattr(env.game, names.ATTR_GRID) != getattr(twin.game, names.ATTR_GRID)
    assert twin.level_index == env.level_index


def test_parameterized_single_level_is_deterministic_json_safe_and_self_validating():
    first = generate(1204, 4, split="validation")
    second = generate(1204, 4, split="validation")
    assert first == second and first is not None
    restored = json.loads(json.dumps(first))
    assert restored["split"] == "validation"
    assert geometry_partition(restored)[1] == "validation"
    assert profile_errors(restored) == []
    assert validate_full_standard(restored, FULL_STANDARD_CONTRACT["curriculum"][3]) == []
    assert restored["solution_mechanics"]["shrink_casts"] >= 1
    assert restored["solution_mechanics"]["grow_casts"] >= 1
    assert restored["solution_mechanics"]["pickups_consumed"] >= 1
    assert restored["solution_mechanics"]["primary_targets_hit"] >= 1


def test_explicit_splits_are_canonical_geometry_partitions_not_rng_labels():
    rows = {split: generate(80, 1, split=split) for split in SPLITS}
    assert all(rows.values())
    for split, row in rows.items():
        fingerprint, partition = geometry_partition(row)
        assert partition == split == row["geometry_split"]
        assert fingerprint == row["geometry_d4_sha256"]
    assert len({row["geometry_d4_sha256"] for row in rows.values()}) == 3


def test_complete_game_has_six_ordered_contexts_and_rebuilds_natively():
    specs = generated_game()
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["context_index"] for spec in specs] == list(range(6))
    assert all(spec["proof"]["sequential_engine_verified"] for spec in specs)
    assert all(spec["sequence_kind"] == "full-official-context" for spec in specs)
    levels = build_game(specs)
    assert len(levels) == 6
    env = Env(levels)
    env.reset()
    for index, spec in enumerate(specs):
        assert env.level_index == index
        replay(env, [tuple(action) for action in spec["solution"]])
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN


def test_build_game_rejects_context_shift_duplicates_order_and_split_mismatch():
    specs = generated_game()
    with pytest.raises(ValueError, match="list or tuple"):
        build_game(None)
    malformed = copy.deepcopy(specs)
    malformed[0] = None
    with pytest.raises(ValueError, match="mapping"):
        build_game(malformed)
    malformed = copy.deepcopy(specs)
    malformed[0]["split"] = []
    with pytest.raises(ValueError, match="same split"):
        build_game(malformed)
    with pytest.raises(ValueError, match="exactly six"):
        build_game(specs[:-1])
    bad = copy.deepcopy(specs)
    bad[0], bad[1] = bad[1], bad[0]
    with pytest.raises(ValueError, match="invalid tier"):
        build_game(bad)
    bad = copy.deepcopy(specs)
    bad[1] = copy.deepcopy(bad[0])
    bad[1]["difficulty"] = 2
    bad[1]["context_index"] = 1
    with pytest.raises(ValueError):
        build_game(bad)
    bad = copy.deepcopy(specs)
    bad[-1]["split"] = "train"
    with pytest.raises(ValueError, match="same split"):
        build_game(bad)


def test_validator_recomputes_hashes_metrics_mechanics_and_native_outcome():
    row = generated_game()[2]
    entry = FULL_STANDARD_CONTRACT["curriculum"][2]
    for field, value in (
        ("visual_density", 0.0),
        ("geometry_d4_sha256", "0" * 64),
        ("gameplay_sha256", "1" * 64),
        ("solution_mechanics", {}),
    ):
        bad = copy.deepcopy(row)
        bad[field] = value
        assert validate_full_standard(bad, entry), field
    bad = copy.deepcopy(row)
    bad["solution"][-1] = [1, None, None]
    errors = validate_full_standard(bad, entry)
    assert any("does not win" in error for error in errors)

    trailing = copy.deepcopy(row)
    trailing["solution"].append([1, None, None])
    trailing["solution_length"] += 1
    errors = validate_full_standard(trailing, entry)
    assert any("before its final action" in error for error in errors)

    game = generated_game()
    game[0]["solution"].append([1, None, None])
    game[0]["solution_length"] += 1
    with pytest.raises(ValueError, match="before its final action"):
        build_game(game)


def test_build_game_native_replay_itself_rejects_post_win_actions(monkeypatch):
    module = importlib.import_module("pebby.games.sc25.generate")
    game = generated_game()
    game[0]["solution"].append([1, None, None])
    game[0]["solution_length"] += 1
    monkeypatch.setattr(module, "validate_full_standard", lambda spec, entry: [])
    with pytest.raises(ValueError, match="before its final action"):
        module.build_game(game)


def test_validator_fails_closed_for_malformed_proof_and_noninteger_tiers():
    row = generated_game()[0]
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    for malformed in (None, [], "verified", True):
        bad = copy.deepcopy(row)
        bad["proof"] = malformed
        errors = validate_full_standard(bad, entry)
        assert "proof must be a mapping" in errors
    for malformed in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="difficulty"):
            generate(9, malformed, split="train")
        assert profile_errors({"difficulty": malformed}) == [
            "difficulty is not an official tier"
        ]
    with pytest.raises(ValueError, match="difficulties"):
        generate_game(9, split="train", difficulties=(True,))

    for malformed in (None, 7, "route"):
        bad = copy.deepcopy(row)
        bad["solution"] = malformed
        errors = validate_full_standard(bad, entry)
        assert "solution must be a nonempty list" in errors
    for malformed in (None, [], "mechanics"):
        bad = copy.deepcopy(row)
        bad["solution_mechanics"] = malformed
        errors = validate_full_standard(bad, entry)
        assert "solution_mechanics must be a mapping" in errors


def test_engine_only_ring_failed_cast_probe_is_labeled_and_recovers_natively():
    row = generate_engine_variant(41, split="validation")
    assert row is not None
    assert row["profile_kind"] == "engine-extension-nonreference"
    assert row["rings"] and row["probe_mechanics"] == {
        "failed_fire_casts": 1,
        "ring_fire_blocks": 1,
        "target_survived_probe": True,
        "native_recovery_completed": True,
    }
    assert profile_errors(row)


def test_spell_budget_pickup_refund_and_teleport_cursor_match_engine():
    spec = generate(2306, 6, split="train")
    assert spec is not None
    env = Env([build_level(spec) for _ in DIFFICULTIES])
    env.set_level(5)
    layout = extract(env)
    predicted = _cast(layout, layout.start, names.SPELL_TELEPORT)
    assert predicted is not None
    replay(env, _spell_actions(names.SPELL_TELEPORT))
    assert (env.player.x, env.player.y) == layout.pads[0]
    assert getattr(env.game, names.ATTR_TELEPORT_INDEX) == predicted[4] == 1
    assert env.used() == len(names.PATTERNS[names.SPELL_TELEPORT]) + 1


def test_tier6_native_certificate_proves_both_target_covered_pickup_dependencies():
    spec = generate(6106, 6, split="test")
    assert spec is not None
    assert {
        (item["x"], item["y"]) for item in spec["targets"]
    } == {
        (item["x"], item["y"]) for item in spec["pickups"]
    }
    causality = spec["solution_causality"]
    assert causality["version"] == "sc25-target-covered-pickup-native-v1"
    assert causality["native_dependencies_verified"] == 2
    assert {pair["family"] for pair in causality["pairs"]} == {
        "primary", "alternate"
    }
    for pair in causality["pairs"]:
        assert pair["target_clear_action"] < pair["pickup_collect_action"]
        assert pair["movement_suffix"]
        assert pair["counterfactual_target_present"]
        assert pair["counterfactual_pickup_present"]
        assert pair["counterfactual_progress_blocked"]
        assert pair["used_after_pickup_action"] == max(
            0, pair["used_before_pickup_action"] + 1 - names.PICKUP_REFUND
        )
    assert validate_full_standard(
        spec, FULL_STANDARD_CONTRACT["curriculum"][5]
    ) == []

    forged = copy.deepcopy(spec)
    forged["solution_causality"]["pairs"][0]["target_clear_action"] += 1
    errors = validate_full_standard(
        forged, FULL_STANDARD_CONTRACT["curriculum"][5]
    )
    assert any("causality does not match native replay" in error for error in errors)


def test_official_tier6_native_prefix_has_the_same_target_then_pickup_rule():
    env = Env()
    env.set_level(5)

    def positions(sprite_name):
        return {
            (sprite.x, sprite.y)
            for sprite in env.game.current_level.get_sprites()
            if sprite.name == sprite_name
        }

    assert positions(names.SPRITE_PICKUP) == {(17, 33), (37, 37)}
    assert positions(names.SPRITE_TARGET) == {(17, 33)}
    assert positions(names.SPRITE_TARGET_ALT) == {(37, 37)}

    for spell in (names.SPELL_GROW, names.SPELL_TELEPORT):
        replay(env, _spell_actions(spell))
    replay(env, [(action, None, None) for action in (4, 4, 1, 1, 1, 1)])
    assert (env.player.x, env.player.y) == (17, 37)
    assert env.used() == 15
    assert (17, 33) in positions(names.SPRITE_TARGET)
    assert (17, 33) in positions(names.SPRITE_PICKUP)

    replay(env, _spell_actions(names.SPELL_FIRE))
    assert env.used() == 19
    assert (17, 33) not in positions(names.SPRITE_TARGET)
    assert (17, 33) in positions(names.SPRITE_PICKUP)
    env.perform(1)
    assert (env.player.x, env.player.y) == (17, 35)
    assert env.used() == 20
    env.perform(1)
    assert (env.player.x, env.player.y) == (17, 33)
    assert env.used() == 11
    assert (17, 33) not in positions(names.SPRITE_PICKUP)


def test_bank_cli_requires_split_and_writes_reconstructable_jsonl(tmp_path):
    output = tmp_path / "sc25.jsonl"
    subprocess.run([
        sys.executable, "-m", "pebby.games.sc25.bank", "--levels", "2",
        "--seed", "30", "--difficulty", "2", "--split", "train",
        "--out", str(output),
    ], check=True, timeout=60)
    specs = [json.loads(line) for line in Path(output).read_text().splitlines() if line]
    assert len(specs) == 2
    assert all(spec["split"] == "train" for spec in specs)
    assert all(build_level(spec) is not None for spec in specs)


def test_invalid_inputs_and_bounded_cutoff_are_explicit():
    with pytest.raises(TypeError):
        generate(1, 1)
    with pytest.raises(ValueError):
        generate(True, 1, split="train")
    with pytest.raises(ValueError):
        generate(1, 7, split="train")
    env = Env()
    result = search(env, limit=1)
    assert result.truncated and not result.unsupported and not result.solved
