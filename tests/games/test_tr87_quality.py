"""Full-mechanics, reference-quality acceptance tests for TR87."""

import copy
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from arcengine import GameState

from pebby.games.tr87.env import Env, official_levels, replay
from pebby.games.tr87.generate import (
    CONTENT_LEFT,
    CONTENT_RIGHT,
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    env_for,
    generate,
    generate_game,
    validate_full_standard,
)
from pebby.games.tr87 import names
from pebby.games.tr87.layout import extract, translation_trace
from pebby.games.tr87.plan import search
from pebby.games.tr87.reference_profiles import PROFILES, profile_errors


def test_all_official_tiers_have_replayed_teachers_and_mechanics():
    assert len(official_levels()) == 6
    assert DIFFICULTIES == tuple(range(1, len(official_levels()) + 1))
    expected_lengths = (14, 25, 21, 21, 14, 35)
    for index, (length, profile) in enumerate(zip(expected_lengths, PROFILES.values())):
        env = Env()
        env.set_level(index)
        layout = extract(env)
        assert {
            "alter_rules": layout.alter_rules,
            "double_translation": layout.double_translation,
            "tree_translation": layout.tree_translation,
        } == profile["flags"]
        assert layout.budget == profile["budget"]
        result = search(env)
        assert not result.truncated and result.length == length
        replay(env, result.actions)
        assert env.levels_completed == 1


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_each_full_tier_is_deterministic_json_roundtrippable_and_certified(difficulty):
    row = generate(700 + difficulty, difficulty, split="validation")
    assert row is not None
    assert row == generate(700 + difficulty, difficulty, split="validation")
    restored = json.loads(json.dumps(row))
    assert profile_errors(restored) == []
    entry = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
    assert validate_full_standard(restored, entry) == []
    assert restored["split"] == restored["geometry_split"] == "validation"
    assert restored["official_copy"] is False
    mechanics = restored["solution_mechanics"]
    if difficulty == 4:
        assert mechanics["translation_branch"] == "double"
        assert mechanics["double_compositions"] == len(restored["source"]["symbols"])
    if difficulty == 6:
        assert mechanics["translation_branch"] == "tree"
        assert mechanics["tree_expansions"] == len(restored["target"]["symbols"])
        assert mechanics["double_compositions"] == 0
    env = Env([level.clone() for level in official_levels()[:difficulty - 1]]
              + [build_level(restored)])
    env.set_level(difficulty - 1)
    replay(env, restored["solution"])
    assert env.levels_completed == 1 and env.state == GameState.WIN


def test_final_tier_native_winning_trace_uses_mixed_and_repeated_children():
    spec = generate(888, 6, split="train")
    assert spec is not None
    env = env_for(spec)
    replay(env, spec["solution"])
    trace = translation_trace(extract(env))
    assert trace["success"]
    assert trace["tree_mixed_child_expansions"] >= 1
    assert trace["tree_repeated_child_expansions"] >= 1
    assert spec["solution_mechanics"]["tree_mixed_child_expansions"] >= 1
    assert spec["solution_mechanics"]["tree_repeated_child_expansions"] >= 1

    false_evidence = copy.deepcopy(spec)
    false_evidence["solution_mechanics"]["tree_repeated_child_expansions"] = 0
    assert any("repeated-child" in error for error in profile_errors(false_evidence))


def test_full_game_advances_through_all_six_native_contexts_without_forcing():
    specs = generate_game(991, split="train")
    assert specs is not None and [row["difficulty"] for row in specs] == list(DIFFICULTIES)
    env = Env(build_game(specs))
    env.reset()
    for index, spec in enumerate(specs):
        assert env.level_index == index
        assert env.levels_completed == index
        before = env.level_index
        replay(env, spec["context_solution"])
        assert env.levels_completed == index + 1
        if index < len(specs) - 1:
            assert env.level_index == before + 1
            assert env.state == GameState.NOT_FINISHED
    assert env.state == GameState.WIN


def test_live_prefix_recovery_uses_current_state_for_target_and_rule_edit_modes():
    for difficulty in (2, 5, 6):
        spec = generate(880 + difficulty, difficulty, split="test")
        env = Env([level.clone() for level in official_levels()[:difficulty - 1]]
                  + [build_level(spec)])
        env.set_level(difficulty - 1)
        prefix = spec["solution"][:max(1, len(spec["solution"]) // 3)]
        replay(env, prefix)
        assert env.levels_completed == 0
        recovery = search(env)
        assert recovery.actions and not recovery.truncated
        replay(env, recovery.actions)
        assert env.levels_completed == 1


def test_shared_search_entry_is_bounded_and_editable_routes_are_diverse():
    for difficulty in (5, 6):
        official = Env()
        official.set_level(difficulty - 1)
        assert search(official, node_limit=0).truncated
        routes = set()
        for seed in range(8):
            spec = generate(9000 + seed, difficulty, split="train")
            routes.add(tuple(step[0] for step in spec["solution"]))
            live = Env([level.clone() for level in official_levels()[:difficulty - 1]]
                       + [build_level(spec)])
            live.set_level(difficulty - 1)
            result = search(live, limit=spec["native_budget"], node_limit=400_000)
            assert result.actions and not result.truncated
            replay(live, result.actions)
            assert live.levels_completed == 1
        assert len(routes) >= 7


def test_split_identity_and_duplicate_gates_are_enforced():
    rows = [generate(1200 + difficulty, difficulty, split="test") for difficulty in DIFFICULTIES]
    assert all(rows)
    assert len({row["geometry_d4_sha256"] for row in rows}) == len(rows)
    assert len({row["gameplay_sha256"] for row in rows}) == len(rows)
    with pytest.raises(ValueError, match="exactly six"):
        build_game(rows[:-1])
    changed = copy.deepcopy(rows)
    changed[1]["split"] = "train"
    with pytest.raises(ValueError, match="same split"):
        build_game(changed)
    changed = copy.deepcopy(rows)
    changed[1]["geometry_d4_sha256"] = changed[0]["geometry_d4_sha256"]
    with pytest.raises(ValueError, match="invalid|duplicate"):
        build_game(changed)
    changed = copy.deepcopy(rows)
    changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(ValueError, match="ordered"):
        build_game(changed)


def test_contract_and_profile_fail_closed_on_false_evidence():
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert FULL_STANDARD_CONTRACT["source_id"] == "tr87-cd924810"
    assert len(FULL_STANDARD_CONTRACT["curriculum"]) == len(official_levels())
    assert [(entry["difficulty"], entry["context_index"])
            for entry in FULL_STANDARD_CONTRACT["curriculum"]] == list(zip(DIFFICULTIES, range(6)))
    assert all(type(entry["search_work"]) is int and 1 <= entry["search_work"] <= 32_000_000
               for entry in FULL_STANDARD_CONTRACT["curriculum"])
    assert all(FULL_STANDARD_CONTRACT["evidence"].get(key) for key in (
        "official_tier_characterization", "solution_mechanics", "native_budget",
        "context_engine_replay", "novelty_split", "bounded_rejections"))
    assert all(isinstance(item, str) and item for item in FULL_STANDARD_CONTRACT["caveats"])
    caveats = " ".join(FULL_STANDARD_CONTRACT["caveats"])
    assert "tiers 5 and 6" in caveats
    for difficulty in (5, 6):
        constructive = generate(1440 + difficulty, difficulty, split="train")
        assert constructive["solution_optimal"] is False
        assert constructive["proof"]["solution_optimal"] is False
    row = generate(1441, 6, split="train")
    for field, value in (
        ("native_budget", 128),
        ("context_engine_verified", False),
        ("official_copy", True),
        ("geometry_d4_sha256", ""),
    ):
        changed = copy.deepcopy(row)
        changed[field] = value
        assert profile_errors(changed), field


def test_validator_recomputes_route_mechanics_identity_and_handles_malformed_specs():
    row = generate(1771, 4, split="validation")
    entry = FULL_STANDARD_CONTRACT["curriculum"][3]
    assert validate_full_standard(row, entry) == []

    wrong_route = copy.deepcopy(row)
    assert wrong_route["solution"][-1][0] in (1, 2)
    wrong_route["solution"][-1][0] = 3 - wrong_route["solution"][-1][0]
    assert any("does not win" in error for error in validate_full_standard(wrong_route, entry))

    wrong_mechanics = copy.deepcopy(row)
    wrong_mechanics["solution_mechanics"]["edited_groups"] += 1
    assert any("mechanic evidence" in error for error in validate_full_standard(wrong_mechanics, entry))

    wrong_geometry = copy.deepcopy(row)
    wrong_geometry["rules"][0]["x"] += 1
    assert any("identity" in error for error in validate_full_standard(wrong_geometry, entry))

    wrong_work = copy.deepcopy(row)
    wrong_work["search_work"] = wrong_work["proof"]["search_work"] = 1
    assert any("measured routing work" in error
               for error in validate_full_standard(wrong_work, entry))

    malformed = copy.deepcopy(row)
    malformed["rules"] = None
    assert validate_full_standard(malformed, entry)
    malformed = copy.deepcopy(row)
    malformed["proof"]["search_work"] = "many"
    assert validate_full_standard(malformed, entry)
    assert validate_full_standard(None, entry)


def test_invalid_split_and_reduced_game_are_explicit():
    with pytest.raises(ValueError, match="split"):
        generate(0, 1, split="heldout")
    reduced = generate_game(0, split="train", difficulties=(1, 3))
    assert [row["difficulty"] for row in reduced] == [1, 3]
    with pytest.raises(ValueError, match="exactly six"):
        build_game(reduced)


@pytest.mark.parametrize("field,value", [
    ("difficulty", True),
    ("difficulty", 1.0),
    ("attempts", True),
    ("attempts", 1.0),
    ("node_limit", True),
    ("node_limit", 1.0),
])
def test_generator_rejects_non_integer_public_bounds(field, value):
    kwargs = {"difficulty": 1, "attempts": 1, "node_limit": 400_000, "split": "train"}
    kwargs[field] = value
    with pytest.raises(ValueError, match="integer|one of"):
        generate(0, **kwargs)


def test_reduced_smoke_tiers_must_be_unique_and_in_official_order():
    for tiers in ((3, 1), (1, 1), (True, 2), (1.0, 2)):
        with pytest.raises(ValueError, match="difficulties|unique|order"):
            generate_game(0, split="train", difficulties=tiers, attempts=1)


def test_constructive_route_honors_node_limit_and_records_measured_work():
    baseline = generate(701, 1, split="validation")
    assert baseline is not None
    assert baseline["search_work"] == baseline["proof"]["search_work"]
    assert baseline["search_work"] > baseline["solution_length"]
    assert baseline["search_work"] <= PROFILES[1]["search_work"]
    stats = {}
    assert generate(
        701, 1, split="validation", node_limit=1,
        attempts=baseline["generation_attempt"], stats=stats,
    ) is None
    assert stats.get("constructive_search_truncated", 0) >= 1


def test_validator_rejects_recomputed_official_identity_even_if_flag_is_false(monkeypatch):
    module = importlib.import_module("pebby.games.tr87.generate")
    row = generate(1881, 2, split="test")
    entry = FULL_STANDARD_CONTRACT["curriculum"][1]
    monkeypatch.setattr(module, "_OFFICIAL_GEOMETRY", {row["geometry_d4_sha256"]})
    errors = validate_full_standard(row, entry)
    assert row["official_copy"] is False
    assert any("matches an official" in error for error in errors)


def _primary_multigame_module():
    path = Path("/home/stepan/Projects/code/Pebby/pebby/multigame.py")
    name = "pebby._primary_multigame_tr87_contract_test"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def test_primary_shared_schema_and_all_three_split_games_are_compatible():
    primary = _primary_multigame_module()
    generate_module = importlib.import_module("pebby.games.tr87.generate")
    env_module = importlib.import_module("pebby.games.tr87.env")
    plan_module = importlib.import_module("pebby.games.tr87.plan")
    source = primary.source_for("tr87")
    contract = primary._full_standard_contract(source, env_module, generate_module)
    assert contract.source_id == "tr87-cd924810"
    assert contract.status == "ready" and contract.official_level_count == 6
    assert primary._solver_adapter(search) == "action_limit+node_limit"

    # Exercise the current primary full-standard collector through the
    # accepted contract. The explicit replacement preserves the earlier
    # collector fixture shape while leaving the ready status unchanged.
    accepted_overlay = replace(contract, status="ready")
    modules = primary.GameModules(
        source=source,
        env=env_module,
        generate=generate_module,
        plan=plan_module,
        solver_adapter="action_limit+node_limit",
        full_standard=accepted_overlay,
    )

    identity_sets = {}
    for game_index, split in enumerate(("train", "validation", "test")):
        collected = primary.collect_generated_game(
            modules,
            master_seed=2027,
            game_index=game_index,
            difficulties=DIFFICULTIES,
            curriculum=accepted_overlay.curriculum,
            split=split,
            require_full_standard=True,
            limits=primary.SearchLimits(max_actions_per_level=256, max_search_work=400_000),
            generator_attempts=180,
        )
        assert collected.record["status"] == "won", collected.record["errors"]
        assert collected.record["levels_completed"] == contract.official_level_count
        assert collected.record["certified_solution_completed"] is True
        assert collected.record["certified_route_levels_completed"] == contract.official_level_count
        assert collected.record["random_steps"] == 0
        specs = collected.specs
        identity_sets[split] = {
            (spec["geometry_d4_sha256"], spec["gameplay_sha256"]) for spec in specs
        }
    assert identity_sets["train"].isdisjoint(identity_sets["validation"])
    assert identity_sets["train"].isdisjoint(identity_sets["test"])
    assert identity_sets["validation"].isdisjoint(identity_sets["test"])


def test_seed_888_full_game_keeps_complete_sprite_extents_inside_render_gutters():
    specs = generate_game(888, split="train")
    assert specs is not None
    for spec, entry in zip(specs, FULL_STANDARD_CONTRACT["curriculum"]):
        assert validate_full_standard(spec, entry) == []
        for sprite in build_level(spec).get_sprites():
            if sprite.name == names.SPRITE_BACKGROUND:
                continue
            assert int(sprite.x) >= CONTENT_LEFT
            assert int(sprite.x + sprite.width) <= CONTENT_RIGHT
            assert int(sprite.y) >= 0
            assert int(sprite.y + sprite.height) <= 64

    clipped = copy.deepcopy(specs[0])
    clipped["source"]["x0"] = 60
    errors = validate_full_standard(clipped, FULL_STANDARD_CONTRACT["curriculum"][0])
    assert any("render-safe bounds" in error for error in errors)
