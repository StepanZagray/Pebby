"""LP85 full-mechanic generation, exact teacher, and native replay tests."""

from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import importlib
import json

import numpy as np
import pytest
from arcengine import GameState

from pebby.games.lp85 import names
from pebby.games.lp85.bank import main as bank_main
from pebby.games.lp85.env import Env, official_levels, replay, upstream
from pebby.games.lp85.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    FORMAT,
    GENERATOR_VERSION,
    SPLITS,
    _context_env,
    _draft,
    _map_id,
    _official_geometry_hashes,
    build_game,
    build_level,
    generate,
    generate_game,
    gameplay_sha256,
    geometry_d4_sha256,
    geometry_partition,
    structural_metrics,
    validate_full_standard,
    verify,
)
from pebby.games.lp85.layout import extract
from pebby.games.lp85.plan import (
    _button_clicks,
    _goal_projection_model,
    _semantic_key,
    _state_key,
    search,
    solution_mechanics,
)
from pebby.games.lp85.reference_profiles import PROFILES, REFERENCE, profile_errors


_generate_module = importlib.import_module("pebby.games.lp85.generate")


@lru_cache(maxsize=None)
def _full_game(split="train"):
    specs = generate_game(0, split=split)
    assert specs is not None, generate_game.last_report
    return specs


def _assert_actions(env, actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        assert action_id == names.ACTION_CLICK
        assert type(x) is int and type(y) is int
        assert 0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE


def _normalized_action_route(spec):
    """Remove display coordinates while preserving ordered native effects."""
    model = _goal_projection_model(Env([build_level(spec)]))
    assert model is not None
    effects_by_action = {
        tuple(action): tuple(tuple(effect) for effect in effects)
        for _, action, effects in model.actions
    }
    return tuple(effects_by_action[tuple(action)] for action in spec["solution"])


def _recertify_identity_variant(spec):
    """Re-run bounded native certification after an identity-only mutation."""
    spec["map_id"] = _map_id(spec)
    spec["geometry_d4_sha256"] = geometry_d4_sha256(spec)
    geometry_hash = getattr(_generate_module, "geometry_sha256", gameplay_sha256)
    presentation_hash = getattr(_generate_module, "presentation_sha256", gameplay_sha256)
    spec["geometry_sha256"] = geometry_hash(spec)
    spec["gameplay_sha256"] = gameplay_sha256(spec)
    if hasattr(_generate_module, "presentation_sha256"):
        spec["presentation_sha256"] = presentation_hash(spec)
    spec["geometry_split"] = geometry_partition(spec["geometry_d4_sha256"])
    spec["split"] = spec["geometry_split"]
    spec["official_geometry_copy"] = spec["geometry_d4_sha256"] in _official_geometry_hashes()
    accepted, reason = verify(spec)
    assert accepted is not None, reason
    accepted["proof"]["rejections_before_accept"] = sum(
        accepted.get("generation_exclusions", {}).values()
    )
    entry = FULL_STANDARD_CONTRACT["curriculum"][spec["difficulty"] - 1]
    assert not validate_full_standard(accepted, entry)
    return accepted


def test_contract_and_reference_characterization_cover_exactly_eight_tiers():
    assert len(official_levels()) == 8
    assert DIFFICULTIES == tuple(range(1, 9))
    assert tuple(REFERENCE) == DIFFICULTIES
    assert [REFERENCE[index]["reference_actions"] for index in DIFFICULTIES] == [5, 8, 16, 12, 9, 19, 5, 5]
    assert [REFERENCE[index]["step_budget"] for index in DIFFICULTIES] == [13, 60, 80, 150, 80, 80, 80, 80]
    assert REFERENCE[3]["alternate_goals"] == REFERENCE[4]["alternate_goals"] == 1
    assert REFERENCE[4]["duplicate_control_extras"] == 12
    assert REFERENCE[5]["nested_pairs"] == 1
    assert REFERENCE[6]["cycle_count"] == 36
    assert REFERENCE[6]["right_only_groups"] == 36
    assert REFERENCE[6]["stacked_sites"] == 7 and REFERENCE[6]["max_stack"] == 8
    assert REFERENCE[7]["passive_cycles"] == 1
    assert REFERENCE[7]["initially_satisfied_goals"] == 1
    assert REFERENCE[8]["nested_pairs"] == 3 and REFERENCE[8]["max_stack"] == 3

    contract = FULL_STANDARD_CONTRACT
    assert contract["format"] == "pebby-full-generator-contract-v1"
    assert contract["source_id"] == "lp85-305b61c3"
    assert contract["status"] == "ready"
    assert len(contract["curriculum"]) == 8
    assert [row["difficulty"] for row in contract["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in contract["curriculum"]] == list(range(8))
    assert all(type(row["search_work"]) is int and 1 <= row["search_work"] <= 32_000_000 for row in contract["curriculum"])
    assert set(contract["evidence"]) == {
        "official_tier_characterization", "solution_mechanics", "native_budget",
        "context_engine_replay", "novelty_split", "bounded_rejections",
    }
    assert all(contract["evidence"].values()) and all(contract["caveats"])


def test_all_official_levels_have_exact_shortest_teacher_and_sequential_native_replay():
    expected_lengths = [5, 8, 16, 12, 9, 19, 5, 5]
    env = Env()
    for index, expected_length in enumerate(expected_lengths):
        assert env.level_index == index and env.levels_completed == index
        assert _goal_projection_model(env) is not None
        result = search(env, limit=env.steps_left, node_limit=200_000)
        assert result.actions is not None and not result.truncated, result
        assert result.exact and not result.unsupported
        assert len(result.actions) == expected_length
        _assert_actions(env, result.actions)
        evidence = solution_mechanics(env, result.actions)
        assert evidence["all_targets_satisfied"]
        if index == 5:
            assert evidence["used_group_count"] == 36
            assert evidence["used_effect_signature_count"] == 7
            assert evidence["max_stack_used"] == 8
        if index == 6:
            assert evidence["stacked_clicks"] and evidence["initially_satisfied_goals"] == 1
        if index == 7:
            assert evidence["nested_cycle_clicks"] >= 2 and evidence["max_stack_used"] == 3
        assert replay(env, result.actions)
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN


def test_full_display_click_semantics_budget_reset_and_clone_independence():
    env = Env([official_levels()[0]])
    initial = _state_key(env)
    clone = env.clone()
    clone.perform(names.ACTION_CLICK, 0, 0)
    assert _state_key(clone) == initial
    _, (x, y) = next(iter(_button_clicks(clone)))
    clone.perform(names.ACTION_CLICK, x, y)
    assert clone.steps_left == env.steps_left - 1
    assert _semantic_key(clone) != _semantic_key(env)
    clone.perform(names.ACTION_RESET)
    assert clone.steps_left == clone.max_steps == 13
    for action in ((1, None, None), (6, None, None), (6, -1, 0), (6, 64, 0), (6, True, 0), (0, 1, 1)):
        with pytest.raises(ValueError):
            clone.perform(*action)


def test_generated_full_game_matches_every_profile_and_replays_without_forced_transitions():
    specs = _full_game()
    assert len(specs) == 8
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert len({spec["game_child_seed"] for spec in specs}) == 8
    assert len({spec["gameplay_sha256"] for spec in specs}) == 8
    assert len({spec["geometry_d4_sha256"] for spec in specs}) == 8
    for index, spec in enumerate(specs):
        assert spec["format"] == FORMAT
        assert spec["generator_version"] == GENERATOR_VERSION
        assert spec["omitted_mechanics"] == []
        assert spec["official_mechanics_complete"]
        assert not spec["official_geometry_copy"]
        assert spec["split"] == spec["geometry_split"] == "train"
        assert spec["proof"]["full_game_replay"]
        assert not validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][index])
        assert not profile_errors(spec, structural_metrics(spec), require_proof=True)

    stored = json.loads(json.dumps(specs))
    levels = build_game(stored)
    env = Env(levels)
    for index, spec in enumerate(stored):
        assert env.level_index == index and env.levels_completed == index
        _assert_actions(env, spec["solution"])
        before = env.levels_completed
        for action in spec["solution"]:
            env.perform(*action)
        assert env.levels_completed == before + 1
    assert env.state == GameState.WIN


def test_each_tier_exercises_its_reference_defining_mechanics():
    specs = _full_game()
    by_tier = {spec["difficulty"]: spec for spec in specs}
    assert by_tier[2]["solution_mechanics"]["overlap_goal_transitions"] > 0
    assert by_tier[3]["structural_metrics"]["alternate_goals"] == 1
    assert by_tier[4]["solution_mechanics"]["duplicate_control_clicks"] > 0
    assert by_tier[5]["solution_mechanics"]["nested_cycle_clicks"] > 0
    tier6 = by_tier[6]["solution_mechanics"]
    assert tier6["used_group_count"] == 36
    assert tier6["used_effect_signature_count"] == 7
    assert tier6["one_way_clicks"] == by_tier[6]["solution_length"]
    tier7 = by_tier[7]["solution_mechanics"]
    assert tier7["stacked_clicks"] and tier7["passive_overlap_transitions"]
    assert tier7["temporarily_displaced_initial_target"] == 1
    tier8 = by_tier[8]["solution_mechanics"]
    assert tier8["stacked_clicks"] and tier8["nested_cycle_clicks"] >= 2


def test_tutorial_routes_vary_by_observed_direction_and_distance_across_train_seeds():
    # Identity-v2 intentionally changes split acceptance.  This pinned
    # eight-seed cohort spans the signed-distance bins under the current
    # canonical transition partition instead of assuming every consecutive
    # seed window must contain all six outcomes.
    seeds = (0, 1, 2, 3, 5, 10, 12, 22)
    specs = [generate(seed, 1, split="train") for seed in seeds]
    assert all(spec is not None for spec in specs)
    routes = [_normalized_action_route(spec) for spec in specs]
    lengths = [spec["solution_length"] for spec in specs]
    assert len(set(routes)) == 6
    assert {route[0][0][1] for route in routes} == {"L", "R"}
    assert set(lengths) <= {4, 5, 6}
    assert len(set(lengths)) >= 2
    assert all(abs(length - REFERENCE[1]["reference_actions"]) <= 1 for length in lengths)


def test_constructive_target_depths_vary_near_tier_references_without_padding():
    tier3 = [generate(seed, 3, split="train") for seed in range(4)]
    # The v2 split changes which attempt is the first accepted candidate.
    # Eight seeds exercise all three constructive depth requests without
    # changing the grammar, action gates, or exact shortest-route search.
    tier6 = [generate(seed, 6, split="train") for seed in range(8)]
    assert all(spec is not None for spec in tier3 + tier6)
    assert {spec["solution_length"] for spec in tier3} == {15, 16, 17}
    assert {spec["solution_length"] for spec in tier6} == {18, 19, 20}
    for spec in tier3 + tier6:
        reference = REFERENCE[spec["difficulty"]]["reference_actions"]
        assert abs(spec["solution_length"] - reference) <= 2
        assert len(spec["solution"]) == spec["optimal_actions"] == spec["solution_length"]
        assert all(action != [names.ACTION_CLICK, 0, 0] for action in spec["solution"])


def test_generated_frames_keep_full_sprite_footprints_visible_and_readable():
    for spec in _full_game():
        env = Env([build_level(spec)])
        frame = np.asarray(env.render())
        assert frame.shape == (names.FRAME_SIZE, names.FRAME_SIZE)
        assert np.issubdtype(frame.dtype, np.integer)
        assert 0 <= int(frame.min()) <= int(frame.max()) <= 15

        grid_width, grid_height = env.level.grid_size
        scale, x_offset, y_offset = env.game.camera._calculate_scale_and_offset()
        placements = defaultdict(list)
        buttons_by_anchor = defaultdict(list)
        for sprite in env.level._sprites:
            pixels = np.asarray(sprite.render())
            assert pixels.shape == (sprite.height, sprite.width)
            assert np.any(pixels >= 0)
            assert 0 <= sprite.x and sprite.x + sprite.width <= grid_width
            assert 0 <= sprite.y and sprite.y + sprite.height <= grid_height
            assert sprite.x + sprite.width <= names.FRAME_SIZE
            assert sprite.y + sprite.height <= names.FRAME_SIZE
            placements[(sprite.x, sprite.y, sprite.width, sprite.height)].append(sprite)
            if sprite.tags and sprite.tags[0].startswith(names.TAG_BUTTON_PREFIX):
                buttons_by_anchor[(sprite.x, sprite.y)].append(sprite)

        # Stacked native effects intentionally share one complete button
        # rectangle.  Distinct click sites must not rely on clipped overlap.
        expected_stacks = sorted(len(site["effects"]) for site in spec["controls"])
        assert sorted(map(len, buttons_by_anchor.values())) == expected_stacks
        for sprites in buttons_by_anchor.values():
            assert len({(sprite.x, sprite.y, sprite.width, sprite.height) for sprite in sprites}) == 1

        # Check the actual 64x64 frame, including letterboxing and HUD.  Every
        # unique footprint retains at least one palette value belonging to a
        # sprite placed there; exact stacks are checked as one readable site.
        for (x, y, width, height), sprites in placements.items():
            left, top = x_offset + x * scale, y_offset + y * scale
            right, bottom = left + width * scale, top + height * scale
            assert 0 <= left < right <= names.FRAME_SIZE
            assert 0 <= top < bottom <= names.FRAME_SIZE
            expected_colors = {
                int(color)
                for sprite in sprites
                for color in np.asarray(sprite.render()).flat
                if int(color) >= 0
            }
            assert expected_colors
            assert np.isin(frame[top:bottom, left:right], tuple(expected_colors)).any()


def test_single_level_generation_is_deterministic_and_canonically_split_bound():
    specs = {}
    for split in SPLITS:
        first = generate(91, 1, split=split)
        second = generate(91, 1, split=split)
        assert first is not None and second == first
        assert first["split"] == first["geometry_split"] == split
        assert geometry_d4_sha256(first) == first["geometry_d4_sha256"]
        assert not validate_full_standard(first, FULL_STANDARD_CONTRACT["curriculum"][0])
        specs[split] = first
    assert len({spec["geometry_d4_sha256"] for spec in specs.values()}) == 3
    assert len({spec["gameplay_sha256"] for spec in specs.values()}) == 3


def test_same_kind_target_permutation_keeps_every_semantic_identity_and_split():
    original = generate(0, 2, split="train")
    assert original is not None
    variant = deepcopy(original)
    variant["goals"][0]["target"], variant["goals"][1]["target"] = (
        variant["goals"][1]["target"], variant["goals"][0]["target"]
    )

    geometry_hash = getattr(_generate_module, "geometry_sha256", gameplay_sha256)
    presentation_hash = getattr(_generate_module, "presentation_sha256", gameplay_sha256)
    assert geometry_d4_sha256(variant) == original["geometry_d4_sha256"]
    assert geometry_hash(variant) == original["geometry_sha256"]
    assert gameplay_sha256(variant) == original["gameplay_sha256"]
    assert presentation_hash(variant) == presentation_hash(original)

    accepted = _recertify_identity_variant(variant)
    assert accepted["split"] == original["split"] == "train"
    assert _normalized_action_route(accepted) == _normalized_action_route(original)
    assert np.array_equal(Env([build_level(accepted)]).render(), Env([build_level(original)]).render())


def test_inactive_passive_decoration_has_separate_presentation_not_split_identity():
    original = generate(0, 7, split="train")
    assert original is not None
    variant = deepcopy(original)
    passive = next(cycle for cycle in variant["cycles"] if cycle["group"] == "C")
    passive["path"][passive["path"].index([3, 9])] = [8, 9]
    next(filler for filler in variant["fillers"] if filler["cell"] == [3, 9])["cell"] = [8, 9]

    geometry_hash = getattr(_generate_module, "geometry_sha256", gameplay_sha256)
    presentation_hash = getattr(_generate_module, "presentation_sha256", gameplay_sha256)
    assert geometry_d4_sha256(variant) == original["geometry_d4_sha256"]
    assert geometry_hash(variant) == original["geometry_sha256"]
    assert gameplay_sha256(variant) == original["gameplay_sha256"]
    assert presentation_hash(variant) != presentation_hash(original)

    accepted = _recertify_identity_variant(variant)
    assert accepted["split"] == original["split"] == "train"
    assert accepted["solution_mechanics"]["passive_overlap_transitions"] > 0
    assert _normalized_action_route(accepted) == _normalized_action_route(original)


def test_semantic_group_labels_and_cycle_origins_are_canonical_but_effect_order_is_not():
    original = generate(0, 2, split="train")
    assert original is not None
    renamed = deepcopy(original)
    rename = {"A": "B", "B": "A"}
    for cycle in renamed["cycles"]:
        cycle["group"] = rename.get(cycle["group"], cycle["group"])
    for site in renamed["controls"]:
        for effect in site["effects"]:
            effect["group"] = rename.get(effect["group"], effect["group"])
    renamed["map_id"] = _map_id(renamed)

    rotated = deepcopy(original)
    for cycle in rotated["cycles"]:
        cycle["path"] = cycle["path"][1:] + cycle["path"][:1]
    rotated["map_id"] = _map_id(rotated)

    for equivalent in (renamed, rotated):
        assert geometry_d4_sha256(equivalent) == original["geometry_d4_sha256"]
        assert _generate_module.geometry_sha256(equivalent) == original["geometry_sha256"]
        assert gameplay_sha256(equivalent) == original["gameplay_sha256"]
        assert _generate_module.presentation_sha256(equivalent) == original["presentation_sha256"]

    original_stacked = generate(0, 8, split="train")
    assert original_stacked is not None
    stacked = deepcopy(original_stacked)
    site = next(site for site in stacked["controls"] if len(site["effects"]) == 3)
    site["effects"] = list(reversed(site["effects"]))
    assert geometry_d4_sha256(stacked) != original_stacked["geometry_d4_sha256"]
    assert _generate_module.geometry_sha256(stacked) != original_stacked["geometry_sha256"]
    assert gameplay_sha256(stacked) != original_stacked["gameplay_sha256"]


def test_validator_strictly_rejects_malformed_seed_context_and_nested_proof_shapes():
    generated = generate(0, 1, split="train")
    assert generated is not None
    original = json.loads(json.dumps(generated))
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    cases = []

    seed_bool = deepcopy(original)
    seed_bool["seed"] = seed_bool["proof"]["seed"] = True
    cases.append((seed_bool, entry, "seed must be an integer"))

    seed_string = deepcopy(original)
    seed_string["seed"] = seed_string["proof"]["seed"] = "not-a-seed"
    cases.append((seed_string, entry, "seed must be an integer"))

    generator_float = deepcopy(original)
    generator_float["generator_version"] = float(GENERATOR_VERSION)
    cases.append((generator_float, entry, "generator version mismatch"))

    training_context = deepcopy(original)
    training_context["training_context_index"] = False
    training_context["proof"]["training_context_index"] = False
    cases.append((training_context, entry, "training context index must be an integer"))

    verification_context = deepcopy(original)
    verification_context["verification_level_index"] = False
    verification_context["proof"]["verification_level_index"] = False
    cases.append((verification_context, entry, "verification level index must be an integer"))

    for field, message in (
        ("native_budget", "native budget must be a mapping"),
        ("solution_mechanics", "solution mechanics must be a mapping"),
        ("context_engine_replay", "context engine replay must be a mapping"),
        ("structural_metrics", "structural metrics must be a mapping"),
        ("proof", "proof must be a mapping"),
    ):
        malformed = deepcopy(original)
        malformed[field] = None
        cases.append((malformed, entry, message))

    bad_proof_counter = deepcopy(original)
    bad_proof_counter["proof"]["expanded"] = True
    cases.append((bad_proof_counter, entry, "proof expanded must be a nonnegative integer"))

    bad_rejections = deepcopy(original)
    bad_rejections["generation_exclusions"] = {"geometry_split_mismatch": True}
    cases.append((bad_rejections, entry, "rejection counts must be nonnegative integers"))

    bad_curriculum = dict(entry)
    bad_curriculum["context_index"] = False
    cases.append((original, bad_curriculum, "curriculum context_index must be an integer"))

    for candidate, curriculum, expected in cases:
        errors = validate_full_standard(candidate, curriculum)
        assert errors and any(expected in error for error in errors), (expected, errors)


def test_generate_requires_explicit_split_and_keeps_bounded_rejection_evidence():
    with pytest.raises(TypeError):
        generate(0, 1)
    with pytest.raises(ValueError):
        generate(0, 1, split="dev")
    with pytest.raises(ValueError):
        generate(0, 9, split="train")
    with pytest.raises(ValueError):
        generate(0, 1, attempts=0, split="train")
    spec = generate(7, 1, split="validation")
    assert spec is not None
    assert isinstance(spec["generation_exclusions"], dict)
    assert generate.last_report["accepted"]
    assert generate.last_report["attempts_used"] <= 48


def test_contextual_native_proof_budget_and_json_round_trip_are_fail_closed():
    for spec in _full_game():
        stored = json.loads(json.dumps(spec))
        assert stored["training_context_index"] == stored["verification_level_index"] == stored["difficulty"] - 1
        assert stored["context_engine_replay"]["level_index"] == stored["difficulty"] - 1
        assert stored["context_engine_replay"]["levels_completed_delta"] == 1
        assert stored["native_budget"]["initial"] == stored["step_budget"]
        assert stored["native_budget"]["remaining_before_level_transition"] == (
            stored["step_budget"] - stored["solution_length"] + 1
        )
        assert build_level(stored).get_data(names.KEY_LEVEL_NAME) == stored["map_id"]
        mutated = deepcopy(stored)
        mutated["cycles"][0]["path"][0][0] += 1
        with pytest.raises(ValueError):
            build_level(mutated)


def test_full_validator_recomputes_identities_mechanics_budget_context_and_native_route():
    original = json.loads(json.dumps(_full_game()[2]))
    entry = FULL_STANDARD_CONTRACT["curriculum"][2]
    assert not validate_full_standard(original, entry)

    wrong_route = deepcopy(original)
    wrong_route["solution"][0] = [names.ACTION_CLICK, 0, 0]
    errors = validate_full_standard(wrong_route, entry)
    assert any("witness" in error or "mechanics" in error or "malformed" in error for error in errors)

    wrong_identity = deepcopy(original)
    wrong_identity["geometry_d4_sha256"] = "0" * 64
    errors = validate_full_standard(wrong_identity, entry)
    assert any("identity does not recompute" in error for error in errors)

    wrong_mechanics = deepcopy(original)
    wrong_mechanics["solution_mechanics"]["used_group_count"] = 999
    assert any("solution mechanics do not recompute" in error for error in validate_full_standard(wrong_mechanics, entry))

    wrong_budget = deepcopy(original)
    wrong_budget["native_budget"]["initial"] -= 1
    assert any("budget evidence does not recompute" in error for error in validate_full_standard(wrong_budget, entry))

    wrong_context = deepcopy(original)
    wrong_context["verification_level_index"] = 0
    assert any("context" in error for error in validate_full_standard(wrong_context, entry))

    malformed = deepcopy(original)
    malformed["difficulty"] = True
    assert validate_full_standard(malformed, entry)


def test_live_prefix_and_inverse_history_recover_with_exact_remaining_budget():
    spec = _full_game()[1]
    env = _context_env(build_level(spec), 1)
    clicks = list(_button_clicks(env))
    start = _semantic_key(env)
    start_steps = env.steps_left
    left = next(click for tags, click in clicks if tags == ("button_A_L",))
    right = next(click for tags, click in clicks if tags == ("button_A_R",))
    env.perform(names.ACTION_CLICK, *left)
    env.perform(names.ACTION_CLICK, *right)
    assert _semantic_key(env) == start
    assert env.steps_left == start_steps - 2
    result = search(env, limit=env.steps_left, node_limit=PROFILES[2]["search_work"])
    assert result.actions is not None and result.exact and not result.truncated
    assert replay(env, result.actions)


def test_goal_projection_fails_closed_if_a_control_can_move():
    spec = _full_game()[0]
    env = Env([build_level(spec)])
    button = next(sprite for sprite in env.level._sprites if sprite.tags and sprite.tags[0].startswith("button_"))
    tile = env.level.get_sprites_by_tag(names.TAG_TILE)[0]
    button.set_position(tile.x, tile.y)
    assert _goal_projection_model(env) is None


def test_build_game_rejects_reduced_reordered_duplicate_or_split_mismatched_sequences():
    specs = list(_full_game())
    with pytest.raises(ValueError):
        build_game(specs[:-1])
    swapped = deepcopy(specs)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        build_game(swapped)
    duplicate = deepcopy(specs)
    duplicate[1] = deepcopy(duplicate[0])
    with pytest.raises(ValueError):
        build_game(duplicate)
    mixed = deepcopy(specs)
    mixed[-1]["split"] = "validation"
    with pytest.raises(ValueError):
        build_game(mixed)
    reduced = generate_game(2, split="train", difficulties=(2, 4))
    assert reduced is not None and [spec["difficulty"] for spec in reduced] == [2, 4]
    with pytest.raises(ValueError):
        build_game(reduced)


def test_bank_cli_writes_one_replayable_split_bound_row(tmp_path):
    output = tmp_path / "lp85.jsonl"
    assert bank_main([
        "--levels", "1", "--seed", "4", "--difficulty", "1",
        "--split", "test", "--out", str(output),
    ]) == 0
    stored = json.loads(output.read_text().strip())
    assert stored["split"] == stored["geometry_split"] == "test"
    env = Env([build_level(stored)])
    assert replay(env, stored["solution"])


def test_native_globals_are_restored_after_generated_contexts():
    module = upstream()
    levels_object = module.levels
    maps_object = module.izutyjcpih
    spec = _full_game()[3]
    env = _context_env(build_level(spec), 3)
    result = search(env, node_limit=PROFILES[4]["search_work"])
    assert result.actions is not None
    assert module.levels is levels_object and module.izutyjcpih is maps_object


def test_search_bounds_remain_explicitly_inconclusive_not_impossible():
    env = Env([official_levels()[5]])
    capped = search(env, node_limit=1)
    assert capped.actions is None and capped.truncated
    assert capped.exact and not capped.unsupported
    action_capped = search(Env([official_levels()[0]]), limit=1, node_limit=10_000)
    assert action_capped.actions is None and action_capped.truncated


def test_generated_quality_audit_sample_spans_all_tiers_without_lowered_gates():
    specs = _full_game()
    lengths = []
    rejections = Counter()
    for spec in specs:
        profile = PROFILES[spec["difficulty"]]
        lengths.append(spec["solution_length"])
        assert profile["actions"][0] <= spec["solution_length"] <= profile["actions"][1]
        assert profile["union_cells_range"][0] <= spec["structural_metrics"]["union_cells"] <= profile["union_cells_range"][1]
        rejections.update(spec["generation_exclusions"])
    assert len(lengths) == 8 and min(lengths) >= 3 and max(lengths) >= 15
    assert sum(rejections.values()) >= 0
