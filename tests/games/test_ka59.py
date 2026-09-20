"""KA59 full-mechanics generation, teacher, and native replay contracts."""

import copy
import json

from arcengine import GameState

from pebby.games.ka59 import names
from pebby.games.ka59.bank import main as bank_main
from pebby.games.ka59.env import Env, official_levels, replay
from pebby.games.ka59.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    generate,
    generate_game,
    validate_full_standard,
)
from pebby.games.ka59 import generate as generate_module
from pebby.games.ka59.layout import extract
from pebby.games.ka59.plan import _selection_clicks, _state_key, search, solve
from pebby.games.ka59.reference_profiles import PROFILES, structural_metrics
from pebby.multigame import MultiGameEnv, collect_generated_game, preflight


def _assert_actions(env, actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        if action_id == names.ACTION_CLICK:
            assert isinstance(x, int) and isinstance(y, int)
            assert 0 <= x < 64 and 0 <= y < 64
        else:
            assert x is None and y is None


def test_full_standard_contract_records_root_readiness_and_bounded_caveats():
    contract = FULL_STANDARD_CONTRACT
    assert contract["status"] == "ready"
    assert contract["evidence"]["root_acceptance"] == (
        "ka59-root-readiness-authorization-2026-09-19"
    )
    assert "21-row" in contract["evidence"]["independent_blast_type_closure"]
    caveats = "\n".join(contract["caveats"])
    for required in (
        "one official level exists per tier",
        "not optimality claims",
        "D4 identity does not prove graph-isomorphism novelty",
        "passive player-goal relation",
        "finite and bounded",
        "off-frame and redundant devices may be nonessential",
        "earlier counterfactual wins",
        "historical 168-row audit predates the blast correction",
        "rejection diagnostics",
    ):
        assert required in caveats


def test_official_inventory_matches_all_seven_reference_profiles():
    levels = official_levels()
    assert DIFFICULTIES == tuple(range(1, 8))
    assert len(levels) == len(PROFILES) == 7
    for difficulty, level in enumerate(levels, 1):
        profile = PROFILES[difficulty]
        assert level.grid_size == (profile["grid_size"],) * 2
        assert level.get_data(names.KEY_STEPS) == profile["step_budget"]
        assert sorted(sprite.name for sprite in level.get_sprites_by_tag(names.TAG_BOX)) == sorted(profile["boxes"])
        assert len(level.get_sprites_by_tag(names.TAG_TARGET)) == profile["targets"]
        assert sorted(sprite.name for sprite in level.get_sprites_by_tag(names.TAG_PLAYER)) == sorted(profile["players"])
        assert len(level.get_sprites_by_tag(names.TAG_PLAYER_TARGET)) == profile["player_targets"]
        assert not level.get_sprites_by_tag(names.TAG_ENEMY)


def test_teacher_has_native_positive_witness_for_every_official_tier_sequentially():
    env = Env()
    measured = []
    for context_index, difficulty in enumerate(DIFFICULTIES):
        assert env.level_index == context_index
        result = search(
            env,
            limit=PROFILES[difficulty]["step_budget"],
            node_limit=120_000,
        )
        assert result.actions is not None and not result.truncated, result
        assert result.exact and not result.unsupported
        _assert_actions(env, result.actions)
        measured.append(len(result.actions))
        assert replay(env, result.actions)
        assert env.levels_completed == context_index + 1
    assert env.state == GameState.WIN
    # These are positive route measurements, not shortest-route assertions.
    assert all(length <= PROFILES[difficulty]["step_budget"] for difficulty, length in zip(DIFFICULTIES, measured))
    assert all(type(PROFILES[difficulty]["reference_actions"]) is int for difficulty in DIFFICULTIES)


def test_move_click_budget_clone_and_full_state_key_cover_dynamic_entities():
    env = Env([official_levels()[6]])
    initial = _state_key(env)
    target_index, (x, y) = next(iter(_selection_clicks(env)))
    clone = env.clone()
    clone.perform(names.ACTION_CLICK, x, y)
    assert clone.steps_left == env.steps_left - 1
    assert _state_key(clone) != initial
    assert _state_key(env) == initial
    assert clone.boxes().index(clone.selected()) == target_index
    clone.perform(names.ACTION_UP)
    assert clone.steps_left == env.steps_left - 2


def test_all_tiers_and_splits_are_json_round_trip_valid_native_witnesses():
    identities = {split: set() for split in ("train", "validation", "test")}
    for split in identities:
        for difficulty in DIFFICULTIES:
            spec = generate(0, difficulty, split=split)
            assert spec is not None
            stored = json.loads(json.dumps(spec))
            entry = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            assert validate_full_standard(stored, entry) == []
            assert replay(Env([build_level(stored)]), stored["solution"])
            assert stored["geometry_split"] == split
            assert stored["solution_length"] == len(stored["solution"])
            identities[split].add(
                (stored["geometry_d4_sha256"], stored["gameplay_sha256"])
            )
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert identities[left].isdisjoint(identities[right])


def test_generated_tiers_are_solved_through_public_search_entry():
    for difficulty in DIFFICULTIES:
        spec = generate(0, difficulty)
        assert spec is not None
        env = Env([build_level(spec)])
        result = search(
            env,
            limit=spec["step_budget"],
            node_limit=20_000 if difficulty == 4 else 5_000,
        )
        assert result.actions is not None and not result.truncated, result
        assert replay(env, result.actions)


def test_complete_game_mode_builds_and_replays_all_contexts_without_forcing_transitions():
    specs = generate_game(91, split="train")
    assert specs is not None and len(specs) == 7
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["game_level_index"] for spec in specs] == list(range(7))
    levels = build_game(json.loads(json.dumps(specs)))
    env = Env(levels)
    for expected_index, spec in enumerate(specs):
        assert env.level_index == expected_index
        assert replay(env, spec["solution"])
        assert env.levels_completed == expected_index + 1
    assert env.state == GameState.WIN


def test_build_game_rejects_shortened_shifted_or_mixed_split_curricula():
    specs = generate_game(7)
    assert specs is not None
    for changed in (
        specs[:3],
        [{**spec, "context_index": 0} if index == 1 else spec for index, spec in enumerate(specs)],
        [{**spec, "split": "validation"} if index == 2 else spec for index, spec in enumerate(specs)],
    ):
        try:
            build_game(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a shortened, shifted, or mixed-split game")


def test_validator_recomputes_route_geometry_partition_and_proof():
    spec = generate(4, 5, split="validation")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][4]
    assert validate_full_standard(spec, entry) == []

    route = json.loads(json.dumps(spec))
    route["solution"][0][0] = (
        names.ACTION_LEFT
        if route["solution"][0][0] == names.ACTION_RIGHT
        else names.ACTION_RIGHT
    )
    assert validate_full_standard(route, entry)

    geometry = json.loads(json.dumps(spec))
    row = next(index for index, value in enumerate(geometry["wall_rows"]) if "1" in value)
    geometry["wall_rows"][row] = geometry["wall_rows"][row].replace("1", "0", 1)
    assert any("recomput" in error or "profile" in error for error in validate_full_standard(geometry, entry))

    proof = json.loads(json.dumps(spec))
    proof["proof"]["mechanics"]["blast_pushes"] = 0
    assert any("proof.mechanics" in error for error in validate_full_standard(proof, entry))

    split = json.loads(json.dumps(spec))
    split["split"] = "train"
    assert validate_full_standard(split, entry)


def test_bounded_sample_has_puzzle_gameplay_and_action_sequence_diversity():
    for difficulty in DIFFICULTIES:
        specs = [generate(seed, difficulty) for seed in range(8)]
        assert all(spec is not None for spec in specs)
        assert len({spec["geometry_d4_sha256"] for spec in specs}) >= 6
        assert len({spec["gameplay_sha256"] for spec in specs}) >= 6
        # Tiny tutorial tiers may repeat route families, but not one canonical
        # fixed action script across the eight-seed sample.
        assert len({spec["action_sha256"] for spec in specs}) >= 3


def test_live_prefix_recovery_and_explicit_bounds():
    spec = generate(2, 7)
    assert spec is not None
    env = Env([build_level(spec)])
    for action in spec["solution"][:5]:
        env.perform(*action)
    result = search(env, limit=env.steps_left, node_limit=5_000)
    assert result.actions is not None and replay(env, result.actions)

    capped = search(Env([official_levels()[0]]), node_limit=1)
    assert capped.actions is None and capped.truncated and not capped.unsupported
    assert solve(Env([official_levels()[0]]), node_limit=1) is None and solve.truncated


def test_profile_metrics_recompute_instead_of_trusting_metadata():
    spec = generate(0, 2)
    assert spec is not None
    assert structural_metrics(spec) == spec["structural_metrics"]
    assert spec["structural_metrics"]["wall_pixels"] == sum(
        row.count("1") for row in spec["wall_rows"]
    )
    assert extract(Env([build_level(spec)])).exact


def test_bank_cli_legacy_preflight_and_collector(tmp_path):
    output = tmp_path / "ka59.jsonl"
    assert bank_main([
        "--levels", "1", "--seed", "4", "--difficulty", "1",
        "--split", "validation", "--out", str(output), "--node-limit", "5000",
    ]) == 0
    stored = json.loads(output.read_text().strip())
    assert stored["split"] == "validation"
    assert replay(Env([build_level(stored)]), stored["solution"])

    diagnostic_output = tmp_path / "ka59-diagnostic.jsonl"
    assert bank_main([
        "--levels", "1", "--seed", "0", "--max-seeds", "20",
        "--difficulty", "2", "--split", "validation", "--attempts", "1",
        "--out", str(diagnostic_output),
    ]) == 0
    rejection_path = tmp_path / "ka59-diagnostic.jsonl.rejections.jsonl"
    failures = [json.loads(line) for line in rejection_path.read_text().splitlines()]
    assert failures
    assert all(failure["attempt_limit"] == 1 and failure["reasons"] for failure in failures)

    modules, = preflight(["ka59"])
    game = MultiGameEnv.from_specs(modules, [stored])
    assert game.progress.level_index == 0
    collected = collect_generated_game(
        modules,
        master_seed=23,
        game_index=0,
        difficulties=[1],
        outer_generation_attempts=2,
        generator_attempts=8,
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_completed"] == 1


def _trajectory(spec, removed_bomb=None):
    level = build_level(spec)
    if removed_bomb is not None:
        bombs = level.get_sprites_by_tag(names.TAG_EXPLOSIVE)
        level.remove_sprite(bombs[removed_bomb])
    env = Env([level])
    states = []
    first_win = None
    for index, action in enumerate(spec["solution"], 1):
        if env.state != GameState.NOT_FINISHED:
            break
        observation = env.perform(*action)
        states.append((
            tuple((box.x, box.y) for box in env.boxes()),
            tuple(
                (player.x, player.y)
                for player in env.level.get_sprites_by_tag(names.TAG_PLAYER)
            ),
        ))
        if observation.state == GameState.WIN or env.levels_completed:
            first_win = index
    return states, first_win


def _trajectory_with_suppressed_detonation(spec, disabled_bomb):
    """Replay while retaining one bomb's native collidable body and timer reset."""
    env = Env([build_level(spec)])
    bombs = env.level.get_sprites_by_tag(names.TAG_EXPLOSIVE)
    disabled = bombs[disabled_bomb]
    native_charge = env.game.lflcissmce

    def charge_without_disabled_blast():
        ready = native_charge()
        if disabled in ready:
            env.game.pxqdkrdaye(disabled)
        return [bomb for bomb in ready if bomb is not disabled]

    env.game.lflcissmce = charge_without_disabled_blast
    states = []
    first_win = None
    for index, action in enumerate(spec["solution"], 1):
        if env.state != GameState.NOT_FINISHED:
            break
        observation = env.perform(*action)
        states.append((
            tuple((box.x, box.y) for box in env.boxes()),
            tuple(
                (player.x, player.y)
                for player in env.level.get_sprites_by_tag(names.TAG_PLAYER)
            ),
        ))
        if observation.state == GameState.WIN or env.levels_completed:
            first_win = index
    return states, first_win


def test_d4_identity_normalizes_native_rectangular_transpose():
    spec = generate(0, 2, split="train")
    assert spec is not None
    transposed = copy.deepcopy(spec)
    grid = spec["grid_size"]
    transposed["wall_rows"] = [
        "".join(spec["wall_rows"][x][y] for x in range(grid))
        for y in range(grid)
    ]
    swapped = {names.BOX_3X6: names.BOX_6X3, names.BOX_6X3: names.BOX_3X6}
    for box in transposed["boxes"]:
        box["start"] = box["start"][::-1]
        box["target"] = box["target"][::-1]
        box["prototype"] = swapped.get(box["prototype"], box["prototype"])
    moves = {(0, -3): names.ACTION_LEFT, (0, 3): names.ACTION_RIGHT,
             (-3, 0): names.ACTION_UP, (3, 0): names.ACTION_DOWN}
    commands = [
        ("select", token[1]) if token[0] == "select"
        else ("move", moves[tuple(token[1:])])
        for token in spec["semantic_solution"]
    ]
    _, actions = generate_module._execute_commands(transposed, commands)
    mechanics, semantic = generate_module._route_certificate(transposed, actions, 1)
    original_ids = generate_module._identities(spec, spec["semantic_solution"])
    transposed_ids = generate_module._identities(transposed, semantic)
    assert mechanics["won"] and len(actions) == spec["solution_length"]
    assert transposed_ids[:2] == original_ids[:2]
    assert transposed_ids[3] == original_ids[3]


def test_validator_rejects_missing_or_contradictory_outcome_and_proof_fields():
    spec = generate(0, 3)
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][2]
    mutations = (
        ("engine_verified", False),
        ("engine_win", False),
        ("context_engine_verified", False),
        ("levels_completed", 99),
        ("search_exact", True),
        ("search_performed", True),
        ("search_truncated", True),
        ("generation_attempt", 0),
    )
    for key, value in mutations:
        changed = copy.deepcopy(spec)
        changed[key] = value
        assert validate_full_standard(changed, entry), key
    changed = copy.deepcopy(spec)
    changed["search_limit"] = -1
    changed["proof"]["search_limit"] = -1
    assert validate_full_standard(changed, entry)
    changed = copy.deepcopy(spec)
    del changed["proof"]["search_truncated"]
    assert validate_full_standard(changed, entry)

    for key in ("context_index", "training_context_index", "verification_level_index"):
        changed = copy.deepcopy(spec)
        changed[key] = float(changed[key])
        assert validate_full_standard(changed, entry), key
    changed = copy.deepcopy(spec)
    changed["generator_version"] = float(changed["generator_version"])
    assert validate_full_standard(changed, entry)
    changed = copy.deepcopy(spec)
    changed["proof"]["mechanics"]["native_budget_spent"] = float(
        changed["proof"]["mechanics"]["native_budget_spent"]
    )
    assert validate_full_standard(changed, entry)


def test_explosive_tiers_store_body_retaining_detonation_and_setup_evidence():
    for difficulty in (5, 6, 7):
        spec = generate(0, difficulty)
        assert spec is not None
        mechanics = spec["solution_mechanics"]
        effects = mechanics["explosive_device_effects"]
        required = names.EXPLOSIVE_MIXED if difficulty == 5 else names.EXPLOSIVE_LARGE
        assert any(
            effect["prototype"] == required
            and effect["detonation_causal"]
            and effect["detonation_changes_boxes"]
            and effect["arranged_before_detonation"]
            and effect["counterfactual_retains_body"] is True
            and effect["counterfactual_uses_native_recharge_reset"] is True
            for effect in effects
        )
    tier5 = generate(0, 5)
    assert tier5 is not None
    mixed_index = next(
        index for index, item in enumerate(tier5["explosives"])
        if item["prototype"] == names.EXPLOSIVE_MIXED
    )
    assert _trajectory(tier5) != _trajectory_with_suppressed_detonation(
        tier5, mixed_index
    )
    substituted = copy.deepcopy(tier5)
    substituted["explosives"][mixed_index]["prototype"] = names.EXPLOSIVE_LARGE
    assert _trajectory(tier5) != _trajectory(substituted)

    tier6 = generate(0, 6)
    assert tier6 is not None
    assert any(
        _trajectory(tier6) != _trajectory_with_suppressed_detonation(tier6, index)
        for index in range(len(tier6["explosives"]))
    )


def test_affected_explosive_tiers_have_native_effects_across_bounded_three_split_cohort():
    for split in ("train", "validation", "test"):
        for seed in range(3):
            for difficulty in (5, 6, 7):
                spec = generate(seed, difficulty, split=split)
                assert spec is not None, (split, seed, difficulty)
                effects = spec["solution_mechanics"]["explosive_device_effects"]
                required = (
                    names.EXPLOSIVE_MIXED
                    if difficulty == 5
                    else names.EXPLOSIVE_LARGE
                )
                assert any(
                    effect["prototype"] == required
                    and effect["arranged_before_detonation"]
                    and effect["detonation_causal"]
                    and effect["detonation_changes_boxes"]
                    for effect in effects
                ), (split, seed, difficulty)
                if difficulty >= 6:
                    assert any(
                        effect["detonation_changes_players"] for effect in effects
                    ), (split, seed, difficulty)


def test_failed_seed_and_game_child_diagnostics_are_retained():
    single = generate_module.generate_with_diagnostics(
        0, 2, attempts=1, split="validation"
    )
    if single.spec is not None:
        single = generate_module.generate_with_diagnostics(
            1, 2, attempts=1, split="validation"
        )
    assert single.spec is None
    assert single.failure is not None
    assert single.failure["attempt_limit"] == 1
    assert single.failure["reasons"]
    assert single.failure["geometry_version"] == generate_module.GEOMETRY_VERSION

    game = generate_module.generate_game_with_diagnostics(
        0, split="validation", attempts=1
    )
    assert game.specs is None
    assert game.failures
    assert all(failure["child_seed"] >= 0 for failure in game.failures)
