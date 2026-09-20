"""LF52 full-mechanics planning, generation, and native replay tests."""

import copy
import json
from pathlib import Path
import tempfile

import pytest
from arcengine import GameState

from pebby.games.lf52 import names
from pebby.games.lf52 import bank as bank_module
from pebby.games.lf52 import generate as generate_module
from pebby.games.lf52.bank import build as build_bank, load as load_bank, save as save_bank
from pebby.games.lf52.env import Env, official_levels, replay
from pebby.games.lf52.generate import (
    DIFFICULTIES,
    FORMAT,
    FULL_STANDARD_CONTRACT,
    GENERATOR_VERSION,
    REFERENCE_PROFILES,
    REQUIRED_MECHANICS,
    build_game,
    build_level,
    effective_seed,
    generate,
    generate_game,
    geometry_identities,
    geometry_partition,
    validate_full_standard,
)
from pebby.games.lf52.layout import extract
from pebby.games.lf52.plan import search, solve


def replay_spec(spec):
    levels = official_levels()
    levels[spec["difficulty"] - 1] = build_level(spec)
    env = Env(levels)
    env.reset()
    env.set_level(spec["difficulty"] - 1)
    before = env.levels_completed
    completed, observation = replay(env, spec["solution"])
    assert completed and observation is not None
    assert env.levels_completed == before + 1
    return env


def native_mover_move_count(spec):
    levels = official_levels()
    levels[spec["difficulty"] - 1] = build_level(spec)
    env = Env(levels)
    env.reset()
    env.set_level(spec["difficulty"] - 1)
    positions = {
        id(entity): tuple(map(int, getattr(entity, names.PROP_GRID_POSITION)))
        for entity in getattr(env.grid, names.METHOD_ENTITIES_NAMED)(names.MOVING_HOLE)
    }
    moves = 0
    for action in spec["solution"]:
        env.perform(*action)
        if env.levels_completed:
            break
        for entity in getattr(env.grid, names.METHOD_ENTITIES_NAMED)(names.MOVING_HOLE):
            key = id(entity)
            position = tuple(map(int, getattr(entity, names.PROP_GRID_POSITION)))
            moves += position != positions[key]
            positions[key] = position
    return moves


def counterfactual_replay(spec, *, remove_moving=None, remove_peg=None):
    changed = copy.deepcopy(spec)
    if remove_moving is not None:
        changed["board"]["moving"] = [
            cell for cell in changed["board"]["moving"] if cell != list(remove_moving)
        ]
    if remove_peg is not None:
        changed["board"]["pegs"] = [
            peg for peg in changed["board"]["pegs"] if peg["cell"] != list(remove_peg)
        ]
    levels = official_levels()
    levels[spec["difficulty"] - 1] = build_level(changed)
    env = Env(levels)
    env.reset()
    env.set_level(spec["difficulty"] - 1)
    return replay(env, spec["solution"])[0]


def test_official_inventory_profiles_and_complete_source_semantics():
    assert len(official_levels()) == 10
    assert DIFFICULTIES == tuple(range(1, 11))
    assert set(REFERENCE_PROFILES) == set(DIFFICULTIES)
    env = Env()
    env.reset()
    for index in range(10):
        env.set_level(index)
        layout = extract(env)
        assert layout.exact, layout.unsupported
        profile = REFERENCE_PROFILES[index + 1]
        occupied = layout.cells | layout.rails
        assert max(x for x, _ in occupied) - min(x for x, _ in occupied) + 1 == profile["width"]
        assert len(layout.moving_cells) == profile["moving"]
        assert len(layout.rails) == profile["rails"]
        assert len(layout.obstacles) == profile["obstacles"]
        assert len(layout.peg_entities) == profile["pegs"]


@pytest.mark.parametrize("level_index", range(10))
def test_bounded_official_teacher_positive_witnesses(level_index):
    env = Env()
    env.reset()
    env.set_level(level_index)
    result = search(env, node_limit=250_000)
    assert result.actions is not None, result.reason
    assert not result.truncated and not result.unsupported and result.exact
    before = env.levels_completed
    assert replay(env, result.actions)[0]
    assert env.levels_completed == before + 1


@pytest.mark.parametrize("level_index", range(10))
def test_tiny_official_teacher_cutoffs_are_unknown_not_unsolvable(level_index):
    env = Env()
    env.reset()
    env.set_level(level_index)
    result = search(env, node_limit=1)
    assert result.actions is None and result.truncated
    assert "limit" in result.reason


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_generated_tier_replays_and_recomputes_full_validation(difficulty):
    spec = generate(10_000 + difficulty, difficulty, attempts=64, split="train")
    assert spec is not None, generate.last_diagnostics
    assert spec["format"] == FORMAT
    assert spec["difficulty"] == difficulty
    assert spec["context_index"] == difficulty - 1
    assert spec["geometry_split"] == "train"
    assert set(REQUIRED_MECHANICS[difficulty]) <= {
        name for name, count in spec["solution_mechanics"].items() if count
    }
    assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]) == []
    replay_spec(json.loads(json.dumps(spec)))


def test_complete_game_mode_is_exactly_ten_increasing_native_contexts():
    specs = generate_game(73, split="train")
    assert specs is not None
    assert generate_game.last_diagnostics["accepted"] is True
    assert generate_game.last_diagnostics["completed"] == 10
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["game_level_index"] for spec in specs] == list(range(10))
    assert all(spec["game_mode"] == "full_game" for spec in specs)
    assert all(spec["game_difficulties"] == list(DIFFICULTIES) for spec in specs)
    assert len({spec["effective_seed"] for spec in specs}) == 10
    env = Env(build_game(specs))
    env.reset()
    for index, spec in enumerate(specs):
        assert env.level_index == index
        assert replay(env, spec["solution"])[0]
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN
    with pytest.raises(ValueError, match="exactly 10"):
        build_game(specs[:3])

    reduced = generate_game(73, difficulties=(1, 3, 6), split="train")
    assert reduced is not None
    assert generate_game.last_diagnostics["mode"] == "reduced_smoke"
    assert all(spec["game_mode"] == "reduced_smoke" for spec in reduced)
    assert all(spec["game_difficulties"] == [1, 3, 6] for spec in reduced)
    with pytest.raises(ValueError, match="exactly 10"):
        build_game(reduced)

    # The collector also assembles complete games from independently generated
    # one-level rows.  Parent provenance is optional, but when present it is an
    # all-or-none strict record.
    game_fields = {
        "game_mode", "game_difficulties", "game_seed", "game_level_index",
        "game_requested_difficulty",
    }
    standalone = [
        {key: value for key, value in spec.items() if key not in game_fields}
        for spec in specs
    ]
    assert len(build_game(standalone)) == len(DIFFICULTIES)
    partially_enriched = copy.deepcopy(standalone)
    partially_enriched[0]["game_seed"] = 73
    with pytest.raises(ValueError, match="absent.*complete"):
        build_game(partially_enriched)


def test_collector_entry_search_solves_every_generated_tier_sequentially():
    specs = generate_game(91, split="train")
    assert specs is not None
    env = Env(build_game(specs))
    env.reset()
    for index in range(10):
        result = search(env, node_limit=250_000)
        assert result.actions is not None, (index + 1, result.reason)
        assert replay(env, result.actions)[0]
        assert env.levels_completed == index + 1


def test_split_mapping_partition_and_json_round_trip_are_canonical():
    specs = {}
    requested = {"train": 404, "validation": 406, "test": 406}
    for split in names.SPLITS:
        spec = generate(requested[split], 8, split=split)
        assert spec is not None
        restored = json.loads(json.dumps(spec))
        assert restored == spec
        assert spec["effective_seed"] == effective_seed(requested[split], split)
        exact, d4 = geometry_identities(spec)
        assert spec["geometry_sha256"] == exact
        assert spec["geometry_d4_sha256"] == d4
        assert geometry_partition(d4) == split == spec["geometry_split"]
        specs[split] = spec
    assert len({spec["effective_seed"] for spec in specs.values()}) == 3
    assert len({spec["geometry_d4_sha256"] for spec in specs.values()}) == 3


def test_validator_rejects_route_geometry_partition_and_proof_tampering():
    spec = generate(505, 9, split="validation")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][8]
    assert validate_full_standard(spec, entry) == []

    changed = copy.deepcopy(spec)
    changed["solution"][0][0] = names.ACTION_RIGHT
    assert validate_full_standard(changed, entry)

    changed = copy.deepcopy(spec)
    changed["board"]["ordinary"].pop()
    assert any("geometry" in error or "profile" in error or "route" in error for error in validate_full_standard(changed, entry))

    changed = copy.deepcopy(spec)
    changed["geometry_split"] = "test"
    assert any("partition" in error for error in validate_full_standard(changed, entry))

    changed = copy.deepcopy(spec)
    changed["proof"]["expanded"] = -1
    assert any("proof" in error for error in validate_full_standard(changed, entry))

    changed = copy.deepcopy(spec)
    changed["solution"].append([names.ACTION_RIGHT, None, None])
    changed["context_solution"] = copy.deepcopy(changed["solution"])
    changed["solution_length"] += 1
    changed["proof"]["route_sha256"] = "forged"
    assert any("final action" in error or "route" in error for error in validate_full_standard(changed, entry))

    changed = copy.deepcopy(spec)
    changed["solution"][0] = [names.ACTION_CLICK, True, 3]
    assert any("schema" in error for error in validate_full_standard(changed, entry))

    for field, value in (
        ("generator_version", 4.0),
        ("native_action_budget", True),
        ("budget_slack", 1.0),
        ("search_limit", True),
        ("context_index", False),
    ):
        changed = copy.deepcopy(spec)
        changed[field] = value
        assert validate_full_standard(changed, entry), field

    changed = copy.deepcopy(spec)
    changed["proof"]["expanded"] += 1
    changed["proof_sha256"] = __import__("hashlib").sha256(
        json.dumps(changed["proof"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert any("constructive work" in error for error in validate_full_standard(changed, entry))


def test_generated_conditional_traps_camera_branch_and_active_counts_are_native():
    tier3 = generate(13_003, 3, attempts=64, split="train")
    tier6 = generate(13_006, 6, attempts=64, split="train")
    assert tier3 is not None and tier6 is not None
    assert tier3["solution_mechanics"]["native_conditional_reset_probes"] >= 1
    assert tier3["solution_mechanics"].get("scripted_reset_landings", 0) == 0
    assert tier6["solution_mechanics"]["native_conditional_reset_probes"] >= 1
    assert tier6["solution_mechanics"]["native_landing_camera_probes"] >= 1
    assert tier6["solution_mechanics"]["blocker_jumps"] == REFERENCE_PROFILES[6]["obstacles"]
    assert tier6["solution_mechanics"]["distinct_moving_holes_moved"] == REFERENCE_PROFILES[6]["moving"]
    assert tier6["solution_mechanics"].get("scripted_reset_landings", 0) == 0

    tier7 = generate(13_007, 7, attempts=64, split="train")
    assert tier7 is not None
    assert tier7["solution_mechanics"]["distinct_blockers_traversed"] == REFERENCE_PROFILES[7]["obstacles"]
    assert tier7["solution_mechanics"]["distinct_moving_holes_moved"] == REFERENCE_PROFILES[7]["moving"]
    assert tier7["solution_mechanics"]["moving_hole_order_events"] >= 1
    assert tier7["solution_mechanics"]["moving_hole_order_task_relations"] >= 1
    assert tier7["solution_mechanics"]["ordered_task_actor_jump_relations"] >= 1
    assert tier7["solution_mechanics"]["moving_hole_moves"] == native_mover_move_count(tier7)
    assert not counterfactual_replay(
        tier7, remove_moving=tier7["construction"]["order_leader_start"],
    )
    assert tier7["solution_length"] >= 80

    tier10 = generate(13_010, 10, attempts=64, split="train")
    assert tier10 is not None
    assert tier10["solution_mechanics"]["moving_hole_order_task_relations"] >= 1
    assert tier10["solution_mechanics"]["ordered_task_actor_jump_relations"] >= 1
    assert tier10["solution_mechanics"]["moving_hole_moves"] == native_mover_move_count(tier10)
    assert not counterfactual_replay(
        tier10, remove_peg=tier10["construction"]["order_task_start"],
    )


def test_tier2_split_support_and_tier4_native_branch_blockers_are_executed():
    # This exact child previously exhausted 64 test attempts without producing
    # a test-partition tier-2 geometry from the fixed four-shape core.
    tier2 = generate(6_015_243_502_515_316_627, 2, attempts=64, split="test")
    assert tier2 is not None
    assert tier2["geometry_split"] == "test"
    assert tier2["construction"]["variant"] in range(32)

    tier4 = generate(10_004, 4, attempts=64, split="train")
    assert tier4 is not None
    mechanics = tier4["solution_mechanics"]
    assert mechanics["distinct_blockers_offered"] == REFERENCE_PROFILES[4]["obstacles"]
    assert mechanics["distinct_blockers_traversed"] < mechanics["distinct_blockers_offered"]


def test_every_tier10_topology_has_a_reachable_ordered_pair_program():
    # The former one-cell terminal leg stranded variants 6..11 before blue
    # rearrangement certification could even begin.
    routes = [generate_module._base_recipe(10, variant)[5] for variant in range(12)]
    assert all(routes)
    assert len({len(route) for route in routes}) == 12


def test_live_prefix_selection_undo_and_reset_recovery():
    # A one-level Env is logical context 1; higher-tier prefix behavior is
    # covered by the context-aware generated replay tests above.
    spec = generate(607, 1)
    env = Env([build_level(spec)])
    env.reset()
    pristine = extract(env)
    route = solve(env)
    assert route
    env.perform(*route[0])
    assert extract(env).selected is not None
    continuation = solve(env)
    assert continuation and continuation[0] == tuple(route[1])
    assert replay(env, continuation)[0]

    env = Env([build_level(spec)])
    env.reset()
    env.perform(*route[0])
    env.perform(names.ACTION_UNDO)
    restored = extract(env)
    assert restored.key == pristine.key
    assert restored.history_depth == 0 and restored.action_count == 0
    assert solve(env) is not None


def test_bank_io_diagnostics_and_hard_bounds():
    specs, tried = build_bank(3, seed=700, difficulty=2, split="train")
    assert len(specs) == 3 and tried >= 3
    assert all(spec["generation_diagnostics"]["attempted"] <= 64 for spec in specs)
    assert bank_module.build.last_diagnostics["complete"] is True
    assert bank_module.build.last_diagnostics["tried"] == tried
    with tempfile.TemporaryDirectory() as tmp:
        path = save_bank(specs, Path(tmp) / "lf52.jsonl")
        assert load_bank(path) == specs
    assert generate(800, 1, attempts=1, node_limit=1) is None
    assert generate.last_diagnostics["attempted"] == 1
    for invalid in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            build_bank(1, max_attempts=invalid)


def test_strict_generator_inputs_and_full_builder_metadata():
    for invalid in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            generate(invalid, 1)
        with pytest.raises(ValueError):
            generate(1, invalid)
        with pytest.raises(ValueError):
            generate_game(invalid, split="train")
        with pytest.raises(ValueError):
            generate_game(1, difficulties=(1, invalid), split="train")

    specs = generate_game(81, split="train")
    assert specs is not None
    changed = copy.deepcopy(specs)
    changed[0]["game_seed"] += 1
    with pytest.raises(ValueError, match="child seed|parent seeds"):
        build_game(changed)


def test_generated_geometry_and_action_sequences_have_real_seed_diversity():
    # Later tiers vary rail travel length, not only palette or translation.
    for difficulty in (4, 6, 8, 10):
        specs = [generate(9000 + seed, difficulty) for seed in range(4)]
        assert all(specs)
        assert len({spec["geometry_d4_sha256"] for spec in specs}) >= 3
        assert len({spec["action_sequence_sha256"] for spec in specs}) >= 2


def test_contract_is_ready_after_root_acceptance():
    assert FORMAT == "pebby.lf52.level.v5"
    assert GENERATOR_VERSION == 5
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert [row["difficulty"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(10))
    assert all(
        isinstance(value, str) and value
        for value in FULL_STANDARD_CONTRACT["evidence"].values()
    )
    assert any("shortest-path optimality" in caveat for caveat in FULL_STANDARD_CONTRACT["caveats"])
    assert any("certificate-guided" in caveat for caveat in FULL_STANDARD_CONTRACT["caveats"])
