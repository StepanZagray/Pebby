"""DC22 full-mechanics generation, native teacher, and whole-game tests."""

from copy import deepcopy
import importlib
import json

import pytest
from arcengine import GameState, Level, Sprite

from pebby.games.dc22 import names
from pebby.games.dc22.bank import main as bank_main
from pebby.games.dc22.env import Env, official_levels, replay
from pebby.games.dc22.env import upstream
from pebby.games.dc22.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    generate,
    generate_game,
    gameplay_hash,
    validate_full_standard,
)
from pebby.games.dc22.layout import extract
from pebby.games.dc22.plan import _button_clicks, _state_key, search, solve
from pebby.games.dc22.reference_profiles import PROFILES
from pebby.multigame import MultiGameEnv, SearchLimits, collect_generated_game, preflight


def _assert_actions(env, actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        if action_id == names.ACTION_CLICK:
            assert isinstance(x, int) and isinstance(y, int)
            assert 0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE
        else:
            assert x is None and y is None


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_official_tier_has_a_native_replayed_teacher_witness(difficulty):
    """The compact teacher is advisory; the unmodified contextual engine is final."""
    profile = PROFILES[difficulty]
    env = Env()
    env.set_level(difficulty - 1)
    result = search(env, limit=200, node_limit=profile["search_limit"])
    assert result.status == "solved" and not result.truncated, result
    assert result.backend == "symbolic-plus-native-replay"
    assert result.optimal is False
    assert len(result.actions) == profile["reference_actions"]
    _assert_actions(env, result.actions)
    assert replay(env, result.actions)


def test_native_click_budget_reset_clone_and_input_validation():
    env = Env([official_levels()[0]])
    assert extract(env).exact
    assert set(_button_clicks(env)) == {(45, 17), (45, 34)}
    start = _state_key(env)

    clone = env.clone()
    clone.perform(names.ACTION_CLICK, 45, 34)
    assert clone.steps_left == start[-1] - 2
    assert _state_key(env) == start
    assert getattr(clone.game, names.ATTR_UNDO) is not None

    clone.perform(names.ACTION_UP)
    observation = clone.perform(names.ACTION_RESET)
    assert observation.state == GameState.NOT_FINISHED
    assert (clone.player.x, clone.player.y) == (10, 30)
    assert clone.steps_left == 128

    for action in ((5, None, None), (6, None, None), (6, -1, 0), (6, 64, 0), (1, 1, None), (7, None, None)):
        with pytest.raises(ValueError):
            clone.perform(*action)
    with pytest.raises(ValueError):
        clone.set_level(1)


def test_single_level_generation_json_roundtrip_and_fail_closed_validation():
    for difficulty in DIFFICULTIES:
        spec = generate(0, difficulty, split="train")
        assert spec is not None
        stored = json.loads(json.dumps(spec))
        entry = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        assert validate_full_standard(stored, entry) == []
        assert stored["reference_action_status"] == (
            "measured positive native witness; optimality not established"
        )
        assert stored["optimality"] == "not_claimed_constructive_witness"
        assert stored["fall_recovery"]["penalty"] == 20
        assert stored["fall_recovery"]["recovered"] is True

        contextual = Env([build_level(stored) for _ in range(difficulty)])
        contextual.set_level(difficulty - 1)
        _assert_actions(contextual, stored["solution"])
        assert replay(contextual, stored["solution"])

    spec = generate(3, 6, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][5]
    mutations = []

    route = deepcopy(spec)
    route["solution"][0][0] = (
        names.ACTION_LEFT
        if route["solution"][0][0] != names.ACTION_LEFT
        else names.ACTION_RIGHT
    )
    mutations.append(route)

    geometry = deepcopy(spec)
    geometry["floors"][0][0] += 2
    mutations.append(geometry)

    mechanics = deepcopy(spec)
    mechanics["solution_mechanics"]["bridge_carry"] = 0
    mutations.append(mechanics)

    proof = deepcopy(spec)
    proof["proof"]["engine_win"] = False
    mutations.append(proof)

    blueprint = deepcopy(spec)
    blueprint["stage_blueprint"]["moving_surface_steps"] += 1
    mutations.append(blueprint)

    for changed in mutations:
        assert validate_full_standard(changed, entry)

    alternate_certificate = deepcopy(spec)
    alternate_certificate["constructed_solution"] = list(reversed(spec["constructed_solution"]))
    assert gameplay_hash(alternate_certificate) == gameplay_hash(spec)


def test_generate_game_is_exactly_six_increasing_contexts_and_replays_sequentially():
    with pytest.raises(ValueError):
        generate_game(11, split="validation", difficulties=(1, 2, 3))
    specs = generate_game(11, split="validation")
    assert specs is not None and len(specs) == len(DIFFICULTIES) == 6
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["context_index"] for spec in specs] == list(range(6))
    assert {spec["split"] for spec in specs} == {"validation"}

    env = Env(build_game(json.loads(json.dumps(specs))))
    for index, spec in enumerate(specs):
        assert env.level_index == index
        before = env.levels_completed
        assert replay(env, spec["solution"])
        assert env.levels_completed == before + 1
    assert env.state == GameState.WIN
    assert env.levels_completed == 6


def test_plan_search_replays_live_generated_prefix_and_fall_undo():
    specs = generate_game(19, split="train")
    assert specs is not None
    env = Env(build_game(specs))

    for action in specs[0]["solution"][:4]:
        env.perform(*action)
    result = search(env, limit=128, node_limit=100)
    assert result.status == "solved"
    assert result.backend == "constructive-native-replay"
    assert replay(env, result.actions)
    assert env.level_index == 1

    env.set_level(5)
    for action in specs[5]["solution"][:5]:
        env.perform(*action)
    result = search(env, limit=192, node_limit=100)
    assert result.status == "solved"
    assert replay(env, result.actions)

    fall = Env(build_game(specs))
    actions = specs[0]["fall_probe"]
    for action in actions[:-1]:
        fall.perform(*action)
    before = (fall.player.x, fall.player.y, fall.steps_left)
    observation = fall.perform(*actions[-1])
    assert observation.state != GameState.GAME_OVER
    assert (fall.player.x, fall.player.y) == before[:2]
    assert before[2] - fall.steps_left == 20


def test_explicit_bounds_and_unsupported_states_are_not_impossibility_claims():
    env = Env([official_levels()[0]])
    capped = search(env, node_limit=1)
    assert capped.status == "truncated" and capped.truncated
    assert capped.actions is None and not capped.unsupported and capped.exact
    assert solve(env, node_limit=1) is None
    assert solve.status == "truncated" and solve.truncated

    generated = generate(0, 1, split="train")
    assert generated is not None
    constructive = search(Env([build_level(generated)]), node_limit=1)
    assert constructive.status == "solved"
    assert constructive.backend == "constructive-native-replay"
    assert constructive.expanded == constructive.generated == 0
    assert "no search nodes expanded" in constructive.reason

    layout = extract(env)
    synthetic = type(layout)(
        snapshot=layout.snapshot,
        exact=False,
        unsupported=("synthetic",),
        level_index=layout.level_index,
        steps_left=layout.steps_left,
        player=layout.player,
        goal=layout.goal,
    )
    unsupported = search(synthetic)
    assert unsupported.status == "unsupported" and unsupported.unsupported
    assert unsupported.actions is None and not unsupported.truncated


def test_bank_cli_and_relevant_shared_collector_integration(tmp_path):
    output = tmp_path / "dc22.jsonl"
    assert bank_main([
        "--levels", "1", "--seed", "4", "--difficulty", "1",
        "--out", str(output), "--node-limit", "10000",
    ]) == 0
    stored = json.loads(output.read_text().strip())
    assert replay(Env([build_level(stored)]), stored["solution"])

    modules, = preflight(["dc22"])
    game = MultiGameEnv.from_specs(modules, [stored])
    assert game.progress.level_index == 0
    assert game.legal_action_ids == names.ACTION_IDS
    collected = collect_generated_game(
        modules,
        master_seed=23,
        game_index=0,
        difficulties=[1],
        limits=SearchLimits(max_actions_per_level=128, max_search_work=10_000),
        outer_generation_attempts=2,
        generator_attempts=8,
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_completed"] == 1


def test_contract_is_ready_after_root_closure():
    contract = FULL_STANDARD_CONTRACT
    assert contract["format"] == "pebby-full-generator-contract-v1"
    assert contract["status"] == "ready"
    assert contract["source_id"] == "dc22-fdcac232"
    assert [row["difficulty"] for row in contract["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in contract["curriculum"]] == list(range(6))
    closure = contract["evidence"]["version_6_closure_review"]
    assert all(
        isinstance(name, str) and name and isinstance(value, str) and value.strip()
        for name, value in contract["evidence"].items()
    )
    assert "40/40 independent device deletions" in closure
    assert "10/10 late-phase omissions" in closure
    assert "300/293/300 actions" in contract["evidence"]["root_integration_review"]
    assert any("historical version-5" in caveat for caveat in contract["caveats"])
    assert any("persistence is opt-in" in caveat for caveat in contract["caveats"])


def test_native_selective_bridge_colour_routing_requires_two_cycles():
    """Public actions must select the heterogeneous colour destination."""
    generated = importlib.import_module("pebby.games.dc22.generate")
    sprites = upstream().sprites
    pieces = [
        Sprite(pixels=[[2, 2], [2, 2]], name=f"probe-floor-{x}",
               visible=True, collidable=False, layer=-2).set_position(x, 4)
        for x in (4, 6, 26, 28)
    ]
    for prototype, x, y in (
        ("plflho1", 4, 4), ("goknoi", 28, 4),
        ("tewfutpibpar1", 4, 4), ("tewfutpibpar2", 12, 4),
        ("tewfutyefmyf2", 24, 4), ("piyqze-buezna-pueite-1", 6, 4),
        ("buezna-matkhq", 46, 5), ("renrjo-buezna", 46, 14),
    ):
        sprite = sprites[prototype].clone().set_position(x, y)
        if prototype == "tewfutpibpar1":
            sprite.tags.extend([names.TAG_BRIDGE_COLOR_CYCLE, "d"])
        pieces.append(sprite)
    level = Level(sprites=pieces, grid_size=(64, 64),
                  data={"StepCounter": 1024}, name="native-color-causal-probe")
    colour = generated._click_action("renrjo-buezna", (46, 14), (64, 64))
    teleport = generated._click_action("buezna-matkhq", (46, 5), (64, 64))
    outcomes = []
    for cycles in range(4):
        env = generated._context_env(level, 5)
        route = [[4, None, None], [3, None, None]] + [colour] * cycles
        route += [teleport, [4, None, None], [4, None, None]]
        for action in route:
            env.perform(*action)
        outcomes.append((env.state.name, env.levels_completed, env.player.x, env.player.y))
    assert [value[0] for value in outcomes] == [
        "NOT_FINISHED", "NOT_FINISHED", "WIN", "NOT_FINISHED",
    ]
    assert outcomes[2][1:] == (1, 28, 4)


def test_validation_rejects_trailing_actions_and_unbound_native_facts():
    generated = importlib.import_module("pebby.games.dc22.generate")
    spec = generate(0, 1, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]

    trailing = deepcopy(spec)
    for key in ("solution", "context_solution", "constructed_solution"):
        trailing[key].append([names.ACTION_UP, None, None])
    trailing["solution_length"] = len(trailing["solution"])
    trailing["mechanic_stage_sha256"], trailing["stage_action_sha256"] = (
        generated.stage_identity_hashes(trailing)
    )
    trailing["proof"]["mechanic_stage_sha256"] = trailing["mechanic_stage_sha256"]
    trailing["proof"]["stage_action_sha256"] = trailing["stage_action_sha256"]
    trailing["solution_mechanics"] = generated._mechanic_certificate(
        generated.build_level(trailing), 0, trailing["solution"]
    )
    assert any("first completion" in error for error in validate_full_standard(trailing, entry))

    mutations = []
    for key, value in (
        ("minimum_steps_remaining", 999999), ("final_steps_remaining", -555),
        ("levels_completed", 37), ("reachable_states", -10),
        ("search_limit", -10), ("seed", -4), ("requested_seed", -400),
        ("generation_attempt", -8), ("engine_verified", 1),
        ("search_truncated", 0),
    ):
        changed = deepcopy(spec)
        changed[key] = value
        if key in changed["proof"]:
            changed["proof"][key] = value
        mutations.append(changed)
    changed = deepcopy(spec)
    changed["generation_exclusions"] = {"invented": -40}
    mutations.append(changed)
    for key in ("seed", "minimum_steps_remaining", "final_steps_remaining"):
        changed = deepcopy(spec)
        changed.pop(key, None)
        changed["proof"].pop(key, None)
        mutations.append(changed)
    for changed in mutations:
        assert validate_full_standard(changed, entry)


def test_generation_failure_diagnostics_survive_single_game_and_bank(tmp_path):
    diagnostic = {}
    # Seed 1's difficulty-1 draft lands outside the train partition on its
    # only attempt, so a single-attempt request is deterministically
    # rejected on geometry_split. (Seed 0 used to land outside train too,
    # but no longer does with the current draft/geometry hashing.)
    assert generate(1, 1, attempts=1, split="train", diagnostics=diagnostic) is None
    assert diagnostic == {
        "status": "rejected", "requested_seed": 1, "difficulty": 1,
        "attempts": 1, "generator_version": 6,
        "reasons": {"geometry_split": 1},
    }

    game_diagnostic = {}
    assert generate_game(0, split="train", attempts=1, diagnostics=game_diagnostic) is None
    assert game_diagnostic["status"] == "rejected"
    assert game_diagnostic["failed_level_index"] >= 0
    assert game_diagnostic["child"]["reasons"]

    output = tmp_path / "dc22.jsonl"
    failures = tmp_path / "dc22-failures.jsonl"
    with pytest.raises(SystemExit):
        bank_main([
            "--levels", "1", "--seed", "1", "--difficulty", "1",
            "--out", str(output), "--max-seeds", "1", "--attempts", "1",
            "--diagnostics-out", str(failures),
        ])
    rows = [json.loads(line) for line in failures.read_text().splitlines()]
    assert rows == [diagnostic]


def test_malformed_nested_rejection_facts_return_errors_without_raising():
    spec = generate(0, 1, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    malformed = (
        {"geometry_split": "bad"},
        {"geometry_split": True},
        {"geometry_split": -1},
        {"invented": 0},
        [],
        "bad",
        None,
    )
    for value in malformed:
        changed = deepcopy(spec)
        changed["generation_exclusions"] = value
        changed["proof"]["generation_exclusions"] = value
        errors = validate_full_standard(changed, entry)
        assert errors
        assert any("rejection" in error for error in errors)
