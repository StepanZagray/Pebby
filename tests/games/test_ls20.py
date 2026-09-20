"""LS20's shared adapter preserves native proofs and live level semantics."""

import json
import importlib
import time

from arcengine import GameState

from pebby.games.ls20.bank import main as bank_main
from pebby.games.ls20.env import Env, official_levels, replay
from pebby.games.ls20.generate import (
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    effective_seed,
    generate,
    replays_to_completion,
    validate_full_standard,
)
from pebby.games.ls20.layout import Layout, extract
from pebby.games.ls20.plan import search, solve
from pebby.multigame import MultiGameEnv, collect_generated_game, preflight


def test_official_first_level_is_solved_from_live_state_and_clone_is_independent():
    env = Env()
    assert len(official_levels()) == 7
    frame = env.reset()
    assert len(frame) == 64 and all(len(row) == 64 for row in frame)
    assert all(isinstance(pixel, int) for row in frame for pixel in row)

    result = search(env, limit=512, node_limit=600_000)
    assert result.actions and not result.truncated and result.reason == "solved"
    assert all(len(action) == 3 and action[0] in env.available_actions for action in result.actions)

    clone = env.clone()
    clone.perform(*result.actions[0])
    remaining = search(clone, limit=512, node_limit=600_000)
    assert remaining.actions and len(remaining.actions) == len(result.actions) - 1
    replay(clone, remaining.actions)
    assert clone.levels_completed == 1 and clone.level_index == 1
    assert env.levels_completed == 0 and env.level_index == 0


def test_three_difficulty_one_specs_are_json_triples_and_replay_within_sixty_seconds():
    started = time.monotonic()
    for seed in range(3):
        spec = generate(seed, 1)
        assert spec is not None
        restored = json.loads(json.dumps(spec))
        assert restored["requested_seed"] == seed
        assert restored["effective_seed"] == seed
        assert restored["effective_split"] == "train"
        assert restored["solution_length"] == len(restored["solution"])
        assert all(len(action) == 3 and action[1:] == [None, None]
                   for action in restored["solution"])
        assert replays_to_completion(restored)
    assert time.monotonic() - started < 60


def test_shared_63_bit_seed_mapping_stays_in_native_training_range():
    original = (1 << 62) + 1_234_567
    mapped, split = effective_seed(original)
    slot = original % 6_000_000
    assert mapped == (slot if slot < 1_000_000 else slot + 2_000_000)
    assert (0 <= mapped < 1_000_000 or 3_000_000 <= mapped < 8_000_000)
    assert split == "train"

    validation, validation_split = effective_seed(original, "validation")
    test, test_split = effective_seed(original, "test")
    assert 1_000_000 <= validation < 2_000_000 and validation_split == "validation"
    assert 2_000_000 <= test < 3_000_000 and test_split == "test"


def test_full_contract_is_accepted_and_declares_exact_official_curriculum():
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert [row["difficulty"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(1, 8))
    assert [row["context_index"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(7))
    modules = preflight(["ls20"], require_full_standard=True)[0]
    assert modules.full_standard is not None and modules.full_standard.ready


def test_whole_game_mode_defaults_to_all_seven_tiers_and_reduced_build_is_rejected(monkeypatch):
    generator = importlib.import_module("pebby.games.ls20.generate")
    calls = []

    def fake(seed, difficulty, attempts, node_limit, *, split):
        calls.append((seed, difficulty, split, attempts, node_limit))
        return {
            "requested_seed": seed,
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "split": split,
        }

    monkeypatch.setattr(generator, "generate", fake)
    specs = generator.generate_game(91, split="validation", attempts=3, node_limit=4)
    assert len(specs) == 7
    assert [spec["difficulty"] for spec in specs] == list(range(1, 8))
    assert [spec["game_level_index"] for spec in specs] == list(range(7))
    assert len({call[0] for call in calls}) == 7
    assert all(call[2:] == ("validation", 3, 4) for call in calls)
    try:
        build_game(specs[:1])
    except ValueError as exc:
        assert "exactly 7" in str(exc)
    else:
        raise AssertionError("reduced specs shifted into a native whole-game context")


def test_full_validator_recomputes_route_and_identities_instead_of_trusting_metadata():
    spec = generate(0, 1, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    assert validate_full_standard(spec, entry) == []

    changed_route = json.loads(json.dumps(spec))
    changed_route["solution"][0][0] = 4 if changed_route["solution"][0][0] != 4 else 3
    assert any("exactly mirror" in error for error in validate_full_standard(changed_route, entry))

    changed_geometry = json.loads(json.dumps(spec))
    changed_geometry["walls"] = changed_geometry["walls"][1:]
    errors = validate_full_standard(changed_geometry, entry)
    assert any("recomputation" in error or "profile" in error for error in errors)

    bool_difficulty = json.loads(json.dumps(spec))
    bool_difficulty["difficulty"] = True
    assert validate_full_standard(bool_difficulty, entry)
    malformed = {"difficulty": 1, "walls": None, "goals": [{"triple": None}]}
    errors = validate_full_standard(malformed, entry)
    assert errors and all(isinstance(error, str) for error in errors)


def test_validation_and_test_specs_use_native_split_namespaces_and_validate():
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    for split, low, high in (
        ("validation", 1_000_000, 2_000_000),
        ("test", 2_000_000, 3_000_000),
    ):
        spec = generate(0, 1, split=split)
        assert spec is not None
        assert low <= spec["effective_seed"] < high
        assert spec["split"] == spec["effective_split"] == split
        assert validate_full_standard(spec, entry) == []


def test_difficulty_two_and_three_stored_proofs_replay_at_their_real_indices():
    for seed, difficulty in ((6, 2), (3, 3)):
        spec = generate(seed, difficulty)
        assert spec is not None
        assert spec["verification_level_index"] == difficulty - 1
        assert replays_to_completion(json.loads(json.dumps(spec)))


def test_three_generated_levels_are_freshly_solved_sequentially_at_real_indices():
    specs = [generate(seed, difficulty) for seed, difficulty in ((0, 1), (6, 2), (3, 3))]
    assert all(spec is not None for spec in specs)
    env = Env([build_level(spec) for spec in specs])

    for expected_index in range(3):
        assert env.level_index == expected_index
        result = search(env, limit=512, node_limit=2_000_000)
        assert result.actions is not None and not result.truncated, result
        assert all(action[0] in env.available_actions for action in result.actions)
        replay(env, result.actions)
        assert env.levels_completed == expected_index + 1

    assert env.state == GameState.WIN


def test_search_bound_and_invalid_actions_are_explicit():
    env = Env()
    result = search(env, node_limit=1)
    assert result.actions is None and result.truncated
    assert solve(env, node_limit=1) is None and solve.truncated
    for action in ((5, None, None), (1, 1, None), (1, None, 1)):
        try:
            env.perform(*action)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action {action}")

    class InexactLayout(Layout):
        @property
        def exact(self):
            return False

    unsupported = search(InexactLayout(**extract(env).__dict__))
    assert unsupported.actions is None
    assert unsupported.unsupported and not unsupported.truncated


def test_bank_cli_preflight_and_shared_collector(tmp_path):
    output = tmp_path / "ls20.jsonl"
    assert bank_main([
        "--levels", "1", "--seed", "0", "--difficulty", "1", "--out", str(output)
    ]) == 0
    stored = json.loads(output.read_text().strip())
    assert replays_to_completion(stored)

    modules, = preflight(["ls20"])
    game = MultiGameEnv.from_specs(modules, [stored])
    assert game.progress.level_index == 0 and game.legal_action_ids == (1, 2, 3, 4)
    collected = collect_generated_game(
        modules,
        master_seed=0,
        game_index=0,
        difficulties=[1],
        outer_generation_attempts=8,
        generator_attempts=50,
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_completed"] == 1


def test_official_layouts_are_hashed_once_and_are_seven_distinct_d4_geometries():
    generator = importlib.import_module("pebby.games.ls20.generate")
    hashes = generator._official_geometry_hashes()
    assert len(hashes) == 7
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes)
    assert generator._official_geometry_hashes() is hashes
    env = Env()
    recomputed = set()
    for index in range(len(official_levels())):
        env.set_level(index)
        recomputed.add(generator._geometry_d4_hash({"walls": sorted(extract(env).walls)}))
    assert recomputed == hashes


def _official_walls(index):
    env = Env()
    env.set_level(index)
    return [list(cell) for cell in sorted(extract(env).walls)]


def test_spec_built_from_an_official_layout_is_rejected_as_official_copy():
    generator = importlib.import_module("pebby.games.ls20.generate")
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    spec = generate(0, 1, split="train")
    assert spec is not None
    for index in range(7):
        copied = json.loads(json.dumps(spec))
        copied["walls"] = _official_walls(index)
        errors = validate_full_standard(copied, entry)
        assert any(
            error.startswith("official_copy:") and "shipped LS20 level" in error
            for error in errors
        ), (index, errors)

    # Acceptance rejects a native draft whose geometry copies a shipped level in
    # every split, before any adapter metadata is attached.
    def native_copy(seed, difficulty, attempts, search_limit, *, split):
        return {"walls": _official_walls(3), "solution": [1], "goals": []}

    generator_calls = []
    original = generator._generate_v4

    def recording(*args, **kwargs):
        generator_calls.append(kwargs["split"])
        return native_copy(*args, **kwargs)

    generator._generate_v4 = recording
    try:
        for split in ("train", "validation", "test"):
            assert generator.generate(5, 4, split=split) is None
    finally:
        generator._generate_v4 = original
    assert generator_calls == ["train", "validation", "test"]


def test_generated_tier_one_spec_records_the_official_copy_check():
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    spec = generate(1, 1, split="train")
    assert spec is not None
    assert spec["official_copy"] is False
    assert spec["geometry_d4_sha256"] not in importlib.import_module(
        "pebby.games.ls20.generate"
    )._official_geometry_hashes()
    assert validate_full_standard(spec, entry) == []

    unrecorded = json.loads(json.dumps(spec))
    del unrecorded["official_copy"]
    assert any("official_copy" in error for error in validate_full_standard(unrecorded, entry))
    assert FULL_STANDARD_CONTRACT["evidence"]["official_copy_exclusion"]
