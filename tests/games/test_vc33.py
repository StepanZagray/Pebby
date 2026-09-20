"""VC33 full-mechanic generation, exact teaching, and native replay tests."""

from copy import deepcopy
import json

from arcengine import GameState
import pytest

from pebby.games.vc33 import names
from pebby.games.vc33.bank import main as bank_main
from pebby.games.vc33.env import Env, official_levels, replay
from pebby.games.vc33.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    REFERENCE_PROFILES,
    SOURCE_ID,
    SPLITS,
    _canonical_split,
    _gameplay_hash,
    _metrics,
    build_game,
    build_level,
    generate,
    generate_game,
    validate_full_standard,
)
from pebby.games.vc33.layout import extract
from pebby.games.vc33.plan import _selection_clicks, search, solve


OFFICIAL_LENGTHS = (3, 7, 23, 21, 44, 20, 49)
OFFICIAL_DENSITIES = (
    0.362305, 0.444336, 0.200074, 0.235352,
    0.426025, 0.338623, 0.443142,
)
OFFICIAL_COUNTS = (
    (1, 1, 2, 0, 0, 1, 2),
    (1, 1, 4, 0, 0, 2, 3),
    (3, 3, 8, 0, 0, 4, 5),
    (1, 1, 6, 2, 0, 4, 5),
    (2, 2, 6, 3, 0, 3, 4),
    (1, 1, 4, 2, 1, 1, 3),
    (3, 3, 8, 3, 2, 2, 5),
)


def _assert_clicks(actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id == names.ACTION_CLICK
        assert isinstance(x, int) and isinstance(y, int)
        assert 0 <= x < 64 and 0 <= y < 64


@pytest.fixture(scope="module")
def full_game_specs():
    specs = generate_game(300, split="validation")
    assert specs is not None
    return specs


def test_contract_matches_all_seven_official_contexts():
    assert len(official_levels()) == 7
    assert DIFFICULTIES == tuple(range(1, 8))
    assert SOURCE_ID == "vc33-5430563c"
    assert FULL_STANDARD_CONTRACT["format"] == "pebby-full-generator-contract-v1"
    assert FULL_STANDARD_CONTRACT["source_id"] == SOURCE_ID
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert tuple(
        (row["difficulty"], row["context_index"])
        for row in FULL_STANDARD_CONTRACT["curriculum"]
    ) == tuple((difficulty, difficulty - 1) for difficulty in DIFFICULTIES)
    assert all(
        type(row["search_work"]) is int and 1 <= row["search_work"] <= 32_000_000
        for row in FULL_STANDARD_CONTRACT["curriculum"]
    )


def test_official_reference_characterization_is_pinned_per_tier():
    for difficulty, level in enumerate(official_levels(), 1):
        metrics = _metrics(level)
        profile = REFERENCE_PROFILES[difficulty]
        counts = tuple(
            metrics[key]
            for key in ("loads", "targets", "buttons", "swaps",
                        "floors", "walls", "supports")
        )
        assert counts == OFFICIAL_COUNTS[difficulty - 1]
        assert metrics["grid_size"] == profile["grid"]
        assert metrics["step_budget"] == profile["budget"]
        assert tuple(metrics["gravity"]) == profile["gravity"]
        assert metrics["visual_density"] == OFFICIAL_DENSITIES[difficulty - 1]
        assert profile["reference_density"] == OFFICIAL_DENSITIES[difficulty - 1]
        assert profile["reference_actions"] == OFFICIAL_LENGTHS[difficulty - 1]


def test_exact_teacher_solves_and_replays_every_official_level_sequentially():
    env = Env()
    measured = []
    for context, expected_length in enumerate(OFFICIAL_LENGTHS):
        assert env.level_index == context
        assert extract(env).exact
        result = search(
            env,
            limit=env.max_steps,
            node_limit=REFERENCE_PROFILES[context + 1]["search_work"],
        )
        assert result.actions is not None, result
        assert result.exact and not result.truncated and not result.unsupported
        assert len(result.actions) == expected_length
        _assert_clicks(result.actions)
        assert replay(env, result.actions)
        measured.append(len(result.actions))
        assert env.levels_completed == context + 1
    assert tuple(measured) == OFFICIAL_LENGTHS
    assert env.state == GameState.WIN


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_each_generated_tier_is_json_stable_validated_and_native_certified(difficulty):
    spec = generate(100 + difficulty, difficulty, split="train")
    assert spec is not None, generate.last_report
    restored = json.loads(json.dumps(spec))
    assert restored == spec
    assert spec["split"] == "train"
    assert spec["difficulty"] == difficulty
    assert spec["training_context_index"] == difficulty - 1
    assert spec["verification_level_index"] == difficulty - 1
    assert spec["geometry_split"] == _canonical_split(spec["geometry_d4_sha256"])
    assert len(spec["geometry_d4_sha256"]) == 64
    assert len(spec["gameplay_sha256"]) == 64
    assert spec["engine_verified"] and spec["search_exact"]
    assert not spec["search_truncated"]
    assert spec["solution_length"] == len(spec["solution"])
    _assert_clicks(spec["solution"])
    assert validate_full_standard(
        restored, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
    ) == ()

    level = build_level(spec)
    native = Env([level])
    for sprite in native.level.get_sprites():
        assert 0 <= sprite.x and sprite.x + sprite.width <= spec["grid_size"]
        assert 0 <= sprite.y and sprite.y + sprite.height <= spec["grid_size"]
    walls = native.level.get_sprites_by_tag(names.TAG_WALL)
    assert walls
    for wall in walls:
        # Native structural walls are solid contiguous bars. This specifically
        # guards against the former arbitrary every-fourth-pixel stippling.
        assert all(pixel == 5 for row in wall.pixels.tolist() for pixel in row)
        assert 0 < native.game.pjfzvvjgud(wall) < spec["grid_size"]
    for target in native.targets():
        assert sum(wall.collides_with(target) for wall in walls) == 1

    trace = spec["proof"]["mechanic_use"]
    assert trace["won"] and trace["changed_actions"] == spec["solution_length"]
    assert trace["distinct_buttons"] > 0
    if difficulty >= 4:
        assert trace["distinct_swaps"] > 0
        assert trace["animation_frames"] > 0
    if difficulty >= 6:
        assert trace["floor_limit_contacts"] == spec["metrics"]["floors"]
    if difficulty == 7:
        assert trace["distinct_swaps"] == 3
        assert trace["coupled_multi_load_transfers"] > 0


def test_explicit_split_is_required_and_all_three_partitions_are_reachable():
    with pytest.raises(ValueError, match="split"):
        generate(0, 1)
    specs = [
        generate(700 + index, 7, split=split)
        for index, split in enumerate(SPLITS)
    ]
    assert all(spec is not None for spec in specs)
    assert {spec["split"] for spec in specs} == set(SPLITS)
    assert len({spec["geometry_d4_sha256"] for spec in specs}) == 3
    assert len({spec["gameplay_sha256"] for spec in specs}) == 3
    assert len({spec["effective_seed"] for spec in specs}) == 3


def test_full_generate_game_builds_and_replays_exact_order(full_game_specs):
    assert len(full_game_specs) == len(DIFFICULTIES)
    assert [spec["difficulty"] for spec in full_game_specs] == list(DIFFICULTIES)
    assert [spec["training_context_index"] for spec in full_game_specs] == list(range(7))
    assert [spec["game_ordinal"] for spec in full_game_specs] == list(range(7))
    assert len({spec["child_seed"] for spec in full_game_specs}) == 7

    levels = build_game(json.loads(json.dumps(full_game_specs)))
    assert len(levels) == 7
    env = Env(levels)
    for index, spec in enumerate(full_game_specs):
        assert env.level_index == index
        assert replay(env, spec["solution"])
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN


def test_reduced_generation_is_ergonomic_but_cannot_shift_build_contexts():
    reduced = generate_game(91, split="test", difficulties=(2, 5, 7))
    assert reduced is not None
    assert [spec["difficulty"] for spec in reduced] == [2, 5, 7]
    with pytest.raises(ValueError, match="exactly"):
        build_game(reduced)
    for invalid in ((5, 2), (2, 2, 5)):
        with pytest.raises(ValueError, match="strictly increasing"):
            generate_game(91, split="test", difficulties=invalid)
    with pytest.raises(ValueError, match="increasing sequence"):
        generate_game(91, split="test", difficulties=2)


def test_build_game_rejects_order_duplicates_and_split_mismatch(full_game_specs):
    with pytest.raises(ValueError, match="sequence"):
        build_game(None)
    with pytest.raises(ValueError, match="exactly"):
        build_game(full_game_specs[:-1])
    with pytest.raises(ValueError, match="object"):
        build_game([None] * len(DIFFICULTIES))

    reordered = deepcopy(full_game_specs)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="ordered"):
        build_game(reordered)

    duplicate = deepcopy(full_game_specs)
    duplicate[1] = deepcopy(duplicate[0])
    duplicate[1]["difficulty"] = 2
    with pytest.raises(ValueError):
        build_game(duplicate)

    mixed = deepcopy(full_game_specs)
    mixed[-1]["split"] = "train"
    with pytest.raises(ValueError, match="one split"):
        build_game(mixed)

    unhashable = deepcopy(full_game_specs)
    unhashable[0]["split"] = []
    with pytest.raises(ValueError, match="valid split"):
        build_game(unhashable)


def test_build_game_independently_rejects_actions_after_first_win(
        full_game_specs, monkeypatch):
    suffixed = deepcopy(full_game_specs)
    suffixed[0]["solution"].append(list(suffixed[0]["solution"][-1]))
    monkeypatch.setattr(
        "pebby.games.vc33.generate.validate_full_standard",
        lambda spec, curriculum: (),
    )
    with pytest.raises(ValueError, match="before its final stored action"):
        build_game(suffixed)


def test_generate_game_preserves_failed_child_diagnostics(monkeypatch):
    def fail_child(seed, difficulty, **kwargs):
        return None

    fail_child.last_report = {
        "accepted": False,
        "attempts_used": 1,
        "rejections": {"forced_test_failure": 1},
    }
    monkeypatch.setattr("pebby.games.vc33.generate.generate", fail_child)
    assert generate_game(42, split="train", difficulties=(2, 5)) is None
    report = generate_game.last_report
    assert report["failure_stage"] == "child_generation"
    assert report["failed_ordinal"] == 0
    assert report["failed_difficulty"] == 2
    assert report["child_report"] == fail_child.last_report


def test_validator_rejects_route_geometry_identity_and_proof_tampering():
    spec = generate(818, 7, split="test")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][6]

    cases = []
    route = deepcopy(spec)
    route["solution"][0] = [names.ACTION_CLICK, 0, 0]
    cases.append(route)
    geometry = deepcopy(spec)
    geometry["plan"]["initial_edges"][0] += 2
    cases.append(geometry)
    identity = deepcopy(spec)
    identity["geometry_d4_sha256"] = "0" * 64
    cases.append(identity)
    proof = deepcopy(spec)
    proof["proof"]["mechanic_use"]["distinct_swaps"] = 0
    cases.append(proof)
    proof_mirror = deepcopy(spec)
    proof_mirror["proof"]["engine_win"] = False
    cases.append(proof_mirror)
    proof_work = deepcopy(spec)
    proof_work["proof"]["search_expanded"] += 1
    cases.append(proof_work)
    nested_identity = deepcopy(spec)
    nested_identity["identities"]["partition"] = "train"
    cases.append(nested_identity)
    rejections = deepcopy(spec)
    rejections["generation"]["rejections"]["fabricated"] = 1
    cases.append(rejections)
    generator_version = deepcopy(spec)
    generator_version["generator_version"] = float(spec["generator_version"])
    cases.append(generator_version)
    context_alias = deepcopy(spec)
    context_alias["context_index"] = float(spec["context_index"])
    cases.append(context_alias)
    proof_alias = deepcopy(spec)
    proof_alias["proof"]["difficulty"] = float(spec["difficulty"])
    cases.append(proof_alias)
    nested_alias = deepcopy(spec)
    nested_alias["proof"]["context_replay"]["verification_level_index"] = False
    cases.append(nested_alias)
    malformed = deepcopy(spec)
    malformed["targets"][0] = []
    cases.append(malformed)
    partition = deepcopy(spec)
    partition["split"] = "validation"
    cases.append(partition)

    for changed in cases:
        assert validate_full_standard(changed, entry)

    aliased_curriculum = deepcopy(entry)
    aliased_curriculum["search_work"] = float(entry["search_work"])
    assert validate_full_standard(spec, aliased_curriculum)


def test_gameplay_identity_excludes_nonexecuted_construction_metadata():
    spec = generate(821, 5, split="validation")
    assert spec is not None
    assert all("load_index" not in target for target in spec["targets"])
    private = deepcopy(spec)
    private["targets"][0]["load_index"] = 999
    assert _gameplay_hash(private) == spec["gameplay_sha256"]
    executed = deepcopy(spec)
    executed["targets"][0]["primary"] += 1
    assert _gameplay_hash(executed) != spec["gameplay_sha256"]


def test_live_prefix_teacher_replans_after_a_wrong_reversible_transfer():
    spec = generate(919, 7, split="validation")
    assert spec is not None
    env = Env([build_level(spec)])
    planned = tuple(spec["solution"][0])
    alternatives = [
        (names.ACTION_CLICK, x, y)
        for effect, (x, y) in _selection_clicks(env)
        if effect >= 0 and (x, y) != planned[1:]
    ]
    assert alternatives
    env.perform(*alternatives[0])
    assert env.stable() and env.levels_completed == 0
    result = search(
        env, limit=env.steps_left,
        node_limit=REFERENCE_PROFILES[7]["search_work"],
    )
    assert result.actions is not None and not result.truncated, result
    assert replay(env, result.actions)


def test_bounds_unsupported_state_and_invalid_actions_are_explicit():
    env = Env([official_levels()[0]])
    capped = search(env, node_limit=1)
    assert capped.actions is None and capped.truncated and not capped.unsupported
    assert solve(env, node_limit=1) is None and solve.truncated

    for action in ((1, None, None), (6, None, None), (6, -1, 0), (6, 0, 64)):
        with pytest.raises(ValueError):
            env.perform(*action)


def test_bank_cli_writes_a_valid_split_qualified_row(tmp_path):
    output = tmp_path / "vc33.jsonl"
    assert bank_main([
        "--levels", "1", "--seed", "4", "--difficulty", "1",
        "--split", "train", "--attempts", "48", "--max-seeds", "20",
        "--out", str(output),
    ]) == 0
    stored = json.loads(output.read_text().strip())
    assert stored["split"] == "train"
    assert validate_full_standard(
        stored, FULL_STANDARD_CONTRACT["curriculum"][0]
    ) == ()
