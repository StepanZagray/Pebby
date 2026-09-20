import copy
import json

import pytest
from arcengine import GameState

from pebby.games.ar25 import names
from pebby.games.ar25.env import Env, replay
from pebby.games.ar25.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    REFERENCE_PROFILES,
    _alternative_solution_audit,
    _context_env,
    _coverage_paths,
    _native_evidence,
    build_game,
    build_level,
    canonical_spec_identity,
    generate,
    generate_game,
    gameplay_identity,
    geometry_identity,
    split_for_identity,
    validate_full_standard,
)
from pebby.games.ar25.plan import OFFICIAL_TARGETS, search


def _replay_spec(spec):
    level = build_level(spec)
    env = Env([level.clone() for _ in range(spec["context_index"] + 1)])
    env.reset()
    env.set_level(spec["context_index"])
    completed, observation = replay(env, [tuple(action) for action in spec["solution"]])
    assert completed
    assert observation is not None
    assert env.levels_completed == 1
    return env


def test_contract_matches_all_eight_official_contexts():
    assert DIFFICULTIES == tuple(range(1, 9))
    assert FULL_STANDARD_CONTRACT["source_id"] == "ar25-0c556536"
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert FULL_STANDARD_CONTRACT["evidence"]["root_acceptance"].endswith("#root-acceptance")
    assert [row["difficulty"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(8))
    assert all(1 <= row["search_work"] <= 32_000_000 for row in FULL_STANDARD_CONTRACT["curriculum"])


def test_official_characterization_matches_native_levels():
    env = Env()
    env.reset()
    for index, difficulty in enumerate(DIFFICULTIES):
        env.set_level(index)
        profile = REFERENCE_PROFILES[difficulty]
        assert env.native_steps_left == profile["steps"]
        assert len(env.goals()) == profile["goals"]
        assert len(env.movables()) == len(profile["shape_cells"])
        assert tuple(
            sum(int(value) != names.TRANSPARENT for row in shape.pixels for value in row)
            for shape in env.movables()
        ) == profile["shape_cells"]
        assert len(env.mirrors()) == len(profile["mirrors"])
        assert sum(names.TAG_FIXED in mirror.tags for mirror in env.mirrors()) == profile["fixed"]


def test_all_official_levels_have_native_witnesses_in_one_game():
    env = Env()
    frame = env.reset()
    assert len(frame) == len(frame[0]) == 64
    expected_lengths = [15, 11, 40, 22, 28, 53, 37, 47]
    for index, expected_length in enumerate(expected_lengths):
        assert env.level_index == index
        result = search(env, limit=10_000)
        assert result.actions is not None
        assert result.exact and not result.truncated and not result.unsupported
        assert len(result.actions) == expected_length
        completed, _ = replay(env, result.actions)
        assert completed
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_default_tier_generates_valid_native_mechanics(difficulty):
    spec = generate(10_000 + difficulty, difficulty, split="train")
    assert spec is not None
    assert spec["difficulty"] == difficulty
    assert spec["context_index"] == difficulty - 1
    assert spec["geometry_sha256"] == spec["geometry_d4_sha256"]
    assert spec["split"] == split_for_identity(spec["geometry_d4_sha256"])
    evidence = spec["solution_mechanics"]
    assert evidence["won"]
    assert evidence["moved_shape_count"] == len(spec["shapes"])
    assert evidence["moved_mirror_count"] == sum(not row["fixed"] for row in spec["mirrors"])
    assert evidence["contributing_shape_count"] == len(spec["shapes"])
    assert evidence["reflected_goal_count"] > 0
    if difficulty >= 5:
        assert evidence["recursive_goal_count"] > 0
    assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]) == []
    _replay_spec(spec)


def test_generation_is_deterministic_and_json_round_trips():
    first = generate(20_404, 8, split="validation")
    second = generate(20_404, 8, split="validation")
    assert first == second
    restored = json.loads(json.dumps(first))
    assert restored == first
    assert validate_full_standard(restored, FULL_STANDARD_CONTRACT["curriculum"][7]) == []


def test_canonical_splits_are_geometry_partitions_not_rng_labels():
    specs = {
        split: generate(30_303, 6, split=split)
        for split in ("train", "validation", "test")
    }
    assert all(specs.values())
    identities = {spec["geometry_d4_sha256"] for spec in specs.values()}
    assert len(identities) == 3
    for split, spec in specs.items():
        assert spec["split"] == split
        assert split_for_identity(geometry_identity(spec)) == split


def test_complete_generate_game_builds_and_wins_sequentially():
    specs = generate_game(40_404, split="train")
    assert specs is not None and len(specs) == 8
    levels = build_game(specs)
    env = Env(levels)
    env.reset()
    for index in range(8):
        assert env.level_index == index
        result = search(env, limit=10_000)
        assert result.actions is not None
        assert replay(env, result.actions)[0]
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN


def test_reduced_game_is_ergonomic_but_not_accepted_as_native_full_game():
    specs = generate_game(41_414, split="train", difficulties=(1, 3, 5))
    assert specs is not None and [spec["difficulty"] for spec in specs] == [1, 3, 5]
    with pytest.raises(ValueError, match="exactly eight"):
        build_game(specs)


@pytest.mark.parametrize("difficulties", ((2, 1), (1, 1), (True, 2)))
def test_explicit_smoke_tiers_must_be_strictly_increasing_unique_integers(difficulties):
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_game(41_415, split="train", difficulties=difficulties)


def test_generation_exhaustion_reports_terminal_cause_work_tier_and_caps():
    assert generate(41_416, 3, split="validation", limit=0) is None
    report = generate.last_report
    assert report["status"] == "exhausted"
    assert report["terminal_cause"] == "search_limit_zero"
    assert report["difficulty"] == 3
    assert report["caps"] == {"attempts": 64, "search_limit": 0}
    assert report["search_work"] == 0

    assert generate_game(
        41_417, split="test", difficulties=(2, 4), attempts=3, limit=0,
    ) is None
    game_report = generate_game.last_report
    assert game_report["status"] == "exhausted"
    assert game_report["caps"] == {"attempts_per_tier": 3, "search_limit": 0}
    assert game_report["terminal_child"]["ordinal"] == 0
    assert game_report["terminal_child"]["difficulty"] == 2
    assert game_report["terminal_child"]["report"]["terminal_cause"] == "search_limit_zero"


def test_tier_two_rejects_known_four_action_public_shortcut_without_padding():
    spec = generate(167, 2, split="train")
    assert spec is not None
    assert spec["attempt"] > 0
    assert spec["rejections"]["below_floor_shortcut"] >= 1
    audit = spec["alternative_solution_audit"]
    assert audit["max_actions"] == REFERENCE_PROFILES[2]["witness_range"][0] - 1
    assert audit["shortcut_actions"] is None
    assert len(spec["solution"]) >= REFERENCE_PROFILES[2]["witness_range"][0]
    assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][1]) == []

    rejected = generate.last_report["shortcut_rejections"][0]
    assert rejected["attempt"] == 0
    assert rejected["route"] == [
        [names.ACTION_LEFT, None, None],
        [names.ACTION_CYCLE, None, None],
        [names.ACTION_RIGHT, None, None],
        [names.ACTION_UP, None, None],
    ]


def test_live_prefix_and_undo_recovery():
    spec = generate(50_505, 7, split="train")
    env = Env([build_level(spec).clone() for _ in range(7)])
    env.reset(); env.set_level(6)
    initial = search(env)
    assert initial.actions
    env.perform(*initial.actions[0])
    env.perform(*initial.actions[1])
    assert env.history_depth >= 1
    env.perform(names.ACTION_UNDO)
    recovered = search(env)
    assert recovered.actions is not None
    assert replay(env, recovered.actions)[0]


def test_display_click_and_cycle_selection_are_both_native_exercised():
    click = generate(60_602, 6, split="train")
    cycle = generate(60_603, 7, split="train")
    assert click["solution_mechanics"]["click_selections"] > 0
    assert click["solution_mechanics"]["cycle_selections"] == 0
    assert cycle["solution_mechanics"]["cycle_selections"] > 0
    assert cycle["solution_mechanics"]["click_selections"] == 0


@pytest.mark.parametrize("restriction", ("horizontal", "vertical"))
def test_zero_incidence_reflection_restrictions_are_explicit_extensions(restriction):
    spec = generate(70_705, 5, split="train", reflection_restriction=restriction)
    assert spec is not None
    assert spec["generation_mode"] == "engine_extension"
    assert spec["mechanics"]["official_incidence"]["restriction_extension"] == 0
    assert spec["solution_mechanics"]["reflection_orientations_used"] == [restriction]
    assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][4]) == []


@pytest.mark.parametrize("difficulty,rotation", ((1, "vertical"), (3, "horizontal")))
def test_zero_incidence_rotation_is_an_explicit_native_extension(difficulty, rotation):
    spec = generate(80_800 + difficulty, difficulty, split="train", rotation=rotation)
    assert spec is not None
    assert spec["generation_mode"] == "engine_extension"
    assert spec["mechanics"]["official_incidence"]["rotation_extension"] == 0
    assert spec["solution_mechanics"]["rotations_observed_before_completion"] >= 4
    assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]) == []


def test_symbolic_reflection_snapshot_matches_native_engine():
    for restriction in (None, "horizontal"):
        spec = generate(90_908, 8 if restriction is None else 5, split="train", reflection_restriction=restriction)
        env = Env([build_level(spec)])
        env.reset()
        mirrors = [(row["orientation"], row["initial_coordinate"]) for row in spec["mirrors"]]
        symbolic = set()
        for row in spec["shapes"]:
            symbolic.update(_coverage_paths(tuple(row["initial_position"]), {(x, y) for y, line in enumerate(row["mask"]) for x, value in enumerate(line) if value == "#"}, mirrors, row["reflection"]))
        native = env.game.naxbskjmlg()
        native_cells = {(x, y) for y in range(native.shape[0]) for x in range(native.shape[1]) if int(native[y, x]) >= 0}
        assert symbolic == native_cells


def test_gameplay_identity_binds_public_puzzle_not_private_teacher_certificate():
    spec = generate(95_905, 6, split="train")
    gameplay = gameplay_identity(spec)
    certificate = canonical_spec_identity(spec)

    private_mutations = []
    changed = copy.deepcopy(spec); changed["shapes"][0]["target_position"] = changed["shapes"][0]["initial_position"]; private_mutations.append(changed)
    changed = copy.deepcopy(spec); changed["mirrors"][0]["target_coordinate"] = changed["mirrors"][0]["initial_coordinate"]; private_mutations.append(changed)
    changed = copy.deepcopy(spec); changed["selection_mode"] = "cycle" if changed["selection_mode"] == "click" else "click"; private_mutations.append(changed)
    changed = copy.deepcopy(spec); changed["proof"]["search_work"] += 1; private_mutations.append(changed)
    changed = copy.deepcopy(spec); changed["solution_mechanics"]["movement_actions"] += 1; private_mutations.append(changed)
    for changed in private_mutations:
        assert gameplay_identity(changed) == gameplay
        assert canonical_spec_identity(changed) != certificate

    public_change = copy.deepcopy(spec)
    public_change["shapes"][0]["color"] = 6
    assert gameplay_identity(public_change) != gameplay


def test_blocked_mirror_move_is_an_attempt_not_actual_mechanic_use():
    spec = generate(95_906, 2, split="train")
    baseline = _native_evidence(spec)
    changed = copy.deepcopy(spec)
    changed["solution"] = [[names.ACTION_UP, None, None], *changed["solution"]]
    evidence = _native_evidence(changed)
    assert evidence["movement_attempts"] == baseline["movement_attempts"] + 1
    assert evidence["movement_actions"] == baseline["movement_actions"]
    assert evidence["moved_mirror_count"] == baseline["moved_mirror_count"]


def test_native_evidence_rejects_actions_after_first_win():
    spec = generate(95_907, 1, split="train")
    changed = copy.deepcopy(spec)
    changed["solution"].append([names.ACTION_CYCLE, None, None])
    with pytest.raises(ValueError, match="after native completion"):
        _native_evidence(changed)


@pytest.mark.parametrize(
    "difficulty,seed,shape_positions,reflected,recursive,contributing,native_remaining",
    (
        (4, 25, ((16, 16), (15, 13)), 11, 0, 2, 103),
        (5, 77, ((4, 12),), 16, 6, 1, 106),
        (7, 23, ((9, 13), (0, 14)), 30, 8, 2, 298),
    ),
)
def test_solution_metrics_use_first_winning_native_state_and_remaining_budget(
    difficulty, seed, shape_positions, reflected, recursive, contributing, native_remaining,
):
    spec = generate(seed, difficulty, split="train")
    evidence = spec["solution_mechanics"]
    assert tuple(map(tuple, evidence["winning_shape_positions"])) == shape_positions
    assert evidence["reflected_goal_count"] == reflected
    assert evidence["recursive_goal_count"] == recursive
    assert evidence["contributing_shape_count"] == contributing
    assert evidence["native_budget_slack"] == native_remaining
    assert evidence["native_budget_remaining"] == native_remaining
    assert evidence["action_count_budget_estimate"] == spec["steps"] - len(spec["solution"])


def test_validator_rejects_geometry_route_proof_and_split_tampering():
    spec = generate(100_100, 6, split="validation")
    row = FULL_STANDARD_CONTRACT["curriculum"][5]
    mutations = []
    changed = copy.deepcopy(spec); changed["goals"][0][0] = (changed["goals"][0][0] + 1) % names.GRID; mutations.append(changed)
    changed = copy.deepcopy(spec); changed["solution"] = changed["solution"][:-1]; changed["context_solution"] = changed["solution"]; changed["solution_length"] -= 1; mutations.append(changed)
    changed = copy.deepcopy(spec); changed["proof"]["search_work"] = 0; mutations.append(changed)
    changed = copy.deepcopy(spec); changed["split"] = "test"; mutations.append(changed)
    changed = copy.deepcopy(spec); changed["solution_mechanics"]["moved_shape_count"] = 0; mutations.append(changed)
    for changed in mutations:
        assert validate_full_standard(changed, row)


def test_validator_requires_strict_curriculum_and_action_integer_types():
    spec = generate(100_101, 1, split="train")
    row = FULL_STANDARD_CONTRACT["curriculum"][0]
    for key, value in (("difficulty", True), ("context_index", False), ("search_work", True)):
        changed_row = dict(row)
        changed_row[key] = value
        assert validate_full_standard(spec, changed_row)

    changed = copy.deepcopy(spec)
    changed["solution"][0][0] = True
    changed["context_solution"] = changed["solution"]
    assert any("action id must be an integer" in error for error in validate_full_standard(changed, row))

    changed = copy.deepcopy(spec)
    changed["solution"][0][1] = False
    changed["context_solution"] = changed["solution"]
    assert any("action x must be an integer or None" in error for error in validate_full_standard(changed, row))


def test_full_standard_modes_versions_and_mechanics_fail_closed():
    row = FULL_STANDARD_CONTRACT["curriculum"][1]
    spec = generate(6, 2, split="train")

    bypass = copy.deepcopy(spec)
    bypass["goals"] = [[11, 11]]
    bypass["generation_mode"] = "garbage"
    # Rebuild the private certificate from actual engine results, so the mode
    # enum—not stale hashes or fabricated evidence—is what rejects this row.
    result = search(_context_env(bypass))
    assert result.actions is not None
    bypass["solution"] = bypass["context_solution"] = [list(action) for action in result.actions]
    bypass["solution_length"] = len(result.actions)
    bypass["solution_mechanics"] = _native_evidence(bypass)
    bypass["geometry_sha256"] = bypass["geometry_d4_sha256"] = geometry_identity(bypass)
    bypass["gameplay_sha256"] = gameplay_identity(bypass)
    bypass["split"] = bypass["geometry_partition"] = split_for_identity(bypass["geometry_sha256"])
    bypass["proof"]["search_work"] = result.expanded
    bypass["canonical_spec_identity"] = canonical_spec_identity(bypass)
    assert any("generation_mode" in error for error in validate_full_standard(bypass, row))

    bad_mechanics = copy.deepcopy(spec)
    bad_mechanics["mechanics"]["movable_shapes"] = 999
    bad_mechanics["canonical_spec_identity"] = canonical_spec_identity(bad_mechanics)
    assert any("mechanics" in error for error in validate_full_standard(bad_mechanics, row))

    bad_version = copy.deepcopy(spec)
    bad_version["generator_version"] = 2.0
    assert any("generator_version" in error for error in validate_full_standard(bad_version, row))


def test_validator_rejects_initially_covered_spawn_with_return_trigger_shortcut():
    spec = copy.deepcopy(generate(1, 1))
    spec["shapes"][0].update(
        mask=[".#.", "###", ".#."],
        initial_position=[3, 5],
        target_position=[15, 5],
    )
    spec["mirrors"][0].update(initial_coordinate=10, target_coordinate=10)
    spec["goals"] = [[4, 5], [3, 6], [4, 6], [5, 6], [4, 7]]
    spec["construction"]["target_shape_positions"] = [[15, 5]]
    spec["construction"]["target_mirrors"] = [["vertical", 10]]

    result = search(_context_env(spec))
    assert result.actions is not None
    assert [action[0] for action in result.actions] == [names.ACTION_RIGHT] * 12
    spec["solution"] = spec["context_solution"] = [list(action) for action in result.actions]
    spec["solution_length"] = len(result.actions)
    spec["solution_mechanics"] = _native_evidence(spec)
    spec["alternative_solution_audit"] = _alternative_solution_audit(spec)
    spec["geometry_sha256"] = spec["geometry_d4_sha256"] = geometry_identity(spec)
    spec["gameplay_sha256"] = gameplay_identity(spec)
    spec["split"] = spec["geometry_partition"] = split_for_identity(spec["geometry_sha256"])
    spec["proof"]["search_work"] = result.expanded
    spec["canonical_spec_identity"] = canonical_spec_identity(spec)

    initial = _context_env(spec)
    assert initial.game.vplrhaovhr()
    assert replay(_context_env(spec), [
        (names.ACTION_RIGHT, None, None),
        (names.ACTION_LEFT, None, None),
    ])[0]
    assert spec["alternative_solution_audit"] == {
        "method": "bounded-public-configuration-bfs-plus-native-replay",
        "work_limit": 25_000,
        "max_actions": 7,
        "status": "initially_covered",
        "work": 0,
        "truncated": False,
        "shortcut_actions": 0,
        "route": [],
    }
    assert any(
        "initial public configuration already covers all goals" in error
        for error in validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][0])
    )


def test_audit_rejects_initial_coverage_from_multi_shape_union():
    spec = copy.deepcopy(generate(3, 3))
    spec["shapes"][0].update(
        mask=["####", "#...", "#...", "#..."],
        initial_position=[2, 3],
    )
    spec["shapes"][1].update(mask=["####", "##.."], initial_position=[12, 4])
    spec["mirrors"][0]["initial_coordinate"] = 10
    spec["goals"] = [
        [2, 3], [2, 4], [2, 5], [2, 6], [2, 14], [2, 15], [2, 16], [2, 17],
        [3, 3], [3, 17], [4, 3], [4, 17], [5, 3], [5, 17],
        [12, 4], [12, 5], [12, 15], [12, 16], [13, 4], [13, 5],
        [13, 15], [13, 16], [14, 4], [14, 16], [15, 4], [15, 16],
    ]

    goals = {tuple(point) for point in spec["goals"]}
    mirror = (("horizontal", 10),)
    per_shape = [
        set(_coverage_paths(tuple(shape["initial_position"]), {
            (x, y)
            for y, row in enumerate(shape["mask"])
            for x, value in enumerate(row)
            if value == "#"
        }, mirror, "both"))
        for shape in spec["shapes"]
    ]
    assert all(not goals <= coverage for coverage in per_shape)
    assert goals <= set().union(*per_shape)
    assert _context_env(spec).game.vplrhaovhr()
    assert _alternative_solution_audit(spec)["status"] == "initially_covered"
    assert any(
        "initial public configuration already covers all goals" in error
        for error in validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][2])
    )


def test_env_rejects_coerced_action_and_coordinate_types():
    env = Env()
    env.reset()
    with pytest.raises(ValueError, match="action_id must be an integer"):
        env.perform(True)
    with pytest.raises(ValueError, match="click x must be an integer"):
        env.perform(names.ACTION_CLICK, True, 1)
    with pytest.raises(ValueError, match="non-click actions require null coordinates"):
        env.perform(names.ACTION_UP, 0, None)


def test_build_game_rejects_order_duplicates_and_split_mismatch():
    specs = generate_game(110_110, split="test")
    changed = copy.deepcopy(specs); changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(ValueError, match="ordered"):
        build_game(changed)
    changed = copy.deepcopy(specs); changed[-1] = copy.deepcopy(changed[0]); changed[-1]["difficulty"] = 8; changed[-1]["context_index"] = 7
    with pytest.raises(ValueError, match="duplicate"):
        build_game(changed)
    changed = copy.deepcopy(specs); changed[-1]["split"] = "train"
    with pytest.raises(ValueError, match="one split"):
        build_game(changed)
    changed = copy.deepcopy(specs); changed[0] = None
    with pytest.raises(ValueError, match=r"specs\[0\] must be a mapping"):
        build_game(changed)
    for field in ("split", "geometry_d4_sha256", "gameplay_sha256"):
        changed = copy.deepcopy(specs); changed[0][field] = []
        with pytest.raises(ValueError, match=field):
            build_game(changed)


def test_build_level_rejects_malformed_specs():
    spec = generate(120_120, 1, split="train")
    bad = copy.deepcopy(spec); bad["shapes"][0]["mask"] = ["...", "..."]
    with pytest.raises(ValueError, match="occupied pixel"):
        build_level(bad)
