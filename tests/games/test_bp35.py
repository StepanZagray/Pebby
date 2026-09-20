import copy
import importlib
import json

from arcengine import GameState
import pytest

from pebby.games.bp35 import names
from pebby.games.bp35.env import Env, official_levels, replay, upstream
from pebby.games.bp35.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    SEARCH_WORK,
    build_game,
    build_level,
    generate,
    generate_game,
    geometry_split,
    validate_full_standard,
)
from pebby.games.bp35.layout import extract
from pebby.games.bp35.plan import (
    OFFICIAL_REFERENCE_PROVENANCE,
    OFFICIAL_ROUTES,
    search,
    solve,
)
from pebby.multigame import (
    RolloutOptions,
    SearchLimits,
    collect_generated_game,
    preflight,
)


bp_generate = importlib.import_module("pebby.games.bp35.generate")


def _context_env(spec):
    difficulty = spec["difficulty"]
    env = Env([build_level(spec) for _ in range(difficulty)])
    env.set_level(difficulty - 1)
    return env


@pytest.fixture(scope="module")
def tier_specs():
    specs = [generate(5000 + difficulty, difficulty) for difficulty in DIFFICULTIES]
    assert all(spec is not None for spec in specs)
    return specs


def test_official_curriculum_is_exactly_nine_and_every_witness_wins_sequentially():
    assert DIFFICULTIES == tuple(range(1, len(official_levels()) + 1))
    assert tuple(FULL_STANDARD_CONTRACT["curriculum"]) == tuple(
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": SEARCH_WORK[difficulty - 1],
        }
        for difficulty in DIFFICULTIES
    )
    assert FULL_STANDARD_CONTRACT["status"] == "ready"

    env = Env()
    env.reset()
    lengths = []
    for difficulty in DIFFICULTIES:
        result = search(env, limit=100)
        assert result.actions == OFFICIAL_ROUTES[difficulty]
        assert result.exact and not result.truncated and not result.unsupported
        assert all(action in env.available_actions for action, _, _ in result.actions)
        completed, _ = replay(env, result.actions)
        assert completed
        lengths.append(len(result.actions))
    assert lengths == [15, 45, 34, 19, 32, 41, 44, 49, 71]
    assert OFFICIAL_REFERENCE_PROVENANCE[9]["native_source_id"] == "bp35-0a0ad940"
    assert OFFICIAL_REFERENCE_PROVENANCE[9]["locally_replayed"] is True
    assert OFFICIAL_REFERENCE_PROVENANCE[9]["optimality_claimed"] is False
    assert env.state == GameState.WIN
    assert env.levels_completed == 9


def test_all_generated_tiers_recompute_full_contract_and_native_evidence(tier_specs):
    for difficulty, spec in zip(DIFFICULTIES, tier_specs):
        assert spec["difficulty"] == difficulty
        assert spec["context_index"] == difficulty - 1
        assert spec["solution_length"] == len(spec["solution"])
        assert spec["proof"]["exact"] is True
        assert spec["proof"]["unsupported"] is False
        assert spec["witness_optimality_claimed"] is False
        assert spec["engine_verified"] is True
        assert spec["context_engine_verified"] is True
        assert spec["solution_mechanics"]["visual_frames_checked"] == len(spec["solution"])
        assert spec["solution_mechanics"]["minimum_click_margin"] >= 3
        if difficulty >= 2:
            assert spec["constraint_probes"]["down_spike_loss"] is True
        if difficulty in (5, 9):
            assert spec["constraint_probes"]["up_spike_loss"] is True
        assert spec["shortcut_probe"]["truncated"] is False
        assert spec["shortcut_probe"]["reduced_length"] >= bp_generate.MIN_ACTIONS[difficulty - 1]
        if difficulty >= 8:
            assert spec["strategy_probes"] == {
                "horizon": 12,
                "expanded": 1,
                "truncated": False,
                "win": False,
                "growth_removed_certified_route_wins": False,
            }
            reduced = spec["shortcut_probe"]["reduced_solution_mechanics"]
            assert reduced["growth_clicks"] >= 20
            assert reduced["bridge_open_to_solid"] >= 1
            assert reduced["bridge_solid_to_open"] >= 1
        if difficulty in (6, 7):
            lattice = spec["lattice_strategy_probes"]
            assert lattice["truncated"] is False
            assert lattice["old_six_action_shortcut_wins"] is False
            assert lattice["initial_final_column_shortcut_wins"] is False
            assert lattice["remote_ascent_shortcut_wins"] is False
            assert lattice["underside_zero_closure_shortcut_wins"] is False
            assert lattice["underside_shifted_shortcut_wins"] is False
            assert lattice["certified_without_gravity_changes_wins"] is False
            assert lattice["certified_without_bridge_closures_wins"] is False
            assert lattice["premature_reversal_loses"] is True
            assert lattice["trap_ablated_premature_reversal_loses"] is False
            assert lattice["initial_static_adjacent_search"] == {
                "method": "native_initial_static_adjacent_reachability_v1",
                "action_classes": ["left", "right", "adjacent_solid_gate_open"],
                "max_expanded": 512,
                "expanded": 23 if difficulty == 6 else 27,
                "frontier_exhausted": True,
                "truncated": False,
                "win": False,
            }
            assert all(
                not row["reached_target_column"] and not row["win"]
                for key in ("arrival_transfers", "ascent_transfers")
                for row in lattice[key]
            )
            assert lattice["arrival_transfers"][0]["opened_first_gates"] == 0
            assert lattice["ascent_transfers"][0]["opened_first_gates"] == 0
            chamber = spec["chamber_structure"]
            assert chamber["method"] == "native_three_chamber_all_rows_v1"
            assert chamber["checked_rows"] == len(spec["rows_bottom_up"])
            assert chamber["partition_cells_checked"] == 2 * len(spec["rows_bottom_up"])
            assert chamber["row_match"] is True
            assert chamber["native_entity_match"] is True
            assert chamber["top_corridor_clear"] is True
            assert chamber["player_in_middle_chamber"] is True
            assert chamber["goal_in_final_chamber"] is True
            assert chamber["unexpected_mutable_barriers"] == []
            assert chamber["valid"] is True
        assert validate_full_standard(
            spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        ) == []


def test_generated_native_layouts_are_not_exact_official_copies(tier_specs):
    for difficulty, spec in zip(DIFFICULTIES, tier_specs):
        official = Env()
        official.set_level(difficulty - 1)
        assert extract(_context_env(spec)).entity_state != extract(official).entity_state


def test_complete_generated_game_uses_native_contexts_and_wins_without_forced_advance():
    specs = generate_game(8181, split="validation")
    assert specs is not None and len(specs) == len(official_levels()) == 9
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["context_index"] for spec in specs] == list(range(9))
    assert [spec["game_level_index"] for spec in specs] == list(range(9))
    assert all(spec["game_scope"] == "full_standard" for spec in specs)
    assert all(spec["is_full_standard_game"] is True for spec in specs)

    env = Env(build_game(specs))
    env.reset()
    for expected_level in range(9):
        assert env.level_index == expected_level
        result = search(env, limit=1_000_000)
        assert result.actions is not None and result.exact and not result.unsupported
        assert replay(env, result.actions)[0]
        assert env.levels_completed == expected_level + 1
    assert env.state == GameState.WIN


def test_build_game_accepts_independently_generated_ordered_specs(tier_specs):
    assert all("game_level_index" not in spec for spec in tier_specs)
    assert len(build_game(tier_specs)) == len(DIFFICULTIES)


def test_known_tier_eight_three_action_growth_shortcut_does_not_win():
    spec = generate(5, 8, split="train")
    assert spec is not None
    env = _context_env(spec)
    start = env.levels_completed
    for action in (
        (names.ACTION_LEFT, None, None),
        (names.ACTION_CLICK, 21, 51),
        (names.ACTION_CLICK, 33, 35),
    ):
        observation = env.perform(*action)
        if observation.state != GameState.NOT_FINISHED:
            break
    assert env.levels_completed == start
    assert env.state != GameState.WIN


@pytest.mark.parametrize("difficulty", (6, 7))
def test_gravity_lattice_six_action_shortcut_does_not_win(difficulty):
    spec = generate(5000 + difficulty, difficulty, split="train")
    assert spec is not None
    env = _context_env(spec)
    start = env.levels_completed
    for action in (
        (names.ACTION_LEFT, None, None),
        (names.ACTION_CLICK, 27, 33),
        *((names.ACTION_RIGHT, None, None),) * 4,
    ):
        observation = env.perform(*action)
        if observation.state != GameState.NOT_FINISHED:
            break
    assert env.levels_completed == start
    assert env.state != GameState.WIN


def test_tier_seven_initial_shelf_final_column_shortcut_does_not_win():
    spec = generate(5007, 7, split="train")
    assert spec is not None
    env = _context_env(spec)
    start = env.levels_completed
    route = (
        *((names.ACTION_RIGHT, None, None),) * 3,
        *((names.ACTION_CLICK, 51, 33),) * 10,
    )
    for action in route:
        observation = env.perform(*action)
        if observation.state != GameState.NOT_FINISHED:
            break
    assert env.levels_completed == start
    assert env.state != GameState.WIN


def test_tier_six_remote_ascent_landing_shortcut_does_not_win():
    spec = generate(5006, 6, split="train")
    assert spec is not None
    env = _context_env(spec)
    start = env.levels_completed
    route = (
        (names.ACTION_LEFT, None, None),
        *((names.ACTION_CLICK, 27, 33),) * 6,
        (names.ACTION_CLICK, 15, 15),
        *((names.ACTION_LEFT, None, None),) * 2,
        (names.ACTION_CLICK, 9, 15),
        *((names.ACTION_RIGHT, None, None),) * 6,
        (names.ACTION_CLICK, 9, 23),
        *((names.ACTION_CLICK, 51, 33),) * 8,
    )
    for action in route:
        observation = env.perform(*action)
        if observation.state != GameState.NOT_FINISHED:
            break
    assert env.levels_completed == start
    assert env.state != GameState.WIN


def test_tier_six_underside_zero_closure_shortcut_does_not_win():
    spec = generate(5006, 6, split="train")
    assert spec is not None
    env = _context_env(spec)
    start = env.levels_completed
    route = (
        (names.ACTION_LEFT, None, None),
        *((names.ACTION_CLICK, 27, 33),) * 7,
        *((names.ACTION_RIGHT, None, None),) * 2,
        (names.ACTION_CLICK, 9, 15),
        *((names.ACTION_RIGHT, None, None),) * 2,
        *((names.ACTION_CLICK, 51, 35),) * 7,
        (names.ACTION_CLICK, 9, 23),
        (names.ACTION_CLICK, 51, 33),
    )
    for action in route:
        observation = env.perform(*action)
        if observation.state != GameState.NOT_FINISHED:
            break
    assert env.levels_completed == start
    assert env.state != GameState.WIN


def test_tier_seven_shifted_underside_shortcut_does_not_win():
    spec = generate(5007, 7, split="train")
    assert spec is not None
    env = _context_env(spec)
    start = env.levels_completed
    route = (
        (names.ACTION_RIGHT, None, None),
        *((names.ACTION_CLICK, 39, 33),) * 9,
        (names.ACTION_CLICK, 33, 33),
        (names.ACTION_LEFT, None, None),
        (names.ACTION_CLICK, 39, 57),
        (names.ACTION_CLICK, 9, 15),
        *((names.ACTION_RIGHT, None, None),) * 3,
        *((names.ACTION_CLICK, 51, 35),) * 9,
        (names.ACTION_CLICK, 9, 23),
        (names.ACTION_CLICK, 51, 33),
    )
    for action in route:
        observation = env.perform(*action)
        if observation.state != GameState.NOT_FINISHED:
            break
    assert env.levels_completed == start
    assert env.state != GameState.WIN


def test_gravity_chamber_certificate_rejects_a_partition_hole():
    spec = generate(5006, 6, split="train")
    assert spec is not None
    broken = copy.deepcopy(spec)
    rows = [list(row) for row in broken["rows_bottom_up"]]
    rows[12][broken["chamber_structure"]["ascent_partition_column"]] = " "
    broken["rows_bottom_up"] = ["".join(row) for row in rows]
    certificate = bp_generate._gravity_chamber_certificate(broken)
    assert certificate["row_match"] is False
    assert certificate["native_entity_match"] is False
    assert certificate["valid"] is False


def test_native_click_and_undo_preserve_visible_chamber_barriers():
    spec = generate(5007, 7, split="train")
    assert spec is not None
    env = _context_env(spec)
    wall = (spec["chamber_structure"]["ascent_partition_column"], 28)
    spike = tuple(spec["chamber_structure"]["middle_trap_cells"][0])
    assert bp_generate._target_name(env, wall) == names.WALL
    assert bp_generate._target_name(env, spike) == names.SPIKE_B
    for cell, expected in ((wall, names.WALL), (spike, names.SPIKE_B)):
        observation = bp_generate._perform_cell_click(env, cell)
        assert observation is not None
        assert observation.state == GameState.NOT_FINISHED
        assert bp_generate._target_name(env, cell) == expected
        env.perform(names.ACTION_UNDO, None, None)
        assert bp_generate._target_name(env, cell) == expected


def test_structural_relation_grammar_has_measured_variant_floor_per_tier():
    observed = {}
    for difficulty in DIFFICULTIES:
        signatures = set()
        for seed in range(70_000, 70_064):
            construction = bp_generate._draft(seed, difficulty, 0)["construction"]
            if difficulty <= 3:
                keys = ("supports", "route_columns", "start_column", "open_spans")
            elif difficulty <= 7:
                keys = ("supports", "gate_columns", "gate_initial")
            else:
                keys = (
                    "growth_columns", "growth_shafts", "bridge_initial",
                    "final_bridge_initial",
                )
            signatures.add(json.dumps({key: construction[key] for key in keys}, sort_keys=True))
        observed[difficulty] = len(signatures)
    assert observed == {
        1: 51, 2: 59, 3: 24, 4: 44, 5: 46, 6: 9, 7: 9, 8: 23, 9: 11,
    }


def test_explicit_reduced_sequence_is_smoke_and_cannot_build_as_full_game():
    specs = generate_game(9191, difficulties=(1, 3, 8), split="train")
    assert specs is not None
    assert [spec["difficulty"] for spec in specs] == [1, 3, 8]
    assert all(spec["game_scope"] == "explicit_smoke_subset" for spec in specs)
    assert all(spec["is_full_standard_game"] is False for spec in specs)
    with pytest.raises(ValueError, match="exactly nine"):
        build_game(specs)
    with pytest.raises(ValueError, match="increasing members"):
        generate_game(1, difficulties=(1, 1, 2))


@pytest.mark.parametrize("difficulty", (6, 7))
def test_split_partition_is_canonical_and_not_only_an_rng_label(difficulty):
    specs = [
        generate(77, difficulty, split=split)
        for split in ("train", "validation", "test")
    ]
    assert all(spec is not None for spec in specs)
    assert {spec["split"] for spec in specs} == {"train", "validation", "test"}
    assert len({spec["geometry_d4_sha256"] for spec in specs}) == 3
    for spec in specs:
        assert spec["geometry_split"] == spec["split"]
        assert geometry_split(spec["geometry_d4_sha256"]) == spec["split"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda spec: spec.__setitem__("geometry_d4_sha256", "0" * 64),
        lambda spec: spec.__setitem__("source_sha256", "0" * 64),
        lambda spec: spec.__setitem__("profile", {"sections": []}),
        lambda spec: spec.__setitem__("effective_seed", spec["effective_seed"] + 1),
        lambda spec: spec.__setitem__("requested_seed", spec["requested_seed"] + 1),
        lambda spec: spec["solution"][0].__setitem__(0, 2),
        lambda spec: spec["solution_mechanics"].__setitem__("bridge_toggles", 999),
        lambda spec: spec["constraint_probes"].__setitem__("down_spike_loss", False),
        lambda spec: spec["construction"].__setitem__("route_columns", [1]),
    ],
)
def test_validator_rejects_identity_route_proof_seed_and_structure_tampering(tier_specs, mutate):
    spec = copy.deepcopy(tier_specs[5])
    mutate(spec)
    assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][5])


def test_validator_rejects_geometry_and_non_integer_action_tampering(tier_specs):
    row_spec = copy.deepcopy(tier_specs[0])
    rows = row_spec["rows_bottom_up"]
    gem_y, gem_x = next(
        (y, row.index("+")) for y, row in enumerate(rows) if "+" in row
    )
    replacement_x = 1 if gem_x != 1 else 2
    row = list(rows[gem_y])
    row[gem_x], row[replacement_x] = " ", "+"
    rows[gem_y] = "".join(row)
    assert validate_full_standard(row_spec, FULL_STANDARD_CONTRACT["curriculum"][0])

    float_action = copy.deepcopy(tier_specs[1])
    float_action["solution"][0][0] = 3.0
    float_action["context_solution"][0][0] = 3.0
    assert validate_full_standard(float_action, FULL_STANDARD_CONTRACT["curriculum"][1])

    bool_curriculum = dict(FULL_STANDARD_CONTRACT["curriculum"][0], difficulty=True)
    assert validate_full_standard(tier_specs[0], bool_curriculum)


def test_validator_rejects_recursive_numeric_type_aliases():
    tier_six = generate(5006, 6, split="train")
    tier_one = generate(5001, 1, split="train")
    assert tier_six is not None and tier_one is not None
    assert validate_full_standard(tier_six, FULL_STANDARD_CONTRACT["curriculum"][5]) == []
    assert validate_full_standard(tier_one, FULL_STANDARD_CONTRACT["curriculum"][0]) == []
    assert tier_one["strategy_probes"] is None
    assert tier_one["lattice_strategy_probes"] is None
    assert tier_one["chamber_structure"] is None

    mutations = []
    proof_flag = copy.deepcopy(tier_six)
    proof_flag["proof"]["chamber_structure"] = copy.deepcopy(
        proof_flag["proof"]["chamber_structure"]
    )
    proof_flag["proof"]["chamber_structure"]["valid"] = 1
    mutations.append((
        proof_flag,
        FULL_STANDARD_CONTRACT["curriculum"][5],
        "proof.chamber_structure does not mirror",
    ))

    chamber_count = copy.deepcopy(tier_six)
    chamber_count["chamber_structure"]["checked_rows"] = 33.0
    chamber_count["proof"]["chamber_structure"]["checked_rows"] = 33.0
    mutations.append((
        chamber_count,
        FULL_STANDARD_CONTRACT["curriculum"][5],
        "three-chamber structure differs",
    ))

    event_flag = copy.deepcopy(tier_one)
    event_flag["solution_mechanics"]["won"] = True
    event_flag["proof"]["solution_mechanics"]["won"] = True
    mutations.append((
        event_flag,
        FULL_STANDARD_CONTRACT["curriculum"][0],
        "solution mechanic census differs",
    ))

    shortcut_count = copy.deepcopy(tier_one)
    shortcut_count["shortcut_probe"]["reduced_length"] = float(
        shortcut_count["shortcut_probe"]["reduced_length"]
    )
    shortcut_count["proof"]["shortcut_probe"]["reduced_length"] = shortcut_count[
        "shortcut_probe"
    ]["reduced_length"]
    mutations.append((
        shortcut_count,
        FULL_STANDARD_CONTRACT["curriculum"][0],
        "shortcut probe differs",
    ))

    for key, alias, diagnostic in (
        ("engine_verified", 1, "engine_verified is missing or inconsistent"),
        ("context_index", 0.0, "context_index is missing or inconsistent"),
        ("requested_seed", float(tier_one["requested_seed"]), "requested/effective seed"),
    ):
        mutated = copy.deepcopy(tier_one)
        mutated[key] = alias
        mutated["proof"][key] = alias
        mutations.append((mutated, FULL_STANDARD_CONTRACT["curriculum"][0], diagnostic))

    for mutated, curriculum, diagnostic in mutations:
        errors = validate_full_standard(mutated, curriculum)
        assert any(diagnostic in error for error in errors), errors

    for key in ("context_index", "search_work"):
        curriculum = dict(FULL_STANDARD_CONTRACT["curriculum"][0])
        curriculum[key] = float(curriculum[key])
        errors = validate_full_standard(tier_one, curriculum)
        assert "curriculum row differs from the calibrated tier" in errors


def test_generated_native_identity_cannot_alias_an_official_level():
    spec = generate(5006, 6, split="train")
    assert spec is not None
    official = bp_generate._official_native_d4_identities()
    assert len(official) == len(official_levels()) == 9
    assert spec["native_initial_d4_sha256"] not in official
    mutated = copy.deepcopy(spec)
    mutated["native_initial_d4_sha256"] = official[0]
    mutated["proof"]["native_initial_d4_sha256"] = official[0]
    errors = validate_full_standard(mutated, FULL_STANDARD_CONTRACT["curriculum"][5])
    assert any("native initial entity-grid identity" in error for error in errors)


def test_native_point_identity_is_d4_translation_order_and_duplicate_invariant():
    official = Env()
    official.set_level(2)
    points = bp_generate._native_initial_points(official)
    transformed = [
        (-y + 41, x - 17, copy.deepcopy(role))
        for x, y, role in reversed(points)
    ]
    transformed.extend(copy.deepcopy(transformed[:5]))
    assert bp_generate._native_points_d4_sha256(transformed) == (
        bp_generate._native_initial_d4_sha256(official)
    )

    integer_role = [["value", ["int", 1]]]
    string_role = [["value", ["str", "1"]]]
    assert bp_generate._native_points_d4_sha256([(0, 0, integer_role)]) != (
        bp_generate._native_points_d4_sha256([(0, 0, string_role)])
    )

    grower_role = [["name", ["str", names.GROWER]]]
    two_cell_grower = [(0, 0, grower_role), (0, 1, grower_role)]
    assert bp_generate._native_points_d4_sha256(two_cell_grower) != (
        bp_generate._native_points_d4_sha256(two_cell_grower[:1])
    )


def test_native_certificate_rejects_illegal_or_post_win_tail_actions(tier_specs):
    illegal = copy.deepcopy(tier_specs[0])
    illegal["solution"].append([999, None, None])
    with pytest.raises(ValueError, match="illegal action"):
        bp_generate._replay_certificate(illegal)

    post_win = copy.deepcopy(tier_specs[0])
    post_win["solution"].append([names.ACTION_LEFT, None, None])
    with pytest.raises(ValueError, match="continues after the first native win"):
        bp_generate._replay_certificate(post_win)


def test_validator_rejects_rehashed_collision_topology_mutation(tier_specs):
    mutated = copy.deepcopy(tier_specs[7])
    rows = mutated["rows_bottom_up"]
    y, x = next(
        (y, x)
        for y, row in enumerate(rows[2:-2], start=2)
        for x, char in enumerate(row[2:-2], start=2)
        if char == " "
    )
    row = list(rows[y])
    row[x] = "o"
    rows[y] = "".join(row)
    raw, d4, gameplay = bp_generate.geometry_identities(mutated)
    mutated.update(
        geometry_sha256=raw,
        geometry_d4_sha256=d4,
        gameplay_sha256=gameplay,
        geometry_split=geometry_split(d4),
        split=geometry_split(d4),
        effective_split=geometry_split(d4),
    )
    for key in (
        "geometry_sha256", "geometry_d4_sha256", "gameplay_sha256", "split",
    ):
        mutated["proof"][key] = mutated[key]
    errors = validate_full_standard(mutated, FULL_STANDARD_CONTRACT["curriculum"][7])
    assert any("rows_bottom_up differs from the versioned tier grammar" in error for error in errors)


def test_json_round_trip_preserves_certificate(tier_specs):
    original = tier_specs[-1]
    restored = json.loads(json.dumps(original))
    assert restored == original
    assert validate_full_standard(
        restored, FULL_STANDARD_CONTRACT["curriculum"][-1]
    ) == []


def test_live_prefix_and_undo_recovery_are_native_replayed(tier_specs):
    spec = tier_specs[3]
    env = _context_env(spec)
    env.perform(names.ACTION_CLICK, 0, 0)
    result = search(env, limit=500)
    assert result.actions is not None and result.exact and not result.unsupported
    assert replay(env, result.actions)[0]

    env = _context_env(spec)
    env.perform(*spec["solution"][0])
    env.perform(names.ACTION_UNDO)
    recovered = search(env, limit=500)
    assert recovered.actions is not None and recovered.exact
    assert replay(env, recovered.actions)[0]

    env = _context_env(spec)
    env.perform(*spec["solution"][0])
    suffix = search(env, limit=500)
    assert suffix.actions is not None and suffix.exact
    assert replay(env, suffix.actions)[0]


def test_search_limits_and_action_triples_are_explicit(tier_specs):
    official_zero = search(Env(), limit=0)
    assert official_zero.actions is None and official_zero.truncated and official_zero.exact

    env = _context_env(tier_specs[1])
    truncated = search(env, limit=0)
    assert truncated.actions is None and truncated.truncated and truncated.exact

    zero_budget = search(env, limit=10, budget=0)
    assert zero_budget.actions is None and not zero_budget.truncated and zero_budget.exact
    assert not zero_budget.unsupported

    actions = solve(_context_env(tier_specs[1]), limit=500)
    assert actions
    for action_id, x, y in actions:
        assert action_id in names.AVAILABLE_ACTIONS
        if action_id == names.ACTION_CLICK:
            assert type(x) is type(y) is int
            assert 0 <= x < 64 and 0 <= y < 64
        else:
            assert x is None and y is None


def test_build_level_rejects_malformed_grids_and_contexts(tier_specs):
    missing_gem = copy.deepcopy(tier_specs[0])
    missing_gem["rows_bottom_up"] = [row.replace("+", " ") for row in missing_gem["rows_bottom_up"]]
    with pytest.raises(ValueError, match="player and gem"):
        build_level(missing_gem)

    no_boundary = copy.deepcopy(tier_specs[0])
    no_boundary["rows_bottom_up"][0] = "o" + " " * 9 + "o"
    with pytest.raises(ValueError, match="boundaries"):
        build_level(no_boundary)

    bad_context = copy.deepcopy(tier_specs[1])
    bad_context["context_index"] = True
    with pytest.raises(ValueError, match="context_index"):
        build_level(bad_context)


def test_official_and_generated_environments_do_not_leak_module_globals(tier_specs):
    module = upstream()
    official_grid1 = module.tjdtolkmxo["grid1"]
    generated = _context_env(tier_specs[4])
    assert module.tjdtolkmxo["grid1"] is official_grid1
    official = Env()
    assert replay(generated, search(generated, limit=500).actions)[0]
    assert replay(official, search(official, limit=100).actions)[0]
    assert module.tjdtolkmxo["grid1"] is official_grid1


def test_shared_collector_calls_the_public_search_entry_and_wins_tier_one():
    modules, = preflight(["bp35"])
    collected = collect_generated_game(
        modules,
        master_seed=0,
        game_index=0,
        difficulties=[1],
        limits=SearchLimits(max_actions_per_level=64, max_search_work=20_000),
        outer_generation_attempts=4,
        generator_attempts=16,
        rollout=RolloutOptions(random_action_probability=0.0, max_game_steps=64),
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_completed"] == 1
    assert collected.record["teacher_steps"] == collected.record["steps"]
    assert collected.record["searches"][0]["exact"] is True


@pytest.mark.parametrize("bad", [True, 1.5, "1"])
def test_generator_rejects_non_integer_difficulty(bad):
    with pytest.raises(ValueError, match="difficulty must be an integer"):
        generate(1, bad)


def test_generation_bounds_are_enforced():
    generate.last_rejections = {"stale": 1}
    assert generate(1, 1, attempts=1, limit=0) is None
    assert generate.last_rejections == {}
    with pytest.raises(ValueError, match="attempts cannot exceed"):
        generate(1, 1, attempts=257)
    with pytest.raises(ValueError, match="split"):
        generate(1, 1, split="official")
