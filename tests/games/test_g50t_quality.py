"""Reference-calibrated full-standard checks for G50T."""

from copy import deepcopy
import json

from arcengine import GameState

from pebby.games.g50t import names
from pebby.games.g50t.env import Env, official_levels, replay
from pebby.games.g50t.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    PROFILES,
    _OFFICIAL_GEOMETRY_SHA256,
    _draft,
    _layout_geometry_spec,
    canonical_action_identity,
    build_game,
    build_level,
    device_relation_identity,
    generate,
    generate_game,
    gameplay_identity,
    geometry_hashes,
    geometry_identity,
    structural_metrics,
    validate_full_standard,
)
from pebby.games.g50t.layout import extract
from pebby.games.g50t.plan import search, transition


def test_contract_matches_the_exact_official_curriculum():
    assert len(official_levels()) == 7
    assert DIFFICULTIES == tuple(range(1, 8))
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert FULL_STANDARD_CONTRACT["source_id"] == "g50t-5849a774"
    rows = FULL_STANDARD_CONTRACT["curriculum"]
    assert [row["difficulty"] for row in rows] == list(DIFFICULTIES)
    assert [row["context_index"] for row in rows] == list(range(7))
    assert all(0 < row["search_work"] <= 32_000_000 for row in rows)


def test_official_exclusion_identities_match_current_relational_hash_schema():
    env = Env()
    measured = set()
    for difficulty in DIFFICULTIES:
        env.set_level(difficulty - 1)
        measured.add(geometry_hashes(
            _layout_geometry_spec(extract(env), difficulty),
        )[1])
    assert measured == set(_OFFICIAL_GEOMETRY_SHA256)


def test_teacher_solves_all_official_levels_sequentially_and_exercises_full_scope():
    expected = {
        1: (20, 1, 1, 0, 0), 2: (36, 2, 2, 0, 0),
        3: (36, 3, 3, 0, 0), 4: (28, 2, 1, 1, 0),
        5: (42, 4, 3, 1, 0), 6: (44, 5, 5, 0, 1),
        7: (42, 4, 2, 2, 1),
    }
    env = Env()
    union = set()
    for difficulty in DIFFICULTIES:
        assert env.level_index == difficulty - 1
        layout = extract(env)
        assert layout.exact, layout.unsupported
        assert (len(layout.allowed), len(layout.switches), len(layout.doors),
                len(layout.teleports), len(layout.enemies)) == expected[difficulty]
        result = search(env, node_limit=PROFILES[difficulty]["search_work"])
        assert result.actions is not None, result.reason
        assert result.exact and not result.truncated and not result.unsupported
        world = layout.world
        for action, _, _ in result.actions:
            outcome = transition(layout, world, action)
            assert outcome is not None
            world, _, events = outcome
            union.update(events)
        assert replay(env, result.actions)
        assert env.levels_completed == difficulty
    assert env.state == GameState.WIN
    assert {"rewind", "ghost_created", "door_changed", "teleport"} <= union


def test_one_generated_row_per_tier_round_trips_and_revalidates():
    for difficulty in DIFFICULTIES:
        spec = generate(700 + difficulty, difficulty, split="train")
        assert spec is not None
        stored = json.loads(json.dumps(spec))
        assert stored == spec
        row = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        assert validate_full_standard(stored, row) == []
        assert stored["context_index"] == difficulty - 1
        assert not stored["optimality_claim"]
        mechanics = stored["solution_mechanics"]
        assert mechanics["rewinds"] == (1 if difficulty == 1 else 2)
        if difficulty in (4, 5, 7):
            assert mechanics["teleports"] >= len(stored["teleports"])
        if difficulty in (6, 7):
            assert mechanics["enemy_exercised"]
        if difficulty == 6:
            assert len(mechanics["enemy_circuit_activations"]) >= 2
        if difficulty == 7:
            assert mechanics["enemy_triggered_teleport_circuits"]


def test_complete_generated_game_has_seven_native_contexts_and_wins_sequentially():
    specs = generate_game(991, split="train")
    assert specs is not None and len(specs) == 7
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["context_index"] for spec in specs] == list(range(7))
    env = Env(build_game(specs))
    for index, spec in enumerate(specs):
        assert env.level_index == index
        assert replay(env, spec["solution"])
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN

    tampered = deepcopy(specs)
    tampered[0]["solution"][0][0] = (
        names.ACTION_LEFT if tampered[0]["solution"][0][0] != names.ACTION_LEFT
        else names.ACTION_RIGHT
    )
    tampered[0]["context_solution"] = deepcopy(tampered[0]["solution"])
    try:
        build_game(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("build_game accepted a tampered native route")

    malformed = deepcopy(specs)
    malformed[0]["game_seed"] = True
    try:
        build_game(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("build_game accepted a boolean parent seed")


def test_canonical_splits_and_tamper_rejection_are_recomputed():
    rows = []
    for split in ("train", "validation", "test"):
        spec = generate(404, 1, split=split)
        assert spec is not None and spec["split"] == split
        identity, partition, _ = geometry_identity(spec)
        assert identity == spec["geometry_d4_sha256"]
        assert spec["geometry_sha256"] == geometry_hashes(spec)[0]
        assert partition == split
        rows.append(spec)
    assert len({spec["geometry_d4_sha256"] for spec in rows}) == 3

    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    changed_route = deepcopy(rows[0])
    changed_route["solution"][0][0] = (
        names.ACTION_LEFT if changed_route["solution"][0][0] != names.ACTION_LEFT
        else names.ACTION_RIGHT
    )
    assert validate_full_standard(changed_route, entry)
    changed_geometry = deepcopy(rows[0])
    changed_geometry["cells"].pop()
    assert validate_full_standard(changed_geometry, entry)
    changed_proof = deepcopy(rows[0])
    changed_proof["proof"]["engine_win"] = False
    assert validate_full_standard(changed_proof, entry)


def test_eight_seed_per_tier_geometry_and_independently_canonicalized_routes():
    measured_route_counts = {}
    for difficulty in DIFFICULTIES:
        specs = [generate(30_000 + difficulty * 100 + offset, difficulty, split="train")
                 for offset in range(8)]
        assert all(spec is not None for spec in specs)
        assert len({spec["geometry_d4_sha256"] for spec in specs}) == 8
        canonical_routes = {
            canonical_action_identity(spec["solution"]) for spec in specs
        }
        measured_route_counts[difficulty] = len(canonical_routes)
    # This is a pinned audit measurement, not an admission floor.  Action-only
    # identity minimizes over all eight transforms independently of decorated
    # geometry; the earlier geometry-selected transform overstated tier 7 as
    # 8/8 when its actual value was 7/8.
    assert measured_route_counts == {1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 7}


def test_structural_bounds_and_tier_five_device_relations_are_measured():
    relation_signatures = set()
    for seed in range(16_000, 16_012):
        spec = generate(seed, 5, split="train")
        assert spec is not None
        metrics = structural_metrics(spec)
        assert metrics == spec["structure"]
        assert len(metrics["toggle_detours"]) == 2
        assert min(metrics["toggle_detours"]) >= 2
        relation_signatures.add(device_relation_identity(spec))
    assert len(relation_signatures) >= 2


def test_relational_identities_ignore_private_names_translation_and_certificates():
    spec = generate(9_102, 2, split="train")
    assert spec is not None
    raw, canonical, _ = geometry_hashes(spec)
    gameplay = gameplay_identity(spec)

    relabelled = deepcopy(spec)
    mapping = {0: 1, 1: 0}
    for key in ("switches", "doors", "teleports"):
        for value in relabelled[key]:
            value["circuit"] = mapping[value["circuit"]]
    assert geometry_hashes(relabelled) == (raw, canonical, geometry_hashes(spec)[2])
    assert geometry_identity(relabelled)[1] == spec["split"]
    assert gameplay_identity(relabelled) == gameplay
    assert replay(Env([build_level(relabelled)]), relabelled["solution"])

    translated = deepcopy(spec)
    for value in translated["cells"]:
        value[0] += 11
        value[1] -= 7
    for key in ("start", "goal"):
        translated[key][0] += 11
        translated[key][1] -= 7
    for value in translated["switches"] + translated["doors"] + translated["enemies"]:
        value["cell"][0] += 11
        value["cell"][1] -= 7
    for value in translated["teleports"]:
        for pad in value["pads"]:
            pad[0] += 11
            pad[1] -= 7
    for value in translated["enemies"]:
        for cell in value["path"]:
            cell[0] += 11
            cell[1] -= 7
    translated_raw, translated_d4, _ = geometry_hashes(translated)
    assert translated_raw != raw
    assert translated_d4 == canonical
    assert gameplay_identity(translated) == gameplay

    certificate_changed = deepcopy(spec)
    certificate_changed["solution"] = certificate_changed["solution"][:-1]
    certificate_changed["proof"]["search_expanded"] += 1
    assert gameplay_identity(certificate_changed) == gameplay

    assignment_changed = deepcopy(spec)
    assignment_changed["doors"][0]["circuit"], assignment_changed["doors"][1]["circuit"] = (
        assignment_changed["doors"][1]["circuit"],
        assignment_changed["doors"][0]["circuit"],
    )
    assert geometry_hashes(assignment_changed)[1] != canonical
    assert gameplay_identity(assignment_changed) != gameplay


def test_live_enemy_rewind_restores_native_origin_and_symbolic_state():
    spec = generate(71, 6, split="train")
    assert spec is not None
    env = Env([build_level(spec) for _ in range(6)])
    env.set_level(5)
    seen_rewind = False
    for step in spec["solution"]:
        env.perform(*step)
        seen_rewind |= step[0] == names.ACTION_REWIND
        snapshot = extract(env)
        if (seen_rewind and snapshot.world.history
                and snapshot.world.enemies[0].position != snapshot.enemies[0].start):
            break
    layout = extract(env)
    assert layout.world.enemies[0].position != layout.enemies[0].start
    expected, _, _ = transition(layout, layout.world, names.ACTION_REWIND)
    env.perform(names.ACTION_REWIND)
    actual = extract(env).world
    assert actual == expected
    assert actual.enemies[0].position == layout.enemies[0].start

    # Regression for a real enemy teleport: rewind is inverse live history,
    # not an immutable-origin reset.
    teleported = generate(72, 7, split="train")
    assert teleported is not None
    base = Env([build_level(teleported)])
    base.perform(*teleported["solution"][0])
    after = extract(base)

    def cell(position):
        return [(position[0] - 1) // names.GRID_STEP,
                (position[1] - 1) // names.GRID_STEP]

    altered = deepcopy(teleported)
    enemy_switch = next(value for value in altered["switches"] if
                        value["circuit"] == altered["teleports"][0]["circuit"])
    later_switch = next(value for value in altered["switches"] if
                        value["circuit"] == altered["teleports"][1]["circuit"])
    old_switch = deepcopy(enemy_switch["cell"])
    enemy_switch["cell"] = cell(after.world.player)
    later_switch["cell"] = old_switch
    altered["teleports"][0]["pads"][0] = cell(after.world.enemies[0].position)
    choices = [value for value in altered["enemies"][0]["path"]
               if value not in (altered["teleports"][0]["pads"][0], old_switch,
                                altered["enemies"][0]["cell"])]
    altered["teleports"][0]["pads"][1] = max(
        choices,
        key=lambda value: sum(abs(left - right) for left, right in zip(
            value, altered["enemies"][0]["cell"])),
    )
    enemy_env = Env([build_level(altered) for _ in range(7)])
    enemy_env.set_level(6)
    for step in altered["solution"][:3]:
        enemy_env.perform(*step)
    enemy_layout = extract(enemy_env)
    enemy = enemy_layout.world.enemies[0]
    inferred = (
        enemy.position[0] - names.GRID_STEP * sum(
            names.ACTION_DELTA[action][0] for action in enemy.history),
        enemy.position[1] - names.GRID_STEP * sum(
            names.ACTION_DELTA[action][1] for action in enemy.history),
    )
    assert inferred != enemy_layout.enemies[0].start
    enemy_expected, _, _ = transition(
        enemy_layout, enemy_layout.world, names.ACTION_REWIND,
    )
    enemy_env.perform(names.ACTION_REWIND)
    enemy_actual = extract(enemy_env).world
    assert enemy_actual == enemy_expected
    assert enemy_actual.enemies[0].position == inferred


def test_rewind_boundary_phases_match_native_on_default_tier_five_row():
    spec = generate(705, 5)
    assert spec is not None
    env = Env([build_level(spec)])
    for step in spec["solution"][:31]:
        env.perform(*step)

    layout = extract(env)
    assert layout.world.player == (49, 49)
    assert layout.world.stage == 2
    assert layout.world.doors == (True, True, True)
    assert len(layout.world.history) == 13
    expected, _, _ = transition(
        layout, layout.world, names.ACTION_REWIND,
    )

    env.perform(names.ACTION_REWIND)
    actual = extract(env)
    assert actual.exact
    assert actual.world.doors == (False, True, True)
    assert actual.world == expected


def test_extra_rewinds_match_full_native_world_on_stored_nonterminal_prefixes():
    for seed, difficulty, expected_prefixes in (
        (705, 5, 35),
        (706, 6, 33),
        (707, 7, 26),
    ):
        spec = generate(seed, difficulty)
        assert spec is not None
        env = Env([build_level(spec)])
        checked = 0
        for prefix, step in enumerate(spec["solution"], 1):
            env.perform(*step)
            if env.state != GameState.NOT_FINISHED:
                continue
            layout = extract(env)
            if not layout.world.history:
                continue

            expected, _, _ = transition(
                layout, layout.world, names.ACTION_REWIND,
            )
            branch = env.clone()
            branch.perform(names.ACTION_REWIND)
            actual = extract(branch)
            assert actual.exact, (seed, difficulty, prefix, actual.unsupported)
            assert actual.world == expected, (seed, difficulty, prefix)
            checked += 1
        assert checked == expected_prefixes


def test_validator_fails_closed_on_typed_evidence_and_wrong_native_route():
    spec = generate(8_008, 1, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    for path, false_value in (
        (("engine_win",), 1),
        (("context_index",), False),
        (("levels_completed",), 99),
        (("native_steps_used",), 999),
        (("search_expanded",), -20),
        (("proof", "engine_win"), 1),
        (("solution_length",), float(spec["solution_length"])),
        (("witness_actions",), float(spec["witness_actions"])),
        (("minimum_steps_left",), float(spec["minimum_steps_left"])),
        (("native_action_budget",), float(spec["native_action_budget"])),
    ):
        changed = deepcopy(spec)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = false_value
        assert validate_full_standard(changed, entry), path
    for malformed_route in (1, True, 1.5):
        changed = deepcopy(spec)
        changed["solution"] = changed["context_solution"] = malformed_route
        assert validate_full_standard(changed, entry)
    for path in (
        ("context_solution", 0, 0),
        ("solution_mechanics", "actions"),
        ("proof", "structure", "free_cells"),
    ):
        changed = deepcopy(spec)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = float(target[path[-1]])
        assert validate_full_standard(changed, entry), path
    excessive = deepcopy(spec)
    excessive["search_expanded"] = excessive["search_limit"] + 1
    excessive["generation_search_expanded"] = excessive["search_expanded"]
    excessive["proof"]["search_expanded"] = excessive["search_expanded"]
    excessive["proof"]["generation_search_expanded"] = excessive[
        "generation_search_expanded"
    ]
    assert validate_full_standard(excessive, entry)
    wrong = deepcopy(spec)
    wrong["solution"][0][0] = (
        names.ACTION_LEFT if wrong["solution"][0][0] != names.ACTION_LEFT
        else names.ACTION_RIGHT
    )
    wrong["context_solution"] = deepcopy(wrong["solution"])
    assert validate_full_standard(wrong, entry)


def test_canonical_circuit_ids_survive_relabeling_in_enemy_evidence():
    spec = generate(706, 6, split="train")
    assert spec is not None
    relabelled = deepcopy(spec)
    mapping = {0: 3, 3: 0, 1: 1, 2: 2, 4: 4}
    for key in ("switches", "doors", "teleports"):
        for value in relabelled[key]:
            value["circuit"] = mapping[value["circuit"]]
    assert gameplay_identity(relabelled) == spec["gameplay_sha256"]
    assert validate_full_standard(
        relabelled, FULL_STANDARD_CONTRACT["curriculum"][5],
    ) == []


def test_tier_six_records_two_native_enemy_gate_interventions():
    spec = generate(707, 6, split="train")
    assert spec is not None
    dependencies = spec["solution_mechanics"]["enemy_gate_dependencies"]
    assert len(dependencies) == 2
    assert len({value["ordinary_circuit"] for value in dependencies}) == 2
    assert len({value["toggle_circuit"] for value in dependencies}) == 2
    assert all(value["baseline_target_triggered"] for value in dependencies)
    assert all(not value["disabled_switch_native_win"] for value in dependencies)
    assert all(not value["disabled_switch_target_triggered"] for value in dependencies)


def test_rejections_are_reported_and_work_is_cumulative():
    draft = _draft(12_345, 3, 0)
    split = geometry_identity(draft)[1]
    rows = []
    assert generate(12_345, 3, attempts=4, node_limit=1, split=split,
                    record_rejection=rows.append) is None
    assert len(rows) == 4
    assert sum(generate.last_rejections.values()) == 4
    assert generate.last_work["attempts"] == 4
    assert generate.last_work["expanded"] == sum(
        row.get("search_expanded", 0) for row in rows
    )
    assert generate.last_work["generated"] == sum(
        row.get("search_generated", 0) for row in rows
    )
    assert all(row["generator_version"] == 4 for row in rows)


def test_circuit_wire_colors_match_native_device_kinds():
    spec = generate(15_007, 7, split="train")
    assert spec is not None
    level = build_level(spec)
    expected = {"ordinary": 8, "toggle": names.TOGGLE_COLOR, "teleport": 15}
    kinds = {value["circuit"]: value["kind"] for value in spec["switches"]}
    circuits = [sprite for sprite in level.get_sprites_by_tag(names.TAG_CIRCUIT)]
    assert len(circuits) == len(kinds)
    for circuit, sprite in zip(sorted(kinds), circuits):
        visible = {int(pixel) for row in sprite.pixels for pixel in row if pixel >= 0}
        assert visible == {expected[kinds[circuit]]}


def test_strict_parameter_types_and_incomplete_build_game_rejection():
    for bad in (True, 1.5, "1"):
        try:
            generate(bad, 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid seed {bad!r}")
    spec = generate(12, 1)
    try:
        build_game([spec])
    except ValueError:
        pass
    else:
        raise AssertionError("build_game accepted a shortened curriculum")
