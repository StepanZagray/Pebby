"""Full-tier FT09 generation, native semantics, and exact teacher coverage."""

import copy
import json

import pytest
from arcengine import GameState

from pebby.games.ft09.bank import main
from pebby.games.ft09.env import Env, official_levels, replay
from pebby.games.ft09.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    _official_identities,
    build_game,
    build_level,
    generate,
    generate_game,
    replays_to_completion,
    validate_full_standard,
    visual_errors,
)
from pebby.games.ft09.generation_quality import gameplay_hash, geometry_hash
from pebby.games.ft09.layout import extract
from pebby.games.ft09.plan import search, solve, verify
from pebby.games.ft09.reference_profiles import PROFILES, REFERENCES, profile_errors


def cycle_spec(colours=2, budget=2):
    return {
        "grid": 32, "palette": [8, 9, 11][:colours], "budget": budget,
        "stencil": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "cells": [{"x": 2, "y": 2}],
        "constraints": [{"x": 6, "y": 2, "colour": 8,
                         "mask": [[1, 1, 1], [0, 0, 1], [1, 1, 1]]}],
    }


def generated_curriculum(split="train", base=1000):
    rows = [generate(base + difficulty, difficulty, attempts=100, split=split)
            for difficulty in DIFFICULTIES]
    assert all(rows)
    return rows


def test_all_six_official_teachers_replay_sequentially_in_live_native_contexts():
    assert len(official_levels()) == 6
    assert DIFFICULTIES == tuple(range(1, len(official_levels()) + 1))
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert [row["difficulty"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(6))
    env = Env()
    for difficulty in DIFFICULTIES:
        assert env.level_index == difficulty - 1
        layout = extract(env)
        result = search(layout, limit=PROFILES[difficulty]["search_work"])
        assert result.actions is not None and not result.truncated
        assert len(result) == REFERENCES[difficulty]["actions"]
        assert verify(layout, result.actions)
        assert all(action[0] in env.available_actions for action in result.actions)
        assert replay(env, result.actions, expect_level=difficulty - 1)[0]
        assert env.levels_completed == difficulty
    assert env.state == GameState.WIN


def test_generated_full_contract_all_tiers_json_round_trip_and_context_replay():
    for difficulty, spec in enumerate(generated_curriculum(), 1):
        restored = json.loads(json.dumps(spec))
        assert profile_errors(restored) == []
        assert validate_full_standard(
            restored, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        ) == []
        assert restored["context_index"] == difficulty - 1
        assert restored["solution_length"] == len(restored["solution"])
        assert restored["geometry_d4_sha256"] == restored["geometry_sha256"]
        assert replays_to_completion(restored)
        assert restored["omitted_mechanics"] == []


def test_generated_full_episode_advances_all_six_tiers_without_forced_transitions():
    specs = generated_curriculum(base=2000)
    env = Env([build_level(spec) for spec in specs])
    for completed, spec in enumerate(specs):
        assert env.level_index == completed == spec["context_index"]
        result = search(env, limit=PROFILES[completed + 1]["search_work"])
        assert result.actions is not None and not result.truncated
        assert replay(env, result.actions, expect_level=completed)[0]
        assert env.levels_completed == completed + 1
    assert env.state == GameState.WIN


def test_generated_mechanics_progress_through_all_official_tiers():
    rows = generated_curriculum(base=3000)
    for spec in rows[:3]:
        assert len(spec["palette"]) == 2
        assert spec["special_cell_count"] == 0
    assert rows[0]["tutorial_hint"]
    assert all(not row["tutorial_hint"] for row in rows[1:])
    three = rows[3]
    assert len(three["palette"]) == 3
    assert three["solution_mechanics"]["third_colour_actions"] > 0
    assert three["solution_mechanics"]["repeat_actions"] > 0
    mixed = rows[4]
    assert mixed["ordinary_cell_count"] and mixed["special_cell_count"] == 3
    assert (mixed["solution_mechanics"]["distinct_special_cells_clicked"]
            == mixed["special_cell_count"])
    directional = rows[5]
    assert directional["special_cell_count"] == directional["cell_count"]
    assert directional["ordinary_cell_count"] == 0
    assert directional["solution_mechanics"]["distinct_coupled_special_cells_clicked"] >= 5


def test_generated_palette_is_vivid_and_rendered_in_order_for_tiers_two_through_six():
    rows = generate_game(888, split="train")
    assert rows is not None
    vivid_official_colours = {8, 9, 11, 12, 14, 15}
    for difficulty, spec in enumerate(rows, 1):
        assert len(set(spec["palette"])) == len(spec["palette"])
        assert set(spec["palette"]) <= vivid_official_colours
        if difficulty == 1:
            continue
        frame = Env([build_level(spec)]).render()
        for index, colour in enumerate(spec["palette"]):
            assert {
                frame[y][x]
                for y in range(index * 4, index * 4 + 4)
                for x in range(60, 64)
            } == {colour}


def test_tier_one_renders_three_truthful_static_examples_around_the_active_board():
    spec = generate(889, 1, attempts=100, split="train")
    assert spec is not None
    assert min(row["x"] for key in ("cells", "constraints") for row in spec[key]) >= 17
    assert min(row["y"] for key in ("cells", "constraints") for row in spec[key]) >= 17

    env = Env([build_level(spec)])
    frame = env.render()
    palette = set(spec["palette"])
    source = tuple(tuple(0 if value == 0 else 2 for value in row)
                   for row in spec["constraints"][0]["mask"])

    def rotate(mask):
        return tuple(tuple(row) for row in zip(*mask[::-1]))

    d4 = set()
    transformed = source
    for _ in range(4):
        d4.add(transformed)
        d4.add(tuple(tuple(reversed(row)) for row in transformed))
        transformed = rotate(transformed)

    def pixel(world_x, world_y):
        return frame[world_y * 2][world_x * 2]

    before = extract(env).initial
    before_budget = env.budget()
    for panel_x, panel_y in ((1, 1), (18, 1), (1, 18)):
        assert {pixel(panel_x + offset, panel_y) for offset in range(13)} == {3}
        assert {pixel(panel_x + offset, panel_y + 12) for offset in range(13)} == {3}
        assert {pixel(panel_x, panel_y + offset) for offset in range(13)} == {3}
        assert {pixel(panel_x + 12, panel_y + offset) for offset in range(13)} == {3}

        rule = tuple(
            tuple(pixel(panel_x + 5 + col, panel_y + 5 + row) for col in range(3))
            for row in range(3)
        )
        centre = rule[1][1]
        assert centre == spec["constraints"][0]["colour"]
        observed_mask = tuple(
            tuple(0 if (row, col) == (1, 1) or rule[row][col] == 0 else 2
                  for col in range(3))
            for row in range(3)
        )
        assert observed_mask in d4
        for row in range(3):
            for col in range(3):
                if (row, col) == (1, 1):
                    continue
                cell_colour = pixel(panel_x + 2 + 4 * col, panel_y + 2 + 4 * row)
                assert cell_colour in palette
                assert (cell_colour == centre) is (rule[row][col] == 0)

        env.click(2 * (panel_x + 2), 2 * (panel_y + 2))
        assert env.budget() == before_budget
        assert extract(env).initial == before


def test_visual_validator_rejects_low_contrast_clipping_and_cue_overlap():
    rows = generate_game(890, split="validation")
    assert rows is not None
    for spec in rows:
        assert visual_errors(spec) == []
        for sprite in build_level(spec).get_sprites():
            assert 0 <= sprite.x < 32 and 0 <= sprite.y < 32
            assert sprite.x + sprite.width <= 32
            assert sprite.y + sprite.height <= 32

    low_contrast = copy.deepcopy(rows[1])
    low_contrast["palette"][0] = 1
    assert any("palette" in error for error in visual_errors(low_contrast))

    clipped = copy.deepcopy(rows[1])
    clipped["cells"][0].update(x=31, y=31)
    assert any("clipped" in error for error in visual_errors(clipped))

    overlapping = copy.deepcopy(rows[1])
    overlapping["cells"][0].update(x=28, y=0)
    assert any("overlaps the palette legend" in error for error in visual_errors(overlapping))


def test_tutorial_flash_and_both_native_no_op_targets_are_budget_free():
    spec = generate(4001, 1, attempts=100)
    env = Env([build_level(spec)])
    before_layout = extract(env)
    before_budget = env.budget()
    rule_no_op, empty_no_op = spec["no_op_actions"]
    rule_observation = env.perform(*rule_no_op)
    assert len(rule_observation.frames) == 1
    assert env.budget() == before_budget and extract(env).initial == before_layout.initial
    empty_observation = env.perform(*empty_no_op)
    assert len(empty_observation.frames) > 1  # one click resolves the complete flash animation
    assert env.budget() == before_budget and extract(env).initial == before_layout.initial
    assert replay(env, [tuple(action) for action in spec["solution"]])[0]


def test_live_prefix_reset_recovers_exact_teacher_and_budget():
    spec = generate(5006, 6, attempts=100)
    env = Env([build_level(spec)])
    for action in spec["solution"][:3]:
        env.perform(*action)
    assert env.budget()[0] < spec["budget"]
    env.perform(0)
    assert env.budget() == (spec["budget"], spec["budget"])
    result = search(env, limit=PROFILES[6]["search_work"])
    assert result.actions is not None and not result.truncated
    assert replay(env, result.actions)[0]


def test_split_is_explicit_deterministic_and_identity_partitioned():
    rows = {}
    for split in ("train", "validation", "test"):
        first = generate(6004, 4, attempts=100, split=split)
        second = generate(6004, 4, attempts=100, split=split)
        assert first == second
        assert first["split"] == first["geometry_partition"] == split
        rows[split] = first
    assert len({row["gameplay_sha256"] for row in rows.values()}) == 3
    with pytest.raises(ValueError, match="split"):
        generate(1, 1, split="dev")


def test_generate_game_has_stable_child_seeds_and_build_game_fails_closed():
    rows = generate_game(12345, split="validation", attempts=100)
    assert rows is not None and len(rows) == 6
    assert rows == generate_game(12345, split="validation", attempts=100)
    assert len({row["seed"] for row in rows}) == 6
    levels = build_game(rows)
    assert len(levels) == 6
    env = Env(levels)
    for index in range(6):
        result = search(env, limit=PROFILES[index + 1]["search_work"])
        assert result.actions is not None and not result.truncated
        assert replay(env, result.actions, expect_level=index)[0]
    assert env.state == GameState.WIN

    with pytest.raises(ValueError, match="exactly"):
        build_game(rows[:-1])
    with pytest.raises(ValueError, match="in order"):
        build_game([rows[1], rows[0], *rows[2:]])
    mismatched = copy.deepcopy(rows)
    mismatched[-1]["split"] = "train"
    with pytest.raises(ValueError, match="share one"):
        build_game(mismatched)
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["gameplay_sha256"] = duplicate[0]["gameplay_sha256"]
    with pytest.raises(ValueError, match="distinct"):
        build_game(duplicate)
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["geometry_sha256"] = duplicate[0]["geometry_sha256"]
    with pytest.raises(ValueError, match="distinct"):
        build_game(duplicate)

    with pytest.raises(ValueError, match="distinct"):
        generate_game(12345, split="validation", difficulties=(1, 1))
    with pytest.raises(ValueError, match="in increasing order"):
        generate_game(12345, split="validation", difficulties=(2, 1))


def test_validator_recomputes_route_mechanics_structure_identity_and_native_win(monkeypatch):
    spec = generate(12346, 5, attempts=100, split="test")
    curriculum = FULL_STANDARD_CONTRACT["curriculum"][4]
    assert validate_full_standard(spec, curriculum) == []

    bad = copy.deepcopy(spec)
    bad["solution"][0] = bad["no_op_actions"][0]
    assert any("route/native replay" in error or "solution" in error
               for error in validate_full_standard(bad, curriculum))
    bad = copy.deepcopy(spec)
    bad["solution_mechanics"]["special_clicks"] += 1
    assert any("mechanics" in error for error in validate_full_standard(bad, curriculum))
    bad = copy.deepcopy(spec)
    bad["active_rule_edges"] += 1
    assert any("recomputed structure" in error for error in validate_full_standard(bad, curriculum))
    bad = copy.deepcopy(spec)
    bad["constraints"][0]["x"] += 4
    assert validate_full_standard(bad, curriculum)
    bad = copy.deepcopy(spec)
    bad["geometry_d4_sha256"] = "0" * 64
    assert any("D4 geometry identity" in error
               for error in validate_full_standard(bad, curriculum))
    bad = copy.deepcopy(spec)
    bad["proof"]["engine_win"] = "yes"
    assert any("proof engine_win" in error for error in validate_full_standard(bad, curriculum))
    assert validate_full_standard({"difficulty": 5, "cells": "malformed"}, curriculum)
    assert validate_full_standard({"difficulty": []}, curriculum)

    # Admission must reject official content from recomputed identities even
    # when the row did not pass through this generator's candidate exclusions.
    monkeypatch.setattr(
        "pebby.games.ft09.generate._official_identities",
        lambda: (frozenset({geometry_hash(spec)}), frozenset({gameplay_hash(spec)})),
    )
    official_errors = validate_full_standard(spec, curriculum)
    assert any("geometry matches an official level" in error for error in official_errors)
    assert any("gameplay matches an official level" in error for error in official_errors)


def test_canonical_identities_ignore_translation_reflection_and_palette_values():
    spec = generate(7005, 5, attempts=100)
    transformed = copy.deepcopy(spec)
    transformed["palette"] = [1, 3]
    old_to_new = dict(zip(spec["palette"], transformed["palette"]))
    for rule in transformed["constraints"]:
        rule["colour"] = old_to_new[rule["colour"]]
    for cell in transformed["cells"]:
        cell["x"] = 100 - cell["x"]
        if "stencil" in cell:
            cell["stencil"] = [list(reversed(row)) for row in cell["stencil"]]
    for rule in transformed["constraints"]:
        rule["x"] = 100 - rule["x"]
        rule["mask"] = [list(reversed(row)) for row in rule["mask"]]
    assert geometry_hash(transformed) == spec["geometry_sha256"]
    assert gameplay_hash(transformed) == spec["gameplay_sha256"]


def test_generated_levels_are_not_official_copies_and_sample_is_duplicate_free():
    official_geometry, official_gameplay = _official_identities()
    rows = generated_curriculum(base=8000)
    assert not ({row["geometry_sha256"] for row in rows} & official_geometry)
    assert not ({row["gameplay_sha256"] for row in rows} & official_gameplay)
    assert len({row["gameplay_sha256"] for row in rows}) == len(rows)


def test_generation_bounds_and_rejection_diagnostics_are_explicit():
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            generate(bad, 1)
    for bad in (0, 7, True):
        with pytest.raises(ValueError, match="difficulty"):
            generate(0, bad)
    with pytest.raises(ValueError, match="attempts"):
        generate(0, 1, attempts=0)
    with pytest.raises(ValueError, match="limit"):
        generate(0, 1, limit=0)
    rejected = []
    spec = generate(9004, 4, attempts=100, split="validation", record_rejection=rejected.append)
    assert spec is not None
    assert spec["generation_attempt"] == len(rejected) + 1
    assert all(row["seed"] == 9004 and row["difficulty"] == 4 and row["reason"] for row in rejected)


def test_initially_satisfied_board_needs_nonempty_cycle():
    env = Env([build_level(cycle_spec())])
    assert env.solved()
    actions = solve(env)
    assert actions is not None and len(actions) == 2
    assert replay(env, actions)[0]


def test_cycle_cannot_overdraw_budget_and_truncation_is_explicit():
    env = Env([build_level(cycle_spec(budget=1))])
    result = search(env)
    assert result.actions is None and not result.truncated
    assert solve(env, limit=0) is None and solve.truncated


def test_midlevel_solver_uses_remaining_budget():
    env = Env([build_level(cycle_spec(colours=3, budget=2))])
    env.perform(6, 6, 6)
    assert extract(env).budget == 1
    assert solve(env) is None and not solve.truncated


def test_invalid_actions_rejected():
    env = Env()
    for action in ((1, None, None), (6, None, None), (6, -1, 5), (6, 64, 5)):
        with pytest.raises(ValueError):
            env.perform(*action)


def test_bank_cli_round_trip_full_split(tmp_path):
    out = tmp_path / "bank.jsonl"
    assert main(["--levels", "3", "--seed", "0", "--difficulty", "6",
                 "--split", "test", "--out", str(out)]) == 0
    specs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(specs) == 3
    assert len({spec["geometry_sha256"] for spec in specs}) == 3
    assert len({spec["gameplay_sha256"] for spec in specs}) == 3
    assert all(spec["split"] == "test" and replays_to_completion(spec) for spec in specs)
