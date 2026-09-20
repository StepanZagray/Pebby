"""Full R11L teacher, generator, contract, and native replay tests."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from arcengine import GameState

from pebby import multigame as M
from pebby.games.r11l import names
from pebby.games.r11l import generate as generate_module
from pebby.games.r11l.env import Env, replay
from pebby.games.r11l.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    REFERENCE_PROFILES,
    build_game,
    build_level,
    canonical_identity,
    generate,
    generate_game,
    geometry_partition,
    official_geometry_hashes,
    profile_errors,
    validate_full_standard,
)
from pebby.games.r11l.layout import extract
from pebby.games.r11l.plan import (
    _hazard_free,
    _overlaps,
    _selection_click,
    search,
    solve,
)


@pytest.fixture(scope="module")
def generated_rows():
    rows = [generate(9100 + difficulty, difficulty, split="train") for difficulty in DIFFICULTIES]
    assert all(row is not None for row in rows)
    return rows


def test_official_six_tier_teacher_replays_sequentially_in_native_engine():
    env = Env()
    frame = env.reset()
    assert len(frame) == len(frame[0]) == 64
    measured = []
    for context_index, difficulty in enumerate(DIFFICULTIES):
        assert env.level_index == context_index
        result = search(env)
        assert result.solved and result.actions
        assert result.exact and not result.truncated and not result.unsupported
        assert all(action[0] == names.ACTION_CLICK for action in result.actions)
        measured.append(len(result.actions))
        assert replay(env, result.actions)
        assert env.levels_completed == difficulty
    assert env.state == GameState.WIN
    assert measured == [REFERENCE_PROFILES[d]["reference_actions"] for d in DIFFICULTIES]


def test_official_pickup_teacher_mutates_core_pixels_and_removes_only_needed_pickups():
    for level_index, expected_absorbed, expected_remaining in ((4, 4, 0), (5, 6, 3)):
        env = Env()
        env.reset()
        env.set_level(level_index)
        result = search(env)
        assert result.solved and not result.unsupported
        removed = 0
        observed_colour_mutation = False
        for action in result.actions:
            before_pickups = len(env.pickups())
            before_colours = {
                name: {int(value) for value in data[names.KEY_CORE].pixels.flat if value > 0}
                for name, data in env.groups().items()
                if data[names.KEY_CORE] is not None
                and data[names.KEY_CORE].name.startswith(names.PREFIX_ABSORBING_CORE)
            }
            outcome = env.perform(*action)
            if env.level_index == level_index:
                removed += before_pickups - len(env.pickups())
                after_colours = {
                    name: {int(value) for value in data[names.KEY_CORE].pixels.flat if value > 0}
                    for name, data in env.groups().items()
                    if data[names.KEY_CORE] is not None
                    and data[names.KEY_CORE].name.startswith(names.PREFIX_ABSORBING_CORE)
                }
                observed_colour_mutation |= before_colours != after_colours
            assert outcome.state != GameState.GAME_OVER
        assert observed_colour_mutation
        assert removed == expected_absorbed
        if level_index == 5:
            assert len(env.pickups()) == expected_remaining


def test_native_hazard_rollback_is_recoverable_by_live_teacher():
    env = Env()
    env.reset()
    env.set_level(1)
    layout = extract(env)
    dangerous = None
    for index, fragment in enumerate(layout.fragments):
        for y in range(64):
            for x in range(64):
                if any(
                    px <= x < px + other.width and py <= y < py + other.height
                    for other, (px, py) in zip(layout.fragments, layout.positions)
                ):
                    continue
                destination = names.click_to_position(x, y)
                if _overlaps(fragment.mask, destination, layout.walls):
                    continue
                moved = list(layout.positions)
                moved[index] = destination
                if not _hazard_free(layout, tuple(moved)):
                    dangerous = index, x, y
                    break
            if dangerous:
                break
        if dangerous:
            break
    assert dangerous is not None
    index, x, y = dangerous
    selection = None if layout.selected == index else _selection_click(layout, layout.positions, index)
    if selection is not None:
        env.perform(names.ACTION_CLICK, *selection)
    losing = env.clone()
    for hit_number in range(1, names.HAZARD_LIMIT + 1):
        outcome = losing.perform(names.ACTION_CLICK, x, y)
        assert losing.hazards_hit() == hit_number
        assert (outcome.state == GameState.GAME_OVER) is (hit_number == names.HAZARD_LIMIT)
    before = [(fragment.x, fragment.y) for fragment in env.fragments()]
    env.perform(names.ACTION_CLICK, x, y)
    assert env.hazards_hit() == 1
    assert [(fragment.x, fragment.y) for fragment in env.fragments()] == before
    recovery = solve(env)
    assert recovery and replay(env, recovery)
    assert env.levels_completed == 1


def test_native_wall_block_and_sixtieth_action_loss_semantics():
    env = Env()
    env.reset()
    layout = extract(env)
    index = layout.selected
    fragment = layout.fragments[index]
    blocked = None
    for y in range(64):
        for x in range(64):
            if any(
                px <= x < px + other.width and py <= y < py + other.height
                for other, (px, py) in zip(layout.fragments, layout.positions)
            ):
                continue
            if _overlaps(fragment.mask, names.click_to_position(x, y), layout.walls):
                blocked = x, y
                break
        if blocked:
            break
    assert blocked is not None
    before = [(item.x, item.y) for item in env.fragments()]
    assert env.perform(names.ACTION_CLICK, *blocked).state != GameState.GAME_OVER
    assert [(item.x, item.y) for item in env.fragments()] == before

    budget_env = Env()
    budget_env.reset()
    selected = budget_env.selected()
    click = (selected.x + selected.width // 2, selected.y + selected.height // 2)
    for action_number in range(1, names.MAX_ACTIONS + 1):
        outcome = budget_env.perform(names.ACTION_CLICK, *click)
        assert (outcome.state == GameState.GAME_OVER) is (action_number == names.MAX_ACTIONS)


def test_all_generated_tiers_are_reference_profiled_json_round_trippable_and_validated(generated_rows):
    assert DIFFICULTIES == tuple(range(1, 7))
    identities = {row["geometry_d4_sha256"] for row in generated_rows}
    assert len(identities) == 6
    assert identities.isdisjoint(official_geometry_hashes())
    for difficulty, row in zip(DIFFICULTIES, generated_rows):
        restored = json.loads(json.dumps(row))
        assert profile_errors(restored) == []
        assert validate_full_standard(
            restored, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        ) == []
        assert restored["difficulty"] == difficulty
        assert restored["context_index"] == difficulty - 1
        assert restored["split"] == restored["geometry_split"] == "train"
        assert restored["engine_verified"]
        assert restored["solution_mechanics"]["replayed_actions"] == len(restored["solution"])


def test_full_generated_game_replays_without_forced_transitions(generated_rows):
    env = Env(build_game(generated_rows))
    env.reset()
    for context_index, row in enumerate(generated_rows):
        assert env.level_index == context_index
        assert row["native_replay"]["context_index"] == context_index
        assert replay(env, [tuple(action) for action in row["solution"]])
        assert env.levels_completed == context_index + 1
    assert env.state == GameState.WIN


def test_generate_game_uses_stable_child_seeds_and_supports_explicit_reduced_sequences():
    full_a = generate_game(812, split="validation")
    full_b = generate_game(812, split="validation")
    assert full_a == full_b
    assert len(full_a) == 6
    assert [row["difficulty"] for row in full_a] == list(DIFFICULTIES)
    assert len({row["game_child_seed"] for row in full_a}) == 6
    reduced = generate_game(812, split="validation", difficulties=(1, 3, 6))
    assert [row["difficulty"] for row in reduced] == [1, 3, 6]
    with pytest.raises(ValueError, match="exactly six"):
        build_game(reduced)


def test_build_game_rejects_shifted_duplicate_and_split_mismatched_sequences(generated_rows):
    with pytest.raises(ValueError, match="ordered"):
        build_game(generated_rows[1:] + generated_rows[:1])
    duplicate = copy.deepcopy(generated_rows)
    duplicate[1]["gameplay_sha256"] = duplicate[0]["gameplay_sha256"]
    with pytest.raises(ValueError, match="unique"):
        build_game(duplicate)
    mismatched = copy.deepcopy(generated_rows)
    mismatched[-1]["split"] = "test"
    with pytest.raises(ValueError, match="common split"):
        build_game(mismatched)


def test_build_game_fails_cleanly_on_malformed_input_and_invokes_full_replay(
    generated_rows, monkeypatch,
):
    assert validate_full_standard(None, FULL_STANDARD_CONTRACT["curriculum"][0])
    assert validate_full_standard({}, None)
    bad_curriculum = dict(FULL_STANDARD_CONTRACT["curriculum"][0])
    bad_curriculum["difficulty"] = True
    assert any(
        "curriculum" in error
        for error in validate_full_standard(generated_rows[0], bad_curriculum)
    )
    with pytest.raises(ValueError, match="sequence"):
        build_game(None)
    with pytest.raises(ValueError, match="mapping"):
        build_game([None] * len(DIFFICULTIES))

    replayed = []

    def replay_full_game(specs, levels):
        replayed.append((specs, levels))

    monkeypatch.setattr(generate_module, "_replay_full_game", replay_full_game)
    levels = build_game(generated_rows)
    assert len(levels) == len(DIFFICULTIES)
    assert len(replayed) == 1


def test_validator_replays_route_and_rejects_tampered_proof_mechanics_and_geometry(generated_rows):
    entry = FULL_STANDARD_CONTRACT["curriculum"][4]
    wrong_route = copy.deepcopy(generated_rows[4])
    wrong_route["solution"][-1][1] = (wrong_route["solution"][-1][1] + 17) % 64
    assert validate_full_standard(wrong_route, entry)

    wrong_mechanics = copy.deepcopy(generated_rows[4])
    wrong_mechanics["solution_mechanics"]["pickups_absorbed"] += 1
    assert any("mechanics" in error for error in validate_full_standard(wrong_mechanics, entry))

    wrong_geometry = copy.deepcopy(generated_rows[4])
    wrong_geometry["absorbers"][0]["fragments"][0][0] += 1
    assert validate_full_standard(wrong_geometry, entry)


def test_explicit_splits_are_geometry_partitions_not_rng_labels():
    rows = {split: generate(7301, 1, split=split) for split in ("train", "validation", "test")}
    assert all(rows.values())
    assert len({row["geometry_d4_sha256"] for row in rows.values()}) == 3
    for split, row in rows.items():
        assert geometry_partition(row)[1] == split
        assert generate(7301, 1, split=split) == row
    with pytest.raises(TypeError):
        generate(1, 1)


def test_canonical_identity_normalizes_rotation_translation_and_ignores_colours(generated_rows):
    original = generated_rows[3]
    changed = copy.deepcopy(original)

    def rotate_point(point):
        x, y = point
        return [63 - y, x]

    for group in changed["groups"]:
        group["target"] = rotate_point(group["target"])
        group["fragments"] = [rotate_point(point) for point in group["fragments"]]
    for decoy in changed["decoys"]:
        decoy["position"] = rotate_point(decoy["position"])
        decoy["colour"] = (decoy["colour"] + 3) % 16
    for key in ("walls", "hazards"):
        for rect in changed[key]:
            x, y, width, height = (rect[name] for name in ("x", "y", "width", "height"))
            rect.update(x=64 - (y + height), y=x, width=height, height=width)
    assert canonical_identity(changed) == canonical_identity(original)


def test_canonical_gameplay_preserves_group_and_native_pickup_semantics(generated_rows):
    ordinary = copy.deepcopy(generated_rows[3])
    regrouped = copy.deepcopy(ordinary)
    regrouped["groups"][0]["fragments"][0], regrouped["groups"][1]["fragments"][0] = (
        regrouped["groups"][1]["fragments"][0],
        regrouped["groups"][0]["fragments"][0],
    )
    assert canonical_identity(regrouped)[0] == canonical_identity(ordinary)[0]
    assert canonical_identity(regrouped)[1] != canonical_identity(ordinary)[1]

    absorption = copy.deepcopy(generated_rows[5])
    reassigned = copy.deepcopy(absorption)
    useful = [pickup for pickup in reassigned["pickups"] if pickup["role"] == "useful"]
    useful[0]["prototype"], useful[-1]["prototype"] = (
        useful[-1]["prototype"], useful[0]["prototype"],
    )
    assert canonical_identity(reassigned)[0] == canonical_identity(absorption)[0]
    assert canonical_identity(reassigned)[1] != canonical_identity(absorption)[1]

    private_metadata = copy.deepcopy(absorption)
    private_metadata["seed"] += 1
    private_metadata["solution"] = [[names.ACTION_CLICK, 0, 0]]
    private_metadata["engine_verified"] = False
    assert canonical_identity(private_metadata) == canonical_identity(absorption)


def test_live_prefix_absorption_teacher_recovers_from_partial_native_state(generated_rows):
    levels = build_game(generated_rows)
    env = Env(levels)
    env.reset()
    env.set_level(4)
    prefix = [tuple(action) for action in generated_rows[4]["solution"][:4]]
    for action in prefix:
        assert env.perform(*action).state != GameState.GAME_OVER
    assert len(env.pickups()) < REFERENCE_PROFILES[5]["pickups"]
    suffix = solve(env)
    assert suffix and replay(env, suffix)
    assert env.levels_completed == 1 and env.level_index == 5


def test_truncation_is_inconclusive_and_contract_shape_is_exact():
    env = Env()
    env.reset()
    result = search(env, limit=0)
    assert result.truncated and not result.solved and not result.exact
    contract = FULL_STANDARD_CONTRACT
    assert contract["format"] == "pebby-full-generator-contract-v1"
    assert contract["source_id"] == M.source_for("r11l").source_id
    assert contract["status"] == "ready"
    assert [row["difficulty"] for row in contract["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in contract["curriculum"]] == list(range(6))
    assert all(type(row["search_work"]) is int and 1 <= row["search_work"] <= 32_000_000
               for row in contract["curriculum"])


def test_bank_cli_requires_split_and_writes_reconstructable_jsonl(tmp_path):
    output = tmp_path / "r11l.jsonl"
    subprocess.run([
        sys.executable, "-m", "pebby.games.r11l.bank", "--levels", "2",
        "--seed", "30", "--difficulty", "2", "--split", "test",
        "--out", str(output),
    ], check=True, timeout=60)
    specs = [json.loads(line) for line in Path(output).read_text().splitlines() if line]
    assert len(specs) == 2
    assert all(spec["split"] == "test" for spec in specs)
    assert all(build_level(spec).grid_size == (64, 64) for spec in specs)
