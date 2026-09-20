"""Full eight-tier RE86 generation, teacher, and native replay tests."""

import json

import numpy as np
import pytest
from arcengine import GameState, Sprite

from pebby.games.re86 import names
from pebby.games.re86.bank import main as bank_main
from pebby.games.re86.env import Env, official_levels, replay
from pebby.games.re86.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    DYES,
    _action_transforms,
    _level_identity,
    _official_geometry_hashes,
    _sprite_identity,
    _transform_rect,
    _transforms,
    build_game,
    build_level,
    env_for,
    generate,
    generate_game,
    geometry_identity,
    gameplay_identity,
    semantic_partition,
    solution_semantic_identity,
    validate_full_standard,
)
from pebby.games.re86.generation_quality import (
    audit_quality,
    semantic_route_signature,
)
from pebby.games.re86.layout import extract
from pebby.games.re86.plan import search, solution_mechanics, solve
from pebby.games.re86.reference_profiles import PROFILES, SOURCE_SHA256


def _assert_actions(env, actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        assert x is None and y is None


def _recertify_route(spec, actions):
    """Update only private route evidence, never public puzzle identity."""
    spec["solution"] = [list(action) for action in actions]
    spec["context_solution"] = [list(action) for action in actions]
    spec["solution_length"] = len(actions)
    spec["budget_remaining"] = spec["native_budget"] - len(actions)
    spec["solution_mechanics"] = solution_mechanics(
        env_for(spec, spec["training_context_index"]), actions
    )
    spec["solution_semantic_sha256"] = solution_semantic_identity(spec)
    spec["proof"]["action_count"] = len(actions)
    spec["proof"]["search_work"] = len(actions)
    spec["proof"]["solution_semantic_sha256"] = spec[
        "solution_semantic_sha256"
    ]


def test_all_eight_official_tiers_have_bounded_native_positive_witnesses():
    levels = official_levels()
    assert len(levels) == 8
    routes = []
    facts = []
    for difficulty, level in zip(DIFFICULTIES, levels):
        env = Env([level])
        assert extract(env).exact
        result = search(env, node_limit=PROFILES[difficulty]["search_work"])
        assert result.actions is not None, result
        assert result.exact and not result.truncated and not result.unsupported
        assert result.optimal is False
        _assert_actions(env, result.actions)
        facts.append(solution_mechanics(Env([level]), result.actions))
        assert replay(env, result.actions)
        routes.append(result.actions)

    # Mechanics are witnessed by the teacher, not inferred from installed tags.
    assert facts[3]["dye_events"] and facts[3]["fixed_center_selections"]
    assert facts[4]["dye_events"] and facts[4]["flexible_selections"]
    assert facts[5]["resize_events"]
    assert facts[6]["dye_events"] and facts[6]["resize_events"]
    assert facts[6]["deformation_events"]
    assert facts[7]["dye_events"] and facts[7]["resize_events"]

    episode = Env()
    for context, route in enumerate(routes):
        assert episode.level_index == context
        assert replay(episode, route)
        assert episode.levels_completed == context + 1
    assert episode.state == GameState.WIN


def test_all_generated_tiers_round_trip_validate_and_exercise_profiles():
    specs = []
    for difficulty in DIFFICULTIES:
        spec = generate(
            70_000 + difficulty, difficulty, attempts=240, split="train"
        )
        assert spec is not None
        stored = json.loads(json.dumps(spec))
        assert stored == spec
        assert stored["source"] == "generated_only"
        assert stored["vendored_source_sha256"] == SOURCE_SHA256
        assert stored["solution_length"] == len(stored["solution"])
        assert stored["proof"]["optimal"] is False
        assert not validate_full_standard(
            stored, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        )
        assert replay(Env([build_level(stored)]), stored["solution"])
        specs.append(stored)
    assert len({spec["geometry_d4_sha256"] for spec in specs}) == 8
    assert len({spec["gameplay_sha256"] for spec in specs}) == 8


def test_complete_generate_game_is_eight_ordered_native_contexts():
    specs = generate_game(812_337, split="validation", attempts=240)
    assert specs is not None and len(specs) == 8
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["training_context_index"] for spec in specs] == list(range(8))
    levels = build_game(json.loads(json.dumps(specs)))
    assert len(levels) == 8
    episode = Env(levels)
    for context, spec in enumerate(specs):
        assert episode.level_index == context
        assert replay(episode, spec["solution"])
    assert episode.state == GameState.WIN and episode.levels_completed == 8


def test_validator_rejects_route_geometry_and_proof_tampering():
    spec = generate(19_771, 7, attempts=240, split="test")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][6]
    assert validate_full_standard(spec, entry) == []

    route_tamper = json.loads(json.dumps(spec))
    route_tamper["solution"].pop()
    assert validate_full_standard(route_tamper, entry)

    geometry_tamper = json.loads(json.dumps(spec))
    geometry_tamper["movables"][0]["position"][0] += 3
    assert validate_full_standard(geometry_tamper, entry)

    proof_tamper = json.loads(json.dumps(spec))
    proof_tamper["proof"]["context_engine_verified"] = False
    assert validate_full_standard(proof_tamper, entry)


def test_public_gameplay_identity_excludes_private_certificate_route():
    spec = generate(70_001, 1, attempts=240, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    base_gameplay = spec["gameplay_sha256"]
    base_route = spec["solution_semantic_sha256"]

    changed = json.loads(json.dumps(spec))
    actions = (
        (names.ACTION_NEXT, None, None),
        (names.ACTION_NEXT, None, None),
        *(tuple(action) for action in changed["solution"]),
    )
    _recertify_route(changed, actions)
    assert gameplay_identity(changed) == base_gameplay
    assert changed["gameplay_sha256"] == base_gameplay
    assert changed["solution_semantic_sha256"] != base_route
    assert validate_full_standard(changed, entry) == []
    private_only = json.loads(json.dumps(spec))
    private_only["seed"] += 1
    private_only["proof"]["kind"] = "private-certificate-change"
    assert gameplay_identity(private_only) == base_gameplay


def test_public_identities_preserve_selection_and_dispatch_semantics():
    tier3 = generate(70_003, 3, attempts=240, split="train")
    assert tier3 is not None
    swapped = json.loads(json.dumps(tier3))
    swapped["movables"][1], swapped["movables"][2] = (
        swapped["movables"][2], swapped["movables"][1]
    )
    assert geometry_identity(swapped) != tier3["geometry_d4_sha256"]
    assert gameplay_identity(swapped) != tier3["gameplay_sha256"]
    original_env = Env([build_level(tier3)])
    swapped_env = Env([build_level(swapped)])
    original_frame = np.asarray(original_env.perform(names.ACTION_NEXT).frame)
    swapped_frame = np.asarray(swapped_env.perform(names.ACTION_NEXT).frame)
    assert not np.array_equal(original_frame, swapped_frame)

    tier4 = generate(70_004, 4, attempts=240, split="train")
    assert tier4 is not None
    base = gameplay_identity(tier4)
    recolored = json.loads(json.dumps(tier4))
    old = recolored["dyes"][0]["prototype"]
    first_group = DYES[:8] if old in DYES[:8] else DYES[8:]
    recolored["dyes"][0]["prototype"] = next(
        prototype for prototype in first_group if prototype != old
    )
    assert geometry_identity(recolored) == tier4["geometry_d4_sha256"]
    assert gameplay_identity(recolored) != base
    reordered = json.loads(json.dumps(tier4))
    reordered["dyes"][0], reordered["dyes"][1] = (
        reordered["dyes"][1], reordered["dyes"][0]
    )
    assert gameplay_identity(reordered) != base
    rotated = json.loads(json.dumps(tier4))
    rotated["dyes"][0]["rotation"] = (
        rotated["dyes"][0]["rotation"] + 90
    ) % 360
    # Shipped dye sprites are rotationally symmetric, so their declared
    # rotation is cosmetic and must not salt the public identity.
    assert gameplay_identity(rotated) == base
    asymmetric = Sprite(
        pixels=np.asarray([
            [names.TRANSPARENT, 2],
            [7, 7],
            [7, 2],
        ], dtype=np.int8),
        name="asymmetric-dye-probe",
        tags=[names.TAG_DYE],
    ).set_position(9, 12)
    unrotated = _sprite_identity(asymmetric, "dye", 0, geometry=False)
    asymmetric.set_rotation(90)
    assert _sprite_identity(asymmetric, "dye", 0, geometry=False) != unrotated


def test_fixed_board_d4_rectangles_and_actions_use_matching_transforms():
    pixels = np.arange(6, dtype=np.int16).reshape(2, 3)

    def board_point(row, col, index):
        return (
            (row, col), (row, 63 - col), (63 - row, col),
            (63 - row, 63 - col), (col, row), (col, 63 - row),
            (63 - col, row), (63 - col, 63 - row),
        )[index]

    for transform_index in range(8):
        transformed, new_x, new_y = _transform_rect(
            pixels, 7, 11, transform_index
        )
        expected_shape = (2, 3) if transform_index < 4 else (3, 2)
        assert transformed.shape == expected_shape
        for row in range(2):
            for col in range(3):
                new_row, new_col = board_point(11 + row, 7 + col,
                                               transform_index)
                assert transformed[new_row - new_y, new_col - new_x] == pixels[row, col]

    actions = (names.ACTION_UP, names.ACTION_RIGHT, names.ACTION_NEXT)
    transformed_actions = _action_transforms(actions)
    for transform_index, route in enumerate(transformed_actions):
        for original, transformed in zip(actions[:2], route[:2]):
            dx, dy = names.ACTION_DELTAS[original]
            new_dy, new_dx = _transforms(dy, dx)[transform_index]
            assert names.ACTION_DELTAS[transformed] == (new_dx, new_dy)
        assert route[-1] == names.ACTION_NEXT

    spec = generate(70_004, 4, attempts=240, split="train")
    assert spec is not None
    level = build_level(spec)
    equivalent = level.clone()
    for sprite in equivalent.get_sprites():
        transformed, x, y = _transform_rect(
            sprite.render(), sprite.x, sprite.y, 5
        )
        sprite.pixels = transformed.copy()
        sprite.set_rotation(0).set_position(x, y)
    assert _level_identity(level, geometry=True) == _level_identity(
        equivalent, geometry=True
    )
    assert _level_identity(
        level, geometry=False, difficulty=4
    ) == _level_identity(equivalent, geometry=False, difficulty=4)


def test_validator_fails_closed_on_provenance_types_palette_and_certificate():
    spec = generate(70_001, 1, attempts=240, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]

    mutations = []
    for path, value in (
        (("source",), "official"),
        (("source_id",), "wrong"),
        (("seed",), True),
        (("generator_version",), 3),
        (("vendored_source_sha256",), "0" * 64),
        (("context_index",), 7),
        (("verification_level_index",), 7),
        (("native_budget",), -999),
        (("engine_verified",), False),
        (("search_truncated",), True),
        (("proof", "engine_win"), False),
        (("proof", "search_limit"), 1),
        (("proof", "search_work"), -1),
        (("proof", "kind"), "other"),
    ):
        changed = json.loads(json.dumps(spec))
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        mutations.append(changed)

    changed = json.loads(json.dumps(spec))
    changed["context_solution"] = []
    mutations.append(changed)
    changed = json.loads(json.dumps(spec))
    changed["solution_length"] += 1
    changed["proof"]["action_count"] += 1
    mutations.append(changed)
    changed = json.loads(json.dumps(spec))
    changed["generation_exclusions"] = {"geometry_split": -1}
    mutations.append(changed)
    changed = json.loads(json.dumps(spec))
    changed["movables"][0]["position"][0] = "3"
    mutations.append(changed)
    changed = json.loads(json.dumps(spec))
    changed["movables"][0]["unsupported"] = 1
    mutations.append(changed)
    changed = json.loads(json.dumps(spec))
    changed["unsupported"] = "field"
    mutations.append(changed)
    changed = json.loads(json.dumps(spec))
    changed["movables"][0]["recolor"] = 16
    changed["target"]["colored"][0][2] = 16
    mutations.append(changed)

    for changed in mutations:
        assert validate_full_standard(changed, entry)
    with pytest.raises(ValueError):
        generate(70_001, 1, attempts=1, node_limit=0)
    with pytest.raises(ValueError):
        generate(70_001, 1, attempts=1, node_limit=-1)
    stats = {}
    assert generate(
        70_001, 1, attempts=1, node_limit=1, stats=stats
    ) is None
    assert stats == {"constructive_work_cap": 1}


def test_validator_curriculum_entry_requires_exact_integer_schema():
    spec = generate(70_001, 1, attempts=240, split="train")
    assert spec is not None
    canonical = FULL_STANDARD_CONTRACT["curriculum"][0]
    assert validate_full_standard(spec, canonical) == []

    for key, value in canonical.items():
        for alias in (bool(value), float(value)):
            changed = dict(canonical)
            changed[key] = alias
            errors = validate_full_standard(spec, changed)
            assert errors
            assert all(type(error) is str for error in errors)

    for changed in (
        {key: value for key, value in canonical.items() if key != "search_work"},
        {**canonical, "unsupported": 1},
        {**canonical, 17: 1},
    ):
        errors = validate_full_standard(spec, changed)
        assert errors
        assert all(type(error) is str for error in errors)

    changed_spec = json.loads(json.dumps(spec))
    changed_spec[17] = "unsupported"
    errors = validate_full_standard(changed_spec, canonical)
    assert errors
    assert all(type(error) is str for error in errors)

    changed_spec = json.loads(json.dumps(spec))
    changed_spec["generation_exclusions"] = {"non_json_count": {1}}
    errors = validate_full_standard(changed_spec, canonical)
    assert errors
    assert all(type(error) is str for error in errors)


def test_official_copy_set_and_public_split_are_recomputed():
    official_hashes = _official_geometry_hashes()
    assert len(official_hashes) == len(DIFFICULTIES)
    assert {
        _level_identity(level, geometry=True) for level in official_levels()
    } == official_hashes

    spec = generate(81_001, 1, attempts=240, split="validation")
    assert spec is not None
    assert semantic_partition(spec["gameplay_sha256"]) == spec["split"]
    assert spec["split_partition_bucket"] == (
        int(spec["gameplay_sha256"], 16) % 3
    )
    tampered = json.loads(json.dumps(spec))
    tampered["official_copy"] = True
    assert validate_full_standard(
        tampered, FULL_STANDARD_CONTRACT["curriculum"][0]
    )


def test_split_partition_and_identity_ignore_certificate_fields():
    rows = [
        generate(55_101, 4, attempts=240, split=split)
        for split in ("train", "validation", "test")
    ]
    assert all(rows)
    assert [row["geometry_split"] for row in rows] == [
        "train", "validation", "test"
    ]
    assert len({row["geometry_d4_sha256"] for row in rows}) == 3
    for row in rows:
        assert semantic_partition(row["gameplay_sha256"]) == row["split"]
        changed = json.loads(json.dumps(row))
        changed["proof"]["kind"] = "irrelevant-certificate-label"
        changed["generation_exclusions"] = {"irrelevant": 999}
        assert geometry_identity(changed) == row["geometry_d4_sha256"]
        assert gameplay_identity(changed) == row["gameplay_sha256"]


def test_collector_teacher_uses_generated_high_tier_mechanics():
    spec = generate(9_807, 7, attempts=240, split="train")
    assert spec is not None
    env = Env([build_level(spec)])
    result = search(env, node_limit=PROFILES[7]["search_work"])
    assert result.actions is not None and not result.truncated, result
    teacher = solution_mechanics(Env([build_level(spec)]), result.actions)
    assert teacher["engine_win"]
    assert teacher["dye_events"] >= 1
    assert teacher["resize_events"] >= 1
    assert teacher["deformation_events"] >= 1
    assert replay(Env([build_level(spec)]), result.actions)


def test_live_prefix_reset_cutoffs_and_invalid_actions_are_explicit():
    spec = generate(310, 3, attempts=240, split="train")
    assert spec is not None
    env = Env([build_level(spec)])
    initial = extract(env)
    env.perform(names.ACTION_NEXT)
    env.perform(names.ACTION_RIGHT)
    assert extract(env).steps_left == initial.steps_left - 2
    result = search(env, node_limit=PROFILES[3]["search_work"])
    assert result.actions is not None and replay(env, result.actions)
    env.reset()
    assert extract(env).steps_left == initial.steps_left

    capped = search(Env([official_levels()[0]]), node_limit=1)
    assert capped.actions is None and capped.truncated
    action_capped = search(
        Env([official_levels()[0]]), limit=1,
        node_limit=PROFILES[1]["search_work"],
    )
    assert action_capped.actions is None and action_capped.truncated
    assert solve(Env([official_levels()[0]]), node_limit=1) is None
    assert solve.truncated

    invalid = (
        (names.ACTION_CLICK, 1, 1),
        (names.ACTION_UP, 1, None),
        (7, None, None),
        (0, 0, 0),
    )
    for action in invalid:
        try:
            env.perform(*action)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action {action}")


def test_bounded_quality_smoke_and_bank_cli(tmp_path):
    report = audit_quality(
        seeds_per_tier=2, split="test", attempts=240, seed_offset=900_000
    )
    assert report["accepted"] == report["requested"] == 16
    assert not report["failures"]
    for row in report["per_tier"].values():
        assert row["distinct_geometry"] == 2
        assert row["distinct_gameplay"] == 2
        assert row["distinct_semantic_routes"] == 2

    output = tmp_path / "re86.jsonl"
    assert bank_main([
        "--levels", "1", "--seed", "21", "--difficulty", "5",
        "--out", str(output), "--attempts", "240",
        "--node-limit", "120000",
    ]) == 0
    stored = json.loads(output.read_text().strip())
    assert replay(Env([build_level(stored)]), stored["solution"])


def test_semantic_route_signature_is_not_geometry_or_palette_jitter():
    first = generate(1_200_001, 8, attempts=240, split="train")
    second = generate(1_200_002, 8, attempts=240, split="train")
    assert first is not None and second is not None
    assert first["geometry_d4_sha256"] != second["geometry_d4_sha256"]
    assert semantic_route_signature(first) != semantic_route_signature(second)
