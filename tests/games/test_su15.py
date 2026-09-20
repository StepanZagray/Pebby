import copy
from collections import Counter
import importlib
import json
import random

import pytest
from arcengine import GameState

from pebby.games.su15 import names
from pebby.games.su15.env import Env, official_levels, replay
from pebby.games.su15.generate import (
    DIFFICULTIES,
    FORMAT,
    FULL_STANDARD_CONTRACT,
    _context_env,
    _draft,
    _mirror_values,
    _verify,
    build_game,
    build_level,
    generate,
    generate_game,
    validate_full_standard,
)
from pebby.games.su15.layout import extract
from pebby.games.su15.plan import search
from pebby.games.su15.quality import REFERENCE_PROFILES, profile_errors
from pebby.games.su15.quality import gameplay_identity, geometry_identity, geometry_partition


def _requirements(env):
    raw = env.game.dsqlbvwaj
    rows = raw if isinstance(raw[0], (list, tuple)) else [raw]
    reverse = {value: key for key, value in names.ENEMY_REQUIREMENT_KEYS.items()}
    return tuple(("enemy", reverse[key], int(count)) if key in reverse
                 else ("fruit", int(key), int(count)) for key, count in rows)


def _replay_spec(spec):
    env = Env([build_level(spec).clone() for _ in range(spec["context_index"] + 1)])
    env.set_level(spec["context_index"])
    completed, observation = replay(env, [tuple(action) for action in spec["solution"]])
    assert completed and observation is not None
    assert env.levels_completed == 1
    return env, observation


def test_full_contract_declares_root_reviewed_ready_nine_tier_curriculum():
    assert len(official_levels()) == 9
    assert DIFFICULTIES == tuple(range(1, 10))
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert [(row["difficulty"], row["context_index"])
            for row in FULL_STANDARD_CONTRACT["curriculum"]] == [
                (difficulty, difficulty - 1) for difficulty in DIFFICULTIES
            ]
    assert all(type(key) is str and key.strip()
               and type(value) is str and value.strip()
               for key, value in FULL_STANDARD_CONTRACT["evidence"].items())
    caveats = "\n".join(FULL_STANDARD_CONTRACT["caveats"])
    for required in (
        "one shipped reference", "not claimed shortest", "finite",
        "two of three installed target zones", "no all-enemy indispensability claim",
        "excludes admission replay, validator replay, and full-game replay",
    ):
        assert required in caveats
    assert "100/98/103 actions" in FULL_STANDARD_CONTRACT["evidence"]["root_primary_collector"]
    assert "pre-final-validation hashes" in FULL_STANDARD_CONTRACT["evidence"]["root_primary_collector"]
    assert "validation-only patch" in FULL_STANDARD_CONTRACT["evidence"]["root_native_frame_review"]


def test_reference_profiles_are_recomputed_from_all_official_starts():
    levels = official_levels()
    for index, profile in REFERENCE_PROFILES.items():
        env = Env(levels)
        env.set_level(index - 1)
        fruit_counts = Counter(int(env.game.kqywaxhmsb[s]) for s in env.fruits())
        enemy_counts = Counter(int(env.game.dfqhmningy(env.game.kcuphgwar[s]))
                               for s in env.enemies())
        assert dict(fruit_counts) == profile["fruit_counts"]
        assert dict(enemy_counts) == profile["enemy_counts"]
        assert len(env.targets()) == profile["target_count"]
        assert _requirements(env) == profile["requirements"]
        assert env.native_steps_left == profile["steps"]


def test_width_aware_reflection_is_one_partition_but_native_order_changes_gameplay():
    spec = generate(6_008, 8, split="train")
    assert spec is not None
    reflected = copy.deepcopy(spec)
    for field in ("fruits", "enemies", "targets"):
        reflected[field] = _mirror_values(reflected[field])
    assert geometry_identity(reflected) == geometry_identity(spec)
    assert gameplay_identity(reflected) == gameplay_identity(spec)
    assert geometry_partition(reflected) == geometry_partition(spec)

    fruit_order = copy.deepcopy(spec)
    fruit_order.update(
        difficulty=4, context_index=3, reference_level=4, steps=48,
        fruits=[{"tier": 2, "position": [20, 30]},
                {"tier": 2, "position": [40, 30]}],
        enemies=[{"kind": 1, "position": [29, 29]}],
        targets=[{"position": [0, 10]}],
        requirements=[{"kind": "fruit", "tier": 2, "count": 2}],
    )
    reversed_fruits = copy.deepcopy(fruit_order)
    reversed_fruits["fruits"].reverse()
    assert gameplay_identity(fruit_order) != gameplay_identity(reversed_fruits)
    env_a, env_b = Env([build_level(fruit_order)]), Env([build_level(reversed_fruits)])
    env_a.reset(), env_b.reset()
    assert env_a.render() == env_b.render()
    env_a.perform(names.ACTION_CLICK, 0, 62)
    env_b.perform(names.ACTION_CLICK, 0, 62)
    assert sorted((int(s.x), int(s.y)) for s in env_a.enemies()) != sorted(
        (int(s.x), int(s.y)) for s in env_b.enemies())

    enemy_order = copy.deepcopy(spec)
    enemy_order.update(
        difficulty=7, context_index=6, reference_level=7, steps=32,
        fruits=[{"tier": 5, "position": [30, 30]}],
        enemies=[{"kind": 1, "position": [25, 30]},
                 {"kind": 1, "position": [37, 30]}],
        targets=[{"position": [0, 10]}],
        requirements=[{"kind": "fruit", "tier": 4, "count": 1}],
    )
    reversed_enemies = copy.deepcopy(enemy_order)
    reversed_enemies["enemies"].reverse()
    assert gameplay_identity(enemy_order) != gameplay_identity(reversed_enemies)
    env_a, env_b = Env([build_level(enemy_order)]), Env([build_level(reversed_enemies)])
    env_a.reset(), env_b.reset()
    assert env_a.render() == env_b.render()
    env_a.perform(names.ACTION_CLICK, 0, 62)
    env_b.perform(names.ACTION_CLICK, 0, 62)
    fruit_a, fruit_b = env_a.fruits()[0], env_b.fruits()[0]
    assert (int(fruit_a.x), int(fruit_a.y)) != (int(fruit_b.x), int(fruit_b.y))


def test_exact_official_semantics_are_rejected_without_using_header_art():
    draft = _draft(random.Random(2), 1)
    official = Env()
    official.reset()
    draft["fruits"] = [
        {"tier": int(official.game.kqywaxhmsb[s]), "position": [int(s.x), int(s.y)]}
        for s in official.fruits()
    ]
    draft["enemies"] = []
    draft["targets"] = [
        {"position": [int(s.x), int(s.y)]} for s in official.targets()
    ]
    accepted, reason = _verify(draft, 500)
    assert accepted is None
    assert reason == "official_gameplay_copy"


def test_teacher_solves_and_replays_all_nine_official_levels_sequentially():
    env = Env()
    env.reset()
    measured = []
    for index in range(9):
        assert env.level_index == index
        result = search(env, limit=50_000)
        assert result.actions is not None
        assert result.exact and not result.truncated and not result.unsupported
        measured.append(len(result.actions))
        assert replay(env, result.actions)[0]
        assert env.levels_completed == index + 1
    assert measured == [REFERENCE_PROFILES[d]["reference_witness_actions"]
                        for d in DIFFICULTIES]
    assert env.state == GameState.WIN


def test_single_level_generation_covers_every_reference_composition_and_replays():
    for difficulty in DIFFICULTIES:
        spec = generate(7_000 + difficulty, difficulty, split="train")
        assert spec is not None, generate.last_report
        assert spec["format"] == FORMAT
        assert spec["difficulty"] == difficulty
        assert spec["context_index"] == difficulty - 1
        assert profile_errors(spec) == []
        assert validate_full_standard(
            spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        ) == []
        env, observation = _replay_spec(spec)
        assert observation.state == GameState.WIN


def test_mechanic_use_progression_is_witness_derived():
    specs = [generate(8_000 + difficulty, difficulty, split="validation")
             for difficulty in DIFFICULTIES]
    assert all(specs)
    mechanics = {spec["difficulty"]: spec["solution_mechanics"] for spec in specs}
    for difficulty in (2, 3, 4, 5, 7, 8, 9):
        assert mechanics[difficulty]["fruit_merges"] >= 1
    for difficulty in (6, 7, 8, 9):
        assert mechanics[difficulty]["fruit_degrades"] >= 1
        assert mechanics[difficulty]["pursuer_motion_actions"] >= 1
    for difficulty in (8, 9):
        assert mechanics[difficulty]["enemy_merges"] >= 1
    assert mechanics[9]["enemy_merges"] >= 3
    for difficulty, trace in mechanics.items():
        assert trace["installed_target_zones"] == REFERENCE_PROFILES[difficulty]["target_count"]
        assert 1 <= trace["witnessed_target_zones"] <= trace["installed_target_zones"]
    for difficulty in (3, 6, 7, 8, 9):
        assert mechanics[difficulty]["witnessed_target_zones"] >= 2


def test_generate_game_and_independently_generated_specs_replay_all_contexts(monkeypatch):
    game = generate_game(889, split="train")
    assert game is not None, generate_game.last_report
    assert generate_game(889, split="train") == game
    assert len(game) == 9
    assert [spec["difficulty"] for spec in game] == list(DIFFICULTIES)
    assert all(validate_full_standard(spec, entry) == [] for spec, entry in zip(
        game, FULL_STANDARD_CONTRACT["curriculum"]
    ))
    assert len(build_game(game)) == 9

    mutations = (
        lambda rows: rows[0].update(game_position=8),
        lambda rows: rows[0].update(game_sequence_sha256="0" * 64),
        lambda rows: rows[0].update(parent_game_seed=890),
        lambda rows: rows[0]["proof"]["full_game_replay"].update(state="WIN"),
        lambda rows: rows[0]["proof"].update(full_game_replay=None),
        lambda rows: rows[8]["proof"]["full_game_replay"].update(state="NOT_FINISHED"),
        lambda rows: rows[0].pop("game_sequence_sha256"),
        lambda rows: rows[0]["proof"].pop("full_game_replay"),
    )
    for mutate in mutations:
        bad = copy.deepcopy(game)
        mutate(bad)
        with pytest.raises(ValueError):
            build_game(bad)

    module = importlib.import_module("pebby.games.su15.generate")
    mixed_parent = copy.deepcopy(game)
    child_seed = module._child_seed(890, 1, 2)
    mixed_parent[1].update(parent_game_seed=890, child_seed=child_seed, seed=child_seed)
    assert validate_full_standard(
        mixed_parent[1], FULL_STANDARD_CONTRACT["curriculum"][1]
    ) == []
    with pytest.raises(ValueError, match="share one parent"):
        build_game(mixed_parent)

    independent = [generate(9_000 + difficulty, difficulty, split="test")
                   for difficulty in DIFFICULTIES]
    assert all(independent)
    assert len(build_game(independent)) == 9

    smoke = generate_game(890, split="validation", difficulties=(1,))
    assert smoke is not None and smoke[0]["sequence_kind"] == "explicit-smoke-subset"
    assert "game_sequence_sha256" not in smoke[0]
    assert "full_game_replay" not in smoke[0]["proof"]
    assert validate_full_standard(smoke[0], FULL_STANDARD_CONTRACT["curriculum"][0]) == []

    native_replay = module._replay_full_game

    def mismatching_replay(specs):
        rows = native_replay(specs)
        rows[0] = dict(rows[0], levels_completed=9)
        return rows

    monkeypatch.setattr(module, "_replay_full_game", mismatching_replay)
    with pytest.raises(ValueError, match="differs from native sequential replay"):
        build_game(game)


def test_json_round_trip_and_certificate_suffix_live_recovery():
    spec = generate(10_003, 3, split="train")
    restored = json.loads(json.dumps(spec))
    assert restored == spec
    assert validate_full_standard(
        restored, FULL_STANDARD_CONTRACT["curriculum"][2]
    ) == []
    env = Env([build_level(restored).clone() for _ in range(3)])
    env.set_level(2)
    prefix = [tuple(action) for action in restored["solution"][:2]]
    for action in prefix:
        env.perform(*action)
    result = search(env, limit=50_000)
    assert result.actions is not None
    assert len(result.actions) < len(restored["solution"])
    assert replay(env, result.actions)[0]


def test_undo_and_unequal_collision_auto_rollback_are_native():
    spec = generate(11_003, 3, split="train")
    env = Env([build_level(spec)])
    env.reset()
    initial = [(int(env.game.kqywaxhmsb[s]), int(s.x), int(s.y)) for s in env.fruits()]
    env.perform(*spec["solution"][0])
    assert env.history_depth >= 2
    env.perform(names.ACTION_UNDO)
    assert [(int(env.game.kqywaxhmsb[s]), int(s.x), int(s.y)) for s in env.fruits()] == initial

    collision = copy.deepcopy(spec)
    fruit0 = next(value for value in collision["fruits"] if value["tier"] == 0)
    fruit1 = next(value for value in collision["fruits"] if value["tier"] == 1)
    fruit0["position"], fruit1["position"] = [20, 30], [21, 30]
    probe = Env([build_level(collision)])
    probe.reset()
    before = sorted((int(probe.game.kqywaxhmsb[s]), int(s.x), int(s.y)) for s in probe.fruits())
    steps = probe.native_steps_left
    probe.perform(names.ACTION_CLICK, 21, 31)
    after = sorted((int(probe.game.kqywaxhmsb[s]), int(s.x), int(s.y)) for s in probe.fruits())
    assert after == before
    assert probe.native_steps_left == steps - 2
    steps = probe.native_steps_left
    probe.perform(names.ACTION_CLICK, 21, 31)
    assert probe.native_steps_left == steps - 4

    # Mixed pursuer classes use the same native flashing penalty path.  Start
    # them overlapped to exercise it directly without depending on a positive
    # certificate deliberately making a losing move.
    pursuer_collision = generate(11_008, 8, split="train")
    assert pursuer_collision is not None
    pursuer_collision = copy.deepcopy(pursuer_collision)
    pursuer_collision["enemies"][0].update(kind=2, position=[20, 30])
    pursuer_collision["enemies"][1].update(kind=1, position=[20, 30])
    probe = Env([build_level(pursuer_collision)])
    probe.reset()
    before = sorted((probe.game.dfqhmningy(probe.game.kcuphgwar[s]), int(s.x), int(s.y))
                    for s in probe.enemies())
    steps = probe.native_steps_left
    probe.perform(names.ACTION_CLICK, 22, 32)
    after = sorted((probe.game.dfqhmningy(probe.game.kcuphgwar[s]), int(s.x), int(s.y))
                   for s in probe.enemies())
    assert after == before
    assert probe.native_steps_left == steps - 2


def test_validator_rejects_geometry_route_split_and_proof_tampering():
    spec = generate(12_008, 8, split="validation")
    entry = FULL_STANDARD_CONTRACT["curriculum"][7]
    bad = copy.deepcopy(spec)
    bad["fruits"][0]["position"][0] += 1
    assert validate_full_standard(bad, entry)
    bad = copy.deepcopy(spec)
    bad["solution"].append([names.ACTION_UNDO, None, None])
    bad["solution_length"] += 1
    assert validate_full_standard(bad, entry)
    bad = copy.deepcopy(spec)
    bad["geometry_split"] = "test"
    assert validate_full_standard(bad, entry)
    bad = copy.deepcopy(spec)
    bad["proof"]["context_engine_verified"] = False
    assert validate_full_standard(bad, entry)
    bad = copy.deepcopy(spec)
    bad["initial_non_background_pixels"] += 1
    assert "initial visual density mismatch" in validate_full_standard(bad, entry)


def test_complete_schema_rejects_all_reviewed_type_proof_and_mirror_tampering():
    spec = generate(12_001, 1, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    mutations = (
        lambda row: row.update(engine_verified=False),
        lambda row: row.update(context_engine_verified=False),
        lambda row: row.update(search_truncated=True),
        lambda row: row.update(search_limit=-1),
        lambda row: row.update(search_work_used=10**20),
        lambda row: row.update(native_budget=999),
        lambda row: row.update(context_solution=[]),
        lambda row: row.update(geometry_version="wrong"),
        lambda row: row.update(gameplay_version="wrong"),
        lambda row: row.update(seed=True),
        lambda row: row.update(generation_attempt=False),
        lambda row: row.update(reference_level=9),
        lambda row: row.update(solution_length=float(row["solution_length"])),
        lambda row: row["proof"].update(context_index=False),
        lambda row: row["proof"].update(levels_completed=999),
        lambda row: row["proof"].update(search_limit=False),
        lambda row: row["proof"].update(search_work_used=-1),
        lambda row: row["proof"].update(witness_kind="optimal"),
        lambda row: row.update(unexpected_certificate=True),
    )
    for mutate in mutations:
        bad = copy.deepcopy(spec)
        mutate(bad)
        assert validate_full_standard(bad, entry), mutate

    old = copy.deepcopy(spec)
    old.update(format="pebby.su15.full-level.v2", generator_version=2,
               geometry_version="su15-playfield-reflection-v1",
               gameplay_version="su15-object-interaction-v1")
    assert validate_full_standard(old, entry)
    bad_entry = dict(entry, context_index=False)
    assert validate_full_standard(spec, bad_entry)


def test_malformed_nested_rows_return_diagnostics_before_native_construction(monkeypatch):
    spec = generate(12_002, 1, split="validation")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    module = importlib.import_module("pebby.games.su15.generate")

    def forbidden(_spec):
        raise AssertionError("schema-invalid row reached native construction")

    monkeypatch.setattr(module, "build_level", forbidden)
    mutations = (
        lambda row: row.update(fruits=[None]),
        lambda row: row.update(enemies=[None]),
        lambda row: row.update(targets=[None]),
        lambda row: row.update(requirements=[None]),
        lambda row: row.update(solution_length="bad"),
        lambda row: row.update(fruits=row["fruits"] * 33),
        lambda row: row["proof"].update(search_work_used="bad"),
    )
    for mutate in mutations:
        bad = copy.deepcopy(spec)
        mutate(bad)
        diagnostics = validate_full_standard(bad, entry)
        assert diagnostics and all(type(value) is str for value in diagnostics)

    for invalid_tier in ("2", None, [], {}):
        bad = copy.deepcopy(spec)
        bad["fruits"][0]["tier"] = invalid_tier
        diagnostics = validate_full_standard(bad, entry)
        assert diagnostics and all(type(value) is str for value in diagnostics)

    for invalid_kind in ([], {}):
        bad = copy.deepcopy(spec)
        bad["sequence_kind"] = invalid_kind
        diagnostics = validate_full_standard(bad, entry)
        assert diagnostics and all(type(value) is str for value in diagnostics)


def test_single_work_meter_caps_every_native_candidate_and_suffix_transition(monkeypatch):
    spec = generate(12_006, 6, split="test")
    assert spec is not None
    draft = copy.deepcopy(spec)
    draft.pop("solution")
    original_perform = Env.perform
    calls = {"count": 0}

    def counted_perform(self, *args, **kwargs):
        calls["count"] += 1
        return original_perform(self, *args, **kwargs)

    monkeypatch.setattr(Env, "perform", counted_perform)
    for limit in (1, 2, 5, 10):
        calls["count"] = 0
        result = search(_context_env(draft), limit=limit)
        assert result.actions is None and result.truncated
        assert result.work == calls["count"] == limit

    calls["count"] = 0
    suffix_result = search(_context_env(spec), limit=1)
    assert suffix_result.actions is None and suffix_result.truncated
    assert suffix_result.work == calls["count"] == 1

    calls["count"] = 0
    positive = search(_context_env(draft), limit=500)
    assert positive.actions is not None and not positive.truncated
    assert positive.work == calls["count"] <= 500


def test_rendered_frames_fit_and_show_rule_cues_without_clipping():
    specs = [generate(13_000 + difficulty, difficulty, split="test")
             for difficulty in DIFFICULTIES]
    assert all(specs)
    official = Env()
    official.reset()
    official_frames = []
    for index in range(len(DIFFICULTIES)):
        official.set_level(index)
        official_frames.append(official.render())
    env = Env([build_level(spec) for spec in specs])
    env.reset()
    for index, spec in enumerate(specs):
        env.set_level(index)
        frame = env.render()
        assert len(frame) == 64 and all(len(row) == 64 for row in frame)
        assert frame not in official_frames
        assert sum(value not in (-1, 3, 4, 5) for row in frame for value in row) == (
            spec["initial_non_background_pixels"]
        )
        assert any(frame[y][x] not in (3, 4, 5, -1)
                   for y in range(0, 10) for x in range(64))
        header = {frame[y][x] for y in range(0, 10) for x in range(0, 26)}
        if spec["difficulty"] >= 2:
            assert {10, 6, 15, 11, 12, 8} <= header
        if spec["difficulty"] >= 8:
            assert {7, 14, 13} <= header
        for field in ("fruits", "enemies", "targets"):
            for value in spec[field]:
                x, y = value["position"]
                assert 0 <= x < 64 and 0 <= y < 64


def test_rejection_reporting_and_argument_bounds_are_explicit():
    assert generate(14_001, 1, attempts=1, limit=1, split="train") is None
    assert generate.last_report["accepted"] is False
    assert generate.last_report["rejections"]
    with pytest.raises(ValueError):
        generate(True, 1, split="train")
    with pytest.raises(ValueError):
        generate(1, 10, split="train")
    with pytest.raises(ValueError):
        generate(1, 1, split="dev")
    with pytest.raises(ValueError):
        generate_game(1, split="train", difficulties=(2, 1))


def test_layout_accepts_full_dynamic_states_and_reports_native_budget():
    spec = generate(15_009, 9, split="train")
    env = Env([build_level(spec).clone() for _ in range(9)])
    env.set_level(8)
    layout = extract(env)
    assert layout.exact_positive_scope
    assert layout.generated
    assert (layout.fruit_count, layout.enemy_count, layout.target_count) == (3, 4, 3)
    assert layout.native_steps_left == spec["steps"]
    assert layout.action_budget == spec["steps"] + env.history_depth
