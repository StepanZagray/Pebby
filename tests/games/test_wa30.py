import copy
import hashlib
import json

import pytest

from pebby.games.wa30 import names
from pebby.games.wa30.env import Env, official_levels, replay
from pebby.games.wa30.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    generate,
    generate_game,
    load_level,
    validate_full_standard,
)
from pebby.games.wa30.layout import extract
from pebby.games.wa30.plan import search, state_of, transition
from pebby.games.wa30.reference_profiles import (
    MECHANIC_INVENTORY,
    PROFILES,
    REFERENCE_MEASUREMENTS,
)


def context_env(spec):
    index = spec["difficulty"] - 1
    env = Env([build_level(spec).clone() for _ in range(index + 1)])
    env.set_level(index)
    return env


def assert_action_triples(actions):
    assert actions
    assert all(
        isinstance(step, (list, tuple))
        and len(step) == 3
        and step[0] in names.ACTION_IDS
        and tuple(step[1:]) == (None, None)
        for step in actions
    )


def test_import_and_exact_nine_tier_contract():
    assert len(official_levels()) == 9
    assert DIFFICULTIES == tuple(range(1, 10))
    assert [row["difficulty"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(DIFFICULTIES)
    assert [row["context_index"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(9))
    assert FULL_STANDARD_CONTRACT["status"] in {"pending_audit", "ready"}
    assert all(entry["source"] for entry in MECHANIC_INVENTORY.values())


def test_every_official_tier_matches_recorded_reference_measurements():
    for difficulty in DIFFICULTIES:
        env = Env()
        env.set_level(difficulty - 1)
        layout = extract(env)
        measured = REFERENCE_MEASUREMENTS[difficulty]
        assert len(layout.boxes) == measured["boxes"]
        assert len(layout.helpers) == measured["helpers"]
        assert len(layout.thieves) == measured["thieves"]
        assert len(layout.walls) == measured["walls"]
        assert len(layout.fences) == measured["fences"]
        assert len(layout.goals) == measured["goal_cells"]
        assert len(layout.bad) == measured["bad_cells"]
        assert layout.max_steps == measured["budget"]
        frame = env.render()
        assert len(frame) == 64 and all(len(row) == 64 for row in frame)
        assert len({pixel for row in frame for pixel in row}) >= 3


def test_exact_teacher_transition_matches_real_engine_on_every_official_tier():
    # Covers all shipped combinations, including stale target caches, fences,
    # ordered multi-robot phases, and thief destruction where reachable.
    actions = (1, 4, 2, 3, 5) * 5
    for difficulty in DIFFICULTIES:
        env = Env()
        env.set_level(difficulty - 1)
        layout = extract(env)
        symbolic = state_of(layout)
        for action in actions[: min(len(actions), env.max_steps() - 1)]:
            symbolic = transition(layout, symbolic, action)
            observation = env.perform(action)
            if observation.finished:
                break
            actual = state_of(extract(env))
            # This fixed trace does not destroy a thief, so actor cardinality is
            # stable and the states should match byte-for-byte.
            assert symbolic == actual


@pytest.mark.parametrize("difficulty", [1, 2, 5, 6, 7])
def test_official_positive_teacher_routes_replay(difficulty):
    env = Env()
    env.set_level(difficulty - 1)
    result = search(env, limit=PROFILES[difficulty]["search_limit"])
    assert result.solved, result.reason
    assert not result.truncated
    assert_action_triples(result.actions)
    proof = Env()
    proof.set_level(difficulty - 1)
    assert replay(proof, result.actions)


@pytest.fixture(scope="module")
def generated_tiers():
    rows = {difficulty: generate(0, difficulty) for difficulty in DIFFICULTIES}
    assert all(rows.values())
    return rows


def test_every_generated_tier_uses_plan_search_and_passes_fail_closed_validation(generated_tiers):
    for difficulty, spec in generated_tiers.items():
        assert_action_triples(spec["solution"])
        errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1])
        assert errors == []
        env = context_env(spec)
        result = search(env, limit=spec["search_limit"])
        assert result.solved, result.reason
        assert result.actions == [tuple(step) for step in spec["solution"]]
        assert replay(context_env(spec), spec["solution"])


def test_json_round_trip_and_validator_rejects_route_geometry_and_proof_tampering(generated_tiers):
    spec = generated_tiers[8]
    restored = json.loads(json.dumps(spec))
    assert restored == spec
    assert load_level(restored).get_data(names.KEY_STEP_COUNTER) == spec["budget"]

    changed = copy.deepcopy(spec)
    changed["solution"][-1][0] = 1 if changed["solution"][-1][0] != 1 else 2
    assert validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][7])

    changed = copy.deepcopy(spec)
    changed["boxes"][0][0] += 1
    assert validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][7])

    changed = copy.deepcopy(spec)
    changed["proof"]["engine_win"] = False
    assert validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][7])


def test_all_splits_are_geometry_partitioned_and_validate():
    rows = [generate(17, 2, split=split) for split in ("train", "validation", "test")]
    assert all(rows)
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert len({row["geometry_d4_sha256"] for row in rows}) == 3
    for row in rows:
        assert validate_full_standard(row, FULL_STANDARD_CONTRACT["curriculum"][1]) == []


def test_generate_game_is_exactly_nine_levels_and_wins_sequentially():
    specs = generate_game(0)
    assert specs is not None and len(specs) == 9
    assert [row["difficulty"] for row in specs] == list(DIFFICULTIES)
    assert [row["game_level_index"] for row in specs] == list(range(9))
    env = Env(build_game(specs))
    for index, spec in enumerate(specs):
        assert env.level_index == index
        assert replay(env, spec["solution"])
        assert env.levels_completed == index + 1
    assert env.finished


def test_live_prefix_recovers_with_same_teacher_entry_point(generated_tiers):
    spec = generated_tiers[8]
    env = context_env(spec)
    prefix = spec["solution"][:12]
    for action, _, _ in prefix:
        assert not env.perform(action).finished
    result = search(env, limit=spec["search_limit"])
    assert result.solved, result.reason
    assert replay(env, result.actions)


def test_generated_sprite_extents_and_visual_cues_are_visible(generated_tiers):
    official_frames = {
        hashlib.sha256(bytes(pixel for row in Env().render() for pixel in row)).hexdigest()
    }
    for index in range(1, 9):
        env = Env()
        env.set_level(index)
        official_frames.add(hashlib.sha256(bytes(pixel for row in env.render() for pixel in row)).hexdigest())
    generated_frames = set()
    for spec in generated_tiers.values():
        env = context_env(spec)
        for sprite in env.sprites():
            assert 0 <= sprite.x and 0 <= sprite.y
            assert sprite.x + sprite.width <= 64
            assert sprite.y + sprite.height <= 64
        frame = env.render()
        generated_frames.add(hashlib.sha256(bytes(pixel for row in frame for pixel in row)).hexdigest())
        assert any(names.GOAL_BORDER_COLOR in row for row in frame)
        assert any(14 in row for row in frame)  # player direction cue
    assert generated_frames.isdisjoint(official_frames)
