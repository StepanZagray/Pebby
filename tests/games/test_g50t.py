"""G50T complete-mechanics planning, generation, and native replay tests."""

import json
import time

from arcengine import GameState

from pebby.games.g50t import names
from pebby.games.g50t.bank import build as build_bank, load as load_bank, save as save_bank
from pebby.games.g50t.env import Env, official_levels, replay
from pebby.games.g50t.generate import build_level, generate
from pebby.games.g50t.layout import extract
from pebby.games.g50t.plan import search, solve
from pebby.multigame import MultiGameEnv, collect_generated_game, preflight


def _assert_actions(env, actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        assert x is None and y is None


def test_official_level_one_solves_and_replays_in_the_real_engine():
    env = Env()
    assert env.level_count == 7
    assert len(official_levels()) == 7
    frame = env.reset()
    assert len(frame) == names.FRAME_SIZE
    assert all(len(row) == names.FRAME_SIZE for row in frame)
    assert all(isinstance(pixel, int) and 0 <= pixel <= 15
               for row in frame for pixel in row)
    assert env.available_actions == names.ACTION_IDS
    assert extract(env).exact

    result = search(env, limit=names.NATIVE_MAX_ACTIONS, node_limit=10_000)
    assert result.actions is not None and result.exact
    assert not result.truncated and not result.unsupported
    assert any(action == names.ACTION_REWIND for action, _, _ in result.actions)
    _assert_actions(env, result.actions)
    assert replay(env, result.actions)
    assert env.levels_completed == 1
    assert env.level_index == 1


def test_clone_native_budget_reset_and_action_validation():
    env = Env([official_levels()[0]])
    clone = env.clone()
    clone_frame = clone.render()
    env.perform(names.ACTION_RIGHT)
    assert env.steps_used == 1 and env.steps_left == names.NATIVE_MAX_ACTIONS - 1
    assert clone.steps_used == 0 and clone.render() == clone_frame
    assert env.render() != clone.render()

    env.perform(names.ACTION_RESET)
    assert env.steps_used == 0
    assert extract(env).history == ()
    invalid = (
        (names.ACTION_CLICK, 0, 0),
        (7, None, None),
        (names.ACTION_UP, 1, None),
        (names.ACTION_RESET, 0, 0),
    )
    for action in invalid:
        try:
            env.perform(*action)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action {action}")


def test_live_replanning_covers_history_good_and_premature_ghosts():
    cases = (
        (names.ACTION_RIGHT, names.ACTION_LEFT, names.ACTION_DOWN),
        (names.ACTION_RIGHT, names.ACTION_REWIND),
        (
            names.ACTION_RIGHT, names.ACTION_RIGHT, names.ACTION_RIGHT,
            names.ACTION_RIGHT, names.ACTION_REWIND, names.ACTION_DOWN,
            names.ACTION_LEFT,
        ),
    )
    for prefix in cases:
        env = Env([official_levels()[0]])
        for action in prefix:
            env.perform(action)
        layout = extract(env)
        assert layout.exact, layout.unsupported
        actions = solve(env, limit=env.steps_left, node_limit=20_000)
        assert actions is not None, solve.reason
        assert solve.exact and not solve.truncated and not solve.unsupported
        assert replay(env, actions)
        assert env.state == GameState.WIN


def test_three_varied_difficulty_one_specs_under_sixty_seconds_and_json_replay():
    started = time.monotonic()
    specs = [generate(seed, 1, attempts=3, node_limit=10_000) for seed in range(3)]
    assert all(spec is not None for spec in specs)
    assert time.monotonic() - started < 60
    signatures = set()
    for spec in specs:
        stored = json.loads(json.dumps(spec))
        assert stored == spec
        assert stored["engine_verified"] and stored["search_exact"]
        assert not stored["search_truncated"]
        assert stored["solution_length"] == len(stored["solution"])
        assert any(action[0] == names.ACTION_REWIND for action in stored["solution"])
        env = Env([build_level(stored)])
        _assert_actions(env, stored["solution"])
        assert replay(env, stored["solution"])
        signatures.add(json.dumps(stored["cells"], sort_keys=True))
    assert len(signatures) == 3


def test_difficulty_two_and_three_smoke_and_declared_scope():
    cell_counts = []
    for difficulty in (2, 3):
        spec = generate(4, difficulty, attempts=3, node_limit=10_000)
        assert spec is not None
        assert spec["difficulty"] == difficulty
        assert spec["mechanics_version"] == "g50t-complete-mechanics-v4"
        assert spec["solution_mechanics"]["rewinds"] == 2
        assert len(spec["switches"]) == difficulty
        assert len(spec["doors"]) == difficulty
        assert replay(Env([build_level(spec)]), spec["solution"])
        cell_counts.append(len(spec["cells"]))
    assert all(count >= 24 for count in cell_counts)


def test_bounded_seed_sample_has_distinct_initial_geometries():
    frames = set()
    topology = set()
    for seed in range(24):
        spec = generate(seed, 1, attempts=8, node_limit=10_000)
        assert spec is not None, seed
        topology.add(spec["geometry_sha256"])
        frame = Env([build_level(spec)]).render()
        frames.add(bytes(pixel for row in frame for pixel in row))
    assert len(topology) >= 22
    assert len(frames) >= 22


def test_three_generated_levels_advance_sequentially_with_live_replanning():
    specs = [generate(seed, difficulty, node_limit=10_000)
             for seed, difficulty in ((8, 1), (9, 2), (10, 3))]
    assert all(spec is not None for spec in specs)
    env = Env([build_level(spec) for spec in specs])
    assert env.level_count == 3
    for expected_index in range(3):
        assert env.level_index == expected_index
        result = search(env, limit=env.steps_left, node_limit=20_000)
        assert result.actions is not None, result.reason
        assert result.exact and not result.truncated and not result.unsupported
        assert replay(env, result.actions)
        assert env.levels_completed == expected_index + 1
    assert env.state == GameState.WIN


def test_all_later_mechanics_are_supported_and_bounds_are_explicit():
    capped = search(Env([official_levels()[0]]), node_limit=1)
    assert capped.actions is None and capped.truncated
    assert not capped.exact and not capped.unsupported
    assert capped.expanded <= capped.node_limit
    assert solve(Env([official_levels()[0]]), node_limit=1) is None
    assert solve.truncated and not solve.exact

    later = search(Env([official_levels()[1]]), node_limit=150_000)
    assert later.actions is not None and later.exact
    assert not later.truncated and not later.unsupported
    assert replay(Env([official_levels()[1]]), later.actions)


def test_bank_preflight_and_shared_collector_smoke(tmp_path):
    specs, tried = build_bank(2, seed=30, difficulty=1, max_seeds=4, node_limit=10_000)
    assert len(specs) == 2 and tried >= 2
    path = save_bank(specs, tmp_path / "g50t.jsonl")
    assert load_bank(path) == specs

    modules, = preflight(["g50t"])
    game = MultiGameEnv.from_specs(modules, [specs[0]])
    assert game.raw_env.level_count == 1
    assert game.legal_action_ids == names.ACTION_IDS
    collected = collect_generated_game(
        modules,
        master_seed=53,
        game_index=0,
        difficulties=[1, 2, 3],
        outer_generation_attempts=2,
        generator_attempts=3,
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_completed"] == 3
