import json
import subprocess
import sys
import time

import pytest
from arcengine import GameState

from pebby.games.tr87.env import Env, official_levels, replay
from pebby.games.tr87.generate import build_level, env_for, generate
from pebby.games.tr87.layout import Layout, extract
from pebby.games.tr87.plan import search, solve


def assert_replay(env, actions):
    before = env.levels_completed
    assert actions
    for action, x, y in actions:
        assert action in env.available_actions
        assert x is None and y is None
    replay(env, actions)
    assert env.levels_completed == before + 1


def test_official_first_level():
    env = Env()
    env.reset()
    assert len(official_levels()) == 6
    initial = extract(env)
    actions = solve(env)
    assert not solve.truncated
    assert extract(env) == initial  # planning must not change the live game
    assert_replay(env, actions)
    assert env.level_index == 1


@pytest.mark.parametrize("difficulty", (1, 2, 3))
def test_generated_roundtrip_and_real_engine_replay(difficulty):
    started = time.monotonic()
    specs = [generate(seed, difficulty) for seed in range(3)]
    assert time.monotonic() - started < 60
    for spec in specs:
        assert spec is not None
        restored = json.loads(json.dumps(spec))
        env = env_for(restored)
        assert extract(env) == Layout.from_dict(restored["initial"])
        assert restored["solution_length"] == len(restored["solution"])
        assert_replay(env, restored["solution"])
        assert env.state == GameState.WIN


def test_clone_reset_and_frame():
    env = Env()
    frame = env.reset()
    assert len(frame) == 64 and all(len(row) == 64 for row in frame)
    assert all(isinstance(pixel, int) and 0 <= pixel <= 15 for row in frame for pixel in row)
    initial = extract(env)
    twin = env.clone()
    twin.perform(2)
    assert extract(env) == initial
    assert extract(twin) != initial
    twin.reset()
    assert extract(twin) == initial
    with pytest.raises(ValueError):
        env.perform(6, 0, 0)


def test_search_limits_report_truncation():
    env = Env()
    env.reset()
    assert solve(env, node_limit=0) is None
    assert solve.truncated and solve.result.truncated
    result = search(env, limit=0)
    assert result.actions is None and not result.truncated
    assert solve(env) is not None
    assert not solve.truncated


def test_sequential_levels_resolve_after_each_transition():
    specs = [generate(0, difficulty) for difficulty in (1, 2, 3)]
    assert all(spec is not None for spec in specs)
    env = Env([build_level(spec) for spec in specs])
    env.reset()
    for index in range(len(specs)):
        assert env.level_index == index
        assert env.levels_completed == index
        actions = solve(env)
        assert not solve.truncated
        assert_replay(env, actions)
    assert env.state == GameState.WIN


def test_bank_cli(tmp_path):
    path = tmp_path / "bank.jsonl"
    subprocess.run([sys.executable, "-m", "pebby.games.tr87.bank", "--levels", "3",
                    "--seed", "0", "--difficulty", "1", "--out", str(path)],
                   check=True, timeout=60, capture_output=True, text=True)
    specs = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(specs) == 3
    assert [spec["seed"] for spec in specs] == [0, 1, 2]
    for spec in specs:
        assert_replay(env_for(spec), spec["solution"])
