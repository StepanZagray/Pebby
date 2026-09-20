"""Bounded all-tier VC33 quality and split-diversity audit."""

from collections import Counter, defaultdict
import json
import time

from pebby.games.vc33 import names
from pebby.games.vc33.env import Env
from pebby.games.vc33.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    SPLITS,
    build_level,
    generate,
    validate_full_standard,
)


def _semantic_route(spec):
    env = Env([build_level(spec)])
    names_used = []
    before = env.levels_completed
    for action in spec["solution"]:
        point = env.game.camera.display_to_grid(action[1], action[2])
        sprite = env.level.get_sprite_at(*point)
        if names.TAG_BUTTON in sprite.tags:
            names_used.append(sprite.name.rsplit("generated-vc33-", 1)[-1])
        elif names.TAG_SWAP in sprite.tags:
            names_used.append(sprite.name.rsplit("generated-vc33-", 1)[-1])
        else:
            names_used.append("other")
        observation = env.perform(*action)
        if env.levels_completed > before or observation.won:
            break
    return tuple(names_used)


def test_bounded_all_tier_all_split_quality_sample():
    """Audit 21 rows: every tier once in each canonical split."""
    started = time.monotonic()
    rows = []
    rejection_counts = Counter()
    route_signatures = defaultdict(set)
    for split_index, split in enumerate(SPLITS):
        for difficulty in DIFFICULTIES:
            seed = 10_000 + split_index * 100 + difficulty
            spec = generate(seed, difficulty, split=split)
            assert spec is not None, generate.last_report
            assert validate_full_standard(
                spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            ) == ()
            assert json.loads(json.dumps(spec)) == spec
            rows.append(spec)
            rejection_counts.update(spec["generation"]["rejections"])
            route_signatures[difficulty].add(_semantic_route(spec))

    assert time.monotonic() - started < 45
    assert len(rows) == 21
    assert len({row["geometry_d4_sha256"] for row in rows}) == len(rows)
    assert len({row["gameplay_sha256"] for row in rows}) == len(rows)
    assert len({row["effective_seed"] for row in rows}) == len(rows)
    assert all(len(route_signatures[difficulty]) >= 2 for difficulty in DIFFICULTIES)

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        assert {row["difficulty"] for row in split_rows} == set(DIFFICULTIES)
        assert all(row["geometry_split"] == split for row in split_rows)

    traces = [row["proof"]["mechanic_use"] for row in rows]
    assert all(trace["distinct_buttons"] > 0 for trace in traces)
    assert all(
        row["proof"]["mechanic_use"]["distinct_swaps"] > 0
        for row in rows if row["difficulty"] >= 4
    )
    assert all(
        row["proof"]["mechanic_use"]["floor_limit_contacts"] == row["metrics"]["floors"]
        for row in rows if row["difficulty"] >= 6
    )
    assert all(
        row["proof"]["mechanic_use"]["coupled_multi_load_transfers"] > 0
        for row in rows if row["difficulty"] == 7
    )
    assert sum(row["generation"]["attempts_used"] for row in rows) >= len(rows)
    assert rejection_counts["geometry_split_mismatch"] > 0
