"""Reference calibration, presentation, and bounded SB26 quality audits."""

import time

from pebby.games.sb26 import names
from pebby.games.sb26.env import Env
from pebby.games.sb26.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    SOURCE_SHA256,
    _official_hashes,
    _structural_metrics,
    build_level,
    generate,
    validate_full_standard,
)
from pebby.games.sb26.layout import extract
from pebby.games.sb26.plan import search
from pebby.games.sb26.reference_profiles import REFERENCE_PROFILES


def test_official_reference_characterization_matches_source_and_teacher():
    assert SOURCE_SHA256 == "dbb4877853a8d30f84e28d26f4a3d6ad7d2d1018602e82ccb5a62284043984f3"
    env = Env()
    env.reset()
    for difficulty in DIFFICULTIES:
        profile = REFERENCE_PROFILES[difficulty]
        layout = extract(env)
        metrics = _structural_metrics(layout, env.render())
        assert tuple(metrics["frame_arities"]) == profile["frame_arities"]
        for field in (
            "connector_count", "fixed_regular", "fixed_links", "movable_regular", "movable_links",
            "goals", "distinct_regular_colours",
        ):
            assert metrics[field] == profile[field]
        assert abs(metrics["visual_density"] - profile["reference_density"]) <= 0.0001
        result = search(env, limit=32, node_limit=500_000)
        assert result.solved and result.length == profile["reference_actions"]
        for action in result.actions:
            env.perform(*action)


def test_eight_semantically_distinct_generated_rows_per_tier_with_visual_rule_cues():
    # Tier 1 has exactly 24 semantic colour-order permutations (9 train,
    # 11 validation, 4 test under v2 identity); one validation row is the
    # rejected official identity. These cover 8/9 admissible train rows.
    tier_one_seeds = (0, 1, 2, 4, 5, 8, 10, 12)
    started = time.monotonic()
    official_geometry, official_gameplay, official_frames = _official_hashes()
    total_rejections = 0
    for difficulty in DIFFICULTIES:
        seeds = tier_one_seeds if difficulty == 1 else tuple(
            10_000 + difficulty * 100 + offset for offset in range(8)
        )
        rows = []
        for seed in seeds:
            row = generate(seed, difficulty, split="train")
            assert row is not None, generate.last_report
            assert validate_full_standard(
                row, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            ) == []
            assert row["geometry_sha256"] not in official_geometry
            assert row["gameplay_sha256"] not in official_gameplay
            assert row["raw_start_frame_sha256"] not in official_frames
            assert row["generation_attempt"] <= 96
            total_rejections += sum(row["generation_rejections"].values())

            env = Env([build_level(row)])
            frame = env.reset()
            layout = extract(env)
            assert layout.connector_count == REFERENCE_PROFILES[difficulty]["connector_count"]
            assert row["structural_metrics"]["visible_bbox"] == [0, 0, 63, 60]
            assert set(frame[53]) == {2}  # full energy is an unbroken visible rule cue
            assert all(frame[frame_data.position[1]][frame_data.position[0]] == frame_data.colour
                       for frame_data in layout.frames)
            for tile in layout.tiles:
                if tile.kind == "link":
                    x, y = tile.position
                    assert frame[y + 1][x + 1] == tile.colour
                    assert frame[y + 2][x + 2] == 4  # hollow centre distinguishes links
            rows.append(row)

        assert len({row["gameplay_sha256"] for row in rows}) == 8
        assert len({row["geometry_sha256"] for row in rows}) == 8
        assert len({row["action_sequence_sha256"] for row in rows}) == 8
    assert total_rejections > 0  # canonical partition/profile rejections are retained
    assert time.monotonic() - started < 60


def test_mechanic_progression_uses_actual_winning_routes_not_installed_decorations():
    rows = [generate(30_000 + difficulty, difficulty, split="validation")
            for difficulty in DIFFICULTIES]
    assert all(row is not None for row in rows)
    for difficulty, row in zip(DIFFICULTIES, rows):
        use = row["solution_mechanics"]
        profile = REFERENCE_PROFILES[difficulty]
        assert use["placements"] == profile["movable_regular"] + profile["movable_links"]
        assert row["solution_length"] == 2 * use["placements"] + 1
        assert use["regular_visits"] == profile["goals"]
        for field, expected in profile["required_use"].items():
            assert use[field] == expected
    assert rows[4]["solution_mechanics"]["repeated_frame_entries"] == 1
    assert rows[5]["solution_mechanics"]["distinct_frames_entered"] == 4
    assert rows[6]["solution_mechanics"]["maximum_depth"] == 3
    assert rows[7]["solution_mechanics"]["cycle_reentries"] == 1
