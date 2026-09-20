"""Bounded generated-quality audit for every official SC25 tier."""

from collections import Counter
from functools import lru_cache

import numpy as np

from pebby.games.sc25 import names
from pebby.games.sc25.env import Env
from pebby.games.sc25.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    SPLITS,
    _official_geometry_hashes,
    build_level,
    generate,
    validate_full_standard,
)


@lru_cache(maxsize=1)
def quality_sample():
    rows = []
    for difficulty in DIFFICULTIES:
        for offset in range(8):
            row = generate(
                73000 + 100 * difficulty + offset,
                difficulty,
                split=SPLITS[offset % len(SPLITS)],
            )
            assert row is not None
            assert validate_full_standard(
                row, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            ) == []
            rows.append(row)
    return rows


def test_eight_seed_quality_sample_covers_every_tier_without_duplicates_or_copies():
    rows = quality_sample()
    assert len({row["geometry_d4_sha256"] for row in rows}) == len(rows)
    assert len({row["gameplay_sha256"] for row in rows}) == len(rows)
    assert not ({row["geometry_d4_sha256"] for row in rows} & _official_geometry_hashes())
    assert all(isinstance(row["generation_rejections"], dict) for row in rows)


def test_required_routes_have_per_tier_topology_and_action_sequence_diversity():
    rows = quality_sample()
    for difficulty in DIFFICULTIES:
        tier = [row for row in rows if row["difficulty"] == difficulty]
        assert len({row["geometry_d4_sha256"] for row in tier}) == 8
        normalized = {
            tuple(action_id for action_id, _x, _y in row["solution"])
            for row in tier
        }
        assert len(normalized) >= 6
    tier5_grammars = Counter(
        row["mechanic"] for row in rows if row["difficulty"] == 5
    )
    assert tier5_grammars["full-composition-small-first"] >= 1
    assert tier5_grammars["full-composition-large-first"] >= 1


def test_native_frames_keep_every_rule_cue_visible_and_inside_the_board():
    first_per_tier = {
        difficulty: next(
            row for row in quality_sample() if row["difficulty"] == difficulty
        )
        for difficulty in DIFFICULTIES
    }
    block_names = {"primary": names.SPRITE_BLOCK,
                   "alternate": names.SPRITE_BLOCK_ALT}
    target_names = {"primary": names.SPRITE_TARGET,
                    "alternate": names.SPRITE_TARGET_ALT}
    for difficulty, spec in first_per_tier.items():
        env = Env([build_level(spec) for _ in DIFFICULTIES])
        env.set_level(difficulty - 1)
        frame = np.asarray(env.render())
        assert frame.shape == (64, 64)
        assert len(np.unique(frame)) >= 8
        sprites = env.game.current_level.get_sprites()
        names_present = Counter(sprite.name for sprite in sprites)
        for spell in spec["spells"]:
            assert names_present[names.SPRITE_ICON_PREFIX + spell] == 1
        if spec["spells"]:
            assert names_present[names.SPRITE_GRID_PANEL] == 1
            assert names_present[names.SPRITE_GRID_CELL] == 9
        for family in ("primary", "alternate"):
            count = sum(item["family"] == family for item in spec["targets"])
            assert names_present[target_names[family]] == count
            assert names_present[block_names[family]] == count
        assert names_present[names.SPRITE_PICKUP] == len(spec["pickups"])
        assert names_present[names.SPRITE_TELEPORT_PAD] == len(spec["pads"])
        assert names_present[names.SPRITE_TELEPORT_PAD_SMALL] == len(spec["small_pads"])
        assert names_present[names.SPRITE_TELEPORT_INDICATOR] == bool(spec["pads"])
        assert names_present[names.SPRITE_TELEPORT_INDICATOR_SMALL] == bool(
            spec["small_pads"]
        )
        if difficulty == 1:
            assert spec["solution_mechanics"]["tutorial_demo_actions"] == 1
        for sprite in sprites:
            rendered = np.asarray(sprite.render())
            ys, xs = np.nonzero(rendered != -1)
            if not len(xs):
                continue
            opaque_box = (
                sprite.x + int(xs.min()), sprite.y + int(ys.min()),
                sprite.x + int(xs.max()), sprite.y + int(ys.max()),
            )
            assert 0 <= opaque_box[0] <= opaque_box[2] < 64
            assert 0 <= opaque_box[1] <= opaque_box[3] < 64


def test_late_tier_solution_certificates_exercise_full_composition():
    tier5 = generate(6105, 5, split="test")
    tier6 = generate(6106, 6, split="test")
    assert tier5 is not None and tier6 is not None
    for row in (tier5, tier6):
        mechanics = row["solution_mechanics"]
        assert mechanics["shrink_casts"] >= 1
        assert mechanics["grow_casts"] >= 1
        assert mechanics["small_teleports"] >= 1
        assert mechanics["large_teleports"] >= 1
        assert mechanics["primary_targets_hit"] >= 1
        assert mechanics["alternate_targets_hit"] >= 1
        assert mechanics["pickups_consumed"] >= 1
        assert mechanics["remaining_budget"] >= 0
    assert tier6["solution_mechanics"]["large_teleports"] >= 2
    assert tier6["solution_mechanics"]["pickups_consumed"] == 2


def test_tier6_cohort_has_two_native_target_clear_then_pickup_dependencies():
    tier6 = [row for row in quality_sample() if row["difficulty"] == 6]
    assert len(tier6) == 8
    for row in tier6:
        target_positions = {(item["x"], item["y"]) for item in row["targets"]}
        pickup_positions = {(item["x"], item["y"]) for item in row["pickups"]}
        assert target_positions == pickup_positions
        assert row["solution_mechanics"]["target_covered_pickups"] == 2
        assert row["solution_mechanics"]["target_clear_before_pickup_pairs"] == 2
        causality = row["solution_causality"]
        assert causality["native_dependencies_verified"] == 2
        assert len(causality["pairs"]) == 2
        for pair in causality["pairs"]:
            assert pair["target_clear_action"] < pair["pickup_collect_action"]
            assert pair["counterfactual_target_present"]
            assert pair["counterfactual_pickup_present"]
            assert pair["counterfactual_progress_blocked"]
