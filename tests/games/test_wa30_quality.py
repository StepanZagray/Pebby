"""Bounded all-tier diversity audit for the full WA30 generator."""

from collections import Counter

from pebby.games.wa30.generate import DIFFICULTIES, generate


def test_eight_seed_sample_per_tier_has_puzzle_and_action_diversity():
    rows = {
        difficulty: [generate(seed, difficulty) for seed in range(8)]
        for difficulty in DIFFICULTIES
    }
    for difficulty, tier in rows.items():
        assert all(tier), f"tier {difficulty} rejected a bounded sample seed"
        assert len({row["seed"] for row in tier}) == 8
        assert len({row["gameplay_sha256"] for row in tier}) == 8, difficulty
        # Closed dividers/chambers and corridor bands deliberately retain one
        # causal topology under translation/reflection, so D4 geometry is not
        # used as a proxy for semantic diversity. Ordered native assignments
        # and the independently certified action witnesses must still vary.
        assert len({row["geometry_d4_sha256"] for row in tier}) >= 2, difficulty
        assert len({row["action_sha256"] for row in tier}) >= 6, difficulty
        assert len({row["solution_length"] for row in tier}) >= (2 if difficulty == 1 else 3), difficulty

    mechanics = Counter()
    for tier in rows.values():
        for row in tier:
            mechanics.update(row["solution_mechanics"])
            assert all(value >= 0 for value in row["generation_exclusions"].values())
    for key in (
        "manual_grabs", "helper_grabs", "helper_deliveries", "fence_box_moves",
        "thief_grabs", "thief_bad_deliveries", "thieves_destroyed",
        "player_steals", "robot_steals", "thief_steals_from_helper",
    ):
        assert mechanics[key] > 0

    # The retained official tier-9 witness has zero steals, so holder stealing
    # is calibrated as a witnessed branch rather than invented as necessary in
    # every mixed-actor tier.  Tiers 6/7 supply player steals; tier 8 supplies
    # the ordered helper-then-thief transfer.
    assert all(row["solution_mechanics"]["player_steals"] >= 1 for row in rows[6])
    assert all(row["solution_mechanics"]["player_steals"] >= 1 for row in rows[7])
    assert all(row["solution_mechanics"]["thief_steals_from_helper"] >= 1 for row in rows[8])
