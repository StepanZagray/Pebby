"""Bounded all-tier generated-quality audit for R11L."""

from collections import Counter, defaultdict
import time

from pebby.games.r11l.generate import DIFFICULTIES, generate, profile_errors


def test_modest_all_tier_quality_audit():
    started = time.monotonic()
    rows = []
    rejection_reasons = Counter()
    lengths = defaultdict(list)
    for difficulty in DIFFICULTIES:
        for sample, split in enumerate(("train", "validation", "test")):
            row = generate(20_000 + difficulty * 100 + sample, difficulty, split=split)
            assert row is not None, generate.last_report
            assert profile_errors(row) == []
            rows.append(row)
            lengths[difficulty].append(row["solution_length"])
            rejection_reasons.update(row["generation_exclusions"])

    assert len(rows) == 18
    assert len({row["geometry_d4_sha256"] for row in rows}) == 18
    assert all(len(lengths[difficulty]) == 3 for difficulty in DIFFICULTIES)
    assert all(
        0 < row["solution_mechanics"]["wall_constrained_witness_drags"]
        <= row["solution_mechanics"]["drag_actions"]
        for row in rows
    )
    assert all(
        0 < row["solution_mechanics"]["hazard_constrained_witness_drags"]
        <= row["solution_mechanics"]["drag_actions"]
        for row in rows if row["difficulty"] in (2, 3, 4)
    )
    assert all(
        "hazard_threatening_goal_moves_avoided" not in row["solution_mechanics"]
        and "wall_blocked_destinations_avoided" not in row["solution_mechanics"]
        for row in rows
    )
    assert all(
        row["solution_mechanics"]["useful_pickups_absorbed"] in (4, 6)
        and row["solution_mechanics"]["decoy_pickups_absorbed"] == 0
        for row in rows if row["difficulty"] in (5, 6)
    )
    assert time.monotonic() - started < 90
