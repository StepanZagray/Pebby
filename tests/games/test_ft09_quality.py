"""Bounded all-tier distribution audit for the full FT09 generator."""

from collections import Counter

from pebby.games.ft09.generate import DIFFICULTIES, generate, replays_to_completion
from pebby.games.ft09.reference_profiles import PROFILES, profile_errors


def test_bounded_all_tier_quality_sample():
    rows = []
    rejections = []
    for difficulty in DIFFICULTIES:
        for sample, split in enumerate(("train", "validation", "test")):
            row = generate(
                50_000 + 100 * difficulty + sample,
                difficulty,
                attempts=100,
                split=split,
                record_rejection=rejections.append,
            )
            assert row is not None
            assert profile_errors(row) == []
            assert replays_to_completion(row)
            rows.append(row)

    assert len(rows) == 18
    assert len({row["gameplay_sha256"] for row in rows}) == len(rows)
    geometry_by_split = {
        split: {row["geometry_sha256"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert not (geometry_by_split["train"] & geometry_by_split["validation"])
    assert not (geometry_by_split["train"] & geometry_by_split["test"])
    assert not (geometry_by_split["validation"] & geometry_by_split["test"])
    assert all(row["generation_attempt"] <= 100 for row in rows)
    assert all(PROFILES[row["difficulty"]]["actions"][0]
               <= row["optimal_actions"]
               <= PROFILES[row["difficulty"]]["actions"][1] for row in rows)
    assert all(row["solution_mechanics"]["exercised_constraints"]
               == row["constraint_count"] for row in rows)
    assert all(row["solution_mechanics"]["third_colour_actions"] > 0
               for row in rows if row["difficulty"] == 4)
    assert all(row["solution_mechanics"]["distinct_special_cells_clicked"]
               == row["special_cell_count"]
               for row in rows if row["difficulty"] == 5)
    assert all(row["solution_mechanics"]["distinct_coupled_special_cells_clicked"] >= 5
               for row in rows if row["difficulty"] == 6)
    assert Counter(row["reason"] for row in rejections)  # diagnostics exercised by split gating
