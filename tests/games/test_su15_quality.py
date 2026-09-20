from collections import Counter
import time

from pebby.games.su15.generate import DIFFICULTIES, generate


def test_bounded_one_per_tier_per_split_correction_smoke():
    identities = {split: set() for split in ("train", "validation", "test")}
    for difficulty in DIFFICULTIES:
        for offset, split in enumerate(identities):
            spec = generate(70_000 + difficulty * 10 + offset, difficulty, split=split)
            assert spec is not None, generate.last_report
            assert spec["geometry_split"] == split
            assert spec["search_work_used"] <= spec["search_limit"]
            assert spec["solution_mechanics"]["won"] is True
            identities[split].add(spec["geometry_sha256"])
    assert identities["train"].isdisjoint(identities["validation"])
    assert identities["train"].isdisjoint(identities["test"])
    assert identities["validation"].isdisjoint(identities["test"])


def _normalized_actions(solution):
    clicks = [(x, y) for action, x, y in solution if action == 6]
    if not clicks:
        return tuple(tuple(value) for value in solution)
    origin_x, origin_y = clicks[0]
    return tuple((action, None if x is None else x - origin_x,
                  None if y is None else y - origin_y)
                 for action, x, y in solution)


def _accepted(difficulty, split, count, seed):
    rows = []
    candidate = seed
    while len(rows) < count and candidate < seed + 200:
        spec = generate(candidate, difficulty, split=split)
        if spec is not None:
            rows.append(spec)
        candidate += 1
    assert len(rows) == count
    return rows


def test_bounded_all_tier_quality_audit_has_puzzle_and_action_diversity():
    started = time.monotonic()
    reports = {}
    for difficulty in DIFFICULTIES:
        rows = _accepted(difficulty, "train", 8, 20_000 + difficulty * 1_000)
        geometry = {row["geometry_d4_sha256"] for row in rows}
        gameplay = {row["gameplay_sha256"] for row in rows}
        actions = {_normalized_actions(row["solution"]) for row in rows}
        # The tiny direct-placement tutorial is finite, but even it is not one
        # fixed route disguised by colors.  Later tiers vary merge/degrade
        # geometry and action semantics more broadly.
        floor = 6 if difficulty == 1 else 7
        assert len(geometry) >= floor
        assert len(gameplay) == len(geometry)
        assert len(actions) >= floor
        assert all(row["solution_mechanics"]["won"] for row in rows)
        reports[difficulty] = {
            "geometry": len(geometry), "gameplay": len(gameplay),
            "actions": len(actions),
            "lengths": Counter(row["solution_length"] for row in rows),
        }
    assert set(reports) == set(DIFFICULTIES)
    assert time.monotonic() - started < 120


def test_canonical_geometry_partitions_do_not_overlap_across_splits():
    identities = {split: set() for split in ("train", "validation", "test")}
    for difficulty in DIFFICULTIES:
        for offset, split in enumerate(identities):
            rows = _accepted(difficulty, split, 2,
                             50_000 + difficulty * 1_000 + offset * 200)
            identities[split].update(row["geometry_d4_sha256"] for row in rows)
    assert identities["train"].isdisjoint(identities["validation"])
    assert identities["train"].isdisjoint(identities["test"])
    assert identities["validation"].isdisjoint(identities["test"])
