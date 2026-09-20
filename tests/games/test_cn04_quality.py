import numpy as np

from pebby.games.cn04 import names
from pebby.games.cn04.generate import (
    _initial_foreground_colours,
    _rendered,
    generate,
)
from pebby.games.cn04.quality import audit_generated


def test_bounded_all_tier_all_split_quality_audit():
    report = audit_generated(seeds=(0,))
    assert report["requested"] == report["accepted"] == 18
    assert report["failures"] == []
    assert report["distinct_geometry_d4"] == 18
    assert report["distinct_gameplay"] == 18
    assert all(not any(overlap.values())
               for overlap in report["cross_split_overlap"].values())
    assert report["by_tier"]["5"]["mechanic_exercise"]["stack_action_rows"] == 3
    assert report["by_tier"]["6"]["mechanic_exercise"]["stack_action_rows"] == 3


def test_eight_seed_solution_and_initial_visual_diversity_per_tier():
    delta = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    by_delta = {value: key for key, value in delta.items()}
    transforms = [
        (swap, sx, sy)
        for swap in (False, True)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]

    def route_identity(solution):
        variants = []
        for swap, sx, sy in transforms:
            actions = []
            for action, _, _ in solution:
                if action in delta:
                    x, y = delta[action]
                    transformed = (
                        sx * (y if swap else x),
                        sy * (x if swap else y),
                    )
                    actions.append(by_delta[transformed])
                else:
                    # Selection coordinates are deliberately normalized away.
                    actions.append(action)
            variants.append(tuple(actions))
        return min(variants)

    for difficulty in range(1, 7):
        rows = [generate(seed, difficulty, split="train") for seed in range(8)]
        assert all(rows)
        assert len({route_identity(row["solution"]) for row in rows}) == 8
        assert len({row["geometry_d4_sha256"] for row in rows}) == 8
        assert len({row["gameplay_sha256"] for row in rows}) == 8
        assert len({
            row["solution_constraints"]["relation_geometry_sha256"]
            for row in rows
        }) >= 7
        if difficulty >= 2:
            assert len({
                row["solution_constraints"]["topology_sha256"]
                for row in rows
            }) >= 2
        if difficulty == 5:
            assert len({
                tuple(row["solution_constraints"]["winning_alternates"])
                for row in rows
            }) >= 3
        if difficulty == 6:
            assert len({
                tuple(row["solution_constraints"]["winning_alternates"])
                for row in rows
            }) >= 4
        for row in rows:
            occupied = set()
            for piece in (piece for piece in row["pieces"] if piece.get("visible", True)):
                rendered = _rendered(piece["pixels"], piece["rotation"])
                height, width = rendered.shape
                x, y = piece["x"], piece["y"]
                assert 0 <= x and x + width <= 20
                assert 0 <= y and y + height <= 20
                cells = {
                    (x + int(px), y + int(py))
                    for py, px in np.argwhere(rendered >= 0)
                }
                assert occupied.isdisjoint(cells)
                occupied.update(cells)
            foreground = _initial_foreground_colours(
                row["pieces"], row["grey_masking"],
            )
            assert row["background"] not in foreground
            assert names.PIN_A in foreground
            if difficulty <= 2:
                assert names.SELECTED_BODY in foreground
            else:
                assert names.GREY in foreground
            if difficulty >= 5:
                assert row["zero_markers"] >= sum(row["stack_sizes"])
