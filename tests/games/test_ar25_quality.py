from collections import Counter

from pebby.games.ar25.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    REFERENCE_PROFILES,
    _coverage_paths,
    _draft,
    _occupied,
    build_game,
    generate,
    generate_game,
    validate_full_standard,
)
from pebby.games.ar25 import names
from pebby.games.ar25.env import Env


def test_bounded_all_tier_quality_sample():
    rows = []
    for difficulty in DIFFICULTIES:
        for offset in range(2):
            spec = generate(200_000 + difficulty * 10 + offset, difficulty, split="train")
            assert spec is not None
            assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]) == []
            rows.append(spec)
    assert len(rows) == 16
    assert len({row["geometry_d4_sha256"] for row in rows}) == len(rows)
    assert len({row["gameplay_sha256"] for row in rows}) == len(rows)
    by_tier = Counter(row["difficulty"] for row in rows)
    assert by_tier == Counter({difficulty: 2 for difficulty in DIFFICULTIES})
    assert all(row["solution_mechanics"]["reflected_goal_count"] > 0 for row in rows)
    assert all(
        row["difficulty"] < 5 or row["solution_mechanics"]["recursive_goal_count"] > 0
        for row in rows
    )


def test_rendered_composition_keeps_native_cues_and_calibrated_extents():
    specs = generate_game(888, split="train")
    levels = build_game(specs)
    env = Env(levels)
    env.reset()
    for index, spec in enumerate(specs):
        profile = REFERENCE_PROFILES[index + 1]
        assert tuple((len(row["mask"][0]), len(row["mask"])) for row in spec["shapes"]) == profile["shape_sizes"]
        assert all(row["color"] == 5 for row in spec["shapes"])
        xs, ys = [point[0] for point in spec["goals"]], [point[1] for point in spec["goals"]]
        bbox = (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
        assert profile["goal_bbox_range"][0][0] <= bbox[0] <= profile["goal_bbox_range"][0][1]
        assert profile["goal_bbox_range"][1][0] <= bbox[1] <= profile["goal_bbox_range"][1][1]
        mirrors = [(row["orientation"], row["initial_coordinate"]) for row in spec["mirrors"]]
        for row in spec["shapes"]:
            paths = _coverage_paths(tuple(row["initial_position"]), _occupied(row["mask"]), mirrors, row["reflection"])
            assert any(path for path in paths.values())
            if index + 1 >= 5:
                assert any(len(path) >= 2 for path in paths.values())
        env.set_level(index)
        frame = env.render()
        assert len(frame) == 64 and all(len(line) == 64 for line in frame)
        # Background, reflected coverage, shape, mirror and goal colors must
        # all remain visibly present in the actual native frame.
        assert {4, 5, 9, 10, 11} <= {value for line in frame for value in line}
        for shape in env.movables():
            assert all(
                0 <= shape.x + x < 21 and 0 <= shape.y + y < 21
                for y in range(shape.height)
                for x in range(shape.width)
                if int(shape.pixels[y, x]) != -1
            )


def test_semantic_solution_diversity_across_eight_seeds_per_tier():
    for difficulty in DIFFICULTIES:
        rows = [
            generate(400_000 + difficulty * 100 + offset, difficulty, split="train")
            for offset in range(8)
        ]
        assert all(rows)
        for row in rows:
            audit = row["alternative_solution_audit"]
            assert audit["method"] == "bounded-public-configuration-bfs-plus-native-replay"
            assert audit["max_actions"] == REFERENCE_PROFILES[difficulty]["witness_range"][0] - 1
            assert audit["shortcut_actions"] is None
            assert audit["work"] <= audit["work_limit"]
        action_programs = {
            tuple(action[0] for action in row["solution"])
            for row in rows
        }
        displacements = {
            (
                tuple(
                    mirror["target_coordinate"] - mirror["initial_coordinate"]
                    for mirror in row["mirrors"] if not mirror["fixed"]
                ),
                tuple(
                    (
                        shape["target_position"][0] - shape["initial_position"][0],
                        shape["target_position"][1] - shape["initial_position"][1],
                    )
                    for shape in row["shapes"]
                ),
            )
            for row in rows
        }
        assert len(action_programs) >= 7
        assert len(displacements) >= 7
        assert len({row["gameplay_sha256"] for row in rows}) == 8


def _four_connected_components(cells):
    """Independent measurement: official art and targets are polyominoes."""
    remaining, count = set(cells), 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    return count


def test_shapes_and_goals_are_official_polyomino_regions_per_tier():
    env = Env()
    env.reset()
    for index, difficulty in enumerate(DIFFICULTIES):
        profile = REFERENCE_PROFILES[difficulty]
        low, high = profile["goal_components"]
        assert low <= high

        # The pinned counts are measurements of the shipped levels themselves.
        env.set_level(index)
        official_shapes = tuple(
            _four_connected_components({
                (x, y)
                for y in range(shape.height)
                for x in range(shape.width)
                if int(shape.pixels[y, x]) != names.TRANSPARENT
            })
            for shape in env.movables()
        )
        official_goals = _four_connected_components(
            {(int(goal.x), int(goal.y)) for goal in env.goals()}
        )
        assert official_shapes == profile["shape_components"]
        assert low <= official_goals <= high

        # Every generated draft (before search/replay acceptance) already
        # carries the same polyomino structure, cell counts and extents.
        drafts, tried = 0, 0
        while drafts < 10:
            assert tried < 400, f"tier {difficulty} draft yield collapsed"
            spec = _draft(700_000 + difficulty * 1_000 + tried, difficulty, 0)
            tried += 1
            if spec is None:
                continue
            drafts += 1
            masks = [_occupied(shape["mask"]) for shape in spec["shapes"]]
            assert tuple(map(_four_connected_components, masks)) == profile["shape_components"]
            assert tuple(map(len, masks)) == profile["shape_cells"]
            assert tuple(
                (len(shape["mask"][0]), len(shape["mask"])) for shape in spec["shapes"]
            ) == profile["shape_sizes"]
            goals = {tuple(goal) for goal in spec["goals"]}
            assert len(goals) == profile["goals"]
            assert low <= _four_connected_components(goals) <= high
