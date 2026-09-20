"""SB26 full-grammar teacher, generation, replay, and pipeline tests."""

import copy
import json

from arcengine import GameState

from pebby.games.sb26 import names
from pebby.games.sb26.env import Env, official_levels
from pebby.games.sb26.generate import (
    DIFFICULTIES,
    FORMAT,
    FULL_STANDARD_CONTRACT,
    _child_seed,
    build_game,
    build_level,
    generate,
    generate_game,
    identity_hashes,
    validate_full_standard,
)
from pebby.games.sb26.layout import extract
from pebby.games.sb26.plan import assignment_from_layout, search, solve, traverse
from pebby.games.sb26.reference_profiles import REFERENCE_PROFILES
from pebby.multigame import (
    MultiGameEnv,
    SearchLimits,
    collect_generated_game,
    preflight,
)


def checked_replay(env, actions):
    assert actions
    start_score = env.levels_completed
    observation = None
    for index, (action_id, x, y) in enumerate(actions):
        assert action_id in env.available_actions
        if action_id == names.ACTION_CLICK:
            assert type(x) is type(y) is int
            assert 0 <= x < 64 and 0 <= y < 64
        else:
            assert x is None and y is None
        observation = env.perform(action_id, x, y)
        if env.levels_completed > start_score:
            assert index == len(actions) - 1
            return True, observation
        assert observation.state != GameState.GAME_OVER
    return False, observation


def test_full_standard_contract_is_root_reviewed_ready():
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert FULL_STANDARD_CONTRACT["evidence"]["independent_closure"] == (
        ".scratch/multigame-resume/full-standard/external-astra-re86-sb26/"
        "sb26-closure-detailed.md"
    )
    assert FULL_STANDARD_CONTRACT["evidence"]["enriched_provenance"] == (
        "tests/games/test_sb26.py::"
        "test_generate_game_is_exactly_eight_increasing_contexts_and_build_replays_it"
    )
    assert any("finite generated support of 23" in caveat
               for caveat in FULL_STANDARD_CONTRACT["caveats"])


def test_all_eight_official_levels_have_full_teacher_and_sequential_native_witnesses():
    first, second = official_levels(), official_levels()
    assert len(first) == len(second) == 8
    assert all(a is not b for a, b in zip(first, second))
    env = Env()
    frame = env.reset()
    assert len(frame) == 64 and all(len(row) == 64 for row in frame)
    assert env.available_actions == names.ACTION_IDS

    for difficulty in DIFFICULTIES:
        assert env.level_index == difficulty - 1
        layout = extract(env)
        assert layout.exact, layout.unsupported
        result = search(env, limit=32, node_limit=500_000)
        assert result.solved and result.exact
        assert not result.truncated and not result.unsupported
        assert result.length == REFERENCE_PROFILES[difficulty]["reference_actions"]
        mechanics = result.traversal.metadata()
        for field, expected in REFERENCE_PROFILES[difficulty]["required_use"].items():
            assert mechanics[field] == expected
        completed, observation = checked_replay(env, result.actions)
        assert completed and env.levels_completed == difficulty
    assert observation.won


def test_clone_display_click_selection_move_undo_and_live_replanning():
    spec = generate(21, 4, split="train")
    assert spec is not None
    env = Env([build_level(spec)])
    frame = env.reset()
    assert all(type(pixel) is int and 0 <= pixel <= 15 for row in frame for pixel in row)
    clone = env.clone()
    initial = extract(clone)
    route = solve(clone)
    clone.perform(*route[0])
    assert extract(clone).selected is not None
    assert extract(env).selected is None
    live = search(clone)
    assert live.solved and checked_replay(clone, live.actions)[0]

    moved = Env([build_level(spec)])
    moved.reset()
    before = extract(moved)
    moved.perform(*spec["solution"][0])
    moved.perform(*spec["solution"][1])
    after = extract(moved)
    assert after.history_depth == before.history_depth + 1
    assert after.energy == before.energy - 1
    moved.perform(names.ACTION_UNDO)
    undone = extract(moved)
    assert undone.history_depth == before.history_depth
    assert undone.energy == after.energy  # native undo deliberately does not refund energy
    assert checked_replay(moved, solve(moved))[0]

    for action in ((1, None, None), (6, None, None), (6, 64, 1), (5, 1, None)):
        try:
            env.perform(*action)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action {action}")


def test_all_tiers_and_splits_are_deterministic_json_roundtrip_and_fail_closed():
    seen = {split: set() for split in ("train", "validation", "test")}
    for difficulty in DIFFICULTIES:
        for split in seen:
            seed = 809 if (difficulty, split) == (8, "validation") else 800 + difficulty
            spec = generate(seed, difficulty, split=split)
            assert spec is not None, generate.last_report
            duplicate = generate(seed, difficulty, split=split)
            assert duplicate == spec
            stored = json.loads(json.dumps(spec))
            assert stored == spec
            assert spec["format"] == FORMAT
            assert spec["difficulty"] == difficulty
            assert spec["context_index"] == difficulty - 1
            assert spec["split"] == spec["geometry_split"] == split
            assert not spec["omitted_mechanics"]
            assert validate_full_standard(
                stored, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            ) == []
            seen[split].add(spec["gameplay_sha256"])
    assert not (seen["train"] & seen["validation"])
    assert not (seen["train"] & seen["test"])
    assert not (seen["validation"] & seen["test"])


def test_validator_rejects_geometry_route_mechanic_partition_and_proof_tampering():
    spec = generate(901, 8, split="test")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][7]
    changes = []

    bad = copy.deepcopy(spec)
    bad["frames"][0]["position"][0] += 1
    changes.append(bad)
    bad = copy.deepcopy(spec)
    bad["solution"][0][1] += 1
    changes.append(bad)
    bad = copy.deepcopy(spec)
    bad["solution_mechanics"]["cycle_reentries"] = 0
    changes.append(bad)
    bad = copy.deepcopy(spec)
    bad["split"] = "train"
    changes.append(bad)
    bad = copy.deepcopy(spec)
    bad["proof"]["context_index"] = 0
    changes.append(bad)
    bad = copy.deepcopy(spec)
    bad["geometry_sha256"] = "0" * 64
    changes.append(bad)

    for tampered in changes:
        assert validate_full_standard(tampered, entry)


def test_validator_rejects_exact_type_presence_cap_energy_and_provenance_mutations():
    spec = generate(0, 1, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]

    mutations = []

    def changed(path, value):
        row = copy.deepcopy(spec)
        target = row
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        mutations.append(row)

    for path, value in (
        (("proof", "engine_win"), 1),
        (("proof", "levels_completed"), True),
        (("context_index",), False),
        (("native_budget",), 64.0),
        (("engine_win",), False),
        (("teacher_model_exact",), False),
        (("search_truncated",), True),
        (("search_unsupported",), True),
        (("solution_energy_cost",), 999),
        (("generation_limits", "action_limit"), 0),
        (("generation_limits", "attempts"), 10_001),
        (("generation_exclusions",), {"imaginary": 999}),
        (("proof", "search_limit"), 500_000.0),
        (("context_solution", 0, 0), 6.0),
        (("frames", 0, "arity"), True),
        (("tray", 0, "colour"), 8.0),
        (("goals", 0), True),
        (("goal_positions", 0, 0), 1.0),
        (("solution_mechanics", "energy_cost"), float(spec["solution_energy_cost"])),
        (("proof", "generator_version"), 2),
    ):
        changed(path, value)

    for path in (("engine_win",), ("proof", "native_budget"),
                 ("generation_limits", "node_limit")):
        row = copy.deepcopy(spec)
        target = row
        for part in path[:-1]:
            target = target[part]
        del target[path[-1]]
        mutations.append(row)

    row = copy.deepcopy(spec)
    row["proof"]["unexpected"] = True
    mutations.append(row)
    row = copy.deepcopy(spec)
    row["proof"]["sequential_context_index"] = 0
    mutations.append(row)
    row = copy.deepcopy(spec)
    key = next(iter(row["generation_exclusions"]), "geometry_split")
    row["generation_exclusions"][key] = row["generation_exclusions"].get(key, 0) + 1
    row["generation_rejections"] = copy.deepcopy(row["generation_exclusions"])
    mutations.append(row)

    for index, tampered in enumerate(mutations):
        assert validate_full_standard(tampered, entry), index

    tier_two = generate(0, 2, split="train")
    assert tier_two is not None
    for path, value in (
        (("fixed", 0, "frame"), False),
        (("connections", 0, "to_frame"), 1.0),
    ):
        row = copy.deepcopy(tier_two)
        target = row
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        assert validate_full_standard(row, FULL_STANDARD_CONTRACT["curriculum"][1])


def test_semantic_identity_excludes_layout_and_unrelated_palette_but_preserves_links():
    def identities(row):
        env = Env([build_level(row)])
        frame = env.reset()
        return identity_hashes(extract(env), frame)

    base = generate(4_104, 4, split="train")
    assert base is not None
    left = copy.deepcopy(base)
    left["frames"][0]["position"] = [12, 18]
    left["frames"][1]["position"] = [24, 33]
    right = copy.deepcopy(base)
    right["frames"][0]["position"] = [18, 18]
    right["frames"][1]["position"] = [18, 33]
    left_ids, right_ids = identities(left), identities(right)
    for field in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256", "geometry_split"):
        assert left_ids[field] == right_ids[field]
    assert left_ids["raw_start_frame_sha256"] != right_ids["raw_start_frame_sha256"]

    tutorial = generate(4_101, 1, split="train")
    assert tutorial is not None
    palette = [colour for colour in names.COLOURS if colour != tutorial["frames"][0]["colour"]][:4]
    old_colours = list(dict.fromkeys(tile["colour"] for tile in tutorial["tray"]))
    base_map = dict(zip(old_colours, palette))
    recoloured = copy.deepcopy(tutorial)
    for tile in recoloured["tray"]:
        tile["colour"] = base_map[tile["colour"]]
    recoloured["goals"] = [base_map[colour] for colour in recoloured["goals"]]
    collision = copy.deepcopy(recoloured)
    replaced = palette[0]
    border = tutorial["frames"][0]["colour"]
    for tile in collision["tray"]:
        if tile["colour"] == replaced:
            tile["colour"] = border
    collision["goals"] = [border if colour == replaced else colour for colour in collision["goals"]]
    recoloured_ids, collision_ids = identities(recoloured), identities(collision)
    assert recoloured_ids["geometry_sha256"] == collision_ids["geometry_sha256"]
    assert recoloured_ids["gameplay_sha256"] == collision_ids["gameplay_sha256"]
    assert recoloured_ids["raw_start_frame_sha256"] != collision_ids["raw_start_frame_sha256"]

    cyclic = generate(8_108, 8, split="train")
    assert cyclic is not None
    relabelled = copy.deepcopy(cyclic)
    old_target = relabelled["frames"][1]["colour"]
    used_frames = {frame["colour"] for frame in relabelled["frames"]}
    new_target = next(colour for colour in names.COLOURS if colour not in used_frames)
    relabelled["frames"][1]["colour"] = new_target
    for tile in relabelled["tray"]:
        if tile["kind"] == "link" and tile["colour"] == old_target:
            tile["colour"] = new_target
    assert identities(cyclic)["gameplay_sha256"] == identities(relabelled)["gameplay_sha256"]
    retargeted = copy.deepcopy(cyclic)
    link = next(tile for tile in retargeted["tray"] if tile["kind"] == "link")
    link["colour"] = next(frame["colour"] for frame in retargeted["frames"]
                          if frame["colour"] != link["colour"])
    assert identities(cyclic)["gameplay_sha256"] != identities(retargeted)["gameplay_sha256"]


def test_tutorial_semantics_are_finite_and_natural_cross_split_leak_is_rejected():
    from collections import Counter
    from itertools import permutations

    def identities(row):
        env = Env([build_level(row)])
        frame = env.reset()
        return identity_hashes(extract(env), frame)

    train = generate(0, 1, split="train")
    test = generate(2, 1, split="test")
    assert train is not None and test is not None

    def goal_to_tray(row):
        return tuple(next(index for index, tile in enumerate(row["tray"])
                          if tile["colour"] == goal) for goal in row["goals"])

    leaked_order = (3, 2, 0, 1)
    assert not (goal_to_tray(train) == goal_to_tray(test) == leaked_order)
    assert train["gameplay_sha256"] != test["gameplay_sha256"]

    base = copy.deepcopy(train)
    positions = [tile["position"] for tile in base["tray"]]
    colours = [tile["colour"] for tile in base["tray"]]
    semantic_ids = set()
    split_counts = Counter()
    for ordering in permutations(colours):
        row = copy.deepcopy(base)
        row["tray"] = [
            {"kind": "regular", "colour": colour, "position": position}
            for colour, position in zip(ordering, positions)
        ]
        row_ids = identities(row)
        semantic_ids.add(row_ids["gameplay_sha256"])
        split_counts[row_ids["geometry_split"]] += 1
    assert len(semantic_ids) == 24
    assert split_counts == {"train": 9, "validation": 11, "test": 4}

    official_env = Env([official_levels()[0]])
    official_frame = official_env.reset()
    official_ids = identity_hashes(extract(official_env), official_frame)
    assert official_ids["gameplay_sha256"] in semantic_ids
    assert official_ids["geometry_split"] == "validation"


def test_tier_eight_exercises_native_cycle_guard_then_undo_recovers():
    spec = generate(1_808, 8, split="train")
    assert spec is not None
    env = Env([build_level(spec)])
    env.reset()
    layout = extract(env)
    root, child = layout.frames
    links = {tile.colour: tile for tile in layout.movable_tiles if tile.kind == "link"}
    regulars = [tile for tile in layout.movable_tiles if tile.kind == "regular"]
    destinations = [
        (root.slots[0], links[child.colour]),
        (child.slots[0], links[root.colour]),
    ]
    free_slots = list(root.slots[1:] + child.slots[1:])
    destinations.extend(zip(free_slots, regulars))
    bad_actions = []
    for destination, tile in destinations:
        bad_actions.extend([
            (names.ACTION_CLICK, *names.click_at(tile.position)),
            (names.ACTION_CLICK, *names.click_at(destination)),
        ])
    for action in bad_actions:
        env.perform(*action)
    filled = extract(env)
    rejected = traverse(filled, assignment_from_layout(filled))
    assert not rejected.won and rejected.cycle_guard_rejections == 1
    before_submit = env.energy
    observation = env.perform(names.ACTION_SUBMIT)
    assert not observation.won and env.levels_completed == 0
    assert env.energy == before_submit - 1

    history = env.history_depth
    env.perform(names.ACTION_UNDO)
    assert env.history_depth == history - 1
    assert env.energy == before_submit - 1
    recovered = search(env, limit=64, node_limit=500_000)
    assert recovered.solved and checked_replay(env, recovered.actions)[0]


def test_generate_game_is_exactly_eight_increasing_contexts_and_build_replays_it():
    specs = generate_game(1_234, split="validation")
    assert specs is not None, generate_game.last_report
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["context_index"] for spec in specs] == list(range(8))
    assert all(spec["sequence_kind"] == "full-official-context" for spec in specs)
    assert all(spec["child_seed"] == spec["requested_seed"] for spec in specs)
    levels = build_game(specs)
    assert len(levels) == 8

    forged = copy.deepcopy(specs)
    for spec in forged:
        spec["game_seed"] = 5_678
        spec["child_seed"] = _child_seed(5_678, spec["difficulty"])
        assert spec["child_seed"] != spec["requested_seed"]
    try:
        build_game(forged)
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError("build_game accepted consistently relabelled seed provenance")

    standalone = [
        generate(spec["requested_seed"], difficulty, split="validation")
        for difficulty, spec in zip(DIFFICULTIES, specs)
    ]
    assert all(spec is not None and "game_seed" not in spec for spec in standalone)
    assert len(build_game(standalone)) == 8

    smoke = generate_game(55, split="train", difficulties=(2, 5, 8))
    assert smoke is not None and all(row["sequence_kind"] == "explicit-smoke-subset" for row in smoke)
    try:
        build_game(smoke)
    except ValueError as exc:
        assert "exactly eight" in str(exc)
    else:
        raise AssertionError("build_game accepted a shortened curriculum")


def test_shared_collector_uses_public_plan_search_for_complete_game():
    modules, = preflight(["sb26"])
    collected = collect_generated_game(
        modules,
        master_seed=91,
        game_index=0,
        difficulties=DIFFICULTIES,
        limits=SearchLimits(max_actions_per_level=64, max_search_work=500_000),
        outer_generation_attempts=2,
        generator_attempts=96,
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_completed"] == 8
    assert collected.record["teacher_steps"] == sum(
        REFERENCE_PROFILES[difficulty]["reference_actions"] for difficulty in DIFFICULTIES
    )


def test_concurrent_official_and_generated_construction_stays_isolated():
    from concurrent.futures import ThreadPoolExecutor

    spec = generate(55, 3, split="train")
    assert spec is not None

    def inventory(generated):
        env = Env([build_level(spec)]) if generated else Env()
        env.reset()
        layout = extract(env)
        return env.level_count, len(layout.frames), len(layout.slots)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inventory, (False, True, False, True) * 4))
    assert results[::2] == [(8, 1, 4)] * 8
    assert results[1::2] == [(1, 3, 9)] * 8
