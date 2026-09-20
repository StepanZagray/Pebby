"""Full-standard S5I5 generation and native-engine certificate tests."""

from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from arcengine import GameState
import pytest

import pebby
from pebby.games.s5i5 import names
from pebby.games.s5i5.bank import load as load_bank
from pebby.games.s5i5.env import Env, official_levels
from pebby.games.s5i5.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    _solution_mechanics,
    _greedy_native_reduction,
    _pin_path_rods,
    _first_native_completion,
    _draft,
    build_game,
    build_level,
    generate,
    generate_game,
    solution_actions,
    validate_full_standard,
    obstacle_sprite,
)
from pebby.games.s5i5.generation_quality import gameplay_hash, solution_semantic_hash
from pebby.games.s5i5.layout import extract
from pebby.games.s5i5.plan import _FastEngine, search
from pebby.games.s5i5.quality_audit import obstacle_pixel_metrics
from pebby.games.s5i5.reference_profiles import PROFILES, REFERENCE_CHARACTERIZATION


def play(env, actions):
    observation = None
    for action in actions:
        assert len(action) == 3
        action_id, x, y = action
        assert action_id == names.ACTION_CLICK
        assert action_id in env.available_actions
        assert 0 <= x < names.FRAME and 0 <= y < names.FRAME
        observation = env.perform(action_id, x, y)
    return observation


@pytest.fixture(scope="module")
def generated_game():
    repo_root = Path(__file__).resolve().parents[2]
    assert Path(pebby.__file__).resolve().is_relative_to(repo_root)
    specs = generate_game(888, split="train")
    assert specs is not None
    return specs


@pytest.fixture(scope="module")
def generated_tier_one():
    spec = generate(0, 1, split="train")
    assert spec is not None
    return spec


@pytest.fixture(scope="module")
def official_teacher_routes():
    """Exercise every official context once; tier 7 is the long bounded case."""
    env = Env()
    routes = []
    for index, difficulty in enumerate(DIFFICULTIES):
        assert env.level_index == index
        before = extract(env).key()
        result = search(env, max_nodes=PROFILES[difficulty]["search_work"])
        assert extract(env).key() == before
        assert result.actions, result.reason
        assert result.exact is True
        assert result.optimal is (difficulty <= 6)
        routes.append(list(result.actions))
        observation = play(env, result.actions)
        assert env.levels_completed == index + 1
        assert observation.state != GameState.GAME_OVER
    assert env.state == GameState.WIN
    return routes


def test_contract_covers_authoritative_official_curriculum():
    assert len(official_levels()) == 8
    assert DIFFICULTIES == tuple(range(1, len(official_levels()) + 1))
    assert FULL_STANDARD_CONTRACT["format"] == "pebby-full-generator-contract-v1"
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert FULL_STANDARD_CONTRACT["source_id"] == "s5i5-18d95033"
    assert FULL_STANDARD_CONTRACT["curriculum"] == [
        {"difficulty": difficulty, "context_index": difficulty - 1,
         "search_work": PROFILES[difficulty]["search_work"]}
        for difficulty in DIFFICULTIES
    ]


def test_all_official_tiers_have_native_positive_teacher_coverage(official_teacher_routes):
    lengths = [len(route) for route in official_teacher_routes]
    assert lengths[:6] == [13, 26, 37, 30, 28, 25]
    assert lengths[6:] == [45, 38]


def test_teacher_continues_from_a_live_late_tier_prefix(official_teacher_routes):
    env = Env()
    env.set_level(7)
    prefix = official_teacher_routes[7][:5]
    assert play(env, prefix).state == GameState.NOT_FINISHED
    before = extract(env).key()
    result = search(env, max_nodes=PROFILES[8]["search_work"])
    assert extract(env).key() == before
    assert result.actions and result.exact and result.optimal is False
    assert play(env, result.actions).state == GameState.WIN


def test_generated_specs_round_trip_validate_and_replay_in_declared_contexts(generated_game):
    assert len(generated_game) == 8
    for index, spec in enumerate(generated_game):
        restored = json.loads(json.dumps(spec))
        assert restored == spec
        assert spec["difficulty"] == index + 1
        assert spec["context_index"] == index
        assert spec["solution_length"] == len(spec["solution"])
        assert spec["proof"]["budget_remaining"] == (
            spec["step_counter"] - spec["solution_length"]
        )
        assert validate_full_standard(
            restored, FULL_STANDARD_CONTRACT["curriculum"][index]
        ) == []

        levels = [build_level(restored) for _ in range(index + 1)]
        env = Env(levels)
        env.set_level(index)
        before = env.levels_completed
        observation = play(env, solution_actions(restored))
        assert env.levels_completed == before + 1
        assert observation.state != GameState.GAME_OVER


def test_complete_generated_game_replays_all_native_contexts(generated_game):
    levels = build_game(generated_game)
    assert len(levels) == 8
    env = Env(levels)
    for index, spec in enumerate(generated_game):
        assert env.level_index == index
        observation = play(env, solution_actions(spec))
        assert env.levels_completed == index + 1
        assert observation.state != GameState.GAME_OVER
    assert env.state == GameState.WIN


def test_generated_routes_cover_each_tiers_required_mechanics(generated_game):
    for spec in generated_game:
        profile = PROFILES[spec["difficulty"]]
        evidence = spec["solution_mechanics"]
        assert all(evidence[kind] for kind in profile["required_kinds"])
        assert evidence["distinct_controls"] >= profile["distinct_controls"]
        assert evidence["all_actions_changed"] is True
        assert evidence["pin_motion_actions"] > 0
        for requirement, counter in (
            ("require_vertical", "vertical_rail_actions"),
            ("require_linked", "linked_actions"),
            ("require_shared", "shared_color_actions"),
            ("require_branch", "branch_actions"),
        ):
            if profile.get(requirement):
                assert evidence[counter] > 0
        collision = spec["collision_rollback"]
        assert collision["native_state_restored"] is True
        assert collision["changed_prefix_actions"] > 0


@pytest.mark.parametrize(
    ("difficulty", "shortcut"),
    [
        (4, [(names.ACTION_CLICK, 12, 57)] * 2),
        (8, [(names.ACTION_CLICK, 8, 44)] * 3
             + [(names.ACTION_CLICK, 10, 51)] * 3),
    ],
)
def test_seed_zero_rejects_reported_native_shortcuts(difficulty, shortcut):
    spec = generate(0, difficulty, split="train")
    assert spec is not None
    env = Env([build_level(spec) for _ in range(difficulty)])
    env.set_level(difficulty - 1)
    before = env.levels_completed
    play(env, shortcut)
    assert env.levels_completed == before


def test_teacher_handles_an_already_aligned_late_context(generated_game):
    aligned = deepcopy(generated_game[7])
    aligned["targets"] = deepcopy(aligned["pins"])
    env = Env([build_level(aligned) for _ in range(8)])
    env.set_level(7)
    assert env.is_won_position()
    result = search(env, max_nodes=100)
    assert result.actions == [(names.ACTION_CLICK, 0, 0)]
    assert result.exact is True and result.optimal is True
    assert play(env, result.actions).state == GameState.WIN


def test_generated_frames_keep_gameplay_and_controls_visible(generated_game):
    levels = [build_level(spec) for spec in generated_game]
    env = Env(levels)
    for index, spec in enumerate(generated_game):
        env.set_level(index)
        frame = np.asarray(env.render(), dtype=np.int16)
        assert frame.shape == (64, 64)
        assert np.all((0 <= frame) & (frame < 16))
        assert np.any(frame[:41] != names.BACKGROUND_COLOR)
        assert np.any(frame[41:63] != names.BACKGROUND_COLOR)
        assert np.count_nonzero(frame[:41] == names.PIN_COLOR) >= len(spec["pins"])
        for color in {row["color"] for row in spec["rails"] + spec["buttons"]}:
            assert np.any(frame[:41] == color), f"tier {index + 1} rod colour hidden"
            assert np.any(frame[41:63] == color), f"tier {index + 1} control cue hidden"
        for sprite in env.level.get_sprites():
            if sprite.name.startswith("boundary"):
                continue
            assert 0 <= sprite.x and 0 <= sprite.y
            assert sprite.x + sprite.width <= 64
            assert sprite.y + sprite.height <= 64


@pytest.mark.parametrize(
    ("difficulty", "official_arena_nonbackground", "official_rendered_obstacle"),
    [(3, 296, 9), (4, 109, 0), (5, 253, 54)],
)
def test_native_pixel_obstacle_density_uses_one_honest_basis(
    difficulty, official_arena_nonbackground, official_rendered_obstacle,
):
    official = Env()
    official.set_level(difficulty - 1)
    official_pixels = obstacle_pixel_metrics(official)
    assert official_pixels["raw_visible_pixels"] == (
        REFERENCE_CHARACTERIZATION[difficulty]["obstacle_pixels"]
    )
    assert official_pixels["rendered_arena_visible_union"] == official_rendered_obstacle
    official_frame = np.asarray(official.render(), dtype=np.int16)
    assert np.count_nonzero(
        official_frame[:41] != names.BACKGROUND_COLOR
    ) == official_arena_nonbackground

    rng = random.Random(f"native-pixel-density:{difficulty}")
    for _ in range(64):
        spec, reason = _draft(rng, difficulty)
        if spec is not None:
            break
    assert spec is not None, reason
    generated_pixels = obstacle_pixel_metrics(Env([build_level(spec)]))
    low, high = PROFILES[difficulty]["obstacle_pixels"]
    assert low <= generated_pixels["raw_visible_pixels"] <= high
    assert generated_pixels["raw_collision_pixels"] == generated_pixels["raw_visible_pixels"]
    assert generated_pixels["background_collision_pixels"] == 0
    assert generated_pixels["visible_collision_fraction"] == 1.0
    assert generated_pixels["rendered_frame_visible_union"] == official_rendered_obstacle
    assert generated_pixels["rendered_arena_visible_union"] == official_rendered_obstacle
    components = generated_pixels["rendered_arena_components"]
    if official_rendered_obstacle:
        assert components == [official_rendered_obstacle]
    else:
        assert components == []
    generated_frame = np.asarray(Env([build_level(spec)]).render(), dtype=np.int16)
    generated_nonbackground = int(np.count_nonzero(
        generated_frame[:41] != names.BACKGROUND_COLOR
    ))
    assert 0.6 <= generated_nonbackground / official_arena_nonbackground <= 1.6


def test_solution_mechanics_deduplicates_physical_controls(generated_tier_one):
    env = Env([build_level(generated_tier_one)])
    controls = {
        (names.ACTION_CLICK, control.click[0], control.click[1]): control
        for control in extract(env).controls
    }
    used = [controls[action] for action in solution_actions(generated_tier_one)]
    assert generated_tier_one["solution_mechanics"]["distinct_controls"] == len({
        control.name for control in used
    })
    assert len({(control.name, control.kind) for control in extract(env).controls}) > len({
        control.name for control in extract(env).controls
    })


def test_validation_recomputes_proofs_identities_routes_and_splits(generated_game):
    curriculum = FULL_STANDARD_CONTRACT["curriculum"][0]
    cases = []

    proof = deepcopy(generated_game[0])
    proof["proof"]["budget_remaining"] += 1
    cases.append(proof)

    geometry = deepcopy(generated_game[0])
    geometry["targets"][0]["x"] += 3
    cases.append(geometry)

    route = deepcopy(generated_game[0])
    route["solution"].pop()
    cases.append(route)

    split = deepcopy(generated_game[0])
    split["split"] = "validation"
    cases.append(split)

    for tampered in cases:
        assert validate_full_standard(tampered, curriculum)


def test_validator_rejects_the_reported_post_win_suffix(generated_tier_one):
    spec = deepcopy(generated_tier_one)
    spec["solution"].append([names.ACTION_CLICK, 14, 57])
    with pytest.raises(ValueError, match="before the final stored action"):
        _solution_mechanics(spec, spec["solution"])
    spec["solution_length"] = len(spec["solution"])
    spec["solution_semantic_sha256"] = solution_semantic_hash(spec)
    spec["proof"]["solution_semantic_sha256"] = spec["solution_semantic_sha256"]
    spec["proof"]["budget_actions"] += 1
    spec["proof"]["budget_remaining"] -= 1
    errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][0])
    assert errors
    assert any("final action" in error or "mechanic evidence" in error
               for error in errors)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("context_index", False),
        lambda row: row.__setitem__("solution_length", 999),
        lambda row: row.__setitem__("optimality_claim", True),
        lambda row: (row.__setitem__("seed", True),
                     row["proof"].__setitem__("seed", True)),
        lambda row: row["proof"].update(
            optimality_claim=0, native_context_replayed=1, engine_win=1
        ),
        lambda row: row.__setitem__("proof", None),
        lambda row: row.__setitem__("step_counter", None),
        lambda row: row.__setitem__("solution_mechanics", None),
        lambda row: row.__setitem__("solution", None),
    ],
)
def test_validator_returns_errors_for_reported_malformed_metadata(generated_tier_one, mutate):
    malformed = deepcopy(generated_tier_one)
    mutate(malformed)
    errors = validate_full_standard(malformed, FULL_STANDARD_CONTRACT["curriculum"][0])
    assert isinstance(errors, list) and errors


def test_semantic_identities_ignore_unused_pin_labels_and_independent_rod_order(
    generated_tier_one,
):
    original = generated_tier_one
    unused_label = deepcopy(original)
    unused_label["pin_rods"] = ["deliberately-unused-label"]
    assert gameplay_hash(unused_label) == original["gameplay_sha256"]
    assert solution_semantic_hash(unused_label) == original["solution_semantic_sha256"]

    reordered = deepcopy(original)
    reordered["rods"].reverse()
    assert gameplay_hash(reordered) == original["gameplay_sha256"]
    assert solution_semantic_hash(reordered) == original["solution_semantic_sha256"]
    assert validate_full_standard(
        reordered, FULL_STANDARD_CONTRACT["curriculum"][0]
    ) == []


def test_tier_four_requires_off_pin_path_shared_constraint_work():
    spec = generate(0, 4, split="train")
    assert spec is not None
    actions = solution_actions(spec)
    evidence = spec["dependency_evidence"]
    pin_path = _pin_path_rods(spec)
    assert len(pin_path) == 1
    parents = {child: parent for parent, child in spec["children"]}
    pin_color = next(rod["color"] for rod in spec["rods"]
                     if rod["name"] in pin_path)
    companions = [rod["name"] for rod in spec["rods"]
                  if rod["name"] not in pin_path and rod["color"] == pin_color]
    assert len(companions) == 4
    assert all(name in parents for name in companions)
    assert all(row["essential"] for row in evidence["recursive_edges"])
    assert any(row["essential"] for row in evidence["shared_companions"])
    assert any(row["essential"] for row in evidence["constraint_order"])
    stripped = deepcopy(spec)
    pin_path = _pin_path_rods(stripped)
    stripped["rods"] = [rod for rod in stripped["rods"]
                         if rod["name"] in pin_path]
    stripped["children"] = [edge for edge in stripped["children"]
                            if set(edge) <= pin_path]
    stripped["obstacles"] = []
    assert (_first_native_completion(stripped, actions) != len(actions)
            or _greedy_native_reduction(stripped, actions) != actions)


def test_official_tier_four_has_standalone_pin_and_four_linked_shared_children():
    env = Env([official_levels()[3]])
    layout = extract(env)
    rod_names = {rod.name for rod in layout.rods if rod.controlled}
    children = getattr(env.game, names.ATTR_CHILDREN)
    rod_parent = {
        child.name: parent.name
        for parent, descendants in children.items()
        for child in descendants
        if parent.name in rod_names and child.name in rod_names
    }
    pin_parent = next(
        parent.name
        for parent, descendants in children.items()
        if any(names.TAG_PIN in child.tags for child in descendants)
    )
    rods = {rod.name: rod for rod in layout.rods}
    assert pin_parent not in rod_parent
    shared = [rod.name for rod in layout.rods
              if rod.controlled and rod.color == rods[pin_parent].color]
    companions = set(shared) - {pin_parent}
    assert len(shared) == 5
    assert len(companions) == 4
    assert companions <= rod_parent.keys()
    assert len({rod_parent[name] for name in companions}) == 4


def test_fast_engine_matches_native_generated_multicolor_prefix():
    spec, reason = _draft(random.Random("multicolor-prefix"), 8)
    assert spec is not None, reason
    spec["targets"] = [{"x": 60, "y": 0}]
    env = Env([build_level(spec) for _ in range(8)])
    env.set_level(7)
    model = _FastEngine(env, extract(env))
    actions = list(model.actions[:8])
    expected = model.start
    for action in actions:
        expected, _ = model.step(expected, model.actions.index(action))
        env.perform(*action)
        actual = _FastEngine(env, extract(env)).start
        assert expected == actual


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("reference_level", 999),
        lambda row: row.__setitem__("certificate_type", "fabricated_shortest_proof"),
        lambda row: row["rods"][0].__setitem__(
            "length", row["rods"][0]["length"] + 0.25
        ),
        lambda row: row["pins"][0].__setitem__("x", str(row["pins"][0]["x"])),
        lambda row: row["rails"][0].__setitem__("x", str(row["rails"][0]["x"])),
    ],
)
def test_validator_rejects_nested_geometry_and_certificate_aliases(
    generated_tier_one, mutate,
):
    malformed = deepcopy(generated_tier_one)
    mutate(malformed)
    assert validate_full_standard(
        malformed, FULL_STANDARD_CONTRACT["curriculum"][0]
    )


def test_validator_short_circuits_a_self_child_graph(generated_tier_one):
    malformed = deepcopy(generated_tier_one)
    name = malformed["rods"][0]["name"]
    malformed["children"].append([name, name])
    errors = validate_full_standard(
        malformed, FULL_STANDARD_CONTRACT["curriculum"][0]
    )
    assert errors and any("cycle" in error or "forest" in error for error in errors)


def test_gameplay_identity_binds_multicolor_control_dispatch(generated_tier_one):
    coupled = deepcopy(generated_tier_one)
    coupled["rails"][0]["secondary_color"] = coupled["rails"][1]["color"]
    assert gameplay_hash(coupled) != gameplay_hash(generated_tier_one)
    assert solution_semantic_hash(coupled) != solution_semantic_hash(generated_tier_one)


def test_gameplay_and_route_identity_ignore_global_translation(generated_tier_one):
    translated = deepcopy(generated_tier_one)
    for key in ("rods", "pins", "targets"):
        for row in translated[key]:
            row["x"] += 3
    for obstacle in translated["obstacles"]:
        obstacle["cells"] = [[x + 3, y] for x, y in obstacle["cells"]]
    for edge in translated["native_relations"]["pin_edges"]:
        edge[1][0] += 3
    assert gameplay_hash(translated) == gameplay_hash(generated_tier_one)
    assert solution_semantic_hash(translated) == solution_semantic_hash(generated_tier_one)


def test_obstacle_collision_cells_are_fully_visible():
    sprite = obstacle_sprite(
        "wall", [[0, 0], [3, 0], [6, 0], [9, 0]], color=8, ink_stride=4
    )
    pixels = np.asarray(sprite.pixels)
    assert np.all(pixels != names.BACKGROUND_COLOR)
    assert np.all(pixels != -1)


def test_generation_is_deterministic_explicit_and_bounded():
    left = generate(123, 1, split="validation")
    right = generate(123, 1, split="validation")
    assert left == right and left is not None
    with pytest.raises(TypeError):
        generate(123, 1)
    with pytest.raises(ValueError):
        generate(123, 9, split="train")
    with pytest.raises(ValueError):
        generate(123, 1, split="train", attempts=0)
    with pytest.raises(ValueError):
        generate(123, 1, split="train", max_transitions=0)


def test_build_game_rejects_shortened_reordered_and_duplicate_games(generated_game):
    with pytest.raises(ValueError, match="exactly eight"):
        build_game(generated_game[:-1])
    reordered = list(generated_game)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="difficulties 1..8"):
        build_game(reordered)
    duplicate = list(generated_game)
    duplicate[1] = deepcopy(duplicate[0])
    duplicate[1]["difficulty"] = 2
    duplicate[1]["context_index"] = 1
    with pytest.raises(ValueError):
        build_game(duplicate)


def test_bank_cli_requires_split_and_round_trips_jsonl(tmp_path):
    output = tmp_path / "bank.jsonl"
    run = subprocess.run(
        [sys.executable, "-m", "pebby.games.s5i5.bank", "--levels", "2",
         "--seed", "0", "--difficulty", "1", "--split", "test",
         "--out", str(output)],
        check=False, capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    specs = load_bank(output)
    assert len(specs) == 2 and {spec["split"] for spec in specs} == {"test"}

    partial = Path(tmp_path) / "partial.jsonl"
    exhausted = subprocess.run(
        [sys.executable, "-m", "pebby.games.s5i5.bank", "--levels", "1",
         "--seed", "0", "--difficulty", "1", "--split", "train",
         "--max-seeds", "0", "--out", str(partial)],
        check=False, capture_output=True, text=True, timeout=15,
    )
    assert exhausted.returncode == 1
    assert partial.exists() and partial.read_text() == ""
    assert "0/1 levels from 0 seeds" in exhausted.stdout


def test_reference_profiles_match_native_budgets_and_counts():
    for difficulty, level in enumerate(official_levels(), 1):
        assert (level._data[names.KEY_STEP_COUNTER]
                == REFERENCE_CHARACTERIZATION[difficulty]["budget"])
