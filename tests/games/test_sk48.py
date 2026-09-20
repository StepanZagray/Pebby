"""Full eight-tier SK48 generation, teacher, and native replay tests."""

from copy import deepcopy
import json
import random

from arcengine import GameState, Level
import pytest

from pebby.games.sk48 import names
from pebby.games.sk48.env import Env, official_levels, replay, upstream
from pebby.games.sk48.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    PROFILES,
    _accepted_spec,
    _counterfactual_evidence,
    _draft,
    _native_route_evidence,
    _official_geometry_identities,
    build_game,
    build_level,
    exact_geometry_identity,
    generate,
    generate_game,
    geometry_identity,
    official_layout_identity,
    gameplay_identity,
    profile_errors,
    split_for_identity,
    structural_metrics,
    validate_full_standard,
)
from pebby.games.sk48.layout import extract
from pebby.games.sk48.model import extract_model, mismatch, select, solved, transition
from pebby.games.sk48.plan import _state_key, search, solve


# Fixed seeds select newly procedural rows; these are not stored routes or
# official templates.  Tiers 3 and 5 intentionally demonstrate bounded
# rejection before a candidate exercises the required coupled mechanic.
_ROWS = (
    (1, "validation", 32),
    (2, "validation", 32),
    (3, "validation", 32),
    (4, "validation", 32),
    (5, "validation", 32),
    (6, "validation", 32),
    (7, "validation", 32),
    (8, "validation", 32),
)


@pytest.fixture(scope="module")
def generated_specs():
    specs = []
    for difficulty, split, attempts in _ROWS:
        spec = generate(
            0, difficulty, split=split, attempts=attempts, node_limit=250_000,
        )
        assert spec is not None, (difficulty, generate.last_rejections)
        specs.append(spec)
    return specs


def _assert_action_triples(env, actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        if action_id == names.ACTION_CLICK:
            assert isinstance(x, int) and isinstance(y, int)
            assert 0 <= x < names.FRAME_SIZE and 0 <= y < names.FRAME_SIZE
        else:
            assert x is None and y is None


def test_contract_profiles_and_official_characterization_are_exactly_eight():
    assert len(official_levels()) == 8
    assert DIFFICULTIES == tuple(range(1, 9))
    assert tuple(PROFILES) == DIFFICULTIES
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert len(FULL_STANDARD_CONTRACT["curriculum"]) == 8
    for index, row in enumerate(FULL_STANDARD_CONTRACT["curriculum"]):
        assert row == {
            "difficulty": index + 1,
            "context_index": index,
            "search_work": 250_000,
        }
    observed = []
    observed_visual = []
    for level in official_levels():
        env = Env([level])
        boundary = env.level.get_sprites_by_tag(names.TAG_BOUNDARY)[0]
        observed.append((
            len(env.heads()), len(env.pairs()), len(env.color_pads()),
            len(env.level.get_sprites_by_tag(names.TAG_RAIL)),
            len(env.level.get_sprites_by_tag(names.TAG_BLOCKER)),
            int(boundary.pixels.shape[0]),
        ))
        frame = env.render()
        observed_visual.append((
            sum(cell != upstream().BACKGROUND_COLOR for row in frame for cell in row),
            len({cell for row in frame for cell in row}),
        ))
    assert observed == [
        (2, 1, 6, 4, 0, 5), (2, 1, 8, 6, 0, 7),
        (3, 1, 8, 6, 0, 7), (5, 2, 8, 6, 0, 7),
        (2, 1, 9, 6, 1, 7), (4, 2, 12, 12, 1, 7),
        (4, 2, 11, 12, 0, 7), (4, 2, 8, 4, 0, 7),
    ]
    assert observed_visual == [
        (1684, 10), (2572, 11), (2578, 12), (2524, 13),
        (2532, 9), (2632, 10), (2668, 11), (2572, 12),
    ]


def test_fast_model_matches_native_across_all_official_mechanics():
    for tier, level in enumerate(official_levels(), 1):
        env = Env([level])
        model = extract_model(env)
        state = model.start
        rng = random.Random(10_000 + tier)
        for _ in range(50):
            if rng.random() < 0.2:
                index = rng.choice([i for i, clickable in enumerate(model.clickable) if clickable])
                x, y = state.heads[index]
                env.perform(names.ACTION_CLICK, x + 2, y + 2)
                state = select(model, state, index)
            else:
                action = rng.choice(names.MOVE_ACTIONS)
                env.perform(action)
                state = transition(model, state, action)
            if env.levels_completed:
                break
            assert extract_model(env).start == state


def test_all_official_tiers_have_bounded_positive_native_teacher_replays():
    limits = (100_000, 100_000, 100_000, 100_000,
              2_000_000, 2_000_000, 100_000, 300_000)
    lengths = []
    for tier, (level, node_limit) in enumerate(zip(official_levels(), limits), 1):
        result = search(Env([level]), limit=names.MOVE_BUDGET, node_limit=node_limit)
        assert result.actions is not None, (tier, result)
        assert not result.truncated and not result.unsupported and result.exact
        env = Env([level])
        _assert_action_triples(env, result.actions)
        assert replay(env, result.actions)
        assert env.levels_completed == 1
        lengths.append(len(result.actions))
    # These are constructive replay lengths, explicitly not optimality claims.
    # Tier 8's constructive search now finds a shorter (still verified exact,
    # non-truncated, replaying) route: 26 actions instead of 28.
    assert lengths == [14, 30, 33, 29, 34, 37, 36, 26]


def test_generated_rows_cover_every_mechanic_and_validate_after_json(generated_specs):
    geometries = set()
    gameplays = set()
    for index, spec in enumerate(generated_specs):
        stored = json.loads(json.dumps(spec))
        row = FULL_STANDARD_CONTRACT["curriculum"][index]
        assert validate_full_standard(stored, row) == ()
        assert profile_errors(stored) == []
        assert stored["training_context_index"] == index
        assert stored["verification_level_index"] == index
        assert stored["native_move_budget"] == names.MOVE_BUDGET
        assert stored["proof"]["shortest_route_claimed"] is False
        assert stored["proof"]["search_truncated"] is False
        assert stored["geometry_identity"] == geometry_identity(stored)
        assert stored["gameplay_identity"] == gameplay_identity(stored)
        assert stored["split"] == split_for_identity(stored["geometry_identity"])
        assert replay(Env([build_level(stored)]), stored["solution"])
        geometries.add(stored["geometry_identity"])
        gameplays.add(stored["gameplay_identity"])

    assert len(geometries) == len(gameplays) == 8
    evidence = [spec["solution_mechanics"] for spec in generated_specs]
    assert evidence[2]["auxiliary_interactions"] > 0
    assert evidence[3]["auxiliary_interactions"] > 0
    assert evidence[4]["blocker_pin_events"] > 0
    assert evidence[5]["blocker_pin_events"] > 0
    assert all(row["click_switches"] > 0 for row in evidence[5:])
    assert all(row["crossing_interactions"] > 0 for row in evidence[6:])
    assert all(row["pause_actions"] > 0 for row in evidence)
    assert all(row["minimum_moves_left"] >= 8 for row in evidence)


def test_complete_generated_game_replays_sequentially_without_forcing(generated_specs):
    levels = build_game(generated_specs)
    assert len(levels) == 8
    env = Env(levels)
    for index, spec in enumerate(generated_specs):
        assert env.level_index == index
        assert replay(env, spec["solution"])
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN


def test_default_generate_game_builds_and_replays_exact_eight_tiers():
    specs = generate_game(888, split="train")
    assert specs is not None
    assert [spec["difficulty"] for spec in specs] == list(DIFFICULTIES)
    assert [spec["training_context_index"] for spec in specs] == list(range(8))
    levels = build_game(specs)
    env = Env(levels)
    for index, spec in enumerate(specs):
        assert env.level_index == index
        assert replay(env, spec["solution"])
        assert env.levels_completed == index + 1
    assert env.state == GameState.WIN


def test_validator_rejects_route_geometry_partition_and_proof_tampering(generated_specs):
    original = generated_specs[0]
    row = FULL_STANDARD_CONTRACT["curriculum"][0]

    route = deepcopy(original)
    route["solution"] = route["solution"][:-1]
    assert validate_full_standard(route, row)

    geometry = deepcopy(original)
    # pads[0] sits exactly names.CELL away from pads[1] in the current
    # footer layout, so nudging it lands it on top of its neighbor and
    # trips the footer-pad-count structural check before the geometry
    # identity check is ever reached. pads[3] (a head-row pad) is not
    # adjacent to any other pad by exactly CELL, so it still isolates the
    # identity/replay mismatch this assertion is meant to exercise.
    geometry["pads"][3]["position"][0] += names.CELL
    assert any("identity" in error or "replay" in error
               for error in validate_full_standard(geometry, row))

    partition = deepcopy(original)
    partition["split"] = next(value for value in ("train", "validation", "test")
                              if value != original["split"])
    assert "canonical split mismatch" in validate_full_standard(partition, row)

    proof = deepcopy(original)
    proof["proof"]["shortest_route_claimed"] = True
    assert "proof metadata mismatch" in validate_full_standard(proof, row)


def test_standard_identities_types_and_exact_first_completion_are_recomputed():
    spec = generate(0, 1, split="train", attempts=32)
    assert spec is not None
    row = FULL_STANDARD_CONTRACT["curriculum"][0]
    assert spec["effective_seed"] == spec["seed"]
    assert spec["context_index"] == spec["training_context_index"] == 0
    assert spec["geometry_d4_sha256"] == spec["geometry_identity"]
    assert spec["gameplay_sha256"] == spec["gameplay_identity"]
    assert spec["geometry_sha256"] != spec["geometry_d4_sha256"]
    assert validate_full_standard(spec, row) == ()

    suffix = deepcopy(spec)
    suffix["solution"].append([names.ACTION_UP, None, None])
    suffix["solution_length"] += 1
    suffix["proof"]["action_count"] += 1
    actions = tuple(tuple(action) for action in suffix["solution"])
    evidence = _native_route_evidence(build_level(suffix), actions, 0)
    evidence.update(_counterfactual_evidence(suffix, actions))
    suffix["solution_mechanics"] = evidence
    assert any("first native completion" in error
               for error in validate_full_standard(suffix, row))

    for key, value in (
        ("context_index", False),
        ("effective_seed", float(spec["effective_seed"])),
        ("split", []),
        ("gameplay_sha256", []),
    ):
        changed = deepcopy(spec)
        changed[key] = value
        assert validate_full_standard(changed, row), key
    changed = deepcopy(spec)
    changed["proof"]["shortcut_node_cap_reached"] = 1
    assert validate_full_standard(changed, row)
    assert validate_full_standard(None, row)
    assert validate_full_standard(spec, None)


def test_full_builder_fails_closed_and_game_failure_keeps_child_diagnostics():
    for malformed in (None, "not-specs", [None] * 8, [{"split": []}] * 8):
        with pytest.raises(ValueError):
            build_game(malformed)
    level = generate(0, 1, split="train", attempts=32)
    assert level is not None
    unhashable_identity = [deepcopy(level) for _ in range(8)]
    unhashable_identity[0]["gameplay_sha256"] = []
    with pytest.raises(ValueError):
        build_game(unhashable_identity)
    assert generate_game(
        0, split="train", difficulties=(6,), attempts=1, node_limit=1,
    ) is None
    report = generate_game.last_failure
    assert report is not None
    assert report["difficulty"] == 6
    assert report["context_index"] == 5
    assert type(report["child_seed"]) is int
    assert report["attempts"] == report["node_limit"] == 1
    assert report["rejections"]


def test_tier6_rejects_the_confirmed_full_rail_underfloor_regression():
    spec = generate(0, 6, split="train", attempts=32, node_limit=250_000)
    assert spec is not None, generate.last_rejections
    assert spec["solution_length"] >= PROFILES[6]["route_actions"][0]
    shortcut = search(
        Env([build_level(spec)]),
        limit=PROFILES[6]["route_actions"][0] - 1,
        node_limit=50_000,
    )
    assert shortcut.actions is None
    # This exact cap is intentionally reported as inconclusive, never proof
    # that no shorter solution exists.
    assert shortcut.truncated
    assert spec["proof"]["shortcut_node_cap_reached"] is True


def test_undo_state_and_random_prefix_recovery_remain_live():
    env = Env([official_levels()[0]])
    initial = _state_key(env)
    env.perform(names.ACTION_UP)
    assert _state_key(env) != initial
    env.perform(names.ACTION_UNDO)
    undone = _state_key(env)
    assert undone[:3] == initial[:3]
    assert undone[3] == initial[3] - 1
    assert undone[4] == initial[4]
    result = search(env, node_limit=100_000)
    assert result.actions is not None and replay(env, result.actions)


def test_bounds_invalid_actions_and_full_game_rejections_are_explicit(generated_specs):
    env = Env([official_levels()[0]])
    capped = search(env, node_limit=1)
    assert capped.actions is None and capped.truncated and not capped.unsupported
    action_capped = search(env, limit=1, node_limit=30_000)
    assert action_capped.actions is None and action_capped.truncated
    assert solve(env, node_limit=1) is None and solve.truncated

    for action in ((5, None, None), (6, None, None), (1, 1, None), (7, None, 1)):
        with pytest.raises(ValueError):
            env.perform(*action)

    with pytest.raises(ValueError, match="exactly eight"):
        build_game(generated_specs[:-1])
    reordered = list(generated_specs)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="1..8 in order"):
        build_game(reordered)
    subset = generate_game(7, split="train", difficulties=(1, 2), attempts=32)
    assert subset is not None and [spec["difficulty"] for spec in subset] == [1, 2]
    with pytest.raises(ValueError, match="exactly eight"):
        build_game(subset)


def test_structural_profiles_expose_expected_progression(generated_specs):
    metrics = [structural_metrics(spec) for spec in generated_specs]
    assert metrics[0]["boundary_cells"] == 5
    assert all(row["boundary_cells"] == 7 for row in metrics[1:])
    assert metrics[2]["auxiliary_heads"] >= 1
    assert metrics[3]["auxiliary_heads"] == 4
    assert metrics[4]["blockers"] == metrics[5]["blockers"] == 1
    assert all(row["clickable_heads"] == 4 for row in metrics[5:])
    assert all(row["orientations"] == 2 for row in metrics[5:])


def test_boundary_metadata_cannot_change_native_identity_or_admit_official_copy():
    env = Env([official_levels()[0]])
    spec = _draft(0, 1, 0)
    boundary = env.level.get_sprites_by_tag(names.TAG_BOUNDARY)[0]
    spec["boundary"] = {
        "prototype": boundary.name,
        "position": [int(boundary.x), int(boundary.y)],
        "cells": int(boundary.pixels.shape[0]),
    }
    spec["heads"] = [
        {
            "prototype": head.name,
            "position": [int(head.x), int(head.y)],
            "rotation": int(head.rotation),
            "length": len(env.lines()[head]),
        }
        for head in env.heads()
    ]
    spec["pads"] = [
        {
            "position": [int(pad.x), int(pad.y)],
            "color": int(pad.pixels[1, 1]),
        }
        for pad in env.color_pads()
    ]
    spec["rails"] = [
        {
            "position": [int(rail.x), int(rail.y)],
            "rotation": int(rail.rotation),
        }
        for rail in env.level.get_sprites_by_tag(names.TAG_RAIL)
    ]
    spec["blockers"] = []
    split = split_for_identity(geometry_identity(spec))
    assert _accepted_spec(spec, split, 250_000)[1] == "official_geometry_copy"

    malformed = deepcopy(spec)
    malformed["boundary"]["cells"] = 3
    accepted, reason = _accepted_spec(malformed, split, 250_000)
    assert accepted is None
    assert reason.startswith("schema:boundary cells do not match")
    with pytest.raises(ValueError, match="boundary cells do not match"):
        geometry_identity(malformed)
    with pytest.raises(ValueError, match="boundary cells do not match"):
        build_level(malformed)


def test_footer_presentation_shift_does_not_change_semantic_identity_or_split():
    spec = generate(0, 1, split="train", attempts=32)
    assert spec is not None
    row = FULL_STANDARD_CONTRACT["curriculum"][0]
    for dx in (-12, -6, 6, 12):
        shifted = deepcopy(spec)
        for head in shifted["heads"]:
            if head["position"][1] >= names.HUD_ROW:
                head["position"][0] += dx
        for pad in shifted["pads"]:
            if pad["position"][1] >= names.HUD_ROW:
                pad["position"][0] += dx
        assert exact_geometry_identity(shifted) != spec["geometry_sha256"]
        assert geometry_identity(shifted) == spec["geometry_identity"]
        assert gameplay_identity(shifted) == spec["gameplay_identity"]
        shifted["geometry_sha256"] = exact_geometry_identity(shifted)
        assert split_for_identity(geometry_identity(shifted)) == spec["split"]
        assert validate_full_standard(shifted, row) == ()

    reordered_target = deepcopy(spec)
    footer_pads = [
        pad for pad in reordered_target["pads"]
        if pad["position"][1] >= names.HUD_ROW
    ]
    footer_pads[0]["color"], footer_pads[1]["color"] = (
        footer_pads[1]["color"], footer_pads[0]["color"],
    )
    assert geometry_identity(reordered_target) == spec["geometry_identity"]
    assert gameplay_identity(reordered_target) != spec["gameplay_identity"]

    changed_selection = _draft(0, 8, 0)
    original_gameplay = gameplay_identity(changed_selection)
    original_layout = official_layout_identity(changed_selection)
    board_indices = [
        index for index, head in enumerate(changed_selection["heads"])
        if head["position"][1] < names.HUD_ROW
    ]
    left, right = board_indices[:2]
    changed_selection["heads"][left], changed_selection["heads"][right] = (
        changed_selection["heads"][right], changed_selection["heads"][left],
    )
    assert gameplay_identity(changed_selection) != original_gameplay
    assert official_layout_identity(changed_selection) == original_layout


def test_model_goal_matches_native_reference_prefix_with_surplus_colors():
    spec = _draft(0, 1, 0)
    spec.update(
        boundary={
            "prototype": names.BOUNDARY_SMALL,
            "position": [11, 12],
            "cells": 5,
        },
        heads=[
            {
                "prototype": names.HEAD_BLUE,
                "position": [5, 24],
                "rotation": 0,
                "length": 5,
            },
            {
                "prototype": names.HEAD_BLUE,
                "position": [5, 56],
                "rotation": 0,
                "length": 3,
            },
        ],
        pads=[
            {"position": [x, y], "color": color}
            for x, y, color in (
                (11, 24, 8), (17, 24, 9), (23, 24, 12),
                (11, 56, 8), (17, 56, 9),
            )
        ],
        rails=[
            {"position": [7, y], "rotation": 0}
            for y in (14, 20, 26, 32)
        ],
        blockers=[],
    )
    env = Env([build_level(spec)])
    model = extract_model(env)
    assert env.game.gvtmoopqgy()
    assert solved(model, model.start)
    assert mismatch(model, model.start) == 0

    observation = env.perform(names.ACTION_RIGHT)
    successor = transition(model, model.start, names.ACTION_RIGHT)
    assert observation.state == GameState.WIN
    assert solved(model, successor)
    assert mismatch(model, successor) == 0


def test_footer_first_selection_is_rejected_before_build_hash_or_admission():
    spec = generate(0, 8, split="train", attempts=32, node_limit=250_000)
    assert spec is not None
    footer_index = next(
        index for index, head in enumerate(spec["heads"])
        if head["position"][1] >= names.HUD_ROW
    )
    malformed = deepcopy(spec)
    malformed["heads"] = [malformed["heads"][footer_index]] + [
        head for index, head in enumerate(malformed["heads"])
        if index != footer_index
    ]
    errors = validate_full_standard(
        malformed, FULL_STANDARD_CONTRACT["curriculum"][7]
    )
    assert "heads[0] must be an initial board head" in errors
    for operation in (
        build_level, exact_geometry_identity, geometry_identity,
        gameplay_identity, official_layout_identity,
    ):
        with pytest.raises(ValueError, match="initial board head"):
            operation(malformed)


def test_official_copy_gate_is_independent_of_initial_board_controller():
    env = Env([official_levels()[3]])
    boundary = env.level.get_sprites_by_tag(names.TAG_BOUNDARY)[0]
    spec = _draft(0, 4, 0)
    spec.update(
        boundary={
            "prototype": boundary.name,
            "position": [int(boundary.x), int(boundary.y)],
            "cells": int(boundary.pixels.shape[0]),
        },
        heads=[
            {
                "prototype": head.name,
                "position": [int(head.x), int(head.y)],
                "rotation": int(head.rotation),
                "length": len(env.lines()[head]),
            }
            for head in env.heads()
        ],
        pads=[
            {
                "position": [int(pad.x), int(pad.y)],
                "color": int(pad.pixels[1, 1]),
            }
            for pad in env.color_pads()
        ],
        rails=[
            {"position": [int(rail.x), int(rail.y)],
             "rotation": int(rail.rotation)}
            for rail in env.level.get_sprites_by_tag(names.TAG_RAIL)
        ],
        blockers=[
            {"position": [int(blocker.x), int(blocker.y)]}
            for blocker in env.level.get_sprites_by_tag(names.TAG_BLOCKER)
        ],
    )
    board_indices = [
        index for index, head in enumerate(spec["heads"])
        if head["position"][1] < names.HUD_ROW
    ]
    assert len(board_indices) >= 2
    original_geometry = geometry_identity(spec)
    official_key = official_layout_identity(spec)
    assert official_key in _official_geometry_identities()

    left, right = board_indices[:2]
    spec["heads"][left], spec["heads"][right] = (
        spec["heads"][right], spec["heads"][left],
    )
    assert geometry_identity(spec) != original_geometry
    assert official_layout_identity(spec) == official_key
    split = split_for_identity(geometry_identity(spec))
    accepted, reason = _accepted_spec(spec, split, 250_000)
    assert accepted is None and reason == "official_geometry_copy"


def test_reference_head_pad_and_surplus_native_color_are_rejected_before_rekey():
    spec = generate(0, 1, split="train", attempts=32)
    assert spec is not None
    reference = next(
        head for head in spec["heads"] if head["position"][1] >= names.HUD_ROW
    )
    for color in names.COLORS:
        malformed = deepcopy(spec)
        for pad in malformed["pads"]:
            if pad["position"][1] >= names.HUD_ROW:
                pad["position"][0] -= names.CELL
        malformed["pads"].append({
            "position": [
                reference["position"][0] + 3 * names.CELL,
                reference["position"][1],
            ],
            "color": color,
        })
        errors = validate_full_standard(
            malformed, FULL_STANDARD_CONTRACT["curriculum"][0]
        )
        assert "footer reference head must not contain a color pad" in errors
        for operation in (
            build_level, exact_geometry_identity, geometry_identity,
            gameplay_identity, official_layout_identity,
        ):
            with pytest.raises(ValueError, match="head must not contain"):
                operation(malformed)


def test_model_uses_native_reference_indicator_count_with_surplus_color():
    prototypes = upstream().sprites
    sprites = []
    for x, y in ((5, 24), (5, 56)):
        sprites.append(prototypes[names.HEAD_BLUE].clone().set_position(x, y))
        for offset in range(3):
            sprites.append(
                prototypes[names.SEGMENT].clone().set_position(
                    x + offset * names.CELL, y
                )
            )
    for x, y, color in (
        (11, 24, 8), (17, 24, 9),
        (5, 56, 8), (11, 56, 9), (17, 56, 12),
    ):
        sprites.append(
            prototypes[names.COLOR_PAD].clone().set_position(x, y)
            .color_remap(None, color)
        )
    sprites.extend((
        prototypes[names.BOUNDARY_SMALL].clone().set_position(11, 12).set_scale(6),
        prototypes[names.FOOTER].clone().set_position(0, 54).set_scale(64),
        prototypes[names.DIVIDER].clone().set_position(0, names.HUD_ROW),
    ))
    env = Env([Level(
        sprites=sprites, grid_size=(names.FRAME_SIZE, names.FRAME_SIZE),
        data={"grouped_pauses": False, "lit_extension": True},
    )])
    model = extract_model(env)
    assert env.game.gvtmoopqgy()
    assert solved(model, model.start)
    assert mismatch(model, model.start) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("boundary", []),
        ("heads", [None]),
        ("pads", [None]),
        ("rails", [None]),
        ("blockers", [None]),
    ),
)
def test_validator_rejects_malformed_nested_components_without_raising(
    field, value,
):
    spec = generate(0, 1, split="train", attempts=32)
    assert spec is not None
    spec[field] = value
    errors = validate_full_standard(
        spec, FULL_STANDARD_CONTRACT["curriculum"][0],
    )
    assert errors
    assert any(field in error for error in errors)
