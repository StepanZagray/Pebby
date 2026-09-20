"""Exact primitive typing for WA30 proof mirrors and route evidence."""

from copy import deepcopy

import pytest

from pebby.games.wa30.generate import (
    FULL_STANDARD_CONTRACT,
    _owner_evidence_errors,
    generate,
    validate_full_standard,
)


@pytest.fixture(scope="module")
def authentic_rows():
    rows = {difficulty: generate(0, difficulty) for difficulty in (1, 2, 3, 6, 7)}
    assert all(rows.values())
    for difficulty, spec in rows.items():
        assert validate_full_standard(
            spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        ) == []
    return rows


def _errors(rows, difficulty, mutate):
    changed = deepcopy(rows[difficulty])
    mutate(changed)
    errors = validate_full_standard(
        changed, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
    )
    assert errors
    return errors


def test_reported_topology_and_mechanic_type_aliases_are_rejected(authentic_rows):
    errors = _errors(
        authentic_rows,
        3,
        lambda row: row["proof"]["topology_counterfactual"].__setitem__(
            "actions_replayed",
            float(row["proof"]["topology_counterfactual"]["actions_replayed"]),
        ),
    )
    assert any("topology_counterfactual" in error for error in errors)

    errors = _errors(
        authentic_rows,
        7,
        lambda row: row["proof"]["topology_counterfactual"].__setitem__(
            "ablated_won",
            int(row["proof"]["topology_counterfactual"]["ablated_won"]),
        ),
    )
    assert any("topology_counterfactual" in error for error in errors)

    def false_zero(row):
        row["native_topology_counterfactual_actions"] = False
        row["proof"]["native_topology_counterfactual_actions"] = False

    errors = _errors(authentic_rows, 2, false_zero)
    assert any("topology counterfactual" in error for error in errors)

    errors = _errors(
        authentic_rows,
        6,
        lambda row: row["solution_mechanics"].__setitem__("player_steals", True),
    )
    assert any("player_steals" in error for error in errors)


@pytest.mark.parametrize(
    ("difficulty", "mutate", "fragment"),
    [
        (
            3,
            lambda row: row["topology_counterfactual"].__setitem__(
                "actions_replayed",
                float(row["topology_counterfactual"]["actions_replayed"]),
            ),
            "counterfactual",
        ),
        (
            3,
            lambda row: row["topology_counterfactual"].__setitem__(
                "baseline_won", 1
            ),
            "counterfactual",
        ),
        (
            7,
            lambda row: row["proof"]["topology_counterfactual"].__setitem__(
                "first_state_divergence_action",
                float(
                    row["proof"]["topology_counterfactual"][
                        "first_state_divergence_action"
                    ]
                ),
            ),
            "topology_counterfactual",
        ),
        (
            1,
            lambda row: row["solution_mechanics"].__setitem__("manual_grabs", True),
            "manual_grabs",
        ),
        (
            1,
            lambda row: row["solution_mechanics"].__setitem__("won", 1),
            "solution_mechanics.won",
        ),
        (
            1,
            lambda row: row["solution_mechanics"].__setitem__("manual_drops", 1.0),
            "manual_drops",
        ),
        (
            1,
            lambda row: row["generation_diagnostics"].__setitem__(
                "search_work_bound", float(row["search_limit"])
            ),
            "search-work bound",
        ),
        (
            1,
            lambda row: row.__setitem__("minimum_steps_left", False),
            "minimum_steps_left",
        ),
    ],
)
def test_nearby_count_boolean_and_recomputed_evidence_aliases_fail_closed(
    authentic_rows, difficulty, mutate, fragment
):
    errors = _errors(authentic_rows, difficulty, mutate)
    assert any(fragment in error for error in errors), errors


def test_owner_evidence_extension_has_exact_types_and_typed_mirrors():
    control = {
        "policy": "repeat_action_5_from_start",
        "actions_replayed": 17,
        "game_over": False,
        "won": False,
        "levels_completed": 0,
        "player_moved": False,
        "player_held_box": False,
    }
    spec = {
        "boxes_in_player_component": 3,
        "boxes_on_fences": 2,
        "native_no_player_control_actions": 17,
        "no_player_control": control,
        "proof": {
            "boxes_in_player_component": 3,
            "boxes_on_fences": 2,
            "native_no_player_control_actions": 17,
            "no_player_control": deepcopy(control),
        },
    }
    assert _owner_evidence_errors(spec, 3) == []

    spec["proof"]["boxes_on_fences"] = 2.0
    spec["proof"]["no_player_control"]["won"] = 0
    errors = _owner_evidence_errors(spec, 3)
    assert any("proof.boxes_on_fences" in error for error in errors)
    assert any("proof.no_player_control" in error for error in errors)

    spec["proof"]["boxes_on_fences"] = 2
    spec["proof"]["no_player_control"] = deepcopy(control)
    spec["no_player_control"]["levels_completed"] = False
    errors = _owner_evidence_errors(spec, 3)
    assert any("native no-player control evidence is malformed" in error for error in errors)


def test_owner_evidence_is_mandatory_and_nonapplicable_control_is_typed():
    assert _owner_evidence_errors({"proof": {}}, 3)
    baseline = {
        "boxes_in_player_component": 0,
        "boxes_on_fences": 0,
        "native_no_player_control_actions": 0,
        "no_player_control": None,
        "proof": {
            "boxes_in_player_component": 0,
            "boxes_on_fences": 0,
            "native_no_player_control_actions": 0,
            "no_player_control": None,
        },
    }
    assert _owner_evidence_errors(baseline, 2) == []
    baseline["native_no_player_control_actions"] = False
    baseline["proof"]["native_no_player_control_actions"] = False
    assert _owner_evidence_errors(baseline, 2)


@pytest.mark.parametrize(("container", "field"), [
    (None, "no_player_control"),
    ("proof", "no_player_control"),
])
def test_nonapplicable_no_player_control_requires_explicit_presence(
    container, field
):
    baseline = {
        "boxes_in_player_component": 0,
        "boxes_on_fences": 0,
        "native_no_player_control_actions": 0,
        "no_player_control": None,
        "proof": {
            "boxes_in_player_component": 0,
            "boxes_on_fences": 0,
            "native_no_player_control_actions": 0,
            "no_player_control": None,
        },
    }
    target = baseline if container is None else baseline[container]
    del target[field]
    errors = _owner_evidence_errors(baseline, 2)
    assert any("no_player_control" in error and "required" in error for error in errors)
