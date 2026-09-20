"""Focused coverage for the constructive native tier-8 route."""

import random

from pebby.games.s5i5 import generate as generator
from pebby.games.s5i5.generation_quality import identity_partition


def _fixed_candidate():
    spec, error = generator._draft(random.Random("tier8-recipe-coverage"), 8)
    assert error is None
    generator._apply_tier8_recipe(
        spec["rods"], ((24, 24, 24), (6, 6, 6, 9))
    )
    witness, reason, work = generator._construct_tier8_recipe(spec, 24)
    assert reason is None
    assert work == 24
    actions, targets = witness
    return dict(spec, targets=targets), actions


def test_tier8_recipe_is_independent_pin_with_causal_off_pin_branch():
    spec, actions = _fixed_candidate()

    assert len(actions) == 24
    assert generator._first_native_completion(spec, actions) == 24
    assert generator._greedy_native_reduction(spec, actions) == actions

    mechanics = generator._solution_mechanics(spec, actions)
    assert mechanics["all_actions_changed"] is True
    assert mechanics["all_actions_relevant"] is True
    assert mechanics["distinct_controls"] == 6
    assert mechanics["constraint_motion_actions"] == 21
    assert {kind for kind in ("extend", "retract", "rotate")
            if mechanics[kind]} == {"extend", "retract", "rotate"}

    dependency = generator._dependency_evidence(spec, actions)
    assert dependency is not None
    assert dependency["native_relations"]["pin_edges"] == [
        ["rod0", [30, 12]]
    ]
    assert all("rod0" not in edge
               for edge in dependency["native_relations"]["rod_edges"])
    assert next(row for row in dependency["shared_companions"]
                if row["rod"] == "rod8") == {
        "rod": "rod8",
        "first_completion": 24,
        "reduced_actions": 15,
        "essential": True,
    }
    assert any(row["child"] == "rod8" and row["essential"]
               for row in dependency["branch_edges"])


def test_tier8_public_generation_passes_full_validator():
    # Seed 1 selects the reviewed 24-action base recipe under the independent
    # recipe selector.  The variation test separately admits every recipe.
    seed = 1
    rng = random.Random(f"{generator.MECHANICS_INVENTORY_VERSION}:{seed}:8")
    draft, error = generator._draft(rng, 8)
    assert error is None
    witness, reason, _ = generator._construct_tier8_recipe(draft, 24)
    assert reason is None
    draft["targets"] = witness[1]
    _, split = identity_partition(draft)

    spec = generator.generate(seed, 8, attempts=1, split=split)
    assert spec is not None
    assert spec["solution_length"] == 24
    assert spec["shortcut_evidence"] == {
        "greedy_fixed_point_actions": 24,
        "bound": 10,
        "compact_nodes": 4000,
        "compact_outcome": "bounded_unknown",
        "optimality_claim": False,
    }
    assert generator.validate_full_standard(
        spec, generator.FULL_STANDARD_CONTRACT["curriculum"][7]
    ) == []
