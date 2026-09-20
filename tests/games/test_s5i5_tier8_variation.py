"""Causally relevant recipe variation for the tier-8 constructor."""

import random

from pebby.games.s5i5 import generate as generator
from pebby.games.s5i5 import names
from pebby.games.s5i5.env import Env
from pebby.games.s5i5.layout import extract


def _candidate(recipe):
    spec, error = generator._draft(random.Random("tier8-recipe-coverage"), 8)
    assert error is None
    generator._apply_tier8_recipe(spec["rods"], recipe)
    spec["targets"] = []
    witness, reason, work = generator._construct_tier8_recipe(spec, 36)
    assert reason is None
    assert work >= 24
    actions, targets = witness
    return dict(spec, targets=targets), actions


def _control_omission_is_fatal(spec, actions, control_name):
    layout = extract(Env([generator.build_level(dict(spec, targets=[]))]))
    clicks = {
        (names.ACTION_CLICK, int(control.click[0]), int(control.click[1]))
        for control in layout.controls if control.name == control_name
    }
    residual = [action for action in actions if tuple(action) not in clicks]
    return generator._first_native_completion(spec, residual) is None


def test_all_tier8_recipes_change_causal_thresholds_and_keep_native_gates():
    programs = set()
    geometries = set()
    for recipe in generator._TIER8_CAUSAL_RECIPES:
        spec, actions = _candidate(recipe)
        stages = generator._tier8_recipe_stages(spec)
        programs.add(tuple(count for _, _, count in stages))
        geometries.add(tuple(
            (rod["name"], rod["x"], rod["length"])
            for rod in spec["rods"] if rod["name"] in {"rod2", "rod3", "rod5", "rod10"}
        ))

        assert 24 <= len(actions) <= 25
        assert generator._first_native_completion(spec, actions) == len(actions)
        assert generator._greedy_native_reduction(spec, actions) == actions
        mechanics = generator._solution_mechanics(spec, actions)
        assert mechanics["all_actions_changed"] is True
        assert mechanics["all_actions_relevant"] is True
        assert mechanics["distinct_controls"] == 6
        assert all(mechanics[kind] for kind in ("extend", "retract", "rotate"))

        dependency = generator._dependency_evidence(spec, actions)
        assert dependency is not None
        assert sum(row["essential"] for row in dependency["branch_edges"]) >= 1
        assert sum(row["essential"] for row in dependency["shared_companions"]) >= 1
        assert all(_control_omission_is_fatal(spec, actions, control)
                   for control in ("rail1", "rail2", "rail3", "rail4", "button0"))

    assert len(programs) == len(generator._TIER8_CAUSAL_RECIPES)
    assert len(geometries) == len(generator._TIER8_CAUSAL_RECIPES)


def test_tier8_recipe_selection_reaches_multiple_relevant_programs():
    observed = set()
    for seed in range(16):
        placed = generator._place_rods(random.Random(seed), 8)
        assert placed is not None
        rods = placed[0]
        spec = {"rods": rods, "pins": [{"x": 30, "y": 12}]}
        stages = generator._tier8_recipe_stages(spec)
        assert stages is not None
        observed.add(tuple(count for _, _, count in stages))
    assert len(observed) >= 3


def test_each_recipe_is_admitted_by_public_generation_and_full_validator():
    expected = {
        tuple(count for _, _, count in generator._tier8_recipe_stages(_candidate(recipe)[0]))
        for recipe in generator._TIER8_CAUSAL_RECIPES
    }
    seeds = {}
    for seed in range(64):
        rng = random.Random(f"{generator.MECHANICS_INVENTORY_VERSION}:{seed}:8")
        draft, error = generator._draft(rng, 8)
        assert error is None
        stages = generator._tier8_recipe_stages(draft)
        signature = tuple(count for _, _, count in stages)
        seeds.setdefault(signature, (seed, draft))
        if set(seeds) == expected:
            break
    assert set(seeds) == expected

    for signature, (seed, draft) in seeds.items():
        witness, reason, _ = generator._construct_tier8_recipe(draft, 36)
        assert reason is None, (signature, seed, reason)
        draft["targets"] = witness[1]
        _, split = generator.identity_partition(draft)
        spec = generator.generate(seed, 8, attempts=1, split=split)
        assert spec is not None, (signature, seed, split)
        assert tuple(count for _, _, count in generator._tier8_recipe_stages(spec)) == signature
        assert generator.validate_full_standard(
            spec, generator.FULL_STANDARD_CONTRACT["curriculum"][7]
        ) == []
