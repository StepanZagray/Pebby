"""Strict publication-boundary regressions from the independent WA30 review."""

import copy

import pytest

from pebby.games.wa30.env import Env
from pebby.games.wa30.generate import (
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    certify,
    _official_semantic_hashes,
    generate,
    generate_game,
    is_official_semantic_copy,
    last_game_generation_report,
    structural_metrics,
    validate_full_standard,
)
from pebby.games.wa30.layout import extract
from pebby.games.wa30.quality import (
    gameplay_hash,
    geometry_hash,
    geometry_partition,
)


def _entry(difficulty):
    return FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]


@pytest.fixture(scope="module")
def tier_two():
    spec = generate(0, 2)
    assert spec is not None
    assert validate_full_standard(spec, _entry(2)) == []
    return spec


def test_strict_schema_and_malformed_collections_return_diagnostics(tier_two):
    mutations = []
    mutations.append(lambda row: (row.__setitem__("seed", True),
                                  row["proof"].__setitem__("seed", True)))
    mutations.append(lambda row: row.__setitem__("requested_seed", "garbage"))
    mutations.append(lambda row: row.__setitem__("effective_seed", -100))
    mutations.append(lambda row: (row.__setitem__("context_index", 1.0),
                                  row["proof"].__setitem__("context_index", 1.0)))
    mutations.append(lambda row: (row.__setitem__("engine_verified", 1),
                                  row["proof"].__setitem__("engine_win", 1)))
    mutations.append(lambda row: (row.__setitem__("expanded_states", -100),
                                  row["proof"].__setitem__("expanded_states", -100)))
    mutations.append(lambda row: (row.__setitem__("planner_backend", "not-a-planner"),
                                  row["proof"].__setitem__("planner_backend", "not-a-planner")))
    mutations.append(lambda row: row["proof"].__setitem__("planner_backend", []))
    mutations.append(lambda row: row["proof"].__setitem__("planner_backend", {}))
    mutations.append(lambda row: row["proof"].__setitem__("rejection_count", -10))
    mutations.append(lambda row: (row.__setitem__("attempt", -1),
                                  row.__setitem__("generation_attempt", 999_999)))
    mutations.append(lambda row: row.__setitem__("effective_split", "garbage"))
    mutations.append(lambda row: row.__setitem__("generation_exclusions", {"x": "bad"}))
    mutations.append(lambda row: row["generation_diagnostics"].__setitem__("rejections", None))
    mutations.append(lambda row: row.__setitem__("solution_mechanics", []))
    mutations.append(lambda row: row.__setitem__("whole_game_context_replay", None))
    for mutate in mutations:
        changed = copy.deepcopy(tier_two)
        mutate(changed)
        assert validate_full_standard(changed, _entry(2))

    for malformed in ([1], [[]], [[1.0, None, None]], [[1, 42, 23]]):
        changed = copy.deepcopy(tier_two)
        changed["solution"] = malformed
        changed["context_solution"] = copy.deepcopy(malformed)
        assert validate_full_standard(changed, _entry(2))

    for field, value in (("context_index", True), ("search_work", 300_000.0)):
        curriculum = copy.deepcopy(_entry(2))
        curriculum[field] = value
        assert validate_full_standard(tier_two, curriculum)

    relabeled = copy.deepcopy(tier_two)
    relabeled["seed"] = relabeled["requested_seed"] = relabeled["effective_seed"] = 999
    relabeled["proof"]["seed"] = 999
    assert any("does not reproduce draft field" in error for error in
               validate_full_standard(relabeled, _entry(2)))

    measured = copy.deepcopy(tier_two)
    work = {
        "search_work": measured["expanded_states"],
        "search_work_limit": measured["search_limit"],
        "search_work_unit": "exact_transition_evaluations",
        "native_admission_replay_actions": measured["solution_length"],
    }
    measured.update(work)
    measured["proof"].update(work)
    assert validate_full_standard(measured, _entry(2)) == []
    measured["search_work"] += 1
    measured["proof"]["search_work"] += 1
    assert any("aliases/unit" in error for error in validate_full_standard(measured, _entry(2)))


def test_semantic_identity_canonicalizes_sets_and_region_unions_but_keeps_actor_order():
    fenced = generate(0, 3)
    assert fenced is not None
    duplicated = copy.deepcopy(fenced)
    duplicated["fences"].extend([copy.deepcopy(fenced["fences"][0])] * 2)
    assert geometry_hash(duplicated) == geometry_hash(fenced)
    assert geometry_partition(duplicated) == geometry_partition(fenced)
    assert gameplay_hash(duplicated) == gameplay_hash(fenced)
    assert any("does not reproduce draft field fences" in error for error in
               validate_full_standard(duplicated, _entry(3)))

    regions = copy.deepcopy(tier := generate(0, 2))
    assert tier is not None
    cells = sorted({
        (col + dx, row + dy)
        for col, row, width, height in tier["goals"]
        for dx in range(width)
        for dy in range(height)
    })
    regions["goals"] = [[col, row, 1, 1] for col, row in reversed(cells)]
    assert geometry_hash(regions) == geometry_hash(tier)
    assert geometry_partition(regions) == geometry_partition(tier)
    assert gameplay_hash(regions) == gameplay_hash(tier)

    reordered = copy.deepcopy(tier)
    reordered["boxes"][0], reordered["boxes"][1] = reordered["boxes"][1], reordered["boxes"][0]
    assert gameplay_hash(reordered) != gameplay_hash(tier)
    # Geometry grouping may ignore assignment order; executable gameplay may not.
    assert geometry_hash(reordered) == geometry_hash(tier)

    assert extract(Env([build_level(tier)])).rotation == 0
    salted = copy.deepcopy(tier)
    salted["player_rotation"] = 90
    assert gameplay_hash(salted) == gameplay_hash(tier)
    assert geometry_hash(salted) == geometry_hash(tier)
    assert any("player_rotation is unsupported" in error for error in
               validate_full_standard(salted, _entry(2)))


def test_official_semantic_copy_is_denied_despite_singleton_goal_art():
    assert len(_official_semantic_hashes()) == 9
    base = generate(0, 1)
    assert base is not None
    layout = extract(Env())
    attack = copy.deepcopy(base)
    attack.update(
        walls=[list(cell) for cell in sorted(layout.walls)],
        fences=[list(cell) for cell in sorted(layout.fences)],
        goals=[[x, y, 1, 1] for x, y in sorted(layout.goals)],
        bad_regions=[[x, y, 1, 1] for x, y in sorted(layout.bad)],
        boxes=[list(cell) for cell in layout.boxes],
        helpers=[list(cell) for cell in layout.helpers],
        thieves=[list(cell) for cell in layout.thieves],
        player=list(layout.player),
        budget=layout.max_steps,
    )
    attack.update(structural_metrics(attack))
    canonical, split = geometry_partition(attack)
    attack.update(
        geometry_sha256=geometry_hash(attack), geometry_d4_sha256=canonical,
        geometry_split=split, split=split, effective_split=split,
        attempt=1, generation_attempt=1,
    )
    certified, reason = certify(attack)
    assert certified is None
    assert reason == "official_semantic_copy"
    assert is_official_semantic_copy(attack)


def test_explicit_subset_is_labeled_smoke_and_child_seed_is_bound():
    specs = generate_game(0, difficulties=(1,))
    assert specs is not None and len(specs) == 1
    report = last_game_generation_report()
    assert report["scope"] == "explicit_smoke_subset"
    assert report["terminal_reason"] == "accepted_explicit_smoke_subset_without_whole_game_claim"
    assert specs[0]["sequence_kind"] == "explicit-smoke-subset"
    assert validate_full_standard(specs[0], _entry(1)) == []
    with pytest.raises(ValueError, match="exactly nine"):
        build_game(specs)

    relabeled = copy.deepcopy(specs[0])
    relabeled["seed"] += 1
    relabeled["requested_seed"] = relabeled["effective_seed"] = relabeled["seed"]
    relabeled["proof"]["seed"] = relabeled["seed"]
    assert any("child seed" in error for error in validate_full_standard(relabeled, _entry(1)))
