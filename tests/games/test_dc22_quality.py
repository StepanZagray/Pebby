"""Bounded reference-calibrated quality audit for generated DC22."""

from collections import Counter
from copy import deepcopy

from pebby.games.dc22 import names
from pebby.games.dc22.env import Env, official_levels
from pebby.games.dc22.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    SPLITS,
    build_level,
    generate,
    validate_full_standard,
)
from pebby.games.dc22.env import replay
from pebby.games.dc22.plan import _button_clicks
from pebby.games.dc22.reference_profiles import PROFILES, structural_metrics


def test_reference_profiles_match_real_context_playfield_sampling():
    levels = official_levels()
    assert len(levels) == len(DIFFICULTIES) == 6
    for difficulty, raw in enumerate(levels, 1):
        env = Env()
        env.set_level(difficulty - 1)
        points = {
            (x, y)
            for y in range(0, env.level.grid_size[1], 2)
            for x in range(0, 40, 2)
            if env.game.sxnzvaqltp(x, y, env.player) is not None
        }
        bbox = (
            min(x for x, _ in points),
            min(y for _, y in points),
            max(x for x, _ in points),
            max(y for _, y in points),
        )
        profile = PROFILES[difficulty]
        assert tuple(raw.grid_size) == profile["grid_size"]
        assert env.max_steps == profile["step_budget"]
        assert len(raw.get_sprites()) == profile["raw_sprites"]
        assert len(env.level.get_sprites()) == profile["engine_sprites"]
        assert len(points) == profile["support_samples"]
        assert bbox == profile["support_bbox"]


def test_eight_seeds_per_tier_have_native_proofs_and_semantic_diversity():
    minimum_stage_relations = {1: 2, 2: 2, 3: 2, 4: 4, 5: 3, 6: 4}
    for difficulty in DIFFICULTIES:
        specs = [generate(seed, difficulty, split="train") for seed in range(8)]
        assert all(spec is not None for spec in specs)
        action_sequences = {tuple(action[0] for action in spec["solution"]) for spec in specs}
        geometry = {spec["geometry_d4_sha256"] for spec in specs}
        gameplay = {spec["gameplay_sha256"] for spec in specs}
        stage_relations = {spec["mechanic_stage_sha256"] for spec in specs}
        stage_actions = {spec["stage_action_sha256"] for spec in specs}
        assert len(action_sequences) >= 6
        assert len(geometry) >= 7
        assert len(gameplay) >= 7
        assert len(stage_relations) >= minimum_stage_relations[difficulty]
        assert len(stage_actions) >= (1 if difficulty == 1 else minimum_stage_relations[difficulty])

        for spec in specs:
            entry = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            assert validate_full_standard(spec, entry) == []
            assert set(PROFILES[difficulty]["required_mechanics"]).issubset(
                set(spec["required_mechanics"])
            )
            for mechanic in spec["required_winning_mechanics"]:
                if mechanic != "fall_recovery":
                    assert spec["solution_mechanics"].get(mechanic, 0) >= 1
            for mechanic in spec["required_interaction_mechanics"]:
                assert spec["interaction_mechanics"][mechanic].get(mechanic, 0) >= 1


def test_every_winning_route_click_is_necessary_not_mechanic_padding():
    for difficulty in DIFFICULTIES:
        for seed in range(8):
            spec = generate(seed, difficulty, split="train")
            assert spec is not None
            for index, action in enumerate(spec["solution"]):
                if action[0] != 6:
                    continue
                contextual = Env([build_level(spec) for _ in range(difficulty)])
                contextual.set_level(difficulty - 1)
                shortened = spec["solution"][:index] + spec["solution"][index + 1:]
                assert not replay(contextual, shortened), (
                    difficulty, seed, index, action, spec["stage_blueprint"]
                )

    tier6 = generate(0, 6, split="train")
    assert tier6 is not None
    assert tier6["interaction_mechanics"] == {}
    assert tier6["interaction_probes"] == {}
    assert tier6["solution_mechanics"]["bridge_colour_cycle"] == 2


def test_canonical_d4_partition_is_disjoint_across_all_tiers_and_splits():
    identities = {split: set() for split in SPLITS}
    for split in SPLITS:
        for difficulty in DIFFICULTIES:
            for seed in range(3):
                spec = generate(seed, difficulty, split=split)
                assert spec is not None
                assert spec["geometry_split"] == split
                identities[split].add(spec["geometry_d4_sha256"])
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            assert identities[left].isdisjoint(identities[right])


def test_generated_frames_keep_native_panel_cues_palette_and_reference_density():
    official = official_levels()
    for difficulty in DIFFICULTIES:
        spec = generate(0, difficulty, split="train")
        assert spec is not None
        level = build_level(spec)
        names_in_level = {sprite.name for sprite in level.get_sprites()}
        expected_panel = "coorbs-bg-1" if difficulty == 6 else "coorbs-bg"
        expected_separator = "merged-sprite-1" if difficulty == 6 else "merged-sprite"
        assert expected_panel in names_in_level
        assert expected_separator in names_in_level

        contextual = Env([level for _ in range(difficulty)])
        contextual.set_level(difficulty - 1)
        frame = contextual.render()
        assert len(frame) == 64 and all(len(row) == 64 for row in frame)
        palette = Counter(pixel for row in frame for pixel in row)
        assert set(palette) <= set(range(16))
        assert len(palette) >= 6
        assert sum(pixel != 4 for row in frame for pixel in row[40:]) >= 100
        clicks = _button_clicks(contextual)
        assert clicks or contextual.level.get_sprites_by_tag("piyqze")
        assert all(frame[y][x] != 4 for x, y in clicks)

        measured = structural_metrics(level)
        reference = PROFILES[difficulty]
        low, high = reference["generated_support_range"]
        assert low <= measured["support_samples"] <= high
        if difficulty >= 2:
            assert measured["support_samples"] >= int(reference["support_samples"] * 0.6)

        official_frame = Env([official[difficulty - 1]]).render()
        assert len({pixel for row in official_frame for pixel in row}) >= 6


def test_rejection_accounting_is_explicit_and_only_partition_rejects_sampled_rows():
    totals = Counter()
    for difficulty in DIFFICULTIES:
        for seed in range(8):
            spec = generate(seed, difficulty, split="train")
            assert spec is not None
            totals.update(spec["generation_exclusions"])
    assert set(totals) <= {"geometry_split"}
    assert totals["geometry_split"] > 0


def test_official_coupled_compositions_are_installed_and_natively_witnessed():
    tier4 = generate(0, 4, split="train")
    tier5 = generate(0, 5, split="train")
    tier6 = generate(0, 6, split="train")
    assert tier4 is not None and tier5 is not None and tier6 is not None

    tier4_prototypes = [row["prototype"] for row in tier4["components"]]
    assert tier4_prototypes.count("drfztmbrixto-1") == 1
    assert tier4_prototypes.count("drfztmbrixto-5") == 1
    assert tier4_prototypes.count("moxubw-plelvb-1") == 2
    assert tier4["solution_mechanics"]["expanding_surface_instances"] >= 2
    assert tier4["solution_mechanics"]["moving_surface_instances"] >= 2

    tier5_prototypes = {row["prototype"] for row in tier5["components"]}
    assert {"drfztmbrixto-1", "drfztmbrixto-buezna", "moxubw-plelvb-1", "sprite-6"} <= tier5_prototypes
    assert tier5["solution_mechanics"]["expanding_surface"] >= 1
    assert tier5["solution_mechanics"]["moving_surface"] >= 1
    assert {"crusher_move", "object_carry", "key_pickup", "bridge_teleport"} <= set(
        tier5["required_winning_mechanics"]
    )

    tier6_prototypes = [row["prototype"] for row in tier6["components"]]
    selective_sources = [
        row for row in tier6["components"]
        if names.TAG_BRIDGE_COLOR_CYCLE in row.get("extra_tags", ())
    ]
    assert len(selective_sources) == 1
    assert tier6_prototypes.count("tewfutpibpar1") == 2
    assert "tewfutyefmyf1" in tier6_prototypes
    assert tier6["solution_mechanics"]["bridge_colour_cycle"] >= 2
    assert tier6["solution_mechanics"]["bridge_teleport"] >= 1
    assert "bridge_colour_cycle" in tier6["required_winning_mechanics"]


def test_tier4_paired_surfaces_are_native_route_dependencies_across_eight_seeds():
    def trace_and_outcome(spec):
        env = Env([build_level(spec) for _ in range(4)])
        env.set_level(3)
        trace = []
        for action in spec["solution"]:
            observation = env.perform(*action)
            trace.append((
                env.player.x, env.player.y, env.steps_left,
                observation.state.name, env.levels_completed,
            ))
        return trace, env.levels_completed

    for seed in range(8):
        spec = generate(seed, 4, split="train")
        assert spec is not None
        certificate = spec["paired_use"]
        assert set(certificate) == {
            "phase1_expander", "phase5_expander", "upper_mover", "lower_mover",
        }
        baseline, completed = trace_and_outcome(spec)
        assert completed == 1
        for role, evidence in certificate.items():
            assert evidence["first_trace_divergence"] >= 1
            assert evidence["won"] is False
            ablated = deepcopy(spec)
            target = evidence["component"]
            ablated["components"] = [
                component for component in ablated["components"]
                if component != target
            ]
            changed, changed_completed = trace_and_outcome(ablated)
            assert changed != baseline, (seed, role)
            assert changed_completed == 0, (seed, role)
