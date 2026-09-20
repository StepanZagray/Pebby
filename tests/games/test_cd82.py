"""CD82: real-engine env, exact planner, verified generator."""

import json
import copy
import time
import unittest

from arcengine import GameState

from pebby.games.cd82 import names
from pebby.games.cd82.bank import build as build_bank, load as load_bank, save as save_bank
from pebby.games.cd82.env import Env, official_levels
from pebby.games.cd82.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    generate,
    generate_episode,
    generate_game,
    generate_report,
    layout_for,
    solution_mechanics,
    tier1_split_capacities,
    tier1_split_support,
    validate_full_standard,
)
from pebby.games.cd82.layout import extract
from pebby.games.cd82.plan import search, solve
from pebby.games.cd82.generation_quality import (
    canonical_identities,
    geometry_split,
    raw_geometry_identity,
)
from pebby.games.cd82.reference_profiles import PROFILES, official_characterization, profile_errors


def replay(env, actions):
    observation = None
    for action_id, x, y in actions:
        assert action_id in env.available_actions
        observation = env.perform(action_id, x, y)
    return observation


class OfficialLevels(unittest.TestCase):
    def test_native_budget_has_100_action_loss_threshold_and_99_usable_actions(self):
        env = Env()
        env.reset()
        for _ in range(names.MAX_ACTIONS):
            observation = env.perform(names.ACTION_UP)
        self.assertFalse(observation.finished)
        self.assertEqual(env.actions_used, 99)
        observation = env.perform(names.ACTION_UP)
        self.assertEqual(env.actions_used, 100)
        self.assertEqual(observation.state, GameState.GAME_OVER)

    def test_measured_teacher_mechanics_replay_at_every_official_tier(self):
        env = Env()
        env.reset()
        for index, reference in enumerate(official_characterization()):
            layout = extract(env)
            result = search(layout)
            self.assertTrue(result.solved, index + 1)
            spec = {
                "target": env.target().tolist(),
                "indicator": env.has_indicator(),
                "palette": [colour for colour, _, _ in env.swatches()],
                "swatches": env.swatches(),
            }
            mechanics = solution_mechanics(spec, result.actions)
            self.assertEqual(result.length, reference["optimal_actions"])
            for name, count in reference["solution_mechanics"].items():
                self.assertEqual(mechanics[name], count)
            replay(env, result.actions)
        self.assertEqual(env.levels_completed, 6)

    def test_reference_contract_covers_all_six_measured_official_tiers(self):
        rows = official_characterization()
        self.assertEqual(DIFFICULTIES, tuple(range(1, 7)))
        self.assertEqual([row["optimal_actions"] for row in rows], [5, 6, 16, 13, 13, 16])
        self.assertEqual([row["palette_count"] for row in rows], [2, 3, 7, 7, 7, 7])
        self.assertEqual([row["indicator"] for row in rows], [False, False, True, True, True, True])
        self.assertEqual([row["target_colours"] for row in rows], [2, 3, 4, 4, 4, 5])
        self.assertEqual(tuple(PROFILES), DIFFICULTIES)

        curriculum = FULL_STANDARD_CONTRACT["curriculum"]
        self.assertEqual(FULL_STANDARD_CONTRACT["format"], "pebby-full-generator-contract-v1")
        self.assertEqual(FULL_STANDARD_CONTRACT["source_id"], "cd82-fb555c5d")
        self.assertEqual(FULL_STANDARD_CONTRACT["status"], "ready")
        self.assertEqual([entry["difficulty"] for entry in curriculum], list(DIFFICULTIES))
        self.assertEqual([entry["context_index"] for entry in curriculum], list(range(6)))
        self.assertTrue(all(type(entry["search_work"]) is int
                            and 1 <= entry["search_work"] <= 32_000_000
                            for entry in curriculum))
        required_evidence = {"official_tier_characterization", "solution_mechanics",
                             "native_budget", "context_engine_replay", "novelty_split",
                             "bounded_rejections"}
        self.assertEqual(set(FULL_STANDARD_CONTRACT["evidence"]), required_evidence)
        self.assertTrue(all(FULL_STANDARD_CONTRACT["evidence"].values()))
        self.assertEqual(FULL_STANDARD_CONTRACT["evidence"], {
            "official_tier_characterization": (
                "docs/generator-evidence/cd82.md"
                "#official-reference-characterization"
            ),
            "solution_mechanics": (
                "docs/generator-evidence/cd82.md"
                "#procedural-generation-and-proof"
            ),
            "native_budget": (
                "docs/generator-evidence/cd82.md"
                "#authoritative-source-and-complete-mechanics"
            ),
            "context_engine_replay": (
                "docs/generator-evidence/cd82.md"
                "#root-acceptance-2026-09-18"
            ),
            "novelty_split": (
                "docs/generator-evidence/cd82.md"
                "#identity-split-and-finite-support"
            ),
            "bounded_rejections": (
                "docs/generator-evidence/cd82.md"
                "#bounded-quality-audit"
            ),
        })
        self.assertTrue(all(FULL_STANDARD_CONTRACT["caveats"]))

    def test_level_1_solution_completes_in_the_real_engine(self):
        env = Env()
        env.reset()
        actions = solve(env)
        self.assertIsNotNone(actions)
        self.assertFalse(solve.truncated)
        self.assertEqual(len(actions), 5)  # click colour, 4 dial presses... exactly optimal
        observation = replay(env, actions)
        self.assertEqual(env.levels_completed, 1)
        self.assertEqual(env.level_index, 1)
        self.assertEqual(observation.state, GameState.NOT_FINISHED)

    def test_level_2_solution_completes(self):
        env = Env()
        env.reset()
        env.set_level(1)
        actions = solve(env)
        self.assertIsNotNone(actions)
        replay(env, actions)
        self.assertEqual(env.levels_completed, 1)
        self.assertEqual(env.level_index, 2)

    def test_whole_game_is_won_level_by_level(self):
        env = Env()
        env.reset()
        total = 0
        for index in range(len(official_levels())):
            result = search(env)
            self.assertTrue(result.solved, f"level {index + 1}: {result.reason}")
            self.assertFalse(result.truncated)
            self.assertLessEqual(len(result.actions), names.MAX_ACTIONS)
            observation = replay(env, result.actions)
            total += len(result.actions)
            self.assertEqual(env.levels_completed, index + 1)
        self.assertTrue(observation.won)
        self.assertEqual(env.state, GameState.WIN)
        self.assertEqual(total, 69)

    def test_indicator_click_matches_the_games_own_coordinates(self):
        env = Env()
        env.reset()
        env.set_level(2)  # first level shipping the indicator
        self.assertTrue(env.has_indicator())
        for dial in names.EDGE_DIALS:
            for action in names.ring_path(env.dial(), dial):
                env.perform(action)
            self.assertEqual(env.dial(), dial)
            self.assertEqual(env.valid_clicks()[-1], names.indicator_click(dial))
        self.assertEqual(env.valid_clicks()[:7], [(x, y) for _, x, y in env.swatches()])

    def test_dial_model_matches_the_engine(self):
        env = Env()
        env.reset()
        dial = env.dial()
        for action in (1, 4, 4, 2, 2, 2, 3, 3, 3, 1, 1, 1, 4, 2):
            dial = names.move_dial(dial, action)
            env.perform(action)
            self.assertEqual(env.dial(), dial)

    def test_clone_is_independent(self):
        env = Env()
        env.reset()
        twin = env.clone()
        env.perform(4)
        self.assertNotEqual(env.dial(), twin.dial())
        self.assertEqual(twin.render(), Env().render())


class Generation(unittest.TestCase):
    def test_generate_report_requires_a_strict_integer_difficulty(self):
        for difficulty in (True, False, 1.0, 2.0):
            with self.subTest(difficulty=difficulty):
                with self.assertRaisesRegex(ValueError, "difficulty must be one of"):
                    generate_report(0, difficulty)

    def test_generate_game_rejects_descending_smoke_difficulties(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            generate_game(0, split="train", difficulties=(2, 1), attempts=1)

    def test_validator_requires_strict_integer_curriculum_coordinates(self):
        spec = generate(7, 1, attempts=120, split="train")
        self.assertIsNotNone(spec)
        base = FULL_STANDARD_CONTRACT["curriculum"][0]
        for field, value in (("difficulty", True), ("difficulty", 1.0),
                             ("context_index", False), ("context_index", 0.0)):
            with self.subTest(field=field, value=value):
                entry = dict(base, **{field: value})
                self.assertTrue(validate_full_standard(spec, entry))

    def test_validator_returns_errors_for_malformed_certificate_fields(self):
        spec = generate(7, 1, attempts=120, split="train")
        self.assertIsNotNone(spec)
        entry = FULL_STANDARD_CONTRACT["curriculum"][0]

        malformed = []
        for field, value in (("proof", None), ("solution_mechanics", []),
                             ("search_limit", "not-an-integer"),
                             ("search_expanded", "not-an-integer"),
                             ("context_index", False), ("target", None),
                             ("palette", None)):
            changed = copy.deepcopy(spec)
            changed[field] = value
            malformed.append((field, changed))

        float_limit = copy.deepcopy(spec)
        float_limit["search_limit"] = float(spec["search_limit"])
        float_limit["proof"]["search_limit"] = float(spec["search_limit"])
        malformed.append(("float search_limit", float_limit))

        for label, changed in malformed:
            with self.subTest(field=label):
                errors = validate_full_standard(changed, entry)
                self.assertIsInstance(errors, list)
                self.assertTrue(errors)

    def test_validator_strictly_rejects_tampered_certificate_metadata(self):
        spec = generate(7, 1, attempts=120, split="train")
        self.assertIsNotNone(spec)
        entry = FULL_STANDARD_CONTRACT["curriculum"][0]
        self.assertEqual(validate_full_standard(spec, entry), [])

        top_level_mutations = {
            "reference_level": True,
            "context_index": False,
            "training_context_index": 0.0,
            "verification_level_index": 5,
            "optimal_actions": float(spec["optimal_actions"]),
            "solution_length": float(spec["solution_length"]),
            "search_limit": float(spec["search_limit"]),
            "search_expanded": float(spec["search_expanded"]),
            "search_truncated": 0,
            "engine_budget": 100.0,
            "usable_actions": 99.0,
            "max_actions": float(spec["max_actions"]),
            "context_engine_verified": "yes",
            "engine_verified": "yes",
        }
        for field, value in top_level_mutations.items():
            with self.subTest(scope="top", field=field, value=value):
                changed = copy.deepcopy(spec)
                changed[field] = value
                self.assertTrue(validate_full_standard(changed, entry))

        required_top_level = (
            "reference_level", "context_index", "training_context_index",
            "verification_level_index", "optimal_actions", "solution_length",
            "search_limit", "search_expanded", "search_truncated", "engine_budget",
            "usable_actions", "max_actions", "context_engine_verified", "engine_verified",
        )
        for field in required_top_level:
            with self.subTest(scope="top missing", field=field):
                changed = copy.deepcopy(spec)
                del changed[field]
                self.assertTrue(validate_full_standard(changed, entry))

        proof_mutations = {
            "seed": float(spec["seed"]),
            "difficulty": True,
            "context_index": False,
            "level_count": 600,
            "native_budget": 100.0,
            "usable_actions": 999,
            "optimal_actions": float(spec["optimal_actions"]),
            "search_limit": float(spec["search_limit"]),
            "search_expanded": float(spec["search_expanded"]),
            "search_truncated": 0,
            "level_advanced": "yes",
            "levels_completed": 0,
            "actual_display_coordinates": False,
        }
        for field, value in proof_mutations.items():
            with self.subTest(scope="proof", field=field, value=value):
                changed = copy.deepcopy(spec)
                changed["proof"][field] = value
                self.assertTrue(validate_full_standard(changed, entry))

        for field in tuple(spec["proof"]):
            with self.subTest(scope="proof missing", field=field):
                changed = copy.deepcopy(spec)
                del changed["proof"][field]
                self.assertTrue(validate_full_standard(changed, entry))

    def test_validator_strictly_checks_optional_episode_certificate(self):
        spec = generate(7, 1, attempts=120, split="train")
        self.assertIsNotNone(spec)
        entry = FULL_STANDARD_CONTRACT["curriculum"][0]
        episode_sha256 = "a" * 64
        spec.update(
            episode_engine_verified=True,
            episode_index=0,
            episode_levels_completed=len(DIFFICULTIES),
            episode_sha256=episode_sha256,
        )
        spec["proof"].update(
            episode_engine_verified=True,
            episode_index=0,
            episode_levels_completed=len(DIFFICULTIES),
            episode_sha256=episode_sha256,
            forced_transitions=0,
        )
        self.assertEqual(validate_full_standard(spec, entry), [])

        top_mutations = {
            "episode_engine_verified": "yes",
            "episode_index": False,
            "episode_levels_completed": float(len(DIFFICULTIES)),
            "episode_sha256": "not-a-sha256",
        }
        proof_mutations = {
            "episode_engine_verified": "yes",
            "episode_index": False,
            "episode_levels_completed": float(len(DIFFICULTIES)),
            "episode_sha256": "b" * 64,
            "forced_transitions": False,
        }
        for scope, mutations in (("top", top_mutations), ("proof", proof_mutations)):
            for field, value in mutations.items():
                with self.subTest(scope=scope, field=field, value=value):
                    changed = copy.deepcopy(spec)
                    target = changed if scope == "top" else changed["proof"]
                    target[field] = value
                    self.assertTrue(validate_full_standard(changed, entry))

        for scope, fields in (("top", tuple(top_mutations)),
                              ("proof", tuple(proof_mutations))):
            for field in fields:
                with self.subTest(scope=f"{scope} missing", field=field):
                    changed = copy.deepcopy(spec)
                    target = changed if scope == "top" else changed["proof"]
                    del target[field]
                    self.assertTrue(validate_full_standard(changed, entry))

    def test_build_game_rejects_unhashable_identity_fields_cleanly(self):
        for field, value in (("seed", []), ("gameplay_sha256", []),
                             ("geometry_sha256", {})):
            with self.subTest(field=field):
                specs = [
                    {
                        "difficulty": difficulty,
                        "context_index": difficulty - 1,
                        "split": "train",
                        "seed": difficulty,
                        "gameplay_sha256": f"gameplay-{difficulty}",
                        "geometry_sha256": f"geometry-{difficulty}",
                    }
                    for difficulty in DIFFICULTIES
                ]
                specs[0][field] = value
                with self.assertRaisesRegex(ValueError, f"invalid {field}"):
                    build_game(specs)

    def test_tier1_bank_reports_exact_finite_split_capacity(self):
        self.assertEqual(tier1_split_support(), {
            "train": {"d4_classes": 1, "raw_capacity": 6},
            "validation": {"d4_classes": 1, "raw_capacity": 6},
            "test": {"d4_classes": 1, "raw_capacity": 4},
        })
        self.assertEqual(tier1_split_capacities(),
                         {"train": 6, "validation": 6, "test": 4})
        build_bank(7, seed=0, difficulty=1, max_attempts=1, split="train")
        self.assertEqual(build_bank.last_report["known_capacity"], 6)
        self.assertTrue(build_bank.last_report["capacity_limited"])

    def test_validator_replays_witness_and_rejects_tampered_or_malformed_routes(self):
        entry = FULL_STANDARD_CONTRACT["curriculum"][0]
        spec = generate(7, 1, attempts=120, split="train")
        self.assertIsNotNone(spec)

        tampered = copy.deepcopy(spec)
        tampered["solution"][0] = [names.ACTION_UP, None, None]
        tampered["solution_mechanics"] = solution_mechanics(
            tampered, [tuple(action) for action in tampered["solution"]])
        self.assertTrue(validate_full_standard(tampered, entry))

        malformed = copy.deepcopy(spec)
        malformed["solution"] = [["click", None, None]] * spec["solution_length"]
        self.assertTrue(validate_full_standard(malformed, entry))

    def test_generate_game_and_build_game_require_exact_ordered_full_curriculum(self):
        specs = generate_game(0, split="train")
        self.assertIsNotNone(specs, generate_game.last_report)
        self.assertEqual(len(specs), len(official_levels()))
        self.assertEqual([spec["difficulty"] for spec in specs], list(DIFFICULTIES))
        self.assertEqual(len({spec["seed"] for spec in specs}), len(DIFFICULTIES))
        self.assertEqual(generate_game(0, split="train"), specs)

        levels = build_game(specs)
        env = Env(levels)
        env.reset()
        for index, spec in enumerate(specs):
            self.assertEqual(env.level_index, index)
            observation = replay(env, [tuple(action) for action in spec["solution"]])
        self.assertTrue(observation.won)
        self.assertEqual(env.levels_completed, len(DIFFICULTIES))

        invalid_sequences = [specs[:-1], specs + [specs[-1]], list(reversed(specs))]
        duplicate = list(specs)
        duplicate[1] = duplicate[0]
        invalid_sequences.append(duplicate)
        mixed = copy.deepcopy(specs)
        mixed[-1]["split"] = "validation"
        invalid_sequences.append(mixed)
        for invalid in invalid_sequences:
            with self.assertRaises(ValueError):
                build_game(invalid)

    def test_explicit_split_requests_are_canonical_and_invalid_values_fail(self):
        specs = [generate(0, 1, attempts=200, split=split_name)
                 for split_name in ("train", "validation", "test")]
        self.assertTrue(all(spec is not None for spec in specs))
        self.assertEqual([spec["split"] for spec in specs], ["train", "validation", "test"])
        self.assertEqual(len({spec["geometry_d4_sha256"] for spec in specs}), 3)
        with self.assertRaises(ValueError):
            generate(0, 1, split="holdout")

    def test_live_prefix_replans_and_reset_recovers_the_generated_level(self):
        spec = generate(4, 1, attempts=200, split="train")
        self.assertIsNotNone(spec)
        actions = [tuple(action) for action in spec["solution"]]

        env = Env([build_level(spec)])
        env.reset()
        replay(env, actions[:3])
        recovery = search(env)
        self.assertTrue(recovery.solved)
        self.assertFalse(recovery.truncated)
        self.assertTrue(replay(env, recovery.actions).won)

        env = Env([build_level(spec)])
        env.reset()
        replay(env, actions[:3])
        env.perform(0)
        self.assertEqual(extract(env).start, layout_for(spec).start)
        self.assertTrue(replay(env, search(env).actions).won)

    def test_generated_episode_advances_all_six_tiers_without_forced_transitions(self):
        specs = generate_episode(0, split="train")
        self.assertIsNotNone(specs)
        self.assertEqual([spec["difficulty"] for spec in specs], list(DIFFICULTIES))
        env = Env([build_level(spec) for spec in specs])
        env.reset()
        for index, spec in enumerate(specs):
            self.assertEqual(env.level_index, index)
            observation = replay(env, [tuple(action) for action in spec["solution"]])
            self.assertEqual(env.levels_completed, index + 1)
        self.assertTrue(observation.won)
        self.assertEqual(env.state, GameState.WIN)
        self.assertTrue(all(spec["episode_engine_verified"] for spec in specs))

    def test_profile_validation_rejects_split_and_identity_relabeling(self):
        spec = generate(2, 1, attempts=200, split="train")
        self.assertIsNotNone(spec)
        relabeled = copy.deepcopy(spec)
        relabeled["split"] = "validation"
        self.assertTrue(profile_errors(relabeled))
        relabeled = copy.deepcopy(spec)
        relabeled["target"][0][1] = 15 if relabeled["target"][0][1] == 0 else 0
        self.assertTrue(profile_errors(relabeled))

    def test_every_full_tier_is_profile_valid_and_exercises_declared_mechanics(self):
        for difficulty, entry in zip(DIFFICULTIES, FULL_STANDARD_CONTRACT["curriculum"]):
            with self.subTest(difficulty=difficulty):
                spec = generate(0, difficulty, attempts=120, split="train")
                self.assertIsNotNone(spec)
                self.assertEqual(profile_errors(spec), [])
                self.assertEqual(validate_full_standard(spec, entry), [])
                self.assertEqual(spec["split"], "train")
                for mechanic, minimum in PROFILES[difficulty]["minimum_mechanics"].items():
                    self.assertGreaterEqual(spec["solution_mechanics"][mechanic], minimum)

    def test_canonical_identities_follow_executable_target_semantics(self):
        target = [[0] * 10 for _ in range(10)]
        for row in range(10):
            for col in range(10):
                if row != col and row + col != 9:
                    if row < 2:
                        target[row][col] = 0
                    elif row < 7 and col < 3:
                        target[row][col] = 15
                    else:
                        target[row][col] = 12
        spec = {"target": target, "indicator": True, "palette": [0, 15, 12, 14]}
        gameplay, geometry_d4 = canonical_identities(spec)
        geometry = raw_geometry_identity(spec)

        cosmetic = [list(row) for row in target]
        for index in range(10):
            cosmetic[index][index] = 9
            cosmetic[index][9 - index] = 11
        cosmetic = [[8 if value == 12 else value for value in row] for row in cosmetic]
        changed = {
            "target": cosmetic,
            "indicator": True,
            "palette": [14, 8, 15, 0],
            "seed": 999,
            "solution": [[names.ACTION_UP, None, None]],
            "proof": {"private": "ignored"},
        }
        self.assertEqual(canonical_identities(changed), (gameplay, geometry_d4))
        self.assertEqual(raw_geometry_identity(changed), geometry)

        rotated = [list(row) for row in zip(*target[::-1])]
        rotated = [[8 if value == 12 else value for value in row] for row in rotated]
        self.assertEqual(canonical_identities({"target": rotated, "indicator": True,
                                               "palette": [8, 0, 14, 15]})[1], geometry_d4)

        for left, right in ((0, 12), (15, 12)):
            relabeled = [[{left: right, right: left}.get(value, value)
                          for value in row] for row in target]
            relabeled_spec = {"target": relabeled, "indicator": True,
                              "palette": [0, 15, 12, 14]}
            with self.subTest(canonical_color_swap=(left, right)):
                self.assertEqual(canonical_identities(relabeled_spec),
                                 (gameplay, geometry_d4))
                self.assertEqual(raw_geometry_identity(relabeled_spec), geometry)

        structural_change = [list(row) for row in target]
        structural_change[0][1] = 15
        structural_spec = {"target": structural_change, "indicator": True,
                           "palette": [0, 15, 12, 14]}
        self.assertNotEqual(canonical_identities(structural_spec)[0], gameplay)
        self.assertNotEqual(canonical_identities(structural_spec)[1], geometry_d4)
        self.assertNotEqual(raw_geometry_identity(structural_spec), geometry)

        self.assertEqual(geometry_split(canonical_identities(changed)[1]),
                         geometry_split(geometry_d4))
        self.assertIn(geometry_split(geometry_d4), ("train", "validation", "test"))

    def test_generated_specs_declare_semantic_identity_version(self):
        spec = generate(7, 1, attempts=120, split="train")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["geometry_version"], "cd82-semantic-color-d4-v2")
        self.assertEqual(spec["proof"]["geometry_version"],
                         "cd82-semantic-color-d4-v2")

    def test_official_copy_detection_uses_semantic_d4_identity(self):
        env = Env()
        env.reset()
        env.set_level(5)
        official = {
            "difficulty": 6,
            "context_index": 5,
            "target": env.target().tolist(),
            "indicator": env.has_indicator(),
            "palette": [colour for colour, _, _ in env.swatches()],
        }
        entry = FULL_STANDARD_CONTRACT["curriculum"][5]
        marker = "target is canonically equivalent to an official level"
        self.assertIn(marker, validate_full_standard(official, entry))

        relabel = {8: 9, 9: 11, 11: 12, 12: 14, 14: 8}
        cosmetic = copy.deepcopy(official)
        cosmetic["target"] = [[relabel.get(value, value) for value in row]
                              for row in cosmetic["target"]]
        self.assertEqual(canonical_identities(cosmetic)[1],
                         canonical_identities(official)[1])
        self.assertIn(marker, validate_full_standard(cosmetic, entry))

        relabeled_special = copy.deepcopy(official)
        relabeled_special["target"] = [
            [{0: 8, 8: 0}.get(value, value) for value in row]
            for row in relabeled_special["target"]
        ]
        self.assertEqual(canonical_identities(relabeled_special)[1],
                         canonical_identities(official)[1])
        self.assertIn(marker, validate_full_standard(relabeled_special, entry))

        structurally_changed = copy.deepcopy(official)
        structurally_changed["target"][0][1] = 15
        self.assertNotEqual(canonical_identities(structurally_changed)[1],
                            canonical_identities(official)[1])
        self.assertNotIn(marker, validate_full_standard(structurally_changed, entry))

    def test_difficulty_1_yields_verified_specs_quickly(self):
        started = time.time()
        accepted = [spec for spec in (generate(seed, 1) for seed in range(6)) if spec is not None]
        self.assertGreaterEqual(len(accepted), 3)
        self.assertLess(time.time() - started, 60)
        for spec in accepted:
            self.assertTrue(spec["engine_verified"])
            self.assertFalse(spec["search_truncated"])
            self.assertEqual(spec["solution_length"], len(spec["solution"]))

    def test_generate_is_deterministic(self):
        self.assertEqual(generate(3, 2), generate(3, 2))

    def test_stored_solutions_replay_to_completion_at_every_difficulty(self):
        for difficulty in DIFFICULTIES:
            spec = generate(0, difficulty)
            self.assertIsNotNone(spec, difficulty)
            env = Env([build_level(spec)])
            env.reset()
            self.assertEqual(env.has_indicator(), spec["indicator"])
            self.assertEqual([c for c, _, _ in env.swatches()], spec["palette"])
            observation = replay(env, [tuple(a) for a in spec["solution"]])
            self.assertTrue(observation.won, difficulty)
            self.assertEqual(env.levels_completed, 1)
            self.assertLessEqual(spec["solution_length"], spec["max_actions"])

    def test_spec_json_round_trip_and_bank_io(self):
        spec = generate(1, 2)
        self.assertIsNotNone(spec)
        again = json.loads(json.dumps(spec))
        self.assertEqual(again, spec)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            specs, attempts = build_bank(3, seed=10, difficulty=1)
            self.assertEqual(len(specs), 3)
            path = save_bank(specs, Path(tmp) / "bank.jsonl")
            self.assertEqual(load_bank(path), specs)
        env = Env([build_level(again)])
        env.reset()
        self.assertTrue(replay(env, [tuple(a) for a in again["solution"]]).won)

    def test_bank_rejects_duplicate_gameplay_and_geometry_within_split(self):
        specs, attempts = build_bank(2, seed=0, difficulty=1, max_attempts=80, split="train")
        self.assertEqual(len(specs), 2)
        self.assertLessEqual(attempts, 80)
        self.assertEqual({spec["split"] for spec in specs}, {"train"})
        self.assertEqual(len({spec["gameplay_sha256"] for spec in specs}), 2)
        self.assertEqual(len({spec["geometry_sha256"] for spec in specs}), 2)

    def test_planner_is_exact_about_the_start_layout(self):
        spec = generate(2, 3)
        self.assertIsNotNone(spec)
        env = Env([build_level(spec)])
        env.reset()
        live = extract(env)
        offline = layout_for(spec)
        self.assertEqual(live.start, offline.start)
        self.assertEqual(live.target, offline.target)
        self.assertEqual(live.palette, offline.palette)
        self.assertEqual(len(live.atom_cells), 16 if spec["indicator"] else 8)

    def test_unmatchable_target_is_rejected_without_truncation(self):
        spec = generate(0, 1)
        broken = dict(spec, target=[list(row) for row in spec["target"]])
        broken["target"][0][1] = 9  # not a swatch colour, and not uniform on its atom
        result = search(layout_for(broken))
        self.assertFalse(result.solved)
        self.assertFalse(result.truncated)

    def test_truncation_is_reported(self):
        env = Env()
        env.reset()
        env.set_level(2)
        result = search(env, limit=10)
        self.assertFalse(result.solved)
        self.assertTrue(result.truncated)


if __name__ == "__main__":
    unittest.main()
