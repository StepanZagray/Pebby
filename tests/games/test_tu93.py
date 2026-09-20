"""TU93 full nine-tier generation, proof, and real-engine replay tests."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest

from arcengine import GameState

from pebby import multigame as shared_multigame
from pebby.games.tu93 import names
from pebby.games.tu93.bank import (
    build as build_bank,
    load as load_bank,
    main as bank_main,
    save as save_bank,
)
from pebby.games.tu93.env import Env, official_levels, replay
from pebby.games.tu93.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    SEARCH_LIMIT,
    SOURCE_ID,
    SPLIT_BUCKETS,
    build_game,
    build_level,
    generate,
    generate_game,
    validate_full_standard,
)
from pebby.games.tu93.layout import extract
from pebby.games.tu93.plan import search, solve, transition
from pebby.games.tu93.quality import (
    REFERENCE_PROFILES,
    canonical_identities,
    geometry_identities,
    measure,
    validate_profile,
)


def checked_replay(env, actions):
    observation = None
    for action, x, y in actions:
        if action not in env.available_actions:
            raise AssertionError(f"planner issued unavailable action {action}")
        before = env.levels_completed
        observation = env.perform(action, x, y)
        if env.levels_completed > before or observation.state == GameState.WIN:
            return True, observation
        if observation.state == GameState.GAME_OVER:
            return False, observation
    return False, observation


class OfficialLevels(unittest.TestCase):
    def test_level_1_is_solved_and_replayed_in_the_real_engine(self):
        env = Env()
        env.reset()
        actions = solve(env)
        self.assertIsNotNone(actions)
        self.assertFalse(solve.truncated)
        self.assertTrue(solve.result.exact)
        completed, _ = checked_replay(env, actions)
        self.assertTrue(completed)
        self.assertEqual(env.levels_completed, 1)
        self.assertEqual(env.level_index, 1)

    def test_all_official_tiers_match_references_and_replay_sequentially(self):
        self.assertEqual(len(official_levels()), 9)
        env = Env()
        env.reset()
        observation = None
        for index, difficulty in enumerate(DIFFICULTIES):
            self.assertEqual(env.level_index, index)
            layout = extract(env)
            result = search(layout, limit=500_000)
            self.assertTrue(result.solved, f"official {difficulty}: {result.reason}")
            self.assertFalse(result.truncated)
            metrics = measure(layout, result.actions)
            reference = REFERENCE_PROFILES[difficulty]["reference"]
            self.assertEqual(metrics["solution_length"], reference["shortest_symbolic_actions"])
            for field in (
                "active_nodes", "edges", "cycle_rank", "dead_ends", "junctions",
                "hunters", "patrollers", "tails", "native_budget",
            ):
                self.assertEqual(metrics[field], reference[field], (difficulty, field))
            self.assertEqual(
                [metrics["grid_width"], metrics["grid_height"]],
                reference["grid_size"],
            )
            self.assertEqual([metrics["origin_x"], metrics["origin_y"]], reference["maze_origin"])
            self.assertEqual((metrics["exits"], metrics["controls"]), (1, 4))
            self.assertEqual(validate_profile(difficulty, metrics), [])
            completed, observation = checked_replay(env, result.actions)
            self.assertTrue(completed, f"official {difficulty}")
            self.assertEqual(env.levels_completed, difficulty)
        self.assertEqual(observation.state, GameState.WIN)

    def test_symbolic_moving_enemies_match_real_engine_each_action(self):
        env = Env()
        env.reset()
        env.set_level(8)
        layout = extract(env)
        result = search(layout)
        self.assertTrue(result.solved)
        state = layout.key()
        for action, _, _ in result.actions:
            outcome = transition(layout, state, action)
            self.assertIsNotNone(outcome)
            state, won, dead = outcome
            self.assertFalse(dead)
            env.perform(action)
            if won:
                self.assertEqual(env.state, GameState.WIN)
                break
            self.assertEqual(extract(env).key(), state)
        else:
            self.fail("symbolic solution did not reach the exit")

    def test_blocked_native_move_consumes_budget_without_advancing_actors(self):
        env = Env()
        env.reset()
        layout = extract(env)
        blocked = next(action for action in names.ACTION_IDS if not layout.open(layout.head, action))
        before_steps = env.steps_left()
        before_key = layout.key()
        env.perform(blocked)
        self.assertEqual(env.steps_left(), before_steps - 1)
        self.assertEqual(extract(env).key(), before_key)

    def test_clone_is_independent(self):
        env = Env()
        env.reset()
        clone = env.clone()
        clone_frame = clone.render()
        env.perform(4)
        self.assertEqual(clone.render(), clone_frame)
        self.assertNotEqual(env.render(), clone.render())
        self.assertEqual(clone.level_index, 0)

    def test_truncation_and_unsupported_states_are_inconclusive(self):
        env = Env()
        env.reset()
        capped = search(env, limit=0)
        self.assertTrue(capped.truncated)
        self.assertFalse(capped.solved)
        unsupported = replace(extract(env), unsupported=("synthetic unknown mechanic",))
        result = search(unsupported)
        self.assertTrue(result.truncated)
        self.assertFalse(result.exact)
        self.assertFalse(result.solved)
        self.assertIsNone(solve(unsupported))
        self.assertTrue(solve.truncated)


class FullGeneration(unittest.TestCase):
    def test_contract_covers_exact_native_curriculum(self):
        self.assertEqual(DIFFICULTIES, tuple(range(1, len(official_levels()) + 1)))
        contract = FULL_STANDARD_CONTRACT
        self.assertEqual(contract["format"], "pebby-full-generator-contract-v1")
        self.assertEqual(SOURCE_ID, shared_multigame.source_for("tu93").source_id)
        self.assertEqual(contract["status"], "ready")
        self.assertTrue(contract["source_id"])
        self.assertTrue(contract["mechanics_inventory_version"])
        self.assertTrue(contract["quality_profile_version"])
        self.assertEqual(
            [(row["difficulty"], row["context_index"]) for row in contract["curriculum"]],
            [(difficulty, difficulty - 1) for difficulty in DIFFICULTIES],
        )
        self.assertTrue(all(0 < row["search_work"] <= 32_000_000
                            for row in contract["curriculum"]))
        self.assertTrue(all(contract["evidence"].values()))
        self.assertTrue(all(contract["caveats"]))

    def test_three_tier_1_seeds_accept_within_sixty_seconds(self):
        started = time.perf_counter()
        specs = [generate(seed, 1) for seed in range(3)]
        self.assertTrue(all(spec is not None for spec in specs))
        self.assertLess(time.perf_counter() - started, 60)
        self.assertEqual(len({spec["gameplay_sha256"] for spec in specs}), 3)
        for spec in specs:
            self.assertEqual(
                validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][0]), []
            )

    def test_every_tier_accepts_and_exercises_its_required_mechanics(self):
        for difficulty in DIFFICULTIES:
            spec = generate(0, difficulty)
            self.assertIsNotNone(spec, (difficulty, generate.last_report))
            self.assertEqual(spec["budget"], REFERENCE_PROFILES[difficulty]["native_budget"])
            for field in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256"):
                self.assertEqual(len(spec[field]), 64)
            self.assertEqual(validate_profile(difficulty, spec["quality"]), [])
            for event, minimum in REFERENCE_PROFILES[difficulty]["required_use"].items():
                self.assertGreaterEqual(spec["quality"]["witness"][event], minimum)
            self.assertEqual(
                validate_full_standard(
                    spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
                ),
                [],
            )

    def test_split_partition_is_deterministic_and_disjoint(self):
        specs = {split: generate(4, 1, split=split) for split in SPLIT_BUCKETS}
        self.assertTrue(all(spec is not None for spec in specs.values()))
        self.assertEqual(specs["train"], generate(4, 1, split="train"))
        identities = {spec["geometry_d4_sha256"] for spec in specs.values()}
        self.assertEqual(len(identities), len(SPLIT_BUCKETS))
        for split, spec in specs.items():
            lower, upper = SPLIT_BUCKETS[split]
            self.assertGreaterEqual(spec["split_partition_bucket"], lower)
            self.assertLess(spec["split_partition_bucket"], upper)

        source = specs["train"]
        rotated = json.loads(json.dumps(source))
        old_cols, old_rows = source["cols"], source["rows"]

        def rotate_cell(cell):
            return [old_rows - 1 - cell[1], cell[0]]

        rotate_direction = {0: 90, 90: 180, 180: 270, 270: 0}
        rotated["cols"], rotated["rows"] = old_rows, old_cols
        rotated["visible_nodes"] = [rotate_cell(cell) for cell in source["visible_nodes"]]
        rotated["edges"] = [
            [rotate_cell(pair[0]), rotate_cell(pair[1])] for pair in source["edges"]
        ]
        rotated["head"] = rotate_cell(source["head"])
        rotated["exit"] = rotate_cell(source["exit"])
        rotated["head_rotation"] = rotate_direction[source["head_rotation"]]
        for field in ("hunters", "patrollers", "tails"):
            rotated[field] = [
                {"cell": rotate_cell(actor["cell"]),
                 "rotation": rotate_direction[actor["rotation"]]}
                for actor in source[field]
            ]
        self.assertEqual(canonical_identities(source), canonical_identities(rotated))
        source_raw, source_d4, source_gameplay = geometry_identities(source)
        rotated_raw, rotated_d4, rotated_gameplay = geometry_identities(rotated)
        self.assertNotEqual(source_raw, rotated_raw)
        self.assertEqual((source_d4, source_gameplay), (rotated_d4, rotated_gameplay))

    def test_json_proof_replay_and_tamper_rejection(self):
        spec = generate(7, 9)
        self.assertIsNotNone(spec, generate.last_report)
        stored = json.loads(json.dumps(spec))
        env = Env([build_level(stored)])
        env.reset()
        self.assertTrue(replay(env, stored["solution"])[0])
        entry = FULL_STANDARD_CONTRACT["curriculum"][8]
        self.assertEqual(validate_full_standard(stored, entry), [])

        tampered = json.loads(json.dumps(stored))
        tampered["solution"][0][0] = next(
            action for action in names.ACTION_IDS if action != tampered["solution"][0][0]
        )
        self.assertTrue(validate_full_standard(tampered, entry))
        forged = json.loads(json.dumps(stored))
        forged["quality"]["witness"]["tail_arms"] += 1
        self.assertIn(
            "stored quality metrics do not match recomputed metrics",
            validate_full_standard(forged, entry),
        )

    def test_full_game_generation_replans_live_at_all_real_indices(self):
        specs = generate_game(123)
        self.assertIsNotNone(specs, generate_game.last_report)
        self.assertEqual(len(specs), len(DIFFICULTIES))
        env = Env(build_game(specs))
        env.reset()
        observation = None
        for index, spec in enumerate(specs):
            self.assertEqual(env.level_index, index)
            first = spec["solution"][0]
            env.perform(*first)
            result = search(env, limit=SEARCH_LIMIT[index + 1])
            self.assertTrue(result.solved, (index + 1, result.reason))
            completed, observation = checked_replay(env, result.actions)
            self.assertTrue(completed, index + 1)
            self.assertEqual(env.levels_completed, index + 1)
            self.assertEqual(spec["proof"]["full_game_replay"]["game_position"], index)
        self.assertEqual(observation.state, GameState.WIN)

        tampered = json.loads(json.dumps(specs))
        tampered[4]["solution"][0][0] = next(
            action
            for action in names.ACTION_IDS
            if action != tampered[4]["solution"][0][0]
        )
        with self.assertRaisesRegex(ValueError, "spec 4 failed full-standard validation"):
            build_game(tampered)

        duplicate = json.loads(json.dumps(specs))
        duplicate[1]["geometry_d4_sha256"] = duplicate[0]["geometry_d4_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate geometry_d4_sha256"):
            build_game(duplicate)

        misordered = json.loads(json.dumps(specs))
        misordered[0], misordered[1] = misordered[1], misordered[0]
        with self.assertRaisesRegex(ValueError, "difficulties must be exactly"):
            build_game(misordered)

        with self.assertRaisesRegex(ValueError, "specs must be mappings"):
            build_game([None] * len(DIFFICULTIES))

    def test_generated_blocked_prefix_can_be_replanned_from_live_state(self):
        spec = generate(8, 1)
        self.assertIsNotNone(spec)
        env = Env([build_level(spec)])
        env.reset()
        layout = extract(env)
        blocked = next(action for action in names.ACTION_IDS if not layout.open(layout.head, action))
        before = env.steps_left()
        env.perform(blocked)
        self.assertEqual(env.steps_left(), before - 1)
        result = search(env, limit=SEARCH_LIMIT[1])
        self.assertTrue(result.solved, result.reason)
        self.assertTrue(checked_replay(env, result.actions)[0])

    def test_reduced_generation_is_allowed_but_not_buildable_as_full_game(self):
        reduced = generate_game(9, difficulties=(1, 3, 7))
        self.assertIsNotNone(reduced)
        self.assertEqual([spec["difficulty"] for spec in reduced], [1, 3, 7])
        with self.assertRaises(ValueError):
            build_game(reduced)
        with self.assertRaises(ValueError):
            generate_game(9, difficulties=(2, 2))
        for invalid in (True, 1.0, "1"):
            with self.assertRaises(ValueError):
                generate(9, invalid)
            with self.assertRaises(ValueError):
                generate_game(9, difficulties=(invalid,))

    def test_bank_io_and_rejection_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs, tried = build_bank(2, seed=30, difficulty=1, split="validation")
            self.assertEqual(len(specs), 2)
            self.assertGreaterEqual(tried, 2)
            self.assertTrue(all(spec["generation_diagnostics"]["accepted"] for spec in specs))
            path = save_bank(specs, Path(tmp) / "tu93.jsonl")
            self.assertEqual(load_bank(path), specs)
            cli_path = Path(tmp) / "cli.jsonl"
            self.assertEqual(
                bank_main([
                    "--levels", "1", "--seed", "40", "--difficulty", "9",
                    "--split", "test", "--out", str(cli_path),
                ]),
                0,
            )
            self.assertEqual(len(load_bank(cli_path)), 1)

    def test_malformed_full_standard_spec_returns_errors(self):
        entry = FULL_STANDARD_CONTRACT["curriculum"][0]
        self.assertTrue(validate_full_standard({}, entry))
        spec = generate(10, 1)
        malformed = dict(spec)
        malformed["solution"] = [[99, None, None]]
        self.assertTrue(validate_full_standard(malformed, entry))


if __name__ == "__main__":
    unittest.main()
