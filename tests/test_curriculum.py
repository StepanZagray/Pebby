"""Generated-only curriculum checks; never load official gameplay levels."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pebby.ls20 import names, rails
from pebby.ls20.bank import load, save
from pebby.ls20.curriculum import DIFFICULTIES, _verify, generate_level
from pebby.ls20.env import Ls20Env, Ls20Scenario
from pebby.ls20.generate import _level_data, build_level, generate_level as original_generate
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {d: generate_level(d, d) for d in DIFFICULTIES}

    def test_every_stage_completes_full_search_and_real_engine_replay(self):
        for difficulty, spec in self.specs.items():
            with self.subTest(difficulty=difficulty):
                env = Ls20Scenario(build_level(spec), spec["seed"] % 7)
                oracle = Oracle(extract(env), limit=spec["search_limit"])
                self.assertFalse(oracle.truncated)
                self.assertTrue(oracle.solvable)
                self.assertEqual(oracle.solution(seed=spec["seed"]), spec["solution"])
                self.assertEqual(oracle.optimal_actions, len(spec["solution"]))
                for action in spec["solution"]:
                    observation = env.perform(action)
                self.assertTrue(observation.won)
                self.assertEqual(env.levels_completed, 1)
                self.assertEqual(env.lives(), 3)
                self.assertGreaterEqual(spec["slack_moves"], 3)
                self.assertFalse(spec["search_truncated"])

    def test_deterministic_and_bank_compatible(self):
        self.assertEqual(self.specs[5], generate_level(5, 5))
        with tempfile.TemporaryDirectory() as directory:
            path = save(list(self.specs.values()), Path(directory) / "bank.jsonl")
            restored = load(path)
            self.assertEqual(len(restored), 5)
            for spec in restored:
                self.assertEqual(spec["curriculum_version"], 2)
                self.assertEqual(spec["generator_version"], 3)
                self.assertEqual(Ls20Env([build_level(spec)]).goal_triples(),
                                 [tuple(goal["triple"]) for goal in spec["goals"]])

    def test_independent_goal_triples_reach_engine_in_order(self):
        spec = self.specs[5]
        self.assertEqual(len(spec["goals"]), 2)
        self.assertNotEqual(spec["goals"][0]["triple"], spec["goals"][1]["triple"])
        changed = copy.deepcopy(spec)
        changed["goals"][0]["triple"] = [1, 2, 3]
        changed["goals"][1]["triple"] = [4, 0, 1]
        self.assertEqual(Ls20Env([build_level(changed)]).goal_triples(),
                         [(1, 2, 3), (4, 0, 1)])

    def test_small_bank_covers_topology_budgets_and_used_mechanics(self):
        specs = [generate_level(seed, seed % 5 + 1) for seed in range(15)]
        self.assertEqual({s["topology"] for s in specs}, {"room", "obstacles", "door"})
        self.assertEqual({s["step_cost"] for s in specs}, {1, 2})
        self.assertEqual({s["rail_mode"] for s in specs}, {"none", "short", "ring"})
        for mechanic in ("moving_cycler", "launcher", "refill"):
            self.assertTrue(any(s["solution_mechanics"][mechanic] for s in specs))
        three_attributes = generate_level(1, 5)
        self.assertEqual({c["kind"] for c in three_attributes["cyclers"]},
                         {"shape", "color", "rotation"})
        self.assertEqual(len(three_attributes["goals"]), 2)

    def test_rails_move_in_real_engine_and_remain_clear_for_entire_schedule(self):
        for difficulty, expected_mode in ((3, "short"), (4, "ring")):
            spec = self.specs[difficulty]
            env = Ls20Scenario(build_level(spec), spec["seed"] % 7)
            layout = extract(env)
            self.assertEqual(spec["rail_mode"], expected_mode)
            self.assertTrue(spec["solution_mechanics"]["moving_cycler"])
            self.assertEqual(len(layout.patrollers), 1)
            states = {tuple(rails.live_states(env.game))}
            special = set(layout.cyclers) | layout.refills | set(layout.goal_at) | layout.walls
            for pad in layout.launchers:
                special.update(pad["triggers"])
            for tick in layout.moving_cyclers:
                self.assertFalse(set(tick) & special)
            for action in spec["solution"][:-1]:
                env.perform(action)
                states.add(tuple(rails.live_states(env.game)))
            self.assertGreater(len(states), 1)
            self.assertGreater(layout.tick_period, 1)
            if expected_mode == "ring":
                self.assertEqual(len(set(layout.patrollers[0]["cells"])), 8)

    def test_fog_changes_actual_frames_and_preserves_engine_win(self):
        spec = self.specs[4]
        fog = Ls20Scenario(build_level(spec), spec["seed"] % 7)
        clear = Ls20Scenario(build_level({**spec, "fog": False}), spec["seed"] % 7)
        self.assertTrue(fog.fog())
        self.assertFalse(np.array_equal(fog.render(), clear.render()))
        for action in spec["solution"]:
            observation = fog.perform(action)
            clear.perform(action)
        self.assertTrue(observation.won)

    def test_truncated_even_solvable_search_is_rejected_before_labeling(self):
        with patch("pebby.ls20.curriculum.Oracle") as oracle:
            oracle.return_value.truncated = True
            oracle.return_value.solvable = True
            self.assertIsNone(_verify(self.specs[1], 1, 0))
            oracle.return_value.solution.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "within 2 attempts and 1 states"):
            generate_level(1, 1, attempts=2, search_limit=1)

    def test_original_generated_specs_keep_scalar_and_shared_goal_behavior(self):
        spec = original_generate(1, 1)
        original = Ls20Scenario(build_level(spec), spec["seed"] % 7)
        self.assertFalse(extract(original).patrollers)
        self.assertEqual(original.goal_triples(), [tuple(spec["goals"][0]["triple"])])
        self.assertIsInstance(_level_data(spec)[names.KEY_GOAL_SHAPE], int)
        duplicate = copy.deepcopy(spec)
        duplicate["goals"].append(copy.deepcopy(duplicate["goals"][0]))
        data = _level_data(duplicate)
        for key in (names.KEY_GOAL_SHAPE, names.KEY_GOAL_COLOR, names.KEY_GOAL_ROTATION):
            self.assertEqual(data[key][0], data[key][1])
        for action in spec["solution"]:
            observation = original.perform(action)
        self.assertTrue(observation.won)

    def test_verification_context_matches_training_budget_rules(self):
        spec = self.specs[1]
        self.assertFalse(spec["verification_match_hint"])
        self.assertEqual(spec["verification_level_index"], spec["seed"] % 7)
        self.assertEqual(spec["context_solution"], spec["solution"])
        self.assertEqual(spec["context_optimal_actions"], spec["optimal_actions"])
        env = Ls20Env([build_level(spec), build_level(spec)])
        self.assertTrue(extract(env).match_hint)
        env.set_level(1)
        self.assertFalse(extract(env).match_hint)

    def test_limits_are_validated(self):
        for kwargs in ({"difficulty": 0}, {"attempts": 0}, {"search_limit": 0}, {"min_slack": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_level(0, **kwargs)


if __name__ == "__main__":
    unittest.main()
