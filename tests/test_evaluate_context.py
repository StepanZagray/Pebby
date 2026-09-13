import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from pebby.agent import evaluate


class ContextualBankTests(unittest.TestCase):
    def spec(self, **extra):
        value = {"seed": 1000000, "difficulty": 1, "optimal_actions": 11,
                 "generator_version": 2}
        value.update(extra)
        return value

    def test_contextual_optimum_and_context_are_used(self):
        spec = self.spec(training_context_index=4, context_optimal_actions=17)
        with patch("pebby.ls20.bank.load", return_value=[spec]), \
                patch("pebby.ls20.generate.build_level", side_effect=lambda value: value) as build:
            levels, optima, specs = evaluate.bank_levels(Path("bank.json"))
        self.assertEqual(optima, [17])
        self.assertEqual(specs, [spec])
        self.assertEqual(levels, [spec])
        build.assert_called_once_with(spec)

    def test_cli_default_caps_follow_the_selected_protocol(self):
        from pebby.ls20 import shipped
        self.assertEqual(evaluate.default_max_actions('generated'), 300)
        self.assertEqual(evaluate.default_max_actions('shipped'),
                         5 * sum(shipped.HUMAN_BASELINE))
        self.assertEqual(evaluate.default_max_actions('shipped_level', 2),
                         5 * shipped.HUMAN_BASELINE[1])
        with self.assertRaises(ValueError):
            evaluate.default_max_actions('shipped_level', 0)

    def test_parameter_count_falls_back_to_loaded_model_when_metadata_is_absent(self):
        self.assertEqual(evaluate.parameter_count(torch.nn.Linear(2, 3)), 9)

    def test_cli_rejects_ignored_depth_override_on_spatial_wrapper(self):
        from types import SimpleNamespace
        policy = SimpleNamespace(config=lambda: {"architecture": "world", "loops": 6})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            checkpoint.touch()
            with patch.object(evaluate, "load_checkpoint", return_value=(policy, {})), \
                    patch.object(evaluate, "completion_rate") as play, \
                    patch.object(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint),
                                               "--device", "cpu", "--loops", "2"]), \
                    patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    evaluate.main()
        self.assertEqual(error.exception.code, 2)
        play.assert_not_called()
        self.assertFalse(hasattr(policy, "loops"))

    def test_ordinary_bank_keeps_context_zero_and_original_optimum(self):
        spec = self.spec()
        with patch("pebby.ls20.bank.load", return_value=[spec]), \
                patch("pebby.ls20.generate.build_level", side_effect=lambda value: value):
            _, optima, specs = evaluate.bank_levels(Path("bank.json"))
        self.assertEqual(optima, [11])
        self.assertEqual([item.get("training_context_index", 0) for item in specs], [0])

    def test_context_tag_without_contextual_optimum_is_rejected(self):
        spec = self.spec(training_context_index=2)
        with patch("pebby.ls20.bank.load", return_value=[spec]), \
                patch("pebby.ls20.generate.build_level", side_effect=lambda value: value):
            with self.assertRaisesRegex(ValueError, "context_optimal_actions"):
                evaluate.bank_levels(Path("bank.json"))

    def test_context_metadata_must_be_an_integer_in_the_game_context_range(self):
        for value in ("2", 7, -1):
            spec = self.spec(training_context_index=value, context_optimal_actions=17)
            with self.subTest(value=value), patch("pebby.ls20.bank.load", return_value=[spec]), \
                    patch("pebby.ls20.generate.build_level", side_effect=lambda item: item), \
                    self.assertRaisesRegex(ValueError, "0..6"):
                evaluate.bank_levels(Path("bank.json"))

    def test_rows_with_both_splits_are_passed_aligned_contexts_by_cli(self):
        spec = self.spec(training_context_index=5, context_optimal_actions=23)
        strict = {"format": evaluate.REPORT_FORMAT, "levels": 1, "runs_played": 1,
                  "max_actions": 200, "completed": 0, "completion_rate": 0.,
                  "goals_cleared": 0, "goals_total": 1, "goal_rate": 0.,
                  "mean_actions": 1., "mean_actions_vs_optimal": 1.,
                  "runs_with_oracle": 1, "won": 0, "game_over": 0, "capped": 1,
                  "stuck": 0, "stalled_actions": 0, "stall_dominated": 0}
        budgeted = {"completed": 0, "levels": 1, "completion_rate": 0.,
                    "goals_cleared": 0, "goals_total": 1, "mean_actions": 1.,
                    "mean_attempts": 1., "multiplier": 5., "temperature": .5}
        class Policy:
            def config(self):
                return {"architecture": "cnn"}

        policy = Policy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            checkpoint.touch()
            bank = Path(directory) / "bank.jsonl"
            bank.write_text(json.dumps(spec) + "\n")
            with patch.object(evaluate, "load_checkpoint", return_value=(policy, {"parameters": 1})), \
                    patch("pebby.ls20.bank.load", return_value=[spec]), \
                    patch("pebby.ls20.generate.build_level", side_effect=lambda value: value), \
                    patch.object(evaluate, "completion_rate", return_value=strict) as strict_call, \
                    patch.object(evaluate, "budgeted_completion", return_value=budgeted) as budget_call, \
                    patch.object(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint),
                                                "--bank", str(bank), "--device", "cpu",
                                                "--max-actions", "17"]):
                evaluate.main()
        self.assertEqual(strict_call.call_args.args[2], 17)
        self.assertEqual(strict_call.call_args.args[4], [23])
        self.assertEqual(strict_call.call_args.args[6], [5])
        self.assertEqual(budget_call.call_args.args[2], [23])
        self.assertEqual(budget_call.call_args.args[8], [5])

    def test_optimality_rate_builds_the_scenario_in_the_supplied_context(self):
        seen_contexts = []

        class Observation:
            frame = [[0] * 64 for _ in range(64)]
            finished = True
            won = True

        class Scenario:
            level_count = 1
            levels_completed = 1

            def reset(self):
                return Observation.frame

            def lives(self):
                return 3

            def perform(self, action):
                return Observation()

        class Oracle:
            truncated = False

            def state_of(self, env):
                return 0

            def distance_for(self, state):
                return 1

        class Policy:
            def __call__(self, frames):
                return torch.tensor([[1., 0., 0., 0.]])

        def scenario(level, context):
            seen_contexts.append(context)
            return Scenario()

        spec = self.spec(training_context_index=6, context_optimal_actions=17)
        with patch("pebby.ls20.generate.build_level", side_effect=lambda value: value), \
                patch("pebby.ls20.plan.oracle_for", return_value=Oracle()), \
                patch.object(evaluate, "Ls20Scenario", side_effect=scenario):
            report = evaluate.optimality_rate([spec], Policy(), max_actions=2,
                                              context_indices=[6])
        self.assertEqual(seen_contexts, [6])
        self.assertEqual(report["optimal_moves"], 1)
        self.assertEqual(report["measured_moves"], 1)

    def test_optimality_rate_defaults_to_each_spec_training_context(self):
        seen_contexts = []

        class Observation:
            frame = [[0] * 64 for _ in range(64)]
            finished = True
            won = True

        class Scenario:
            levels_completed = 1

            def reset(self):
                return Observation.frame

            def lives(self):
                return 3

            def perform(self, action):
                return Observation()

        class Oracle:
            truncated = False

            def state_of(self, env):
                return 0

            def distance_for(self, state):
                return 1

        class Policy:
            def __call__(self, frames):
                return torch.tensor([[1., 0., 0., 0.]])

        spec = self.spec(training_context_index=6, context_optimal_actions=17)
        with patch("pebby.ls20.generate.build_level", side_effect=lambda value: value), \
                patch("pebby.ls20.plan.oracle_for", return_value=Oracle()), \
                patch.object(evaluate, "Ls20Scenario",
                              side_effect=lambda level, context: (seen_contexts.append(context)
                                                                  or Scenario())):
            evaluate.optimality_rate([spec], Policy(), max_actions=2)
        self.assertEqual(seen_contexts, [6])


if __name__ == "__main__":
    unittest.main()
