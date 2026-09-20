"""`DiverseLivesDecision` in a competition session and `Engine.act` on finished games.

Only generated fixtures and the shipped level list run here; no checkpoint is
loaded. Every controller behaviour asserted is a runtime heuristic, not learning.
"""

import json
import unittest
from pathlib import Path

import torch
from arcengine import GameState

from inference import Engine
from pebby.agent.competition import (CompetitionSession, DiverseLivesDecision, FourMovementDecision,
                                     run_competition)
from pebby.ls20 import names
from pebby.ls20.generate import build_level

ROOT = Path(__file__).resolve().parents[1]


def generated_level():
    # A two-cell corridor: RIGHT wins; DOWN hits a wall and spends a move.
    return build_level(dict(walls=[(1, 3), (0, 2), (1, 1)], start=(1, 2),
                            start_triple=(0, 0, 0), goals=[dict(cell=(2, 2), triple=(0, 0, 0))],
                            cyclers=[], refills=[], launchers=[], step_counter=2, step_cost=1, fog=False))


class FixedPolicy:
    def __init__(self, logits):
        self.logits = list(logits)

    def __call__(self, frames):
        return torch.tensor([self.logits], dtype=torch.float32)


class DiverseLivesDecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_reset_is_returned_only_on_game_over(self):
        # DOWN by a margin no temperature-0.5 sample overturns: nine deaths, RESET, one more.
        session = CompetitionSession([generated_level()])
        decision = DiverseLivesDecision(FixedPolicy([0., 60., 0., 0.]))
        report = run_competition(decision, session, per_level_caps=[11])
        self.assertEqual([row["action"] for row in report["ledger"]], [2] * 9 + [0, 2])
        for row in report["ledger"]:
            self.assertEqual(row["action"] == 0, row["state_before"] == GameState.GAME_OVER.value)
        self.assertEqual(report["decision"]["action_selection"], "diverse_lives")
        self.assertEqual(report["decision"]["reset_controller"], "game_over_only")
        self.assertFalse(report["decision"]["voluntary_reset_learned"])
        self.assertFalse(report["decision"]["learned"])
        self.assertTrue(report["decision"]["runtime_heuristics"])
        self.assertEqual(report["decision"]["stall_controller"], "unchanged_frame_mask")
        self.assertEqual(report["decision"]["controller"]["controller"], "diverse_lives")
        # Three life losses and one RESET are four boundaries on the same level, so the
        # trajectory after the RESET samples from diversity index 4 (index 3 belonged to
        # the GAME_OVER state, where no decision is ever taken).
        self.assertEqual(decision.controller.diversity_index, 4)
        self.assertEqual(decision.controller.level_index, 0)
        json.dumps(report, allow_nan=False)

    def test_decide_never_returns_reset_before_game_over(self):
        session = CompetitionSession([generated_level()])
        decision = DiverseLivesDecision(FixedPolicy([0., 2., 1., -1.]))
        decision.start(session.context())
        while session.state == GameState.NOT_FINISHED and session.actions < 40:
            action = decision.decide(session.context())
            self.assertIn(action, names.ACTION_IDS)
            observation, event = session.step(action)
            decision.observe(observation, event)
        if session.state == GameState.GAME_OVER:
            self.assertEqual(decision.decide(session.context()), 0)

    def test_completes_a_generated_level_session_with_a_stub_policy(self):
        session = CompetitionSession([generated_level(), generated_level()])
        report = run_competition(DiverseLivesDecision(FixedPolicy([0., 0., 0., 5.])), session,
                                 per_level_caps=[10, 10])
        self.assertTrue(report["completed"])
        self.assertEqual((report["levels_completed"], report["actions"], report["resets"]), (2, 2, 0))
        self.assertEqual([row["action"] for row in report["ledger"]], [4, 4])
        self.assertEqual(report["final_state"], GameState.WIN.value)

    def test_later_lives_diverge_from_the_first_and_runs_are_reproducible(self):
        # Strict argmax replays three identical DOWN deaths; the diverse controller
        # samples lives two and three, so the same session no longer repeats itself.
        logits = [0., 2., 1., -1.]
        strict = run_competition(FourMovementDecision(FixedPolicy(logits)),
                                 CompetitionSession([generated_level()]), per_level_caps=[11])
        self.assertEqual([row["action"] for row in strict["ledger"]], [2] * 9 + [0, 2])
        runs = [run_competition(DiverseLivesDecision(FixedPolicy(logits), base_seed=0),
                                CompetitionSession([generated_level()]), per_level_caps=[11])
                for _ in range(2)]
        self.assertEqual([row["action"] for row in runs[0]["ledger"]],
                         [row["action"] for row in runs[1]["ledger"]])
        first_life = [row["action"] for row in runs[0]["ledger"]][:3]
        self.assertEqual(first_life, [2, 2, 2])
        later = [row for row in runs[0]["ledger"][3:] if row["action"] != 0]
        self.assertTrue(later)
        self.assertNotEqual([row["action"] for row in later], [2] * len(later))

    def test_four_movement_decision_metadata_is_unchanged(self):
        self.assertEqual(FourMovementDecision.metadata, dict(
            reset_controller="game_over_only", voluntary_reset_learned=False,
            policy_action_ids=[1, 2, 3, 4], decision_action_ids=[0, 1, 2, 3, 4],
            action_selection="strict_argmax", stall_controller="none",
            history_boundary_semantics="clear_on_life_loss_level_transition_and_reset",
            persistent_game_memory=False))


class EngineControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.generated = Engine(None).dispatch(json.loads((ROOT / "examples/generate.json").read_text()))["level"]

    @staticmethod
    def engine(logits, controller="diverse"):
        engine = Engine(None, controller=controller)
        engine.agent._model = FixedPolicy(logits)
        engine.agent.loaded, engine.agent._attempted, engine.agent.reason = True, True, None
        return engine

    def test_controller_flag_is_validated_and_reported(self):
        with self.assertRaisesRegex(ValueError, "controller"):
            Engine(None, controller="random")
        self.assertEqual(Engine(None).info()["controller"], "diverse")
        self.assertEqual(Engine(None, controller="argmax").info()["controller"], "argmax")
        self.assertEqual(Engine.CONTROLLERS, ("diverse", "argmax"))

    def test_act_returns_reset_on_game_over_before_the_policy(self):
        class Exploding:
            def __call__(self, frames):
                raise AssertionError("the policy must not be consulted on a finished game")

        for controller in Engine.CONTROLLERS:
            with self.subTest(controller=controller):
                engine = self.engine([0.] * 4, controller)
                engine.agent._model = Exploding()
                result = engine.dispatch({"op": "agent", "level": {"shipped": 0},
                                          "actions": [1, 2] * 1024})
                self.assertEqual(result["action"], 0)
                self.assertTrue(result["finished"])
                self.assertTrue(result["loaded"])
                self.assertIsNone(result["probabilities"])
                self.assertIn("RESET", result["reason"])
                self.assertEqual(result["status"]["state"], "GAME_OVER")
                self.assertFalse(result["status"]["won"])

    def test_act_reports_finished_with_no_action_on_win(self):
        for controller in Engine.CONTROLLERS:
            with self.subTest(controller=controller):
                engine = self.engine([0.] * 4, controller)
                result = engine.dispatch({"op": "agent", "level": self.generated,
                                          "actions": self.generated["solution"]})
                self.assertIsNone(result["action"])
                self.assertTrue(result["finished"])
                self.assertTrue(result["status"]["won"])
                self.assertIsNone(result["probabilities"])

    def test_open_game_answers_carry_the_selection_record(self):
        engine = self.engine([0., 3., 2., 1.])
        result = engine.dispatch({"op": "agent", "level": {"shipped": 0}, "actions": []})
        self.assertEqual(result["action"], 2)
        self.assertFalse(result["finished"])
        self.assertEqual(result["selection"]["controller"], "diverse_lives")
        self.assertEqual(result["selection"]["diversity_index"], 0)
        self.assertFalse(result["selection"]["sampled"])
        self.assertAlmostEqual(sum(result["probabilities"]), 1., places=6)
        self.assertEqual(engine.dispatch({"op": "agent", "level": {"shipped": 0}, "actions": []}),
                         result)  # Stateless: the same request replays to the same answer.
        argmax = self.engine([0., 3., 2., 1.], "argmax").dispatch(
            {"op": "agent", "level": {"shipped": 0}, "actions": []})
        self.assertEqual(argmax["action"], 2)
        self.assertEqual(argmax["selection"], {"controller": "argmax"})

    def test_replayed_moves_feed_the_controller_penalties_on_later_lives_only(self):
        # A replayed LEFT that moved the player makes RIGHT (its undo) pay the
        # reversal penalty on the next decision -- but only once a life has been
        # lost: the first life is strict argmax under the stall mask alone.
        engine = self.engine([0., 0., 0., 0.])
        first = engine.dispatch({"op": "agent", "level": {"shipped": 0}, "actions": [3]})
        self.assertEqual(first["selection"]["penalties"], [0., 0., 0., 0.])
        self.assertEqual(first["selection"]["diversity_index"], 0)
        self.assertFalse(first["selection"]["sampled"])
        later = engine.dispatch({"op": "agent", "level": {"shipped": 0},
                                 "actions": [1, 2] * 40 + [3]})
        self.assertEqual(later["status"]["lives"], 2)
        self.assertEqual(later["selection"]["diversity_index"], 1)
        self.assertEqual(later["selection"]["penalties"], [0., 0., 0., .5])
        self.assertTrue(later["selection"]["sampled"])

    def test_replayed_life_loss_switches_to_seeded_sampling(self):
        # Eighty alternating moves burn one life on shipped level one.
        engine = self.engine([0., 0., 0., 0.])
        request = {"op": "agent", "level": {"shipped": 0}, "actions": [1, 2] * 40}
        result = engine.dispatch(request)
        self.assertEqual(result["status"]["lives"], 2)
        self.assertFalse(result["finished"])
        self.assertTrue(result["selection"]["sampled"])
        self.assertEqual(result["selection"]["diversity_index"], 1)
        self.assertEqual(engine.dispatch(request), result)  # Stateless across requests.
        # Under argmax the same replay gets no selection record and no sampling.
        argmax = self.engine([0., 0., 0., 0.], "argmax").dispatch(request)
        self.assertEqual(argmax["selection"], {"controller": "argmax"})
        self.assertEqual(argmax["status"]["lives"], 2)


if __name__ == "__main__":
    unittest.main()
