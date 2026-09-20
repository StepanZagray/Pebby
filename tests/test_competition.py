"""Real engine qualification using only small generated level specifications."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arcengine import GameState

from pebby.agent.competition import CompetitionSession, FourMovementDecision, run_competition
from pebby.ls20.generate import build_level


def generated_level():
    # A new two-cell corridor: RIGHT wins; DOWN hits a wall and spends a move.
    return build_level(dict(walls=[(1, 3), (0, 2), (1, 1)], start=(1, 2),
                            start_triple=(0, 0, 0), goals=[dict(cell=(2, 2), triple=(0, 0, 0))],
                            cyclers=[], refills=[], launchers=[], step_counter=2, step_cost=1, fog=False))


class ScriptedDecision:
    metadata = {"reset_controller": "synthetic_test_script", "voluntary_reset_learned": False}

    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, context):
        return next(self.actions)


class CompetitionEngineTests(unittest.TestCase):
    def session(self, count=2):
        return CompetitionSession([generated_level() for _ in range(count)])

    def test_second_level_and_consecutive_resets_preserve_real_engine_progress(self):
        before_environment = dict(os.environ)
        session = self.session()
        session.step(4)
        self.assertEqual((session.level_index, session.levels_completed), (1, 1))
        for _ in range(3):
            session.step(0)
            self.assertEqual((session.level_index, session.levels_completed, session.lives), (1, 1, 3))
        self.assertEqual((session.actions, session.resets, session.per_level_actions), (4, 3, [1, 3]))
        self.assertEqual(dict(os.environ), before_environment)
        with self.assertRaisesRegex(RuntimeError, "cannot restart"):
            session._env.reset()

    def test_native_three_lives_game_over_and_reset_charge(self):
        session = self.session()
        self.assertEqual(session.lives, 3)
        seen = []
        for _ in range(30):
            _, event = session.step(2)
            if event["lives_after"] < event["lives_before"]:
                seen.append(event["lives_after"])
            if session.state == GameState.GAME_OVER:
                break
        self.assertEqual(seen, [2, 1, 0])
        self.assertEqual(session.actions, 9)
        with self.assertRaisesRegex(ValueError, "only RESET"):
            session.step(4)
        self.assertEqual(session.actions, 9)
        session.step(0)
        self.assertEqual((session.actions, session.resets, session.lives), (10, 1, 3))
        self.assertEqual(session.state, GameState.NOT_FINISHED)

    def test_reset_has_zero_tank_cost_but_one_scorecard_action(self):
        session = self.session()
        session.step(2)
        self.assertEqual(session._env.steps_left(), 1)
        self.assertEqual(session._env.game._action_count, 1)
        session.step(0)
        self.assertEqual(session._env.steps_left(), 2)
        self.assertEqual(session._env.game._action_count, 0)
        self.assertEqual((session.actions, session.resets), (2, 1))

    def test_seven_levels_win_in_one_continuous_session_and_ledger(self):
        session = self.session(7)
        report = run_competition(ScriptedDecision([4, 0, 0] + [4] * 6), session,
                                 per_level_caps=[20] * 7)
        self.assertTrue(report["completed"])
        self.assertEqual((report["levels_completed"], report["actions"], report["resets"]), (7, 9, 2))
        self.assertEqual(report["per_level_actions"], [1, 3, 1, 1, 1, 1, 1])
        self.assertIsNone(report["reset_count_cap"])
        self.assertEqual([row["charged_actions"] for row in report["ledger"]], list(range(1, 10)))
        for before, after in zip(report["ledger"], report["ledger"][1:]):
            self.assertEqual(before["frame_after_sha256"], after["frame_before_sha256"])
        with self.assertRaises(ValueError):
            session.step(0)
        with self.assertRaisesRegex(ValueError, "fresh"):
            run_competition(ScriptedDecision([]), session, per_level_caps=[20] * 7)

    def test_per_level_cap_includes_resets_and_stops_without_restart(self):
        session = self.session()
        report = run_competition(ScriptedDecision([4, 0, 0]), session, per_level_caps=[1, 2])
        self.assertEqual((report["levels_completed"], report["actions"], report["resets"]), (1, 3, 2))
        self.assertFalse(report["completed"])
        self.assertEqual(report["ending"], "per_level_action_cap")
        self.assertEqual(report["full_game_initializations"], 1)
        self.assertEqual(report["full_game_resets_after_initialization"], 0)
        self.assertEqual((session.level_index, session.levels_completed), (1, 1))

    def test_cap_at_game_over_does_not_buy_a_fresh_attempt(self):
        session = self.session()
        report = run_competition(ScriptedDecision([2] * 9), session, per_level_caps=[9, 9])
        self.assertEqual(report["final_state"], GameState.GAME_OVER.value)
        self.assertEqual((report["actions"], report["resets"]), (9, 0))

    def test_invalid_actions_and_caps_are_not_silently_coerced(self):
        session = self.session()
        for action in (True, 1.5, -1, 5, "0"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                session.step(action)
        self.assertEqual(session.actions, 0)
        for caps in ([1], [0, 1], [True, 1]):
            with self.subTest(caps=caps), self.assertRaises(ValueError):
                run_competition(ScriptedDecision([]), session, per_level_caps=caps)


class FourMovementAdapterTests(unittest.TestCase):
    def test_strict_argmax_and_game_over_only_reset_are_distinguished(self):
        import torch

        class Policy:
            def __call__(self, frames):
                return torch.tensor([[0., 2., 1., -1.]])

        session = CompetitionSession([generated_level()])
        adapter = FourMovementDecision(Policy())
        report = run_competition(adapter, session, per_level_caps=[11])
        self.assertEqual([row["action"] for row in report["ledger"]], [2] * 9 + [0, 2])
        self.assertEqual(report["decision"]["reset_controller"], "game_over_only")
        self.assertFalse(report["decision"]["voluntary_reset_learned"])
        self.assertFalse(report["decision"]["persistent_game_memory"])
        self.assertEqual(report["decision"]["stall_controller"], "none")

    def test_existing_history_boundary_behavior_is_kept(self):
        from unittest.mock import Mock
        history = Mock()
        adapter = FourMovementDecision(object())
        adapter.history = history
        observation = Mock(frame=[[0]])
        base = dict(action=2, reset=False, lives_before=3, lives_after=3, level_before=1, level_after=1)
        adapter.observe(observation, base)
        history.observe.assert_called_with([[0]], 1, reset=False)
        for update in (dict(lives_after=2), dict(level_after=2), dict(action=0, reset=True)):
            event = {**base, **update}
            adapter.observe(observation, event)
            history.observe.assert_called_with([[0]], event["action"] - 1, reset=True)

    def test_checkpoint_hash_mismatch_stops_before_model_or_game_creation(self):
        from tools.evaluate_reference_competition import main
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"not a checkpoint")
            with patch("pebby.agent.competition.CompetitionSession") as constructor:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    main(["--checkpoint", str(checkpoint), "--checkpoint-sha256", "0" * 64,
                          "--report-out", str(Path(directory) / "report.json"),
                          "--per-level-max-actions", "10"])
                constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
