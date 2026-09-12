"""Focused tests for the generated life-loss diagnostic collector."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import collect_structured_event_transitions as events


class _FakeResult:
    def __init__(self, frame, *, finished=False, won=False):
        self.frame = frame
        self.finished = finished
        self.won = won


class _FakeEnv:
    """Small cloneable engine-shaped fixture for unreachable/terminal branches."""

    def __init__(self):
        self.module = object()
        self._lives = 3
        self._state = SimpleNamespace(value="NOT_FINISHED")
        self._action = 0

    @property
    def state(self):
        return self._state

    def perform(self, action):
        self._action = int(action)
        if action == 4:
            self._lives = 0
            self._state = SimpleNamespace(value="GAME_OVER")
            return _FakeResult(np.full((64, 64), action, dtype=np.uint8), finished=True)
        return _FakeResult(np.full((64, 64), action, dtype=np.uint8))

    def player_cell(self):
        return (self._action, 0)

    def triple(self):
        return (0, 0, 0)

    def steps_left(self):
        return 10

    def lives(self):
        return self._lives

    def goals_solved(self):
        return []


class _UnreachableOracle:
    truncated = False
    start = (0,)
    layout = SimpleNamespace()
    refills = ()

    def state_of(self, env):
        return None

    def distance_for(self, state):
        return None


class _ReachableOracle:
    truncated = False
    start = ("state",)
    layout = SimpleNamespace()
    refills = ()

    def state_of(self, env):
        return ("state",)

    def distance_for(self, state):
        return 1


class _SafeFakeEnv(_FakeEnv):
    def perform(self, action):
        self._action = int(action)
        return _FakeResult(np.full((64, 64), action, dtype=np.uint8))


class _WinningFakeEnv(_FakeEnv):
    def perform(self, action):
        self._action = int(action)
        if action == 4:
            self._state = SimpleNamespace(value="WIN")
            return _FakeResult(np.full((64, 64), action, dtype=np.uint8),
                               finished=True, won=True)
        return _FakeResult(np.full((64, 64), action, dtype=np.uint8))


class StructuredEventTransitionTests(unittest.TestCase):
    def test_doomed_state_is_transition_only_and_terminal_branch_is_checked(self):
        env = _FakeEnv()
        env._lives = 1
        oracle = _UnreachableOracle()
        # A doomed state has no policy target, but its concrete logical tuple
        # still receives a mechanics comparison for every branch.
        def predicted(_layout, _state, action, _refills):
            return (None, "died" if action == 3 else "moved")

        with patch.object(events, "simulate", side_effect=predicted):
            captured = events.capture_branches(env, oracle, seed=123, step=7)
        self.assertFalse(bool(captured["current_reachable"]))
        self.assertEqual(int(captured["optimal"]), 0)
        self.assertEqual(int(captured["branch_checks"]), 4)
        self.assertEqual(int(captured["oracle_unverified_branches"]), 0)
        self.assertEqual(captured["terminal"].tolist(), [False, False, False, True])
        self.assertEqual(captured["lost_life"].tolist(), [False, False, False, True])
        self.assertEqual(captured["next_lives"].tolist(), [1, 1, 1, 0])
        self.assertEqual(int(captured["branch_events"][3]), events.EVENT_TERMINAL_LOSS)
        self.assertTrue(np.all(captured["next_optimal"] == 0))

    def test_winning_successor_has_logical_distance_zero(self):
        result = _FakeResult(np.zeros((64, 64), dtype=np.uint8), finished=True, won=True)
        distance, reachable, mask, _ = events._oracle_next(_UnreachableOracle(), _FakeEnv(), result)
        self.assertEqual(distance, 0)
        self.assertFalse(reachable)
        self.assertEqual(mask, 0)

    def test_winning_branch_remains_an_optimal_action_at_distance_one(self):
        def predicted(_layout, _state, action, _refills):
            return (("state",), "won" if action == 3 else "moved")

        with patch.object(events, "simulate", side_effect=predicted), \
                patch.object(events.world_data, "successor_optimal_mask", return_value=1):
            captured = events.capture_branches(_WinningFakeEnv(), _ReachableOracle(), seed=123, step=2)
        self.assertTrue(bool(captured["current_reachable"]))
        self.assertEqual(int(captured["current_distance"]), 1)
        self.assertEqual(int(captured["optimal"]), 0b1000)
        self.assertEqual(int(captured["next_distance"][3]), 0)

    def test_reachable_state_without_an_optimal_branch_fails_closed(self):
        with patch.object(events, "simulate", return_value=(("state",), "moved")), \
                patch.object(events.world_data, "successor_optimal_mask", return_value=1):
            with self.assertRaisesRegex(events.TransitionMismatch, "no safe optimal action"):
                events.capture_branches(_SafeFakeEnv(), _ReachableOracle(), seed=123, step=2)

    def test_event_codes_keep_reset_and_terminal_losses_distinct(self):
        self.assertEqual(events._event_code(False, False, False), events.EVENT_LIVE)
        self.assertEqual(events._event_code(True, False, False), events.EVENT_RESET_LOSS)
        self.assertEqual(events._event_code(True, True, False), events.EVENT_TERMINAL_LOSS)
        self.assertEqual(events._event_code(False, True, True), events.EVENT_WIN)

    def test_source_proof_rejects_truncated_rows(self):
        source = Path("data/ls20-mechanism-training-pilot100.jsonl")
        row = json.loads(source.read_text().splitlines()[0])
        row["search_truncated"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "complete contextual engine proof"):
                events.load_verified_specs(path, "train", 1)

    def test_train_and_validation_namespaces_are_disjoint(self):
        source = Path("data/ls20-mechanism-training-pilot100.jsonl")
        row = json.loads(source.read_text().splitlines()[0])
        row["seed"] = 1_000_001
        row["training_context_index"] = row["seed"] % 7
        row["proof"]["context_index"] = row["seed"] % 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-namespace.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "training seed .*namespace"):
                events.load_verified_specs(path, "train", 1)

    def test_context_failure_keeps_five_value_collection_contract(self):
        row = json.loads(Path("data/ls20-mechanism-training-pilot100.jsonl").read_text().splitlines()[0])
        with patch.object(events.world_data, "verified_context", return_value=(None, None, {"seed": row["seed"]})):
            result = events.collect_level(row, "train", 8, 2, 100, np.random.default_rng(1))
        self.assertEqual(len(result), 5)
        self.assertEqual(result[1]["shortfall"], "context_verification_failed")

    def test_valid_source_selection_is_bounded_and_unique(self):
        rows = events.load_verified_specs(
            Path("data/ls20-mechanism-training-pilot100.jsonl"), "train", 5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({int(row["seed"]) for row in rows}), 5)
        self.assertTrue(all(int(row["seed"]) % 7 == row["training_context_index"] for row in rows))


if __name__ == "__main__":
    unittest.main()
