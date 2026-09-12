import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np

from tools import collect_structured_event_transitions as base
from tools import collect_structured_event_transitions_fast as fast


class _Result:
    def __init__(self, frame, *, finished=False, won=False):
        self.frame = frame
        self.finished = finished
        self.won = won


class _Env:
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
            self._lives -= 1
            self._state = SimpleNamespace(value="GAME_OVER")
            return _Result(np.full((64, 64), action, dtype=np.uint8), finished=True)
        return _Result(np.full((64, 64), action, dtype=np.uint8))

    def player_cell(self): return (self._action, 0)
    def triple(self): return (0, 0, 0)
    def steps_left(self): return 10
    def lives(self): return self._lives
    def goals_solved(self): return []
    def render(self): return np.full((64, 64), self._action, dtype=np.uint8)


class _Oracle:
    truncated = False
    start = (0,)
    layout = SimpleNamespace()
    refills = ()

    def state_of(self, _env): return None
    def distance_for(self, _state): return None


class _FixedRng:
    def __init__(self, action): self.action = action
    def integers(self, _low, _high): return self.action


class FastStructuredEventTransitionTests(unittest.TestCase):
    def test_output_metadata_is_generic_without_a_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation = root / "train.jsonl", root / "validation.jsonl"
            train.write_text("{}\n")
            validation.write_text("{}\n")
            args = SimpleNamespace(train=train, validation=validation,
                                   reference_data=None, reference_report=None)
            metadata = fast.build_output_metadata(args, [])
            encoded = json.dumps(metadata)
            self.assertNotIn("pilot-v3", encoded)
            self.assertFalse(metadata["reference"]["requested"])
            self.assertIsNone(metadata["reference"]["data"])
            self.assertEqual(metadata["source_paths"]["train"], str(train))

    def test_reference_comparison_requires_run_identity_and_exact_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_data = root / "reference.npz"
            current_data = root / "current.npz"
            np.savez_compressed(reference_data, frames=np.arange(8, dtype=np.int16),
                                meta=np.array("reference"))
            np.savez_compressed(current_data, frames=np.arange(8, dtype=np.int16),
                                meta=np.array("current"))
            sources = {
                "train": {"sha256_before": "train", "selected": [1, 2]},
                "validation": {"sha256_before": "validation", "selected": [3, 4]},
            }
            config = {"rng_seed": 20260912, "history": 8, "max_actions": 300,
                      "train_levels": 2, "validation_levels": 2}
            reference_report = root / "reference.json"
            reference_report.write_text(__import__("json").dumps({
                "run_config": config, "sources": sources,
                "output": {"sha256": fast.digest(reference_data)}, "elapsed_seconds": 10.,
            }))
            current_report = {"run_config": dict(config), "sources": sources,
                              "output": {"sha256": fast.digest(current_data)},
                              "elapsed_seconds": 5.}
            comparison = fast.compare_reference(current_report, current_data,
                                                reference_data, reference_report)
            self.assertTrue(comparison["matched"])
            self.assertEqual(comparison["speedup_vs_reference"], 2.)
            altered = dict(config, rng_seed=7)
            current_report["run_config"] = altered
            mismatch = fast.compare_reference(current_report, current_data,
                                              reference_data, reference_report)
            self.assertFalse(mismatch["matched"])
            self.assertNotIn("speedup_vs_reference", mismatch)

    def test_ordinary_step_checks_one_branch_without_four_way_capture(self):
        spec = {"seed": 700001, "difficulty": 5}
        env = _Env()
        env._state = SimpleNamespace(value="NOT_FINISHED")
        oracle = _Oracle()
        def moved(_layout, _state, _action, _refills): return (None, "moved")
        with patch.object(fast.world_data, "verified_context", return_value=(env, oracle, {"seed": 700001})), \
                patch.object(fast.base, "capture_branches", side_effect=AssertionError("unexpected capture")), \
                patch.object(fast, "simulate", side_effect=moved):
            rows, report, ordinary, retained, engine, unverified = fast.collect_level_fast(
                spec, "train", 8, 1, 100, _FixedRng(0))
        self.assertEqual(rows, [])
        self.assertEqual(ordinary, 1)
        self.assertEqual(retained, 0)
        self.assertEqual(engine, 1)
        self.assertEqual(unverified, 0)
        self.assertEqual(report["ordinary_mechanics_checks"], 1)

    def test_actual_loss_defers_to_complete_capture_and_retains_all_four_checks(self):
        spec = {"seed": 700001, "difficulty": 5}
        env = _Env()
        oracle = _Oracle()
        def died_or_moved(_layout, _state, action, _refills):
            return (None, "died" if action == 3 else "moved")
        with patch.object(fast.world_data, "verified_context", return_value=(env, oracle, {"seed": 700001})), \
                patch.object(fast, "simulate", side_effect=died_or_moved), \
                patch.object(fast.base, "simulate", side_effect=died_or_moved):
            rows, report, ordinary, retained, engine, unverified = fast.collect_level_fast(
                spec, "train", 8, 1, 100, _FixedRng(3))
        self.assertEqual(len(rows), 1)
        self.assertEqual(ordinary, 1)
        self.assertEqual(retained, 4)
        self.assertEqual(engine, 5)
        self.assertEqual(unverified, 0)
        self.assertEqual(int(rows[0]["selected_action"]), 3)
        self.assertEqual(int(rows[0]["retained_oracle_checked_branches"]), 4)
        self.assertEqual(report["retained_branch_checks"], 4)


if __name__ == "__main__":
    unittest.main()
