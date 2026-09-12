import unittest

import numpy as np
import torch

from tools.structured_onpolicy_sampling import PairedRows, PairedStateSampler
from tools.train_structured_onpolicy_dynamics import (
    INITIAL,
    INITIAL_SHA256,
    build_schedule,
    make_local_model,
    prepare_dynamics_batch,
    schedule_event_counts,
    _used_rows_digest,
    _load_initial,
)


def _view(offset, count=8):
    fields = np.zeros((count, 148, 96), dtype=np.float32)
    next_fields = np.zeros((count, 4, 148, 96), dtype=np.float32)
    fields[:, :, 48:85] = .5
    next_fields[:, :, :, 48:85] = .5
    fields[:, 0, 0] = np.arange(count) + offset
    next_fields[:, :, 0, 0] = np.arange(count)[:, None] + offset + np.arange(4)
    data = {
        "fields": fields,
        "next_fields": next_fields,
        "seeds": np.arange(count, dtype=np.int64) + 100 + offset * 100,
        "difficulties": np.arange(count, dtype=np.int8) % 5 + 1,
        "player_cell": np.zeros((count, 2), dtype=np.int64),
        "next_player_cell": np.zeros((count, 4, 2), dtype=np.int64),
        "triple": np.zeros((count, 3), dtype=np.int64),
        "next_triple": np.zeros((count, 4, 3), dtype=np.int64),
        "steps": np.full(count, 10, dtype=np.int64),
        "next_steps": np.full((count, 4), 9, dtype=np.int64),
        "lives": np.full(count, 3, dtype=np.int64),
        "next_lives": np.full((count, 4), 3, dtype=np.int64),
        "optimal": np.ones(count, dtype=np.uint8),
        "next_optimal": np.ones((count, 4), dtype=np.uint8),
        "lost_life": np.zeros((count, 4), dtype=bool),
        "terminal": np.zeros((count, 4), dtype=bool),
        "won": np.zeros((count, 4), dtype=bool),
    }
    return data


def _trajectory(base):
    data = {key: value.copy() for key, value in base.items()}
    data["seeds"] = np.array([base["seeds"][0], base["seeds"][2]], dtype=np.int64)
    data["on_policy"] = np.array([True, False], dtype=bool)
    data["fields"] = data["fields"][:2] + 100
    data["next_fields"] = data["next_fields"][:2] + 100
    for key in ("difficulties", "player_cell", "next_player_cell", "triple",
                "next_triple", "steps", "next_steps", "lives", "next_lives",
                "optimal", "next_optimal", "lost_life", "terminal", "won"):
        data[key] = data[key][:2].copy()
    return data


class PairedDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_is_reproducible_and_level_balanced(self):
        seeds = np.arange(32, dtype=np.int64)
        visited = np.repeat(seeds[:16], 2)
        sampler = PairedStateSampler(seeds, seeds % 5 + 1, visited,
                                     np.ones(len(visited), dtype=bool))
        first = build_schedule(sampler, 8, 4, 5, seed=17)
        second = build_schedule(sampler, 8, 4, 5, seed=17)
        for (a, aa), (b, bb) in zip(first, second):
            np.testing.assert_array_equal(a.base_rows, b.base_rows)
            np.testing.assert_array_equal(a.base_views, b.base_views)
            np.testing.assert_array_equal(a.trajectory_rows, b.trajectory_rows)
            np.testing.assert_array_equal(aa, bb)
            self.assertEqual(len(np.unique(a.base_rows)), 8)
            self.assertEqual(int((a.trajectory_rows >= 0).sum()), 4)
            self.assertTrue(np.all((aa >= 0) & (aa < 4)))

    def test_treatment_replaces_only_verified_on_policy_rows(self):
        view0 = _view(0)
        view1 = _view(1)
        # Make both legacy views represent the same level namespace while
        # retaining distinct values, as load_inputs/validate_pair does.
        view1["seeds"] = view0["seeds"].copy()
        trajectory = _trajectory(view0)
        trajectory["on_policy"][:] = True
        selection = PairedRows(
            base_rows=np.array([0, 1, 2, 3]),
            base_views=np.array([0, 1, 0, 1], dtype=np.int8),
            trajectory_rows=np.array([0, -1, 1, -1]),
        )
        actions = np.array([3, 2, 1, 0], dtype=np.int64)
        control = prepare_dynamics_batch([view0, view1], trajectory, selection, actions, False)
        treatment = prepare_dynamics_batch([view0, view1], trajectory, selection, actions, True)
        for key in control:
            if key == "actions":
                np.testing.assert_array_equal(control[key], treatment[key])
            else:
                np.testing.assert_array_equal(control[key][[1, 3]], treatment[key][[1, 3]])
        np.testing.assert_array_equal(treatment["fields"][[0, 2]], trajectory["fields"][[0, 1]])
        np.testing.assert_array_equal(
            treatment["next_fields"][[0, 2]],
            trajectory["next_fields"][[0, 1], actions[[0, 2]]],
        )
        self.assertFalse(np.array_equal(control["fields"][0], treatment["fields"][0]))

    def test_batch_branch_shapes_and_event_counts(self):
        view = _view(0)
        trajectory = _trajectory(view)
        trajectory["on_policy"][:] = True
        selection = PairedRows(np.array([0, 1]), np.array([0, 0], dtype=np.int8),
                               np.array([0, -1]))
        actions = np.array([2, 1], dtype=np.int64)
        data = prepare_dynamics_batch([view, view], trajectory, selection, actions, True)
        self.assertEqual(data["fields"].shape, (2, 148, 96))
        self.assertEqual(data["next_fields"].shape, (2, 148, 96))
        self.assertEqual(data["next_optimal"].shape, (2,))
        schedule = [(selection, actions)]
        counts = schedule_event_counts([view, view], trajectory, schedule, True)
        self.assertEqual(counts["branches"], 2)
        self.assertEqual(counts["terminal_failure"], 0)
        self.assertNotEqual(_used_rows_digest(schedule, False),
                            _used_rows_digest(schedule, True))

    def test_expert_anchor_cannot_be_replaced(self):
        view = _view(0)
        trajectory = _trajectory(view)
        trajectory["on_policy"][0] = False
        selection = PairedRows(np.array([0]), np.array([0], dtype=np.int8), np.array([0]))
        with self.assertRaisesRegex(ValueError, "expert anchor"):
            prepare_dynamics_batch([view, view], trajectory, selection,
                                   np.array([0], dtype=np.int64), True)

    def test_real_initializer_is_exact_local_capacity(self):
        if not INITIAL.exists():
            self.skipTest("generated dynamics initializer absent")
        saved, sha = _load_initial(INITIAL)
        self.assertEqual(sha, INITIAL_SHA256)
        model = make_local_model(saved, "cpu")
        self.assertEqual(model.parameter_count(), 294664)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
