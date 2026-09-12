import unittest

import numpy as np

from tools.structured_onpolicy_sampling import PairedStateSampler


class PairedStateSamplingTests(unittest.TestCase):
    def test_distinct_levels_and_same_level_replacement_excluding_anchors(self):
        seeds = np.arange(2048, dtype=np.int64)
        visited = np.repeat(seeds[:1024], 3)
        flags = np.tile([False, True, True], 1024)
        sampler = PairedStateSampler(seeds, seeds % 5 + 1, visited, flags)
        batch = sampler.draw(1024, 512, .7, np.random.default_rng(23))
        selected = batch.trajectory_rows >= 0
        self.assertEqual(len(np.unique(seeds[batch.base_rows])), 1024)
        self.assertEqual(int(selected.sum()), 512)
        np.testing.assert_array_equal(visited[batch.trajectory_rows[selected]], seeds[batch.base_rows[selected]])
        self.assertTrue(flags[batch.trajectory_rows[selected]].all())
        self.assertEqual(int(batch.base_views.sum()), 512)

    def test_long_trajectories_do_not_multiply_level_sampling_weight(self):
        # Equal difficulty;100 eligible levels, one has100 times more states.
        seeds = np.arange(100)
        visited = np.concatenate((np.zeros(100, dtype=int), np.arange(1, 100)))
        sampler = PairedStateSampler(seeds, np.ones(100, int), visited, np.ones(len(visited), bool))
        rng = np.random.default_rng(4)
        counts = np.zeros(100, int)
        for _ in range(1000):
            batch = sampler.draw(8, 4, .5, rng)
            counts[batch.base_rows[batch.trajectory_rows >= 0]] += 1
        self.assertLess(counts[0], 80)
        self.assertGreater(counts[0], 10)

    def test_reproducible_and_rejects_unpaired_or_infeasible(self):
        seeds = np.arange(16)
        sampler = PairedStateSampler(seeds, seeds % 5 + 1, seeds[:4], np.ones(4, bool))
        a = sampler.draw(8, 4, 0, np.random.default_rng(2))
        b = sampler.draw(8, 4, 0, np.random.default_rng(2))
        np.testing.assert_array_equal(a.base_rows, b.base_rows)
        np.testing.assert_array_equal(a.trajectory_rows, b.trajectory_rows)
        for batch, count in [(7, 4), (32, 4), (8, 5)]:
            with self.assertRaises(ValueError):
                sampler.draw(batch, count, 0, np.random.default_rng(2))
        with self.assertRaises(ValueError):
            PairedStateSampler(seeds, seeds % 5 + 1, np.array([999]), np.ones(1, bool))


if __name__ == '__main__':
    unittest.main()
