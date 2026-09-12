"""Tests for level-first generated curriculum sampling."""

import unittest

import torch

from pebby.agent.curriculum_sampling import CurriculumSampler


def dataset(levels, rows_per_level=1):
    rows = []
    metadata = []
    for seed, difficulty in levels:
        metadata.append({"seed": seed, "difficulty": difficulty})
        rows.extend([seed] * rows_per_level if isinstance(rows_per_level, int)
                     else [seed] * rows_per_level.get(seed, 1))
    return {"seeds": torch.tensor(rows, dtype=torch.int64),
            "meta": {"levels": metadata}}


class CurriculumSamplingTests(unittest.TestCase):
    def test_ratios_interpolate_between_default_endpoints(self):
        sampler = CurriculumSampler(dataset([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]))
        dtype = torch.float64
        self.assertTrue(torch.allclose(sampler.ratios(0.), torch.tensor([.55, .25, .12, .06, .02], dtype=dtype)))
        self.assertTrue(torch.allclose(sampler.ratios(1.), torch.tensor([.05, .10, .20, .25, .40], dtype=dtype)))
        self.assertTrue(torch.allclose(sampler.ratios(.5), torch.tensor([.30, .175, .16, .155, .21], dtype=dtype)))

    def test_indices_are_level_first_and_do_not_repeat_a_seed(self):
        sampler = CurriculumSampler(dataset([(seed, 1) for seed in range(8)]),
                                    start=[1., 0., 0., 0., 0.], end=[1., 0., 0., 0., 0.])
        data = dataset([(seed, 1) for seed in range(8)])
        indices = sampler.indices(8, 0., torch.Generator().manual_seed(4))
        chosen = data["seeds"][indices].tolist()
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(set(chosen)), 8)
        self.assertEqual(sampler.last_difficulty_counts, {1: 8, 2: 0, 3: 0, 4: 0, 5: 0})
        self.assertEqual(sampler.last_distinct_levels, 8)

    def test_uniform_level_sampling_ignores_frame_count(self):
        levels = [(11, 1), (12, 1), (13, 1), (14, 1)]
        sampler = CurriculumSampler(dataset(levels, {11: 1, 12: 8, 13: 32, 14: 64}),
                                    start=[1., 0., 0., 0., 0.], end=[1., 0., 0., 0., 0.])
        counts = {seed: 0 for seed, _ in levels}
        generator = torch.Generator().manual_seed(9)
        data = dataset(levels, {11: 1, 12: 8, 13: 32, 14: 64})
        for _ in range(400):
            indices = sampler.indices(2, 0., generator)
            for seed in data["seeds"][indices].tolist():
                counts[seed] += 1
        self.assertTrue(all(150 <= count <= 250 for count in counts.values()), counts)

    def test_progression_moves_exact_quotas_from_easy_to_hard(self):
        levels = [(difficulty * 100 + seed, difficulty)
                  for difficulty in range(1, 6) for seed in range(20)]
        sampler = CurriculumSampler(dataset(levels))
        early = sampler.indices(10, 0., torch.Generator().manual_seed(1))
        early_counts = sampler.last_difficulty_counts.copy()
        late = sampler.indices(10, 1., torch.Generator().manual_seed(1))
        late_counts = sampler.last_difficulty_counts.copy()
        self.assertEqual(sum(early_counts.values()), 10)
        self.assertEqual(sum(late_counts.values()), 10)
        self.assertGreater(early_counts[1], late_counts[1])
        self.assertLess(early_counts[5], late_counts[5])
        self.assertNotEqual(early.tolist(), late.tolist())
        self.assertEqual(len(sampler.last_level_seeds), 10)

    def test_same_cpu_generator_seed_reproduces_indices_and_reports(self):
        sampler = CurriculumSampler(dataset([(seed, seed % 5 + 1) for seed in range(20)]))
        first = sampler.indices(7, .37, torch.Generator().manual_seed(22))
        first_counts = sampler.last_difficulty_counts.copy()
        first_levels = sampler.last_level_seeds
        second = sampler.indices(7, .37, torch.Generator().manual_seed(22))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_counts, sampler.last_difficulty_counts)
        self.assertEqual(first_levels, sampler.last_level_seeds)

    def test_metadata_validation_and_unknown_rows(self):
        data = {"seeds": torch.tensor([10, 10, 99]),
                "meta": {"levels": [{"seed": 10, "difficulty": 1},
                                      {"seed": 20, "difficulty": 2}]}}
        sampler = CurriculumSampler(data, start=[1., 0., 0., 0., 0.], end=[1., 0., 0., 0., 0.])
        self.assertEqual(sampler.seeds.tolist(), [10])
        data['meta']['levels'].extend([{'seed': 30, 'excluded': 'truncated source proof'},
                                       {'seed': 40}])
        sampler = CurriculumSampler(data)
        self.assertEqual(sampler.seeds.tolist(), [10])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            CurriculumSampler({"seeds": [1], "meta": {"levels": [
                {"seed": 1, "difficulty": 1}, {"seed": 1, "difficulty": 2}]}})
        with self.assertRaisesRegex(ValueError, "difficulty"):
            CurriculumSampler({"seeds": [1], "meta": {"levels": [
                {"seed": 1, "difficulty": 6}]}})
        with self.assertRaisesRegex(ValueError, "no rows"):
            CurriculumSampler({"seeds": [1], "meta": {"levels": [
                {"seed": 2, "difficulty": 1}]}})

    def test_insufficient_distinct_levels_is_a_clear_error(self):
        sampler = CurriculumSampler(dataset([(1, 1), (2, 1)]),
                                    start=[1., 0., 0., 0., 0.], end=[1., 0., 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, "distinct level"):
            sampler.indices(3, 0., torch.Generator().manual_seed(0))
        insufficient_stage = CurriculumSampler(dataset([(1, 1), (2, 2)]),
                                               start=[1., 0., 0., 0., 0.],
                                               end=[1., 0., 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, "difficulty 1"):
            insufficient_stage.indices(2, 0., torch.Generator().manual_seed(0))
        with self.assertRaisesRegex(ValueError, "1024"):
            sampler.indices(1025, 0., torch.Generator().manual_seed(0))

    def test_schedule_coverage_accounts_for_fractional_quota_rounding(self):
        sampler = CurriculumSampler(dataset([(1, 1), *[(seed, 2) for seed in range(2, 12)]]),
                                    start=[.15, .85, 0., 0., 0.], end=[.15, .85, 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, 'needs 2 distinct levels'):
            sampler.check_coverage(8)

    def test_ratio_and_generator_validation(self):
        data = dataset([(1, 1), (2, 2)])
        with self.assertRaisesRegex(ValueError, "five"):
            CurriculumSampler(data, start=[1., 0.], end=[1., 0.])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            CurriculumSampler(data, start=[-1., 1., 0., 0., 0.])
        sampler = CurriculumSampler(data, start=[1., 0., 0., 0., 0.],
                                    end=[1., 0., 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, "progress"):
            sampler.ratios(1.1)
        with self.assertRaisesRegex(ValueError, "Generator"):
            sampler.indices(1, 0., None)


if __name__ == "__main__":
    unittest.main()
