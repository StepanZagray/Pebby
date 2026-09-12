import copy
import unittest

import numpy as np
import torch

from pebby.agent.on_policy_sampling import OnPolicySampler, mixed_batch


def fixture():
    levels = [{'seed': stage * 1000 + i, 'difficulty': stage} for stage in range(1, 6) for i in range(500)]
    base = {'seeds': np.repeat([x['seed'] for x in levels], 2), 'meta': {'levels': levels}}
    subset = [x for x in levels if x['seed'] % 1000 < 205]
    supplement = {'seeds': np.repeat([x['seed'] for x in subset], 3),
                  'meta': {'levels': subset, 'on_policy_rows': [i for i in range(len(subset)*3) if i % 3 != 2]}}
    return base, supplement


class OnPolicySamplingTests(unittest.TestCase):
    def test_auxiliary_rows_enter_training_without_changing_policy_quota_or_level_uniqueness(self):
        base, supplemental = fixture()
        supplemental['meta']['auxiliary_rows'] = list(range(2, len(supplemental['seeds']), 3))
        sampler = OnPolicySampler(base, supplemental, fraction=.25, auxiliary_fraction=1.,
                                   start=[.35,.25,.20,.15,.05], end=[.05,.10,.20,.25,.40])
        indices = sampler.indices(1024, .5, torch.Generator().manual_seed(7))
        policy = set(supplemental['meta']['on_policy_rows'])
        auxiliary = set(supplemental['meta']['auxiliary_rows'])
        self.assertEqual(sum(int(i) in policy for i in indices[1]), 256)
        self.assertGreater(sum(int(i) in auxiliary for i in indices[1]), 0)
        self.assertEqual(sampler.last_on_policy_count, 256)
        self.assertEqual(sampler.last_auxiliary_count, sum(int(i) in auxiliary for i in indices[1]))
        batch = mixed_batch({'seed': torch.tensor(base['seeds'])},
                            {'seed': torch.tensor(supplemental['seeds'])}, indices)
        self.assertEqual(len(torch.unique(batch['seed'])), 1024)
        self.assertEqual(tuple(batch['seed'].tolist()), sampler.last_level_seeds)

    def sampler(self, base, supplement):
        return OnPolicySampler(base, supplement, fraction=.25,
                               start=[.35,.25,.20,.15,.05], end=[.05,.10,.20,.25,.40])

    def test_exact_1024_distinct_levels_256_policy_rows_and_difficulty_schedule(self):
        base, supplement = fixture()
        sampler = self.sampler(base, supplement)
        for progress in (0., .25, .5, .75, 1.):
            indices = sampler.indices(1024, progress, torch.Generator().manual_seed(7))
            a, b, order = indices
            self.assertEqual(len(a), 768)
            self.assertEqual(len(b), 256)
            self.assertTrue(all(int(i) in supplement['meta']['on_policy_rows'] for i in b))
            batch = mixed_batch({'seed': torch.tensor(base['seeds'])},
                                {'seed': torch.tensor(supplement['seeds'])}, indices)
            self.assertEqual(len(torch.unique(batch['seed'])), 1024)
            self.assertEqual(tuple(batch['seed'].tolist()), sampler.last_level_seeds)
            actual = torch.bincount(batch['seed'] // 1000, minlength=6)[1:]
            expected = sampler._quotas(sampler.ratios(progress), 1024)
            torch.testing.assert_close(actual, expected)
            repeated = sampler.indices(1024, progress, torch.Generator().manual_seed(7))
            for left, right in zip(indices, repeated):
                torch.testing.assert_close(left, right)

    def test_bad_metadata_and_inadequate_coverage_rejected(self):
        base, supplement = fixture()
        for rows in ([], [True], [-1], [0, 0], [len(supplement['seeds'])]):
            wrong = copy.deepcopy(supplement)
            wrong['meta']['on_policy_rows'] = rows
            with self.assertRaises(ValueError):
                self.sampler(base, wrong)
        wrong = copy.deepcopy(supplement)
        wrong['meta']['levels'][0]['difficulty'] = 5
        with self.assertRaisesRegex(ValueError, 'match base'):
            self.sampler(base, wrong)
        wrong = copy.deepcopy(supplement)
        wrong['meta']['on_policy_rows'] = [0]
        with self.assertRaisesRegex(ValueError, 'on-policy difficulty'):
            self.sampler(base, wrong).check_coverage(1024)

    def test_mismatched_tensor_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fields differ'):
            mixed_batch({'x': torch.arange(2)}, {}, (torch.tensor([0]), torch.tensor([0]), torch.arange(2)))
        with self.assertRaisesRegex(ValueError, 'dtypes differ'):
            mixed_batch({'x': torch.arange(2)}, {'x': torch.arange(2).float()},
                        (torch.tensor([0]), torch.tensor([0]), torch.arange(2)))

    def test_custom_nearly_all_reserved_batch_preserves_quotas(self):
        base, supplemental = fixture()
        for seed in range(25):
            ratios = torch.rand(5, generator=torch.Generator().manual_seed(seed)).tolist()
            sampler = OnPolicySampler(base, supplemental, fraction=.95, start=ratios, end=ratios)
            a, b, order = sampler.indices(16, .5, torch.Generator().manual_seed(seed))
            self.assertEqual((len(a), len(b)), (1, 15))
            self.assertEqual(len(set(sampler.last_level_seeds)), 16)


if __name__ == '__main__':
    unittest.main()
