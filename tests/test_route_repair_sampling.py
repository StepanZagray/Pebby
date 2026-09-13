import unittest

import numpy as np

from pebby.agent.route_repair_sampling import RouteReplaySampler


class RouteReplaySamplerTests(unittest.TestCase):
    def fixture(self):
        seeds = np.arange(28)
        tiers = {int(seed): int(seed // 4 + 1) for seed in seeds}
        base = dict(seeds=np.repeat(seeds, 2), optimal=np.tile([0, 1], len(seeds)))
        recent = dict(seeds=np.repeat(seeds, 3), row_kind=np.tile([0, 1, 2], len(seeds)),
                      optimal=np.tile([1, 2, 0], len(seeds)), policy_valid=np.tile([True, True, False], len(seeds)),
                      dynamics_valid=np.ones(len(seeds) * 3, dtype=bool))
        return base, recent, tiers

    def test_matched_distinct_levels_and_tier_counts(self):
        base, recent, tiers = self.fixture()
        sampler = RouteReplaySampler(base, recent, tiers, tier_counts=(4,) * 7, recent_fraction=.5)
        a, b = sampler.sample(np.random.default_rng(3)), sampler.sample(np.random.default_rng(3))
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(len(np.unique(a['seeds'])), 28)
        self.assertEqual(np.bincount([tiers[int(s)] for s in a['seeds']])[1:].tolist(), [4] * 7)
        self.assertEqual(int((a['recent_rows'] >= 0).sum()), 14)
        np.testing.assert_array_equal(base['seeds'][a['base_rows']], a['seeds'])
        mask = a['recent_rows'] >= 0
        np.testing.assert_array_equal(recent['seeds'][a['recent_rows'][mask]], a['seeds'][mask])

    def test_zero_policy_failure_rows_remain_available(self):
        base, recent, tiers = self.fixture()
        sampler = RouteReplaySampler(base, recent, tiers, tier_counts=(4,) * 7,
                                     recent_fraction=1., kind_weights=(0., 0., 1.))
        selection = sampler.sample(np.random.default_rng(1))
        self.assertTrue((recent['optimal'][selection['recent_rows']] == 0).all())
        self.assertTrue((recent['row_kind'][selection['recent_rows']] == 2).all())
        self.assertGreater(int((base['optimal'][selection['base_rows']] == 0).sum()), 0)

    def test_rejects_nontrain_and_mislabelled_eligibility(self):
        base, recent, tiers = self.fixture()
        recent['policy_valid'][0] = False
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            RouteReplaySampler(base, recent, tiers, tier_counts=(4,) * 7)
        recent['policy_valid'][0] = True
        recent['seeds'][0] = 1000
        with self.assertRaisesRegex(ValueError, 'original TRAIN'):
            RouteReplaySampler(base, recent, tiers, tier_counts=(4,) * 7)

    def test_no_replay_qualification_still_keeps_all_base_rows(self):
        base, _, tiers = self.fixture()
        sampler = RouteReplaySampler(base, None, tiers, tier_counts=(2,) * 7, recent_fraction=0.)
        sample = sampler.sample(np.random.default_rng(2))
        self.assertTrue((sample['recent_rows'] == -1).all())
        self.assertGreater(int((base['optimal'][sample['base_rows']] == 0).sum()), 0)

    def test_rejects_insufficient_distinct_tier_levels(self):
        base, recent, tiers = self.fixture()
        with self.assertRaisesRegex(ValueError, 'distinct levels'):
            RouteReplaySampler(base, recent, tiers, tier_counts=(5, 4, 4, 4, 4, 4, 4))

    def test_quality_allowlists_control_sampling_without_discarding_zero_policy(self):
        base, recent, tiers = self.fixture()
        base_rows = np.arange(0, len(base['seeds']), 2)
        recent_rows = np.arange(2, len(recent['seeds']), 3)
        sampler = RouteReplaySampler(base, recent, tiers, tier_counts=(4,) * 7,
            recent_fraction=1., base_rows=base_rows, recent_rows=recent_rows)
        sample = sampler.sample(np.random.default_rng(2))
        self.assertTrue(np.isin(sample['base_rows'], base_rows).all())
        self.assertTrue(np.isin(sample['recent_rows'], recent_rows).all())
        self.assertTrue((base['optimal'][sample['base_rows']] == 0).all())
        self.assertTrue((recent['optimal'][sample['recent_rows']] == 0).all())


if __name__ == '__main__':
    unittest.main()
