import unittest

from tools.spatial_paired_statistics import (exact_binomial_interval,
                                             exact_mcnemar_two_sided,
                                             paired_win_statistics)


class SpatialPairedStatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_uses_only_discordant_pairs(self):
        self.assertEqual(exact_mcnemar_two_sided(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_two_sided(1, 1), 1.0)
        self.assertAlmostEqual(exact_mcnemar_two_sided(0, 3), .25)

    def test_clopper_pearson_boundaries_and_order(self):
        self.assertEqual(exact_binomial_interval(0, 0), (None, None))
        self.assertEqual(exact_binomial_interval(0, 10)[0], 0.0)
        self.assertEqual(exact_binomial_interval(10, 10)[1], 1.0)
        low, high = exact_binomial_interval(3, 10)
        self.assertLess(low, .3)
        self.assertGreater(high, .3)

    def test_paired_rows_report_gains_losses_unpaired_and_transformed_interval(self):
        reference = [
            {'level': 1, 'completed': True},
            {'level': 2, 'completed': False},
            {'level': 3, 'completed': False},
            {'level': 4, 'completed': True},
        ]
        candidate = [
            {'level': 1, 'completed': True},
            {'level': 2, 'completed': True},
            {'level': 4, 'completed': False},
            {'level': 5, 'completed': True},
        ]
        result = paired_win_statistics(reference, candidate, identifier='level')
        self.assertEqual(result['paired_levels'], 3)
        self.assertEqual(result['candidate_only_wins'], 1)
        self.assertEqual(result['reference_only_wins'], 1)
        self.assertEqual(result['net_wins'], 0)
        self.assertEqual(result['unpaired_reference'], [3])
        self.assertEqual(result['unpaired_candidate'], [5])
        self.assertEqual(result['mcnemar_exact_two_sided_p'], 1.0)
        self.assertEqual(len(result['net_win_rate_difference_conservative_interval']), 2)

    def test_no_observed_discordance_does_not_imply_zero_generalization_uncertainty(self):
        result = paired_win_statistics([True] * 70, [True] * 70)
        low, high = result['net_win_rate_difference_conservative_interval']
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        self.assertEqual(result['conditional_candidate_win_share_exact_interval'], (None, None))

    def test_exact_binomial_known_bounds_and_symmetric_net_interval(self):
        self.assertAlmostEqual(exact_binomial_interval(0, 10)[1], 1 - .025 ** .1)
        self.assertAlmostEqual(exact_binomial_interval(10, 10)[0], .025 ** .1)
        result = paired_win_statistics([True, False], [False, True])
        low, high = result['net_win_rate_difference_conservative_interval']
        self.assertAlmostEqual(low, -high)

    def test_large_panel_does_not_overflow_binomial_coefficients(self):
        low, high = exact_binomial_interval(1000, 2000)
        self.assertTrue(.47 < low < .5 < high < .53)
        self.assertEqual(exact_mcnemar_two_sided(1000, 1000), 1.0)
        self.assertLess(exact_mcnemar_two_sided(0, 2000), 1e-100)
        result = paired_win_statistics([True, False] * 1000, [False, True] * 1000)
        self.assertEqual(result['mcnemar_exact_two_sided_p'], 1.0)

    def test_positional_pairing_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            paired_win_statistics([True], [True, False])

    def test_extreme_confidence_widens_net_interval_without_rounding_error(self):
        result = paired_win_statistics([True], [False], confidence=1 - 2 ** -53)
        self.assertEqual(result['net_win_rate_difference_conservative_interval'], (-1., 1.))


if __name__ == '__main__':
    unittest.main()
