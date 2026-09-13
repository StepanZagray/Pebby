"""CPU checks of probability semantics and gradients, without model training."""
import math
import unittest

import torch

from pebby.agent.optimal_set_objective import optimal_action_loss


class OptimalSetObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def test_set_is_indifferent_to_allocation_inside_ties_uniform_penalizes_imbalance(self):
        balanced = torch.tensor([[.4, .4, .1, .1]]).log()
        skewed = torch.tensor([[.79, .01, .1, .1]]).log()
        mask = torch.tensor([3])
        torch.testing.assert_close(optimal_action_loss(balanced, mask, 'set'),
                                   optimal_action_loss(skewed, mask, 'set'))
        self.assertAlmostEqual(float(optimal_action_loss(balanced, mask, 'set')), -math.log(.8), places=6)
        self.assertGreater(float(optimal_action_loss(skewed, mask, 'uniform')),
                           float(optimal_action_loss(balanced, mask, 'uniform')))

    def test_set_loss_measures_total_suboptimal_probability(self):
        mask = torch.tensor([5])
        for probabilities, optimal_mass in (([.3, .2, .1, .4], .4), ([.5, .1, .3, .1], .8)):
            logits = torch.tensor([probabilities]).log()
            self.assertAlmostEqual(float(optimal_action_loss(logits, mask, 'set')), -math.log(optimal_mass), places=6)

    def test_single_optimal_action_matches_cross_entropy_in_both_modes(self):
        logits = torch.tensor([[2., -1., 3., 0.], [-3., 1., 2., 4.]])
        masks = torch.tensor([4, 2])
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 1]))
        for mode in ('uniform', 'set'):
            torch.testing.assert_close(optimal_action_loss(logits, masks, mode), expected)

    def test_all_optimal_is_zero_constraint_only_for_set_mode(self):
        logits = torch.tensor([[3., 0., -1., 7.], [-10000., 10000., 4., 0.]], requires_grad=True)
        masks = torch.tensor([15, 15])
        loss = optimal_action_loss(logits, masks, 'set')
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=0, rtol=0)
        self.assertGreater(float(optimal_action_loss(logits, masks, 'uniform').detach()), math.log(4))

    def test_all_undefined_is_differentiable_zero_for_each_mode(self):
        for mode in ('uniform', 'set'):
            logits = torch.randn(3, 4, requires_grad=True)
            loss = optimal_action_loss(logits, torch.zeros(3, dtype=torch.long), mode)
            self.assertEqual(float(loss.detach()), 0.)
            self.assertTrue(loss.requires_grad)
            loss.backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=0, rtol=0)

    def test_defined_rows_have_equal_weight_and_undefined_rows_no_gradient(self):
        logits = torch.tensor([[1., 2., 3., 4.], [10000., -10000., 0., 3.], [3., 4., 1., 2.]], requires_grad=True)
        masks = torch.tensor([1, 0, 7])
        for mode in ('uniform', 'set'):
            expected = (optimal_action_loss(logits[:1], masks[:1], mode) +
                        optimal_action_loss(logits[2:], masks[2:], mode)) / 2
            actual = optimal_action_loss(logits, masks, mode)
            torch.testing.assert_close(actual, expected)
            gradient, = torch.autograd.grad(actual, logits)
            torch.testing.assert_close(gradient[1], torch.zeros(4), atol=0, rtol=0)

    def test_set_gradient_moves_mass_into_optimal_set_and_preserves_device_autograd(self):
        logits = torch.tensor([[.3, -.2, .8, .1]], dtype=torch.float64, requires_grad=True)
        loss = optimal_action_loss(logits, torch.tensor([5]), 'set')
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.device, logits.device)
        loss.backward()
        self.assertEqual(logits.grad.dtype, logits.dtype)
        self.assertTrue((logits.grad[0, [0, 2]] < 0).all())
        self.assertTrue((logits.grad[0, [1, 3]] > 0).all())
        improved = logits.detach() - .1 * logits.grad
        self.assertLess(float(optimal_action_loss(improved, torch.tensor([5]), 'set')), float(loss.detach()))

    def test_extreme_finite_logits_and_low_precision_inputs_are_stable(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for mode in ('uniform', 'set'):
                logits = torch.tensor([[10000., -10000., -10000., 9999.]], dtype=dtype, requires_grad=True)
                loss = optimal_action_loss(logits, torch.tensor([6]), mode)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                torch.testing.assert_close(logits.grad.sum().float(), torch.tensor(0.), atol=.01, rtol=0)

    def test_action_permutations_and_constant_score_shifts_preserve_loss(self):
        logits = torch.tensor([[2., 5., -3., 1.], [-1., 0., 2., 3.]])
        masks = torch.tensor([5, 10])
        order = torch.tensor([2, 0, 3, 1])
        bits = (masks[:, None] & (1 << torch.arange(4))) != 0
        permuted = (bits[:, order].long() * (1 << torch.arange(4))).sum(-1)
        for mode in ('uniform', 'set'):
            expected = optimal_action_loss(logits, masks, mode)
            torch.testing.assert_close(optimal_action_loss(logits[:, order], permuted, mode), expected)
            torch.testing.assert_close(optimal_action_loss(logits + 100, masks, mode), expected)

    def test_invalid_contracts_are_rejected(self):
        valid_scores, valid_masks = torch.zeros(2, 4), torch.tensor([1, 3])
        for logits in (torch.zeros(2, 3), torch.zeros(0, 4), torch.zeros(2, 1, 4),
                       torch.zeros(2, 4, dtype=torch.long), torch.full((2, 4), float('nan')),
                       torch.full((2, 4), float('inf')), torch.full((2, 4), 1e100, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                optimal_action_loss(logits, valid_masks)
        for optimal in (torch.tensor([1, -1]), torch.tensor([1, 16]), torch.zeros(2, 1, dtype=torch.long),
                        torch.ones(2), torch.ones(2, dtype=torch.bool), torch.tensor([1]),
                        torch.empty(2, dtype=torch.long, device='meta')):
            with self.assertRaises(ValueError):
                optimal_action_loss(valid_scores, optimal)
        for mode in ('other', None, True):
            with self.assertRaises(ValueError):
                optimal_action_loss(valid_scores, valid_masks, mode)


if __name__ == '__main__':
    unittest.main()
