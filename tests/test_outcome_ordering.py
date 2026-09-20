"""Tiny CPU checks for opt-in comparator ordering; no checkpoint or data access."""
import math
import unittest

import torch
from torch.nn import functional as F

from pebby.agent.outcome_ordering import (masked_optimal_set_cross_entropy,
                                        pairwise_safe_ordering_loss)


class OutcomeOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_unsafe_and_unreachable_branches_never_form_pairs(self):
        scores = torch.tensor([[0., 1., 100., -100.]], requires_grad=True)
        loss = pairwise_safe_ordering_loss(scores, torch.tensor([[2, 5, 0, -1]]),
                                          torch.tensor([[0, 0, 1, 0]]), torch.tensor([1]))
        torch.testing.assert_close(loss, F.softplus(torch.tensor(math.log(4) + 1)))
        loss.backward()
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertGreater(scores.grad[0, 1].item(), 0)
        torch.testing.assert_close(scores.grad[0, 2:], torch.zeros(2))

    def test_ties_have_no_constraint_and_all_strict_pairs_are_used(self):
        scores = torch.tensor([[0., 2., 1., 4.]], requires_grad=True)
        # Two tied shortest routes: five strict pairs, not just optimal-vs-rest.
        distance = torch.tensor([[1, 1, 3, 4]])
        loss = pairwise_safe_ordering_loss(scores, distance, torch.zeros(1, 4),
                                          torch.tensor([3]), margin_scale=0)
        expected = torch.stack([F.softplus(scores[0, j] - scores[0, i])
                                for i, j in [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]).mean()
        torch.testing.assert_close(loss, expected)

    def test_roots_are_equally_weighted_and_zero_optimal_is_excluded(self):
        scores = torch.tensor([[0., 3., 8., 9.], [4., 2., 1., -2.], [40., -40., 0., 0.]])
        distance = torch.tensor([[1, 2, -1, -1], [1, 2, 3, 4], [1, 2, 3, 4]])
        lost = torch.zeros(3, 4)
        optimal = torch.tensor([1, 1, 0])
        combined = pairwise_safe_ordering_loss(scores, distance, lost, optimal)
        separate = torch.stack([pairwise_safe_ordering_loss(scores[i:i+1], distance[i:i+1],
                                                            lost[i:i+1], optimal[i:i+1])
                                for i in (0, 1)]).mean()
        torch.testing.assert_close(combined, separate)

    def test_action_permutation_preserves_loss_and_permutates_gradients(self):
        scores = torch.tensor([[.3, -.7, .2, .9]], requires_grad=True)
        distance = torch.tensor([[3, 1, 1, 5]])
        optimal = torch.tensor([6])
        lost = torch.zeros(1, 4)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = scores.detach()[:, permutation].requires_grad_()
        permuted_optimal = torch.tensor([9])
        for function in ('ordering', 'ce'):
            if function == 'ordering':
                before = pairwise_safe_ordering_loss(scores, distance, lost, optimal)
                after = pairwise_safe_ordering_loss(permuted, distance[:, permutation], lost[:, permutation], permuted_optimal)
            else:
                before = masked_optimal_set_cross_entropy(scores, optimal)
                after = masked_optimal_set_cross_entropy(permuted, permuted_optimal)
            torch.testing.assert_close(before, after)
            gradient = torch.autograd.grad(before, scores)[0]
            permuted_gradient = torch.autograd.grad(after, permuted)[0]
            torch.testing.assert_close(gradient[:, permutation], permuted_gradient)

    def test_no_pair_batches_are_finite_and_differentiable(self):
        for distance, lost, optimal in [([1, 2, 3, 4], [0, 0, 0, 0], 0),
                                        ([-1, -1, -1, -1], [0, 0, 0, 0], 0),
                                        ([1, 2, 3, 4], [1, 1, 1, 1], 0),
                                        ([2, 2, 2, 2], [0, 0, 0, 0], 15),
                                        ([2, -1, -1, -1], [0, 0, 0, 0], 1)]:
            with self.subTest(distance=distance, optimal=optimal):
                scores = torch.randn(1, 4, requires_grad=True)
                loss = pairwise_safe_ordering_loss(scores, torch.tensor([distance]),
                                                  torch.tensor([lost]), torch.tensor([optimal]))
                self.assertEqual(loss.item(), 0)
                loss.backward()
                torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_cross_entropy_uniform_ties_valid_root_denominator_and_zero_gradient(self):
        scores = torch.tensor([[1., 2., 3., 4.], [40., -40., 0., 0.]], requires_grad=True)
        loss = masked_optimal_set_cross_entropy(scores, torch.tensor([3, 0]))
        torch.testing.assert_close(loss, -scores[0].log_softmax(-1)[:2].mean())
        loss.backward()
        torch.testing.assert_close(scores.grad[1], torch.zeros(4))
        scores.grad = None
        empty = masked_optimal_set_cross_entropy(scores, torch.zeros(2, dtype=torch.long))
        self.assertEqual(empty.item(), 0)
        empty.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_incomplete_or_inconsistent_nonzero_optimal_masks_are_rejected(self):
        for optimal in (1, 4, 15):
            with self.subTest(optimal=optimal), self.assertRaisesRegex(ValueError, 'optimal set'):
                pairwise_safe_ordering_loss(torch.zeros(1, 4), torch.tensor([[1, 1, 3, -1]]),
                                           torch.zeros(1, 4), torch.tensor([optimal]))

    def test_invalid_contracts_are_rejected(self):
        args = [torch.zeros(1, 4), torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 4), torch.tensor([1])]
        for index, bad in [(0, torch.zeros(1, 3)), (0, torch.zeros(1, 4, dtype=torch.long)),
                           (0, torch.full((1, 4), float('nan'))), (0, torch.full((1, 4), float('inf'))),
                           (1, torch.ones(1, 4)), (2, torch.full((1, 4), 2)),
                           (3, torch.tensor([16])), (3, torch.tensor([1.]))]:
            altered = list(args)
            altered[index] = bad
            with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                pairwise_safe_ordering_loss(*altered)
        for margin in (-1., float('nan'), float('inf'), True):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                pairwise_safe_ordering_loss(*args, margin_scale=margin)

    def test_tiny_optimization_reverses_reproduced_route_inversion(self):
        # Scalar scores/distances from the frozen checkpoint audit's equal-field
        # example. This tests loss direction, not a trained comparator or gameplay.
        scores = torch.nn.Parameter(torch.tensor([[2.974335, 2.974335, 2.452938, 3.246965]]))
        distance = torch.tensor([[19, 19, 18, 20]])
        optimizer = torch.optim.SGD([scores], lr=.2)
        self.assertEqual(scores.argmax(-1).item(), 3)
        for _ in range(40):
            optimizer.zero_grad()
            loss = pairwise_safe_ordering_loss(scores, distance, torch.zeros(1, 4), torch.tensor([4]))
            loss.backward()
            optimizer.step()
        self.assertEqual(scores.argmax(-1).item(), 2)
        self.assertGreater(scores[0, 0].item(), scores[0, 3].item())
        self.assertGreater(scores[0, 1].item(), scores[0, 3].item())


if __name__ == '__main__':
    unittest.main()
