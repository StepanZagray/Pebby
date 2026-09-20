import unittest
import torch
from pebby.agent.planning_decision_metrics import decision_metrics


class DecisionMetricsTests(unittest.TestCase):
    def test_bad_bad_ranking_does_not_count(self):
        distances = torch.tensor([[0, 1, 2, 3]])
        losses = torch.zeros(1, 4, dtype=torch.bool)
        optimal = torch.tensor([1])
        a = decision_metrics(torch.tensor([[4., 3., 2., 1.]]), distances, losses, optimal)
        b = decision_metrics(torch.tensor([[4., 1., 2., 3.]]), distances, losses, optimal)
        self.assertEqual(a, b)
        self.assertEqual(a['optimal_boundary_pairs'], 3)

    def test_finite_regret_and_failure_counts_are_separate(self):
        scores = torch.tensor([[0., 2., 1., -1.], [0., 1., 3., -1.]])
        distances = torch.tensor([[5, 8, -1, 7], [5, 8, -1, 7]])
        lost = torch.tensor([[False, False, True, False]]).expand(2, -1)
        result = decision_metrics(scores, distances, lost, torch.tensor([1, 1]))
        self.assertEqual(result['safe_selected_roots'], 1)
        self.assertEqual(result['safe_selected_regret_sum'], 3)
        self.assertEqual(result['selected_unreachable'], 1)
        self.assertEqual(result['selected_life_loss'], 1)

    def test_tied_optima_and_unsupervised_roots(self):
        scores = torch.zeros(2, 4)
        distances = torch.tensor([[0, 0, 1, 2], [-1, -1, -1, -1]])
        result = decision_metrics(scores, distances, torch.zeros(2, 4), torch.tensor([3, 0]))
        self.assertEqual(result['supervised_roots'], 1)
        self.assertEqual(result['optimal_boundary_pairs'], 4)
        self.assertEqual(result['optimal_boundary_ties'], 4)
        with self.assertRaises(ValueError):
            decision_metrics(scores, distances, torch.zeros(2, 4), torch.tensor([1, 0]))


if __name__ == '__main__':
    unittest.main()
