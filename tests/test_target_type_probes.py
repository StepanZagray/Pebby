"""Synthetic route-label and query-probe mechanics, CPU only."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from tools import target_type_probes as p


class TargetTypeProbeTests(unittest.TestCase):
    def test_effects_are_explicit_and_simultaneous(self):
        before = ((0, 0), 0, 0, 0, 0, 0, 20, 0)
        after = ((0, 1), 1, 2, 3, 1, 1, 19, 1)
        self.assertEqual(p.effect_types(before, after, 'launched'), set(p.TYPES))
        self.assertEqual(p.effect_types(before, before, 'moved'), set())

    def test_labels_union_optimal_prefixes_and_stop_at_first_event(self):
        start = ((0, 0), 0, 0, 0, 0, 0, 20, 0)
        plain = ((0, 1), 0, 0, 0, 0, 0, 19, 0)
        shape = ((0, 2), 1, 0, 0, 0, 0, 18, 0)
        refill = ((1, 0), 0, 0, 0, 0, 1, 20, 0)
        goal = ((0, 3), 1, 0, 0, 1, 0, 17, 0)
        graph = {(start, 0): plain, (plain, 0): shape, (plain, 1): refill, (shape, 0): goal}
        distances = {start: 3, plain: 2, shape: 1, refill: 1, goal: 0}
        oracle = SimpleNamespace(start=start, layout=None, refills=(), truncated=False, solvable=True,
                                 distance_for=distances.get)
        def advance(layout, state, action, refills):
            return graph.get((state, action))
        def simulate(layout, state, action, refills):
            return graph[(state, action)], 'moved'
        with patch.object(p, 'advance', side_effect=advance), patch.object(p, 'simulate', side_effect=simulate):
            labels, seen, _ = p.first_types(oracle)
        self.assertEqual({name for name, yes in zip(p.TYPES, labels) if yes}, {'shape', 'refill'})
        self.assertEqual(seen, 2)
        oracle.truncated = True
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            p.first_types(oracle)

    def test_query_shapes_gradient_and_distinct_batch_guard(self):
        prior = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            features = torch.randn(8, 9, 16)
            glyph = torch.randn(8, 14)
            probe = p.QueryProbe(16, True)
            scores = probe(features, glyph)
            self.assertEqual(scores.shape, (8, len(p.TYPES)))
            scores.square().mean().backward()
            self.assertGreater(probe.query.weight.grad.abs().sum(), 0)
            labels = torch.zeros(8, len(p.TYPES))
            labels[:, 0] = 1
            with self.assertRaisesRegex(ValueError, 'distinct'):
                p.fit(features, features, glyph, glyph, labels, labels, 1, 16, 0)
            result = p.fit(features, features, glyph, glyph, labels, labels, 2, 4, 0,
                           torch.arange(8) < 4, torch.arange(8) < 4)
            self.assertEqual(result['validation']['by_visibility']['fog']['count'], 4)
        finally:
            torch.set_num_threads(prior)


if __name__ == '__main__':
    unittest.main()
