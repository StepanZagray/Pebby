"""CPU resource metric tests, without checkpoints or game execution."""
import unittest

import torch

from tools.audit_spatial_route_resources import COHORTS, ResourceMetrics, resource_targets


def fixture():
    return dict(current_steps=torch.tensor([4, 4]), next_steps=torch.tensor([[3, 4, 8, -1], [42, 3, 3, 3]]),
                next_lives=torch.tensor([[3, 3, 3, 2], [3, 3, 3, 3]]),
                lost_life=torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]]),
                terminal=torch.tensor([[0, 0, 0, 1], [1, 1, 0, 0]]),
                won=torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]]))


def predictions(steps, lives):
    step_logits = torch.full((*steps.shape, 44), -10.)
    step_logits.scatter_(-1, (steps + 1)[..., None], 10.)
    lives_logits = torch.full((*lives.shape, 4), -10.)
    lives_logits.scatter_(-1, lives[..., None], 10.)
    return dict(field_logits=(None, None, None, None, step_logits, lives_logits),
                event_logits=torch.full((*steps.shape, 3), -1.))


class ResourceAuditTests(unittest.TestCase):
    def test_cohorts_are_disjoint_with_explicit_death_win_terminal_precedence(self):
        items = fixture()
        _, _, masks = resource_targets(items)
        self.assertEqual(set(masks), set(COHORTS))
        self.assertTrue((torch.stack(list(masks.values())).sum(0) == 1).all())
        self.assertTrue(masks['life_loss'][0, 3])  # Triple event overlap belongs only to death cohort.
        self.assertFalse(masks['win_without_life_loss'][0, 3])
        self.assertTrue(masks['win_without_life_loss'][1, 0])
        self.assertTrue(masks['terminal_without_life_loss_or_win'][1, 1])
        self.assertEqual([int(masks[name].sum()) for name in COHORTS], [3, 1, 1, 1, 1, 1])

    def test_mae_uses_real_budget_span_and_empty_support_is_null(self):
        items = {key: value[:1].clone() for key, value in fixture().items()}
        items['next_steps'][0, 3] = -7  # Same category as -1 in training.
        predicted_steps = torch.tensor([[2, 6, 8, 42]])
        result = ResourceMetrics()
        result.update(predictions(predicted_steps, items['next_lives']), items)
        report = result.result()
        self.assertEqual(report['cohorts']['all']['budget_absolute_error'], 46)  # 1 + 2 + 0 + 43.
        self.assertEqual(report['cohorts']['all']['budget_mae'], 11.5)
        self.assertEqual(report['cohorts']['all']['budget_accuracy'], .25)
        self.assertEqual(report['cohorts']['all']['budget_within_one_accuracy'], .5)
        self.assertEqual(report['negative_budget_labels_clamped'], 1)
        self.assertIsNone(report['cohorts']['win_without_life_loss']['budget_accuracy'])
        self.assertIsNone(report['events']['won']['precision'])
        self.assertEqual(report['events']['won']['recall'], 0.)

    def test_batch_partition_does_not_change_metrics_or_overlapping_event_supports(self):
        items = fixture()
        prediction = predictions(items['next_steps'], items['next_lives'])
        full, divided = ResourceMetrics(), ResourceMetrics()
        full.update(prediction, items)
        for row in range(2):
            subset = {key: value[row:row + 1] for key, value in items.items()}
            divided.update(predictions(subset['next_steps'], subset['next_lives']), subset)
        self.assertEqual(full.result(), divided.result())
        self.assertEqual(full.result()['events']['terminal']['positive_support'], 3)
        self.assertEqual(full.result()['events']['won']['positive_support'], 2)

    def test_invalid_overflow_and_nonfinite_logits_fail(self):
        items = fixture()
        items['next_steps'][0, 0] = 43
        with self.assertRaisesRegex(ValueError, 'overflow'):
            resource_targets(items)
        items = fixture()
        prediction = predictions(items['next_steps'], items['next_lives'])
        prediction['event_logits'][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            ResourceMetrics().update(prediction, items)


if __name__ == '__main__':
    unittest.main()
