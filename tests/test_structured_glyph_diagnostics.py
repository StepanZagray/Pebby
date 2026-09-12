import unittest

import torch
from torch.nn import functional as F

from tools.structured_glyph_diagnostics import GlyphDiagnosticAccumulator, balanced_glyph_loss


def readout(batch, predictions):
    values = []
    for classes, column in ((6, 0), (4, 1), (4, 2)):
        logits = torch.zeros(batch, classes)
        for row, prediction in enumerate(predictions[:, column].tolist()):
            logits[row, prediction] = 3.
        values.append(logits)
    return dict(zip(("carried_shape_logits", "carried_color_logits", "carried_rotation_logits"), values))


class StructuredGlyphDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_balanced_loss_matches_equal_group_means(self):
        current = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 2], [2, 2, 3]])
        target = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 2, 2], [2, 2, 0]])
        generator = torch.Generator().manual_seed(12)
        logits = {
            "carried_shape_logits": torch.randn(4, 6, generator=generator, requires_grad=True),
            "carried_color_logits": torch.randn(4, 4, generator=generator, requires_grad=True),
            "carried_rotation_logits": torch.randn(4, 4, generator=generator, requires_grad=True),
        }
        actual = balanced_glyph_loss(logits, current, target)
        expected = []
        for column, key in enumerate(("carried_shape_logits", "carried_color_logits", "carried_rotation_logits")):
            row_loss = F.cross_entropy(logits[key], target[:, column], reduction="none")
            changed = target[:, column] != current[:, column]
            expected.append(torch.stack((row_loss[changed].mean(), row_loss[~changed].mean())).mean())
        torch.testing.assert_close(actual, torch.stack(expected).mean())
        actual.backward()
        self.assertTrue(all(value.grad is not None for value in logits.values()))

    def test_balanced_loss_uses_present_group_when_one_group_empty(self):
        current = torch.tensor([[0, 1, 2], [1, 2, 3]])
        target = current.clone()
        logits = {
            "carried_shape_logits": torch.randn(2, 6, requires_grad=True),
            "carried_color_logits": torch.randn(2, 4, requires_grad=True),
            "carried_rotation_logits": torch.randn(2, 4, requires_grad=True),
        }
        expected = torch.stack([
            F.cross_entropy(logits["carried_shape_logits"], target[:, 0]),
            F.cross_entropy(logits["carried_color_logits"], target[:, 1]),
            F.cross_entropy(logits["carried_rotation_logits"], target[:, 2]),
        ]).mean()
        torch.testing.assert_close(balanced_glyph_loss(logits, current, target), expected)

    def test_loss_rejects_invalid_labels_and_does_not_mutate_targets(self):
        logits = {
            "carried_shape_logits": torch.zeros(2, 6),
            "carried_color_logits": torch.zeros(2, 4),
            "carried_rotation_logits": torch.zeros(2, 4),
        }
        current = torch.tensor([[0, 1, 2], [1, 2, 3]])
        target = torch.tensor([[0, 1, 2], [1, 2, 3]])
        before = (current.clone(), target.clone())
        with self.assertRaises(ValueError):
            balanced_glyph_loss(logits, current, torch.tensor([[6, 0, 0], [0, 0, 0]]))
        with self.assertRaises(ValueError):
            balanced_glyph_loss(logits, current.float(), target)
        with self.assertRaises(ValueError):
            balanced_glyph_loss(logits, current, target[:, :2])
        with self.assertRaises(ValueError):
            balanced_glyph_loss(logits, torch.zeros(2, 4, dtype=torch.long),
                                torch.zeros(2, 4, dtype=torch.long))
        self.assertTrue(torch.equal(current, before[0]))
        self.assertTrue(torch.equal(target, before[1]))

    def test_accumulator_is_additive_and_reports_persistence(self):
        current = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        target = torch.tensor([[1, 0, 0], [1, 1, 1], [2, 3, 2], [3, 0, 3]])
        predictions = torch.tensor([[1, 0, 0], [1, 1, 1], [2, 3, 2], [3, 3, 3]])
        fields = torch.zeros(4, 148, 96)
        # action target row-major cells: row0 direct shape, row1 launcher,
        # row2 out of bounds, row3 direct color.
        fields[0, 1, 48 + 2] = 1.
        fields[1, 13, 48 + 5] = 1.
        fields[3, 38, 48 + 3] = 1.
        player = torch.tensor([[0, 0], [1, 1], [0, 0], [2, 2]])
        actions = torch.tensor([3, 0, 0, 1])
        one = GlyphDiagnosticAccumulator().update(readout(4, predictions), current, target,
                                                  current_fields=fields, player_cell=player, actions=actions)
        split = GlyphDiagnosticAccumulator()
        split.update(readout(2, predictions[:2]), current[:2], target[:2],
                     current_fields=fields[:2], player_cell=player[:2], actions=actions[:2])
        split.update(readout(2, predictions[2:]), current[2:], target[2:],
                     current_fields=fields[2:], player_cell=player[2:], actions=actions[2:])
        self.assertEqual(one.summary(), split.summary())
        result = one.summary()
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["attributes"]["shape"]["changed"]["predicted"]["count"], 1)
        self.assertEqual(result["attributes"]["shape"]["changed"]["persistence"]["correct_sum"], 0)
        self.assertEqual(result["role_partition"]["direct_cycler"]["shape"]["changed"]["predicted"]["count"], 1)
        self.assertEqual(result["role_partition"]["launcher"]["shape"]["changed"]["predicted"]["count"], 0)
        self.assertEqual(result["role_partition"]["other"]["shape"]["changed"]["predicted"]["count"], 0)

    def test_role_partition_is_public_argmax_and_validates_grouped_inputs(self):
        current = torch.tensor([[0, 0, 0]])
        target = torch.tensor([[1, 0, 0]])
        fields = torch.zeros(1, 148, 96)
        fields[0, 1, 48 + 5] = .9
        acc = GlyphDiagnosticAccumulator()
        acc.update(readout(1, target), current, target, current_fields=fields,
                   player_cell=torch.tensor([[0, 0]]), actions=torch.tensor([3]))
        result = acc.summary()
        self.assertEqual(result["role_partition"]["launcher"]["shape"]["changed"]["predicted"]["count"], 1)
        self.assertIn("not engine ground truth", result["role_partition_definition"])
        with self.assertRaises(ValueError):
            acc.update(readout(1, target), current, target, current_fields=fields,
                       player_cell=torch.tensor([[0, 0]]))


if __name__ == "__main__":
    unittest.main()
