import inspect
import unittest

import torch

from pebby.agent.event_calibration import PositiveSlopePlatt, calibration_bce


class EventCalibrationTests(unittest.TestCase):
    def test_identity_initialization_and_positive_slopes(self):
        model = PositiveSlopePlatt()
        logits = torch.tensor([[-3., 0., 2.], [1., -2., .5]], requires_grad=True)
        torch.testing.assert_close(model(logits), logits)
        self.assertTrue(bool((model.slope() > 0).all()))

    def test_conditional_win_outcomes_sum_to_one(self):
        model = PositiveSlopePlatt()
        logits = torch.tensor([[-3., 0., 2.], [1., -2., .5]])
        outcomes = model.coherent_probabilities(logits)
        torch.testing.assert_close(outcomes["loss"] + outcomes["win"] + outcomes["continue"],
                                   torch.ones(2))
        torch.testing.assert_close(outcomes["win"],
                                   (1 - outcomes["loss"]) * outcomes["conditional_win"])

    def test_forward_has_no_label_or_teacher_argument(self):
        names = tuple(inspect.signature(PositiveSlopePlatt.forward).parameters)
        self.assertEqual(names, ("self", "logits"))
        with self.assertRaises(ValueError):
            PositiveSlopePlatt()(torch.zeros(2, 4))

    def test_unweighted_bce_detaches_labels_and_propagates_logits(self):
        model = PositiveSlopePlatt()
        logits = torch.zeros(4, 3, requires_grad=True)
        labels = torch.tensor([[1., 0., 0.]] * 4, requires_grad=True)
        loss = calibration_bce(model, logits, labels)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNone(labels.grad)
        self.assertTrue(bool(torch.isfinite(model.raw_slope.grad).all()))
        self.assertTrue(bool(torch.isfinite(model.intercept.grad).all()))

    def test_conditional_mask_selects_only_no_loss_win_targets(self):
        model = PositiveSlopePlatt()
        logits = torch.zeros(2, 3, requires_grad=True)
        labels = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        mask[:, 2] = labels[:, 0] == 0
        loss = calibration_bce(model, logits, labels, mask)
        self.assertAlmostEqual(float(loss.detach()), float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.zeros(4), torch.tensor([1., 0., 0., 1.]))))

    def test_label_shape_and_domain_are_checked(self):
        model = PositiveSlopePlatt()
        logits = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            calibration_bce(model, logits, torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            calibration_bce(model, logits, torch.full((2, 3), 2.))
        with self.assertRaises(ValueError):
            calibration_bce(model, logits, torch.full((2, 3), .5))
        with self.assertRaises(ValueError):
            calibration_bce(model, logits, torch.tensor([[0, 1, -1], [0, 1, 0]]))

    def test_fit_schedule_rejects_non_power_two_or_nonfinite_lr(self):
        from tools.calibrate_structured_events import fit_train_only, _assert_hashes_unchanged
        train = {"levels": 8, "logits": torch.zeros(8, 4, 3).numpy(),
                 "labels": torch.zeros(8, 4, 3).numpy()}
        with self.assertRaises(ValueError):
            fit_train_only(train, updates=1, batch_levels=3)
        with self.assertRaises(ValueError):
            fit_train_only(train, updates=1, batch_levels=4, lr=float("nan"))

        from pathlib import Path
        temporary = Path("/tmp/event-calibration-source-mutation-test")
        try:
            temporary.write_text("before")
            expected = {str(temporary): __import__("hashlib").sha256(b"before").hexdigest()}
            temporary.write_text("after")
            with self.assertRaises(ValueError):
                _assert_hashes_unchanged(expected)
        finally:
            temporary.unlink()


if __name__ == "__main__":
    unittest.main()
