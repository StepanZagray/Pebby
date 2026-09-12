import copy
import unittest

import numpy as np
import torch

from pebby.agent.structured_global_glyph import GlobalGlyphTransition
from pebby.agent.structured_sequence_objective import sequence_objective
from tools.train_structured_factored_sequences import (
    GLOBAL_FORMAT, ObjectiveView, _check_initial_model_sources, _model_for_checkpoint, _direct_readout,
    h4_glyph_auxiliary_losses,
)
from tools.train_structured_mixed_sequences import load_exploratory_cache, mixed_batch, mixed_rows
from tools.train_structured_sequences import load_cache
from tools.structured_glyph_diagnostics import balanced_glyph_loss, LOGIT_KEYS


def field_fixture(batch=2, seed=51):
    generator = torch.Generator().manual_seed(seed)
    field = torch.randn(batch, 148, 96, generator=generator)
    field[:, :144, 48:56] = torch.rand(batch, 144, 8, generator=generator)
    for start, stop in ((56, 62), (62, 66), (66, 70), (70, 76), (76, 80), (80, 84)):
        field[:, :, start:stop] = torch.rand(batch, 148, stop - start,
                                             generator=generator).softmax(-1)
    field[:, :144, 84] = torch.rand(batch, 144, generator=generator)
    field[:, 144:, 48:70] = 0
    field[:, 144:, 84] = 1
    field[..., 85:96] = 0
    next_fields = torch.stack([field + .01 * (step + 1) for step in range(4)], 1)
    # Keep observed target probabilities valid while changing only unobserved core/padding.
    next_fields[..., 48:85] = torch.stack([field[..., 48:85] for _ in range(4)], 1)
    labels = {
        "player_cell": torch.tensor([[1, 2], [3, 4]][:batch]),
        "triple": torch.tensor([[0, 1, 2], [3, 2, 1]][:batch]),
        "steps": torch.tensor([12, 20][:batch]), "lives": torch.tensor([3, 2][:batch]),
        "next_player_cell": torch.tensor([[[1, 2], [1, 3], [2, 3], [2, 4]],
                                            [[3, 4], [4, 4], [4, 5], [5, 5]]][:batch]),
        "next_triple": torch.tensor([[[0, 1, 2], [1, 1, 2], [1, 2, 2], [2, 2, 3]],
                                      [[3, 2, 1], [3, 2, 1], [4, 3, 1], [4, 3, 2]]][:batch]),
        "next_steps": torch.tensor([[11, 10, 9, 8], [19, 18, 17, 16]][:batch]),
        "next_lives": torch.tensor([[3, 3, 3, 3], [2, 2, 2, 2]][:batch]),
        "lost_life": torch.zeros(batch, 4, dtype=torch.long),
        "terminal": torch.zeros(batch, 4, dtype=torch.long),
        "won": torch.zeros(batch, 4, dtype=torch.long),
    }
    actions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]][:batch])
    return field, next_fields, actions, labels, torch.ones(48)


class FactoredSequenceTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_auxiliary_loss_uses_previous_actual_triple_and_has_early_gradients(self):
        fields, next_fields, actions, labels, scale = field_fixture(batch=1)
        torch.manual_seed(812)
        model = GlobalGlyphTransition(loops=1)
        result = sequence_objective(ObjectiveView(model), fields, next_fields, actions, labels, scale)
        extra = h4_glyph_auxiliary_losses(result, labels, direct_weight=1.,
                                          predicted_readout_weight=1.)
        self.assertTrue(torch.isfinite(extra["total"]))
        (result["total"] + extra["total"]).backward()
        self.assertGreater(float(model.position.grad.abs().sum()), 0.)
        self.assertGreater(float(model.glyph_head[-1].weight.grad.abs().sum()), 0.)

    def test_future_labels_change_only_auxiliary_value_not_model_rollout(self):
        fields, next_fields, actions, labels, scale = field_fixture(batch=1)
        model = GlobalGlyphTransition(loops=1).eval()
        first = sequence_objective(ObjectiveView(model), fields, next_fields, actions, labels, scale)
        altered = copy.deepcopy(labels)
        altered["next_triple"] = torch.tensor([[[5, 3, 3], [5, 3, 3], [5, 3, 3], [5, 3, 3]]])
        second = sequence_objective(ObjectiveView(model), fields, next_fields, actions, altered, scale)
        torch.testing.assert_close(first["outputs"]["predicted"]["fields"],
                                   second["outputs"]["predicted"]["fields"], atol=0, rtol=0)
        extra_first = h4_glyph_auxiliary_losses(first, labels, direct_weight=1.)
        extra_second = h4_glyph_auxiliary_losses(second, altered, direct_weight=1.)
        self.assertGreater(float((extra_first["total"] - extra_second["total"]).detach().abs()), 1e-8)

    def test_direct_probability_logits_match_log_probability_gradient(self):
        current = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        target = torch.tensor([[1, 0, 0], [1, 2, 1], [2, 2, 3]])
        generator = torch.Generator().manual_seed(73)
        logits = [torch.randn(3, count, generator=generator, requires_grad=True)
                  for count in (6, 4, 4)]
        probabilities = torch.cat([value.softmax(-1) for value in logits], -1)
        fields = torch.cat((torch.zeros(3, 4, 148, 70),
                            probabilities[:, None, None].expand(-1, 4, 148, -1),
                            torch.zeros(3, 4, 148, 12)), -1)
        actual = balanced_glyph_loss(_direct_readout(fields, 3), current, target)
        expected = balanced_glyph_loss(dict(zip(LOGIT_KEYS, logits)), current, target)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, logits)
        for observed, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(observed, reference, atol=1e-7, rtol=1e-6)

    def test_direct_loss_rejects_non_probability_field_channels(self):
        fields, next_fields, actions, labels, scale = field_fixture(batch=1)
        model = GlobalGlyphTransition(loops=1).eval()
        result = sequence_objective(ObjectiveView(model), fields, next_fields, actions, labels, scale)
        result["outputs"]["predicted"]["fields"][:, 0, 0, 70] = 1.5
        with self.assertRaisesRegex(ValueError, "outside"):
            h4_glyph_auxiliary_losses(result, labels, direct_weight=1.)

    def test_objective_view_removes_optional_direct_logits_only(self):
        model = GlobalGlyphTransition(loops=1)
        view = ObjectiveView(model)
        field = field_fixture(batch=1)[0]
        output = view(field, torch.tensor([0]))
        self.assertEqual(set(output), {"field", "readout", "events"})
        self.assertNotIn("glyph_logits", output)

    def test_verified_cache_loaders_and_mixed_batch_are_compatible(self):
        live, live_manifest = load_exploratory_cache(
            "data/structured-field-h4-exploratory-smoke8")
        closing, closing_manifest = load_cache(
            "data/structured-field-h4-closing-train-smoke8", "train")
        validation, validation_manifest = load_cache(
            "data/structured-field-h4-validation-smoke8", "validation")
        self.assertEqual(live_manifest["field_encoder"], closing_manifest["field_encoder"])
        self.assertEqual(live_manifest["field_encoder"], validation_manifest["field_encoder"])
        self.assertEqual(len(set(live["seeds"]) & set(validation["seeds"])), 0)
        rows = mixed_rows(live, closing, 8, 0.0, np.random.default_rng(17))
        batch = mixed_batch(live, closing, rows, "cpu")
        self.assertEqual(tuple(batch[0].shape), (8, 148, 96))
        self.assertEqual(tuple(batch[1].shape), (8, 4, 148, 96))
        self.assertEqual(tuple(batch[2].shape), (8, 4))
        self.assertEqual(tuple(batch[3]["next_triple"].shape), (8, 4, 3))

    def test_global_loader_accepts_exact_base_warmstart_only(self):
        saved = torch.load("checkpoints/ls20-structured-mixed-h4-800.pt",
                           map_location="cpu", weights_only=True)
        _check_initial_model_sources(saved, "global")
        model = _model_for_checkpoint(saved, "global", "cpu")
        self.assertEqual(model.checkpoint_format,
                         GLOBAL_FORMAT)
        self.assertEqual(model.parameter_count(), 211432)
        bad = dict(saved, format="unrelated.format.v1")
        with self.assertRaisesRegex(ValueError, "global arm"):
            _model_for_checkpoint(bad, "global", "cpu")

    def test_local_warmstart_preserves_global_checkpoint_config(self):
        saved = torch.load("checkpoints/ls20-structured-glyph-global300.pt",
                           map_location="cpu", weights_only=True)
        _check_initial_model_sources(saved, "local")
        model = _model_for_checkpoint(saved, "local", "cpu")
        self.assertEqual(model.config()["loops"], saved["config"]["loops"])
        self.assertEqual(model.config()["glyph_hidden"], saved["config"]["glyph_hidden"])
        self.assertEqual(model.parameter_count(), 294664)


if __name__ == "__main__":
    unittest.main()
