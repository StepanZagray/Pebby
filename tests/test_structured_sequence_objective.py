import copy
import unittest

import torch

from pebby.agent.structured_recall_transition import StructuredRecallTransition
from pebby.agent.structured_sequence_objective import sequence_objective
from pebby.agent.structured_transition import StructuredTransition


def make_field(batch, seed):
    generator = torch.Generator().manual_seed(seed)
    field = torch.randn(batch, 148, 96, generator=generator)
    field[:, :144, 48:56] = torch.rand(batch, 144, 8, generator=generator)
    for start, stop in ((56, 62), (62, 66), (66, 70)):
        field[:, :144, start:stop] = torch.rand(batch, 144, stop - start,
                                                generator=generator).softmax(-1)
    for start, stop in ((70, 76), (76, 80), (80, 84)):
        field[:, :, start:stop] = torch.rand(batch, 148, stop - start,
                                             generator=generator).softmax(-1)
    field[:, :144, 84] = torch.rand(batch, 144, generator=generator)
    field[:, 144:, 48:70] = 0
    field[:, 144:, 84] = 1
    field[..., 85:96] = 0
    return field


def fixture(batch=2):
    fields = make_field(batch, 71)
    next_fields = torch.stack([make_field(batch, 100 + step) for step in range(4)], 1)
    actions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]][:batch], dtype=torch.long)
    labels = {
        "player_cell": torch.tensor([[1, 2], [3, 4]][:batch]),
        "triple": torch.tensor([[0, 1, 2], [3, 2, 1]][:batch]),
        "steps": torch.tensor([12, 20][:batch]),
        "lives": torch.tensor([3, 2][:batch]),
        "next_player_cell": torch.tensor([
            [[1, 2], [1, 3], [2, 3], [2, 4]],
            [[3, 4], [4, 4], [4, 5], [5, 5]],
        ][:batch]),
        "next_triple": torch.tensor([
            [[0, 1, 2], [1, 1, 2], [1, 2, 2], [2, 2, 3]],
            [[3, 2, 1], [3, 2, 1], [4, 3, 1], [4, 3, 2]],
        ][:batch]),
        "next_steps": torch.tensor([[11, 10, 9, 8], [19, 18, 17, 16]][:batch]),
        "next_lives": torch.tensor([[3, 3, 3, 3], [2, 2, 2, 2]][:batch]),
        "lost_life": torch.zeros(batch, 4, dtype=torch.long),
        "terminal": torch.zeros(batch, 4, dtype=torch.long),
        "won": torch.zeros(batch, 4, dtype=torch.long),
    }
    labels["terminal"][0, -1] = 1
    labels["won"][0, -1] = 1
    if batch > 1:
        labels["lost_life"][1, -1] = 1
    return fields, next_fields, actions, labels, torch.ones(48)


class StructuredSequenceObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_autoregressive_h4_has_finite_gradients_through_early_prediction(self):
        fields, next_fields, actions, labels, scale = fixture()
        model = StructuredTransition(loops=1)
        result = sequence_objective(model, fields, next_fields, actions, labels, scale)
        self.assertEqual(result["outputs"]["predicted"]["fields"].shape, (2, 4, 148, 96))
        self.assertEqual(set(result["losses"]), {
            "field_core", "field_appearance", "field_carried", "field_visibility",
            "field_padding", "field_changed", "readout_player", "readout_glyph",
            "readout_steps", "readout_lives", "readout_roles", "readout_goal", "events",
        })
        self.assertTrue(torch.isfinite(result["total"]))
        predicted_fields = result["outputs"]["predicted"]["fields"]
        predicted_fields.retain_grad()
        result["total"].backward()
        self.assertIsNotNone(predicted_fields.grad)
        self.assertGreater(float(predicted_fields.grad[:, 0].abs().sum()), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_future_targets_and_labels_never_enter_autoregressive_inputs(self):
        fields, next_fields, actions, labels, scale = fixture()
        model = StructuredTransition(loops=1).eval()
        first = sequence_objective(model, fields, next_fields, actions, labels, scale)
        altered_targets = next_fields.clone()
        altered_targets[:, 1:, :, 48:70] = 1 - altered_targets[:, 1:, :, 48:70]
        altered_labels = copy.deepcopy(labels)
        altered_labels["next_player_cell"] = torch.zeros_like(labels["next_player_cell"])
        altered_labels["next_steps"] = torch.full_like(labels["next_steps"], 42)
        second = sequence_objective(model, fields, altered_targets, actions, altered_labels, scale)
        torch.testing.assert_close(first["outputs"]["predicted"]["fields"],
                                   second["outputs"]["predicted"]["fields"], atol=0, rtol=0)
        torch.testing.assert_close(first["outputs"]["predicted"]["events"]["won_logits"],
                                   second["outputs"]["predicted"]["events"]["won_logits"],
                                   atol=0, rtol=0)

    def test_rollout_composition_matches_model_rollout_and_actual_readouts_are_separate(self):
        fields, next_fields, actions, labels, scale = fixture()
        model = StructuredTransition(loops=1).eval()
        result = sequence_objective(model, fields, next_fields, actions, labels, scale)
        expected = model.rollout(fields, actions)
        torch.testing.assert_close(result["outputs"]["predicted"]["fields"], expected["fields"],
                                   atol=0, rtol=0)
        for name in expected["readout"]:
            torch.testing.assert_close(result["outputs"]["predicted"]["readout"][name],
                                       expected["readout"][name], atol=0, rtol=0)
        self.assertEqual(result["outputs"]["actual"]["readout"]["player_logits"].shape,
                         (2, 4, 144))
        self.assertEqual(result["outputs"]["current"]["readout"]["player_logits"].shape,
                         (2, 144))
        self.assertIsNot(result["outputs"]["actual"]["readout"],
                         result["outputs"]["predicted"]["readout"])

    def test_checkpointed_steps_match_loss_and_gradients(self):
        fields, next_fields, actions, labels, scale = fixture(batch=1)
        torch.manual_seed(918)
        ordinary = StructuredTransition(loops=1)
        checkpointed = StructuredTransition(loops=1)
        checkpointed.load_state_dict(ordinary.state_dict())
        plain = sequence_objective(ordinary, fields, next_fields, actions, labels, scale,
                                   checkpoint_steps=False)
        checked = sequence_objective(checkpointed, fields, next_fields, actions, labels, scale,
                                     checkpoint_steps=True)
        torch.testing.assert_close(plain["total"], checked["total"], atol=1e-6, rtol=1e-6)
        plain["total"].backward()
        checked["total"].backward()
        for name, parameter in ordinary.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(checkpointed.named_parameters())[name].grad,
                                       atol=1e-5, rtol=1e-5, msg=name)

    def test_terminal_lost_life_and_win_are_allowed_only_at_final_horizon(self):
        fields, next_fields, actions, labels, scale = fixture(batch=1)
        model = StructuredTransition(loops=1)
        sequence_objective(model, fields, next_fields, actions, labels, scale)
        for name in ("lost_life", "terminal", "won"):
            changed = copy.deepcopy(labels)
            changed[name][0, 1] = 1
            if name == "won":
                changed["terminal"][0, 1] = 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                sequence_objective(model, fields, next_fields, actions, changed, scale)
        changed = copy.deepcopy(labels)
        changed["won"][0, -1] = 1
        changed["terminal"][0, -1] = 0
        with self.assertRaises(ValueError):
            sequence_objective(model, fields, next_fields, actions, changed, scale)

    def test_fixed_initial_memory_is_passed_to_every_step(self):
        fields, next_fields, actions, labels, scale = fixture(batch=1)
        memory = make_field(1, 442)
        model = StructuredRecallTransition(loops=1).eval()
        with torch.no_grad():
            model.memory_output.weight.copy_(torch.eye(96) * .1)
        result = sequence_objective(model, fields, next_fields, actions, labels, scale,
                                    initial_field=memory)
        changed_memory = memory.clone()
        changed_memory[..., :48] += 1
        changed = sequence_objective(model, fields, next_fields, actions, labels, scale,
                                     initial_field=changed_memory)
        self.assertGreater(float((result["outputs"]["predicted"]["fields"]
                                  - changed["outputs"]["predicted"]["fields"]).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
