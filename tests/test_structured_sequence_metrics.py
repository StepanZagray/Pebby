import unittest

import torch

from tools.structured_sequence_metrics import evaluate


class MetricFake(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def readout(self, field):
        value = field[:, 0, 0].round().long().clamp(0, 143)
        output = {}
        for name, classes in (("player", 144), ("steps", 44),
                              ("carried_shape", 6), ("carried_color", 4),
                              ("carried_rotation", 4)):
            target = value if name in ("player", "steps") else torch.zeros_like(value)
            output[name + "_logits"] = torch.full((len(field), classes), -20.0, device=field.device)
            output[name + "_logits"].scatter_(1, target[:, None], 20.0)
        output["lives_logits"] = torch.full((len(field), 4), -20.0, device=field.device)
        output["lives_logits"][:, 2] = 20.0
        output["role_logits"] = torch.zeros(len(field), 144, 8, device=field.device)
        output["goal_shape_logits"] = torch.zeros(len(field), 144, 6, device=field.device)
        output["goal_color_logits"] = torch.zeros(len(field), 144, 4, device=field.device)
        output["goal_rotation_logits"] = torch.zeros(len(field), 144, 4, device=field.device)
        return output

    def rollout(self, current, actions):
        fields = current[:, None] + actions.cumsum(1)[:, :, None, None]
        return {
            "fields": fields,
            "readout": {key: value.reshape(len(current), 4, *value.shape[1:])
                         for key, value in self.readout(fields.flatten(0, 1)).items()},
            "events": {name + "_logits": torch.full((len(current), 4), -1.0, device=current.device)
                       for name in ("lost_life", "terminal", "won")},
        }


def fixture():
    base = torch.tensor([0., 1.])[:, None, None].expand(2, 148, 96).clone()
    next_fields = torch.stack([base + (step + 1) for step in range(4)], 1)
    return {
        "fields": base,
        "next_fields": next_fields,
        "actions": torch.ones(2, 4, dtype=torch.long),
        "next_player_cell": torch.stack([
            torch.stack([torch.tensor([step + 1, 0]) for step in range(4)]),
            torch.stack([torch.tensor([step + 2, 0]) for step in range(4)]),
        ]),
        "next_triple": torch.zeros(2, 4, 3, dtype=torch.long),
        "next_steps": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        "next_lives": torch.full((2, 4), 2, dtype=torch.long),
        "lost_life": torch.zeros(2, 4, dtype=torch.long),
        "terminal": torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]]),
        "won": torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]]),
    }


class StructuredSequenceMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_batch_size_invariance_and_final_event_confusions(self):
        data = fixture()
        model = MetricFake()
        one = evaluate(model, data, torch.ones(48), batch_size=1)
        two = evaluate(model, data, torch.ones(48), batch_size=2)
        self.assertEqual(one["levels"], 2)
        self.assertEqual(one["transitions"], 8)
        self.assertEqual(one["events"], two["events"])
        self.assertEqual(one["readout"], two["readout"])
        self.assertEqual(one["field_mse"], two["field_mse"])
        self.assertEqual(one["changed_appearance"], two["changed_appearance"])
        self.assertEqual(one["readout"]["predicted"]["player"]["accuracy"], [1.] * 4)
        self.assertEqual(one["readout"]["actual"]["steps"]["accuracy"], [1.] * 4)
        self.assertEqual(one["readout"]["initial_copy"]["player"]["accuracy"], [0.] * 4)
        self.assertEqual(one["events"]["terminal"]["fn"], [0, 0, 0, 1])
        self.assertEqual(one["events"]["won"]["fn"], [0, 0, 0, 1])
        self.assertEqual(one["events"]["terminal"]["positive"], [0, 0, 0, 1])
        self.assertEqual(one["field_mse"]["predicted"]["core"]["mse"], [0.] * 4)
        self.assertEqual(one["field_mse"]["initial_copy"]["core"]["mse"], [1., 4., 9., 16.])
        self.assertEqual(one["changed_appearance"]["predicted"]["mse"], [0.] * 4)

    def test_actual_readout_is_reported_separately_from_dynamics(self):
        result = evaluate(MetricFake(), fixture(), torch.ones(48), batch_size=2)
        self.assertEqual(result["readout"]["actual"]["player"]["correct_sum"], [2] * 4)
        self.assertEqual(result["readout"]["predicted"]["carried_joint"]["correct_sum"], [2] * 4)
        self.assertEqual(result["readout"]["actual"]["lives"]["cross_entropy"], [0.] * 4)
        self.assertIn("actual-target readouts", " ".join(result["notes"]))


if __name__ == "__main__":
    unittest.main()
