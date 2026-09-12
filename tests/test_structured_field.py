"""Contract tests for the frozen public structured field encoder."""

import inspect
import unittest

import torch

from pebby.agent.structured_field import FIELD_FORMAT, StructuredFieldEncoder


class StructuredFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.encoder = StructuredFieldEncoder()

    @staticmethod
    def inputs(batch=2):
        torch.manual_seed(7301)
        frames = torch.randint(0, 16, (batch, 8, 64, 64), dtype=torch.uint8)
        valid = torch.ones(batch, 8, dtype=torch.bool)
        actions = torch.full((batch, 8), -1, dtype=torch.long)
        return frames, valid, actions

    def test_config_sources_and_frozen_eval_contract(self):
        encoder = self.encoder
        config = encoder.config()
        self.assertEqual(config["format"], FIELD_FORMAT)
        self.assertEqual(config["tokens"], 148)
        self.assertEqual(config["state_channels"], 48)
        self.assertEqual(config["world"]["history"], 8)
        self.assertEqual(config["world"]["hud_tokens"], 16)
        self.assertEqual(encoder.metadata()["sources"]["cell_appearance"]["train_levels"], 2000)
        self.assertEqual(encoder.metadata()["sources"]["cell_appearance"]["validation_levels"], 0)
        self.assertEqual(encoder.parameter_counts()["world"], 514283)
        self.assertEqual(encoder.parameter_counts()["visibility"], 112772)
        self.assertFalse(encoder.training)
        self.assertFalse(encoder.world.training)
        self.assertFalse(encoder.visibility.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertFalse(encoder.world.training)
        self.assertFalse(encoder.visibility.training)

    def test_channel_order_probabilities_and_hud_pool(self):
        frames, valid, actions = self.inputs(2)
        with torch.no_grad():
            fields = self.encoder(frames, valid, actions)
        self.assertEqual(tuple(fields.shape), (2, 148, 96))
        self.assertTrue(torch.isfinite(fields).all())
        self.assertTrue(torch.all(fields[:, :, 85:] == 0))
        self.assertTrue(torch.all(fields[:, 144:, 48:70] == 0))
        self.assertTrue(torch.all(fields[:, 144:, 84] == 1))
        # The carried glyph is global public evidence, broadcast to board/HUD.
        torch.testing.assert_close(fields[:, :1, 70:84].expand(-1, 148, -1), fields[:, :, 70:84])
        appearance = fields[:, :144, 48:70]
        self.assertTrue(bool(((appearance >= 0) & (appearance <= 1)).all()))
        for start, width in ((8, 6), (14, 4), (18, 4)):
            torch.testing.assert_close(appearance[:, :, start:start + width].sum(-1),
                                      torch.ones(2, 144), atol=2e-6, rtol=2e-6)

    def test_public_history_only_and_reset_aware_padding(self):
        signature = inspect.signature(StructuredFieldEncoder.forward)
        self.assertEqual(list(signature.parameters), ["self", "frames", "history_valid", "previous_actions"])
        frames, _, _ = self.inputs(1)
        valid = torch.tensor([[False, False, False, False, True, True, True, True]])
        actions = torch.tensor([[-1, -1, -1, -1, -1, 0, 3, 1]])
        before = frames.clone()
        result = self.encoder(frames, valid, actions)
        self.assertEqual(tuple(result.shape), (1, 148, 96))
        self.assertTrue(torch.equal(frames, before))
        with self.assertRaises(TypeError):
            self.encoder(frames, valid, actions, labels=torch.zeros(1))
        with self.assertRaises(TypeError):
            self.encoder(frames, valid, actions, visibility_mask=torch.ones(1, 144))
        with self.assertRaises(ValueError):
            self.encoder(frames[:, :4], valid[:, :4], actions[:, :4])
        with self.assertRaises(ValueError):
            self.encoder(frames, valid, torch.full_like(actions, 4))
        with self.assertRaises(ValueError):
            self.encoder(frames, torch.tensor([[True, False, True, True, True, True, True, True]]), actions)
        with self.assertRaises(ValueError):
            self.encoder(frames, valid.float(), actions)
        with self.assertRaises(ValueError):
            self.encoder(frames, valid, actions.float())

    def test_frozen_determinism_and_batch_single_tolerance(self):
        frames, valid, actions = self.inputs(2)
        with torch.enable_grad():
            first = self.encoder(frames, valid, actions)
            second = self.encoder(frames, valid, actions)
        self.assertFalse(first.requires_grad)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        for index in range(2):
            with torch.no_grad():
                single = self.encoder(frames[index:index + 1], valid[index:index + 1], actions[index:index + 1])
            torch.testing.assert_close(first[index:index + 1], single, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
