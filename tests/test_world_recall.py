"""State-recall path of the LS20 world policy (``WorldModelConfig.state_recall``).

Covers: the default stays off and old checkpoints without the key still load
and compute the same logits; an enabled network round-trips through the
checkpoint format; a plain checkpoint migrates into an enabled network with
zero-padded projector columns so latents and logits are unchanged on padded
multi-frame histories; reverse and unrelated migrations fail closed; the
recall inputs are exactly the raw current HUD tokens and the player-weighted
refined cells; gradients reach the new columns and both recall inputs; the
encoder chunk / checkpoint paths keep outputs and gradients; and the trainer
flag records and migrates. Synthetic mazes only, CPU only, no official level.
"""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import world_model as wm
from pebby.agent import world_train as trainer
from pebby.agent.world_model import ACTION_COUNT, CELLS, WorldModelConfig, world_losses
from tests.test_world_model import TINY, make_model, make_synthetic, to_batch

CHANNELS, HUD_TOKENS = TINY["channels"], TINY["hud_tokens"]
HUD_INPUTS = HUD_TOKENS * CHANNELS


def run(model, batch, **kwargs):
    return model(batch["frames"], batch["history_valid"], batch["previous_actions"], **kwargs)


def encode(model, batch, **kwargs):
    return model.encode(batch["frames"], batch["history_valid"], batch["previous_actions"], **kwargs)


class StateRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.data = make_synthetic(seed=31, levels=2, steps=6, history=4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def setUp(self):
        torch.manual_seed(5)

    def padded_batch(self, count=4):
        """The first states of a level: histories with 3, 2, 1 and 0 padded slots."""
        batch = to_batch(self.data, slice(0, count))
        self.assertFalse(bool(batch["history_valid"].all()), "the slice must contain padded history")
        self.assertTrue(bool(batch["history_valid"][:, -1].all()))
        return batch

    # ------------------------------------------------------------ compatibility
    def test_default_is_off_and_old_checkpoints_without_the_key_still_load(self):
        config = WorldModelConfig(**TINY)
        self.assertFalse(config.state_recall)
        without_key = {key: value for key, value in config.as_dict().items() if key != "state_recall"}
        self.assertFalse(WorldModelConfig.from_dict(without_key).state_recall)
        self.assertEqual(WorldModelConfig.from_dict(without_key), config)
        with self.assertRaisesRegex(ValueError, "state_recall must be boolean"):
            WorldModelConfig(**TINY, state_recall=1)
        model = make_model().eval()
        self.assertEqual(model.recall_inputs, 0)
        self.assertEqual(model.projector[0].in_features, model.tokens * TINY["reduce"])
        # Recall widens one existing tensor; it adds no parameter names.
        self.assertEqual(set(make_model(state_recall=True).state_dict()), set(model.state_dict()))
        batch = self.padded_batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            saved = wm.save_world_checkpoint(path, model)
            saved["config"].pop("state_recall")
            torch.save(saved, path)
            restored, loaded = wm.load_world_checkpoint(path)
        self.assertFalse(restored.cfg.state_recall)
        self.assertNotIn("state_recall", loaded["config"])
        with torch.inference_mode():
            self.assertTrue(torch.equal(run(model, batch), run(restored, batch)))

    def test_recall_enabled_round_trip_and_strict_shape_rejections(self):
        model = make_model(state_recall=True).eval()
        self.assertEqual(model.recall_inputs, HUD_INPUTS + CHANNELS)
        self.assertEqual(model.projector[0].in_features, model.base_inputs + model.recall_inputs)
        batch = self.padded_batch()
        with torch.inference_mode():
            logits = run(model, batch)
            single = model(batch["frames"][:, -1])  # history of one frame
        self.assertEqual(tuple(logits.shape), (4, ACTION_COUNT))
        self.assertEqual(tuple(single.shape), (4, ACTION_COUNT))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recall.pt"
            saved = wm.save_world_checkpoint(path, model)
            self.assertTrue(saved["config"]["state_recall"])
            restored, loaded = wm.load_world_checkpoint(path)
            self.assertTrue(restored.cfg.state_recall)
            self.assertEqual(loaded["parameters"], model.parameter_count())
            self.assertEqual(loaded["config"], model.config())
            with torch.inference_mode():
                self.assertTrue(torch.equal(logits, run(restored, batch)))
            # Dropping the flag from the config must not silently load the wider weights.
            saved["config"]["state_recall"] = False
            torch.save(saved, path)
            with self.assertRaises((ValueError, RuntimeError)):
                wm.load_world_checkpoint(path)
        # Strict loads across the two shapes fail in both directions.
        with self.assertRaises(RuntimeError):
            make_model().load_state_dict(model.state_dict())
        with self.assertRaises(RuntimeError):
            model.load_state_dict(make_model().state_dict())

    # ---------------------------------------------------------------- migration
    def test_migration_from_plain_checkpoint_preserves_latents_and_logits(self):
        source = make_model().eval()
        with torch.no_grad():  # wake the AdaLN-zero gates so the ranker path is not trivial
            for block in source.predictor.blocks:
                torch.nn.init.normal_(block.modulation[-1].weight, std=.5)
        target = make_model(state_recall=True).eval()
        base = target.base_inputs
        self.assertGreater(target.projector[0].weight.detach()[:, base:].abs().sum().item(), 0.,
                           "a fresh recall network uses the usual random initialization")
        self.assertEqual(wm.initialize_from_checkpoint(target, source), ["projector.0.weight"])
        weight = target.projector[0].weight.detach()
        self.assertTrue(torch.equal(weight[:, :base], source.projector[0].weight.detach()))
        self.assertEqual(weight[:, base:].abs().sum().item(), 0., "only the new columns are zeroed")
        target_parameters = dict(target.named_parameters())
        for name, parameter in source.named_parameters():
            if name != "projector.0.weight":
                self.assertTrue(torch.equal(parameter.detach(), target_parameters[name].detach()), name)
        batch = self.padded_batch()
        with torch.inference_mode():
            for loops in (1, 3):
                with self.subTest(loops=loops):
                    before, after = encode(source, batch, loops=loops), encode(target, batch, loops=loops)
                    self.assertTrue(torch.equal(after["state"], before["state"]))
                    torch.testing.assert_close(after["latent"], before["latent"], atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(target.logits_from(after)[0], source.logits_from(before)[0],
                                               atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(run(target, batch), run(source, batch), atol=1e-6, rtol=1e-6)
        with torch.no_grad():
            plain_out = world_losses(source, batch, sigreg_generator=torch.Generator().manual_seed(1))
            recall_out = world_losses(target, batch, sigreg_generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(recall_out["targets"], plain_out["targets"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(recall_out["total"], plain_out["total"], atol=1e-5, rtol=1e-5)

    def test_migration_rejects_reverse_and_unrelated_configs_but_allows_grounding(self):
        plain, recall = make_model(), make_model(state_recall=True)
        with self.assertRaisesRegex(ValueError, "without state_recall"):
            wm.initialize_from_checkpoint(make_model(), recall)
        for override in (dict(latent=32), dict(hud_tokens=8), dict(channels=32, heads=4), dict(loops=3)):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "differs beyond"):
                wm.initialize_from_checkpoint(make_model(state_recall=True, **override), plain)
        with self.assertRaisesRegex(ValueError, "without grounding"):
            wm.initialize_from_checkpoint(make_model(state_recall=True), make_model(grounding=True))
        # Identical configs copy every tensor verbatim and migrate nothing.
        same = make_model(state_recall=True)
        self.assertEqual(wm.initialize_from_checkpoint(same, recall), [])
        for (name, mine), (_, theirs) in zip(same.state_dict().items(), recall.state_dict().items()):
            self.assertTrue(torch.equal(mine, theirs), name)
        # Grounding and recall may turn on together; the latent is still the source's.
        both = make_model(state_recall=True, grounding=True).eval()
        self.assertEqual(wm.initialize_from_checkpoint(both, plain), ["projector.0.weight"])
        batch = self.padded_batch()
        with torch.inference_mode():
            torch.testing.assert_close(encode(both, batch)["latent"], encode(plain.eval(), batch)["latent"],
                                       atol=1e-6, rtol=1e-6)
        # Foreign or missing tensors are refused rather than partially loaded.
        foreign = copy.deepcopy(plain)
        foreign.extra = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            wm.initialize_from_checkpoint(make_model(state_recall=True), foreign)
        truncated = copy.deepcopy(plain)
        del truncated.player_head
        with self.assertRaisesRegex(ValueError, "missing"):
            wm.initialize_from_checkpoint(make_model(state_recall=True), truncated)

    # ------------------------------------------------------------ recall inputs
    def test_recall_inputs_are_raw_current_hud_and_player_weighted_cells_only(self):
        model = make_model(state_recall=True).eval()
        base = model.base_inputs
        batch = self.padded_batch()
        captured = []
        handle = model.projector[0].register_forward_hook(lambda module, args, output: captured.append(args[0]))
        with torch.no_grad():
            encoding = encode(model, batch)
            handle.remove()
            self.assertEqual(len(captured), 1)
            inputs = captured[0]
            self.assertEqual(tuple(inputs.shape), (4, base + HUD_INPUTS + CHANNELS))
            # Base inputs first, untouched by recall.
            torch.testing.assert_close(inputs[:, :base], model.reduce(encoding["state"]).flatten(1),
                                       atol=1e-6, rtol=1e-6)
            # Then the current frame's HUD tokens exactly as they leave the stem:
            # no age or action embedding, no temporal memory, no refinement.
            raw = model.frame_tokens(batch["frames"][:, -1].long())[:, CELLS:].flatten(1)
            torch.testing.assert_close(inputs[:, base:base + HUD_INPUTS], raw, atol=1e-6, rtol=1e-6)
            # Then the refined cells averaged under the learned player softmax.
            _, weights = model.player_weights(encoding["cells"])
            expected = torch.einsum("bp,bpc->bc", weights, encoding["cells"])
            torch.testing.assert_close(inputs[:, base + HUD_INPUTS:], expected, atol=1e-6, rtol=1e-6)
            # Padding stays inert and older frames' HUDs do not enter the recall slice.
            older = batch["frames"].clone()
            older[:, :-1, wm.HUD_TOP:, :] = 15
            valid = batch["history_valid"].clone()
            valid[:, 0] = False
            reference = model(batch["frames"], valid, batch["previous_actions"])
            frames = batch["frames"].clone()
            frames[:, 0] = 7
            torch.testing.assert_close(model(frames, valid, batch["previous_actions"]), reference, atol=1e-6, rtol=1e-6)
            captured.clear()
            handle = model.projector[0].register_forward_hook(lambda module, args, output: captured.append(args[0]))
            model.encode(older, batch["history_valid"], batch["previous_actions"])
            handle.remove()
            torch.testing.assert_close(captured[0][:, base:base + HUD_INPUTS], raw, atol=1e-6, rtol=1e-6)
            # Labels and successor frames never reach the logits.
            out = world_losses(model, batch)
            torch.testing.assert_close(out["logits"], run(model, batch), atol=1e-5, rtol=1e-5)
            shuffled = {**batch, "next_frames": torch.full_like(batch["next_frames"], 6),
                        "optimal": batch["optimal"].flip(0), "distances": batch["distances"].flip(0)}
            torch.testing.assert_close(world_losses(model, shuffled)["logits"], out["logits"], atol=1e-5, rtol=1e-5)

    def test_gradients_reach_new_projector_columns_and_both_recall_inputs(self):
        model = make_model(state_recall=True).train()
        base = model.base_inputs
        batch = self.padded_batch()
        captured = []
        handle = model.projector[0].register_forward_hook(lambda module, args, output: captured.append(args[0]))
        latent = encode(model, batch)["latent"]
        handle.remove()
        inputs = captured[0]
        # A latent-only objective: nothing here reaches player_head except the recall path.
        objective = latent.square().sum()
        input_grad, weight_grad, player_grad, hud_grad = torch.autograd.grad(
            objective, [inputs, model.projector[0].weight, model.player_head.weight, model.hud[0].weight],
            retain_graph=True, allow_unused=True)
        self.assertIsNotNone(input_grad)
        self.assertGreater(input_grad[:, base:base + HUD_INPUTS].abs().sum().item(), 0., "raw HUD recall")
        self.assertGreater(input_grad[:, base + HUD_INPUTS:].abs().sum().item(), 0., "player-weighted recall")
        self.assertGreater(weight_grad[:, base:].abs().sum().item(), 0., "new projector columns")
        self.assertIsNotNone(player_grad, "the player softmax is trained through the latent")
        self.assertGreater(player_grad.abs().sum().item(), 0.)
        self.assertGreater(hud_grad.abs().sum().item(), 0.)
        plain = make_model().train()
        plain_latent = encode(plain, batch)["latent"]
        self.assertIsNone(torch.autograd.grad(plain_latent.square().sum(), plain.player_head.weight,
                                              allow_unused=True)[0],
                          "without recall the latent does not depend on player_head")
        # The full training objective moves the new columns and the recall path too.
        model.zero_grad(set_to_none=True)
        world_losses(model, batch)["total"].backward()
        self.assertGreater(model.projector[0].weight.grad[:, base:].abs().sum().item(), 0.)
        self.assertGreater(model.projector[0].weight.grad[:, :base].abs().sum().item(), 0.)

    def test_encoder_chunking_and_checkpointing_keep_recall_outputs_and_gradients(self):
        model = make_model(state_recall=True).train()
        other = copy.deepcopy(model)
        other.checkpoint_encoder = other.checkpoint_loops = True
        other.encoder_chunk_size = 1
        batch = self.padded_batch(3)
        outputs = []
        for policy in (model, other):
            torch.manual_seed(1)
            out = world_losses(policy, batch, sigreg_generator=torch.Generator().manual_seed(1))
            out["total"].backward()
            outputs.append(out)
        reference, recomputed = outputs
        torch.testing.assert_close(recomputed["latent"], reference["latent"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(recomputed["targets"], reference["targets"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(recomputed["logits"], reference["logits"], atol=1e-6, rtol=1e-5)
        for (name, first), (_, second) in zip(model.named_parameters(), other.named_parameters()):
            if first.grad is not None:
                torch.testing.assert_close(second.grad, first.grad, atol=1e-6, rtol=1e-5, msg=name)
        self.assertGreater(model.projector[0].weight.grad[:, model.base_inputs:].abs().sum().item(), 0.)

    # ------------------------------------------------------------------ trainer
    def test_trainer_flag_records_recall_and_migrates_a_plain_checkpoint(self):
        train = make_synthetic(seed=41, levels=2, steps=6, history=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "train.npz", meta=np.array(json.dumps({"source": "synthetic"})), **train)
            flags = ["--train", str(root / "train.npz"), "--epochs", "1", "--batch-size", "8",
                     "--device", "cpu", "--seed", "0"]
            for key, value in TINY.items():
                flags += [f"--{key.replace('_', '-')}", str(value)]
            plain_path, recall_path = root / "plain.pt", root / "recall.pt"
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                self.assertEqual(trainer.main(flags + ["--checkpoint-out", str(plain_path)]), 0, buffer.getvalue())
                self.assertEqual(trainer.main(flags + ["--checkpoint-out", str(recall_path), "--state-recall",
                                                       "--initialize-checkpoint", str(plain_path)]), 0,
                                 buffer.getvalue())
            self.assertIn("zero-padded new input columns of projector.0.weight", buffer.getvalue())
            plain_model, plain_checkpoint = wm.load_world_checkpoint(plain_path)
            recall_model, checkpoint = wm.load_world_checkpoint(recall_path)
            self.assertFalse(plain_checkpoint["config"]["state_recall"])
            self.assertTrue(checkpoint["config"]["state_recall"])
            self.assertTrue(recall_model.cfg.state_recall)
            self.assertEqual(checkpoint["initialize_checkpoint"], str(plain_path))
            self.assertEqual(checkpoint["checkpoint_loops"], False)
            self.assertEqual(checkpoint["encoder_chunk_size"], 0)
            report = json.loads(recall_path.with_suffix(".training.json").read_text())
            self.assertTrue(report["config"]["state_recall"])
            self.assertEqual(report["parameters"], recall_model.parameter_count())
            self.assertGreater(recall_model.parameter_count(), plain_model.parameter_count())
            with torch.inference_mode():
                self.assertEqual(tuple(recall_model(torch.from_numpy(train["frames"][:2])).shape), (2, ACTION_COUNT))
            # Reverse and unrelated initializations fail closed.
            for extra in (["--initialize-checkpoint", str(recall_path)],
                          ["--state-recall", "--initialize-checkpoint", str(plain_path), "--latent", "32"]):
                with self.subTest(extra=extra), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        trainer.main(flags + ["--checkpoint-out", str(root / "bad.pt")] + extra)
                    self.assertNotEqual(failure.exception.code, 0)
            self.assertFalse((root / "bad.pt").exists())


if __name__ == "__main__":
    unittest.main()
