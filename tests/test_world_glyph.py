"""Learned glyph perception in the LS20 world policy (``WorldModelConfig.glyph_recall``).

Covers: the default stays off and old checkpoints without the key load and
compute the same logits; an enabled network adds exactly ``glyph_encoder`` and
``glyph_context`` and 14 projector columns AFTER the state-recall block; the
grouped softmax of the CURRENT crop is what reaches the projector and the
context is one vector broadcast to every source token; zero-migration parity
for every flag order (plain -> glyph, plain -> glyph + recall (+ grounding),
recall -> recall + glyph, glyph -> glyph + recall with the glyph block moved);
reverse/unrelated migrations fail closed; copying a pretrained classifier
keeps the migrated outputs; successor frames and teacher triples cannot change
the current logits; gradients reach the context, the new projector columns and
the encoder through both the direct and the latent path; the glyph term is
purely visual; chunk/checkpoint parity; and the trainer flags, provenance and
guards. Synthetic mazes only, CPU only, no official level.
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

from pebby.agent import glyph_model as gm
from pebby.agent import glyph_train as gt
from pebby.agent import world_model as wm
from pebby.agent import world_train as trainer
from pebby.agent.glyph_model import GLYPH_CLASSES, GlyphEncoder, glyph_probabilities
from pebby.agent.world_model import ACTION_COUNT, CELLS, WorldModelConfig, world_losses
from tests.test_glyph_model import write_glyph_npz
from tests.test_world_model import TINY, make_model, make_synthetic, to_batch

CHANNELS, HUD_TOKENS = TINY["channels"], TINY["hud_tokens"]
RECALL_INPUTS = HUD_TOKENS * CHANNELS + CHANNELS


def triples_from_tokens(tokens):
    """The synthetic renderer fills the glyph crop with one colour; derive a consistent triple from it."""
    tokens = np.asarray(tokens, dtype=np.int64)
    return np.stack([tokens % 6, tokens % 4, (tokens // 2) % 4], axis=-1).astype(np.int16)


def make_glyph_synthetic(**kwargs):
    data = make_synthetic(**kwargs)
    data["current_triple"] = triples_from_tokens(data["frames"][:, -1, 55, 3])
    data["next_triple"] = triples_from_tokens(data["next_frames"][:, :, 55, 3])
    return data


def glyph_arrays(data):
    """Glyph NPZ arrays (current + four successors per state) from a world synthetic split."""
    glyphs = np.concatenate((data["frames"][:, -1, 55:61, 3:9][:, None], data["next_frames"][:, :, 55:61, 3:9]),
                            axis=1).reshape(-1, 6, 6)
    triples = np.concatenate((data["current_triple"][:, None], data["next_triple"]), axis=1).reshape(-1, 3)
    return glyphs, triples, np.repeat(data["seeds"], 5), np.tile(np.arange(5), len(data["seeds"]))


def run(model, batch, **kwargs):
    return model(batch["frames"], batch["history_valid"], batch["previous_actions"], **kwargs)


def encode(model, batch, **kwargs):
    return model.encode(batch["frames"], batch["history_valid"], batch["previous_actions"], **kwargs)


def wake(model):
    """AdaLN-zero gates start at zero; wake them so the ranker path is not trivial."""
    with torch.no_grad():
        for block in model.predictor.blocks:
            torch.nn.init.normal_(block.modulation[-1].weight, std=.5)
    return model


def randomize_glyph_path(model):
    """Make a glyph network's context and glyph projector columns non-trivial."""
    with torch.no_grad():
        torch.nn.init.normal_(model.glyph_context.weight, std=.5)
        torch.nn.init.normal_(model.projector[0].weight[:, -GLYPH_CLASSES:], std=.5)
    return model


class WorldGlyphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.data = make_glyph_synthetic(seed=31, levels=2, steps=6, history=4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def setUp(self):
        torch.manual_seed(5)

    def padded_batch(self, count=4):
        batch = to_batch(self.data, slice(0, count))
        self.assertFalse(bool(batch["history_valid"].all()), "the slice must contain padded history")
        self.assertIn("current_triple", batch)
        self.assertEqual(tuple(batch["next_triple"].shape), (count, ACTION_COUNT, 3))
        return batch

    def assert_parity(self, target, source, batch, compare_losses=True):
        with torch.inference_mode():
            for loops in (1, 3):
                before, after = encode(source, batch, loops=loops), encode(target, batch, loops=loops)
                self.assertTrue(torch.equal(after["state"], before["state"]), f"state at loops={loops}")
                torch.testing.assert_close(after["latent"], before["latent"], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(target.logits_from(after)[0], source.logits_from(before)[0],
                                           atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(run(target, batch), run(source, batch), atol=1e-6, rtol=1e-6)
        if compare_losses:
            with torch.no_grad():
                before = world_losses(source, batch, sigreg_generator=torch.Generator().manual_seed(1))
                after = world_losses(target, batch, sigreg_generator=torch.Generator().manual_seed(1))
            torch.testing.assert_close(after["targets"], before["targets"], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(after["logits"], before["logits"], atol=1e-5, rtol=1e-5)
            for name, value in before["losses"].items():
                torch.testing.assert_close(after["losses"][name], value, atol=1e-5, rtol=1e-5, msg=name)

    # ------------------------------------------------------------ compatibility
    def test_default_is_off_and_old_checkpoints_without_the_key_still_load(self):
        config = WorldModelConfig(**TINY)
        self.assertFalse(config.glyph_recall)
        self.assertEqual(wm.BOOLEAN_FLAGS, ("grounding", "state_recall", "glyph_recall", "query_readout", "cell_recall"))
        without_key = {key: value for key, value in config.as_dict().items() if key != "glyph_recall"}
        self.assertEqual(WorldModelConfig.from_dict(without_key), config)
        with self.assertRaisesRegex(ValueError, "glyph_recall must be boolean"):
            WorldModelConfig(**TINY, glyph_recall=1)
        model = make_model().eval()
        self.assertEqual(model.glyph_inputs, 0)
        self.assertFalse(hasattr(model, "glyph_encoder"))
        self.assertEqual(model.projector_layout(), [("base", 0, model.base_inputs), ("recall", model.base_inputs, 0),
                                                    ("glyph", model.base_inputs, 0)])
        with self.assertRaisesRegex(ValueError, "glyph logits need glyph_recall"):
            model.glyph_logits(torch.zeros(1, 64, 64, dtype=torch.long))
        batch = self.padded_batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            saved = wm.save_world_checkpoint(path, model)
            saved["config"].pop("glyph_recall")
            torch.save(saved, path)
            restored, loaded = wm.load_world_checkpoint(path)
        self.assertFalse(restored.cfg.glyph_recall)
        with torch.inference_mode():
            self.assertTrue(torch.equal(run(model, batch), run(restored, batch)))
            # A plain network refuses glyph logits; a glyph network requires them.
            tokens = model.frame_tokens(batch["frames"].flatten(0, 1)).view(4, 4, model.tokens, -1)
            with self.assertRaisesRegex(ValueError, "without glyph_recall"):
                model.assemble(tokens, batch["history_valid"], batch["previous_actions"],
                               glyph_logits=torch.zeros(4, GLYPH_CLASSES))
            glyph = make_model(glyph_recall=True).eval()
            tokens = glyph.frame_tokens(batch["frames"].flatten(0, 1)).view(4, 4, glyph.tokens, -1)
            with self.assertRaisesRegex(ValueError, "needs the current frame's glyph logits"):
                glyph.assemble(tokens, batch["history_valid"], batch["previous_actions"])
            with self.assertRaisesRegex(ValueError, "needs the current frame's glyph logits"):
                glyph.assemble(tokens, batch["history_valid"], batch["previous_actions"],
                               glyph_logits=torch.zeros(4, GLYPH_CLASSES + 1))
        # The plain training objective ignores triples entirely.
        self.assertNotIn("glyph", world_losses(model, batch)["losses"])

    def test_enabled_network_adds_encoder_context_and_last_projector_columns(self):
        plain = make_model()
        model = make_model(glyph_recall=True).eval()
        new_keys = set(model.state_dict()) - set(plain.state_dict())
        self.assertEqual(new_keys, {"glyph_context.weight", "glyph_encoder.mlp.0.weight", "glyph_encoder.mlp.0.bias",
                                    "glyph_encoder.mlp.2.weight", "glyph_encoder.mlp.2.bias"})
        self.assertEqual(model.glyph_inputs, GLYPH_CLASSES)
        self.assertEqual(model.projector[0].in_features, model.base_inputs + GLYPH_CLASSES)
        self.assertEqual(model.glyph_context.weight.abs().sum().item(), 0., "context starts at zero")
        self.assertIsNone(model.glyph_context.bias)
        self.assertGreater(model.projector[0].weight[:, -GLYPH_CLASSES:].abs().sum().item(), 0.,
                           "a fresh glyph network uses the usual projector initialization")
        both = make_model(glyph_recall=True, state_recall=True)
        self.assertEqual(both.projector[0].in_features, both.base_inputs + RECALL_INPUTS + GLYPH_CLASSES)
        self.assertEqual(both.projector_layout(), [("base", 0, both.base_inputs),
                                                   ("recall", both.base_inputs, RECALL_INPUTS),
                                                   ("glyph", both.base_inputs + RECALL_INPUTS, GLYPH_CLASSES)])
        self.assertEqual(model.tokens, plain.tokens, "no extra transformer token")
        batch = self.padded_batch()
        with torch.inference_mode():
            self.assertEqual(tuple(model.frame_tokens(batch["frames"][:, -1]).shape),
                             tuple(plain.frame_tokens(batch["frames"][:, -1]).shape))
        # The projector's last 14 inputs are the grouped softmax of the CURRENT crop's logits.
        for network in (model, both.eval()):
            captured = []
            handle = network.projector[0].register_forward_hook(lambda module, args, output: captured.append(args[0]))
            with torch.no_grad():
                encoding = encode(network, batch)
                handle.remove()
                inputs = captured[0]
                logits = network.glyph_encoder(batch["frames"][:, -1, 55:61, 3:9].long())
                expected = glyph_probabilities(logits)
                torch.testing.assert_close(inputs[:, -GLYPH_CLASSES:], expected, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(inputs[:, :network.base_inputs],
                                           network.reduce(encoding["state"]).flatten(1), atol=1e-6, rtol=1e-6)
                if network.cfg.state_recall:
                    raw = network.frame_tokens(batch["frames"][:, -1].long())[:, CELLS:].flatten(1)
                    base = network.base_inputs
                    torch.testing.assert_close(inputs[:, base:base + HUD_TOKENS * CHANNELS], raw, atol=1e-6, rtol=1e-6)
                # Older frames' glyphs never enter: only the current crop is classified.
                older = batch["frames"].clone()
                older[:, :-1, 55:61, 3:9] = 15
                captured.clear()
                handle = network.projector[0].register_forward_hook(lambda module, args, output: captured.append(args[0]))
                network.encode(older, batch["history_valid"], batch["previous_actions"])
                handle.remove()
                torch.testing.assert_close(captured[0][:, -GLYPH_CLASSES:], expected, atol=1e-6, rtol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glyph.pt"
            saved = wm.save_world_checkpoint(path, model)
            self.assertTrue(saved["config"]["glyph_recall"])
            restored, loaded = wm.load_world_checkpoint(path)
            self.assertEqual(loaded["parameters"], model.parameter_count())
            with torch.inference_mode():
                self.assertTrue(torch.equal(run(model, batch), run(restored, batch)))
            saved["config"]["glyph_recall"] = False
            torch.save(saved, path)
            with self.assertRaises((ValueError, RuntimeError)):
                wm.load_world_checkpoint(path)
        with self.assertRaises(RuntimeError):
            make_model().load_state_dict(model.state_dict())

    def test_context_is_one_vector_broadcast_to_every_current_source_token(self):
        model = make_model(glyph_recall=True).eval()
        batch = self.padded_batch()
        sources = []
        handle = model.core[0].register_forward_hook(lambda module, args, output: sources.append(args[1].clone()))
        with torch.no_grad():
            encode(model, batch, loops=1)
            torch.nn.init.normal_(model.glyph_context.weight, std=.5)
            encode(model, batch, loops=1)
            handle.remove()
            zero_context, with_context = sources
            features = glyph_probabilities(model.glyph_logits(batch["frames"][:, -1]))
            shift = model.glyph_context(features)  # [B, C]
        difference = with_context - zero_context  # [B, tokens, C]
        self.assertEqual(tuple(difference.shape), (4, model.tokens, CHANNELS))
        torch.testing.assert_close(difference, shift[:, None, :].expand_as(difference), atol=1e-6, rtol=1e-6)
        self.assertGreater(difference.abs().sum().item(), 0.)
        # Every loop recalls that same source (the context is not re-added per loop).
        sources.clear()
        handle = model.core[0].register_forward_hook(lambda module, args, output: sources.append(args[1]))
        with torch.no_grad():
            encode(model, batch, loops=3)
        handle.remove()
        self.assertEqual(len(sources), 3)
        self.assertTrue(all(source is sources[0] for source in sources))

    # ---------------------------------------------------------------- migration
    def test_zero_migration_parity_for_every_flag_order(self):
        batch = self.padded_batch()
        plain = wake(make_model()).eval()
        recall = wake(make_model(state_recall=True)).eval()
        glyph = randomize_glyph_path(wake(make_model(glyph_recall=True))).eval()
        cases = {
            "plain -> glyph": (plain, dict(glyph_recall=True)),
            "plain -> glyph + recall": (plain, dict(glyph_recall=True, state_recall=True)),
            "plain -> glyph + recall + grounding": (plain, dict(glyph_recall=True, state_recall=True, grounding=True)),
            "recall -> recall + glyph": (recall, dict(state_recall=True, glyph_recall=True)),
            "glyph -> glyph + recall": (glyph, dict(glyph_recall=True, state_recall=True)),
        }
        for name, (source, flags) in cases.items():
            with self.subTest(case=name):
                target = make_model(**flags).eval()
                self.assertEqual(wm.initialize_from_checkpoint(target, source), ["projector.0.weight"])
                weight, source_weight = target.projector[0].weight.detach(), source.projector[0].weight.detach()
                base = target.base_inputs
                self.assertTrue(torch.equal(weight[:, :base], source_weight[:, :base]))
                if flags.get("state_recall") and not source.cfg.state_recall:
                    self.assertEqual(weight[:, base:base + RECALL_INPUTS].abs().sum().item(), 0.)
                if source.cfg.glyph_recall:
                    # The glyph block moved behind the inserted recall block, values intact.
                    self.assertTrue(torch.equal(weight[:, -GLYPH_CLASSES:], source_weight[:, -GLYPH_CLASSES:]))
                    self.assertTrue(torch.equal(target.glyph_context.weight.detach(), source.glyph_context.weight.detach()))
                    self.assertGreater(target.glyph_context.weight.abs().sum().item(), 0.)
                else:
                    self.assertEqual(weight[:, -GLYPH_CLASSES:].abs().sum().item(), 0.)
                    self.assertEqual(target.glyph_context.weight.abs().sum().item(), 0.)
                    self.assertGreater(sum(p.abs().sum().item() for p in target.glyph_encoder.parameters()), 0.,
                                       "the fresh classifier keeps its usual initialization")
                target_parameters = dict(target.named_parameters())
                for parameter_name, parameter in source.named_parameters():
                    if parameter_name != "projector.0.weight":
                        self.assertTrue(torch.equal(parameter.detach(), target_parameters[parameter_name].detach()),
                                        parameter_name)
                # Bias and second layer are copied unchanged; latents, logits, targets
                # and every shared loss term are the source's.
                self.assertTrue(torch.equal(target.projector[0].bias.detach(), source.projector[0].bias.detach()))
                self.assertTrue(torch.equal(target.projector[2].weight.detach(), source.projector[2].weight.detach()))
                self.assert_parity(target, source, batch, compare_losses=not flags.get("grounding"))
        # A grounding target needs grounding labels for its losses; parity of the policy path suffices above.

    def test_migration_rejects_reverse_and_unrelated_configs(self):
        glyph = make_model(glyph_recall=True)
        with self.assertRaisesRegex(ValueError, "without glyph_recall"):
            wm.initialize_from_checkpoint(make_model(), glyph)
        with self.assertRaisesRegex(ValueError, "without glyph_recall"):
            wm.initialize_from_checkpoint(make_model(state_recall=True), glyph)
        for override in (dict(latent=32), dict(hud_tokens=8), dict(channels=32, heads=4), dict(loops=3)):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "differs beyond"):
                wm.initialize_from_checkpoint(make_model(glyph_recall=True, **override), make_model())
        # Identical configs copy every tensor verbatim, classifier and context included.
        same = make_model(glyph_recall=True)
        self.assertEqual(wm.initialize_from_checkpoint(same, glyph), [])
        for (name, mine), (_, theirs) in zip(same.state_dict().items(), glyph.state_dict().items()):
            self.assertTrue(torch.equal(mine, theirs), name)
        foreign = copy.deepcopy(make_model())
        foreign.extra = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            wm.initialize_from_checkpoint(make_model(glyph_recall=True), foreign)
        truncated = copy.deepcopy(make_model())
        del truncated.player_head
        with self.assertRaisesRegex(ValueError, "missing"):
            wm.initialize_from_checkpoint(make_model(glyph_recall=True), truncated)
        # A glyph source whose own classifier is incomplete is refused, not partially loaded.
        broken = copy.deepcopy(glyph)
        del broken.glyph_encoder
        with self.assertRaisesRegex(ValueError, "missing"):
            wm.initialize_from_checkpoint(make_model(glyph_recall=True, state_recall=True), broken)

    def test_pretrained_classifier_initialization_keeps_migrated_outputs(self):
        batch = self.padded_batch()
        plain = wake(make_model()).eval()
        target = make_model(glyph_recall=True).eval()
        wm.initialize_from_checkpoint(target, plain)
        torch.manual_seed(77)
        pretrained = GlyphEncoder()
        with torch.no_grad():
            for parameter in pretrained.parameters():
                torch.nn.init.normal_(parameter, std=.3)
        copied = wm.initialize_glyph_encoder(target, pretrained)
        self.assertEqual(copied, sorted(f"glyph_encoder.{key}" for key in pretrained.state_dict()))
        for key, value in pretrained.state_dict().items():
            self.assertTrue(torch.equal(target.glyph_encoder.state_dict()[key], value))
        with torch.inference_mode():
            self.assertFalse(torch.equal(target.glyph_logits(batch["frames"][:, -1]),
                                         GlyphEncoder()(batch["frames"][:, -1, 55:61, 3:9])))
        self.assert_parity(target, plain, batch)
        with self.assertRaisesRegex(ValueError, "needs a network with glyph_recall"):
            wm.initialize_glyph_encoder(make_model(), pretrained)
        with self.assertRaisesRegex(ValueError, "architecture"):
            wm.initialize_glyph_encoder(target, GlyphEncoder(hidden=32))
        with self.assertRaisesRegex(ValueError, "architecture"):
            wm.initialize_glyph_encoder(target, torch.nn.Linear(576, 14))

    # ------------------------------------------------------------ training terms
    def test_successors_and_teacher_triples_cannot_change_current_logits(self):
        model = randomize_glyph_path(make_model(glyph_recall=True)).eval()
        batch = self.padded_batch()
        with torch.no_grad():
            reference = run(model, batch)
            out = world_losses(model, batch)
            torch.testing.assert_close(out["logits"], reference, atol=1e-5, rtol=1e-5)
            self.assertEqual(tuple(out["glyph_logits"].shape), (4, GLYPH_CLASSES))
            torch.testing.assert_close(out["glyph_logits"], model.glyph_logits(batch["frames"][:, -1]),
                                       atol=1e-6, rtol=1e-6)
            self.assertIn("glyph", out["losses"])
            for prefix in ("current", "actual"):
                for field in ("shape", "color", "rotation"):
                    self.assertIn(f"glyph_{prefix}_{field}_accuracy", out["diagnostics"])
            shuffled = {**batch, "next_frames": torch.full_like(batch["next_frames"], 6),
                        "optimal": batch["optimal"].flip(0), "distances": batch["distances"].flip(0),
                        "current_triple": (batch["current_triple"] + 1) % 4,
                        "next_triple": (batch["next_triple"] + 1) % 4}
            leaked = world_losses(model, shuffled)
            torch.testing.assert_close(leaked["logits"], reference, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(leaked["glyph_logits"], out["glyph_logits"], atol=1e-6, rtol=1e-6)
            self.assertFalse(torch.allclose(leaked["losses"]["glyph"], out["losses"]["glyph"]))
            self.assertFalse(torch.allclose(leaked["targets"], out["targets"]))
        # Missing or invalid teacher triples fail closed before any forward pass.
        for missing in ("current_triple", "next_triple"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "glyph_recall requires"):
                world_losses(model, {key: value for key, value in batch.items() if key != missing})
        bad = {**batch, "current_triple": batch["current_triple"].clone()}
        bad["current_triple"][0, 0] = 6
        with self.assertRaises(ValueError):
            world_losses(model, bad)
        with self.assertRaisesRegex(ValueError, "next_triple must be"):
            world_losses(model, {**batch, "next_triple": batch["next_triple"][:, :3]})
        # The term enters the total through its weight (default 1); plain networks never use it.
        self.assertEqual(wm.DEFAULT_WEIGHTS["glyph"], 1.)
        with torch.no_grad():
            seeded = world_losses(model, batch, sigreg_generator=torch.Generator().manual_seed(1))
            without = world_losses(model, batch, {"glyph": 0.}, sigreg_generator=torch.Generator().manual_seed(1))
            torch.testing.assert_close(without["total"] + seeded["losses"]["glyph"], seeded["total"],
                                       atol=1e-5, rtol=1e-5)

    def test_gradients_reach_context_new_projector_columns_and_the_encoder(self):
        model = randomize_glyph_path(make_model(glyph_recall=True)).train()
        batch = self.padded_batch()
        encoder_parameters = list(model.glyph_encoder.parameters())
        # Latent path: the new projector columns and, through them, the classifier.
        latent = encode(model, batch)["latent"]
        weight_grad, *encoder_grads = torch.autograd.grad(
            latent.square().sum(), [model.projector[0].weight] + encoder_parameters, allow_unused=True)
        self.assertGreater(weight_grad[:, -GLYPH_CLASSES:].abs().sum().item(), 0., "new projector columns")
        self.assertTrue(all(g is not None and g.abs().sum().item() > 0 for g in encoder_grads))
        # Direct policy path: through the broadcast context into the classifier.
        direct = model.direct_logits(encode(model, batch)["cells"])[0]
        context_grad, *encoder_grads = torch.autograd.grad(
            direct.sum(), [model.glyph_context.weight] + encoder_parameters, allow_unused=True)
        self.assertGreater(context_grad.abs().sum().item(), 0.)
        self.assertTrue(all(g is not None and g.abs().sum().item() > 0 for g in encoder_grads))
        # Even a zero context receives gradient (so migration can start learning it).
        fresh = make_model(glyph_recall=True).train()
        direct = fresh.direct_logits(encode(fresh, batch)["cells"])[0]
        self.assertGreater(torch.autograd.grad(direct.sum(), fresh.glyph_context.weight)[0].abs().sum().item(), 0.)
        # The glyph classification term is purely visual: no gradient into the frame encoder.
        out = world_losses(model, batch)
        stem_grad = torch.autograd.grad(out["losses"]["glyph"], model.stem[0].weight, retain_graph=True,
                                        allow_unused=True)[0]
        self.assertIsNone(stem_grad)
        model.zero_grad(set_to_none=True)
        out["total"].backward()
        self.assertGreater(model.projector[0].weight.grad[:, -GLYPH_CLASSES:].abs().sum().item(), 0.)
        self.assertGreater(model.glyph_context.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum().item() > 0 for p in encoder_parameters))

    def test_encoder_chunking_and_checkpointing_keep_glyph_outputs_and_gradients(self):
        model = randomize_glyph_path(make_model(glyph_recall=True, state_recall=True)).train()
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
        for key in ("latent", "targets", "logits", "glyph_logits"):
            torch.testing.assert_close(recomputed[key], reference[key], atol=1e-6, rtol=1e-5, msg=key)
        for (name, first), (_, second) in zip(model.named_parameters(), other.named_parameters()):
            if first.grad is not None:
                torch.testing.assert_close(second.grad, first.grad, atol=1e-6, rtol=1e-5, msg=name)
        self.assertGreater(model.glyph_context.weight.grad.abs().sum().item(), 0.)

    # ------------------------------------------------------------------ trainer
    def test_trainer_flags_record_glyph_provenance_and_guard_misuse(self):
        train = make_glyph_synthetic(seed=41, levels=2, steps=6, history=4)
        validation = make_glyph_synthetic(seed=42, levels=1, steps=4, history=4)
        validation["seeds"] = validation["seeds"] + 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (("train", train), ("validation", validation)):
                np.savez(root / f"{name}.npz", meta=np.array(json.dumps({"source": "synthetic"})), **payload)
            bare = {key: value for key, value in train.items() if key not in ("current_triple", "next_triple")}
            np.savez(root / "bare.npz", **bare)
            write_glyph_npz(root / "glyph-train.npz", *glyph_arrays(train), split="train")
            write_glyph_npz(root / "glyph-validation.npz", *glyph_arrays(validation), split="validation")
            glyph_path = root / "glyph.pt"
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                self.assertEqual(gt.main(["--train", str(root / "glyph-train.npz"), "--validation",
                                          str(root / "glyph-validation.npz"), "--checkpoint-out", str(glyph_path),
                                          "--steps", "5", "--batch-size", "2", "--threads", "4"]), 0, buffer.getvalue())
            pretrained, glyph_checkpoint = gm.load_glyph_checkpoint(glyph_path)
            self.assertEqual(glyph_checkpoint["train_seeds"], [0, 1])
            flags = ["--train", str(root / "train.npz"), "--validation", str(root / "validation.npz"),
                     "--epochs", "1", "--batch-size", "8", "--device", "cpu", "--seed", "0"]
            for key, value in TINY.items():
                flags += [f"--{key.replace('_', '-')}", str(value)]
            plain_path, fresh_path, init_path = root / "plain.pt", root / "fresh.pt", root / "init.pt"
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                self.assertEqual(trainer.main(flags + ["--checkpoint-out", str(plain_path)]), 0, buffer.getvalue())
                self.assertEqual(trainer.main(flags + ["--checkpoint-out", str(fresh_path), "--glyph-recall"]), 0,
                                 buffer.getvalue())
                self.assertEqual(trainer.main(flags + ["--checkpoint-out", str(init_path), "--glyph-recall",
                                                       "--initialize-checkpoint", str(plain_path),
                                                       "--initialize-glyph-checkpoint", str(glyph_path)]), 0,
                                 buffer.getvalue())
            log = buffer.getvalue()
            self.assertIn("zero-padded new input columns of projector.0.weight", log)
            self.assertIn("Initialized 4 glyph encoder tensors", log)
            self.assertIn("glyph recall: 14 glyph probabilities", log)
            fresh_model, fresh_checkpoint = wm.load_world_checkpoint(fresh_path)
            self.assertTrue(fresh_model.cfg.glyph_recall)
            self.assertIsNone(fresh_checkpoint["glyph_source"])
            self.assertIsNone(fresh_checkpoint["initialize_glyph_checkpoint"])
            init_model, checkpoint = wm.load_world_checkpoint(init_path)
            self.assertTrue(checkpoint["config"]["glyph_recall"])
            self.assertEqual(checkpoint["initialize_checkpoint"], str(plain_path))
            self.assertEqual(checkpoint["initialize_glyph_checkpoint"], str(glyph_path))
            source = checkpoint["glyph_source"]
            self.assertEqual(source["path"], str(glyph_path))
            self.assertEqual(source["sha256"], gt.file_digest(glyph_path))
            self.assertEqual((source["train_levels"], source["validation_levels"]), (2, 1))
            self.assertEqual(source["results"], glyph_checkpoint["results"])
            self.assertEqual(source["format"], gm.GLYPH_FORMAT)
            report = json.loads(init_path.with_suffix(".training.json").read_text())
            for key in ("glyph", "glyph_current_shape_accuracy", "glyph_actual_rotation_accuracy"):
                self.assertIn(key, report["history"][0]["train"])
                self.assertIn(key, report["history"][0]["validation"])
            self.assertIn("--glyph-weight", " ".join(trainer.build_parser().format_help().split()))
            with torch.inference_mode():
                self.assertEqual(tuple(init_model(torch.from_numpy(train["frames"][:2])).shape), (2, ACTION_COUNT))
            # Guards fail closed and write nothing.
            # A classifier whose own splits are disjoint but that saw the WORLD validation level 100.
            leaky = root / "leaky-glyph.pt"
            metadata = {k: v for k, v in glyph_checkpoint.items() if k not in ("format", "config", "parameters", "weights")}
            gm.save_glyph_checkpoint(leaky, pretrained, **{**metadata, "train_seeds": [0, 1, 100],
                                                            "validation_seeds": [200]})
            cases = {
                "glyph init without flag": ["--initialize-glyph-checkpoint", str(glyph_path)],
                "no triples": ["--glyph-recall", "--train", str(root / "bare.npz")],
                "reverse": ["--initialize-checkpoint", str(init_path)],
                "world checkpoint as glyph": ["--glyph-recall", "--initialize-glyph-checkpoint", str(plain_path)],
                "glyph trained on validation levels": ["--glyph-recall", "--initialize-glyph-checkpoint", str(leaky)],
                "unrelated config": ["--glyph-recall", "--initialize-checkpoint", str(plain_path), "--latent", "32"],
            }
            for name, extra in cases.items():
                with self.subTest(name=name), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        trainer.main(flags + ["--checkpoint-out", str(root / "bad.pt")] + extra)
                    self.assertNotEqual(failure.exception.code, 0)
            self.assertFalse((root / "bad.pt").exists())


if __name__ == "__main__":
    unittest.main()
