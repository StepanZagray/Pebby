"""Optional learned query readout of the LS20 world policy (``WorldModelConfig.query_readout``).

Covers: the default stays off, adds no state-dict key and old checkpoints
without the key load and compute the same logits; an enabled network adds
exactly the ``query_head`` tensors with a zero output layer; migration from
plain, state-recall, glyph and glyph+recall sources (alone or combined with
adding recall/glyph/grounding) preserves states, latents, logits, targets and
losses; reverse/unrelated/partial migrations fail closed; gradients reach the
output layer first and the queries, relative geometry, raw stream and the
shared core once the output is nonzero; the head reads exactly the raw current
tokens and geometry relative to the learned player softmax; labels, successor
frames and older frames' HUDs cannot change the current logits; the
actual-successor policy uses the same head on its own history only;
chunk/checkpoint/life-reset parity of outputs and gradients; the trainer flag;
and a CPU migration smoke of the real base checkpoint when present (no world
data is loaded). Synthetic mazes only, CPU only, no official level.
"""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent import world_model as wm
from pebby.agent import world_readout as wr
from pebby.agent import world_train as trainer
from pebby.agent.glyph_model import GLYPH_CLASSES
from pebby.agent.world_model import ACTION_COUNT, CELLS, WorldModelConfig, world_losses
from tests.test_world_glyph import make_glyph_synthetic, randomize_glyph_path, wake
from tests.test_world_model import TINY, make_model, make_synthetic, to_batch

BASE_CHECKPOINT = Path("checkpoints/ls20-world-mixedpath-b1024.epoch1.pt")


def run(model, batch, **kwargs):
    return model(batch["frames"], batch["history_valid"], batch["previous_actions"], **kwargs)


def encode(model, batch, **kwargs):
    return model.encode(batch["frames"], batch["history_valid"], batch["previous_actions"], **kwargs)


def randomize_output(model, std=.5):
    """Make the residual correction non-trivial (a trained head)."""
    with torch.no_grad():
        torch.nn.init.normal_(model.query_head.output[-1].weight, std=std)
        torch.nn.init.normal_(model.query_head.output[-1].bias, std=std)
    return model


def successor_masks(data):
    masks = np.tile(np.array([1, 3, 0, 8], dtype=np.uint8), (len(data["frames"]), 1))
    masks[data["terminal"]] = 0
    return masks


class QueryReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.data = make_glyph_synthetic(seed=31, levels=2, steps=6, history=4)
        cls.data["next_optimal"] = successor_masks(cls.data)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def setUp(self):
        torch.manual_seed(5)

    def padded_batch(self, count=4):
        batch = to_batch(self.data, slice(0, count))
        self.assertFalse(bool(batch["history_valid"].all()), "the slice must contain padded history")
        return batch

    def assert_parity(self, target, source, batch, compare_losses=True):
        with torch.inference_mode():
            for loops in (1, 3):
                before, after = encode(source, batch, loops=loops), encode(target, batch, loops=loops)
                self.assertTrue(torch.equal(after["state"], before["state"]), f"state at loops={loops}")
                self.assertTrue(torch.equal(after["raw"], before["raw"]))
                torch.testing.assert_close(after["latent"], before["latent"], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(target.logits_from(after)[0], source.logits_from(before)[0],
                                           atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(run(target, batch), run(source, batch), atol=1e-6, rtol=1e-6)
        if compare_losses:
            with torch.no_grad():
                before = world_losses(source, batch, {"successor_policy": 1.},
                                      sigreg_generator=torch.Generator().manual_seed(1))
                after = world_losses(target, batch, {"successor_policy": 1.},
                                     sigreg_generator=torch.Generator().manual_seed(1))
            torch.testing.assert_close(after["targets"], before["targets"], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(after["logits"], before["logits"], atol=1e-5, rtol=1e-5)
            # A target that also adds glyph_recall gains its visual glyph term; every shared term matches.
            self.assertLessEqual(set(before["losses"]), set(after["losses"]))
            self.assertLessEqual(set(after["losses"]) - set(before["losses"]), {"glyph"})
            for name, value in before["losses"].items():
                torch.testing.assert_close(after["losses"][name], value, atol=1e-5, rtol=1e-5, msg=name)

    # ------------------------------------------------------------ compatibility
    def test_default_is_off_and_old_state_dicts_and_forward_are_preserved(self):
        config = WorldModelConfig(**TINY)
        self.assertFalse(config.query_readout)
        self.assertIn("query_readout", wm.BOOLEAN_FLAGS)
        without_key = {key: value for key, value in config.as_dict().items() if key != "query_readout"}
        self.assertEqual(WorldModelConfig.from_dict(without_key), config)
        with self.assertRaisesRegex(ValueError, "query_readout must be boolean"):
            WorldModelConfig(**TINY, query_readout=1)
        model = make_model().eval()
        self.assertFalse(hasattr(model, "query_head"))
        self.assertFalse(any(key.startswith("query_head.") for key in model.state_dict()))
        batch = self.padded_batch()
        with torch.inference_mode():
            encoding = encode(model, batch)
            self.assertEqual(set(encoding), {"state", "latent", "cells", "raw", "glyph"})
            self.assertIsNone(encoding["glyph"])
            # The raw tokens are the current frame's stem tokens, untouched by age/action/refinement.
            torch.testing.assert_close(encoding["raw"], model.frame_tokens(batch["frames"][:, -1].long()),
                                       atol=1e-6, rtol=1e-6)
            logits, extra = model.logits_from(encoding)
            self.assertNotIn("query", extra)
            torch.testing.assert_close(logits, extra["direct"] + model.ranker(extra["features"]).squeeze(-1),
                                       atol=1e-6, rtol=1e-6)
            with self.assertRaisesRegex(ValueError, "need query_readout"):
                model.query_logits(encoding)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            saved = wm.save_world_checkpoint(path, model)
            saved["config"].pop("query_readout")
            torch.save(saved, path)
            restored, loaded = wm.load_world_checkpoint(path)
        self.assertFalse(restored.cfg.query_readout)
        self.assertNotIn("query_readout", loaded["config"])
        with torch.inference_mode():
            self.assertTrue(torch.equal(run(model, batch), run(restored, batch)))

    def test_enabled_network_adds_only_the_head_with_a_zero_output_and_round_trips(self):
        plain = make_model()
        model = make_model(query_readout=True).eval()
        new_keys = set(model.state_dict()) - set(plain.state_dict())
        self.assertTrue(new_keys and all(key.startswith("query_head.") for key in new_keys))
        self.assertEqual(set(plain.state_dict()), set(model.state_dict()) - new_keys)
        self.assertEqual(model.tokens, plain.tokens, "no extra transformer token")
        self.assertEqual(model.projector[0].in_features, plain.projector[0].in_features, "latent path untouched")
        self.assertEqual(model.query_head.output[-1].weight.abs().sum().item(), 0.)
        self.assertEqual(model.query_head.output[-1].bias.abs().sum().item(), 0.)
        self.assertGreater(model.query_head.query_embedding.abs().sum().item(), 0.)
        self.assertEqual(model.query_head.glyph_inputs, 0)
        self.assertEqual(make_model(query_readout=True, glyph_recall=True).query_head.glyph_inputs, GLYPH_CLASSES)
        self.assertEqual(model.query_head.parameter_count(),
                         model.parameter_count() - plain.parameter_count())
        # The learned queries and stream embeddings are excluded from weight decay like other embeddings.
        decayed = {id(p) for p in wm.parameter_groups(model, .05)[0]["params"]}
        self.assertNotIn(id(model.query_head.query_embedding), decayed)
        self.assertNotIn(id(model.query_head.stream_embedding), decayed)
        self.assertIn(id(model.query_head.blocks[0].mlp[0].weight), decayed)
        batch = self.padded_batch()
        with torch.inference_mode():
            logits, extra = model.logits_from(encode(model, batch))
            self.assertEqual(tuple(extra["query"].shape), (4, ACTION_COUNT))
            self.assertEqual(extra["query"].abs().sum().item(), 0., "fresh head contributes nothing")
            self.assertEqual(tuple(model(batch["frames"][:, -1]).shape), (4, ACTION_COUNT))
        randomize_output(model)
        with torch.inference_mode():
            logits = run(model, batch)
            extra = model.logits_from(encode(model, batch))[1]
            self.assertGreater(extra["query"].abs().sum().item(), 0.)
            self.assertFalse(torch.allclose(extra["query"][0], extra["query"][1]), "input dependent")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.pt"
            saved = wm.save_world_checkpoint(path, model)
            self.assertTrue(saved["config"]["query_readout"])
            restored, loaded = wm.load_world_checkpoint(path)
            self.assertTrue(restored.cfg.query_readout)
            self.assertEqual(loaded["parameters"], model.parameter_count())
            with torch.inference_mode():
                self.assertTrue(torch.equal(logits, run(restored, batch)))
            saved["config"]["query_readout"] = False
            torch.save(saved, path)
            with self.assertRaises((ValueError, RuntimeError)):
                wm.load_world_checkpoint(path)
        with self.assertRaises(RuntimeError):
            make_model().load_state_dict(model.state_dict())
        with self.assertRaises(RuntimeError):
            model.load_state_dict(make_model().state_dict())

    # ---------------------------------------------------------------- migration
    def test_migration_preserves_outputs_for_every_source_and_flag_combination(self):
        batch = self.padded_batch()
        plain = wake(make_model()).eval()
        recall = wake(make_model(state_recall=True)).eval()
        glyph = randomize_glyph_path(wake(make_model(glyph_recall=True))).eval()
        both = randomize_glyph_path(wake(make_model(glyph_recall=True, state_recall=True))).eval()
        cases = {
            "plain -> query": (plain, dict(query_readout=True), []),
            "plain -> query + recall": (plain, dict(query_readout=True, state_recall=True), ["projector.0.weight"]),
            "plain -> query + glyph": (plain, dict(query_readout=True, glyph_recall=True), ["projector.0.weight"]),
            "plain -> query + glyph + recall + grounding": (
                plain, dict(query_readout=True, glyph_recall=True, state_recall=True, grounding=True),
                ["projector.0.weight"]),
            "recall -> recall + query": (recall, dict(state_recall=True, query_readout=True), []),
            "glyph -> glyph + query": (glyph, dict(glyph_recall=True, query_readout=True), []),
            "glyph -> glyph + recall + query": (glyph, dict(glyph_recall=True, state_recall=True, query_readout=True),
                                                ["projector.0.weight"]),
            "glyph + recall -> + query": (both, dict(glyph_recall=True, state_recall=True, query_readout=True), []),
        }
        for name, (source, flags, migrated) in cases.items():
            with self.subTest(case=name):
                target = make_model(**flags).eval()
                self.assertEqual(wm.initialize_from_checkpoint(target, source), migrated)
                self.assertEqual(target.query_head.output[-1].weight.abs().sum().item(), 0.)
                self.assertEqual(target.query_head.output[-1].bias.abs().sum().item(), 0.)
                # Every other head tensor keeps a fresh (nonzero) initialization.
                fresh = [p for key, p in target.query_head.named_parameters()
                         if not key.startswith("output.2.") and p.ndim == 2]
                self.assertTrue(all(p.abs().sum().item() > 0 for p in fresh))
                target_parameters = dict(target.named_parameters())
                for parameter_name, parameter in source.named_parameters():
                    if parameter_name not in migrated:
                        self.assertTrue(torch.equal(parameter.detach(), target_parameters[parameter_name].detach()),
                                        parameter_name)
                if flags.get("glyph_recall") and not source.cfg.glyph_recall:
                    self.assertEqual(target.glyph_context.weight.abs().sum().item(), 0.)
                self.assert_parity(target, source, batch, compare_losses=not flags.get("grounding"))

    def test_existing_trained_query_can_gain_glyph_without_changing_logits(self):
        batch = self.padded_batch()
        for recall in (False, True):
            with self.subTest(state_recall=recall):
                source = randomize_output(make_model(query_readout=True, state_recall=recall)).eval()
                target = make_model(query_readout=True, glyph_recall=True, state_recall=recall).eval()
                migrated = wm.initialize_from_checkpoint(target, source)
                self.assertIn("query_head.context.0.weight", migrated)
                self.assertTrue(torch.equal(target.query_head.output[-1].weight,
                                            source.query_head.output[-1].weight))
                self.assertEqual(target.query_head.context[0].weight[:, -GLYPH_CLASSES:].abs().sum().item(), 0.)
                self.assert_parity(target, source, batch, compare_losses=False)

    def test_migration_rejects_reverse_unrelated_and_partial_sources(self):
        query = randomize_output(make_model(query_readout=True))
        with self.assertRaisesRegex(ValueError, "without query_readout"):
            wm.initialize_from_checkpoint(make_model(), query)
        with self.assertRaisesRegex(ValueError, "without query_readout"):
            wm.initialize_from_checkpoint(make_model(state_recall=True, glyph_recall=True), query)
        for override in (dict(latent=32), dict(readout_hidden=16), dict(channels=32, heads=4), dict(loops=3)):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "differs beyond"):
                wm.initialize_from_checkpoint(make_model(query_readout=True, **override), make_model())
        # Identical configs copy every tensor verbatim, the trained output layer included.
        same = make_model(query_readout=True)
        self.assertEqual(wm.initialize_from_checkpoint(same, query), [])
        for (name, mine), (_, theirs) in zip(same.state_dict().items(), query.state_dict().items()):
            self.assertTrue(torch.equal(mine, theirs), name)
        self.assertGreater(same.query_head.output[-1].weight.abs().sum().item(), 0.)
        # A source whose own head is incomplete is refused, not partially loaded.
        broken = copy.deepcopy(query)
        del broken.query_head.blocks[0]
        with self.assertRaisesRegex(ValueError, "missing"):
            wm.initialize_from_checkpoint(make_model(query_readout=True, state_recall=True), broken)
        foreign = copy.deepcopy(make_model())
        foreign.extra = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            wm.initialize_from_checkpoint(make_model(query_readout=True), foreign)
        # A shape mismatch elsewhere is never hidden by the head addition.
        resized = copy.deepcopy(make_model())
        resized.player_head = torch.nn.Linear(TINY["channels"], 2)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            wm.initialize_from_checkpoint(make_model(query_readout=True), resized)

    # ------------------------------------------------------------------ gradients
    def test_gradients_reach_the_output_first_then_queries_geometry_raw_stream_and_core(self):
        model = make_model(query_readout=True).train()
        batch = self.padded_batch()
        head = model.query_head
        inner = [head.query_embedding, head.query_condition.weight, head.context[0].weight, head.relative.weight,
                 head.raw_cell.weight, head.refined_cell.weight, head.raw_hud.weight, head.stream_embedding,
                 head.blocks[0].attention.in_proj_weight, head.blocks[-1].mlp[0].weight]
        # Zero output: only the output layer moves; nothing behind it receives signal yet.
        query = model.logits_from(encode(model, batch))[1]["query"]
        output_grad, *inner_grads = torch.autograd.grad(query.sum(), [head.output[-1].weight] + inner,
                                                        allow_unused=True)
        self.assertGreater(output_grad.abs().sum().item(), 0.)
        self.assertTrue(all(g is None or g.abs().sum().item() == 0. for g in inner_grads))
        # Nonzero output: the queries, conditioning, geometry, all four streams and the
        # shared encoder (stem through the raw stream, core through the refined stream).
        randomize_output(model)
        query = model.logits_from(encode(model, batch))[1]["query"]
        grads = torch.autograd.grad(query.sum(), inner + [model.stem[0].weight, model.core[0].mlp[0].weight,
                                                          model.player_head.weight], allow_unused=True)
        for parameter, grad in zip(inner + ["stem", "core", "player_head"], grads):
            with self.subTest(parameter=parameter if isinstance(parameter, str) else tuple(parameter.shape)):
                self.assertIsNotNone(grad)
                self.assertGreater(grad.abs().sum().item(), 0.)
        # The raw stream is a genuinely separate path: with the refined stream and the
        # refined context zeroed, the stem still receives gradient from the correction.
        raw_only = copy.deepcopy(model)
        with torch.no_grad():
            raw_only.query_head.refined_cell.weight.zero_()
            raw_only.query_head.refined_hud.weight.zero_()
        query = raw_only.logits_from(encode(raw_only, batch))[1]["query"]
        self.assertGreater(torch.autograd.grad(query.sum(), raw_only.stem[0].weight)[0].abs().sum().item(), 0.)
        # The full objective moves the head and, through it, the successor policy too.
        model.zero_grad(set_to_none=True)
        out = world_losses(model, batch, {"successor_policy": 1.})
        out["losses"]["successor_policy"].backward(retain_graph=True)
        self.assertGreater(head.query_embedding.grad.abs().sum().item(), 0., "successor policy trains the head")
        model.zero_grad(set_to_none=True)
        out["total"].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertGreater(head.output[-1].weight.grad.abs().sum().item(), 0.)

    # -------------------------------------------------------------- what it reads
    def test_head_reads_raw_current_tokens_and_geometry_relative_to_the_learned_player(self):
        model = randomize_output(make_model(query_readout=True, glyph_recall=True)).eval()
        batch = self.padded_batch()
        captured = []
        handle = model.query_head.register_forward_hook(lambda module, args, output: captured.append(args))
        with torch.no_grad():
            encoding = encode(model, batch)
            model.logits_from(encoding)
            handle.remove()
            raw, state, weights, glyph = captured[0]
            self.assertIs(state, encoding["state"])
            torch.testing.assert_close(raw, model.frame_tokens(batch["frames"][:, -1].long()), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(weights, model.player_weights(encoding["cells"])[1], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(glyph, encoding["glyph"], atol=1e-6, rtol=1e-6)
            # Older frames' pixels reach the head only through the refined stream: the raw
            # argument is a function of the current frame alone.
            older = batch["frames"].clone()
            older[:, :-1] = 15
            captured.clear()
            handle = model.query_head.register_forward_hook(lambda module, args, output: captured.append(args))
            model.logits_from(model.encode(older, batch["history_valid"], batch["previous_actions"]))
            handle.remove()
            self.assertTrue(torch.equal(captured[0][0], raw))
            # Padding stays inert.
            valid = batch["history_valid"].clone()
            valid[:, 0] = False
            frames = batch["frames"].clone()
            frames[:, 0] = 7
            torch.testing.assert_close(model(frames, valid, batch["previous_actions"]),
                                       model(batch["frames"], valid, batch["previous_actions"]), atol=1e-6, rtol=1e-6)
            # Glyph handling is strict in both directions.
            with self.assertRaisesRegex(ValueError, "glyph probabilities"):
                model.query_head(raw, state, weights, None)
            with self.assertRaisesRegex(ValueError, "without glyph inputs"):
                make_model(query_readout=True).query_head(raw, state, weights, glyph)
            with self.assertRaisesRegex(ValueError, "raw current tokens"):
                model.query_logits({key: value for key, value in encoding.items() if key != "raw"})
        # Geometry: a one-hot player at (row, col) gives every cell its offset from it, and the
        # features are translation-equivariant (no absolute board coordinate enters).
        one_hot = torch.zeros(2, CELLS)
        one_hot[0, 3 * 12 + 4] = 1.
        one_hot[1, 5 * 12 + 6] = 1.
        torch.testing.assert_close(wr.expected_player_position(one_hot), torch.tensor([[3., 4.], [5., 6.]]))
        features = wr.relative_position_features(one_hot)
        self.assertEqual(tuple(features.shape), (2, CELLS, wr.RELATIVE_FEATURES))
        torch.testing.assert_close(features[0, 7 * 12 + 1, :2], torch.tensor([4. / 11, -3. / 11]))
        torch.testing.assert_close(features[0, 7 * 12 + 1], features[1, 9 * 12 + 3])
        torch.testing.assert_close(features[0, 3 * 12 + 4], torch.tensor([0., 0., 0., 0., 0., 1.]))
        soft = torch.full((1, CELLS), 1. / CELLS)
        torch.testing.assert_close(wr.expected_player_position(soft), torch.tensor([[5.5, 5.5]]))
        with self.assertRaises(ValueError):
            wr.relative_position_features(torch.zeros(1, CELLS + 1))

    def test_labels_and_next_frames_cannot_change_current_logits(self):
        model = randomize_output(randomize_glyph_path(make_model(query_readout=True, glyph_recall=True,
                                                                  state_recall=True))).eval()
        batch = self.padded_batch()
        with torch.no_grad():
            reference = run(model, batch)
            out = world_losses(model, batch, {"successor_policy": 1.})
            torch.testing.assert_close(out["logits"], reference, atol=1e-5, rtol=1e-5)
            shuffled = {**batch, "next_frames": torch.full_like(batch["next_frames"], 6),
                        "optimal": batch["optimal"].flip(0), "distances": batch["distances"].flip(0),
                        "next_optimal": torch.zeros_like(batch["next_optimal"]),
                        "player_cell": batch["player_cell"].flip(0),
                        "current_triple": (batch["current_triple"] + 1) % 4,
                        "next_triple": (batch["next_triple"] + 1) % 4}
            leaked = world_losses(model, shuffled, {"successor_policy": 1.})
            torch.testing.assert_close(leaked["logits"], reference, atol=1e-5, rtol=1e-5)
            self.assertFalse(torch.allclose(leaked["targets"], out["targets"]))
            self.assertFalse(torch.allclose(leaked["losses"]["policy"], out["losses"]["policy"]))

    def test_successor_policy_uses_the_head_on_each_successor_own_history_only(self):
        model = randomize_output(wake(make_model(query_readout=True))).eval()
        batch = self.padded_batch()
        encodings, results = [], []
        original = model.logits_from

        def capture(encoding, *args, **kwargs):
            result = original(encoding, *args, **kwargs)
            encodings.append(encoding)
            results.append(result)
            return result

        with torch.no_grad(), patch.object(model, "logits_from", side_effect=capture):
            world_losses(model, batch, {"successor_policy": 1.})
        self.assertEqual(len(results), 2)
        current_logits, current_extra = results[0]
        next_logits, next_extra = results[1]
        self.assertEqual(tuple(next_extra["query"].shape), (4 * ACTION_COUNT, ACTION_COUNT))
        with torch.no_grad():
            # The successor correction is the head applied to the successor encodings' own raw tokens.
            torch.testing.assert_close(next_extra["query"], model.query_logits(encodings[1]), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(encodings[1]["raw"].view(4, ACTION_COUNT, model.tokens, -1),
                                       model.frame_tokens(batch["next_frames"].flatten(0, 1).long())
                                       .view(4, ACTION_COUNT, model.tokens, -1), atol=1e-6, rtol=1e-6)
            # Row 3 has a full history: each successor's logits equal a standalone forward on the
            # history shifted by one and ending in that successor (no other row, no label).
            row = 3
            self.assertTrue(bool(batch["history_valid"][row].all()))
            for action in range(ACTION_COUNT):
                frames = torch.cat((batch["frames"][row, 1:], batch["next_frames"][row, action][None]))[None]
                actions = torch.cat((batch["previous_actions"][row, 1:], torch.tensor([action])))[None]
                standalone = model(frames, torch.ones(1, frames.size(1), dtype=torch.bool), actions)
                torch.testing.assert_close(next_logits[row * ACTION_COUNT + action][None], standalone,
                                           atol=1e-5, rtol=1e-5)
            # Changing another row's successor frames cannot change this row's successor logits.
            altered = {**batch, "next_frames": batch["next_frames"].clone()}
            altered["next_frames"][0] = 6
            with patch.object(model, "logits_from", side_effect=capture):
                world_losses(model, altered, {"successor_policy": 1.})
            torch.testing.assert_close(results[-1][0][row * ACTION_COUNT:(row + 1) * ACTION_COUNT],
                                       next_logits[row * ACTION_COUNT:(row + 1) * ACTION_COUNT], atol=1e-6, rtol=1e-6)
            self.assertFalse(torch.allclose(results[-1][0][:ACTION_COUNT], next_logits[:ACTION_COUNT]))
            torch.testing.assert_close(results[-2][0], current_logits, atol=1e-6, rtol=1e-6)

    def test_encoder_chunking_checkpointing_and_life_reset_keep_outputs_and_gradients(self):
        model = randomize_output(randomize_glyph_path(wake(make_model(
            query_readout=True, glyph_recall=True, state_recall=True)))).train()
        other = copy.deepcopy(model)
        other.checkpoint_encoder = other.checkpoint_loops = True
        other.encoder_chunk_size = 1
        batch = self.padded_batch(3)
        batch["lost_life"] = torch.zeros_like(batch["terminal"])
        batch["lost_life"][0, 0] = True
        outputs = []
        for policy in (model, other):
            torch.manual_seed(1)
            out = world_losses(policy, batch, {"successor_policy": 1.},
                               sigreg_generator=torch.Generator().manual_seed(1))
            out["total"].backward()
            outputs.append(out)
        reference, recomputed = outputs
        for key in ("latent", "targets", "logits"):
            torch.testing.assert_close(recomputed[key], reference[key], atol=1e-6, rtol=1e-5, msg=key)
        torch.testing.assert_close(recomputed["losses"]["successor_policy"], reference["losses"]["successor_policy"],
                                   atol=1e-6, rtol=1e-5)
        for (name, first), (_, second) in zip(model.named_parameters(), other.named_parameters()):
            if first.grad is not None:
                torch.testing.assert_close(second.grad, first.grad, atol=1e-6, rtol=1e-5, msg=name)
        self.assertGreater(model.query_head.query_embedding.grad.abs().sum().item(), 0.)
        # The chunked encoding still carries raw tokens and glyph probabilities for every row.
        with torch.no_grad():
            chunked, whole = encode(other, batch), encode(model, batch)
        torch.testing.assert_close(chunked["raw"], whole["raw"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(chunked["glyph"], whole["glyph"], atol=1e-6, rtol=1e-6)
        # A life-reset successor's readout sees only the repeated reset frame (its own history).
        with torch.no_grad():
            reset = batch["next_frames"][:1, 0:1].expand(-1, 4, -1, -1)
            expected = model(reset, torch.tensor([[False, False, False, True]]), torch.full((1, 4), -1))
            encodings = []
            with patch.object(model, "logits_from", wraps=model.logits_from) as spy:
                world_losses(model, batch, {"successor_policy": 1.})
                successor_logits = spy.call_args_list[1].args[0]
                successor_logits = model.logits_from(successor_logits)[0]
        torch.testing.assert_close(successor_logits[:1], expected, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------ trainer
    def test_trainer_flag_migrates_a_plain_checkpoint_and_records_the_head(self):
        train = make_synthetic(seed=41, levels=2, steps=6, history=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "train.npz", meta=np.array(json.dumps({"source": "synthetic"})), **train)
            flags = ["--train", str(root / "train.npz"), "--epochs", "1", "--batch-size", "8",
                     "--device", "cpu", "--seed", "0", "--select-on", "last"]
            for key, value in TINY.items():
                flags += [f"--{key.replace('_', '-')}", str(value)]
            plain_path, query_path = root / "plain.pt", root / "query.pt"
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                self.assertEqual(trainer.main(flags + ["--checkpoint-out", str(plain_path)]), 0, buffer.getvalue())
                self.assertEqual(trainer.main(flags + ["--checkpoint-out", str(query_path), "--query-readout",
                                                       "--state-recall", "--initialize-checkpoint", str(plain_path)]),
                                 0, buffer.getvalue())
            log = buffer.getvalue()
            self.assertIn("query readout:", log)
            self.assertIn("queries x 2 blocks", log)
            plain_model, plain_checkpoint = wm.load_world_checkpoint(plain_path)
            query_model, checkpoint = wm.load_world_checkpoint(query_path)
            self.assertFalse(plain_checkpoint["config"]["query_readout"])
            self.assertTrue(checkpoint["config"]["query_readout"] and checkpoint["config"]["state_recall"])
            self.assertTrue(query_model.cfg.query_readout)
            self.assertEqual(checkpoint["initialize_checkpoint"], str(plain_path))
            report = json.loads(query_path.with_suffix(".training.json").read_text())
            self.assertTrue(report["config"]["query_readout"])
            self.assertEqual(report["parameters"], query_model.parameter_count())
            self.assertGreater(query_model.parameter_count(), plain_model.parameter_count())
            args = trainer.build_parser().parse_args(["--train", "x.npz"])
            self.assertFalse(args.query_readout)
            with torch.inference_mode():
                self.assertEqual(tuple(query_model(torch.from_numpy(train["frames"][:2])).shape), (2, ACTION_COUNT))
            for extra in (["--initialize-checkpoint", str(query_path)],
                          ["--query-readout", "--initialize-checkpoint", str(plain_path), "--latent", "32"]):
                with self.subTest(extra=extra), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        trainer.main(flags + ["--checkpoint-out", str(root / "bad.pt")] + extra)
                    self.assertNotEqual(failure.exception.code, 0)
            self.assertFalse((root / "bad.pt").exists())

    @unittest.skipUnless(BASE_CHECKPOINT.exists(), "base checkpoint not present")
    def test_real_base_checkpoint_migrates_on_cpu_with_identical_logits(self):
        source, checkpoint = wm.load_world_checkpoint(BASE_CHECKPOINT)  # read only; no world data
        target = wm.build_world_policy({**checkpoint["config"], "query_readout": True}).eval()
        self.assertEqual(wm.initialize_from_checkpoint(target, source), [])
        torch.manual_seed(0)
        frames = torch.randint(0, 16, (2, 3, 64, 64), dtype=torch.uint8)
        valid = torch.tensor([[False, True, True], [True, True, True]])
        actions = torch.tensor([[-1, 0, 3], [1, 2, 2]])
        with torch.inference_mode():
            torch.testing.assert_close(target(frames, valid, actions), source(frames, valid, actions),
                                       atol=1e-6, rtol=1e-6)
        self.assertEqual(target.parameter_count(), checkpoint["parameters"] + target.query_head.parameter_count())


if __name__ == "__main__":
    unittest.main()
