"""CPU-only contracts for experimental public position recall and strict IO."""

import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pebby.agent.world_model import CELLS, WorldModelConfig, WorldPolicy
from pebby.agent.world_position_recall import (
    POSITION_INPUTS, POSITION_RECALL_FORMAT, POSITION_RECALL_VERSION,
    PositionRecallPolicy, initialize_from_base, load_checkpoint, save_checkpoint,
)


TINY = dict(channels=8, blocks=1, heads=2, expansion=2, loops=1, history=8,
            temporal_layers=1, hud_channels=4, hud_tokens=2, latent=8, reduce=1,
            predictor_blocks=1, predictor_hidden=16, value_hidden=8, max_distance=8,
            lookahead_depth=2, summary=2, readout_hidden=4, ranker_hidden=4,
            sigreg_projections=8, sigreg_knots=3,
            state_recall=True, glyph_recall=True, query_readout=True, grounding=True)


def config(**overrides):
    return WorldModelConfig(**{**TINY, **overrides})


class PositionRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(731)

    def test_known_softmax_order_and_original_prefix(self):
        model = PositionRecallPolicy(config())
        state = torch.randn(2, model.tokens, model.cfg.channels)
        probabilities = torch.arange(1, CELLS + 1, dtype=torch.float32)
        probabilities /= probabilities.sum()
        state[:, :CELLS, 0] = probabilities.log()
        with torch.no_grad():
            model.player_head.weight.zero_()
            model.player_head.weight[0, 0] = 1
            model.player_head.bias.zero_()
        hud = torch.randn(2, model.cfg.hud_tokens * model.cfg.channels)
        glyph = torch.randn(2, 14)
        original = WorldPolicy.projector_inputs(model, state, hud, glyph)
        actual = model.projector_inputs(state, hud, glyph)
        torch.testing.assert_close(actual[:, :-24], original, rtol=0, atol=0)
        grid = probabilities.reshape(12, 12)
        expected = torch.cat((grid.sum(1), grid.sum(0))).expand(2, -1)
        torch.testing.assert_close(actual[:, -24:], expected)
        torch.testing.assert_close(actual[:, -24:-12].sum(-1), torch.ones(2))
        torch.testing.assert_close(actual[:, -12:].sum(-1), torch.ones(2))
        self.assertEqual(model.projector_layout(), [*WorldPolicy.projector_layout(model),
                                                   ("position", original.size(1), 24)])
        self.assertEqual(actual.size(1), model.projector[0].in_features)
        default = PositionRecallPolicy(WorldModelConfig(state_recall=True, glyph_recall=True))
        self.assertEqual(default.projector_layout()[-1], ("position", 1742, 24))

    def test_detaches_only_appended_copy_existing_recall_still_trains_player(self):
        model = PositionRecallPolicy(config())
        state = torch.randn(2, model.tokens, model.cfg.channels, requires_grad=True)
        hud = torch.randn(2, model.cfg.hud_tokens * model.cfg.channels, requires_grad=True)
        glyph = torch.randn(2, 14, requires_grad=True)
        self.assertFalse(model.position_features(state).requires_grad)
        inputs = model.projector_inputs(state, hud, glyph)
        inputs[:, -24:].square().sum().backward()
        for tensor in (state, hud, glyph, model.player_head.weight):
            self.assertIsNotNone(tensor.grad)
            self.assertEqual(torch.count_nonzero(tensor.grad).item(), 0)
        model.zero_grad(set_to_none=True)
        state.grad = hud.grad = glyph.grad = None
        inputs = model.projector_inputs(state, hud, glyph)
        player_start = model.base_inputs + model.cfg.hud_tokens * model.cfg.channels
        inputs[:, player_start:player_start + model.cfg.channels].square().sum().backward()
        self.assertGreater(model.player_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(state.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        state.grad = None
        model.direct_logits(state[:, :CELLS])[0].square().sum().backward()
        self.assertGreater(model.player_head.weight.grad.abs().sum().item(), 0)

    def test_no_optional_recall_flags_required(self):
        model = PositionRecallPolicy(config(state_recall=False, glyph_recall=False,
                                           query_readout=False, grounding=False))
        state = torch.randn(2, model.tokens, model.cfg.channels)
        self.assertEqual(model.projector_inputs(state).shape, (2, model.base_inputs + 24))

    def test_dimensions_fail_closed(self):
        model = PositionRecallPolicy(config())
        state = torch.randn(2, model.tokens, model.cfg.channels)
        hud = torch.randn(2, model.cfg.hud_tokens * model.cfg.channels)
        glyph = torch.randn(2, 14)
        for bad_state, bad_hud, bad_glyph in ((state[:, :-1], hud, glyph),
                                            (state, hud[:, :-1], glyph),
                                            (state, hud, glyph[:, :-1]),
                                            (state, None, glyph), (state, hud, None)):
            with self.subTest(shapes=(bad_state.shape,
                                     None if bad_hud is None else bad_hud.shape)), \
                    self.assertRaises(ValueError):
                model.projector_inputs(bad_state, bad_hud, bad_glyph)

    def test_migration_copies_all_tensors_and_preserves_awake_public_h8_outputs(self):
        source = WorldPolicy(config()).eval()
        # Wake every zero-initialized predictor gate, glyph context and query
        # output so preservation cannot pass merely because those paths are zero.
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if "predictor" in name or "query_head.output" in name or "glyph_context" in name:
                    parameter.add_(torch.randn_like(parameter) * .05)
        target = PositionRecallPolicy(config()).eval()
        self.assertEqual(initialize_from_base(target, source), ["projector.0.weight"])
        self.assertEqual(target.parameter_count() - source.parameter_count(), 2 * source.cfg.latent * 24)
        for key, value in source.state_dict().items():
            actual = target.state_dict()[key]
            if key == "projector.0.weight":
                self.assertEqual(torch.count_nonzero(actual[:, -24:]).item(), 0)
                actual = actual[:, :-24]
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        frames = torch.randint(0, 16, (2, 8, 64, 64), dtype=torch.uint8)
        valid = torch.ones(2, 8, dtype=torch.bool)
        valid[0, :4] = False
        actions = torch.randint(0, 4, (2, 8))
        actions[~valid] = -1
        with torch.no_grad():
            expected = source.encode(frames, valid, actions)
            actual = target.encode(frames, valid, actions)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
            logits, extra = source.logits_from(expected)
            migrated_logits, migrated_extra = target.logits_from(actual)
            for key in extra:
                torch.testing.assert_close(migrated_extra[key], extra[key], rtol=2e-5, atol=2e-6)
            self.assertGreater(extra["query"].abs().sum().item(), 0)
            self.assertGreater((extra["successors"][:, 0] - extra["successors"][:, 1]).abs().sum().item(), 0)
            torch.testing.assert_close(migrated_logits, logits, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(target(frames, valid, actions), logits, rtol=2e-5, atol=2e-6)
        self.assertEqual(inspect.signature(target.forward), inspect.signature(source.forward))
        with self.assertRaises(TypeError):
            target(frames, valid, actions, player_cell=torch.zeros(2, 2))

    def test_migration_rejects_config_keys_and_shapes_before_mutation(self):
        target = PositionRecallPolicy(config())
        with self.assertRaisesRegex(ValueError, "identical base configs"):
            initialize_from_base(target, WorldPolicy(config(grounding=False)))
        with self.assertRaises(ValueError):
            initialize_from_base(target, PositionRecallPolicy(config()))
        for kind in ("missing", "extra", "shape"):
            source = WorldPolicy(config())
            if kind == "missing":
                del source.player_head.bias
            elif kind == "extra":
                source.register_buffer("unexpected", torch.ones(1))
            else:
                source.player_head.weight = torch.nn.Parameter(torch.zeros(1, 3))
            before = {key: value.clone() for key, value in target.state_dict().items()}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                initialize_from_base(target, source)
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_config_and_checkpoint_roundtrip(self):
        model = PositionRecallPolicy(config())
        self.assertEqual(model.config()["architecture"], "world")
        self.assertEqual(model.config()["position_recall"], POSITION_RECALL_VERSION)
        self.assertEqual(PositionRecallPolicy(model.config()).config(), model.config())
        with self.assertRaises(ValueError):
            PositionRecallPolicy({**model.config(), "position_recall": False})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            saved = save_checkpoint(path, model, experiment={"stage": "cpu-test", "steps": 0})
            rebuilt, metadata = load_checkpoint(path)
            self.assertEqual(metadata["format"], POSITION_RECALL_FORMAT)
            self.assertFalse(rebuilt.training)
            self.assertEqual(rebuilt.config(), model.config())
            self.assertEqual(metadata["experiment"], saved["experiment"])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(rebuilt.state_dict()[key], value, rtol=0, atol=0)
            frames = torch.randint(0, 16, (1, 64, 64), dtype=torch.uint8)
            with torch.no_grad():
                torch.testing.assert_close(rebuilt(frames), model.eval()(frames), rtol=0, atol=0)

    def test_corrupt_checkpoints_rejected(self):
        model = PositionRecallPolicy(config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            original = save_checkpoint(path, model)
            corruptions = [
                lambda item: item.update(format="pebby.ls20-world-policy.v1"),
                lambda item: item.update(parameters=item["parameters"] + 1),
                lambda item: item.update(parameters=float(item["parameters"])),
                lambda item: item["config"].pop("position_recall"),
                lambda item: item["config"].update(position_recall="unknown"),
                lambda item: item["config"].update(unknown=True),
                lambda item: item["config"].pop("history"),
                lambda item: item["position_source"].update(detach="all_player_paths"),
                lambda item: item["weights"].pop("player_head.bias"),
                lambda item: item["weights"].update(extra=torch.ones(1)),
                lambda item: item["weights"].update({"player_head.bias": torch.zeros(2)}),
                lambda item: item["weights"].update({"player_head.bias": torch.full((1,), float("nan"))}),
                lambda item: item.update(metric=float("inf")),
            ]
            for index, mutate in enumerate(corruptions):
                corrupted = copy.deepcopy(original)
                mutate(corrupted)
                torch.save(corrupted, path)
                with self.subTest(corruption=index), self.assertRaises(ValueError):
                    load_checkpoint(path)

    def test_failed_saves_are_atomic_and_reserved_metadata_cannot_override(self):
        model = PositionRecallPolicy(config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model)
            original_bytes = path.read_bytes()
            for metadata in ({"metric": float("nan")}, {"format": "bad"},
                             {"position_source": {}}, {"config": {}}):
                with self.assertRaises(ValueError):
                    save_checkpoint(path, model, **metadata)
                self.assertEqual(path.read_bytes(), original_bytes)
            with patch("pebby.agent.world_position_recall.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, model)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with self.assertRaises(ValueError):
                save_checkpoint(path, WorldPolicy(config()))


if __name__ == "__main__":
    unittest.main()
