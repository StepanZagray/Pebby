"""Focused regression tests for the shared-depth LS20 transformer policy."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from pebby.agent import evaluate, model as policy, train as trainer
from pebby.agent.looped import LOOPED_MODEL_FORMAT, LoopedLs20Policy


TINY = {
    "architecture": "looped",
    "channels": 16,
    "blocks": 2,
    "heads": 4,
    "expansion": 2,
    "loops": 4,
    "hud_channels": 8,
    "reduce_channels": 4,
    "hidden": 16,
}


def make_policy(**overrides):
    config = {**TINY, **overrides}
    return policy.build_policy(config)


def frames(seed=0, batch=2):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, policy.PALETTE, (batch, 64, 64), generator=generator)


class LoopedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._torch_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_shared_core_reuses_parameters_at_all_requested_depths(self):
        models = [LoopedLs20Policy(**{key: value for key, value in TINY.items()
                                      if key != "architecture"} | {"loops": depth})
                  for depth in (1, 2, 4, 8)]
        counts = {model.parameter_count() for model in models}
        key_sets = {tuple(model.state_dict()) for model in models}
        self.assertEqual(len(counts), 1)
        self.assertEqual(len(key_sets), 1)
        for depth, model in zip((1, 2, 4, 8), models):
            self.assertEqual(model.config()["loops"], depth)
            self.assertEqual(len(model.core), TINY["blocks"])

        shared = make_policy()
        calls = []
        handle = shared.core[0].register_forward_hook(lambda module, args, output: calls.append(args))
        outputs = [shared(frames(3), loops=depth, return_all=True) for depth in (1, 2, 4, 8)]
        handle.remove()
        self.assertEqual(len(calls), 1 + 2 + 4 + 8)
        self.assertTrue(all(call[1] is calls[-1][1] for call in calls[-8:]))
        self.assertEqual([tuple(output.shape) for output in outputs],
                         [(depth, 2, 4) for depth in (1, 2, 4, 8)])
        self.assertFalse(torch.allclose(outputs[0][-1], outputs[-1][-1]))

    def test_early_and_final_depths_backpropagate_finite_gradients(self):
        model = make_policy().train()
        batch = frames(4, batch=1)
        for depth in (1, 4):
            with self.subTest(depth=depth):
                model.zero_grad(set_to_none=True)
                model(batch, loops=depth).square().mean().backward()
                for name in ("stem.0.weight", "source_norm.weight",
                             "core.0.recall.weight", "action.weight"):
                    gradient = dict(model.named_parameters())[name].grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    self.assertGreater(gradient.abs().sum().item(), 0., name)

        model.zero_grad(set_to_none=True)
        states = []
        def retain(module, args, output):
            output.retain_grad()
            states.append(output)
        handle = model.core[-1].register_forward_hook(retain)
        model(batch, loops=4).square().mean().backward()
        handle.remove()
        self.assertEqual(len(states), 4)
        for state in (states[0], states[-1]):
            self.assertIsNotNone(state.grad)
            self.assertTrue(torch.isfinite(state.grad).all())
            self.assertGreater(state.grad.abs().sum().item(), 0.)

    def test_source_hud_palette_and_position_each_influence_logits(self):
        model = make_policy().eval()
        original = frames(5, batch=1)
        with torch.inference_mode():
            baseline = model(original)
            source = model.encode(original)
            state = source
            for _ in range(model.loops):
                for block in model.core:
                    state = block(state, torch.zeros_like(source))
            source_without_content = model.readout(state)

            playfield = original.clone()
            row, column = policy.PLAY_TOP + 2, policy.PLAY_LEFT + 2
            playfield[:, row, column] = (playfield[:, row, column] + 1) % policy.PALETTE
            play_logits = model(playfield)

            hud = original.clone()
            # Carried-token pixels below the playfield: the board is identical.
            hud[:, 60, 3:9] = (hud[:, 60, 3:9] + 1) % policy.PALETTE
            self.assertTrue(torch.equal(original[:, policy.PLAY_TOP:policy.PLAY_BOTTOM,
                                                 policy.PLAY_LEFT:policy.PLAY_RIGHT],
                                        hud[:, policy.PLAY_TOP:policy.PLAY_BOTTOM,
                                            policy.PLAY_LEFT:policy.PLAY_RIGHT]))
            hud_logits = model(hud)

            palette = original.clone()
            palette[original == 0] = 1
            palette[original == 1] = 0
            palette_logits = model(palette)

            position_model = copy.deepcopy(model)
            position_model.row_position[0, 0].add_(1.)
            position_logits = position_model(original)

        for name, logits in {
            "source": source_without_content,
            "playfield": play_logits,
            "HUD": hud_logits,
            "palette": palette_logits,
            "position": position_logits,
        }.items():
            with self.subTest(name=name):
                self.assertTrue(torch.any(torch.abs(baseline - logits) > 1e-7), name)

    def test_repeated_frames_are_deterministic_and_have_no_cross_frame_memory(self):
        model = make_policy().eval()
        first = frames(6, batch=1)
        second = torch.full_like(first, 15)
        with torch.inference_mode():
            before = model(first)
            other = model(second)
            after = model(first)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(torch.allclose(before, other))

    def test_input_depth_and_config_validation(self):
        model = make_policy().eval()
        for bad in ((64, 64), (2, 63, 64), (2, 64, 64, 1)):
            with self.subTest(shape=bad), self.assertRaisesRegex(ValueError, "frames must be"):
                model(torch.zeros(bad, dtype=torch.int64))
        for depth in (0, -1, True, 1.5):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, "positive integer"):
                model(frames(7, batch=1), loops=depth)
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_policy(channels=15, heads=4)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_policy(blocks=0)
        with self.assertRaisesRegex(ValueError, "unknown policy architecture"):
            policy.build_policy({"architecture": "unknown"})

    def test_checkpoint_round_trip_and_architecture_mismatch_rejection(self):
        model = make_policy().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "looped.pt"
            saved = policy.save_checkpoint(path, model, epochs=1, train_seeds=[1])
            restored, loaded = policy.load_checkpoint(path)
            self.assertEqual(loaded["format"], LOOPED_MODEL_FORMAT)
            self.assertEqual(loaded["config"], model.config())
            self.assertEqual(loaded["parameters"], model.parameter_count())
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(frames(8)), restored(frames(8))))

            bad_format = Path(directory) / "cnn-format-looped-config.pt"
            mismatched = dict(saved)
            mismatched["format"] = policy.MODEL_FORMAT
            torch.save(mismatched, bad_format)
            with self.assertRaisesRegex(ValueError, "format and architecture disagree"):
                policy.load_checkpoint(bad_format)

            bad_legacy = Path(directory) / "looped-format-legacy-config.pt"
            legacy_config = {key: value for key, value in model.config().items()
                             if key != "architecture"}
            mismatched = dict(saved)
            mismatched["config"] = legacy_config
            torch.save(mismatched, bad_legacy)
            with self.assertRaisesRegex(ValueError, "format and architecture disagree"):
                policy.load_checkpoint(bad_legacy)

    def test_run_epoch_reports_final_and_all_exit_losses_distinctly(self):
        model = make_policy().eval()
        batch = frames(9, batch=2)
        actions = torch.tensor([0, 1], dtype=torch.long)
        masks = torch.tensor([0b0001, 0b0011], dtype=torch.uint8)
        loader = DataLoader(TensorDataset(batch, actions, masks), batch_size=2)
        target = trainer.bits_of(masks, torch.device("cpu"))
        target = target / target.sum(1, keepdim=True)
        with torch.inference_mode():
            exits = model(batch, loops=4, return_all=True)
            expected_final = (-(target * torch.log_softmax(exits[-1], dim=-1)).sum(-1).mean())
            expected_all = (-(target * torch.log_softmax(exits, dim=-1)).sum(-1).mean())
        self.assertNotAlmostEqual(expected_final.item(), expected_all.item(), places=6)

        all_model = copy.deepcopy(model)
        final_model = copy.deepcopy(model)
        all_stats = trainer.run_epoch(
            all_model, loader, torch.device("cpu"),
            optimizer=torch.optim.SGD(all_model.parameters(), lr=0.),
            loop_loss="all", train_min_loops=4,
            depth_generator=torch.Generator().manual_seed(0))
        final_stats = trainer.run_epoch(
            final_model, loader, torch.device("cpu"),
            optimizer=torch.optim.SGD(final_model.parameters(), lr=0.),
            loop_loss="final", train_min_loops=4,
            depth_generator=torch.Generator().manual_seed(0))
        self.assertEqual(all_stats["depth_batches"], {"4": 1})
        self.assertEqual(final_stats["depth_batches"], {"4": 1})
        self.assertAlmostEqual(all_stats["cross_entropy"], expected_final.item(), places=6)
        self.assertAlmostEqual(all_stats["objective_cross_entropy"], expected_all.item(), places=6)
        self.assertAlmostEqual(final_stats["cross_entropy"], expected_final.item(), places=6)
        self.assertAlmostEqual(final_stats["objective_cross_entropy"], expected_final.item(), places=6)

    @staticmethod
    def _empty_evaluation_report():
        return {
            "format": "pebby.ls20-evaluation.v1",
            "levels": 0,
            "completed": 0,
            "completion_rate": 0.,
            "goals_cleared": 0,
            "goals_total": 0,
            "goal_rate": 0.,
            "mean_actions": 0.,
            "mean_actions_vs_optimal": None,
            "runs_with_oracle": 0,
            "won": 0,
            "game_over": 0,
            "capped": 0,
            "stuck": 0,
            "stall_dominated": 0,
            "on_stall": "next-best",
            "runs": [],
        }

    def test_evaluate_cli_depth_override_is_recorded_on_looped_checkpoint(self):
        model = make_policy().eval()
        checkpoint = {"format": LOOPED_MODEL_FORMAT, "config": model.config(),
                      "parameters": model.parameter_count()}
        observed = {}
        def completion(policy_instance, *args):
            observed["loops"] = policy_instance.loops
            return self._empty_evaluation_report()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"placeholder")
            output = io.StringIO()
            with patch.object(sys, "argv", ["evaluate", "--checkpoint", str(path),
                                              "--loops", "2", "--protocol", "strict"]), \
                 patch.object(evaluate, "load_checkpoint", return_value=(model, checkpoint)), \
                 patch.object(evaluate, "completion_rate", side_effect=completion), \
                 patch.object(evaluate, "shipped_table", return_value=[]), \
                 redirect_stdout(output):
                evaluate.main()
        self.assertEqual(observed["loops"], 2)
        self.assertIn('"inference_loops": 2', output.getvalue())
        self.assertIn('"checkpoint_loops": 4', output.getvalue())

    def test_evaluate_cli_rejects_depth_override_for_cnn_checkpoint(self):
        model = policy.build_policy().eval()
        checkpoint = {"format": policy.MODEL_FORMAT, "config": model.config(),
                      "parameters": model.parameter_count()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"placeholder")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["evaluate", "--checkpoint", str(path),
                                              "--loops", "2"]), \
                 patch.object(evaluate, "load_checkpoint", return_value=(model, checkpoint)), \
                 redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                evaluate.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires a checkpoint with mutable inference loops", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
