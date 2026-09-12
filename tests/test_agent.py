"""Regression checks for the LS20 policy: shapes, one real step, shards, checkpoints.

These run on CPU. The end-to-end case needs `pebby.ls20.generate` and
`pebby.ls20.plan`; it is gated on `GENERATOR` and skips with a message rather
than failing, so this file stays useful in a checkout where the level generator
has not landed.

Two of these tests guard decisions that came from measurements rather than
taste, and would otherwise be quietly undone: the model has no auxiliary value
head, and its colour encoding is not shared across colours.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import data, model as policy, train as trainer
from pebby.ls20 import names

ROOT = Path(__file__).resolve().parents[1]
# Controllers at 309k, 2.03M and 7.81M parameters all scored identically on the
# prior ARC-AGI-3 work on this machine, so the policy is deliberately small and
# a jump back to millions of parameters should have to argue for itself here.
PARAMETER_BUDGET = 500_000


def generator_ready():
    try:
        from pebby.ls20 import generate, plan  # noqa: F401
    except ImportError:
        return False
    return True


GENERATOR = generator_ready()
NO_GENERATOR = "pebby.ls20.generate / pebby.ls20.plan are not implemented yet"
TINY = {"channels": 16, "blocks": 2, "hud_channels": 16, "reduce_channels": 8, "hidden": 32}


def fake_shard(seeds, per_level=7, rng_seed=0):
    """Arrays in exactly the shard layout, without needing the real game."""
    rng = np.random.default_rng(rng_seed)
    count = len(seeds) * per_level
    actions = rng.integers(0, len(names.ACTION_IDS), count).astype(np.uint8)
    # The label's own bit is always set; a second optimal action sometimes joins
    # it, which is what the real oracle's ties look like.
    optimal = ((1 << actions) | np.where(rng.random(count) < .3, 1 << ((actions + 1) % 4), 0)).astype(np.uint8)
    return {"frames": rng.integers(0, 16, (count, names.FRAME_SIZE, names.FRAME_SIZE), dtype=np.uint8),
            "actions": actions,
            "optimal": optimal,
            "to_go": rng.integers(0, 50, count).astype(np.int16),
            "seeds": np.repeat(np.asarray(seeds, dtype=np.int32), per_level),
            "meta": {"format": data.DATA_FORMAT, "generator_version": "test", "seeds": list(seeds),
                     "difficulty": 1, "epsilon": .05, "rng_seed": rng_seed, "samples": count,
                     "episodes": len(seeds), "completed_episodes": len(seeds),
                     "created": "2026-01-01T00:00:00+00:00"}}


class PolicyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = policy.build_policy().eval()

    def test_output_shape_and_parameter_budget(self):
        for batch in (1, 2, 5):
            with self.subTest(batch=batch):
                frames = torch.randint(0, policy.PALETTE, (batch, names.FRAME_SIZE, names.FRAME_SIZE))
                with torch.no_grad():
                    logits = self.model(frames)
                self.assertIsInstance(logits, torch.Tensor)  # One head, not a tuple.
                self.assertEqual(logits.shape, (batch, len(names.ACTION_IDS)))
                self.assertTrue(torch.isfinite(logits).all())
        count = self.model.parameter_count()
        self.assertEqual(count, sum(p.numel() for p in self.model.parameters()))
        self.assertLess(count, PARAMETER_BUDGET)
        self.assertGreater(count, 50_000)

    def test_no_auxiliary_value_head(self):
        # Auxiliary prediction heads measured as failures on this workload; if
        # one comes back it must come back with evidence, not by accident.
        self.assertFalse(hasattr(self.model, "value"))
        children = [name for name, _ in self.model.named_children()]
        self.assertNotIn("value", children)
        self.assertEqual(children[-1], "action")  # The action logits are the only output.

    def test_colours_are_not_shared_across_the_palette(self):
        """Swapping two colours must change the decision.

        A colour-equivariant encoder pools over the palette axis and produces
        bitwise identical logits under a colour swap. In LS20 colour is one
        third of the goal triple, so that symmetry would destroy the task.
        """
        torch.manual_seed(0)
        frames = torch.randint(0, policy.PALETTE, (4, names.FRAME_SIZE, names.FRAME_SIZE))
        swapped = frames.clone()
        first, second = names.COLORS[0], names.COLORS[1]  # Two really-used carry colours.
        swapped[frames == first] = second
        swapped[frames == second] = first
        with torch.no_grad():
            self.assertFalse(torch.equal(self.model(frames), self.model(swapped)))

    def test_hud_broadcasting_changes_the_network_and_still_round_trips(self):
        conditioned = policy.build_policy({"broadcast_hud": True}).eval()
        self.assertGreater(conditioned.parameter_count(), self.model.parameter_count())
        self.assertLess(conditioned.parameter_count(), PARAMETER_BUDGET)
        frames = torch.randint(0, policy.PALETTE, (2, names.FRAME_SIZE, names.FRAME_SIZE))
        with torch.no_grad():
            self.assertEqual(conditioned(frames).shape, (2, len(names.ACTION_IDS)))
        rebuilt = policy.build_policy(conditioned.config())
        self.assertTrue(rebuilt.config()["broadcast_hud"])
        self.assertEqual(rebuilt.parameter_count(), conditioned.parameter_count())

    def test_config_rebuilds_the_same_network(self):
        rebuilt = policy.build_policy(self.model.config())
        self.assertEqual(rebuilt.config(), self.model.config())
        self.assertEqual(rebuilt.parameter_count(), self.model.parameter_count())
        self.assertEqual([tuple(p.shape) for p in rebuilt.parameters()],
                         [tuple(p.shape) for p in self.model.parameters()])

    def test_forward_rejects_frames_that_are_not_64x64_batches(self):
        for shape in ((64, 64), (2, 32, 32), (2, 64, 64, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "frames must be"):
                    self.model(torch.zeros(shape, dtype=torch.int64))

    def test_frames_to_tensor_accepts_one_frame_and_a_batch(self):
        frame = [[(row + col) % policy.PALETTE for col in range(names.FRAME_SIZE)]
                 for row in range(names.FRAME_SIZE)]
        single = policy.frames_to_tensor(frame)
        self.assertEqual(single.shape, (1, names.FRAME_SIZE, names.FRAME_SIZE))
        self.assertEqual(single.dtype, torch.int64)
        batch = policy.frames_to_tensor(np.zeros((3, names.FRAME_SIZE, names.FRAME_SIZE), dtype=np.uint8))
        self.assertEqual(batch.shape, (3, names.FRAME_SIZE, names.FRAME_SIZE))
        self.assertEqual(batch.dtype, torch.int64)
        with torch.no_grad():
            self.model(single)  # What it returns must be forward-ready.

    def test_one_optimizer_step_changes_weights(self):
        # A deliberately tiny trunk: this asserts that gradients flow, not that
        # the real network fits.
        torch.manual_seed(0)
        model = policy.build_policy(TINY)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        frames = torch.randint(0, policy.PALETTE, (4, names.FRAME_SIZE, names.FRAME_SIZE))
        actions = torch.tensor([0, 1, 2, 3])
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        loss = torch.nn.functional.cross_entropy(model(frames), actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertGreater(model.action.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.stem[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        changed = [name for name, parameter in model.named_parameters()
                   if not torch.equal(before[name], parameter.detach())]
        for name in ("action.weight", "stem.0.weight", "hud.0.weight"):
            self.assertIn(name, changed)

    def test_checkpoint_round_trip_preserves_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            model = policy.build_policy(TINY).eval()
            policy.save_checkpoint(path, model, train_seeds=[0, 1], validation_seeds=[2],
                                   epochs=3, generator_version="test")
            restored, checkpoint = policy.load_checkpoint(path)
            self.assertEqual(checkpoint["format"], policy.MODEL_FORMAT)
            self.assertEqual(checkpoint["config"], model.config())
            self.assertEqual(checkpoint["train_seeds"], [0, 1])
            self.assertEqual(checkpoint["validation_seeds"], [2])
            self.assertEqual(checkpoint["epochs"], 3)
            self.assertEqual(checkpoint["parameters"], model.parameter_count())
            frames = torch.randint(0, policy.PALETTE, (3, names.FRAME_SIZE, names.FRAME_SIZE))
            with torch.no_grad():
                self.assertTrue(torch.equal(model(frames), restored(frames)))
            self.assertFalse(restored.training)  # A loaded checkpoint must not be in train mode.
            torch.save({"format": "something-else", "weights": {}}, path)
            with self.assertRaisesRegex(ValueError, "formats differ"):
                policy.load_checkpoint(path)


class ShardTests(unittest.TestCase):
    def test_npz_round_trip_preserves_dtypes_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = fake_shard([11, 12, 13])
            path = data.write_shard(Path(directory) / "shard.npz", shard)
            self.assertTrue(Path(path).exists())
            restored = data.read_shard(path)
            for key, dtype in (("frames", np.uint8), ("actions", np.uint8), ("optimal", np.uint8),
                               ("to_go", np.int16), ("seeds", np.int32)):
                with self.subTest(key=key):
                    self.assertEqual(restored[key].dtype, dtype)
                    np.testing.assert_array_equal(restored[key], shard[key])
            self.assertEqual(restored["meta"]["format"], data.DATA_FORMAT)
            self.assertEqual(restored["meta"]["seeds"], [11, 12, 13])
            json.dumps(restored["meta"], allow_nan=False)

    def test_load_shards_concatenates_and_rejects_a_foreign_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = data.write_shard(root / "a.npz", fake_shard([1, 2], rng_seed=1))
            second = data.write_shard(root / "b.npz", fake_shard([3], rng_seed=2))
            loaded = data.load_shards([first, second])
            self.assertEqual(len(loaded["frames"]), 3 * 7)
            self.assertEqual(len(loaded["meta"]), 2)
            self.assertEqual(sorted(set(loaded["seeds"].tolist())), [1, 2, 3])
            self.assertEqual(loaded["frames"].dtype, np.uint8)
            self.assertEqual(len(loaded["to_go"]), 3 * 7)  # Kept for analysis, unused by training.
            stale = fake_shard([9])
            stale["meta"]["format"] = "pebby.ls20-oracle.v0"
            data.write_shard(root / "c.npz", stale)
            with self.assertRaises(ValueError):
                data.load_shards([first, root / "c.npz"])

    @unittest.skipIf(GENERATOR, "the generator exists, so the degraded path is unreachable")
    def test_generation_fails_with_a_message_naming_the_missing_module(self):
        with self.assertRaises(RuntimeError) as raised:
            data.generate_shard(seeds=[0], difficulty=1, epsilon=0.)
        self.assertRegex(str(raised.exception), r"pebby\.ls20\.(generate|plan)")


class SplitAndBaselineTests(unittest.TestCase):
    def test_the_split_never_straddles_a_level_seed(self):
        for levels, per_level in ((20, 5), (7, 13), (2, 3)):
            with self.subTest(levels=levels):
                seeds = np.repeat(np.arange(levels, dtype=np.int32), per_level)
                train_mask, validation_mask, train_seeds, validation_seeds = trainer.split_by_seed(seeds)
                self.assertTrue((train_mask ^ validation_mask).all())  # A partition, not a sample.
                self.assertTrue(train_seeds and validation_seeds)
                self.assertEqual(set(train_seeds) & set(validation_seeds), set())
                self.assertEqual(sorted(train_seeds + validation_seeds), list(range(levels)))
                self.assertEqual(set(seeds[train_mask].tolist()), set(train_seeds))
                self.assertEqual(set(seeds[validation_mask].tolist()), set(validation_seeds))

    def test_a_single_level_reports_no_validation_rather_than_leaking(self):
        train_mask, validation_mask, train_seeds, validation_seeds = trainer.split_by_seed(
            np.zeros(9, dtype=np.int32))
        self.assertTrue(train_mask.all())
        self.assertFalse(validation_mask.any())
        self.assertEqual((train_seeds, validation_seeds), ([0], []))

    def test_the_split_is_deterministic_for_a_seed(self):
        seeds = np.repeat(np.arange(10, dtype=np.int32), 3)
        first = trainer.split_by_seed(seeds, rng_seed=5)[3]
        self.assertEqual(first, trainer.split_by_seed(seeds, rng_seed=5)[3])
        self.assertNotEqual(first, trainer.split_by_seed(seeds, rng_seed=6)[3])

    def test_the_target_puts_equal_mass_on_every_optimal_action(self):
        """30% of oracle labels are one of several equally optimal moves, so a
        single-label target docks the model for playing correctly."""
        targets = trainer.optimal_targets(np.array([0b0001, 0b0011, 0b1111], dtype=np.uint8))
        np.testing.assert_allclose(targets, [[1, 0, 0, 0], [.5, .5, 0, 0], [.25] * 4])
        np.testing.assert_allclose(targets.sum(1), 1.)
        # A single optimal action must reduce to exactly one-hot, so a shard
        # without ties trains identically to plain cross-entropy.
        single = trainer.optimal_targets((1 << np.arange(4)).astype(np.uint8))
        np.testing.assert_array_equal(single, np.eye(4, dtype=np.float32))

    def test_the_prior_baseline_scores_a_frame_blind_predictor(self):
        actions = np.array([0, 0, 0, 0, 1, 1, 2, 3], dtype=np.int64)
        prior = trainer.label_prior(actions)
        np.testing.assert_allclose(prior, [.5, .25, .125, .125])
        scores = trainer.prior_scores(prior, actions)
        self.assertAlmostEqual(scores["accuracy"], .5)  # Always guess the majority action.
        self.assertAlmostEqual(scores["cross_entropy"], float(-np.log(prior[actions]).mean()))
        # A uniform prior must score exactly ln(4); anything below that on a
        # balanced label set would mean the baseline itself was miscomputed.
        uniform = trainer.label_prior(np.array([0, 1, 2, 3]))
        self.assertAlmostEqual(trainer.prior_scores(uniform, np.array([0, 1, 2, 3]))["cross_entropy"],
                               float(np.log(4)))

    def test_a_model_that_ties_the_prior_is_reported_as_having_learned_nothing(self):
        baseline = {"train": {"cross_entropy": 1.65815, "accuracy": .3},
                    "validation": {"cross_entropy": 1.65815, "accuracy": .3}}
        worse = trainer.verdict([{"epoch": 1, "validation": {"cross_entropy": 1.66739, "accuracy": .57}}], baseline)
        self.assertFalse(worse["beats_prior"])
        self.assertIn("does NOT beat", worse["message"])
        # Accuracy can beat the prior while cross-entropy does not; the verdict
        # must show both rather than let one of them tell the whole story.
        self.assertIn("0.5700", worse["message"])
        better = trainer.verdict([{"epoch": 1, "validation": {"cross_entropy": .4, "accuracy": .8}}], baseline)
        self.assertTrue(better["beats_prior"])
        self.assertIn("beats", better["message"])

    def test_the_verdict_judges_the_kept_epoch_not_the_last_one(self):
        history = [{"epoch": 1, "validation": {"cross_entropy": .5, "accuracy": .8}},
                   {"epoch": 2, "validation": {"cross_entropy": 2.5, "accuracy": .7}}]
        baseline = {"train": {"cross_entropy": 1.3, "accuracy": .3},
                    "validation": {"cross_entropy": 1.3, "accuracy": .3}}
        self.assertTrue(trainer.verdict(history, baseline, {"epoch": 1})["beats_prior"])
        self.assertFalse(trainer.verdict(history, baseline)["beats_prior"])  # Last epoch overfitted.

    def test_validation_shards_sharing_a_level_with_training_are_refused(self):
        first = {"seeds": np.array([1, 1, 2], dtype=np.int32)}
        trainer.disjoint_seeds(first, {"seeds": np.array([7, 8], dtype=np.int32)})  # Disjoint: fine.
        with self.assertRaisesRegex(ValueError, "both the training and validation"):
            trainer.disjoint_seeds(first, {"seeds": np.array([2, 9], dtype=np.int32)})


@unittest.skipUnless(GENERATOR, NO_GENERATOR)
class ShippedLevelTests(unittest.TestCase):
    """The seven levels the real game ships are the ultimate target."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shipped_optima_come_from_the_cache_and_never_from_a_search(self):
        """Planning shipped level 7 costs 21.7M states and 6.8 GiB; the table must
        read `pebby.ls20.shipped` instead, and must do it for all seven."""
        from pebby.ls20 import shipped
        from pebby.agent import evaluate
        self.assertEqual(len(evaluate.shipped_levels()), 7)
        for index in range(7):
            with self.subTest(level=index + 1):
                optimum, reason = evaluate.level_optimum(index)
                self.assertEqual(optimum, shipped.OPTIMAL_ACTIONS[index])
                self.assertIn("cached", reason)
        # The teacher beats the human median everywhere, so a shipped failure is
        # the policy's, never a weak oracle's.
        for optimum, human in zip(shipped.OPTIMAL_ACTIONS, shipped.HUMAN_BASELINE):
            self.assertLess(optimum, human)

    def test_a_stub_policy_rolls_out_without_crashing_and_reports_json(self):
        from pebby.ls20.env import Ls20Env
        from pebby.agent import evaluate

        class Stub(torch.nn.Module):
            def forward(self, frames):
                return torch.zeros(len(frames), len(names.ACTION_IDS))

        run = evaluate.rollout(Stub(), Ls20Env(levels=[evaluate.shipped_levels()[0]]), 3)
        self.assertEqual(run["actions"], 3)
        self.assertIn(run["ending"], ("win", "game_over", "capped", "stuck"))
        self.assertEqual(run["goals_total"], 1)
        json.dumps(run, allow_nan=False)

    def test_sampling_varies_the_action_while_argmax_does_not(self):
        """Retries only mean something if the policy can play differently."""
        from pebby.agent import evaluate

        class Tilted(torch.nn.Module):
            def forward(self, frames):
                return torch.tensor([[1., .9, .8, .7]] * len(frames))

        frame = [[0] * names.FRAME_SIZE for _ in range(names.FRAME_SIZE)]
        greedy = {evaluate.choose_action(Tilted(), frame)[1] for _ in range(20)}
        self.assertEqual(greedy, {0})
        generator = torch.Generator().manual_seed(0)
        sampled = {evaluate.choose_action(Tilted(), frame, temperature=1., generator=generator)[1]
                   for _ in range(40)}
        self.assertGreater(len(sampled), 1)

    def test_the_budgeted_protocol_retries_inside_one_action_budget(self):
        """The benchmark scores against a budget with RESET and lives, not one
        flawless attempt; humans take 2-4x optimal on these levels."""
        from pebby.ls20 import shipped
        from pebby.agent import evaluate

        class Stub(torch.nn.Module):
            def forward(self, frames):
                return torch.zeros(len(frames), len(names.ACTION_IDS))

        report = evaluate.budgeted_completion(Stub(), evaluate.shipped_levels()[:1],
                                              shipped.HUMAN_BASELINE[:1], multiplier=.5,
                                              temperature=.5)
        run = report["runs"][0]
        self.assertLessEqual(run["actions"], run["budget"])
        self.assertGreaterEqual(run["attempts"], 1)
        self.assertEqual(len(run["endings"]), run["attempts"])
        self.assertEqual(report["protocol"], "budgeted")
        json.dumps(report, allow_nan=False)

    def test_a_stalling_policy_is_not_allowed_to_repeat_a_dead_action(self):
        """A refused goal-pad bump costs no budget and leaves the frame identical,
        so plain greedy argmax repeats it until the cap. Measured with a trained
        policy: 74.2% of all actions produced a byte-identical frame, every
        checked one a zero-budget bump against a goal pad."""
        from pebby.ls20.env import Ls20Env
        from pebby.agent import evaluate

        class Constant(torch.nn.Module):
            def forward(self, frames):
                return torch.tensor([[9., 1., 1., 1.]] * len(frames))

        level = evaluate.shipped_levels()[0]
        repeating = evaluate.rollout(Constant(), Ls20Env(levels=[level]), 30, on_stall="repeat")
        masking = evaluate.rollout(Constant(), Ls20Env(levels=[level]), 30)
        self.assertEqual(repeating["on_stall"], "repeat")
        self.assertEqual(masking["on_stall"], "next-best")
        self.assertGreater(repeating["stalls"], masking["stalls"])


class CommandLineTests(unittest.TestCase):
    def test_every_entry_point_imports_and_parses_arguments(self):
        # --help exercises the whole import chain without a GPU.
        for module in ("pebby.agent.data", "pebby.agent.train", "pebby.agent.evaluate"):
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-m", module, "--help"],
                                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage", result.stdout)

    def test_training_refuses_a_missing_shard(self):
        result = subprocess.run([sys.executable, "-m", "pebby.agent.train", "--shards", "/nonexistent.npz"],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing shards", result.stderr)


@unittest.skipUnless(GENERATOR, NO_GENERATOR)
class EndToEndTests(unittest.TestCase):
    """Real generated levels, real oracle labels, real rollout."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle_shard_trains_and_evaluates_on_generated_levels(self):
        from pebby.ls20 import generate
        from pebby.agent import evaluate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = list(range(3))
            shard = data.generate_shard(seeds=seeds, difficulty=1, epsilon=0., seed=0, workers=1)
            self.assertGreater(len(shard["frames"]), 0)
            self.assertEqual(shard["frames"].dtype, np.uint8)
            self.assertTrue(set(shard["seeds"].tolist()) <= set(seeds))
            self.assertTrue(((shard["actions"] >= 0) & (shard["actions"] < len(names.ACTION_IDS))).all())
            path = data.write_shard(root / "shard.npz", shard)
            result = subprocess.run(
                [sys.executable, "-m", "pebby.agent.train", "--shards", str(path), "--epochs", "1",
                 "--batch-size", "16", "--channels", "16", "--blocks", "1", "--device", "cpu",
                 "--loader-workers", "0", "--checkpoint-out", str(root / "policy.pt")],
                cwd=ROOT, capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("prior", result.stdout)  # The baseline is never optional.
            model, _ = policy.load_checkpoint(root / "policy.pt")
            levels, oracles = evaluate.generated_levels(1, 1, 900_000)
            report = evaluate.completion_rate(model, levels, max_actions=60, oracles=oracles)
            self.assertEqual(report["levels"], 1)
            self.assertLessEqual(report["completion_rate"], 1.)
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
