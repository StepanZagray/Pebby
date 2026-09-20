"""CPU-only replay tests using deliberately shuffled source/outcome row IDs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pebby.agent.full_game_replay import PUBLIC, PUBLIC_SCHEMA, TARGET_SCHEMA, RawOutcomeReplay, _stat


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "outcomes"
        (self.cache / "train").mkdir(parents=True)
        self.source_sha = "a" * 64
        self.source = self.root / (self.source_sha + "-fixture")
        self.source.mkdir()
        self.bank = self.root / "train.jsonl"
        self.bank.write_text(''.join(json.dumps(dict(seed=seed, difficulty=tier, private_geometry="ignored")) + '\n'
                                     for seed, tier in [(10, 1), (20, 2), (30, 2)]))
        self.rows = np.array([8, 3, 7, 0, 5, 1], dtype=np.int64)
        self.seeds = np.array([10, 20, 30, 10, 20, 30], dtype=np.int64)
        source_seeds = np.zeros(10, dtype=np.int32)
        source_seeds[self.rows] = self.seeds
        source_info = {}
        for key, (dtype, tail) in PUBLIC_SCHEMA.items():
            values = np.zeros((10, *tail), dtype=dtype)
            for row in range(10):
                values[row] = row if key != "history_valid" else row % 2
            source_info[key] = self.save(self.source, key, values)
        source_info["seeds"] = self.save(self.source, "seeds", source_seeds)
        (self.source / "manifest.json").write_text(json.dumps(dict(source_sha256=self.source_sha, arrays=source_info)))
        arrays = {}
        for key, (dtype, tail) in {"rows": ("int64", ()), "seeds": ("int64", ()), **TARGET_SCHEMA}.items():
            values = np.zeros((6, *tail), dtype=dtype)
            if key == "rows":
                values = self.rows
            elif key == "seeds":
                values = self.seeds
            elif key == "next_steps":
                values[:] = np.arange(6)[:, None]
            elif key == "current_steps":
                values[:] = 99
            elif key == "won":
                values[0, 1] = True
                values[2, 3] = True
            arrays[key] = self.save(self.cache / "train", key, values)
        bindings = {str(path): digest(path) for path in self.source.iterdir()}
        bindings[str(self.root / "train.npz")] = self.source_sha
        self.manifest = dict(status="complete", sources_unchanged=True, validation_disjoint=True,
                             public_inputs=list(PUBLIC), current_source_id=0, current_branch=-1,
                             official_frames_or_routes_used=False,
                             source_sha256=bindings,
                             source_stats_after={str(path): _stat(path) for path in self.source.iterdir()},
                             arrays={"train": arrays}, selection={"train": dict(rows=6, levels=3, roots_per_level=2,
                                source_rows_sha256=hashlib.sha256(self.rows.tobytes()).hexdigest())})
        self.write_manifest()

    def save(self, root, key, values):
        path = root / f"{key}.npy"
        np.save(path, values)
        return dict(shape=list(values.shape), dtype=values.dtype.str, sha256=digest(path))

    def write_manifest(self):
        (self.cache / "manifest.json").write_text(json.dumps(self.manifest))

    def replay(self, **kwargs):
        return RawOutcomeReplay(self.cache, "train", source_root=self.source, bank_path=self.bank, **kwargs)

    def test_alignment_public_boundary_and_zero_optimal_retained(self):
        with self.replay() as replay:
            public, target = replay.batch(np.array([4, 0, 4, 2]))
            self.assertEqual(set(public), set(PUBLIC))
            self.assertEqual(set(target), set(TARGET_SCHEMA))
            np.testing.assert_array_equal(public["frames"][:, 0, 0, 0], [5, 8, 5, 7])
            np.testing.assert_array_equal(target["next_steps"][:, 0], [4, 0, 4, 2])
            self.assertEqual(int(target["optimal"].sum()), 0)
            self.assertEqual(replay.metadata["zero_optimal_rows_retained"], 6)
            self.assertFalse(replay._arrays["source_frames"].flags.writeable)
            self.assertNotIn("raw", replay._arrays)
            self.assertNotIn("state", replay._arrays)
            self.assertNotIn("glyph", replay._arrays)
            json.dumps(replay.metadata, allow_nan=False)

    def test_tensor_mutation_does_not_mutate_mmaps_and_batches_survive_close(self):
        replay = self.replay(drop_pages_every=1)
        public, target = replay.batch([0])
        public["frames"].fill_(77)
        target["optimal"].fill_(15)
        again, labels = replay.batch([0])
        self.assertEqual(int(again["frames"][0, 0, 0, 0]), 8)
        self.assertEqual(int(labels["optimal"][0]), 0)
        mappings = [array._mmap for array in replay._arrays.values()]
        replay.close()
        replay.close()
        self.assertTrue(all(backing.closed for backing in mappings))
        self.assertEqual(int(public["frames"][0, 0, 0, 0]), 77)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            replay.batch([0])

    def test_sampling_tiers_levels_events_and_determinism(self):
        with self.replay() as replay:
            rng = np.random.default_rng(31)
            sampled = np.concatenate([replay.sample(1000, rng, {2: 1.}, event_fraction=1) for _ in range(8)])
            self.assertTrue(np.all(replay.tiers[sampled] == 2))
            # Level 20 has no events: it stays equally represented using normal roots.
            self.assertLess(abs(np.mean(replay.seeds[sampled] == 20) - .5), .025)
            self.assertTrue(np.all(sampled[replay.seeds[sampled] == 30] == 2))
            default = replay.sample(4000, rng, event_fraction=0)
            self.assertLess(abs(np.mean(replay.tiers[default] == 1) - .5), .035)
            np.testing.assert_array_equal(replay.sample(100, np.random.default_rng(9)),
                                          replay.sample(100, np.random.default_rng(9)))
            self.assertEqual(set(np.unique(default)), set(range(6)))

    def test_invalid_indices_and_sampling_parameters(self):
        with self.replay(max_batch_size=8) as replay:
            for indices in ([.5], [[0]], [True], list(range(9))):
                with self.assertRaises(ValueError):
                    replay.batch(indices)
            for indices in ([-1], [6], np.array([2**64 - 1], dtype=np.uint64)):
                with self.assertRaises(IndexError):
                    replay.batch(indices)
            for kwargs in (dict(batch_size=0), dict(batch_size=9), dict(event_fraction=1.1),
                           dict(tier_weights={3: 1}), dict(tier_weights={1: -1}),
                           dict(tier_weights={1: float("nan")}), dict(tier_weights={1: 0})):
                with self.assertRaises(ValueError):
                    replay.sample(rng=np.random.default_rng(0), **(dict(batch_size=2) | kwargs))

    def test_hashes_small_arrays_and_optional_raw_pixels(self):
        with self.replay() as replay:
            self.assertNotIn(str(self.source / "frames.npy"), replay.metadata["verified_sha256"])
            self.assertIn(str(self.source / "previous_actions.npy"), replay.metadata["verified_sha256"])
            self.assertEqual(replay.metadata["pixel_binding"], "recorded_stats_only")
        with self.replay(verify_hashes=True) as replay:
            self.assertIn(str(self.source / "frames.npy"), replay.metadata["verified_sha256"])
        path = self.cache / "train/next_steps.npy"
        with path.open("r+b") as handle:
            handle.seek(-2, 2)
            handle.write(b"xx")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.replay()

    def test_raw_stats_mutation_rejected_at_open_and_during_use(self):
        with self.replay() as replay:
            path = self.source / "frames.npy"
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "stats changed"):
                replay.batch([0])
        with self.assertRaisesRegex(ValueError, "stats changed"):
            self.replay()

    def test_device_number_change_is_explicit_but_other_stat_changes_fail(self):
        for key in (*PUBLIC, "seeds"):
            self.manifest["source_stats_after"][str(self.source / f"{key}.npy")][0] += 100
        self.write_manifest()
        with self.replay() as replay:
            self.assertEqual(set(replay.metadata["source_device_number_changes"]), {*PUBLIC, "seeds"})
        self.manifest["source_stats_after"][str(self.source / "frames.npy")][1] += 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "stats changed"):
            self.replay()

    def test_bank_changes_during_replay_fail(self):
        with self.replay() as replay:
            self.bank.write_text(self.bank.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "stats changed"):
                replay.sample(2, np.random.default_rng(0))

    def test_wrong_alignment_even_with_updated_target_hash_rejected(self):
        bad = self.seeds.copy()
        bad[0] = 30
        self.manifest["arrays"]["train"]["seeds"] = self.save(self.cache / "train", "seeds", bad)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.replay()

    def test_partial_initialization_failure_closes_mmaps(self):
        opened = []
        original = np.load
        def load(*args, **kwargs):
            value = original(*args, **kwargs)
            opened.append(value._mmap)
            return value
        self.bank.write_text('{"seed":10,"difficulty":8}\n')
        with patch("pebby.agent.full_game_replay.np.load", side_effect=load):
            with self.assertRaisesRegex(ValueError, "bank"):
                self.replay()
        self.assertTrue(opened)
        self.assertTrue(all(backing.closed for backing in opened))

    def test_guard_invoked_during_open_and_batch(self):
        calls = []
        with self.replay(guard=lambda: calls.append(1)) as replay:
            before = len(calls)
            replay.batch([0])
            self.assertGreater(len(calls), before)


if __name__ == "__main__":
    unittest.main()
