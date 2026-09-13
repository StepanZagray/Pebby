"""CPU-only cache preparation checks using synthetic arrays, never the real parent."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import cache_reference_outcome_inputs as cache


class SelectionTests(unittest.TestCase):
    def fixture(self):
        seeds = np.repeat([101, 102], 8).astype(np.int32)
        position = dict(source=np.r_[np.zeros(32, np.uint8), 1],
                        branch=np.r_[np.tile([-1, 0], 16), -1],
                        rows=np.r_[np.repeat(np.arange(16), 2), 0],
                        seeds=np.r_[np.repeat(seeds, 2), 101])
        return position, {"seeds": seeds}, np.arange(16, dtype=np.int64)

    def test_uses_only_original_current_rows_and_every_seed_eight_times(self):
        position, source, fixed = self.fixture()
        rows, indices, seeds = cache.select_current(position, source, 16, 2, fixed)
        np.testing.assert_array_equal(rows, fixed)
        np.testing.assert_array_equal(indices, np.arange(0, 32, 2))
        np.testing.assert_array_equal(seeds, [101, 102])

    def test_duplicate_source_rows_and_wrong_published_selection_fail(self):
        position, source, fixed = self.fixture()
        position["rows"][2] = 0
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cache.select_current(position, source, 16, 2, fixed)
        position, source, fixed = self.fixture()
        with self.assertRaisesRegex(ValueError, "published"):
            cache.select_current(position, source, 16, 2, fixed[::-1])

    def test_missing_source_level_and_wrong_cached_seed_fail(self):
        position, source, fixed = self.fixture()
        source["seeds"] = np.r_[source["seeds"], 103]
        with self.assertRaisesRegex(ValueError, "omits"):
            cache.select_current(position, source, 16, 2, fixed)
        position, source, fixed = self.fixture()
        position["seeds"][0] = 999
        with self.assertRaisesRegex(ValueError, "seeds disagree"):
            cache.select_current(position, source, 16, 2, fixed)


class FrozenEncodingTests(unittest.TestCase):
    def test_encoder_receives_only_three_current_public_arrays(self):
        import torch
        captured = []
        class Model:
            def encode(self, *args):
                captured.append(args)
                return {name: torch.zeros((len(args[0]), *cache.SCHEMA[name][1]), dtype=torch.float32)
                        for name in ("raw", "state", "glyph")}
        arrays = dict(frames=np.zeros((3, 8, 64, 64), np.uint8),
                      history_valid=np.ones((3, 8), bool), previous_actions=np.full((3, 8), -1, np.int64),
                      next_frames=object(), labels=object())
        with torch.inference_mode():
            result = cache.encode_current(Model(), arrays, np.array([2, 0]), "cpu")
        self.assertEqual(len(captured[0]), 3)
        self.assertEqual(tuple(captured[0][0].shape), (2, 8, 64, 64))
        self.assertEqual(tuple(result["raw"].shape), (2, 160, 64))
        self.assertEqual(tuple(result["glyph"].shape), (2, 14))

    def test_original_projector_uses_only_cached_components_and_rejects_position_extra(self):
        import torch
        from unittest.mock import Mock
        model = Mock()
        model.projector_inputs.return_value = torch.zeros((2, 1742))
        encoding = dict(state=torch.zeros((2, 160, 64)), raw=torch.zeros((2, 160, 64)), glyph=torch.zeros((2, 14)))
        self.assertEqual(tuple(cache.original_inputs(model, encoding).shape), (2, 1742))
        self.assertEqual(tuple(model.projector_inputs.call_args.args[1].shape), (2, 1024))
        model.projector_inputs.return_value = torch.zeros((2, 1766))
        with self.assertRaisesRegex(ValueError, "1742"):
            cache.original_inputs(model, encoding)


class IntegrityTests(unittest.TestCase):
    def test_bindings_hash_once_and_reject_stat_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"unchanged")
            bindings = cache.Bindings()
            with patch.object(cache, "sha", wraps=cache.sha) as digest:
                expected = bindings.add(path)
                bindings.add(path, expected)
                bindings.verify()
                self.assertEqual(digest.call_count, 1)
            path.write_bytes(b"different content")
            with self.assertRaisesRegex(ValueError, "stats changed"):
                bindings.verify()

    def test_publication_never_replaces_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            staging, output = Path(directory) / "stage", Path(directory) / "published"
            staging.mkdir(); output.mkdir()
            (staging / "new").write_text("new")
            (output / "old").write_text("old")
            with self.assertRaises(FileExistsError):
                cache.publish_directory(staging, output)
            self.assertTrue((output / "old").exists())
            target = Path(directory) / "fresh"
            cache.publish_directory(staging, target)
            self.assertTrue((target / "new").exists())
            self.assertFalse(staging.exists())

    def test_cli_refuses_overwrite_before_reading_parent(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cache, "sha") as digest:
            with self.assertRaises(FileExistsError):
                cache.main(["--out-dir", directory])
            digest.assert_not_called()

    def make_published(self, root, *, overlap=False):
        counts = {"train": (16, 2), "validation": (8, 1)}
        report = dict(status="complete", parent_sha256=cache.PARENT_SHA, sources_unchanged=True,
                      validation_disjoint=True, public_inputs=list(cache.PUBLIC), current_source_id=0,
                      current_branch=-1, arrays={}, selection={})
        for split, (count, _) in counts.items():
            directory = root / split
            directory.mkdir()
            rows = np.arange(count, dtype=np.int64)
            seeds = (np.repeat([101, 102], 8) if split == "train" else np.repeat(101 if overlap else 201, 8)).astype(np.int64)
            report["selection"][split] = dict(source_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
            report["arrays"][split] = {}
            for key, (dtype, tail) in cache.SCHEMA.items():
                values = seeds if key == "seeds" else rows if key == "rows" else np.zeros((count, *tail), dtype=dtype)
                path = directory / f"{key}.npy"
                np.save(path, values)
                report["arrays"][split][key] = dict(shape=list(values.shape), dtype=values.dtype.str,
                                                     sha256=cache.sha(path), size_bytes=path.stat().st_size)
        (root / "manifest.json").write_text(json.dumps(report))
        return counts

    def test_published_validator_checks_hashes_shapes_seed_disjointness_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = self.make_published(root)
            with patch.object(cache, "COUNTS", counts):
                result = cache.validate_published(root)
                self.assertEqual(len(result["validated_output_hashes"]), 1 + 2 * len(cache.SCHEMA))
                path = root / "train/raw.npy"
                with path.open("r+b") as handle:
                    handle.seek(-4, 2); handle.write(b"oops")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    cache.validate_published(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = self.make_published(root, overlap=True)
            with patch.object(cache, "COUNTS", counts), self.assertRaisesRegex(ValueError, "overlap"):
                cache.validate_published(root)


if __name__ == "__main__":
    unittest.main()
