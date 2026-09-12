import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pebby.ls20.generate import FORMAT, GENERATOR_VERSION
from tools.collect_workspace_onpolicy import (
    _cached_seed_array,
    _load_allowed_seeds,
    _validate_bank,
    select_sources,
)
from tools.goal_attribute_probes import digest


def _write_bank(path, start):
    rows = []
    for difficulty in range(1, 6):
        for offset in range(2):
            rows.append({
                "format": FORMAT,
                "generator_version": GENERATOR_VERSION,
                "seed": start + difficulty * 10 + offset,
                "difficulty": difficulty,
                "launchers": [],
                "official_inputs_used": False,
                "training_context_index": (start + difficulty * 10 + offset) % 7,
                "context_engine_verified": True,
                "search_truncated": False,
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def _write_seed_cache(root, source, seeds):
    cache = root / "cache"
    cache.mkdir()
    manifest_dir = cache / "bound"
    manifest_dir.mkdir()
    seed_path = manifest_dir / "seeds.npy"
    np.save(seed_path, np.asarray(seeds, dtype=np.int64))
    manifest = {
        "source_sha256": digest(source),
        "arrays": {"seeds": {
            "dtype": "<i8", "shape": [len(seeds)], "sha256": digest(seed_path),
        }},
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
    return cache


class WorkspaceOnPolicyCollectorTests(unittest.TestCase):
    def test_mmap_seed_cache_requires_matching_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            combined = root / "combined.npz"
            combined.write_bytes(b"compressed source placeholder")
            cache = _write_seed_cache(root, combined, [10, 11, 12])
            source_sha, found = _cached_seed_array(combined, cache)
            self.assertEqual(source_sha, digest(combined))
            manifest, seed_path, _, seeds = found
            self.assertEqual(manifest.name, "manifest.json")
            self.assertEqual(seed_path.name, "seeds.npy")
            self.assertEqual(seeds.tolist(), [10, 11, 12])

            payload = json.loads(manifest.read_text())
            payload["arrays"]["seeds"]["shape"] = [99]
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                _cached_seed_array(combined, cache)

    def test_allowed_seed_intersection_is_mmap_bound_and_nonempty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            combined = root / "combined.npz"
            combined.write_bytes(b"source")
            cache = _write_seed_cache(root, combined, [1, 2, 3, 4])
            eligible = root / "eligible.npy"
            np.save(eligible, np.asarray([3, 4, 9], dtype=np.int64))
            allowed, metadata = _load_allowed_seeds(combined, eligible, cache)
            self.assertEqual(allowed, {3, 4})
            self.assertEqual(metadata["intersection_levels"], 2)
            self.assertEqual(metadata["combined_seed_cache"]["rows"], 4)

    def test_source_selection_is_disjoint_and_stratified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.jsonl"
            extended = root / "extended.jsonl"
            old_rows = _write_bank(old, 100)
            extended_rows = _write_bank(extended, 1000)
            combined = root / "combined.npz"
            combined.write_bytes(b"source")
            all_seeds = [row["seed"] for row in old_rows + extended_rows]
            cache = _write_seed_cache(root, combined, all_seeds)
            eligible = root / "eligible.npy"
            np.save(eligible, np.asarray(all_seeds, dtype=np.int64))

            specs, records, membership = select_sources(
                old, extended, combined, eligible, cache, per_source=5,
                selection_seed=1234,
            )
            self.assertEqual(len(specs), 10)
            self.assertEqual(len({row["seed"] for row in specs}), 10)
            self.assertEqual([record["name"] for record in records], ["original", "extended"])
            self.assertEqual(records[0]["selected_difficulties"], {1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
            self.assertEqual(records[1]["selected_difficulties"], {1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
            self.assertEqual(membership["intersection_levels"], 20)

    def test_bank_rejects_official_or_wrong_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bank.jsonl"
            row = {"format": FORMAT, "generator_version": GENERATOR_VERSION,
                   "seed": 1, "difficulty": 1, "official_inputs_used": True}
            with self.assertRaisesRegex(ValueError, "official"):
                _validate_bank(path, [row])
            row["official_inputs_used"] = False
            row["format"] = "official"
            with self.assertRaisesRegex(ValueError, "current generated TRAIN"):
                _validate_bank(path, [row])


if __name__ == "__main__":
    unittest.main()
