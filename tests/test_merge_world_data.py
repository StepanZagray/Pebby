import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.merge_world_data import MergeError, merge
from tools import merge_world_data as merger


FORMAT = "pebby.ls20-world-transitions.v1"


class MergeWorldDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pebby-merge-world-")
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.audit = self.root / "validity.json"
        self._write_audit({10: (True, 1), 11: (False, 2), 12: (True, 3)})

    def tearDown(self):
        self.assertEqual(list(self.root.glob(".merge-world-data-*")), [])

    @staticmethod
    def _proof(seed):
        return {"context_engine_verified": True, "context_index": seed % 7,
                "search_truncated": False, "engine_win": True,
                "lives": 3, "completed": 1}

    def _write_audit(self, records):
        self.audit.write_text(json.dumps({
            "format": "pebby.ls20-curriculum-validity.v1",
            "status": "complete",
            "split": "train",
            "levels": [
                {"seed": seed, "accepted": accepted, "difficulty": difficulty,
                 "proof": self._proof(seed)}
                for seed, (accepted, difficulty) in records.items()
            ],
        }))

    def _write_shard(self, name, seeds, *, shape=(2,), dtype=np.int16, levels=None,
                     context_values=None):
        path = self.root / name
        seeds = np.asarray(seeds, dtype=np.int32)
        count = len(seeds)
        values = np.arange(count * int(np.prod(shape)), dtype=dtype).reshape((count, *shape))
        meta_levels = []
        for seed in sorted(set(int(value) for value in seeds)):
            meta_levels.append({"seed": seed, "difficulty": seed - 9,
                                "samples": int(np.count_nonzero(seeds == seed)),
                                **self._proof(seed)})
        if levels is not None:
            meta_levels = levels
        meta = {"format": FORMAT, "source": "generated_only",
                "oracle_search": "complete_only", "levels": meta_levels}
        arrays = {"frames": values, "optimal": values[:, 0], "seeds": seeds}
        if context_values is not None:
            arrays["context_index"] = np.asarray(context_values, dtype=np.int8)
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays, meta=np.array(json.dumps(meta)))
        return path

    def test_filters_rejected_rows_preserves_arrays_and_attaches_proof(self):
        first = self._write_shard("first.npz", [10, 10, 11])
        second = self._write_shard("second.npz", [12, 12])
        output = self.root / "merged.npz"
        output_meta = merge([first, second], output, self.audit, "train", min_levels=2)

        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["seeds"], [10, 10, 12, 12])
            np.testing.assert_array_equal(archive["frames"], [[0, 1], [2, 3], [0, 1], [2, 3]])
            np.testing.assert_array_equal(archive["optimal"], [0, 2, 0, 2])
            meta = json.loads(str(archive["meta"]))
        self.assertEqual(output_meta["samples"], 4)
        self.assertEqual(meta["seeds"], [10, 12])
        self.assertEqual([level["seed"] for level in meta["levels"]], [10, 12])
        self.assertEqual(meta["levels"][0]["difficulty"], 1)
        self.assertEqual(meta["levels"][0]["proof"]["context_index"], 3)
        self.assertTrue(meta["levels"][0]["context_engine_verified"])
        self.assertEqual(meta["levels"][0]["samples"], 2)
        self.assertEqual(meta["audit_sha256"], hashlib.sha256(self.audit.read_bytes()).hexdigest())
        self.assertEqual(meta["source_sha256"][str(first)], hashlib.sha256(first.read_bytes()).hexdigest())
        self.assertEqual(meta["source_sha256"][str(second)], hashlib.sha256(second.read_bytes()).hexdigest())

    def test_duplicate_seed_across_shards_is_rejected_and_cleans_temp(self):
        first = self._write_shard("first.npz", [10, 10])
        second = self._write_shard("second.npz", [10, 12])
        with self.assertRaisesRegex(MergeError, "more than one input shard"):
            merge([first, second], self.root / "merged.npz", self.audit, "train")
        self.assertFalse((self.root / "merged.npz").exists())

    def _add_arrays(self, path, **extra):
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        values.update(extra)
        np.savez_compressed(path, **values)

    def _sidecar(self, source, masks, **overrides):
        with np.load(source, allow_pickle=False) as archive:
            seeds = archive['seeds']
        meta = {'format': 'pebby.ls20-successor-labels.v1', 'source': 'generated_only',
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'rows': len(seeds), 'levels': len(np.unique(seeds)), **overrides}
        path = self.root / 'labels.npz'
        np.savez_compressed(path, seeds=seeds, next_optimal=masks, meta=np.array(json.dumps(meta)))
        return path

    def test_streams_filtered_rows_without_materializing_image_members(self):
        source = self._write_shard('stream.npz', [10, 11, 10, 12, 11, 12])
        output = self.root / 'streamed.npz'
        original_get = np.lib.npyio.NpzFile.__getitem__
        def guarded(archive, key):
            if key == 'frames':
                raise AssertionError('image members must be streamed, not loaded with NpzFile')
            return original_get(archive, key)
        with patch.object(np.lib.npyio.NpzFile, '__getitem__', guarded), \
                patch.object(merger, 'ROW_CHUNK_BYTES', 8):
            merge([source], output, self.audit, 'train')
        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['seeds'], [10, 10, 12, 12])
            np.testing.assert_array_equal(archive['frames'], [[0, 1], [4, 5], [6, 7], [10, 11]])

    def test_merges_bound_sidecar_with_embedded_labels_and_filters_both(self):
        first = self._write_shard('old.npz', [10, 11, 10])
        second = self._write_shard('new.npz', [12, 12])
        old_labels = np.array([[1, 2, 4, 8], [15, 0, 0, 0], [3, 5, 6, 9]], dtype=np.uint8)
        new_labels = np.array([[8, 4, 2, 1], [7, 7, 7, 7]], dtype=np.uint8)
        self._add_arrays(first, terminal=np.zeros((3, 4), dtype=bool))
        self._add_arrays(second, terminal=np.zeros((2, 4), dtype=bool), next_optimal=new_labels)
        labels = self._sidecar(first, old_labels)
        output = self.root / 'with-labels.npz'
        meta = merge([first, second], output, self.audit, 'train',
                     successor_sidecars={first: labels})
        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['next_optimal'], np.concatenate((old_labels[[0, 2]], new_labels)))
        self.assertEqual(meta['successor_label_sources'][str(first)]['sha256'],
                         hashlib.sha256(labels.read_bytes()).hexdigest())

    def test_sidecar_guards_source_hash_terminal_masks_and_row_order(self):
        source = self._write_shard('old.npz', [10, 11, 10])
        terminal = np.zeros((3, 4), dtype=bool)
        terminal[0, 0] = True
        self._add_arrays(source, terminal=terminal)
        masks = np.ones((3, 4), dtype=np.uint8)
        output = self.root / 'invalid.npz'
        for overrides, message in (({'source_sha256': 'wrong'}, 'hash mismatch'),
                                   ({'rows': 4}, 'rows must equal'),
                                   ({}, 'terminal successors')):
            with self.subTest(overrides=overrides):
                sidecar = self._sidecar(source, masks, **overrides)
                with self.assertRaisesRegex(MergeError, message):
                    merge([source], output, self.audit, 'train', successor_sidecars={source: sidecar})
                self.assertFalse(output.exists())
        masks[0, 0] = 0
        sidecar = self._sidecar(source, masks)
        self._add_arrays(sidecar, seeds=np.array([11, 10, 10], dtype=np.int32))
        with self.assertRaisesRegex(MergeError, 'exact order'):
            merge([source], output, self.audit, 'train', successor_sidecars={source: sidecar})

    def test_source_mutation_prevents_replacing_existing_output(self):
        source = self._write_shard('source.npz', [10])
        output = self.root / 'kept.npz'
        output.write_bytes(b'previous output')
        original = merger._copy_selected_rows
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            original(*args, **kwargs)
            if not changed:
                with source.open('ab') as stream:
                    stream.write(b'changed')
                changed = True
        with patch.object(merger, '_copy_selected_rows', mutate):
            with self.assertRaisesRegex(MergeError, 'source changed'):
                merge([source], output, self.audit, 'train')
        self.assertEqual(output.read_bytes(), b'previous output')

    def test_member_order_is_irrelevant_and_audit_cannot_be_overwritten(self):
        first = self._write_shard('one.npz', [10])
        second = self._write_shard('two.npz', [12])
        with np.load(second, allow_pickle=False) as archive:
            values = {name: archive[name] for name in reversed(archive.files)}
        np.savez_compressed(second, **values)
        merge([first, second], self.root / 'reordered.npz', self.audit, 'train')
        audit_bytes = self.audit.read_bytes()
        with self.assertRaisesRegex(MergeError, 'validity audit'):
            merge([first], self.audit, self.audit, 'train')
        self.assertEqual(self.audit.read_bytes(), audit_bytes)

    def test_sidecar_cannot_replace_existing_labels_or_target_unknown_source(self):
        source = self._write_shard('embedded.npz', [10])
        self._add_arrays(source, terminal=np.zeros((1, 4), dtype=bool),
                         next_optimal=np.ones((1, 4), dtype=np.uint8))
        labels = self._sidecar(source, np.ones((1, 4), dtype=np.uint8))
        with self.assertRaisesRegex(MergeError, 'replace embedded'):
            merge([source], self.root / 'bad.npz', self.audit, 'train',
                  successor_sidecars={source: labels})
        with self.assertRaisesRegex(MergeError, 'unknown input'):
            merge([source], self.root / 'bad.npz', self.audit, 'train',
                  successor_sidecars={self.root / 'unknown.npz': labels})

    def test_minimum_distinct_levels_is_checked_after_filtering(self):
        first = self._write_shard("first.npz", [10, 11])
        with self.assertRaisesRegex(MergeError, "accepted levels remain"):
            merge([first], self.root / "merged.npz", self.audit, "train", min_levels=2)
        self.assertFalse((self.root / "merged.npz").exists())

    def test_schema_mismatch_and_split_mismatch_are_rejected(self):
        first = self._write_shard("first.npz", [10, 11])
        second = self._write_shard("second.npz", [12], dtype=np.int32)
        with self.assertRaisesRegex(MergeError, "dtypes or trailing shapes"):
            merge([first, second], self.root / "merged.npz", self.audit, "train")

        wrong_split = self.root / "validation.json"
        wrong_split.write_text(json.dumps({"status": "complete", "split": "validation", "levels": []}))
        with self.assertRaisesRegex(MergeError, "not 'train'"):
            merge([first], self.root / "split.npz", wrong_split, "train")

    def test_nested_audit_status_normalizes_complete_engine_proof(self):
        audit = self.root / "nested.json"
        audit.write_text(json.dumps({"status": "complete", "validation": {
            "levels": [{"seed": 10, "status": "verified", "difficulty": 1,
                         "context_validity": True, "engine_win": True,
                         "replay_lives": 3, "levels_completed": 1,
                         "context_index": 3, "search_truncated": False}],
        }}))
        shard = self._write_shard("one.npz", [10])
        output = self.root / "nested-output.npz"
        merge([shard], output, audit, "validation")
        with np.load(output, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
        self.assertTrue(meta["levels"][0]["context_engine_verified"])
        self.assertFalse(meta["levels"][0]["search_truncated"])

    def test_rows_audit_filters_other_split_and_checks_npz_context(self):
        audit = self.root / "rows.json"
        audit.write_text(json.dumps({"status": "complete", "rows": [
            {"split": "train", "seed": 10, "difficulty": 1,
             "context_validity": True, "engine_win": True,
             "replay_lives": 3, "levels_completed": 1, "context_index": 3, "search_truncated": False},
            {"split": "validation", "seed": 12, "difficulty": 3,
             "context_validity": True, "engine_win": True,
             "replay_lives": 3, "levels_completed": 1, "context_index": 5, "search_truncated": False},
        ]}))
        shard = self._write_shard("context.npz", [10], context_values=[3])
        output = self.root / "rows-output.npz"
        merge([shard], output, audit, "train")
        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["seeds"], [10])

        bad = self._write_shard("bad-context.npz", [10], context_values=[0])
        with self.assertRaisesRegex(MergeError, "context_index disagrees"):
            merge([bad], self.root / "bad-output.npz", audit, "train")


if __name__ == "__main__":
    unittest.main()
