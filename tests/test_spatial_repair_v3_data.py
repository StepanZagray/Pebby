"""CPU contract tests; no production collection, checkpoint load or CUDA call."""
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from tests.test_cache_spatial_recovery import PublicEncoder, source
from tools import cache_spatial_repair_v3 as cache
from tools import collect_spatial_repair_v3 as collect


def fixture():
    result = source()
    result['chosen_action'] = np.array([0, 1, 2, 3], dtype=np.int8)
    result['optimal'][:] = [1, 3, 0, 0]
    result['policy_valid'] = result['optimal'] != 0
    result['dynamics_valid'] = np.ones(4, dtype=bool)
    result['current_lives'][:] = 3
    result['next_lives'][:] = 3
    result['next_lives'][2] = 2
    result['lost_life'][2] = True
    return result


class SpatialRepairDataTests(unittest.TestCase):
    def test_full_root_payload_preserves_exhaustion_zero_masks_and_real_actions(self):
        arrays = fixture()
        self.assertEqual(cache.validate_rows(arrays, 7), 4)
        result = cache.payload(PublicEncoder(), arrays, np.array([2, 3, 0]), 10, 'cpu')
        self.assertEqual(set(result), set(cache.SCHEMA))
        np.testing.assert_array_equal(result['rows'], [12, 13, 10])
        np.testing.assert_array_equal(result['optimal'], [0, 0, 1])
        np.testing.assert_array_equal(result['policy_valid'], [False, False, True])
        self.assertTrue(result['dynamics_valid'].all())
        self.assertTrue(result['lost_life'][0].all())
        np.testing.assert_array_equal(result['chosen_action'], [2, 3, 0])
        self.assertEqual(int(result['row_kind'][0]), 2)

    def test_target_changes_cannot_enter_frozen_public_encoder(self):
        arrays = fixture()
        encoder = PublicEncoder()
        a = cache.payload(encoder, arrays, np.arange(4), 0, 'cpu')
        arrays['distances'][:] = 97
        arrays['optimal'][:] = 0
        b = cache.payload(encoder, arrays, np.arange(4), 0, 'cpu')
        for key in ('raw', 'state', 'glyph'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(all(len(call) == 3 for call in encoder.calls))

    def test_invalid_objective_masks_or_life_labels_rejected(self):
        for key, value in [('policy_valid', np.ones(4, dtype=bool)),
                           ('dynamics_valid', np.zeros(4, dtype=bool)),
                           ('chosen_action', np.full(4, 4, dtype=np.int8)),
                           ('lost_life', np.zeros((4, 4), dtype=bool))]:
            with self.subTest(key=key):
                arrays = fixture()
                arrays[key] = value
                with self.assertRaises(ValueError):
                    cache.validate_rows(arrays, 7)

    def test_dedup_keeps_base_and_zero_policy_targets_without_assuming_mask(self):
        a, b, conflicts, duplicates = cache.approved_rows(
            [b'a', b'b', b'a', b'c', b'c'], [b'x', b'y', b'x', b'zero', b'zero'],
            2, [0, 0, 0, 2, 0])
        np.testing.assert_array_equal(a, [0, 1])
        np.testing.assert_array_equal(b, [2])
        self.assertEqual(len(conflicts), 0)
        self.assertEqual(duplicates, 2)

    def test_conflicting_public_targets_exclude_both_sources(self):
        a, b, conflicts, duplicates = cache.approved_rows(
            [b'a', b'b', b'a', b'c'], [b'left', b'y', b'right', b'z'], 2, [0] * 4)
        np.testing.assert_array_equal(a, [1])
        np.testing.assert_array_equal(b, [1])
        np.testing.assert_array_equal(conflicts, [0, 2])
        self.assertEqual(duplicates, 0)

    def test_public_identity_includes_history_validity_and_actions(self):
        arrays = fixture()
        before = cache.fingerprint(arrays, 0, cache.base.PUBLIC)
        for key, index in [('frames', (0, 0, 0, 0)), ('history_valid', (0, 0)), ('previous_actions', (0, 0))]:
            copy = {k: v.copy() for k, v in arrays.items()}
            copy[key][index] = not copy[key][index] if copy[key].dtype == bool else copy[key][index] + 1
            self.assertNotEqual(before, cache.fingerprint(copy, 0, cache.base.PUBLIC))
        arrays['distances'][:] = 17
        self.assertEqual(before, cache.fingerprint(arrays, 0, cache.base.PUBLIC))

    def test_fingerprint_level_decompresses_each_column_once(self):
        arrays, reads = fixture(), Counter()
        class Archive:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def __getitem__(self, key):
                reads[key] += 1
                return arrays[key].copy()
        with patch.object(cache.np, 'load', return_value=Archive()):
            result = cache.fingerprint_level(Path('fake-compressed-level.npz'))
        keys = {*cache.base.PUBLIC, *cache.LABEL_KEYS, 'row_kind', 'seeds'}
        self.assertEqual(dict(reads), dict.fromkeys(keys, 1))
        self.assertEqual(result['public'], [cache.fingerprint(arrays, row, cache.base.PUBLIC) for row in range(4)])
        self.assertEqual(result['labels'], [cache.fingerprint(arrays, row, cache.LABEL_KEYS) for row in range(4)])
        self.assertEqual(result['zero'], [False, False, True, True])
        self.assertEqual(result['loss_counts'], [0, 0, 4, 0])
        self.assertTrue(all(not isinstance(value, np.ndarray) for values in result.values() for value in values))

    def test_pilot_and_full_quotas_and_high_tier_concurrency(self):
        self.assertEqual(sum(collect.PILOT_QUOTAS), 64)
        self.assertEqual(sum(collect.FULL_QUOTAS), 512)
        selected = [{'difficulty': d, 'seed': d} for d in range(1, 8)]
        shards = collect.shard_levels(selected, 2)
        self.assertEqual([[s['difficulty'] for s in group] for group in shards], [[6, 7], [1, 2, 3, 4, 5]])
        self.assertEqual(collect.shard_levels(selected, 1), [selected])
        with self.assertRaises(ValueError):
            collect.shard_levels(selected, 3)

    def test_zero_mask_life_loss_cohort_counts_are_not_filtered(self):
        got = collect.cohorts(fixture())
        self.assertEqual(got['rows'], 4)
        self.assertEqual(got['zero_optimal_rows'], 2)
        self.assertEqual(got['life_loss_branches'], 4)
        self.assertEqual(got['exhaustion_rows'], 1)

    def test_worker_uses_explicit_new_checkpoint_and64_step_suffix_preserving_raw_rows(self):
        arrays = fixture()
        raw = [{key: value[i] for key, value in arrays.items() if key not in ('policy_valid', 'dynamics_valid')}
               for i in range(4)]
        proof = dict(seed=7, rows=4, policy_rows=1, branch_checks={'branches': 16})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'levels').mkdir()
            checkpoint = path / 'checkpoint.pt'
            checkpoint.write_bytes(b'explicit fixture checkpoint; never loaded')
            sha = cache.base.sha(checkpoint)
            bank = path / 'bank.jsonl'
            bank.write_text(json.dumps({'seed': 7, 'difficulty': 1, 'generator_version': 3}) + '\n')
            (path / 'report.json').write_text(json.dumps({'source_checkpoint_sha256': sha,
                'source_bindings': {str(checkpoint): sha}}))
            args = SimpleNamespace(out_dir=path, checkpoint=checkpoint, checkpoint_sha256=sha,
                                   worker_bank=bank, max_actions=150, recovery_horizon=64)
            with patch.object(collect, 'load_checkpoint') as loader, patch.object(collect.legacy, 'collect_level', return_value=(raw, proof)) as run, patch.object(collect, 'memory_available', return_value=30 * 2**30), patch.object(collect.torch, 'set_num_threads'):
                loader.return_value = (unittest.mock.MagicMock(), {})
                collect.worker(args)
                loader.assert_called_once_with(checkpoint, 'cpu')
                self.assertEqual(run.call_args.args[2:], (150, 64))
            saved = json.loads((path / 'levels/7.json').read_text())
            self.assertEqual(saved['source_checkpoint_sha256'], sha)
            with np.load(path / 'levels/7.npz', allow_pickle=False) as output:
                self.assertEqual(len(output['optimal']), 4)
                np.testing.assert_array_equal(output['optimal'], arrays['optimal'])
                np.testing.assert_array_equal(output['lost_life'], arrays['lost_life'])
                np.testing.assert_array_equal(output['row_kind'], arrays['row_kind'])
                np.testing.assert_array_equal(output['policy_valid'], [True, True, False, False])

    def test_stale_checkpoint_and_existing_output_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(collect, 'worker') as worker:
            with self.assertRaises(SystemExit):
                collect.main(['--out-dir', folder, '--checkpoint-sha256', '5ea4' * 16])
            worker.assert_not_called()
        with tempfile.TemporaryDirectory() as folder, patch.object(cache.base, 'require_no_foreign_cuda') as gpu:
            with self.assertRaises(FileExistsError):
                cache.main(['--source-dir', 'absent', '--out-dir', folder])
            gpu.assert_not_called()

    def test_incomplete_publication_is_rejected_without_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
            with self.assertRaisesRegex(ValueError, 'incomplete or invalid'):
                cache.validate_published(path)


if __name__ == '__main__':
    unittest.main()
