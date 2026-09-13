"""CPU checks for matching, TRAIN isolation, disposable updates, and provenance."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from tools import train_spatial_recovery_comparison as trainer


class RecoveryComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def fixture(self):
        base = {'seeds': np.repeat(np.arange(1500), 2), 'optimal': np.ones(3000, dtype=np.uint8)}
        supplement = {'seeds': np.tile(np.arange(512), 3),
                      'row_kind': np.repeat(np.arange(3, dtype=np.uint8), 512), 'optimal': np.full(1536, 2, dtype=np.uint8)}
        return base, supplement

    def sampler(self, base, supplement):
        return trainer.MatchedSampler(base, supplement, base_rows=np.arange(len(base['seeds']), dtype=np.int64),
                                      supplement_rows=np.flatnonzero(supplement['row_kind'] < 2).astype(np.int64))

    def test_true_b1024_matches_seeds_and_reproducible_schedule(self):
        base, supplement = self.fixture()
        sampler = self.sampler(base, supplement)
        control_rng, recovery_rng = np.random.default_rng(42), np.random.default_rng(42)
        for _ in range(3):
            control, recovery = sampler.sample(control_rng), sampler.sample(recovery_rng)
            for key in control:
                np.testing.assert_array_equal(control[key], recovery[key])
            self.assertEqual(len(np.unique(control['seeds'])), 1024)
            slots = np.flatnonzero(control['replacement_rows'] >= 0)
            self.assertEqual(len(slots), 256)
            np.testing.assert_array_equal(base['seeds'][control['base_rows']], control['seeds'])
            rows = control['replacement_rows'][slots]
            np.testing.assert_array_equal(supplement['seeds'][rows], control['seeds'][slots])
            self.assertTrue(np.isin(supplement['row_kind'][rows], (0, 1)).all())
            self.assertGreater(np.count_nonzero(supplement['row_kind'][rows] == 0), 100)
            self.assertGreater(np.count_nonzero(supplement['row_kind'][rows] == 1), 20)

    def test_eligibility_needs_policy_rows_and_never_repeats_levels(self):
        base, supplement = self.fixture()
        supplement['row_kind'][:300] = 1
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            self.sampler(base, supplement)
        base, supplement = self.fixture()
        supplement['seeds'][0] = 99999
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            self.sampler(base, supplement)
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            self.sampler({'seeds': np.arange(100), 'optimal': np.ones(100)}, {'seeds': np.arange(100), 'row_kind': np.zeros(100), 'optimal': np.ones(100)})

    def test_only_reserved_rows_change_and_inputs_are_not_mutated(self):
        base, supplement = self.fixture()
        for name in ('raw', 'state', 'glyph'):
            base[name] = np.arange(len(base['seeds']), dtype=np.float32)[:, None]
            supplement[name] = -1 - np.arange(len(supplement['seeds']), dtype=np.float32)[:, None]
        selection = self.sampler(base, supplement).sample(np.random.default_rng(42))
        with patch.object(trainer, 'SCHEMA', {name: None for name in base}):
            control = trainer.matched_batch(base, supplement, selection, 'control', 'cpu')
            recovery = trainer.matched_batch(base, supplement, selection, 'recovery', 'cpu')
        replaced = selection['replacement_rows'] >= 0
        for name in control:
            self.assertTrue(torch.equal(control[name][~replaced], recovery[name][~replaced]))
            self.assertTrue(bool((control[name][replaced] != recovery[name][replaced]).all()))
            self.assertTrue((base[name] >= 0).all())
        self.assertNotIn('row_kind', recovery)
        bad = {**selection, 'seeds': selection['seeds'] + 1}
        with self.assertRaisesRegex(ValueError, 'matched seeds'):
            trainer.matched_batch(base, supplement, bad, 'recovery', 'cpu')

    def test_fresh_arms_discard_updates_and_checkpoint_preserves_contract(self):
        from tests.test_spatial_outcome_objective import fixture
        source, items, weights = fixture()
        checkpoint = dict(planner_config=source.config(), planner_weights=copy.deepcopy(source.state_dict()),
                          objective_weights=weights, planner_horizon=1, planner_refinement_loops=1,
                          encoder_weights={'player_head.weight': torch.randn(1, 4), 'player_head.bias': torch.zeros(1)},
                          encoder_frozen=True, persistent_game_memory=False, learned_voluntary_reset=False)
        original = trainer.weights_sha256(checkpoint['planner_weights'])
        for _ in range(2):
            model, optimizer = trainer.fresh_arm(checkpoint, .0001, 'cpu')
            self.assertEqual(trainer.weights_sha256(model.state_dict()), original)
            self.assertFalse(optimizer.state)
            record, norm = trainer.fit_step(model, optimizer, items, checkpoint['encoder_weights'], weights, 'float32')
            self.assertTrue(torch.isfinite(record['total']))
            self.assertTrue(torch.isfinite(norm))
            self.assertNotEqual(trainer.weights_sha256(model.state_dict()), original)
            self.assertEqual(trainer.weights_sha256(checkpoint['planner_weights']), original)
        args = SimpleNamespace(steps=780, lr=.0001, cache=Path('base'), supplement=Path('extra'),
                               recovery_fraction=.25, policy_row_fraction=.75)
        report = dict(initial_planner_weights_sha256=original, cache_manifest_sha256='base', train_levels=1234,
                      supplemental_manifest_sha256='supp', source_sha256={'source': 'hash'},
                      quality_manifest='quality.json', quality_manifest_sha256='quality',
                      quality_row_file_sha256={'base': 'a', 'supplement': 'b'}, quality_filter={'allowed': True})
        output = trainer.checkpoint_for(checkpoint, model, 'recovery', args, report)
        for key in ('objective_weights', 'planner_horizon', 'planner_refinement_loops', 'encoder_frozen',
                    'persistent_game_memory', 'learned_voluntary_reset'):
            self.assertEqual(output[key], checkpoint[key])
        for key in ('quality_manifest', 'quality_manifest_sha256', 'quality_row_file_sha256', 'quality_filter'):
            self.assertEqual(output[key], report[key])
        self.assertEqual(output['source_checkpoint_sha256'], trainer.CHECKPOINT_SHA)
        self.assertEqual(output['batch_size'], 1024)
        self.assertEqual(output['optimizer_steps'], 780)
        self.assertEqual(output['train_levels'], 1234)

    def test_supplement_checks_train_provenance_schema_and_file_integrity(self):
        checkpoint = dict(encoder_parent_sha256='encoder', encoder_weights_sha256='weights', encoder_runtime={'mode': 'fixed'})
        schema = {'seeds': ('int64', ()), 'raw': ('float32', (2,)), 'state': ('float32', (2,)), 'glyph': ('float32', (2,))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = dict(status='complete', split='train', official_inputs_used=False,
                            source_checkpoint_sha256=trainer.CHECKPOINT_SHA, **checkpoint, files={})
            arrays = {'seeds': np.array([2, 3, 3], dtype=np.int64), 'row_kind': np.array([0, 1, 2], np.uint8)}
            arrays.update({key: np.ones((3, 2), np.float32) for key in ('raw', 'state', 'glyph')})
            for key, value in arrays.items():
                target = path / (key + '.npy')
                np.save(target, value)
                manifest['files'][target.name] = dict(sha256=trainer.sha(target), shape=list(value.shape), dtype=value.dtype.str)
            def load():
                (path / 'manifest.json').write_text(json.dumps(manifest))
                with patch.object(trainer, 'SCHEMA', schema), patch.object(trainer, 'guard'):
                    return trainer.load_supplement(path, checkpoint, {'seeds': np.array([2, 3])}, trainer.Bindings())
            loaded, _ = load()
            self.assertEqual(loaded['row_kind'].tolist(), [0, 1, 2])
            for key, wrong in [('split', 'validation'), ('official_inputs_used', True),
                               ('source_checkpoint_sha256', 'different'), ('encoder_runtime', {})]:
                original = manifest[key]
                manifest[key] = wrong
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    load()
                manifest[key] = original
            manifest['files']['raw.npy']['dtype'] = 'float64'
            with self.assertRaisesRegex(ValueError, 'schema'):
                load()
            manifest['files']['raw.npy']['dtype'] = '<f4'
            np.save(path / 'seeds.npy', np.array([2, 3, 999], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load()
            manifest['files']['seeds.npy']['sha256'] = trainer.sha(path / 'seeds.npy')
            with self.assertRaisesRegex(ValueError, 'non-TRAIN'):
                load()

    def test_quality_rows_reject_invalid_and_useless_rows(self):
        base, supplement = self.fixture()
        train_seeds = np.unique(base['seeds'])
        for rows, message in [(np.array([], np.int64), 'nonempty'), (np.array([0, 0]), 'duplicate'),
                              (np.array([-1]), 'bounds'), (np.array([len(base['seeds'])]), 'bounds'),
                              (np.array([0.]), 'int64'), (np.array([[0]]), 'one-dimensional')]:
            with self.subTest(rows=rows.tolist()), self.assertRaisesRegex(ValueError, message):
                trainer.quality_mask(base, rows, train_seeds, 'base')
        base['optimal'][0] = 0
        with self.assertRaisesRegex(ValueError, 'zero optimal'):
            trainer.quality_mask(base, np.array([0]), train_seeds, 'base')
        supplement['optimal'][1] = 0
        with self.assertRaisesRegex(ValueError, 'zero optimal'):
            trainer.quality_mask(supplement, np.array([1]), train_seeds, 'supplement')
        with self.assertRaisesRegex(ValueError, 'kind 0 or 1'):
            trainer.quality_mask(supplement, np.array([1024]), train_seeds, 'supplement')
        supplement['seeds'][0] = 99999
        with self.assertRaisesRegex(ValueError, 'non-TRAIN'):
            trainer.quality_mask(supplement, np.array([0]), train_seeds, 'supplement')

    def test_quality_masks_apply_before_grouping_and_preserve_matched_levels(self):
        base, supplement = self.fixture()
        base['optimal'][::2] = 0
        supplement['optimal'][511] = 0
        base_rows = np.flatnonzero(base['optimal'])
        supplement_rows = np.flatnonzero((supplement['optimal'] != 0) & (supplement['row_kind'] < 2))
        sampler = trainer.MatchedSampler(base, supplement, base_rows=base_rows, supplement_rows=supplement_rows)
        self.assertNotIn(511, sampler.eligible)
        for _ in range(3):
            selected = sampler.sample(np.random.default_rng(42))
            chosen = selected['replacement_rows'][selected['replacement_rows'] >= 0]
            self.assertEqual(len(np.unique(selected['seeds'])), 1024)
            self.assertTrue(np.isin(selected['base_rows'], base_rows).all())
            self.assertTrue(np.isin(chosen, supplement_rows).all())
            self.assertTrue((base['optimal'][selected['base_rows']] != 0).all())
            self.assertTrue((supplement['optimal'][chosen] != 0).all())
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            trainer.MatchedSampler(base, supplement, base_rows=base_rows,
                                   supplement_rows=np.arange(255, dtype=np.int64))

    def test_quality_manifest_binds_indices_and_refuses_old_qualification(self):
        base, supplement = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = dict(status='complete', base_cache_manifest_sha256='base',
                            supplemental_manifest_sha256='supp', files={})
            for name, rows in [('base', np.arange(1, 3000, 2, dtype=np.int64)),
                               ('supplement', np.arange(1024, dtype=np.int64))]:
                target = path / (name + '_rows.npy')
                np.save(target, rows)
                manifest['files'][target.name] = dict(path=target.name, sha256=trainer.sha(target))
            manifest_path = path / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            report = dict(cache_manifest_sha256='base', supplemental_manifest_sha256='supp', source_sha256={})
            bindings = trainer.Bindings()
            rows = trainer.load_quality_manifest(manifest_path, base, supplement, report, bindings)
            self.assertEqual(len(rows['base']), 1500)
            self.assertEqual(report['quality_filter']['base']['excluded_rows'], 1500)
            self.assertEqual(len(report['source_sha256']), 3)
            self.assertEqual(report['quality_manifest_sha256'], trainer.sha(manifest_path))
            self.assertEqual(len(report['quality_row_file_sha256']), 2)
            manifest['base_cache_manifest_sha256'] = 'changed'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'different caches'):
                trainer.load_quality_manifest(manifest_path, base, supplement, report, trainer.Bindings())
            manifest['base_cache_manifest_sha256'] = 'base'
            manifest_path.write_text(json.dumps(manifest))
            np.save(path / 'base_rows.npy', np.array([0], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                trainer.load_quality_manifest(manifest_path, base, supplement, report, trainer.Bindings())
        report.update(objective_weights={'unchanged': True}, initial_planner_weights_sha256='head',
                      learning_rate=.0001, precision='bf16', batch_size=1024, planned_steps=780,
                      recovery_fraction=.25, policy_row_fraction=.75)
        qualification = dict(report, status='complete', qualification_passed=True)
        trainer.validate_qualification(qualification, report)
        for key in ('quality_manifest_sha256', 'quality_row_file_sha256', 'quality_filter'):
            old = {k: v for k, v in qualification.items() if k != key}
            with self.assertRaisesRegex(ValueError, 'quality-gated'):
                trainer.validate_qualification(old, report)

    def test_source_bindings_reject_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.py'
            path.write_text('before')
            bindings = trainer.Bindings()
            bindings.add(path)
            path.write_text('after!')
            with self.assertRaisesRegex(ValueError, 'stats changed'):
                bindings.verify()


if __name__ == '__main__':
    unittest.main()
