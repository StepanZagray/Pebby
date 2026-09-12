"""Versioned generated contexts must never silently fall back to seed modulo."""
import copy
import unittest

import numpy as np
import torch

from pebby.agent.curriculum_sampling import CurriculumSampler
from pebby.agent.world_train import require_verified_data

VERSION = 'ls20-reference-v1'


class CalibratedConsumerTests(unittest.TestCase):
    def test_seven_stage_sampling_is_selected_by_provenance(self):
        levels = [{'seed': stage * 100 + offset, 'difficulty': stage,
                   'difficulty_version': VERSION}
                  for stage in range(1, 8) for offset in range(40)]
        data = {'seeds': np.array([level['seed'] for level in levels]), 'meta': {'levels': levels}}
        sampler = CurriculumSampler(data)
        early = sampler.indices(32, 0, torch.Generator().manual_seed(42))
        early_counts = sampler.last_difficulty_counts.copy()
        sampler.indices(32, 1, torch.Generator().manual_seed(42))
        self.assertEqual(set(early_counts), set(range(1, 8)))
        self.assertEqual(len(np.unique(data['seeds'][early])), 32)
        self.assertGreater(early_counts[1], sampler.last_difficulty_counts[1])
        self.assertLess(early_counts[7], sampler.last_difficulty_counts[7])
        legacy = copy.deepcopy(data)
        for level in legacy['meta']['levels']:
            del level['difficulty_version']
        with self.assertRaisesRegex(ValueError, 'difficulty'):
            CurriculumSampler(legacy)

    def test_training_checks_versioned_context_and_actual_row_context(self):
        proof = {'seed': 10, 'difficulty': 7, 'difficulty_version': VERSION,
                 'context_index': 6, 'context_engine_verified': True, 'search_truncated': False}
        data = {'seeds': np.array([10]), 'context_index': np.array([6]),
                'meta': {'source': 'generated_only', 'oracle_search': 'complete_only', 'levels': [proof]}}
        require_verified_data(data)
        for mutation in ('legacy_context', 'wrong_rows', 'unknown_version'):
            bad = copy.deepcopy(data)
            if mutation == 'legacy_context': bad['meta']['levels'][0]['context_index'] = 3
            if mutation == 'wrong_rows': bad['context_index'][0] = 3
            if mutation == 'unknown_version': bad['meta']['levels'][0]['difficulty_version'] = 'future'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                require_verified_data(bad)

    def test_mixed_legacy_and_calibrated_tiers_are_not_relabelled(self):
        data = {'seeds': np.array([10, 11]), 'meta': {'levels': [
            {'seed': 10, 'difficulty': 1, 'difficulty_version': VERSION},
            {'seed': 11, 'difficulty': 1}]}}
        with self.assertRaisesRegex(ValueError, 'mixed'):
            CurriculumSampler(data)


    def test_real_engine_collectors_use_tier_context_and_preserve_version(self):
        import random
        from pebby.agent import data, world_data
        from tests.test_policy_history import corridor
        spec = corridor(budget=40) | {'seed': 10, 'difficulty': 7,
                                      'difficulty_version': VERSION, 'search_limit': 1000}
        rows, proof = world_data.collect_level(spec, history=2, samples=2, coverage='mixed')
        self.assertNotIn('excluded', proof)
        self.assertEqual(proof['difficulty_version'], VERSION)
        self.assertEqual(proof['context_index'], 6)
        self.assertTrue(rows)
        self.assertTrue(all(row['context_index'] == 6 for row in rows))
        legacy_rows = data.episode_from_spec(spec, 0, random.Random(42), deviations=0, pads=0)
        self.assertEqual(legacy_rows['context_index'], 6)
        self.assertEqual(legacy_rows['difficulty_version'], VERSION)
        with self.assertRaisesRegex(ValueError, 'calibrated level context'):
            world_data.verified_context(spec, context_index=3)

    def test_calibrated_proof_budget_is_honored_without_changing_legacy(self):
        from pebby.ls20.provenance import search_limit_for
        self.assertEqual(search_limit_for({'search_limit': 1_000_000}, 600_000), 600_000)
        self.assertEqual(search_limit_for({'difficulty_version': VERSION, 'search_limit': 22_000_000}, 600_000), 22_000_000)
        with self.assertRaisesRegex(ValueError, 'search_limit'):
            search_limit_for({'difficulty_version': VERSION, 'search_limit': 100_000_000}, 600_000)
        for requested in (0, True, 600_000.5, 32_000_001):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, 'search_limit'):
                search_limit_for({'difficulty_version': VERSION, 'search_limit': 600_000}, requested)

    def test_stratified_collection_includes_all_seven_tiers(self):
        from tools.goal_attribute_probes import stratified_seeds
        specs = {stage * 100 + offset: {'difficulty': stage, 'difficulty_version': VERSION}
                 for stage in range(1, 8) for offset in range(2)}
        chosen = stratified_seeds(specs, 7, 42)
        self.assertEqual([specs[seed]['difficulty'] for seed in chosen], list(range(1, 8)))

    def test_inference_rejects_incorrect_calibrated_bank_context(self):
        from pathlib import Path
        from unittest.mock import patch
        from pebby.agent.evaluate import bank_levels
        spec = {'seed': 10, 'difficulty': 7, 'difficulty_version': VERSION,
                'training_context_index': 3, 'context_optimal_actions': 3}
        with patch('pebby.ls20.bank.load', return_value=[spec]), self.assertRaisesRegex(ValueError, 'calibrated context'):
            bank_levels(Path('unused.jsonl'))


    def test_public_inference_accepts_and_binds_calibrated_metadata(self):
        import inference
        from tests.test_policy_history import corridor
        spec = corridor() | {'seed': 10, 'difficulty': 7, 'difficulty_version': VERSION,
                              'reference_profile': {'reference_level': 7}, 'training_context_index': 6}
        checked = inference.validate_level(spec)
        self.assertEqual(checked['difficulty_version'], VERSION)
        self.assertEqual(checked['training_context_index'], 6)
        with self.assertRaisesRegex(Exception, 'difficulty - 1'):
            inference.validate_level(spec | {'training_context_index': 3})

    def test_event_cache_proofs_accept_seventh_tier_and_refuse_seed_context(self):
        from tools.build_structured_event_cache import _level_index
        proof = {'seed': 10, 'difficulty': 7, 'difficulty_version': VERSION,
                 'context_index': 6, 'context_engine_verified': True}
        self.assertEqual(_level_index({'levels': [proof]})[10], proof)
        with self.assertRaisesRegex(ValueError, 'wrong context'):
            _level_index({'levels': [proof | {'context_index': 3}]})

    def test_public_inference_rails_reach_engine_and_bind_cache(self):
        import inference
        from pebby.ls20.layout import extract
        from tests.test_policy_history import corridor
        spec = corridor() | {'seed': 10, 'difficulty': 7, 'difficulty_version': VERSION,
                              'training_context_index': 6, 'rails': [{'cells': [[3, 4], [4, 4]]}],
                              'cyclers': [{'cell': [3, 4], 'kind': 'shape'}]}
        spec['walls'] = [cell for cell in spec['walls'] if tuple(cell) not in {(3, 4), (4, 4), (3, 5)}]
        level = inference.validate_level(spec)
        self.assertEqual(len(extract(inference.build_env(level)).patrollers), 1)
        shifted = inference.validate_level(spec | {'rails': [{'cells': [[3, 4], [3, 5]]}]})
        self.assertNotEqual(inference.cache_key(level), inference.cache_key(shifted))
        reordered = inference.validate_level(spec | {'rails': [{'cells': [[4, 4], [3, 4]]}]})
        self.assertEqual(inference.cache_key(level), inference.cache_key(reordered))
        with self.assertRaisesRegex(ValueError, 'inside a wall'):
            inference.validate_level(spec | {'rails': [{'cells': [[3, 4], [4, 5]]}]})


    def test_disk_loader_retains_context_for_provenance_validation(self):
        import json
        import tempfile
        from pathlib import Path
        from pebby.agent.world_data import build
        from pebby.agent.world_train import load_dataset
        from tests.test_policy_history import corridor
        spec = corridor() | {'seed': 10, 'difficulty': 7, 'difficulty_version': VERSION}
        arrays = build([spec], workers=1, history=2, samples=2, coverage='mixed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibrated.npz'
            np.savez(path, **(arrays | {'meta': json.dumps(arrays['meta'])}))
            loaded = load_dataset(path)
            self.assertTrue((loaded['context_index'] == 6).all())
            require_verified_data(loaded)
            loaded['context_index'][:] = 3
            with self.assertRaisesRegex(ValueError, 'context_index'):
                require_verified_data(loaded)

    def test_paired_state_sampling_preserves_high_tiers_and_high_train_seeds(self):
        from tools.structured_onpolicy_sampling import PairedStateSampler
        seeds = np.arange(900_000, 900_014)
        difficulties = np.tile(np.arange(1, 8), 2)
        metadata = {'levels': [{'seed': int(s), 'difficulty': int(d), 'difficulty_version': VERSION}
                               for s, d in zip(seeds, difficulties)]}
        sampler = PairedStateSampler(seeds, difficulties, seeds, np.ones(len(seeds), bool),
                                     difficulty_metadata=metadata)
        seen = set()
        for _ in range(20):
            batch = sampler.draw(8, 4, 1, np.random.default_rng(_))
            seen.update(difficulties[batch.base_rows])
            self.assertEqual(len(np.unique(seeds[batch.base_rows])), 8)
        self.assertEqual(seen, set(range(1, 8)))
        with self.assertRaisesRegex(ValueError, 'difficulty'):
            PairedStateSampler(seeds, difficulties, seeds, np.ones(len(seeds), bool))

    def test_structured_selectors_keep_all_seven_tiers(self):
        from types import SimpleNamespace
        from tools.build_structured_training_sequences import select_rows as closing_rows
        from tools.build_structured_field_cache import select_rows as field_rows, LABELS
        from tools.score_world_sequences import select_anchors
        seeds = np.arange(900_000, 900_014)
        levels = [{'seed': int(s), 'difficulty': int(i % 7 + 1), 'difficulty_version': VERSION}
                  for i, s in enumerate(seeds)]
        data = {'seeds': seeds, 'meta': {'source': 'generated_only', 'oracle_search': 'complete_only',
                                       'split': 'train', 'levels': levels}}
        for name in LABELS.values(): data[name] = np.zeros((14, 4), dtype=bool)
        index = SimpleNamespace(anchor_row=np.arange(14), meta={'mode': 'closing_only_train', 'split': 'train'})
        rows = closing_rows(data, index, 14)
        self.assertEqual(set(seeds[rows]), set(seeds))
        rows, _ = field_rows(data, 14, 42, 'train')
        self.assertEqual(set(seeds[rows]), set(seeds))
        data['seeds'] = seeds + 1_000_000
        for level in levels: level['seed'] += 1_000_000
        index.meta = {'mode': 'explore_validation', 'split': 'validation'}
        rows = select_anchors(data, index, 14)
        self.assertEqual(set(data['seeds'][rows]), set(seeds + 1_000_000))

    def test_seven_tier_sequence_cache_loads_and_samples_without_losing_labels(self):
        import json
        import tempfile
        from pathlib import Path
        from tests.test_train_structured_sequences import SequenceTrainerTests
        from tools.train_structured_sequences import load_cache
        from tools.train_structured_transition import digest, sample_rows
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data, manifest = SequenceTrainerTests().fixture(path)
            data['seeds'] = np.array([900_006, 900_007])
            data['difficulties'] = np.array([6, 7])
            for name in ('seeds', 'difficulties'):
                np.save(path / (name + '.npy'), data[name])
                manifest['arrays'][name]['sha256'] = digest(path / (name + '.npy'))
            manifest['difficulty_levels'] = [{'seed': int(s), 'difficulty': int(d), 'difficulty_version': VERSION}
                                             for s, d in zip(data['seeds'], data['difficulties'])]
            manifest['difficulty_version'] = VERSION
            (path / 'manifest.json').write_text(json.dumps(manifest))
            loaded, _ = load_cache(path, 'train')
            rows = sample_rows(loaded, 2, 1, np.random.default_rng(42))
            self.assertEqual(set(loaded['difficulties'][rows]), {6, 7})
            del manifest['difficulty_levels'][0]['difficulty_version']
            (path / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'mixed'):
                load_cache(path, 'train')

    def test_frozen_experiments_reject_seven_tier_inputs_with_migration_path(self):
        from pebby.ls20.provenance import require_legacy_experiment
        from tools.prepare_world_combined_inputs import require_frozen_legacy_bank
        levels = [{'seed': 900_007, 'difficulty': 7, 'difficulty_version': VERSION}]
        with self.assertRaisesRegex(ValueError, 'train_structured_sequences.py'):
            require_legacy_experiment({'source_metadata': {'levels': levels}})
        with self.assertRaisesRegex(RuntimeError, 'merge_world_data.py'):
            require_frozen_legacy_bank(levels)

    def test_cache_builder_preserves_selected_seven_tier_provenance(self):
        import tempfile
        from pathlib import Path
        from pebby.agent import world_data
        from pebby.agent.world_sequences import build_sidecar, load_sidecar
        from tools.build_structured_sequence_cache import write_cache
        from tools.train_structured_sequences import load_cache
        from tests.test_structured_sequence_cache import RecordingAssembler
        from tests.test_policy_history import corridor
        specs = []
        for stage in range(1, 8):
            spec = corridor(budget=40) | {'seed': 1_900_000 + stage, 'difficulty': stage,
                                         'difficulty_version': VERSION}
            spec['goals'][0]['cell'] = (8, 3)
            specs.append(spec)
        data = world_data.build(specs, workers=1, history=8, samples=16, coverage='mixed', epsilon=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, sidecar, output = root / 'source.npz', root / 'index.npz', root / 'cache'
            world_data.save(source, data)
            build_sidecar(data, source, sidecar, 'explore_validation')
            index = load_sidecar(sidecar, source, data, mode='explore_validation')
            manifest = write_cache(data, index, source, sidecar, output, RecordingAssembler(),
                                   {'fixture': 'public-only'}, levels=7)
            self.assertEqual(manifest['difficulty_version'], VERSION)
            self.assertEqual(len(manifest['difficulty_levels']), 7)
            self.assertEqual(set(manifest['difficulty_counts']), set(map(str, range(1, 8))))
            loaded, _ = load_cache(output, 'validation')
            self.assertEqual(set(loaded['difficulties']), set(range(1, 8)))
