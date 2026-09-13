"""CPU trainer contracts using synthetic generated-data-shaped fixtures."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_policy import FORMAT as ORIGINAL_FORMAT
from pebby.agent.spatial_route_outcome_policy import FORMAT as ROUTE_FORMAT
from tools.cache_reference_outcome_inputs import SCHEMA
from tools.train_spatial_route_repair import build_model, checkpoint, matched_items, tiny_selection


def cache(seeds):
    result = {name: np.zeros((len(seeds), *shape), dtype=dtype)
              for name, (dtype, shape) in SCHEMA.items()}
    result['seeds'][:] = seeds
    result['optimal'][:] = 1
    return result


class SpatialRouteRepairTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def test_matched_items_replace_only_same_train_level_and_keep_zero_policy_labels(self):
        base, recent = cache([11, 11, 22, 22]), cache([22, 11])
        base['raw'][:] = np.arange(4)[:, None, None]
        recent['raw'][:] = 99
        recent['optimal'][0] = 0
        recent['lost_life'][0] = True
        selection = dict(seeds=np.array([11, 22]), base_rows=np.array([1, 3]),
                         recent_rows=np.array([-1, 0]))
        result = matched_items(base, recent, selection, 'cpu')
        self.assertNotIn('seeds', result)
        self.assertNotIn('rows', result)
        self.assertEqual(result['optimal'].tolist(), [1, 0])
        self.assertTrue(result['lost_life'][1].all())
        self.assertTrue((result['raw'][0] == 1).all())
        self.assertTrue((result['raw'][1] == 99).all())
        result['raw'].zero_()
        self.assertTrue((base['raw'][1] == 1).all())
        self.assertTrue((recent['raw'][0] == 99).all())

    def test_matched_items_reject_seed_mixing_including_foreign_validation_seed(self):
        base, recent = cache([11, 22]), cache([999])
        selection = dict(seeds=np.array([11]), base_rows=np.array([0]), recent_rows=np.array([0]))
        with self.assertRaisesRegex(ValueError, 'recent rows'):
            matched_items(base, recent, selection, 'cpu')
        with self.assertRaisesRegex(ValueError, 'recent rows'):
            matched_items(base, None, selection, 'cpu')
        selection['seeds'] = np.array([22])
        with self.assertRaisesRegex(ValueError, 'base rows'):
            matched_items(base, recent, selection, 'cpu')

    def test_tiny_panel_is_deterministic_distinct_balanced_and_covers_failures(self):
        levels = np.arange(84) + 100
        tiers = {int(level): index // 12 + 1 for index, level in enumerate(levels)}
        arrays = cache(np.repeat(levels, 8))
        failures, wins = np.arange(1, len(arrays['seeds']), 8), np.arange(2, len(arrays['seeds']), 8)
        arrays['optimal'][failures] = 0
        arrays['lost_life'][failures] = True
        arrays['terminal'][failures] = True
        arrays['next_triple'][failures] = 1
        arrays['won'][wins] = True
        arrays['terminal'][wins] = True
        rows = tiny_selection(arrays, tiers)
        np.testing.assert_array_equal(rows, tiny_selection(arrays, tiers))
        selected_seeds = arrays['seeds'][rows]
        self.assertEqual(len(set(selected_seeds)), 64)
        self.assertEqual(Counter(tiers[int(s)] for s in selected_seeds), {1: 10, 2: 9, 3: 9, 4: 9, 5: 9, 6: 9, 7: 9})
        for name in ('lost_life', 'terminal', 'won'):
            self.assertTrue(arrays[name][rows].any(), name)
        self.assertTrue((arrays['optimal'][rows] == 0).any())
        for field in range(3):
            self.assertTrue((arrays['next_triple'][rows, :, field] != arrays['current_triple'][rows, None, field]).any())
        arrays['won'][:] = False
        with self.assertRaisesRegex(ValueError, 'cohorts'):
            tiny_selection(arrays, tiers)

    def parent(self):
        torch.manual_seed(9)
        planner = SpatialOutcomePlanner(channels=8, width=8, hud_width=8, summary=16, comparator_hidden=16).eval()
        return dict(format=ORIGINAL_FORMAT, planner_config=planner.config(),
                    planner_weights={k: v.clone() for k, v in planner.state_dict().items()},
                    encoder_frozen=True, official_training_inputs=False,
                    sampling={'old': True}, precision='float32', seed=42,
                    quality_row_file_sha256={'old': 'digest'})

    def test_build_model_matched_warm_start_all_planner_trainable(self):
        parent = self.parent()
        control = build_model(parent, 'control', 'cpu', seed=17)
        route = build_model(parent, 'route', 'cpu', seed=17)
        repeat = build_model(parent, 'route', 'cpu', seed=17)
        inputs = (torch.randn(2, 160, 8), torch.randn(2, 160, 8),
                  torch.randn(2, 14), torch.randn(2, 144).softmax(-1))
        with torch.no_grad():
            expected, actual = control(*inputs), route(*inputs)
        for name in ('action_logits', 'value_logits', 'event_logits'):
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        for model in (control, route):
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            for name, value in parent['planner_weights'].items():
                torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                self.assertNotEqual(model.state_dict()[name].data_ptr(), value.data_ptr())
        for name, value in route.state_dict().items():
            torch.testing.assert_close(value, repeat.state_dict()[name], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'arm'):
            build_model(parent, 'unknown', 'cpu')

    def test_checkpoint_both_formats_record_latest_training_stage_without_tensor_aliases(self):
        parent = self.parent()
        args = SimpleNamespace(steps=5, lr=.0001, tier_counts=[10, 9, 9, 9, 9, 9, 9],
                               precision='bf16', seed=17, cache=Path('/generated/base'), recent=Path('/generated/recent'))
        report = dict(source_sha256={'source': 'digest'}, sampling={'current': True},
                      quality_row_file_sha256={'base_rows': 'base', 'supplement_rows': 'recent'},
                      base_manifest_sha256='base-manifest', recent_manifest_sha256='recent-manifest',
                      objective_weights={'value': 1}, qualification_sha256='qualification')
        for arm, expected_format in (('control', ORIGINAL_FORMAT), ('route', ROUTE_FORMAT)):
            with self.subTest(arm=arm):
                model = build_model(parent, arm, 'cpu')
                result = checkpoint(parent, model, arm, args, report)
                self.assertEqual(result['format'], expected_format)
                for key in ('sampling', 'quality_row_file_sha256', 'objective_weights'):
                    self.assertEqual(result[key], report[key])
                self.assertEqual(result['precision'], 'bf16')
                self.assertEqual(result['seed'], 17)
                self.assertEqual(result['parent_training_metadata']['precision'], 'float32')
                self.assertEqual(result['parent_training_metadata']['sampling'], {'old': True})
                self.assertEqual(result['optimizer_steps'], 5)
                self.assertEqual(result['weight_decay'], .05)
                self.assertIn('cosine', result['learning_rate_schedule'])
                self.assertTrue(result['encoder_frozen'])
                self.assertTrue(result['actual_outcome_comparator_auxiliary_training'])
                self.assertTrue(result['route_repair']['all_planner_parameters_trained'])
                self.assertFalse(result['route_repair']['gameplay_gain_established'])
                for name, value in result['planner_weights'].items():
                    torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
                    self.assertNotEqual(value.data_ptr(), model.state_dict()[name].data_ptr())
        self.assertEqual(parent['seed'], 42)


if __name__ == '__main__':
    unittest.main()
