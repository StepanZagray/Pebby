"""Actual failure archives remain usable by the active structured cache/trainer."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import world_data as wd
from tests.test_policy_history import corridor, RecordingPolicy
from tests.test_structured_onpolicy_field_cache import PublicPolicy
from tools.collect_onpolicy_world import collect_level
from tools import build_structured_onpolicy_field_cache as fp16
from tools import build_structured_onpolicy_fp32_cache as fp32


class StructuredFailureCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        spec = {**corridor(budget=4), 'seed': 8, 'difficulty': 1}
        rows, proof, count = collect_level(spec, RecordingPolicy(history=8))
        proof.update(row_start=0, row_count=len(rows), on_policy_row_count=count)
        data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        data['meta'] = {'format': wd.FORMAT, 'source': 'generated_only', 'oracle_search': 'complete_only',
                        'history': 8, 'alternatives_per_state': 4, 'split': 'train', 'levels': [proof],
                        'on_policy_rows': list(range(count)), 'auxiliary_rows': list(range(count, len(rows))),
                        'collection_policy': 'model_greedy'}
        return data

    def test_all_real_failure_rows_survive_both_cache_formats_and_distance_labels(self):
        data = self.data()
        self.assertTrue((data['terminal'] & ~data['won']).any())
        for builder in (fp16, fp32):
            with self.subTest(builder=builder.FORMAT), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'source.npz'
                wd.save(source, data)
                cache = root / 'cache'
                manifest = builder.build(data, cache, PublicPolicy(), {str(source): fp16.digest(source)}, batch_size=2)
                self.assertEqual(manifest['rows'], len(data['seeds']))
                np.testing.assert_array_equal(np.load(cache / 'optimal.npy'), data['optimal'])
                np.testing.assert_array_equal(np.load(cache / 'lost_life.npy'), data['lost_life'])
                current = np.load(cache / 'current_distance.npy')
                self.assertTrue((current[data['optimal'] == 0] == -1).all())
                self.assertTrue((np.load(cache / 'terminal.npy') & ~np.load(cache / 'won.npy')).any())

    def test_failure_cache_reaches_structured_event_objective_and_masked_actor(self):
        from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition
        from pebby.agent.structured_objective import objective
        from pebby.agent.structured_workspace_policy import StructuredWorkspaceReadout
        from tools.structured_onpolicy_sampling import PairedStateSampler
        from tools.train_structured_onpolicy_comparison import validate_rows, prepare_batch
        from tools.train_structured_onpolicy_dynamics import prepare_dynamics_batch, tensors_for_batch, schedule_event_counts
        from tools.preflight_structured_workspace import backward
        data = self.data()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache'
            manifest = fp32.build(data, cache, PublicPolicy(), {}, batch_size=2)
            arrays = {key: np.load(cache / (key + '.npy')) for key in manifest['arrays']}
            base = {'seeds': np.array([8]), 'difficulties': np.array([1])}
            validate_rows(arrays, manifest, base)
            # Use the actual game-over sample as an explicitly authorized auxiliary row.
            auxiliary = np.zeros(len(arrays['seeds']), bool)
            auxiliary[-1] = True
            arrays['auxiliary'] = auxiliary
            sampler = PairedStateSampler(base['seeds'], base['difficulties'], arrays['seeds'], arrays['on_policy'],
                                         auxiliary=auxiliary, auxiliary_fraction=1.)
            selection = sampler.draw(1, 1, .5, np.random.default_rng(4))
            view = {key: value[:1].copy() for key, value in arrays.items() if key not in ('level_seeds','level_offsets','level_rows')}
            batch = prepare_dynamics_batch([view, view], arrays, selection, np.array([0]), True)
            self.assertEqual(batch['optimal'].tolist(), [0])
            self.assertEqual(batch['lost_life'].tolist(), [True])
            self.assertEqual(batch['terminal'].tolist(), [True])
            fields, following, actions, labels = tensors_for_batch(batch, 'cpu')
            model = LocalGlobalGlyphTransition(loops=1, expansion=1, event_hidden=8, glyph_hidden=8)
            result = objective(model, fields, following, actions, labels, torch.ones(48), pos_weight=torch.ones(3))
            self.assertTrue(torch.isfinite(result['total']))
            result['losses']['events'].backward()
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for name, parameter in model.named_parameters() if 'event' in name))
            counts = schedule_event_counts([view, view], arrays, [(selection, np.array([0]))], True)
            self.assertEqual(counts['auxiliary_branches'], 1)
            self.assertEqual(counts['terminal_failure'], 1)
            policy_batch = prepare_batch([view, view], arrays, selection, True)
            head = StructuredWorkspaceReadout({'mode': 'successors', 'loops': 1, 'expansion': 1})
            losses = backward(head, policy_batch, 'cpu', 1)
            self.assertEqual(losses['policy_valid_fraction'], 0.)
            self.assertEqual(losses['actual'], 0.)
            self.assertEqual(losses['imagined'], 0.)


if __name__ == '__main__':
    unittest.main()
