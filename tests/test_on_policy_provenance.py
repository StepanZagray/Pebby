import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pebby.agent.on_policy_provenance import file_digest, validate_on_policy_provenance


class OnPolicyProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.behaviors = []
        self.raw = []
        for i in range(2):
            checkpoint = self.root / f'behavior{i}.pt'
            checkpoint.write_bytes(f'frozen behavior {i}'.encode())
            behavior = {'path': str(checkpoint), 'sha256': file_digest(checkpoint)}
            self.behaviors.append(behavior)
            meta = {'source': 'generated_only', 'collection_policy': 'model_greedy',
                    'on_policy_provenance': {'official_inputs_used': False,
                                             'oracle_actions_in_policy_rollout': 0},
                    'behavior_checkpoint': behavior, 'on_policy_rows': [0, 2]}
            data = {'meta': meta, 'seeds': np.array([10 + i, 10 + i, 10 + i])}
            path = self.root / f'source{i}.npz'
            np.savez(path, seeds=data['seeds'], meta=json.dumps(meta))
            self.raw.append((path, data))
        meta = copy.deepcopy(self.raw[0][1]['meta'])
        del meta['behavior_checkpoint']
        meta.update(behavior_checkpoints=self.behaviors, on_policy_rows=[0, 2, 3, 5],
                    on_policy_sources=[{'path': str(path), 'sha256': file_digest(path),
                        'row_start': i * 3, 'row_stop': (i + 1) * 3,
                        'behavior_index': i, 'on_policy_rows': [i * 3, i * 3 + 2]}
                        for i, (path, _) in enumerate(self.raw)])
        self.aggregate = {'meta': meta, 'seeds': np.array([10, 10, 10, 11, 11, 11])}

    def test_raw_and_source_bound_aggregate(self):
        result = validate_on_policy_provenance(self.raw[0][1])
        self.assertEqual(result['behavior_checkpoint'], self.behaviors[0])
        result = validate_on_policy_provenance(self.aggregate)
        self.assertEqual(result['behavior_checkpoints'], self.behaviors)
        self.assertEqual(result['on_policy_sources'][1]['on_policy_rows'], [3, 5])

    def test_false_provenance_checkpoint_and_source_mutation_rejected(self):
        for key, value in [('official_inputs_used', True), ('oracle_actions_in_policy_rollout', 1)]:
            data = copy.deepcopy(self.aggregate)
            data['meta']['on_policy_provenance'][key] = value
            with self.assertRaisesRegex(ValueError, 'public-policy provenance'):
                validate_on_policy_provenance(data)
        path = self.raw[1][0]
        path.write_bytes(path.read_bytes() + b'mutation')
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            validate_on_policy_provenance(self.aggregate)
        Path(self.behaviors[0]['path']).write_bytes(b'changed behavior')
        with self.assertRaisesRegex(ValueError, 'checkpoint hash mismatch'):
            validate_on_policy_provenance(self.raw[0][1])

    def test_misbound_behavior_expert_row_and_incomplete_ranges_rejected(self):
        changes = [
            ('behavior_index', 0, 'assigned checkpoint'),
            ('on_policy_rows', [3, 4], 'marked rows'),
            ('row_start', 2, 'source ranges'),
        ]
        for key, value, message in changes:
            data = copy.deepcopy(self.aggregate)
            data['meta']['on_policy_sources'][1][key] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_on_policy_provenance(data)
        data = copy.deepcopy(self.aggregate)
        data['meta']['on_policy_rows'] = [0, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, 'coverage is incomplete'):
            validate_on_policy_provenance(data)

    def test_split_and_row_alignment_rejected(self):
        data = copy.deepcopy(self.aggregate)
        data['seeds'][0] = 1_000_001
        with self.assertRaisesRegex(ValueError, 'training seeds'):
            validate_on_policy_provenance(data)
        data['seeds'][0] = 12
        with self.assertRaisesRegex(ValueError, 'row seeds'):
            validate_on_policy_provenance(data)


if __name__ == '__main__':
    unittest.main()
