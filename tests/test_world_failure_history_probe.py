import unittest
import torch
from tests.test_policy_history import corridor, RecordingPolicy
from tools.world_failure_history_probe import probe_level, CATEGORIES


class FailureHistoryProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_repeated_refusal_categories_deduplicate_and_cache_public_inputs(self):
        spec = corridor()
        spec.update(seed=1000001, difficulty=1, start=(5,3), cyclers=[{'cell':(4,3),'kind':'shape'}])
        spec['goals'][0]['triple'] = [1,0,0]
        policy = RecordingPolicy(history=8)
        rows, report = probe_level(spec, policy, 20)
        self.assertEqual(report['actions'],20)
        self.assertEqual(len(policy.calls),9)
        self.assertEqual(len(rows),5)
        self.assertEqual(report['events']['initial']['row_index'], report['events']['first_wrong_action']['row_index'])
        e=report['events']['after_8_unchanged']
        self.assertEqual(e['step'],8)
        self.assertEqual(e['status'],'valid')
        self.assertFalse(e['actor_optimal'])
        self.assertTrue((rows[e['row_index']]['previous_actions']==3).all())
        self.assertEqual(set(report['events']),set(CATEGORIES))

    def test_initial_correct_winning_route_has_no_first_wrong(self):
        spec=corridor(); spec.update(seed=1000001,difficulty=1)
        rows,r=probe_level(spec,RecordingPolicy(history=8))
        self.assertEqual(r['ending'],'won')
        self.assertEqual(len(rows),1)
        self.assertEqual(r['events']['first_wrong_action']['status'],'not_observed')
        self.assertTrue(r['events']['initial']['actor_optimal'])

    def test_unreachable_labels_excluded_and_life_reset_clears_history(self):
        spec=corridor();spec.update(seed=1000001,difficulty=1,start=(8,3))
        rows,r=probe_level(spec,RecordingPolicy(history=8),60)
        event=r['events']['first_unreachable']
        self.assertEqual(event['status'],'unreachable')
        self.assertEqual(int(rows[event['row_index']]['optimal']),0)
        self.assertIsNone(event['actor_optimal'])
        reset=r['events']['first_life_reset']
        self.assertEqual(reset['status'],'valid')
        row=rows[reset['row_index']]
        self.assertEqual(int(row['history_valid'].sum()),1)
        self.assertTrue((row['previous_actions']==-1).all())
        self.assertEqual(r['stats']['checked_behavior_transitions'],r['actions'])

    def test_training_seeds_rejected(self):
        spec=corridor();spec.update(seed=8,difficulty=1)
        with self.assertRaisesRegex(ValueError,'validation seeds'):
            probe_level(spec,RecordingPolicy(history=8))

if __name__=='__main__': unittest.main()
