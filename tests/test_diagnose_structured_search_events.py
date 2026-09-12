import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools.diagnose_structured_search_events import next_history,selected_specs
from tools.collect_structured_event_transitions import capture_branches
from tools.build_structured_field_cache import actual_histories
from pebby.agent import world_data as wd
from tests.test_policy_history import corridor

class DiagnosticTests(unittest.TestCase):
    def test_real_four_branch_history_and_causal_actions(self):
        spec=corridor();spec.update(seed=8,difficulty=1)
        env,oracle,proof=wd.verified_context(spec,context_index=1,search_limit=600000)
        self.assertIsNotNone(env)
        frames,valid,actions=wd.history_arrays([env.render()],[-1],8)
        cap=capture_branches(env,oracle,8,0)
        self.assertEqual(cap['branch_checks'],4);self.assertEqual(cap['engine_branch_checks'],4)
        h,v,a=actual_histories({'frames':frames[None],'history_valid':valid[None],'previous_actions':actions[None],'next_frames':cap['next_frames'][None],'lost_life':cap['lost_life'][None]})
        for action in range(4):
            x=next_history(frames,valid,actions,cap['next_frames'][action],action,cap['lost_life'][action])
            for lhs,rhs in zip(x,(h[0,action],v[0,action],a[0,action])):np.testing.assert_array_equal(lhs,rhs)
            np.testing.assert_array_equal(x[0][-1],cap['_results'][action].frame)
    def test_reset_and_eight_refusals_history(self):
        f=np.full((8,64,64),2,np.uint8);v=np.array([False]*7+[True]);a=np.full(8,-1)
        for i in range(8):f,v,a=next_history(f,v,a,f[-1],3,False)
        self.assertTrue(v.all());np.testing.assert_array_equal(a,np.full(8,3))
        f,v,a=next_history(f,v,a,np.full((64,64),7,np.uint8),1,True)
        self.assertEqual(v.sum(),1);self.assertTrue(v[-1]);self.assertTrue((a==-1).all());self.assertTrue((f==7).all())
    def test_train_split_and_stratification_fail_closed(self):
        good=[{'seed':8+i,'difficulty':i+1} for i in range(5)]
        with patch('tools.diagnose_structured_search_events.select',return_value=good):self.assertEqual(selected_specs('unused'),good)
        for change in [{'seed':1000001,'difficulty':1},{'seed':8,'difficulty':2}]:
            bad=[change,*good[1:]]
            with patch('tools.diagnose_structured_search_events.select',return_value=bad):
                with self.assertRaises(ValueError):selected_specs('unused')
if __name__=='__main__':torch.set_num_threads(1);unittest.main()
