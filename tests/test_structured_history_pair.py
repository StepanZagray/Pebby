import copy
import unittest
import numpy as np
import torch
from tests import test_world_exploratory_sequences as fixture
from tools.prepare_structured_history_pair import current_only,pair_inputs,encode_pairs,choose_pairs

class RecordingEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.calls=[]
    def forward(self,frames,valid,actions):
        self.calls.append((frames.clone(),valid.clone(),actions.clone()))
        return frames[:,-1,0,0].float()[:,None,None].expand(-1,148,96)/16

class HistoryPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.ExploratoryTrainingTests.setUpClass();cls.data=fixture.ExploratoryTrainingTests.data
    def test_current_only_exact_boundary(self):
        f=torch.arange(2*8*64*64).reshape(2,8,64,64)%16
        out,valid,actions=current_only(f)
        torch.testing.assert_close(out,f[:,-1:].expand_as(f))
        self.assertEqual(valid.sum(1).tolist(),[1,1]);self.assertTrue(valid[:,-1].all());self.assertTrue((actions==-1).all())
    def test_actual_branch_reset_and_no_future_source_leak(self):
        data=copy.deepcopy(self.data);rows=np.array([0]);actions=np.array([3]);data['lost_life'][0,3]=True
        inputs=pair_inputs(data,rows,actions)
        target=inputs['h8'][1];self.assertEqual(target[1].sum(),1);self.assertTrue((target[2]==-1).all())
        torch.testing.assert_close(target[0],torch.from_numpy(data['next_frames'][0,3]).expand(1,8,64,64))
        encoder=RecordingEncoder();result=encode_pairs(encoder,inputs)
        self.assertEqual(len(encoder.calls),4);self.assertEqual(result['h8']['fields'].shape,(1,148,96))
        data['next_frames'][0,3]^=1;changed=pair_inputs(data,rows,actions)
        for mode in inputs:
            for a,b in zip(inputs[mode][0],changed[mode][0]):torch.testing.assert_close(a,b)
            self.assertFalse(torch.equal(inputs[mode][1][0],changed[mode][1][0]))
        with self.assertRaises(ValueError):pair_inputs(data,rows,np.array([4]))
    def test_balanced_distinct_stationary_selection(self):
        n=20;parent={'seeds':np.arange(n),'source_rows':np.arange(n),'difficulties':np.arange(n)%5+1}
        data={'seeds':np.arange(n),'frames':np.zeros((n,8,64,64),np.uint8),'next_frames':np.zeros((n,4,64,64),np.uint8)}
        rows,actions,info=choose_pairs(parent,data,10)
        self.assertEqual(len(np.unique(rows)),10);self.assertEqual(info['actual_stationary_rows'],10)
        self.assertEqual(np.bincount(parent['difficulties'][rows],minlength=6)[1:].tolist(),[2]*5)
        aa,bb,_=choose_pairs(parent,data,10);np.testing.assert_array_equal(rows,aa);np.testing.assert_array_equal(actions,bb)
        parent['seeds']+=1000000
        with self.assertRaisesRegex(ValueError,'TRAIN'):choose_pairs(parent,data,10)
