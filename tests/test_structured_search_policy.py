import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from pebby.agent.structured_search_policy import StructuredSearchPolicy,SearchPolicyConfig,load_search_policy_checkpoint,FORMAT
from pebby.agent.event_calibration import PositiveSlopePlatt

class Encoder(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(1));self.seen=None
    def forward(self,frames,valid,actions):
        self.seen=(frames.clone(),valid.clone(),actions.clone());return torch.zeros(len(frames),148,96)
class Dynamics(nn.Module):
    def forward(self,f,a):
        out=f.clone();out[:,0,0]=a
        return {'field':out,'events':{k:torch.full((len(f),),-30.) for k in ['lost_life_logits','terminal_logits','won_logits']}}
class Head(nn.Module):
    def forward(self,f):
        z=torch.full((len(f),5),-1000.);z[torch.arange(len(f)),3-f[:,0,0].long()]=0;return z
class PolicyTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_public_history_adapter_frozen_and_scores(self):
        e=Encoder();p=StructuredSearchPolicy(e,Dynamics(),Head(),PositiveSlopePlatt(),{'depth':1})
        frames=torch.zeros(2,8,64,64,dtype=torch.uint8);valid=torch.ones(2,8,dtype=torch.bool);actions=torch.full((2,8),-1)
        result=p(frames,valid,actions)
        self.assertEqual(result.shape,(2,4));self.assertEqual(result.argmax(-1).tolist(),[3,3])
        self.assertFalse(result.requires_grad);self.assertTrue(torch.equal(e.seen[0],frames));self.assertTrue(torch.equal(e.seen[1],valid));self.assertTrue(torch.equal(e.seen[2],actions))
        p.train();self.assertFalse(p.training);self.assertTrue(all(not x.requires_grad for x in p.parameters()))
        self.assertEqual(p.parameter_counts()['trainable'],0)
        with self.assertRaises(TypeError):p(frames,valid,actions,next_frames=frames)
    def test_float32_evaluator_preserves_near_tie_core_order(self):
        p=StructuredSearchPolicy(Encoder(),Dynamics(),Head(),PositiveSlopePlatt(),{'depth':1})
        roots=[{'root_action':i,'score':.5+(1e-12 if i==3 else 0)} for i in range(4)]
        with patch.object(p,'search_fields',return_value={'results':[{'roots':roots}]}):
            scores=p(torch.zeros(1,8,64,64),torch.ones(1,8,dtype=torch.bool),torch.full((1,8),-1))
        self.assertEqual(int(scores.float().argmax()),3)
        self.assertEqual(scores.tolist(),[[3.,2.,1.,4.]])

    def test_bad_config(self):
        for c in [{'depth':0},{'beam':5},{'gamma':.9},{'history':1},{'chunk_size':129}]:
            with self.assertRaises(ValueError):SearchPolicyConfig(**c)
    def test_calibration_v1_or_unconditional_won_rejected(self):
        from pebby.agent.structured_search_policy import _calibration
        from pebby.agent.event_calibration import FORMAT as calibration_format,EVENT_NAMES
        saved={'format':calibration_format,'parameters':6,'event_names':list(EVENT_NAMES),'config':{'initial_slope':1.,'positive_slope':True,'conditional_won':False}}
        with self.assertRaisesRegex(ValueError,'conditional'):_calibration(saved,None,None)
        saved['format']='pebby.structured-event-platt.v1'
        with self.assertRaisesRegex(ValueError,'v2'):_calibration(saved,None,None)

    def test_generic_dispatch_and_fail_closed(self):
        from pebby.agent.model import load_checkpoint
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.pt';torch.save({'format':FORMAT,'official_inputs_used':False,'sources':{},'config':{}},path)
            with self.assertRaises(ValueError):load_search_policy_checkpoint(path)
            with patch('pebby.agent.structured_search_policy.load_search_policy_checkpoint',return_value=('model','saved')) as loader:
                self.assertEqual(load_checkpoint(path),('model','saved'));loader.assert_called_once_with(path,'cpu')
            torch.save({'format':'wrong'},path)
            with self.assertRaises(ValueError):load_search_policy_checkpoint(path)
if __name__=='__main__':unittest.main()
