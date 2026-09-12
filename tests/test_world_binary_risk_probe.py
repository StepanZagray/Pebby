import unittest
import numpy as np
from tools.probe_world_binary_risk import metrics,unsafe,features
import torch
from tests import test_world_sequences as sequence_fixture
from tests.test_world_model import make_model
from tests.test_world_glyph import wake
from pebby.agent import world_data as wd

class RiskMetricTests(unittest.TestCase):
    def test_auroc_ties_and_missing_class(self):
        y=np.array([0,0,1,1],bool)
        m=metrics(y,[.1,.2,.8,.9]);self.assertEqual(m['auroc'],1);self.assertEqual(m['balanced_accuracy'],1)
        m=metrics(y,[.5,.5,.5,.5]);self.assertEqual(m['auroc'],.5);self.assertEqual(m['balanced_accuracy'],.5)
        self.assertIsNone(metrics([0,0],[.1,.3])['auroc'])
        with self.assertRaises(ValueError):metrics(y,[.1,.2,float('nan'),.9])

    def test_unsafe_includes_life_loss_but_excludes_won(self):
        a={'distances':np.array([1,-1,0,7]),'lost_life':np.array([0,0,0,1],bool),'terminal':np.array([0,0,1,0],bool),'won':np.array([0,0,1,0],bool)}
        np.testing.assert_array_equal(unsafe(a),[0,1,0,1])
    def test_actual_feature_histories_match_independent_encode_including_reset(self):
        torch.set_num_threads(1)
        fixture=sequence_fixture.WorldSequenceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        batch={k:v[:1].copy() for k,v in fixture.data.items() if k in ('frames','history_valid','previous_actions','next_frames','lost_life')}
        batch['lost_life'][0,0]=True
        model=wake(make_model(history=8,glyph_recall=True)).eval()
        with torch.no_grad():
            x,_=features(model,batch)
            for a in range(4):
                if a==0:f,v,p=wd.history_arrays([batch['next_frames'][0,a]],[-1],8)
                else:
                    f=np.concatenate((batch['frames'][0,1:],batch['next_frames'][0,a][None]))
                    v=np.r_[batch['history_valid'][0,1:],True];p=np.r_[batch['previous_actions'][0,1:],a]
                expected=model.encode(torch.from_numpy(f[None]).long(),torch.from_numpy(v[None]),torch.from_numpy(p[None]))['latent'][0]
                torch.testing.assert_close(x['actual'][a],expected,atol=2e-6,rtol=2e-5)

if __name__=='__main__':unittest.main()
