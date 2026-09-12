import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from tools.score_world_fixed_fatal_histories import verify_reference_alignment,verify_cache_digest,score
from tools.probe_world_binary_risk import features
from tools.collect_onpolicy_world import collect_level
from tests.test_world_closing_sequences import PublicHistoryDeadendPolicy
from tests.test_policy_history import corridor
from tests.test_world_model import make_model
from pebby.agent import world_data as wd
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from pebby.ls20 import names
from pebby.agent.on_policy_provenance import file_digest

class FixedFatalScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_real_deadend_reference_actions_labels_and_histories_bound(self):
        spec=corridor(budget=5);spec.update(seed=8,difficulty=1);spec['goals'][0]['cell']=(8,3)
        rows,proof,count=collect_level(spec,PublicHistoryDeadendPolicy(history=8),max_actions=5);row=rows[count-1]
        self.assertEqual(proof['stop'],'action_limit')
        data={key:np.asarray(row[key])[None] for key in ('frames','history_valid','previous_actions','next_frames','distances','lost_life','terminal','won')}
        data.update(seed=np.array([8]),step=np.array([4]),reference_action=np.array([2]),first_unreachable=np.array([True]))
        saved={key:data[key].copy() for key in ('frames','history_valid','previous_actions','seed','step')}
        event={'history_row':0,'step':4,'actor_action':2,**{name:row[key].tolist() for name,key in [('true_distances','distances'),('true_lost_life','lost_life'),('true_terminal','terminal'),('true_won','won')]}}
        reference={'levels':[{'seed':8,'events':{'first_reachable_to_unreachable':event}}]}
        verify_reference_alignment(data,reference,saved)
        doubled={key:np.repeat(value,2,axis=0) for key,value in data.items()}
        model=make_model(history=8).eval().requires_grad_(False)
        with patch.object(model,'encode',wraps=model.encode) as calls:
            scored=score(model,doubled)
        self.assertEqual(calls.call_count,2)
        self.assertTrue(all(call.args[0].shape[0]==1 for call in calls.call_args_list))
        self.assertNotIn('query',scored['component_preferences'])
        for key in ('reference_action','distances','frames'):
            changed=copy.deepcopy(data);changed[key].flat[0]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):verify_reference_alignment(changed,reference,saved)

    def test_real_life_reset_branch_encoding_uses_reset_history(self):
        env=Ls20Scenario(build_level(corridor(budget=0)),1)
        f,v,p=wd.history_arrays([env.render()],[-1],8);frames=[];lost=[]
        for action in names.ACTION_IDS:
            branch=wd.clone_env(env);result=branch.perform(action)
            frames.append(result.frame);lost.append(branch.lives()<env.lives())
        self.assertTrue(all(lost))
        batch={'frames':f[None],'history_valid':v[None],'previous_actions':p[None],'next_frames':np.asarray(frames)[None],'lost_life':np.asarray(lost)[None]}
        model=make_model(history=8).eval()
        with torch.no_grad():
            actual,_=features(model,batch)
            for i,frame in enumerate(frames):
                ef,ev,ep=wd.history_arrays([frame],[-1],8)
                expected=model.encode(torch.from_numpy(ef[None]),torch.from_numpy(ev[None]),torch.from_numpy(ep[None]))['latent'][0]
                torch.testing.assert_close(actual['actual'][i],expected,atol=2e-6,rtol=2e-5)

    def test_cache_bytes_cannot_change_under_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cache.npz';p.write_bytes(b'captured real branches')
            receipt={'cache_sha256':file_digest(p)};verify_cache_digest(p,receipt)
            p.write_bytes(b'changed successor pixels')
            with self.assertRaisesRegex(ValueError,'checksum'):verify_cache_digest(p,receipt)

if __name__=='__main__':unittest.main()
