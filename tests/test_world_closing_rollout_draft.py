"""Closing-K4 draft tests use only constructed generated-engine corridors."""
import copy,importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent import world_data as wd,world_model as legacy
from pebby.ls20 import names
from tests.test_policy_history import corridor
from tests.test_world_model import make_model
from tests.test_world_glyph import wake


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,Path(path));module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
helper=load('pebby.agent._world_rollout_closing_draft','artifacts/world_rollout_closing_draft.py')
draft=load('pebby.agent._world_model_closing_draft','artifacts/world_model_closing_draft.py')


def trajectory(kind):
    spec=corridor(budget=3 if kind=='reset' else 5 if kind=='deadend' else 20);spec.update(seed=8,difficulty=1)
    if kind in ('win','live'):spec['goals'][0]['cell']=(7 if kind=='win' else 8,3)
    env,oracle,proof=wd.verified_context(spec)
    assert env is not None and oracle.solvable and not oracle.truncated
    frame=env.render();f,v,p=wd.history_arrays([frame],[-1],8)
    initial,_,_,_=wd._expand(env,oracle,oracle.distance_for(oracle.state_of(env)),8,0)
    row=wd._row(initial,f,v,p,8,proof['context_index']);sequence=copy.deepcopy(row)
    keys=['next_frames','next_player_cell','next_triple','next_steps','next_lives','distances','next_optimal','terminal','won','lost_life']
    values={k:[] for k in keys};frames=[frame];actions=[-1];windows=[]
    chosen=[3,3,2,3] if kind=='reset' else [3,3,2,2] if kind=='deadend' else [3]*4
    for h,a in enumerate(chosen):
        lives=env.lives();result=env.perform(names.ACTION_IDS[a]);lost=env.lives()<lives
        if h<3:assert not result.finished and not lost
        distance=0 if result.won else oracle.distance_for(oracle.state_of(env))
        label=-1 if (result.finished and not result.won) or distance is None else distance
        vals=[result.frame,env.player_cell(),env.triple(),env.steps_left(),env.lives(),label,wd.successor_optimal_mask(oracle,oracle.state_of(env),terminal=result.finished),result.finished,result.won,lost]
        for k,value in zip(keys,vals):values[k].append(value)
        if lost:frames,actions=[result.frame],[-1]
        else:frames,actions=(frames+[result.frame])[-8:],(actions+[a])[-8:]
        windows.append(wd.history_arrays(frames,actions,8))
    for k,value in values.items():sequence[k]=np.asarray(value,dtype=row[k].dtype)
    batch={k:torch.as_tensor(np.stack([sequence[k],row[k]])) for k in row}
    batch.update(rollout_mask=torch.tensor([True,False]),rollout_actions=torch.tensor([chosen,[-1]*4]))
    return batch,windows


class ClosingRolloutDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(324);return wake(make_model(history=8,grounding=True,glyph_recall=True,query_readout=True)).eval()

    def loss(self,model,batch):return draft.world_losses(model,batch,{'successor_policy':1.},sigreg_generator=torch.Generator().manual_seed(8))

    def test_actual_generated_final_win_and_reset_target_histories_and_labels(self):
        for kind in ('win','reset','deadend'):
            with self.subTest(kind=kind):
                batch,windows=trajectory(kind);model=self.model();out=self.loss(model,batch)
                self.assertTrue(torch.isfinite(out['total']))
                if kind=='win':self.assertTrue(batch['won'][0,3]);self.assertEqual(int(batch['distances'][0,3]),0);self.assertEqual(int(batch['next_optimal'][0,3]),0)
                if kind=='deadend':self.assertEqual(int(batch['distances'][0,3]),-1);self.assertEqual(int(batch['next_optimal'][0,3]),0);self.assertFalse(batch['terminal'][0].any())
                if kind=='reset':
                    self.assertTrue(batch['lost_life'][0,3]);self.assertFalse(batch['terminal'][0,3]);self.assertEqual(int(batch['distances'][0,3]),3)
                    f,v,p=windows[3];self.assertTrue(np.all(f==f[-1]));self.assertEqual(v.tolist(),[False]*7+[True]);self.assertEqual(p.tolist(),[-1]*8)
                with torch.no_grad():
                    for h,(f,v,p) in enumerate(windows):
                        expected=model.encode(torch.from_numpy(f[None]).long(),torch.from_numpy(v[None]),torch.from_numpy(p[None]))['latent'][0]
                        torch.testing.assert_close(out['targets'][0,h],expected,atol=2e-6,rtol=2e-5)
                expected=draft._value_loss(model,out['predicted'].flatten(0,1),batch['distances'].long().flatten(),batch['terminal'].flatten(),batch['won'].flatten())[0]
                torch.testing.assert_close(out['losses']['imagined_value'],expected)
                from pebby.agent.world_grounding import world_grounding_losses
                torch.testing.assert_close(out['losses']['grounding'],world_grounding_losses(model,batch,out['latent'],out['targets'],out['predicted'])[0])
                glyph=torch.cat((model.glyph_logits(batch['frames'][:,-1]),model.glyph_logits(batch['next_frames'].flatten(0,1))))
                labels=torch.cat((batch['current_triple'],batch['next_triple'].flatten(0,1)))
                torch.testing.assert_close(out['losses']['glyph'],draft.GlyphEncoder.loss(glyph,labels))

    def test_flags_and_future_labels_never_enter_autoregressive_predictions_or_current_policy(self):
        batch,_=trajectory('live');changed={k:v.clone() for k,v in batch.items()};changed['lost_life'][0,3]=True
        changed['next_frames'][0,3]=6;changed['terminal'][0,3]=True;changed['distances'][0,3]=-1;changed['next_optimal'][0,3]=0
        model=self.model();a=self.loss(model,batch);b=self.loss(model,changed)
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0);torch.testing.assert_close(a['predicted'],b['predicted'],atol=0,rtol=0)
        self.assertFalse(torch.allclose(a['targets'],b['targets']))

    def test_forbid_interior_closing_flags_and_keep_legacy_parity(self):
        batch,_=trajectory('live');model=self.model()
        for name in ('terminal','won','lost_life'):
            for h in range(3):
                changed={k:v.clone() for k,v in batch.items()};changed[name][0,h]=True
                with self.assertRaisesRegex(ValueError,'interior'):self.loss(model,changed)
        a=self.loss(model,batch);b=legacy.world_losses(model,batch,{'successor_policy':1.},sigreg_generator=torch.Generator().manual_seed(8))
        for name in ('total','logits','targets','predicted'):torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)
        plain={k:v for k,v in batch.items() if k not in ('rollout_mask','rollout_actions')}
        a=self.loss(model,plain);b=legacy.world_losses(model,plain,{'successor_policy':1.},sigreg_generator=torch.Generator().manual_seed(8))
        for name in ('total','logits','targets','predicted'):torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)

    def test_final_closing_gradient_flows_through_all_four_predictions_and_encoder(self):
        batch,_=trajectory('reset');model=self.model();values=[]
        def capture(_m,_i,o):
            if o.requires_grad:o.retain_grad();values.append(o)
        hook=model.predictor.register_forward_hook(capture);out=self.loss(model,batch)
        F.mse_loss(out['predicted'][:1,3],out['targets'][:1,3].detach()).backward();hook.remove()
        for value in values[:4]:self.assertGreater(float(value.grad.abs().sum()),0)
        self.assertGreater(float(model.predictor.action_embedding.weight.grad.abs().sum()),0)
        model.zero_grad(set_to_none=True);out=self.loss(model,batch)
        F.mse_loss(out['targets'][:1,3],out['predicted'][:1,3].detach()).backward()
        self.assertGreater(float(model.stem[0].weight.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
