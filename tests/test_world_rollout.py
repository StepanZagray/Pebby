"""Production mixed K4 losses; real generated engine trajectories only."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent import world_model, world_data, world_grounding
from pebby.ls20 import generate, names
from pebby.agent import world_training_objectives as objectives
from tests.test_world_model import make_model
from tests.test_world_glyph import wake



class MixedRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        generated = generate.generate_level(0, 1)
        env, oracle, proof = world_data.verified_context(generated, context_index=3)
        assert env is not None and proof['context_engine_verified']
        route = oracle.solution()
        assert len(route) > 4 and not oracle.truncated
        rows, _ = world_data.collect_level(generated, history=8, samples=1, epsilon=0, context_index=3)
        ordinary = rows[0]
        sequence = copy.deepcopy(ordinary)
        frames, actions = [env.render()], [-1]
        future = {name: [] for name in ('next_frames','next_player_cell','next_triple',
                  'next_steps','next_lives','distances','next_optimal')}
        cls.windows = []
        for action in route[:4]:
            result = env.perform(action)
            assert not result.finished and env.lives() == 3
            frames.append(result.frame);actions.append(names.ACTION_IDS.index(action))
            cls.windows.append(world_data.history_arrays(frames, actions, 8))
            state = oracle.state_of(env)
            values = (result.frame, env.player_cell(), env.triple(), env.steps_left(),
                      env.lives(), oracle.distance_for(state), world_data.successor_optimal_mask(oracle,state))
            for key, value in zip(future, values):future[key].append(value)
        for key, value in future.items():sequence[key] = np.asarray(value, dtype=ordinary[key].dtype)
        for key in ('terminal','won','lost_life'):sequence[key] = np.zeros(4,dtype=bool)
        cls.batch = {key:torch.as_tensor(np.stack([sequence[key],ordinary[key]]))
                     for key in ordinary if isinstance(ordinary[key],(np.ndarray,int,np.integer,bool))}
        cls.batch.update(rollout_mask=torch.tensor([True,False]),
                         rollout_actions=torch.tensor([actions[1:],[-1,-1,-1,-1]]))

    def model(self):
        torch.manual_seed(123)
        return wake(make_model(history=8, grounding=True, glyph_recall=True, query_readout=True)).eval()

    def loss(self, model, batch=None, **kwargs):
        return objectives.world_losses(model,self.batch if batch is None else batch,
            {'successor_policy':1.},sigreg_generator=torch.Generator().manual_seed(7),**kwargs)

    def test_histories_match_independent_real_engine_encodes_and_cost(self):
        model = self.model()
        original_sigreg = model.sigreg.forward
        with patch.object(model,'frame_tokens',wraps=model.frame_tokens) as token_calls, \
             patch.object(model.sigreg,'forward',wraps=original_sigreg) as sigreg:
            out = self.loss(model)
        self.assertEqual(token_calls.call_count,2)
        self.assertEqual(tuple(sigreg.call_args.args[0].shape),(5,2,model.cfg.latent))
        with torch.no_grad():
            for horizon,(frames,valid,actions) in enumerate(self.windows):
                expected=model.encode(torch.as_tensor(frames)[None],torch.as_tensor(valid)[None],
                                      torch.as_tensor(actions)[None])['latent'][0]
                torch.testing.assert_close(out['targets'][0,horizon],expected,atol=2e-6,rtol=2e-5)
        ordinary={key:value[1:] for key,value in self.batch.items() if key not in ('rollout_mask','rollout_actions')}
        base=objectives.world_losses(model,ordinary,{'successor_policy':1.},sigreg_generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(out['targets'][1],base['targets'][0],atol=2e-6,rtol=2e-5)

    def test_four_step_recursion_and_current_action_alternatives(self):
        model=self.model();passed=[];original=model.logits_from
        def capture(encoding,*args,**kwargs):
            passed.append(kwargs.get('successors'))
            return original(encoding,*args,**kwargs)
        with patch.object(model,'logits_from',side_effect=capture):out=self.loss(model)
        first=model.predict_successors(out['latent'])
        torch.testing.assert_close(passed[0],first)
        state=out['latent'][:1]
        for h in range(4):
            state=model.predict_successors(state,self.batch['rollout_actions'][:1,h])
            torch.testing.assert_close(out['predicted'][:1,h],state)
        torch.testing.assert_close(out['predicted'][1],first[1])

    def test_gradients_through_unroll_current_and_online_targets(self):
        model=self.model();intermediates=[]
        def capture(_module,_args,output):
            if output.requires_grad:
                output.retain_grad();intermediates.append(output)
        handle=model.predictor.register_forward_hook(capture)
        out=self.loss(model);out['latent'].retain_grad()
        F.mse_loss(out['predicted'][:1,3],out['targets'][:1,3].detach()).backward()
        handle.remove()
        self.assertGreater(out['latent'].grad[0].abs().sum().item(),0)
        # All first + three recurrent prediction calls participate; no detach/teacher forcing.
        for value in intermediates[:4]:self.assertGreater(value.grad.abs().sum().item(),0)
        self.assertGreater(model.predictor.action_embedding.weight.grad.abs().sum().item(),0)
        model.zero_grad(set_to_none=True);token_outputs=[];original=model.frame_tokens
        def token_capture(*args,**kwargs):
            value=original(*args,**kwargs);value.retain_grad();token_outputs.append(value);return value
        with patch.object(model,'frame_tokens',side_effect=token_capture):out=self.loss(model)
        F.mse_loss(out['predicted'].detach(),out['targets']).backward()
        self.assertGreater(token_outputs[1].grad.abs().sum().item(),0)
        self.assertGreater(model.stem[0].weight.grad.abs().sum().item(),0)

    def test_future_actions_frames_and_labels_do_not_leak_to_current_policy(self):
        model=self.model();changed={key:value.clone() for key,value in self.batch.items()}
        changed['next_frames']=torch.full_like(changed['next_frames'],6)
        changed['rollout_actions'][0]=(changed['rollout_actions'][0]+1)%4
        changed['distances']+=3;changed['next_optimal']=torch.zeros_like(changed['next_optimal'])
        for key in ('current_triple','next_triple'):
            changed[key]=(changed[key]+1)%torch.tensor([6,4,4])
        changed['optimal']=torch.full_like(changed['optimal'],15)
        changed['next_player_cell']=(changed['next_player_cell']+1)%12
        with torch.no_grad():a=self.loss(model);b=self.loss(model,changed)
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)
        self.assertFalse(torch.allclose(a['targets'],b['targets']))
        self.assertFalse(torch.allclose(a['predicted'][0],b['predicted'][0]))

    def test_chronological_grounding_value_glyph_and_successor_policy_alignment(self):
        model=self.model();actual=[];original=model.logits_from
        def capture(encoding,*args,**kwargs):
            result=original(encoding,*args,**kwargs);actual.append(result[0]);return result
        with patch.object(model,'logits_from',side_effect=capture):out=self.loss(model)
        b=self.batch
        expected=world_grounding.world_grounding_losses(model,b,out['latent'],out['targets'],out['predicted'])[0]
        torch.testing.assert_close(out['losses']['grounding'],expected)
        expected=objectives._value_loss(model,out['predicted'].flatten(0,1),b['distances'].long().flatten(),
                                  b['terminal'].flatten(),b['won'].flatten())[0]
        torch.testing.assert_close(out['losses']['imagined_value'],expected)
        masks=b['next_optimal'].flatten();valid=masks!=0;bits=world_model.optimal_bits(masks[valid])
        expected=-(bits/bits.sum(-1,keepdim=True)*F.log_softmax(actual[1][valid],dim=-1)).sum(-1).mean()
        torch.testing.assert_close(out['losses']['successor_policy'],expected)
        glyph=torch.cat((model.glyph_logits(b['frames'][:,-1]),model.glyph_logits(b['next_frames'].flatten(0,1))))
        labels=torch.cat((b['current_triple'],b['next_triple'].flatten(0,1)))
        torch.testing.assert_close(out['losses']['glyph'],world_model.GlyphEncoder.loss(glyph,labels))

    def test_counterfactual_metrics_exclude_sequence_and_horizons_are_explicit(self):
        out=self.loss(self.model());d=out['diagnostics']
        rank,top=world_model._rank_of_true(out['predicted'][1:],out['targets'][1:])
        torch.testing.assert_close(d['counterfactual_mean_rank'],rank)
        torch.testing.assert_close(d['counterfactual_top1'],top)
        for h in range(4):
            torch.testing.assert_close(d[f'rollout_prediction_mse_h{h+1}'],F.mse_loss(out['predicted'][:1,h],out['targets'][:1,h]))
        all_sequence={key:value[:1] for key,value in self.batch.items()}
        d=self.loss(self.model(),all_sequence)['diagnostics']
        self.assertNotIn('counterfactual_mean_rank',d)
        self.assertEqual(d['counterfactual_rows'],0)
        self.assertEqual(out['diagnostic_weights']['counterfactual_top1'],1)
        self.assertEqual(out['diagnostic_weights']['rollout_prediction_mse_h4'],1)

    def test_absent_and_false_masks_keep_outputs_losses_and_gradients(self):
        model = self.model()
        ordinary = {key: value for key, value in self.batch.items()
                    if key not in ('rollout_mask', 'rollout_actions')}
        false = {**ordinary, 'rollout_mask': torch.zeros(2, dtype=torch.bool),
                 'rollout_actions': torch.full((2, 4), -1)}
        absent = self.loss(model, ordinary)
        masked = self.loss(model, false)
        for key in ('logits', 'latent', 'targets', 'predicted', 'total'):
            torch.testing.assert_close(masked[key], absent[key], atol=0, rtol=0)
        for group in ('losses', 'diagnostics'):
            for key in absent[group]:
                torch.testing.assert_close(masked[group][key], absent[group][key], atol=0, rtol=0)
        absent['total'].backward()
        gradients = {key: p.grad.clone() if p.grad is not None else None
                     for key, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        masked['total'].backward()
        for key, p in model.named_parameters():
            if gradients[key] is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, gradients[key], atol=0, rtol=0)

    def test_invalid_rollout_contract_fails_before_encoder(self):
        b=self.batch;bad=[{**b,'rollout_mask':b['rollout_mask'].long()},
            {**b,'rollout_actions':b['rollout_actions'].float()},
            {**b,'rollout_actions':torch.full((2,4),-1)},
            {k:v for k,v in b.items() if k!='rollout_actions'}]
        for key in ('terminal','won','lost_life'):
            changed={**b,key:b[key].clone()};changed[key][0,2]=True;bad.append(changed)
        model=self.model()
        for data in bad:
            with self.subTest(keys=data.keys()), patch.object(model,'frame_tokens',side_effect=AssertionError('encoder called')):
                with self.assertRaises(ValueError):self.loss(model,data)


if __name__=='__main__':unittest.main()
