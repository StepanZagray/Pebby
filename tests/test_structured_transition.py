"""Standalone spatial transitions: recurrence, source recall and trainable heads."""
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from pebby.agent.structured_transition import (StructuredTransition, StructuredTransitionConfig,
                                                StructuredFieldReadout, steps_targets)


class StructuredTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(723)
        self.model = StructuredTransition()
        self.field = torch.randn(2,148,96)
        self.actions = torch.tensor([0,3])

    def test_shapes_and_all_heads_receive_finite_supervised_gradients(self):
        output = self.model(self.field, self.actions)
        self.assertEqual(output['field'].shape, (2,148,96))
        heads = output['readout']
        shapes = {'player_logits':(2,144),'role_logits':(2,144,8),
                  'goal_shape_logits':(2,144,6),'goal_color_logits':(2,144,4),'goal_rotation_logits':(2,144,4),
                  'carried_shape_logits':(2,6),'carried_color_logits':(2,4),'carried_rotation_logits':(2,4),
                  'steps_logits':(2,44),'lives_logits':(2,4)}
        loss = output['field'].square().mean()
        for name, shape in shapes.items():
            self.assertEqual(heads[name].shape,shape)
            if name=='role_logits':
                loss = loss + F.binary_cross_entropy_with_logits(heads[name],torch.randint(2,shape).float())
            else:
                logits=heads[name].reshape(-1,shape[-1]);labels=torch.arange(len(logits))%shape[-1]
                loss = loss + F.cross_entropy(logits,labels)
        for logits in output['events'].values():
            self.assertEqual(logits.shape,(2,))
            loss = loss + F.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.]))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad,name)
            self.assertTrue(torch.isfinite(parameter.grad).all(),name)
            self.assertGreater(float(parameter.grad.abs().sum()),0,name)
        for name in ('cell','player'):
            grad=getattr(self.model.readout,name).weight.grad
            self.assertTrue((grad.abs().sum(-1)>0).all())
        self.assertTrue((self.model.readout.global_head[-1].weight.grad.abs().sum(-1)>0).all())
        self.assertTrue((self.model.event_head[-1].weight.grad.abs().sum(-1)>0).all())

    def test_action_dependence_and_loop_count_share_weights(self):
        a=self.model.predict(self.field,torch.zeros(2,dtype=torch.long))
        b=self.model.predict(self.field,torch.ones(2,dtype=torch.long))
        self.assertGreater(float((a-b).detach().abs().max()),1e-5)
        deep=StructuredTransition(loops=5)
        self.assertEqual(self.model.parameter_count(),deep.parameter_count())
        self.assertEqual(set(self.model.state_dict()),set(deep.state_dict()))
        deep.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(deep.predict(self.field,self.actions,loops=2),
                                   self.model.predict(self.field,self.actions),atol=0,rtol=0)
        self.assertFalse(torch.equal(a,self.model.predict(self.field,torch.zeros(2,dtype=torch.long),loops=1)))

    def test_rollout_composes_predictions_and_recalls_each_transition_source(self):
        self.field.requires_grad_(True)
        actions=torch.tensor([[0,1,2,3],[3,2,1,0]])
        sources=[]
        handle=self.model.block.register_forward_pre_hook(lambda _m,args:sources.append(args[1].detach().clone()))
        try: result=self.model.rollout(self.field,actions)
        finally:handle.remove()
        self.assertEqual(result['fields'].shape,(2,4,148,96))
        self.assertEqual(len(sources),8)
        current=self.field
        for step in range(4):
            expected_source=current+self.model.position[None]+self.model.action_embedding(actions[:,step])[:,None]
            for index in (2*step,2*step+1):torch.testing.assert_close(sources[index],expected_source,atol=0,rtol=0)
            direct=self.model(current,actions[:,step]);current=direct['field']
            torch.testing.assert_close(result['fields'][:,step],current,atol=0,rtol=0)
            for name,scores in direct['events'].items():torch.testing.assert_close(result['events'][name][:,step],scores,atol=0,rtol=0)
        result['fields'].retain_grad()
        result['fields'][:,-1].square().mean().backward()
        self.assertGreater(float(self.model.block.attention.in_proj_weight.grad.abs().sum()),0)
        self.assertIsNotNone(self.field.grad)
        self.assertGreater(float(self.field.grad.abs().sum()),0)

    def test_actual_field_readout_does_not_call_transition_or_backprop_into_detached_input(self):
        actual=torch.randn(2,148,96,requires_grad=True)
        with patch.object(self.model,'predict',side_effect=AssertionError('unexpected transition')):
            heads=self.model.readout(actual.detach())
        sum(x.square().mean() for x in heads.values()).backward()
        self.assertIsNone(actual.grad)
        self.assertIsNone(self.model.action_embedding.weight.grad)
        self.assertGreater(float(self.model.readout.cell.weight.grad.abs().sum()),0)

    def test_events_depend_on_source_and_action_even_with_fixed_successor(self):
        with patch.object(self.model,'predict',return_value=self.field):
            baseline=self.model(self.field,self.actions)['events']
            altered=self.model(self.field+1,self.actions)['events']
            other_action=self.model(self.field,1-self.actions.remainder(2))['events']
        for name in baseline:
            self.assertFalse(torch.equal(baseline[name],altered[name]))
            self.assertFalse(torch.equal(baseline[name],other_action[name]))

    def test_state_dict_roundtrip_and_no_target_argument(self):
        buffer=io.BytesIO();torch.save({'config':self.model.config(),'weights':self.model.state_dict()},buffer);buffer.seek(0)
        checkpoint=torch.load(buffer,weights_only=True);restored=StructuredTransition(checkpoint['config']);restored.load_state_dict(checkpoint['weights'])
        torch.testing.assert_close(restored.predict(self.field,self.actions),self.model.predict(self.field,self.actions),atol=0,rtol=0)
        with self.assertRaises(TypeError):self.model(self.field,self.actions,targets=self.field)
        with self.assertRaises(TypeError):self.model.rollout(self.field,self.actions[:,None],target_fields=self.field)

    def test_contract_rejects_wrong_shapes_types_actions_and_empty_rollout(self):
        for field in (self.field[:,:144],self.field[:,:,:64],self.field.long()):
            with self.assertRaises(ValueError):self.model(field,self.actions)
        for actions in (self.actions.float(),self.actions[:,None],torch.tensor([-1,4]),torch.tensor([0])):
            with self.assertRaises(ValueError):self.model(self.field,actions)
        for loops in (0,True,1.5):
            with self.assertRaises(ValueError):self.model.predict(self.field,self.actions,loops=loops)
        with self.assertRaises(ValueError):self.model.rollout(self.field,torch.empty(2,0,dtype=torch.long))
        for config in ({'loops':0},{'heads':5},{'expansion':True}):
            with self.assertRaises(ValueError):StructuredTransitionConfig(**config)

    def test_budget_boundaries_underflow_is_distinct_from_zero(self):
        values=torch.tensor([[-3,-2,-1,0],[1,40,41,42]],dtype=torch.int16)
        expected=torch.tensor([[43,43,43,0],[1,40,41,42]])
        actual=steps_targets(values)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertEqual(actual.shape,values.shape)
        self.assertEqual(actual.device,values.device)
        self.assertEqual(actual.dtype,torch.long)
        self.assertEqual(steps_targets(torch.tensor(-1)).item(),43)
        self.assertEqual(steps_targets(torch.empty(0,dtype=torch.long)).numel(),0)
        logits=self.model.readout(self.field)['steps_logits']
        loss=F.cross_entropy(logits,steps_targets(torch.tensor([-3,0])))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(self.model.readout.global_head[-1].weight.grad.abs().sum()),0)

    def test_budget_mapper_rejects_unobserved_values_and_noninteger_labels(self):
        for values in (torch.tensor([-4]),torch.tensor([43]),torch.tensor([0.,42.]),
                       torch.tensor([True]),torch.tensor([complex(1,0)]),[-1,0]):
            with self.subTest(values=values),self.assertRaises(ValueError):steps_targets(values)
        for dtype in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
            torch.testing.assert_close(steps_targets(torch.tensor([0,42],dtype=dtype)),torch.tensor([0,42]))
        with self.assertRaises(ValueError):StructuredTransitionConfig(steps_classes=43)
        with self.assertRaises(ValueError):StructuredFieldReadout(steps_classes=43)


if __name__=='__main__':unittest.main()
