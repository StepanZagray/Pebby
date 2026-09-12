"""Public initial recall: neutral migration, shared memory and causal rollouts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pebby.agent.structured_transition import StructuredTransition
from pebby.agent.structured_recall_transition import (StructuredRecallTransition, StructuredRecallConfig,
                                                     save_recall_checkpoint, load_recall_checkpoint, FORMAT)


class StructuredRecallTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(62)
        self.field=torch.randn(2,148,96)
        self.memory=torch.randn(2,148,96)
        self.actions=torch.tensor([0,3])

    def test_same_seed_base_weights_and_zero_projection_bit_exact_outputs(self):
        torch.manual_seed(42);base=StructuredTransition()
        torch.manual_seed(42);recall=StructuredRecallTransition()
        for name,value in base.state_dict().items():
            torch.testing.assert_close(recall.state_dict()[name],value,atol=0,rtol=0)
        for loops in (1,2,4):
            reference=base(self.field,self.actions,loops=loops)
            actual=recall(self.field,self.actions,self.memory,loops=loops)
            torch.testing.assert_close(actual['field'],reference['field'],atol=0,rtol=0)
            for group in ('readout','events'):
                for name in reference[group]:torch.testing.assert_close(actual[group][name],reference[group][name],atol=0,rtol=0)
        memory_swapped=recall.predict(self.field,self.actions,self.memory.flip(0))
        torch.testing.assert_close(memory_swapped,base.predict(self.field,self.actions),atol=0,rtol=0)

    def test_zero_gate_learns_and_enabled_memory_receives_gradients(self):
        model=StructuredRecallTransition()
        output=model(self.field,self.actions,self.memory)
        output['field'].square().mean().backward()
        self.assertGreater(float(model.memory_output.weight.grad.abs().sum()),0)
        self.assertEqual(float(model.memory_attention.in_proj_weight.grad.abs().sum()),0)
        with torch.no_grad():model.memory_output.weight.copy_(torch.eye(96)*.2)
        model.zero_grad(set_to_none=True)
        memory=self.memory.clone().requires_grad_()
        result=model(self.field,self.actions,memory)
        loss=result['field'].square().mean()+sum(x.square().mean() for x in result['events'].values())
        loss=loss+sum(x.square().mean() for x in result['readout'].values());loss.backward()
        self.assertGreater(float(memory.grad.abs().sum()),0)
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().sum()),0,name)

    def test_enabled_recall_responds_to_example_and_spatial_memory_permutations(self):
        model=StructuredRecallTransition()
        with torch.no_grad():model.memory_output.weight.copy_(torch.eye(96)*.2)
        baseline=model.predict(self.field,self.actions,self.memory)
        for changed in (self.memory.flip(0),self.memory.flip(1)):
            different=model.predict(self.field,self.actions,changed)
            self.assertGreater(float((different-baseline).detach().abs().max()),1e-6)

    def test_rollout_keeps_initial_memory_fixed_and_recalls_own_current_prediction(self):
        model=StructuredRecallTransition(loops=3)
        with torch.no_grad():model.memory_output.weight.copy_(torch.eye(96)*.1)
        actions=torch.tensor([[0,1,2,3],[3,2,1,0]])
        field=self.field.clone().requires_grad_();memory=self.memory.clone().requires_grad_()
        with patch.object(model,'predict',wraps=model.predict) as predictions, \
                patch.object(model.memory_attention,'forward',wraps=model.memory_attention.forward) as attention:
            result=model.rollout(field,actions,memory)
        self.assertEqual(len(predictions.call_args_list),4);self.assertEqual(len(attention.call_args_list),12)
        for step,call in enumerate(predictions.call_args_list):
            self.assertIs(call.args[2],memory)
            expected=field if step==0 else result['fields'][:,step-1]
            torch.testing.assert_close(call.args[0],expected,atol=0,rtol=0)
        first=attention.call_args_list[0].args[1]
        for call in attention.call_args_list:
            torch.testing.assert_close(call.args[1],first,atol=0,rtol=0)
            self.assertIs(call.args[1],call.args[2])
        result['fields'][:,-1].square().mean().backward()
        self.assertGreater(float(field.grad.abs().sum()),0)
        self.assertGreater(float(memory.grad.abs().sum()),0)

    def test_loop_parameter_count_config_and_distinct_checkpoint_roundtrip(self):
        model=StructuredRecallTransition(loops=4,memory_heads=8)
        self.assertEqual(model.parameter_count(),StructuredRecallTransition(loops=1).parameter_count())
        self.assertGreater(model.parameter_count(),StructuredTransition().parameter_count())
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'recall.pt';save_recall_checkpoint(path,model)
            restored,checkpoint=load_recall_checkpoint(path)
            self.assertEqual(checkpoint['format'],FORMAT);self.assertEqual(restored.config(),model.config())
            torch.testing.assert_close(restored.predict(self.field,self.actions,self.memory),
                                       model.predict(self.field,self.actions,self.memory),atol=0,rtol=0)
            checkpoint['format']='pebby.structured-transition.v1';torch.save(checkpoint,path)
            with self.assertRaises(ValueError):load_recall_checkpoint(path)
        with self.assertRaises(TypeError):StructuredTransition(model.config())

    def test_required_memory_and_no_teacher_or_event_inputs(self):
        model=StructuredRecallTransition()
        with self.assertRaises(TypeError):model(self.field,self.actions)
        with self.assertRaises(TypeError):model.predict(self.field,self.actions)
        with self.assertRaises(TypeError):model.rollout(self.field,self.actions[:,None])
        for extras in ({'lost_life':torch.zeros(2)},{'target_fields':self.field},{'player_cell':torch.zeros(2,2)}):
            with self.assertRaises(TypeError):model(self.field,self.actions,self.memory,**extras)
        for memory in (self.memory[:1],self.memory[:,:144],self.memory.long()):
            with self.assertRaises(ValueError):model(self.field,self.actions,memory)
        with self.assertRaises(ValueError):StructuredRecallConfig(memory_heads=5)
        with self.assertRaises(ValueError):StructuredRecallConfig(memory_heads=True)


if __name__=='__main__':unittest.main()
