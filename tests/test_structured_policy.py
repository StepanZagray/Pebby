import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from pebby.agent.structured_policy import (
    STRUCTURED_POLICY_FORMAT, StructuredPolicyReadout, StructuredFieldPolicy,
    save_structured_policy_checkpoint, load_structured_policy_checkpoint, _digest,
)
from pebby.agent.structured_transition import StructuredTransition


class TinyPublicEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.calls=[]
    def metadata(self):
        return {'config':{'history':8},'sources':{'fixture':'public synthetic frames'},'parameter_counts':{'total':1}}
    def forward(self,frames,history_valid=None,previous_actions=None):
        self.calls.append((frames,history_valid,previous_actions))
        value=frames[:,-1].float().mean((1,2))*self.weight
        return value[:,None,None].expand(-1,148,96).clone()


def activate(head):
    with torch.no_grad():
        head.scorer.weight.copy_(torch.linspace(-.2,.2,96)[None])


class StructuredPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):torch.manual_seed(42)

    def test_neutral_initialization_exact_counts_and_gradient_after_head_move(self):
        field=torch.randn(2,148,96)
        for mode in ('direct','successors'):
            head=StructuredPolicyReadout({'mode':mode})
            self.assertEqual(head.parameter_count(),108097)
            inputs=field if mode=='direct' else field[:,None].expand(-1,4,-1,-1)
            logits=head(inputs)
            torch.testing.assert_close(logits,torch.zeros(2,4),atol=0,rtol=0)
            F.cross_entropy(logits,torch.tensor([0,1])).backward()
            self.assertGreater(float(head.scorer.weight.grad.abs().sum()),0)
            self.assertEqual(float(head.attention.in_proj_weight.grad.abs().sum()),0)
            head.zero_grad();activate(head)
            F.cross_entropy(head(inputs),torch.tensor([0,1])).backward()
            self.assertGreater(float(head.attention.in_proj_weight.grad.abs().sum()),0)
            self.assertGreater(float(head.action_queries.grad.abs().sum()),0)

    def test_direct_batch_and_spatial_position_sensitivity(self):
        head=StructuredPolicyReadout();activate(head);field=torch.randn(3,148,96)
        batched=head(field);single=torch.cat([head(row[None]) for row in field])
        torch.testing.assert_close(batched,single,atol=2e-6,rtol=2e-6)
        changed=head(field.roll(1,1))
        self.assertGreater(float((batched-changed).detach().abs().max()),1e-6)

    def test_successor_query_permutation_and_isolated_future_sensitivity(self):
        head=StructuredPolicyReadout({'mode':'successors'});activate(head)
        fields=torch.randn(2,4,148,96);before=head(fields)
        altered=fields.clone();altered[:,2,8:12,3]+=20
        after=head(altered)
        torch.testing.assert_close(before[:,[0,1,3]],after[:,[0,1,3]],atol=0,rtol=0)
        self.assertGreater(float((before[:,2]-after[:,2]).detach().abs().max()),1e-6)
        permutation=torch.tensor([3,0,2,1])
        reordered=head(fields[:,permutation],permutation[None].expand(2,-1))
        torch.testing.assert_close(reordered,before[:,permutation],atol=2e-6,rtol=2e-6)
        repeated=fields[:,:1].expand(-1,4,-1,-1)
        self.assertGreater(float(head(repeated).detach().std(-1).max()),1e-6)
        direct=StructuredPolicyReadout({'mode':'direct'});direct.load_state_dict(head.state_dict())
        torch.testing.assert_close(direct(fields[:,0]),head(repeated),atol=2e-6,rtol=2e-6)

    def test_wrapper_frozen_boundaries_bounded_successors_and_public_signature(self):
        encoder=TinyPublicEncoder();dynamics=StructuredTransition(loops=1)
        calls=[];original=dynamics.predict
        def recording(this,field,actions,**kwargs):
            calls.append((len(field),actions.detach().clone()))
            return original(field,actions,**kwargs)
        dynamics.predict=types.MethodType(recording,dynamics)
        policy=StructuredFieldPolicy(encoder,dynamics,{'mode':'successors','successor_batch_size':3})
        policy.train();activate(policy.readout)
        self.assertTrue(policy.readout.training);self.assertFalse(encoder.training);self.assertFalse(dynamics.training)
        self.assertTrue(all(not p.requires_grad for m in (encoder,dynamics) for p in m.parameters()))
        current=torch.randn(2,148,96,requires_grad=True)
        result=policy.forward_fields(current);F.cross_entropy(result,torch.tensor([0,1])).backward()
        self.assertIsNone(current.grad)
        self.assertTrue(all(p.grad is None for m in (encoder,dynamics) for p in m.parameters()))
        self.assertGreater(float(policy.readout.scorer.weight.grad.abs().sum()),0)
        self.assertEqual([x[0] for x in calls],[3,3,2])
        self.assertEqual(torch.cat([x[1] for x in calls]).tolist(),[0,1,2,3,0,1,2,3])
        policy(torch.zeros(1,8,64,64),torch.ones(1,8,dtype=torch.bool),torch.full((1,8),-1))
        self.assertEqual(len(encoder.calls[-1]),3)
        with self.assertRaises(TypeError):policy(torch.zeros(1,8,64,64),labels=torch.zeros(1))

    def test_successors_wrapper_has_no_current_direct_shortcut(self):
        dynamics=StructuredTransition(loops=1)
        def constant(this,fields,actions,**kwargs):return torch.zeros_like(fields)
        dynamics.predict=types.MethodType(constant,dynamics)
        policy=StructuredFieldPolicy(TinyPublicEncoder(),dynamics,{'mode':'successors'})
        activate(policy.readout)
        torch.testing.assert_close(policy.forward_fields(torch.randn(2,148,96)),
                                   policy.forward_fields(torch.randn(2,148,96)*3),atol=0,rtol=0)
        direct=StructuredFieldPolicy(TinyPublicEncoder(),config={'mode':'direct'})
        self.assertIsNone(direct.dynamics);self.assertEqual(direct.parameter_counts()['dynamics'],0)

    def test_checkpoint_roundtrip_source_pin_counts_and_encoder_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);world=root/'world.pt';visibility=root/'visibility.pt'
            world.write_bytes(b'public encoder fixture');visibility.write_bytes(b'visibility fixture')
            encoder=TinyPublicEncoder();bound=encoder.metadata()
            bound.update(code_hashes={str(world):_digest(world)},checkpoint_hashes={str(visibility):_digest(visibility)})
            dynamics=StructuredTransition(loops=1);dpath=root/'dynamics.pt'
            torch.save({'format':'pebby.structured-transition.v1','config':dynamics.config(),
                        'weights':dynamics.state_dict(),'cache_manifests':{'train':{'field_encoder':bound}}},dpath)
            with patch('pebby.agent.structured_policy.load_structured_field_encoder',side_effect=lambda *a,**kw:TinyPublicEncoder()):
                for mode in ('direct','successors'):
                    policy=StructuredFieldPolicy.from_checkpoints(world,visibility,dpath if mode=='successors' else None,{'mode':mode})
                    activate(policy.readout);destination=root/(mode+'.pt')
                    save_structured_policy_checkpoint(destination,policy,{'fixture_only':True})
                    restored,checkpoint=load_structured_policy_checkpoint(destination)
                    self.assertEqual(checkpoint['format'],STRUCTURED_POLICY_FORMAT)
                    self.assertEqual(checkpoint['parameters'],policy.parameter_count())
                    self.assertEqual(restored.config()['architecture'],'structured')
                    self.assertEqual(restored.config()['history'],8)
                    self.assertEqual(restored.parameter_counts(),policy.parameter_counts())
                    fields=torch.randn(2,148,96)
                    torch.testing.assert_close(restored.forward_fields(fields),policy.forward_fields(fields),atol=0,rtol=0)
                direct=torch.load(root/'direct.pt',map_location='cpu',weights_only=True)
                names={Path(p).name for p in direct['sources']['code_hashes']}
                self.assertTrue({'structured_field.py','world_model.py','world_readout.py',
                    'world_grounding.py','world_rollout.py','cell_appearance.py',
                    'cell_appearance_dense.py','glyph_model.py','cell_visibility.py'}<=names)
                # Simulate implementation drift without modifying active source.
                original_digest=_digest
                def changed_digest(path):
                    return 'f'*64 if Path(path).name=='world_model.py' else original_digest(path)
                with patch('pebby.agent.structured_policy._digest',side_effect=changed_digest):
                    with self.assertRaisesRegex(ValueError,'code hash mismatch'):
                        load_structured_policy_checkpoint(root/'direct.pt')
                damaged=copy.deepcopy(direct);damaged['parameters']+=1
                torch.save(damaged,root/'wrong-count.pt')
                with self.assertRaisesRegex(ValueError,'parameter counts mismatch'):
                    load_structured_policy_checkpoint(root/'wrong-count.pt')
                broken=copy.deepcopy(bound);broken['config']={'history':4}
                torch.save({'format':'pebby.structured-transition.v1','config':dynamics.config(),
                            'weights':dynamics.state_dict(),'cache_manifests':{'train':{'field_encoder':broken}}},dpath)
                with self.assertRaisesRegex(ValueError,'encoder mismatch'):
                    StructuredFieldPolicy.from_checkpoints(world,visibility,dpath,{'mode':'successors'})
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    load_structured_policy_checkpoint(root/'successors.pt')
                world.write_bytes(b'tampered public source')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    load_structured_policy_checkpoint(root/'direct.pt')

    def test_real_generated_public_history_with_bound_frozen_checkpoints(self):
        from pebby.ls20.env import Ls20Scenario
        from pebby.ls20.generate import build_level
        from pebby.agent.world_data import history_arrays
        from tests.test_policy_history import corridor
        env=Ls20Scenario(build_level(corridor()),1)
        history=tuple(torch.from_numpy(x)[None] for x in history_arrays([env.render()],[-1],8))
        for mode,total in [('direct',735152),('successors',926121)]:
            policy=StructuredFieldPolicy.from_checkpoints(
                'checkpoints/ls20-world-cell-recall-b1024.pt',
                'checkpoints/ls20-cell-visibility-initial-200.pt',
                'checkpoints/ls20-structured-transition-fit2000.pt' if mode=='successors' else None,
                {'mode':mode,'successor_batch_size':2})
            policy.train()
            with torch.no_grad():logits=policy(*history)
            torch.testing.assert_close(logits,torch.zeros(1,4),atol=0,rtol=0)
            self.assertEqual(policy.parameter_count(),total)
            self.assertEqual(policy.parameter_counts()['trainable'],108097)
            self.assertFalse(policy.encoder.training)


if __name__=='__main__':unittest.main()
