import unittest

import torch
from torch.nn import functional as F

from pebby.agent.structured_transition import StructuredTransition
from pebby.agent.structured_global_glyph import (
    GlobalGlyphTransition, GLOBAL_GLYPH_FORMAT,
)


class GlobalGlyphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.field = torch.randn(2,148,96)
        self.actions = torch.tensor([[0,1,2,3],[3,2,1,0]])

    def test_warmstart_preserves_other_channels_and_readout(self):
        base = StructuredTransition()
        model = GlobalGlyphTransition()
        new_keys = model.warmstart_from_base_state_dict(base.state_dict())
        self.assertTrue(all(k.startswith(('glyph_head.','glyph_pool_score.')) for k in new_keys))
        action = self.actions[:,0]
        expected = base(self.field,action)
        actual = model(self.field,action)
        for start,stop in ((0,70),(84,96)):
            torch.testing.assert_close(actual['field'][...,start:stop],expected['field'][...,start:stop],atol=0,rtol=0)
        for key,value in model.readout(actual['field']).items():
            torch.testing.assert_close(actual['readout'][key],value,atol=0,rtol=0)
        self.assertEqual(model.parameter_count(),211432)
        self.assertEqual(model.config()['variant'],'global_glyph')
        self.assertEqual(model.checkpoint_format,GLOBAL_GLYPH_FORMAT)
        wrong = dict(base.state_dict());wrong.pop(next(iter(wrong)))
        with self.assertRaisesRegex(ValueError,'exactly'):model.warmstart_from_base_state_dict(wrong)
        clone = GlobalGlyphTransition(model.config())
        clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(clone.predict(self.field,action),actual['field'],atol=0,rtol=0)

    def test_h4_is_chronological_with_one_bounded_global_distribution(self):
        model = GlobalGlyphTransition(loops=1)
        rollout = model.rollout(self.field,self.actions)
        current = self.field
        for step in range(4):
            one = model(current,self.actions[:,step])
            torch.testing.assert_close(one['field'],rollout['fields'][:,step],atol=0,rtol=0)
            current = one['field']
            for start,stop in ((70,76),(76,80),(80,84)):
                values = current[...,start:stop]
                self.assertTrue(bool(((values>=0)&(values<=1)).all()))
                torch.testing.assert_close(values,values[:,:1].expand_as(values),atol=0,rtol=0)
                torch.testing.assert_close(values.sum(-1),torch.ones(2,148))
        changed = self.actions.clone();changed[:,1] = (changed[:,1]+1)%4
        second = model.rollout(self.field,changed)
        torch.testing.assert_close(rollout['fields'][:,0],second['fields'][:,0],atol=0,rtol=0)
        self.assertFalse(torch.equal(rollout['fields'][:,1],second['fields'][:,1]))

    def test_final_readout_loss_reaches_earlier_states_global_head_and_spatial_trunk(self):
        model = GlobalGlyphTransition(loops=1)
        field = self.field.clone().requires_grad_(True)
        intermediates = []
        def record(module,args,output):
            output['field'].retain_grad();intermediates.append(output['field'])
        hook = model.register_forward_hook(record)
        output = model.rollout(field,self.actions)
        hook.remove()
        # The experiment uses the unchanged predicted READOUT, not direct glyph logits.
        loss = sum(F.cross_entropy(output['readout']['carried_'+name+'_logits'][:,-1],torch.tensor([0,1]))
                   for name in ('shape','color','rotation'))
        loss.backward()
        for state in intermediates[:-1]:
            self.assertIsNotNone(state.grad)
            self.assertGreater(float(state.grad.abs().sum()),0)
        self.assertGreater(float(field.grad.abs().sum()),0)
        for prefix in ('glyph_pool_score','glyph_head','block','action_embedding','readout'):
            norm = sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
                       if name.startswith(prefix) and p.grad is not None)
            self.assertGreater(norm,0,prefix)


if __name__=='__main__':unittest.main()
