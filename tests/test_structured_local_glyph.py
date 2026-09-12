import inspect
import unittest
import torch
from pebby.agent.structured_global_glyph import GlobalGlyphTransition
from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition,LOCAL_GLYPH_FORMAT

class LocalGlyphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def pair(self,loops=2):
        torch.manual_seed(42);base=GlobalGlyphTransition(loops=loops)
        torch.manual_seed(42);local=LocalGlobalGlyphTransition(loops=loops)
        return base,local
    def test_exact_neutral_predictions_actions_batches_and_warmstart(self):
        for loops in (1,2,4):
            base,local=self.pair(loops)
            for batch in (1,4):
                field=torch.randn(batch,148,96);actions=torch.arange(batch)%4
                torch.testing.assert_close(base.predict(field,actions),local.predict(field,actions),atol=0,rtol=0)
            with torch.no_grad():base.output[1].bias.add_(.1)
            local.warmstart_from_global_state_dict(base.state_dict())
            torch.testing.assert_close(base.predict(field,actions),local.predict(field,actions),atol=0,rtol=0)
    def test_local_gradients_and_hud_direct_branch(self):
        base,local=self.pair();state=torch.randn(3,148,96,requires_grad=True);source=torch.randn_like(state)
        result=local.block(state,source);result[:,:144].square().mean().backward()
        self.assertGreater(float(local.block.local_conv.weight.grad.abs().sum()),0)
        self.assertGreater(float(local.block.local_conv.bias.grad.abs().sum()),0)
        self.assertEqual(float(local.block.local_norm.weight.grad.abs().sum()),0)
        with torch.no_grad():local.block.local_conv.weight.normal_(0,.001)
        local.zero_grad();result=local.block(state,source)
        torch.testing.assert_close(result[:,144:],local.block.base(state,source)[:,144:],atol=0,rtol=0)
        result.square().mean().backward();self.assertGreater(float(local.block.local_norm.weight.grad.abs().sum()),0)
    def test_shared_parameters_h4_causality_and_roundtrip(self):
        _,model=self.pair();_,more=self.pair(4);self.assertEqual(model.parameter_count(),more.parameter_count())
        with torch.no_grad():model.block.local_conv.weight.normal_(0,.001)
        field=torch.randn(2,148,96);actions=torch.tensor([[0,1,2,3],[3,2,1,0]])
        recorded=[]
        def capture(_module,_args,out):out['field'].retain_grad();recorded.append(out['field'])
        hook=model.register_forward_hook(capture);roll=model.rollout(field,actions);hook.remove();roll['fields'][:,-1].square().mean().backward()
        self.assertEqual(len(recorded),4)
        self.assertTrue(all(x.grad is not None and x.grad.abs().sum()>0 for x in recorded))
        with torch.no_grad():
            current=field
            for h in range(4):
                current=model.predict(current,actions[:,h]);torch.testing.assert_close(current,roll['fields'][:,h],atol=0,rtol=0)
            other=LocalGlobalGlyphTransition(model.config());other.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(other.predict(field,actions[:,0]),model.predict(field,actions[:,0]),atol=0,rtol=0)
        self.assertEqual(model.checkpoint_format,LOCAL_GLYPH_FORMAT)
        with self.assertRaises(TypeError):model(field,actions[:,0],next_fields=field)
    def test_bad_warmstart_fails_before_mutation(self):
        base,local=self.pair();before={k:v.clone() for k,v in local.state_dict().items()};bad=dict(base.state_dict());bad.pop('position')
        with self.assertRaises(ValueError):local.warmstart_from_global_state_dict(bad)
        for key,value in local.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        bad=dict(base.state_dict());bad['position']=torch.zeros(1)
        with self.assertRaisesRegex(ValueError,'shape'):local.warmstart_from_global_state_dict(bad)
