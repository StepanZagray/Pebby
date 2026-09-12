import copy
import tempfile
import unittest
from pathlib import Path
import torch
from tests import test_world_exploratory_sequences as generated
from pebby.agent.structured_factored_policy import (StructuredFactoredPolicy,save_factored_policy_checkpoint,load_factored_policy_checkpoint,state_digest,FACTORED_POLICY_FORMAT)
from pebby.agent.model import load_checkpoint

WORLD='checkpoints/ls20-world-cell-recall-b1024.pt'
VIS='checkpoints/ls20-cell-visibility-initial-200.pt'

@unittest.skipUnless(all(Path(p).exists() for p in (WORLD,VIS,'checkpoints/ls20-structured-glyph-local300.pt','checkpoints/ls20-structured-glyph-global300.pt')), 'requires local generated diagnostic checkpoints')
class FactoredPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);generated.ExploratoryTrainingTests.setUpClass();cls.data=generated.ExploratoryTrainingTests.data
        cls.local=StructuredFactoredPolicy.from_checkpoints(WORLD,VIS,'checkpoints/ls20-structured-glyph-local300.pt')
        cls.global_model=StructuredFactoredPolicy.from_checkpoints(WORLD,VIS,'checkpoints/ls20-structured-glyph-global300.pt')
    def test_real_generated_public_history_frozen_gradient_and_counts(self):
        for policy,total in [(self.local,1029816),(self.global_model,946584)]:
            policy.train();policy.zero_grad();frames=torch.tensor(self.data['frames'][:2]);valid=torch.tensor(self.data['history_valid'][:2]);actions=torch.tensor(self.data['previous_actions'][:2])
            logits=policy(frames,valid,actions);self.assertEqual(tuple(logits.shape),(2,4));self.assertTrue(torch.isfinite(logits).all())
            torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])).backward()
            self.assertGreater(float(policy.readout.scorer.weight.grad.abs().sum()),0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in policy.encoder.parameters()))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in policy.dynamics.parameters()))
            self.assertEqual(policy.parameter_count(),total);self.assertEqual(policy.parameter_counts()['trainable'],108097)
            with self.assertRaises(TypeError):policy(frames,next_fields=frames)
    def test_real_custom_and_generic_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt';save_factored_policy_checkpoint(path,self.local)
            for loader in (load_factored_policy_checkpoint,load_checkpoint):
                policy,saved=loader(path);self.assertEqual(saved['format'],FACTORED_POLICY_FORMAT);self.assertEqual(policy.parameter_count(),1029816)
                field=torch.randn(2,148,96)
                with torch.no_grad():torch.testing.assert_close(policy.forward_fields(field),self.local.forward_fields(field),atol=0,rtol=0)
    def test_provenance_weights_counts_config_tampering_failclosed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt';save_factored_policy_checkpoint(path,self.global_model);saved=torch.load(path,weights_only=True)
            for what in ('code','weights','counts','config','format'):
                bad=copy.deepcopy(saved)
                if what=='code':bad['sources']['code_hashes'].pop(next(iter(bad['sources']['code_hashes'])))
                elif what=='weights':bad['readout_weights']['scorer.bias']+=1
                elif what=='counts':bad['parameters']+=1
                elif what=='config':bad['config']['mode']='direct'
                else:bad['format']='wrong'
                torch.save(bad,path)
                with self.assertRaises(ValueError):load_factored_policy_checkpoint(path)
    def test_altered_frozen_weights_and_source_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.pt';saved=self.local.dynamics.position.detach().clone()
            with torch.no_grad():self.local.dynamics.position.add_(1)
            try:
                with self.assertRaisesRegex(ValueError,'frozen'):save_factored_policy_checkpoint(path,self.local)
            finally:
                with torch.no_grad():self.local.dynamics.position.copy_(saved)
            source=copy.deepcopy(self.local.sources);self.local.sources['artifacts']['dynamics']['sha256']='0'*64
            try:
                with self.assertRaisesRegex(ValueError,'hash'):save_factored_policy_checkpoint(path,self.local)
            finally:self.local.sources=source
    def test_unsupported_dynamics_format_rejected(self):
        with self.assertRaisesRegex(ValueError,'format'):
            StructuredFactoredPolicy.from_checkpoints(WORLD,VIS,'checkpoints/ls20-structured-transition-fit2000.pt')
