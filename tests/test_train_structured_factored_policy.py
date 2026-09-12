"""Explicit dynamics dispatch and path-only normalization in the policy trainer."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from pebby.agent.structured_policy import save_structured_policy_checkpoint
from pebby.agent.structured_factored_policy import canonical_metadata, save_factored_policy_checkpoint
from tools.train_structured_policy import build_training_policy, check_policy_encoder, digest


class FactoredPolicyTrainerTests(unittest.TestCase):
    def test_direct_and_base_successors_keep_original_factory_and_serializer(self):
        dummy=SimpleNamespace(sources={})
        with patch('tools.train_structured_policy.StructuredFieldPolicy.from_checkpoints',return_value=dummy) as base:
            model,save,factored=build_training_policy('world','visibility',None,{'mode':'direct'})
            self.assertIs(model,dummy);self.assertIs(save,save_structured_policy_checkpoint);self.assertFalse(factored)
            self.assertEqual(base.call_args.kwargs,{'device':'cpu'})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'base.pt';torch.save({'format':'pebby.structured-transition.v1'},path)
            dummy.sources={'artifacts':{'dynamics':{'sha256':digest(path)}}}
            with patch('tools.train_structured_policy.StructuredFieldPolicy.from_checkpoints',return_value=dummy):
                _,save,factored=build_training_policy('world','visibility',path,{'mode':'successors'})
                self.assertIs(save,save_structured_policy_checkpoint);self.assertFalse(factored)

    def test_global_and_local_select_factored_serializer_and_unknown_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'dynamics.pt'
            for fmt in ('pebby.structured-transition-global-glyph.v1','pebby.structured-transition-local-global-glyph.v1'):
                torch.save({'format':fmt},path)
                dummy=SimpleNamespace(sources={'artifacts':{'dynamics':{'sha256':digest(path)}}})
                with patch('pebby.agent.structured_factored_policy.StructuredFactoredPolicy.from_checkpoints',return_value=dummy) as factory:
                    model,save,factored=build_training_policy('world','visibility',path,{'mode':'successors'})
                    self.assertIs(model,dummy);self.assertIs(save,save_factored_policy_checkpoint);self.assertTrue(factored)
                    self.assertEqual(factory.call_args.kwargs,{'device':'cpu'})
            torch.save({'format':'unrecognized.future-format'},path)
            with patch('tools.train_structured_policy.StructuredFieldPolicy.from_checkpoints') as base:
                with self.assertRaisesRegex(ValueError,'unsupported dynamics'):
                    build_training_policy('world','visibility',path,{'mode':'successors'})
                base.assert_not_called()
            with self.assertRaisesRegex(ValueError,'only successors'):
                build_training_policy('world','visibility',path,{'mode':'direct'})

    def test_loading_race_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'base.pt';torch.save({'format':'pebby.structured-transition.v1'},path)
            dummy=SimpleNamespace(sources={'artifacts':{'dynamics':{'sha256':'changed'}}})
            with patch('tools.train_structured_policy.StructuredFieldPolicy.from_checkpoints',return_value=dummy):
                with self.assertRaisesRegex(ValueError,'changed during'):
                    build_training_policy('world','visibility',path,{'mode':'successors'})

    def test_factored_normalizes_paths_not_encoder_values(self):
        cache={'config':{'history':8},'sources':{'world_checkpoint':{'path':'checkpoints/example.pt','sha256':'same'}},
               'parameter_counts':{'total':10},'code_hashes':{'pebby/agent/structured_field.py':'code'}}
        actual=canonical_metadata({k:v for k,v in cache.items() if k!='code_hashes'})
        model=SimpleNamespace(sources={'encoder_metadata':actual})
        check_policy_encoder(cache,model,True)
        with self.assertRaisesRegex(ValueError,'encoders differ'):check_policy_encoder(cache,model,False)
        for kind in ('hash','config','count'):
            changed=copy.deepcopy(cache)
            if kind=='hash':changed['sources']['world_checkpoint']['sha256']='other'
            elif kind=='config':changed['config']['history']=1
            else:changed['parameter_counts']['total']=11
            with self.assertRaisesRegex(ValueError,'encoders differ'):check_policy_encoder(changed,model,True)


if __name__=='__main__':unittest.main()
