"""Policy targets stay outside frozen public feature computation."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.structured_policy import StructuredPolicyReadout
from pebby.agent.structured_transition import StructuredTransition
from tools.train_structured_policy import (
    digest, load_policy_cache, loss_for_rows, outputs_for_rows, policy_terms,
)


class StructuredPolicyTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uniform_target_ce_and_entropy_are_not_optimal_mass_loss(self):
        terms = policy_terms(torch.zeros(2,4), torch.tensor([3,15]))
        torch.testing.assert_close(terms['ce'], torch.full((2,), math.log(4)))
        torch.testing.assert_close(terms['entropy'], torch.tensor([math.log(2),math.log(4)]))
        torch.testing.assert_close(terms['optimal_probability'], torch.tensor([.5,1.]))
        self.assertEqual(terms['correct'].tolist(), [1.,1.])
        for masks in (torch.tensor([0,1]),torch.tensor([16,1])):
            with self.assertRaises(ValueError):policy_terms(torch.zeros(2,4), masks)

    def test_extra_label_arrays_require_matching_hashes_and_valid_masks(self):
        # Isolate the added guard: the historical loader returns validated base arrays.
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            data={'seeds':np.array([10,20])}
            extra={'optimal':np.array([1,3]),'next_optimal':np.ones((2,4),np.int16),
                   'distances':np.zeros((2,4),np.int16),'source_rows':np.array([4,9])}
            manifest={'arrays':{}}
            for name,array in extra.items():
                target=path/(name+'.npy');np.save(target,array)
                manifest['arrays'][name]={'sha256':digest(target),'shape':list(array.shape),'dtype':str(array.dtype)}
            with patch('tools.train_structured_policy.load_cache',side_effect=lambda *a:(dict(data),manifest)):
                loaded,_=load_policy_cache(path,'train')
                np.testing.assert_array_equal(loaded['optimal'],[1,3])
                for name in ('optimal','distances'):
                    target=path/(name+'.npy');original=target.read_bytes()
                    target.write_bytes(original+b'changed')
                    with self.assertRaisesRegex(ValueError,'digest mismatch'):load_policy_cache(path,'train')
                    target.write_bytes(original)
                np.save(path/'optimal.npy',np.array([0,3]))
                manifest['arrays']['optimal']['sha256']=digest(path/'optimal.npy')
                with self.assertRaisesRegex(ValueError,'outside permitted'):load_policy_cache(path,'train')

    def test_future_and_teacher_isolation_and_frozen_successor_gradients(self):
        torch.manual_seed(42)
        data={'fields':np.random.default_rng(42).normal(size=(2,148,96)).astype(np.float32),
              'next_fields':np.random.default_rng(43).normal(size=(2,4,148,96)).astype(np.float32),
              'optimal':np.array([1,2])}
        direct=StructuredPolicyReadout({'mode':'direct','loops':1})
        successor=StructuredPolicyReadout({'mode':'successors','loops':1})
        # Neutral initialization has no feature effect; a moved scalar head exposes it.
        with torch.no_grad():
            direct.scorer.weight.normal_(std=.1)
            successor.scorer.weight.normal_(std=.1)
        dynamics=StructuredTransition(loops=1).eval().requires_grad_(False)
        rows=np.arange(2)
        before,_=outputs_for_rows(direct,data,rows,'cpu')
        predicted_before,_=outputs_for_rows(successor,data,rows,'cpu',dynamics,3)
        changed=dict(data, next_fields=data['next_fields']+10,optimal=np.array([4,8]))
        after,_=outputs_for_rows(direct,changed,rows,'cpu')
        predicted_after,_=outputs_for_rows(successor,changed,rows,'cpu',dynamics,3)
        torch.testing.assert_close(before['direct'],after['direct'],atol=0,rtol=0)
        torch.testing.assert_close(predicted_before['imagined'],predicted_after['imagined'],atol=0,rtol=0)
        self.assertFalse(torch.equal(predicted_before['actual'],predicted_after['actual']))
        loss,parts=loss_for_rows(successor,data,rows,'cpu',dynamics,3)
        torch.testing.assert_close(loss,(parts['actual']+parts['imagined'])/2)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in successor.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in dynamics.parameters()))
        dynamics.train()
        with self.assertRaisesRegex(ValueError,'frozen/eval'):
            outputs_for_rows(successor,data,rows,'cpu',dynamics,3)


if __name__=='__main__':unittest.main()
