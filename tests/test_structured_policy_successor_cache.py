"""Real generated-cache parity and fail-closed imagined-cache validation."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.cache_structured_policy_successors import load_imagined_cache
from tools.train_structured_policy import build_training_policy,digest,load_policy_cache,outputs_for_rows,imagined_fields


class SuccessorCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source=Path('data/structured-field-cache-smoke8/train')
        cls.cache=Path('data/structured-policy-imagined-smoke8')
        if not cls.cache.exists():raise unittest.SkipTest('run real eight-level imagined cache CPU builder first')
        cls.policy,_,_=build_training_policy('checkpoints/ls20-world-cell-recall-b1024.pt',
            'checkpoints/ls20-cell-visibility-initial-200.pt','checkpoints/ls20-structured-glyph-local300.pt',{'mode':'successors'})
        cls.data,cls.manifest=load_policy_cache(cls.source,'train')

    def test_actual_cpu_predictions_losses_and_gradients_match_on_the_fly(self):
        cache,_=load_imagined_cache(self.cache,'train',self.source,self.data,self.manifest,self.policy)
        current=torch.tensor(np.array(self.data['fields'])).float()
        actual=imagined_fields(self.policy.dynamics,current,2)
        torch.testing.assert_close(actual,torch.tensor(np.array(cache)),atol=0,rtol=0)
        head=self.policy.readout
        with torch.no_grad():head.scorer.weight.normal_(std=.01)
        rows=np.array([3,0]);live,labels=outputs_for_rows(head,self.data,rows,'cpu',self.policy.dynamics,2)
        cached=dict(self.data,imagined_fields=cache)
        with patch.object(self.policy.dynamics,'predict',side_effect=AssertionError('cached training must not predict again')):
            saved,savedlabels=outputs_for_rows(head,cached,rows,'cpu',self.policy.dynamics,2)
        torch.testing.assert_close(labels,savedlabels,atol=0,rtol=0)
        for key in live:torch.testing.assert_close(live[key],saved[key],atol=0,rtol=0)
        live['imagined'].sum().backward();grad=head.scorer.weight.grad.clone();head.zero_grad()
        saved['imagined'].sum().backward();torch.testing.assert_close(grad,head.scorer.weight.grad,atol=0,rtol=0)

    def test_wrong_model_and_wrong_source_binding_rejected(self):
        wrong=SimpleNamespace(sources=copy.deepcopy(self.policy.sources))
        wrong.sources['artifacts']['dynamics']['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'model binding'):
            load_imagined_cache(self.cache,'train',self.source,self.data,self.manifest,wrong)
        manifest=copy.deepcopy(self.manifest);manifest['arrays']['fields']['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'source/order'):
            load_imagined_cache(self.cache,'train',self.source,self.data,manifest,self.policy)

    def test_corrupted_hash_shape_and_order_rejected_even_with_updated_manifest(self):
        for fault in ('hash','shape','order'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)/'cache';shutil.copytree(self.cache,root)
                subpath=root/'train/manifest.json';sub=json.loads(subpath.read_text())
                name='seeds' if fault=='order' else 'imagined_fields';path=root/'train'/(name+'.npy')
                if fault=='hash':path.write_bytes(path.read_bytes()+b'corrupt')
                else:
                    values=np.load(path)
                    values=values[::-1] if fault=='order' else values[:1]
                    np.save(path,values)
                    sub['arrays'][name].update(sha256=digest(path),shape=list(values.shape))
                    subpath.write_text(json.dumps(sub))
                    mainpath=root/'manifest.json';main=json.loads(mainpath.read_text());main['splits']['train']['sha256']=digest(subpath)
                    mainpath.write_text(json.dumps(main))
                with self.assertRaises(ValueError):load_imagined_cache(root,'train',self.source,self.data,self.manifest,self.policy)


if __name__=='__main__':unittest.main()
