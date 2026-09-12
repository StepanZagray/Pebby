"""V2 removes only imagined storage quantization, with original guards intact."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests import test_structured_onpolicy_field_cache as fixtures
from tools import build_structured_onpolicy_field_cache as v1
from tools import build_structured_onpolicy_fp32_cache as v2


class FractionalDynamics(torch.nn.Module):
    def predict(self,fields,actions):
        return fields*1.0031+actions[:,None,None].float()/10+0.00001


class FP32OnpolicyCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OnpolicyFieldCacheTests.setUpClass()
        cls.fixture=fixtures.OnpolicyFieldCacheTests.fixture

    save_source=fixtures.OnpolicyFieldCacheTests.save_source

    def policy(self):
        result=fixtures.PublicPolicy();result.dynamics=FractionalDynamics()
        return result

    def test_fp32_output_exact_and_other_fields_identical_to_v1(self):
        original_v1_hash = v2.digest(v1.__file__)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,report,checkpoint=self.save_source(root)
            data,hashes=v2.load_source(source,report,checkpoint)
            result=v2.build(data,root/'v2',self.policy(),hashes,batch_size=2,max_encoder_batch=3)
            v1.build(data,root/'v1',self.policy(),hashes,batch_size=2,max_encoder_batch=3)
            self.assertEqual(result['format'],'pebby.structured-onpolicy-field-cache.v2')
            self.assertEqual(result['precision']['imagined_storage_dtype'],'float32')
            self.assertFalse(result['precision']['imagined_quantization'])
            self.assertNotIn('imagined_quantization_max_error',result['precision'])
            self.assertEqual(result['precision']['current_storage_dtype'],'float16')
            self.assertEqual(result['precision']['actual_storage_dtype'],'float16')
            current=np.load(root/'v2/fields.npy');actual=np.load(root/'v2/next_fields.npy')
            imagined=np.load(root/'v2/imagined_fields.npy')
            self.assertEqual(current.dtype,np.float16);self.assertEqual(actual.dtype,np.float16)
            self.assertEqual(imagined.dtype,np.float32)
            self.assertEqual(imagined.shape,(len(current),4,148,96))
            expected=FractionalDynamics().predict(torch.from_numpy(current).float().repeat_interleave(4,0),
                torch.arange(4).repeat(len(current))).reshape(imagined.shape).numpy()
            np.testing.assert_array_equal(imagined,expected)
            self.assertTrue(np.any(imagined!=expected.astype(np.float16).astype(np.float32)))
            for name,entry in result['arrays'].items():
                self.assertEqual(v2.digest(root/'v2'/f'{name}.npy'),entry['sha256'])
                if name!='imagined_fields':
                    np.testing.assert_array_equal(np.load(root/'v2'/f'{name}.npy'),np.load(root/'v1'/f'{name}.npy'))
            self.assertIn(str(Path(v2.__file__).resolve()),result['source_hashes'])
            self.assertEqual(v2.digest(v1.__file__), original_v1_hash)

    def test_source_drift_aborts_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,report,checkpoint=self.save_source(root)
            data,hashes=v2.load_source(source,report,checkpoint);policy=self.policy()
            original=policy.dynamics.predict
            def predict(*args):
                checkpoint.write_bytes(b'changed after load');return original(*args)
            policy.dynamics.predict=predict
            with self.assertRaisesRegex(ValueError,'source changed'):v2.build(data,root/'failed',policy,hashes)
            self.assertFalse((root/'failed').exists());self.assertFalse(list(root.glob('.failed-*')))

    def test_cli_publishes_v2_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,report,checkpoint=self.save_source(root)
            arguments=['--source',str(source),'--report',str(report),'--checkpoint',str(checkpoint),
                       '--out',str(root/'cli'),'--seconds','30']
            with patch('pebby.agent.model.load_checkpoint',return_value=(self.policy(),{})):
                self.assertEqual(v2.main(arguments),0)
            result=json.loads((root/'cli/manifest.json').read_text())
            self.assertEqual(result['arrays']['imagined_fields']['dtype'],'float32')
            with self.assertRaises(SystemExit):v2.main(arguments)


if __name__=='__main__':unittest.main()
