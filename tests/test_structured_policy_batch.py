import copy
import unittest
import numpy as np
import torch
from tools.structured_policy_batch import prepared_policy_inputs,outputs_for_prepared_batch
from tools.train_structured_policy import new_head,outputs_for_rows,policy_terms


class PreparedBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def batch(self):
        rng=np.random.default_rng(42)
        return {'fields':rng.normal(size=(2,148,96)).astype(np.float16),
                'next_fields':rng.normal(size=(2,4,148,96)).astype(np.float16),
                'imagined_fields':rng.normal(size=(2,4,148,96)).astype(np.float32),
                'optimal':np.array([3,8],np.uint8)}

    def test_successor_does_not_read_current_and_shares_cpu_fp32_memory(self):
        batch=self.batch();batch['fields']=object()
        batch['next_triple']=object()  # labels other than optimal never read
        inputs,masks=prepared_policy_inputs(batch,'successors')
        np.testing.assert_array_equal(inputs['actual'],batch['next_fields'].astype(np.float32))
        np.testing.assert_array_equal(inputs['imagined'],batch['imagined_fields'])
        self.assertEqual(inputs['imagined'].data_ptr(),batch['imagined_fields'].ctypes.data)
        self.assertTrue(all(not x.requires_grad for x in inputs.values()))
        self.assertEqual(masks.tolist(),[3,8])

    def test_exact_inputs_logits_loss_and_head_gradients_against_old_path(self):
        for mode in ('direct','successors'):
            batch=self.batch();head=new_head({'mode':mode},42,'cpu')
            with torch.no_grad():
                for p in head.parameters():p.add_(.001*torch.randn_like(p))
            other=copy.deepcopy(head)
            original,masks=outputs_for_rows(head,batch,np.arange(2),'cpu')
            prepared,new_masks=outputs_for_prepared_batch(other,batch)
            torch.testing.assert_close(masks,new_masks,atol=0,rtol=0)
            for key in original:torch.testing.assert_close(original[key],prepared[key],atol=0,rtol=0)
            old_loss=sum(policy_terms(v,masks)['ce'].mean() for v in original.values())/len(original)
            new_loss=sum(policy_terms(v,new_masks)['ce'].mean() for v in prepared.values())/len(prepared)
            torch.testing.assert_close(old_loss,new_loss,atol=0,rtol=0)
            old_loss.backward();new_loss.backward()
            for a,b in zip(head.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)

    def test_reject_noncontiguous_quantized_imagined_or_invalid_masks(self):
        batch=self.batch();batch['imagined_fields']=batch['imagined_fields'][:,:,::-1,:]
        with self.assertRaisesRegex(ValueError,'contiguous'):prepared_policy_inputs(batch,'successors')
        batch=self.batch();batch['imagined_fields']=batch['imagined_fields'].astype(np.float16)
        with self.assertRaisesRegex(ValueError,'float32'):prepared_policy_inputs(batch,'successors')
        batch=self.batch();batch['optimal'][0]=0
        with self.assertRaisesRegex(ValueError,'1..15'):prepared_policy_inputs(batch,'direct')


if __name__=='__main__':unittest.main()
