"""Equivalent frozen execution, palette-zero boundaries and bounded one-hot work."""
import unittest
from unittest.mock import patch
import torch
from pebby.agent.cell_appearance import CellAppearance
from pebby.agent.cell_appearance_dense import DenseCellAppearance


class DenseCellAppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_random_frames_all_cells_match_and_chunks_are_bounded(self):
        torch.manual_seed(319);source=CellAppearance().eval();dense=DenseCellAppearance(source,3)
        frames=torch.randint(16,(11,64,64),dtype=torch.uint8)
        with torch.no_grad(),patch.object(dense,'_chunk',wraps=dense._chunk) as calls:
            expected=source(frames);actual=dense(frames)
        self.assertEqual([len(c.args[0]) for c in calls.call_args_list],[3,3,3,2])
        for a,b,c in zip(actual,expected,(8,6,4,4)):
            self.assertEqual(tuple(a.shape),(11,144,c));torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
        for a,b in zip(actual,DenseCellAppearance(source,1)(frames)):
            torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)

    def test_palette_zero_padding_matches_at_every_boundary(self):
        source=CellAppearance().eval()
        with torch.no_grad():
            for p in source.parameters():p.zero_()
            # Every7x7 color0 pixel counts, so all-zero one-hot padding is detectably wrong.
            source.network[0].weight[0].reshape(7,7,16)[:,:,0]=.1
            source.network[2].weight[0,0]=1;source.network[4].weight[:,0]=1
        frame=torch.ones(1,64,64,dtype=torch.uint8);dense=DenseCellAppearance(source)
        with torch.no_grad():expected=source(frame);actual=dense(frame)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
        logits=actual[0][0,:,0].view(12,12)
        self.assertTrue((logits[0]>0).all());self.assertTrue((logits[:,-1]>0).all())
        self.assertTrue((logits[1:,:-1]==0).all())

    def test_copy_is_independent_frozen_and_preserves_source(self):
        source=CellAppearance();before={k:v.clone() for k,v in source.state_dict().items()}
        dense=DenseCellAppearance(source)
        self.assertEqual(dense.parameter_count(),source.parameter_count())
        self.assertTrue(all(not p.requires_grad for p in dense.parameters()))
        self.assertTrue(all(p.requires_grad for p in source.parameters()))
        self.assertTrue(source.training);self.assertFalse(dense.training)
        for k,v in source.state_dict().items():torch.testing.assert_close(v,before[k])
        original=dense.network[0].weight.clone()
        with torch.no_grad():source.network[0].weight.add_(1)
        torch.testing.assert_close(dense.network[0].weight,original)

    def test_invalid_inputs_and_empty_batch(self):
        source=CellAppearance();dense=DenseCellAppearance(source)
        for chunk in (0,-1,True,1.5):
            with self.assertRaises(ValueError):DenseCellAppearance(source,chunk)
        for frame in (torch.zeros(2,63,64,dtype=torch.long),torch.zeros(2,64,64),torch.zeros(2,64,64,dtype=torch.bool),torch.full((1,64,64),-1),torch.full((1,64,64),16)):
            with self.assertRaises(ValueError):dense(frame)
        for output,c in zip(dense(torch.empty(0,64,64,dtype=torch.uint8)),(8,6,4,4)):
            self.assertEqual(tuple(output.shape),(0,144,c))
        with self.assertRaises(TypeError):dense(torch.zeros(1,64,64,dtype=torch.uint8),labels=None)

    def test_default_constructor_reload_and_state_dict_round_trip(self):
        source=CellAppearance().eval();dense=DenseCellAppearance(chunk_size=2)
        self.assertEqual(dense.parameter_count(),110166)
        self.assertIs(dense.load_from(source),dense)
        restored=DenseCellAppearance(chunk_size=1)
        restored.load_state_dict(dense.state_dict(),strict=True)
        self.assertEqual(set(dense.state_dict()),{
            f'network.{layer}.{name}' for layer in (0,2,4) for name in ('weight','bias')})
        frames=torch.randint(16,(3,64,64),dtype=torch.uint8)
        with torch.no_grad():expected=source(frames)
        for outputs in (dense(frames),restored(frames)):
            for actual,target in zip(outputs,expected):
                torch.testing.assert_close(actual,target,atol=2e-5,rtol=2e-5)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))

if __name__=='__main__':unittest.main()
