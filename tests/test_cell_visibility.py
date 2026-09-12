"""Pixel-only inference and label/provenance checks for the global pilot."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.cell_visibility import CellVisibility
from tools.train_cell_visibility import binary_metrics, digest, load_data, save_progress


class CellVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_frame_only_global_receptive_field_and_gradients(self):
        torch.manual_seed(9)
        model = CellVisibility()
        self.assertLess(model.parameter_count(), 150000)
        self.assertEqual(list(inspect.signature(model.forward).parameters), ['frames'])
        frame = torch.zeros(2,64,64,dtype=torch.uint8); frame[1,50:60,40:50] = 8
        logits = model(frame)
        self.assertEqual(tuple(logits.shape),(2,144))
        # A distant region affects the top-left output, demonstrating that the
        # head can use global evidence beyond its corresponding local patch.
        self.assertNotEqual(float(logits[0,0].detach()),float(logits[1,0].detach()))
        labels = torch.arange(144).remainder(2).expand(2,-1).float()
        F.binary_cross_entropy_with_logits(logits,labels).backward()
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()),0)
        with self.assertRaises(TypeError):model(frame,player_cell=torch.zeros(2,2))
        for bad in (frame.float(),frame.bool(),frame[:,:,:63],torch.full_like(frame,16)):
            with self.assertRaises(ValueError):model(bad)
        restored=CellVisibility();restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(frame),restored(frame),atol=0,rtol=0)

    def test_progress_checkpoint_preserves_weights_without_final_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.pt'; model = CellVisibility()
            save_progress(path, model, 100, 200, {'fixture': 'hash'}, [{'step': 100}])
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['status'], 'training_progress')
            self.assertEqual(saved['completed_updates'], 100)
            self.assertEqual(saved['planned_updates'], 200)
            restored = CellVisibility(); restored.load_state_dict(saved['weights'])
            frame = torch.zeros(1,64,64,dtype=torch.uint8)
            torch.testing.assert_close(model(frame), restored(frame), atol=0, rtol=0)
            self.assertFalse(path.with_suffix('.tmp.pt').exists())

    def test_false_positive_partition_and_exact_boards(self):
        labels=np.array([[1,0,0,0],[1,0,0,0]],bool)
        predicted=np.array([[1,1,0,0],[0,0,1,1]],bool)
        reasons={name:np.tile(np.arange(4)==column,(2,1)) for name,column in [('fog',1),('hud',2),('boundary',3)]}
        result=binary_metrics(predicted,labels,reasons)
        self.assertEqual(result['false_positive_by_reason'],{'fog':1,'hud':1,'boundary':1})
        self.assertEqual(result['false_negative'],1)
        self.assertEqual(result['exact_boards'],0)
        self.assertEqual(binary_metrics(labels,labels,reasons)['exact_boards'],2)

    def test_loader_proof_split_and_no_label_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'data.npz';proof=root/'proof.json'
            mask=np.zeros((2,144),bool);mask[:,13]=True
            data=dict(frames=np.zeros((2,64,64),np.uint8),seeds=np.array([100,1000100]),
                      split=np.array(['train','validation']),fully_visible=mask.copy(),
                      hud_overlap=np.zeros_like(mask),label_mask=mask.copy(),support7_label_mask=mask.copy())
            def save():
                np.savez(path,**data)
                proof.write_text(json.dumps(dict(status='complete',initial_state_only=True,output_npz=dict(sha256=digest(path)))))
            save();loaded,reasons,_=load_data(path,proof)
            self.assertEqual(loaded['support7_label_mask'].shape,(2,144))
            self.assertEqual(sum(int(r.sum()) for r in reasons.values()),286)
            data['frames'][0,0,0]=1;np.savez(path,**data)
            with self.assertRaisesRegex(ValueError,'proof'):load_data(path,proof)
            data['seeds'][1]=100;save()
            with self.assertRaisesRegex(ValueError,'distinct'):load_data(path,proof)
            data['seeds'][1]=1000100;data['support7_label_mask'][:,0]=True;data['label_mask'][:,0]=True;data['fully_visible'][:,0]=True;save()
            with self.assertRaisesRegex(ValueError,'fixed excluded'):load_data(path,proof)


if __name__ == '__main__':
    unittest.main()
