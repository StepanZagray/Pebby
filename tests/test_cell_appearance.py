"""Public-pixel decoder contracts; synthetic fixtures contain no engine inputs."""
import inspect,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.cell_appearance import CellAppearance,cell_patches,ATTRIBUTE_SIZES
from tools.train_cell_appearance import load_examples,digest

class CellAppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name)
        rng=np.random.default_rng(18);frames=rng.integers(0,16,(2,64,64),dtype=np.uint8)
        full=np.zeros((2,144),bool);full[:,[26,27,28,29]]=True
        support=full.copy();support[:,29]=False # 5x5 public, outer ring not public
        role=np.zeros((2,144),np.uint16);role[:,26]=2;role[:,27]=1;role[:,29]=128
        attrs=np.full((2,144,3),-1,np.int8);attrs[:,26]=[[2,1,3],[4,3,1]]
        data={'format':np.array(['pebby.visible-cell-labels.v4']),
              'frames':frames,'fully_visible':full,'hud_overlap':np.zeros_like(full),'label_mask':full.copy(),
              'support7_label_mask':support,'support9_label_mask':np.zeros_like(full),
              'split':np.array(['train','validation']),'seeds':np.array([18,1_000_018]),'roles':role,'goal_attrs':attrs}
        path=root/'labels.npz';proof=root/'proof.json'
        def save(update_proof=True):
            np.savez_compressed(path,**data)
            if update_proof:proof.write_text(json.dumps({'status':'complete','format':'pebby.visible-cell-label-feasibility.v4','initial_state_only':True,'output_npz':{'path':str(path),'sha256':digest(path)}}))
        save();return data,path,proof,save

    def test_exact_cell_spatial_placement_and_zero_padding(self):
        frame=torch.arange(64*64,dtype=torch.int64).reshape(64,64)
        frames=torch.stack((frame,frame+10000));patches=cell_patches(frames)
        self.assertEqual(tuple(patches.shape),(2,144,7,7))
        for batch in range(2):
            for row in range(12):
                for col in range(12):
                    expected=torch.zeros(7,7,dtype=torch.int64)
                    for dy in range(7):
                        for dx in range(7):
                            y=5*row-1+dy;x=4+5*col-1+dx
                            if 0<=y<64 and 0<=x<64:expected[dy,dx]=frames[batch,y,x]
                    torch.testing.assert_close(patches[batch,12*row+col],expected)
        self.assertTrue((patches[:,0,0]==0).all())
        self.assertTrue((patches[:,11,:,-1]==0).all())

    def test_dense_forward_matches_independent_public_crops(self):
        torch.manual_seed(73);model=CellAppearance().eval()
        frames=torch.randint(16,(2,64,64),dtype=torch.uint8);pieces=[]
        for image in frames:
            for row in range(12):
                for col in range(12):
                    patch=torch.zeros(7,7,dtype=image.dtype)
                    for dy in range(7):
                        for dx in range(7):
                            y=row*5-1+dy;x=col*5+3+dx
                            if 0<=y<64 and 0<=x<64:patch[dy,dx]=image[y,x]
                    pieces.append(patch)
        with torch.no_grad():
            dense=torch.cat(model(frames),-1)
            independently_cropped=model.patch_logits(torch.stack(pieces)).view(2,144,22)
        torch.testing.assert_close(dense,independently_cropped,atol=0,rtol=0)
        self.assertEqual(list(inspect.signature(model.forward).parameters),['frames'])
        with self.assertRaises(TypeError):model(frames,label_mask=torch.ones(2,144,dtype=torch.bool))

    def test_loader_excludes_outer_ring_unknown_and_preserves_floor(self):
        data,path,proof,_=self.fixture();examples,_=load_examples(path,proof)
        for split,row in [('train',0),('validation',1)]:
            e=examples[split];self.assertEqual(e['levels'],1);self.assertEqual(len(e['patches']),3)
            self.assertEqual(e['role_masks'].tolist(),[2,1,0]);self.assertEqual(e['goals'].tolist(),[True,False,False])
            x,y=4+5*2,5*2
            torch.testing.assert_close(e['patches'][0],torch.from_numpy(data['frames'][row,y-1:y+6,x-1:x+6]))
            self.assertFalse(e['role_bits'][:,7].any()) # excluded5x5-only player label

    def test_visible_targets_send_gradients_to_all_role_and_attribute_outputs(self):
        _,path,proof,_=self.fixture();examples,_=load_examples(path,proof)
        e=examples['train'];torch.manual_seed(4);model=CellAppearance()
        roles,*attrs=model.patch_logits(e['patches']).split((8,*ATTRIBUTE_SIZES),-1)
        loss=F.binary_cross_entropy_with_logits(roles,e['role_bits'].float())
        loss+=sum(F.cross_entropy(logits[e['goals']],e['attributes'][e['goals'],j]) for j,logits in enumerate(attrs))/3
        loss.backward();gradient=model.network[-1].weight.grad
        self.assertTrue(torch.isfinite(loss));self.assertTrue((gradient.abs().sum(1)>0).all())
        self.assertGreater(float(model.network[0].weight.grad.abs().sum()),0)

    def test_altered_data_or_incomplete_proof_rejected(self):
        data,path,proof,save=self.fixture();data['frames'][0,0,0]^=1;save(False)
        with self.assertRaisesRegex(ValueError,'audit'):load_examples(path,proof)
        save();p=json.loads(proof.read_text());p['status']='running';proof.write_text(json.dumps(p))
        with self.assertRaisesRegex(ValueError,'audit'):load_examples(path,proof)

    def test_invalid_splits_and_duplicate_seeds_rejected(self):
        for which in ('split','boundary','duplicate'):
            with self.subTest(which=which):
                data,path,proof,save=self.fixture()
                if which=='split':data['split'][1]='test'
                elif which=='boundary':data['seeds'][1]=123
                else:data['seeds'][1]=data['seeds'][0]
                save()
                with self.assertRaises(ValueError):load_examples(path,proof)

    def test_invalid_masks_and_missing_support_rejected(self):
        for which in ('base','support7','support9','missing'):
            with self.subTest(which=which):
                data,path,proof,save=self.fixture()
                if which=='base':data['label_mask'][0,0]=True
                elif which=='support7':data['support7_label_mask'][0,0]=True
                elif which=='support9':data['support9_label_mask'][0,29]=True
                else:del data['support7_label_mask']
                save()
                with self.assertRaises(ValueError):load_examples(path,proof)

    def test_palette_and_shape_inputs_rejected(self):
        model=CellAppearance()
        for pixels in (torch.zeros(1,7,7),torch.zeros(1,7,7,dtype=torch.bool),torch.zeros(1,5,5,dtype=torch.long)):
            with self.assertRaises(ValueError):model.patch_logits(pixels)
        for value in (-1,16):
            with self.assertRaises((ValueError,RuntimeError)):model.patch_logits(torch.full((1,7,7),value))
        with self.assertRaises(ValueError):cell_patches(torch.zeros(1,63,64,dtype=torch.uint8))

if __name__=='__main__':unittest.main()
