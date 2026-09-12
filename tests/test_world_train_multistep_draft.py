"""Focused trainer-draft review using real generated corridor trajectories."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from pebby.agent import world_train as production
from pebby.agent.world_sequences import FourStepSampler
from pebby.agent.on_policy_provenance import file_digest
from tests.test_world_model import TINY, make_model
from tests import test_world_sequences as sequence_fixtures
from tests.test_world_rollout_draft import draft as draft_model


path=Path(__file__).resolve().parents[1]/'artifacts/world_train_multistep_draft.py'
spec=importlib.util.spec_from_file_location('pebby.agent._trainer_multistep_draft',path)
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


class TrainerMultistepDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.fixture=sequence_fixtures.WorldSequenceTests()
        # This archived trainer intentionally receives its original failure-free row contract.
        with patch('pebby.agent.world_data._failures', return_value=([], {'failure_samples': 0})):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.data=self.fixture.data;self.tensors=trainer.as_tensors(self.data)
        self.index=self.fixture.index()

    def sampler(self):
        return FourStepSampler(self.data,self.data,self.index,
                               fraction=.5,start=[1,0,0,0,0],end=[1,0,0,0,0])

    def test_real_sequence_sampler_path_reaches_loss_and_retains_current_inputs(self):
        sampler=self.sampler();captured=[]
        def loss(model,batch,*args,**kwargs):
            captured.append(({key:value.detach().clone() for key,value in batch.items()},
                             list(sampler.last_level_seeds)))
            return draft_model.world_losses(model,batch,*args,**kwargs)
        model=make_model(history=8)
        with patch.object(trainer,'world_losses',side_effect=loss):
            stats=trainer.run_epoch(model,self.tensors,torch.device('cpu'),{},2,
                optimizer=torch.optim.SGD(model.parameters(),lr=0),generator=torch.Generator().manual_seed(6),
                curriculum=sampler,steps_per_epoch=2,start_step=0,total_steps=2,on_policy_tensors=self.tensors)
        self.assertEqual(len(captured),2)
        self.assertEqual(stats['rollout_rows'],2)
        self.assertEqual(stats['counterfactual_rows'],2)
        self.assertEqual(stats['distinct_levels_per_batch'],2)
        for batch,seeds in captured:
            mask=batch['rollout_mask'];self.assertEqual(mask.sum(),1)
            self.assertEqual(batch['rollout_actions'][mask].tolist(),[[3,3,3,3]])
            self.assertTrue((batch['rollout_actions'][~mask]==-1).all())
            self.assertEqual(batch['distances'][mask].tolist(),[[4,3,2,1]])
            self.assertEqual(len(set(seeds)),2)
            anchor=next(int(row) for row in self.index.anchor_row
                        if torch.equal(batch['frames'][mask][0],self.tensors['frames'][row]))
            torch.testing.assert_close(batch['frames'][mask][0],self.tensors['frames'][anchor])
            torch.testing.assert_close(batch['optimal'][mask][0],self.tensors['optimal'][anchor])

    def test_metrics_use_eligible_counts_and_sum_row_counts(self):
        # Unequal eligible fractions expose the erroneous B-weighted average.
        outputs=[]
        for cf,seq,rank,error in [(2,1,1.,10.),(1,2,3.,20.)]:
            outputs.append({'total':torch.tensor(0.),'losses':{},
                'diagnostics':{'counterfactual_mean_rank':torch.tensor(rank),
                               'rollout_prediction_mse_h4':torch.tensor(error),
                               'counterfactual_rows':torch.tensor(cf),'rollout_rows':torch.tensor(seq)},
                'diagnostic_weights':{'counterfactual_mean_rank':torch.tensor(cf),
                                      'rollout_prediction_mse_h4':torch.tensor(seq)}})
        data={key:value[:6] for key,value in self.tensors.items()}
        with patch.object(trainer,'world_losses',side_effect=outputs):
            stats=trainer.run_epoch(make_model(history=8),data,torch.device('cpu'),{},3)
        self.assertAlmostEqual(stats['counterfactual_mean_rank'],5/3)
        self.assertAlmostEqual(stats['rollout_prediction_mse_h4'],50/3)
        self.assertEqual(stats['counterfactual_rows'],3)
        self.assertEqual(stats['rollout_rows'],3)
        self.assertEqual(stats['samples'],6)

    def test_legacy_evaluation_and_training_are_exact(self):
        torch.manual_seed(44);first=make_model(history=8);second=copy.deepcopy(first)
        data={key:value[:4] for key,value in self.tensors.items()}
        a=production.run_epoch(first,data,torch.device('cpu'),{},2)
        with patch.object(trainer,'world_losses',draft_model.world_losses):
            b=trainer.run_epoch(second,data,torch.device('cpu'),{},2)
        self.assertEqual({key:a[key] for key in b},b)
        torch.manual_seed(87)
        a=production.run_epoch(first,data,torch.device('cpu'),{},2,
            optimizer=torch.optim.SGD(first.parameters(),lr=.001),generator=torch.Generator().manual_seed(4))
        torch.manual_seed(87)
        with patch.object(trainer,'world_losses',draft_model.world_losses):
            b=trainer.run_epoch(second,data,torch.device('cpu'),{},2,
                optimizer=torch.optim.SGD(second.parameters(),lr=.001),generator=torch.Generator().manual_seed(4))
        self.assertEqual({key:a[key] for key in b},b)
        for key,value in first.state_dict().items():
            torch.testing.assert_close(value,second.state_dict()[key],atol=0,rtol=0)

    def test_cli_persists_source_bound_index_in_all_checkpoints_and_report(self):
        out=self.fixture.root/'result.pt'
        args=['--train',str(self.fixture.source),'--on-policy-data',str(self.fixture.source),
              '--rollout-index',str(self.fixture.root/'index.npz'),'--curriculum',
              '--on-policy-fraction','.5',
              '--curriculum-start','1','0','0','0','0','--curriculum-end','1','0','0','0','0',
              '--batch-size','2','--epochs','1','--device','cpu','--drop-last',
              '--checkpoint-out',str(out),'--seed','5']
        for key,value in {**TINY,'history':8}.items():args.extend(['--'+key.replace('_','-'),str(value)])
        with redirect_stdout(io.StringIO()),patch.object(trainer,'world_losses',draft_model.world_losses):
            self.assertEqual(trainer.main(args),0)
        reference=None
        for path in (out,out.with_name('result.running.pt'),out.with_name('result.last.pt')):
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            index=checkpoint['on_policy_source']['rollout_index']
            self.assertEqual(index['sha256'],file_digest(self.fixture.root/'index.npz'))
            self.assertEqual(index['source_sha256'],file_digest(self.fixture.source))
            self.assertEqual(index['K'],4);self.assertEqual(index['mode'],'on_policy_train')
            self.assertEqual(index['anchors'],2)
            if reference is None:reference=index
            else:self.assertEqual(index,reference)
        report=json.loads(out.with_suffix('.training.json').read_text())
        self.assertEqual(report['on_policy_source']['rollout_index'],reference)
        self.assertEqual(report['history'][0]['train']['rollout_rows'],5)
        self.assertEqual(report['history'][0]['train']['counterfactual_rows'],5)


if __name__=='__main__':unittest.main()
