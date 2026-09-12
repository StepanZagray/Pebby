"""Closing trainer/preflight draft integration; one generated CPU optimizer update."""
import copy,importlib.util,io,json,sys,unittest
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from pebby.agent import world_train as production,world_data as wd
from pebby.agent.world_closing_sequences import ClosingSampler
from pebby.agent.world_sequences import FourStepSampler
from pebby.agent.on_policy_provenance import file_digest
from tests import test_world_closing_sequences as fixtures
from tests.test_world_closing_rollout_draft import draft as draft_model
from tests.test_world_model import TINY,make_model


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,Path(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
trainer=load('pebby.agent._closing_trainer_draft','artifacts/world_train_closing_draft.py')
preflight=load('closing_preflight_draft','artifacts/world_continuation_preflight_closing_draft.py')

class ClosingTrainerDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.fixture=fixtures.ClosingSequenceTests()
        # Preserve the archived trainer's original failure-free fixture contract.
        with patch('pebby.agent.world_data._failures', return_value=([], {'failure_samples': 0})):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.index,self.attestation,self.sha=self.fixture.index()
        self.base=self.fixture.root/'base.npz'
        data={k:v[[4,9]] for k,v in self.fixture.data.items() if k!='meta'}
        meta=self.fixture.data['meta'];data['meta']={k:copy.deepcopy(meta[k]) for k in ('format','source','oracle_search','history','levels')}
        for level in data['meta']['levels']:level.update(samples=1,on_policy_samples=0,expert_samples=1)
        wd.save(self.base,data)

    def args(self):
        out=self.fixture.root/'result.pt'
        args=['--train',str(self.base),'--on-policy-data',str(self.fixture.source),
              '--closing-rollout-index',str(self.fixture.root/'closing.npz'),
              '--closing-action-attestation',str(self.attestation),'--closing-action-sha256',self.sha,
              '--curriculum','--on-policy-fraction','.5','--curriculum-start','1','0','0','0','0',
              '--curriculum-end','1','0','0','0','0','--batch-size','2','--epochs','1','--device','cpu',
              '--drop-last','--checkpoint-out',str(out),'--seed','5']
        for key,value in {**TINY,'history':8}.items():args.extend(['--'+key.replace('_','-'),str(value)])
        return args,out

    def test_cli_one_cpu_update_and_all_checkpoint_provenance(self):
        args,out=self.args();captured=[]
        def loss(model,batch,*a,**kw):
            captured.append({k:v.detach().clone() for k,v in batch.items()});return draft_model.world_losses(model,batch,*a,**kw)
        with redirect_stdout(io.StringIO()),patch.object(trainer,'world_losses',side_effect=loss):self.assertEqual(trainer.main(args),0)
        self.assertEqual(len(captured),1);batch=captured[0];mask=batch['rollout_mask']
        self.assertEqual(int(mask.sum()),1);self.assertTrue(batch['won'][mask,3].all());self.assertEqual(int(batch['next_optimal'][mask,3]),0)
        reference=None
        for path in (out,out.with_name('result.running.pt'),out.with_name('result.last.pt')):
            ckpt=torch.load(path,map_location='cpu',weights_only=True);index=ckpt['on_policy_source']['rollout_index']
            self.assertEqual(index['mode'],'closing_only_train');self.assertEqual(index['source_sha256'],file_digest(self.fixture.source))
            self.assertEqual(index['sha256'],file_digest(self.fixture.root/'closing.npz'));self.assertEqual(index['attestation_sha256'],self.sha)
            self.assertEqual(Path(index['attestation_path']).resolve(),self.attestation.resolve())
            if reference is None:reference=index
            else:self.assertEqual(index,reference)
        report=json.loads(out.with_suffix('.training.json').read_text());self.assertEqual(report['on_policy_source']['rollout_index'],reference)
        self.assertEqual(report['history'][0]['train']['rollout_rows'],1)
        self.assertEqual(ckpt['optimizer_steps'],1)
        self.assertEqual(preflight.closing_batch_counts(batch,1)['final_won'],1)

    def test_unpaired_mutually_exclusive_and_bad_attestation_fail_before_fit(self):
        args,_=self.args()
        bad=[]
        for flag in ('--closing-rollout-index','--closing-action-attestation','--closing-action-sha256','--on-policy-data'):
            changed=args.copy();pos=changed.index(flag);del changed[pos:pos+2];bad.append(changed)
        bad.append(args+['--rollout-index','other.npz'])
        changed=args.copy();changed[changed.index('--closing-action-sha256')+1]='0'*64;bad.append(changed)
        for argv in bad:
            with self.subTest(argv=argv),redirect_stderr(io.StringIO()),patch.object(trainer,'run_epoch') as run:
                with self.assertRaises(SystemExit):trainer.main(argv)
                run.assert_not_called()
        for flags in (['--closing-rollout-index','x'],['--closing-rollout-index','x','--closing-action-attestation','y','--closing-action-sha256','z','--rollout-index','old']):
            argv=['preflight','--source','x','--train','x','--config','x','--out','x']+flags
            with patch.object(sys,'argv',argv),redirect_stderr(io.StringIO()),patch.object(torch.cuda,'is_available') as cuda:
                with self.assertRaises(SystemExit):preflight.main()
                cuda.assert_not_called()

    def test_preflight_counts_final_win_and_unsafe_and_rejects_interior(self):
        b={'rollout_mask':torch.tensor([True,True,False]),'won':torch.zeros(3,4,dtype=torch.bool),'terminal':torch.zeros(3,4,dtype=torch.bool),'lost_life':torch.zeros(3,4,dtype=torch.bool),'distances':torch.ones(3,4,dtype=torch.long)}
        b['won'][0,3]=b['terminal'][0,3]=True;b['distances'][1,3]=-1
        stats=preflight.closing_batch_counts(b,2);self.assertEqual(stats['final_won'],1);self.assertEqual(stats['final_unsafe'],1)
        b['lost_life'][0,0]=True
        with self.assertRaisesRegex(ValueError,'interior'):preflight.closing_batch_counts(b,2)
        with self.assertRaisesRegex(ValueError,'active'):preflight.closing_batch_counts(b,1)

    def test_existing_live_sequence_and_ordinary_paths_exact_parity(self):
        raw=self.fixture.fixture;index=raw.index();data=trainer.as_tensors(raw.data)
        for seq in (False,True):
            if seq:
                sampler=FourStepSampler(raw.data,raw.data,index,fraction=.5,start=[1,0,0,0,0],end=[1,0,0,0,0])
                a=next(trainer.curriculum_batches(data,sampler,2,torch.Generator().manual_seed(8),1,0,1,data))
                b=next(production.curriculum_batches(data,sampler,2,torch.Generator().manual_seed(8),1,0,1,data))
                for name in a:torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)
            else:
                model=make_model(history=8);other=copy.deepcopy(model)
                a=trainer.run_epoch(model,data,torch.device('cpu'),{},2)
                b=production.run_epoch(other,data,torch.device('cpu'),{},2)
                # Current training adds failure-coverage diagnostics; every archived metric remains exact.
                self.assertEqual(a,{key:b[key] for key in a})

if __name__=='__main__':unittest.main()
