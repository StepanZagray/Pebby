import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from pebby.agent.structured_factored_policy import state_digest
from tools.evaluate_structured_onpolicy_gameplay import validate_fit, checked_fit, checked_baseline, BANK_SHA, FORMAT, digest


def fit_template():
    report={'status':'complete','source':'generated_only','smoke':False,'official_inputs_used':False,
        'sources_unchanged':True,'paired_selections_exact':True,'initializations_exact':True,'primary_depth':2,
        'trained_depths':[1,2,4],'replacements':512,'trajectory_levels':1024,
        'args':{'updates':200,'batch_size':1024},'warmstart':{'path':'warm.pt','sha256':'a'*64,'state_sha256':'b'*64},'arms':{}}
    for name,used in [('replay','c'),('onpolicy','d')]:
        report['arms'][name]={'status':'complete','completed_updates':200,'parameters':201315,'active_parameters':201315,
            'initial_state_sha256':'b'*64,'depth_draws':{'1':67,'2':67,'4':66},'selection_sha256':'e'*64,
            'used_state_rows_sha256':used*64,'checkpoint':{'strict_reload_exact':True}}
    return report


class OnPolicyGameplayTests(unittest.TestCase):
    def test_only_matched_complete_production_fit(self):
        validate_fit(fit_template())
        for key,value in [('status','running'),('smoke',True),('replacements',256),('initializations_exact',False),('trajectory_levels',64)]:
            report=fit_template();report[key]=value
            with self.assertRaises(ValueError):validate_fit(report)
        for key,value in [('selection_sha256','f'*64),('initial_state_sha256','f'*64),('depth_draws',{'1':0,'2':100,'4':100})]:
            report=fit_template();report['arms']['onpolicy'][key]=value
            with self.assertRaises(ValueError):validate_fit(report)
        report=fit_template();report['args']['updates']=2
        with self.assertRaises(ValueError):validate_fit(report)

    def files(self,root):
        actor=root/'actor.pt';actor.write_bytes(b'fixture actor')
        warm=root/'warm.pt';weights={'fixture':torch.ones(1)};state=state_digest(weights)
        torch.save({'format':FORMAT,'weights':weights,'state_sha256':state,'config':{'memory_mode':'evolving'}},warm)
        cache=root/'cache';cache.mkdir();(cache/'manifest.json').write_text('{}')
        report=fit_template();report['warmstart']={'path':str(warm),'sha256':digest(warm),'state_sha256':state}
        report['sources']={str(actor):digest(actor),str(warm):digest(warm),str(cache/'manifest.json'):digest(cache/'manifest.json')}
        report['args']['trajectory_cache']=str(cache)
        for name in ('replay','onpolicy'):
            arm=report['arms'][name];arm['initial_state_sha256']=state
            provenance={'arm':name,'experiment':'matched_onpolicy_state_distribution','updates':200,'final_update':200,'batch_size':1024,
                'replacements':512,'primary_depth':2,'depths':[1,2,4],'depth_draws':arm['depth_draws'],
                'evolving_warmstart':report['warmstart'],'selection_sha256':arm['selection_sha256'],
                'used_state_rows_sha256':arm['used_state_rows_sha256'],'smoke':False,'fixed_final':True,'source':'generated_only',
                'official_inputs_used':False,'encoder_and_dynamics_frozen':True,'trajectory_cache':str(cache),
                'trajectory_manifest_sha256':digest(cache/'manifest.json')}
            saved={'format':FORMAT,'config':{'memory_mode':'evolving'},'parameters':201315,'active_trainable_parameters':201315,
                'weights':weights,'state_sha256':state,'actor_checkpoint':str(actor),'actor_sha256':digest(actor),
                'sources':report['sources'],'source_unchanged':True,'official_inputs_used':False,'training_provenance':provenance}
            target=root/(name+'.pt');torch.save(saved,target)
            arm['checkpoint']={'path':str(target),'sha256':digest(target),'strict_reload_exact':True}
        path=root/'fit.json';path.write_text(json.dumps(report));return actor,path,report

    def test_serialized_fit_checkpoint_and_source_guards(self):
        # Metadata fixture only; actual architecture reconstruction is generic loader's job.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actor,path,report=self.files(root)
            with patch('tools.evaluate_structured_onpolicy_gameplay.ACTOR',str(actor)):
                _,checkpoints,_=checked_fit(path);self.assertEqual(set(checkpoints),{'replay','onpolicy'})
                target=Path(checkpoints['onpolicy']['path']);saved=torch.load(target,weights_only=True)
                saved['training_provenance']['replacements']=256;torch.save(saved,target)
                with self.assertRaisesRegex(ValueError,'checkpoint hash'):checked_fit(path)
                report['arms']['onpolicy']['checkpoint']['sha256']=digest(target);path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'provenance mismatch'):checked_fit(path)
                actor.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError,'source changed'):checked_fit(path)

    def test_baseline_reuse_requires_exact_protocol_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_text('stable')
            checkpoint={'path':str(root/'warm.pt'),'sha256':'a'*64}
            specs=[{'seed':1_000_000+i,'training_context_index':i%7,'context_optimal_actions':20} for i in range(100)]
            runs=[{'seed':s['seed'],'context':s['training_context_index'],'optimal':20,'on_stall':'repeat','temperature':0.,
                   'actions':30,'completed':i<36,'goals_cleared':int(i<36),'goals_total':1,'losses':0,'stalls':0,
                   'ending':'win' if i<36 else 'capped'} for i,s in enumerate(specs)]
            baseline={'status':'complete','sources_unchanged':True,'bank_sha256':BANK_SHA,'max_actions':200,
                'protocol':'strict','on_stall':'repeat','temperature':0.,'primary_depth':2,'device':'cuda','precision':'FP32; TF32 off',
                'official_inputs_used':False,'training_performed':False,'sources':{str(source):digest(source)},
                'arms':{'evolving':{'status':'complete','checkpoint':checkpoint,'runs':runs}}}
            path=root/'baseline.json';path.write_text(json.dumps(baseline))
            result=checked_baseline(path,checkpoint,specs,{})
            self.assertTrue(result['reused']);self.assertEqual(result['summary']['all']['completed'],36)
            for key,value in [('device','cpu'),('max_actions',201),('primary_depth',4)]:
                changed=copy.deepcopy(baseline);changed[key]=value;path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):checked_baseline(path,checkpoint,specs,{})
            changed=copy.deepcopy(baseline);changed['arms']['evolving']['runs'][0]['seed']+=1;path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'order/context'):checked_baseline(path,checkpoint,specs,{})


if __name__=='__main__':unittest.main()
