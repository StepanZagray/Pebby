"""Strict generated100 gameplay of C/T repaired dynamics with one fixed head.

B0 is reused only from an explicitly hash-bound identical-protocol report.
This dedicated composition is not a generic model.py or UI checkpoint format.
"""
import argparse
import gc
import json
import io
import hashlib
import os
from pathlib import Path
import signal
import time

import torch

from pebby.agent.evaluate import rollout
from pebby.agent.structured_factored_policy import state_digest
from pebby.agent.world_data import history_arrays
from pebby.ls20.env import Ls20Scenario
from tools.compose_structured_workspace_dynamics import load_composition,public_wiring_check,WARM_SHA,INITIAL_SHA,digest,normalized_sources
from tools.evaluate_workspace_reserved_zero import checked_baselines
from tools.evaluate_structured_workspace_gameplay import checked_bank,summarize,paired,verify_sources,BANK,BANK_SHA
from tools.train_structured_transition import atomic_json


def merge_sources(target, incoming):
    """Canonical merge rejects conflicting path aliases rather than overwriting."""
    merged=normalized_sources(target) if target else {}
    for path,sha in normalized_sources(incoming).items():
        if path in merged and merged[path]!=sha:raise ValueError('conflicting source fingerprints')
        merged[path]=sha
    target.clear();target.update(merged)


def validate_fit_report(report):
    if (report.get('status')!='complete' or report.get('source')!='generated_only'
            or report.get('official_inputs_used') is not False
            or report.get('source_unchanged') is not True or report.get('paired_initializations_exact') is not True
            or report.get('paired_schedule_exact') is not True):raise ValueError('complete paired production repair required')
    args=report.get('args',{})
    if (args.get('smoke') is not False or args.get('preflight_only') is not False
            or args.get('device')!='cuda' or args.get('batch_size')!=1024 or args.get('updates')!=200):
        raise ValueError('complete paired production repair requires CUDA B1024/200')
    if (report.get('replacements')!=512 or report.get('trajectory_levels',0)<1024
            or report.get('validation_levels')!=512):raise ValueError('production source populations mismatch')
    if (report.get('initialization',{}).get('sha256')!=INITIAL_SHA
            or report.get('workspace_readout',{}).get('sha256')!=WARM_SHA):raise ValueError('paired repair parent mismatch')
    hashes=[report.get('source_schedule_sha256'), report['initialization'].get('state_sha256'),
            report['workspace_readout'].get('loaded_state_sha256'),report.get('frozen_encoder_state_sha256')]
    used=[]
    for name,replacements in (('control',0),('onpolicy',102400)):
        arm=report.get('arms',{}).get(name,{})
        training=arm.get('training',[])
        if (arm.get('status')!='complete' or arm.get('parameters')!=294664
                or arm.get('trainable_parameters')!=294664
                or arm.get('initial_state_sha256')!=report['initialization'].get('state_sha256')
                or arm.get('schedule_sha256')!=report.get('source_schedule_sha256')
                or arm.get('replacement_rows')!=replacements
                or [item.get('step') for item in training]!=list(range(1,201))
                or any(item.get('replaced')!=(512 if name=='onpolicy' else 0) for item in training)):
            raise ValueError('paired arm initialization/schedule/update witness mismatch')
        used.append(arm.get('used_state_rows_sha256'))
    hashes+=used
    if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
        raise ValueError('invalid paired state/schedule SHA witness')
    if used[0]==used[1]:raise ValueError('control and onpolicy actual state rows must differ')


def checked_fit(path,expected_sha):
    path=Path(path);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('explicit fit report SHA mismatch')
    report=json.loads(raw);validate_fit_report(report)
    sources=normalized_sources(report.get('sources'))
    arms={}
    for name in ('control','onpolicy'):
        item=report['arms'][name];checkpoint=item.get('checkpoint',{})
        if checkpoint.get('strict_reload_exact') is not True:raise ValueError('exact repair reload required')
        raw_checkpoint=Path(checkpoint['path']).read_bytes()
        if hashlib.sha256(raw_checkpoint).hexdigest()!=checkpoint['sha256']:raise ValueError('repair checkpoint changed')
        saved=torch.load(io.BytesIO(raw_checkpoint),map_location='cpu',weights_only=True)
        expected={'arm':name,'updates':200,'batch_size':1024,'smoke':False,
            'selection_sha256':report['source_schedule_sha256'],
            'used_state_rows_sha256':item['used_state_rows_sha256'],
            'frozen_workspace_state_sha256':report['workspace_readout']['loaded_state_sha256'],
            'frozen_encoder_state_sha256':report['frozen_encoder_state_sha256']}
        if any(saved.get(k)!=v for k,v in expected.items()) or normalized_sources(saved.get('sources'))!=sources:
            raise ValueError('saved repair and completed fit provenance differ')
        arms[name]=checkpoint
    if digest(path)!=expected_sha:raise ValueError('fit report changed during load')
    return report,arms


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-report',type=Path,required=True);p.add_argument('--fit-sha256',required=True)
    p.add_argument('--baseline-report',type=Path,default=Path('artifacts/structured-onpolicy-gameplay100.json'))
    p.add_argument('--baseline-sha256',required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=600);args=p.parse_args(argv)
    if args.report.exists() or not 1<=args.seconds<=600:p.error('new report and bounded seconds1..600 required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();print('PID',os.getpid(),flush=True)
    def expired(*_):raise TimeoutError('composed gameplay deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    report={'status':'validating','pid':os.getpid(),'source':'generated_only','split':'validation',
        'official_inputs_used':False,'training_performed':False,'device':'cuda','precision':'FP32; TF32 off',
        'max_actions':200,'protocol':'strict','on_stall':'repeat','temperature':0.,'primary_depth':2,
        'bank':str(BANK),'bank_sha256':BANK_SHA,'arms':{},'dedicated_composition_only':True,
        'limits':['Reused monitor is diagnostic, not an untouched final test.','H1 dynamics repair is not an H4 training experiment.',
                  'Only predicted successors reach the fixed action head; actual future labels never enter inference.']}
    def persist():report.update(elapsed_seconds=time.monotonic()-started);atomic_json(args.report,report)
    policy=None;persist()
    try:
        fit,checkpoints=checked_fit(args.fit_report,args.fit_sha256)
        if digest(args.baseline_report)!=args.baseline_sha256:raise ValueError('explicit B0 report SHA mismatch')
        levels,optima,specs=checked_bank();baselines,sources=checked_baselines(args.baseline_report,specs)
        baseline=baselines['warmstart']
        if baseline['checkpoint']['sha256']!=WARM_SHA or summarize(baseline['runs'])['all']['completed']!=36:
            raise ValueError('exact warm B0 baseline required')
        merge_sources(sources,{str(args.fit_report):args.fit_sha256,str(Path(__file__).resolve()):digest(__file__)})
        for path in ('tools/evaluate_workspace_reserved_zero.py','tools/evaluate_structured_workspace_gameplay.py',
                     'pebby/agent/evaluate.py','pebby/agent/history.py','pebby/agent/world_data.py'):
            merge_sources(sources,{str(Path(path).resolve()):digest(path)})
        report.update(B0={'checkpoint':baseline['checkpoint'],'summary':summarize(baseline['runs']),
                           'runs':baseline['runs'],'reused_report_sha256':args.baseline_sha256},sources=sources)
        for arm,checkpoint in checkpoints.items():
            policy,composition=load_composition(checkpoint['path'],checkpoint['sha256'],'cuda')
            if (composition['replacement']['arm']!=arm or composition['replacement']['updates']!=fit['args']['updates']
                    or composition['replacement']['smoke'] is not False
                    or composition['replacement']['selection_sha256']!=fit['source_schedule_sha256']):
                raise ValueError('checkpoint and paired fit schedule/arm differ')
            merge_sources(sources,composition['runtime_source_hashes'])
            entry={'status':'running','composition':composition,'runs':[]};report['arms'][arm]=entry
            frame=Ls20Scenario(levels[0],specs[0]['training_context_index']).reset()
            f,v,a=history_arrays([frame],[-1],8)
            entry['wiring_check']=public_wiring_check(policy,torch.as_tensor(f,device='cuda')[None],
                torch.as_tensor(v,device='cuda')[None],torch.as_tensor(a,device='cuda')[None])
            before=state_digest(policy.state_dict())
            for index,(level,optimum,spec) in enumerate(zip(levels,optima,specs)):
                entry['incomplete_level']=spec['seed']
                run=rollout(policy,Ls20Scenario(level,spec['training_context_index']),200,'cuda',optimum,'repeat',temperature=0.)
                run.update(seed=spec['seed'],difficulty=spec['difficulty'],context=spec['training_context_index'],losses=3-run['lives_left'])
                entry['runs'].append(run);entry.pop('incomplete_level');entry['summary']=summarize(entry['runs'])
                report['status']='running';persist()
                if (index+1)%10==0:print(json.dumps({'arm':arm,**entry['summary']['all']}),flush=True)
            if state_digest(policy.state_dict())!=before:raise ValueError('composed frozen weights changed')
            entry.update(status='complete',weights_unchanged=True,paired_vs_B0=paired(entry['runs'],baseline['runs']))
            verify_sources(sources);persist();policy=None;gc.collect();torch.cuda.empty_cache()
        report['paired_T_vs_C']=paired(report['arms']['onpolicy']['runs'],report['arms']['control']['runs'])
        verify_sources(sources);report.update(status='complete',sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed_partial',error=str(error));raise
    finally:
        policy=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        signal.alarm(0);persist()
    return 0


if __name__=='__main__':main()
