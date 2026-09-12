"""Strict fixed-monitor gameplay after the matched200 on-policy continuation.

Two arms, identical generated levels/contexts and public causal H8 inference.
No training, Oracle action selection, retries, or action masking.
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.evaluate import rollout
from pebby.agent.model import load_checkpoint
from pebby.agent.structured_factored_policy import state_digest
from pebby.ls20.env import Ls20Scenario
from tools.evaluate_structured_workspace_gameplay import (
    ACTOR, BANK, BANK_SHA, FORMAT, DecisionTimer, checked_bank, paired, summarize, verify_sources,
)
from tools.train_structured_transition import digest, atomic_json


def is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_fit(report):
    if (report.get('status')!='complete' or report.get('source')!='generated_only' or report.get('smoke') is not False
            or report.get('official_inputs_used') is not False or report.get('sources_unchanged') is not True
            or report.get('paired_selections_exact') is not True or report.get('initializations_exact') is not True
            or report.get('primary_depth')!=2 or report.get('trained_depths')!=[1,2,4]
            or report.get('replacements')!=512 or report.get('trajectory_levels',0)<1024):
        raise ValueError('completed matched generated production fit required')
    args=report.get('args',{})
    if args.get('updates')!=200 or args.get('batch_size')!=1024:
        raise ValueError('fixed200 updates/B1024 required')
    warm=report.get('warmstart',{})
    if not isinstance(warm.get('path'),str) or not is_sha(warm.get('sha256')) or not is_sha(warm.get('state_sha256')):
        raise ValueError('exact evolving initializer binding required')
    selections=[];depths=[]
    for name in ('replay','onpolicy'):
        arm=report.get('arms',{}).get(name,{})
        draw=arm.get('depth_draws',{})
        if (arm.get('status')!='complete' or arm.get('completed_updates')!=200
                or arm.get('parameters')!=201315 or arm.get('active_parameters')!=201315
                or arm.get('initial_state_sha256')!=warm['state_sha256']
                or set(draw)!={'1','2','4'} or any(type(v) is not int or v<=0 for v in draw.values())
                or sum(draw.values())!=200 or not is_sha(arm.get('selection_sha256'))
                or not is_sha(arm.get('used_state_rows_sha256'))
                or arm.get('checkpoint',{}).get('strict_reload_exact') is not True):
            raise ValueError(f'invalid completed {name} arm')
        selections.append(arm['selection_sha256']);depths.append(draw)
    if selections[0]!=selections[1] or depths[0]!=depths[1]:
        raise ValueError('paired candidate/view/depth sampling mismatch')
    if report['arms']['replay']['used_state_rows_sha256']==report['arms']['onpolicy']['used_state_rows_sha256']:
        raise ValueError('both arms claim identical used-state selection')


def checked_fit(path):
    path=Path(path);raw=path.read_bytes();report=json.loads(raw);validate_fit(report)
    sources=dict(report['sources']);verify_sources(sources)
    if ACTOR not in sources:raise ValueError('base actor source missing')
    warm=report['warmstart'];warm_path=Path(warm['path'])
    warm_bytes=warm_path.read_bytes()
    if (hashlib.sha256(warm_bytes).hexdigest()!=warm['sha256']
            or sources.get(str(warm_path))!=warm['sha256']):raise ValueError('warmstart hash/source mismatch')
    warm_saved=torch.load(io.BytesIO(warm_bytes),map_location='cpu',weights_only=True)
    if (warm_saved.get('format')!=FORMAT or state_digest(warm_saved['weights'])!=warm['state_sha256']
            or warm_saved.get('state_sha256')!=warm['state_sha256']
            or warm_saved.get('config',{}).get('memory_mode')!='evolving'):
        raise ValueError('warmstart field architecture/state mismatch')
    checkpoints={}
    for name in ('replay','onpolicy'):
        arm=report['arms'][name];item=arm['checkpoint'];checkpoint=Path(item['path']);payload=checkpoint.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('final checkpoint hash mismatch')
        saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True);p=saved.get('training_provenance',{})
        if (saved.get('format')!=FORMAT or saved.get('config',{}).get('memory_mode')!='evolving'
                or saved.get('parameters')!=201315 or saved.get('active_trainable_parameters')!=201315
                or state_digest(saved['weights'])!=saved.get('state_sha256')
                or saved.get('actor_checkpoint')!=ACTOR or saved.get('actor_sha256')!=sources[ACTOR]
                or saved.get('sources')!=report['sources'] or saved.get('source_unchanged') is not True
                or saved.get('official_inputs_used') is not False or p.get('arm')!=name
                or p.get('experiment')!='matched_onpolicy_state_distribution'
                or p.get('updates')!=200 or p.get('final_update')!=200 or p.get('batch_size')!=1024
                or p.get('replacements')!=512 or p.get('primary_depth')!=2 or p.get('depths')!=[1,2,4]
                or p.get('depth_draws')!=arm['depth_draws'] or p.get('evolving_warmstart')!=warm
                or p.get('selection_sha256')!=arm['selection_sha256']
                or p.get('used_state_rows_sha256')!=arm['used_state_rows_sha256']
                or p.get('smoke') is not False or p.get('fixed_final') is not True
                or p.get('source')!='generated_only' or p.get('official_inputs_used') is not False
                or p.get('encoder_and_dynamics_frozen') is not True):
            raise ValueError(f'{name} checkpoint/fit provenance mismatch')
        cache_manifest=str(Path(p['trajectory_cache'])/'manifest.json')
        if (p.get('trajectory_cache')!=report['args'].get('trajectory_cache')
                or sources.get(cache_manifest)!=p.get('trajectory_manifest_sha256')):
            raise ValueError('trajectory cache source mismatch')
        checkpoints[name]={'path':str(checkpoint),'sha256':item['sha256']}
        sources[str(checkpoint)]=item['sha256']
    sha=hashlib.sha256(raw).hexdigest()
    if digest(path)!=sha:raise ValueError('fit report changed during inspection')
    sources[str(path)]=sha
    return report,checkpoints,sources


def checked_baseline(path, warmstart, specs, sources):
    path=Path(path);raw=path.read_bytes();record=json.loads(raw)
    arm=record.get('arms',{}).get('evolving',{})
    checkpoint=arm.get('checkpoint',{})
    if (record.get('status')!='complete' or record.get('sources_unchanged') is not True
            or record.get('bank_sha256')!=BANK_SHA or record.get('max_actions')!=200
            or record.get('protocol')!='strict' or record.get('on_stall')!='repeat'
            or record.get('temperature')!=0 or record.get('primary_depth')!=2
            or record.get('device')!='cuda' or record.get('precision')!='FP32; TF32 off'
            or record.get('official_inputs_used') is not False or record.get('training_performed') is not False
            or arm.get('status')!='complete' or checkpoint.get('sha256')!=warmstart['sha256']
            or Path(checkpoint.get('path','')).resolve()!=Path(warmstart['path']).resolve()):
        raise ValueError('warmstart monitor is not the identical verified protocol/checkpoint')
    runs=arm.get('runs',[])
    if len(runs)!=100 or len(specs)!=100:raise ValueError('complete100-level baseline required')
    for run,spec in zip(runs,specs):
        if (run.get('seed')!=spec['seed'] or run.get('context')!=spec['training_context_index']
                or run.get('optimal')!=spec['context_optimal_actions'] or run.get('on_stall')!='repeat'
                or run.get('temperature')!=0 or not 0<run.get('actions',0)<=200):
            raise ValueError('baseline monitor order/context/protocol mismatch')
    verify_sources(record['sources'])
    for source,sha in record['sources'].items():
        if source in sources and sources[source]!=sha:raise ValueError('baseline source binding conflict')
        sources[source]=sha
    sha=hashlib.sha256(raw).hexdigest()
    if digest(path)!=sha:raise ValueError('baseline report changed while checking')
    sources[str(path)]=sha
    return {'reused':True,'path':str(path),'sha256':sha,'checkpoint':checkpoint,'runs':runs,'summary':summarize(runs),
        'limits':'Same fixed monitor, reused prior GPU FP32 rollout; not an independent final test.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-report',type=Path,default=Path('artifacts/structured-onpolicy-comparison-200.json'))
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--baseline-report',type=Path,default=Path('artifacts/structured-workspace-gameplay100.json'))
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--seconds',type=int,default=1200)
    args=parser.parse_args()
    if args.report.exists() or not 1<=args.seconds<=1200:parser.error('new report/deadline1..1200 required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();print('PID',os.getpid(),flush=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('paired gameplay deadline')))
    signal.alarm(args.seconds)
    report={'status':'validating','pid':os.getpid(),'source':'generated_only','split':'validation','official_inputs_used':False,
        'training_performed':False,'device':args.device,'precision':'FP32; TF32 off','max_actions':200,
        'protocol':'strict','on_stall':'repeat','temperature':0.,'primary_depth':2,'bank':str(BANK),'bank_sha256':BANK_SHA,'arms':{},
        'limits':['Fixed generated monitor100 only; no official completion or training.',
                  'Both arms use the same evolving architecture and warmstart; training state distribution differs.',
                  'Model forward timing excludes H8 tensor assembly.', 'Partial interrupted levels are excluded from completed-run aggregates.',
                  'Reuses the same generated monitor and prior warmstart rollout; this is not an independent final test.']}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started;atomic_json(args.report,report)
    policy=timer=None;persist()
    try:
        fit,checkpoints,sources=checked_fit(args.fit_report)
        levels,optima,specs=checked_bank()
        for path in (str(BANK),__file__,'tools/evaluate_structured_workspace_gameplay.py','pebby/agent/evaluate.py',
                     'pebby/agent/history.py','pebby/agent/model.py','pebby/agent/structured_workspace_controller.py',
                     'pebby/ls20/env.py','pebby/ls20/generate.py','pebby/ls20/bank.py','pebby/ls20/names.py','third_party/ls20/ls20.py'):
            sha=digest(path)
            if path in sources and sources[path]!=sha:raise ValueError('evaluation source changed')
            sources[str(path)]=sha
        train_seed_path='data/structured-field-16384/train/seeds.npy'
        if not any(Path(path).resolve()==Path(train_seed_path).resolve() for path in sources):raise ValueError('training seed guard missing')
        train=np.load(train_seed_path,allow_pickle=False)
        if set(map(int,train)) & {s['seed'] for s in specs}:raise ValueError('monitor/TRAIN overlap')
        report['reused_warmstart']=checked_baseline(args.baseline_report,fit['warmstart'],specs,sources)
        report['sources']=sources;report['fit_report']=str(args.fit_report);report['warmstart']=fit['warmstart']
        for arm,checkpoint in checkpoints.items():
            report.update(status='running',active_arm=arm);persist()
            if digest(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('checkpoint changed before reconstruction')
            policy,_=load_checkpoint(checkpoint['path'],args.device)
            policy.float().eval().requires_grad_(False)
            if digest(checkpoint['path'])!=checkpoint['sha256'] or policy.config().get('loops')!=2:
                raise ValueError('checkpoint changed or wrong primary depth')
            entry={'status':'running','checkpoint':checkpoint,'parameters':policy.parameter_count(),'runs':[]}
            report['arms'][arm]=entry;timer=DecisionTimer(policy,args.device)
            for index,(level,optimum,spec) in enumerate(zip(levels,optima,specs)):
                first=len(timer.durations);entry['incomplete_level']={'index':index,'seed':spec['seed']}
                result=rollout(policy,Ls20Scenario(level,spec['training_context_index']),200,args.device,optimum,'repeat',temperature=0.)
                durations=timer.durations[first:]
                if len(durations)!=result['actions']:raise ValueError('decision/action census mismatch')
                result.update(seed=spec['seed'],difficulty=spec['difficulty'],context=spec['training_context_index'],
                    losses=3-result['lives_left'],decision_count=len(durations),decision_seconds=sum(durations),
                    decision_median_seconds=float(np.median(durations)) if durations else None)
                entry['runs'].append(result);entry.pop('incomplete_level',None);entry['summary']=summarize(entry['runs']);persist()
                if (index+1)%10==0:print(json.dumps({'arm':arm,**entry['summary']['all']}),flush=True)
            timer.close();timer=None;entry['status']='complete'
            entry['decision_seconds']=sum(r['decision_seconds'] for r in entry['runs'])
            verify_sources(sources);persist();del policy;policy=None;gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
        report['paired_onpolicy_vs_replay']=paired(report['arms']['onpolicy']['runs'],report['arms']['replay']['runs'])
        report['paired_vs_reused_warmstart']={arm:paired(entry['runs'],report['reused_warmstart']['runs'])
            for arm,entry in report['arms'].items()}
        verify_sources(sources);report.update(status='complete',sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed_partial',error=str(error));raise
    finally:
        if timer is not None:timer.close()
        policy=None;gc.collect()
        if args.device=='cuda' and torch.cuda.is_available():torch.cuda.empty_cache()
        signal.alarm(0);persist()


if __name__=='__main__':main()
