"""Tiny fixed TRAIN-only capacity check; not the B1024 production protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from pebby.agent.structured_transition import StructuredTransition
from pebby.agent.structured_objective import objective
from tools.train_structured_transition import load_cache, branch_batch, training_scale, evaluate, event_counts


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('data/structured-field-pilot-2048/train'))
    parser.add_argument('--report',type=Path,default=Path('artifacts/structured-trainability-16.json'))
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--seconds',type=int,default=120)
    args=parser.parse_args()
    if not 1<=args.updates<=600 or not 1<=args.seconds<=120:parser.error('updates<=600 and seconds<=120 required')
    if args.report.exists():raise FileExistsError(args.report)
    torch.set_num_threads(1);torch.manual_seed(42)
    started=time.monotonic();print('PID',os.getpid(),flush=True)
    r=dict(status='running',pid=os.getpid(),device='cpu',threads=1,seed=42,lr=.001,weight_decay=.01,
           levels=16,transitions=64,checkpoint_saved=False,validation_used=False,official_inputs_used=False,
           scope='Fixed tiny training-set capacity diagnostic, not distinct-level B1024 production training.',curve=[])
    def persist():
        r['elapsed_seconds']=time.monotonic()-started
        tmp=args.report.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(args.report)
    def timeout(*_):raise TimeoutError('CPU trainability deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.seconds)
    model=small=scale=pos_weight=None
    completed=0
    try:
        files=[Path(__file__),Path('pebby/agent/structured_transition.py'),Path('pebby/agent/structured_objective.py'),
               Path('tools/train_structured_transition.py'),args.cache/'manifest.json',*sorted(args.cache.glob('*.npy'))]
        r['source_hashes']={str(p):digest(p) for p in files}
        data,manifest=load_cache(args.cache,'train')
        rng=np.random.default_rng(42)
        ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(data['difficulties']==d),4 if d==3 else 3,replace=False) for d in range(1,6)]))
        small={k:np.array(v[ids]) for k,v in data.items()}
        r.update(indices=ids.tolist(),seeds=small['seeds'].tolist(),difficulties=small['difficulties'].tolist(),
                 field_encoder=manifest['field_encoder'],events=event_counts(small))
        scale=torch.tensor(training_scale(data))
        prevalence=np.array([np.asarray(data[k],dtype=bool).mean() for k in ('lost_life','terminal','won')])
        pos_weight=torch.tensor(np.clip((1-prevalence)/np.maximum(prevalence,1e-8),1,20),dtype=torch.float32)
        r.update(feature_scale=scale.tolist(),event_positive_weights=pos_weight.tolist())
        del data
        batch=branch_batch(small,np.repeat(np.arange(16),4),np.tile(np.arange(4),16),'cpu')
        model=StructuredTransition();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        def measure(step):
            measured=evaluate(model,small,scale,pos_weight,'cpu',batch_size=16)
            with torch.no_grad():
                field=torch.tensor(small['fields']).float()
                predicted=torch.stack([model.predict(field,torch.full((16,),a)) for a in range(4)],1)
                actual=torch.tensor(small['next_fields']).float()
                measured['action_variance']={group:{'predicted':float(predicted[...,s:e].var(1,unbiased=False).mean()),
                                                   'actual':float(actual[...,s:e].var(1,unbiased=False).mean())}
                                            for group,s,e in (('core',0,48),('appearance',48,70),('carried',70,84))}
                predicted_player=model.readout(predicted.flatten(0,1))['player_logits'].argmax(-1).view(16,4)
                measured['predicted_distinct_player_positions_mean']=float(np.mean([len(set(x)) for x in predicted_player.tolist()]))
                actual_player=small['next_player_cell'][...,1]*12+small['next_player_cell'][...,0]
                measured['actual_distinct_player_positions_mean']=float(np.mean([len(set(x)) for x in actual_player.tolist()]))
            r['curve'].append(dict(step=step,elapsed_seconds=time.monotonic()-started,evaluation=measured));persist()
            print(json.dumps({'step':step,'seconds':r['elapsed_seconds'],'player':measured['metrics']['predicted_player_accuracy'],
                              'moved':measured['metrics']['predicted_moved_player_accuracy'],
                              'wrong_action':measured['metrics']['permuted_action_moved_player_accuracy']}),flush=True)
        measure(0)
        for step in range(1,args.updates+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            result=objective(model,*batch,scale,pos_weight=pos_weight)
            if not torch.isfinite(result['total']):raise ValueError('nonfinite loss')
            result['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
            completed=step
            if step in (100,300,600) or step==args.updates:measure(step)
        r['status']='complete'
    except TimeoutError as error:
        r.update(status='bounded_timeout',error=str(error))
        signal.alarm(0)
        if model is not None and (not r['curve'] or r['curve'][-1]['step']!=completed):measure(completed)
    except BaseException as error:
        r.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.alarm(0)
        r['completed_updates']=completed
        r['source_unchanged']=all(digest(p)==h for p,h in r.get('source_hashes',{}).items())
        if not r['source_unchanged']:r['status']='failed_source_changed'
        persist()
    return 0 if r['status'] in ('complete','bounded_timeout') else 1

if __name__=='__main__':raise SystemExit(main())
