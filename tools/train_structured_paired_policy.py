"""Fixed-schedule policy training on two paired H1 TRAIN states per level.

Level sampling remains without replacement. A separate RNG chooses exactly
half the sampled levels from each view; the loss is computed once over full B.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from tools.train_structured_policy import (build_training_policy,check_policy_encoder,
    load_policy_cache,new_head,evaluate,policy_terms)
from tools.train_structured_transition import sample_rows,digest,atomic_json
from tools.structured_policy_batch import outputs_for_prepared_batch


def validate_pair(original, additional, first, second, original_path):
    binding=second.get('paired_source',{})
    expected=Path(original_path)/'manifest.json'
    if (Path(binding.get('manifest_path','')).resolve()!=expected.resolve()
            or binding.get('manifest_sha256')!=digest(expected)
            or binding.get('seeds_sha256')!=first['arrays']['seeds']['sha256']
            or binding.get('source_rows_sha256')!=first['arrays']['source_rows']['sha256']):
        raise ValueError('additional cache original manifest/order binding mismatch')
    if (first['source_sha256']!=second['source_sha256']
            or Path(first['source_path']).resolve()!=Path(second['source_path']).resolve()
            or first['field_encoder']!=second['field_encoder']):
        raise ValueError('paired source or encoder mismatch')
    for key in ('seeds','difficulties'):
        if not np.array_equal(original[key],additional[key]):
            raise ValueError(f'paired ordered {key} mismatch')
    n=len(original['seeds'])
    if (not binding.get('identical_ordered_seeds') or binding.get('different_source_rows')!=n
            or binding.get('singleton_seeds') or np.any(original['source_rows']==additional['source_rows'])):
        raise ValueError('paired policy requires a distinct second row for every level')


def paired_batch(views,rows,view_rng,mode):
    rows=np.asarray(rows)
    if rows.ndim!=1 or len(rows)<2 or len(rows)%2:
        raise ValueError('paired batch must have a positive even level count')
    if len(np.unique(views[0]['seeds'][rows]))!=len(rows):
        raise ValueError('duplicate level in paired batch')
    view=np.zeros(len(rows),np.int8);view[view_rng.permutation(len(rows))[:len(rows)//2]]=1
    names=['fields','optimal','seeds','difficulties','source_rows']
    if mode=='successors':
        if any('imagined_fields' not in source for source in views):
            raise ValueError('both views require verified imagined caches')
        names+=['next_fields','imagined_fields']
    elif mode!='direct':raise ValueError('unknown policy mode')
    result={}
    for key in names:
        first=views[0][key]
        result[key]=np.empty((len(rows),*first.shape[1:]),dtype=first.dtype)
        for which in (0,1):
            chosen=np.flatnonzero(view==which)
            result[key][chosen]=views[which][key][rows[chosen]]
    if not np.array_equal(result['seeds'],views[0]['seeds'][rows]):
        raise ValueError('paired seed order drift')
    return result,view


def loss_for_prepared_batch(head,batch,device):
    outputs,masks=outputs_for_prepared_batch(head,batch,device)
    losses={name:policy_terms(logits,masks)['ce'].mean() for name,logits in outputs.items()}
    return sum(losses.values())/len(losses),losses


def probe_batch(views,policy,args):
    size=2**(min(args.max_batch,len(views[0]['seeds'])).bit_length()-1);attempts=[]
    while size>=2:
        head=optimizer=loss=parts=batch=None;started=time.monotonic()
        try:
            batch,view=paired_batch(views,np.arange(size),np.random.default_rng(args.view_seed),args.mode)
            head=new_head(policy.config(),args.seed,args.device)
            optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.01)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            loss,parts=loss_for_prepared_batch(head,batch,args.device)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite preflight loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            if args.device=='cuda':torch.cuda.synchronize()
            attempts.append({'batch_size':size,'view_counts':np.bincount(view,minlength=2).tolist(),
                'status':'fits','seconds':time.monotonic()-started,'loss':float(loss.detach()),
                'gradient_norm':float(norm),'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None})
            return size,attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({'batch_size':size,'status':'out_of_memory'});size//=2
        finally:
            del head,optimizer,loss,parts,batch
            gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
    raise RuntimeError('no even power-of-two paired policy batch fits')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache',default='data/structured-field-16384/train')
    p.add_argument('--validation-cache',default='data/structured-field-16384/validation')
    p.add_argument('--additional-cache',required=True)
    p.add_argument('--additional-imagined-cache')
    p.add_argument('--view-seed',type=int,default=43)
    p.add_argument('--mode',choices=('direct','successors'),default='direct')
    p.add_argument('--world',default='checkpoints/ls20-world-cell-recall-b1024.pt')
    p.add_argument('--visibility',default='checkpoints/ls20-cell-visibility-initial-200.pt')
    p.add_argument('--dynamics')
    p.add_argument('--imagined-cache',help='verified training-only FP32 successor cache root')
    p.add_argument('--checkpoint',required=True);p.add_argument('--report',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--updates',type=int,default=600);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--max-batch',type=int,default=1024);p.add_argument('--eval-batch',type=int,default=128)
    p.add_argument('--dynamics-batch',type=int,default=128);p.add_argument('--loops',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    if not 2<=args.max_batch<=1024 or args.max_batch&(args.max_batch-1):p.error('max batch must bepower-of-two<=1024')
    if not 1<=args.updates<=2000 or not 1<=args.seconds<=1800 or args.eval_batch<1 or not 1<=args.dynamics_batch<=128:
        p.error('boundedupdates/time/batchesrequired')
    if not np.isfinite(args.lr) or args.lr<=0:p.error('positive finite learning rate required')
    if (args.mode=='successors')!=(args.dynamics is not None):p.error('only successors mode requires --dynamics')
    if args.mode=='successors' and not (args.imagined_cache and args.additional_imagined_cache):p.error('successors require both verified imagined cache roots')
    if args.mode=='direct' and (args.imagined_cache or args.additional_imagined_cache):p.error('imagined caches require successors mode')
    if Path(args.report).exists() or Path(args.checkpoint).exists():p.error('refusing existing outputs')
    torch.set_num_threads(1);start=time.monotonic();print('PID',os.getpid(),flush=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report={'status':'running','pid':os.getpid(),'args':vars(args),'training':[],
            'scope':'Learned action readout on verified generated cached fields; no game rollout/control result.',
            'loss_definition':'CE to uniform distribution over optimal actions; irreducible target entropy log(number optimal actions).',
            'limitations':['Frozen perception/dynamics are not optimized.','Successor actual inputs are training/diagnostic only; playable inference uses imagined successors.',
                           'Event-enriched cache sampling changes state priors; validation is not official gameplay.']}
    atomic_json(args.report,report)
    def expired(*_):raise TimeoutError('bounded structured policy deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    try:
        train,tm=load_policy_cache(args.train_cache,'train');val,vm=load_policy_cache(args.validation_cache,'validation')
        additional,am=load_policy_cache(args.additional_cache,'train')
        validate_pair(train,additional,tm,am,args.train_cache)
        views=(train,additional)
        if tm['field_encoder']!=vm['field_encoder'] or np.intersect1d(train['seeds'],val['seeds']).size:
            raise ValueError('encoder mismatch or split leakage')
        policy,save_policy,factored=build_training_policy(args.world,args.visibility,args.dynamics,
            {'mode':args.mode,'loops':args.loops,'successor_batch_size':args.dynamics_batch})
        check_policy_encoder(tm['field_encoder'],policy,factored)
        report['policy_backend']='factored' if factored else 'base'
        if not tm['field_encoder'].get('code_hashes'):
            raise ValueError('cached encoder implementation fingerprints required')
        # Older verified CPU caches bind checkpoint hashes under sources instead
        # of the redundant checkpoint_hashes map. Exact metadata equality above
        # and from_checkpoints already verify those actual checkpoint sources.
        for group in ('code_hashes','checkpoint_hashes'):
            if any(digest(f)!=sha for f,sha in tm['field_encoder'].get(group,{}).items()):
                raise ValueError('cached encoder source drift')
        sources={}
        for directory,manifest in ((args.train_cache,tm),(args.additional_cache,am),(args.validation_cache,vm)):
            sources[str(Path(directory)/'manifest.json')]=digest(Path(directory)/'manifest.json')
            sources.update({str(Path(directory)/(name+'.npy')):info['sha256'] for name,info in manifest['arrays'].items()})
        sources.update(policy.sources['code_hashes'])
        sources.update({r['path']:r['sha256'] for r in policy.sources['artifacts'].values()})
        sources.update(am.get('source_hashes',{}))
        sources[tm['source_path']]=tm['source_sha256']
        for file in (__file__,'tools/train_structured_policy.py','tools/train_structured_transition.py','tools/structured_policy_batch.py'):
            sources[str(file)]=digest(file)
        if args.imagined_cache:
            from tools.cache_structured_policy_successors import load_imagined_cache
            for split,data,manifest,path in (('train',train,tm,args.train_cache),('validation',val,vm,args.validation_cache)):
                data['imagined_fields'],cache_sources=load_imagined_cache(args.imagined_cache,split,path,data,manifest,policy)
                sources.update(cache_sources)
            sources['tools/cache_structured_policy_successors.py']=digest('tools/cache_structured_policy_successors.py')
            additional['imagined_fields'],cache_sources=load_imagined_cache(args.additional_imagined_cache,'train',args.additional_cache,additional,am,policy)
            sources.update(cache_sources)
            report['imagined_cache']=args.imagined_cache
            report['additional_imagined_cache']=args.additional_imagined_cache
        if policy.dynamics is not None:
            policy.dynamics.to('cpu' if args.imagined_cache else args.device).eval().requires_grad_(False)
        report.update(sources=sources,train_levels=len(train['seeds']),validation_levels=len(val['seeds']),
                      parameter_counts=policy.parameter_counts(),precision='float32; autocast/TF32 disabled',
                      encoder_device='cpu',dynamics_device=('cpu' if args.imagined_cache else args.device) if policy.dynamics is not None else None)
        if any(digest(f)!=sha for f,sha in sources.items()):raise ValueError('paired source drift before preflight')
        size,attempts=probe_batch(views,policy,args);report.update(batch_size=size,batch_probe=attempts)
        atomic_json(args.report,report)
        if args.preflight_only:report['status']='preflight_complete'
        else:
            policy.readout=new_head(policy.config(),args.seed,args.device);policy.train()
            optimizer=torch.optim.AdamW(policy.readout.parameters(),lr=args.lr,weight_decay=.01)
            rng=np.random.default_rng(args.seed);view_rng=np.random.default_rng(args.view_seed)
            seen=set();seen_pairs=set();seen_views=[set(),set()];view_draws=np.zeros(2,np.int64);draws=np.zeros(5,np.int64)
            for step in range(args.updates):
                rows=sample_rows(train,size,step/max(args.updates-1,1),rng)
                if len(np.unique(train['seeds'][rows]))!=size:raise RuntimeError('duplicate level in policy batch')
                batch,which=paired_batch(views,rows,view_rng,args.mode)
                optimizer.zero_grad(set_to_none=True)
                loss,parts=loss_for_prepared_batch(policy.readout,batch,args.device)
                if not bool(torch.isfinite(loss)):raise ValueError('nonfinite policy loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.readout.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();seen.update(map(int,train['seeds'][rows]));draws+=np.bincount(train['difficulties'][rows],minlength=6)[1:6]
                view_draws+=np.bincount(which,minlength=2)
                seen_pairs.update(zip(map(int,batch['seeds']),map(int,batch['source_rows'])))
                for v in (0,1):seen_views[v].update(map(int,batch['seeds'][which==v]))
                if step==0 or (step+1)%20==0 or step+1==args.updates:
                    report['training'].append({'step':step+1,'loss':float(loss.detach()),
                        'branch_losses':{k:float(v.detach()) for k,v in parts.items()},'gradient_norm':float(norm),
                        'elapsed_seconds':time.monotonic()-start,'distinct_train_levels_seen':len(seen),'distinct_seed_row_pairs_seen':len(seen_pairs),
                        'view_draws':view_draws.tolist(),'batch_view_counts':np.bincount(which,minlength=2).tolist()})
                    report.update(completed_updates=step+1,distinct_train_levels_seen=len(seen),difficulty_draws=draws.tolist())
                    atomic_json(args.report,report);print(json.dumps(report['training'][-1]),flush=True)
                del loss,parts,batch
            if any(digest(f)!=sha for f,sha in sources.items()):raise ValueError('source changed during policy training')
            if any(p.requires_grad or p.grad is not None for p in policy.encoder.parameters()):raise RuntimeError('encoder training leak')
            if policy.dynamics is not None and any(p.requires_grad or p.grad is not None for p in policy.dynamics.parameters()):
                raise RuntimeError('dynamics training leak')
            report['seen_train_seeds']=sorted(seen)
            report['seen_seed_row_pairs']=[list(x) for x in sorted(seen_pairs)]
            report['seen_levels_by_view']=[sorted(x) for x in seen_views]
            report['view_draws']=view_draws.tolist()
            provenance={'sources':sources,'updates':args.updates,'batch_size':size,'seed':args.seed,'mode':args.mode,
                        'uniform_optimal_cross_entropy':True,'successor_actual_imagined_weights':[.5,.5] if args.mode=='successors' else None,
                        'distinct_train_levels_seen':len(seen),'seen_train_seeds':sorted(seen),'official_inputs_used':False,
                        'paired_views':True,'view_seed':args.view_seed,'view_draws':view_draws.tolist(),
                        'seen_seed_row_pairs':report['seen_seed_row_pairs'],'view_batch_sizes':[size//2,size//2],
                        'fixed_final_no_validation_selection':True,'encoder_and_dynamics_frozen':True}
            target=Path(args.checkpoint);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            save_policy(temporary,policy,provenance);os.replace(temporary,target)
            report.update(status='evaluation_running',checkpoint=str(target),checkpoint_sha256=digest(target))
            atomic_json(args.report,report)
            report['validation']=evaluate(policy.readout,val,args.device,policy.dynamics,args.eval_batch,args.dynamics_batch)
            report['train_evaluation']=evaluate(policy.readout,train,args.device,policy.dynamics,args.eval_batch,args.dynamics_batch)
            report['additional_train_evaluation']=evaluate(policy.readout,additional,args.device,policy.dynamics,args.eval_batch,args.dynamics_batch)
            report['status']='complete'
        if any(digest(f)!=sha for f,sha in sources.items()):raise ValueError('policy experiment sources changed')
        report.update(source_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-start;atomic_json(args.report,report)


if __name__=='__main__':main()
