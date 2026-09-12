"""Fixed-schedule learned action readout training on verified generated H1 fields.

Loss is cross entropy to the UNIFORM distribution over all optimal actions,
not negative log total optimal probability mass. No controller rollouts here.
"""
from pebby.ls20.provenance import metadata_difficulty_stages, metadata_difficulty_version

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
from torch.nn import functional as F

from pebby.agent.structured_policy import (
    StructuredFieldPolicy, StructuredPolicyReadout, save_structured_policy_checkpoint,
)
from tools.train_structured_transition import load_cache, sample_rows, digest, atomic_json


def build_training_policy(world, visibility, dynamics, config):
    """Select the public wrapper AND serializer by explicit dynamics format."""
    mode = config.get('mode', 'direct')
    if mode not in ('direct', 'successors') or (mode == 'successors') != (dynamics is not None):
        raise ValueError('only successors mode requires a dynamics checkpoint')
    factory, save, factored = StructuredFieldPolicy, save_structured_policy_checkpoint, False
    source_hash = None
    if dynamics is not None:
        raw = Path(dynamics).read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        metadata = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        fmt = metadata.get('format') if isinstance(metadata, dict) else None
        if fmt in ('pebby.structured-transition-global-glyph.v1',
                   'pebby.structured-transition-local-global-glyph.v1'):
            from pebby.agent.structured_factored_policy import (
                StructuredFactoredPolicy, save_factored_policy_checkpoint,
            )
            factory, save, factored = StructuredFactoredPolicy, save_factored_policy_checkpoint, True
        elif fmt != 'pebby.structured-transition.v1':
            raise ValueError(f'unsupported dynamics checkpoint format: {fmt!r}')
    policy = factory.from_checkpoints(world, visibility, dynamics, config, device='cpu')
    if source_hash is not None:
        bound = policy.sources['artifacts']['dynamics']['sha256']
        if bound != source_hash or digest(dynamics) != source_hash:
            raise ValueError('dynamics changed during policy dispatch/loading')
    return policy, save, factored


def check_policy_encoder(cache_encoder, policy, factored=False):
    """Normalize only path spelling for factored wrappers; retain all values."""
    actual = policy.sources['encoder_metadata']
    cached = cache_encoder
    if factored:
        from pebby.agent.structured_factored_policy import canonical_metadata
        cached, actual = canonical_metadata(cached), canonical_metadata(actual)
    if any(cached.get(key) != value for key, value in actual.items()):
        raise ValueError('cached and public inference encoders differ')


def load_policy_cache(path,split):
    data,manifest=load_cache(path,split)
    path=Path(path);n=len(data['seeds'])
    if not n:raise ValueError('empty policy cache')
    # The H1 dynamics loader historically omitted optimal/distances/source_rows.
    # Validate EVERY additional manifest member before exposing any to this task.
    for name,info in manifest['arrays'].items():
        file=path/(name+'.npy')
        if name not in data:
            if digest(file)!=info['sha256']:raise ValueError(f'cache array digest mismatch: {name}')
            data[name]=np.load(file,mmap_mode='r',allow_pickle=False)
        value=data[name]
        if list(value.shape)!=info['shape'] or str(value.dtype)!=info['dtype']:
            raise ValueError(f'cache shape/dtype metadata mismatch: {name}')
        for start in range(0,len(value),64):
            if not np.isfinite(value[start:start+64]).all():raise ValueError(f'nonfinite cache array: {name}')
    for name,shape,low,high in [('optimal',(n,),1,15),('next_optimal',(n,4),0,15),
                               ('distances',(n,4),-1,None),('source_rows',(n,),0,None)]:
        value=data.get(name)
        if value is None or value.shape!=shape or not np.issubdtype(value.dtype,np.integer):
            raise ValueError(f'checked integer policy array required: {name}')
        if (value<low).any() or (high is not None and (value>high).any()):
            raise ValueError(f'{name} outside permitted range')
    if len(np.unique(data['source_rows']))!=n:raise ValueError('one unique source row per distinct level required')
    return data,manifest


def policy_terms(logits,masks,*,allow_unreachable=False):
    if logits.ndim!=2 or logits.shape[1]!=4 or masks.shape!=(len(logits),):
        raise ValueError('policy logits[B,4] and masks[B] required')
    if masks.dtype not in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
        raise ValueError('optimal masks must be integers')
    if bool(((masks<(0 if allow_unreachable else 1))|(masks>15)).any()):raise ValueError('optimal masks must be1..15 (0 only for explicitly masked failure rows)')
    if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite policy logits')
    bits=((masks.long()[:,None]>>torch.arange(4,device=logits.device))&1).float()
    count=bits.sum(-1);log_prob=F.log_softmax(logits.float(),-1)
    ce=-(bits/count[:,None].clamp_min(1)*log_prob).sum(-1)
    return {'ce':ce,'entropy':count.clamp_min(1).log(),'correct':bits.gather(1,logits.argmax(-1)[:,None]).squeeze(1),
            'optimal_probability':(bits*log_prob.exp()).sum(-1),
            **({'valid':count>0} if allow_unreachable else {})}


@torch.no_grad()
def imagined_fields(dynamics,current,max_batch=128):
    if dynamics is None or not 1<=max_batch<=128:raise ValueError('frozen dynamics and batch1..128 required')
    if dynamics.training or any(p.requires_grad for p in dynamics.parameters()):
        raise ValueError('dynamics must remain frozen/eval')
    output=[]
    for first in range(0,len(current)*4,max_batch):
        ids=torch.arange(first,min(first+max_batch,len(current)*4),device=current.device)
        output.append(dynamics.predict(current[ids//4],ids%4).detach())
    result=torch.cat(output).reshape(len(current),4,148,96)
    if not bool(torch.isfinite(result).all()):raise ValueError('nonfinite imagined fields')
    return result


def outputs_for_rows(head,data,rows,device,dynamics=None,dynamics_batch=128):
    current=torch.as_tensor(np.array(data['fields'][rows]),device=device).float()
    masks=torch.as_tensor(np.array(data['optimal'][rows]),device=device).long()
    if head.cfg.mode=='direct':return {'direct':head(current.detach())},masks
    predicted=(torch.as_tensor(np.array(data['imagined_fields'][rows]),device=device)
               if 'imagined_fields' in data else imagined_fields(dynamics,current,dynamics_batch))
    actual=torch.as_tensor(np.array(data['next_fields'][rows]),device=device).float()
    return {'actual':head(actual.detach()),'imagined':head(predicted)},masks


def loss_for_rows(head,data,rows,device,dynamics=None,dynamics_batch=128):
    outputs,masks=outputs_for_rows(head,data,rows,device,dynamics,dynamics_batch)
    losses={name:policy_terms(logits,masks)['ce'].mean() for name,logits in outputs.items()}
    return sum(losses.values())/len(losses),losses


def new_head(config,seed,device):
    torch.manual_seed(seed)
    return StructuredPolicyReadout(config).to(device).train()


def probe_batch(data,policy,args):
    size=2**(min(args.max_batch,len(data['seeds'])).bit_length()-1);attempts=[]
    while size:
        head=optimizer=loss=parts=None;started=time.monotonic()
        try:
            head=new_head(policy.config(),args.seed,args.device)
            optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.01)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            loss,parts=loss_for_rows(head,data,np.arange(size),args.device,policy.dynamics,args.dynamics_batch)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite preflight loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True)
            optimizer.step()
            if args.device=='cuda':torch.cuda.synchronize()
            attempts.append({'batch_size':size,'status':'fits','seconds':time.monotonic()-started,
                             'loss':float(loss.detach()),'gradient_norm':float(norm),
                             'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None})
            return size,attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({'batch_size':size,'status':'out_of_memory'});size//=2
        finally:
            del head,optimizer,loss,parts
            gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
    raise RuntimeError('no power-of-two policy batch fits')


@torch.no_grad()
def evaluate(head,data,device,dynamics=None,batch_size=128,dynamics_batch=128):
    head.eval();sums={};n=len(data['seeds'])
    for first in range(0,n,batch_size):
        rows=np.arange(first,min(first+batch_size,n))
        outputs,masks=outputs_for_rows(head,data,rows,device,dynamics,dynamics_batch)
        for name,logits in outputs.items():
            terms=policy_terms(logits,masks)
            aggregate=sums.setdefault(name,{key:0. for key in terms})
            for key,values in terms.items():aggregate[key]+=float(values.sum())
    result={'levels':n,'input_kind':'cached public current fields' if head.cfg.mode=='direct' else 'actual successor diagnostic / imagined successor inference inputs',
            'metrics':{name:{'set_accuracy':v['correct']/n,'correct':int(v['correct']),
                'uniform_optimal_cross_entropy':v['ce']/n,'uniform_target_entropy':v['entropy']/n,
                'cross_entropy_minus_target_entropy':(v['ce']-v['entropy'])/n,
                'optimal_probability':v['optimal_probability']/n} for name,v in sums.items()}}
    if 'actual' in result['metrics']:
        a,b=result['metrics']['actual'],result['metrics']['imagined']
        result['actual_minus_imagined_accuracy']=a['set_accuracy']-b['set_accuracy']
        result['imagined_minus_actual_cross_entropy']=b['uniform_optimal_cross_entropy']-a['uniform_optimal_cross_entropy']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache',default='data/structured-field-16384/train')
    p.add_argument('--validation-cache',default='data/structured-field-16384/validation')
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
    if not 1<=args.max_batch<=1024 or args.max_batch&(args.max_batch-1):p.error('max batch must bepower-of-two<=1024')
    if not 1<=args.updates<=2000 or not 1<=args.seconds<=1800 or args.eval_batch<1 or not 1<=args.dynamics_batch<=128:
        p.error('boundedupdates/time/batchesrequired')
    if not np.isfinite(args.lr) or args.lr<=0:p.error('positive finite learning rate required')
    if (args.mode=='successors')!=(args.dynamics is not None):p.error('only successors mode requires --dynamics')
    if args.imagined_cache and args.mode!='successors':p.error('imagined cache requires successors mode')
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
        if metadata_difficulty_version(tm) != metadata_difficulty_version(vm):
            raise ValueError('train and validation difficulty versions differ')
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
        for directory,manifest in ((args.train_cache,tm),(args.validation_cache,vm)):
            sources[str(Path(directory)/'manifest.json')]=digest(Path(directory)/'manifest.json')
            sources.update({str(Path(directory)/(name+'.npy')):info['sha256'] for name,info in manifest['arrays'].items()})
        sources.update(policy.sources['code_hashes'])
        sources.update({r['path']:r['sha256'] for r in policy.sources['artifacts'].values()})
        for file in (__file__,'tools/train_structured_transition.py'):
            sources[str(file)]=digest(file)
        if args.imagined_cache:
            from tools.cache_structured_policy_successors import load_imagined_cache
            for split,data,manifest,path in (('train',train,tm,args.train_cache),('validation',val,vm,args.validation_cache)):
                data['imagined_fields'],cache_sources=load_imagined_cache(args.imagined_cache,split,path,data,manifest,policy)
                sources.update(cache_sources)
            sources['tools/cache_structured_policy_successors.py']=digest('tools/cache_structured_policy_successors.py')
            report['imagined_cache']=args.imagined_cache
        if policy.dynamics is not None:
            policy.dynamics.to('cpu' if args.imagined_cache else args.device).eval().requires_grad_(False)
        report.update(sources=sources,train_levels=len(train['seeds']),validation_levels=len(val['seeds']),
                      parameter_counts=policy.parameter_counts(),precision='float32; autocast/TF32 disabled',
                      encoder_device='cpu',dynamics_device=('cpu' if args.imagined_cache else args.device) if policy.dynamics is not None else None)
        size,attempts=probe_batch(train,policy,args);report.update(batch_size=size,batch_probe=attempts)
        atomic_json(args.report,report)
        if args.preflight_only:report['status']='preflight_complete'
        else:
            policy.readout=new_head(policy.config(),args.seed,args.device);policy.train()
            optimizer=torch.optim.AdamW(policy.readout.parameters(),lr=args.lr,weight_decay=.01)
            rng=np.random.default_rng(args.seed);seen=set();draws=np.zeros(len(metadata_difficulty_stages(tm)),np.int64)
            for step in range(args.updates):
                rows=sample_rows(train,size,step/max(args.updates-1,1),rng)
                if len(np.unique(train['seeds'][rows]))!=size:raise RuntimeError('duplicate level in policy batch')
                optimizer.zero_grad(set_to_none=True)
                loss,parts=loss_for_rows(policy.readout,train,rows,args.device,policy.dynamics,args.dynamics_batch)
                if not bool(torch.isfinite(loss)):raise ValueError('nonfinite policy loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.readout.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();seen.update(map(int,train['seeds'][rows]));draws+=np.bincount(train['difficulties'][rows],minlength=len(draws)+1)[1:len(draws)+1]
                if step==0 or (step+1)%20==0 or step+1==args.updates:
                    report['training'].append({'step':step+1,'loss':float(loss.detach()),
                        'branch_losses':{k:float(v.detach()) for k,v in parts.items()},'gradient_norm':float(norm),
                        'elapsed_seconds':time.monotonic()-start,'distinct_train_levels_seen':len(seen)})
                    report.update(completed_updates=step+1,distinct_train_levels_seen=len(seen),difficulty_draws=draws.tolist())
                    atomic_json(args.report,report);print(json.dumps(report['training'][-1]),flush=True)
                del loss,parts
            if any(digest(f)!=sha for f,sha in sources.items()):raise ValueError('source changed during policy training')
            if any(p.requires_grad or p.grad is not None for p in policy.encoder.parameters()):raise RuntimeError('encoder training leak')
            if policy.dynamics is not None and any(p.requires_grad or p.grad is not None for p in policy.dynamics.parameters()):
                raise RuntimeError('dynamics training leak')
            report['seen_train_seeds']=sorted(seen)
            provenance={'sources':sources,'updates':args.updates,'batch_size':size,'seed':args.seed,'mode':args.mode,
                        'uniform_optimal_cross_entropy':True,'successor_actual_imagined_weights':[.5,.5] if args.mode=='successors' else None,
                        'distinct_train_levels_seen':len(seen),'seen_train_seeds':sorted(seen),'official_inputs_used':False,
                        'fixed_final_no_validation_selection':True,'encoder_and_dynamics_frozen':True}
            target=Path(args.checkpoint);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            save_policy(temporary,policy,provenance);os.replace(temporary,target)
            report.update(status='evaluation_running',checkpoint=str(target),checkpoint_sha256=digest(target))
            atomic_json(args.report,report)
            report['validation']=evaluate(policy.readout,val,args.device,policy.dynamics,args.eval_batch,args.dynamics_batch)
            report['train_evaluation']=evaluate(policy.readout,train,args.device,policy.dynamics,args.eval_batch,args.dynamics_batch)
            report['status']='complete'
        if any(digest(f)!=sha for f,sha in sources.items()):raise ValueError('policy experiment sources changed')
        report.update(source_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-start;atomic_json(args.report,report)


if __name__=='__main__':main()
