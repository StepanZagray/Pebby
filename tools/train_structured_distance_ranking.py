"""Matched distance/ranking diagnostic; no planner, rollout or policy integration."""
import argparse, gc, hashlib, io, os, signal, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools import train_structured_distance as base
from pebby.agent.structured_factored_policy import canonical_metadata

INIT_SHA='63c9d71d78fe4524a2a8621993283ba5679e5f61588f57a716c2d628ccf3f59d'

def ranking_loss(logits, optimal, lost_life, gamma=.99):
    """Uniform complete-optimal target, conditioned on no life loss. Labels only."""
    if logits.ndim!=3 or logits.shape[1]!=4 or not 0<gamma<1:raise ValueError('expected B4C logits and gamma in (0,1)')
    optimal=optimal.detach();lost_life=lost_life.detach()
    if optimal.shape!=logits.shape[:1] or optimal.dtype not in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64) or lost_life.shape!=logits.shape[:2] or lost_life.dtype!=torch.bool:raise ValueError('invalid ranking labels')
    if ((optimal<1)|(optimal>15)).any():raise ValueError('complete nonzero masks required')
    bits=(optimal[:,None]&(1<<torch.arange(4,device=logits.device)))!=0
    eligible=~lost_life
    if (bits&~eligible).any():raise ValueError('optimal action loses life')
    valid=(eligible&~bits).any(-1)
    logp=F.log_softmax(logits.float(),-1)
    support=torch.arange(logits.shape[-1]-1,device=logits.device)*np.log(gamma)
    scores=torch.logsumexp(logp[...,:-1]+support,-1)/(-np.log(gamma))
    logchoice=F.log_softmax(scores.masked_fill(~eligible,-torch.inf),-1)
    perrow=-torch.where(bits,logchoice,0.).sum(-1)/bits.sum(-1)
    # Clamp denominator keeps no-information batches differentiable and finite.
    return (perrow*valid).sum()/valid.sum().clamp_min(1), valid.sum()

def objective(head,data,rows,device,weight):
    if len(np.unique(data['seeds'][rows]))!=len(rows):raise ValueError('distinct levels required')
    ce=[];ranks=[];count=None
    for key,target in [('fields','current_targets'),('next_fields','next_targets'),('imagined_fields','next_targets')]:
        x=torch.as_tensor(data[key][rows],device=device).float().reshape(-1,148,96)
        y=torch.as_tensor(data[target][rows],device=device).long().reshape(-1)
        z=head(x);ce.append(F.cross_entropy(z,y))
        if key!='fields':
            r,count=ranking_loss(z.reshape(len(rows),4,-1),torch.as_tensor(data['optimal'][rows],device=device),torch.as_tensor(data['lost_life'][rows],device=device).bool());ranks.append(r)
    absolute=sum(ce)/3;relative=sum(ranks)/2
    return absolute+weight*relative, {'absolute_ce':absolute.detach(),'ranking_loss':relative.detach(),'ranking_rows':count.detach()}

def load_initial(path,policy,D):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INIT_SHA:raise ValueError('initial checkpoint SHA mismatch')
    c=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if c.get('format')!=base.FORMAT or c.get('official_inputs_used') is not False or c.get('policy_integrated') is not False:raise ValueError('invalid initializer format/provenance')
    if canonical_metadata(c.get('frozen_binding'))!=canonical_metadata(policy.sources):raise ValueError('initializer frozen binding mismatch')
    if not c.get('sources') or any(base.digest(p)!=h for p,h in c['sources'].items()):raise ValueError('initializer source drift')
    h=base.StructuredDistanceReadout(c['config'])
    if h.cfg.max_distance!=D or c.get('parameters')!=h.parameter_count():raise ValueError('initializer support/count mismatch')
    h.load_state_dict(c['weights'],strict=True)
    return c

def make_head(initial,device):
    torch.manual_seed(42);h=base.StructuredDistanceReadout(initial['config']);h.load_state_dict(initial['weights'],strict=True);return h.to(device)

def preflight(data,initial,args):
    base.validate_batch_size(args.max_batch);limit=min(args.max_batch,len(data['seeds']));size=1<<(limit.bit_length()-1);attempts=[]
    while size:
        h=opt=loss=detail=None
        try:
            h=make_head(initial,args.device);opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.01)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            rows=base.sample_rows(data,size,0.,np.random.default_rng(42));loss,detail=objective(h,data,rows,args.device,1.)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(h.parameters(),10.,error_if_nonfinite=True);opt.step()
            attempts.append({'batch_size':size,'status':'fits','loss':float(loss.detach()),'gradient_norm':float(norm)})
            return {'batch_size':size,'attempts':attempts,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None}
        except torch.cuda.OutOfMemoryError:attempts.append({'batch_size':size,'status':'oom'});size//=2
        finally:
            del h,opt,loss,detail;gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
    raise RuntimeError('no optimizer batch fits')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','imagined-cache','dynamics','out','report'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--initial',type=Path,default=Path('checkpoints/ls20-structured-distance-local-h4-600.pt'))
    p.add_argument('--world',default='checkpoints/ls20-world-cell-recall-b1024.pt');p.add_argument('--visibility',default='checkpoints/ls20-cell-visibility-initial-200.pt')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--max-batch',type=int,default=1024);p.add_argument('--updates',type=int,default=200);p.add_argument('--smoke',action='store_true');p.add_argument('--seconds',type=int,default=900);a=p.parse_args()
    try:base.validate_batch_size(a.max_batch)
    except ValueError as e:p.error(str(e))
    if not 1<=a.seconds<=1800 or (not a.smoke and (a.updates!=200 or a.max_batch!=1024)) or (a.smoke and (a.device!='cpu' or not 1<=a.updates<=2 or a.max_batch>8)):p.error('fixed200/B1024 main experiment; CPU smoke <=2 updates/B8')
    if a.out.exists() or a.report.exists():p.error('refusing output overwrite')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();print('PID',os.getpid(),flush=True)
    report={'status':'running','pid':os.getpid(),'smoke':a.smoke,'arms':{},'official_inputs_used':False,'policy_integrated':False,'precision':'FP32, no autocast, TF32 off','gamma':.99,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    def persist():report['elapsed_seconds']=time.monotonic()-start;base.atomic_json(a.report,report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('deadline')));signal.alarm(a.seconds);persist()
    try:
        policy=base.StructuredFactoredPolicy.from_checkpoints(a.world,a.visibility,a.dynamics,device='cpu');data={};manifests={};sources=dict(policy.sources['code_hashes']);sources.update({v['path']:v['sha256'] for v in policy.sources['artifacts'].values()})
        for split in ('train','validation'):
            path=a.cache/split;d,m=base.load_policy_cache(path,split);base.check_policy_encoder(m['field_encoder'],policy,True)
            d['imagined_fields'],guards=base.load_imagined_cache(a.imagined_cache,split,path,d,m,policy);sources.update(guards)
            sources[str(path/'manifest.json')]=base.digest(path/'manifest.json');sources.update({str(path/(k+'.npy')):v['sha256'] for k,v in m['arrays'].items()})
            d['current_distance'],d['next_distance']=base.label_data(d);data[split]=d;manifests[split]=m
        if np.intersect1d(data['train']['seeds'],data['validation']['seeds']).size:raise ValueError('split leakage')
        D=base.support_from_train(data['train']['current_distance'],data['train']['next_distance'])
        for d in data.values():
            for k in ('current','next'):d[k+'_targets']=base.distance_targets(torch.tensor(d[k+'_distance']),D).numpy()
        initial=load_initial(a.initial,policy,D);sources.update(initial['sources']);sources[str(a.initial)]=INIT_SHA
        for f in (__file__,base.__file__):sources[str(f)]=base.digest(f)
        report.update(source_hashes=sources,initial_sha256=INIT_SHA,max_distance=D,frozen_binding=policy.sources)
        report['preflight']=preflight(data['train'],initial,a);size=report['preflight']['batch_size'];report['batch_size']=size;persist();a.out.mkdir(parents=True)
        for arm,weight in [('control',0.),('ranking',1.)]:
            arm_start=time.monotonic();report['active_arm']=arm;report['arms'][arm]={'status':'training','completed_updates':0,'weight':weight,'training':[]};persist()
            h=make_head(initial,a.device);opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.01);rng=np.random.default_rng(42);draws=np.zeros(5,np.int64);seen=set();curve=[];rank_rows=0;selection_hash=hashlib.sha256()
            for step in range(a.updates):
                rows=base.sample_rows(data['train'],size,step/max(a.updates-1,1),rng);selection_hash.update(rows.astype('<i8').tobytes());opt.zero_grad(set_to_none=True)
                loss,detail=objective(h,data['train'],rows,a.device,weight);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),10.,error_if_nonfinite=True);opt.step()
                draws+=np.bincount(data['train']['difficulties'][rows],minlength=6)[1:6];seen.update(map(int,data['train']['seeds'][rows]));rank_rows+=int(detail['ranking_rows'])
                if step==0 or (step+1)%100==0 or step+1==a.updates:
                    curve.append({'step':step+1,'elapsed_seconds':time.monotonic()-arm_start,'loss':float(loss.detach()),**{k:float(v) for k,v in detail.items()}})
                    report['arms'][arm].update(completed_updates=step+1,elapsed_seconds=time.monotonic()-arm_start,training=curve);persist()
            report['arms'][arm]['status']='evaluating';persist()
            trainrows=base.diagnostic_rows(data['train'],8 if a.smoke else 1024);valrows=base.diagnostic_rows(data['validation'],8) if a.smoke else None
            result={'status':'complete','elapsed_seconds':time.monotonic()-arm_start,'completed_updates':a.updates,'weight':weight,'training':curve,'difficulty_draws':draws.tolist(),'action_draws':[size*a.updates]*4,'ranking_rows':rank_rows,'selection_sha256':selection_hash.hexdigest(),'train_rows':trainrows.tolist(),'train_selection_seed':20260912,'train_diagnostic':base.evaluate(h,data['train'],D,a.device,2 if a.smoke else 32,selected_rows=trainrows),'validation':base.evaluate(h,data['validation'],D,a.device,2 if a.smoke else 32,selected_rows=valrows)}
            if any(base.digest(f)!=v for f,v in sources.items()):raise ValueError('source drift')
            target=a.out/(arm+'.pt');temp=target.with_suffix('.tmp')
            torch.save(dict(format=base.FORMAT,config=h.config(),weights=h.state_dict(),parameters=h.parameter_count(),frozen_binding=policy.sources,sources=sources,cache_manifests=manifests,updates=a.updates,batch_size=size,seed=42,official_inputs_used=False,policy_integrated=False,fixed_final_no_validation_selection=True,ranking_weight=weight,initial_sha256=INIT_SHA,seen_train_seeds=sorted(seen)),temp);os.link(temp,target);temp.unlink()
            result['checkpoint_sha256']=base.digest(target);report['arms'][arm]=result;persist();del h,opt,loss,detail;gc.collect()
            if a.device=='cuda':torch.cuda.empty_cache()
        if report['arms']['control']['selection_sha256']!=report['arms']['ranking']['selection_sha256']:raise ValueError('unmatched samples')
        report.update(status='complete',source_unchanged=True,active_arm=None)
    except BaseException as e:report.update(status='failed',error=str(e));raise
    finally:signal.alarm(0);persist()

if __name__=='__main__':main()
