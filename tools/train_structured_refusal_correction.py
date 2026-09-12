"""Matched same-level H1 refusal correction with ordinary/closing H4 replay.

Only the H1 action distribution differs. No planner, calibrated event or
actor update is part of this experiment. Production batches preserve4:21:7.
"""
import argparse,gc,hashlib,io,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from tools import train_structured_factored_sequences as h4
from tools import train_structured_transition as h1
from tools.train_structured_policy import load_policy_cache
from tools.train_structured_glyph_ablation import losses as h1_losses,evaluate as evaluate_h1
from tools.structured_sequence_metrics import evaluate as evaluate_h4
from tools.train_structured_distance import diagnostic_rows


def counts(size,smoke=False):
    if type(size)!=int or size<8 or size>1024 or size&(size-1) or (size<32 and not smoke):raise ValueError('production power2 B32..1024; B8/16 smoke only')
    one=size//8;four=size-one;close=(four+2)//4
    return one,four-close,close

def sample(h1data,live,closing,candidates,size,progress,rng,smoke=False):
    one,nlive,nclose=counts(size,smoke)
    eligible=np.unique(candidates['cache_rows']);small={k:h1data[k][eligible] for k in ['seeds','difficulties']}
    if one>len(eligible):raise h4.InsufficientMixedLevels('insufficient eligible H1 levels')
    r1=eligible[h1.sample_rows(small,one,progress,rng)];exclude=h1data['seeds'][r1]
    available=np.flatnonzero(~np.isin(closing['seeds'],exclude))
    if nclose>len(available):raise h4.InsufficientMixedLevels('insufficient closing')
    rc=available[h1.sample_rows({k:closing[k][available] for k in ['seeds','difficulties']},nclose,progress,rng)]
    available=np.flatnonzero(~np.isin(live['seeds'],np.r_[exclude,closing['seeds'][rc]]))
    if nlive>len(available):raise h4.InsufficientMixedLevels('insufficient live')
    rl=available[h1.sample_rows({k:live[k][available] for k in ['seeds','difficulties']},nlive,progress,rng)]
    seeds=np.r_[exclude,live['seeds'][rl],closing['seeds'][rc]]
    if len(np.unique(seeds))!=size:raise ValueError('duplicate cross-source levels')
    return r1,rl,rc

def actions_for(rows,candidates,rng,refusal):
    if not refusal:return rng.integers(4,size=len(rows),dtype=np.int64)
    return np.array([rng.choice(candidates['actions'][candidates['cache_rows']==r]) for r in rows],np.int64)

def backward_groups(model,one_batch,four_batch,scale,positive,device,measure=False):
    """Two weighted backward calls, one logical B and one eventual optimizer step."""
    with h1.autocast(device):
        result=h4.sequence_objective(h4.ObjectiveView(model),*four_batch,scale,pos_weight=positive,checkpoint_steps=True)
        extra=h4.h4_glyph_auxiliary_losses(result,four_batch[3],direct_weight=1.,predicted_readout_weight=1.)
        four=.875*(result['total']+extra['total'])
    four.backward();four_value=float(four.detach());before=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()] if measure else None
    del result,extra,four
    with h1.autocast(device):
        one,_,_=h1_losses(model,one_batch,scale,positive,'local-balanced',direct_glyph_weight=1.);one=.125*one
    one.backward();result={'weighted_h4_loss':four_value,'weighted_h1_loss':float(one.detach())}
    if measure:
        after=[p.grad.detach() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()];four_sq=sum(x.float().square().sum() for x in before);one_sq=sum((y-x).float().square().sum() for x,y in zip(before,after));dot=sum((x.float()*(y-x).float()).sum() for x,y in zip(before,after))
        result.update(weighted_h4_gradient_l2=float(four_sq.sqrt()),weighted_h1_gradient_l2=float(one_sq.sqrt()),gradient_cosine=float(dot/(four_sq*one_sq).sqrt().clamp_min(1e-20)),h1_to_h4_gradient_ratio=float(one_sq.sqrt()/four_sq.sqrt().clamp_min(1e-20)))
    return result

def load_candidates(path,data,manifest_path,split="train"):
    raw=Path(path).read_bytes()
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:c={k:z[k] for k in z.files if k!='meta'};meta=json.loads(str(z['meta']))
    # Full proof schema checked by the builder; source file hashes are mandatory.
    if meta.get('status')!='complete' or meta.get('source')!='generated_only' or meta.get('split')!=split or meta.get('h1_manifest_sha256')!=h1.digest(manifest_path):raise ValueError('refusal index split/manifest binding')
    parent=json.loads(Path(manifest_path).read_text())
    if parent['source_sha256']!=meta.get('source_npz_sha256'):raise ValueError('refusal source NPZ mismatch')
    hashes=meta.get('source_array_hashes_verified')
    if not isinstance(hashes,dict) or not hashes or any(h1.digest(p)!=v for p,v in hashes.items()):raise ValueError('refusal proof sources missing/drifted')
    rows=c['cache_rows'];actions=c['actions']
    if rows.ndim!=1 or not np.issubdtype(rows.dtype,np.integer) or not len(rows) or actions.shape!=rows.shape or not np.issubdtype(actions.dtype,np.integer) or ((rows<0)|(rows>=len(data['seeds']))).any() or ((actions<0)|(actions>3)).any():raise ValueError('invalid refusal candidates')
    if not np.array_equal(c['seeds'],data['seeds'][rows]) or not ((c['seeds']>=(0 if split=='train' else 1000000))&(c['seeds']<(1000000 if split=='train' else 2000000))).all():raise ValueError('refusal TRAIN seed alignment')
    if len(set(zip(rows.tolist(),actions.tolist())))!=len(rows):raise ValueError('duplicate refusal candidates')
    for key in ['terminal','won','lost_life']:
        if np.asarray(data[key][rows,actions]).any():raise ValueError('eventful refusal')
    for key in ['player_cell','triple','steps','lives']:
        if not np.array_equal(data[key][rows],data['next_'+key][rows,actions]):raise ValueError('state-changing refusal')
    if not np.array_equal(c['source_rows'],data['source_rows'][rows]):raise ValueError('refusal source row mismatch')
    frame_paths=[p for p in hashes if Path(p).name=='frames.npy'];next_paths=[p for p in hashes if Path(p).name=='next_frames.npy']
    if len(frame_paths)!=1 or len(next_paths)!=1:raise ValueError('exact rendered proof arrays missing')
    frame_manifest=Path(frame_paths[0]).parent/'manifest.json'
    if h1.digest(frame_manifest)!=meta.get('source_manifest_sha256') or json.loads(frame_manifest.read_text()).get('source_sha256')!=meta.get('source_npz_sha256'):raise ValueError('refusal frame manifest/source hash mismatch')
    dependencies={str(frame_manifest):h1.digest(frame_manifest),str(manifest_path):h1.digest(manifest_path)}
    if 'event_report_sha256' in meta:
        discovery=Path('artifacts/structured-events-paired-train.json')
        if h1.digest(discovery)!=meta['event_report_sha256']:raise ValueError('candidate discovery report drift')
        dm=json.loads(discovery.read_text());event_path=dm['arrays']['path']
        if dm['arrays']['sha256']!=meta['event_npz_sha256'] or h1.digest(event_path)!=meta['event_npz_sha256']:raise ValueError('candidate discovery data drift')
        dependencies.update({str(discovery):meta['event_report_sha256'],event_path:meta['event_npz_sha256']})
    meta={**meta,'revalidated_dependency_hashes':dependencies}
    frames=np.load(frame_paths[0],mmap_mode='r');following=np.load(next_paths[0],mmap_mode='r')
    if not np.array_equal(frames[c['source_rows'],-1],following[c['source_rows'],actions]):raise ValueError('refusal actually changes frame')
    if not np.array_equal(c['next_distance'],data['distances'][rows,actions]) or not np.array_equal(c['current_distance'],c['next_distance']):raise ValueError('refusal distance labels inconsistent')
    return c,meta,hashlib.sha256(raw).hexdigest()

def make(saved,device):
    torch.manual_seed(42);return h4._model_for_checkpoint(saved,'local',device)

def preflight(data,live,closing,candidates,saved,scale,positive,args):
    size=args.max_batch;attempts=[]
    while size>=(8 if args.smoke else 32):
        model=opt=b1=b4=None
        try:
            rows=sample(data,live,closing,candidates,size,0.,np.random.default_rng(42),args.smoke);action=actions_for(rows[0],candidates,np.random.default_rng(43),True)
            model=make(saved,args.device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            b1=h1.branch_batch(data,rows[0],action,args.device);b4=h4.mixed_batch(live,closing,rows[1:],args.device)
            stats=backward_groups(model,b1,b4,scale,positive,args.device,True);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
            attempts.append({'batch':size,'status':'fits','gradient_norm':float(norm),**stats,'peak_allocated':torch.cuda.max_memory_allocated() if args.device=='cuda' else None});return size,attempts
        except (torch.cuda.OutOfMemoryError,h4.InsufficientMixedLevels):attempts.append({'batch':size,'status':'oom_or_insufficient'});size//=2
        finally:
            del model,opt,b1,b4;gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
    raise RuntimeError('no allowed complete batch fits')

@torch.inference_mode()
def refusal_metrics(model,data,candidates,device,batch=32,limit=None):
    rows=candidates['cache_rows'];actions=candidates['actions']
    if limit is not None:rows=rows[:limit];actions=actions[:limit]
    sums={'branches':0,'field_mse_sum':0.,'copy_mse_sum':0.,'raw_false_wins':0,'player_correct':0,'glyph_correct':0}
    for first in range(0,len(rows),batch):
        x,y,a,l=h1.branch_batch(data,rows[first:first+batch],actions[first:first+batch],device)
        with h1.autocast(device):o=model(x,a)
        count=len(x);sums['branches']+=count;sums['field_mse_sum']+=float((o['field'].float()-y).square().mean((1,2)).sum());sums['copy_mse_sum']+=float((x-y).square().mean((1,2)).sum());sums['raw_false_wins']+=int(((o['events']['won_logits']>0)&~l['won'].bool()).sum());sums['player_correct']+=int((o['readout']['player_logits'].argmax(-1)==l['next_player_cell'][:,1]*12+l['next_player_cell'][:,0]).sum())
        glyph=torch.stack([o['field'][:,0,lo:hi].argmax(-1) for lo,hi in [(70,76),(76,80),(80,84)]],-1);sums['glyph_correct']+=int((glyph==l['next_triple']).all(-1).sum())
    return sums

@torch.inference_mode()
def event_summary(model,data,rows,device,batch=32):
    from tools.evaluate_structured_event_head import _event_metrics
    probs=[];targets=[]
    for start in range(0,len(rows),batch):
        chosen=rows[start:start+batch]
        for action in range(4):
            x,y,a,l=h1.branch_batch(data,chosen,np.full(len(chosen),action,np.int64),device)
            with h1.autocast(device):o=model(x,a)
            probs.append(o['events']['won_logits'].float().sigmoid().cpu().numpy());targets.append(l['won'].cpu().numpy())
    return _event_metrics(np.concatenate(probs),np.concatenate(targets).astype(np.int8))

@torch.inference_mode()
def search_diagnostic(model,path,device,smoke=False):
    from tools.evaluate_structured_event_head import _event_metrics
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['meta']))
        if meta.get('split')!='train' or meta.get('source')!='generated_only':raise ValueError('TRAIN search diagnostic required')
        fields=z['fields'];actual=z['actual_fields'];selected=z['selected_action'];stall=z['selected_stall'];won=z['won']
        if fields.dtype!=np.float16 or actual.dtype!=np.float16 or fields.shape!=(346,148,96) or actual.shape!=(346,4,148,96) or meta.get('official_inputs_used') is not False or meta.get('training_performed') is not False or not ((z['seed']>=0)&(z['seed']<1000000)).all():raise ValueError('expected immutable TRAIN5 FP16 diagnostic contract')
    n=min(len(fields),8) if smoke else len(fields);predprobs=[];actualprobs=[];mse=[]
    for first in range(0,n,8):
        end=min(first+8,n);x=torch.as_tensor(fields[first:end],device=device).float();target=torch.as_tensor(actual[first:end],device=device).float();a=torch.arange(4,device=device).repeat(end-first);current=x[:,None].expand(-1,4,-1,-1).reshape(-1,148,96)
        with h1.autocast(device):
            o=model(current,a);raw=model.event_head(torch.cat([model.readout.summary(current),model.readout.summary(target.flatten(0,1)),model.action_embedding(a)],-1))
        predprobs.append(o['events']['won_logits'].float().sigmoid().reshape(-1,4).cpu().numpy());actualprobs.append(raw[:,2].float().sigmoid().reshape(-1,4).cpu().numpy());mse.append((o['field'].float()-target.flatten(0,1)).square().mean((1,2)).reshape(-1,4).cpu().numpy())
    pp=np.concatenate(predprobs);ap=np.concatenate(actualprobs);errors=np.concatenate(mse);result={}
    for key,mask in [('all',np.ones(n,bool)),('selected_stalls',stall[:n])]:
        rows=np.flatnonzero(mask);act=selected[rows]
        result[key]={'states':len(rows),'raw_predicted_win':_event_metrics(pp[rows,act],won[rows,act].astype(np.int8)) if len(rows) else None,'raw_actual_win':_event_metrics(ap[rows,act],won[rows,act].astype(np.int8)) if len(rows) else None,'selected_field_mse':float(errors[rows,act].mean()) if len(rows) else None,'selected_mean_raw_win_probability':float(pp[rows,act].mean()) if len(rows) else None}
    return {'same_float16_fields_for_baseline_and_arms':True,'old_calibration_applied':False,'metrics':result}

@torch.inference_mode()
def all_metrics(model,one,oneval,candidates,val_candidates,closing,val,scale,args):
    from types import SimpleNamespace
    args=SimpleNamespace(**(vars(args)|{'device':'cpu'}));model.to('cpu').eval();scale=scale.to('cpu');batch=2 if args.smoke else 32
    vr=diagnostic_rows(oneval,8 if args.smoke else 512)
    # Preserve the genuinely positive closing population, not empty H4 VAL event metrics.
    cr=np.flatnonzero(np.asarray(closing['won']).any(-1))
    if args.smoke:cr=cr[:2]
    cv={k:v[cr] for k,v in closing.items()}
    vv={k:v[:2] for k,v in val.items()} if args.smoke else val
    return {'refusal_train':refusal_metrics(model,one,candidates,args.device,batch,8 if args.smoke else None),'refusal_validation':refusal_metrics(model,oneval,val_candidates,args.device,batch,8 if args.smoke else None),'ordinary_h1_validation':evaluate_h1(model,oneval,vr,args.device,batch),'h1_validation_win':event_summary(model,oneval,vr,args.device,batch),'h4_validation':evaluate_h4(model,vv,scale,device=args.device,batch_size=batch),'closing_train_positive':evaluate_h4(model,cv,scale,device=args.device,batch_size=batch) if len(cr) else None,'closing_train_positive_levels':len(cr),'search_train5':search_diagnostic(model,args.search_diagnostic,args.device,args.smoke)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['h1-cache','live-cache','closing-cache','validation-cache','refusal-index','validation-refusal-index','out','report']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--world',default='checkpoints/ls20-world-cell-recall-b1024.pt');p.add_argument('--visibility',default='checkpoints/ls20-cell-visibility-initial-200.pt');p.add_argument('--search-diagnostic',type=Path,default=Path('data/search-events-train5.npz'));p.add_argument('--initialize',type=Path,default=Path('checkpoints/ls20-factored-local-h4-400.pt'));p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--max-batch',type=int,default=1024);p.add_argument('--updates',type=int,default=200);p.add_argument('--smoke',action='store_true');p.add_argument('--seconds',type=int,default=1800);a=p.parse_args()
    try:counts(a.max_batch,a.smoke)
    except ValueError as e:p.error(str(e))
    if (not a.smoke and (a.updates!=200 or a.max_batch!=1024)) or (a.smoke and (a.device!='cpu' or a.updates!=2 or a.max_batch!=8)) or not 1<=a.seconds<=1800:p.error('fixed200/B1024 production or CPU2updateB8 smoke')
    if a.out.exists() or a.report.exists():p.error('refusing overwrite')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.monotonic();print('PID',os.getpid(),flush=True)
    report={'status':'running','pid':os.getpid(),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'arms':{},'official_inputs_used':False,'policy_integrated':False,'loss_weights':{'H4':.875,'H1':.125},'glyph_weights':{'direct':1,'predicted_readout':1},'precision':'BF16 CUDA/FP32 CPU; checkpointed H4; TF32 disabled','calibration_valid':False,'fixed_final_no_validation_selection':True}
    def persist():report['elapsed_seconds']=time.monotonic()-start;h1.atomic_json(a.report,report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded fit deadline')));signal.alarm(a.seconds);persist()
    try:
        one,m1=load_policy_cache(a.h1_cache/'train','train');oneval,m1v=load_policy_cache(a.h1_cache/'validation','validation');live,ml=h4.load_exploratory_cache(a.live_cache);closing,mc=h4.load_cache(a.closing_cache,'train');val,mv=h4.load_cache(a.validation_cache,'validation')
        manifests={'h1_train':m1,'h1_validation':m1v,'live_train':ml,'closing_train':mc,'validation':mv}
        if any(m['field_encoder']!=ml['field_encoder'] for m in manifests.values()):raise ValueError('encoder cache mismatch')
        if np.intersect1d(np.unique(np.r_[one['seeds'],live['seeds'],closing['seeds']]),np.unique(np.r_[oneval['seeds'],val['seeds']])).size:raise ValueError('validation leakage')
        candidates,proof,indexsha=load_candidates(a.refusal_index,one,a.h1_cache/'train'/'manifest.json')
        val_candidates,valproof,valindexsha=load_candidates(a.validation_refusal_index,oneval,a.h1_cache/'validation'/'manifest.json','validation')
        if (len(candidates['cache_rows']),len(np.unique(candidates['seeds'])),len(val_candidates['cache_rows']),len(np.unique(val_candidates['seeds'])))!=(1113,1103,36,34):raise ValueError('unexpected fixed refusal coverage')
        if len(oneval['seeds'])!=512 or int(oneval['won'].sum())!=43 or any(np.asarray(val[k]).any() for k in h1.EVENTS):raise ValueError('unexpected H1/H4 validation coverage')
        raw=a.initialize.read_bytes();initsha=hashlib.sha256(raw).hexdigest();saved=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu');h4._check_encoder(saved,ml['field_encoder']);h4._check_initial_model_sources(saved,'local')
        if saved.get('format')!=h4.LOCAL_FORMAT or saved.get('parameters')!=294664 or saved.get('official_inputs_used') is not False:raise ValueError('strict local generated initializer required')
        scale=torch.as_tensor(saved['feature_scale'],device=a.device);positive=torch.as_tensor(saved['event_positive_weights'],device=a.device)
        if not torch.equal(positive,torch.full_like(positive,20.)):raise ValueError('expected fixed old eventpositive20')
        sources={str(a.initialize):initsha,str(a.refusal_index):indexsha,str(a.validation_refusal_index):valindexsha,**proof['source_array_hashes_verified'],**valproof['source_array_hashes_verified'],**proof['revalidated_dependency_hashes'],**valproof['revalidated_dependency_hashes']}
        for directory,m in [(a.h1_cache/'train',m1),(a.h1_cache/'validation',m1v),(a.live_cache,ml),(a.closing_cache,mc),(a.validation_cache,mv)]:
            sources[str(directory/'manifest.json')]=h1.digest(directory/'manifest.json');sources.update({str(directory/(k+'.npy')):v['sha256'] for k,v in m['arrays'].items()})
        sources.update(saved['sources']);sources[str(a.search_diagnostic)]=h1.digest(a.search_diagnostic)
        for file in [__file__,h4.__file__,h1.__file__,'tools/train_structured_glyph_ablation.py','pebby/agent/structured_objective.py','pebby/agent/structured_sequence_objective.py','tools/evaluate_structured_event_head.py']:sources[str(file)]=h1.digest(file)
        if any(h1.digest(f)!=v for f,v in sources.items()):raise ValueError('source drift before preflight')
        report.update(sources=sources,refusal_proof=proof,initial_sha256=initsha)
        size,attempts=preflight(one,live,closing,candidates,saved,scale,positive,a);report.update(batch_size=size,source_counts=counts(size,a.smoke),preflight=attempts);a.out.mkdir(parents=True);persist()
        baseline=make(saved,a.device).eval();report['baseline']=all_metrics(baseline,one,oneval,candidates,val_candidates,closing,val,scale,a);del baseline;gc.collect();persist()
        for arm,refusal in [('control',False),('refusal',True)]:
            model=make(saved,a.device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);rng=np.random.default_rng(42);arng=np.random.default_rng(43);selection=hashlib.sha256();actionhash=hashlib.sha256();seen=set();draws={k:np.zeros(5,np.int64) for k in ['h1','live','closing']};curve=[]
            report['active_arm']=arm;report['arms'][arm]={'status':'training'};persist()
            for step in range(a.updates):
                rows=sample(one,live,closing,candidates,size,step/max(a.updates-1,1),rng,a.smoke);action=actions_for(rows[0],candidates,arng,refusal)
                for r in rows:selection.update(r.astype('<i8').tobytes())
                actionhash.update(action.astype('<i8').tobytes());b1=h1.branch_batch(one,rows[0],action,a.device);b4=h4.mixed_batch(live,closing,rows[1:],a.device);opt.zero_grad(set_to_none=True)
                stats=backward_groups(model,b1,b4,scale,positive,a.device,step==0 or (step+1)%100==0);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step();del b1,b4
                for key,d,r in zip(['h1','live','closing'],[one,live,closing],rows):seen.update(map(int,d['seeds'][r]));draws[key]+=np.bincount(d['difficulties'][r],minlength=6)[1:6]
                if step==0 or (step+1)%50==0 or step+1==a.updates:
                    curve.append({'step':step+1,**stats,'gradient_norm':float(norm),'elapsed_seconds':time.monotonic()-start});report['arms'][arm].update(completed_updates=step+1,training=curve);persist()
                if (step+1)%50==0 or step+1==a.updates:
                    payload={'format':h4.LOCAL_FORMAT,'config':model.config(),'weights':model.state_dict(),'parameters':model.parameter_count(),'sources':sources,'cache_manifests':manifests,'feature_scale':scale.cpu(),'event_positive_weights':positive.cpu(),'updates':step+1,'batch_size':size,'seed':42,'initialize':str(a.initialize),'official_inputs_used':False,'policy_integrated':False,'objective':'matched_same_level_refusal_H1_plus_H4','direct_glyph_weight':1.,'predicted_readout_glyph_weight':1.,'arm':arm,'selection_sha256':selection.hexdigest(),'action_sha256':actionhash.hexdigest(),'seen_train_seeds':sorted(seen),'calibration_valid':False,'fixed_final_no_validation_selection':True}
                    target=a.out/f'{arm}-step{step+1}.pt';tmp=target.with_suffix('.tmp');torch.save(payload,tmp);os.link(tmp,target);tmp.unlink()
            model.eval();report['arms'][arm].update(status='evaluating');persist()
            r=report['arms'][arm];r.update(selection_sha256=selection.hexdigest(),action_sha256=actionhash.hexdigest(),difficulty_draws={k:v.tolist() for k,v in draws.items()},distinct_levels=len(seen))
            r['diagnostics']=all_metrics(model,one,oneval,candidates,val_candidates,closing,val,scale,a)
            from pebby.agent.structured_factored_policy import StructuredFactoredPolicy,state_digest
            final_path=a.out/f'{arm}-step{a.updates}.pt'
            rebuilt=StructuredFactoredPolicy.from_checkpoints(a.world,a.visibility,final_path,device='cpu')
            if rebuilt.dynamics.parameter_count()!=294664 or state_digest(rebuilt.dynamics.state_dict())!=state_digest(model.state_dict()):raise ValueError('strict final dynamics reconstruction mismatch')
            r.update(checkpoint_path=str(final_path),checkpoint_sha256=h1.digest(final_path),strict_factory_reload=True,state_sha256=state_digest(model.state_dict()))
            del rebuilt
            r['status']='complete';persist();del model,opt;gc.collect()
            if a.device=='cuda':torch.cuda.empty_cache()
        if report['arms']['control']['selection_sha256']!=report['arms']['refusal']['selection_sha256']:raise ValueError('unmatched level draws')
        if any(h1.digest(f)!=v for f,v in sources.items()):raise ValueError('source drift')
        report.update(status='complete',source_unchanged=True,active_arm=None)
    except BaseException as e:report.update(status='failed',error=str(e));raise
    finally:signal.alarm(0);persist()
if __name__=='__main__':main()
