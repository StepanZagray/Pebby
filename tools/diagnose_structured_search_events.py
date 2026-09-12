"""TRAIN-only search-history event/field diagnosis; no optimizer or policy labels as inputs."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from pebby.agent import world_data as wd
from pebby.agent.structured_search_policy import load_search_policy_checkpoint
from tools.collect_onpolicy_world import select
from tools.collect_structured_event_transitions import capture_branches
from tools.build_structured_field_cache import actual_histories
from tools.train_structured_transition import atomic_json,digest

FORMAT='pebby.structured-search-event-diagnostic.v1'

def next_history(frames,valid,actions,frame,action,lost):
    if lost:
        return np.repeat(frame[None],8,0),np.array([False]*7+[True]),np.full(8,-1,np.int64)
    return np.concatenate([frames[1:],frame[None]]),np.r_[valid[1:],True],np.r_[actions[1:],action]

def selected_specs(path):
    specs=select(path,5,20260912)
    if {int(s['difficulty']) for s in specs}!={1,2,3,4,5} or len({s['seed'] for s in specs})!=5 or any(not 0<=int(s['seed'])<1000000 for s in specs):raise ValueError('five distinct difficulty-stratified TRAIN levels required')
    return specs

@torch.inference_mode()
def encode(policy,h,v,a,device):
    return policy.encoder(torch.as_tensor(h,device=device),torch.as_tensor(v,device=device),torch.as_tensor(a,device=device)).float()

@torch.inference_mode()
def raw_events(policy,current,following,actions):
    d=policy.dynamics
    return d.event_head(torch.cat([d.readout.summary(current),d.readout.summary(following),d.action_embedding(actions)],-1))

def as_numpy(x,dtype=None):return x.detach().cpu().numpy().astype(dtype,copy=False)

@torch.inference_mode()
def collect_level(spec,policy,device,max_actions=96):
    if not 0<=int(spec['seed'])<1000000 or not 1<=max_actions<=96:raise ValueError('bounded TRAIN episode required')
    env,oracle,proof=wd.verified_context(spec,context_index=int(spec['seed'])%7,search_limit=600000)
    if env is None:raise ValueError('context verification failed')
    frames,valid,actions=wd.history_arrays([env.render()],[-1],8);seen=set();rows=[];traces=[];refusals=0;checks=0;missing=0;stop='action_cap'
    for step in range(max_actions):
        key=wd.state_history_key(env,oracle,frames,valid,actions)
        if key in seen:stop='exact_history_repeat';break
        seen.add(key)
        current=encode(policy,frames[None],valid[None],actions[None],device)
        result=policy.search_fields(current);choice=int(result['actions'][0]);plan=result['results'][0]['roots'][choice]['actions']
        # Everything below is offline label capture AFTER the public action.
        capture=capture_branches(env,oracle,int(spec['seed']),step);checks+=capture['branch_checks'];missing+=capture['oracle_unverified_branches']
        batch={'frames':frames[None],'history_valid':valid[None],'previous_actions':actions[None],'next_frames':capture['next_frames'][None],'lost_life':capture['lost_life'][None]}
        h,v,a=actual_histories(batch);actual=encode(policy,h.flatten(0,1),v.flatten(0,1),a.flatten(0,1),device)
        act=torch.arange(4,device=device);pred=policy.dynamics(current.expand(4,-1,-1),act);imagined=pred['field']
        raw_actual=raw_events(policy,current.expand(4,-1,-1),actual,act);raw_imagined=torch.stack([pred['events'][k+'_logits'] for k in ('lost_life','terminal','won')],-1)
        row={k:np.asarray(capture[k]) for k in ['next_frames','lost_life','terminal','won','next_player_cell','next_triple','next_steps','next_lives','next_distance','next_reachable','next_optimal','current_reachable','current_distance','optimal']}
        row.update(frames=frames.copy(),history_valid=valid.copy(),previous_actions=actions.copy(),fields=as_numpy(current[0],np.float16),actual_fields=as_numpy(actual,np.float16),imagined_fields=as_numpy(imagined,np.float16),raw_actual=as_numpy(raw_actual),raw_imagined=as_numpy(raw_imagined),calibrated_actual=as_numpy(policy._outcomes(raw_actual)),calibrated_imagined=as_numpy(policy._outcomes(raw_imagined)),seed=np.int64(spec['seed']),difficulty=np.int8(spec['difficulty']),context=np.int8(spec['seed']%7),step=np.int16(step),selected_action=np.int8(choice),prior_refusals=np.int16(refusals),selected_stall=np.bool_(not capture['lost_life'][choice] and np.array_equal(frames[-1],capture['next_frames'][choice])),branch_unchanged=np.array([np.array_equal(frames[-1],f) for f in capture['next_frames']]),missing_oracle_branches=np.int8(capture['oracle_unverified_branches']))
        for name,field in [('actual',actual),('imagined',imagined)]:
            rd=policy.dynamics.readout(field);row[name+'_player_correct']=as_numpy(rd['player_logits'].argmax(-1))==(capture['next_player_cell'][:,1]*12+capture['next_player_cell'][:,0])
            triple=torch.stack([rd['carried_'+k+'_logits'].argmax(-1) for k in ['shape','color','rotation']],-1);row[name+'_glyph_correct']=(as_numpy(triple)==capture['next_triple']).all(-1)
        row['field_mse']=as_numpy((imagined-actual).square().mean((1,2)));row['copy_mse']=as_numpy((current-actual).square().mean((1,2)))
        # Variable-length trace prefix: no padded fake live transitions.
        clone=wd.clone_env(env);th,tv,ta=frames.copy(),valid.copy(),actions.copy();imag=current;prefix=0
        for horizon,action in enumerate(plan):
            cap=capture if horizon==0 else capture_branches(clone,oracle,int(spec['seed']),step)
            if horizon:checks+=cap['branch_checks'];missing+=cap['oracle_unverified_branches']
            nh,nv,na=next_history(th,tv,ta,cap['next_frames'][action],action,bool(cap['lost_life'][action]))
            nextfield=encode(policy,nh[None],nv[None],na[None],device);out=policy.dynamics(imag,torch.tensor([action],device=device));imag=out['field']
            traces.append({'parent_local_row':np.int32(len(rows)),'seed':np.int64(spec['seed']),'horizon':np.int8(horizon+1),'action':np.int8(action),'frames':nh,'history_valid':nv,'previous_actions':na,'actual_fields':as_numpy(nextfield[0],np.float16),'imagined_fields':as_numpy(imag[0],np.float16),'events':np.array([cap[k][action] for k in ['lost_life','terminal','won']]),'raw_imagined':as_numpy(torch.stack([out['events'][k+'_logits'] for k in ['lost_life','terminal','won']],-1)[0])})
            prefix+=1;clone=cap['_branches'][action];th,tv,ta=nh,nv,na
            if cap['lost_life'][action] or cap['terminal'][action]:break
        row['trace_length']=np.int8(prefix);rows.append(row)
        branch=capture['_branches'][choice];event=capture['_results'][choice]
        refusals=refusals+1 if row['selected_stall'] else 0
        frames,valid,actions=next_history(frames,valid,actions,capture['next_frames'][choice],choice,bool(capture['lost_life'][choice]));env=branch
        if event.finished:stop='won' if event.won else 'terminal_failure';break
    return rows,traces,{'seed':int(spec['seed']),'difficulty':spec['difficulty'],'proof':proof,'rows':len(rows),'stop':stop,'branch_checks':checks,'missing_oracle_branches':missing}

def metrics(arrays):
    out={};selected=arrays['selected_action'].astype(int);n=len(selected)
    masks={'all':np.ones(n,bool),'selected_stalls':arrays['selected_stall'],'selected_nonstalls':~arrays['selected_stall']}
    for k in [0,1,2,4,8]:masks['prior_refusals_'+str(k)]=(arrays['prior_refusals']==k) if k<8 else (arrays['prior_refusals']>=8)
    for name,mask in masks.items():
        ids=np.flatnonzero(mask);r={'states':len(ids),'branches':4*len(ids)}
        for arm in ['actual','imagined']:
            prob=arrays['calibrated_'+arm][ids];truth=arrays['won'][ids];high=prob[:,:,1]>.5
            r[arm]={'player_correct':int(arrays[arm+'_player_correct'][ids].sum()),'glyph_correct':int(arrays[arm+'_glyph_correct'][ids].sum()),'predicted_win_above_half':int(high.sum()),'false_win_above_half':int((high&~truth).sum()),'true_wins':int(truth.sum()),'mean_win_probability':float(prob[:,:,1].mean()) if len(ids) else None,'selected_false_win_above_half':int((high[np.arange(len(ids)),selected[ids]]&~truth[np.arange(len(ids)),selected[ids]]).sum())}
        r.update(field_mse=float(arrays['field_mse'][ids].mean()) if len(ids) else None,copy_mse=float(arrays['copy_mse'][ids].mean()) if len(ids) else None);out[name]=r
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,default=Path('data/ls20-verified-train.jsonl'));p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/ls20-structured-search-ranking-d4.pt'));p.add_argument('--out',type=Path,default=Path('data/search-events-train5.npz'));p.add_argument('--report',type=Path,default=Path('artifacts/search-events-train5.json'));p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--max-actions',type=int,default=96);p.add_argument('--seconds',type=int,default=180);a=p.parse_args()
    if not 1<=a.max_actions<=96 or not 1<=a.seconds<=180:p.error('bounded96actions/180seconds required')
    if a.out.exists() or a.report.exists():p.error('refusing overwrite')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.monotonic();print('PID',os.getpid(),flush=True)
    report={'status':'running','pid':os.getpid(),'split':'train','source':'generated_only','format':FORMAT,'training_performed':False,'official_inputs_used':False,'device':a.device,'precision':'FP32 inference TF32 disabled; fields float16 storage','levels':[]}
    sources={str(x):digest(x) for x in [a.bank,a.checkpoint,Path(__file__),Path('tools/collect_structured_event_transitions.py'),Path('tools/build_structured_field_cache.py'),Path('pebby/agent/world_data.py')]}
    def persist():report['elapsed_seconds']=time.monotonic()-start;atomic_json(a.report,report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('pilot180s deadline')));signal.alarm(a.seconds);persist()
    try:
        specs=selected_specs(a.bank);report['seeds']=[s['seed'] for s in specs];policy,saved=load_search_policy_checkpoint(a.checkpoint,a.device)
        if policy.cfg.depth!=4:raise ValueError('depth4 behavior required')
        report['policy_sources']=saved['sources'];rows=[];traces=[]
        for spec in specs:
            r,t,info=collect_level(spec,policy,a.device,a.max_actions)
            for x in t:x['parent_row']=x.pop('parent_local_row')+len(rows)
            rows.extend(r);traces.extend(t);report['levels'].append(info);persist()
        arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        arrays.update({'trace_'+k:np.stack([r[k] for r in traces]) for k in traces[0]})
        if any(digest(x)!=h for x,h in sources.items()):raise ValueError('source drift')
        report.update(status='complete',rows=len(rows),trace_transitions=len(traces),metrics=metrics(arrays),source_hashes=sources,source_unchanged=True,winning_coverage_required=False,scope='TRAIN diagnostic only; reachable policy labels may be absent; no expert supplementation or world-training schema claim')
        arrays['meta']=np.array(json.dumps(report));a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix('.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        os.link(tmp,a.out);tmp.unlink();report['npz_sha256']=digest(a.out)
    except BaseException as e:report.update(status='failed',error=str(e));raise
    finally:signal.alarm(0);persist()
if __name__=='__main__':main()
