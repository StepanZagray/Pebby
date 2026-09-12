"""Standalone generated distance head; no planner or controller integration."""
import argparse,gc,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.structured_distance import (StructuredDistanceReadout,current_distance_labels,next_distance_labels,distance_targets,distance_loss,discounted_value)
from pebby.agent.structured_factored_policy import StructuredFactoredPolicy
from tools.train_structured_policy import load_policy_cache,check_policy_encoder
from tools.cache_structured_policy_successors import load_imagined_cache
from tools.train_structured_transition import sample_rows,digest,atomic_json

FORMAT='pebby.structured-distance-readout.v1'


def label_data(data):
    value={k:torch.as_tensor(np.array(data[k])) for k in ('optimal','distances','terminal','won','lost_life')}
    return current_distance_labels(value['optimal'],value['distances'],value['terminal'],value['won'],value['lost_life']).numpy(),next_distance_labels(value['distances'],value['terminal'],value['won']).numpy()


def support_from_train(current,following):
    return max(1,int(np.max(current)),int(np.max(following)))


def batch_loss(head,data,rows,actions,device):
    """Only three CE terms in the hot path; no metric synchronizations."""
    rows=np.asarray(rows);actions=np.asarray(actions)
    if rows.ndim!=1 or actions.shape!=rows.shape or not np.issubdtype(actions.dtype,np.integer) or not np.isin(actions,np.arange(4)).all() or len(np.unique(data['seeds'][rows]))!=len(rows):raise ValueError('one action per distinct level required')
    terms=[]
    for key,target_key in [('fields','current_targets'),('next_fields','next_targets'),('imagined_fields','next_targets')]:
        fields=data[key][rows] if key=='fields' else data[key][rows,actions]
        targets=data[target_key][rows] if key=='fields' else data[target_key][rows,actions]
        fields=torch.as_tensor(np.array(fields),device=device).float();targets=torch.as_tensor(np.array(targets),device=device).long()
        terms.append(F.cross_entropy(head(fields),targets))
    return sum(terms)/3


def new_head(max_distance,device):
    torch.manual_seed(42)
    return StructuredDistanceReadout(max_distance=max_distance).to(device)


def validate_batch_size(size):
    if type(size)!=int or not 1<=size<=1024 or size&(size-1):
        raise ValueError('max batch must be a power of two1..1024')


def preflight(data,D,args):
    validate_batch_size(args.max_batch)
    limit=min(args.max_batch,len(data['seeds']))
    if limit<1:raise ValueError('empty training bank')
    size=1<<(limit.bit_length()-1);attempts=[]
    while size:
        head=opt=loss=None
        try:
            head=new_head(D,args.device);opt=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.01)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            rows=sample_rows(data,size,0.,np.random.default_rng(42));actions=np.random.default_rng(43).integers(4,size=len(rows))
            loss=batch_loss(head,data,rows,actions,args.device)
            if not torch.isfinite(loss):raise ValueError('nonfinite preflight')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True);opt.step()
            result={'batch_size':size,'unique_levels':len(np.unique(data['seeds'][rows])),'loss':float(loss.detach()),'gradient_norm':float(norm),'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None}
            attempts.append({**result,'status':'fits'});result['attempts']=attempts
            return result
        except torch.cuda.OutOfMemoryError:
            attempts.append({'batch_size':size,'status':'out_of_memory'});size//=2
        finally:
            del head,opt,loss;gc.collect()
            if args.device=='cuda':torch.cuda.empty_cache()
    raise RuntimeError('no optimizer batch fits')


def diagnostic_rows(data,limit=1024,seed=20260912):
    rng=np.random.default_rng(seed);size=min(limit,len(data['seeds']))
    groups=[list(rng.permutation(np.flatnonzero(data['difficulties']==d))) for d in range(1,6)]
    rows=[]
    while len(rows)<size:
        progressed=False
        for group in groups:
            if group and len(rows)<size:rows.append(group.pop());progressed=True
        if not progressed:raise ValueError('invalid difficulty support')
    return np.asarray(rows,np.int64)


def ranking(values,data,rows):
    masks=np.asarray(data['optimal'][rows]);bits=(masks[:,None]&(1<<np.arange(4)))!=0
    best=values.max(-1,keepdims=True);ties=np.isclose(values,best,rtol=0,atol=1e-7);choice=values.argmax(-1);ix=np.arange(len(rows));lost=np.asarray(data['lost_life'][rows]);dist=np.asarray(data['distances'][rows])
    return {'count':len(rows),'argmax_optimal':int(bits[ix,choice].sum()),'tie_rows':int((ties.sum(-1)>1).sum()),'tie_has_optimal':int((ties&bits).any(-1).sum()),'all_tied_actions_optimal':int((~ties|bits).all(-1).sum()),'argmax_lost_life':int(lost[ix,choice].sum()),'argmax_unreachable':int((dist[ix,choice]<0).sum()),'argmax_won':int(np.asarray(data['won'][rows])[ix,choice].sum())}


@torch.inference_mode()
def evaluate(head,data,D,device,batch_size=32,gamma=.99,selected_rows=None):
    head.eval();metrics={key:{} for key in ('current','actual','imagined')};ranks={key:{} for key in ('actual','imagined')}
    selected=np.arange(len(data['seeds'])) if selected_rows is None else np.asarray(selected_rows)
    if selected.ndim!=1 or not np.issubdtype(selected.dtype,np.integer) or len(selected)==0 or np.any((selected<0)|(selected>=len(data['seeds']))) or len(np.unique(data['seeds'][selected]))!=len(selected):raise ValueError('distinct valid evaluation rows required')
    for begin in range(0,len(selected),batch_size):
        rows=selected[begin:begin+batch_size]
        for mode,key,label in [('current','fields','current_distance'),('actual','next_fields','next_distance'),('imagined','imagined_fields','next_distance')]:
            fields=torch.tensor(np.array(data[key][rows]),device=device).float().reshape(-1,148,96);labels=torch.tensor(np.array(data[label][rows]),device=device).reshape(-1);logits=head(fields);r=distance_loss(logits,labels,D)
            sums={**r['metrics'],'ce_sum':float(r['loss'])*len(labels)}
            for name,value in sums.items():metrics[mode][name]=metrics[mode].get(name,0)+value
            if mode!='current':
                scores=discounted_value(logits,gamma).reshape(len(rows),4).cpu().numpy()
                for name,value in ranking(scores,data,rows).items():ranks[mode][name]=ranks[mode].get(name,0)+value
    for x in metrics.values():x.update(ce=x['ce_sum']/x['count'],finite_mae=x['finite_mae_sum']/x['finite_count'] if x['finite_count'] else None,finite_within1=x['finite_within1_count']/x['finite_count'] if x['finite_count'] else None)
    for x in ranks.values():x['set_accuracy']=x['argmax_optimal']/x['count']
    return {'distance':metrics,'raw_state_value_ranking':ranks,'gamma':gamma,'note':'Diagnostic only: finite reset-state value can prefer life-losing actions; no event gate, planner or controller. Ties use abs tolerance1e-7; argmax tie break follows action order.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','imagined-cache','dynamics','checkpoint','report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--support-train-cache',type=Path,help='Optional verified TRAIN-only vocabulary support; useful for tiny smoke batches')
    p.add_argument('--world',default='checkpoints/ls20-world-cell-recall-b1024.pt');p.add_argument('--visibility',default='checkpoints/ls20-cell-visibility-initial-200.pt');p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--updates',type=int,default=600);p.add_argument('--max-batch',type=int,default=1024);p.add_argument('--eval-batch',type=int,default=32);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seconds',type=int,default=600);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    if not 1<=args.updates<=600 or not 1<=args.max_batch<=1024 or not 1<=args.eval_batch<=128 or not 1<=args.seconds<=1800 or not np.isfinite(args.lr) or args.lr<=0:p.error('invalid bounded run arguments')
    try:validate_batch_size(args.max_batch)
    except ValueError as error:p.error(str(error))
    if args.report.exists() or args.checkpoint.exists():p.error('refusing output overwrite')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();print('PID',os.getpid(),flush=True);report={'status':'running','pid':os.getpid(),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'training':[],'official_inputs_used':False,'policy_integrated':False,'precision':'FP32, no autocast, both TF32 off','source_unchanged':False}
    def persist():report['elapsed_seconds']=time.monotonic()-start;atomic_json(args.report,report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('distance trainer deadline')));signal.alarm(args.seconds);persist()
    try:
        policy=StructuredFactoredPolicy.from_checkpoints(args.world,args.visibility,args.dynamics,device='cpu');data={};manifests={};sources=dict(policy.sources['code_hashes']);sources.update({v['path']:v['sha256'] for v in policy.sources['artifacts'].values()})
        for split in ('train','validation'):
            path=args.cache/split;d,m=load_policy_cache(path,split);check_policy_encoder(m['field_encoder'],policy,True);imagined,guards=load_imagined_cache(args.imagined_cache,split,path,d,m,policy);d['imagined_fields']=imagined;sources.update(guards);sources[str(path/'manifest.json')]=digest(path/'manifest.json');sources.update({str(path/(k+'.npy')):v['sha256'] for k,v in m['arrays'].items()});d['current_distance'],d['next_distance']=label_data(d);data[split]=d;manifests[split]=m
        if np.intersect1d(data['train']['seeds'],data['validation']['seeds']).size:raise ValueError('split leakage')
        support=data['train']
        if args.support_train_cache is not None:
            support,sm=load_policy_cache(args.support_train_cache,'train');check_policy_encoder(sm['field_encoder'],policy,True)
            support['current_distance'],support['next_distance']=label_data(support)
            sources[str(args.support_train_cache/'manifest.json')]=digest(args.support_train_cache/'manifest.json')
            sources.update({str(args.support_train_cache/(k+'.npy')):v['sha256'] for k,v in sm['arrays'].items()})
        D=support_from_train(support['current_distance'],support['next_distance'])
        report['support_train_levels']=len(support['seeds'])
        if args.support_train_cache is not None:del support
        for split,d in data.items():
            for key in ('current','next'):d[key+'_targets']=distance_targets(torch.tensor(d[key+'_distance']),D).numpy()
        for file in (__file__,'pebby/agent/structured_distance.py','tools/train_structured_policy.py','tools/cache_structured_policy_successors.py','tools/train_structured_transition.py'):sources[str(file)]=digest(file)
        report.update(max_distance=D,frozen_binding=policy.sources,source_hashes=sources,levels={k:len(v['seeds']) for k,v in data.items()},loss_weights=[1/3]*3,seed=42,fixed_final_no_validation_selection=True)
        report['preflight']=preflight(data['train'],D,args);size=report['preflight']['batch_size'];report['batch_size']=size;persist()
        if args.preflight_only:report['status']='preflight_complete'
        else:
            head=new_head(D,args.device);opt=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.01);rng=np.random.default_rng(42);seen=set();difficulty_draws=np.zeros(5,np.int64);action_draws=np.zeros(4,np.int64)
            for step in range(args.updates):
                head.train();rows=sample_rows(data['train'],size,step/max(args.updates-1,1),rng);actions=rng.integers(4,size=len(rows));opt.zero_grad(set_to_none=True);loss=batch_loss(head,data['train'],rows,actions,args.device)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True);opt.step();seen.update(map(int,data['train']['seeds'][rows]));report['completed_updates']=step+1
                difficulty_draws+=np.bincount(data['train']['difficulties'][rows],minlength=6)[1:6];action_draws+=np.bincount(actions,minlength=4);report.update(difficulty_draws=difficulty_draws.tolist(),action_draws=action_draws.tolist())
                if step==0 or (step+1)%100==0 or step+1==args.updates:report['training'].append({'step':step+1,'loss':float(loss.detach())});persist()
            report['validation']=evaluate(head,data['validation'],D,args.device,args.eval_batch)
            chosen=diagnostic_rows(data['train']);report['train_diagnostic']=evaluate(head,data['train'],D,args.device,args.eval_batch,selected_rows=chosen);report['train_diagnostic'].update(selection_seed=20260912,rows=chosen.tolist(),seeds=data['train']['seeds'][chosen].tolist())
            if any(digest(f)!=h for f,h in sources.items()):raise ValueError('sources changed')
            args.checkpoint.parent.mkdir(parents=True,exist_ok=True);temp=args.checkpoint.with_name(args.checkpoint.name+'.tmp')
            torch.save(dict(format=FORMAT,config=head.config(),weights=head.state_dict(),parameters=head.parameter_count(),frozen_binding=policy.sources,sources=sources,cache_manifests=manifests,updates=args.updates,batch_size=size,seed=42,difficulty_draws=difficulty_draws.tolist(),action_draws=action_draws.tolist(),seen_train_seeds=sorted(seen),official_inputs_used=False,policy_integrated=False,fixed_final_no_validation_selection=True),temp);os.link(temp,args.checkpoint);temp.unlink();report.update(status='complete',checkpoint_sha256=digest(args.checkpoint),parameters=head.parameter_count(),distinct_train_levels_seen=len(seen))
        if any(digest(f)!=h for f,h in sources.items()):raise ValueError('sources changed')
        report['source_unchanged']=True
    except BaseException as error:report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:signal.alarm(0);persist()

if __name__=='__main__':main()
