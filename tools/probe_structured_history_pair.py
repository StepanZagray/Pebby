"""Two matched fixed TRAIN64 fits; diagnostic only, no saved model weights."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from pebby.agent.structured_transition import StructuredTransition, steps_targets
from pebby.agent.structured_objective import objective,diagnostics
from tools.train_structured_transition import training_scale,LABELS
from tools.build_structured_field_cache import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path('data/structured-history-pair-train64'));p.add_argument('--report',type=Path,default=Path('artifacts/structured-history-pair-fit400.json'));args=p.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();print('PID',os.getpid(),flush=True)
    r=dict(status='running',pid=os.getpid(),device='cuda',compute_dtype='float32',autocast=False,tf32=False,batch_size=64,updates_per_arm=400,lr=.001,weight_decay=.01,clip_norm=10.,seed=42,arms={},weights_saved=False,validation_used=False,official_inputs_used=False,scope='Fixed conditional TRAIN64 trainability comparison; not main B1024 training or generalization evidence.')
    def persist():
        r['elapsed_seconds']=time.monotonic()-start;tmp=args.report.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(args.report)
    def expired(*_):raise TimeoutError('paired GPU diagnostic180second deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(180)
    try:
        m=json.loads((args.cache/'manifest.json').read_text())
        if m.get('format')!='pebby.structured-history-pair.v1' or m.get('status')!='complete' or m.get('split')!='train' or m.get('source')!='generated_only' or m.get('levels')!=64:raise ValueError('requires paired64 TRAIN cache')
        guards={str(args.cache/'manifest.json'):digest(args.cache/'manifest.json')}
        arrays={}
        for name,info in m['arrays'].items():
            file=args.cache/name
            if digest(file)!=info['sha256']:raise ValueError('paired array drift')
            guards[str(file)]=info['sha256'];a=np.load(file)
            if list(a.shape)!=info['shape'] or str(a.dtype)!=info['dtype'] or not np.isfinite(a).all():raise ValueError('bad array')
            arrays[name]=a
        for file in (__file__,'pebby/agent/structured_transition.py','pebby/agent/structured_objective.py','tools/train_structured_transition.py'):
            guards[str(file)]=digest(file)
        guards.update(m['source_hashes'])
        if any(digest(file)!=sha for file,sha in guards.items()):raise ValueError('parent sources changed')
        seeds=arrays['seeds.npy'];assert len(np.unique(seeds))==64 and ((seeds>=0)&(seeds<1000000)).all()
        # Shared H8 TRAIN-current standard deviation, never arm-specific or target-derived.
        scale=torch.tensor(training_scale({'fields':arrays['h8/fields.npy'],'seeds':seeds}),device='cuda')
        labels={key:torch.as_tensor(arrays[key+'.npy'],device='cuda') for key in LABELS}
        actions=torch.tensor(arrays['actions.npy'],device='cuda').long()
        rates=np.array([arrays[k+'.npy'].mean() for k in ('lost_life','terminal','won')]);pos=torch.tensor(np.clip((1-rates)/np.maximum(rates,1e-8),1,20),device='cuda',dtype=torch.float32)
        source_manifest=json.loads(Path(next(k for k in m['source_hashes'] if k.endswith('/train/manifest.json'))).read_text())
        cached=next(x for x in Path('data/world-array-cache').glob(source_manifest['source_sha256']+'-*') if x.is_dir())
        frames=np.load(cached/'frames.npy',mmap_mode='r');nxt=np.load(cached/'next_frames.npy',mmap_mode='r');rows=arrays['source_rows.npy'];act=arrays['actions.npy']
        stationary=torch.tensor((frames[rows,-1]==nxt[rows,act]).all((-1,-2)),device='cuda')
        r.update(source_hashes=guards,cache_source_hashes=m['source_hashes'],seeds=seeds.tolist(),source_rows=rows.tolist(),parent_rows=arrays['parent_rows.npy'].tolist(),actions=act.tolist(),selection=m['selection'],stationary_rows=stationary.nonzero().flatten().tolist(),scale=scale.tolist(),scale_source='H8 paired TRAIN current fields across64levels/148tokens; std floor.1; shared by both arms',event_positive_weights=pos.tolist(),event_rates=rates.tolist())
        for mode in ('h8','current_only'):
            torch.manual_seed(42);torch.cuda.manual_seed_all(42);model=StructuredTransition().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            field=torch.tensor(arrays[mode+'/fields.npy'],device='cuda').float();target=torch.tensor(arrays[mode+'/next_fields.npy'],device='cuda').float()
            arm={'curve':[],'completed_updates':0,'parameters':model.parameter_count()};r['arms'][mode]=arm
            def measure(step):
                model.eval()
                with torch.no_grad():
                    result=objective(model,field,target,actions,labels,scale,pos_weight=pos);pred=result['output']['field']
                    metrics=diagnostics(result,field,target,labels,scale)
                    raw={}
                    for subset,mask in [('all',torch.ones(64,device='cuda',dtype=torch.bool)),('stationary',stationary),('nonstationary',~stationary)]:
                        raw[subset]={'count':int(mask.sum())}
                        if mask.any():
                            for group,s,e in [('core',0,48),('semantic',48,85)]:
                                raw[subset][group+'_mse']=float((pred[mask,:,s:e]-target[mask,:,s:e]).square().mean())
                                raw[subset][group+'_copy_mse']=float((field[mask,:,s:e]-target[mask,:,s:e]).square().mean())
                            out=result['readouts']['predicted'];player=labels['next_player_cell'][:,1]*12+labels['next_player_cell'][:,0]
                            raw[subset]['player_accuracy']=float((out['player_logits'].argmax(-1)[mask]==player[mask]).float().mean())
                            hits=torch.stack([out['carried_'+name+'_logits'].argmax(-1)==labels['next_triple'][:,j] for j,name in enumerate(('shape','color','rotation'))],-1).all(-1)
                            raw[subset]['glyph_joint_accuracy']=float(hits[mask].float().mean())
                            raw[subset]['steps_accuracy']=float((out['steps_logits'].argmax(-1)[mask]==steps_targets(labels['next_steps'])[mask]).float().mean())
                    arm['curve'].append(dict(step=step,total=float(result['total']),losses={k:float(v) for k,v in result['losses'].items()},metrics=metrics,raw=raw))
                persist();print(json.dumps({'arm':mode,'step':step,'seconds':r['elapsed_seconds'],'raw':raw}),flush=True)
            measure(0);torch.cuda.reset_peak_memory_stats()
            for step in range(1,401):
                model.train();opt.zero_grad(set_to_none=True);result=objective(model,field,target,actions,labels,scale,pos_weight=pos)
                if not torch.isfinite(result['total']):raise ValueError('nonfinite objective')
                result['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step();arm['completed_updates']=step
                if step in (100,300,400):measure(step)
            arm['peak_allocated_bytes']=torch.cuda.max_memory_allocated();del result,model,opt,field,target;gc.collect();torch.cuda.empty_cache();persist()
        if any(digest(file)!=sha for file,sha in guards.items()):raise ValueError('sources changed')
        r.update(status='complete',source_unchanged=True)
    except BaseException as error:r.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:signal.alarm(0);persist()

if __name__=='__main__':main()
