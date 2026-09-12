"""Frozen HUD budget decodability, generated source-bound fields only."""
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pebby.agent.structured_transition import steps_targets
from tools.build_structured_field_cache import digest

ROOT=Path('data/structured-field-pilot-2048')
OUT=Path('artifacts/structured-budget-frozen-hud-probe.json')


def read_split(split, count):
    directory=ROOT/split;manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='complete' or manifest.get('source')!='generated_only' or manifest.get('split')!=split:
        raise ValueError('requires complete generated split')
    config=manifest['field_encoder']['config']
    if (config['tokens'],config['hud_tokens'],config['channels'],config['state_channels'])!=(148,4,96,48):
        raise ValueError('unexpected frozen HUD layout')
    arrays={};hashes={str(manifest_path):digest(manifest_path)}
    for name in ('fields','next_fields','steps','next_steps','seeds','difficulties','source_rows'):
        path=directory/f'{name}.npy';value=digest(path)
        if value!=manifest['arrays'][name]['sha256']:raise ValueError('array checksum mismatch')
        hashes[str(path)]=value;arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    if arrays['fields'].shape!=(count,148,96) or arrays['next_fields'].shape!=(count,4,148,96):
        raise ValueError('field shape mismatch')
    seeds=np.array(arrays['seeds']);lo,hi=(0,1_000_000) if split=='train' else (1_000_000,2_000_000)
    if len(set(map(int,seeds)))!=count or not ((seeds>=lo)&(seeds<hi)).all():raise ValueError('distinct generated split seeds required')
    # Copy only192HUD features per view, never the large full field arrays.
    current=np.array(arrays['fields'][:,-4:,:48],dtype=np.float32).reshape(count,192)
    actual=np.array(arrays['next_fields'][:,:,-4:,:48],dtype=np.float32).reshape(count,4,192)
    features=torch.from_numpy(np.concatenate((current[:,None],actual),axis=1))
    raw=torch.from_numpy(np.concatenate((np.array(arrays['steps'])[:,None],np.array(arrays['next_steps'])),axis=1))
    labels=steps_targets(raw)
    if not bool(torch.isfinite(features).all()):raise ValueError('nonfinite frozen HUD features')
    return features,labels,raw,seeds,hashes,manifest


@torch.no_grad()
def evaluate(model, features, labels, majority, prior):
    logits=torch.cat([model(batch) for batch in features.flatten(0,1).split(1024)]).reshape(len(features),5,44)
    result={}
    for category,columns in [('current',slice(0,1)),('all4_actual',slice(1,5)),('all5_views',slice(0,5))]:
        target=labels[:,columns].flatten();pred=logits[:,columns].reshape(-1,44)
        counts=torch.bincount(target,minlength=44)
        result[category]={'examples':len(target),'correct':int(pred.argmax(-1).eq(target).sum()),
                          'accuracy':float(pred.argmax(-1).eq(target).float().mean()),
                          'ce':float(F.cross_entropy(pred,target)),
                          'training_majority_accuracy':float(target.eq(majority).float().mean()),
                          'training_smoothed_classprior_ce':float(-prior[target].log().mean()),
                          'target_class_counts':counts.tolist(),
                          'underflow_examples':int(target.eq(43).sum()),
                          'underflow_correct':int((pred.argmax(-1).eq(target)&target.eq(43)).sum())}
    return result


def main():
    start=time.monotonic();print('PID',os.getpid(),flush=True);torch.set_num_threads(1)
    def timeout(*_):raise TimeoutError('frozen budget probe120second deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    if OUT.exists():raise FileExistsError(OUT)
    train,train_labels,train_raw,train_seeds,hashes,train_meta=read_split('train',2048)
    val,val_labels,val_raw,val_seeds,val_hashes,val_meta=read_split('validation',512);hashes.update(val_hashes)
    if set(map(int,train_seeds))&set(map(int,val_seeds)):raise ValueError('train/validation seed overlap')
    if train_meta['field_encoder']!=val_meta['field_encoder']:raise ValueError('different frozen field encoders')
    for path in [Path(__file__),Path('pebby/agent/structured_transition.py'),Path('tools/build_structured_field_cache.py')]:hashes[str(path)]=digest(path)
    flat=train.flatten(0,1);variance=flat.var(0,unbiased=False)
    variance_info={'features':192,'all_finite':True,'mean':float(variance.mean()),'minimum':float(variance.min()),
                   'maximum':float(variance.max()),'below_1e8':int((variance<1e-8).sum()),
                   'per_hud_token_mean':variance.reshape(4,48).mean(1).tolist()}
    if not bool((variance>1e-8).any()):raise ValueError('HUD features have no variation')
    mean=flat.mean(0);scale=variance.sqrt().clamp_min(1e-6)
    train=(train-mean)/scale;val=(val-mean)/scale
    torch.manual_seed(42);generator=torch.Generator().manual_seed(42)
    model=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,44))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    seen=torch.zeros(2048,dtype=torch.bool);view_counts=torch.zeros(5,dtype=torch.long);curve=[];max_grad_norm=0.
    print(json.dumps({'variance':variance_info,'parameters':sum(p.numel() for p in model.parameters()),'loaded_seconds':time.monotonic()-start}),flush=True)
    for step in range(200):
        levels=torch.randperm(2048,generator=generator)[:1024];slots=torch.randint(5,(1024,),generator=generator)
        assert len(levels.unique())==1024
        seen[levels]=True;view_counts+=torch.bincount(slots,minlength=5)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(train[levels,slots]),train_labels[levels,slots]);loss.backward()
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('nonfinite probe loss or gradient')
        grad_norm=float(torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt())
        max_grad_norm=max(max_grad_norm,grad_norm);optimizer.step()
        if step==0 or (step+1)%50==0:
            row={'updates':step+1,'train_minibatch_ce':float(loss.detach()),'elapsed_seconds':time.monotonic()-start}
            curve.append(row);print(json.dumps(row),flush=True)
    counts=torch.bincount(train_labels.flatten(),minlength=44);majority=int(counts.argmax())
    prior=(counts.float()+1)/(counts.sum()+44)
    model.eval();results={'train':evaluate(model,train,train_labels,majority,prior),
                          'validation':evaluate(model,val,val_labels,majority,prior)}
    if any(digest(path)!=sha for path,sha in hashes.items()):raise ValueError('probe source changed')
    report={'status':'complete','source_hashes':hashes,'field_encoder':train_meta['field_encoder'],
            'architecture':'Linear192to128,GELU,Linear128to44','parameters':sum(p.numel() for p in model.parameters()),
            'features':'concatenate four frozen HUD tokens, state channels0:48 only;192features',
            'feature_variance_before_fit':variance_info,'normalization':'mean/std from ALL5TRAINviews only; standard deviation floor1e-6',
            'label_contract':'exact budgets0..42; class43 groups actual negative budgets−3,−2,−1; zero remains class0',
            'training':{'optimizer':'Adam','learning_rate':.001,'updates':200,'seed':42,'batch_size':1024,
                        'distinct_base_levels_every_update':True,'uniform_random_view':'currentoroneof4actualsuccessors',
                        'view_draw_counts_current_then_actions0to3':view_counts.tolist(),'distinct_train_levels_seen':int(seen.sum()),
                        'finite_gradients_all_updates':True,'maximum_gradient_norm':max_grad_norm},
            'results':results,'training_curve':curve,'training_majority_class':majority,
            'train_levels':2048,'validation_levels':512,'split_seed_overlap':0,
            'negative_raw_steps_counts':{'train':{str(v):int(train_raw.eq(v).sum()) for v in [-3,-2,-1]},
                                         'validation':{str(v):int(val_raw.eq(v).sum()) for v in [-3,-2,-1]}},
            'weights_saved':False,'encoder_updated':False,'validation_used_for_training_or_selection':False,
            'pid':os.getpid(),'cpu_threads':1,'device':'cpu','elapsed_seconds':time.monotonic()-start,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'limits':['Frozen-feature decodability, not transition fidelity or controller completion.',
                      'Cache deliberately enriches life-loss levels; classpriors/accuracy are conditional on this sample.',
                      'Four successor views of each level are correlated; validation levels are disjoint from training.',
                      'A positive result does not prove the current structured global-pool readout can learn or uses these features.',
                      'No glyph/role probabilities, supplied budgets, player coordinates, official data or engine state enter the classifier.']}
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');signal.alarm(0)
    print(json.dumps({split:{k:{metric:v[metric] for metric in ['accuracy','ce','training_majority_accuracy']} for k,v in rows.items()} for split,rows in results.items()}),flush=True)


if __name__=='__main__':main()
