"""Retain every verified TRAIN trajectory/anchor row in a distinct field cache.

Four targets remain action alternatives. Repeated seeds are intentional; the
level index supports distinct-level sampling followed by a within-level draw.
Imagined targets are computed in FP32 from stored-current FP16 fields promoted
to FP32, matching existing cached policy training, then stored in FP16.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

import numpy as np
import torch

from pebby.agent.world_data import FORMAT as WORLD_FORMAT
from pebby.agent.world_train import load_dataset, require_verified_data
from pebby.agent.structured_distance import current_distance_labels
from pebby.agent.structured_transition import steps_targets
from tools.build_structured_field_cache import LABELS, digest, encode_fields, exact_labels, event_counts
from tools.train_structured_policy import imagined_fields

FORMAT = 'pebby.structured-onpolicy-field-cache.v1'
PUBLIC = ('frames', 'history_valid', 'previous_actions', 'next_frames', 'lost_life')


def check_hashes(hashes):
    for path, expected in hashes.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f'source changed or missing: {path}')


def fingerprints(metadata):
    result = {}
    def visit(value):
        if isinstance(value, dict):
            for key in ('path', 'checkpoint'):
                if isinstance(value.get(key), str) and isinstance(value.get('sha256'), str):
                    result[value[key]] = value['sha256']
            for key in ('bank', 'proof'):
                if isinstance(value.get(key), str) and isinstance(value.get(key+'_sha256'), str):
                    result[value[key]] = value[key+'_sha256']
            for key in ('code_hashes', 'source_hashes'):
                if isinstance(value.get(key), dict):
                    result.update({p:h for p,h in value[key].items() if isinstance(h,str)})
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(metadata)
    return result


def row_metadata(data):
    meta = data['meta']; n = len(data['frames'])
    if n<1:raise ValueError('nonempty source required')
    if (meta.get('format') != WORLD_FORMAT or meta.get('source') != 'generated_only'
            or meta.get('history') != 8 or meta.get('alternatives_per_state') != 4):
        raise ValueError('standard generated H8/four-alternative source required')
    if meta.get('split', 'train') != 'train' or meta.get('official_inputs_used', False):
        raise ValueError('TRAIN generated source only')
    require_verified_data(data)
    for name,shape in {'next_frames':(n,4,64,64),'history_valid':(n,8),'previous_actions':(n,8),
                       'terminal':(n,4),'won':(n,4),'lost_life':(n,4)}.items():
        if data.get(name) is None or data[name].shape!=shape:raise ValueError(f'bad shape: {name}')
    if data['frames'].shape != (n,8,64,64) or data['frames'].dtype != np.uint8:
        raise ValueError('current frames must uint8[N,8,64,64]')
    if data['next_frames'].dtype != np.uint8 or np.any(data['frames'] > 15) or np.any(data['next_frames'] > 15):
        raise ValueError('public frames must palette uint8')
    for key in ('history_valid','terminal','won','lost_life'):
        if data.get(key) is None or data[key].dtype != np.bool_:
            raise ValueError(f'boolean array required: {key}')
    if not np.issubdtype(data['previous_actions'].dtype,np.integer):
        raise ValueError('integer previous actions required')
    valid,previous=data['history_valid'],data['previous_actions']
    if (not valid[:,-1].all() or np.any(valid[:,:-1]&~valid[:,1:])
            or np.any((previous < -1)|(previous > 3)) or np.any(previous[~valid]!=-1)):
        raise ValueError('invalid causal history padding/actions')
    for name, shape in {'player_cell':(n,2),'next_player_cell':(n,4,2),'current_triple':(n,3),
            'next_triple':(n,4,3),'current_steps':(n,),'next_steps':(n,4),'current_lives':(n,),
            'next_lives':(n,4),'optimal':(n,),'next_optimal':(n,4),'distances':(n,4),'seeds':(n,)}.items():
        value=data.get(name)
        if value is None or value.shape != shape or not np.issubdtype(value.dtype,np.integer):
            raise ValueError(f'exact integer label required: {name} {shape}')
    if np.any((data['optimal'] < 0) | (data['optimal'] > 15)):
        raise ValueError('current optimal masks must be0..15')
    distance=current_distance_labels(*(data[k] for k in ('optimal','distances','terminal','won','lost_life')), allow_unreachable=True).numpy()
    for name in ('player_cell','next_player_cell'):
        if np.any((data[name]<0)|(data[name]>=12)):raise ValueError('player cell outside12x12')
    for name in ('current_triple','next_triple'):
        if np.any((data[name]<0)|(data[name]>=np.array([6,4,4]))):raise ValueError('triple outside6/4/4')
    for name in ('current_steps','next_steps'):steps_targets(torch.as_tensor(data[name]))
    lives,next_lives=data['current_lives'],data['next_lives'];lost=data['lost_life'];terminal=data['terminal'];won=data['won']
    if (np.any((lives<1)|(lives>3)) or np.any((next_lives<0)|(next_lives>3))
            or np.any(next_lives != lives[:,None]-lost.astype(np.int16))
            or np.any((terminal&~won)!=(lost&(next_lives==0))) or np.any(won&lost)):
        raise ValueError('inconsistent life decrement/terminal labels')
    masks=data['next_optimal'];reachable=(data['distances']>0)&~terminal
    if np.any((masks<0)|(masks>15)) or np.any(reachable!=(masks!=0)):
        raise ValueError('successor optimal/reachability mismatch')
    if np.any((data['seeds'] < 0) | (data['seeds'] >= 1000000)):
        raise ValueError('TRAIN seed namespace required')
    marked = meta.get('on_policy_rows')
    if (not isinstance(marked,list) or not marked or any(type(x) is not int for x in marked)
            or len(set(marked)) != len(marked) or min(marked)<0 or max(marked)>=n):
        raise ValueError('unique in-range on_policy_rows indices required')
    on_policy=np.zeros(n,dtype=bool);on_policy[marked]=True
    levels={int(x['seed']):x for x in meta['levels']}
    difficulties=np.asarray([levels[int(s)]['difficulty'] for s in data['seeds']],dtype=np.int8)
    if np.any((difficulties<1)|(difficulties>5)):raise ValueError('difficulty outside1..5')
    seeds, counts=np.unique(data['seeds'],return_counts=True)
    result=dict(source_rows=np.arange(n,dtype=np.int64),seeds=np.array(data['seeds'],copy=True),
        difficulties=difficulties,on_policy=on_policy,level_seeds=seeds.astype(np.int64),
        level_offsets=np.r_[0,np.cumsum(counts)].astype(np.int64),
        level_rows=np.argsort(data['seeds'],kind='stable').astype(np.int64),current_distance=distance)
    if data.get('context_index') is not None:
        context=data['context_index'];expected=np.array([levels[int(s)]['context_index'] for s in data['seeds']])
        if (context.shape!=(n,) or not np.issubdtype(context.dtype,np.integer)
                or np.any((context<0)|(context>6)) or not np.array_equal(context,expected)):
            raise ValueError('context_index must match exact per-level proof')
        result['context_index']=np.array(context,copy=True)
    return result


def load_source(source, report, checkpoint):
    source, report, checkpoint=map(Path,(source,report,checkpoint))
    hashes={str(p.resolve()):digest(p) for p in (source,report,checkpoint)}
    record=json.loads(report.read_text())
    if (record.get('status')!='complete' or record.get('output_sha256')!=digest(source)
            or record.get('source_unchanged') is not True):
        raise ValueError('completed unchanged-source collector report required')
    bound=record.get('source_hashes')
    if not isinstance(bound,dict) or not bound:raise ValueError('collector source hashes required')
    hashes.update(bound);check_hashes(hashes)
    data=load_dataset(source)
    # world_train's fixed optional inventory omits this collector diagnostic.
    with np.load(source,allow_pickle=False) as archive:
        if 'context_index' in archive.files:data['context_index']=archive['context_index']
    row_metadata(data)
    behavior=data['meta'].get('behavior_checkpoint',{})
    if (behavior.get('sha256')!=hashes[str(checkpoint.resolve())]
            or Path(behavior.get('path','')).resolve()!=checkpoint.resolve()):
        raise ValueError('source behavior checkpoint mismatch')
    check_hashes(hashes)
    return data,hashes


def build(data,out,policy,hashes,*,device='cpu',batch_size=32,max_encoder_batch=128):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    if not 1<=batch_size<=128 or not 1<=max_encoder_batch<=128:
        raise ValueError('bounded row/encoder batches1..128 required')
    if getattr(policy,'encoder',None) is None or getattr(policy,'dynamics',None) is None:
        raise ValueError('public structured actor with frozen encoder/dynamics required')
    index=row_metadata(data);n=len(index['seeds']);rows=index['source_rows']
    labels=exact_labels(data,rows)
    hashes=dict(hashes)
    for path in (__file__,'tools/build_structured_field_cache.py','tools/train_structured_policy.py',
                 'pebby/agent/world_train.py','pebby/agent/model.py','pebby/agent/structured_workspace_controller.py',
                 'pebby/agent/structured_workspace_policy.py','pebby/agent/structured_distance.py',
                 'pebby/agent/structured_transition.py'):
        hashes[str(Path(path).resolve())]=digest(path)
    hashes.update(fingerprints(policy.sources));hashes.update(fingerprints(policy.encoder.metadata()))
    check_hashes(hashes)
    policy.eval().requires_grad_(False)
    policy.encoder.to(device);policy.dynamics.to(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+out.name+'-',dir=out.parent));started=time.monotonic()
    try:
        fields=np.lib.format.open_memmap(temporary/'fields.npy',mode='w+',dtype=np.float16,shape=(n,148,96))
        next_fields=np.lib.format.open_memmap(temporary/'next_fields.npy',mode='w+',dtype=np.float16,shape=(n,4,148,96))
        imagined=np.lib.format.open_memmap(temporary/'imagined_fields.npy',mode='w+',dtype=np.float16,shape=(n,4,148,96))
        actual_error=imagined_error=0.
        with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            for first in range(0,n,batch_size):
                end=min(first+batch_size,n)
                current,future,error=encode_fields(policy.encoder,{k:data[k][first:end] for k in PUBLIC},max_encoder_batch)
                actual_error=max(actual_error,error)
                predictions=imagined_fields(policy.dynamics,torch.as_tensor(current,device=device).float(),max_encoder_batch)
                if predictions.dtype!=torch.float32:raise ValueError('predictions must compute FP32')
                predicted=predictions.cpu().numpy();quantized=predicted.astype(np.float16)
                if not np.isfinite(quantized).all():raise ValueError('imagined FP16 overflow')
                imagined_error=max(imagined_error,float(np.abs(predicted-quantized.astype(np.float32)).max()))
                fields[first:end]=current;next_fields[first:end]=future;imagined[first:end]=quantized
        for array in (fields,next_fields,imagined):array.flush()
        del fields,next_fields,imagined
        for name,value in {**labels,**index}.items():np.save(temporary/f'{name}.npy',value,allow_pickle=False)
        inventory={}
        for path in sorted(temporary.glob('*.npy')):
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            for first in range(0,len(value),32):
                if not np.isfinite(value[first:first+32]).all():raise ValueError('nonfinite saved array')
            inventory[path.stem]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':digest(path)}
        check_hashes(hashes)
        manifest={'format':FORMAT,'status':'complete','source':'generated_only','split':'train',
            'rows':n,'levels':len(index['level_seeds']),'on_policy_rows':int(index['on_policy'].sum()),
            'expert_rows':int((~index['on_policy']).sum())-sum(level.get('failure_samples',0) for level in data['meta']['levels']),
            'failure_rows':sum(level.get('failure_samples',0) for level in data['meta']['levels']),
            'policy_unlabelled_rows':int((data['optimal']==0).sum()),'arrays':inventory,'source_hashes':hashes,
            'field_encoder':policy.encoder.metadata(),'actor_sources':policy.sources,
            'source_metadata':data['meta'],'event_coverage':event_counts(labels),
            'grouping':'sorted level_seeds; level_rows[level_offsets[g]:level_offsets[g+1]] gives stable source-order row IDs',
            'precision':{'compute_dtype':'float32','storage_dtype':'float16','autocast':False,
                'matmul_tf32':False,'cudnn_tf32':False,'actual_quantization_max_error':actual_error,
                'imagined_quantization_max_error':imagined_error,'imagined_input':'stored current FP16 promoted to FP32'},
            'history_contract':'stored causal H8 current; four independent action alternatives; loss targets reset to actual post-loss frame, last slot valid/actions-1',
            'supervision_contract':'exact labels copied unchanged; only public histories enter encoder; current fields+action IDs enter dynamics',
            'no_future_inputs_to_current_or_imagined_fields':True,'device':device,'row_batch':batch_size,
            'max_encoder_batch':max_encoder_batch,'elapsed_seconds':time.monotonic()-started}
        (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.rename(out)
        return manifest
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','report','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-encoder-batch',type=int,default=128)
    p.add_argument('--seconds',type=int,default=600);args=p.parse_args(argv)
    if not 1<=args.seconds<=600:p.error('seconds must1..600')
    if args.out.exists():p.error('refusing existing output')
    torch.set_num_threads(1);print('PID',os.getpid(),flush=True)
    def expired(*_):raise TimeoutError('on-policy cache deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    try:
        from pebby.agent.model import load_checkpoint
        data,hashes=load_source(args.source,args.report,args.checkpoint)
        policy,_=load_checkpoint(args.checkpoint,device='cpu')
        check_hashes(hashes)
        result=build(data,args.out,policy,hashes,device=args.device,batch_size=args.batch_size,max_encoder_batch=args.max_encoder_batch)
        print(json.dumps({'status':'complete','rows':result['rows'],'levels':result['levels']}),flush=True)
    finally:signal.alarm(0)
    return 0


if __name__=='__main__':main()
