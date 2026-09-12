"""Held-out chronological K4 fields, separate from one-step action-alternative caches."""
from pebby.ls20.provenance import metadata_difficulty_stages, cache_difficulty_metadata

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import tempfile
import time
import numpy as np
import torch
from pebby.agent.world_sequences import load_sidecar, sequence_targets
from pebby.agent.world_rollout import chronological_histories
from pebby.agent.world_train import load_dataset, as_tensors
from tools.build_structured_field_cache import digest
from tools.score_world_sequences import select_anchors

FORMAT='pebby.structured-sequence-cache.v1'
CURRENT={'player_cell':'player_cell','triple':'current_triple','steps':'current_steps','lives':'current_lives','optimal':'optimal'}
FUTURE={'next_player_cell':'player_cell','next_triple':'current_triple','next_steps':'current_steps','next_lives':'current_lives','next_optimal':'optimal'}


def sequence_batch(data,tensors,index,rows):
    """Source-indexed chronological targets and independently checked public H8 windows."""
    if index.meta.get('mode')!='explore_validation' or index.meta.get('split')!='validation':
        raise ValueError('only held-out exploratory K4 indices accepted')
    rows=np.asarray(rows,dtype=np.int64);future=index.lookup(rows)
    if not np.array_equal(future,rows[:,None]+np.arange(1,5)):
        raise ValueError('index must identify four consecutive source rows')
    for name in ('frames','history_valid','previous_actions',*CURRENT.values(),*FUTURE.values(),'next_player_cell'):
        if data.get(name) is None:raise ValueError(f'necessary source array missing: {name}')
    if not np.all(data['seeds'][future]==data['seeds'][rows,None]):
        raise ValueError('sequence crosses level identity')
    target=sequence_targets(tensors,index,torch.from_numpy(rows.copy()))
    current=tuple(tensors[key][rows] for key in ('frames','history_valid','previous_actions'))
    histories=chronological_histories(current[0],target['next_frames'][:,:,None],current[1],current[2],target['rollout_actions'])
    for computed,key in zip(histories,('frames','history_valid','previous_actions')):
        if not torch.equal(computed,tensors[key][future]):raise ValueError(f'chronological {key} do not match actual source histories')
    branch_rows=np.concatenate((rows[:,None],future[:,:-1]),axis=1)
    actions=target['rollout_actions'].numpy()
    for name in ('terminal','won','lost_life'):
        actual=np.asarray(data[name])[branch_rows,actions]
        if actual.any():raise ValueError('held-out live K4 index cannot include terminal/life reset')
        if not np.array_equal(target[name].numpy(),actual):raise ValueError('chronological event target mismatch')
    labels={out:np.array(data[source][rows],copy=True) for out,source in CURRENT.items()}
    for out,source in FUTURE.items():
        value=np.array(data[source][future],copy=True)
        if not np.array_equal(target[out].numpy(),value):raise ValueError('chronological teacher target mismatch')
        labels[out]=value
    labels.update({key:target[key].numpy().copy() for key in ('distances','terminal','won','lost_life')})
    labels['actions']=actions.copy()
    return current,histories,labels,future


def encode_sequence(assembler,data,tensors,index,rows):
    current,histories,labels,future=sequence_batch(data,tensors,index,rows)
    with torch.inference_mode():
        field=assembler(*current)
        target=assembler(*(x.flatten(0,1) for x in histories))
    n=len(rows)
    if field.shape!=(n,148,96) or target.shape!=(n*4,148,96):raise ValueError('field shape must be148x96')
    outputs=[];error=0.
    for item in (field,target.reshape(n,4,148,96)):
        f=item.float().cpu().numpy();q=f.astype(np.float16)
        if not np.isfinite(f).all() or not np.isfinite(q).all():raise ValueError('nonfinite field or float16 overflow')
        error=max(error,float(np.abs(f-q.astype(np.float32)).max()));outputs.append(q)
    return outputs[0],outputs[1],labels,future,error


def write_cache(data,index,source,index_path,out,assembler,provenance,levels=8,seed=42,batch_size=8):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    hashes={str(p):digest(p) for p in (Path(source),Path(index_path),Path(__file__))}
    if index.meta.get('source_sha256')!=hashes[str(Path(source))]:raise ValueError('index source checksum mismatch')
    rows=select_anchors(data,index,levels,seed)
    if len(np.unique(data['seeds'][rows]))!=levels:raise ValueError('one anchor per distinct validation level required')
    tensors=as_tensors(data);out.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=f'.{out.name}-',dir=out.parent));start=time.monotonic()
    try:
        fields=np.lib.format.open_memmap(temporary/'fields.npy',mode='w+',dtype=np.float16,shape=(levels,148,96))
        next_fields=np.lib.format.open_memmap(temporary/'next_fields.npy',mode='w+',dtype=np.float16,shape=(levels,4,148,96))
        collected=[];future_rows=[];encoding_seconds=0.;quantization=0.
        for begin in range(0,levels,batch_size):
            selected=rows[begin:begin+batch_size];t=time.monotonic()
            current,targets,labels,future,error=encode_sequence(assembler,data,tensors,index,selected)
            encoding_seconds+=time.monotonic()-t;quantization=max(error,quantization)
            fields[begin:begin+len(selected)]=current;next_fields[begin:begin+len(selected)]=targets
            collected.append(labels);future_rows.append(future)
        fields.flush();next_fields.flush();del fields,next_fields
        arrays={key:np.concatenate([batch[key] for batch in collected]) for key in collected[0]}
        difficulty={int(level['seed']):int(level['difficulty']) for level in data['meta']['levels']}
        arrays.update(seeds=np.array(data['seeds'][rows],copy=True),source_rows=rows,
                      future_rows=np.concatenate(future_rows),
                      difficulties=np.array([difficulty[int(s)] for s in data['seeds'][rows]],np.int8))
        for key,value in arrays.items():np.save(temporary/f'{key}.npy',value,allow_pickle=False)
        inventory={}
        for path in sorted(temporary.glob('*.npy')):
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            inventory[path.stem]={'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest(path)}
        if any(digest(path)!=sha for path,sha in hashes.items()):raise ValueError('source/index/builder changed')
        manifest={**cache_difficulty_metadata(data['meta'], arrays['seeds']), 'format':FORMAT,'status':'complete','source':'generated_only','split':'validation','mode':'heldout_chronological_K4',
                  'levels':levels,'source_hashes':hashes,'source_index_metadata':index.meta,'field_encoder':provenance,
                  'arrays':inventory,'selection_seed':seed,'difficulty_counts':{str(d):int((arrays['difficulties']==d).sum()) for d in metadata_difficulty_stages(data['meta'])},
                  'chronological_actions':'actions[N,4] from actual future_rows producing actions; values0..3. Never action alternatives.',
                  'history_verified_against_actual_source_rows':True,
                  'exact_label_mapping':{'current':CURRENT,'future':FUTURE,'distances':'exact future current-state distance via complete optimal branch distances+1'},
                  'event_coverage':{'transitions':levels*4,'terminal':0,'won':0,'lost_life':0,
                                    'unreachable':int((arrays['distances']<0).sum())},
                  'target_kind':'Frozen public encoder semantic distillation, separately scored against exact current/future state labels.',
                  'no_future_inputs_to_current_field':True,'encoding_seconds':encoding_seconds,
                  'float16_max_absolute_error':quantization,'elapsed_seconds':time.monotonic()-start,
                  'limitations':['All four transitions are live/reachable with unchanged lives; this index excludes reset, terminal and source/expert boundaries.',
                                 'Actual future histories only build targets. Future fields/flags/labels must not enter autonomous transition inputs.',
                                 'Generated validation only; not a training bank, rollout completion result, search or control result.']}
        (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');temporary.rename(out)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/ls20-world-combined-validation.npz'))
    p.add_argument('--index',type=Path,default=Path('data/ls20-world-combined-validation-k4.npz'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--levels',type=int,default=8)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--world',type=Path,default=Path('checkpoints/ls20-world-cell-recall-b1024.pt'))
    p.add_argument('--visibility',type=Path,default=Path('checkpoints/ls20-cell-visibility-initial-200.pt'))
    args=p.parse_args();torch.set_num_threads(1);start=time.monotonic();print('PID',os.getpid(),flush=True)
    def timeout(*_):raise TimeoutError('held-out H4 cache60second deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
    if args.batch_size<1:raise ValueError('batch size must be positive')
    from pebby.agent.structured_field import load_structured_field_encoder
    assembler=load_structured_field_encoder(args.world,args.visibility,device='cpu')
    provenance=assembler.metadata()
    source_paths=[Path('pebby/agent')/name for name in ('structured_field.py','world_model.py','world_readout.py','world_grounding.py','world_rollout.py','cell_appearance.py','cell_appearance_dense.py','glyph_model.py','cell_visibility.py')]
    provenance['code_hashes']={str(path):digest(path) for path in source_paths}
    provenance['checkpoint_hashes']={str(path):digest(path) for path in (args.world,args.visibility)}
    guards={**provenance['code_hashes'],**provenance['checkpoint_hashes'],
            'pebby/agent/world_sequences.py':digest('pebby/agent/world_sequences.py')}
    data=load_dataset(args.source,history=8,cache_dir=Path('data/world-array-cache'))
    index=load_sidecar(args.index,args.source,data,mode='explore_validation')
    manifest=write_cache(data,index,args.source,args.index,args.out,assembler,provenance,args.levels,args.seed,args.batch_size)
    if any(digest(path)!=sha for path,sha in guards.items()):raise ValueError('encoder/index helper changed during cache build')
    report={'status':'complete','pid':os.getpid(),'cpu_threads':1,'levels':args.levels,
            'manifest_sha256':digest(args.out/'manifest.json'),'elapsed_seconds':time.monotonic()-start,
            'encoding_seconds':manifest['encoding_seconds'],'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'source_hashes':guards,'event_coverage':manifest['event_coverage']}
    (args.out/'build-report.json').write_text(json.dumps(report,indent=2)+'\n');signal.alarm(0);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
