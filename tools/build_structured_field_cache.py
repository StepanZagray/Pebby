"""Separate generated field caches: learned public fields and exact teacher labels.

Actual branch targets use H8 append/shift or life-reset histories. No labels
enter current-field assembly. One current row per distinct generated level.
"""
from pebby.ls20.provenance import metadata_difficulty_stages, cache_difficulty_metadata

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import resource
import signal
import tempfile
import time
import numpy as np
import torch

FORMAT = 'pebby.structured-field-cache.v1'
LABELS = {'player_cell':'player_cell','next_player_cell':'next_player_cell',
          'triple':'current_triple','next_triple':'next_triple','steps':'current_steps',
          'next_steps':'next_steps','lives':'current_lives','next_lives':'next_lives',
          'lost_life':'lost_life','terminal':'terminal','won':'won','optimal':'optimal',
          'next_optimal':'next_optimal','distances':'distances'}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def actual_histories(batch):
    """Return independent four-branch histories [B,4,H,...], action order0..3."""
    frames=torch.as_tensor(batch['frames']);valid=torch.as_tensor(batch['history_valid'])
    previous=torch.as_tensor(batch['previous_actions']);future=torch.as_tensor(batch['next_frames'])
    lost=torch.as_tensor(batch['lost_life'])
    b,h=frames.shape[:2]
    if frames.shape != (b,h,64,64) or h != 8 or future.shape != (b,4,64,64):
        raise ValueError('expected current H8 and four actual frames')
    if valid.shape!=(b,h) or valid.dtype!=torch.bool or previous.shape!=(b,h):
        raise ValueError('invalid public history metadata')
    if lost.shape!=(b,4) or lost.dtype!=torch.bool:
        raise ValueError('actual lost_life labels must be boolean[B,4]')
    if not bool(valid[:,-1].all()) or bool(((previous < -1)|(previous>3)).any()):
        raise ValueError('invalid current history/actions')
    histories=torch.cat((frames[:,None,1:].expand(-1,4,-1,-1,-1),future[:,:,None]),2)
    validity=torch.cat((valid[:,None,1:].expand(-1,4,-1),torch.ones((b,4,1),dtype=torch.bool)),2)
    actions=torch.cat((previous[:,None,1:].expand(-1,4,-1),torch.arange(4,dtype=previous.dtype)[None,:,None].expand(b,-1,-1)),2)
    reset_valid=torch.zeros_like(validity);reset_valid[:,:,-1]=True
    return (torch.where(lost[:,:,None,None,None],future[:,:,None].expand(-1,-1,h,-1,-1),histories),
            torch.where(lost[:,:,None],reset_valid,validity),
            torch.where(lost[:,:,None],-torch.ones_like(actions),actions))


def select_rows(data, count, seed, split, include_life_loss_levels=False):
    meta=data['meta'];seeds=np.asarray(data['seeds'])
    if meta.get('source')!='generated_only' or meta.get('oracle_search')!='complete_only' or meta.get('split')!=split:
        raise ValueError('requires complete generated source in requested split')
    lo,hi=(0,1_000_000) if split=='train' else (1_000_000,2_000_000)
    if not ((seeds>=lo)&(seeds<hi)).all():raise ValueError('generated split namespace mismatch')
    if type(count)!=int or count<1:raise ValueError('level count must be positive')
    for output,source in LABELS.items():
        if data.get(source) is None:raise ValueError(f'exact source label missing: {source}')
    difficulty={int(item['seed']):int(item['difficulty']) for item in meta['levels']}
    available=np.unique(seeds);rng=np.random.default_rng(seed);selected=[]
    loss_rows=np.asarray(data['lost_life']).any(1)
    loss_levels=set(map(int,seeds[loss_rows])) if include_life_loss_levels else set()
    stages = metadata_difficulty_stages(data['meta'])
    for tier in stages:
        quota=count//len(stages)+int(tier<=count%len(stages))
        pool=[int(s) for s in available if difficulty[int(s)]==tier]
        if len(pool)<quota:raise ValueError(f'insufficient distinct levels at difficulty{tier}')
        forced=sorted(set(pool)&loss_levels)
        if len(forced)>quota:raise ValueError(f'life-loss levels exceed difficulty{tier} quota; increase level count')
        remaining=[s for s in pool if s not in loss_levels]
        selected.extend(forced+list(map(int,rng.choice(remaining,quota-len(forced),replace=False))))
    # Preserve ascending source-row order and every RNG draw from the original
    # repeated-flatnonzero implementation, but group eligible rows in one pass.
    grouped={s:[] for s in selected}
    for row,value in enumerate(seeds):
        level=int(value)
        if level in grouped and (level not in loss_levels or loss_rows[row]):
            grouped[level].append(row)
    rows=np.array([rng.choice(grouped[s]) for s in selected],dtype=np.int64)
    return rows,np.array([difficulty[s] for s in selected],dtype=np.int8)


def exact_labels(data, rows):
    result={}
    for output,source in LABELS.items():
        if data.get(source) is None:raise ValueError(f'exact source label missing: {source}')
        result[output]=np.array(data[source][rows],copy=True)
    return result


def encode_fields(assembler,batch,max_encoder_batch=None):
    """Only public histories reach assembler; teacher flags affect targets only."""
    frames=torch.as_tensor(batch['frames']);valid=torch.as_tensor(batch['history_valid'])
    previous=torch.as_tensor(batch['previous_actions'])
    if max_encoder_batch is not None and (type(max_encoder_batch)!=int or not 1<=max_encoder_batch<=1024):
        raise ValueError('max encoder batch must be1..1024')
    def encode(inputs):
        if max_encoder_batch is None or len(inputs[0])<=max_encoder_batch:
            return assembler(*inputs).cpu()
        return torch.cat([assembler(*(value[first:first+max_encoder_batch] for value in inputs)).cpu()
                          for first in range(0,len(inputs[0]),max_encoder_batch)])
    with torch.inference_mode():
        current=encode((frames,valid,previous))
        histories,validity,actions=actual_histories(batch)
        target=encode((histories.flatten(0,1),validity.flatten(0,1),actions.flatten(0,1)))
    b=len(frames)
    if current.shape!=(b,148,96) or target.shape!=(b*4,148,96):raise ValueError('field shape must be148x96')
    values=[];errors=[]
    for field in (current,target.reshape(b,4,148,96)):
        array=field.detach().cpu().float().numpy()
        quantized=array.astype(np.float16)
        if not np.isfinite(array).all() or not np.isfinite(quantized).all():raise ValueError('nonfinite field/float16 overflow')
        errors.append(float(np.max(np.abs(array-quantized.astype(np.float32)))))
        values.append(quantized)
    return *values,max(errors)


def event_counts(labels):
    lost=labels['lost_life'];terminal=labels['terminal'];won=labels['won'];distances=labels['distances']
    return {'rows':len(lost),'branches':int(lost.size),'won':int(won.sum()),'lost_life':int(lost.sum()),
            'terminal_loss':int((terminal&~won).sum()),'unreachable':int((distances<0).sum()),
            'unsafe_union':int(((distances<0)|lost|(terminal&~won)).sum()),
            'rows_with_win':int(won.any(1).sum()),'rows_with_life_loss':int(lost.any(1).sum())}


def write_split(data,source_path,out,assembler,count,seed,split,assembler_provenance,batch_size=8,include_life_loss_levels=False,
                max_encoder_batch=None,device='cpu'):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    source_hash=digest(source_path);start=time.monotonic()
    rows,difficulties=select_rows(data,count,seed,split,include_life_loss_levels);labels=exact_labels(data,rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=f'.{out.name}-',dir=out.parent))
    try:
        fields=np.lib.format.open_memmap(temporary/'fields.npy',mode='w+',dtype=np.float16,shape=(count,148,96))
        next_fields=np.lib.format.open_memmap(temporary/'next_fields.npy',mode='w+',dtype=np.float16,shape=(count,4,148,96))
        quantization=0.;encoding_seconds=0.
        for begin in range(0,count,batch_size):
            selected=rows[begin:begin+batch_size]
            batch={key:np.array(data[key][selected],copy=True) for key in ('frames','history_valid','previous_actions','next_frames','lost_life')}
            encoding_start=time.monotonic()
            current,future,error=encode_fields(assembler,batch,max_encoder_batch)
            encoding_seconds+=time.monotonic()-encoding_start
            fields[begin:begin+len(selected)]=current;next_fields[begin:begin+len(selected)]=future
            quantization=max(quantization,error)
            completed=begin+len(selected)
            if completed%256==0 or completed==count:
                print(json.dumps({'split':split,'rows_encoded':completed,'target_rows':count,
                                  'encoding_seconds':encoding_seconds}),flush=True)
        fields.flush();next_fields.flush();del fields,next_fields
        arrays={'seeds':np.array(data['seeds'][rows],copy=True),'difficulties':difficulties,'source_rows':rows,**labels}
        if data.get('context_index') is not None:arrays['context_index']=np.array(data['context_index'][rows],copy=True)
        for name,value in arrays.items():np.save(temporary/f'{name}.npy',value,allow_pickle=False)
        inventory={}
        for path in sorted(temporary.glob('*.npy')):
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            inventory[path.stem]={'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest(path)}
        if digest(source_path)!=source_hash:raise ValueError('source changed while caching')
        manifest={**cache_difficulty_metadata(data['meta'], arrays['seeds']), 'format':FORMAT,'status':'complete','split':split,'source':'generated_only',
                  'source_path':str(source_path),'source_sha256':source_hash,'builder_sha256':digest(__file__),
                  'field_encoder':assembler_provenance,'arrays':inventory,'selected_levels':count,
                  'encoding_precision':{'device':device,'compute_dtype':'float32','cache_dtype':'float16',
                                        'autocast':False,'tf32':False,'max_encoder_batch':max_encoder_batch},
                  'selection':{'seed':seed,'difficulty_counts':{str(d):int((difficulties==d).sum()) for d in metadata_difficulty_stages(data['meta'])},
                               'method':'include all life-loss levels within difficulty quotas and choose loss-containing row, remainder uniform' if include_life_loss_levels else 'uniform level within difficulty, then uniform source row per level',
                               'event_enrichment':include_life_loss_levels,
                               'available_life_loss_levels':len(set(map(int,np.asarray(data['seeds'])[np.asarray(data['lost_life']).any(1)]))),
                               'selected_life_loss_levels':int(labels['lost_life'].any(1).sum()),
                               'calibration_limit':'Event-enriched sampling changes priors; event metrics are conditional, not natural-prevalence calibration.' if include_life_loss_levels else 'Uniform one-row-per-level sampling may omit rare events.'},
                  'event_coverage':event_counts(labels),
                  'source_event_coverage':event_counts({target:data[source] for target,source in LABELS.items()}),
                  'encoding_seconds':encoding_seconds,'float16_max_absolute_quantization_error':quantization,
                  'field_contract':'learned public-input semantic distillation; fields/current and next_fields/actual targets are not exact world labels',
                  'supervision_contract':{'exact_label_source_mapping':LABELS,'negative_steps_preserved':True,
                                          'labels_not_assembler_inputs':True},
                  'history_contract':'H8; independent actions0..3 append actual frames; lost_life targets repeat resetframe H8, onlylastvalid, allprevious_actions-1; current untouched',
                  'elapsed_seconds':time.monotonic()-start}
        (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.rename(out)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,default=Path('data/ls20-world-combined-train.npz'))
    parser.add_argument('--validation',type=Path,default=Path('data/ls20-world-combined-validation.npz'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--train-levels',type=int,default=2048)
    parser.add_argument('--validation-levels',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--include-life-loss-levels',action='store_true',
                        help='Explicitly include all source life-loss levels within difficulty quotas and select a loss-containing row.')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--max-encoder-batch',type=int,default=128,
                        help='Maximum number of public histories per encoder call, including actual successor histories (1..1024).')
    parser.add_argument('--world',type=Path,required=True)
    parser.add_argument('--visibility',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if args.batch_size<1:raise ValueError('batch size must be positive')
    if not 1<=args.max_encoder_batch<=1024:raise ValueError('max encoder batch must be1..1024')
    print('PID',os.getpid(),flush=True);start=time.monotonic();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    def timeout(*_):raise TimeoutError('field cache builder600second budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    from pebby.agent.world_train import load_dataset
    from pebby.agent.structured_field import load_structured_field_encoder
    assembler=load_structured_field_encoder(args.world,args.visibility,device=args.device).eval()
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    provenance=assembler.metadata()
    encoder_paths=[Path('pebby/agent')/name for name in ('structured_field.py','world_model.py',
        'world_readout.py','world_grounding.py','world_rollout.py','cell_appearance.py',
        'cell_appearance_dense.py','glyph_model.py','cell_visibility.py')]
    provenance['code_hashes']={str(path):digest(path) for path in encoder_paths}
    provenance['checkpoint_hashes']={str(path):digest(path) for path in (args.world,args.visibility)}
    reports={};all_seeds={}
    for split,path,count in [('train',args.train,args.train_levels),('validation',args.validation,args.validation_levels)]:
        data=load_dataset(path,history=8,cache_dir=Path('data/world-array-cache'))
        reports[split]=write_split(data,path,args.out/split,assembler,count,args.seed,split,provenance,args.batch_size,args.include_life_loss_levels,
                                   args.max_encoder_batch,args.device)
        all_seeds[split]=set(map(int,np.load(args.out/split/'seeds.npy',allow_pickle=False)))
        del data
    assert not all_seeds['train']&all_seeds['validation']
    if any(digest(path)!=sha for path,sha in provenance['code_hashes'].items()):
        raise ValueError('field encoder code changed during build')
    if any(digest(path)!=sha for path,sha in provenance['checkpoint_hashes'].items()):
        raise ValueError('field encoder checkpoint changed during build')
    result={'status':'complete','format':FORMAT,'pid':os.getpid(),'cpu_threads':1,
            'device':args.device,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
            'split_overlap':0,'train':reports['train'],'validation':reports['validation'],
            'elapsed_seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'limits':'One source row per selected generated level, all4branches. Learned field distillation is not exact semantic ground truth. Event sampling method/enrichment and calibration limits are explicit in each split manifest.'}
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');signal.alarm(0)
    print(json.dumps({'status':'complete','elapsed_seconds':result['elapsed_seconds'],
                      'train_events':reports['train']['event_coverage'],'validation_events':reports['validation']['event_coverage']}),flush=True)


if __name__=='__main__':main()
