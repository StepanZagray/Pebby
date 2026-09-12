"""Frozen public initial-field sidecars aligned to immutable structured caches.

Initial histories come only from stored generated observations, never re-rendering.
The memory persists across life loss; fog-limited initial pixels are not an oracle.
"""
import argparse
import hashlib
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

FORMAT='pebby.structured-initial-cache.v1'
SOURCE_ARRAYS=('frames','history_valid','previous_actions','current_lives','seeds','meta','won','terminal')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _split_seeds(seeds,split):
    if split not in ('train','validation'):raise ValueError('split must be train or validation')
    low,high=(0,1000000) if split=='train' else (1000000,2000000)
    if seeds.ndim!=1 or not np.issubdtype(seeds.dtype,np.integer) or not ((seeds>=low)&(seeds<high)).all():
        raise ValueError('integer seed array violates requested split')


def select_initial_rows(data,ordered_seeds,split):
    """Return unique initial source indices in EXACT supplied cache seed order."""
    seeds=np.asarray(data['seeds']);ordered_seeds=np.asarray(ordered_seeds)
    _split_seeds(seeds,split);_split_seeds(ordered_seeds,split)
    if len(np.unique(ordered_seeds))!=len(ordered_seeds):raise ValueError('duplicate cache seeds')
    valid=np.asarray(data['history_valid']);previous=np.asarray(data['previous_actions']);lives=np.asarray(data['current_lives'])
    if valid.shape!=(len(seeds),8) or valid.dtype!=np.bool_ or previous.shape!=valid.shape or not np.issubdtype(previous.dtype,np.integer):
        raise ValueError('initial selection requires boolean H8 and integer actions')
    if lives.shape!=(len(seeds),) or not np.issubdtype(lives.dtype,np.integer):raise ValueError('initial selection needs integer lives')
    candidates=np.flatnonzero((valid.sum(1)==1)&valid[:,-1]&(lives==3)&(previous==-1).all(1))
    mapping={}
    for row in candidates:mapping.setdefault(int(seeds[row]),[]).append(int(row))
    selected=[]
    for seed in ordered_seeds:
        matches=mapping.get(int(seed),[])
        if len(matches)!=1:raise ValueError(f'seed {seed} has {len(matches)} initial rows; exactly one required')
        selected.append(matches[0])
    return np.asarray(selected,dtype=np.int64)


def parent_metadata(cache,split):
    cache=Path(cache);path=cache/'manifest.json';raw=path.read_bytes();manifest=json.loads(raw)
    formats={'pebby.structured-field-cache.v1','pebby.structured-closing-sequence-cache.v1','pebby.structured-sequence-cache.v1'}
    if manifest.get('format') not in formats or manifest.get('status')!='complete' or manifest.get('source')!='generated_only' or manifest.get('split')!=split:
        raise ValueError('requires complete generated structured cache in requested split')
    watched={str(path):hashlib.sha256(raw).hexdigest()}
    sequence=manifest['format']!='pebby.structured-field-cache.v1'
    closing=manifest['format']=='pebby.structured-closing-sequence-cache.v1'
    if sequence:
        expected_split='train' if closing else 'validation'
        expected_mode='closing_only_chronological_K4' if closing else 'heldout_chronological_K4'
        index=manifest.get('source_index_metadata',{})
        if (split!=expected_split or manifest.get('mode')!=expected_mode or index.get('K')!=4 or index.get('history')!=8
                or index.get('source')!='generated_only' or index.get('split')!=split
                or index.get('mode')!=('closing_only_train' if closing else 'explore_validation')
                or index.get('format')!=('pebby.ls20-closing-four-step-index.v1' if closing else 'pebby.ls20-four-step-index.v1')
                or manifest.get('history_verified_against_actual_source_rows') is not True):
            raise ValueError('invalid H4 split/mode/index provenance')
        if closing and manifest.get('final_history_verified_independent_append_reset') is not True:
            raise ValueError('closing cache lacks final-history verification')
        # Producer code hashes describe historical builds; do not require that
        # producer source (including this evolving tool) remains installed.
        # Actual data/index/attestation bytes are immutable inputs and checked.
        hashes=manifest.get('source_hashes',{})
        for source,expected in hashes.items():
            if Path(source).suffix in ('.npz','.json'):
                if digest(source)!=expected:raise ValueError(f'H4 parent source hash mismatch: {source}')
                watched[source]=expected
        mapping=manifest.get('initial_source_mapping',{}) if closing else index
        if closing and mapping.get('array')!='source_initial_rows':raise ValueError('closing initial mapping must name source_initial_rows')
        for record in (index,mapping):
            source=Path(record.get('source_path',''));expected=record.get('source_sha256')
            recorded={h for p,h in hashes.items() if Path(p).resolve()==source.resolve()}
            if not expected or recorded!={expected} or digest(source)!=expected:
                raise ValueError('H4 source mapping is not hash-bound')
            watched[str(source)]=expected
        manifest={**manifest,'source_path':mapping['source_path'],'source_sha256':mapping['source_sha256'],
                  'parent_branch_source_path':index['source_path'],'parent_branch_source_sha256':index['source_sha256']}
    values={}
    for name in ('seeds','source_rows',*(('source_initial_rows',) if closing else ())):
        p=cache/(name+'.npy');expected=manifest['arrays'][name]['sha256']
        if digest(p)!=expected:raise ValueError(f'parent cache {name} hash mismatch')
        watched[str(p)]=expected;values[name]=np.load(p,allow_pickle=False)
    _split_seeds(values['seeds'],split)
    if values['source_rows'].shape!=values['seeds'].shape or not np.issubdtype(values['source_rows'].dtype,np.integer):raise ValueError('invalid parent source row order')
    if len(np.unique(values['seeds']))!=len(values['seeds']):raise ValueError('duplicate parent seeds')
    if closing:
        initial=values['source_initial_rows']
        if initial.shape!=values['seeds'].shape or not np.issubdtype(initial.dtype,np.integer):raise ValueError('invalid source_initial_rows')
        with np.load(manifest['parent_branch_source_path'],allow_pickle=False) as archive:
            branch_seeds=archive['seeds']
        source_rows=values['source_rows']
        if np.any((source_rows<0)|(source_rows>=len(branch_seeds))) or not np.array_equal(branch_seeds[source_rows],values['seeds']):
            raise ValueError('closing branch source seed/order mismatch')
    return manifest,values,watched


def source_arrays(manifest,array_cache):
    source=Path(manifest['source_path']);expected=manifest['source_sha256']
    if digest(source)!=expected:raise ValueError('source NPZ hash mismatch')
    choices=[p for p in Path(array_cache).glob(expected+'-*') if p.is_dir() and all((p/(name+'.npy')).exists() for name in SOURCE_ARRAYS)]
    if not choices:raise ValueError('existing checked source array cache required; builder never rewrites image banks')
    root=sorted(choices)[0];path=root/'manifest.json';raw=path.read_bytes();cache=json.loads(raw)
    if cache['source_sha256']!=expected:raise ValueError('array cache source hash mismatch')
    watched={str(source):expected,str(path):hashlib.sha256(raw).hexdigest()};data={}
    for name in SOURCE_ARRAYS:
        path=root/(name+'.npy');expected=cache['arrays'][name]['sha256']
        if digest(path)!=expected:raise ValueError(f'source array hash mismatch: {name}')
        watched[str(path)]=expected;value=np.load(path,mmap_mode='r',allow_pickle=False)
        data[name]=json.loads(str(value)) if name=='meta' else value
    from pebby.agent.world_train import require_verified_data,require_winning_coverage
    require_verified_data(data);require_winning_coverage(data)
    if data['meta'].get('split')!=manifest['split']:raise ValueError('source metadata split mismatch')
    return data,watched


def encoder_for(manifest,device):
    from pebby.agent.structured_field import load_structured_field_encoder
    expected=manifest['field_encoder'];watched={}
    for group in ('code_hashes','checkpoint_hashes'):
        if not expected.get(group):raise ValueError('parent lacks complete encoder hashes')
        for path,value in expected[group].items():
            if digest(path)!=value:raise ValueError(f'encoder source drift: {path}')
            watched[path]=value
    encoder=load_structured_field_encoder(expected['sources']['world_checkpoint']['path'],
                                         expected['sources']['visibility']['checkpoint'],device=device).eval()
    if encoder.metadata()!={k:expected[k] for k in encoder.metadata()}:
        raise ValueError('frozen encoder metadata differs from parent')
    if any(p.requires_grad for p in encoder.parameters()):raise ValueError('field encoder must be frozen')
    return encoder,watched


def encode_public_initial(encoder,data,rows):
    """Only the three stored PUBLIC history inputs cross the encoder boundary."""
    frames=np.array(data['frames'][rows],copy=True)
    valid=np.array(data['history_valid'][rows],copy=True)
    actions=np.array(data['previous_actions'][rows],copy=True)
    if frames.shape!=(len(rows),8,64,64) or frames.dtype!=np.uint8:raise ValueError('expected uint8 public H8 frames')
    if not (valid.sum(1)==1).all() or not valid[:,-1].all() or not (actions==-1).all():raise ValueError('not initial public histories')
    with torch.inference_mode():value=encoder(torch.from_numpy(frames),torch.from_numpy(valid),torch.from_numpy(actions))
    if value.shape!=(len(rows),148,96):raise ValueError('field encoder shape mismatch')
    value=value.detach().cpu().float().numpy();result=value.astype(np.float16)
    if not np.isfinite(value).all() or not np.isfinite(result).all():raise ValueError('nonfinite/overflowed field')
    return result,float(np.max(np.abs(value-result.astype(np.float32))))


def build_split(cache,out,split,*,device='cpu',batch_size=128,limit=None,array_cache='data/world-array-cache'):
    start=time.monotonic();out=Path(out)
    if out.exists():raise FileExistsError(out)
    manifest,parent,watched=parent_metadata(cache,split)
    if type(batch_size)!=int or batch_size<1:raise ValueError('batch size must be positive')
    count=len(parent['seeds']) if limit is None else limit
    if type(count)!=int or not 1<=count<=len(parent['seeds']):raise ValueError('limit must be positive and fit parent')
    data,source_watched=source_arrays(manifest,array_cache);watched.update(source_watched)
    original_rows=parent.get('source_initial_rows',parent['source_rows'])
    if np.any((original_rows<0)|(original_rows>=len(data['seeds']))) or not np.array_equal(data['seeds'][original_rows],parent['seeds']):
        raise ValueError('parent seed/source row ordering mismatch')
    seeds=parent['seeds'][:count];rows=select_initial_rows(data,seeds,split)
    if 'source_initial_rows' in parent and not np.array_equal(rows,parent['source_initial_rows'][:count]):
        raise ValueError('declared source_initial_rows differ from unique verified initial rows')
    encoder,encoder_watched=encoder_for(manifest,device);watched.update(encoder_watched)
    watched[str(Path(__file__))]=digest(__file__)
    out.parent.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(prefix='.'+out.name+'-',dir=out.parent))
    try:
        fields=np.lib.format.open_memmap(temporary/'initial_fields.npy',mode='w+',dtype=np.float16,shape=(count,148,96))
        error=0.;encoding=0.
        for first in range(0,count,batch_size):
            tick=time.monotonic();value,quantization=encode_public_initial(encoder,data,rows[first:first+batch_size]);encoding+=time.monotonic()-tick
            fields[first:first+len(value)]=value;error=max(error,quantization)
            print(json.dumps({'split':split,'encoded':first+len(value),'total':count}),flush=True)
        fields.flush();del fields
        np.save(temporary/'seeds.npy',seeds,allow_pickle=False)
        np.save(temporary/'source_initial_rows.npy',rows,allow_pickle=False)
        arrays={}
        for path in sorted(temporary.glob('*.npy')):
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            arrays[path.stem]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':digest(path)}
        if any(digest(p)!=h for p,h in watched.items()):raise ValueError('input/code/hash drift while building initial memory')
        result={'format':FORMAT,'status':'complete','source':'generated_only','split':split,'pid':os.getpid(),
                'parent_cache':str(cache),'parent_manifest_sha256':watched[str(Path(cache)/'manifest.json')],
                'parent_format':manifest['format'],'parent_mode':manifest.get('mode'),
                'parent_branch_source_path':manifest.get('parent_branch_source_path',manifest['source_path']),
                'parent_branch_source_sha256':manifest.get('parent_branch_source_sha256',manifest['source_sha256']),
                'parent_seed_sha256':manifest['arrays']['seeds']['sha256'],'parent_levels':len(parent['seeds']),
                'rows':count,'complete_parent_alignment':count==len(parent['seeds']),
                'source_path':manifest['source_path'],'source_sha256':manifest['source_sha256'],
                'field_encoder':manifest['field_encoder'],'arrays':arrays,'watched_hashes':watched,
                'source_unchanged':True,'device':device,'batch_size':batch_size,'encoding_seconds':encoding,
                'encoding_precision':{'device':device,'compute_dtype':'float32','cache_dtype':'float16','autocast':False,
                                      'matmul_tf32_allowed':torch.backends.cuda.matmul.allow_tf32,
                                      'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32},
                'float16_max_absolute_quantization_error':error,'elapsed_seconds':time.monotonic()-start,
                'memory_contract':'Stored public initial H8/valid/actions only; retain across life losses; initial fog limits information.',
                'initial_selection':'Exactly one source row per seed: valid.sum1,lastvalid,lives3,previousall-1; source labels only identify row, never encoder inputs.'}
        (temporary/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');temporary.rename(out)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('data/structured-field-pilot-2048'))
    parser.add_argument('--train-cache',type=Path,help='Explicit train parent, including closing H4 cache')
    parser.add_argument('--validation-cache',type=Path,help='Explicit validation parent, including live H4 cache')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=('train','validation','both'),default='both')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--array-cache',type=Path,default=Path('data/world-array-cache'))
    args=parser.parse_args();print('PID',os.getpid(),flush=True);torch.set_num_threads(1)
    def expired(*_):raise TimeoutError('initial-cache600second deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    try:
        for split in (('train','validation') if args.split=='both' else (args.split,)):
            parent=(args.train_cache if split=='train' else args.validation_cache) or args.cache/split
            result=build_split(parent,args.out/split,split,device=args.device,batch_size=args.batch_size,
                               limit=args.limit,array_cache=args.array_cache)
            print(json.dumps({k:result[k] for k in ('split','rows','encoding_seconds','elapsed_seconds','complete_parent_alignment')}),flush=True)
    finally:signal.alarm(0)

if __name__=='__main__':main()
