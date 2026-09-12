"""Build/validate FP32 imagined successors for cached-field policy training only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from tools.train_structured_transition import digest, atomic_json

FORMAT='pebby.structured-policy-imagined-cache.v1'
PRECISION={'compute_dtype':'float32','storage_dtype':'float32','autocast':False,'matmul_tf32':False,'cudnn_tf32':False}


def binding(policy):
    from pebby.agent.structured_factored_policy import canonical_metadata
    return canonical_metadata(policy.sources)


def source_binding(path,manifest):
    path=Path(path).resolve()
    return {'manifest_path':str(path/'manifest.json'),'manifest_sha256':digest(path/'manifest.json'),
            'arrays':{key:manifest['arrays'][key] for key in ('fields','seeds','source_rows')}}


def load_imagined_cache(root,split,source,data,manifest,policy):
    """Fail closed; caller adds returned fingerprints to training source guards."""
    root=Path(root);main_path=root/'manifest.json';main=json.loads(main_path.read_text())
    if (main.get('format')!=FORMAT or main.get('status')!='complete' or main.get('source')!='generated_only'
            or main.get('precision')!=PRECISION or main.get('binding')!=binding(policy)):
        raise ValueError('imagined cache format/precision/model binding mismatch')
    if split not in ('train','validation'):raise ValueError('invalid imagined split')
    subpath=root/split/'manifest.json'
    if digest(subpath)!=main['splits'][split]['sha256']:raise ValueError('imagined split manifest hash mismatch')
    sub=json.loads(subpath.read_text())
    if sub.get('split')!=split or sub.get('source_binding')!=source_binding(source,manifest):
        raise ValueError('imagined cache source/order binding mismatch')
    if set(sub.get('arrays',{}))!={'imagined_fields','seeds','source_rows'}:raise ValueError('imagined array inventory mismatch')
    fingerprints={str(main_path):digest(main_path),str(subpath):digest(subpath)};arrays={};n=len(data['seeds'])
    for name,info in sub['arrays'].items():
        path=root/split/(name+'.npy')
        if digest(path)!=info['sha256']:raise ValueError('imagined array hash mismatch')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        expected=(n,4,148,96) if name=='imagined_fields' else (n,)
        if array.shape!=expected or list(array.shape)!=info['shape'] or str(array.dtype)!=info['dtype']:
            raise ValueError('imagined array shape mismatch')
        if name=='imagined_fields':
            if array.dtype!=np.float32:raise ValueError('imagined field dtype must float32')
            for first in range(0,n,32):
                if not np.isfinite(array[first:first+32]).all():raise ValueError('nonfinite imagined fields')
        elif not np.array_equal(array,data[name]):raise ValueError('imagined seed/source row order mismatch')
        arrays[name]=array;fingerprints[str(path)]=info['sha256']
    return arrays['imagined_fields'],fingerprints


def build(root,source_root,policy,*,device='cpu',max_batch=128):
    from tools.train_structured_policy import load_policy_cache,check_policy_encoder
    from pebby.agent.structured_factored_policy import StructuredFactoredPolicy
    if policy.dynamics is None or not 1<=max_batch<=128:raise ValueError('frozen dynamics and batch1..128 required')
    root=Path(root);source_root=Path(source_root)
    if root.exists():raise ValueError('refusing existing imagined cache')
    temporary=root.with_name(root.name+f'.building-{os.getpid()}')
    if temporary.exists():raise ValueError('temporary cache already exists')
    root.parent.mkdir(parents=True,exist_ok=True);temporary.mkdir()
    policy.dynamics.to(device).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources={str(Path(__file__).resolve()):digest(__file__),'tools/train_structured_policy.py':digest('tools/train_structured_policy.py')}
    sources.update(policy.sources['code_hashes']);sources.update({v['path']:v['sha256'] for v in policy.sources['artifacts'].values()})
    report={'format':FORMAT,'status':'building','source':'generated_only','precision':PRECISION,
            'device':device,'max_batch':max_batch,'binding':binding(policy),'splits':{},'pid':os.getpid(),
            'scope':'Training-only imagined fields; public inference always predicts successors from current public fields.'}
    seen=set();encoder=None
    try:
        for split in ('train','validation'):
            source=source_root/split;data,manifest=load_policy_cache(source,split)
            check_policy_encoder(manifest['field_encoder'],policy,isinstance(policy,StructuredFactoredPolicy))
            if encoder is not None and encoder!=manifest['field_encoder']:raise ValueError('split encoder mismatch')
            encoder=manifest['field_encoder'];seeds=set(map(int,data['seeds']))
            if seen&seeds:raise ValueError('split seed leakage')
            seen|=seeds
            sub={'split':split,'source_binding':source_binding(source,manifest),'arrays':{}}
            for name in ('manifest.json','fields.npy','seeds.npy','source_rows.npy'):
                sources[str(source/name)]=digest(source/name)
            directory=temporary/split;directory.mkdir();n=len(data['seeds'])
            target=np.lib.format.open_memmap(directory/'imagined_fields.npy',mode='w+',dtype=np.float32,shape=(n,4,148,96))
            flat=target.reshape(n*4,148,96)
            with torch.inference_mode():
                for first in range(0,n*4,max_batch):
                    ids=np.arange(first,min(first+max_batch,n*4))
                    current=torch.as_tensor(np.array(data['fields'][ids//4]),device=device).float()
                    actions=torch.as_tensor(ids%4,device=device).long()
                    predicted=policy.dynamics.predict(current,actions)
                    if predicted.dtype!=torch.float32 or not bool(torch.isfinite(predicted).all()):
                        raise ValueError('imagined predictions must finite FP32')
                    flat[first:first+len(ids)]=predicted.cpu().numpy()
            target.flush();del flat,target
            for name in ('seeds','source_rows'):np.save(directory/(name+'.npy'),np.array(data[name]),allow_pickle=False)
            for name in ('imagined_fields','seeds','source_rows'):
                path=directory/(name+'.npy');array=np.load(path,mmap_mode='r',allow_pickle=False)
                for first in range(0,len(array),32):
                    if not np.isfinite(array[first:first+32]).all():raise ValueError('nonfinite saved array')
                sub['arrays'][name]={'shape':list(array.shape),'dtype':str(array.dtype),'sha256':digest(path)}
            atomic_json(directory/'manifest.json',sub)
            report['splits'][split]={'levels':n,'sha256':digest(directory/'manifest.json')}
            print(json.dumps({'split':split,'levels':n,'branches':n*4}),flush=True)
        if any(digest(p)!=sha for p,sha in sources.items()):raise ValueError('source changed during imagined caching')
        report.update(status='complete',source_hashes=sources)
        atomic_json(temporary/'manifest.json',report)
        os.rename(temporary,root)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report


def main():
    from tools.train_structured_policy import build_training_policy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-cache',default='data/structured-field-16384');p.add_argument('--output',required=True)
    p.add_argument('--dynamics',required=True);p.add_argument('--world',default='checkpoints/ls20-world-cell-recall-b1024.pt')
    p.add_argument('--visibility',default='checkpoints/ls20-cell-visibility-initial-200.pt')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--max-batch',type=int,default=128)
    p.add_argument('--seconds',type=int,default=300);args=p.parse_args()
    if not 1<=args.seconds<=600 or not 1<=args.max_batch<=128:p.error('bounded seconds1..600/batch1..128 required')
    if Path(args.output).exists():p.error('refusing existing output')
    torch.set_num_threads(1);start=time.monotonic();print('PID',os.getpid(),flush=True)
    def expired(*_):raise TimeoutError('imagined cache deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    try:
        policy,_,_=build_training_policy(args.world,args.visibility,args.dynamics,{'mode':'successors'})
        build(args.output,args.source_cache,policy,device=args.device,max_batch=args.max_batch)
        print(json.dumps({'status':'complete','elapsed_seconds':time.monotonic()-start}),flush=True)
    finally:signal.alarm(0)

if __name__=='__main__':main()
