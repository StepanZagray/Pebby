"""Bounded sequential CUDA parity/timing benchmark; never publishes trained weights.

Run only after the current training job releases the GPU. Two fresh, identical
readout heads take the same two AdamW steps on one fixed prepared B1024 batch.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from tools.train_structured_policy import (build_training_policy, check_policy_encoder,
    load_policy_cache, loss_for_rows, new_head, policy_terms)
from tools.train_structured_paired_policy import paired_batch, validate_pair
from tools.structured_policy_batch import outputs_for_prepared_batch
from tools.cache_structured_policy_successors import load_imagined_cache
from tools.train_structured_transition import atomic_json, digest, sample_rows


def prepared_loss(head, batch):
    outputs, masks = outputs_for_prepared_batch(head, batch, 'cuda')
    parts = {k: policy_terms(v, masks)['ce'].mean() for k, v in outputs.items()}
    return sum(parts.values()) / len(parts), parts


def run_arm(kind, batch, config, seed, lr):
    head = new_head(config, seed, 'cuda')
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    rows = np.arange(1024)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        if kind == 'original':
            loss, parts = loss_for_rows(head, batch, rows, 'cuda')
        else:
            loss, parts = prepared_loss(head, batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite benchmark loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        steps.append({'update':update+1,'uniform_optimal_ce':float(loss.detach()),
                      'parts':{k:float(v.detach()) for k,v in parts.items()},
                      'gradient_norm':float(norm),'synchronized_update_seconds':elapsed,
                      'cuda_stream_interval_ms':begin.elapsed_time(end),
                      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        del loss, parts, begin, end
    weights = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    result = {'steps':steps,'parameters':sum(p.numel() for p in head.parameters()),
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    del optimizer, head
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    return result, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', default='data/structured-field-16384/train')
    parser.add_argument('--additional-cache', default='data/structured-field-additional-state-16384/train')
    parser.add_argument('--validation-cache', default='data/structured-field-16384/validation')
    parser.add_argument('--imagined-cache', default='data/structured-policy-imagined-local-h4-400')
    parser.add_argument('--additional-imagined-cache', default='data/structured-policy-imagined-additional-local-h4-400')
    parser.add_argument('--world', default='checkpoints/ls20-world-cell-recall-b1024.pt')
    parser.add_argument('--visibility', default='checkpoints/ls20-cell-visibility-initial-200.pt')
    parser.add_argument('--dynamics', default='checkpoints/ls20-factored-local-h4-400.pt')
    parser.add_argument('--report', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--view-seed', type=int, default=43)
    parser.add_argument('--loops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seconds', type=int, default=180)
    args = parser.parse_args()
    if Path(args.report).exists():parser.error('refusing existing report')
    if not 1<=args.seconds<=300 or not np.isfinite(args.lr) or args.lr<=0:
        parser.error('bounded seconds1..300 and finite positive LR required')
    torch.set_num_threads(1)
    report={'status':'running','pid':os.getpid(),'args':vars(args),'batch_size':1024,
            'protocol':'Two sequential fresh heads, same seed/config and prepared1024 distinct TRAIN levels,512 per view, two AdamW steps each, no weights published.',
            'timing_limit':'CUDA event intervals include host submission gaps, not pure GPU-active kernel time. Wall times include CPU preparation plus synchronized transfer/forward/backward/step. First update may include cold-start effects; compare second update separately. Only two updates per arm, not a throughput estimate.',
            'tolerance':{'atol':1e-6,'rtol':1e-5}}
    atomic_json(args.report,report);started=time.monotonic();print('PID',os.getpid(),flush=True)
    def expired(*_):raise TimeoutError('bounded prepared-policy benchmark deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        # This script never stops another process. The operator must authorize
        # launch after GPU release; fail closed if another compute PID is visible.
        import subprocess
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if any(int(pid.strip())!=os.getpid() for pid in active.splitlines() if pid.strip()):
            raise RuntimeError('another GPU compute process is active')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        first,fm=load_policy_cache(args.train_cache,'train')
        second,sm=load_policy_cache(args.additional_cache,'train')
        validate_pair(first,second,fm,sm,args.train_cache)
        valpath=Path(args.validation_cache)
        vm=json.loads((valpath/'manifest.json').read_text())
        if (vm.get('status')!='complete' or vm.get('source')!='generated_only'
                or vm.get('split')!='validation' or vm.get('field_encoder')!=fm['field_encoder']):
            raise ValueError('validation provenance mismatch')
        if digest(valpath/'seeds.npy')!=vm['arrays']['seeds']['sha256']:
            raise ValueError('validation seed hash mismatch')
        if np.intersect1d(first['seeds'],np.load(valpath/'seeds.npy',mmap_mode='r')).size:
            raise ValueError('TRAIN/validation level leakage')
        policy,_,factored=build_training_policy(args.world,args.visibility,args.dynamics,
                                               {'mode':'successors','loops':args.loops})
        for manifest in (fm,sm):check_policy_encoder(manifest['field_encoder'],policy,factored)
        sources={str(valpath/'manifest.json'):digest(valpath/'manifest.json'),
                 str(valpath/'seeds.npy'):vm['arrays']['seeds']['sha256']}
        for cache,data,manifest,imagined in ((args.train_cache,first,fm,args.imagined_cache),
                                            (args.additional_cache,second,sm,args.additional_imagined_cache)):
            data['imagined_fields'],bound=load_imagined_cache(imagined,'train',cache,data,manifest,policy)
            sources.update(bound)
            sources[str(Path(cache)/'manifest.json')]=digest(Path(cache)/'manifest.json')
            # Payloads already checked once by strict loaders; retain their
            # inventories in report without rehashing them on every arm/trial.
        sources.update(policy.sources['code_hashes'])
        sources.update({v['path']:v['sha256'] for v in policy.sources['artifacts'].values()})
        for path in (__file__,'tools/train_structured_paired_policy.py','tools/train_structured_policy.py',
                     'tools/structured_policy_batch.py','tools/cache_structured_policy_successors.py'):
            sources[str(path)]=digest(path)
        rows=sample_rows(first,1024,0.,np.random.default_rng(args.seed))
        tick=time.monotonic()
        batch,view=paired_batch((first,second),rows,np.random.default_rng(args.view_seed),'successors')
        report.update(paired_batch_seconds=time.monotonic()-tick,level_indices=rows.tolist(),
                      seeds=batch['seeds'].tolist(),source_rows=batch['source_rows'].tolist(),
                      view_assignments=view.tolist(),view_counts=np.bincount(view,minlength=2).tolist(),
                      sources=sources,cache_inventory={'original':fm['arrays'],'additional':sm['arrays']},
                      parameter_counts=policy.parameter_counts(),precision='FP32; autocast/TF32 off')
        config=policy.config()
        if any(p.requires_grad for p in policy.encoder.parameters()) or any(p.requires_grad for p in policy.dynamics.parameters()):
            raise ValueError('source encoder/dynamics must remain frozen')
        # Both frozen source modules remain on CPU throughout this benchmark.
        report['original'],original=run_arm('original',batch,config,args.seed,args.lr)
        atomic_json(args.report,report)
        report['prepared'],prepared=run_arm('prepared',batch,config,args.seed,args.lr)
        names=list(original);assert names==list(prepared)
        report['parameter_parity']={name:{'bitexact':torch.equal(original[name],prepared[name]),
            'max_abs':float((original[name]-prepared[name]).abs().max()),
            'within_tolerance':torch.allclose(original[name],prepared[name],atol=1e-6,rtol=1e-5)} for name in names}
        report['loss_parity']=[{'update':i+1,'absolute_difference':abs(a['uniform_optimal_ce']-b['uniform_optimal_ce']),
                               'within_tolerance':bool(np.isclose(a['uniform_optimal_ce'],b['uniform_optimal_ce'],atol=1e-6,rtol=1e-5))}
                              for i,(a,b) in enumerate(zip(report['original']['steps'],report['prepared']['steps']))]
        if not all(x['within_tolerance'] for x in report['parameter_parity'].values()) or not all(x['within_tolerance'] for x in report['loss_parity']):
            raise ValueError('two-step parameter/loss parity failed')
        # Only small manifests/code/checkpoints are rechecked here. Large
        # immutable payload hashes were checked once at load, before both arms.
        small={p:s for p,s in sources.items() if not p.endswith('.npy')}
        if any(digest(p)!=s for p,s in small.items()):raise ValueError('benchmark source drift')
        report.update(status='complete',source_bindings_unchanged=True,
                      all_parameters_bitexact=all(x['bitexact'] for x in report['parameter_parity'].values()))
        del original,prepared,batch,policy;gc.collect();torch.cuda.empty_cache()
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;atomic_json(args.report,report)
    print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
                      'all_parameters_bitexact':report['all_parameters_bitexact']}),flush=True)


if __name__=='__main__':main()
