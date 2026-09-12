"""Measure the largest power-of-two TRUE world-model batch up to 1024.

Every candidate runs two full optimizer steps, including float32 SIGReg across
all examples and four successor branches. No gradient accumulation is used.
This is a fit probe, not a trained checkpoint; all probe weights are discarded.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import torch

from .world_model import (WorldModelConfig, WorldPolicy, parameter_groups)
from .world_training_objectives import world_losses
from .world_train import load_dataset, as_tensors


def candidate(config, tensors, size, precision, checkpoint_encoder, checkpoint_loops, encoder_chunk_size=0):
    model = optimizer = batch = result = None
    try:
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = WorldPolicy(config).cuda().train()
        model.checkpoint_encoder = checkpoint_encoder
        model.checkpoint_loops = checkpoint_loops
        model.encoder_chunk_size = encoder_chunk_size
        optimizer = torch.optim.AdamW(parameter_groups(model, .05), lr=1e-4)
        indices = torch.arange(size) % len(tensors['frames'])
        batch = {key:value[indices].cuda() for key,value in tensors.items()}
        started = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                result = world_losses(model, batch)
            result['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        return {'batch_size':size, 'fits':True, 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'seconds_two_steps':time.perf_counter()-started}
    except torch.OutOfMemoryError:
        return {'batch_size':size, 'fits':False, 'reason':'CUDA out of memory'}
    finally:
        result = batch = optimizer = model = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--config-checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--precision',choices=('float32','bf16'),default='bf16')
    parser.add_argument('--grounding',action='store_true')
    parser.add_argument('--checkpoint-encoder',action='store_true')
    parser.add_argument('--checkpoint-loops',action='store_true')
    parser.add_argument('--encoder-chunk-size',type=int,default=0)
    parser.add_argument('--start',type=int,default=64)
    parser.add_argument('--cap',type=int,default=1024)
    args=parser.parse_args()
    if args.encoder_chunk_size<0:parser.error('encoder-chunk-size must not be negative')
    if not torch.cuda.is_available():parser.error('CUDA required')
    if args.start<1 or args.cap>1024 or args.start>args.cap or args.start&(args.start-1) or args.cap&(args.cap-1):
        parser.error('start/cap must be powers of two, start <= cap <= 1024')
    if args.precision=='bf16' and not torch.cuda.is_bf16_supported():parser.error('native bf16 unsupported')
    saved=torch.load(args.config_checkpoint,weights_only=True,map_location='cpu')
    config=WorldModelConfig.from_dict({**saved['config'],'grounding':args.grounding})
    tensors=as_tensors(load_dataset(args.data,config.history))
    report={'config':config.as_dict(),'precision':args.precision,'checkpoint_encoder':args.checkpoint_encoder,
            'checkpoint_loops':args.checkpoint_loops,'gradient_accumulation':False,'trials':[],
            'encoder_chunk_size':args.encoder_chunk_size,
            'gpu':torch.cuda.get_device_name(),'largest_fitting_batch':None}
    size=args.start
    while size<=args.cap:
        trial=candidate(config,tensors,size,args.precision,args.checkpoint_encoder,args.checkpoint_loops,
                        args.encoder_chunk_size)
        report['trials'].append(trial)
        print(json.dumps(trial),flush=True)
        if not trial['fits']:break
        report['largest_fitting_batch']=size
        size*=2
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    if report['largest_fitting_batch'] is None:raise SystemExit('No candidate fit; retry smaller start')
    print(f"Largest fitting true batch: {report['largest_fitting_batch']}",flush=True)


if __name__=='__main__': main()
