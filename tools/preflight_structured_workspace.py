"""Disposable real-optimizer capacity/throughput probe for spatial readouts."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.model import load_checkpoint
from pebby.agent.structured_workspace_policy import StructuredWorkspaceReadout
from tools.cache_structured_policy_successors import load_imagined_cache
from tools.structured_policy_batch import prepared_policy_inputs
from tools.train_structured_paired_policy import paired_batch, validate_pair
from tools.train_structured_policy import check_policy_encoder, load_policy_cache, policy_terms
from tools.train_structured_transition import atomic_json, digest


def backward(head, batch, device, loops):
    inputs, masks = prepared_policy_inputs(batch, 'successors', device, allow_unreachable=True)
    result = {}
    # Both views contribute equally to ONE optimizer step over distinct levels.
    for name, fields in inputs.items():
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            terms = policy_terms(head(fields, loops=loops), masks, allow_unreachable=True)
            loss = .5 * terms['ce'].sum() / terms['valid'].sum().clamp_min(1)
        loss.backward()
        result[name] = float(loss.detach())
    result['policy_valid_fraction'] = float((masks != 0).float().mean())
    return result


def load_inputs():
    actor_path = 'checkpoints/ls20-structured-policy-paired-local-h4-600.pt'
    actor_sha = digest(actor_path)
    policy, saved = load_checkpoint(actor_path, 'cpu')
    if digest(actor_path) != actor_sha:
        raise ValueError('actor changed during load')
    views, manifests = [], []
    sources = {actor_path: actor_sha, **policy.sources['code_hashes']}
    sources.update({info['path']: info['sha256'] for info in policy.sources['artifacts'].values()})
    for directory, imagined in [
        ('data/structured-field-16384/train', 'data/structured-policy-imagined-local-h4-400'),
        ('data/structured-field-additional-state-16384/train', 'data/structured-policy-imagined-additional-local-h4-400'),
    ]:
        data, manifest = load_policy_cache(directory, 'train')
        check_policy_encoder(manifest['field_encoder'], policy, True)
        data['imagined_fields'], hashes = load_imagined_cache(imagined, 'train', directory, data, manifest, policy)
        sources.update(hashes)
        sources[str(Path(directory)/'manifest.json')] = digest(Path(directory)/'manifest.json')
        sources.update({str(Path(directory)/(name+'.npy')): info['sha256'] for name, info in manifest['arrays'].items()})
        views.append(data)
        manifests.append(manifest)
    validate_pair(*views, *manifests, 'data/structured-field-16384/train')
    for path in [__file__, 'pebby/agent/structured_workspace_policy.py', 'pebby/agent/structured_policy.py',
                 'tools/train_structured_paired_policy.py', 'tools/structured_policy_batch.py',
                 'tools/train_structured_policy.py', 'tools/cache_structured_policy_successors.py',
                 'tools/train_structured_transition.py']:
        sources[str(path)] = digest(path)
    return policy, views, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--max-batch', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.max_batch < 2 or args.max_batch > 1024 or args.max_batch & (args.max_batch-1):
        parser.error('new output and power-of-two batch2..1024 required')
    if not 2 <= args.steps <= 10 or not 1 <= args.seconds <= 600:
        parser.error('bounded steps/time required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('preflight bound')))
    signal.alarm(args.seconds)
    report = {'status': 'running', 'pid': os.getpid(), 'args': vars(args) | {'report': str(args.report)},
              'official_inputs_used': False, 'checkpoints_published': False, 'attempts': [],
              'precision': 'CUDA BF16, CPU FP32; TF32 disabled',
              'limits': ['Disposable steps do not establish training convergence or gameplay success.',
                         'Static workspace is frozen and inactive; active parameter capacity differs.']}
    def persist():
        report['elapsed_seconds'] = time.monotonic()-start
        atomic_json(args.report, report)
    persist()
    try:
        policy, views, sources = load_inputs()
        report['sources'] = sources
        # Exact neutral parity on actual verified TRAIN fields before any probe.
        sample, _ = paired_batch(views, np.arange(4), np.random.default_rng(43), 'successors')
        fields = torch.from_numpy(sample['next_fields']).float()
        with torch.inference_mode():
            neutral = StructuredWorkspaceReadout.from_readout(policy.readout, evolving=True)
            old = policy.readout(fields)
            new = neutral(fields)
            if not torch.equal(old, new):
                raise ValueError(f'neutral warmstart not exact: {float((old-new).abs().max())}')
        report['neutral_cpu_parity_exact'] = True
        del neutral, fields, old, new, sample
        size = args.max_batch
        while size >= 2:
            head = optimizer = batch = None
            try:
                batch, _ = paired_batch(views, np.arange(size), np.random.default_rng(43), 'successors')
                # Evolving arm has the larger memory footprint; test worst depth4.
                torch.manual_seed(42)
                head = StructuredWorkspaceReadout.from_readout(policy.readout, evolving=True, checkpoint_workspace=True).to(args.device).train()
                optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=.0003, weight_decay=.01)
                if args.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                records = []
                for step in range(args.steps):
                    tick = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    losses = backward(head, batch, args.device, 4)
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step()
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    records.append({'step': step+1, 'seconds': time.monotonic()-tick,
                                    'losses': losses, 'gradient_norm': float(norm)})
                report['attempts'].append({'batch_size': size, 'status': 'fits', 'records': records,
                    'parameters': head.parameter_count(), 'active_trainable_parameters': head.trainable_parameter_count(),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None})
                report['batch_size'] = size
                report['steady_step_seconds_mean'] = float(np.mean([r['seconds'] for r in records[1:]]))
                break
            except torch.cuda.OutOfMemoryError:
                report['attempts'].append({'batch_size': size, 'status': 'out_of_memory'})
                size //= 2
                persist()
            finally:
                del head, optimizer, batch
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
        if size < 2:
            raise RuntimeError('no fitting even batch')
        if any(digest(path) != expected for path, expected in sources.items()):
            raise ValueError('preflight source drift')
        report.update(status='complete', source_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        signal.alarm(0)
        persist()


if __name__ == '__main__':
    main()
