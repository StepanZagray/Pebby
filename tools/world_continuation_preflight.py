"""Disposable true-batch CUDA check before a world-policy continuation.

Loads generated training data only, samples one row from each of distinct
levels, and discards all updated weights. No checkpoint is saved. This checks
memory and finite gradients, not controller quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from pebby.agent.world_model import (WorldModelConfig, WorldPolicy, initialize_from_checkpoint, initialize_glyph_encoder, load_world_checkpoint, parameter_groups)
from pebby.agent.world_training_objectives import world_losses
from pebby.agent.world_train import (load_dataset, require_verified_data,
    require_winning_coverage, attach_successor_labels, as_tensors)
from pebby.agent.glyph_model import load_glyph_checkpoint


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def closing_batch_counts(batch, expected):
    mask = batch.get('rollout_mask')
    if mask is None or mask.dtype != torch.bool or mask.ndim != 1 or int(mask.sum()) != expected or expected <= 0:
        raise ValueError('closing rollout rows are not active')
    for key in ('won','terminal','lost_life'):
        if bool(batch[key][mask,:3].any()):
            raise ValueError('closing batch has an interior terminal/reset event')
    final = {key: batch[key][mask,3] for key in ('won','terminal','lost_life','distances')}
    unsafe = (final['distances'] < 0) | final['lost_life'] | (final['terminal'] & ~final['won'])
    return {'rows': expected, 'final_won': int(final['won'].sum()),
            'final_unsafe': int(unsafe.sum()), 'final_deadend': int((final['distances'] < 0).sum()),
            'final_lost_life': int(final['lost_life'].sum()), 'final_terminal': int(final['terminal'].sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--config', required=True, help='JSON config for the target architecture')
    parser.add_argument('--glyph-source')
    parser.add_argument('--successor-labels')
    parser.add_argument('--data-cache-dir')
    parser.add_argument('--on-policy-data')
    parser.add_argument('--on-policy-fraction', type=float, default=.25)
    parser.add_argument('--rollout-index')
    parser.add_argument('--closing-rollout-index')
    parser.add_argument('--closing-action-attestation')
    parser.add_argument('--closing-action-sha256')
    parser.add_argument('--successor-policy-weight', type=float, default=0.)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--encoder-chunk-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    size = args.batch_size
    if size < 1 or size > 1024 or size & (size - 1):
        parser.error('batch must be a power of two from 1 through 1024')
    if args.encoder_chunk_size < 1 or args.lr <= 0:
        parser.error('encoder chunk and learning rate must be positive')
    closing = (args.closing_rollout_index, args.closing_action_attestation, args.closing_action_sha256)
    if any(closing) and not all(closing):
        parser.error('closing rollout index, action attestation and SHA256 are required together')
    if args.closing_rollout_index and (args.rollout_index or not args.on_policy_data):
        parser.error('closing rollout requires --on-policy-data and excludes --rollout-index')
    if args.rollout_index and not args.on_policy_data:
        parser.error('--rollout-index requires --on-policy-data')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('native bf16 CUDA is required')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = WorldModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    source_hash = digest(args.source)
    source, _ = load_world_checkpoint(args.source)
    model = WorldPolicy(config)
    migrated = initialize_from_checkpoint(model, source)
    del source
    if args.glyph_source:
        glyph, _ = load_glyph_checkpoint(args.glyph_source)
        initialize_glyph_encoder(model, glyph)
        del glyph
    data = load_dataset(args.train, config.history, args.data_cache_dir)
    require_verified_data(data)
    require_winning_coverage(data)
    if args.successor_labels:
        data = attach_successor_labels(data, args.train, args.successor_labels)
    unique = np.unique(data['seeds'])
    if len(unique) < max(10000, size):
        raise ValueError('preflight requires at least 10000 verified generated training levels')
    rng = np.random.default_rng(args.seed)
    on_policy_count = 0
    if args.on_policy_data:
        from pebby.agent.on_policy_sampling import OnPolicySampler, mixed_batch
        from pebby.agent.on_policy_provenance import validate_on_policy_provenance
        supplemental = load_dataset(args.on_policy_data, config.history, args.data_cache_dir)
        require_verified_data(supplemental)
        require_winning_coverage(supplemental)
        validate_on_policy_provenance(supplemental)
        if args.closing_rollout_index:
            from pebby.agent.world_closing_sequences import load_sidecar, ClosingSampler, closing_mixed_batch
            index = load_sidecar(args.closing_rollout_index, args.on_policy_data, supplemental,
                                 attestation=args.closing_action_attestation,
                                 attestation_sha256=args.closing_action_sha256)
            sampler = ClosingSampler(data, supplemental, index, fraction=args.on_policy_fraction,
                                      start=[.35,.25,.20,.15,.05], end=[.05,.10,.20,.25,.40])
        elif args.rollout_index:
            from pebby.agent.world_sequences import load_sidecar, FourStepSampler, four_step_mixed_batch
            index = load_sidecar(args.rollout_index, args.on_policy_data, supplemental)
            sampler = FourStepSampler(data, supplemental, index, fraction=args.on_policy_fraction,
                                      start=[.35,.25,.20,.15,.05], end=[.05,.10,.20,.25,.40])
        else:
            sampler = OnPolicySampler(data, supplemental, fraction=args.on_policy_fraction,
                                      start=[.35,.25,.20,.15,.05], end=[.05,.10,.20,.25,.40])
        selection = sampler.indices(size, .5, torch.Generator().manual_seed(args.seed))
        selected_batch = (closing_mixed_batch(as_tensors(data), as_tensors(supplemental), selection, index)
                          if args.closing_rollout_index else four_step_mixed_batch(as_tensors(data), as_tensors(supplemental), selection, index)
                          if args.rollout_index else
                          mixed_batch(as_tensors(data), as_tensors(supplemental), selection))
        batch = {name: value.cuda() for name, value in selected_batch.items()}
        assert len(set(sampler.last_level_seeds)) == size
        on_policy_count = sampler.last_on_policy_count
        del supplemental
    else:
        chosen = rng.choice(unique, size=size, replace=False)
        indices = np.asarray([rng.choice(np.flatnonzero(data['seeds'] == seed)) for seed in chosen])
        batch = {key: torch.as_tensor(value[indices]).cuda() for key, value in data.items()
                 if isinstance(value, np.ndarray) and value.ndim and len(value) == len(data['seeds'])}
    levels = len(unique)
    del data
    model = model.cuda().train()
    model.checkpoint_encoder = True
    model.encoder_chunk_size = args.encoder_chunk_size
    optimizer = torch.optim.AdamW(parameter_groups(model, .05), lr=args.lr)
    weights = {'sigreg': .0125, 'grounding': 1., 'glyph': 1.,
               'successor_policy': args.successor_policy_weight}
    report = {'pid': os.getpid(), 'config': model.config(), 'parameters': model.parameter_count(),
              'source_checkpoint': args.source, 'source_sha256': source_hash,
              'batch_size': size, 'batch_unit': 'distinct_generated_training_level',
              'training_levels': levels, 'training_source': args.train,
              'on_policy_data': args.on_policy_data, 'on_policy_rows_per_batch': on_policy_count,
              'on_policy_data_sha256': digest(args.on_policy_data) if args.on_policy_data else None,
              'rollout_index': args.closing_rollout_index or args.rollout_index,
              'rollout_index_sha256': digest(args.closing_rollout_index or args.rollout_index) if args.closing_rollout_index or args.rollout_index else None,
              'rollout_index_metadata': index.meta if args.closing_rollout_index or args.rollout_index else None,
              'closing_action_attestation': args.closing_action_attestation,
              'closing_action_sha256': args.closing_action_sha256,
              'closing_batch': closing_batch_counts(batch, on_policy_count) if args.closing_rollout_index else None,
              'encoder_chunk_size': args.encoder_chunk_size, 'checkpoint_encoder': True,
              'precision': 'bf16', 'gradient_accumulation': False, 'weights': weights,
              'migrated_keys': migrated, 'checkpoint_saved': False, 'steps': [],
              'source_hashes': {path: digest(path) for path in
                                ('pebby/agent/world_model.py', 'pebby/agent/world_training_objectives.py',
                                 'pebby/agent/world_readout.py',
                                 'pebby/agent/world_train.py', 'pebby/agent/on_policy_sampling.py',
                                 'pebby/agent/on_policy_provenance.py',
                                 'pebby/agent/world_sequences.py', 'pebby/agent/world_closing_sequences.py', 'pebby/agent/world_rollout.py',
                                 'pebby/agent/world_cache.py', __file__)},
              'limitations': ['Two disposable optimizer updates establish fit and gradients only.',
                              'No validation or official data; no quality estimate.']}
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = world_losses(model, batch, weights)
        if (args.rollout_index or args.closing_rollout_index) and int(result['diagnostics'].get('rollout_rows', -1)) != on_policy_count:
            raise ValueError('chronological rollout loss is not active')
        result['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                key = name.split('.')[0]
                gradients[key] = gradients.get(key, 0.) + float(parameter.grad.abs().sum())
        optimizer.step()
        torch.cuda.synchronize()
        report['steps'].append({'step': step + 1, 'seconds': time.perf_counter() - started,
                                'gradient_norm_before_clip': float(norm),
                                'gradient_l1_after_clip_by_module': gradients,
                                'losses': {k: float(v.detach()) for k, v in result['losses'].items()},
                                'rollout_diagnostics': {k: float(v.detach()) for k, v in result['diagnostics'].items()
                                                        if k.startswith('rollout_')},
                                'loss': float(result['total'].detach())})
        print(json.dumps(report['steps'][-1]), flush=True)
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    assert digest(args.source) == source_hash, 'source checkpoint changed'
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Completed disposable preflight:', report['parameters'], 'parameters,',
          report['peak_allocated_gib'], 'GiB peak', flush=True)


if __name__ == '__main__':
    main()
