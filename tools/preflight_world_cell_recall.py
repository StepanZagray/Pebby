"""Two disposable B1024 training updates; test memory and wiring, not quality."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.world_cell_recall import initialize_cell_encoder, load_cell_source
from pebby.agent.world_model import (WorldPolicy, initialize_from_checkpoint, load_world_checkpoint, parameter_groups)
from pebby.agent.world_training_objectives import world_losses
from pebby.agent.world_train import load_dataset, require_verified_data, require_winning_coverage
from tools.train_cell_appearance import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('checkpoints/ls20-world-k4-b1024.epoch1.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--closing', action='store_true', help='Use the matched half closing-sequence batch')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600-second preflight limit')))
    signal.alarm(600)
    print('PID', os.getpid(), flush=True)
    torch.set_num_threads(1); torch.manual_seed(9)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('native CUDA bf16 required')
    started = time.monotonic()
    train_path = Path('data/ls20-world-combined-train.npz')
    cell_path = Path('checkpoints/cell-appearance-2k-400.pt')
    files = [args.source, train_path, cell_path, Path(__file__),
             *(Path('pebby/agent') / name for name in ('world_model.py', 'world_training_objectives.py', 'world_train.py',
                 'world_cell_recall.py', 'cell_appearance.py', 'cell_appearance_dense.py'))]
    if args.closing:
        files += [Path(path) for path in ('data/ls20-world-onpolicy-aggregate2-train.npz',
                  'data/ls20-world-onpolicy-aggregate2-closing-k4.npz',
                  'artifacts/world-closing-actions-batch1.json',
                  'pebby/agent/world_closing_sequences.py', 'pebby/agent/world_rollout.py')]
    hashes = {str(path): digest(path) for path in files}
    source, _ = load_world_checkpoint(args.source)
    model = WorldPolicy(dict(source.config(), cell_recall=True))
    initialize_from_checkpoint(model, source)
    del source
    encoder, provenance = load_cell_source(cell_path, 'data/ls20-visible-cell-labels-2k.npz',
                                          'artifacts/world-visible-cell-labels-2k-proof.json', [])
    initialize_cell_encoder(model, encoder); del encoder
    data = load_dataset(train_path, model.cfg.history, 'data/world-array-cache')
    require_verified_data(data); require_winning_coverage(data)
    unique = np.unique(data['seeds'])
    if len(unique) < 10000 or np.any((unique < 0) | (unique >= 1_000_000)):
        raise ValueError('need 10k+ verified generated training levels')
    rng = np.random.default_rng(9)
    seeds = rng.choice(unique, size=1024, replace=False)
    if args.closing:
        from pebby.agent.world_train import as_tensors
        from pebby.agent.on_policy_provenance import validate_on_policy_provenance
        from pebby.agent.world_closing_sequences import load_sidecar, ClosingSampler, closing_mixed_batch
        supplemental_path = 'data/ls20-world-onpolicy-aggregate2-train.npz'
        supplemental = load_dataset(supplemental_path, model.cfg.history, 'data/world-array-cache')
        require_verified_data(supplemental); require_winning_coverage(supplemental)
        validate_on_policy_provenance(supplemental)
        index = load_sidecar('data/ls20-world-onpolicy-aggregate2-closing-k4.npz', supplemental_path,
                            supplemental, attestation='artifacts/world-closing-actions-batch1.json',
                            attestation_sha256='eeb73a76c9db26e3b23cf4961be822c2cee4c346067c73c37ee3f89e051290d0')
        sampler = ClosingSampler(data, supplemental, index, fraction=.5,
                                 start=[.35, .25, .20, .15, .05], end=[.05, .10, .20, .25, .40])
        selection = sampler.indices(1024, .5, torch.Generator().manual_seed(8))
        batch = {key: value.cuda() for key, value in closing_mixed_batch(
            as_tensors(data), as_tensors(supplemental), selection, index).items()}
        seeds = sampler.last_level_seeds
        if sampler.last_on_policy_count != 512:
            raise ValueError('expected 512 closing rows')
        del supplemental
    else:
        rows = np.array([rng.choice(np.flatnonzero(data['seeds'] == seed)) for seed in seeds])
        batch = {name: torch.as_tensor(value[rows]).cuda() for name, value in data.items()
                 if isinstance(value, np.ndarray) and value.ndim and len(value) == len(data['seeds'])}
    if len(set(map(int, seeds))) != 1024:
        raise ValueError('batch must contain 1024 distinct generated training levels')
    del data
    model = model.cuda().train()
    model.checkpoint_encoder = True; model.encoder_chunk_size = 128
    optimizer = torch.optim.AdamW(parameter_groups(model, .05), lr=.0003)
    frozen = {key: value.detach().clone() for key, value in model.cell_appearance.state_dict().items()}
    report = dict(status='running', pid=os.getpid(), source_hashes=hashes, cell_source=provenance,
                  config=model.config(), parameters=model.parameter_count(),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  batch_size=1024, distinct_levels_in_batch=len(set(map(int, seeds))),
                  closing_rows=512 if args.closing else 0,
                  training_levels=len(unique), gradient_accumulation=False, precision='bf16',
                  checkpoint_saved=False, steps=[],
                  limitations=['Memory and gradient validation only; updated weights are discarded.',
                               'Appearance decoder still has initial-state-only training.'])
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = world_losses(model, batch, {'sigreg': .0125, 'grounding': 1., 'glyph': 1.})
        if args.closing and int(output['diagnostics'].get('rollout_rows', -1)) != 512:
            raise ValueError('closing rollout loss inactive')
        output['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        context_grad = float(model.cell_context.weight.grad.abs().sum())
        if context_grad <= 0 or any(p.grad is not None for p in model.cell_appearance.parameters()):
            raise ValueError('appearance recall gradient/freeze contract violated')
        optimizer.step(); torch.cuda.synchronize()
        report['steps'].append(dict(step=step + 1, seconds=time.monotonic() - tick,
                                    loss=float(output['total'].detach()), gradient_norm=float(norm),
                                    cell_context_gradient_l1=context_grad))
        print(json.dumps(report['steps'][-1]), flush=True)
    for key, value in model.cell_appearance.state_dict().items():
        if not torch.equal(value, frozen[key]):
            raise ValueError('frozen decoder changed')
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError('preflight inputs changed during execution')
    report.update(status='complete', peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  elapsed_seconds=time.monotonic() - started, decoder_unchanged=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    signal.alarm(0)
    print(json.dumps({key: report[key] for key in ('status', 'parameters', 'trainable_parameters',
          'peak_allocated_gib', 'peak_reserved_gib', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
