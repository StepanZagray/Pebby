"""Generated-only projector grounding before mandatory joint policy adaptation.

The encoder stays frozen. Normalization is fitted on TRAIN features, used only
while optimizing the projector, and folded back into its first linear layer.
This intermediate checkpoint is not a promoted or adapted gameplay policy.
"""

import argparse
from datetime import datetime
import hashlib
import json
import mmap
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent import world_position_recall as position
from pebby.agent.world_grounding import SIZES
from pebby.agent.world_model import load_world_checkpoint
from tools import train_reference_repair as repair

ROOT = repair.ROOT
NAMES = ('player', 'shape', 'color', 'rotation', 'steps', 'lives')
ARRAYS = ('inputs', 'labels', 'seeds', 'source', 'rows', 'branch', 'terminal', 'lost_life', 'won')
FORMAT = 'pebby_position_recall_warmup_report_v1'


def now():
    return datetime.now().astimezone().isoformat()


def finite(value):
    # torch.isfinite may allocate several full-sized intermediates. The feature
    # matrix is >3 GiB, while a training minibatch is small; keep verification
    # from consuming more GPU memory than the actual optimization.
    for chunk in value.reshape(-1).split(1048576):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError('nonfinite projector warmup value')


def normalize_projector(first, mean, scale):
    """Preserve raw-input outputs while changing its input to (x-mean)/scale."""
    with torch.no_grad():
        first.bias.add_(first.weight @ mean)
        first.weight.mul_(scale[None])


def fold_projector(first, mean, scale):
    """Restore raw-input inference without adding a normalizer to the model."""
    with torch.no_grad():
        first.weight.div_(scale[None])
        first.bias.sub_(first.weight @ mean)


def memory_guard():
    available = repair.memory_check()
    if available < 7 * 2**30:
        raise MemoryError('warmup stops at 7 GiB MemAvailable to preserve 6 GiB reserve')
    if torch.cuda.max_memory_allocated() > 7 * 2**30:
        raise MemoryError('warmup GPU allocation exceeded 7 GiB')
    return available


def score(model, features, labels, populations):
    """Keep observation populations separate; no gameplay inference is implied."""
    sums = {name: torch.zeros(6, 2, dtype=torch.float64, device='cuda') for name in populations}
    counts = {name: int(mask.sum()) for name, mask in populations.items()}
    with torch.no_grad():
        for start in range(0, len(features), 1024):
            target = labels[start:start + 1024]
            output = model.grounding_head(model.projector(features[start:start + 1024]))
            for field, logits in enumerate(output):
                correct = (logits.argmax(-1) == target[:, field]).double()
                ce = F.cross_entropy(logits, target[:, field], reduction='none').double()
                for name, mask in populations.items():
                    selected = mask[start:start + 1024]
                    sums[name][field, 0] += correct[selected].sum()
                    sums[name][field, 1] += ce[selected].sum()
    result = {}
    for name, values in sums.items():
        data = values.cpu().tolist()
        result[name] = {'count': counts[name], 'fields': {
            field: {'accuracy': data[i][0] / counts[name] if counts[name] else None,
                    'cross_entropy': data[i][1] / counts[name] if counts[name] else None}
            for i, field in enumerate(NAMES)}}
    return result


def gate(validation):
    thresholds = dict(player=.90, shape=.95, color=.95, rotation=.95, steps=.90, lives=.95)
    checks = {}
    for population in ('current', 'actual_nonterminal'):
        summary = validation[population]
        checks[population] = {field: (summary['count'] > 0 and
                                     summary['fields'][field]['accuracy'] >= threshold)
                              for field, threshold in thresholds.items()}
    return {'passed': all(v for fields in checks.values() for v in fields.values()),
            'thresholds': thresholds, 'checks': checks,
            'scope': 'frozen public state grounding; mandatory joint adaptation and gameplay remain'}


def unchanged_nonheads(model, parent):
    actual, expected = model.state_dict(), parent.state_dict()
    if set(actual) != set(expected):
        raise ValueError('position model has unexpected weight keys')
    for key in expected:
        if key.startswith(('projector.', 'grounding_head.')):
            continue
        if not torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu()):
            raise ValueError(f'frozen parameter changed: {key}')


def load_features(directory, parent):
    """Bind every feature array independently instead of trusting completion flags."""
    datasets, bindings, stats = {}, {}, {}
    manifests = [directory / 'manifest.json']
    for path in manifests:
        if not path.is_file():
            raise ValueError(f'completed feature cache receipt missing: {path}')
        bindings[str(path)] = repair.digest(path)
    report = json.loads(manifests[0].read_text())
    if (report.get('status') != 'complete' or report.get('parent_sha256') != repair.PARENT_SHA or
            Path(report.get('parent_path', '')).resolve() != repair.PARENT or
            report.get('official_inputs_used') is not False or
            report.get('feature_width') != 1766 or report.get('label_sizes') != list(SIZES)):
        raise ValueError('feature cache must bind the exact fresh generated-only parent and schema')
    for key in ('sources_unchanged', 'policy_gradients_absent', 'finite_features',
                'public_input_only', 'no_engine_or_teacher_encoder_input',
                'learned_player_marginals_detached', 'policy_frozen'):
        if report.get(key) is not True:
            raise ValueError(f'feature cache lacks verified {key}')
    if report.get('source_bindings') != report.get('source_bindings_after'):
        raise ValueError('feature extraction source bindings changed')
    if report.get('large_array_stats_before') != report.get('large_array_stats_after'):
        raise ValueError('feature extraction source array statistics changed')
    for source in ('base', 'supplement', 'validation'):
        parity = report.get('history_parity', {}).get(source, {})
        if any(parity.get(k) is not True for k in ('tokens_match', 'validity_exact',
                                                  'actions_exact', 'features_match')):
            raise ValueError('actual public histories lack verified reset/normal parity')
        if min(parity.get('reset_observations', 0), parity.get('normal_observations', 0)) <= 0:
            raise ValueError('both reset and non-reset history parity required')
    for path, expected in report['source_bindings'].items():
        if repair.digest(path) != expected:
            raise ValueError(f'feature extraction source differs: {path}')
    for source, rows, seeds in (('base', 80000, 10000), ('supplement', 8000, 1000),
                                 ('validation', 4000, 500)):
        selected = report['selection'][source]
        path = directory / selected['path']
        if (selected['rows'] != rows or selected['seeds'] != seeds or
                selected['min_rows_per_seed'] != 8 or selected['max_rows_per_seed'] != 8 or
                repair.digest(path) != selected['sha256']):
            raise ValueError('feature selection differs from fixed eight-row-per-seed plan')
        bindings[str(path)] = selected['sha256']
    for split in ('train', 'validation'):
        arrays = {}
        for name in ARRAYS:
            path = directory / split / (name + '.npy')
            bindings[str(path)] = repair.digest(path)
            stats[str(path)] = repair.SourceGuard.stat(path)
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
            expected = report['outputs'][split][name]
            if (bindings[str(path)] != expected['sha256'] or list(arrays[name].shape) != expected['shape'] or
                    arrays[name].dtype.str != expected['dtype'] or path.stat().st_size != expected['size_bytes']):
                raise ValueError(f'feature array differs from completed manifest: {split}/{name}')
        n = len(arrays['inputs'])
        if n != (440000 if split == 'train' else 20000):
            raise ValueError('feature count differs from fixed source selection')
        if arrays['inputs'].shape != (n, 1766) or arrays['inputs'].dtype != np.float32:
            raise ValueError('expected FP32 public projector inputs with 24 position features')
        if arrays['labels'].shape != (n, 6) or arrays['labels'].dtype != np.int64:
            raise ValueError('expected six categorical state labels')
        if any(arrays[k].shape != (n,) for k in ARRAYS if k not in ('inputs', 'labels')):
            raise ValueError('feature metadata dimensions differ')
        expected = set(parent[split + '_seeds'])
        if set(map(int, np.unique(arrays['seeds']))) != expected:
            raise ValueError('feature cache seed set differs from fresh parent')
        for i, size in enumerate(SIZES):
            values = arrays['labels'][:, i]
            if np.any(values < 0) or np.any(values >= size):
                raise ValueError('feature state labels outside supported categories')
        if np.any(~np.isin(arrays['branch'], [-1, 0, 1, 2, 3])):
            raise ValueError('unknown observation branch')
        if np.any(~np.isin(arrays['source'], [0, 1])) or (split == 'validation' and np.any(arrays['source'])):
            raise ValueError('unknown feature source or supplemental validation input')
        datasets[split] = arrays
    return datasets, bindings, stats, report


def to_gpu(array):
    # Bound host copies to 64K feature rows; do not materialize another 3 GiB
    # host feature matrix while the GPU holds its training copy.
    tensor = torch.empty(array.shape, dtype=torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype,
                         device='cuda')
    for first in range(0, len(array), 65536):
        tensor[first:first + 65536].copy_(torch.from_numpy(np.array(array[first:first + 65536], copy=True)))
        array._mmap.madvise(mmap.MADV_DONTNEED)
        memory_guard()
    return tensor


def populations(arrays):
    branch = torch.from_numpy(np.array(arrays['branch'], copy=True)).cuda()
    terminal = torch.from_numpy(np.array(arrays['terminal'], copy=True)).cuda().bool()
    return {'all': torch.ones(len(branch), dtype=torch.bool, device='cuda'),
            'current': branch == -1, 'actual': branch >= 0,
            'actual_nonterminal': (branch >= 0) & ~terminal,
            'actual_terminal': (branch >= 0) & terminal}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-position-inputs-v1')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.cache, args.out_dir = args.cache.resolve(), args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(format=FORMAT, status='starting', pid=os.getpid(), started_local=now(),
                  source_checkpoint=str(repair.PARENT), source_checkpoint_sha256=repair.PARENT_SHA,
                  batch_size=1024, batch_unit='cached public observations', drop_last=True, epochs=100, seed=42,
                  optimizer='AdamW', learning_rate=.001, weight_decay=.0001,
                  objectives='mean of six state cross-entropies only; encoder frozen',
                  official_inputs_used=False, gameplay_claim=False, history=[],
                  limits=['Warmup changes the latent coordinate system; old dynamics/value/ranker need joint adaptation.',
                          'Shuffled state batches can include multiple observations from a level; no SIGReg is used here.',
                          'Generated validation is a reused development set, not an untouched final test.'])
    checkpoint_path = args.out_dir / 'model.pt'
    output = args.out_dir / 'report.json'
    repair.write(output, report)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('warmup 600-second deadline')))
    signal.alarm(600)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        memory_guard()
        if repair.digest(repair.PARENT) != repair.PARENT_SHA:
            raise ValueError('fresh parent checkpoint changed')
        parent, metadata = load_world_checkpoint(repair.PARENT)
        repair.validate_parent(metadata)
        arrays, cache_hashes, cache_stats, cache_report = load_features(args.cache, metadata)
        source_paths = [Path(__file__).resolve(), Path(position.__file__).resolve(),
                        ROOT / 'pebby/agent/world_model.py', ROOT / 'pebby/agent/world_grounding.py',
                        ROOT / 'tools/train_reference_repair.py']
        code = {str(path): repair.digest(path) for path in source_paths}
        binding = {'path': str(args.cache), 'manifest_sha256': cache_hashes[str(args.cache / 'manifest.json')]}
        report.update(status='loading_features', train_seeds=metadata['train_seeds'],
                      validation_seeds=metadata['validation_seeds'], source_code_bindings=code,
                      feature_cache_binding=binding, consumed_cache_sha256=cache_hashes,
                      base_initialization=metadata['initialization'],
                      upstream_bank_calibration=cache_report['upstream_bank_calibration'])
        repair.write(output, report)
        model = position.PositionRecallPolicy(parent.cfg)
        position.initialize_from_base(model, parent)
        model.requires_grad_(False)
        model.projector.requires_grad_(True)
        model.grounding_head.requires_grad_(True)
        unchanged_nonheads(model, parent)
        model.to('cuda').eval()
        torch.cuda.reset_peak_memory_stats()
        features = {split: to_gpu(data['inputs']) for split, data in arrays.items()}
        labels = {split: to_gpu(data['labels']) for split, data in arrays.items()}
        masks = {split: populations(data) for split, data in arrays.items()}
        for x in features.values():
            finite(x)
        # The raw model before warmup is the exact migrated source. All later
        # state decoding measures use the same frozen public encoder features.
        report['before'] = {split: score(model, features[split], labels[split], masks[split])
                            for split in features}
        mean = features['train'].mean(0)
        scale = features['train'].std(0, correction=0).clamp_min(1e-6)
        finite(mean)
        finite(scale)
        raw_check = features['validation'][:1024].clone()
        with torch.no_grad():
            original_latent = model.projector(raw_check)
        normalize_projector(model.projector[0], mean, scale)
        with torch.no_grad():
            normalized_latent = model.projector((raw_check - mean) / scale)
        torch.testing.assert_close(normalized_latent, original_latent, atol=1e-4, rtol=1e-4)
        for x in features.values():
            x.sub_(mean).div_(scale)
        np.savez(args.out_dir / 'normalization.npz', mean=mean.cpu().numpy(), scale=scale.cpu().numpy())
        parameters = [p for p in model.parameters() if p.requires_grad]
        expected_ids = {id(p) for module in (model.projector, model.grounding_head) for p in module.parameters()}
        assert {id(p) for p in parameters} == expected_ids
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
        generator = torch.Generator(device='cuda').manual_seed(42)
        updates, train_started = 0, time.monotonic()
        report.update(status='training_projector', trainable_parameters=sum(p.numel() for p in parameters),
                      parameters=model.parameter_count(), config=model.config(), optimizer_steps=0)
        repair.write(output, report)
        for epoch in range(1, 101):
            memory_guard()
            model.projector.train()
            model.grounding_head.train()
            order = torch.randperm(len(features['train']), generator=generator, device='cuda')
            epoch_loss = torch.zeros((), device='cuda')
            examples = 0
            for selected in order[:len(order) // 1024 * 1024].split(1024):
                optimizer.zero_grad(set_to_none=True)
                logits = model.grounding_head(model.projector(features['train'][selected]))
                target = labels['train'][selected]
                loss = torch.stack([F.cross_entropy(scores, target[:, i])
                                    for i, scores in enumerate(logits)]).mean()
                loss.backward()
                optimizer.step()
                epoch_loss.add_(loss.detach() * len(selected))
                examples += len(selected)
                updates += 1
            finite(epoch_loss)
            if epoch in (1, 5, 10, 20, 40, 60, 80, 100):
                model.eval()
                entry = {'epoch': epoch, 'optimizer_steps': updates, 'training_cross_entropy': float(epoch_loss / examples),
                         'elapsed_training_seconds': time.monotonic() - train_started,
                         'validation': score(model, features['validation'], labels['validation'], masks['validation'])}
                # Full TRAIN scoring at the endpoints distinguishes decoding
                # failures from a simple failure to fit even the training data.
                if epoch in (1, 100):
                    entry['train'] = score(model, features['train'], labels['train'], masks['train'])
                report['history'].append(entry)
                report['optimizer_steps'] = updates
                repair.write(output, report)
                print(json.dumps({'time': now(), 'epoch': epoch, 'optimizer_steps': updates,
                                  'training_cross_entropy': entry['training_cross_entropy'],
                                  'validation_current': entry['validation']['current']['fields'],
                                  'elapsed_training_seconds': entry['elapsed_training_seconds']}), flush=True)
        model.eval()
        with torch.no_grad():
            normalized_latent = model.projector((raw_check - mean) / scale)
            normalized_logits = model.grounding_head(normalized_latent)
        fold_projector(model.projector[0], mean, scale)
        with torch.no_grad():
            folded_latent = model.projector(raw_check)
            folded_logits = model.grounding_head(folded_latent)
        torch.testing.assert_close(folded_latent, normalized_latent, atol=5e-4, rtol=1e-4)
        for actual, expected in zip(folded_logits, normalized_logits):
            torch.testing.assert_close(actual, expected, atol=5e-4, rtol=1e-4)
        fold_validation = {'passed': True, 'max_latent_absolute_difference':
                           float((folded_latent - normalized_latent).abs().max()),
                           'max_head_absolute_difference': max(float((a - b).abs().max())
                            for a, b in zip(folded_logits, normalized_logits)),
                           'validation_rows': len(raw_check), 'atol': 5e-4, 'rtol': 1e-4}
        unchanged_nonheads(model, parent)
        assert all(p.grad is None for name, p in model.named_parameters()
                   if not name.startswith(('projector.', 'grounding_head.')))
        for p in model.parameters():
            finite(p)
        for path, expected in code.items():
            assert repair.digest(path) == expected
        for path, expected in cache_stats.items():
            assert repair.SourceGuard.stat(path) == expected
        assert repair.digest(args.cache / 'manifest.json') == binding['manifest_sha256']
        assert repair.digest(repair.PARENT) == repair.PARENT_SHA
        raw_validation = to_gpu(arrays['validation']['inputs'])
        folded_validation = score(model, raw_validation, labels['validation'], masks['validation'])
        passed = gate(folded_validation)
        provenance = dict(source_checkpoint=str(repair.PARENT), source_checkpoint_sha256=repair.PARENT_SHA,
                          feature_cache_binding=binding, source_code_bindings=code, normalization_folded=True,
                          base_initialization=metadata['initialization'], report_path=str(output),
                          warmup_epochs=100, batch_size=1024, batch_unit='cached public observations')
        position.save_checkpoint(checkpoint_path, model, warmup_provenance=provenance,
                                 train_seeds=metadata['train_seeds'], validation_seeds=metadata['validation_seeds'],
                                 initialization={'kind': 'fresh_reference_position_grounding_warmup', 'seed': 42},
                                 stage='grounding_warmup_requires_joint_adaptation')
        restored, saved = position.load_checkpoint(checkpoint_path)
        unchanged_nonheads(restored, parent)
        report.update(status='complete', warmup_checkpoint=str(checkpoint_path),
                      warmup_checkpoint_sha256=repair.digest(checkpoint_path),
                      finite_gate_validation=passed, normalizer_fold_validation=fold_validation,
                      normalization_folded=True, only_projector_grounding_updated=True,
                      sources_unchanged=True, frozen_encoder_gradients_absent=True,
                      peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      last_validation=folded_validation,
                      next='joint_adaptation' if passed['passed'] else 'assess_failed_grounding_gate')
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        signal.alarm(0)
        report.update(finished_local=now(), elapsed_seconds=time.monotonic() - started)
        repair.write(output, report)


if __name__ == '__main__':
    main()
