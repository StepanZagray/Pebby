"""Controlled generated-only H1 glyph ablation; no policy or official inputs.

Three independently runnable arms share data, initialization of existing weights,
sample order, optimizer and fixed update count. Only global-balanced changes the
representation. Both balanced arms add the SAME predicted-readout glyph loss.
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

from pebby.agent.structured_objective import objective
from pebby.agent.structured_transition import StructuredTransition, steps_targets
from tools.train_structured_policy import load_policy_cache
from tools.train_structured_transition import (
    atomic_json, autocast, branch_batch, digest, sample_rows, training_scale,
)


def check_initial_encoder(initial, encoder):
    manifests = initial.get('cache_manifests', {})
    keys = ('config', 'sources', 'parameter_counts', 'code_hashes')
    if not manifests or any(not encoder.get(key) for key in keys):
        raise ValueError('complete initialization and cache encoder provenance required')
    for manifest in manifests.values():
        bound = manifest.get('field_encoder', {})
        if any(bound.get(key) != encoder[key] for key in keys):
            raise ValueError('initialization was trained with a different field encoder')


def evaluation_rows(data, limit, seed):
    """Approximately balanced across available difficulties, filling sparse groups."""
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(np.flatnonzero(data['difficulties'] == difficulty)))
              for difficulty in range(1, 6)]
    chosen = []
    limit = min(limit, len(data['seeds']))
    while len(chosen) < limit:
        progressed = False
        for group in groups:
            if group and len(chosen) < limit:
                chosen.append(group.pop())
                progressed = True
        if not progressed:
            raise ValueError('evaluation rows contain unsupported difficulties')
    return np.asarray(chosen, dtype=np.int64)


def make_model(initial, arm, seed, device):
    torch.manual_seed(seed)
    if arm == 'local-balanced':
        from pebby.agent.structured_local_glyph import LocalGlobalGlyphTransition
        if initial['format'] != 'pebby.structured-transition-global-glyph.v1':
            raise ValueError('local comparison requires global glyph initialization')
        model = LocalGlobalGlyphTransition(initial['config'] | {'variant': 'local_global_glyph'})
        model.warmstart_from_global_state_dict(initial['weights'])
    elif arm == 'global-balanced':
        from pebby.agent.structured_global_glyph import GlobalGlyphTransition
        model = GlobalGlyphTransition(initial['config'])
        if initial['format'] == 'pebby.structured-transition.v1':
            model.warmstart_from_base_state_dict(initial['weights'])
        elif initial['format'] == 'pebby.structured-transition-global-glyph.v1':
            model.load_state_dict(initial['weights'], strict=True)
        else:
            raise ValueError('unsupported global initialization')
    else:
        if initial['format'] != 'pebby.structured-transition.v1':
            raise ValueError('base comparison requires base initialization')
        model = StructuredTransition(initial['config'])
        model.load_state_dict(initial['weights'], strict=True)
    return model.to(device).train()


def losses(model, batch, scale, positive, arm, direct_glyph_weight=0.):
    from tools.structured_glyph_diagnostics import balanced_glyph_loss
    result = objective(model, *batch, scale, pos_weight=positive)
    extra = balanced_glyph_loss(result['readouts']['predicted'],
                                batch[3]['triple'], batch[3]['next_triple'])
    total = result['total'] + (extra if arm != 'baseline' else 0)
    if direct_glyph_weight:
        if arm not in ('global-balanced', 'local-balanced'):
            raise ValueError('direct categorical glyph supervision requires global variant')
        direct = result['output']['glyph_logits']
        direct_readout = {f'carried_{name}_logits': direct[name]
                          for name in ('shape', 'color', 'rotation')}
        direct_loss = balanced_glyph_loss(direct_readout, batch[3]['triple'], batch[3]['next_triple'])
        total = total + direct_glyph_weight*direct_loss
        result['direct_glyph_balanced_ce'] = direct_loss
    return total, result, extra


def preflight(initial, train, scale, positive, args):
    limit = min(args.max_batch, len(train['seeds']))
    size = 2 ** (limit.bit_length() - 1)
    attempts = []
    while size:
        model = optimizer = batch = result = extra = total = None
        try:
            model = make_model(initial, args.arm, args.seed, args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            batch = branch_batch(train, np.arange(size), np.arange(size) % 4, args.device)
            with autocast(args.device):
                total, result, extra = losses(model, batch, scale, positive, args.arm, args.direct_glyph_weight)
            if not bool(torch.isfinite(total)):
                raise ValueError('nonfinite preflight loss')
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            attempts.append({'batch_size': size, 'status': 'fits',
                             'seconds': time.monotonic()-started,
                             'loss': float(total.detach()), 'gradient_norm': float(norm),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated()
                             if args.device == 'cuda' else None})
            return size, attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({'batch_size': size, 'status': 'out_of_memory'})
            size //= 2
        finally:
            del model, optimizer, batch, result, extra, total
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    raise RuntimeError('no power of two batch fits')


@torch.no_grad()
def evaluate(model, data, rows, device, batch_size):
    """All four actions, conditional glyph counts and independent physical heads."""
    from tools.structured_glyph_diagnostics import GlyphDiagnosticAccumulator
    model.eval()
    glyph = GlyphDiagnosticAccumulator()
    field_glyph = GlyphDiagnosticAccumulator()
    counts = dict(branches=0, predicted_player=0, actual_player=0,
                  predicted_budget=0, actual_budget=0)
    carried = dict(values=0, outside_probability_range=0, token_range_sum=0.,
                   distributions=0)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start+batch_size]
        for action in range(4):
            field, following, actions, labels = branch_batch(
                data, chunk, np.full(len(chunk), action, dtype=np.int64), device)
            with autocast(device):
                output = model(field, actions)
                actual = model.readout(following)
            predicted = output['readout']
            glyph.update(predicted, labels['triple'], labels['next_triple'],
                         current_fields=field, player_cell=labels['player_cell'], actions=actions)
            target_player = labels['next_player_cell'][:, 1]*12 + labels['next_player_cell'][:, 0]
            target_budget = steps_targets(labels['next_steps'])
            for name, readout in (('predicted', predicted), ('actual', actual)):
                counts[name+'_player'] += int((readout['player_logits'].argmax(-1) == target_player).sum())
                counts[name+'_budget'] += int((readout['steps_logits'].argmax(-1) == target_budget).sum())
            counts['branches'] += len(chunk)
            values = output['field'][..., 70:84].float()
            means = values.mean(1)
            field_glyph.update(
                dict(zip(('carried_shape_logits', 'carried_color_logits', 'carried_rotation_logits'),
                         means.split((6, 4, 4), -1))),
                labels['triple'], labels['next_triple'], current_fields=field,
                player_cell=labels['player_cell'], actions=actions)
            carried['values'] += values.numel()
            carried['outside_probability_range'] += int(((values < 0) | (values > 1)).sum())
            carried['token_range_sum'] += float((values.amax(1)-values.amin(1)).sum())
            carried['distributions'] += len(chunk)*14
    return {'levels': len(rows), 'glyph': glyph.summary(), 'physical_counts': counts,
            'field_glyph_argmax': field_glyph.summary(),
            'field_glyph_note': 'Argmax of mean carried channels; base residual values are scores, not probabilities.',
            'carried_field': carried,
            'scope': 'H1 generated transitions only; no control or multi-step claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('baseline', 'balanced', 'global-balanced', 'local-balanced'), required=True)
    parser.add_argument('--cache', default='data/structured-field-16384')
    parser.add_argument('--initialize', default='checkpoints/ls20-structured-mixed-h4-800.pt')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--max-batch', type=int, default=1024)
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--train-eval-levels', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--direct-glyph-weight', type=float, default=0.,
                        help='Additional balanced CE on the global state logits themselves (global variant only)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_batch <= 1024 or args.max_batch & (args.max_batch-1):
        parser.error('max batch must be power of two in1..1024')
    if (not 1 <= args.updates <= 1000 or not 1 <= args.seconds <= 1800
            or args.eval_batch < 1 or args.train_eval_levels < 1
            or not np.isfinite(args.lr) or args.lr <= 0):
        parser.error('positive bounded updates, time, evaluation sizes and learning rate required')
    if Path(args.report).exists() or Path(args.checkpoint).exists():
        parser.error('refusing existing outputs')
    if (not np.isfinite(args.direct_glyph_weight) or args.direct_glyph_weight < 0
            or (args.direct_glyph_weight and args.arm not in ('global-balanced', 'local-balanced'))):
        parser.error('nonnegative direct-glyph-weight requires a global glyph variant when nonzero')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('PID', os.getpid(), flush=True)
    started = time.monotonic()
    report = {'status': 'running', 'pid': os.getpid(), 'args': vars(args), 'training': [],
              'official_inputs_used': False, 'policy_integrated': False,
              'precision': 'BF16 autocast on CUDA; FP32 CPU; TF32 disabled',
              'limitations': ['One current state per distinct level; four available branches.',
                             'Launcher geometry is not explicit in semantic channels; latent visual features may carry it.',
                             'Action-target role partitions use predicted visible roles, not engine truth.',
                             'H1 improvement alone does not establish H4 fidelity or gameplay skill.']}
    atomic_json(args.report, report)
    def expired(*_):
        raise TimeoutError('bounded glyph ablation deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    try:
        train, tm = load_policy_cache(Path(args.cache)/'train', 'train')
        val, vm = load_policy_cache(Path(args.cache)/'validation', 'validation')
        if tm['field_encoder'] != vm['field_encoder'] or np.intersect1d(train['seeds'], val['seeds']).size:
            raise ValueError('cache encoder mismatch or split leakage')
        initial_sha = digest(args.initialize)
        initial = torch.load(args.initialize, map_location='cpu', weights_only=True)
        if digest(args.initialize) != initial_sha:
            raise ValueError('initial checkpoint changed while loading')
        if initial.get('format') not in ('pebby.structured-transition.v1', 'pebby.structured-transition-global-glyph.v1'):
            raise ValueError('base or global structured transition initialization required')
        check_initial_encoder(initial, tm['field_encoder'])
        sources = {args.initialize: initial_sha}
        for split, manifest in (('train', tm), ('validation', vm)):
            directory = Path(args.cache)/split
            sources[str(directory/'manifest.json')] = digest(directory/'manifest.json')
            sources.update({str(directory/(name+'.npy')): info['sha256']
                            for name, info in manifest['arrays'].items()})
        for group in ('code_hashes', 'checkpoint_hashes'):
            for file, expected in tm['field_encoder'].get(group, {}).items():
                if digest(file) != expected:
                    raise ValueError('encoder provenance source drift')
                sources[file] = expected
        files = [__file__, 'tools/train_structured_policy.py', 'tools/train_structured_transition.py',
                 'tools/structured_glyph_diagnostics.py', 'pebby/agent/structured_transition.py',
                 'pebby/agent/structured_objective.py']
        if args.arm in ('global-balanced', 'local-balanced'):
            files.append('pebby/agent/structured_global_glyph.py')
        if args.arm == 'local-balanced':
            files.append('pebby/agent/structured_local_glyph.py')
        for file in files:
            sources[str(file)] = digest(file)
        for file in ('pebby/agent/structured_transition.py', 'pebby/agent/structured_objective.py'):
            if initial.get('sources', {}).get(file) != digest(file):
                raise ValueError('initial model implementation drift')
        if initial['format'] == 'pebby.structured-transition-global-glyph.v1':
            file = 'pebby/agent/structured_global_glyph.py'
            if initial.get('sources', {}).get(file) != digest(file):
                raise ValueError('initial global model implementation drift')
        scale = torch.tensor(training_scale(train), device=args.device)
        flags = np.stack([train[key] for key in ('lost_life', 'terminal', 'won')], -1)
        positives = flags.sum((0, 1))
        positive = torch.tensor(np.minimum((flags.shape[0]*4-positives)/np.maximum(positives, 1), 20),
                                device=args.device, dtype=torch.float32).clamp_min(1)
        report.update(sources=sources, train_levels=len(train['seeds']), validation_levels=len(val['seeds']),
                      feature_scale=scale.tolist(), event_positive_weights=positive.tolist())
        size, attempts = preflight(initial, train, scale, positive, args)
        report.update(batch_size=size, batch_probe=attempts)
        atomic_json(args.report, report)
        if args.preflight_only:
            report['status'] = 'preflight_complete'
        else:
            model = make_model(initial, args.arm, args.seed, args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            rng = np.random.default_rng(args.seed)
            seen = set()
            for step in range(args.updates):
                rows = sample_rows(train, size, step/max(args.updates-1, 1), rng)
                actions = rng.integers(0, 4, size=size)
                batch = branch_batch(train, rows, actions, args.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(args.device):
                    total, result, extra = losses(model, batch, scale, positive, args.arm, args.direct_glyph_weight)
                if not bool(torch.isfinite(total)):
                    raise ValueError('nonfinite training loss')
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                seen.update(map(int, train['seeds'][rows]))
                if step == 0 or (step+1) % 20 == 0 or step+1 == args.updates:
                    row = {'step': step+1, 'loss': float(total.detach()),
                           'balanced_glyph_ce': float(extra.detach()), 'gradient_norm': float(norm),
                           'distinct_train_levels_seen': len(seen), 'elapsed_seconds': time.monotonic()-started}
                    if 'direct_glyph_balanced_ce' in result:
                        row['direct_glyph_balanced_ce'] = float(result['direct_glyph_balanced_ce'].detach())
                    report['training'].append(row)
                    atomic_json(args.report, report)
                    print(json.dumps(row), flush=True)
                del total, result, extra, batch
            if any(digest(file) != sha for file, sha in sources.items()):
                raise ValueError('source changed during experiment')
            checkpoint = {'format': getattr(model, 'checkpoint_format', 'pebby.structured-transition.v1'),
                          'config': model.config(), 'weights': model.state_dict(), 'sources': sources,
                          'cache_manifests': {'train': tm, 'validation': vm},
                          'feature_scale': scale.cpu(), 'event_positive_weights': positive.cpu(),
                          'updates': args.updates, 'batch_size': size, 'seed': args.seed,
                          'arm': args.arm, 'parameters': model.parameter_count(),
                          'direct_glyph_weight': args.direct_glyph_weight,
                          'seen_train_seeds': sorted(seen), 'official_inputs_used': False,
                          'policy_integrated': False, 'fixed_final_no_validation_selection': True}
            target = Path(args.checkpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+f'.{os.getpid()}.tmp')
            torch.save(checkpoint, temporary)
            os.replace(temporary, target)
            report.update(status='evaluation_running', parameters=model.parameter_count(),
                          checkpoint_sha256=digest(target), distinct_train_levels_seen=len(seen),
                          seen_train_seeds=sorted(seen), completed_updates=args.updates)
            atomic_json(args.report, report)
            report['validation'] = evaluate(model, val, np.arange(len(val['seeds'])), args.device, args.eval_batch)
            # Fixed difficulty-stratified TRAIN sample, independent of validation and training RNG.
            chosen = evaluation_rows(train, args.train_eval_levels, args.seed+1)
            report['train_evaluation'] = evaluate(model, train, chosen, args.device, args.eval_batch)
            report['train_evaluation']['seeds'] = train['seeds'][chosen].tolist()
            report['status'] = 'complete'
        if any(digest(file) != sha for file, sha in sources.items()):
            raise ValueError('source changed during experiment')
        report['source_unchanged'] = True
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(args.report, report)


if __name__ == '__main__':
    main()
