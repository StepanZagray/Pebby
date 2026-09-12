"""Matched static/writable-memory readout experiment; gameplay evaluated separately.

Fixed final checkpoints, no validation/depth selection. Both arms warmstart the
same actor; static has fewer active parameters. Actual fields are training-only.
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.structured_workspace_policy import StructuredWorkspaceReadout
from pebby.agent.structured_factored_policy import state_digest
from pebby.agent.structured_distance import current_distance_labels
from tools.preflight_structured_workspace import load_inputs, backward
from tools.cache_structured_policy_successors import load_imagined_cache
from tools.structured_policy_batch import prepared_policy_inputs
from tools.train_structured_paired_policy import paired_batch, validate_pair
from tools.train_structured_policy import load_policy_cache, check_policy_encoder, policy_terms
from tools.train_structured_transition import sample_rows, digest, atomic_json
from tools.train_structured_distance import diagnostic_rows

FORMAT = 'pebby.structured-workspace-readout.v1'
DEPTHS = (1, 2, 4)


def verify_sources(sources):
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError(f'source changed: {path}')


def draws(data, size, updates, seed=42):
    """Independent reproducible streams; draw ordering never depends on arm."""
    level_rng = np.random.default_rng(seed)
    view_rng = np.random.default_rng(seed + 1)
    depth_rng = np.random.default_rng(seed + 2)
    for step in range(updates):
        rows = sample_rows(data, size, step / max(updates - 1, 1), level_rng)
        view_seed = int(view_rng.integers(0, 2**63 - 1))
        yield rows, view_seed, int(depth_rng.choice(DEPTHS))


def selection_bytes(rows, view, depth):
    return np.asarray(rows, dtype='<i8').tobytes() + np.asarray(view, dtype='i1').tobytes() + bytes([depth])


def load_validation(policy, sources):
    result, manifests = [], []
    directories = ('data/structured-field-16384/validation', 'data/structured-field-additional-state-16384/validation')
    imagined = ('data/structured-policy-imagined-local-h4-400', 'data/structured-policy-imagined-additional-local-h4-400')
    for directory, future in zip(directories, imagined):
        data, manifest = load_policy_cache(directory, 'validation')
        check_policy_encoder(manifest['field_encoder'], policy, True)
        data['imagined_fields'], hashes = load_imagined_cache(future, 'validation', directory, data, manifest, policy)
        sources.update(hashes)
        sources[str(Path(directory) / 'manifest.json')] = digest(Path(directory) / 'manifest.json')
        sources.update({str(Path(directory) / (key + '.npy')): info['sha256'] for key, info in manifest['arrays'].items()})
        result.append(data); manifests.append(manifest)
    identical = all(manifests[0]['arrays'][key]['sha256'] == manifests[1]['arrays'][key]['sha256']
                    for key in manifests[0]['arrays'])
    identical = identical and np.array_equal(result[0]['imagined_fields'], result[1]['imagined_fields'])
    if not identical:
        validate_pair(*result, *manifests, directories[0])
    return result[:1] if identical else result


@torch.no_grad()
def evaluate(head, data, rows, batch_size=16):
    """All sums counted per state; FP32 actual/imagined on identical rows."""
    head.eval()
    totals = {str(depth): {kind: {group: {'count': 0, 'correct': 0, 'ce_sum': 0.}
              for group in ('all', 'distance_ge17')} for kind in ('actual', 'imagined')} for depth in DEPTHS}
    for begin in range(0, len(rows), batch_size):
        ids = rows[begin:begin + batch_size]
        batch = {key: np.ascontiguousarray(data[key][ids]) for key in ('optimal', 'next_fields', 'imagined_fields')}
        inputs, masks = prepared_policy_inputs(batch, 'successors', 'cpu')
        distance = current_distance_labels(*(np.asarray(data[key][ids]) for key in ('optimal', 'distances', 'terminal', 'won', 'lost_life')))
        for depth in DEPTHS:
            for kind, fields in inputs.items():
                terms = policy_terms(head(fields, loops=depth), masks)
                for group, selected in [('all', torch.ones(len(ids), dtype=torch.bool)), ('distance_ge17', distance >= 17)]:
                    rec = totals[str(depth)][kind][group]
                    rec['count'] += int(selected.sum())
                    rec['correct'] += int(terms['correct'][selected].sum())
                    rec['ce_sum'] += float(terms['ce'][selected].double().sum())
    for per_depth in totals.values():
        for per_kind in per_depth.values():
            for rec in per_kind.values():
                rec['accuracy'] = rec['correct'] / rec['count'] if rec['count'] else None
                rec['ce'] = rec['ce_sum'] / rec['count'] if rec['count'] else None
    return totals


def save_head(path, head, sources, provenance, actor_checkpoint='checkpoints/ls20-structured-policy-paired-local-h4-600.pt'):
    path = Path(path)
    if path.exists(): raise ValueError('checkpoint exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    record = dict(format=FORMAT, config=head.config(), parameters=head.parameter_count(),
                  active_trainable_parameters=head.trainable_parameter_count(), weights=weights,
                  actor_checkpoint=actor_checkpoint, actor_sha256=sources[actor_checkpoint],
                  source_unchanged=True, official_inputs_used=False, state_sha256=state_digest(weights),
                  sources=sources, training_provenance=provenance)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temp.open('xb') as stream:
            torch.save(record, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    loaded, _ = load_head(path)
    for key, value in weights.items():
        if not torch.equal(value, loaded.state_dict()[key]): raise ValueError('reload weight mismatch')
    return {'path': str(path), 'sha256': digest(path), 'strict_reload_exact': True}


def load_head(path):
    raw = Path(path).read_bytes()
    saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT: raise ValueError('wrong workspace format')
    verify_sources(saved['sources'])
    if saved.get('official_inputs_used') is not False or saved.get('source_unchanged') is not True:
        raise ValueError('invalid source attestation')
    if state_digest(saved['weights']) != saved.get('state_sha256'):
        raise ValueError('weight digest mismatch')
    if (not isinstance(saved.get('actor_sha256'), str) or len(saved['actor_sha256']) != 64
            or saved.get('actor_sha256') != saved['sources'].get(saved.get('actor_checkpoint'))):
        raise ValueError('actor binding mismatch')
    model = StructuredWorkspaceReadout(saved['config'])
    model.load_state_dict(saved['weights'], strict=True)
    if model.parameter_count() != saved['parameters'] or model.trainable_parameter_count() != saved['active_trainable_parameters']:
        raise ValueError('workspace parameter count mismatch')
    if digest(path) != hashlib.sha256(raw).hexdigest(): raise ValueError('checkpoint changed while loading')
    return model.eval(), saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--updates', type=int, default=600)
    p.add_argument('--seconds', type=int, default=2400)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--checkpoint-prefix', type=Path, required=True)
    p.add_argument('--preflight', type=Path, default=Path('artifacts/structured-workspace-preflight-gpu.json'))
    args = p.parse_args()
    if args.report.exists() or any(Path(str(args.checkpoint_prefix) + '-' + arm + '.pt').exists() for arm in ('static', 'evolving')):
        p.error('outputs must not exist')
    if not 2 <= args.batch_size <= 1024 or args.batch_size & (args.batch_size - 1): p.error('power-of-two batch2..1024 required')
    if not 1 <= args.seconds <= 2400: p.error('seconds must be1..2400')
    if args.smoke:
        if args.device != 'cpu' or args.batch_size > 8 or args.updates != 2: p.error('smoke requires CPU batch<=8 updates2')
    elif args.updates != 600 or args.batch_size != 1024:
        p.error('production comparison fixed600 updates B1024; fallback requires a reviewed new plan')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('comparison deadline')))
    signal.alarm(args.seconds)
    report = {'status': 'loading', 'pid': os.getpid(), 'smoke': args.smoke, 'source': 'generated_only',
              'official_inputs_used': False, 'primary_depth': 2, 'trained_depths': list(DEPTHS),
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'precision': 'training CUDA BF16 or CPU FP32; evaluation CPU FP32; TF32 off',
              'limits': ['Public factory is available; this training report contains no gameplay result.', 'Static has fewer active parameters.',
                         'Fixed final steps; no validation or depth selection.', 'Smoke caps each diagnostic at8 rows.'], 'arms': {}}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.report, report)
    persist()
    try:
        policy, views, sources = load_inputs()
        validations = load_validation(policy, sources)
        report['validation_views_identical'] = len(validations) == 1
        report['validation_distinct_levels'] = len(validations[0]['seeds'])
        if any(set(map(int, v['seeds'])) & set(map(int, w['seeds'])) for v in views for w in validations):
            raise ValueError('train/validation seed overlap')
        for path in (__file__, 'tools/train_structured_distance.py', 'pebby/agent/structured_distance.py',
                     'pebby/agent/structured_workspace_controller.py', 'pebby/agent/model.py'):
            sources[str(path)] = digest(path)
        if not args.smoke:
            capacity = json.loads(args.preflight.read_text())
            if (capacity.get('status') != 'complete' or capacity.get('batch_size', 0) < args.batch_size
                    or capacity.get('args', {}).get('device') != 'cuda'
                    or not capacity.get('source_unchanged') or not capacity.get('neutral_cpu_parity_exact')):
                raise ValueError('completed fitting neutral GPU preflight required')
            for bound in ('pebby/agent/structured_workspace_policy.py',
                          'checkpoints/ls20-structured-policy-paired-local-h4-600.pt'):
                if capacity.get('sources', {}).get(bound) != sources.get(bound):
                    raise ValueError('preflight architecture/initializer mismatch')
            sources[str(args.preflight)] = digest(args.preflight)
            report['preflight'] = {'path': str(args.preflight), 'sha256': sources[str(args.preflight)]}
        verify_sources(sources)
        report['sources'] = sources
        train_rows = diagnostic_rows(views[0], 8 if args.smoke else 1024)
        val_rows = np.arange(min(8, len(validations[0]['seeds'])) if args.smoke else len(validations[0]['seeds']))
        report['diagnostic_rows'] = {'train': train_rows.tolist(), 'validation': val_rows.tolist(), 'train_seed': 20260912}
        reference = None
        for arm in ('static', 'evolving'):
            torch.manual_seed(42)
            head = StructuredWorkspaceReadout.from_readout(policy.readout, evolving=arm == 'evolving', checkpoint_workspace=True).to(args.device).train()
            optimizer = torch.optim.AdamW((q for q in head.parameters() if q.requires_grad), lr=.0003, weight_decay=.01)
            entry = {'status': 'training', 'parameters': head.parameter_count(), 'active_parameters': head.trainable_parameter_count(), 'progress': []}
            report['arms'][arm] = entry
            hasher = hashlib.sha256(); seen = set(); depth_draws = {str(d): 0 for d in DEPTHS}
            difficulty_draws = np.zeros(5, np.int64); view_draws = np.zeros(2, np.int64)
            for step, (rows, view_seed, depth) in enumerate(draws(views[0], args.batch_size, args.updates)):
                tick = time.monotonic()
                batch, which = paired_batch(views, rows, np.random.default_rng(view_seed), 'successors')
                hasher.update(selection_bytes(rows, which, depth))
                optimizer.zero_grad(set_to_none=True)
                losses = backward(head, batch, args.device, depth)
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                seen.update(map(int, batch['seeds'])); depth_draws[str(depth)] += 1
                difficulty_draws += np.bincount(batch['difficulties'], minlength=6)[1:6]
                view_draws += np.bincount(which, minlength=2)
                if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.updates:
                    if args.device == 'cuda': torch.cuda.synchronize()
                    entry.update(completed_updates=step + 1, selection_sha256=hasher.hexdigest(), distinct_levels_seen=len(seen),
                                 depth_draws=depth_draws, difficulty_draws=difficulty_draws.tolist(), view_draws=view_draws.tolist())
                    entry['progress'].append({'step': step + 1, 'seconds': time.monotonic() - tick, 'losses': losses, 'gradient_norm': float(norm)})
                    report['status'] = arm + '_training'; persist(); print(json.dumps({'arm': arm, **entry['progress'][-1]}), flush=True)
                del batch
            if reference is not None and reference != hasher.hexdigest(): raise ValueError('arm selection streams differ')
            reference = hasher.hexdigest()
            del optimizer
            head.to('cpu').eval(); gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()
            entry['status'] = 'evaluating'; report['status'] = arm + '_evaluation'; persist()
            entry['evaluation'] = {}
            for split, banks, ids in [('train', views, train_rows), ('validation', validations, val_rows)]:
                entry['evaluation'][split] = [evaluate(head, bank, ids) for bank in banks]
            verify_sources(sources)
            entry['checkpoint'] = save_head(str(args.checkpoint_prefix) + '-' + arm + '.pt', head, sources,
                {'arm': arm, 'updates': args.updates, 'batch_size': args.batch_size, 'seed': 42, 'depths': list(DEPTHS),
                 'selection_sha256': hasher.hexdigest(), 'primary_depth': 2, 'smoke': args.smoke, 'fixed_final': True,
                 'source': 'generated_only', 'official_inputs_used': False, 'encoder_and_dynamics_frozen': True,
                 'trained_depths': [int(depth) for depth, count in depth_draws.items() if count],
                 'depth_draws': depth_draws, 'final_update': args.updates,
                 'level_seed': 42, 'view_seed': 43, 'depth_seed': 44})
            entry['status'] = 'complete'; persist(); del head; gc.collect()
        verify_sources(sources)
        report.update(status='complete', sources_unchanged=True, paired_selections_exact=True)
    except BaseException as error:
        report.update(status='failed', error=str(error)); raise
    finally:
        signal.alarm(0); persist()


if __name__ == '__main__': main()
