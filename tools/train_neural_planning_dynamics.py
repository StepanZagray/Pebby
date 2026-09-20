"""Repair recurrent predictions using generated model-selected action sequences.

This updates dynamics only. It emits a dynamics checkpoint, not a promoted
controller. Train and validation roots remain separated by generated level.
"""
import argparse
import copy
import hashlib
from datetime import datetime
import json
import os
import signal
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from pebby.agent.neural_imagination_policy import load_checkpoint
from pebby.agent.neural_planning_objective import planning_sequence_loss, EVENTS
from pebby.agent.structured_factored_policy import state_digest
from pebby.agent.structured_transition import steps_targets
from tools.train_navigation_probe import Budget, digest, write_json

FORMAT = 'pebby.neural-planning-dynamics.v1'
LABELS = ('next_player_cell', 'next_triple', 'next_steps', 'next_lives', *EVENTS)
KEYS = ('fields', 'actions', 'next_fields', 'transition_valid', 'next_field_valid', 'seeds', 'difficulties', *LABELS)


SCHEMA = {
    'fields': ('float32', (148, 96)), 'actions': ('int64', (4, 4)),
    'next_fields': ('float32', (4, 4, 148, 96)),
    'transition_valid': ('bool', (4, 4)), 'next_field_valid': ('bool', (4, 4)),
    'seeds': ('int64', ()), 'difficulties': ('int8', ()),
    'next_player_cell': ('int16', (4, 4, 2)), 'next_triple': ('int16', (4, 4, 3)),
    'next_steps': ('int16', (4, 4)), 'next_lives': ('int16', (4, 4)),
    **{name: ('bool', (4, 4)) for name in EVENTS},
}


def guarded_digest(path, guard):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1024 * 1024):
            guard()
            value.update(block)
    guard()
    return value.hexdigest()


def reserve_allocation(size, guard):
    guard()
    available = next(int(line.split()[1]) * 1024 for line in
                     Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available - size < getattr(guard, 'reserve', 7 * 2**30):
        raise MemoryError('sequence allocation would breach host memory reserve')


def validate_sequences(data, split):
    count = len(data['seeds'])
    for key, (dtype, tail) in SCHEMA.items():
        if data[key].shape != (count, *tail) or data[key].dtype != np.dtype(dtype):
            raise ValueError('fixed four-branch K4 sequence schema required: ' + key)
    low, high = (0, 1_000_000) if split == 'train' else (1_000_000, 2_000_000)
    if not count or not ((data['seeds'] >= low) & (data['seeds'] < high)).all():
        raise ValueError('invalid generated split namespace')
    if not np.isin(data['difficulties'], np.arange(1, 8)).all():
        raise ValueError('difficulty must be in 1..7')
    for seed in np.unique(data['seeds']):
        if len(np.unique(data['difficulties'][data['seeds'] == seed])) != 1:
            raise ValueError('one level has conflicting difficulty labels')
    actions = data['actions']
    if ((actions < 0) | (actions > 3)).any() or not (actions[:, :, 0] == np.arange(4)).all():
        raise ValueError('four canonical first actions and actions in 0..3 required')
    valid, observed = data['transition_valid'], data['next_field_valid']
    if (not valid[..., 0].all() or (valid[..., 1:] & ~valid[..., :-1]).any()
            or (observed & ~valid).any() or ((~observed) & valid & ~data['terminal']).any()):
        raise ValueError('invalid transition/field masks')
    if ((data['won'] & ~data['terminal'] & valid).any()
            or (data['terminal'][..., :-1] & valid[..., 1:]).any()):
        raise ValueError('win must terminate; terminal tails must be masked')
    for name in EVENTS:
        if (data[name] & ~valid).any():
            raise ValueError('event labels in unexecuted tails')
    for key in ('fields', 'next_fields'):
        if not np.isfinite(data[key]).all():
            raise ValueError('nonfinite field: ' + key)
    cell, triple = data['next_player_cell'][valid], data['next_triple'][valid]
    if ((cell < 0) | (cell >= 12)).any() or (triple < 0).any() or (triple[:, 0] >= 6).any() or (triple[:, 1:] >= 4).any():
        raise ValueError('physical labels outside grid/glyph domain')
    if ((data['next_lives'][valid] < 0) | (data['next_lives'][valid] > 3)).any():
        raise ValueError('lives outside 0..3')
    if ((data['next_steps'][valid] < -3) | (data['next_steps'][valid] > 42)).any():
        raise ValueError('step budget outside -3..42')


def load_sequences(paths, split, policy, guard=lambda: None):
    if split not in ('train', 'validation') or policy.planner.cfg.horizon != 4:
        raise ValueError('train/validation and K4 parent required')
    if not paths or len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError('distinct sequence directories required')
    parts, sources = [], {}
    encoder_sha = state_digest(policy.encoder.state_dict())
    dynamics_sha = state_digest(policy.planner.dynamics.state_dict())
    continuation_sha = state_digest(policy.planner.continuation.state_dict())
    for directory in paths:
        guard()
        path = Path(directory) / 'manifest.json'
        manifest_sha = guarded_digest(path, guard)
        manifest = json.loads(path.read_text())
        if (manifest.get('format') != 'pebby.neural-planning-sequences.v1' or manifest.get('status') != 'complete'
                or manifest.get('source') != 'generated_only' or manifest.get('split') != split
                or manifest.get('horizon') != 4
                or manifest.get('official_inputs_used') is not False or manifest.get('sources_unchanged') is not True
                or manifest.get('encoder_state_sha256') != encoder_sha
                or manifest.get('dynamics_state_sha256') != dynamics_sha
                or manifest.get('continuation_state_sha256') != continuation_sha
                or manifest.get('dynamics_config') != policy.planner.dynamics.config()):
            raise ValueError('completed matching generated K4 sequence cache required: ' + str(directory))
        count = manifest['arrays']['seeds']['shape'][0]
        for key, (dtype, tail) in SCHEMA.items():
            info = manifest['arrays'][key]
            if info['shape'] != [count, *tail] or info['dtype'] != dtype:
                raise ValueError('fixed four-branch K4 sequence schema required: ' + key)
        estimated = sum(count * int(np.prod(tail)) * np.dtype(dtype).itemsize for dtype, tail in SCHEMA.values())
        reserve_allocation(2 * estimated, guard)
        data_path = Path(directory) / manifest['data_file']
        if guarded_digest(data_path, guard) != manifest['data_sha256']:
            raise ValueError('sequence data checksum differs')
        with np.load(data_path, allow_pickle=False) as data:
            part = {}
            for key in KEYS:
                guard()
                part[key] = data[key]  # NPZ decoding already owns its array; avoid a second copy.
        validate_sequences(part, split)
        if guarded_digest(path, guard) != manifest_sha:
            raise ValueError('sequence manifest changed while loading')
        parts.append(part)
        sources[str(path.resolve())] = manifest_sha
        sources[str(data_path.resolve())] = manifest['data_sha256']
    reserve_allocation(sum(value.nbytes for part in parts for value in part.values()), guard)
    combined = {key:np.concatenate([part[key] for part in parts]) for key in KEYS}
    validate_sequences(combined, split)
    return combined, sources


def tensor_batch(data, rows, branches, device):
    public = torch.as_tensor(data['fields'][rows], device=device).float()
    arrays = {key:torch.as_tensor(data[key][rows, branches], device=device) for key in
              ('actions', 'next_fields', 'transition_valid', 'next_field_valid', *LABELS)}
    arrays['actions'] = arrays['actions'].long()
    return public, arrays


def close_mmaps(data):
    for value in (data or {}).values():
        backing = getattr(value, '_mmap', None)
        if backing is not None and not backing.closed:
            backing.close()


def load_h1_replay(path, policy, validation_seeds, guard):
    from tools.train_structured_policy import load_policy_cache, check_policy_encoder
    path = Path(path)
    directory = path / 'train' if (path / 'train' / 'manifest.json').is_file() else path
    guard()
    data = None
    try:
        data, manifest = load_policy_cache(directory, 'train')
        guard()
        check_policy_encoder(manifest['field_encoder'],
                             SimpleNamespace(sources={'encoder_metadata':policy.encoder.metadata()}), True)
        if np.intersect1d(data['seeds'], validation_seeds).size:
            raise ValueError('H1 replay and causal validation levels overlap')
        if not manifest['field_encoder'].get('code_hashes'):
            raise ValueError('H1 cached encoder code bindings required')
        sources = {str((directory / 'manifest.json').resolve()):guarded_digest(directory / 'manifest.json', guard)}
        for key, info in manifest['arrays'].items():
            sources[str((directory / (key + '.npy')).resolve())] = info['sha256']
        for group in ('code_hashes', 'checkpoint_hashes'):
            for source, expected in manifest['field_encoder'].get(group, {}).items():
                if guarded_digest(source, guard) != expected:
                    raise ValueError('H1 encoder binding changed')
                sources[str(Path(source).resolve())] = expected
        return data, sources
    except BaseException:
        close_mmaps(data)
        raise


def h1_tensor_batch(data, rows, branches, device):
    fields = torch.as_tensor(np.array(data['fields'][rows]), device=device).float()
    items = {key:torch.as_tensor(np.array(data[key][rows, branches])[:, None], device=device) for key in LABELS}
    items.update(actions=torch.as_tensor(branches[:, None], device=device).long(),
                 next_fields=torch.as_tensor(np.array(data['next_fields'][rows, branches])[:, None], device=device),
                 transition_valid=torch.ones((len(rows), 1), dtype=torch.bool, device=device),
                 next_field_valid=torch.ones((len(rows), 1), dtype=torch.bool, device=device))
    return fields, items


def _record():
    return dict(count=0, **{key + '_correct': 0 for key in ('player', 'glyph', 'budget', 'lives')},
                events={name:dict(positives=0, tp=0, fp=0, fn=0) for name in EVENTS})


def _finish(record):
    result = copy.deepcopy(record)
    for key in ('player', 'glyph', 'budget', 'lives'):
        result[key + '_accuracy'] = result[key + '_correct'] / result['count'] if result['count'] else None
    for event in result['events'].values():
        event['recall'] = event['tp'] / event['positives'] if event['positives'] else None
        event['precision'] = event['tp'] / (event['tp'] + event['fp']) if event['tp'] + event['fp'] else None
    return result


@torch.no_grad()
def evaluate(model, data, size, device, guard):
    model.eval()
    result = [_record() for _ in range(4)]
    levels = {int(seed): [_record() for _ in range(4)] for seed in np.unique(data['seeds'])}
    level_tiers = {int(seed):int(tier) for seed, tier in zip(data['seeds'], data['difficulties'])}
    strata = {name: [_record() for _ in range(4)] for name in (*EVENTS, 'after_life_loss')}
    field_errors = [dict(count=0, prediction_sum=0., identity_sum=0.) for _ in range(4)]
    for begin in range(0, len(data['seeds']) * 4, size):
        guard()
        ids = np.arange(begin, min(begin + size, len(data['seeds']) * 4))
        field, items = tensor_batch(data, ids // 4, ids % 4, device)
        prediction = model.rollout(field, items['actions'])
        heads = prediction['readout']
        glyph = torch.stack([prediction['glyph_logits'][key].argmax(-1) for key in ('shape', 'color', 'rotation')], -1)
        batch_seeds = data['seeds'][ids // 4]
        for step, rec in enumerate(result):
            valid = items['transition_valid'][:, step].bool()
            cell = items['next_player_cell'][:, step].long()
            correct = dict(player=heads['player_logits'][:, step].argmax(-1) == cell[:, 1] * 12 + cell[:, 0],
                           glyph=(glyph[:, step] == items['next_triple'][:, step]).all(-1),
                           budget=heads['steps_logits'][:, step].argmax(-1) == steps_targets(torch.where(valid, items['next_steps'][:, step], 0)),
                           lives=heads['lives_logits'][:, step].argmax(-1) == items['next_lives'][:, step])
            def add(record, mask):
                record['count'] += int(mask.sum())
                for key, values in correct.items():
                    record[key + '_correct'] += int((values & mask).sum())
                for name in EVENTS:
                    truth = items[name][:, step].bool()
                    estimate = prediction['events'][name + '_logits'][:, step] >= 0
                    counts = record['events'][name]
                    for key, value in dict(positives=truth & mask, tp=truth & estimate & mask,
                                           fp=~truth & estimate & mask, fn=truth & ~estimate & mask).items():
                        counts[key] += int(value.sum())
            add(rec, valid)
            for seed in np.unique(batch_seeds):
                add(levels[int(seed)][step], valid & torch.as_tensor(batch_seeds == seed, device=device))
            for name in EVENTS:
                add(strata[name][step], valid & items[name][:, step].bool())
            after_loss = items['lost_life'][:, :step].any(1) if step else torch.zeros_like(valid)
            add(strata['after_life_loss'][step], valid & after_loss)
            observed = items['next_field_valid'][:, step]
            if observed.any():
                target = items['next_fields'][observed, step].float()
                predicted = prediction['fields'][observed, step].float()
                identity = field[observed]
                # Carried channels have exact supervision and imperfect encoder targets.
                channels = torch.cat((torch.arange(70, device=device), torch.arange(84, 96, device=device)))
                record = field_errors[step]
                record['count'] += int(observed.sum())
                record['prediction_sum'] += float((predicted[..., channels] - target[..., channels]).square().mean((1, 2)).sum())
                record['identity_sum'] += float((identity[..., channels] - target[..., channels]).square().mean((1, 2)).sum())
    def macro(seeds):
        output = []
        for step in range(4):
            records = [_finish(levels[seed][step]) for seed in seeds if levels[seed][step]['count']]
            item = dict(levels=len(records), transitions=sum(record['count'] for record in records))
            for key in ('player', 'glyph', 'budget', 'lives'):
                item[key + '_accuracy'] = float(np.mean([r[key + '_accuracy'] for r in records])) if records else None
            output.append(item)
        return output
    for rec in field_errors:
        for key in ('prediction', 'identity'):
            rec[key + '_mse'] = rec[key + '_sum'] / rec['count'] if rec['count'] else None
    return dict(micro_by_horizon=[_finish(rec) for rec in result],
                level_macro_by_horizon=macro(list(levels)),
                tier_level_macro_by_horizon={str(tier):macro([seed for seed in levels if level_tiers[seed] == tier])
                                              for tier in sorted(set(level_tiers.values()))},
                per_level={str(seed):[_finish(rec) for rec in records] for seed, records in levels.items()},
                positive_event_strata={name:[_finish(rec) for rec in records] for name, records in strata.items()},
                noncarried_field_error_by_horizon=field_errors)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--train', type=Path, nargs='+', required=True)
    p.add_argument('--validation', type=Path, nargs='+', required=True)
    p.add_argument('--h1-replay', type=Path, help='optional generated TRAIN field cache to retain broad one-step dynamics')
    p.add_argument('--h1-fraction', type=float, default=.5, help='probability an update uses H1 replay, when provided')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--max-seconds', type=float, default=600)
    args = p.parse_args(argv)
    if (not 1 <= args.updates <= 2000 or not 1 <= args.batch_size <= 64 or not np.isfinite(args.lr)
            or not np.isfinite(args.h1_fraction) or not 0 <= args.h1_fraction < 1
            or args.lr <= 0 or not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 1800):
        p.error('invalid bounded training settings')
    if args.device == 'cpu':
        if torch.cuda.is_initialized():
            raise RuntimeError('fresh CPU process required')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    guard = Budget(args.max_seconds, reserve_gib=7)
    guard()
    report = dict(format=FORMAT, status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
                  parent=str(args.parent.resolve()), parent_sha256=args.parent_sha256, steps=[],
                  official_training_inputs=False, gameplay_evaluated=False,
                  settings={key: ([str(p.resolve()) for p in value] if isinstance(value, list) else
                                   str(value.resolve()) if isinstance(value, Path) else value)
                            for key, value in vars(args).items()},
                  optimizer=dict(name='AdamW', betas=[.9, .999], eps=1e-8, weight_decay=.01, grad_clip_norm=1.),
                  sampling='uniform level, then uniform root, then uniform first-action branch; repeated roots share one level mass',
                  limits=['Only learned dynamics is updated; no controller promotion or seven-level claim.',
                          'Repeated roots/branches share level identities; validation is grouped generated development evidence.',
                          'Exact reset labels do not restore initial-state information missing from H8.',
                          'Core/appearance targets distill imperfect public-encoder outputs; physical/event targets are exact.',
                          'Masked terminal tails are untrained; positive event support is reported explicitly.'])
    write_json(args.out / 'report.json', report)
    print(json.dumps(dict(event='started', pid=os.getpid())), flush=True)
    def timeout(*_):
        raise TimeoutError('planning training wall-clock budget exhausted')
    h1 = None
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        if guarded_digest(args.parent, guard) != args.parent_sha256:
            raise ValueError('parent checksum differs')
        guard()
        policy, _ = load_checkpoint(args.parent, 'cpu')
        guard()
        train, ts = load_sequences(args.train, 'train', policy, guard)
        validation, vs = load_sequences(args.validation, 'validation', policy, guard)
        if np.intersect1d(train['seeds'], validation['seeds']).size:
            raise ValueError('training and validation levels overlap')
        paths = [Path(__file__), Path('pebby/agent/neural_planning_objective.py'),
                 Path('pebby/agent/neural_imagination_policy.py'), Path('tools/train_navigation_probe.py'),
                 Path('pebby/agent/structured_transition.py'), Path('pebby/agent/structured_global_glyph.py'),
                 Path('pebby/agent/structured_local_glyph.py'),
                 Path('pebby/agent/structured_factored_policy.py'), Path('pebby/agent/structured_policy.py'),
                 Path('pebby/agent/neural_imagination.py'), Path('pebby/agent/structured_objective.py')]
        sources = {**ts, **vs, str(args.parent.resolve()):args.parent_sha256,
                   **{str(path.resolve()):guarded_digest(path, guard) for path in paths}}
        ancestry = torch.load(args.parent, map_location='cpu', weights_only=True)
        factored_path = Path(ancestry['parent_path'])
        sources[str(factored_path.resolve())] = ancestry['parent_sha256']
        factored = torch.load(factored_path, map_location='cpu', weights_only=True)
        sources.update(factored['sources']['code_hashes'])
        sources.update({str(Path(value['path']).resolve()):value['sha256']
                        for value in factored['sources']['artifacts'].values()})
        del ancestry, factored
        if args.h1_replay is not None:
            h1, h1_sources = load_h1_replay(args.h1_replay, policy, validation['seeds'], guard)
            sources.update(h1_sources)
            if args.batch_size > len(h1['seeds']):
                raise ValueError('H1 replay batch exceeds distinct levels')
            for filename in ('tools/train_structured_policy.py', 'tools/train_structured_transition.py'):
                sources[str(Path(filename).resolve())] = guarded_digest(filename, guard)
        report['h1_replay'] = dict(enabled=h1 is not None, levels=len(h1['seeds']) if h1 is not None else 0,
                                   fraction=args.h1_fraction if h1 is not None else 0.,
                                   event_weight_basis='K4 TRAIN transition frequencies')
        positive_counts = np.asarray([(train[name] & train['transition_valid']).sum() for name in EVENTS])
        valid_count = int(train['transition_valid'].sum())
        report.update(sources=sources, train_levels=len(np.unique(train['seeds'])),
                      validation_levels=len(np.unique(validation['seeds'])), train_roots=len(train['seeds']),
                      validation_roots=len(validation['seeds']), train_events=dict(zip(EVENTS, positive_counts.tolist())))
        # A repair advertised to cover endings must actually receive each event.
        if np.any(positive_counts == 0):
            raise ValueError('training needs positive life-loss, terminal and win sequences before joint event repair')
        scale = np.maximum(train['fields'][..., :48].std((0, 1)), .1)
        positive = np.minimum((valid_count - positive_counts) / positive_counts, 50).clip(.01)
        report.update(core_scale=scale.tolist(), event_positive_weights=positive.tolist())
        model = copy.deepcopy(policy.planner.dynamics).to(args.device).requires_grad_(True)
        # This experiment teaches exact state and latent fields; the unused
        # semantic readout is preserved rather than receiving weight decay only.
        model.readout.cell.requires_grad_(False)
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=args.lr)
        report['before'] = evaluate(model, validation, args.batch_size, args.device, guard)
        model.train()
        rng = np.random.default_rng(args.seed)
        seeds = np.unique(train['seeds'])
        grouped = {seed:np.flatnonzero(train['seeds'] == seed) for seed in seeds}
        report['draws'] = []
        for step in range(args.updates):
            guard()
            replay_update = h1 is not None and rng.random() < args.h1_fraction
            branches = rng.integers(0, 4, args.batch_size)
            if replay_update:
                rows = rng.choice(len(h1['seeds']), size=args.batch_size, replace=False)
                field, items = h1_tensor_batch(h1, rows, branches, args.device)
            else:
                selected_seeds = rng.choice(seeds, size=args.batch_size, replace=args.batch_size > len(seeds))
                rows = np.asarray([rng.choice(grouped[seed]) for seed in selected_seeds])
                field, items = tensor_batch(train, rows, branches, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = planning_sequence_loss(model, field, items['actions'], items['next_fields'],
                                          {name:items[name] for name in LABELS}, items['transition_valid'],
                                          items['next_field_valid'], core_scale=scale, event_positive_weights=positive)
            if not torch.isfinite(loss['total']):
                raise FloatingPointError('nonfinite planning loss')
            loss['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            rec = dict(step=step + 1, total=float(loss['total'].detach()), gradient_norm=float(norm),
                       losses={name:float(value.detach()) for name,value in loss['losses'].items()},
                       transition_count=loss['transition_count'], field_count=loss['field_count'],
                       change_comparison_count=loss['change_comparison_count'],
                       changed_transition_count=loss['changed_transition_count'],
                       changed_cell_count=loss['changed_cell_count'],
                       event_positive_counts=loss['event_positive_counts'])
            report['steps'].append(rec)
            report['draws'].append(dict(source='h1' if replay_update else 'k4', rows=rows.tolist(), branches=branches.tolist()))
            if (step + 1) % 25 == 0:
                print(json.dumps(rec), flush=True)
                write_json(args.out / 'report.json', report)
        report['after'] = evaluate(model, validation, args.batch_size, args.device, guard)
        weights = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        for path, expected in sources.items():
            guard()
            if guarded_digest(path, guard) != expected:
                raise ValueError('training source changed: ' + path)
        envelope = dict(format=FORMAT, config=model.config(), weights=weights, weights_sha256=state_digest(weights),
                        parent=str(args.parent.resolve()), parent_sha256=args.parent_sha256,
                        encoder_state_sha256=state_digest(policy.encoder.state_dict()), sources=sources,
                        continuation_state_sha256=state_digest(policy.planner.continuation.state_dict()),
                        settings=report['settings'], optimizer=report['optimizer'],
                        official_training_inputs=False, updates=args.updates, train_events=report['train_events'])
        checkpoint = args.out / 'dynamics.pt'
        torch.save(envelope, checkpoint)
        restored = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if state_digest(restored['weights']) != envelope['weights_sha256']:
            raise ValueError('checkpoint roundtrip changed weights')
        report.update(status='complete', sources_unchanged=True, checkpoint=str(checkpoint.resolve()),
                      checkpoint_sha256=digest(checkpoint), weights_sha256=envelope['weights_sha256'])
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        close_mmaps(h1)
        signal.signal(signal.SIGALRM, previous_handler)
        report.update(elapsed_seconds=time.monotonic() - guard.started, finished_local=datetime.now().astimezone().isoformat())
        write_json(args.out / 'report.json', report)
    return report


if __name__ == '__main__':
    main()
