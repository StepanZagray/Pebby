"""Generated-only, frozen CPU probes of the first active special effect on any optimal route.

Teacher labels are set-valued across all optimal prefixes. Public initial frames
and learned player/glyph readouts are the only feature inputs; no route or target
coordinates enter a probe. A failed probe is not evidence that information is absent.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.goal_attribute_probes import digest, read_specs, stratified_seeds, load_glyph_classifier
from pebby.agent.world_model import load_world_checkpoint
from pebby.agent.glyph_model import glyph_probabilities
from pebby.ls20.generate import build_level
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, advance, simulate

TYPES = ('matching_goal', 'shape', 'color', 'rotation', 'refill', 'launcher')


def effect_types(before, after, outcome):
    result = set()
    for index, name in ((1, 'shape'), (2, 'color'), (3, 'rotation')):
        if before[index] != after[index]:
            result.add(name)
    if before[4] != after[4]:
        result.add('matching_goal')
    if before[5] != after[5]:
        result.add('refill')
    if outcome == 'launched':
        result.add('launcher')
    return result


def first_types(oracle, prefix_limit=100000):
    """Explore the decreasing-distance DAG only until first active effects.

    Multiple effects in one action are simultaneous labels (no ordering claim).
    Ordinary movement/walls and already-inactive cells are not special effects.
    """
    if oracle.truncated or not oracle.solvable:
        raise ValueError('incomplete_or_unsolved_oracle')
    todo, seen, types = [oracle.start], {oracle.start}, set()
    simultaneous = 0
    while todo:
        before = todo.pop()
        distance = oracle.distance_for(before)
        for action in range(4):
            after = advance(oracle.layout, before, action, oracle.refills)
            if after is None or oracle.distance_for(after) != distance - 1:
                continue
            exact, outcome = simulate(oracle.layout, before, action, oracle.refills)
            if exact != after:
                raise AssertionError('transition disagreement')
            events = effect_types(before, after, outcome)
            if events:
                types.update(events)
                simultaneous += int(len(events) > 1)
            elif after not in seen:
                seen.add(after)
                if len(seen) > prefix_limit:
                    raise ValueError('optimal_prefix_limit')
                todo.append(after)
    if not types:
        raise ValueError('no_special_effect')
    return [int(name in types) for name in TYPES], len(seen), simultaneous


def collect(specs, seeds):
    frames, masks, records = [], [], []
    for index, seed in enumerate(seeds):
        spec = specs[seed]
        record = {'seed': seed, 'difficulty': spec['difficulty'], 'fog': bool(spec.get('fog'))}
        if not spec.get('context_engine_verified') or spec.get('search_truncated', False):
            raise ValueError('bank lacks complete contextual verification')
        env = Ls20Scenario(build_level(spec), int(spec['training_context_index']))
        frame = np.asarray(env.reset(), dtype=np.uint8)
        try:
            oracle = Oracle(extract(env), limit=600000, engine='fast')
            labels, prefixes, simultaneous = first_types(oracle)
            masks.append(labels)
            frames.append(frame)
            record.update(labels=[name for name, yes in zip(TYPES, labels) if yes],
                          prefix_states=prefixes, simultaneous_effect_edges=simultaneous,
                          optimal_actions=oracle.optimal_actions, excluded=None)
        except ValueError as error:
            record['excluded'] = str(error)
        records.append(record)
        if index % 100 == 0:
            print('labels', index, '/', len(seeds), flush=True)
    return np.asarray(frames), torch.tensor(masks, dtype=torch.float32), records


def features(model, classifier, frames, batch_size):
    result = {name: [] for name in ('raw_board', 'refined_board', 'refined_local3', 'latent', 'glyph')}
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            current = torch.as_tensor(frames[start:start + batch_size]).long()
            # Repeated initial padding has identical frame tokens; encode pixels once.
            raw = model.frame_tokens(current)
            tokens = raw[:, None].expand(-1, model.cfg.history, -1, -1)
            valid = torch.zeros(len(current), model.cfg.history, dtype=torch.bool)
            valid[:, -1] = True
            actions = torch.full(valid.shape, -1, dtype=torch.long)
            own_glyph = model.glyph_logits(current) if model.cfg.glyph_recall else None
            encoded = model.assemble(tokens, valid, actions, glyph_logits=own_glyph)
            _, player_weights = model.player_weights(encoded['cells'])
            center = player_weights.argmax(-1)
            rows, cols = center // 12, center % 12
            offsets = torch.tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            rr, cc = rows[:, None] + offsets[:, 0], cols[:, None] + offsets[:, 1]
            local_valid = (rr >= 0) & (rr < 12) & (cc >= 0) & (cc < 12)
            indices = rr.clamp(0, 11) * 12 + cc.clamp(0, 11)
            local = encoded['cells'][torch.arange(len(current))[:, None], indices]
            local = local * local_valid[..., None]
            crop = current[:, 55:61, 3:9]
            carried = glyph_probabilities(classifier(F.one_hot(crop, 16).flatten(1).float()))
            for key, value in (('raw_board', raw[:, :144]), ('refined_board', encoded['cells']),
                               ('refined_local3', local), ('latent', encoded['latent']), ('glyph', carried)):
                result[key].append(value.clone())
    return {key: torch.cat(values) for key, values in result.items()}


class QueryProbe(nn.Module):
    def __init__(self, channels, token_input):
        super().__init__()
        self.token_input = token_input
        if token_input:
            self.query = nn.Linear(14, channels)
        self.head = nn.Sequential(nn.Linear(channels + 14, 64), nn.GELU(), nn.Linear(64, len(TYPES)))

    def forward(self, feature, glyph):
        if self.token_input:
            scores = (feature * self.query(glyph)[:, None]).sum(-1) / feature.shape[-1] ** .5
            feature = (scores.softmax(-1)[..., None] * feature).sum(1)
        return self.head(torch.cat((feature, glyph), -1))


def accuracy(scores, labels):
    chosen = scores.argmax(-1)
    hit = labels.gather(1, chosen[:, None]).squeeze(1)
    singleton = labels.sum(-1) == 1
    return {'set_accuracy': float(hit.mean()), 'singleton_accuracy': float(hit[singleton].mean()) if singleton.any() else None,
            'predicted_counts': torch.bincount(chosen, minlength=len(TYPES)).tolist()}


def fit(train, validation, glyph_train, glyph_validation, y, vy, steps, batch_size, seed, train_fog=None, validation_fog=None):
    if len(train) < batch_size:
        raise ValueError('insufficient distinct training levels for true batch')
    torch.manual_seed(seed)
    dims = (0, 1) if train.ndim == 3 else (0,)
    mean, scale = train.mean(dims), train.std(dims).clamp_min(1e-5)
    train, validation = (train - mean) / scale, (validation - mean) / scale
    gm, gs = glyph_train.mean(0), glyph_train.std(0).clamp_min(1e-5)
    glyph_train, glyph_validation = (glyph_train - gm) / gs, (glyph_validation - gm) / gs
    probe = QueryProbe(train.shape[-1], train.ndim == 3)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=.003, weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed)
    targets = y / y.sum(-1, keepdim=True)
    for _ in range(steps):
        indices = torch.randperm(len(train), generator=rng)[:batch_size]
        logits = probe(train[indices], glyph_train[indices])
        loss = -(targets[indices] * logits.log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    def evaluate(scores, labels, fog):
        metrics = accuracy(scores, labels)
        metrics['count'] = len(labels)
        if fog is not None:
            metrics['by_visibility'] = {}
            for name, subset in (('fog', fog), ('nonfog', ~fog)):
                metrics['by_visibility'][name] = ({**accuracy(scores[subset], labels[subset]),
                                                  'count': int(subset.sum())} if subset.any() else {'count': 0})
        return metrics
    with torch.no_grad():
        return {'train': evaluate(probe(train, glyph_train), y, train_fog),
                'validation': evaluate(probe(validation, glyph_validation), vy, validation_fog),
                'parameters': sum(p.numel() for p in probe.parameters()), 'last_batch_loss': float(loss.detach()),
                'input_shape': list(train.shape[1:]), 'glyph_inputs': 14, 'steps': steps,
                'batch_size': batch_size, 'distinct_levels_per_batch': batch_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/ls20-world-mixedpath-b1024.epoch1.pt')
    parser.add_argument('--glyph-checkpoint', default='checkpoints/ls20-glyph-prototype.pt')
    parser.add_argument('--train-bank', default='data/ls20-verified-train.jsonl')
    parser.add_argument('--validation-bank', default='data/ls20-verified-validation.jsonl')
    parser.add_argument('--out', default='artifacts/world-target-type-probes.json')
    parser.add_argument('--train-count', type=int, default=1000)
    parser.add_argument('--validation-count', type=int, default=512)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--feature-batch-size', type=int, default=16)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    started = time.monotonic()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'pid': os.getpid(), 'device': 'cpu', 'threads': 1, 'types': TYPES,
              'seed': args.seed, 'official_inputs_used': False, 'results': {},
              'label_contract': 'Set of first active special effects across ALL optimal prefixes; simultaneous effects share labels.',
              'protocol': {'normalization': 'training statistics only, pooled across tokens per channel',
                           'heads': 'glyph-conditioned query pooling then MLP64 for token inputs; MLP64 for latent',
                           'sampling': 'one initial history per distinct verified generated level; equal difficulty quotas',
                           'local_center': 'frozen player-head argmax; no oracle coordinates',
                           'carried_glyph': 'public current 6x6 HUD crop classified by frozen pretrained generated glyph model'},
              'limitations': ['Predicting the first effect type is not navigation or closed-loop completion.',
                             'Low accuracy under this bounded probe does not establish absence of information.',
                             'Set accuracy credits any tied optimal first type; singleton accuracy and class coverage are reported.',
                             'Local probe is confounded by learned player localization errors; it is not an oracle-centered local upper bound.',
                             'Raw board and refined board share probe architecture; local has only 9 tokens; latent has different width and no query.',
                             'Initial fog is retained, including potentially hidden optimal targets; no visibility oracle is given to probes.',
                             'All inputs include public carried-glyph probabilities; there is no standalone glyph-only baseline.',
                             'Inactive special cells are not events; launchers are explicit other mechanics; no missing types silently dropped.'],
              'hashes': {str(path): digest(path) for path in (args.checkpoint, args.glyph_checkpoint,
                          args.train_bank, args.validation_bank, __file__, 'pebby/agent/world_model.py', 'pebby/ls20/plan.py')}}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(output)
    def timeout(*_):
        raise TimeoutError('CPU diagnostic time cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    persist()
    print('PID', os.getpid(), flush=True)
    try:
        banks = [read_specs(args.train_bank), read_specs(args.validation_bank)]
        seeds = [stratified_seeds(banks[0], args.train_count, args.seed),
                 stratified_seeds(banks[1], args.validation_count, args.seed + 1)]
        if set(seeds[0]) & set(seeds[1]):
            raise ValueError('overlapping split seeds')
        collected = []
        for split, bank, selected in zip(('train', 'validation'), banks, seeds):
            frames, labels, records = collect(bank, selected)
            collected.append((frames, labels))
            report[split] = {'selected_levels': len(selected), 'retained_levels': len(frames),
                             'records': records, 'exclusions': dict(Counter(r['excluded'] for r in records if r['excluded'])),
                             'tied_levels': int((labels.sum(-1) > 1).sum()),
                             'class_memberships': labels.sum(0).tolist()}
            persist()
        y, vy = collected[0][1], collected[1][1]
        report['unsupported_classes'] = {split: [name for name, count in zip(TYPES, labels.sum(0)) if count == 0]
                                         for split, labels in (('train', y), ('validation', vy))}
        report['limitations'].append('Classes with zero label support cannot be assessed; see unsupported_classes.')
        fog = [torch.tensor([r['fog'] for r in report[split]['records'] if r['excluded'] is None])
               for split in ('train', 'validation')]
        majority = int(y.mean(0).argmax())
        report['majority'] = {'selected_using': 'training set-accuracy maximizing constant class', 'class': TYPES[majority],
                              'train_accuracy': float(y[:, majority].mean()), 'validation_accuracy': float(vy[:, majority].mean())}
        report['majority']['by_visibility'] = {
            split: {name: {'count': int(subset.sum()),
                           'accuracy': float(labels[subset, majority].mean()) if subset.any() else None}
                    for name, subset in (('fog', visible), ('nonfog', ~visible))}
            for split, labels, visible in zip(('train', 'validation'), (y, vy), fog)}
        model, checkpoint = load_world_checkpoint(args.checkpoint, 'cpu')
        classifier, _ = load_glyph_classifier(args.glyph_checkpoint)
        model.requires_grad_(False)
        classifier.requires_grad_(False)
        report['checkpoint_config'] = model.config()
        f, vf = [features(model, classifier, frames, args.feature_batch_size) for frames, _ in collected]
        del model, classifier
        persist()
        for name in ('raw_board', 'refined_board', 'refined_local3', 'latent'):
            report['results'][name] = fit(f[name], vf[name], f['glyph'], vf['glyph'], y, vy,
                                          args.steps, args.batch_size, args.seed, *fog)
            print(name, report['results'][name], flush=True)
            persist()
        report['status'] = 'complete'
    except TimeoutError:
        report['status'] = 'incomplete_time_cap'
    except Exception as error:
        report['status'], report['error'] = 'failed', repr(error)
        raise
    finally:
        signal.alarm(0)
        persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
