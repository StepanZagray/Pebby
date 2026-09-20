"""CPU successor-resource diagnostics on frozen public feature caches; no fitting."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.neural_outcome_policy import ENCODER_RUNTIME, PARENT_SHA as ENCODER_PARENT_SHA, weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.spatial_outcome_policy import FORMAT as SPATIAL_FORMAT
from pebby.agent.spatial_route_outcome_planner import SpatialRouteOutcomePlanner
from pebby.agent.spatial_route_outcome_policy import FORMAT as ROUTE_FORMAT
from pebby.agent.world_grounding import SIZES
from tools.cache_reference_outcome_inputs import Bindings, release, stat
from tools.train_reference_outcomes import ARRAYS, batch, guard, load_data, sha, write
from tools.train_reference_spatial_outcomes import player_probabilities

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ('live_decrease', 'live_unchanged', 'live_refill', 'life_loss',
           'win_without_life_loss', 'terminal_without_life_loss_or_win')
EVENTS = ('lost_life', 'terminal', 'won')


def resource_targets(items):
    """Disjoint cohorts, preserving training's negative-budget clamp to -1."""
    following = items['next_steps']
    if following.ndim != 2 or following.shape[1] != 4 or not len(following):
        raise ValueError('next_steps must be nonempty [B,4]')
    for name in ('next_steps', 'next_lives', 'current_steps', *EVENTS):
        value = items[name]
        shape = (len(following),) if name == 'current_steps' else following.shape
        if value.shape != shape or value.is_floating_point() or value.is_complex():
            raise ValueError('resource labels must have integer/boolean target shapes')
    if bool((following > SIZES[4] - 2).any()):
        raise ValueError('positive budget overflow is outside the training categories')
    lives = items['next_lives'].long()
    if bool(((lives < 0) | (lives >= SIZES[5])).any()):
        raise ValueError('next_lives outside training categories')
    for name in EVENTS:
        if not bool(((items[name] == 0) | (items[name] == 1)).all()):
            raise ValueError('event labels must be binary')
    lost, terminal, won = (items[name].bool() for name in EVENTS)
    live = ~lost & ~terminal & ~won
    current = items['current_steps'][:, None]
    masks = (live & (following < current), live & (following == current), live & (following > current),
             lost, won & ~lost, terminal & ~lost & ~won)
    if not bool((torch.stack(masks).sum(0) == 1).all()):
        raise ValueError('resource cohorts must partition every branch')
    return following.long().clamp_min(-1), lives, dict(zip(COHORTS, masks))


class ResourceMetrics:
    """Accumulate integer totals so batches and empty cohorts are unbiased."""
    def __init__(self):
        self.cohorts = {name: dict(support=0, budget_correct=0, budget_within_one=0,
                                  budget_absolute_error=0, lives_correct=0)
                        for name in ('all', *COHORTS)}
        self.events = {name: dict(support=0, positive_support=0, predicted_positive=0,
                                 true_positive=0, correct=0) for name in EVENTS}
        self.negative_budgets_clamped = 0

    def update(self, prediction, items):
        target, lives, masks = resource_targets(items)
        steps_logits, lives_logits = prediction['field_logits'][4:6]
        events = prediction['event_logits']
        for name, value, shape in (('steps', steps_logits, (*target.shape, SIZES[4])),
                                   ('lives', lives_logits, (*target.shape, SIZES[5])),
                                   ('events', events, (*target.shape, len(EVENTS)))):
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite or incorrectly shaped {name} predictions')
        error = (steps_logits.argmax(-1) - 1 - target).abs()
        lives_correct = lives_logits.argmax(-1) == lives
        self.negative_budgets_clamped += int((items['next_steps'] < -1).sum())
        for name, mask in {'all': torch.ones_like(target, dtype=torch.bool), **masks}.items():
            counts = self.cohorts[name]
            counts['support'] += int(mask.sum())
            counts['budget_correct'] += int((error[mask] == 0).sum())
            counts['budget_within_one'] += int((error[mask] <= 1).sum())
            counts['budget_absolute_error'] += int(error[mask].sum())
            counts['lives_correct'] += int(lives_correct[mask].sum())
        for index, name in enumerate(EVENTS):
            actual, predicted = items[name].bool(), events[..., index] >= 0
            counts = self.events[name]
            counts['support'] += actual.numel()
            counts['positive_support'] += int(actual.sum())
            counts['predicted_positive'] += int(predicted.sum())
            counts['true_positive'] += int((actual & predicted).sum())
            counts['correct'] += int((actual == predicted).sum())

    def result(self):
        cohorts = {}
        for name, count in self.cohorts.items():
            n = count['support']
            cohorts[name] = {**count, **{metric: count[key] / n if n else None for metric, key in
                [('budget_accuracy', 'budget_correct'), ('budget_within_one_accuracy', 'budget_within_one'),
                 ('budget_mae', 'budget_absolute_error'), ('next_lives_accuracy', 'lives_correct')]}}
        events = {}
        for name, count in self.events.items():
            events[name] = {**count,
                'recall': count['true_positive'] / count['positive_support'] if count['positive_support'] else None,
                'precision': count['true_positive'] / count['predicted_positive'] if count['predicted_positive'] else None,
                'accuracy': count['correct'] / count['support'] if count['support'] else None}
        return dict(cohorts=cohorts, events=events, negative_budget_labels_clamped=self.negative_budgets_clamped)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/reference-outcome-inputs-v1')
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--recent', action='store_true', help='cache is a v3 supplement; apply its verified quality allowlist')
    parser.add_argument('--batch-size', type=int, choices=(32, 64, 128), default=64)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 600 or (args.recent and args.split != 'train'):
        parser.error('deadline must be 1..600 seconds; recent caches require --split train')
    if args.out.exists():
        raise FileExistsError(args.out)
    started, previous_threads = time.monotonic(), torch.get_num_threads()
    old_handler = signal.getsignal(signal.SIGALRM)
    arrays, all_arrays, stats = {}, {}, {}
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
                  checkpoint=str(args.checkpoint.resolve()), split=args.split, recent=args.recent,
                  training=False, official_inputs=False, device='cpu', batch_size=args.batch_size,
                  cohort_precedence='life loss; win without life loss; other terminal; live decrease/unchanged/refill',
                  event_metrics='Independent binary labels; event overlaps are retained.',
                  budget_decode='category 0 is -1; maximum category 43 is budget 42; negative labels clamp at -1')
    def timeout(*_):
        raise TimeoutError('bounded CPU resource audit expired')
    try:
        signal.signal(signal.SIGALRM, timeout); signal.alarm(args.max_seconds)
        torch.set_num_threads(1); guard()
        bindings = Bindings(); bindings.add(args.checkpoint, args.checkpoint_sha256)
        for path in [Path(__file__), *sorted((ROOT / 'pebby/agent').glob('*.py')),
                     *[ROOT / 'tools' / name for name in ('cache_reference_outcome_inputs.py',
                         'train_reference_outcomes.py', 'train_reference_spatial_outcomes.py')]]:
            bindings.add(path)
        parent = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if (parent.get('encoder_frozen') is not True or parent.get('official_training_inputs') is not False
                or parent.get('privileged_inference_inputs', False) is not False
                or parent.get('encoder_runtime') != ENCODER_RUNTIME
                or parent.get('encoder_parent_sha256') != ENCODER_PARENT_SHA
                or weights_sha256(parent['encoder_weights']) != parent['encoder_weights_sha256']):
            raise ValueError('verified frozen public encoder checkpoint required')
        cls = {SPATIAL_FORMAT: SpatialOutcomePlanner, ROUTE_FORMAT: SpatialRouteOutcomePlanner}.get(parent['format'])
        if cls is None:
            raise ValueError('unsupported spatial checkpoint format')
        model = cls(parent['planner_config']).eval()
        model.load_state_dict(parent['planner_weights'], strict=True)
        if args.recent:
            from tools import cache_spatial_repair_v3
            bindings.add(Path(cache_spatial_repair_v3.__file__))
            manifest = cache_spatial_repair_v3.validate_published(args.cache)
            hashes, stats = manifest['validated_output_hashes'], manifest['validated_output_stats']
            arrays = {name: np.load(args.cache / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in ARRAYS}
            rows = np.load(args.cache / manifest['quality']['files']['supplement_rows']['path'], allow_pickle=False)
        else:
            all_arrays, hashes, stats = load_data(args.cache)
            arrays = all_arrays[args.split]
            rows = np.arange(len(arrays['seeds']))
            manifest = json.loads((args.cache / 'manifest.json').read_text())
        if parent['encoder_runtime'] != manifest.get('encoder_runtime', manifest.get('settings')) and args.recent:
            raise ValueError('encoder runtime differs from cache')
        manifest_hash = bindings.add(args.cache / 'manifest.json')
        if not args.recent and parent['cache_manifest_sha256'] != manifest_hash:
            raise ValueError('original feature cache differs from checkpoint lineage')
        if args.recent and parent['encoder_weights_sha256'] != manifest['encoder_weights_sha256']:
            raise ValueError('recent cache uses different encoder weights')
        report.update(format=parent['format'], source_sha256={**hashes, **bindings.hashes},
                      cache_manifest_sha256=manifest_hash, rows=len(rows), levels=len(np.unique(arrays['seeds'][rows])))
        encoder = {key: value for key, value in parent['encoder_weights'].items() if key.startswith('player_head.')}
        metrics = ResourceMetrics()
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                guard()
                items = batch(arrays, rows[start:start + args.batch_size], 'cpu')
                prediction = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder))
                metrics.update(prediction, items)
        bindings.verify()
        if any(stat(path) != value for path, value in stats.items()):
            raise ValueError('input arrays changed during resource audit')
        if torch.cuda.is_initialized():
            raise ValueError('CPU audit initialized CUDA')
        report.update(status='complete', sources_unchanged=True, **metrics.result())
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        for group in all_arrays.values() if all_arrays else [arrays]:
            release(group, close=True)
        torch.set_num_threads(previous_threads)
        report.update(seconds=time.monotonic() - started, finished_local=datetime.now().astimezone().isoformat())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write(args.out, report)
    return report


if __name__ == '__main__':
    main()
