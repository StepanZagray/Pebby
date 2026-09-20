"""Matched comparator-only continuation on verified generated TRAIN outcomes.

The frozen predictor is evaluated once. Both arms receive identical public
predictions and teacher labels, level draws, initialization and optimizer budget.
The ordering arm adds safe pair ranking with actual and neutral physical fields.
Neither arm can improve the frozen physical/event predictor in this experiment.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.outcome_ordering import masked_optimal_set_cross_entropy, pairwise_safe_ordering_loss
from pebby.agent.spatial_outcome_objective import actual_outcomes
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from pebby.agent.world_grounding import SIZES
from tools.audit_spatial_decisions import outcome_probabilities, score_probabilities
from tools.cache_reference_outcome_inputs import Bindings
from tools.train_reference_outcomes import batch, gpu_available, guard, load_data, optimizer_for, sha, write
from tools.train_reference_spatial_outcomes import player_probabilities
from tools.train_spatial_recovery_comparison import MatchedSampler, load_quality_manifest, load_supplement, schedule, validate_checkpoint

COMPARATOR = ('outcome_projection.', 'comparator.')


def freeze_predictor(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(COMPARATOR))


def predictor_digest(model):
    return weights_sha256({k: v for k, v in model.state_dict().items() if not k.startswith(COMPARATOR)})


def compact_cache(model, arrays, indices, encoder_weights, directory):
    """Preserve original row IDs; unselected rows may never be sampled."""
    directory.mkdir()
    width = 144 + 6 + 4 + 4 + 44 + 4 + 130 + 3
    paths = {key: directory / (key + '.npy') for key in ('predicted', 'actual')}
    cache = {key: np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                           shape=(len(arrays['seeds']), 4, width)) for key, path in paths.items()}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            guard()
            rows = indices[start:start + 128]
            items = batch(arrays, rows, 'cuda')
            prediction = model(items['raw'], items['state'], items['glyph'], player_probabilities(items, encoder_weights))
            probabilities = outcome_probabilities(prediction['field_logits'], prediction['value_logits'], prediction['event_logits'])
            torch.testing.assert_close(score_probabilities(model, probabilities), prediction['action_logits'], atol=0, rtol=0)
            cache['predicted'][rows] = probabilities.cpu().numpy()
            cache['actual'][rows] = outcome_probabilities(*actual_outcomes(items)).cpu().numpy()
    for value in cache.values():
        value.flush()
    return cache, {str(p.resolve()): sha(p) for p in paths.values()}


def selected_items(base, supplement, compact, selection):
    rows, replacements = selection['base_rows'], selection['replacement_rows']
    slots = np.flatnonzero(replacements >= 0)
    result = {}
    for key in ('predicted', 'actual', 'distances', 'lost_life', 'optimal'):
        left, right = (compact['base'], compact['supplement']) if key in ('predicted', 'actual') else (base, supplement)
        values = np.array(left[key][rows])
        values[slots] = right[key][replacements[slots]]
        result[key] = torch.from_numpy(values).cuda()
    return result


def comparator_loss(model, items, ordering_weight):
    native = score_probabilities(model, items['predicted'])
    teacher = score_probabilities(model, items['actual'])
    ce = masked_optimal_set_cross_entropy(native, items['optimal']) + masked_optimal_set_cross_entropy(teacher, items['optimal'])
    if ordering_weight:
        # Neutralization is TRAIN-only augmentation. Distances/events stay real;
        # physical fields are held constant to remove shortcuts in their use.
        neutral = items['actual'].clone()
        physical_width = sum(SIZES)
        neutral[..., :physical_width] = neutral[:, :1, :physical_width].clone().expand(-1, 4, -1)
        neutral_scores = score_probabilities(model, neutral)
        ordering = .5 * (pairwise_safe_ordering_loss(teacher, items['distances'], items['lost_life'], items['optimal']) +
                         pairwise_safe_ordering_loss(neutral_scores, items['distances'], items['lost_life'], items['optimal']))
    else:
        ordering = native.sum() * 0
    return ce + ordering_weight * ordering, ce, ordering


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--cache', type=Path, default=Path('data/reference-outcome-inputs-v1'))
    p.add_argument('--supplement', type=Path, default=Path('data/spatial-recovery-inputs-v1'))
    p.add_argument('--quality', type=Path, default=Path('artifacts/spatial-recovery-v1/quality/manifest.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-seconds', type=int, default=600)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.steps < 1 or not 0 < args.lr < 1 or args.max_seconds < 1:
        raise ValueError('positive bounded training settings required')
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('checkpoint differs from pinned parent')
    gpu_available(); guard()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('training deadline exceeded')))
    signal.alarm(args.max_seconds)
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
                  parent_sha256=args.checkpoint_sha256, steps=args.steps, learning_rate=args.lr, seed=args.seed,
                  batch_size=1024, precision='float32', arms={}, source_sha256={},
                  frozen_encoder=True, frozen_outcome_predictor=True, official_training_inputs=False,
                  training_objective='native CE + actual-outcome CE; ordering adds actual/neutral safe-pair softplus ranking',
                  limits=['Frozen refill and life-loss prediction errors cannot improve in these arms.',
                          'Fixed generated TRAIN recovery mixture; no new current-policy collection in this experiment.',
                          'No architecture, temporal horizon, voluntary reset, or persistent-memory change.'])
    try:
        cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint(cp)
        arrays, hashes, _ = load_data(args.cache)
        report['source_sha256'].update(hashes)
        report['cache_manifest_sha256'] = sha(args.cache / 'manifest.json')
        if report['cache_manifest_sha256'] != cp['cache_manifest_sha256']:
            raise ValueError('parent/cache mismatch')
        bindings = Bindings()
        supplement, manifest = load_supplement(args.supplement, cp, arrays['train'], bindings)
        report['supplemental_manifest_sha256'] = sha(args.supplement / 'manifest.json')
        approved = load_quality_manifest(args.quality, arrays['train'], supplement, report, bindings)
        bindings.verify()
        report['source_sha256'].update(bindings.hashes)
        for name, source in [('base', arrays['train']), ('supplement', supplement)]:
            for start in range(0, len(approved[name]), 4096):
                rows = approved[name][start:start+4096]
                pairwise_safe_ordering_loss(torch.zeros(len(rows), 4),
                    torch.from_numpy(np.array(source['distances'][rows])),
                    torch.from_numpy(np.array(source['lost_life'][rows])),
                    torch.from_numpy(np.array(source['optimal'][rows])))
        paths = [Path(__file__), args.checkpoint, args.cache/'manifest.json', args.supplement/'manifest.json',
                 *sorted(Path('pebby/agent').glob('*.py')), Path('tools/audit_spatial_decisions.py'),
                 Path('tools/train_spatial_recovery_comparison.py'), Path('tools/train_reference_outcomes.py'),
                 Path('tools/train_reference_spatial_outcomes.py'), Path('tools/cache_reference_outcome_inputs.py')]
        report['source_sha256'].update({str(path.resolve()): sha(path) for path in paths})
        sampler = MatchedSampler(arrays['train'], supplement, base_rows=approved['base'], supplement_rows=approved['supplement'])
        model = SpatialOutcomePlanner(cp['planner_config']).cuda().eval()
        model.load_state_dict(cp['planner_weights'], strict=True)
        frozen_sha = predictor_digest(model)
        report['frozen_predictor_sha256'] = frozen_sha
        encoder_weights = {k: v.cuda() for k, v in cp['encoder_weights'].items() if k.startswith('player_head.')}
        compact = {}
        for name, source in [('base', arrays['train']), ('supplement', supplement)]:
            compact[name], compact_hashes = compact_cache(model, source, approved[name], encoder_weights, args.out/name)
            report['source_sha256'].update(compact_hashes)
            print(json.dumps(dict(event='cache_complete', source=name, rows=len(approved[name]))), flush=True)
        write(args.out/'report.json', report)
        for arm, weight in [('control', 0.), ('ordering', 1.)]:
            model.load_state_dict(cp['planner_weights'], strict=True)
            freeze_predictor(model)
            optimizer = optimizer_for(model, args.lr)
            rng = np.random.default_rng(args.seed)
            arm_report = dict(status='running', ordering_weight=weight, new_optimizer=True,
                              initial_sha256=weights_sha256(model.state_dict()), updates=[])
            report['arms'][arm] = arm_report
            selection_hash = __import__('hashlib').sha256()
            for step in range(args.steps):
                guard()
                selection = sampler.sample(rng)
                if len(np.unique(selection['seeds'])) != 1024:
                    raise ValueError('batch must contain 1024 distinct levels')
                for values in selection.values():
                    selection_hash.update(values.tobytes())
                items = selected_items(arrays['train'], supplement, compact, selection)
                for group in optimizer.param_groups:
                    group['lr'] = args.lr * schedule(step, args.steps)
                optimizer.zero_grad(set_to_none=True)
                loss, ce, order = comparator_loss(model, items, weight)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite loss')
                loss.backward()
                for parameter in model.parameters():
                    if parameter.requires_grad and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())):
                        raise ValueError('missing or nonfinite comparator gradient')
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step()
                if step == 0 or (step+1) % 100 == 0 or step+1 == args.steps:
                    row = dict(step=step+1, loss=float(loss), cross_entropy=float(ce), ordering_loss=float(order))
                    arm_report['updates'].append(row)
                    write(args.out/'report.json', report)
                    print(json.dumps(dict(event='update', arm=arm, **row)), flush=True)
            if predictor_digest(model) != frozen_sha:
                raise ValueError('frozen predictor changed')
            checkpoint = copy.copy(cp)
            checkpoint.update(planner_weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                continuation_arm=arm, source_checkpoint_sha256=args.checkpoint_sha256,
                training_stage='comparator_continuation; legacy fields describe the inherited predictor stage',
                comparator_continuation=dict(kind='safe_ordering' if weight else 'matched_ce_control', steps=args.steps,
                    learning_rate=args.lr, seed=args.seed, batch_size=1024, ordering_weight=weight,
                    predictor_frozen=True, predictor_sha256=frozen_sha, precision='float32'),
                source_sha256=report['source_sha256'])
            destination = args.out/(arm+'.pt')
            torch.save(checkpoint, destination)
            arm_report.update(status='complete', checkpoint_sha256=sha(destination),
                              sampling_sha256=selection_hash.hexdigest(), predictor_unchanged=True)
        if len({a['sampling_sha256'] for a in report['arms'].values()}) != 1:
            raise ValueError('matched arms sampled different training examples')
        bindings.verify()
        if any(sha(path) != digest for path, digest in report['source_sha256'].items()):
            raise ValueError('source/cache changed during experiment')
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, finished_local=datetime.now().astimezone().isoformat())
        write(args.out/'report.json', report)


if __name__ == '__main__':
    main()
