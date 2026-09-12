"""Instrument first-error states with actual/imagined field diagnostics.

This sibling imports the frozen first-error diagnostic and intercepts only its
post-decision successor scoring.  It never supplies actual futures to the
actor.  Channel replacement is an offline sensitivity probe: mixed fields can
be off-manifold, so an intervention is not a causal retraining result.
"""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch

from pebby.agent.model import load_checkpoint
from tools import diagnose_workspace_first_errors as base
from tools.train_structured_transition import atomic_json, digest


FIELD_GROUPS = {
    'encoder_state': (0, 48),
    'appearance': (48, 70),
    'carried_glyph': (70, 84),
    'visibility': (84, 85),
    'reserved': (85, 96),
}


def replace_actual_group(imagined, actual, start, end):
    """Replace a channel interval while preserving every action and token."""
    if not isinstance(imagined, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise ValueError('field intervention expects tensors')
    if imagined.shape != actual.shape or imagined.ndim != 3 or tuple(imagined.shape[1:]) != (148, 96):
        raise ValueError('field intervention expects matching [4,148,96] tensors')
    if not 0 <= start < end <= 96:
        raise ValueError('invalid field channel interval')
    hybrid = imagined.clone()
    hybrid[..., start:end] = actual[..., start:end]
    return hybrid


def _group_error(actual, imagined, start, end):
    delta = (imagined[..., start:end] - actual[..., start:end]).astype(np.float64)
    flat = delta.reshape(delta.shape[0], -1)
    return {'mae': float(np.abs(flat).mean()), 'mse': float(np.square(flat).mean()),
            'max_abs': float(np.abs(flat).max()),
            'per_branch_mse': np.square(flat).mean(1).tolist()}


def field_group_errors(actual, imagined):
    """Return compact per-channel-group errors for [4,148,96] fields."""
    actual = np.asarray(actual, dtype=np.float32)
    imagined = np.asarray(imagined, dtype=np.float32)
    if actual.shape != (4, 148, 96) or imagined.shape != actual.shape:
        raise ValueError('expected four actual and imagined [148,96] fields')
    if not np.isfinite(actual).all() or not np.isfinite(imagined).all():
        raise ValueError('nonfinite field diagnostic')
    return {name: _group_error(actual, imagined, *span) for name, span in FIELD_GROUPS.items()}


def _triple_predictions(readout):
    return np.stack([readout[f'carried_{name}_logits'].argmax(-1).cpu().numpy()
                     for name in ('shape', 'color', 'rotation')], axis=-1)


def semantic_readout(readout, player_cells, triples):
    """Compare frozen dynamics readouts with engine labels for diagnostics."""
    cells = np.asarray(player_cells, dtype=np.int64).reshape(-1, 2)
    target_player = cells[:, 1] * 12 + cells[:, 0]
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    predicted_player = readout['player_logits'].argmax(-1).cpu().numpy()
    predicted_triple = _triple_predictions(readout)
    return {
        'player_correct': (predicted_player == target_player).tolist(),
        'glyph_shape_correct': (predicted_triple[:, 0] == triples[:, 0]).tolist(),
        'glyph_color_correct': (predicted_triple[:, 1] == triples[:, 1]).tolist(),
        'glyph_rotation_correct': (predicted_triple[:, 2] == triples[:, 2]).tolist(),
        'glyph_joint_correct': (predicted_triple == triples).all(1).tolist(),
        'player_prediction': predicted_player.tolist(),
        'glyph_prediction': predicted_triple.tolist(),
        'target_player': target_player.tolist(),
        'target_triple': triples.tolist(),
    }


def raw_field_semantics(fields, player_cells, triples):
    """Decode only the documented raw player/glyph field channels.

    These are direct field probes, separate from the learned transition
    readout above.  Carried glyph channels are pooled because the public
    encoder broadcasts them; predicted fields are checked for token drift.
    """
    values = fields.detach().cpu().numpy() if isinstance(fields, torch.Tensor) else np.asarray(fields)
    if values.ndim != 3 or values.shape[1:] != (148, 96):
        raise ValueError('raw field semantics expects [B,148,96]')
    cells = np.asarray(player_cells, dtype=np.int64).reshape(-1, 2)
    target_player = cells[:, 1] * 12 + cells[:, 0]
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    player_prediction = values[:, :144, 55].argmax(1)
    pooled = values[:, :, 70:84].mean(1)
    glyph_prediction = np.stack([pooled[:, :6].argmax(1), pooled[:, 6:10].argmax(1),
                                 pooled[:, 10:14].argmax(1)], axis=-1)
    token_prediction = np.stack([values[:, :, 70:76].argmax(-1),
                                 values[:, :, 76:80].argmax(-1),
                                 values[:, :, 80:84].argmax(-1)], axis=-1)
    pooled_choice = glyph_prediction[:, None, :]
    return {
        'player_correct': (player_prediction == target_player).tolist(),
        'glyph_shape_correct': (glyph_prediction[:, 0] == triples[:, 0]).tolist(),
        'glyph_color_correct': (glyph_prediction[:, 1] == triples[:, 1]).tolist(),
        'glyph_rotation_correct': (glyph_prediction[:, 2] == triples[:, 2]).tolist(),
        'glyph_joint_correct': (glyph_prediction == triples).all(1).tolist(),
        'player_prediction': player_prediction.tolist(),
        'glyph_prediction': glyph_prediction.tolist(),
        'target_player': target_player.tolist(), 'target_triple': triples.tolist(),
        'glyph_token_consistency': (token_prediction == pooled_choice).all(-1).mean(1).tolist(),
        'glyph_token_std': values[:, :, 70:84].std(1).mean(1).tolist(),
    }


def _choice(logits, optimal_mask, unsafe):
    values = np.asarray(logits, dtype=np.float64)
    choice = int(values.argmax())
    return {'choice': choice, 'optimal': bool(int(optimal_mask) & (1 << choice)),
            'unsafe': bool(np.asarray(unsafe, dtype=bool)[choice]),
            'changed_from_imagined': None, 'scores': values.tolist()}


@torch.inference_mode()
def instrumented_successor_scores(policy, observed, valid, previous, targets, public_logits, original):
    """Call the frozen scorer, then collect actual/imagined field evidence."""
    views, counts = original(policy, observed, valid, previous, targets, public_logits)
    histories, history_valid, previous_actions = base.successor_histories(
        observed, valid, previous, targets['next_frames'], targets['lost_life'])
    current = policy.encoder(torch.from_numpy(np.asarray(observed)[None]).long(),
                             torch.from_numpy(np.asarray(valid)[None]).bool(),
                             torch.from_numpy(np.asarray(previous)[None]).long())
    imagined = policy.successor_fields(current)[0]
    actual = policy.encoder(torch.from_numpy(histories.reshape(4, 8, 64, 64)).long(),
                            torch.from_numpy(history_valid.reshape(4, 8)).bool(),
                            torch.from_numpy(previous_actions.reshape(4, 8)).long())
    actual_np, imagined_np = actual.cpu().numpy(), imagined.cpu().numpy()
    target_player = np.asarray(targets['next_player_cell'])
    target_triple = np.asarray(targets['next_triple'])
    actual_readout = policy.dynamics.readout(actual)
    imagined_readout = policy.dynamics.readout(imagined)
    current_readout = policy.dynamics.readout(current)
    current_player = np.asarray(targets['player_cell'])[None]
    current_triple = np.asarray(targets['current_triple'])[None]
    unsafe = np.asarray(targets['lost_life'], dtype=bool) | (np.asarray(targets['distances']) < 0)
    actual_action_logits = policy.readout(actual[None])[0].cpu().numpy()
    imagined_action_logits = policy.readout(imagined[None])[0].cpu().numpy()
    baseline_choice = int(np.asarray(public_logits).argmax())
    interventions = {}
    for name, (start, end) in FIELD_GROUPS.items():
        hybrid = replace_actual_group(imagined, actual, start, end)
        logits = policy.readout(hybrid[None])[0].cpu().numpy()
        item = _choice(logits, targets['optimal'], unsafe)
        item['changed_from_imagined'] = item['choice'] != baseline_choice
        interventions[name] = item
    full_actual = _choice(actual_action_logits, targets['optimal'], unsafe)
    full_actual['changed_from_imagined'] = full_actual['choice'] != baseline_choice
    interventions['all_actual'] = full_actual
    branch_rows = []
    for action in range(4):
        branch_rows.append({
            'action': action,
            'chosen': action == baseline_choice,
            'optimal': bool(int(targets['optimal']) & (1 << action)),
            'unsafe': bool(unsafe[action]),
            'field_error': {name: values['per_branch_mse'][action]
                            for name, values in field_group_errors(actual_np, imagined_np).items()},
            'actual_semantics': {key: values[action] for key, values in
                                 semantic_readout(actual_readout, target_player, target_triple).items()},
            'imagined_semantics': {key: values[action] for key, values in
                                  semantic_readout(imagined_readout, target_player, target_triple).items()},
        })
    return views, {**counts,
        'field_groups': field_group_errors(actual_np, imagined_np),
        'current_semantics': semantic_readout(current_readout, current_player, current_triple),
        'current_raw_semantics': raw_field_semantics(current, current_player, current_triple),
        'actual_next_semantics': semantic_readout(actual_readout, target_player, target_triple),
        'imagined_next_semantics': semantic_readout(imagined_readout, target_player, target_triple),
        'actual_raw_next_semantics': raw_field_semantics(actual, target_player, target_triple),
        'imagined_raw_next_semantics': raw_field_semantics(imagined, target_player, target_triple),
        'branch_rows': branch_rows,
        'interventions': interventions,
        'target_next_player_cell': target_player.tolist(),
        'target_next_triple': target_triple.tolist(),
        'target_distances': np.asarray(targets['distances']).tolist(),
        'target_lost_life': np.asarray(targets['lost_life']).tolist(),
        'target_terminal': np.asarray(targets['terminal']).tolist(),
        'target_won': np.asarray(targets['won']).tolist(),
        'imagined_matches_public': views.get('imagined_matches_public'),
    }


def _summarize(levels):
    errors = [row['first_error']['field_diagnostics'] for row in levels
              if row.get('outcome') == 'first_error' and 'first_error' in row
              and 'field_diagnostics' in row['first_error']]
    result = {'first_error_states': len(errors)}
    for name in FIELD_GROUPS:
        values = [item['field_groups'][name]['mse'] for item in errors]
        result[name + '_mse_mean'] = float(np.mean(values)) if values else None
    result['actual_next_player_correct'] = sum(
        sum(item['actual_next_semantics']['player_correct']) for item in errors)
    result['imagined_next_player_correct'] = sum(
        sum(item['imagined_next_semantics']['player_correct']) for item in errors)
    result['actual_next_glyph_joint_correct'] = sum(
        sum(item['actual_next_semantics']['glyph_joint_correct']) for item in errors)
    result['imagined_next_glyph_joint_correct'] = sum(
        sum(item['imagined_next_semantics']['glyph_joint_correct']) for item in errors)
    result['actual_raw_next_glyph_joint_correct'] = sum(
        sum(item['actual_raw_next_semantics']['glyph_joint_correct']) for item in errors)
    result['imagined_raw_next_glyph_joint_correct'] = sum(
        sum(item['imagined_raw_next_semantics']['glyph_joint_correct']) for item in errors)
    result['actual_raw_next_player_correct'] = sum(
        sum(item['actual_raw_next_semantics']['player_correct']) for item in errors)
    result['imagined_raw_next_player_correct'] = sum(
        sum(item['imagined_raw_next_semantics']['player_correct']) for item in errors)
    result['intervention_choice_changes'] = {
        name: sum(bool(item['interventions'][name]['changed_from_imagined']) for item in errors)
        for name in (*FIELD_GROUPS, 'all_actual')}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--checkpoint-sha256', default=None)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--levels', type=int, default=100)
    parser.add_argument('--max-actions', type=int, default=200)
    parser.add_argument('--search-limit', type=int, default=600000)
    parser.add_argument('--seconds', type=int, default=300)
    args = parser.parse_args(argv)
    if (args.report.exists() or not 1 <= args.levels <= 100 or not 1 <= args.max_actions <= 200
            or not 1 <= args.seconds <= 300 or not 1 <= args.search_limit <= 600000):
        parser.error('new report and bounded generated monitor settings required')
    try:
        checkpoint, checkpoint_sha = base.resolve_checkpoint(args.checkpoint, args.checkpoint_sha256)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic(); print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('field diagnostic deadline')))
    signal.alarm(args.seconds)
    paths = [checkpoint, base.BANK, Path(__file__), Path(base.__file__),
             Path('tools/build_structured_field_cache.py'),
             Path('tools/validate_extended_collector.py'), Path('pebby/agent/world_data.py'),
             Path('pebby/agent/structured_transition.py'), Path('pebby/agent/structured_policy.py'),
             Path('pebby/agent/structured_factored_policy.py'),
             Path('pebby/agent/structured_workspace_controller.py')]
    sources = {str(path): digest(path) for path in paths}
    report = {'status': 'running', 'pid': os.getpid(), 'source': 'generated_only',
              'official_inputs_used': False, 'training_performed': False,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': sources[str(checkpoint)],
              'checkpoint_expected_sha256': checkpoint_sha, 'bank': str(base.BANK),
              'bank_sha256': base.BANK_SHA, 'levels_requested': args.levels,
              'max_actions': args.max_actions, 'device': 'cpu', 'precision': 'FP32, TF32 off',
              'levels': [], 'intervention_caveat':
              'Replacing predicted field groups with actual groups creates hybrid fields; choice changes are sensitivity evidence, not causal retraining results.',
              'limits': ['Actual futures and engine targets are post-decision diagnostic privileges only.',
                         'First-error selection and engine semantic labels do not prove a unique cause.',
                         'Carried/player semantic scores are diagnostic and can be hidden or occluded in public pixels.']}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        atomic_json(args.report, report)
    persist()
    policy = None
    original_scorer = base.matched_successor_scores
    captured = []
    try:
        policy, info = load_checkpoint(checkpoint, 'cpu')
        if digest(checkpoint) != checkpoint_sha:
            raise ValueError('checkpoint changed while loading')
        if (info.get('format') != 'pebby.structured-workspace-readout.v1'
                or info.get('readout_config', {}).get('memory_mode') != 'evolving'
                or info.get('readout_config', {}).get('loops') != 2):
            raise ValueError('evolving depth-2 generated-only workspace controller required')
        policy.float().eval().requires_grad_(False)
        sources.update(policy.sources['code_hashes'])

        def instrumented(policy_, observed, valid, previous, targets, public_logits=None):
            views, evidence = instrumented_successor_scores(
                policy_, observed, valid, previous, targets, public_logits, original_scorer)
            captured.append(evidence)
            return views, evidence

        base.matched_successor_scores = instrumented
        levels, _, specs = base.checked_bank()
        report['selected_seeds'] = [int(spec['seed']) for spec in specs[:args.levels]]
        checks = Counter()
        for level, spec in zip(levels[:args.levels], specs[:args.levels]):
            row = base.run_level(policy, level, spec, args.max_actions, args.search_limit, checks)
            if row.get('outcome') == 'first_error':
                if not captured:
                    raise ValueError('first-error row lacked instrumentation')
                row['first_error']['field_diagnostics'] = captured.pop(0)
            report['levels'].append(row)
            report['aggregate'] = base.aggregate_levels(report['levels'])
            report['field_summary'] = _summarize(report['levels'])
            report['branch_checks'] = dict(checks)
            persist()
            print(json.dumps({'seed': row['seed'], 'outcome': row['outcome']}), flush=True)
        if captured:
            raise ValueError('instrumentation records were not attached to first-error rows')
        for path, sha in sources.items():
            if digest(path) != sha:
                raise ValueError(f'source changed during field diagnostic: {path}')
        report.update(status='complete', source_unchanged=True, source_hashes=sources,
                      aggregate=base.aggregate_levels(report['levels']),
                      field_summary=_summarize(report['levels']), branch_checks=dict(checks))
    except BaseException as error:
        report.update(status='failed_partial', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        base.matched_successor_scores = original_scorer
        policy = None; gc.collect(); signal.alarm(0); persist()


if __name__ == '__main__':
    main()
