"""Matched generated first-error diagnostic for a frozen workspace controller.

The actor sees only its public H8 history.  At the first reachable decision at
which its committed action is not an oracle-optimal action, this diagnostic
privileges the engine only to encode the four real successor histories.  The
same frozen readout scores those actual fields and the controller's imagined
fields.  Actual futures are evidence after action selection; they are never
fed back into the actor or used to choose an action.
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

from pebby.agent import world_data as wd
from pebby.agent.model import load_checkpoint
from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from tools.build_structured_field_cache import actual_histories
from tools.diagnose_structured_workspace_trajectory import CompleteTeacher
from tools.evaluate_structured_workspace_gameplay import BANK, BANK_SHA, checked_bank
from tools.train_structured_transition import atomic_json, digest
from tools.validate_extended_collector import checked_expansion


CHECKPOINT = Path('checkpoints/ls20-structured-workspace-comparison-600-evolving.pt')
CHECKPOINT_SHA = '6a5d7716fca4403048434a8100e26770dc26fb9d19ce66600fb3eccca7d22e0c'


def resolve_checkpoint(path, expected_sha):
    """Resolve the default checkpoint or require an explicit custom binding."""
    if path is None:
        if expected_sha is not None and str(expected_sha).lower() != CHECKPOINT_SHA:
            raise ValueError('an explicit SHA for the default checkpoint must match its pinned SHA')
        return CHECKPOINT, CHECKPOINT_SHA
    path = Path(path)
    if expected_sha is None:
        raise ValueError('--checkpoint-sha256 is required with a supplied checkpoint')
    expected_sha = str(expected_sha).lower()
    if len(expected_sha) != 64 or any(char not in '0123456789abcdef' for char in expected_sha):
        raise ValueError('checkpoint SHA must be a 64-character hexadecimal digest')
    return path, expected_sha


def successor_histories(observed, valid, previous, next_frames, lost_life):
    """Build the four public H8 successor histories with the collector reset rule.

    ``lost_life`` is used only to reconstruct the public history after an
    actual branch.  It is not an actor input or an action-selection signal.
    The implementation delegates to the existing cache helper so a diagnostic
    and a future cache cannot silently disagree about reset semantics.
    """
    frames = np.asarray(observed)
    validity = np.asarray(valid)
    actions = np.asarray(previous)
    futures = np.asarray(next_frames)
    losses = np.asarray(lost_life)
    if frames.shape != (8, 64, 64) or validity.shape != (8,) or actions.shape != (8,):
        raise ValueError('expected one public H8 history')
    if validity.dtype != np.bool_ or actions.dtype.kind not in 'iu':
        raise ValueError('invalid public history dtypes')
    if futures.shape != (4, 64, 64) or futures.dtype != np.uint8:
        raise ValueError('expected four uint8 successor frames')
    if losses.shape != (4,) or losses.dtype != np.bool_:
        raise ValueError('expected four boolean life-loss labels')
    batch = {'frames': frames[None], 'history_valid': validity[None],
             'previous_actions': actions[None], 'next_frames': futures[None],
             'lost_life': losses[None]}
    histories, history_valid, previous_actions = actual_histories(batch)
    return (histories[0].numpy(), history_valid[0].numpy(), previous_actions[0].numpy())


def _choice_metrics(logits, optimal_mask, unsafe):
    values = np.asarray(logits, dtype=np.float64)
    mask = int(optimal_mask)
    unsafe = np.asarray(unsafe, dtype=np.bool_)
    if values.shape != (4,) or unsafe.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('successor scores must be finite vectors of length four')
    choice = int(values.argmax())
    return {'choice': choice, 'optimal': bool(mask & (1 << choice)),
            'unsafe': bool(unsafe[choice]), 'scores': values.tolist()}


def score_successor_views(actual_logits, imagined_logits, optimal_mask, unsafe):
    """Compare readout choices for actual and imagined four-action successors."""
    actual = _choice_metrics(actual_logits, optimal_mask, unsafe)
    imagined = _choice_metrics(imagined_logits, optimal_mask, unsafe)
    return {'actual': actual, 'imagined': imagined,
            'choice_disagrees': actual['choice'] != imagined['choice']}


def mechanic_flags(spec):
    """Stable generated-only groups available in the bank specification."""
    return {
        'fog': bool(spec.get('fog')),
        'cyclers': bool(spec.get('cyclers')),
        'launchers': bool(spec.get('launchers')),
        'refills': bool(spec.get('refills')),
        'rails': bool(spec.get('rails', spec.get('rail'))),
        'two_goals': len(spec.get('goals') or ()) >= 2,
        'long_route_ge48': int(spec.get('context_optimal_actions', 0)) >= 48,
    }


def aggregate_levels(levels):
    """Aggregate first-error outcomes and mechanic subgroups without rates-only loss."""
    counts = Counter(row.get('outcome') for row in levels)
    result = {'levels': len(levels), 'first_error': counts['first_error'],
              'no_error': counts['no_error'], 'unknown': counts['unknown'],
              'outcome_counts': dict(counts), 'mechanics': {}}
    first_errors = [row for row in levels if row.get('outcome') == 'first_error']
    diagnostic_errors = [row for row in first_errors
                         if isinstance(row.get('first_error'), dict)
                         and isinstance(row['first_error'].get('successor_views'), dict)]
    result['first_error_successors'] = {
        'states': len(diagnostic_errors),
        'actual_optimal': sum(bool(row['first_error']['successor_views']['actual']['optimal'])
                              for row in diagnostic_errors),
        'imagined_optimal': sum(bool(row['first_error']['successor_views']['imagined']['optimal'])
                                for row in diagnostic_errors),
        'actual_unsafe': sum(bool(row['first_error']['successor_views']['actual']['unsafe'])
                             for row in diagnostic_errors),
        'imagined_unsafe': sum(bool(row['first_error']['successor_views']['imagined']['unsafe'])
                               for row in diagnostic_errors),
        'choice_disagrees': sum(bool(row['first_error']['successor_views']['choice_disagrees'])
                                for row in diagnostic_errors),
        'imagined_public_mismatches': sum(
            row['first_error']['successor_views'].get('imagined_matches_public') is False
            for row in diagnostic_errors),
    }
    keys = ('fog', 'cyclers', 'launchers', 'refills', 'rails', 'two_goals', 'long_route_ge48')
    for key in keys:
        subset = [row for row in levels if row.get('mechanics', {}).get(key)]
        sub = Counter(row.get('outcome') for row in subset)
        result['mechanics'][key] = {'levels': len(subset),
                                    'first_error': sub['first_error'],
                                    'no_error': sub['no_error'], 'unknown': sub['unknown']}
    return result


def _public_policy_scores(policy, observed, valid, previous):
    frames = torch.from_numpy(np.asarray(observed)[None]).long()
    validity = torch.from_numpy(np.asarray(valid)[None]).bool()
    actions = torch.from_numpy(np.asarray(previous)[None]).long()
    scores = policy(frames, history_valid=validity, previous_actions=actions)
    if tuple(scores.shape) != (1, 4) or not bool(torch.isfinite(scores).all()):
        raise ValueError('invalid frozen public policy scores')
    return scores[0]


@torch.inference_mode()
def matched_successor_scores(policy, observed, valid, previous, targets, public_logits=None):
    """Score actual and model-imagined successor fields with one frozen readout."""
    next_frames = np.asarray(targets['next_frames'], dtype=np.uint8)
    lost_life = np.asarray(targets['lost_life'], dtype=np.bool_)
    histories, validity, actions = successor_histories(observed, valid, previous,
                                                        next_frames, lost_life)
    current = policy.encoder(torch.from_numpy(np.asarray(observed)[None]).long(),
                             torch.from_numpy(np.asarray(valid)[None]).bool(),
                             torch.from_numpy(np.asarray(previous)[None]).long())
    imagined_fields = policy.successor_fields(current)
    actual_fields = policy.encoder(torch.from_numpy(histories.reshape(4, 8, 64, 64)).long(),
                                   torch.from_numpy(validity.reshape(4, 8)).bool(),
                                   torch.from_numpy(actions.reshape(4, 8)).long())
    imagined_logits = policy.readout(imagined_fields)[0]
    actual_logits = policy.readout(actual_fields[None])[0]
    # StructuredWorkspaceReadout returns one scalar per successor.  Keep the
    # check explicit: a changed readout API must not silently alter this audit.
    if tuple(actual_logits.shape) != (4,) or tuple(imagined_logits.shape) != (4,):
        raise ValueError('workspace successor readout must return four scalar scores')
    imagined_numpy = imagined_logits.cpu().numpy()
    public_match = None
    if public_logits is not None:
        public_numpy = np.asarray(public_logits, dtype=np.float64)
        if public_numpy.shape != (4,) or not np.isfinite(public_numpy).all():
            raise ValueError('committed public scores must be a finite length-four vector')
        public_match = bool(np.allclose(public_numpy, imagined_numpy, rtol=1e-5, atol=1e-5))
        if not public_match or int(public_numpy.argmax()) != int(imagined_numpy.argmax()):
            raise ValueError('reconstructed imagined readout disagrees with committed public logits')
    distances = np.asarray(targets['distances'])
    unsafe = lost_life | (distances < 0)
    views = score_successor_views(actual_logits.cpu().numpy(), imagined_numpy,
                                  int(targets['optimal']), unsafe)
    views['imagined_matches_public'] = public_match
    return views, {
        'actual_reset_branches': int(lost_life.sum()),
        'actual_terminal_branches': int(np.asarray(targets['terminal']).sum()),
        'actual_winning_branches': int(np.asarray(targets['won']).sum()),
        'actual_unsafe_branches': int(unsafe.sum()),
    }


def _diagnose_level(policy, env, teacher, spec, max_actions, seed, expansion_checks):
    """Run public actor until its first reachable nonoptimal committed action."""
    frame = env.reset()
    frames, actions = [frame], [-1]
    decisions = []
    ending = 'capped'
    first = None
    for step in range(max_actions):
        observed, valid, previous = wd.history_arrays(frames, actions, 8)
        # This is the only action-selection call.  The teacher is queried only
        # after the public action has been committed.
        public_scores = _public_policy_scores(policy, observed, valid, previous)
        choice = int(public_scores.argmax())
        label = teacher.label(env)
        if label['status'] != 'reachable':
            ending = label['status']
            break
        evidence = teacher.before(env, choice)
        old_lives, old_level = env.lives(), env.level_index
        row = {'step': step, 'choice': choice, 'distance': label['distance'],
               'optimal_mask': label['optimal_mask'],
               'optimal': bool(label['optimal_mask'] & (1 << choice))}
        if not row['optimal']:
            # The actor choice is already fixed. Branch expansion and actual
            # successor encoding are post-decision privileged diagnostics.
            original = wd._expand
            def checked(*args, **kwargs):
                return checked_expansion(original, expansion_checks, *args, **kwargs)
            from unittest.mock import patch
            with patch.object(wd, '_expand', side_effect=checked):
                targets, branches, results, branch_mask = wd._expand(
                    env, teacher.oracle, int(label['distance']), seed, step)
            if int(branch_mask) != int(label['optimal_mask']):
                raise ValueError('first-error branch mask disagrees with teacher label')
            views, branch_counts = matched_successor_scores(
                policy, observed, valid, previous, targets, public_scores.detach().cpu().numpy())
            first = {**row, 'successor_views': views, 'successor_counts': branch_counts,
                     'public_scores': public_scores.detach().cpu().tolist(),
                     'targets': {'distances': np.asarray(targets['distances']).tolist(),
                                 'terminal': np.asarray(targets['terminal']).tolist(),
                                 'won': np.asarray(targets['won']).tolist(),
                                 'lost_life': np.asarray(targets['lost_life']).tolist(),
                                 'next_optimal': np.asarray(targets['next_optimal']).tolist()}}
            # Check the committed action's actual transition, then stop.  No
            # actual branch is selected in place of the actor's choice.
            observation = env.perform(names.ACTION_IDS[choice])
            teacher.after(env, observation, evidence)
            ending = 'first_error'
            break
        observation = env.perform(names.ACTION_IDS[choice])
        teacher.after(env, observation, evidence)
        if observation.frame is None:
            raise ValueError('live actor action omitted public successor')
        loss = env.lives() < old_lives
        decisions.append(row)
        if observation.finished:
            ending = 'win' if observation.won else 'game_over'
            break
        if loss:
            frames, actions = [observation.frame], [-1]
        else:
            frames, actions = (frames + [observation.frame])[-8:], (actions + [choice])[-8:]
        if env.level_index != old_level:
            frames, actions = [observation.frame], [-1]
    if first is not None:
        return {'outcome': 'first_error', 'first_error': first, 'actions_before_error': len(decisions),
                'ending': ending}
    if ending in ('unknown', 'unknown_graph_coverage', 'unknown_truncated', 'unknown_unsupported', 'unreachable'):
        return {'outcome': 'unknown', 'unknown_status': ending, 'actions_before_error': len(decisions),
                'ending': ending}
    return {'outcome': 'no_error', 'actions_before_error': len(decisions), 'ending': ending,
            'completed': ending == 'win'}


def run_level(policy, level, spec, max_actions, search_limit, checks):
    env = Ls20Scenario(level, spec['training_context_index'])
    env.reset()
    teacher = CompleteTeacher(env, spec, search_limit)
    row = {'seed': int(spec['seed']), 'difficulty': int(spec['difficulty']),
           'context': int(spec['training_context_index']), 'mechanics': mechanic_flags(spec),
           'teacher': teacher.proof}
    if not teacher.covered:
        row.update(outcome='unknown', unknown_status=teacher.proof['status'], ending='teacher_unavailable')
        return row
    row.update(_diagnose_level(policy, env, teacher, spec, max_actions, int(spec['seed']), checks))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='workspace checkpoint; custom paths require --checkpoint-sha256')
    parser.add_argument('--checkpoint-sha256', default=None,
                        help='SHA-256 of the exact checkpoint bytes consumed')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--levels', type=int, default=100)
    parser.add_argument('--max-actions', type=int, default=200)
    parser.add_argument('--search-limit', type=int, default=600000)
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args(argv)
    if (args.report.exists() or not 1 <= args.levels <= 100 or not 1 <= args.max_actions <= 200
            or not 1 <= args.search_limit <= 600000 or not 1 <= args.seconds <= 1200):
        parser.error('new report and bounded generated monitor settings required')
    try:
        checkpoint, checkpoint_sha = resolve_checkpoint(args.checkpoint, args.checkpoint_sha256)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic(); print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('first-error deadline')))
    signal.alarm(args.seconds)
    paths = [checkpoint, BANK, Path(__file__),
             Path('tools/diagnose_structured_workspace_trajectory.py'),
             Path('tools/evaluate_structured_workspace_gameplay.py'),
             Path('tools/build_structured_field_cache.py'), Path('tools/validate_extended_collector.py'),
             Path('pebby/agent/world_data.py'), Path('pebby/agent/model.py'),
             Path('pebby/agent/history.py'), Path('pebby/agent/structured_policy.py'),
             Path('pebby/agent/structured_factored_policy.py'),
             Path('pebby/agent/structured_workspace_controller.py'),
             Path('pebby/ls20/env.py'), Path('pebby/ls20/plan.py'), Path('pebby/ls20/layout.py')]
    sources = {str(p): digest(p) for p in paths}
    report = {'status': 'running', 'pid': os.getpid(), 'source': 'generated_only',
              'official_inputs_used': False, 'training_performed': False,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': sources[str(checkpoint)],
              'checkpoint_expected_sha256': checkpoint_sha,
              'bank': str(BANK), 'bank_sha256': BANK_SHA, 'levels_requested': args.levels,
              'max_actions': args.max_actions, 'search_limit': args.search_limit,
              'device': 'cpu', 'precision': 'FP32, TF32 off', 'levels': [],
              'limits': ['Actor choices use only public H8 frames, validity, and previous actions.',
                         'Oracle and actual successors are post-decision diagnostics only.',
                         'First-error counts are diagnostic associations, not unique causal proof.',
                         'Unreachable means no completion before another life loss; it is not whole-episode impossibility.']}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        atomic_json(args.report, report)
    persist()
    policy = None
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
        levels, _, specs = checked_bank()
        report['selected_seeds'] = [int(s['seed']) for s in specs[:args.levels]]
        checks = Counter()
        for level, spec in zip(levels[:args.levels], specs[:args.levels]):
            entry = run_level(policy, level, spec, args.max_actions, args.search_limit, checks)
            report['levels'].append(entry)
            report['aggregate'] = aggregate_levels(report['levels'])
            report['branch_checks'] = dict(checks)
            persist()
            print(json.dumps({'seed': entry['seed'], 'outcome': entry['outcome'],
                              'ending': entry.get('ending')}), flush=True)
        for path, sha in sources.items():
            if digest(path) != sha:
                raise ValueError(f'source changed during diagnostic: {path}')
        report.update(status='complete', source_unchanged=True, source_hashes=sources,
                      aggregate=aggregate_levels(report['levels']), branch_checks=dict(checks))
    except BaseException as error:
        report.update(status='failed_partial', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        policy = None; gc.collect(); signal.alarm(0); persist()


if __name__ == '__main__':
    main()
