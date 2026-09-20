"""Actual sequential gameplay is the admission gate; offline scores are not.

This local runner uses native lives and the existing GAME_OVER-only RESET
adapter. It does not produce an official API scorecard, learn voluntary RESET,
or add persistent game memory. A better single session is eligible for further
gameplay confirmation, never automatic promotion.
"""
import hashlib
import json
from numbers import Integral
from pathlib import Path

from . import competition


GATE_FORMAT = 'pebby.gameplay-gate.v1'
EVALUATION_KIND = 'actual_sequential_gameplay'
_SOURCE_KEYS = {'gameplay_gate', 'competition', 'history', 'frame_adapter',
                'environment', 'game', 'engine'}
_CONTRACT_KEYS = (
    'format', 'protocol', 'official_api_scorecard', 'full_game_initializations',
    'full_game_resets_after_initialization', 'reset_count_cap',
    'native_lives_per_level_initialization', 'action_accounting',
    'per_level_caps', 'levels_total', 'initial_frame_sha256', 'decision',
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources():
    from arcengine import base_game
    from . import history, model
    from pebby.ls20 import env
    paths = dict(gameplay_gate=__file__, competition=competition.__file__,
                 history=history.__file__, frame_adapter=model.__file__,
                 environment=env.__file__, game=env.UPSTREAM,
                 engine=base_game.__file__)
    return {name: dict(path=str(Path(path).resolve()), sha256=_digest(path))
            for name, path in paths.items()}


def _identity(report, sources):
    payload = dict(contract={key: report[key] for key in _CONTRACT_KEYS},
                   source_sha256={key: value['sha256'] for key, value in sources.items()})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def evaluate_sequential(policy, device, guard=lambda: None, per_level_cap=300):
    """Run exactly one fresh shipped seven-level game, using public pixels.

    ``guard`` may raise a caller-owned budget/cancellation exception. It runs
    before construction, at every decision boundary and after observations,
    including the winning observation. Exceptions propagate without returning
    a partial report as successful evidence. Policy checkpoint binding remains
    the caller's responsibility; evaluator/engine sources are bound here.
    """
    if isinstance(per_level_cap, bool) or not isinstance(per_level_cap, Integral) or per_level_cap < 1:
        raise ValueError('per_level_cap must be a positive integer')
    if not callable(guard):
        raise ValueError('guard must be callable')
    guard()
    sources = _sources()
    guard()

    class GuardedDecision(competition.FourMovementDecision):
        def start(self, context):
            guard()
            super().start(context)

        def decide(self, context):
            guard()
            return super().decide(context)

        def observe(self, observation, event):
            super().observe(observation, event)
            guard()

    session = competition.CompetitionSession()
    if session.level_count != 7:
        raise ValueError('sequential gameplay gate requires exactly seven shipped levels')
    report = competition.run_competition(GuardedDecision(policy, device), session,
                                         per_level_caps=[int(per_level_cap)] * 7)
    guard()
    if any(_digest(value['path']) != value['sha256'] for value in sources.values()):
        raise RuntimeError('gameplay evaluator or engine sources changed during evaluation')
    report.update(evaluation_kind=EVALUATION_KIND, source_bindings=sources,
                  source_unchanged=True, protocol_identity=_identity(report, sources),
                  execution_device=str(device),
                  limitations=[
                      'Local actual gameplay; no official API scorecard.',
                      'GAME_OVER RESET is an adapter; voluntary RESET is not learned.',
                      'History clears at boundaries; persistent game memory is absent.',
                      'One session alone cannot automatically promote a checkpoint.',
                  ])
    return report


def _evidence_problem(report):
    """Check accounting and protocol provenance; never interpret offline fields."""
    if not isinstance(report, dict):
        return 'missing session report'
    try:
        sources = report['source_bindings']
        if (not isinstance(sources, dict) or set(sources) != _SOURCE_KEYS
                or any(not isinstance(value['sha256'], str) or len(value['sha256']) != 64
                       or any(char not in '0123456789abcdef' for char in value['sha256'])
                       for value in sources.values())):
            return 'missing or invalid gameplay source bindings'
        if (report.get('evaluation_kind') != EVALUATION_KIND
                or report.get('source_unchanged') is not True
                or report['protocol_identity'] != _identity(report, report['source_bindings'])):
            return 'missing or inconsistent gameplay protocol identity'
        if (report['format'] != competition.REPORT_FORMAT
                or report['protocol'] != 'single_sequential_level_reset_session'
                or report['official_api_scorecard'] is not False
                or report['full_game_initializations'] != 1
                or report['full_game_resets_after_initialization'] != 0
                or report['reset_count_cap'] is not None
                or report['native_lives_per_level_initialization'] != 3
                or report['decision'] != competition.FourMovementDecision.metadata
                or report['action_accounting']['scorecard_reset_action_cost'] != 1
                or report['action_accounting']['engine_reset_budget_cost'] != 0):
            return 'unsupported sequential gameplay protocol'
        total, progress, actions = report['levels_total'], report['levels_completed'], report['actions']
        if (type(total) is not int or total != 7 or type(progress) is not int
                or not 0 <= progress <= total or type(actions) is not int or actions < 0
                or type(report['completed']) is not bool
                or type(report['lives_left']) is not int or not 0 <= report['lives_left'] <= 3):
            return 'invalid gameplay outcome'
        won = report['completed']
        if (won != (progress == total) or won != (report['ending'] == 'win')
                or won != (report['final_state'] == competition.GameState.WIN.value)
                or report['ending'] not in ('win', 'per_level_action_cap')):
            return 'inconsistent completion or interrupted session'
        caps, counts, ledger = report['per_level_caps'], report['per_level_actions'], report['ledger']
        if (len(caps) != total or len(counts) != total or len(ledger) != actions
                or any(type(cap) is not int or cap < 1 for cap in caps)
                or any(type(count) is not int or not 0 <= count <= cap
                       for count, cap in zip(counts, caps)) or sum(counts) != actions):
            return 'inconsistent action accounting'
        if report['resets'] != sum(row['action'] == 0 for row in ledger):
            return 'inconsistent RESET accounting'
        previous_hash, previous_progress = report['initial_frame_sha256'], 0
        for index, row in enumerate(ledger, 1):
            if (row['charged_actions'] != index or row['frame_before_sha256'] != previous_hash
                    or row['levels_completed_before'] != previous_progress
                    or row['levels_completed'] not in (previous_progress, previous_progress + 1)):
                return 'discontinuous gameplay ledger'
            previous_hash, previous_progress = row['frame_after_sha256'], row['levels_completed']
        if previous_progress != progress:
            return 'gameplay ledger disagrees with completion'
        if ledger and (ledger[-1]['state_after'] != report['final_state']
                       or ledger[-1]['lives_after'] != report['lives_left']):
            return 'gameplay ledger disagrees with final state'
    except (KeyError, TypeError, ValueError, AttributeError):
        return 'incomplete gameplay evidence'
    return None


def assess_gameplay(baseline, candidate):
    """Compare whole sessions under identical rules; no offline tie-breaker.

    A strict progress gain only qualifies for further gameplay confirmation.
    Generated-game comparisons, repeated sessions and checkpoint promotion are
    separate caller-owned decisions. A seven-level win is objective evidence
    only when both reports carry a matching, valid protocol identity.
    """
    result = dict(format=GATE_FORMAT, status='no_evidence', promoted=False,
                  objective_evidence=False, offline_metrics_used=False,
                  confirmation_required=True)
    for name, report in (('baseline', baseline), ('candidate', candidate)):
        problem = _evidence_problem(report)
        if problem:
            return {**result, 'reason': f'{name}: {problem}'}
    if baseline['protocol_identity'] != candidate['protocol_identity']:
        return {**result, 'reason': 'gameplay protocol identities differ'}
    before, after = baseline['levels_completed'], candidate['levels_completed']
    result.update(baseline_levels_completed=before, candidate_levels_completed=after,
                  protocol_identity=candidate['protocol_identity'],
                  objective_evidence=candidate['completed'] and after == 7)
    if after < before:
        result.update(status='reject', reason='candidate completes fewer sequential levels')
    elif after == before:
        result.update(status='tied', reason='equal sequential progress; offline scores cannot promote')
    else:
        result.update(status='eligible_for_gameplay_confirmation',
                      reason='strict sequential progress gain; further gameplay confirmation required')
    return result
