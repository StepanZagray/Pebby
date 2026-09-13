"""Paired native evaluation for retained and route-repair spatial policies.

Generated mode plays one isolated three-life episode per bank row.  Shipped mode
plays one numerical, sequential seven-level competition session.  Both modes
use the same public-history loop, strict argmax, action caps, and optional
public-frame-only ``next-best`` stall controller for every checkpoint arm.

The evaluator is deliberately passive.  Engine state is read after an action
only for diagnostics; it is never supplied to the policy.  In particular,
``blocked_no_player_motion_budget_spending`` counts a movement action when the
engine reports the same player cell, level, lives, carried triple, and solved
goals, while ``steps_left`` decreases and the action is not terminal.  This
captures charged wall/obstacle bumps that can change the HUD while separating
them from exact-frame refusals.  Transform/refill/goal changes, life loss,
level transitions, wins, resets, and non-charged refusals are excluded.
A charged stationary action can still be intentional game behavior (for
example, a patroller wait); this diagnostic describes the transition and does
not label it a policy error.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from pebby.agent import competition, evaluate
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent import spatial_outcome_policy
from pebby.agent import spatial_route_outcome_policy
from pebby.agent import spatial_semantic_outcome_policy
from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from tools import evaluate_spatial_recovery as recovery_tool
from tools.evaluate_reference_spatial_outcomes import guard, sha, write
from tools.spatial_paired_statistics import paired_win_statistics


ROOT = Path(__file__).resolve().parents[1]
ROUTE_FORMAT = spatial_route_outcome_policy.FORMAT
SPATIAL_FORMAT = spatial_outcome_policy.FORMAT
SEMANTIC_FORMAT = spatial_semantic_outcome_policy.FORMAT


def _proc_start_ticks():
    """Read this process' Linux start tick without relying on a PID alone."""
    fields = Path('/proc/self/stat').read_text().rpartition(')')[2].split()
    return int(fields[19])


def _snapshot(env):
    """Engine-observed state used only for post-action diagnostics."""
    return dict(level=int(env.level_index), cell=tuple(env.player_cell()),
                triple=tuple(env.triple()), goals=tuple(map(bool, env.goals_solved())),
                steps=int(env.steps_left()), lives=int(env.lives()),
                state=getattr(env.state, 'value', str(env.state)))


def _frame_equal(left, right):
    if left is None or right is None:
        return False
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _public_history_digest(history, frame, device):
    """Digest the same public tensors supplied by ``PolicyHistory.scores``."""
    if history is None:
        digest = hashlib.sha256()
        digest.update(b'frame')
        digest.update(np.asarray(frame).tobytes())
        return digest.hexdigest()
    padding = history.length - len(history.frames)
    frames = [history.frames[0]] * padding + history.frames
    valid = [False] * padding + [True] * len(history.frames)
    actions = [-1] * padding + history.actions
    frame_tensor = torch.as_tensor(frames, device=device)[None]
    valid_tensor = torch.tensor(valid, device=device)[None]
    action_tensor = torch.tensor(actions, device=device)[None]
    return recovery_tool.input_digest(
        (frame_tensor,),
        {'history_valid': valid_tensor, 'previous_actions': action_tensor})


class RouteDiagnostics(recovery_tool.RecoveryDiagnostics):
    """Legacy recovery metrics plus engine-observed charged blocked moves."""

    EXCLUSION_NAMES = ('terminal', 'level_transition', 'life_change', 'player_moved',
                       'transform', 'goal_change', 'budget_not_decreased', 'missing_state')

    def __init__(self):
        super().__init__()
        self.blocked_exclusions = Counter()
        self.blocked_budget_units = 0

    def _record_blocked(self, action_index, observation, old_lives, before_state,
                        after_state):
        """Record the engine-only charged-bump classification.

        Keeping this separate from the legacy hook-backed transition lets the
        shipped numerical adapter use the same definition without pretending it
        has policy forward-hook outputs.
        """
        before_level = before_state.get('level')
        after_level = after_state.get('level')
        before_cell = before_state.get('cell')
        after_cell = after_state.get('cell')
        before_steps = before_state.get('steps')
        after_steps = after_state.get('steps')
        before_triple = before_state.get('triple')
        after_triple = after_state.get('triple')
        before_goals = before_state.get('goals')
        after_goals = after_state.get('goals')
        if action_index not in range(4):
            return
        if observation.finished:
            self.blocked_exclusions['terminal'] += 1
        elif before_level != after_level:
            self.blocked_exclusions['level_transition'] += 1
        elif old_lives != after_state.get('lives'):
            self.blocked_exclusions['life_change'] += 1
        elif before_cell is None or after_cell is None:
            self.blocked_exclusions['missing_state'] += 1
        elif before_cell != after_cell:
            self.blocked_exclusions['player_moved'] += 1
        elif before_triple != after_triple:
            self.blocked_exclusions['transform'] += 1
        elif before_goals != after_goals:
            self.blocked_exclusions['goal_change'] += 1
        elif before_steps is None or after_steps is None or after_steps >= before_steps:
            self.blocked_exclusions['budget_not_decreased'] += 1
        else:
            self.counts['blocked_no_player_motion_budget_spending'] += 1
            self.blocked_budget_units += before_steps - after_steps

    def transition(self, action_index, before_frame, observation, old_lives, new_lives,
                   goals, *, before_state, after_state):
        # The old evaluator owns the exact-frame refusal definition and policy
        # hook consistency checks.  Reuse it without changing its bound file.
        super().transition(action_index, before_frame, observation, old_lives, new_lives, goals)
        self._record_blocked(action_index, observation, old_lives, before_state, after_state)

    def report(self):
        result = super().report()
        result['blocked_no_player_motion_budget_spending'] = self.counts[
            'blocked_no_player_motion_budget_spending']
        result['blocked_no_player_motion_budget_units'] = self.blocked_budget_units
        result['blocked_no_player_motion_exclusions'] = {
            name: self.blocked_exclusions[name] for name in self.EXCLUSION_NAMES
        }
        result['blocked_definition'] = (
            'movement action; same engine player cell, level, lives, carried triple and '
            'solved-goal tuple; nonterminal; steps_left strictly decreases'
        )
        result['blocked_definition_caveat'] = (
            'a charged stationary action may be intentional engine behavior; this count '
            'is diagnostic evidence, not an error label'
        )
        return result


class ObservedEnvironment(recovery_tool.ObservedEnvironment):
    """Old passive wrapper with snapshots and the native action sequence."""

    def __init__(self, env, diagnostics, check, on_stall='repeat'):
        super().__init__(env, diagnostics, check)
        self.on_stall = on_stall
        self.native_action_sequence = []
        self.native_policy_action_indices = []

    def reset(self):
        frame = self.env.reset()
        self.frame = frame
        self.diagnostics.observe_goals(self.env.goals_solved())
        return frame

    def perform(self, action):
        self.check()
        if action not in names.ACTION_IDS:
            raise ValueError('generated native evaluation accepts movement actions only')
        before_frame = np.asarray(self.frame).copy()
        before_state = _snapshot(self.env)
        old_lives = self.env.lives()
        observation = self.env.perform(action)
        after_state = _snapshot(self.env)
        action_index = names.ACTION_IDS.index(action)
        self.native_action_sequence.append(int(action))
        self.native_policy_action_indices.append(action_index)
        # evaluate.rollout's public next-best controller may select an action
        # other than the raw policy argmax.  The legacy diagnostic is hook
        # backed and checks the action it is asked to observe; record the actual
        # public controller action while retaining the raw output hook values.
        if self.on_stall == 'next-best':
            self.diagnostics.pending = action_index
        self.diagnostics.transition(action_index, before_frame, observation, old_lives,
                                    self.env.lives(), self.env.goals_solved(),
                                    before_state=before_state, after_state=after_state)
        self.frame = observation.frame
        return observation


def rollout(policy, env, *, max_actions=300, device='cpu', optimal=None,
            on_stall='repeat', check=lambda: None):
    """Run the old strict rollout while retaining native actions and new metrics."""
    diagnostics = RouteDiagnostics()
    hooks = [policy.register_forward_pre_hook(diagnostics.pre_policy, with_kwargs=True),
             policy.planner.register_forward_hook(diagnostics.planner_output),
             policy.register_forward_hook(diagnostics.policy_output)]
    observed = ObservedEnvironment(env, diagnostics, check, on_stall=on_stall)
    try:
        run = evaluate.rollout(policy, observed, max_actions, torch.device(device), optimal,
                               on_stall=on_stall, temperature=0.)
        run['recovery'] = diagnostics.report()
        run['native_action_sequence'] = list(observed.native_action_sequence)
        run['native_policy_action_indices'] = list(observed.native_policy_action_indices)
        return run
    finally:
        for hook in hooks:
            hook.remove()


def aggregate(runs):
    """Aggregate native completion and passive diagnostics, including new count."""
    return recovery_tool.aggregate(runs)


def per_tier(runs):
    return {str(tier): aggregate([run for run in runs if run.get('difficulty') == tier])
            for tier in range(1, 8)}


def validate_specs(specs):
    """Reject official/legacy rows before constructing any generated level."""
    recovery_tool.validate_specs(specs)
    if any(s.get('difficulty_version') not in (None, 'ls20-reference-v1') for s in specs):
        raise ValueError('generated rows use an unsupported difficulty contract')


def _read_specs(path):
    specs = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                specs.append(json.loads(line))
    return specs


def _loader_for_metadata(metadata):
    checkpoint_format = metadata.get('format')
    if checkpoint_format == SPATIAL_FORMAT:
        return spatial_outcome_policy.load_checkpoint
    if checkpoint_format == ROUTE_FORMAT:
        return spatial_route_outcome_policy.load_checkpoint
    if checkpoint_format == SEMANTIC_FORMAT:
        return spatial_semantic_outcome_policy.load_checkpoint
    raise ValueError('only retained, route, and semantic spatial outcome checkpoints are supported')


def load_policy(path, device='cpu'):
    """Dispatch the exact retained or route loader without reinterpreting formats."""
    metadata = torch.load(Path(path), map_location='cpu', weights_only=True)
    loader = _loader_for_metadata(metadata)
    policy, loaded = loader(Path(path).resolve(), device=device)
    if loaded.get('format') != metadata.get('format'):
        raise ValueError('checkpoint format changed during loader dispatch')
    return policy, loaded


def _source_paths(mode):
    # Bind every local module that can affect the public loop or engine.  The
    # vendored engine is a direct source dependency of Ls20Env, while helper
    # modules are included because their imported behavior is part of the
    # checkpoint protocol.
    from pebby.ls20 import env as env_module
    from pebby.agent import history, model, neural_outcome_policy
    from pebby.agent import spatial_route_outcome_planner
    from tools import evaluate_reference_spatial_outcomes as reference_tool
    paths = [Path(__file__).resolve(), Path(recovery_tool.__file__).resolve(),
             Path(evaluate.__file__).resolve(), Path(history.__file__).resolve(),
             Path(model.__file__).resolve(), Path(neural_outcome_policy.__file__).resolve(),
             Path(spatial_outcome_policy.__file__).resolve(),
             Path(spatial_route_outcome_policy.__file__).resolve(),
             Path(spatial_semantic_outcome_policy.__file__).resolve(),
             Path(spatial_route_outcome_planner.__file__).resolve(),
             Path(competition.__file__).resolve(), Path(names.__file__).resolve(),
             Path(env_module.__file__).resolve(), env_module.UPSTREAM]
    paths.extend([Path(reference_tool.__file__).resolve(), ROOT / 'tools/spatial_paired_statistics.py'])
    for module_path in sorted((ROOT / 'pebby' / 'agent').glob('*.py')):
        if module_path not in paths:
            paths.append(module_path)
    for module_path in sorted((ROOT / 'pebby' / 'ls20').glob('*.py')):
        if module_path not in paths:
            paths.append(module_path)
    if mode in ('shipped', 'shipped-isolated'):
        from pebby.ls20 import shipped
        paths.extend([Path(shipped.__file__).resolve()])
    return paths


def _check_hashes(paths, expected=None):
    result = {str(Path(path).resolve()): sha(path) for path in paths}
    if expected is not None:
        for path, digest in expected.items():
            if sha(path) != digest:
                raise ValueError(f'bound source changed: {path}')
    return result


def _checkpoint_args(parser):
    parser.add_argument('--checkpoint', action='append', nargs=3, required=True,
                        metavar=('NAME', 'PATH', 'SHA256'),
                        help='paired arm name, checkpoint path and exact SHA256')


def _new_report(args, specs, selected, bindings):
    local_start = datetime.now().astimezone()
    return dict(status='running', pid=os.getpid(), start_ticks=_proc_start_ticks(),
                started_local=local_start.isoformat(),
                deadline_local=(local_start + timedelta(seconds=args.max_seconds)).isoformat(),
                mode=args.mode, source_sha256=bindings,
                bank_sha256=args.bank_sha256.lower() if args.mode == 'generated' else None,
                bank_levels=len(specs) if args.mode == 'generated' else None,
                selected_levels=len(selected),
                selected_seeds=[s.get('seed') for s in selected] if args.mode == 'generated' else None,
                complete_bank_selected=(len(selected) == len(specs) if args.mode == 'generated' else None),
                device=args.device, training=False, official_inputs_used=args.mode in ('shipped', 'shipped-isolated'),
                official_training_inputs=False,
                oracle_calls=0, max_seconds=args.max_seconds,
                runtime=dict(precision='float32', execution='native_eager', history=8,
                             temporal_backend='auto', matmul_tf32=False, cudnn_tf32=True),
                protocol=dict(kind=('paired_isolated_generated' if args.mode == 'generated'
                                    else ('paired_isolated_shipped' if args.mode == 'shipped-isolated'
                                          else 'paired_single_sequential_shipped')),
                              max_actions=args.max_actions if args.mode == 'generated' else None,
                              on_stall=args.on_stall, temperature=0., native_lives=3,
                              public_history=8, action_caps=(
                                  '5x_vendored_human_baseline' if args.foundation_human_baseline_caps
                                  else args.per_level_max_actions if args.mode in ('shipped', 'shipped-isolated')
                                  else args.max_actions)),
                arms={}, paired={}, limits=[
                    'Diagnostics read engine state after actions; engine state is never a policy input.',
                    'Exact-frame refusal means unchanged frame with no life loss or terminal event.',
                    'Blocked charged moves require unchanged cell/level/lives/triple/goals and lower steps_left.',
                    'Generated runs are isolated one-level episodes; shipped mode is one continuous session.',
                    'Isolated shipped mode runs each official level in a fresh three-life scenario; '
                    'it reports per-level capability and is not a sequential score.',
                    'Scores and refusal win outputs are not calibrated probabilities.',
                    'Route readout is learned scene aggregation, not explicit multi-step search.',
                    'Generated next-best stops as stuck after all four actions are masked; '
                    'shipped next-best clears that public mask and continues to its cap.',
                    'Persistent game memory and learned voluntary reset are absent.',
                ])


def _validate_common(args, parser):
    if not args.checkpoint:
        parser.error('at least one checkpoint arm is required')
    if len({arm[0] for arm in args.checkpoint}) != len(args.checkpoint):
        parser.error('checkpoint arm names must be unique')
    if args.max_seconds < 1 or args.max_actions < 1:
        parser.error('positive action and deadline bounds required')
    if args.report_out.exists():
        raise FileExistsError(args.report_out)
    if args.mode == 'generated':
        if args.bank is None or args.bank_sha256 is None:
            parser.error('generated mode requires --bank and --bank-sha256')
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
    else:
        if args.limit is not None or args.bank is not None or args.bank_sha256 is not None:
            parser.error('--bank/--limit are only valid in generated mode')
        if args.foundation_human_baseline_caps == (args.per_level_max_actions is not None):
            parser.error('shipped mode requires exactly one cap basis')
        if args.per_level_max_actions is not None and args.per_level_max_actions < 1:
            parser.error('--per-level-max-actions must be positive')
    for _, path, digest in args.checkpoint:
        if not re.fullmatch(r'[0-9a-fA-F]{64}', digest):
            parser.error('checkpoint SHA256 must be exactly 64 hexadecimal characters')
        if not Path(path).is_file():
            parser.error(f'checkpoint not found: {path}')
    if args.mode == 'generated':
        if not re.fullmatch(r'[0-9a-fA-F]{64}', args.bank_sha256):
            parser.error('--bank-sha256 must be exactly 64 hexadecimal characters')
        if not args.bank.is_file():
            parser.error(f'bank not found: {args.bank}')


def _paired_generated(reference, candidate):
    comparison = recovery_tool.paired(reference, candidate)
    comparison['statistics'] = paired_win_statistics(reference, candidate, identifier='seed')
    comparison['per_tier'] = {
        str(tier): _paired_generated_tier(reference, candidate, tier)
        for tier in range(1, 8)
    }
    return comparison


def _paired_generated_tier(reference, candidate, tier):
    left = [run for run in reference if run.get('difficulty') == tier]
    right = [run for run in candidate if run.get('difficulty') == tier]
    comparison = recovery_tool.paired(left, right)
    comparison['statistics'] = paired_win_statistics(left, right, identifier='seed')
    return comparison


class CompetitionTraceDecision:
    """Numerical shipped adapter with optional public-frame-only next-best."""

    def __init__(self, policy, device, on_stall, session, check, diagnostics):
        self.base = competition.FourMovementDecision(policy, device)
        self.device = device
        self.on_stall = on_stall
        self.session = session
        self.check = check
        self.diagnostics = diagnostics
        self.blocked = set()
        self.pending = None
        self.native_action_sequence = []
        self.native_policy_action_indices = []
        self.metadata = dict(self.base.metadata)
        self.metadata['stall_controller'] = ('exact_frame_next_best' if on_stall == 'next-best' else 'none')

    def start(self, context):
        self.base.start(context)
        self.blocked.clear()
        self.pending = None
        self.diagnostics.observe_level_goals(self.session._env.level_index,
                                             self.session._env.goals_solved())

    def _scores(self, context):
        with torch.inference_mode():
            if self.base.history is not None:
                scores = self.base.history.scores()
            else:
                from pebby.agent.model import frames_to_tensor
                scores = self.base.policy(frames_to_tensor(context.frame, self.device))[0].float()
        if tuple(scores.shape) != (4,) or not bool(torch.isfinite(scores).all()):
            raise ValueError('existing checkpoint must produce four finite movement logits')
        return scores

    def decide(self, context):
        self.check()
        before = _snapshot(self.session._env)
        before_frame = np.asarray(context.frame).copy()
        state_name = getattr(context.state, 'value', context.state)
        if state_name == 'GAME_OVER':
            action, index = 0, None
        else:
            fingerprint = _public_history_digest(self.base.history, context.frame, self.device)
            scores = self._scores(context)
            if self.blocked:
                mask = torch.tensor([index in self.blocked for index in range(4)],
                                    device=scores.device)
                if bool(mask.all()):
                    self.blocked.clear()
                else:
                    scores = scores.masked_fill(mask, float('-inf'))
            index = int(scores.argmax())
            action = index + 1
        self.pending = dict(before=before, before_frame=before_frame, action=action, index=index,
                            fingerprint=(fingerprint if state_name != 'GAME_OVER' else
                                         _public_history_digest(self.base.history, context.frame,
                                                                 self.device)))
        self.native_action_sequence.append(action)
        self.native_policy_action_indices.append(index)
        return action

    def observe(self, observation, event):
        if self.pending is None:
            raise ValueError('competition observation has no pending action')
        after = _snapshot(self.session._env)
        fake = SimpleNamespace(frame=observation.frame, finished=observation.finished, won=observation.won)
        self.diagnostics.transition(self.pending['index'], self.pending['before_frame'], fake,
                                    self.pending['before']['lives'], after['lives'],
                                    after['goals'], before_state=self.pending['before'],
                                    after_state=after, fingerprint=self.pending['fingerprint'])
        unchanged = _frame_equal(self.pending['before_frame'], observation.frame)
        boundary = (event['reset'] or event['lives_after'] < event['lives_before']
                    or event['level_after'] != event['level_before'])
        if unchanged and not observation.finished and not boundary and self.on_stall == 'next-best':
            if self.pending['index'] is not None:
                self.blocked.add(self.pending['index'])
        elif not unchanged:
            self.blocked.clear()
        self.base.observe(observation, event)
        self.pending = None


class CompetitionDiagnostics(RouteDiagnostics):
    """RouteDiagnostics without policy hooks for the numerical shipped session."""

    def __init__(self):
        super().__init__()
        self.event_outputs_available = False
        self.goal_level = None

    def observe_level_goals(self, level, goals, won=False):
        """Keep the legacy goal census scoped to one sequential level."""
        level = int(level)
        if self.goal_level != level:
            self.goal_level = level
            self.ever_goals = []
            self.all_goals_simultaneously_observed = False
        self.observe_goals(goals, won)

    def transition(self, action_index, before_frame, observation, old_lives, new_lives,
                   goals, *, before_state, after_state, fingerprint=None):
        """Record shipped transitions without requiring neural forward hooks."""
        unchanged = _frame_equal(before_frame, observation.frame)
        refusal = unchanged and not observation.finished and new_lives == old_lives
        self.counts['lives_lost'] += max(0, old_lives - new_lives)
        # RESET is a charged competition command, but it has no four-logit
        # policy choice.  Keep it out of refusal/fixed-point/public-input
        # counters and break any preceding movement refusal run.
        if action_index not in range(4):
            self.refusal_run = 0
            self.last_fingerprint = None
            self.last_action = None
            self.last_unchanged = False
            self.observe_level_goals(after_state.get('level', 0), goals, observation.won)
            self.pending = None
            return
        self.counts['unchanged_actions'] += unchanged
        if refusal:
            self.counts['refusal_actions'] += 1
            self.counts['refusal_episodes'] += self.refusal_run == 0
            self.refusal_run += 1
            self.counts['max_consecutive_refusals'] = max(
                self.counts['max_consecutive_refusals'], self.refusal_run)
        else:
            self.counts['refusal_episodes_followed_by_change'] += (
                self.refusal_run > 0 and not unchanged)
            self.refusal_run = 0
        if fingerprint is not None:
            self.counts['repeated_public_inputs'] += fingerprint in self.seen
            self.counts['exact_fixedpoint_steps'] += bool(
                refusal and self.last_unchanged and fingerprint == self.last_fingerprint
                and action_index == self.last_action)
            self.seen.add(fingerprint)
            self.last_fingerprint = fingerprint
        else:
            self.counts['repeated_public_inputs'] += 0
        self.last_action = action_index
        self.last_unchanged = refusal
        self._record_blocked(action_index, observation, old_lives, before_state, after_state)
        self.observe_level_goals(after_state.get('level', 0), goals, observation.won)
        self.pending = None

    def report(self):
        result = super().report()
        result.update(refusal_predicted_win_ge_0_5=None,
                      refusal_predicted_no_win_lt_0_5=None,
                      refusal_win_output_sum=None, refusal_win_output_mean=None,
                      refusal_win_output_max=None,
                      event_outputs_available=False)
        return result


def _shipped_level_rows(competition_report):
    """Derive per-level sequential rows from the competition ledger and caps."""
    completed = int(competition_report['levels_completed'])
    rows = []
    ledger = competition_report['ledger']
    for index, actions in enumerate(competition_report['per_level_actions']):
        events = [event for event in ledger if event['level_before'] == index + 1]
        game_over_seen = any(event['state_after'] == 'GAME_OVER' for event in events)
        if index < completed:
            ending = 'win'
        elif actions >= competition_report['per_level_caps'][index]:
            # A prior GAME_OVER can be followed by charged RESET commands.  The
            # final cap governs this row; the transient terminal is retained in
            # game_over_seen rather than replacing the final outcome.
            ending = 'capped'
        elif game_over_seen:
            ending = 'game_over'
        elif events:
            ending = 'in_progress'
        else:
            ending = 'unplayed'
        rows.append(dict(level=index + 1, actions=actions, cap=competition_report['per_level_caps'][index],
                         completed=ending == 'win', ending=ending,
                         game_over_seen=game_over_seen,
                         lives_left=(events[-1]['lives_after'] if events else None)))
    return rows


def _paired_shipped(reference, candidate):
    left = {row['level']: row for row in reference}
    right = {row['level']: row for row in candidate}
    levels = sorted(left.keys() & right.keys())
    gains = [level for level in levels if right[level]['completed'] and not left[level]['completed']]
    losses = [level for level in levels if left[level]['completed'] and not right[level]['completed']]
    comparison = dict(paired_levels=len(levels), win_gains=gains, win_losses=losses,
                net_wins=len(gains) - len(losses),
                reference_only=sorted(left.keys() - right.keys()),
                candidate_only=sorted(right.keys() - left.keys()))
    comparison['statistics'] = paired_win_statistics(reference, candidate, identifier='level')
    return comparison


def _run_generated_arm(name, path, expected, specs, levels, optima, args, report, check):
    policy, metadata = load_policy(path, args.device)
    checkpoint_format = metadata['format']
    if metadata.get('encoder_frozen') is not True or metadata.get('official_training_inputs') is not False:
        raise ValueError('generated-only frozen encoder metadata required')
    if metadata.get('encoder_weights_sha256') is None:
        raise ValueError('checkpoint has no frozen encoder digest')
    arm = dict(status='running', checkpoint=str(Path(path).resolve()), checkpoint_sha256=expected.lower(),
               checkpoint_format=checkpoint_format, checkpoint_config=policy.config(), runs=[])
    report['arms'][name] = arm
    before_weights = weights_sha256(policy.state_dict())
    try:
        for index, (level, optimum, spec) in enumerate(zip(levels, optima, specs)):
            check()
            run = rollout(policy, Ls20Scenario(level, int(spec.get('training_context_index', 0))),
                          max_actions=args.max_actions, device=args.device, optimal=optimum,
                          on_stall=args.on_stall, check=check)
            run.update(run=index, seed=spec['seed'], difficulty=spec['difficulty'])
            arm['runs'].append(run)
            arm['summary'] = aggregate(arm['runs'])
            arm['per_tier'] = per_tier(arm['runs'])
            report['elapsed_seconds'] = time.monotonic() - report['_started_monotonic']
            write(args.report_out, report)
        if weights_sha256(policy.state_dict()) != before_weights:
            raise ValueError('model weights changed during evaluation')
        arm.update(status='complete', weights_unchanged=True)
    finally:
        del policy
        gc.collect()
        if args.device == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _run_shipped_arm(name, path, expected, args, report, check):
    policy, metadata = load_policy(path, args.device)
    if metadata.get('encoder_frozen') is not True or metadata.get('official_training_inputs') is not False:
        raise ValueError('generated-only frozen encoder metadata required')
    from pebby.ls20 import shipped
    session = competition.CompetitionSession()
    if session.level_count != shipped.LEVEL_COUNT:
        raise ValueError('expected exactly seven shipped levels')
    diagnostics = CompetitionDiagnostics()
    decision = CompetitionTraceDecision(policy, args.device, args.on_stall, session, check, diagnostics)
    caps = ([5 * baseline for baseline in shipped.HUMAN_BASELINE]
            if args.foundation_human_baseline_caps else [args.per_level_max_actions] * shipped.LEVEL_COUNT)
    before_weights = weights_sha256(policy.state_dict())
    arm = dict(status='running', checkpoint=str(Path(path).resolve()), checkpoint_sha256=expected.lower(),
               checkpoint_format=metadata['format'], checkpoint_config=policy.config())
    report['arms'][name] = arm
    write(args.report_out, report)
    try:
        result = competition.run_competition(decision, session, per_level_caps=caps)
        result['native_action_sequence'] = list(decision.native_action_sequence)
        result['native_policy_action_indices'] = list(decision.native_policy_action_indices)
        result['recovery'] = diagnostics.report()
        result['per_level'] = _shipped_level_rows(result)
        result['human_baseline_actions'] = list(shipped.HUMAN_BASELINE)
        result['checkpoint_format'] = metadata['format']
        arm.update(result=result, summary=dict(levels=result['levels_total'], completed=result['levels_completed'],
                                               actions=result['actions'], resets=result['resets'],
                                               endings={result['ending']: 1},
                                               recovery=result['recovery']),
                   native_action_sequence=result['native_action_sequence'],
                   native_policy_action_indices=result['native_policy_action_indices'])
        report['elapsed_seconds'] = time.monotonic() - report['_started_monotonic']
        write(args.report_out, report)
        if weights_sha256(policy.state_dict()) != before_weights:
            raise ValueError('model weights changed during evaluation')
        arm.update(status='complete', weights_unchanged=True)
    finally:
        del policy, decision, session
        gc.collect()
        if args.device == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _run_isolated_shipped_arm(name, path, expected, args, report, check):
    """Evaluate all seven official levels in fresh, independent scenarios.

    This is deliberately a companion to the sequential score protocol.  A
    level that is unreachable after an earlier sequential life loss is still
    measured here, while RESET commands and cross-level progress are absent by
    construction.  The per-level goal denominator comes from the real scenario
    rollout, so level 6's two goals cannot be silently flattened to seven.
    """
    policy, metadata = load_policy(path, args.device)
    if metadata.get('encoder_frozen') is not True or metadata.get('official_training_inputs') is not False:
        raise ValueError('generated-only frozen encoder metadata required')
    from pebby.ls20 import shipped
    caps = ([5 * baseline for baseline in shipped.HUMAN_BASELINE]
            if args.foundation_human_baseline_caps else [args.per_level_max_actions] * shipped.LEVEL_COUNT)
    before_weights = weights_sha256(policy.state_dict())
    arm = dict(status='running', checkpoint=str(Path(path).resolve()), checkpoint_sha256=expected.lower(),
               checkpoint_format=metadata['format'], checkpoint_config=policy.config(), per_level=[])
    report['arms'][name] = arm
    write(args.report_out, report)
    try:
        rows = []
        for index, level in enumerate(evaluate.shipped_levels()):
            check()
            optimum, reason = evaluate.level_optimum(index)
            run = evaluate.rollout(policy, Ls20Scenario(level, index), caps[index],
                                   args.device, optimum, args.on_stall, 0.)
            row = dict(level=index + 1, optimum=optimum, optimal_reason=reason, cap=caps[index],
                       human_baseline=shipped.HUMAN_BASELINE[index], **run)
            rows.append(row)
            arm['per_level'] = rows
            arm['summary'] = dict(levels=len(rows), completed=sum(r['completed'] for r in rows),
                                  goals_cleared=sum(r['goals_cleared'] for r in rows),
                                  goals_total=sum(r['goals_total'] for r in rows),
                                  actions=sum(r['actions'] for r in rows),
                                  endings=dict(Counter(r['ending'] for r in rows)),
                                  resets=0)
            report['elapsed_seconds'] = time.monotonic() - report['_started_monotonic']
            write(args.report_out, report)
        if weights_sha256(policy.state_dict()) != before_weights:
            raise ValueError('model weights changed during evaluation')
        arm.update(status='complete', weights_unchanged=True)
    finally:
        del policy
        gc.collect()
        if args.device == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('generated', 'shipped', 'shipped-isolated'), default='generated')
    _checkpoint_args(parser)
    parser.add_argument('--bank', type=Path, default=None)
    parser.add_argument('--bank-sha256')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-actions', type=int, default=300)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--on-stall', choices=('repeat', 'next-best'), default='repeat')
    caps = parser.add_mutually_exclusive_group(required=False)
    caps.add_argument('--foundation-human-baseline-caps', action='store_true')
    caps.add_argument('--per-level-max-actions', type=int)
    args = parser.parse_args(argv)
    if args.mode == 'generated' and args.bank is None:
        args.bank = ROOT / 'artifacts/reference-grounding-repair-v1/validation70.jsonl'
    _validate_common(args, parser)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_specs = [(name, Path(path), digest.lower()) for name, path, digest in args.checkpoint]
    expected_inputs = [(path, digest) for _, path, digest in checkpoint_specs]
    if args.mode == 'generated':
        expected_inputs.append((args.bank, args.bank_sha256.lower()))
    bindings = _check_hashes(_source_paths(args.mode) + [path for _, path, _ in checkpoint_specs]
                             + ([args.bank] if args.mode == 'generated' else []))
    for path, expected in expected_inputs:
        if sha(path) != expected:
            raise ValueError(f'exact hash mismatch: {path}')
    specs = selected = []
    levels = optima = None
    if args.mode == 'generated':
        specs = _read_specs(args.bank)
        validate_specs(specs)
        if args.limit is not None and args.limit > len(specs):
            parser.error('--limit exceeds bank rows')
        selected = specs[:args.limit] if args.limit is not None else specs
        levels, optima, loaded = evaluate.bank_levels(args.bank, args.limit)
        if loaded != selected:
            raise ValueError('bank changed while loading generated levels')
    else:
        from pebby.ls20 import shipped
        selected = [dict(level=index + 1) for index in range(shipped.LEVEL_COUNT)]
    report = _new_report(args, specs, selected, bindings)
    started_monotonic = time.monotonic()
    report['_started_monotonic'] = started_monotonic
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    write(args.report_out, {k: v for k, v in report.items() if k != '_started_monotonic'})
    old_alarm = signal.getsignal(signal.SIGALRM)
    policy = None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError(
            'bounded spatial route evaluation deadline reached')))
        signal.alarm(args.max_seconds)
        report['minimum_memavailable_bytes'] = guard()
        if args.device == 'cuda':
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True, timeout=10)
            foreign = [int(value) for value in result.stdout.splitlines()
                       if value.strip() and int(value) != os.getpid()]
            if foreign:
                raise RuntimeError(f'foreign CUDA compute processes active: {foreign}')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')

        def check():
            if time.monotonic() - report['_started_monotonic'] > args.max_seconds:
                raise TimeoutError('bounded spatial route evaluation deadline reached')
            available = guard()
            report['minimum_memavailable_bytes'] = min(report.get('minimum_memavailable_bytes', available), available)

        for name, path, expected in checkpoint_specs:
            check()
            if args.mode == 'generated':
                _run_generated_arm(name, path, expected, selected, levels, optima, args, report, check)
            elif args.mode == 'shipped':
                _run_shipped_arm(name, path, expected, args, report, check)
            else:
                _run_isolated_shipped_arm(name, path, expected, args, report, check)
            if len(report['arms']) > 1:
                names_seen = list(report['arms'])
                left = report['arms'][names_seen[0]]
                right = report['arms'][names_seen[-1]]
                if args.mode == 'generated':
                    report['paired'][names_seen[-1]] = _paired_generated(left['runs'], right['runs'])
                elif args.mode == 'shipped':
                    report['paired'][names_seen[-1]] = _paired_shipped(left['result']['per_level'],
                                                                          right['result']['per_level'])
                else:
                    report['paired'][names_seen[-1]] = _paired_shipped(left['per_level'],
                                                                          right['per_level'])
            write(args.report_out, {k: v for k, v in report.items() if k != '_started_monotonic'})
        if any(sha(path) != expected for path, expected in bindings.items()):
            raise ValueError('source, bank, or checkpoint changed during evaluation')
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        for arm in report['arms'].values():
            if arm.get('status') == 'running':
                arm['status'] = 'partial_failed'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        report.pop('_started_monotonic', None)
        report.update(elapsed_seconds=time.monotonic() - started_monotonic,
                      finished_local=datetime.now().astimezone().isoformat(),
                      process_cleanup='No persistent children; caller must verify exact evaluator PID exit.')
        if not math.isfinite(report['elapsed_seconds']):
            report['elapsed_seconds'] = 0.
        write(args.report_out, report)
    return report


if __name__ == '__main__':
    main()
