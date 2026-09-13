"""One continuous LS20 session with the official competition reset semantics.

The game keeps its native lives. Every command, including a current-level RESET,
is charged once. There is no retry-count cap or aggregation across fresh games.
This local runner does not create an official API scorecard.

The engine's per-life movement counter and LS20 tank do not charge RESET. The
upstream scorecard does: Card.inc_reset_count increments both resets and actions
(arcprize/ARC-AGI, arc_agi/scorecard.py). Keep these separate accounting domains.
"""

from dataclasses import dataclass
import hashlib
from numbers import Integral
from types import MethodType

from arcengine import GameState

from pebby.ls20.env import Ls20Env


REPORT_FORMAT = "pebby.ls20-competition.v1"


def frame_digest(frame):
    return hashlib.sha256(bytes(cell for row in frame for cell in row)).hexdigest()


def _level_only_reset(game):
    game.level_reset()


def _forbid_full_reset(game):
    raise RuntimeError("a competition session cannot restart the full game")


@dataclass(frozen=True)
class DecisionContext:
    """Public observations only; return the exact game action ID, RESET=0."""

    frame: list
    state: GameState
    levels_completed: int
    win_levels: int
    available_actions: tuple


class CompetitionSession:
    """Own one real engine instance; optional levels are generated test fixtures."""

    def __init__(self, levels=None):
        self._env = Ls20Env(levels=levels)
        self.frame = self._env.reset()
        # Instance methods only: no global class or process environment mutation.
        self._env.game.handle_reset = MethodType(_level_only_reset, self._env.game)
        self._env.game.full_reset = MethodType(_forbid_full_reset, self._env.game)
        self.initial_frame_sha256 = frame_digest(self.frame)
        self.actions = 0
        self.resets = 0
        self.per_level_actions = [0] * self.level_count
        self.ledger = []
        self._run_started = False

    @property
    def level_count(self):
        return self._env.level_count

    @property
    def level_index(self):
        return self._env.level_index

    @property
    def levels_completed(self):
        return self._env.levels_completed

    @property
    def state(self):
        return self._env.state

    @property
    def lives(self):
        return self._env.lives()

    def context(self):
        legal = () if self.state == GameState.WIN else ((0,) if self.state == GameState.GAME_OVER
                                                       else (0, 1, 2, 3, 4))
        return DecisionContext(self.frame, self.state, self.levels_completed, self.level_count, legal)

    def step(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in range(5):
            raise ValueError("decision must be an integer game action ID 0..4")
        action = int(action)
        if action not in self.context().available_actions:
            raise ValueError("only RESET is legal after GAME_OVER; no action is legal after WIN")
        before_level, before_progress, before_lives = self.level_index, self.levels_completed, self.lives
        before_hash, before_state = frame_digest(self.frame), self.state.value
        observation = self._env.perform(action)
        if observation.frame is None:
            raise RuntimeError("accepted action produced no observation")
        if self.levels_completed < before_progress or self.level_index < before_level:
            raise RuntimeError("sequential game progress regressed")
        if action == 0 and (self.level_index != before_level or self.levels_completed != before_progress):
            raise RuntimeError("RESET changed completed progress or the current level")
        self.actions += 1
        self.resets += action == 0
        self.per_level_actions[before_level] += 1
        event = dict(action=action, charged_actions=self.actions, level_before=before_level + 1,
                     level_after=self.level_index + 1, level_actions=self.per_level_actions[before_level],
                     levels_completed_before=before_progress, levels_completed=self.levels_completed,
                     lives_before=before_lives, lives_after=self.lives, state_before=before_state,
                     state_after=self.state.value, frame_before_sha256=before_hash,
                     frame_after_sha256=frame_digest(observation.frame), reset=action == 0)
        self.ledger.append(event)
        self.frame = observation.frame
        return observation, event


class FourMovementDecision:
    """Existing four-logit policy plus declared GAME_OVER-only RESET controller.

    Voluntary RESET is not learned or available through these four logits. History
    clears at life loss, level transition, and explicit reset for compatibility;
    persistent game memory remains unimplemented.
    """

    metadata = dict(reset_controller="game_over_only", voluntary_reset_learned=False,
                    policy_action_ids=[1, 2, 3, 4], decision_action_ids=[0, 1, 2, 3, 4],
                    action_selection="strict_argmax", stall_controller="none",
                    history_boundary_semantics="clear_on_life_loss_level_transition_and_reset",
                    persistent_game_memory=False)

    def __init__(self, policy, device="cpu"):
        self.policy = policy
        self.device = device
        self.history = None

    def start(self, context):
        from .history import for_policy
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        self.history = for_policy(self.policy, context.frame, self.device)

    def decide(self, context):
        if context.state == GameState.GAME_OVER:
            return 0
        import torch
        from .model import frames_to_tensor
        with torch.inference_mode():
            scores = (self.history.scores() if self.history is not None else
                      self.policy(frames_to_tensor(context.frame, self.device))[0].float())
            if tuple(scores.shape) != (4,) or not bool(torch.isfinite(scores).all()):
                raise ValueError("existing checkpoint must produce four finite movement logits")
            return int(scores.argmax()) + 1

    def observe(self, observation, event):
        if self.history is not None:
            boundary = (event["reset"] or event["lives_after"] < event["lives_before"]
                        or event["level_after"] != event["level_before"])
            self.history.observe(observation.frame, event["action"] - 1, reset=boundary)


def run_competition(decision, session, *, per_level_caps):
    """Run a fresh session once. Caps charge every command and never restart a game."""
    if session._run_started or session.actions:
        raise ValueError("evaluation requires a fresh, previously unrun session")
    caps = list(per_level_caps)
    if len(caps) != session.level_count or any(isinstance(n, bool) or not isinstance(n, Integral)
                                              or n <= 0 for n in caps):
        raise ValueError("provide one positive integer action cap for each sequential level")
    caps = [int(n) for n in caps]
    session._run_started = True
    if hasattr(decision, "start"):
        decision.start(session.context())
    ending = "per_level_action_cap"
    while session.state != GameState.WIN:
        if session.per_level_actions[session.level_index] >= caps[session.level_index]:
            break
        action = decision.decide(session.context())
        observation, event = session.step(action)
        if hasattr(decision, "observe"):
            decision.observe(observation, event)
    if session.state == GameState.WIN:
        ending = "win"
    return dict(format=REPORT_FORMAT, protocol="single_sequential_level_reset_session",
                official_api_scorecard=False, full_game_initializations=1,
                action_accounting=dict(scorecard_reset_action_cost=1, engine_reset_budget_cost=0,
                    initial_game_initialization_charged=False,
                    source="https://github.com/arcprize/ARC-AGI/blob/main/arc_agi/scorecard.py"),
                full_game_resets_after_initialization=0, reset_count_cap=None,
                native_lives_per_level_initialization=3, actions=session.actions,
                resets=session.resets, per_level_actions=session.per_level_actions.copy(),
                per_level_caps=caps, levels_completed=session.levels_completed,
                levels_total=session.level_count, completed=session.state == GameState.WIN,
                ending=ending, final_state=session.state.value, lives_left=session.lives,
                initial_frame_sha256=session.initial_frame_sha256,
                decision=dict(getattr(decision, "metadata", {"reset_controller": "caller_supplied",
                                                            "learned_reset_claim": "unspecified"})),
                ledger=list(session.ledger))
