"""Runtime decision heuristics that stop three lives and a RESET replaying one trajectory.

Nothing here is learned. A deterministic argmax policy whose history is flushed
at every life loss and RESET is fed the same frames again and emits the same
actions again, so the three lives of a level and every RESET after a GAME_OVER
replay one identical death. This controller sits between any four-logit policy
and the game, reads only public frames, and applies three rules in order:

1. Unchanged-frame mask (the rule `pebby.agent.evaluate.rollout` credits with
   its completion gain): an action whose last attempt from this exact frame
   left the frame byte-identical is masked out; the mask clears as soon as the
   frame moves.
2. Penalties, on later lives only: the action that undoes the previous move
   (up/down, left/right) loses `reversal_penalty` when that previous move
   actually changed the frame, and every earlier attempt of the same action
   from the same frame digest this life costs `repeat_penalty`, so a loop that
   revisits a state pays more on each pass.
3. Selection: strict argmax on the first life of a level. After a life loss or
   a RESET on the same level the action is sampled from a tempered softmax
   using a torch.Generator seeded from (base_seed, level_index, diversity
   index), so each life differs from the last and every run is reproducible.

Gating: the first life of every level (diversity index 0) is the bare policy's
strict argmax plus the stall mask and nothing else, so the controller can never
regress the retained strict baseline on a level that argmax already solves. The
reversal penalty, the repeat penalty and the temperature sampling all switch on
together, and only after a life loss or a RESET on that level.

The controller never emits RESET: `decide` returns a movement index 0..3 and
refuses a finished game. The caller owns RESET and terminal states.
"""

import hashlib

import torch

from ..ls20 import names
from .history import for_policy
from .model import frames_to_tensor

CONTROLLER_NAME = "diverse_lives"

# Index of the move that undoes each movement index: up<->down, left<->right.
OPPOSITE = tuple(names.ACTION_DELTAS.index((-dx, -dy)) for dx, dy in names.ACTION_DELTAS)

DEFAULTS = dict(stall_mask=True, reversal_penalty=0.5, repeat_penalty=0.5, temperature=0.5,
                base_seed=0)


def frame_digest(frame):
    """SHA-256 of the raw cell bytes; the same digest the competition ledger records."""
    return hashlib.sha256(bytes(cell for row in frame for cell in row)).hexdigest()


def derive_seed(base_seed, level_index, diversity_index):
    """A reproducible 63-bit generator seed for one (level, life-or-reset) pair."""
    text = f"{int(base_seed)}:{int(level_index)}:{int(diversity_index)}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & ((1 << 63) - 1)


class DiverseLivesController:
    """Wrap a four-logit policy with per-life stall, anti-loop and diversity rules.

    Knobs (all documented in `metadata`):

    * `stall_mask` (True): mask actions already shown to leave this exact frame
      unchanged; clears when the frame moves. Same rule as `evaluate.rollout`.
    * `reversal_penalty` (0.5): logit penalty on the move that undoes the
      previous frame-changing move. Later lives only.
    * `repeat_penalty` (0.5): logit penalty per earlier attempt of the same
      action from the same frame digest within the current life. Later lives
      only.
    * `temperature` (0.5): softmax temperature for lives after the first on a
      level; 0 makes every life argmax (with the penalties still applied on
      later lives).

    The first life of a level (diversity index 0) sees none of the penalties
    and no sampling: strict argmax under the stall mask, exactly the retained
    strict baseline where that baseline does not stall.
    * `base_seed` (0): the root of every derived sampling seed.

    Protocol: `start(frame)` once per game, `decide(frame, ...)` for a movement
    index, then `observe(frame, action, ...)` with the frame that action
    produced (`action=None` for a RESET performed by the caller).
    """

    def __init__(self, policy, device=None, *, stall_mask=DEFAULTS["stall_mask"],
                 reversal_penalty=DEFAULTS["reversal_penalty"],
                 repeat_penalty=DEFAULTS["repeat_penalty"],
                 temperature=DEFAULTS["temperature"], base_seed=DEFAULTS["base_seed"]):
        if reversal_penalty < 0 or repeat_penalty < 0 or temperature < 0:
            raise ValueError("penalties and temperature must not be negative")
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise ValueError("base_seed must be an integer")
        self.policy = policy
        self.device = device
        self.stall_mask = bool(stall_mask)
        self.reversal_penalty = float(reversal_penalty)
        self.repeat_penalty = float(repeat_penalty)
        self.temperature = float(temperature)
        self.base_seed = int(base_seed)
        self.history = None
        self.level_index = 0
        self.diversity_index = 0
        self.last_lives = None
        self.last = None  # Diagnostics for the most recent decision.
        self._generator = None
        self._frame = None
        self._digest = None
        self._blocked = set()
        self._previous = None  # (movement index, frame changed?) of the last move this life.
        self._tried = {}  # (frame digest, movement index) -> attempts this life.
        self._seen = {}  # frame digest -> visits this life.

    # -- lifecycle ------------------------------------------------------------

    def start(self, frame, *, level_index=0):
        """Begin a game on `frame`; the first life of `level_index` selects by argmax."""
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        self.history = for_policy(self.policy, frame, self.device)
        self.level_index = int(level_index)
        self.diversity_index = 0
        self._generator = None
        self.last = None
        self._begin_life(frame)

    def _begin_life(self, frame):
        self._blocked = set()
        self._previous = None
        self._tried = {}
        self._seen = {}
        self._set_frame(frame)

    def _set_frame(self, frame):
        self._frame = frame
        self._digest = frame_digest(frame)
        self._seen[self._digest] = self._seen.get(self._digest, 0) + 1

    def _new_life(self, *, new_level):
        if new_level:
            self.level_index += 1
            self.diversity_index = 0
        else:
            self.diversity_index += 1
        self._generator = None

    # -- decisions ------------------------------------------------------------

    def raw_scores(self, frame):
        """The policy's four logits for the current position, validated."""
        with torch.inference_mode():
            scores = (self.history.scores() if self.history is not None else
                      self.policy(frames_to_tensor(frame, self.device))[0].float())
        if tuple(scores.shape) != (4,) or not bool(torch.isfinite(scores).all()):
            raise ValueError("policy must produce four finite movement logits")
        return scores.detach().cpu().float()

    def penalties(self, digest):
        """Per-action logit penalties for the current life from `digest`.

        All zero on a level's first life: the penalties are gated to lives after
        a life loss or a RESET (diversity index > 0), together with sampling.
        """
        penalties = [0.0] * 4
        if self.diversity_index == 0:
            return penalties
        if self._previous is not None and self._previous[1]:
            penalties[OPPOSITE[self._previous[0]]] += self.reversal_penalty
        for index in range(4):
            penalties[index] += self.repeat_penalty * self._tried.get((digest, index), 0)
        return penalties

    def decide(self, frame, *, level_index=None, lives=None, state=None):
        """Return a movement index 0..3 for `frame`; never RESET.

        `frame` is the frame last passed to `start`/`observe`. `level_index`,
        when given, overrides the tracked level (a new value starts a new
        level's argmax life). `lives` is recorded for diagnostics only: life
        boundaries come from `observe`. A finished `state` is refused because
        the caller, not this controller, owns RESET.
        """
        if state is not None and getattr(state, "name", state) in ("WIN", "GAME_OVER"):
            raise ValueError("no movement decision for a finished game; the caller owns RESET")
        if level_index is not None and int(level_index) != self.level_index:
            self.level_index = int(level_index)
            self.diversity_index = 0
            self._generator = None
        self.last_lives = lives
        digest = self._digest if frame is self._frame else frame_digest(frame)
        raw = self.raw_scores(frame)
        penalties = self.penalties(digest)
        adjusted = raw - torch.tensor(penalties)
        blocked = sorted(self._blocked) if self.stall_mask else []
        masked = bool(blocked) and len(blocked) < len(names.ACTION_IDS)
        if masked:
            adjusted = adjusted.masked_fill(torch.tensor([i in self._blocked for i in range(4)]),
                                            float("-inf"))
        sampled = self.diversity_index > 0 and self.temperature > 0
        seed = None
        if sampled:
            if self._generator is None:
                seed = derive_seed(self.base_seed, self.level_index, self.diversity_index)
                self._generator = torch.Generator(device="cpu").manual_seed(seed)
            probabilities = (adjusted / self.temperature).softmax(0)
            index = int(torch.multinomial(probabilities, 1, generator=self._generator))
        else:
            index = int(adjusted.argmax())
        self.last = dict(raw=raw.tolist(), penalties=penalties, adjusted=adjusted.tolist(),
                         blocked=blocked, mask_applied=masked,
                         mask_exhausted=bool(blocked) and not masked, sampled=sampled,
                         level_index=self.level_index, diversity_index=self.diversity_index,
                         action_index=index)
        return index

    def observe(self, frame, action, *, life_lost=False, level_changed=False, reset=False):
        """Record the frame that `action` (movement index, or None for RESET) produced."""
        boundary = bool(life_lost or level_changed or reset)
        if action is not None and action not in range(4):
            raise ValueError("action must be a movement index 0..3 or None for RESET")
        if self.history is not None:
            self.history.observe(frame, -1 if action is None else int(action), reset=boundary)
        if action is not None:
            key = (self._digest, int(action))
            self._tried[key] = self._tried.get(key, 0) + 1
            changed = frame_digest(frame) != self._digest
            self._blocked = self._blocked | {int(action)} if not changed else set()
            self._previous = (int(action), changed)
        if boundary:
            self._new_life(new_level=bool(level_changed))
            self._begin_life(frame)
        else:
            self._set_frame(frame)

    # -- description ----------------------------------------------------------

    @property
    def metadata(self):
        return dict(controller=CONTROLLER_NAME, learned=False, runtime_heuristics=True,
                    emits_reset=False, stall_mask=self.stall_mask,
                    stall_mask_rule="mask actions whose last attempt from this exact frame left it "
                                    "byte-identical; clear when the frame changes",
                    reversal_penalty=self.reversal_penalty,
                    reversal_penalty_rule="subtract from the move that undoes the previous "
                                          "frame-changing move (up/down, left/right); later "
                                          "lives only",
                    repeat_penalty=self.repeat_penalty,
                    repeat_penalty_rule="subtract per earlier attempt of the same action from the "
                                        "same frame digest within the current life; later lives "
                                        "only",
                    penalty_gating="diversity_index_above_zero_only",
                    temperature=self.temperature, base_seed=self.base_seed,
                    first_life_selection="strict_argmax_with_stall_mask_no_penalties",
                    later_life_selection="seeded_tempered_softmax_sampling" if self.temperature > 0
                                         else "strict_argmax",
                    seed_derivation="sha256(base_seed:level_index:diversity_index)",
                    diversity_index="0 on a level's first life; +1 per life loss and per RESET on "
                                    "the same level; 0 again on a new level",
                    penalty_scope="current life only",
                    history="pebby.agent.history.for_policy when the policy declares a history, "
                            "else a single frame",
                    history_boundary_semantics="clear_on_life_loss_level_transition_and_reset",
                    persistent_game_memory=False)
