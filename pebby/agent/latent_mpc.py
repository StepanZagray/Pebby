"""Small inference-only latent beam search for the world policy.

The controller deliberately has a narrow contract.  It receives one public
history, obtains the learned current-policy logits once, and expands only
``predict_successors`` from the model's latent.  Imagined latents do not have
cell or raw-token readouts, so the current policy is retained as a root prior
and the value heads score the imagined branches.  The caller replans after
each real observation.

This is a diagnostic controller, not a game rule implementation.  Its fixed
weights are intentionally generic and should be calibrated on generated data
before being used for a larger comparison.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MPCConfig:
    """Bounded search and generic value-score settings."""

    horizon: int = 2
    beam_width: int = 4
    prior_only: bool = False
    prior_weight: float = 0.25
    distance_weight: float = 1.0
    unreachable_weight: float = 2.0
    terminal_weight: float = 2.0
    win_weight: float = 2.0
    step_cost: float = 0.05
    discount: float = 0.95
    terminal_threshold: float = 0.80

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) \
                or not 1 <= self.horizon <= 4:
            raise ValueError("horizon must be an integer in 1..4")
        if isinstance(self.beam_width, bool) or not isinstance(self.beam_width, int) \
                or not 1 <= self.beam_width <= 64:
            raise ValueError("beam_width must be an integer in 1..64")
        if not isinstance(self.prior_only, bool):
            raise ValueError("prior_only must be boolean")
        for name in ("prior_weight", "distance_weight", "unreachable_weight",
                     "terminal_weight", "win_weight", "step_cost"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("discount must be finite and in (0, 1]")
        if not math.isfinite(self.terminal_threshold) or not 0 < self.terminal_threshold <= 1:
            raise ValueError("terminal_threshold must be finite and in (0, 1]")


@dataclass(frozen=True)
class MPCDecision:
    """Observable result of one root decision."""

    action: int
    prior_action: int
    root_action_rank: int
    overrode_prior: bool
    logits: torch.Tensor
    action_scores: torch.Tensor
    best_score: float
    beam_nodes: int
    expanded_depth: int


@dataclass
class _Node:
    latent: torch.Tensor
    sequence: tuple
    prior_log_probability: float
    risk_score: float
    expected_distance: float
    survival_probability: float
    terminal: bool

    def score(self, config):
        return (self.risk_score
                + config.prior_weight * self.prior_log_probability
                - config.distance_weight * self.expected_distance
                - config.step_cost * len(self.sequence))


def _value_statistics(model, latent):
    """Return normalized finite distance and separate learned risk signals."""
    distance_logits, terminal_logits, won_logits = model.value(latent)
    probabilities = distance_logits.softmax(-1)
    unreachable = probabilities[..., -1]
    # The final bin is the unreachable bucket.  Exclude it from the finite
    # expectation instead of treating it as a finite distance of max_distance.
    bins = getattr(model, "bin_values", None)
    if bins is None or bins.numel() != distance_logits.shape[-1]:
        bins = torch.arange(distance_logits.shape[-1], device=latent.device,
                             dtype=distance_logits.dtype)
    bins = bins.to(device=latent.device, dtype=distance_logits.dtype)
    finite_probability = (1 - unreachable).clamp_min(torch.finfo(probabilities.dtype).eps)
    finite_distance = (probabilities[..., :-1] * bins[:-1]).sum(-1) / finite_probability
    max_distance = float(getattr(getattr(model, "cfg", None), "max_distance",
                                 max(1, distance_logits.shape[-1] - 1)))
    finite_distance = finite_distance / max_distance
    terminal = terminal_logits.sigmoid()
    won = won_logits.sigmoid()
    return finite_distance, unreachable, terminal, won


@torch.inference_mode()
def plan_from_encoding(model, encoding, config=None):
    """Plan one action from an encoding produced by the public history encoder.

    The return value's ``action_scores`` are four root-action scores and can be
    passed through the existing rollout's unchanged-frame mask.  ``prior_only``
    returns the exact logits from ``model.logits_from`` and performs no search.
    """
    config = MPCConfig() if config is None else config
    if not isinstance(config, MPCConfig):
        raise TypeError("config must be MPCConfig")
    latent = encoding.get("latent")
    if latent is None or latent.ndim != 2 or latent.shape[0] != 1:
        raise ValueError("encoding['latent'] must have shape [1, latent]")
    logits, _ = model.logits_from(encoding)
    if logits.shape != (1, 4):
        raise ValueError(f"model logits must have shape [1, 4], got {tuple(logits.shape)}")
    logits = logits[0].detach()
    prior_action = int(logits.argmax())
    prior_rank = 1 + int((logits > logits[prior_action]).sum())
    if config.prior_only:
        return MPCDecision(action=prior_action, prior_action=prior_action,
                           root_action_rank=prior_rank, overrode_prior=False,
                           logits=logits, action_scores=logits.clone(),
                           best_score=float(logits[prior_action]), beam_nodes=1,
                           expanded_depth=0)

    prior_log_probability = logits.log_softmax(-1)
    # Keep an independent beam for each possible first action.  This prevents
    # a high-prior root from consuming all four slots before every root has a
    # comparable full-horizon candidate.
    root_node = _Node(latent=latent[0], sequence=(), prior_log_probability=0.0,
                      risk_score=0.0, expected_distance=0.0,
                      survival_probability=1.0, terminal=False)
    root_beams = {}
    best_by_root = torch.full((4,), -torch.inf, dtype=logits.dtype, device=logits.device)
    expanded_depth = 0

    for depth in range(config.horizon):
        current = ([root_node] if depth == 0 else
                   [node for nodes in root_beams.values() for node in nodes])
        active = [node for node in current if not node.terminal]
        finished = [node for node in current if node.terminal]
        if not active:
            break
        parents = torch.stack([node.latent for node in active], dim=0)
        successors = model.predict_successors(parents)
        if successors.ndim != 3 or successors.shape[1:] != (4, latent.shape[-1]):
            raise ValueError("predict_successors must return [B, 4, latent]")
        flat = successors.flatten(0, 1)
        expected, unreachable, terminal, won = _value_statistics(model, flat)
        children = []
        for parent_index, parent in enumerate(active):
            for action in range(4):
                index = parent_index * 4 + action
                sequence = parent.sequence + (action,)
                first_action = sequence[0]
                # Terminal/win risk is charged once through the parent's
                # surviving probability.  The final distance and step cost
                # are charged only by the eventual leaf, never once per
                # live prefix.
                terminal_mass = parent.survival_probability * float(terminal[index])
                unreachable_mass = parent.survival_probability * float(unreachable[index])
                risk = (config.win_weight * terminal_mass * float(won[index])
                        - config.terminal_weight * terminal_mass * float(1 - won[index])
                        - config.unreachable_weight * unreachable_mass)
                prior_term = (float(prior_log_probability[first_action])
                              if depth == 0 else parent.prior_log_probability)
                survival = parent.survival_probability * max(
                    0.0, 1.0 - float(terminal[index]) - float(unreachable[index]))
                stopped = bool(terminal[index] >= config.terminal_threshold
                               or unreachable[index] >= config.terminal_threshold)
                child = _Node(
                    latent=flat[index], sequence=sequence,
                    prior_log_probability=prior_term,
                    risk_score=parent.risk_score + (config.discount ** depth) * risk,
                    expected_distance=float(expected[index]),
                    survival_probability=survival, terminal=stopped,
                )
                children.append(child)
        candidates = {action: [] for action in range(4)}
        for node in finished + children:
            candidates[node.sequence[0]].append(node)
        root_beams = {}
        for action, nodes in candidates.items():
            nodes.sort(key=lambda node: node.score(config), reverse=True)
            root_beams[action] = nodes[:config.beam_width]
        expanded_depth = depth + 1

    # Only complete-horizon leaves and genuinely stopped terminal/unreachable
    # leaves are eligible for the root decision.  A live h=1 prefix must not
    # win merely because every h=2 continuation is poor.
    for action, nodes in root_beams.items():
        eligible = [node for node in nodes
                    if len(node.sequence) == config.horizon or node.terminal]
        if eligible:
            best_by_root[action] = max(node.score(config) for node in eligible)
    finite_scores = best_by_root
    if not bool(torch.isfinite(finite_scores).any()):
        # This is defensive for malformed model adapters; normal models always
        # produce four first-action candidates.
        finite_scores = logits.clone()
    action = int(finite_scores.argmax())
    return MPCDecision(action=action, prior_action=prior_action,
                       root_action_rank=1 + int((logits > logits[action]).sum()),
                       overrode_prior=action != prior_action, logits=logits,
                       action_scores=finite_scores.detach().clone(),
                       best_score=float(finite_scores[action]),
                       beam_nodes=sum(len(nodes) for nodes in root_beams.values()),
                       expanded_depth=expanded_depth)


@torch.inference_mode()
def choose_action(model, frames, history_valid=None, previous_actions=None, config=None):
    """Encode a public history and return an inference-only ``MPCDecision``."""
    encoding = model.encode(frames, history_valid, previous_actions)
    return plan_from_encoding(model, encoding, config)
