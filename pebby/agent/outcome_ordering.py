"""Opt-in comparator supervision from generated labels, never inference inputs.

These losses operate on comparator scores only. Callers must freeze/detach the
dynamics predictor when using them for a comparator-only training experiment.
"""
import math

import torch
from torch.nn import functional as F

from .neural_outcome_planner import ACTION_COUNT


def _score_contract(scores: torch.Tensor) -> int:
    if (not torch.is_tensor(scores) or scores.ndim != 2
            or scores.shape[1] != ACTION_COUNT or scores.shape[0] < 1
            or not scores.is_floating_point()):
        raise ValueError('scores must be a nonempty floating [B,4] tensor')
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('scores must be finite')
    return scores.shape[0]


def _integers(target: torch.Tensor, shape: tuple[int, ...],
              scores: torch.Tensor, name: str) -> torch.Tensor:
    if (not torch.is_tensor(target) or tuple(target.shape) != shape
            or target.device != scores.device
            or target.dtype not in (torch.uint8, torch.int8, torch.int16,
                                    torch.int32, torch.int64)):
        raise ValueError(f'{name} must be an integer tensor of shape {shape} on the scores device')
    return target.long()


def _optimal_bits(scores: torch.Tensor, optimal: torch.Tensor) -> torch.Tensor:
    masks = _integers(optimal, (scores.shape[0],), scores, 'optimal')
    if bool(((masks < 0) | (masks > 15)).any()):
        raise ValueError('optimal masks must be in 0..15')
    return (masks[:, None] & (1 << torch.arange(ACTION_COUNT, device=scores.device))) != 0


def masked_optimal_set_cross_entropy(scores: torch.Tensor,
                                     optimal: torch.Tensor) -> torch.Tensor:
    """Uniform-target cross entropy over each nonzero optimal action bitmask.

    Roots have equal weight; optimal=0 contributes neither loss nor denominator.
    All-unsupervised batches return a graph-connected zero. This matches the
    existing policy CE, including uniform weighting among tied optimal actions.
    """
    _score_contract(scores)
    bits = _optimal_bits(scores, optimal)
    valid = bits.any(-1)
    selected = scores[valid].float()
    distribution = bits[valid].float() / bits[valid].sum(-1, keepdim=True).clamp_min(1)
    return -(distribution * selected.log_softmax(-1)).sum() / valid.sum().clamp_min(1)


def pairwise_safe_ordering_loss(scores: torch.Tensor, distances: torch.Tensor,
                               lost_life: torch.Tensor, optimal: torch.Tensor,
                               *, margin_scale: float = 1.0) -> torch.Tensor:
    """Rank strictly shorter safe reachable successors above longer ones.

    Inputs are scores/distances/lost_life [B,4] and optimal [B] bitmasks. Every
    ordered shorter/longer pair receives softplus(margin - score_difference),
    with margin = margin_scale * log1p(distance_difference). Set margin_scale=0
    for ordinary logistic ranking. Negative-distance and life-loss branches,
    ties, and roots with optimal=0 have no ordering supervision. A nonzero
    optimal mask must contain exactly the shortest safe reachable action set;
    inconsistent or incomplete masks raise ValueError.

    Average pairs within each root, then average roots with at least one pair.
    Thus roots with more alternatives do not receive greater aggregate weight.
    An all-invalid/tied batch returns a differentiable graph-connected zero.
    Labels describe generated training targets; none is a policy forward input.
    """
    batch = _score_contract(scores)
    if isinstance(margin_scale, bool) or not math.isfinite(margin_scale) or margin_scale < 0:
        raise ValueError('margin_scale must be finite and nonnegative')
    distance = _integers(distances, (batch, ACTION_COUNT), scores, 'distances')
    if (not torch.is_tensor(lost_life) or lost_life.shape != scores.shape
            or lost_life.device != scores.device
            or bool(((lost_life != 0) & (lost_life != 1)).any())):
        raise ValueError('lost_life must be binary [B,4] on the scores device')
    bits = _optimal_bits(scores, optimal)
    supervised = bits.any(-1)
    safe = (distance >= 0) & ~lost_life.bool()
    best = distance.masked_fill(~safe, torch.iinfo(torch.long).max).min(-1).values
    expected = safe & (distance == best[:, None])
    if bool(((bits != expected) & supervised[:, None]).any()):
        raise ValueError('incomplete/inconsistent optimal set')

    pairs = (supervised[:, None, None] & safe[:, :, None] & safe[:, None, :]
             & (distance[:, :, None] < distance[:, None, :]))
    values = scores.float()
    gap = distance.double()[:, None, :] - distance.double()[:, :, None]
    margin = gap.clamp_min(0).log1p().to(values.dtype) * margin_scale
    difference = values[:, :, None] - values[:, None, :]
    penalties = F.softplus((margin - difference).masked_fill(~pairs, 0))
    counts = pairs.sum(dim=(1, 2))
    per_root = penalties.masked_fill(~pairs, 0).sum(dim=(1, 2)) / counts.clamp_min(1)
    return per_root.sum() / (counts > 0).sum().clamp_min(1)
