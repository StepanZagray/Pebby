"""Small positive-slope Platt calibration for frozen event logits."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


EVENT_NAMES = ("lost_life", "terminal", "won")
EVENT_COUNT = len(EVENT_NAMES)
FORMAT_V1 = "pebby.structured-event-platt.v1"
FORMAT = "pebby.structured-event-platt.v2"


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("initial slope must be finite and positive")
    # log(expm1(x)) is stable for the small positive values accepted here.
    return math.log(math.expm1(value)) if value < 20 else value


class PositiveSlopePlatt(nn.Module):
    """Per-event affine logit calibration with strictly positive slopes.

    The six learned values are three unconstrained softplus slope parameters
    and three intercepts.  Positive slopes preserve the event ranking for each
    event while allowing its probability scale and prior to move.
    """

    format = FORMAT

    def __init__(self, initial_slope: float = 1.0):
        super().__init__()
        raw = _inverse_softplus(float(initial_slope))
        self.raw_slope = nn.Parameter(torch.full((EVENT_COUNT,), raw, dtype=torch.float32))
        self.intercept = nn.Parameter(torch.zeros(EVENT_COUNT, dtype=torch.float32))

    def slope(self) -> torch.Tensor:
        return F.softplus(self.raw_slope)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise ValueError("event logits must be a floating tensor")
        if logits.ndim < 1 or logits.shape[-1] != EVENT_COUNT:
            raise ValueError("event logits must have final dimension 3")
        return logits * self.slope().to(logits) + self.intercept.to(logits)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self(logits).sigmoid()

    def coherent_probabilities(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return mutually exclusive loss/win/continue probabilities.

        The calibrated win logit is interpreted conditionally on no life loss.
        The terminal logit remains a separate diagnostic and is intentionally
        not forced into this three-way normalization because no terminal-loss
        labels are present in the current generated caches.
        """
        probability = self.probabilities(logits)
        loss = probability[..., 0]
        conditional_win = probability[..., 2]
        win = (1. - loss) * conditional_win
        continue_ = (1. - loss) * (1. - conditional_win)
        return {"loss": loss, "conditional_win": conditional_win,
                "win": win, "continue": continue_}


def calibration_bce(calibrator: PositiveSlopePlatt, logits: torch.Tensor,
                    labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Unweighted BCE target loss; labels are targets and never receive grads.

    ``mask`` supports the conditional win fit: it has the same shape as labels
    and marks independently available event targets.  The loss is averaged
    over selected event examples, so no class weight or validation prior enters.
    """
    if not isinstance(labels, torch.Tensor) or labels.shape != logits.shape:
        raise ValueError("event labels must match logits shape [...,3]")
    accepted = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64, torch.float16, torch.float32, torch.float64, torch.bfloat16)
    if labels.dtype not in accepted:
        raise ValueError("event labels must be binary")
    target = labels.detach().to(logits)
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("event labels must be binary")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("event logits/labels must be finite")
    if mask is None:
        selected = torch.ones_like(target, dtype=torch.bool)
    elif not isinstance(mask, torch.Tensor) or mask.shape != labels.shape or mask.dtype != torch.bool:
        raise ValueError("event mask must be boolean and match labels shape")
    else:
        selected = mask
    losses = F.binary_cross_entropy_with_logits(calibrator(logits), target, reduction="none")
    if not bool(selected.any()):
        raise ValueError("event mask selects no targets")
    return losses[selected].mean()
