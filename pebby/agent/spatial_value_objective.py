"""Training-only value-head CE and finite-conditional ordinal supervision."""
import math

import torch
from torch.nn import functional as F


def value_targets(logits: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Map negative distances to unreachable; reachable overflow clips to 128."""
    if (not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[1:] != (4, 130)
            or not len(logits) or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('value logits must be finite floating nonempty [B,4,130]')
    if (not torch.is_tensor(distances) or distances.shape != logits.shape[:2]
            or distances.device != logits.device
            or distances.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError('distances must be integer [B,4] on the logits device')
    return torch.where(distances < 0, 129, distances.long().clamp(0, 128))


def finite_cdf_mse(logits: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Mean squared CDF error across reachable branches and thresholds 0..127.

    The finite conditional distribution is softmax(logits[..., :129]); category
    129 means unreachable and never represents a numeric distance. Unreachable
    targets have no ordinal loss or gradient. All-unreachable batches yield a
    graph-connected zero. Reachable distances above 128 explicitly clip to 128.
    """
    targets = value_targets(logits, distances)
    reachable = distances >= 0
    finite = logits.float()[..., :129][reachable].softmax(-1)
    cumulative = finite.cumsum(-1)[..., :128]
    truth = targets[reachable, None] <= torch.arange(128, device=logits.device)
    error = (cumulative - truth.to(cumulative.dtype)).square()
    return error.sum() / (reachable.sum().clamp_min(1) * 128)


def spatial_value_loss(logits: torch.Tensor, distances: torch.Tensor,
                       *, ordinal_weight: float = 0.) -> dict[str, torch.Tensor]:
    """Natural 130-class CE on every branch, plus optional conditional CDF MSE.

    No optimal-action mask is accepted or consulted: roots without a supervised
    policy action still carry valid reachable/unreachable value supervision.
    """
    if not math.isfinite(ordinal_weight) or ordinal_weight < 0:
        raise ValueError('ordinal_weight must be finite and nonnegative')
    targets = value_targets(logits, distances)
    ce = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
    ordinal = finite_cdf_mse(logits, distances)
    return dict(total=ce + ordinal_weight * ordinal, cross_entropy=ce, cdf_mse=ordinal)
