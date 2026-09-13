"""Opt-in action supervision; labels are training targets, never policy inputs."""
from typing import Literal

import torch


def optimal_action_loss(logits: torch.Tensor, optimal: torch.Tensor,
                        mode: Literal['uniform', 'set'] = 'uniform') -> torch.Tensor:
    """Mean action loss over roots whose optimal-action bitmask is nonzero.

    ``logits`` is nonempty floating [B,4]; ``optimal`` is integer [B] on the
    same device, with masks in 0..15. Scores must remain finite in FP32.
    Uniform mode is cross entropy against a uniform distribution over optimal
    actions. Set mode is minus log total probability of the optimal set: it
    imposes no preference among tied optimal actions. Neither adds regularizers.

    Arithmetic uses FP32 with autograd preserved. Undefined roots contribute
    neither loss nor denominator; an all-undefined batch returns a graph-
    connected zero. As with ordinary FP32 CE, a mathematically unrepresentable
    loss (e.g. a gap exceeding the FP32 maximum) is outside its numeric range.
    """
    if mode not in ('uniform', 'set'):
        raise ValueError("mode must be 'uniform' or 'set'")
    if (not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape[1] != 4
            or not len(logits) or not logits.is_floating_point()):
        raise ValueError('logits must be nonempty floating [B,4]')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('logits must be finite')
    if (not torch.is_tensor(optimal) or optimal.shape != (len(logits),)
            or optimal.device != logits.device or optimal.dtype not in
            (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError('optimal must be integer [B] on the logits device')
    masks = optimal.long()
    if bool(((masks < 0) | (masks > 15)).any()):
        raise ValueError('optimal masks must be in 0..15')
    scores = logits.float()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('logits must remain finite when converted to FP32')
    defined = masks != 0
    selected = scores[defined]
    bits = (masks[defined, None] & (1 << torch.arange(4, device=logits.device))) != 0
    log_probabilities = selected.log_softmax(-1)
    if mode == 'uniform':
        per_root = -log_probabilities.masked_fill(~bits, 0).sum(-1) / bits.sum(-1)
    else:
        per_root = -torch.logsumexp(log_probabilities.masked_fill(~bits, -torch.inf), -1)
        # All actions optimal means exactly no decision constraint, even when
        # roundoff in logsumexp(log_softmax(.)) would leave a tiny residual.
        per_root = torch.where(bits.all(-1), selected[:, 0] * 0, per_root)
    return per_root.sum() / defined.sum().clamp_min(1)
