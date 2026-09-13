"""Experimental prediction-only stop-gradient, not a LeWM replication.

Keep actual-state supervision and regularization attached to the shared encoder.
This changes the prediction gradient contract only; it is not a proven repair.
"""

from torch.nn import functional as F

from . import world_training_objectives as base
from .world_model import DEFAULT_WEIGHTS


def world_losses(model, batch, weights=None, *, loops=None, sigreg_generator=None):
    out = base.world_losses(model, batch, weights, loops=loops,
                           sigreg_generator=sigreg_generator)
    out["losses"]["prediction"] = F.mse_loss(out["predicted"], out["targets"].detach())
    merged_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    out["total"] = sum(merged_weights[name] * value for name, value in out["losses"].items())
    return out
