"""Decision diagnostics from generated labels, never inference inputs.

Optimal-boundary ordering excludes comparisons between two suboptimal actions.
Regret is measured only for safe reachable selections; failures to stay in that
set have their own counts rather than an invented finite distance penalty.
"""
import torch
from .outcome_ordering import _score_contract, _integers, _optimal_bits


@torch.no_grad()
def decision_metrics(scores, distances, lost_life, optimal):
    batch = _score_contract(scores)
    distance = _integers(distances, (batch, 4), scores, 'distances')
    bits = _optimal_bits(scores, optimal)
    if (lost_life.shape != scores.shape or lost_life.device != scores.device
            or not ((lost_life == 0) | (lost_life == 1)).all()):
        raise ValueError('lost_life must be binary [B,4] on scores device')
    supervised = bits.any(-1)
    safe = (distance >= 0) & ~lost_life.bool()
    best = distance.masked_fill(~safe, torch.iinfo(torch.long).max).min(-1).values
    if ((bits != (safe & (distance == best[:, None]))) & supervised[:, None]).any():
        raise ValueError('incomplete/inconsistent optimal set')
    chosen = scores.argmax(-1)
    rows = torch.arange(batch, device=scores.device)
    valid_chosen = supervised & safe[rows, chosen]
    boundary = bits[:, :, None] & ~bits[:, None, :] & supervised[:, None, None]
    delta = scores[:, :, None] - scores[:, None, :]
    finite_regret = (distance[rows, chosen] - best)[valid_chosen]
    return dict(roots=batch, supervised_roots=int(supervised.sum()),
                optimal_selections=int((bits[rows, chosen] & supervised).sum()),
                selected_life_loss=int((lost_life[rows, chosen].bool() & supervised).sum()),
                selected_unreachable=int(((distance[rows, chosen] < 0) & supervised).sum()),
                safe_selected_roots=int(valid_chosen.sum()),
                safe_selected_regret_sum=int(finite_regret.sum()),
                optimal_boundary_pairs=int(boundary.sum()),
                optimal_boundary_correct=int(((delta > 0) & boundary).sum()),
                optimal_boundary_ties=int(((delta == 0) & boundary).sum()))
