"""Generated-only supervision for publicly visible dynamic state in a latent.

The model receives no teacher fields at inference. These losses discourage a
predictive representation that preserves a level's static layout while dropping
its moving player, carried attributes, or budget. They are task-specific
supervision, not part of the LeWorldModel two-term objective.
"""
import torch
from torch import nn
from torch.nn import functional as F

FIELDS = ('next_player_cell', 'current_triple', 'next_triple', 'current_steps',
          'next_steps', 'current_lives', 'next_lives')
SIZES = (144, 6, 4, 4, 44, 4)


class StateGrounding(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                  nn.Linear(hidden, sum(SIZES)))

    def forward(self, latent):
        return self.head(latent).split(SIZES, dim=-1)


def labels(batch, next_state=False, device=None):
    player = torch.as_tensor(batch['next_player_cell' if next_state else 'player_cell'],
                             device=device).long().reshape(-1, 2)
    prefix = 'next' if next_state else 'current'
    triple = torch.as_tensor(batch[prefix + '_triple'], device=device).long().reshape(-1, 3)
    steps = torch.as_tensor(batch[prefix + '_steps'], device=device).long().flatten()
    lives = torch.as_tensor(batch[prefix + '_lives'], device=device).long().flatten()
    if bool(((player < 0) | (player >= 12)).any()):
        raise ValueError('grounding player cells must be inside the 12x12 grid')
    # Victory precedes exhaustion; cost-3 winning moves can leave -3 steps.
    # All negative values share the exhausted category (the public bar is empty).
    targets = (player[:, 1] * 12 + player[:, 0], *triple.unbind(-1), steps.clamp_min(-1) + 1, lives)
    for target, size in zip(targets, SIZES):
        if bool(((target < 0) | (target >= size)).any()):
            raise ValueError('grounding label outside public-state category range')
    return targets


def loss(head, latent, targets):
    logits = head(latent)
    losses = [F.cross_entropy(scores, target) for scores, target in zip(logits, targets)]
    accuracies = [(scores.argmax(-1) == target).float().mean()
                  for scores, target in zip(logits, targets)]
    return torch.stack(losses).mean(), torch.stack(accuracies)


def world_grounding_losses(model, batch, current, actual, imagined):
    missing = [key for key in ('player_cell', *FIELDS) if batch.get(key) is None]
    if missing:
        raise ValueError(f'grounded world policy requires generated teacher fields: {missing}')
    current_labels = labels(batch, device=current.device)
    next_labels = labels(batch, next_state=True, device=current.device)
    current_loss, current_accuracy = loss(model.grounding_head, current, current_labels)
    actual_loss, actual_accuracy = loss(model.grounding_head, actual.flatten(0, 1), next_labels)
    imagined_loss, imagined_accuracy = loss(model.grounding_head, imagined.flatten(0, 1), next_labels)
    diagnostics = {}
    names = ('player', 'shape', 'color', 'rotation', 'steps', 'lives')
    for prefix, accuracy in [('latent', current_accuracy), ('actual', actual_accuracy),
                              ('imagined', imagined_accuracy)]:
        diagnostics.update({f'{prefix}_{name}_accuracy': value.detach()
                            for name, value in zip(names, accuracy)})
    return (current_loss + actual_loss + imagined_loss) / 3, diagnostics
