"""Learned appearance of visible LS20 cells, independent of the game engine.

This component decodes public pixels only. It is not a dynamics model or policy.
Initial-state validation does not establish correctness on fogged, overlapped,
solved-goal or animated cells. Visibility labels are training masks, never inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F

ROLE_NAMES = ('wall', 'goal', 'cycler_shape', 'cycler_color',
              'cycler_rotation', 'launcher', 'refill', 'player')
ATTRIBUTE_SIZES = (6, 4, 4)
FORMAT = 'pebby.cell-appearance.v1'


def cell_patches(frames):
    """Return 7x7 public neighborhoods in row-major order, [B,144,7,7].

    Each neighborhood includes the 5x5 cell and one surrounding pixel. Outside
    frame padding is zero; such cells must be excluded from validated training.
    """
    if frames.ndim != 3 or tuple(frames.shape[-2:]) != (64, 64):
        raise ValueError('frames must be [B,64,64]')
    padded = F.pad(frames, (1, 1, 1, 1), value=0)
    support = padded[:, :62, 4:66]
    return support.unfold(1, 7, 5).unfold(2, 7, 5).reshape(-1, 144, 7, 7)


class CellAppearance(nn.Module):
    """Shared pixel MLP with independent role bits and goal-attribute heads."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16 * 7 * 7, 128), nn.GELU(),
                                     nn.Linear(128, 64), nn.GELU(),
                                     nn.Linear(64, len(ROLE_NAMES) + sum(ATTRIBUTE_SIZES)))

    def patch_logits(self, patches):
        if patches.ndim != 3 or tuple(patches.shape[-2:]) != (7, 7):
            raise ValueError('patches must be [N,7,7]')
        if patches.dtype.is_floating_point or patches.dtype == torch.bool:
            raise ValueError('patches must contain integer palette indices')
        pixels = F.one_hot(patches.long(), 16).flatten(1).float()
        return self.network(pixels)

    def forward(self, frames):
        patches = cell_patches(frames)
        logits = self.patch_logits(patches.flatten(0, 1)).reshape(frames.size(0), 144, -1)
        return logits.split((len(ROLE_NAMES), *ATTRIBUTE_SIZES), dim=-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
