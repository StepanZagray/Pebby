"""Experimental full-frame visibility classifier; public pixels are its only input.

Initial-state fitting is not a certified visibility mask or a dynamics model.
"""
import torch
from torch import nn

FORMAT = 'pebby.cell-visibility.v1'


class CellVisibility(nn.Module):
    """Predict 144 row-major support visibility logits from a 64x64 frame."""

    def __init__(self):
        super().__init__()
        self.palette = nn.Embedding(16, 4)
        self.features = nn.Sequential(
            nn.Conv2d(4, 12, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(24 * 8 * 8, 64), nn.GELU(),
            nn.Linear(64, 144),
        )

    def forward(self, frames):
        if frames.ndim != 3 or tuple(frames.shape[-2:]) != (64, 64):
            raise ValueError('frames must be [B,64,64]')
        if frames.dtype.is_floating_point or frames.dtype == torch.bool:
            raise ValueError('frames must contain integer palette indices')
        if bool(((frames < 0) | (frames > 15)).any()):
            raise ValueError('palette indices must be within 0..15')
        pixels = self.palette(frames.long()).permute(0, 3, 1, 2)
        return self.features(pixels)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
