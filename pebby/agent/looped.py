"""Shared-depth spatial transformer for LS20. Recurrence stays inside one frame.

The immutable encoded observation is recalled inside both residual branches;
each update has outer normalization (arXiv:2604.15259v2, Definition 5.2).
This is an LS20 adaptation, not a reproduction or a claim of useful extra depth.
No hidden state survives an action, no oracle enters inference, and every loop
uses the same blocks and action readout. Training uses full backpropagation.
"""

import torch
from torch import nn

from ..ls20 import names
from .model import (ACTION_COUNT, HUD_BOTTOM, HUD_COLUMNS, HUD_TOP, PALETTE,
                    PLAY_BOTTOM, PLAY_LEFT, PLAY_RIGHT, PLAY_TOP, _channels, _norm)

LOOPED_MODEL_FORMAT = "pebby.ls20-looped-policy.v1"


def positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class RecallBlock(nn.Module):
    """Internal recall: input changes the update, while state is the residual."""

    def __init__(self, channels, heads, expansion):
        super().__init__()
        self.recall = nn.Linear(2 * channels, channels, bias=False)
        self.attention_input = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.attention_output = nn.LayerNorm(channels)
        self.mlp_input = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * expansion), nn.GELU(),
                                 nn.Linear(channels * expansion, channels))
        self.mlp_output = nn.LayerNorm(channels)

    def forward(self, state, source):
        recalled = self.attention_input(self.recall(torch.cat((state, source), dim=-1)))
        update = self.attention(recalled, recalled, recalled, need_weights=False)[0]
        state = self.attention_output(state + update)
        recalled = self.mlp_input(self.recall(torch.cat((state, source), dim=-1)))
        return self.mlp_output(state + self.mlp(recalled))


class LoopedLs20Policy(nn.Module):
    """One frame -> four logits; ``loops`` controls shared computational depth.

    ``return_all=True`` returns [loops, batch, actions] for exit supervision and
    diagnostics. Defaults are deterministic in both train and eval modes; only
    the trainer samples depth. There is no learned or confidence-based halting.
    """

    checkpoint_format = LOOPED_MODEL_FORMAT

    def __init__(self, channels=64, blocks=2, heads=4, expansion=4, loops=4,
                 hud_channels=32, reduce_channels=8, hidden=64):
        super().__init__()
        config = dict(channels=channels, blocks=blocks, heads=heads, expansion=expansion,
                      loops=loops, hud_channels=hud_channels,
                      reduce_channels=reduce_channels, hidden=hidden)
        for name, value in config.items():
            positive_integer(name, value)
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if hud_channels < 2 or hidden < 2:
            raise ValueError("hud_channels and hidden must be at least 2")
        self.hyper = {"architecture": "looped", **config}
        self.loops = loops
        self.stem = nn.Sequential(
            nn.Conv2d(PALETTE, channels, names.CELL, stride=names.CELL, bias=False),
            _norm(channels), nn.GELU())
        self.hud = nn.Sequential(
            nn.Conv2d(PALETTE, hud_channels // 2, 3, padding=1, bias=False),
            _norm(hud_channels // 2), nn.GELU(),
            nn.Conv2d(hud_channels // 2, hud_channels, 3, stride=2, padding=1, bias=False),
            _norm(hud_channels), nn.GELU(), nn.AdaptiveAvgPool2d((1, HUD_COLUMNS)))
        self.hud_projection = nn.Linear(hud_channels, channels)
        self.row_position = nn.Parameter(torch.empty(names.GRID_ROWS, channels))
        self.column_position = nn.Parameter(torch.empty(names.GRID_COLS, channels))
        self.hud_position = nn.Parameter(torch.empty(HUD_COLUMNS, channels))
        for position in (self.row_position, self.column_position, self.hud_position):
            nn.init.normal_(position, std=.02)
        self.source_norm = nn.LayerNorm(channels)
        self.core = nn.ModuleList([RecallBlock(channels, heads, expansion) for _ in range(blocks)])
        # Preserve every cell/column in the readout: no pooled CLS bottleneck.
        self.reduce = nn.Sequential(nn.Linear(channels, reduce_channels), nn.GELU())
        tokens = names.GRID_ROWS * names.GRID_COLS + HUD_COLUMNS
        self.head = nn.Sequential(nn.Linear(tokens * reduce_channels, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden // 2), nn.GELU())
        self.action = nn.Linear(hidden // 2, ACTION_COUNT)

    def encode(self, frames):
        if frames.dim() != 3 or frames.shape[-2:] != (names.FRAME_SIZE, names.FRAME_SIZE):
            raise ValueError(f"frames must be [B, {names.FRAME_SIZE}, {names.FRAME_SIZE}], "
                             f"got {tuple(frames.shape)}")
        cells = self.stem(_channels(frames[:, PLAY_TOP:PLAY_BOTTOM, PLAY_LEFT:PLAY_RIGHT]))
        cells = cells.permute(0, 2, 3, 1)
        cells = cells + self.row_position[None, :, None, :] + self.column_position[None, None, :, :]
        hud = self.hud(_channels(frames[:, HUD_TOP:HUD_BOTTOM, :])).squeeze(2).transpose(1, 2)
        hud = self.hud_projection(hud) + self.hud_position
        return self.source_norm(torch.cat((cells.flatten(1, 2), hud), dim=1))

    def readout(self, state):
        return self.action(self.head(self.reduce(state).flatten(1)))

    def forward(self, frames, *, loops=None, return_all=False):
        depth = positive_integer("loops", self.loops if loops is None else loops)
        source = self.encode(frames)
        state = source
        exits = []
        for _ in range(depth):
            for block in self.core:
                state = block(state, source)
            if return_all:
                exits.append(self.readout(state))
        return torch.stack(exits) if return_all else self.readout(state)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def config(self):
        return {**self.hyper, "loops": self.loops}
