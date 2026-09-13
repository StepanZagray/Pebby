"""Fixed H1 spatial outcome network; public inputs only, no temporal rollout.

The frozen parent's learned player probabilities are an input, never a target.
Three distinct dilated residual blocks retain the board's 12x12 cell ordering.
Action/HUD/glyph conditioning enters every block. Only decoded outcome
probabilities reach the action comparator. ``neural_outcome_losses`` accepts
this module's unchanged typed output contract.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .neural_outcome_planner import VALUE_BINS, EVENT_NAMES, _outcome_contract
from .world_grounding import SIZES


@dataclass(frozen=True)
class SpatialOutcomePlannerConfig:
    channels: int = 64
    width: int = 48
    hud_width: int = 32
    summary: int = 64
    comparator_hidden: int = 64

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        for key, expected in (('architecture', 'spatial_outcome_planner'), ('horizon', 1),
                              ('spatial_blocks', 3), ('dilations', [1, 2, 4])):
            if key in config and config.pop(key) != expected:
                raise ValueError(f'{key} must be {expected!r}')
        return cls(**config)


def _norm(width):
    return nn.GroupNorm(8 if width % 8 == 0 else 1, width)


class _SpatialBlock(nn.Module):
    def __init__(self, width, conditioning, dilation):
        super().__init__()
        self.norm1, self.norm2 = _norm(width), _norm(width)
        self.condition = nn.Linear(conditioning, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)

    def forward(self, grid, condition):
        # Inject after normalization; each block owns its condition projection.
        update = self.norm1(grid) + self.condition(condition)[:, :, None, None]
        update = self.conv1(F.silu(update))
        return grid + self.conv2(F.silu(self.norm2(update)))


class SpatialOutcomePlanner(nn.Module):
    """Public raw/state [B,160,C], glyph [B,14], player probabilities [B,144]."""

    def __init__(self, config=None, **overrides):
        super().__init__()
        if isinstance(config, dict):
            config = SpatialOutcomePlannerConfig.from_dict({**config, **overrides})
        elif config is None:
            config = SpatialOutcomePlannerConfig(**overrides)
        elif overrides:
            raise ValueError('pass a config or keyword overrides, not both')
        if not isinstance(config, SpatialOutcomePlannerConfig):
            raise ValueError('expected SpatialOutcomePlannerConfig')
        self.cfg = config
        width, condition = config.width, config.hud_width + 14 + 4
        self.context_projection = nn.Sequential(nn.Conv2d(2 * config.channels + 1, width, 1),
                                                _norm(width), nn.SiLU())
        self.hud_projection = nn.Sequential(nn.Linear(32 * config.channels, config.hud_width),
                                            nn.LayerNorm(config.hud_width), nn.SiLU())
        self.blocks = nn.ModuleList([_SpatialBlock(width, condition, dilation) for dilation in (1, 2, 4)])
        self.output_norm = _norm(width)
        self.player_head = nn.Conv2d(width, 1, 1)
        self.summary_head = nn.Sequential(nn.Linear(2 * width + condition, config.summary),
                                          nn.LayerNorm(config.summary), nn.SiLU(),
                                          nn.Linear(config.summary, config.summary), nn.SiLU())
        self.field_heads = nn.ModuleList([nn.Linear(config.summary, size) for size in SIZES[1:]])
        self.value_head = nn.Linear(config.summary, VALUE_BINS)
        self.event_head = nn.Linear(config.summary, len(EVENT_NAMES))
        outcome_width = sum(SIZES) + VALUE_BINS + len(EVENT_NAMES)
        hidden = config.comparator_hidden
        self.outcome_projection = nn.Sequential(nn.Linear(outcome_width, hidden), nn.GELU())
        # A shared final scalar bias would cancel from every action softmax.
        self.comparator = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1, bias=False))

    def config(self):
        return dict(architecture='spatial_outcome_planner', horizon=1, spatial_blocks=3,
                    dilations=[1, 2, 4], **asdict(self.cfg))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def score_outcomes(self, field_logits, value_logits, event_logits):
        """Whole-branch equivariant comparison; no context or action-ID input."""
        _outcome_contract(field_logits, value_logits, event_logits)
        probabilities = torch.cat([*[x.float().softmax(-1) for x in field_logits],
                                   value_logits.float().softmax(-1), event_logits.float().sigmoid()], -1)
        branches = self.outcome_projection(probabilities)
        pooled = branches.mean(1, keepdim=True).expand_as(branches)
        return self.comparator(torch.cat((branches, pooled), -1)).squeeze(-1)

    def _inputs(self, raw, state, glyph, player_probabilities, action_rows):
        if not torch.is_tensor(raw) or raw.ndim != 3 or raw.shape[1:] != (160, self.cfg.channels) or raw.size(0) < 1:
            raise ValueError('raw must be nonempty [B,160,channels]')
        batch = raw.size(0)
        for name, value, shape in (('raw', raw, raw.shape), ('state', state, raw.shape),
                                   ('glyph', glyph, (batch, 14)),
                                   ('player_probabilities', player_probabilities, (batch, 144))):
            if not torch.is_tensor(value) or value.shape != shape or value.device != raw.device or not value.is_floating_point():
                raise ValueError(f'{name} must be a floating tensor of shape {tuple(shape)} on the input device')
        player = player_probabilities.float()
        if not bool(torch.isfinite(player).all()) or bool((player < 0).any()) or bool((player > 1).any()):
            raise ValueError('player_probabilities must be finite probabilities')
        if not torch.allclose(player.sum(-1), torch.ones(batch, device=player.device), atol=1e-4, rtol=0):
            raise ValueError('player_probabilities must sum to one per root')
        if action_rows is None:
            actions = torch.eye(4, device=raw.device, dtype=raw.dtype)[None].expand(batch, -1, -1)
        else:
            actions = action_rows
            if not torch.is_tensor(actions) or actions.device != raw.device or not actions.is_floating_point():
                raise ValueError('action_rows must be floating one-hot rows on the input device')
            if actions.shape == (4, 4):
                actions = actions[None].expand(batch, -1, -1)
            if actions.shape != (batch, 4, 4) or not bool(((actions == 0) | (actions == 1)).all()):
                raise ValueError('action_rows must be [4,4] or [B,4,4] one-hot permutations')
            if not bool((actions.sum(-1) == 1).all() and (actions.sum(-2) == 1).all()):
                raise ValueError('action_rows must contain every action exactly once')
        return actions

    def forward(self, raw, state, glyph, player_probabilities, *, action_rows=None):
        actions = self._inputs(raw, state, glyph, player_probabilities, action_rows)
        batch, width = len(raw), self.cfg.width
        cell_features = torch.cat((raw[:, :144], state[:, :144]), -1).transpose(1, 2).reshape(batch, -1, 12, 12)
        player = player_probabilities.to(raw.dtype).reshape(batch, 1, 12, 12)
        grid = self.context_projection(torch.cat((cell_features, player), 1).contiguous(memory_format=torch.channels_last))
        hud = self.hud_projection(torch.cat((raw[:, 144:], state[:, 144:]), -1).flatten(1))
        shared = torch.cat((hud, glyph.to(hud.dtype)), -1)[:, None].expand(-1, 4, -1)
        condition = torch.cat((shared, actions.to(shared.dtype)), -1).reshape(batch * 4, -1)
        grid = grid[:, None].expand(-1, 4, -1, -1, -1).reshape(batch * 4, width, 12, 12)
        grid = grid.contiguous(memory_format=torch.channels_last)
        for block in self.blocks:  # Three different blocks, each evaluated once.
            grid = block(grid, condition)
        grid = F.silu(self.output_norm(grid))
        player_logits = self.player_head(grid).reshape(batch, 4, 144)
        cells = grid.reshape(batch, 4, width, 144)
        next_weights = player_logits.float().softmax(-1).to(cells.dtype)
        next_context = torch.einsum('bap,bacp->bac', next_weights, cells)
        current_context = torch.einsum('bp,bacp->bac', player_probabilities.to(cells.dtype), cells)
        summary = self.summary_head(torch.cat((next_context, current_context, condition.reshape(batch, 4, -1)), -1))
        fields = (player_logits, *(head(summary) for head in self.field_heads))
        value, events = self.value_head(summary), self.event_head(summary)
        return dict(action_logits=self.score_outcomes(fields, value, events), field_logits=fields,
                    value_logits=value, event_logits=events)
