"""Versioned spatial outcome model with a trainable all-cell value readout.

The public interface and typed outcomes match the original spatial model. A
zero-initialized residual preserves parent predictions before training. This
adds learned scene aggregation, not simulator access or explicit search.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .spatial_outcome_planner import SpatialOutcomePlanner, SpatialOutcomePlannerConfig


@dataclass(frozen=True)
class SpatialRouteOutcomePlannerConfig(SpatialOutcomePlannerConfig):
    route_channels: int = 64
    route_heads: int = 4
    route_expansion: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.route_channels % self.route_heads:
            raise ValueError('route_channels must be divisible by route_heads')

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        for key, expected in (('architecture', 'spatial_route_outcome_planner'), ('horizon', 1),
                              ('spatial_blocks', 3), ('dilations', [1, 2, 4]), ('route_blocks', 1),
                              ('route_version', 1)):
            if key in config and config.pop(key) != expected:
                raise ValueError(f'{key} must be {expected!r}')
        return cls(**config)


class _RouteReadout(nn.Module):
    """One action-conditioned query attends all 144 public scene cells."""

    def __init__(self, config):
        super().__init__()
        width = config.route_channels
        condition = config.hud_width + 14 + 4
        self.query_projection = nn.Linear(config.summary + condition, width)
        # Splitting this projection avoids materializing four copies of the
        # raw/state concatenation; it equals a projection of [grid, raw, state].
        self.public_projection = nn.Linear(2 * config.channels, width)
        self.spatial_projection = nn.Linear(config.width, width, bias=False)
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, config.route_heads, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * config.route_expansion), nn.SiLU(),
                                 nn.Linear(width * config.route_expansion, width))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, 130)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, summary, condition, cells, raw_cells, refined_cells):
        batch = summary.shape[0]
        query = self.query_projection(torch.cat((summary, condition), -1)).reshape(batch * 4, 1, -1)
        public = self.public_projection(torch.cat((raw_cells, refined_cells), -1))
        memory = self.spatial_projection(cells.transpose(-1, -2)) + public[:, None]
        memory = self.memory_norm(memory).flatten(0, 1)
        route = query + self.attention(self.query_norm(query), memory, memory, need_weights=False)[0]
        route = route + self.ffn(self.ffn_norm(route))
        return self.output_projection(self.output_norm(route)).reshape(batch, 4, 130)


class SpatialRouteOutcomePlanner(SpatialOutcomePlanner):
    """Public raw/state [B,160,C], glyph [B,14], player probabilities [B,144].

    Added parameters are all named ``route_readout.*``. Constructor and transfer
    leave every planner parameter trainable; the trainer owns any freezing.
    """

    def __init__(self, config=None, **overrides):
        if isinstance(config, dict):
            config = SpatialRouteOutcomePlannerConfig.from_dict({**config, **overrides})
        elif config is None:
            config = SpatialRouteOutcomePlannerConfig(**overrides)
        elif overrides:
            raise ValueError('pass a config or keyword overrides, not both')
        if not isinstance(config, SpatialRouteOutcomePlannerConfig):
            raise ValueError('expected SpatialRouteOutcomePlannerConfig')
        super().__init__(config)
        self.route_readout = _RouteReadout(config)

    def config(self):
        return dict(architecture='spatial_route_outcome_planner', horizon=1, spatial_blocks=3,
                    dilations=[1, 2, 4], route_blocks=1, route_version=1, **asdict(self.cfg))

    @classmethod
    def from_parent(cls, parent: SpatialOutcomePlanner, *, route_channels: int = 64,
                    route_heads: int = 4, route_expansion: int = 2):
        """Copy a plain spatial parent's weights exactly and add a zero residual.

        Device, floating dtype, and train/eval mode follow the parent. Parameter
        freezing does not: all weights start trainable for the caller to control.
        Parent tensors are copied, not shared with the returned model.
        """
        if type(parent) is not SpatialOutcomePlanner:
            raise ValueError('warm start requires the original SpatialOutcomePlanner')
        config = SpatialRouteOutcomePlannerConfig(**asdict(parent.cfg), route_channels=route_channels,
                                                 route_heads=route_heads, route_expansion=route_expansion)
        anchor = next(parent.parameters())
        model = cls(config).to(device=anchor.device, dtype=anchor.dtype)
        missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
        expected = {name for name in model.state_dict() if name.startswith('route_readout.')}
        if set(missing) != expected or unexpected:
            raise ValueError('parent transfer changed an existing spatial parameter contract')
        return model.train(parent.training)

    def forward(self, raw, state, glyph, player_probabilities, *, action_rows=None):
        # Preserve the original operations through every original head. Keeping
        # this version local avoids changing existing checkpoint implementations.
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
        for block in self.blocks:
            grid = block(grid, condition)
        grid = F.silu(self.output_norm(grid))
        player_logits = self.player_head(grid).reshape(batch, 4, 144)
        cells = grid.reshape(batch, 4, width, 144)
        next_weights = player_logits.float().softmax(-1).to(cells.dtype)
        next_context = torch.einsum('bap,bacp->bac', next_weights, cells)
        current_context = torch.einsum('bp,bacp->bac', player_probabilities.to(cells.dtype), cells)
        branch_condition = condition.reshape(batch, 4, -1)
        summary = self.summary_head(torch.cat((next_context, current_context, branch_condition), -1))
        fields = (player_logits, *(head(summary) for head in self.field_heads))
        value, events = self.value_head(summary), self.event_head(summary)
        correction = self.route_readout(summary, branch_condition, cells, raw[:, :144], state[:, :144])
        value = value + correction.to(value.dtype)
        return dict(action_logits=self.score_outcomes(fields, value, events), field_logits=fields,
                    value_logits=value, event_logits=events)
