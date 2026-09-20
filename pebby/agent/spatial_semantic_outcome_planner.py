"""Public learned scene evidence and an optional direct scene-attention actor.

Attention refines queries over the current scene; it does not roll out future
states. Zero output projections preserve the original spatial policy at birth.
"""
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from .spatial_outcome_planner import SpatialOutcomePlanner, SpatialOutcomePlannerConfig
from .spatial_route_outcome_planner import SpatialRouteOutcomePlanner, SpatialRouteOutcomePlannerConfig, _RouteReadout


@dataclass(frozen=True)
class SpatialSemanticOutcomePlannerConfig(SpatialRouteOutcomePlannerConfig):
    actor: bool = False
    actor_channels: int = 64
    actor_heads: int = 4
    actor_blocks: int = 3
    actor_expansion: int = 2

    def __post_init__(self):
        # Parent validation iterates asdict(self), so validate its fields using
        # a plain parent config rather than interpreting the actor boolean as a width.
        base = {field.name: getattr(self, field.name) for field in fields(SpatialOutcomePlannerConfig)}
        SpatialOutcomePlannerConfig(**base)
        if type(self.actor) is not bool:
            raise ValueError('actor must be boolean')
        for name in ('route_channels', 'route_heads', 'route_expansion', 'actor_channels',
                     'actor_heads', 'actor_blocks', 'actor_expansion'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.route_channels % self.route_heads or self.actor_channels % self.actor_heads:
            raise ValueError('attention channels must be divisible by their head count')

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        for key, expected in (('architecture', 'spatial_semantic_outcome_planner'), ('semantic_version', 1),
                              ('semantic_channels', 22), ('horizon', 1), ('spatial_blocks', 3),
                              ('dilations', [1, 2, 4]), ('route_blocks', 1)):
            if key in config and config.pop(key) != expected:
                raise ValueError(f'{key} must be {expected!r}')
        return cls(**config)


class _SemanticRouteReadout(_RouteReadout):
    def __init__(self, config):
        super().__init__(config)
        self.semantic_projection = nn.Linear(22, config.route_channels, bias=False)
        nn.init.zeros_(self.semantic_projection.weight)

    def forward(self, summary, condition, cells, raw_cells, refined_cells, semantic):
        batch = len(summary)
        query = self.query_projection(torch.cat((summary, condition), -1)).reshape(batch * 4, 1, -1)
        public = self.public_projection(torch.cat((raw_cells, refined_cells), -1))
        public = public + self.semantic_projection(semantic.to(public.dtype))
        memory = self.spatial_projection(cells.transpose(-1, -2)) + public[:, None]
        memory = self.memory_norm(memory).flatten(0, 1)
        route = query + self.attention(self.query_norm(query), memory, memory, need_weights=False)[0]
        route = route + self.ffn(self.ffn_norm(route))
        return self.output_projection(self.output_norm(route)).reshape(batch, 4, 130)


class _ActorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.actor_channels
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, config.actor_heads, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * config.actor_expansion), nn.SiLU(),
                                 nn.Linear(width * config.actor_expansion, width))

    def forward(self, query, memory):
        query = query + self.attention(self.query_norm(query), memory, memory, need_weights=False)[0]
        return query + self.ffn(self.ffn_norm(query))


class _SceneActor(nn.Module):
    """Four action queries share one 144-cell memory, without typed-outcome inputs."""
    def __init__(self, config):
        super().__init__()
        width = config.actor_channels
        condition = config.hud_width + 14 + 4
        self.query_projection = nn.Linear(config.summary + condition, width)
        self.public_projection = nn.Linear(2 * config.channels + 22, width)
        self.spatial_projection = nn.Linear(config.width, width, bias=False)
        self.memory_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([_ActorBlock(config) for _ in range(config.actor_blocks)])
        self.output_norm = nn.LayerNorm(width)
        # A shared scalar bias would cancel from every action softmax.
        self.output_projection = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, summary, condition, cells, raw_cells, refined_cells, semantic):
        query = self.query_projection(torch.cat((summary, condition), -1))
        memory = self.public_projection(torch.cat((raw_cells, refined_cells, semantic.to(raw_cells.dtype)), -1))
        memory = self.memory_norm(memory + self.spatial_projection(cells.mean(1).transpose(-1, -2)))
        for block in self.blocks:
            query = block(query, memory)
        return self.output_projection(self.output_norm(query)).squeeze(-1)


class SpatialSemanticOutcomePlanner(SpatialRouteOutcomePlanner):
    """Original spatial inputs plus learned probabilities ``semantic[B,144,22]``.

    Both arms own semantic_grid_projection and route_readout. Only actor=True
    owns actor_readout parameters. All planner parameters start trainable.
    ``return_components=True`` additionally exposes outcome_action_logits and
    actor_action_logits for passive diagnostics; the control actor is zero.
    """
    def __init__(self, config=None, **overrides):
        if isinstance(config, dict):
            config = SpatialSemanticOutcomePlannerConfig.from_dict({**config, **overrides})
        elif config is None:
            config = SpatialSemanticOutcomePlannerConfig(**overrides)
        elif overrides:
            raise ValueError('pass a config or keyword overrides, not both')
        if not isinstance(config, SpatialSemanticOutcomePlannerConfig):
            raise ValueError('expected SpatialSemanticOutcomePlannerConfig')
        # Construct shared modules in identical order for both arms, actor last.
        SpatialOutcomePlanner.__init__(self, config)
        self.route_readout = _SemanticRouteReadout(config)
        self.semantic_grid_projection = nn.Linear(22, config.width, bias=False)
        nn.init.zeros_(self.semantic_grid_projection.weight)
        self.actor_readout = _SceneActor(config) if config.actor else None

    def config(self):
        return dict(architecture='spatial_semantic_outcome_planner', semantic_version=1, semantic_channels=22,
                    horizon=1, spatial_blocks=3, dilations=[1, 2, 4], route_blocks=1, **asdict(self.cfg))

    @classmethod
    def from_parent(cls, parent: SpatialOutcomePlanner, **overrides):
        """Copy the original spatial parent, without tensor aliases or freeze flags."""
        if type(parent) is not SpatialOutcomePlanner:
            raise ValueError('warm start requires the original SpatialOutcomePlanner')
        config = SpatialSemanticOutcomePlannerConfig(**{**asdict(parent.cfg), **overrides})
        anchor = next(parent.parameters())
        model = cls(config).to(device=anchor.device, dtype=anchor.dtype)
        missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
        added = ('route_readout.', 'semantic_grid_projection.', 'actor_readout.')
        if unexpected or set(missing) != {name for name in model.state_dict() if name.startswith(added)}:
            raise ValueError('parent transfer changed an original spatial tensor contract')
        return model.train(parent.training)

    def _semantic_inputs(self, raw, state, glyph, semantic):
        for name, value in (('raw', raw), ('state', state), ('glyph', glyph)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite')
        if (not torch.is_tensor(semantic) or semantic.shape != (len(raw), 144, 22)
                or not semantic.is_floating_point() or semantic.device != raw.device):
            raise ValueError('semantic must be floating [B,144,22] on the public input device')
        if not bool(torch.isfinite(semantic).all()) or bool(((semantic < 0) | (semantic > 1)).any()):
            raise ValueError('semantic must contain finite probabilities in [0,1]')
        for values in semantic[..., 8:].split((6, 4, 4), -1):
            if not torch.allclose(values.float().sum(-1), torch.ones_like(values[..., 0]).float(), atol=1e-4, rtol=0):
                raise ValueError('semantic attribute distributions must each sum to one')

    def forward(self, raw, state, glyph, player_probabilities, semantic, *, action_rows=None, return_components=False):
        if type(return_components) is not bool:
            raise ValueError('return_components must be boolean')
        actions = self._inputs(raw, state, glyph, player_probabilities, action_rows)
        self._semantic_inputs(raw, state, glyph, semantic)
        batch, width = len(raw), self.cfg.width
        cell_features = torch.cat((raw[:, :144], state[:, :144]), -1).transpose(1, 2).reshape(batch, -1, 12, 12)
        player = player_probabilities.to(raw.dtype).reshape(batch, 1, 12, 12)
        grid = self.context_projection(torch.cat((cell_features, player), 1).contiguous(memory_format=torch.channels_last))
        residual = self.semantic_grid_projection(semantic.to(grid.dtype)).transpose(1, 2).reshape(batch, width, 12, 12)
        grid = grid + residual
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
        scene = (summary, branch_condition, cells, raw[:, :144], state[:, :144], semantic)
        value = value + self.route_readout(*scene).to(value.dtype)
        outcome_scores = self.score_outcomes(fields, value, events)
        scores = outcome_scores
        actor_scores = torch.zeros_like(scores)
        if self.actor_readout is not None:
            actor_scores = self.actor_readout(*scene).to(scores.dtype)
            scores = scores + actor_scores
        result = dict(action_logits=scores, field_logits=fields, value_logits=value, event_logits=events)
        if return_components:
            result.update(outcome_action_logits=outcome_scores, actor_action_logits=actor_scores)
        return result
