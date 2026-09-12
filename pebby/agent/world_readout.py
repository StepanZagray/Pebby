"""Task-specific learned global attention readout for the LS20 world policy.

This is an OPTIONAL addition (``WorldModelConfig.query_readout``, default off)
and is NOT a LeWorldModel mechanism: it is a small task-specific policy head
that leaves the weight-shared looped refinement, the online latent predictor,
SIGReg and every other head untouched. ``QueryReadout`` produces a residual
four-action correction that ``WorldPolicy.logits_from`` adds to the existing
``direct + ranker`` logits; its output layer starts at zero, so turning the flag
on (fresh or by checkpoint migration) leaves every prediction unchanged until
training moves it.

What it sees, all from the same public history the rest of the network sees
------------------------------------------------------------------------
* ``raw``: the CURRENT frame's stem tokens (12x12 cells + HUD tokens, with the
  spatial position embeddings) exactly as they enter ``assemble``, before any
  age/action embedding, temporal memory or refinement. These keep the per-cell
  attributes the refinement is free to compress away.
* ``state``: the refined tokens after the looped core (cells + HUD).
* ``weights``: the softmax of the existing ``player_head`` over the refined
  cells. Only geometry derived from it enters (expected player row/column and
  every cell's offset from it); no teacher player or goal coordinate is used.
* ``glyph``: with ``glyph_recall``, the grouped softmax of the current crop's
  glyph logits (14 values). The head works with and without it.

How it reads
------------
A context vector is built from the player-weighted refined cell, the
player-weighted raw cell, the mean raw HUD token, the expected player position
and the optional glyph probabilities. ``QUERY_COUNT`` learned queries are
conditioned on that context and, over ``QUERY_LAYERS`` cross-attention blocks,
attend to a memory of four token streams: raw cells, refined cells, raw HUD and
refined HUD. Cell entries also carry a projection of their relative-position
features, so a query that encodes "the attributes I carry" can match a cell by
dot product and read off where that cell lies relative to the player. The
concatenated query outputs and the context go through a small MLP to the
four-logit correction. No hand-coded glyph parser, equality rule, planner,
oracle, hidden state or cached external state is involved: every call is a
pure function of its arguments.

Depends on torch only.
"""

import torch
from torch import nn

ACTION_COUNT = 4
GRID_ROWS = 12
GRID_COLS = 12
CELLS = GRID_ROWS * GRID_COLS
QUERY_COUNT = 4
QUERY_LAYERS = 2
RELATIVE_FEATURES = 6
STREAMS = 4  # raw cells, refined cells, raw HUD, refined HUD


def expected_player_position(weights):
    """Player softmax ``[B, 144]`` -> expected ``(row, col)`` ``[B, 2]`` in cell units."""
    if weights.dim() != 2 or weights.size(1) != CELLS:
        raise ValueError(f"player weights must be [B, {CELLS}], got {tuple(weights.shape)}")
    index = torch.arange(CELLS, device=weights.device, dtype=weights.dtype)
    rows, cols = torch.div(index, GRID_COLS, rounding_mode="floor"), index % GRID_COLS
    return torch.stack((weights @ rows, weights @ cols), dim=-1)


def relative_position_features(weights):
    """Per-cell geometry relative to the expected player cell: ``[B, 144, 6]``.

    Features per cell: row offset, column offset (both divided by 11 so they lie
    in [-1, 1]), their absolute values, the normalised Manhattan distance and
    the cell's own player probability. Derived from the learned player softmax
    only; nothing here reads a label.
    """
    position = expected_player_position(weights)  # [B, 2]
    index = torch.arange(CELLS, device=weights.device, dtype=weights.dtype)
    rows, cols = torch.div(index, GRID_COLS, rounding_mode="floor"), index % GRID_COLS
    d_row = (rows[None] - position[:, :1]) / (GRID_ROWS - 1)
    d_col = (cols[None] - position[:, 1:]) / (GRID_COLS - 1)
    return torch.stack((d_row, d_col, d_row.abs(), d_col.abs(), (d_row.abs() + d_col.abs()) / 2, weights),
                       dim=-1)


class QueryBlock(nn.Module):
    """Queries attend to the token memory, then a residual MLP; outer norms as in the core."""

    def __init__(self, channels, heads, expansion=2):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.attention_output = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * expansion), nn.GELU(),
                                 nn.Linear(channels * expansion, channels))
        self.mlp_output = nn.LayerNorm(channels)

    def forward(self, queries, memory):
        update = self.attention(self.query_norm(queries), memory, memory, need_weights=False)[0]
        queries = self.attention_output(queries + update)
        return self.mlp_output(queries + self.mlp(queries))


class QueryReadout(nn.Module):
    """Learned queries over raw and refined current tokens -> residual ``[B, 4]`` correction.

    ``channels`` is the token width, ``heads`` the attention heads, ``hidden``
    the output MLP width and ``glyph_inputs`` 14 with ``glyph_recall`` else 0.
    The output layer is zero-initialised (``reset_output``), so a fresh head
    contributes exactly zero.
    """

    def __init__(self, channels, heads, hidden, glyph_inputs=0, queries=QUERY_COUNT, layers=QUERY_LAYERS):
        super().__init__()
        for name, value in (("channels", channels), ("heads", heads), ("hidden", hidden),
                            ("queries", queries), ("layers", layers)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if isinstance(glyph_inputs, bool) or not isinstance(glyph_inputs, int) or glyph_inputs < 0:
            raise ValueError("glyph_inputs must be a nonnegative integer")
        self.channels, self.queries, self.glyph_inputs = channels, queries, glyph_inputs
        # Memory streams: separate projections so raw and refined features are not conflated.
        self.raw_cell = nn.Linear(channels, channels)
        self.refined_cell = nn.Linear(channels, channels)
        self.raw_hud = nn.Linear(channels, channels)
        self.refined_hud = nn.Linear(channels, channels)
        self.relative = nn.Linear(RELATIVE_FEATURES, channels)
        self.stream_embedding = nn.Parameter(torch.empty(STREAMS, channels))
        nn.init.normal_(self.stream_embedding, std=.02)
        self.memory_norm = nn.LayerNorm(channels)
        # Context: player-weighted refined cell, player-weighted raw cell, mean raw HUD,
        # expected player (row, col), optional glyph probabilities.
        self.context_inputs = 3 * channels + 2 + glyph_inputs
        self.context = nn.Sequential(nn.Linear(self.context_inputs, channels), nn.GELU(),
                                     nn.Linear(channels, channels))
        self.query_embedding = nn.Parameter(torch.empty(queries, channels))
        nn.init.normal_(self.query_embedding, std=.02)
        self.query_condition = nn.Linear(channels, queries * channels)
        self.blocks = nn.ModuleList([QueryBlock(channels, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(queries * channels + channels)
        self.output = nn.Sequential(nn.Linear(queries * channels + channels, hidden), nn.GELU(),
                                    nn.Linear(hidden, ACTION_COUNT))
        self.reset_output()

    def reset_output(self):
        """Zero the final layer: the correction is exactly zero until training moves it."""
        with torch.no_grad():
            self.output[-1].weight.zero_()
            self.output[-1].bias.zero_()

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, raw, state, weights, glyph=None):
        """``raw``/``state`` ``[B, T, C]`` (T = 144 + HUD tokens), ``weights`` ``[B, 144]``, ``glyph`` ``[B, 14]`` or None."""
        if raw.dim() != 3 or raw.shape != state.shape or raw.size(1) <= CELLS or raw.size(-1) != self.channels:
            raise ValueError(f"raw and refined tokens must both be [B, >{CELLS}, {self.channels}], "
                             f"got {tuple(raw.shape)} and {tuple(state.shape)}")
        if self.glyph_inputs:
            if glyph is None or tuple(glyph.shape) != (raw.size(0), self.glyph_inputs):
                raise ValueError(f"query readout needs the current glyph probabilities [B, {self.glyph_inputs}]")
        elif glyph is not None:
            raise ValueError("glyph probabilities were given to a query readout without glyph inputs")
        dtype = state.dtype
        raw = raw.to(dtype)
        weights = weights.to(dtype)
        raw_cells, raw_hud = raw[:, :CELLS], raw[:, CELLS:]
        cells, hud = state[:, :CELLS], state[:, CELLS:]
        relative = relative_position_features(weights.float()).to(dtype)  # [B, 144, 6]
        offsets = self.relative(relative)
        stream = self.stream_embedding.to(dtype)
        memory = torch.cat((self.raw_cell(raw_cells) + offsets + stream[0],
                            self.refined_cell(cells) + offsets + stream[1],
                            self.raw_hud(raw_hud) + stream[2],
                            self.refined_hud(hud) + stream[3]), dim=1)
        memory = self.memory_norm(memory)
        position = expected_player_position(weights.float()).to(dtype) / (GRID_ROWS - 1)
        parts = [torch.einsum("bp,bpc->bc", weights, cells), torch.einsum("bp,bpc->bc", weights, raw_cells),
                 raw_hud.mean(1), position]
        if self.glyph_inputs:
            parts.append(glyph.to(dtype))
        context = self.context(torch.cat(parts, dim=-1))  # [B, C]
        queries = self.query_embedding.to(dtype)[None] + self.query_condition(context).view(
            raw.size(0), self.queries, self.channels)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.output(self.output_norm(torch.cat((queries.flatten(1), context), dim=-1)))
