"""LS20 looped transformer with a LeWorldModel-style latent world model.

Self-contained: this module depends on torch only. The geometry constants
below mirror ``pebby.ls20.names`` and ``tests/test_world_model.py`` asserts
they agree, so nothing here imports the game or the older policy modules.

What is LeWorldModel (arXiv:2603.19312v2, ``jepa.py``/``module.py``) here
--------------------------------------------------------------------------
* One online encoder ``enc`` maps an observation to a per-frame vector
  ``z``; the SAME weights encode the current observation and every target
  observation. No EMA teacher, no stop-gradient: the prediction loss sends
  gradients through both the predicted and the target branch.
* A predictor ``pred(z, a)`` (AdaLN-zero action conditioning, zero-initialised
  gates, as in LeWM) predicts the latent of the next observation. Loss is the
  squared error against ``enc(next observation)``.
* SIGReg (Epps-Pulley sketch, ``SIGReg`` below) is applied to exactly the
  vectors the predictor consumes and predicts, per frame slot, with the batch
  as the population: slot 0 is the current latent, slots 1..4 are the four
  counterfactual successor latents. Nothing is pooled over spatial positions.

What is NOT LeWorldModel (task-specific additions, kept separable)
------------------------------------------------------------------
* ``value_head``: distance-bin / terminal / won prediction from a latent.
  Trained on real successor latents and on imagined ones (``predict``).
* ``player_head`` + ``move_head``: player-relative spatial readout. Move
  logits are computed by a shared 3x3 readout over the 12x12 cell grid and
  weighted by a softmax over the predicted player cell, so no weight is tied
  to an absolute board position.
* ``ranker``: an MLP over lookahead features of the imagined successors
  (expected distance, unreachable / terminal / won probabilities, a small
  latent summary, recursively aggregated to ``lookahead_depth``). The final
  action logits are ``direct + ranker``, so the learned dynamics influence
  the ranking and the policy gradient reaches predictor and encoder.
* ``state_recall`` (config flag, default off): the latent projector also
  sees (1) the current frame's HUD tokens exactly as they enter ``assemble``
  (stem features with HUD positions, before age/action embedding, temporal
  memory and refinement) and (2) the refined board cell vector averaged
  under the ``player_head`` softmax. Both bypass the per-token ``reduce``
  bottleneck for the cues the HUD and the player cell carry; they are
  appended AFTER the ``tokens * reduce`` base inputs so a checkpoint without
  recall migrates by zero-padding the new columns
  (``initialize_from_checkpoint``). Nothing from the future or from labels
  enters: both inputs are functions of the same public history.
* ``glyph_recall`` (config flag, default off): a ``glyph_model.GlyphEncoder``
  classifies the CURRENT frame's fixed 6x6 carried-glyph crop into 14 logits
  (shape 6 | colour 4 | rotation 4). Their grouped softmax is (1) projected by
  the zero-initialised, bias-free ``glyph_context`` and added to every current
  source token after ``source_norm`` and before the shared refinement, and
  (2) appended LAST to the projector inputs (after the reduced tokens and the
  optional state-recall features). No token is added, no successor glyph
  reaches the policy, and no label enters the forward pass; ``world_losses``
  adds a classification term on the current and actual-next crops only.
* ``query_readout`` (config flag, default off): a task-specific learned
  global attention readout (``world_readout.QueryReadout``, not a LeWM
  mechanism). Learned queries conditioned on the player-weighted features,
  the raw current HUD and the optional glyph probabilities attend over BOTH
  the current frame's raw stem tokens (as they enter ``assemble``, before
  age/action embedding, temporal memory and refinement) and the refined
  tokens, with geometry-derived relative-position features under the learned
  player softmax. The result is a residual four-action correction added to
  ``direct + ranker`` by ``logits_from`` for the current policy and for the
  optional actual-successor policy alike. Its output layer is zero at
  construction and after migration, so enabling it preserves every logit
  until training moves it. ``assemble`` returns the raw current tokens as
  ``"raw"`` (and ``"glyph"``) so nothing is re-encoded.

Observation contract
--------------------
``forward(frames, history_valid=None, previous_actions=None, *, loops=None)``
* ``frames``: uint8/int64 ``[B, 64, 64]`` (one current frame) or
  ``[B, H, 64, 64]`` chronological public history, last index = current.
  ``H`` may be anything from 1 to ``config.history``.
* ``history_valid``: bool ``[B, H]``; padded slots are ``False``; the current
  slot must be ``True``. Default: all valid.
* ``previous_actions``: int64 ``[B, H]``; the action 0..3 that produced that
  frame, ``-1`` for the first frame or padding. Default: all ``-1``.
* Returns exactly four logits ``[B, 4]``. No planner, no oracle, no game
  internals, and no state survives between calls: the caller owns history.

Optional ``cell_recall`` adds a frozen learned public-pixel appearance decoder
to every frame's spatial stem. Eight role probabilities and three categorical
attribute distributions feed a zero-initialized 22-to-channels projection.
These are fallible appearance features on every cell, including obscured cells;
they are not an engine state or a visibility mask. The existing temporal memory
and shared refinement process this evidence. The decoder weights are embedded
in checkpoints, excluded from optimizer groups, and require generated-data
provenance when the trainer imports them. Enabling the option preserves the
source policy until training moves the projection.

``encode`` exposes ``{"state", "latent", "cells", "raw", "glyph"}``; ``predict_successors``
rolls the predictor for recursive imagination; ``lookahead`` returns the
ranking features; ``world_losses`` builds every training term from a batch of
the NPZ contract (see ``world_train``).

Memory expectations (float32, eager, full backpropagation)
----------------------------------------------------------
Per training example the encoder runs 5 times (current + 4 successors), each
``loops * blocks`` block passes over ``144 + hud_tokens`` tokens. Rough
activation footprint per block pass is ``tokens * (8 + 2 * expansion) *
channels + heads * tokens**2`` floats (``~0.27M`` floats for the default
config), i.e. about ``5 * loops * blocks * 1.1 MB`` per example, so batch 64
at the defaults needs roughly 4.7 GiB before parameters and optimizer state
(0.7M parameters, negligible). This is an estimate, not a measurement: the
trainer prints the real CUDA peak after epoch 1.
``checkpoint_loops=True`` recomputes each loop in backward and divides that by
about ``loops``. ``activation_estimate_gib`` reports the estimate.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

WORLD_MODEL_FORMAT = "pebby.ls20-world-policy.v1"

# Geometry: mirrors pebby.ls20.names (asserted by the tests).
FRAME_SIZE = 64
CELL = 5
X_ORIGIN = 4
Y_ORIGIN = 0
GRID_ROWS = 12
GRID_COLS = 12
CELLS = GRID_ROWS * GRID_COLS
PALETTE = 16
ACTION_COUNT = 4
PLAY_TOP = Y_ORIGIN
PLAY_BOTTOM = Y_ORIGIN + CELL * GRID_ROWS
PLAY_LEFT = X_ORIGIN
PLAY_RIGHT = X_ORIGIN + CELL * GRID_COLS
HUD_TOP = 52
HUD_BOTTOM = FRAME_SIZE

REQUIRED_ARRAYS = ("frames", "history_valid", "previous_actions", "next_frames",
                   "terminal", "won", "optimal", "distances", "seeds")
from .world_grounding import FIELDS as GROUNDING_FIELDS
from .glyph_model import (GLYPH_CLASSES, GLYPH_FIELDS, GlyphEncoder, check_triples, crop_glyph,
                          glyph_probabilities)
from .world_readout import QueryReadout

OPTIONAL_ARRAYS = ("player_cell", "lost_life", "next_optimal", *GROUNDING_FIELDS)

# Boolean config flags that ``initialize_from_checkpoint`` may turn on (never off).
BOOLEAN_FLAGS = ("grounding", "state_recall", "glyph_recall", "query_readout", "cell_recall")


@dataclass(frozen=True)
class WorldModelConfig:
    """Everything needed to rebuild the network; ``config()`` returns it as a dict."""

    channels: int = 64
    blocks: int = 2
    heads: int = 4
    expansion: int = 4
    loops: int = 6
    history: int = 8
    temporal_layers: int = 1
    hud_channels: int = 32
    hud_tokens: int = 16
    latent: int = 128
    reduce: int = 4
    predictor_blocks: int = 2
    predictor_hidden: int = 256
    value_hidden: int = 128
    max_distance: int = 64
    lookahead_depth: int = 1
    summary: int = 8
    readout_hidden: int = 32
    ranker_hidden: int = 32
    sigreg_projections: int = 256
    sigreg_knots: int = 17
    grounding: bool = False
    state_recall: bool = False
    glyph_recall: bool = False
    query_readout: bool = False
    cell_recall: bool = False

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in BOOLEAN_FLAGS:
                if not isinstance(value, bool):
                    raise ValueError(f"{field.name} must be boolean")
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field.name} must be a positive integer, got {value!r}")
        if self.channels % self.heads:
            raise ValueError("channels must be divisible by heads")
        if self.hud_channels < 2:
            raise ValueError("hud_channels must be at least 2")
        if self.sigreg_knots < 3:
            raise ValueError("sigreg_knots must be at least 3")

    def as_dict(self):
        return {"architecture": "world", **asdict(self)}

    @classmethod
    def from_dict(cls, config):
        config = dict(config or {})
        architecture = config.pop("architecture", "world")
        if architecture != "world":
            raise ValueError(f"not a world-policy config: architecture {architecture!r}")
        unknown = set(config) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**config)


def _norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


def _planes(patch):
    """[n, h, w] colour indices -> [n, 16, h, w] one-hot planes."""
    return F.one_hot(patch.long(), num_classes=PALETTE).permute(0, 3, 1, 2).float()


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (LeWM ``module.py``, Epps-Pulley).

    ``forward(embeddings)`` takes ``[T, B, D]``: ``T`` frame slots, each a
    population of ``B`` examples. Every slot is projected on ``projections``
    random unit directions and the univariate Epps-Pulley statistic of each
    projection against N(0, 1) is integrated over ``knots`` nodes on [0, 3]
    with the Gaussian window; the result is scaled by ``B`` so it is the test
    statistic, then averaged over slots and directions. Zero iff the batch
    marginals are standard normal; a constant or low-variance batch scores
    far higher. ``[B, D]`` is accepted as a single slot.
    """

    def __init__(self, projections=256, knots=17):
        super().__init__()
        if projections < 1 or knots < 3:
            raise ValueError("projections must be positive and knots at least 3")
        self.projections = projections
        t = torch.linspace(0., 3., knots, dtype=torch.float32)
        dt = 3. / (knots - 1)
        weights = torch.full((knots,), 2. * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings, generator=None):
        # Keep the characteristic-function statistic in float32 under AMP.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            return self._statistic(embeddings.float(), generator)

    def _statistic(self, embeddings, generator=None):
        if embeddings.dim() == 2:
            embeddings = embeddings[None]
        if embeddings.dim() != 3:
            raise ValueError(f"SIGReg expects [T, B, D] or [B, D], got {tuple(embeddings.shape)}")
        population = embeddings.size(1)
        directions = torch.randn(embeddings.size(-1), self.projections, generator=generator,
                                 device=embeddings.device, dtype=embeddings.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
        x_t = (embeddings @ directions).unsqueeze(-1) * self.t  # [T, B, M, K]
        error = (x_t.cos().mean(1) - self.phi).square() + x_t.sin().mean(1).square()
        statistic = (error @ self.weights) * population  # [T, M]
        return statistic.mean()


class RefineBlock(nn.Module):
    """Weight-shared refinement step: input recall in both branches, outer norm.

    ``state`` is the residual stream; ``source`` is the immutable encoded
    observation that every loop recalls, so depth cannot forget the input.
    """

    def __init__(self, channels, heads, expansion):
        super().__init__()
        self.attention_recall = nn.Linear(2 * channels, channels, bias=False)
        self.attention_input = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.attention_output = nn.LayerNorm(channels)
        self.mlp_recall = nn.Linear(2 * channels, channels, bias=False)
        self.mlp_input = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * expansion), nn.GELU(),
                                 nn.Linear(channels * expansion, channels))
        self.mlp_output = nn.LayerNorm(channels)

    def forward(self, state, source):
        recalled = self.attention_input(self.attention_recall(torch.cat((state, source), dim=-1)))
        state = self.attention_output(state + self.attention(recalled, recalled, recalled,
                                                             need_weights=False)[0])
        recalled = self.mlp_input(self.mlp_recall(torch.cat((state, source), dim=-1)))
        return self.mlp_output(state + self.mlp(recalled))


class TemporalMemory(nn.Module):
    """Per-position attention of the current frame's token over its own history.

    Each of the ``P`` token positions attends only along time (``H`` slots),
    so a cell fogged now can be recalled from an earlier frame and a moving
    rail patroller leaves a per-cell trace. Padding slots are masked out.
    """

    def __init__(self, channels, heads, expansion):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.attention_output = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * expansion), nn.GELU(),
                                 nn.Linear(channels * expansion, channels))
        self.mlp_output = nn.LayerNorm(channels)

    def forward(self, query, tokens, valid):
        """query [B, P, C], tokens [B, H, P, C], valid [B, H] -> [B, P, C]."""
        batch, history, positions, channels = tokens.shape
        memory = tokens.permute(0, 2, 1, 3).reshape(batch * positions, history, channels)
        query = query.reshape(batch * positions, 1, channels)
        padding = (~valid).repeat_interleave(positions, dim=0)  # True = ignore
        memory = self.memory_norm(memory)
        update = self.attention(self.query_norm(query), memory, memory,
                                key_padding_mask=padding, need_weights=False)[0]
        state = self.attention_output(query + update)
        state = self.mlp_output(state + self.mlp(state))
        return state.reshape(batch, positions, channels)


class PredictorBlock(nn.Module):
    """AdaLN-zero residual MLP block; starts as the identity (LeWM predictor)."""

    def __init__(self, latent, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(latent, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(latent, 3 * latent))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.mlp = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, latent))

    def forward(self, latent, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        return latent + gate * self.mlp(self.norm(latent) * (1 + scale) + shift)


class LatentPredictor(nn.Module):
    """``pred(z, a)`` -> next latent. Action enters through AdaLN modulation."""

    def __init__(self, latent, hidden, blocks, actions=ACTION_COUNT):
        super().__init__()
        self.action_embedding = nn.Embedding(actions, latent)
        self.blocks = nn.ModuleList([PredictorBlock(latent, hidden) for _ in range(blocks)])

    def forward(self, latent, actions):
        condition = self.action_embedding(actions)
        for block in self.blocks:
            latent = block(latent, condition)
        return latent


def _mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class WorldPolicy(nn.Module):
    """History of frames -> four action logits, with a latent world model inside."""

    checkpoint_format = WORLD_MODEL_FORMAT

    def __init__(self, config=None, **overrides):
        super().__init__()
        if isinstance(config, dict):
            config = WorldModelConfig.from_dict({**config, **overrides})
        elif config is None:
            config = WorldModelConfig(**overrides)
        elif overrides:
            raise ValueError("pass either a config or keyword overrides, not both")
        self.cfg = config
        self.loops = config.loops
        # Set by the trainer only; recomputes each loop during backward to save memory.
        self.checkpoint_loops = False
        self.checkpoint_encoder = False
        self.encoder_chunk_size = 0
        channels = config.channels

        # --- encoder: stem, HUD, positions, temporal memory, looped core ------
        self.stem = nn.Sequential(nn.Conv2d(PALETTE, channels, CELL, stride=CELL, bias=False),
                                  _norm(channels), nn.GELU())
        if config.cell_recall:
            from .cell_appearance_dense import DenseCellAppearance
            self.cell_appearance = DenseCellAppearance()
            self.cell_context = nn.Linear(22, channels, bias=False)
            nn.init.zeros_(self.cell_context.weight)
        self.hud = nn.Sequential(
            nn.Conv2d(PALETTE, config.hud_channels // 2, 3, padding=1, bias=False),
            _norm(config.hud_channels // 2), nn.GELU(),
            nn.Conv2d(config.hud_channels // 2, config.hud_channels, 3, stride=2, padding=1, bias=False),
            _norm(config.hud_channels), nn.GELU(), nn.AdaptiveAvgPool2d((1, config.hud_tokens)))
        self.hud_projection = nn.Linear(config.hud_channels, channels)
        self.row_position = nn.Parameter(torch.empty(GRID_ROWS, channels))
        self.column_position = nn.Parameter(torch.empty(GRID_COLS, channels))
        self.hud_position = nn.Parameter(torch.empty(config.hud_tokens, channels))
        for position in (self.row_position, self.column_position, self.hud_position):
            nn.init.normal_(position, std=.02)
        # Age 0 is the current frame; the action index is shifted by one so -1 maps to "none".
        self.age_embedding = nn.Embedding(config.history, channels)
        self.action_embedding = nn.Embedding(ACTION_COUNT + 1, channels)
        nn.init.normal_(self.age_embedding.weight, std=.02)
        nn.init.normal_(self.action_embedding.weight, std=.02)
        self.temporal = nn.ModuleList([TemporalMemory(channels, config.heads, config.expansion)
                                       for _ in range(config.temporal_layers)])
        self.source_norm = nn.LayerNorm(channels)
        self.core = nn.ModuleList([RefineBlock(channels, config.heads, config.expansion)
                                   for _ in range(config.blocks)])

        # --- LeWM latent: reduce every token, flatten, project (no final norm) --
        self.tokens = CELLS + config.hud_tokens
        self.reduce = nn.Sequential(nn.Linear(channels, config.reduce), nn.GELU())
        # Base projector inputs come first; state recall appends the raw current
        # HUD tokens and the player-weighted refined cell so migration is a pad.
        self.base_inputs = self.tokens * config.reduce
        self.recall_inputs = (config.hud_tokens * channels + channels) if config.state_recall else 0
        # Glyph probabilities come last so any earlier block can be added later.
        self.glyph_inputs = GLYPH_CLASSES if config.glyph_recall else 0
        self.projector = _mlp(self.base_inputs + self.recall_inputs + self.glyph_inputs,
                              2 * config.latent, config.latent)
        if config.glyph_recall:
            self.glyph_encoder = GlyphEncoder()
            self.glyph_context = nn.Linear(GLYPH_CLASSES, channels, bias=False)
            nn.init.zeros_(self.glyph_context.weight)
        self.predictor = LatentPredictor(config.latent, config.predictor_hidden, config.predictor_blocks)
        self.sigreg = SIGReg(config.sigreg_projections, config.sigreg_knots)

        # --- task-specific heads (not part of LeWM) ---------------------------
        self.bins = config.max_distance + 1  # 0..max-1 exact, last = unreachable / lost
        self.value_head = _mlp(config.latent, config.value_hidden, self.bins + 2)
        self.register_buffer("bin_values", torch.arange(self.bins, dtype=torch.float32))
        self.latent_summary = nn.Linear(config.latent, config.summary)
        self.feature_size = 4 + config.summary
        self.player_head = nn.Linear(channels, 1)
        self.move_head = nn.Sequential(nn.Conv2d(channels, config.readout_hidden, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(config.readout_hidden, ACTION_COUNT, 1))
        self.ranker = _mlp(self.feature_size * config.lookahead_depth, config.ranker_hidden, 1)
        if config.query_readout:
            # Residual correction over raw + refined current tokens; zero output at start.
            self.query_head = QueryReadout(channels, config.heads, config.readout_hidden,
                                           glyph_inputs=self.glyph_inputs)
        if config.grounding:
            from .world_grounding import StateGrounding
            self.grounding_head = StateGrounding(config.latent, config.value_hidden)

    # ------------------------------------------------------------------ config
    def config(self):
        return self.cfg.as_dict()

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def projector_layout(self):
        """Named input blocks of the first projector layer: ``(name, offset, width)``, width 0 = absent."""
        layout, offset = [], 0
        for name, width in (("base", self.base_inputs), ("recall", self.recall_inputs), ("glyph", self.glyph_inputs)):
            layout.append((name, offset, width))
            offset += width
        return layout

    def activation_estimate_gib(self, batch_size, loops=None):
        """Rough float32 activation footprint of one training step (see module doc)."""
        cfg = self.cfg
        depth = self.loops if loops is None else loops
        per_pass = self.tokens * (8 + 2 * cfg.expansion) * cfg.channels + cfg.heads * self.tokens ** 2
        passes = 5 * (depth if not self.checkpoint_loops else 1) * cfg.blocks
        stems = 5 * (cfg.history + 1) * (PALETTE * FRAME_SIZE * FRAME_SIZE + 2 * self.tokens * cfg.channels)
        return batch_size * 4 * (passes * per_pass + stems) / 2 ** 30

    # ------------------------------------------------------------------ inputs
    def _prepare(self, frames, history_valid, previous_actions):
        frames = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
        if frames.dim() == 3:
            frames = frames[:, None]
        if frames.dim() != 4 or frames.shape[-2:] != (FRAME_SIZE, FRAME_SIZE):
            raise ValueError(f"frames must be [B, {FRAME_SIZE}, {FRAME_SIZE}] or "
                             f"[B, H, {FRAME_SIZE}, {FRAME_SIZE}], got {tuple(frames.shape)}")
        batch, history = frames.shape[:2]
        if not 1 <= history <= self.cfg.history:
            raise ValueError(f"history length {history} must be between 1 and {self.cfg.history}")
        device = self.age_embedding.weight.device
        frames = frames.to(device=device, dtype=torch.long)
        if history_valid is None:
            history_valid = torch.ones(batch, history, dtype=torch.bool, device=device)
        else:
            history_valid = torch.as_tensor(history_valid, device=device).bool()
        if previous_actions is None:
            previous_actions = torch.full((batch, history), -1, dtype=torch.long, device=device)
        else:
            previous_actions = torch.as_tensor(previous_actions, device=device).long()
        if history_valid.shape != (batch, history) or previous_actions.shape != (batch, history):
            raise ValueError("history_valid and previous_actions must be [B, H]")
        if not bool(history_valid[:, -1].all()):
            raise ValueError("the current (last) history slot must be valid")
        if bool(((previous_actions < -1) | (previous_actions >= ACTION_COUNT)).any()):
            raise ValueError("previous_actions must be in -1..3")
        return frames, history_valid, previous_actions

    # ----------------------------------------------------------------- encoder
    def frame_tokens(self, frames):
        """[n, 64, 64] -> [n, tokens, C] with spatial positions, no age/action."""
        if self.encoder_chunk_size and frames.size(0) > self.encoder_chunk_size:
            return torch.cat([self.frame_tokens(chunk)
                              for chunk in frames.split(self.encoder_chunk_size)], dim=0)
        if self.checkpoint_encoder and torch.is_grad_enabled():
            return checkpoint(self._frame_tokens, frames, use_reentrant=False)
        return self._frame_tokens(frames)

    def _frame_tokens(self, frames):
        cells = self.stem(_planes(frames[:, PLAY_TOP:PLAY_BOTTOM, PLAY_LEFT:PLAY_RIGHT]))
        cells = cells.permute(0, 2, 3, 1)
        if self.cfg.cell_recall:
            # Learned appearance evidence, including uncertain/obscured cells.
            # No engine visibility mask, role label or goal matcher enters here.
            with torch.no_grad():
                role, *attributes = self.cell_appearance(frames)
                evidence = torch.cat([role.sigmoid(), *(x.softmax(-1) for x in attributes)], -1)
            cells = cells + self.cell_context(evidence).reshape(-1, GRID_ROWS, GRID_COLS, self.cfg.channels)
        cells = (cells + self.row_position[None, :, None, :].to(cells.dtype)
                 + self.column_position[None, None, :, :].to(cells.dtype))
        hud = self.hud(_planes(frames[:, HUD_TOP:HUD_BOTTOM, :])).squeeze(2).transpose(1, 2)
        hud = self.hud_projection(hud)
        hud = hud + self.hud_position.to(hud.dtype)
        return torch.cat((cells.flatten(1, 2), hud), dim=1)

    def _refine(self, state, source, depth):
        for _ in range(depth):
            if self.checkpoint_loops and torch.is_grad_enabled():
                state = checkpoint(self._loop, state, source, use_reentrant=False)
            else:
                state = self._loop(state, source)
        return state

    def _loop(self, state, source):
        for block in self.core:
            state = block(state, source)
        return state

    def glyph_logits(self, frames):
        """``[n, 64, 64]`` frames -> ``[n, 14]`` logits of their carried glyph (``glyph_recall`` only)."""
        if not self.cfg.glyph_recall:
            raise ValueError("glyph logits need glyph_recall")
        return self.glyph_encoder(crop_glyph(frames))

    def assemble(self, tokens, history_valid, previous_actions, loops=None, *, glyph_logits=None):
        """tokens [B, H, P, C] (+valid/actions [B, H]) -> {"state", "latent", "cells", "raw", "glyph"}.

        ``glyph_logits`` ``[B, 14]`` are the CURRENT frame's glyph logits; required
        with ``glyph_recall`` and refused without it. ``"raw"`` is the current
        (last) slot of ``tokens`` untouched (before age/action embedding), and
        ``"glyph"`` its grouped probabilities or None; ``logits_from`` reads both
        for the optional ``query_readout``.
        """
        size = self.encoder_chunk_size
        if size and tokens.size(0) > size:
            glyph_chunks = (glyph_logits.split(size) if glyph_logits is not None
                            else [None] * math.ceil(tokens.size(0) / size))
            chunks = [self.assemble(t, v, a, loops, glyph_logits=g) for t, v, a, g in zip(
                tokens.split(size), history_valid.split(size), previous_actions.split(size), glyph_chunks)]
            state = torch.cat([chunk["state"] for chunk in chunks], dim=0)
            return {"state": state, "cells": state[:, :CELLS],
                    "latent": torch.cat([chunk["latent"] for chunk in chunks], dim=0),
                    "raw": torch.cat([chunk["raw"] for chunk in chunks], dim=0),
                    "glyph": (torch.cat([chunk["glyph"] for chunk in chunks], dim=0)
                              if chunks[0]["glyph"] is not None else None)}
        if self.checkpoint_encoder and torch.is_grad_enabled():
            return checkpoint(self._assemble, tokens, history_valid, previous_actions, loops, glyph_logits,
                              use_reentrant=False)
        return self._assemble(tokens, history_valid, previous_actions, loops, glyph_logits)

    def _assemble(self, tokens, history_valid, previous_actions, loops=None, glyph_logits=None):
        depth = self.loops if loops is None else loops
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("loops must be a positive integer")
        history = tokens.size(1)
        # Raw current-frame tokens: as they enter, before age/action embedding,
        # temporal memory and refinement (state recall reads the HUD part, the
        # optional query readout reads cells and HUD).
        raw = tokens[:, -1]
        raw_hud = raw[:, CELLS:].flatten(1) if self.cfg.state_recall else None
        glyph = None
        if self.cfg.glyph_recall:
            if glyph_logits is None or tuple(glyph_logits.shape) != (tokens.size(0), GLYPH_CLASSES):
                raise ValueError("glyph_recall needs the current frame's glyph logits [B, 14]")
            glyph = glyph_probabilities(glyph_logits.float()).to(tokens.dtype)
        elif glyph_logits is not None:
            raise ValueError("glyph logits were given to a network without glyph_recall")
        ages = torch.arange(history - 1, -1, -1, device=tokens.device)
        tokens = (tokens + self.age_embedding(ages)[None, :, None, :].to(tokens.dtype)
                  + self.action_embedding(previous_actions + 1)[:, :, None, :].to(tokens.dtype))
        current = tokens[:, -1]
        for layer in self.temporal:
            current = layer(current, tokens, history_valid)
        source = self.source_norm(current)
        if glyph is not None:
            # Broadcast the learned attributes into every current source token; every loop recalls it.
            source = source + self.glyph_context(glyph)[:, None, :].to(source.dtype)
        state = self._refine(source, source, depth)
        cells = state[:, :CELLS]
        latent = self.projector(self.projector_inputs(state, raw_hud, glyph))
        return {"state": state, "latent": latent, "cells": cells, "raw": raw, "glyph": glyph}

    def projector_inputs(self, state, raw_hud=None, glyph=None):
        """Flattened reduced tokens, then (with ``state_recall``) the recall features,
        then (with ``glyph_recall``) the 14 glyph probabilities."""
        reduced = self.reduce(state).flatten(1)
        parts = [reduced]
        if self.cfg.state_recall:
            if raw_hud is None:
                raise ValueError("state_recall needs the raw current HUD tokens")
            cells = state[:, :CELLS]
            weights = self.player_weights(cells)[1]
            player_cell = torch.einsum("bp,bpc->bc", weights.to(cells.dtype), cells)
            parts += [raw_hud.to(reduced.dtype), player_cell.to(reduced.dtype)]
        if self.cfg.glyph_recall:
            if glyph is None:
                raise ValueError("glyph_recall needs the current glyph probabilities")
            parts.append(glyph.to(reduced.dtype))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def player_weights(self, cells):
        """Player logits [B, 144] and their softmax from the refined cell tokens."""
        player = self.player_head(cells).squeeze(-1)
        return player, player.softmax(-1)

    def encode(self, frames, history_valid=None, previous_actions=None, *, loops=None):
        frames, history_valid, previous_actions = self._prepare(frames, history_valid, previous_actions)
        batch, history = frames.shape[:2]
        tokens = self.frame_tokens(frames.flatten(0, 1)).view(batch, history, self.tokens, -1)
        # Only the current public frame's glyph is classified; older frames are not.
        glyph_logits = self.glyph_logits(frames[:, -1]) if self.cfg.glyph_recall else None
        return self.assemble(tokens, history_valid, previous_actions, loops, glyph_logits=glyph_logits)

    # --------------------------------------------------------- world model API
    def predict_successors(self, latent, actions=None):
        """``pred(z, a)``. actions None -> all four: [B, 4, D]; [B] -> [B, D]; [B, k] -> [B, k, D]."""
        if actions is None:
            actions = torch.arange(ACTION_COUNT, device=latent.device)[None].expand(latent.size(0), -1)
        actions = torch.as_tensor(actions, device=latent.device).long()
        if actions.dim() == 1:
            return self.predictor(latent, actions)
        expanded = latent[:, None, :].expand(-1, actions.size(1), -1).reshape(-1, latent.size(-1))
        return self.predictor(expanded, actions.reshape(-1)).view(latent.size(0), actions.size(1), -1)

    def value(self, latent):
        """Task-specific value head: distance-bin logits [.., bins], terminal, won logits."""
        out = self.value_head(latent)
        return out[..., :self.bins], out[..., self.bins], out[..., self.bins + 1]

    def value_features(self, latent):
        distance_logits, terminal, won = self.value(latent)
        probabilities = distance_logits.softmax(-1)
        expected = (probabilities * self.bin_values).sum(-1, keepdim=True) / self.cfg.max_distance
        return torch.cat((expected, probabilities[..., -1:], terminal.sigmoid()[..., None],
                          won.sigmoid()[..., None], self.latent_summary(latent)), dim=-1)

    def lookahead(self, latent, depth=None, successors=None):
        """Features [B, 4, F * depth] of imagined successors, and the successors [B, 4, D]."""
        depth = self.cfg.lookahead_depth if depth is None else depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("lookahead depth must be a positive integer")
        if successors is None:
            successors = self.predict_successors(latent)
        flat = successors.flatten(0, 1)
        features = self.value_features(flat)
        if depth > 1:
            children, _ = self.lookahead(flat, depth - 1)  # [B*4, 4, F*(depth-1)]
            # Soft-min over the child's expected distance: the best continuation dominates.
            weights = torch.softmax(-children[..., 0] * self.cfg.max_distance / 4., dim=1)
            continuation = (weights[..., None] * children).sum(1)
            # Terminal states are absorbing: do not rank a win through arbitrary
            # descendants of a latent for which no next action exists.
            terminal = self.value(flat)[1].sigmoid()[..., None]
            absorbing = features.repeat(1, depth - 1)
            continuation = (1 - terminal) * continuation + terminal * absorbing
            features = torch.cat((features, continuation), dim=-1)
        return features.view(latent.size(0), ACTION_COUNT, -1), successors

    # ---------------------------------------------------------------- readout
    def direct_logits(self, cells):
        """Player-relative move readout over the 12x12 cell tokens."""
        grid = cells.view(-1, GRID_ROWS, GRID_COLS, cells.size(-1)).permute(0, 3, 1, 2)
        player, weights = self.player_weights(cells)  # [B, 144]
        moves = self.move_head(grid).flatten(2)  # [B, 4, 144]
        return torch.einsum("bp,bap->ba", weights, moves), player

    def query_logits(self, encoding, weights=None):
        """Residual correction ``[B, 4]`` of the optional ``query_readout`` from one encoding.

        ``encoding`` must come from ``assemble``/``encode`` (it needs ``"raw"``,
        ``"state"`` and, with ``glyph_recall``, ``"glyph"``); ``weights`` is the
        player softmax, recomputed from the refined cells when not given.
        """
        if not self.cfg.query_readout:
            raise ValueError("query logits need query_readout")
        if encoding.get("raw") is None:
            raise ValueError("query_readout needs the raw current tokens that assemble returns as 'raw'")
        if weights is None:
            weights = self.player_weights(encoding["cells"])[1]
        return self.query_head(encoding["raw"], encoding["state"], weights, encoding.get("glyph"))

    def logits_from(self, encoding, successors=None):
        direct, player = self.direct_logits(encoding["cells"])
        features, successors = self.lookahead(encoding["latent"], successors=successors)
        logits = direct + self.ranker(features).squeeze(-1)
        extra = {"direct": direct, "player": player, "features": features, "successors": successors}
        if self.cfg.query_readout:
            # The same readout serves the current policy and the actual-successor
            # policy: each encoding carries only its own current frame and history.
            extra["query"] = self.query_logits(encoding, player.softmax(-1))
            logits = logits + extra["query"]
        return logits, extra

    def forward(self, frames, history_valid=None, previous_actions=None, *, loops=None):
        return self.logits_from(self.encode(frames, history_valid, previous_actions, loops=loops))[0]


# ----------------------------------------------------------------- training
DEFAULT_WEIGHTS = {"prediction": 1., "sigreg": .1, "policy": 1., "value": .5,
                   "imagined_value": .5, "player": .1, "grounding": 1., "glyph": 1.,
                   "successor_policy": 0.}


def glyph_labels(batch, batch_size, device):
    """``current_triple [B, 3]`` and ``next_triple [B, 4, 3]`` as one ``[5B, 3]`` block, current first."""
    missing = [key for key in ("current_triple", "next_triple") if batch.get(key) is None]
    if missing:
        raise ValueError(f"glyph_recall requires generated glyph labels: {missing}")
    current = check_triples(torch.as_tensor(batch["current_triple"], device=device), batch_size)
    following = torch.as_tensor(batch["next_triple"], device=device)
    if tuple(following.shape) != (batch_size, ACTION_COUNT, 3):
        raise ValueError(f"next_triple must be [B, 4, 3], got {tuple(following.shape)}")
    return torch.cat((current, check_triples(following.flatten(0, 1))), dim=0)


def optimal_bits(masks):
    powers = torch.arange(ACTION_COUNT, device=masks.device)
    return ((masks.long()[:, None] >> powers) & 1).float()


def successor_policy_masks(batch, batch_size, device, terminal):
    """Validate optional actual-successor action labels before running the encoder."""
    if batch.get("next_optimal") is None:
        raise ValueError("positive successor_policy weight requires next_optimal [B, 4]")
    masks = torch.as_tensor(batch["next_optimal"], device=device)
    if tuple(masks.shape) != (batch_size, ACTION_COUNT):
        raise ValueError("next_optimal must be [B, 4]")
    if masks.dtype not in (torch.uint8, torch.uint16, torch.uint32, torch.uint64,
                            torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("next_optimal must contain integer 4-bit masks in 0..15")
    masks = masks.long()
    if bool(((masks < 0) | (masks > 15)).any()):
        raise ValueError("next_optimal must contain integer 4-bit masks in 0..15")
    if terminal.shape != masks.shape or bool((terminal & (masks != 0)).any()):
        raise ValueError("terminal successors must have next_optimal zero (terminal shape [B, 4])")
    return masks.flatten()


def _value_loss(model, latent, distances, terminal, won):
    distance_logits, terminal_logit, won_logit = model.value(latent)
    unreachable = distances < 0
    bins = torch.where(unreachable, torch.full_like(distances, model.bins - 1),
                       distances.clamp(0, model.bins - 2))
    loss = (F.cross_entropy(distance_logits, bins)
            + F.binary_cross_entropy_with_logits(terminal_logit, terminal.float())
            + F.binary_cross_entropy_with_logits(won_logit, won.float()))
    return loss, (distance_logits.argmax(-1) == bins).float().mean()


def _rank_of_true(predicted, targets):
    """predicted [B, 4, D], targets [B, 4, D] -> mean rank (1 = best) and top-1 rate."""
    distances = torch.cdist(predicted, targets)  # [B, 4(pred), 4(target)]
    own = distances.diagonal(dim1=1, dim2=2)[..., None]
    tied = torch.isclose(distances, own, atol=1e-7, rtol=1e-5)
    better = ((distances < own) & ~tied).sum(-1).float()
    ties = tied.sum(-1).clamp_min(1).float()
    rank = 1 + better + (ties - 1) / 2
    # Uniform tie-breaking: collapse scores chance (1/4), never perfect.
    top1 = (better == 0).float() / ties
    return rank.mean(), top1.mean()


def world_losses(model, batch, weights=None, *, loops=None, sigreg_generator=None):
    """Every training term from one batch of the NPZ contract.

    ``batch`` holds tensors ``frames [B, H, 64, 64]``, ``history_valid [B, H]``,
    ``previous_actions [B, H]``, ``next_frames [B, 4, 64, 64]``, ``terminal``
    and ``won`` ``[B, 4]``, ``optimal [B]`` bitmask, ``distances [B, 4]`` and
    optionally ``player_cell [B, 2]`` (col, row) or None. A positive
    ``successor_policy`` weight requires integer ``next_optimal [B, 4]``
    masks (0..15); zero masks are excluded and terminal masks must be zero.
    Optional rollout_mask [B] and rollout_actions [B,4] turn selected rows into
    four chronological steps; only step four may terminate or reset a life. Current policy
    still ranks all four first-action successors. Sequence labels are chronological.
    This auxiliary policy reuses actual successor encodings and imagines
    their own next latents; chronological rows separately supervise multistep predictions.
    Labels never touch logits: each policy sees only its own public history.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    frames, history_valid, previous_actions = model._prepare(
        batch["frames"], batch["history_valid"], batch["previous_actions"])
    device = frames.device
    batch_size, history = frames.shape[:2]
    from .world_rollout import (rollout_contract, replace_sequence_histories,
                                mixed_predictions, prediction_diagnostics, diagnostic_weights)
    rollout_mask, rollout_actions = rollout_contract(batch, batch_size, device)
    next_frames = torch.as_tensor(batch["next_frames"], device=device).long()
    if next_frames.shape != (batch_size, ACTION_COUNT, FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"next_frames must be [B, 4, 64, 64], got {tuple(next_frames.shape)}")
    terminal = torch.as_tensor(batch["terminal"], device=device).bool()
    won = torch.as_tensor(batch["won"], device=device).bool()
    distances = torch.as_tensor(batch["distances"], device=device).long()
    optimal = torch.as_tensor(batch["optimal"], device=device)

    if not math.isfinite(weights["successor_policy"]) or weights["successor_policy"] < 0:
        raise ValueError("successor_policy weight must be finite and nonnegative")
    next_optimal = None
    if weights["successor_policy"] > 0:
        next_optimal = successor_policy_masks(batch, batch_size, device, terminal)

    # Glyph perception: the current crop feeds the policy branch, the actual
    # successor crops feed only the target branch (one shared encoder, gradients on).
    current_glyph = next_glyph = None
    if model.cfg.glyph_recall:
        triples = glyph_labels(batch, batch_size, device)  # fail closed before any forward
        current_glyph = model.glyph_logits(frames[:, -1])
        next_glyph = model.glyph_logits(next_frames.flatten(0, 1))  # [B*4, 14]

    # Current observation: one encoder pass.
    tokens = model.frame_tokens(frames.flatten(0, 1)).view(batch_size, history, model.tokens, -1)
    current = model.assemble(tokens, history_valid, previous_actions, loops, glyph_logits=current_glyph)

    # Target observations: the history shifted by one, ending in each counterfactual
    # successor, through the SAME encoder (shared frame tokens, no stop-gradient).
    successor_tokens = model.frame_tokens(next_frames.flatten(0, 1)).view(
        batch_size, ACTION_COUNT, 1, model.tokens, -1)
    tail = tokens[:, 1:][:, None].expand(-1, ACTION_COUNT, -1, -1, -1)
    target_tokens = torch.cat((tail, successor_tokens), dim=2).flatten(0, 1)
    all_actions = torch.arange(ACTION_COUNT, device=device)[None].expand(batch_size, -1)
    target_valid = torch.cat((history_valid[:, 1:][:, None].expand(-1, ACTION_COUNT, -1),
                              torch.ones(batch_size, ACTION_COUNT, 1, dtype=torch.bool, device=device)),
                             dim=2).flatten(0, 1)
    target_actions = torch.cat((previous_actions[:, 1:][:, None].expand(-1, ACTION_COUNT, -1),
                                all_actions[..., None]), dim=2).flatten(0, 1)
    if batch.get("lost_life") is not None:
        reset = torch.as_tensor(batch["lost_life"], device=device).bool().flatten()
        if reset.shape != (batch_size * ACTION_COUNT,):
            raise ValueError("lost_life must be [B, 4]")
        # Match collector and inference: repeated reset frame, only the final
        # slot valid, no producing action retained across a lost-life boundary.
        reset_tokens = successor_tokens.flatten(0, 1).expand(-1, history, -1, -1)
        target_tokens = torch.where(reset[:, None, None, None], reset_tokens, target_tokens)
        reset_valid = torch.zeros_like(target_valid)
        reset_valid[:, -1] = True
        target_valid = torch.where(reset[:, None], reset_valid, target_valid)
        target_actions = torch.where(reset[:, None], -torch.ones_like(target_actions), target_actions)
    target_tokens, target_valid, target_actions = replace_sequence_histories(
        target_tokens, target_valid, target_actions, tokens, successor_tokens,
        history_valid, previous_actions, rollout_mask, rollout_actions, batch.get("lost_life"))
    actual_encoding = model.assemble(target_tokens, target_valid, target_actions, loops,
                                     glyph_logits=next_glyph)
    targets = actual_encoding["latent"].view(batch_size, ACTION_COUNT, -1)

    # LeWM terms.
    first_successors = model.predict_successors(current["latent"])  # four first-action alternatives
    predicted = mixed_predictions(model, current["latent"], first_successors,
                                  rollout_mask, rollout_actions)
    prediction = F.mse_loss(predicted, targets)
    slots = torch.cat((current["latent"][None], targets.transpose(0, 1)), dim=0)  # [5, B, D]
    sigreg = model.sigreg(slots, generator=sigreg_generator)

    # Task-specific terms.
    logits, extra = model.logits_from(current, successors=first_successors)
    bits = optimal_bits(optimal)
    target_policy = bits / bits.sum(1, keepdim=True).clamp_min(1.)
    log_probabilities = F.log_softmax(logits, dim=-1)
    policy = -(target_policy * log_probabilities).sum(-1).mean()
    value, distance_accuracy = _value_loss(model, targets.flatten(0, 1), distances.flatten(),
                                           terminal.flatten(), won.flatten())
    imagined_value, imagined_accuracy = _value_loss(model, predicted.flatten(0, 1), distances.flatten(),
                                                    terminal.flatten(), won.flatten())
    losses = {"prediction": prediction, "sigreg": sigreg, "policy": policy, "value": value,
              "imagined_value": imagined_value}
    grounding_diagnostics = {}
    if next_optimal is not None:
        # Reuse the actual encodings. Their policy imagines its own successors;
        # teacher labels and actual future images never enter the current logits.
        next_logits, _ = model.logits_from(actual_encoding)
        valid_next = next_optimal != 0
        next_bits = optimal_bits(next_optimal[valid_next])
        next_targets = next_bits / next_bits.sum(-1, keepdim=True).clamp_min(1.)
        selected_logits = next_logits[valid_next]
        losses["successor_policy"] = (-(next_targets * F.log_softmax(selected_logits, dim=-1))
                                      .sum() / valid_next.sum().clamp_min(1))
        with torch.no_grad():
            correct = next_bits.gather(1, selected_logits.argmax(-1)[:, None]).sum()
            grounding_diagnostics["successor_policy_set_accuracy"] = correct / valid_next.sum().clamp_min(1)
            grounding_diagnostics["successor_policy_valid_fraction"] = valid_next.float().mean()
    if model.cfg.grounding:
        from .world_grounding import world_grounding_losses
        losses["grounding"], latent_diagnostics = world_grounding_losses(
            model, batch, current["latent"], targets, predicted)
        grounding_diagnostics.update(latent_diagnostics)
    if model.cfg.glyph_recall:
        # Visual classification of the current and the four ACTUAL next crops;
        # imagined successors have no pixels and get no glyph term.
        glyph_logits_all = torch.cat((current_glyph, next_glyph), dim=0)
        losses["glyph"] = GlyphEncoder.loss(glyph_logits_all, triples)
        with torch.no_grad():
            for prefix, scores, labels in (("current", current_glyph, triples[:batch_size]),
                                           ("actual", next_glyph, triples[batch_size:])):
                per_field = GlyphEncoder.accuracies(scores, labels)[0]
                grounding_diagnostics.update({f"glyph_{prefix}_{name}_accuracy": value
                                              for name, value in zip(GLYPH_FIELDS, per_field)})
    player_cell = batch.get("player_cell")
    if player_cell is not None:
        player_cell = torch.as_tensor(player_cell, device=device).long()
        index = player_cell[:, 1] * GRID_COLS + player_cell[:, 0]
        if bool(((index < 0) | (index >= CELLS)).any()):
            raise ValueError("player_cell must hold (col, row) inside the 12x12 grid")
        losses["player"] = F.cross_entropy(extra["player"], index)
    total = sum(weights[name] * value_ for name, value_ in losses.items())

    with torch.no_grad():
        prediction_metrics = prediction_diagnostics(
            model, current["latent"], predicted, targets, rollout_mask, rollout_actions, _rank_of_true)
        chosen = logits.argmax(-1)
        diagnostics = {
            **grounding_diagnostics,
            **prediction_metrics,
            "target_variance_mean": targets.flatten(0, 1).var(0, correction=0).mean(),
            "target_variance_min": targets.flatten(0, 1).var(0, correction=0).min(),
            "set_accuracy": bits.gather(1, chosen[:, None]).mean(),
            "optimal_probability": (bits * log_probabilities.exp()).sum(-1).mean(),
            "distance_accuracy": distance_accuracy, "imagined_distance_accuracy": imagined_accuracy,
            "lookahead_span": (extra["features"][..., 0].max(-1).values
                               - extra["features"][..., 0].min(-1).values).mean(),
        }
        if player_cell is not None:
            diagnostics["player_accuracy"] = (extra["player"].argmax(-1) == index).float().mean()
    return {"total": total, "losses": losses, "diagnostics": diagnostics, "logits": logits,
            "latent": current["latent"], "targets": targets, "predicted": predicted,
            "glyph_logits": current_glyph,
            **({"diagnostic_weights": diagnostic_weights(rollout_mask, diagnostics)}
               if rollout_mask is not None else {})}


def parameter_groups(model, weight_decay):
    """Matrix weight decay only: LayerNorm/GroupNorm scales, biases and embedding
    tables are excluded."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "position" in name or "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.}]


# ------------------------------------------------------------------- IO
def build_world_policy(config=None):
    return WorldPolicy(WorldModelConfig.from_dict(config))


def save_world_checkpoint(path, model, **metadata):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reserved = {"format", "config", "parameters", "weights"} & metadata.keys()
    if reserved:
        raise ValueError(f"reserved checkpoint metadata: {sorted(reserved)}")
    checkpoint_ = {"format": WORLD_MODEL_FORMAT, "config": model.config(),
                   "parameters": model.parameter_count(), **metadata,
                   "weights": {key: value.detach().to("cpu") for key, value in model.state_dict().items()}}
    json.dumps({key: value for key, value in checkpoint_.items() if key != "weights"}, allow_nan=False)
    # Evaluation may read the previous epoch while training writes the next one.
    # A failed or interrupted write must also preserve the last usable snapshot.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint_, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint_


def load_world_checkpoint(path, device="cpu"):
    """Rebuild the saved network in eval mode. Returns (model, checkpoint)."""
    checkpoint_ = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint_.get("format") != WORLD_MODEL_FORMAT:
        raise ValueError(f"unsupported checkpoint format {checkpoint_.get('format')!r}, "
                         f"expected {WORLD_MODEL_FORMAT!r}")
    config = checkpoint_.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a dictionary")
    model = build_world_policy(config)
    model.load_state_dict(checkpoint_["weights"])
    if checkpoint_.get("parameters") != model.parameter_count():
        raise ValueError("checkpoint parameter count does not match the rebuilt network")
    return model.to(device).eval(), checkpoint_


def initialize_from_checkpoint(model, source):
    """Weights-only warm start of ``model`` from ``source`` (a ``WorldPolicy``).

    The only tolerated config differences are ``BOOLEAN_FLAGS`` switched from
    off to on: ``grounding`` adds a randomly initialised ``grounding_head``;
    ``state_recall`` and ``glyph_recall`` widen the first projector layer. Its
    input is a sequence of named blocks (``projector_layout``: base reduced
    tokens, HUD/player recall, glyph probabilities); every block the source has
    is copied to the block's offset in the target and every new block is zero,
    so adding recall in front of an existing glyph block moves that block
    rather than misaligning it. ``glyph_recall`` also adds a fresh
    ``glyph_encoder`` and a ``glyph_context`` that is zeroed here, so the
    migrated network computes exactly the source's latents and logits until
    training moves them. ``query_readout`` adds a fresh ``query_head`` whose
    output layer is zeroed here, so its residual correction starts at exactly
    zero; every other tensor of the head keeps its fresh initialization so the
    first nonzero output step immediately trains the queries. Any combination
    of these additions is handled together. Every other difference, every
    unexpected or missing key and every shape mismatch raises ``ValueError``.
    Returns the keys whose tensors were migrated rather than copied verbatim.
    ``cell_recall`` may also be added: its frozen decoder starts fresh and its
    projection is zeroed. The trainer separately requires a proven generated
    decoder import before optimizing a model with this flag enabled.
    """
    target_config, source_config = model.config(), source.config()
    for flag in BOOLEAN_FLAGS:
        if source_config[flag] and not target_config[flag]:
            raise ValueError(f"cannot initialize a network without {flag} from one with it")
    differing = sorted(key for key in target_config if key not in BOOLEAN_FLAGS
                       and target_config[key] != source_config[key])
    if differing:
        raise ValueError(f"initialization config differs beyond {list(BOOLEAN_FLAGS)}: {differing}")
    weights = {key: value.detach().to("cpu").clone() for key, value in source.state_dict().items()}
    migrated = []
    target_layout, source_layout = model.projector_layout(), source.projector_layout()
    if target_layout != source_layout:
        key = "projector.0.weight"
        shape = tuple(model.projector[0].weight.shape)
        expected = (shape[0], sum(width for _, _, width in source_layout))
        if key not in weights or tuple(weights[key].shape) != expected:
            found = tuple(weights[key].shape) if key in weights else None
            raise ValueError(f"{key}: expected {expected} in the source, got {found}")
        padded = torch.zeros(shape, dtype=weights[key].dtype)
        for (name, offset, width), (_, source_offset, source_width) in zip(target_layout, source_layout):
            if source_width == 0:
                continue  # a new block: its columns stay zero
            if source_width != width:
                raise ValueError(f"{key}: block {name} is {source_width} wide in the source but {width} here")
            padded[:, offset:offset + width] = weights[key][:, source_offset:source_offset + width]
        weights[key] = padded
        migrated.append(key)
    expected_keys = model.state_dict()
    allowed_missing = set()
    if target_config["grounding"] and not source_config["grounding"]:
        allowed_missing |= {key for key in expected_keys if key.startswith("grounding_head.")}
    adding_glyph = target_config["glyph_recall"] and not source_config["glyph_recall"]
    if adding_glyph:
        allowed_missing |= {key for key in expected_keys
                            if key.startswith("glyph_encoder.") or key.startswith("glyph_context.")}
        if source_config["query_readout"]:
            # An existing trained query head gains glyph probabilities at the
            # end of its context vector. Preserve its old computation exactly.
            key = "query_head.context.0.weight"
            target_shape = tuple(expected_keys[key].shape)
            source_shape = (target_shape[0], target_shape[1] - GLYPH_CLASSES)
            if key not in weights or tuple(weights[key].shape) != source_shape:
                raise ValueError(f"{key}: expected source shape {source_shape}")
            padded = torch.zeros(target_shape, dtype=weights[key].dtype)
            padded[:, :source_shape[1]] = weights[key]
            weights[key] = padded
            migrated.append(key)
    adding_query = target_config["query_readout"] and not source_config["query_readout"]
    if adding_query:
        allowed_missing |= {key for key in expected_keys if key.startswith("query_head.")}
    adding_cells = target_config["cell_recall"] and not source_config["cell_recall"]
    if adding_cells:
        allowed_missing |= {key for key in expected_keys
                            if key.startswith("cell_appearance.") or key.startswith("cell_context.")}
    unexpected = sorted(set(weights) - set(expected_keys))
    missing = sorted(set(expected_keys) - set(weights) - allowed_missing)
    if unexpected or missing:
        raise ValueError(f"incompatible initialization weights: unexpected {unexpected}, missing {missing}")
    mismatched = sorted(key for key, value in weights.items() if tuple(value.shape) != tuple(expected_keys[key].shape))
    if mismatched:
        raise ValueError(f"initialization weight shapes differ: {mismatched}")
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != allowed_missing:
        raise ValueError(f"incompatible initialization weights: {result}")
    if adding_glyph:
        with torch.no_grad():  # output preservation: the fresh classifier cannot reach the tokens yet
            model.glyph_context.weight.zero_()
    if adding_query:
        model.query_head.reset_output()  # output preservation: the correction starts at zero
    if adding_cells:
        with torch.no_grad():
            model.cell_context.weight.zero_()
    return migrated


def initialize_glyph_encoder(model, encoder):
    """Copy a pretrained ``GlyphEncoder``'s weights into ``model.glyph_encoder`` (strict)."""
    if not model.cfg.glyph_recall:
        raise ValueError("glyph encoder initialization needs a network with glyph_recall")
    if not isinstance(encoder, GlyphEncoder) or encoder.config() != model.glyph_encoder.config():
        raise ValueError("glyph encoder architecture does not match the network's glyph_encoder")
    model.glyph_encoder.load_state_dict(
        {key: value.detach().to("cpu").clone() for key, value in encoder.state_dict().items()}, strict=True)
    return sorted(f"glyph_encoder.{key}" for key in encoder.state_dict())
