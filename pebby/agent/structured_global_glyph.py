"""Shared-loop field transition with one learned categorical carried state.

The base spatial model is unchanged. A new attention-pooling head predicts one
shape/color/rotation distribution and broadcasts it over the output field.
No engine rules, exact state inputs, teacher forcing, or policy are used here.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .structured_transition import (
    FIELD_TOKENS, FIELD_WIDTH, STEPS_CLASSES, StructuredTransition,
    StructuredTransitionConfig, _actions, _field,
)

GLOBAL_GLYPH_FORMAT = 'pebby.structured-transition-global-glyph.v1'


@dataclass(frozen=True)
class GlobalGlyphConfig:
    loops: int = 2
    heads: int = 4
    expansion: int = 2
    event_hidden: int = 128
    steps_classes: int = STEPS_CLASSES
    glyph_hidden: int = 96
    variant: str = 'global_glyph'

    def __post_init__(self):
        StructuredTransitionConfig(**self.base_config())
        if type(self.glyph_hidden) is not int or self.glyph_hidden < 1:
            raise ValueError('glyph_hidden must be a positive integer')
        if self.variant != 'global_glyph':
            raise ValueError('variant must be global_glyph')

    def base_config(self):
        return {key: getattr(self, key) for key in
                ('loops', 'heads', 'expansion', 'event_hidden', 'steps_classes')}


class GlobalGlyphTransition(StructuredTransition):
    """Base field dynamics plus a shared source/state -> global glyph head.

    ``readout`` remains the identical base readout on current, actual, and
    predicted fields. ``glyph_logits`` exposes the new head separately for
    diagnostics; it does not silently replace readout logits or add a loss.
    """
    checkpoint_format = GLOBAL_GLYPH_FORMAT

    def __init__(self, config=None, **overrides):
        if config is None:
            config = GlobalGlyphConfig(**overrides)
        elif isinstance(config, dict):
            config = GlobalGlyphConfig(**(config | overrides))
        elif overrides or not isinstance(config, GlobalGlyphConfig):
            raise ValueError('pass a global glyph config or keyword overrides')
        super().__init__(config.base_config())
        self.cfg = config
        self.glyph_pool_score = nn.Linear(2 * FIELD_WIDTH, 1)
        self.glyph_head = nn.Sequential(
            nn.LayerNorm(2 * FIELD_WIDTH),
            nn.Linear(2 * FIELD_WIDTH, config.glyph_hidden), nn.GELU(),
            nn.Linear(config.glyph_hidden, 14),
        )

    def config(self):
        return asdict(self.cfg)

    def warmstart_from_base_state_dict(self, state_dict):
        """Copy ALL base parameters; preserve only randomly initialized new heads.

        Fail before loading if keys or shapes differ. Callers bind the source
        checkpoint hash separately; this helper never reads or writes files.
        """
        own = self.state_dict()
        new = {key for key in own if key.startswith(('glyph_pool_score.', 'glyph_head.'))}
        expected = set(own) - new
        if set(state_dict) != expected:
            raise ValueError('warmstart must contain exactly the complete base state dict')
        if any(state_dict[key].shape != own[key].shape for key in expected):
            raise ValueError('base warmstart parameter shape mismatch')
        result = self.load_state_dict(state_dict, strict=False)
        if set(result.missing_keys) != new or result.unexpected_keys:
            raise RuntimeError('unexpected global glyph warmstart mismatch')
        return sorted(new)

    def _prediction(self, field, actions, loops=None):
        _field(field)
        actions = _actions(actions, len(field), field.device)
        depth = self.cfg.loops if loops is None else loops
        if type(depth) is not int or depth < 1:
            raise ValueError('loops must be a positive integer')
        source = field + self.position.to(field.dtype)[None] + self.action_embedding(actions).to(field.dtype)[:, None]
        state = source
        for _ in range(depth):
            state = self.block(state, source)
        ordinary = field + self.output(state)
        evidence = torch.cat((source, state), -1)
        pooled = (self.glyph_pool_score(evidence).softmax(1) * evidence).sum(1)
        shape, color, rotation = self.glyph_head(pooled).split((6, 4, 4), -1)
        logits = dict(shape=shape, color=color, rotation=rotation)
        probabilities = torch.cat([value.softmax(-1) for value in logits.values()], -1)
        # Concatenation preserves autograd through both spatial and glyph paths.
        predicted = torch.cat((ordinary[..., :70],
                               probabilities[:, None].expand(-1, FIELD_TOKENS, -1),
                               ordinary[..., 84:]), -1)
        return predicted, logits

    def predict(self, field, actions, *, loops=None):
        return self._prediction(field, actions, loops)[0]

    def forward(self, field, actions, *, loops=None):
        predicted, glyph_logits = self._prediction(field, actions, loops)
        events = self.event_head(torch.cat((self.readout.summary(field), self.readout.summary(predicted),
                                           self.action_embedding(actions.long())), -1))
        return dict(field=predicted, readout=self.readout(predicted), glyph_logits=glyph_logits,
                    events=dict(lost_life_logits=events[:, 0], terminal_logits=events[:, 1], won_logits=events[:, 2]))

    def rollout(self, field, actions, *, loops=None):
        """Four or any positive number of autoregressive steps, without detach."""
        _field(field)
        actions = _actions(actions, len(field), field.device, ndim=2)
        if actions.shape[1] < 1:
            raise ValueError('rollout requires at least one action')
        outputs = []
        for step in range(actions.shape[1]):
            output = self(field, actions[:, step], loops=loops)
            outputs.append(output)
            field = output['field']
        return dict(fields=torch.stack([x['field'] for x in outputs], 1),
                    readout={key: torch.stack([x['readout'][key] for x in outputs], 1) for key in outputs[0]['readout']},
                    events={key: torch.stack([x['events'][key] for x in outputs], 1) for key in outputs[0]['events']},
                    glyph_logits={key: torch.stack([x['glyph_logits'][key] for x in outputs], 1) for key in outputs[0]['glyph_logits']})
