"""Local glyph dynamics with learned physical predictions carried forward.

Observed fields have zero padding at85:96. An imagined output instead carries
player probabilities at85, a learned six-dimensional budget embedding at86:92,
and lives probabilities at92:96. No exact state, engine rule, or future label
enters this module. First-step inputs stay unchanged; subsequent calls consume
the previous output, including its feedback, without reinitializing it.

Training must exclude85:96 from observed-field distillation and supervise
``feedback_logits`` with physical labels, in addition to the final readout.
This is an untrained architectural alternative, not a calibrated state model.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .structured_local_glyph import LocalGlyphConfig, LocalGlobalGlyphTransition
from .structured_transition import BOARD_TOKENS, FIELD_TOKENS, _actions, _field


PHYSICAL_FEEDBACK_FORMAT = 'pebby.structured-transition-physical-feedback.v1'
_BASE_CONFIG_KEYS = ('loops', 'heads', 'expansion', 'event_hidden',
                     'steps_classes', 'glyph_hidden')


@dataclass(frozen=True)
class PhysicalFeedbackConfig:
    loops: int = 2
    heads: int = 4
    expansion: int = 2
    event_hidden: int = 128
    steps_classes: int = 44
    glyph_hidden: int = 96
    variant: str = 'physical_feedback'

    def __post_init__(self):
        LocalGlyphConfig(**self.local_config())
        if self.variant != 'physical_feedback':
            raise ValueError('variant must be physical_feedback')

    def local_config(self):
        return {key: getattr(self, key) for key in _BASE_CONFIG_KEYS}


class PhysicalFeedbackTransition(LocalGlobalGlyphTransition):
    """Preserve the public transition interface and the learned spatial model.

The ordinary next-field readout supplies differentiable distributions to the
feedback channels. The returned generic readout and events are evaluated on
the final field, while ``feedback_logits`` exposes the distributions actually
carried into the next transition. Like ``glyph_logits``, these have their own
supervision; they are not silently substituted for the generic readout.

All original parameters are constructed before the new budget embedding, so
same-seed base and feedback models have identical shared initialization. Their
full predictions are intentionally different even before training.
"""
    checkpoint_format = PHYSICAL_FEEDBACK_FORMAT

    def __init__(self, config=None, **overrides):
        if config is None:
            config = PhysicalFeedbackConfig(**overrides)
        elif isinstance(config, dict):
            config = PhysicalFeedbackConfig(**(config | overrides))
        elif overrides or not isinstance(config, PhysicalFeedbackConfig):
            raise ValueError('pass a physical feedback config or keyword overrides')
        super().__init__(config.local_config())
        self.cfg = config
        self.budget_feedback = nn.Linear(config.steps_classes, 6, bias=False)

    def config(self):
        return asdict(self.cfg)

    def warmstart_from_local_state_dict(self, state_dict):
        """Validate the complete base model before copying; retain new weights.

        The caller owns checkpoint/source provenance. This method performs no
        file access and rejects partial or already-feedback checkpoints.
        """
        own = self.state_dict()
        new = {'budget_feedback.weight'}
        expected = set(own) - new
        if set(state_dict) != expected:
            raise ValueError('warmstart requires exactly complete local glyph state')
        if any(own[key].shape != state_dict[key].shape for key in expected):
            raise ValueError('local warmstart shape mismatch')
        result = self.load_state_dict(state_dict, strict=False)
        if set(result.missing_keys) != new or result.unexpected_keys:
            raise RuntimeError('unexpected physical feedback warmstart mismatch')
        return sorted(new)

    def forward(self, field, actions, *, loops=None):
        _field(field)
        actions = _actions(actions, len(field), field.device)
        ordinary, glyph_logits = super()._prediction(field, actions, loops)
        before_feedback = self.readout(ordinary)
        feedback_logits = {name: before_feedback[name + '_logits']
                           for name in ('player', 'steps', 'lives')}
        player = torch.cat((feedback_logits['player'].softmax(-1),
                            ordinary.new_zeros(len(field), FIELD_TOKENS - BOARD_TOKENS)), -1)
        budget = self.budget_feedback(feedback_logits['steps'].softmax(-1))
        lives = feedback_logits['lives'].softmax(-1)
        predicted = torch.cat((ordinary[..., :85], player[..., None],
                               budget[:, None].expand(-1, FIELD_TOKENS, -1),
                               lives[:, None].expand(-1, FIELD_TOKENS, -1)), -1)
        events = self.event_head(torch.cat((self.readout.summary(field),
                                           self.readout.summary(predicted),
                                           self.action_embedding(actions)), -1))
        return dict(field=predicted, readout=self.readout(predicted),
                    glyph_logits=glyph_logits, feedback_logits=feedback_logits,
                    events=dict(lost_life_logits=events[:, 0],
                                terminal_logits=events[:, 1], won_logits=events[:, 2]))

    def predict(self, field, actions, *, loops=None):
        return self(field, actions, loops=loops)['field']

    def rollout(self, field, actions, *, loops=None):
        """Use only each preceding predicted field; no reset or teacher forcing."""
        _field(field)
        actions = _actions(actions, len(field), field.device, ndim=2)
        if actions.shape[1] < 1:
            raise ValueError('rollout requires at least one action')
        outputs = []
        for step in range(actions.shape[1]):
            output = self(field, actions[:, step], loops=loops)
            outputs.append(output)
            field = output['field']
        result = dict(fields=torch.stack([output['field'] for output in outputs], 1))
        for group in ('readout', 'events', 'glyph_logits', 'feedback_logits'):
            result[group] = {key: torch.stack([output[group][key] for output in outputs], 1)
                             for key in outputs[0][group]}
        return result
