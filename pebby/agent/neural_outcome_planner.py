"""One-step neural outcome planner over a frozen encoder's public features.

Two fixed attention decoder blocks predict four first-action outcomes. A
permutation-equivariant comparator sees only their decoded probabilities. There
is no recurrent refinement, temporal rollout, environment access or policy-logit
bypass. Targets belong exclusively to ``neural_outcome_losses``.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .world_grounding import SIZES

ACTION_COUNT = 4
SLOT_COUNT = 8
VALUE_BINS = 130
EVENT_NAMES = ('lost_life', 'terminal', 'won')
FIELD_NAMES = ('player', 'shape', 'color', 'rotation', 'steps', 'lives')
DEFAULT_WEIGHTS = {'physical': 1., 'value': 1., 'events': 1., 'policy': 1.}


@dataclass(frozen=True)
class NeuralOutcomePlannerConfig:
    channels: int = 64
    heads: int = 4
    expansion: int = 2
    context_tokens: int = 160
    glyph_inputs: int = 14
    comparator_hidden: int = 64

    def __post_init__(self):
        for name in ('channels', 'heads', 'expansion', 'context_tokens', 'comparator_hidden'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.channels % self.heads:
            raise ValueError('channels must be divisible by heads')
        if isinstance(self.glyph_inputs, bool) or self.glyph_inputs not in (0, 14):
            raise ValueError('glyph_inputs must be 0 or 14')

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        for key, expected in (('architecture', 'neural_outcome_planner'), ('horizon', 1), ('decoder_blocks', 2)):
            if key in config and config.pop(key) != expected:
                raise ValueError(f'{key} must be {expected!r}')
        return cls(**config)


class _OutcomeBlock(nn.Module):
    def __init__(self, channels, heads, expansion):
        super().__init__()
        self.self_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * expansion), nn.GELU(),
                                 nn.Linear(channels * expansion, channels))

    def forward(self, queries, context):
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        queries = queries + self.cross_attention(self.cross_norm(queries), context, context,
                                                 need_weights=False)[0]
        return queries + self.ffn(self.ffn_norm(queries))


def _outcome_contract(field_logits, value_logits, event_logits):
    if not isinstance(field_logits, (tuple, list)) or len(field_logits) != len(SIZES):
        raise ValueError('field_logits must contain the six physical fields')
    if not torch.is_tensor(value_logits) or value_logits.ndim != 3 or value_logits.shape[1:] != (ACTION_COUNT, VALUE_BINS):
        raise ValueError('value_logits must be [B,4,130]')
    batch = value_logits.size(0)
    if batch < 1:
        raise ValueError('outcomes require a nonempty batch')
    for name, tensor, size in [*(zip(FIELD_NAMES, field_logits, SIZES)), ('events', event_logits, 3)]:
        if not torch.is_tensor(tensor) or tensor.shape != (batch, ACTION_COUNT, size):
            raise ValueError(f'{name} logits must be [B,4,{size}]')
        if tensor.device != value_logits.device or not tensor.is_floating_point():
            raise ValueError('outcome logits must be floating tensors on one device')
    if not value_logits.is_floating_point():
        raise ValueError('value_logits must be floating point')
    return batch


class NeuralOutcomePlanner(nn.Module):
    """Current ``raw/state [B,T,C]`` and optional glyph -> four learned outcomes."""

    def __init__(self, config=None, **overrides):
        super().__init__()
        if isinstance(config, dict):
            config = NeuralOutcomePlannerConfig.from_dict({**config, **overrides})
        elif config is None:
            config = NeuralOutcomePlannerConfig(**overrides)
        elif overrides:
            raise ValueError('pass a config or keyword overrides, not both')
        if not isinstance(config, NeuralOutcomePlannerConfig):
            raise TypeError('config must be a NeuralOutcomePlannerConfig or dict')
        self.cfg = config
        width = config.channels
        self.context_projection = nn.Linear(2 * width, width)
        self.context_norm = nn.LayerNorm(width)
        self.glyph_projection = (nn.Linear(config.glyph_inputs, width, bias=False)
                                 if config.glyph_inputs else None)
        self.action_embedding = nn.Parameter(torch.empty(ACTION_COUNT, width))
        self.slot_embedding = nn.Parameter(torch.empty(SLOT_COUNT, width))
        nn.init.normal_(self.action_embedding, std=.02)
        nn.init.normal_(self.slot_embedding, std=.02)
        self.blocks = nn.ModuleList([_OutcomeBlock(width, config.heads, config.expansion) for _ in range(2)])
        self.output_norm = nn.LayerNorm(width)
        self.field_heads = nn.ModuleList([nn.Linear(width, size) for size in SIZES])
        self.value_head = nn.Linear(width, VALUE_BINS)
        self.event_head = nn.Linear(width, len(EVENT_NAMES))
        # The comparator receives no action embedding or encoder features.
        outcome_width = sum(SIZES) + VALUE_BINS + len(EVENT_NAMES)
        hidden = config.comparator_hidden
        self.outcome_projection = nn.Sequential(nn.Linear(outcome_width, hidden), nn.GELU())
        self.comparator = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def config(self):
        return dict(architecture='neural_outcome_planner', horizon=1, decoder_blocks=2, **asdict(self.cfg))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def score_outcomes(self, field_logits, value_logits, event_logits):
        """Score typed logits using only probabilities; whole-branch permutations commute.

        All field/value tensors are categorical logits. Events are three
        independent binary logits in lost_life, terminal, won order. The caller
        can substitute measured actual outcomes for a privileged diagnostic.
        """
        _outcome_contract(field_logits, value_logits, event_logits)
        probabilities = torch.cat([*[x.float().softmax(-1) for x in field_logits],
                                   value_logits.float().softmax(-1), event_logits.float().sigmoid()], -1)
        branches = self.outcome_projection(probabilities)
        pooled = branches.mean(1, keepdim=True).expand_as(branches)
        return self.comparator(torch.cat((branches, pooled), -1)).squeeze(-1)

    def forward(self, raw, state, glyph=None):
        expected_tail = (self.cfg.context_tokens, self.cfg.channels)
        if not torch.is_tensor(raw) or raw.ndim != 3 or raw.shape[1:] != expected_tail or raw.size(0) < 1:
            raise ValueError(f'raw must be a nonempty [B,{expected_tail[0]},{expected_tail[1]}] tensor')
        if not torch.is_tensor(state) or state.shape != raw.shape or state.device != raw.device:
            raise ValueError('state must match raw shape and device')
        if not raw.is_floating_point() or not state.is_floating_point():
            raise ValueError('raw and state must be floating point')
        batch = raw.size(0)
        context = self.context_norm(self.context_projection(torch.cat((raw, state), -1)))
        queries = self.action_embedding[:, None] + self.slot_embedding[None]
        queries = queries[None].expand(batch, -1, -1, -1).to(context.dtype)
        if glyph is not None:
            if self.glyph_projection is None:
                raise ValueError('glyph is disabled by config')
            if not torch.is_tensor(glyph) or glyph.shape != (batch, self.cfg.glyph_inputs) or glyph.device != raw.device:
                raise ValueError('glyph must be [B,14] on the context device')
            if not glyph.is_floating_point():
                raise ValueError('glyph must be floating point')
            queries = queries + self.glyph_projection(glyph)[:, None, None].to(context.dtype)
        queries = queries.reshape(batch * ACTION_COUNT, SLOT_COUNT, self.cfg.channels)
        memory = context[:, None].expand(-1, ACTION_COUNT, -1, -1).reshape(
            batch * ACTION_COUNT, self.cfg.context_tokens, self.cfg.channels)
        for block in self.blocks:  # two distinct fixed blocks, no refinement loop
            queries = block(queries, memory)
        slots = self.output_norm(queries).reshape(batch, ACTION_COUNT, SLOT_COUNT, self.cfg.channels)
        fields = tuple(head(slots[:, :, index]) for index, head in enumerate(self.field_heads))
        value = self.value_head(slots[:, :, 6])
        events = self.event_head(slots[:, :, 7])
        return dict(action_logits=self.score_outcomes(fields, value, events),
                    field_logits=fields, value_logits=value, event_logits=events)


def _integer_target(batch, key, shape, device):
    if key not in batch:
        raise ValueError(f'missing target {key}')
    tensor = torch.as_tensor(batch[key], device=device)
    if tuple(tensor.shape) != shape or tensor.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f'{key} must be an integer tensor with shape {shape}')
    return tensor.long()


def neural_outcome_losses(predictions, batch, weights=None):
    """Losses from generated first-successor labels; targets never enter forward.

    ``distances`` describes the physical post-action state. Only negative
    distance maps to unreachable; a reachable life-loss/reset successor keeps
    its reachable state-value label. Zero optimal masks have no policy target.
    Undefined supervised metrics are None and carry zero diagnostic weight.
    """
    supplied = {} if weights is None else dict(weights)
    if set(supplied) - DEFAULT_WEIGHTS.keys():
        raise ValueError('unknown loss family')
    weights = {**DEFAULT_WEIGHTS, **supplied}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError('loss weights must be finite and nonnegative')
    fields, value, events = (predictions[k] for k in ('field_logits', 'value_logits', 'event_logits'))
    size = _outcome_contract(fields, value, events)
    logits = predictions['action_logits']
    if logits.shape != (size, ACTION_COUNT) or logits.device != value.device:
        raise ValueError('action_logits must be [B,4] on the outcome device')
    device = value.device
    player = _integer_target(batch, 'next_player_cell', (size, 4, 2), device)
    triple = _integer_target(batch, 'next_triple', (size, 4, 3), device)
    steps = _integer_target(batch, 'next_steps', (size, 4), device)
    lives = _integer_target(batch, 'next_lives', (size, 4), device)
    distances = _integer_target(batch, 'distances', (size, 4), device)
    optimal = _integer_target(batch, 'optimal', (size,), device)
    if bool(((player < 0) | (player >= 12)).any()):
        raise ValueError('next_player_cell must be (col,row) inside the 12x12 grid')
    if bool(((optimal < 0) | (optimal > 15)).any()):
        raise ValueError('optimal masks must be in 0..15')
    targets = (player[..., 1] * 12 + player[..., 0], *triple.unbind(-1), steps.clamp_min(-1) + 1, lives)
    for target, classes in zip(targets, SIZES):
        if bool(((target < 0) | (target >= classes)).any()):
            raise ValueError('physical target outside category range')
    event_targets = []
    for name in EVENT_NAMES:
        if name not in batch:
            raise ValueError(f'missing target {name}')
        target = torch.as_tensor(batch[name], device=device)
        if target.shape != (size, ACTION_COUNT) or bool(((target != 0) & (target != 1)).any()):
            raise ValueError(f'{name} must be binary [B,4]')
        event_targets.append(target.float())
    event_targets = torch.stack(event_targets, -1)
    bins = torch.where(distances < 0, 129, distances.clamp(0, 128))
    bits = (optimal[:, None] & (1 << torch.arange(ACTION_COUNT, device=device))) != 0
    valid = bits.any(-1)
    valid_count = valid.sum()
    policy_targets = bits.float() / bits.sum(-1, keepdim=True).clamp_min(1)
    policy_log_probabilities = logits.float().log_softmax(-1)
    losses = dict(
        physical=torch.stack([F.cross_entropy(scores.float().flatten(0, 1), target.flatten())
                              for scores, target in zip(fields, targets)]).mean(),
        value=F.cross_entropy(value.float().flatten(0, 1), bins.flatten()),
        events=F.binary_cross_entropy_with_logits(events.float(), event_targets),
        policy=-(policy_targets * policy_log_probabilities).sum() / valid_count.clamp_min(1))
    with torch.no_grad():
        chosen = logits.argmax(-1)
        correct = bits.gather(1, chosen[:, None]).squeeze(-1)
        diagnostics = dict(policy_valid_count=valid_count, policy_valid_fraction=valid.float().mean(),
            set_accuracy=correct[valid].float().mean() if bool(valid_count) else None,
            optimal_probability=(bits * policy_log_probabilities.exp()).sum(-1)[valid].mean() if bool(valid_count) else None,
            value_accuracy=(value.argmax(-1) == bins).float().mean(),
            value_overflow_fraction=(distances > 128).float().mean(),
            unreachable_fraction=(distances < 0).float().mean(),
            life_loss_fraction=event_targets[..., 0].mean())
        diagnostics.update({f'{name}_accuracy': (scores.argmax(-1) == target).float().mean()
                            for name, scores, target in zip(FIELD_NAMES, fields, targets)})
        metric_weights = {name: size * ACTION_COUNT for name in diagnostics}
        for index, name in enumerate(EVENT_NAMES):
            positives = event_targets[..., index].bool()
            support = positives.sum()
            diagnostics[name + '_support'] = support
            diagnostics[name + '_recall'] = ((events[..., index] >= 0) & positives).sum() / support if bool(support) else None
            metric_weights[name + '_recall'] = support
            metric_weights[name + '_support'] = 1
        metric_weights.update(policy=valid_count, set_accuracy=valid_count,
                              optimal_probability=valid_count, policy_valid_count=1, policy_valid_fraction=size)
    return dict(total=sum(weights[name] * loss for name, loss in losses.items()), losses=losses,
                diagnostics=diagnostics, diagnostic_weights=metric_weights)
