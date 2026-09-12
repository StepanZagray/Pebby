"""Learned transitions of public 148-token fields; no pixel or engine access.

Input assembly is external: 144 board tokens and four HUD tokens, width96.
Observed channel groups are world48, appearance22, carried14, visibility1,
and eleven padding channels. Predicted fields are continuous residual states;
their channels are not asserted to remain calibrated appearance probabilities.
No policy, search, teacher forcing, or loss is implemented here.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

FIELD_TOKENS = 148
BOARD_TOKENS = 144
FIELD_WIDTH = 96
STEPS_CLASSES = 44
STEPS_UNDERFLOW_CLASS = 43


def steps_targets(steps):
    """Map integer budgets -3..42 to 44 classes, preserving tensor shape/device.

    Classes0..42 are exact budgets. Class43 groups the observed negative
    budgets (-3,-2,-1); it is distinct from an exhausted but nonnegative zero.
    Unexpected values are rejected, never silently clamped. The domain was
    checked against combined train and aggregate2 actual successor labels.
    """
    if not isinstance(steps, torch.Tensor) or steps.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('steps targets must be an integer tensor')
    values = steps.long()
    if bool(((values < -3) | (values > 42)).any()):
        raise ValueError('steps targets must be in observed domain -3..42')
    return torch.where(values < 0, STEPS_UNDERFLOW_CLASS, values)


@dataclass(frozen=True)
class StructuredTransitionConfig:
    loops: int = 2
    heads: int = 4
    expansion: int = 2
    event_hidden: int = 128
    steps_classes: int = STEPS_CLASSES

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if FIELD_WIDTH % self.heads:
            raise ValueError('heads must divide field width96')
        if self.steps_classes != STEPS_CLASSES:
            raise ValueError('steps_classes must be44: budgets0..42 plus underflow')


def _field(field):
    if not isinstance(field, torch.Tensor) or field.ndim != 3 or tuple(field.shape[1:]) != (FIELD_TOKENS, FIELD_WIDTH):
        raise ValueError('field must be [B,148,96]')
    if not field.is_floating_point():
        raise ValueError('field must be floating point')


def _actions(actions, batch, device, ndim=1):
    if (not isinstance(actions, torch.Tensor) or actions.ndim != ndim or actions.shape[0] != batch
            or actions.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError('actions must be integer [B] or rollout integer [B,K] as appropriate')
    if actions.device != device:
        raise ValueError('actions and fields must share a device')
    if bool(((actions < 0) | (actions > 3)).any()):
        raise ValueError('actions must be in0..3')
    return actions.long()


class SharedFieldBlock(nn.Module):
    """One pre-norm block recalling the same transition source in both branches."""
    def __init__(self, heads, expansion):
        super().__init__()
        self.attention_recall = nn.Linear(2 * FIELD_WIDTH, FIELD_WIDTH, bias=False)
        self.attention_norm = nn.LayerNorm(FIELD_WIDTH)
        self.attention = nn.MultiheadAttention(FIELD_WIDTH, heads, dropout=0., batch_first=True)
        self.mlp_recall = nn.Linear(2 * FIELD_WIDTH, FIELD_WIDTH, bias=False)
        self.mlp_norm = nn.LayerNorm(FIELD_WIDTH)
        self.mlp = nn.Sequential(nn.Linear(FIELD_WIDTH, FIELD_WIDTH * expansion), nn.GELU(),
                                 nn.Linear(FIELD_WIDTH * expansion, FIELD_WIDTH))

    def forward(self, state, source):
        query = self.attention_norm(self.attention_recall(torch.cat((state, source), -1)))
        state = state + self.attention(query, query, query, need_weights=False)[0]
        value = self.mlp_norm(self.mlp_recall(torch.cat((state, source), -1)))
        return state + self.mlp(value)


class StructuredFieldReadout(nn.Module):
    """State heads usable on predicted OR externally supplied detached actual fields."""
    def __init__(self, steps_classes=STEPS_CLASSES):
        super().__init__()
        if type(steps_classes) is not int or steps_classes != STEPS_CLASSES:
            raise ValueError('steps_classes must be44: budgets0..42 plus underflow')
        self.player = nn.Linear(FIELD_WIDTH, 1)
        self.cell = nn.Linear(FIELD_WIDTH, 22)
        self.pool_score = nn.Linear(FIELD_WIDTH, 1)
        self.global_head = nn.Sequential(nn.LayerNorm(FIELD_WIDTH), nn.Linear(FIELD_WIDTH, FIELD_WIDTH),
                                         nn.GELU(), nn.Linear(FIELD_WIDTH, 14 + steps_classes + 4))
        self.steps_classes = steps_classes

    def summary(self, field):
        weights = self.pool_score(field).softmax(1)
        return (weights * field).sum(1)

    def forward(self, field):
        _field(field)
        cells = field[:, :BOARD_TOKENS]
        role, shape, color, rotation = self.cell(cells).split((8, 6, 4, 4), -1)
        carried_shape, carried_color, carried_rotation, steps, lives = self.global_head(self.summary(field)).split(
            (6, 4, 4, self.steps_classes, 4), -1)
        return dict(player_logits=self.player(cells).squeeze(-1), role_logits=role,
                    goal_shape_logits=shape, goal_color_logits=color, goal_rotation_logits=rotation,
                    carried_shape_logits=carried_shape, carried_color_logits=carried_color,
                    carried_rotation_logits=carried_rotation, steps_logits=steps, lives_logits=lives)


class StructuredTransition(nn.Module):
    def __init__(self, config=None, **overrides):
        super().__init__()
        if config is None:
            config = StructuredTransitionConfig(**overrides)
        elif isinstance(config, dict):
            config = StructuredTransitionConfig(**(config | overrides))
        elif overrides or not isinstance(config, StructuredTransitionConfig):
            raise ValueError('pass a config or keyword overrides')
        self.cfg = config
        self.action_embedding = nn.Embedding(4, FIELD_WIDTH)
        self.position = nn.Parameter(torch.empty(FIELD_TOKENS, FIELD_WIDTH))
        nn.init.normal_(self.position, std=.02)
        nn.init.normal_(self.action_embedding.weight, std=.02)
        self.block = SharedFieldBlock(config.heads, config.expansion)
        self.output = nn.Sequential(nn.LayerNorm(FIELD_WIDTH), nn.Linear(FIELD_WIDTH, FIELD_WIDTH))
        self.readout = StructuredFieldReadout(config.steps_classes)
        self.event_head = nn.Sequential(nn.Linear(3 * FIELD_WIDTH, config.event_hidden), nn.GELU(),
                                        nn.Linear(config.event_hidden, 3))

    def config(self):
        return asdict(self.cfg)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def predict(self, field, actions, *, loops=None):
        _field(field)
        actions = _actions(actions, len(field), field.device)
        depth = self.cfg.loops if loops is None else loops
        if type(depth) is not int or depth < 1:
            raise ValueError('loops must be a positive integer')
        source = field + self.position.to(field.dtype)[None] + self.action_embedding(actions).to(field.dtype)[:, None]
        state = source
        for _ in range(depth):
            state = self.block(state, source)
        return field + self.output(state)

    def forward(self, field, actions, *, loops=None):
        predicted = self.predict(field, actions, loops=loops)
        events = self.event_head(torch.cat((self.readout.summary(field), self.readout.summary(predicted),
                                           self.action_embedding(actions.long())), -1))
        return dict(field=predicted, readout=self.readout(predicted),
                    events=dict(lost_life_logits=events[:, 0], terminal_logits=events[:, 1], won_logits=events[:, 2]))

    def rollout(self, field, actions, *, loops=None):
        """Open-loop [B,K] action sequence; each next input is its own prediction."""
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
                    readout={k: torch.stack([x['readout'][k] for x in outputs], 1) for k in outputs[0]['readout']},
                    events={k: torch.stack([x['events'][k] for x in outputs], 1) for k in outputs[0]['events']})
