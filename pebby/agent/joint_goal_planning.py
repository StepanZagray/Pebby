"""Fresh public-pixel perception, recurrent dynamics and neural goal planning.

No checkpoint is loaded during construction. Every registered parameter is
trainable; patch decoders are training-only. Hard continuation choices need
their own supervised loss. H8 memory ends at caller-declared history boundaries.
Optional learned terminal gating discounts later GRU evidence; it does not
truncate dynamics or certify future trajectories.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from .cell_appearance import CellAppearance, cell_patches
from .glyph_model import GlyphEncoder, crop_glyph
from .structured_physical_feedback import PhysicalFeedbackTransition
from .structured_policy import StructuredPolicyReadout, _fields
from .world_model import RefineBlock, TemporalMemory


FORMAT = 'pebby.joint-goal-planning.v1'
_RUNTIME_CONFIG = dict(architecture='structured', history=8,
                       decision_architecture='joint_goal_planning', field_layout_version=1)


@dataclass(frozen=True)
class JointGoalConfig:
    horizon: int = 4
    hidden: int = 96
    encoder_loops: int = 2
    dynamics_loops: int = 2
    relation_feedback: bool = True
    terminal_gating: bool = False

    def __post_init__(self):
        for key in ('horizon', 'hidden', 'encoder_loops', 'dynamics_loops'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.horizon > 16:
            raise ValueError('horizon must be in1..16')
        if type(self.relation_feedback) is not bool:
            raise ValueError('relation_feedback must be boolean')
        if type(self.terminal_gating) is not bool:
            raise ValueError('terminal_gating must be boolean')


def _config(config):
    if config is None:
        return JointGoalConfig()
    if isinstance(config, dict):
        config = dict(config)
        for key, expected in _RUNTIME_CONFIG.items():
            if key in config and config.pop(key) != expected:
                raise ValueError(f'{key} must be {expected!r}')
        return JointGoalConfig(**config)
    if not isinstance(config, JointGoalConfig):
        raise ValueError('JointGoalConfig or a configuration dictionary required')
    return config


def hud_patches(frames):
    """Public HUD strips [...,4,12,16], left to right; no engine state."""
    if tuple(frames.shape[-2:]) != (64, 64):
        raise ValueError('frames must end in64x64')
    strips = frames[..., 52:64, :].reshape(*frames.shape[:-2], 12, 4, 16)
    return strips.transpose(-3, -2)


class _PublicPlanningEncoder(nn.Module):
    def __init__(self, loops):
        super().__init__()
        self.loops = loops
        self.appearance = CellAppearance()
        self.cell_projection = nn.Linear(64, 48)
        self.hud = nn.Sequential(nn.Linear(3072, 64), nn.GELU(), nn.Linear(64, 48))
        self.position = nn.Parameter(torch.randn(148, 48) * .02)
        self.age = nn.Parameter(torch.randn(8, 48) * .02)
        self.previous_action = nn.Embedding(5, 48)
        self.temporal = TemporalMemory(48, 4, 2)
        self.refine = RefineBlock(48, 4, 2)
        self.norm = nn.LayerNorm(48)
        self.glyph = GlyphEncoder()
        self.visibility = nn.Linear(48, 1)

    def inputs(self, frames, history_valid, previous_actions):
        device = self.position.device
        frames = torch.as_tensor(frames, device=device)
        if (frames.ndim != 4 or tuple(frames.shape[1:]) != (8, 64, 64) or not len(frames)
                or frames.dtype.is_floating_point or frames.dtype == torch.bool
                or bool(((frames < 0) | (frames > 15)).any())):
            raise ValueError('frames must be integer palette indices [B,8,64,64] in0..15')
        batch = len(frames)
        valid = (torch.ones(batch, 8, dtype=torch.bool, device=device) if history_valid is None
                 else torch.as_tensor(history_valid, device=device))
        actions = (torch.full((batch, 8), -1, dtype=torch.long, device=device) if previous_actions is None
                   else torch.as_tensor(previous_actions, device=device))
        if (valid.shape != (batch, 8) or valid.dtype != torch.bool or not bool(valid[:, -1].all())
                or bool((valid[:, :-1] & ~valid[:, 1:]).any())):
            raise ValueError('history_valid must be bool [B,8], a nonempty left-padded suffix')
        if (actions.shape != (batch, 8) or actions.dtype.is_floating_point or actions.dtype == torch.bool
                or bool(((actions < -1) | (actions > 3)).any())
                or bool((actions[~valid] != -1).any())):
            raise ValueError('previous_actions must be integer [B,8] in-1..3 with -1 padding')
        return frames.long(), valid, actions.long()

    def forward(self, frames, history_valid=None, previous_actions=None):
        frames, valid, actions = self.inputs(frames, history_valid, previous_actions)
        batch = len(frames)
        flat = frames.flatten(0, 1)
        pixels = F.one_hot(cell_patches(flat).long(), 16).flatten(-3).to(self.position.dtype)
        hidden = pixels
        for layer in list(self.appearance.network.children())[:-1]:
            hidden = layer(hidden)
        # The same learned patch features serve typed perception and spatial state.
        semantic = self.appearance.network[-1](hidden.reshape(batch, 8, 144, 64)[:, -1])
        role, shape, color, rotation = semantic.split((8, 6, 4, 4), -1)
        board = self.cell_projection(hidden)
        hud_pixels = F.one_hot(hud_patches(flat).long(), 16).flatten(-3).to(self.position.dtype)
        tokens = torch.cat((board, self.hud(hud_pixels)), 1).reshape(batch, 8, 148, 48)
        tokens = (tokens + self.position[None, None] + self.age[None, :, None]
                  + self.previous_action(actions + 1)[:, :, None])
        source = self.temporal(tokens[:, -1], tokens, valid)
        core = source
        for _ in range(self.loops):
            core = self.refine(core, source)
        core = self.norm(core)
        carried = self.glyph(crop_glyph(frames[:, -1])).split((6, 4, 4), -1)
        appearance = torch.cat((role.sigmoid(), shape.softmax(-1), color.softmax(-1), rotation.softmax(-1)), -1)
        appearance = torch.cat((appearance, core.new_zeros(batch, 4, 22)), 1)
        glyph = torch.cat([scores.softmax(-1) for scores in carried], -1)[:, None].expand(-1, 148, -1)
        visible_logits = self.visibility(core[:, :144]).squeeze(-1)
        visible = torch.cat((visible_logits.sigmoid(), core.new_ones(batch, 4)), 1)
        field = torch.cat((core, appearance, glyph, visible[..., None], core.new_zeros(batch, 148, 11)), -1)
        return dict(field=field, role_logits=role, goal_shape_logits=shape,
                    goal_color_logits=color, goal_rotation_logits=rotation,
                    carried_shape_logits=carried[0], carried_color_logits=carried[1],
                    carried_rotation_logits=carried[2], visibility_logits=visible_logits)


class _PixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.board = nn.Sequential(nn.Linear(48, 128), nn.GELU(), nn.Linear(128, 784))
        self.hud = nn.Sequential(nn.Linear(48, 128), nn.GELU(), nn.Linear(128, 3072))

    def forward(self, fields):
        return dict(board=self.board(fields[:, :144, :48]).reshape(-1, 144, 7, 7, 16),
                    hud=self.hud(fields[:, 144:, :48]).reshape(-1, 4, 12, 16, 16))


class JointGoalPlanning(nn.Module):
    """Fully trainable public-observation model; auxiliary targets are external.

    ``relation_feedback=False`` constructs identical parameters but zeros only
    the relation residual into policy/value summaries. Relation supervision is
    still available. The caller supplies all losses and owns training budgets.
    """
    checkpoint_format = FORMAT

    def __init__(self, config=None):
        super().__init__()
        self.cfg = _config(config)
        self.encoder = _PublicPlanningEncoder(self.cfg.encoder_loops)
        self.dynamics = PhysicalFeedbackTransition(loops=self.cfg.dynamics_loops)
        self.continuation = StructuredPolicyReadout()
        # A shared scalar bias cancels from four-action CE. Avoid that dead parameter.
        # A fresh nonzero weight lets continuation CE reach its encoder immediately.
        self.continuation.scorer = nn.Linear(96, 1, bias=False)
        self.relation_mlp = nn.Sequential(nn.Linear(133, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.compatibility_head = nn.Linear(64, 1)
        self.solved_head = nn.Linear(64, 1)
        self.relation_projection = nn.Linear(66, 96, bias=False)
        self.summary_position = nn.Parameter(torch.randn(148, 96) * .02)
        self.summary_query = nn.Parameter(torch.randn(1, 1, 96) * .02)
        self.summary_norm = nn.LayerNorm(96)
        self.summary_attention = nn.MultiheadAttention(96, 4, dropout=0., batch_first=True)
        self.value_head = nn.Linear(96, 130)
        self.trajectory = nn.GRUCell(229, self.cfg.hidden)
        self.scorer = nn.Sequential(nn.Linear(2 * self.cfg.hidden + 96, self.cfg.hidden),
                                    nn.GELU(), nn.Linear(self.cfg.hidden, 1, bias=False))
        self.pixel_decoder = _PixelDecoder()

    def config(self):
        return {**asdict(self.cfg), **_RUNTIME_CONFIG}

    def parameter_counts(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        decoder = sum(parameter.numel() for parameter in self.pixel_decoder.parameters())
        return dict(total=total, trainable=sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
                    decoder_only=decoder, deployed=total - decoder)

    def parameter_count(self):
        return self.parameter_counts()['total']

    def encode_details(self, frames, history_valid=None, previous_actions=None):
        return self.encoder(frames, history_valid, previous_actions)

    def encode(self, frames, history_valid=None, previous_actions=None):
        return self.encode_details(frames, history_valid, previous_actions)['field']

    def readout(self, field):
        _fields(field)
        return self.dynamics.readout(field)

    def visibility_logits(self, field):
        _fields(field)
        return self.encoder.visibility(field[:, :144, :48]).squeeze(-1)

    def relation(self, field):
        physical = self.readout(field)
        batch = len(field)
        evidence = torch.cat((field[:, :144, :84], physical['player_logits'].softmax(-1)[..., None],
                              physical['steps_logits'].softmax(-1)[:, None].expand(-1, 144, -1),
                              physical['lives_logits'].softmax(-1)[:, None].expand(-1, 144, -1)), -1)
        hidden = self.relation_mlp(evidence)
        return dict(hidden=hidden, compatibility_logits=self.compatibility_head(hidden).reshape(batch, 144),
                    solved_logits=self.solved_head(hidden).reshape(batch, 144))

    def conditioned_field(self, field):
        relation = self.relation(field)
        features = torch.cat((relation['hidden'], relation['compatibility_logits'].sigmoid()[..., None],
                              relation['solved_logits'].sigmoid()[..., None]), -1)
        residual = self.relation_projection(features) * float(self.cfg.relation_feedback)
        return field + torch.cat((residual, field.new_zeros(len(field), 4, 96)), 1)

    def pixel_logits(self, field):
        _fields(field)
        return self.pixel_decoder(field)

    def _summary(self, field):
        memory = self.summary_norm(self.conditioned_field(field) + self.summary_position[None])
        return self.summary_attention(self.summary_query.expand(len(field), -1, -1), memory, memory,
                                      need_weights=False)[0][:, 0]

    def value_logits(self, field):
        return self.value_head(self._summary(field))

    def continuation_logits(self, field):
        scores = self.continuation(self.conditioned_field(field))
        if scores.shape != (len(field), 4) or not bool(torch.isfinite(scores).all()):
            raise ValueError('continuation must return finite [B,4] logits')
        return scores

    def imagine(self, field, *, root_actions=None, horizon=None, tail_ablation=False):
        _fields(field)
        depth = self.cfg.horizon if horizon is None else horizon
        if type(depth) is not int or not 1 <= depth <= 16 or type(tail_ablation) is not bool:
            raise ValueError('horizon must be in1..16 and tail_ablation must be boolean')
        batch = len(field)
        expected = torch.arange(4, device=field.device)[None].expand(batch, -1)
        roots = expected if root_actions is None else root_actions
        if (not isinstance(roots, torch.Tensor) or roots.shape != (batch, 4) or roots.dtype != torch.long
                or roots.device != field.device or not torch.equal(roots.sort(-1).values, expected)):
            raise ValueError('root_actions must be long [B,4] permutations of0..3')
        current = field[:, None].expand(-1, 4, -1, -1).reshape(batch * 4, 148, 96)
        actions = roots.reshape(-1)
        inputs, traces, predicted_fields, event_traces, values = [], [], [], [], []
        for step in range(depth):
            if step:
                with torch.no_grad():
                    actions = self.continuation_logits(current.detach()).argmax(-1)
            prediction = self.dynamics(current, actions)
            current = prediction['field']
            summary = self._summary(current)
            value = self.value_head(summary)
            events = torch.stack([prediction['events'][name + '_logits']
                                  for name in ('lost_life', 'terminal', 'won')], -1)
            inputs.append(torch.cat((summary, events.sigmoid(), value.softmax(-1)), -1))
            traces.append(actions.reshape(batch, 4))
            predicted_fields.append(current.reshape(batch, 4, 148, 96))
            event_traces.append(events.reshape(batch, 4, 3))
            values.append(value.reshape(batch, 4, 130))
        evidence = [inputs[0]] * depth if tail_ablation else inputs
        hidden = field.new_zeros(batch * 4, self.cfg.hidden)
        for item in reversed(evidence):
            if self.cfg.terminal_gating:
                # Input = summary96 + (lost_life, terminal, won) probabilities
                # + value130. Gate only later evidence; retain this transition
                # and its terminal/win evidence. A life loss alone does not stop.
                hidden = hidden * (1 - item[:, 97:98])
            hidden = self.trajectory(item, hidden)
        branches = hidden.reshape(batch, 4, -1)
        pooled = branches.mean(1, keepdim=True).expand_as(branches)
        observed = self._summary(field)[:, None].expand(-1, 4, -1)
        scores = self.scorer(torch.cat((branches, pooled, observed), -1)).squeeze(-1)
        return dict(action_logits=torch.zeros_like(scores).scatter(1, roots, scores),
                    root_scores=scores, root_actions=roots, imagined_actions=torch.stack(traces, 2),
                    imagined_fields=torch.stack(predicted_fields, 2),
                    imagined_event_logits=torch.stack(event_traces, 2),
                    imagined_value_logits=torch.stack(values, 2),
                    horizon=depth, transition_count_per_root=depth, transition_count_per_decision=4 * depth)

    def forward(self, frames, history_valid=None, previous_actions=None):
        return self.imagine(self.encode(frames, history_valid, previous_actions))['action_logits']


def save_checkpoint(path, model, metadata=None, **extra):
    if not isinstance(model, JointGoalPlanning):
        raise ValueError('JointGoalPlanning model required')
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError('checkpoint metadata must be a dictionary')
    metadata = dict(metadata or {})
    if metadata.keys() & extra.keys():
        raise ValueError('duplicate checkpoint metadata keys')
    metadata.update(extra)
    reserved = {'format', 'config', 'weights', 'parameter_counts'} & metadata.keys()
    if reserved:
        raise ValueError(f'reserved checkpoint metadata: {sorted(reserved)}')
    payload = dict(format=FORMAT, config=model.config(), parameter_counts=model.parameter_counts(),
                   weights={key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, **metadata)
    json.dumps({key: value for key, value in payload.items() if key != 'weights'}, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.joint-goal-', suffix='.pt', delete=False) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError('unsupported joint goal checkpoint format')
    with torch.random.fork_rng(devices=[]):
        model = JointGoalPlanning(payload['config'])
    model.load_state_dict(payload['weights'], strict=True)
    if payload.get('parameter_counts') != model.parameter_counts():
        raise ValueError('checkpoint parameter accounting differs')
    return model.to(device).eval(), payload
