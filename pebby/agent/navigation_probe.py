"""Public-history navigation experiment, independent of production frozen envelopes.

Encoder comparisons share the entire initialized graph; only active encoder
parameters' gradient eligibility changes. Both modes keep the encoder in eval.
The direct control shares the spatial trunk and learned raw/state/glyph/player
inputs, but replaces decoded outcomes with a fresh summary comparator. It has a
different terminal-head capacity/initialization, so is a separate readout ablation.
Labels are accepted only by ``stage_loss``. No privileged coordinates enter a
policy call. ``outcomes`` is unweighted physical + value + events + policy;
``policy`` is the common objective for the readout comparison.
"""
from contextlib import contextmanager, nullcontext
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .neural_outcome_planner import neural_outcome_losses
from .neural_outcome_policy import ENCODER_RUNTIME as PARENT_RUNTIME, encoder_execution, weights_sha256
from .spatial_outcome_planner import SpatialOutcomePlanner
from .world_model import WorldModelConfig, WorldPolicy

FORMAT = 'pebby.ls20-navigation-probe.v1'
ENCODER_RUNTIME = {**PARENT_RUNTIME, 'mha_fastpath': False}


@contextmanager
def _same_attention_path():
    # PyTorch otherwise uses a distinct eval/no_grad self-attention kernel in
    # the frozen arm. Keep the differentiable kernel in both, restoring the
    # process setting afterward (this research runner is single-threaded).
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)

# encode() also computes an unused latent: its reduce/projector and all legacy
# policy/dynamics/grounding heads are deliberately excluded. cell_appearance is
# internally no_grad in WorldPolicy, even when the rest of the encoder is tuned.
_ENCODER_FEATURES = frozenset(('stem', 'hud', 'hud_projection', 'row_position',
    'column_position', 'hud_position', 'age_embedding', 'action_embedding',
    'temporal', 'source_norm', 'core', 'glyph_encoder', 'glyph_context',
    'cell_context', 'player_head'))
_DIRECT_UNUSED = frozenset(('field_heads', 'value_head', 'event_head',
                            'outcome_projection', 'comparator'))


class NavigationProbe(nn.Module):
    """Four primitive action scores, using the WorldPolicy public-history API."""

    checkpoint_format = FORMAT

    def __init__(self, encoder, planner, *, encoder_mode='frozen', readout='outcomes'):
        super().__init__()
        if encoder_mode not in ('frozen', 'finetune'):
            raise ValueError('encoder_mode must be frozen or finetune')
        if readout not in ('outcomes', 'direct'):
            raise ValueError('readout must be outcomes or direct')
        if not isinstance(encoder, WorldPolicy) or not isinstance(planner, SpatialOutcomePlanner):
            raise ValueError('expected WorldPolicy and SpatialOutcomePlanner')
        if encoder.cfg.channels != planner.cfg.channels or encoder.tokens != 160 or not encoder.cfg.glyph_recall:
            raise ValueError('encoder must supply matching channels, 160 tokens and glyph recall')
        if encoder.loops != encoder.cfg.loops:
            raise ValueError('encoder runtime loops must match its serialized config')
        self.encoder_mode, self.readout = encoder_mode, readout
        self.encoder, self.planner = encoder.float().eval(), planner.float()
        for name, parameter in self.encoder.named_parameters():
            active = name.split('.')[0] in _ENCODER_FEATURES and name != 'player_head.bias'
            parameter.requires_grad_(encoder_mode == 'finetune' and active)
        for name, parameter in self.planner.named_parameters():
            active = readout == 'outcomes' or name.split('.')[0] not in _DIRECT_UNUSED
            # This shared scalar bias cancels in player softmax and categorical CE.
            parameter.requires_grad_(active and name != 'player_head.bias')
        if readout == 'direct':
            hidden = planner.cfg.comparator_hidden
            self.direct_head = nn.Sequential(nn.Linear(2 * planner.cfg.summary, hidden),
                                              nn.GELU(), nn.Linear(hidden, 1, bias=False))

    @classmethod
    def from_parent(cls, policy, **options):
        """Copy a retained parent; never modify its production freeze contract."""
        return cls(copy.deepcopy(policy.encoder), copy.deepcopy(policy.planner), **options)

    def config(self):
        return {**self.encoder.config(), 'decision_architecture': 'navigation_probe',
                'encoder_mode': self.encoder_mode, 'readout': self.readout}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def public_features(self, frames, history_valid=None, previous_actions=None):
        """The same four learned tensors in both readout/gradient modes."""
        device = next(self.encoder.parameters()).device
        context = torch.no_grad() if self.encoder_mode == 'frozen' else nullcontext()
        with context, encoder_execution(device), _same_attention_path():
            encoded = self.encoder.encode(frames, history_valid, previous_actions)
            player = self.encoder.player_weights(encoded['cells'])[1]
        return encoded['raw'], encoded['state'], encoded['glyph'], player

    def _direct(self, raw, state, glyph, player):
        # Identical trunk computation to SpatialOutcomePlanner.forward, stopping
        # at its pre-outcome summary. This consumes no teacher/goal coordinates.
        planner = self.planner
        actions = planner._inputs(raw, state, glyph, player, None)
        batch, width = len(raw), planner.cfg.width
        cells = torch.cat((raw[:, :144], state[:, :144]), -1).transpose(1, 2).reshape(batch, -1, 12, 12)
        grid = planner.context_projection(torch.cat((cells, player.to(raw.dtype).reshape(batch, 1, 12, 12)), 1)
                                          .contiguous(memory_format=torch.channels_last))
        hud = planner.hud_projection(torch.cat((raw[:, 144:], state[:, 144:]), -1).flatten(1))
        shared = torch.cat((hud, glyph.to(hud.dtype)), -1)[:, None].expand(-1, 4, -1)
        condition = torch.cat((shared, actions.to(shared.dtype)), -1).reshape(batch * 4, -1)
        grid = grid[:, None].expand(-1, 4, -1, -1, -1).reshape(batch * 4, width, 12, 12)
        grid = grid.contiguous(memory_format=torch.channels_last)
        for block in planner.blocks:
            grid = block(grid, condition)
        grid = F.silu(planner.output_norm(grid))
        next_weights = planner.player_head(grid).reshape(batch, 4, 144).float().softmax(-1)
        cells = grid.reshape(batch, 4, width, 144)
        next_context = torch.einsum('bap,bacp->bac', next_weights.to(cells.dtype), cells)
        current_context = torch.einsum('bp,bacp->bac', player.to(cells.dtype), cells)
        summary = planner.summary_head(torch.cat((next_context, current_context, condition.reshape(batch, 4, -1)), -1))
        pooled = summary.mean(1, keepdim=True).expand_as(summary)
        return dict(action_logits=self.direct_head(torch.cat((summary, pooled), -1)).squeeze(-1))

    def predict(self, frames, history_valid=None, previous_actions=None):
        features = self.public_features(frames, history_valid, previous_actions)
        return self.planner(*features) if self.readout == 'outcomes' else self._direct(*features)

    def forward(self, frames, history_valid=None, previous_actions=None):
        return self.predict(frames, history_valid, previous_actions)['action_logits']

    def parameter_groups(self, encoder_lr, controller_lr):
        for value in (encoder_lr, controller_lr):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('learning rates must be finite and positive')
        groups = []
        for name, rate in (('encoder', encoder_lr), ('controller', controller_lr)):
            parameters = [p for key, p in self.named_parameters() if p.requires_grad
                          and key.startswith('encoder.') == (name == 'encoder')]
            if parameters:
                groups.append(dict(name=name, params=parameters, lr=rate))
        return groups

    def parameter_audit(self):
        """Counts/names report eligibility, not a claim that every gradient is nonzero."""
        trainable = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        frozen = {name: p.numel() for name, p in self.named_parameters() if not p.requires_grad}
        return dict(encoder_mode=self.encoder_mode, readout=self.readout,
                    trainable_parameters=sum(trainable.values()), frozen_parameters=sum(frozen.values()),
                    trainable=trainable, frozen=frozen,
                    cell_appearance_always_frozen=bool(self.encoder.cfg.cell_recall))


def stage_loss(predicted, items, *, objective='policy'):
    """Uniform CE over optimal actions; zero bitmasks carry no policy target."""
    if objective == 'outcomes':
        if not all(key in predicted for key in ('field_logits', 'value_logits', 'event_logits')):
            raise ValueError('outcomes objective requires the outcome readout')
        return neural_outcome_losses(predicted, items)
    if objective != 'policy':
        raise ValueError('objective must be policy or outcomes')
    logits = predicted['action_logits']
    if not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape[1] != 4 or not len(logits) or not logits.is_floating_point():
        raise ValueError('action_logits must be nonempty floating [B,4]')
    if 'optimal' not in items:
        raise ValueError('missing target optimal')
    optimal = torch.as_tensor(items['optimal'], device=logits.device)
    if optimal.shape != (len(logits),) or optimal.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('optimal must be an integer [B] tensor')
    if bool(((optimal < 0) | (optimal > 15)).any()):
        raise ValueError('optimal masks must be in 0..15')
    bits = (optimal.long()[:, None] & (1 << torch.arange(4, device=logits.device))) != 0
    valid = bits.any(-1)
    count = valid.sum()
    distribution = bits.float() / bits.sum(-1, keepdim=True).clamp_min(1)
    log_probs = logits.float().log_softmax(-1)
    loss = -(distribution * log_probs).sum() / count.clamp_min(1)
    with torch.no_grad():
        correct = bits.gather(1, logits.argmax(-1)[:, None]).squeeze(-1)
        diagnostics = dict(policy_valid_count=count, policy_valid_fraction=valid.float().mean(),
            set_accuracy=correct[valid].float().mean() if bool(count) else None,
            optimal_probability=(bits * log_probs.exp()).sum(-1)[valid].mean() if bool(count) else None)
    return dict(total=loss, losses=dict(policy=loss), diagnostics=diagnostics,
                diagnostic_weights=dict(policy=count, policy_valid_count=1,
                    policy_valid_fraction=len(logits), set_accuracy=count, optimal_probability=count))


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError('metadata must be a JSON object')
    try:
        normalized = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('metadata must contain finite JSON values') from exc
    return normalized


def make_checkpoint(model, metadata):
    """Standalone weights snapshot; no optimizer state or production lineage gate."""
    if not isinstance(model, NavigationProbe):
        raise ValueError('expected NavigationProbe')
    weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return dict(format=FORMAT, encoder_mode=model.encoder_mode, readout=model.readout,
                encoder_runtime=dict(ENCODER_RUNTIME), encoder_config=model.encoder.config(),
                planner_config=model.planner.config(), weights=weights,
                weights_sha256=weights_sha256(weights), metadata=_metadata(metadata))


def load_checkpoint(path, device='cpu'):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    required = {'format', 'encoder_mode', 'readout', 'encoder_runtime', 'encoder_config',
                'planner_config', 'weights', 'weights_sha256', 'metadata'}
    if not isinstance(data, dict) or set(data) != required or data.get('format') != FORMAT:
        raise ValueError('invalid navigation probe checkpoint schema/format')
    if data['encoder_runtime'] != ENCODER_RUNTIME:
        raise ValueError('navigation encoder runtime differs')
    if not isinstance(data['weights'], dict) or not data['weights'] or any(
            not isinstance(k, str) or not torch.is_tensor(v) for k, v in data['weights'].items()):
        raise ValueError('weights must be a nonempty tensor dictionary')
    if data['weights_sha256'] != weights_sha256(data['weights']):
        raise ValueError('navigation weights digest differs')
    metadata = _metadata(data['metadata'])
    try:
        encoder = WorldPolicy(WorldModelConfig.from_dict(data['encoder_config']))
        planner = SpatialOutcomePlanner(data['planner_config'])
        model = NavigationProbe(encoder, planner, encoder_mode=data['encoder_mode'], readout=data['readout'])
        model.load_state_dict(data['weights'], strict=True)
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('invalid navigation probe config/weights') from exc
    return model.to(device).eval(), metadata
