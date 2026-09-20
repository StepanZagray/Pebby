"""Spatial outcome policy v2: exact decoded HUD scalars, optionally tuned encoder.

Differences from ``spatial_outcome_policy`` (v1):

* The planner receives ``hud_decoder.decode_hud`` scalars of the current frame
  (exact remaining steps and lives) next to the learned HUD projection, so the
  budget is no longer known only to the encoder's pooled ~4-step granularity.
* ``encoder_mode='finetune'`` makes the feature-relevant encoder parameters
  trainable, following ``navigation_probe``: the same parameter eligibility,
  ``player_head.bias`` frozen, encoder norms kept in eval, and the same
  differentiable attention kernel in both modes so a tuned checkpoint scores
  identically under ``no_grad`` at inference.
* Checkpoints carry their own format, encoder mode and (when warm-started)
  the parent v1 path/digest; there is no direct/query score mixing.

The encoder always runs through ``world_features.encode_features`` (the
source-pinned feature adapter); ``world_model.py`` is not modified.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .hud_decoder import HUD_SCALARS, decode_hud
from .navigation_probe import ENCODER_RUNTIME, _ENCODER_FEATURES, _same_attention_path
from .neural_outcome_policy import encoder_execution, weights_sha256
from .spatial_outcome_planner import SpatialOutcomePlanner
from .spatial_outcome_policy import load_checkpoint as load_v1_checkpoint
from .world_features import encode_features
from .world_model import WorldModelConfig, WorldPolicy

FORMAT = 'pebby.ls20-spatial-outcome-policy.v2'
DECISION_ARCHITECTURE = 'spatial_outcomes_v2'
ENCODER_MODES = ('frozen', 'finetune')
# Linear layers whose input columns widen by ``hud_scalars`` between v1 and v2,
# and the number of trailing v1 columns (action one-hot) that shift right.
_WIDENED = {'blocks.0.condition.weight': 4, 'blocks.1.condition.weight': 4,
            'blocks.2.condition.weight': 4, 'summary_head.0.weight': 4}


class SpatialOutcomePolicyV2(nn.Module):
    """Public-history action scores from decoded outcomes plus exact HUD scalars."""

    checkpoint_format = FORMAT

    def __init__(self, encoder, planner, *, encoder_mode='frozen'):
        super().__init__()
        if encoder_mode not in ENCODER_MODES:
            raise ValueError('encoder_mode must be frozen or finetune')
        if not isinstance(encoder, WorldPolicy) or not isinstance(planner, SpatialOutcomePlanner):
            raise ValueError('expected WorldPolicy and SpatialOutcomePlanner')
        if encoder.cfg.channels != planner.cfg.channels or encoder.tokens != 160 or not encoder.cfg.glyph_recall:
            raise ValueError('encoder must supply matching channels, 160 tokens and glyph recall')
        if encoder.loops != encoder.cfg.loops:
            raise ValueError('encoder runtime loops must match its serialized config')
        if planner.cfg.hud_scalars not in (0, HUD_SCALARS):
            raise ValueError(f'planner hud_scalars must be 0 or {HUD_SCALARS}')
        self.encoder_mode = encoder_mode
        self.encoder, self.planner = encoder.float().eval(), planner.float()
        for name, parameter in self.encoder.named_parameters():
            active = name.split('.')[0] in _ENCODER_FEATURES and name != 'player_head.bias'
            parameter.requires_grad_(encoder_mode == 'finetune' and active)
        for name, parameter in self.planner.named_parameters():
            # This shared scalar bias cancels in the player softmax and its CE.
            parameter.requires_grad_(name != 'player_head.bias')
        self.provenance = {}

    @property
    def hud_scalars(self):
        return self.planner.cfg.hud_scalars

    def config(self):
        return {**self.encoder.config(), 'decision_architecture': DECISION_ARCHITECTURE,
                'encoder_mode': self.encoder_mode, 'hud_scalars': self.hud_scalars}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Norm statistics never update; gradients still flow when tuning.
        return self

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def public_features(self, frames, history_valid=None, previous_actions=None):
        device = next(self.encoder.parameters()).device
        context = torch.no_grad() if self.encoder_mode == 'frozen' else nullcontext()
        with context, encoder_execution(device), _same_attention_path():
            encoded = encode_features(self.encoder, frames, history_valid, previous_actions)
            player = self.encoder.player_weights(encoded['cells'])[1]
        return encoded['raw'], encoded['state'], encoded['glyph'], player

    def predict(self, frames, history_valid=None, previous_actions=None):
        """Planner output dict: action_logits, field_logits, value_logits, event_logits."""
        raw, state, glyph, player = self.public_features(frames, history_valid, previous_actions)
        scalars = None
        if self.hud_scalars:
            frames = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
            scalars = decode_hud(frames).to(raw.device)
        return self.planner(raw, state, glyph, player, hud_scalars=scalars)

    def forward(self, frames, history_valid=None, previous_actions=None):
        return self.predict(frames, history_valid, previous_actions)['action_logits']

    def parameter_groups(self, encoder_lr, controller_lr):
        groups = []
        for name, rate in (('encoder', encoder_lr), ('planner', controller_lr)):
            parameters = [p for key, p in self.named_parameters() if p.requires_grad
                          and key.startswith('encoder.') == (name == 'encoder')]
            if parameters:
                groups.append(dict(name=name, params=parameters, lr=rate))
        return groups


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def warm_start_planner(source, hud_scalars=HUD_SCALARS, direct_readout=False):
    """A planner with ``hud_scalars`` whose v1-shaped weights come from ``source``.

    Tensors with identical names and shapes are copied. The four condition
    inputs that widen (three block projections and the summary input) get
    the v1 columns at their original positions, with the new scalar columns
    zero-initialised, so the warm start reproduces the parent's outputs
    exactly until the scalar columns learn. Anything else stays freshly
    initialised and is listed in the returned report.
    """
    if not isinstance(source, SpatialOutcomePlanner) or source.cfg.hud_scalars or source.cfg.direct_readout:
        raise ValueError('warm start expects a v1 planner without hud_scalars or direct_readout')
    planner = SpatialOutcomePlanner({**source.config(), 'hud_scalars': hud_scalars,
                                     'direct_readout': direct_readout})
    old, new = source.state_dict(), planner.state_dict()
    copied, widened, fresh = [], [], []
    with torch.no_grad():
        for name, value in new.items():
            if name not in old:
                fresh.append(name)
            elif old[name].shape == value.shape:
                value.copy_(old[name])
                copied.append(name)
            elif name in _WIDENED and value.shape[0] == old[name].shape[0]:
                tail = _WIDENED[name]
                keep = old[name].shape[1] - tail
                value.zero_()
                value[:, :keep] = old[name][:, :keep]
                value[:, keep + hud_scalars:] = old[name][:, keep:]
                widened.append(name)
            else:
                fresh.append(name)
    planner.load_state_dict(new, strict=True)
    return planner, dict(copied=copied, widened=widened, fresh=fresh)


def from_v1_checkpoint(path, *, encoder_mode='frozen', hud_scalars=HUD_SCALARS, direct_readout=False):
    """New v2 policy warm-started from a strict v1 checkpoint; ``policy.provenance`` holds the report."""
    path = Path(path)
    parent, _ = load_v1_checkpoint(path, 'cpu')
    if hud_scalars or direct_readout:
        planner, report = warm_start_planner(parent.planner, hud_scalars, direct_readout)
    else:
        planner, report = parent.planner, dict(copied=list(parent.planner.state_dict()), widened=[], fresh=[])
    policy = SpatialOutcomePolicyV2(parent.encoder, planner, encoder_mode=encoder_mode)
    policy.provenance = dict(parent_path=str(path), parent_sha256=_file_sha256(path), **report)
    return policy


def _metadata(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('metadata must contain finite JSON values') from exc


def make_checkpoint(policy, **metadata):
    if not isinstance(policy, SpatialOutcomePolicyV2):
        raise ValueError('expected SpatialOutcomePolicyV2')
    encoder_weights = {k: v.detach().cpu().clone() for k, v in policy.encoder.state_dict().items()}
    planner_weights = {k: v.detach().cpu().clone() for k, v in policy.planner.state_dict().items()}
    provenance = policy.provenance or {}
    return dict(format=FORMAT, decision_architecture=DECISION_ARCHITECTURE,
                encoder_mode=policy.encoder_mode, hud_scalars=policy.hud_scalars,
                encoder_runtime=dict(ENCODER_RUNTIME),
                encoder_config=policy.encoder.config(), encoder_weights=encoder_weights,
                encoder_weights_sha256=weights_sha256(encoder_weights),
                planner_config=policy.planner.config(), planner_weights=planner_weights,
                planner_weights_sha256=weights_sha256(planner_weights),
                parent_path=provenance.get('parent_path'), parent_sha256=provenance.get('parent_sha256'),
                metadata=_metadata(metadata))


def save_checkpoint(path, policy, **metadata):
    data = make_checkpoint(policy, **metadata)
    torch.save(data, Path(path))
    return data


def load_checkpoint(path, device='cpu'):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        raise ValueError('not a spatial outcome policy v2 checkpoint')
    required = ('encoder_mode', 'hud_scalars', 'encoder_runtime', 'encoder_config', 'encoder_weights',
                'encoder_weights_sha256', 'planner_config', 'planner_weights', 'planner_weights_sha256')
    if any(key not in data for key in required):
        raise ValueError('incomplete spatial outcome policy v2 checkpoint')
    if data['encoder_runtime'] != ENCODER_RUNTIME:
        raise ValueError('encoder runtime differs')
    for name in ('encoder', 'planner'):
        if data[f'{name}_weights_sha256'] != weights_sha256(data[f'{name}_weights']):
            raise ValueError(f'embedded {name} digest differs')
    if data['hud_scalars'] != data['planner_config'].get('hud_scalars', 0):
        raise ValueError('hud_scalars disagrees with the planner config')
    encoder = WorldPolicy(WorldModelConfig.from_dict(data['encoder_config']))
    encoder.load_state_dict(data['encoder_weights'], strict=True)
    planner = SpatialOutcomePlanner(data['planner_config'])
    planner.load_state_dict(data['planner_weights'], strict=True)
    policy = SpatialOutcomePolicyV2(encoder, planner, encoder_mode=data['encoder_mode'])
    policy.provenance = {k: data.get(k) for k in ('parent_path', 'parent_sha256') if data.get(k)}
    return policy.to(device).eval(), data
