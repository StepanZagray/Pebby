"""Public-history policy adapter for a separately trained one-step outcome planner.

The original regenerated-data encoder is frozen. The planner is initialized and
trained separately; no old ranker contributes to its scores. Reset remains the
competition runner's explicit GAME_OVER adapter, not a learned fifth action.
"""

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .world_model import WorldPolicy, WorldModelConfig
from .neural_outcome_planner import NeuralOutcomePlanner

FORMAT = 'pebby.ls20-neural-outcome-policy.v1'
PARENT_SHA = '6db7f40d4008ff809e9b3a8b05d6a9a18801b343f02e52a9f585eece5e31cff9'
ENCODER_RUNTIME = dict(precision='float32', temporal_backend='auto', matmul_tf32=False, cudnn_tf32=True)


def weights_sha256(weights):
    digest = hashlib.sha256()
    for name, value in weights.items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def encoder_execution(device):
    previous = None
    if device.type == 'cuda':
        previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    try:
        with torch.autocast(device.type, enabled=False):
            yield
    finally:
        if previous is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous



class NeuralOutcomePolicy(nn.Module):
    def __init__(self, encoder, planner, *, planner_weight=1., direct_weight=0.):
        super().__init__()
        if any(not math.isfinite(w) or w < 0 for w in (planner_weight, direct_weight)):
            raise ValueError('score weights must be finite and nonnegative')
        if planner_weight + direct_weight == 0:
            raise ValueError('at least one score weight must be positive')
        self.encoder = encoder.float().eval().requires_grad_(False)
        self.planner = planner
        self.planner_weight, self.direct_weight = planner_weight, direct_weight

    def config(self):
        # The legacy public-history wrapper recognizes architecture=world.
        # The checkpoint FORMAT, not this history compatibility field, selects
        # this new architecture at load time.
        return {**self.encoder.config(), 'decision_architecture': 'neural_outcomes',
                'planner_horizon': 1, 'planner_refinement_loops': 1}

    def forward(self, frames, history_valid=None, previous_actions=None):
        device = next(self.encoder.parameters()).device
        with torch.no_grad(), encoder_execution(device):
            encoding = self.encoder.encode(frames, history_valid, previous_actions)
            direct = None
            if self.direct_weight:
                direct, player = self.encoder.direct_logits(encoding['cells'])
                if self.encoder.cfg.query_readout:
                    direct = direct + self.encoder.query_logits(encoding, player.softmax(-1))
        predicted = self.planner(encoding['raw'], encoding['state'], encoding['glyph'])
        scores = self.planner_weight * predicted['action_logits']
        if direct is not None:
            scores = scores + self.direct_weight * direct

        return scores

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self


def load_checkpoint(path, device='cpu', *, direct_weight=None, planner_weight=None):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if data.get('format') != FORMAT:
        raise ValueError('not a neural outcome policy checkpoint')
    if data.get('encoder_parent_sha256') != PARENT_SHA or data.get('encoder_frozen') is not True:
        raise ValueError('protected fresh frozen encoder lineage required')
    if data.get('encoder_runtime') != ENCODER_RUNTIME:
        raise ValueError('encoder runtime differs from trained feature cache')
    if data.get('encoder_weights_sha256') != weights_sha256(data['encoder_weights']):
        raise ValueError('serialized encoder weights hash differs')
    encoder = WorldPolicy(WorldModelConfig.from_dict(data['encoder_config']))
    encoder.load_state_dict(data['encoder_weights'], strict=True)
    planner = NeuralOutcomePlanner(data['planner_config'])
    planner.load_state_dict(data['planner_weights'], strict=True)
    weights = data['score_weights']
    policy = NeuralOutcomePolicy(encoder, planner,
        direct_weight=weights['direct'] if direct_weight is None else direct_weight,
        planner_weight=weights['planner'] if planner_weight is None else planner_weight)
    return policy.to(device).eval(), data
