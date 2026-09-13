"""Public-history adapter for spatial neural outcomes and their neural comparator."""
from pathlib import Path
import torch

from .world_model import WorldPolicy, WorldModelConfig
from .neural_outcome_policy import NeuralOutcomePolicy, PARENT_SHA, ENCODER_RUNTIME, encoder_execution, weights_sha256
from .spatial_outcome_planner import SpatialOutcomePlanner

FORMAT = 'pebby.ls20-spatial-outcome-policy.v1'


class SpatialOutcomePolicy(NeuralOutcomePolicy):
    def config(self):
        return {**super().config(), 'decision_architecture': 'spatial_outcomes'}

    def forward(self, frames, history_valid=None, previous_actions=None):
        device = next(self.encoder.parameters()).device
        with torch.no_grad(), encoder_execution(device):
            encoding = self.encoder.encode(frames, history_valid, previous_actions)
            player = self.encoder.player_weights(encoding['cells'])[1]
            direct = None
            if self.direct_weight:
                direct = self.encoder.direct_logits(encoding['cells'])[0]
                if self.encoder.cfg.query_readout:
                    direct = direct + self.encoder.query_logits(encoding, player)
        result = self.planner(encoding['raw'], encoding['state'], encoding['glyph'], player)
        scores = self.planner_weight * result['action_logits']
        return scores if direct is None else scores + self.direct_weight * direct


def load_checkpoint(path, device='cpu', *, direct_weight=None, planner_weight=None):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (data.get('format') != FORMAT or data.get('encoder_parent_sha256') != PARENT_SHA
            or data.get('encoder_frozen') is not True or data.get('official_training_inputs') is not False):
        raise ValueError('generated-only spatial outcome policy with frozen fresh encoder required')
    if data.get('encoder_runtime') != ENCODER_RUNTIME:
        raise ValueError('encoder runtime differs from feature cache')
    if data.get('encoder_weights_sha256') != weights_sha256(data['encoder_weights']):
        raise ValueError('embedded encoder digest differs')
    encoder = WorldPolicy(WorldModelConfig.from_dict(data['encoder_config']))
    encoder.load_state_dict(data['encoder_weights'], strict=True)
    planner = SpatialOutcomePlanner(data['planner_config'])
    planner.load_state_dict(data['planner_weights'], strict=True)
    weights = data['score_weights']
    policy = SpatialOutcomePolicy(encoder, planner,
        direct_weight=weights['direct'] if direct_weight is None else direct_weight,
        planner_weight=weights['planner'] if planner_weight is None else planner_weight)
    return policy.to(device).eval(), data
