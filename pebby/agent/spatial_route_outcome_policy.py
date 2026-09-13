"""Public-history adapter and strict versioned checkpoint loader for route values."""
import copy
from pathlib import Path

import torch

from .neural_outcome_policy import ENCODER_RUNTIME, PARENT_SHA, weights_sha256
from .spatial_outcome_policy import FORMAT as PARENT_FORMAT, SpatialOutcomePolicy
from .spatial_route_outcome_planner import SpatialRouteOutcomePlanner
from .world_model import WorldModelConfig, WorldPolicy

FORMAT = 'pebby.ls20-spatial-route-outcome-policy.v1'


class SpatialRouteOutcomePolicy(SpatialOutcomePolicy):
    def config(self):
        return {**super().config(), 'decision_architecture': 'spatial_route_outcomes', 'route_version': 1}


def checkpoint_from_parent(parent: dict, planner: SpatialRouteOutcomePlanner,
                           *, parent_checkpoint_sha256: str) -> dict:
    """Build the route inference envelope while preserving the encoder lineage.

    This copies planner tensors and records the source checkpoint. The caller
    must write truthful latest-stage training metadata after optimization; this
    helper does not claim a training run or a gameplay improvement.
    """
    if parent.get('format') != PARENT_FORMAT or not isinstance(planner, SpatialRouteOutcomePlanner):
        raise ValueError('route checkpoint migration requires an original spatial parent and route planner')
    if len(parent_checkpoint_sha256) != 64 or any(char not in '0123456789abcdef' for char in parent_checkpoint_sha256):
        raise ValueError('parent checkpoint SHA256 must be lowercase hexadecimal')
    result = copy.copy(parent)
    result.update(format=FORMAT, planner_config=planner.config(),
                  planner_weights={name: value.detach().cpu().clone() for name, value in planner.state_dict().items()},
                  source_checkpoint_sha256=parent_checkpoint_sha256,
                  route_architecture=dict(version=1, source_format=PARENT_FORMAT,
                                          parent_planner_weights_sha256=weights_sha256(parent['planner_weights']),
                                          memory='all 144 public raw/refined/action-conditioned scene cells',
                                          residual_target='130 value logits', parameter_prefix='route_readout.',
                                          explicit_search=False, privileged_inference_inputs=False))
    return result


def load_checkpoint(path, device='cpu', *, direct_weight=None, planner_weight=None):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (data.get('format') != FORMAT or data.get('encoder_parent_sha256') != PARENT_SHA
            or data.get('encoder_frozen') is not True or data.get('official_training_inputs') is not False):
        raise ValueError('generated-only route outcome policy with frozen fresh encoder required')
    if data.get('privileged_inference_inputs', False) is not False:
        raise ValueError('route policy must use public inference inputs only')
    if data.get('encoder_runtime') != ENCODER_RUNTIME:
        raise ValueError('encoder runtime differs from feature cache')
    if data.get('encoder_weights_sha256') != weights_sha256(data['encoder_weights']):
        raise ValueError('embedded encoder digest differs')
    encoder = WorldPolicy(WorldModelConfig.from_dict(data['encoder_config']))
    encoder.load_state_dict(data['encoder_weights'], strict=True)
    planner = SpatialRouteOutcomePlanner(data['planner_config'])
    planner.load_state_dict(data['planner_weights'], strict=True)
    weights = data['score_weights']
    policy = SpatialRouteOutcomePolicy(encoder, planner,
        direct_weight=weights['direct'] if direct_weight is None else direct_weight,
        planner_weight=weights['planner'] if planner_weight is None else planner_weight)
    return policy.to(device).eval(), data
