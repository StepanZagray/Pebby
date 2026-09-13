"""Versioned public-history policy with an embedded frozen pixel perceptor."""
import copy
import hashlib
import io
from pathlib import Path

import torch

from .cell_appearance import CellAppearance, FORMAT as PERCEPTOR_FORMAT, ROLE_NAMES, ATTRIBUTE_SIZES
from .neural_outcome_policy import ENCODER_RUNTIME, PARENT_SHA, encoder_execution, weights_sha256
from .spatial_outcome_policy import FORMAT as PARENT_FORMAT, SpatialOutcomePolicy
from .spatial_semantic_outcome_planner import SpatialSemanticOutcomePlanner
from .world_model import WorldModelConfig, WorldPolicy

FORMAT = 'pebby.ls20-spatial-semantic-outcome-policy.v1'
TEACHER_SHA256 = '143e0cdde09889ed11e938a004e77b5ed968e93f0cd8353be74f82bf5370082b'
TEACHER_WEIGHTS_SHA256 = '46d6edce2c45f48557a16e284a73e2de4dff78e508dc74c106b6f05fbf0f1302'
RECOVERY_SHA256 = 'ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8'
PERCEPTOR_CONFIG = dict(architecture='cell_appearance', patch_size=7, cells=144,
                        role_names=list(ROLE_NAMES), attribute_sizes=list(ATTRIBUTE_SIZES),
                        probabilities='independent role sigmoid; separate attribute softmax',
                        frame='current public image only')


def _architecture(planner):
    return dict(version=1, actor=planner.cfg.actor, semantic_channels=22,
                current_scene_attention=True, temporal_rollout=False, hidden_scene_reconstruction=False,
                privileged_inference_inputs=False)


def load_perceptor(path: str | Path) -> CellAppearance:
    """Import only the exact generated-TRAIN teacher, independently of old loaders."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != TEACHER_SHA256:
        raise ValueError('perceptor checkpoint differs from the approved fresh teacher')
    data = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if data.get('format') != PERCEPTOR_FORMAT or weights_sha256(data['weights']) != TEACHER_WEIGHTS_SHA256:
        raise ValueError('perceptor format or protected weight digest differs')
    model = CellAppearance()
    model.load_state_dict(data['weights'], strict=True)
    return model.eval().requires_grad_(False)


def semantic_probabilities(perceptor: CellAppearance, frames: torch.Tensor) -> torch.Tensor:
    """Learned [B,144,22] evidence from current pixels; no masks or engine inputs."""
    with torch.no_grad(), torch.autocast(frames.device.type, enabled=False):
        roles, *attributes = perceptor(frames)
        return torch.cat((roles.float().sigmoid(), *(value.float().softmax(-1) for value in attributes)), -1)


class SpatialSemanticOutcomePolicy(SpatialOutcomePolicy):
    def __init__(self, encoder, planner, perceptor, *, direct_weight=0., planner_weight=1.):
        if not isinstance(planner, SpatialSemanticOutcomePlanner) or not isinstance(perceptor, CellAppearance):
            raise ValueError('semantic policy needs its versioned planner and pixel perceptor')
        super().__init__(encoder, planner, direct_weight=direct_weight, planner_weight=planner_weight)
        self.perceptor = perceptor.float().eval().requires_grad_(False)

    def config(self):
        return {**super().config(), 'decision_architecture': 'spatial_semantic_outcomes',
                'semantic_version': 1, 'direct_scene_actor': self.planner.cfg.actor}

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.perceptor.eval()
        return self

    def forward(self, frames, history_valid=None, previous_actions=None):
        device = next(self.encoder.parameters()).device
        frames = torch.as_tensor(frames, device=device)
        if frames.ndim not in (3, 4):
            raise ValueError('public frames must be [B,64,64] or [B,H,64,64]')
        current = frames[:, -1] if frames.ndim == 4 else frames
        with torch.no_grad(), encoder_execution(device):
            encoding = self.encoder.encode(frames, history_valid, previous_actions)
            player = self.encoder.player_weights(encoding['cells'])[1]
            semantic = semantic_probabilities(self.perceptor, current)
            direct = None
            if self.direct_weight:
                direct = self.encoder.direct_logits(encoding['cells'])[0]
                if self.encoder.cfg.query_readout:
                    direct = direct + self.encoder.query_logits(encoding, player)
        result = self.planner(encoding['raw'], encoding['state'], encoding['glyph'], player, semantic)
        scores = self.planner_weight * result['action_logits']
        return scores if direct is None else scores + self.direct_weight * direct


def checkpoint_from_parent(parent: dict, planner: SpatialSemanticOutcomePlanner, perceptor: CellAppearance,
                           *, parent_checkpoint_sha256: str = RECOVERY_SHA256) -> dict:
    """Clone the inference envelope; caller supplies truthful latest training metadata."""
    if (parent.get('format') != PARENT_FORMAT or parent_checkpoint_sha256 != RECOVERY_SHA256
            or not isinstance(planner, SpatialSemanticOutcomePlanner) or not isinstance(perceptor, CellAppearance)):
        raise ValueError('semantic migration requires retained spatial lineage and matching modules')
    teacher_weights = {name: value.detach().cpu().clone() for name, value in perceptor.state_dict().items()}
    if weights_sha256(teacher_weights) != TEACHER_WEIGHTS_SHA256:
        raise ValueError('perceptor weights differ from the protected fresh teacher')
    result = copy.copy(parent)
    result.pop('parameters', None)  # Parent counts describe a different complete model.
    result.update(format=FORMAT, planner_config=planner.config(),
        planner_weights={name: value.detach().cpu().clone() for name, value in planner.state_dict().items()},
        planner_parameters=planner.parameter_count(), source_checkpoint_sha256=parent_checkpoint_sha256,
        encoder_weights={name: value.detach().cpu().clone() for name, value in parent['encoder_weights'].items()},
        perceptor_format=PERCEPTOR_FORMAT, perceptor_config=copy.deepcopy(PERCEPTOR_CONFIG),
        perceptor_weights=teacher_weights, perceptor_weights_sha256=TEACHER_WEIGHTS_SHA256,
        perceptor_checkpoint_sha256=TEACHER_SHA256, perceptor_frozen=True,
        perceptor_parameters=perceptor.parameter_count(),
        semantic_architecture=_architecture(planner))
    return result


def load_checkpoint(path, device='cpu', *, direct_weight=None, planner_weight=None):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (data.get('format') != FORMAT or data.get('encoder_parent_sha256') != PARENT_SHA
            or data.get('source_checkpoint_sha256') != RECOVERY_SHA256
            or data.get('encoder_frozen') is not True or data.get('official_training_inputs') is not False
            or data.get('privileged_inference_inputs', False) is not False):
        raise ValueError('generated-only semantic policy with retained frozen-encoder lineage required')
    if data.get('encoder_runtime') != ENCODER_RUNTIME:
        raise ValueError('encoder runtime differs from the public feature contract')
    if data.get('encoder_weights_sha256') != weights_sha256(data['encoder_weights']):
        raise ValueError('embedded encoder digest differs')
    if (data.get('perceptor_format') != PERCEPTOR_FORMAT or data.get('perceptor_config') != PERCEPTOR_CONFIG
            or data.get('perceptor_checkpoint_sha256') != TEACHER_SHA256 or data.get('perceptor_frozen') is not True
            or data.get('perceptor_weights_sha256') != TEACHER_WEIGHTS_SHA256
            or weights_sha256(data['perceptor_weights']) != TEACHER_WEIGHTS_SHA256):
        raise ValueError('embedded perceptor differs from the protected fresh pixel teacher')
    encoder = WorldPolicy(WorldModelConfig.from_dict(data['encoder_config']))
    encoder.load_state_dict(data['encoder_weights'], strict=True)
    planner = SpatialSemanticOutcomePlanner(data['planner_config'])
    if data.get('semantic_architecture') != _architecture(planner):
        raise ValueError('semantic architecture declaration differs from the versioned planner')
    planner.load_state_dict(data['planner_weights'], strict=True)
    perceptor = CellAppearance()
    perceptor.load_state_dict(data['perceptor_weights'], strict=True)
    weights = data['score_weights']
    policy = SpatialSemanticOutcomePolicy(encoder, planner, perceptor,
        direct_weight=weights['direct'] if direct_weight is None else direct_weight,
        planner_weight=weights['planner'] if planner_weight is None else planner_weight)
    return policy.to(device).eval(), data
