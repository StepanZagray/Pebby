"""Public-H8 neural selector over a separately fitted, frozen learned D.

The v1 imagination implementation and its checkpoints remain unchanged. This
format binds the repaired dynamics artifact, its original encoder ancestry,
and fresh selector weights. It adds no engine access, search, or game memory.
"""
import copy
import hashlib
import io
from pathlib import Path

import torch
from torch import nn

from .neural_imagination_policy import NeuralImaginationPolicy, digest
from .neural_imagination_policy import load_checkpoint as load_original_checkpoint
from .structured_factored_policy import canonical_metadata, state_digest
from .structured_local_glyph import LocalGlobalGlyphTransition

FORMAT = 'pebby.repaired-neural-imagination.v1'
DYNAMICS_FORMAT = 'pebby.neural-planning-dynamics.v1'


def _read(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError('checkpoint checksum differs')
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), sha256


def _sources():
    paths = [Path(__file__), Path(__file__).with_name('neural_imagination.py'),
             Path(__file__).with_name('neural_imagination_policy.py')]
    return {str(path.resolve()): digest(path) for path in paths}


def _dynamics_source_paths():
    root = Path(__file__).resolve().parents[2]
    return {str(root / path) for path in (
        'tools/train_neural_planning_dynamics.py', 'pebby/agent/neural_planning_objective.py',
        'pebby/agent/neural_imagination_policy.py', 'tools/train_navigation_probe.py',
        'pebby/agent/structured_transition.py', 'pebby/agent/structured_global_glyph.py',
        'pebby/agent/structured_local_glyph.py')}


def _finite(state):
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('nonfinite checkpoint weights')


class _DynamicsParent(nn.Module):
    def __init__(self, encoder, dynamics, sources):
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.sources = sources
        self.requires_grad_(False)
        self.eval()


def load_dynamics_parent(path, device='cpu', expected_sha256=None):
    """Restore fixed perception and strict repaired D for fresh selector training.

    The returned parent's encoder/dynamics/sources interface matches the
    factored parent consumed by the existing generated-field selector trainer.
    Every recorded dynamics training source is checked before use.
    """
    saved, sha256 = _read(path, expected_sha256)
    if (saved.get('format') != DYNAMICS_FORMAT
            or saved.get('official_training_inputs') is not False
            or state_digest(saved['weights']) != saved['weights_sha256']):
        raise ValueError('invalid repaired dynamics provenance')
    if type(saved.get('updates')) is not int or saved['updates'] < 1:
        raise ValueError('positive dynamics update count required')
    events = saved.get('train_events', {})
    if any(type(events.get(name)) is not int or events[name] < 1
           for name in ('lost_life', 'terminal', 'won')):
        raise ValueError('positive generated dynamics event support required')
    sources = saved.get('sources')
    parent_path = str(Path(saved['parent']).resolve())
    if (not isinstance(sources, dict) or not _dynamics_source_paths().issubset(sources)
            or sources.get(parent_path) != saved['parent_sha256']):
        raise ValueError('incomplete repaired dynamics source binding')
    for source, expected in sources.items():
        if digest(source) != expected:
            raise ValueError('repaired dynamics source changed: ' + source)
    if digest(parent_path) != saved['parent_sha256']:
        raise ValueError('original imagination parent checksum differs')
    original, _ = load_original_checkpoint(parent_path, 'cpu')
    encoder = original.encoder
    dynamics = original.planner.dynamics
    if state_digest(encoder.state_dict()) != saved['encoder_state_sha256']:
        raise ValueError('repaired dynamics encoder differs')
    if type(dynamics) is not LocalGlobalGlyphTransition or dynamics.config() != saved['config']:
        raise ValueError('repaired dynamics architecture differs')
    _finite(saved['weights'])
    dynamics.load_state_dict(saved['weights'], strict=True)
    if state_digest(dynamics.state_dict()) != saved['weights_sha256']:
        raise ValueError('repaired dynamics weights changed during restore')
    _finite(encoder.state_dict())
    provenance = dict(
        encoder_metadata=canonical_metadata(encoder.metadata()),
        code_hashes={**{p: h for p, h in sources.items() if p.endswith('.py')}, **_sources()},
        encoder_state_sha256=saved['encoder_state_sha256'],
        dynamics_state_sha256=saved['weights_sha256'],
        dynamics_checkpoint=dict(path=str(Path(path).resolve()), sha256=sha256),
        original_parent=dict(path=parent_path, sha256=saved['parent_sha256']))
    metadata = {key: copy.deepcopy(value) for key, value in saved.items() if key != 'weights'}
    return _DynamicsParent(encoder, dynamics, provenance).to(device), metadata


class RepairedImaginationPolicy(NeuralImaginationPolicy):
    def config(self):
        return dict(history=8, architecture='structured',
                    decision_architecture='repaired_neural_imagination', **self.planner.config())


def save_checkpoint(path, policy, parent_path, metadata):
    """Save selector weights bound to an unchanged repaired-D artifact."""
    if Path(path).exists():
        raise FileExistsError(path)
    if type(policy) not in (RepairedImaginationPolicy, NeuralImaginationPolicy):
        raise ValueError('neural imagination policy required')
    if metadata.get('official_training_inputs') is not False:
        raise ValueError('generated-only training attestation required')
    parent_path = Path(parent_path).resolve()
    parent, _ = load_dynamics_parent(parent_path)
    if (policy.planner.dynamics.config() != parent.dynamics.config()
            or policy.encoder.metadata() != parent.encoder.metadata()):
        raise ValueError('frozen perception/dynamics configuration changed')
    for trained, original in ((policy.encoder, parent.encoder),
                              (policy.planner.dynamics, parent.dynamics)):
        if (state_digest(trained.state_dict()) != state_digest(original.state_dict())
                or any(p.requires_grad for p in trained.parameters())):
            raise ValueError('frozen perception/dynamics changed')
    _finite(policy.state_dict())
    weights = {key: value.detach().cpu().clone() for key, value in policy.planner.state_dict().items()
               if not key.startswith('dynamics.')}
    envelope = dict(format=FORMAT, config=policy.planner.config(), weights=weights,
                    weights_sha256=state_digest(weights), parent_path=str(parent_path),
                    parent_sha256=parent.sources['dynamics_checkpoint']['sha256'],
                    encoder_state_sha256=parent.sources['encoder_state_sha256'],
                    dynamics_state_sha256=parent.sources['dynamics_state_sha256'],
                    original_parent=copy.deepcopy(parent.sources['original_parent']),
                    metadata=copy.deepcopy(metadata), sources=_sources())
    with Path(path).open('xb') as stream:
        torch.save(envelope, stream)


def load_checkpoint(path, device='cpu'):
    saved, _ = _read(path)
    if (saved.get('format') != FORMAT
            or saved['metadata'].get('official_training_inputs') is not False
            or state_digest(saved['weights']) != saved['weights_sha256']):
        raise ValueError('invalid repaired imagination provenance')
    if saved.get('sources') != _sources():
        raise ValueError('repaired imagination implementation changed')
    parent, _ = load_dynamics_parent(saved['parent_path'], expected_sha256=saved['parent_sha256'])
    if any(saved[key] != parent.sources[key] for key in
           ('encoder_state_sha256', 'dynamics_state_sha256', 'original_parent')):
        raise ValueError('repaired imagination frozen ancestry differs')
    policy = RepairedImaginationPolicy(parent, saved['config'])
    current = policy.planner.state_dict()
    if set(saved['weights']) != {key for key in current if not key.startswith('dynamics.')}:
        raise ValueError('missing or unexpected learned selector weights')
    current.update(saved['weights'])
    policy.planner.load_state_dict(current, strict=True)
    _finite(policy.state_dict())
    return policy.to(device).eval(), saved['metadata']
