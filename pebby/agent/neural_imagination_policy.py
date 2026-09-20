"""Public-observation wrapper and pinned experimental imagination checkpoint."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from .neural_imagination import NeuralImagination
from .structured_factored_policy import load_factored_policy_checkpoint, state_digest

FORMAT = 'pebby.neural-imagination.v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class NeuralImaginationPolicy(nn.Module):
    def __init__(self, parent, config=None):
        super().__init__()
        self.encoder = parent.encoder
        self.planner = NeuralImagination(parent.dynamics, config)
        self.encoder.requires_grad_(False)
        self.planner.dynamics.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.planner.dynamics.eval()
        return self

    def config(self):
        return dict(history=8, architecture='structured', decision_architecture='neural_imagination',
                    **self.planner.config())

    def forward(self, frames, history_valid=None, previous_actions=None):
        with torch.no_grad():
            field = self.encoder(frames, history_valid, previous_actions)
        return self.planner(field)


def save_checkpoint(path, policy, parent_path, metadata):
    if Path(path).exists():
        raise FileExistsError(path)
    if metadata.get('official_training_inputs') is not False:
        raise ValueError('generated-only training attestation required')
    parent_path = Path(parent_path).resolve()
    parent, _ = load_factored_policy_checkpoint(parent_path)
    for trained, original in ((policy.encoder, parent.encoder),
                               (policy.planner.dynamics, parent.dynamics)):
        if state_digest(trained.state_dict()) != state_digest(original.state_dict()):
            raise ValueError('frozen perception/dynamics changed')
    weights = {key: value.detach().cpu().clone() for key, value in policy.planner.state_dict().items()
               if not key.startswith('dynamics.')}
    source_paths = [Path(__file__), Path(__file__).with_name('neural_imagination.py')]
    envelope = dict(format=FORMAT, config=policy.planner.config(), weights=weights,
                    weights_sha256=state_digest(weights), parent_path=str(parent_path),
                    parent_sha256=digest(parent_path), metadata=metadata,
                    sources={str(p.resolve()): digest(p) for p in source_paths})
    with Path(path).open('xb') as stream:
        torch.save(envelope, stream)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('format') != FORMAT or saved['metadata'].get('official_training_inputs') is not False
            or digest(saved['parent_path']) != saved['parent_sha256']
            or state_digest(saved['weights']) != saved['weights_sha256']):
        raise ValueError('invalid imagination checkpoint provenance')
    expected = {str(Path(__file__).resolve()), str(Path(__file__).with_name('neural_imagination.py').resolve())}
    if set(saved['sources']) != expected or any(digest(p) != h for p, h in saved['sources'].items()):
        raise ValueError('imagination implementation changed')
    parent, _ = load_factored_policy_checkpoint(saved['parent_path'], device)
    policy = NeuralImaginationPolicy(parent, saved['config'])
    current = policy.planner.state_dict()
    if set(saved['weights']) != {k for k in current if not k.startswith('dynamics.')}:
        raise ValueError('missing or unexpected learned selector weights')
    current.update(saved['weights'])
    policy.planner.load_state_dict(current, strict=True)
    if any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
        raise ValueError('nonfinite checkpoint weights')
    return policy.to(device).eval(), saved['metadata']
