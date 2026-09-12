"""Assemble a public-H8 controller from a trained workspace action head.

The head's actor checkpoint fixes perception and dynamics. Actual successors
and training labels are never accepted by the assembled policy forward method.
"""
from dataclasses import replace
import hashlib
import io
from pathlib import Path

import torch

from .structured_factored_policy import load_factored_policy_checkpoint, state_digest
from .structured_workspace_policy import StructuredWorkspaceReadout


FORMAT = 'pebby.structured-workspace-readout.v1'


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_workspace_policy_checkpoint(path, device='cpu', *, loops=None):
    raw = Path(path).read_bytes()
    saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT:
        raise ValueError('wrong workspace controller format')
    provenance = saved.get('training_provenance', {})
    if (saved.get('official_inputs_used') is not False
            or provenance.get('official_inputs_used') is not False
            or provenance.get('source') != 'generated_only'
            or saved.get('source_unchanged') is not True):
        raise ValueError('verified generated-only training provenance required')
    trained = provenance.get('depths')
    depth = provenance.get('primary_depth') if loops is None else loops
    if type(depth) is not int or depth not in (1, 2, 4) or trained != [1, 2, 4]:
        raise ValueError('controller requires a declared trained depth 1,2,4')
    actor = saved.get('actor_checkpoint')
    if not isinstance(actor, str) or _digest(actor) != saved.get('actor_sha256'):
        raise ValueError('workspace actor binding mismatch')
    sources = saved.get('sources', {})
    if not isinstance(sources, dict) or sources.get(actor) != saved['actor_sha256']:
        raise ValueError('actor missing from training source binding')
    # Bind the actual head implementation, not every large training array at
    # inference. Full training provenance remains stored in the checkpoint.
    for name in ('structured_workspace_policy.py', 'structured_policy.py'):
        module = Path(__file__).with_name(name).resolve()
        matches = [value for key, value in sources.items() if Path(key).resolve() == module]
        if not matches or any(value != _digest(module) for value in matches):
            raise ValueError('workspace implementation source mismatch')
    head = StructuredWorkspaceReadout(saved['config'])
    head.load_state_dict(saved['weights'], strict=True)
    if state_digest(head.state_dict()) != saved.get('state_sha256'):
        raise ValueError('workspace state digest mismatch')
    if (head.parameter_count() != saved.get('parameters')
            or head.trainable_parameter_count() != saved.get('active_trainable_parameters')):
        raise ValueError('workspace head parameter counts mismatch')
    if head.cfg.mode != 'successors':
        raise ValueError('workspace controller requires successor mode')
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError('nonfinite workspace state')
    policy, actor_record = load_factored_policy_checkpoint(actor, device)
    if _digest(actor) != saved['actor_sha256']:
        raise ValueError('actor changed during reconstruction')
    expected = dict(head.config())
    expected.pop('memory_mode')
    expected.pop('checkpoint_workspace')
    if expected != policy.readout.config():
        raise ValueError('workspace and frozen actor configuration mismatch')
    head.cfg = replace(head.cfg, loops=depth, checkpoint_workspace=False)
    policy.readout = head.to(device)
    policy.cfg = replace(policy.cfg, loops=depth)
    policy.eval().requires_grad_(False)
    if _digest(path) != hashlib.sha256(raw).hexdigest():
        raise ValueError('workspace checkpoint changed during reconstruction')
    info = {'format': FORMAT, 'config': policy.config(), 'parameters': policy.parameter_count(),
            'parameter_counts': policy.parameter_counts(), 'readout_config': head.config(),
            'readout_parameters': head.parameter_count(), 'actor_checkpoint': actor,
            'actor_sha256': saved['actor_sha256'], 'checkpoint_sha256': hashlib.sha256(raw).hexdigest(),
            'training_provenance': provenance, 'inference_only': True,
            'static_unused_workspace_parameters': (head.parameter_count()-108097 if head.cfg.memory_mode == 'static' else 0)}
    return policy, info
