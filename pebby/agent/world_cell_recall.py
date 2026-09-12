"""Generated-data provenance for importing a frozen public-pixel decoder.

Training utilities only: inference needs the weights embedded in WorldPolicy,
and never loads labels, a bank, the game engine, or an external checkpoint.
"""
from pathlib import Path
import hashlib
import io
import json
import re

import numpy as np
import torch

from .cell_appearance import CellAppearance, FORMAT


def cell_weight_digest(encoder):
    """Canonical digest of the embedded convolutional decoder, independent of device."""
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode() + b'\n')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_cell_source(source, validation_seeds, encoder=None):
    if not isinstance(source, dict) or source.get('format') != FORMAT:
        raise ValueError('cell recall needs generated decoder provenance')
    seeds = source.get('train_seeds')
    if not isinstance(seeds, list) or not seeds or any(type(x) is not int or not 0 <= x < 1_000_000 for x in seeds):
        raise ValueError('cell decoder training seeds must be in the generated training split')
    if len(seeds) != len(set(seeds)):
        raise ValueError('duplicate cell decoder training seeds')
    if set(seeds) & set(map(int, validation_seeds)):
        raise ValueError('cell decoder training overlaps world validation levels')
    if source.get('validation_used_for_training_or_selection') is not False:
        raise ValueError('cell decoder validation must not be used for training or selection')
    if source.get('frozen') is not True or source.get('parameters') != 110166:
        raise ValueError('cell decoder must record its frozen architecture')
    for key in ('path', 'bank', 'proof'):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f'cell decoder provenance missing {key}')
    for key in ('sha256', 'bank_sha256', 'proof_sha256', 'embedded_weights_sha256'):
        if not isinstance(source.get(key), str) or re.fullmatch('[0-9a-f]{64}', source[key]) is None:
            raise ValueError(f'cell decoder provenance missing valid {key}')
    if encoder is not None and cell_weight_digest(encoder) != source['embedded_weights_sha256']:
        raise ValueError('embedded cell decoder weights differ from imported frozen decoder')
    return source


def load_cell_source(checkpoint_path, bank_path, proof_path, validation_seeds):
    # The checked initial appearance loader verifies masks, seed ranges, proof
    # completion and the raw bank hash. Labels never enter the imported model.
    from tools.train_cell_appearance import digest, load_examples

    checkpoint_path, bank_path, proof_path = map(Path, (checkpoint_path, bank_path, proof_path))
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != FORMAT:
        raise ValueError('unsupported cell decoder checkpoint')
    hashes = checkpoint.get('source_hashes', {})
    input_hashes = {path: digest(path) for path in (bank_path, proof_path, Path('pebby/agent/cell_appearance.py'))}
    for path, expected_hash in input_hashes.items():
        # Match both relative and absolute spellings of the recorded input.
        matches = [value for key, value in hashes.items() if Path(key).resolve() == path.resolve()]
        if matches != [expected_hash]:
            raise ValueError(f'cell decoder provenance mismatch: {path}')
    load_examples(bank_path, proof_path)
    with np.load(bank_path, allow_pickle=False) as data:
        seeds = sorted(map(int, data['seeds'][data['split'] == 'train']))
    encoder = CellAppearance()
    encoder.load_state_dict(checkpoint['weights'], strict=True)
    if checkpoint.get('parameters') != encoder.parameter_count():
        raise ValueError('cell decoder parameter count mismatch')
    from .cell_appearance_dense import DenseCellAppearance
    embedded_weights_sha256 = cell_weight_digest(DenseCellAppearance(encoder))
    if any(digest(path) != expected_hash for path, expected_hash in input_hashes.items()):
        raise ValueError('cell decoder inputs changed while loading')
    source = dict(format=FORMAT, path=str(checkpoint_path), sha256=checkpoint_sha256,
                  bank=str(bank_path), bank_sha256=input_hashes[bank_path], proof=str(proof_path),
                  proof_sha256=input_hashes[proof_path], train_seeds=seeds,
                  parameters=encoder.parameter_count(), frozen=True,
                  embedded_weights_sha256=embedded_weights_sha256,
                  validation_used_for_training_or_selection=checkpoint.get('validation_used_for_training_or_selection'),
                  initial_state_only=checkpoint.get('initial_state_only'),
                  source_hashes=hashes)
    verify_cell_source(source, validation_seeds)
    return encoder.eval(), source


def initialize_cell_encoder(model, encoder):
    from .cell_appearance_dense import DenseCellAppearance

    if not model.cfg.cell_recall or not isinstance(encoder, CellAppearance):
        raise ValueError('cell initialization needs cell_recall and CellAppearance')
    converted = DenseCellAppearance(encoder)
    model.cell_appearance.load_state_dict(converted.state_dict(), strict=True)
    return sorted(f'cell_appearance.{key}' for key in converted.state_dict())
