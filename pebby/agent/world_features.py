"""Feature-only compatibility path for the source-pinned WorldPolicy encoder.

Spatial outcomes use raw/refined tokens and glyph probabilities, never the
LeWM latent. The original WorldPolicy assembles those features and projects the
latent in one method. Changing that source would invalidate existing structured
encoder/cache/checkpoint provenance. This narrow adapter therefore mirrors its
feature assembly, guarded by exact method-source fingerprints, while reusing
all learned modules and input/frame/refinement methods directly.

No modules, parameters, methods, or runtime hooks are replaced. The returned
mapping intentionally has no latent. Consolidate this assembly into a shared
WorldPolicy.encode_features seam during a versioned source migration; until
then changes to the mirrored upstream methods require a new parity review.
"""
from functools import lru_cache, partial
import hashlib
import inspect
import math

import torch
from torch.utils.checkpoint import checkpoint

from .world_model import WorldPolicy, CELLS, GLYPH_CLASSES, glyph_probabilities


SOURCE_CONTRACT = {
    'encode': '93bb415fefac68c46cee8a7c769965b5c1ce43781a9465dd5f103eae21e3d0e7',
    'assemble': '6d3f4a9b0b6051754da2eddc843739da797ad58b194835b0fe8ccb2bfd116779',
    '_assemble': '5dc76f5ae61b41e3086ed64e570447cf2bc2c64ed3b63fd77bfab8406f2339ba',
}


@lru_cache(maxsize=8)
def _check_contract(methods):
    # Function and code identities are cache keys: replacing either requires
    # verification again, without repeated source I/O during each decision.
    for name, function, _code in methods:
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError) as error:
            raise ValueError('feature adapter requires inspectable WorldPolicy source') from error
        if hashlib.sha256(source.encode()).hexdigest() != SOURCE_CONTRACT[name]:
            raise ValueError('WorldPolicy feature source contract changed: ' + name)


def _assemble_one(encoder, tokens, history_valid, previous_actions, loops, glyph_logits):
    """Mirrors the pinned _assemble prefix through refined feature construction."""
    depth = encoder.loops if loops is None else loops
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError('loops must be a positive integer')
    history = tokens.size(1)
    raw = tokens[:, -1]
    glyph = None
    if encoder.cfg.glyph_recall:
        if glyph_logits is None or tuple(glyph_logits.shape) != (tokens.size(0), GLYPH_CLASSES):
            raise ValueError('glyph_recall needs the current frame\'s glyph logits [B, 14]')
        glyph = glyph_probabilities(glyph_logits.float()).to(tokens.dtype)
    elif glyph_logits is not None:
        raise ValueError('glyph logits were given to a network without glyph_recall')
    ages = torch.arange(history - 1, -1, -1, device=tokens.device)
    tokens = (tokens + encoder.age_embedding(ages)[None, :, None, :].to(tokens.dtype)
              + encoder.action_embedding(previous_actions + 1)[:, :, None, :].to(tokens.dtype))
    current = tokens[:, -1]
    for layer in encoder.temporal:
        current = layer(current, tokens, history_valid)
    source = encoder.source_norm(current)
    if glyph is not None:
        source = source + encoder.glyph_context(glyph)[:, None, :].to(source.dtype)
    state = encoder._refine(source, source, depth)
    return dict(state=state, cells=state[:, :CELLS], raw=raw, glyph=glyph)


def _assemble(encoder, tokens, history_valid, previous_actions, loops, glyph_logits):
    size = encoder.encoder_chunk_size
    if size and tokens.size(0) > size:
        glyph_chunks = (glyph_logits.split(size) if glyph_logits is not None
                        else [None] * math.ceil(tokens.size(0) / size))
        chunks = [_assemble(encoder, t, v, a, loops, g) for t, v, a, g in zip(
            tokens.split(size), history_valid.split(size), previous_actions.split(size), glyph_chunks)]
        state = torch.cat([chunk['state'] for chunk in chunks], dim=0)
        return dict(state=state, cells=state[:, :CELLS],
                    raw=torch.cat([chunk['raw'] for chunk in chunks], dim=0),
                    glyph=torch.cat([chunk['glyph'] for chunk in chunks], dim=0) if chunks[0]['glyph'] is not None else None)
    assemble = partial(_assemble_one, encoder)
    if encoder.checkpoint_encoder and torch.is_grad_enabled():
        return checkpoint(assemble, tokens, history_valid, previous_actions, loops, glyph_logits,
                          use_reentrant=False)
    return assemble(tokens, history_valid, previous_actions, loops, glyph_logits)


def encode_features(encoder, frames, history_valid=None, previous_actions=None, *, loops=None):
    """Return raw/state/cells/glyph, skipping reduce/projector and latent recall.

    Supports the exact WorldPolicy class, including its chunking and gradient
    checkpoint settings. Subclasses and instance assembly overrides need their
    own compatibility review and are rejected instead of silently bypassed.
    """
    if type(encoder) is not WorldPolicy or any(name in encoder.__dict__ for name in SOURCE_CONTRACT):
        raise ValueError('feature adapter requires an unmodified WorldPolicy assembly implementation')
    _check_contract(tuple((name, getattr(WorldPolicy, name), getattr(WorldPolicy, name).__code__)
                          for name in SOURCE_CONTRACT))
    frames, history_valid, previous_actions = encoder._prepare(frames, history_valid, previous_actions)
    batch, history = frames.shape[:2]
    tokens = encoder.frame_tokens(frames.flatten(0, 1)).view(batch, history, encoder.tokens, -1)
    glyph_logits = encoder.glyph_logits(frames[:, -1]) if encoder.cfg.glyph_recall else None
    return _assemble(encoder, tokens, history_valid, previous_actions, loops, glyph_logits)
