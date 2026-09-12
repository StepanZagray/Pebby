"""Zero redundant host gathers for already materialized policy training batches.

This helper is for the next training run, not public inference. Actual successor
fields remain supervised training inputs; only imagined successors are playable
policy inputs. No source cache validation is replaced by this local shape check.
"""
import numpy as np
import torch


def prepared_policy_inputs(batch, mode, device='cpu', *, allow_unreachable=False):
    """Convert contiguous NumPy batch arrays directly, without fancy indexing.

    CPU float32 arrays are shared with tensors and must not be mutated during
    use. GPU copies/conversions preserve the existing float32 head inputs.
    """
    if mode not in ('direct', 'successors'):
        raise ValueError('unknown policy mode')
    masks = batch['optimal']
    if not isinstance(masks, np.ndarray) or masks.ndim != 1 or masks.dtype.kind not in 'iu':
        raise ValueError('integer optimal masks[B] required')
    if not masks.flags.c_contiguous or not len(masks) or np.any((masks < (0 if allow_unreachable else 1)) | (masks > 15)):
        raise ValueError('contiguous nonempty optimal masks in1..15 required')
    n = len(masks)
    names = {'direct': ('fields',)} if mode == 'direct' else {'actual': ('next_fields',), 'imagined': ('imagined_fields',)}
    inputs = {}
    for kind, (key,) in names.items():
        array = batch[key]
        shape = (n,148,96) if mode == 'direct' else (n,4,148,96)
        if not isinstance(array, np.ndarray) or array.shape != shape or not array.flags.c_contiguous:
            raise ValueError(f'{key} requires contiguous prepared batch with shape{shape}')
        if array.dtype not in (np.dtype('float16'), np.dtype('float32')):
            raise ValueError(f'{key} requires float16/float32')
        if kind == 'imagined' and array.dtype != np.float32:
            raise ValueError('imagined cache precision must remain float32')
        inputs[kind] = torch.as_tensor(array, device=device).float().detach()
    return inputs, torch.as_tensor(masks, device=device).long()


def outputs_for_prepared_batch(head, batch, device='cpu'):
    """Same head calls as outputs_for_rows, after paired_batch has gathered once."""
    inputs, masks = prepared_policy_inputs(batch, head.cfg.mode, device)
    return {kind: head(value) for kind, value in inputs.items()}, masks
