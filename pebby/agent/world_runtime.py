"""Optional execution settings for the existing differentiable WorldPolicy.

These wrap methods, not modules or parameters, so checkpoint keys and the
inference architecture stay identical. Call once before optimizer updates.
"""
from functools import wraps
import os

import torch


def configure_execution(model, *, compile_core=False, temporal_backend='auto'):
    if temporal_backend not in ('auto', 'math', 'cudnn', 'flash'):
        raise ValueError('unknown temporal attention backend')
    if getattr(model, '_execution_configured', False):
        raise ValueError('execution already configured on this model')
    if temporal_backend != 'auto':
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend = {'math': SDPBackend.MATH, 'cudnn': SDPBackend.CUDNN_ATTENTION,
                   'flash': SDPBackend.FLASH_ATTENTION}[temporal_backend]

        def wrap_attention(forward):
            @wraps(forward)
            def attention(*args, **kwargs):
                with sdpa_kernel(backend):
                    return forward(*args, **kwargs)
            return attention

        for layer in model.temporal:
            layer.attention.forward = wrap_attention(layer.attention.forward)
    if compile_core:
        # Loss code calls assemble/_loop directly, bypassing model.forward.
        # Keep random SIGReg and input validation outside this pure region.
        eager_loop = model._loop
        compiled_loop = torch.compile(eager_loop, fullgraph=True, dynamic=False, mode='default')

        @wraps(eager_loop)
        def loop(*args, **kwargs):
            # Validation includes partial batches. Keep inference eager instead
            # of compiling additional graphs for every tail shape.
            if torch.is_grad_enabled():
                return compiled_loop(*args, **kwargs)
            return eager_loop(*args, **kwargs)

        model._loop = loop
    model._execution_configured = True
    return dict(compile_core=bool(compile_core), temporal_backend=temporal_backend,
                compile_scope='gradient_enabled_core',
                allocator_environment={key: os.environ.get(key) for key in
                                       ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')},
                compile_threads=os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'))
