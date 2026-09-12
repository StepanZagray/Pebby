"""Frozen convolutional execution of CellAppearance's existing pixel MLP.

Checkpoint ownership remains with CellAppearance. This adapter copies its weights:
7x7 position-major/channel-last linear weights become a stride-5 convolution;
the remaining linear layers become 1x1 convolutions. No policy integration occurs.
FP32 accumulation order differs: use atol=2e-5, rtol=2e-5 for logits, not bit parity.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .cell_appearance import CellAppearance, ROLE_NAMES, ATTRIBUTE_SIZES


class DenseCellAppearance(nn.Module):
    """Public [B,64,64] palette frames -> role/shape/color/rotation [B,144,C].

    At most chunk_size frames are one-hot encoded at once (default128).
    For a caller's B*H flattened histories, only the small outputs grow with B*H.
    The adapter is frozen and does not retain or modify the source module.
    """
    def __init__(self, appearance=None, chunk_size=128):
        super().__init__()
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        source=CellAppearance() if appearance is None else appearance
        self._check_source(source)
        layers=source.network
        device,dtype=layers[0].weight.device,layers[0].weight.dtype
        self.network=nn.Sequential(nn.Conv2d(16,128,7,stride=5,device=device,dtype=dtype),nn.GELU(),
                                   nn.Conv2d(128,64,1,device=device,dtype=dtype),nn.GELU(),
                                   nn.Conv2d(64,22,1,device=device,dtype=dtype))
        self.chunk_size=chunk_size
        self.load_from(source)
        self.requires_grad_(False);self.eval()

    @staticmethod
    def _check_source(source):
        if not isinstance(source,CellAppearance):
            raise TypeError('appearance must be CellAppearance')
        layers=source.network
        if (len(layers)!=5 or not all(isinstance(layers[i],nn.Linear) for i in (0,2,4))
                or not all(isinstance(layers[i],nn.GELU) and layers[i].approximate=='none' for i in (1,3))
                or [tuple(layers[i].weight.shape) for i in (0,2,4)]!=[(128,784),(64,128),(22,64)]):
            raise ValueError('unsupported CellAppearance network layout')

    def load_from(self, appearance):
        """Copy an original-format decoder without retaining it or changing it."""
        self._check_source(appearance)
        layers=appearance.network
        with torch.no_grad():
            self.network[0].weight.copy_(layers[0].weight.reshape(128,7,7,16).permute(0,3,1,2))
            for i in (2,4):self.network[i].weight.copy_(layers[i].weight[:,:,None,None])
            for i in (0,2,4):self.network[i].bias.copy_(layers[i].bias)
        self.requires_grad_(False)
        return self

    def _chunk(self, frames):
        # Pad palette INDICES before one-hot: outside-frame zero is color0,
        # whose channel must be1, rather than an all-zero feature vector.
        indices=F.pad(frames,(1,1,1,1),value=0)[:,:62,4:66]
        pixels=F.one_hot(indices.long(),16).permute(0,3,1,2).to(self.network[0].weight.dtype)
        return self.network(pixels).flatten(2).transpose(1,2)

    def forward(self, frames):
        if not isinstance(frames,torch.Tensor) or frames.ndim!=3 or tuple(frames.shape[-2:])!=(64,64):
            raise ValueError('frames must be [B,64,64]')
        if frames.dtype not in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
            raise ValueError('frames must contain integer palette indices')
        if frames.device!=self.network[0].weight.device:
            raise ValueError('frames and adapter must be on the same device')
        if frames.numel() and bool(((frames<0)|(frames>15)).any()):
            raise ValueError('palette indices must be in0..15')
        if len(frames):
            logits=torch.cat([self._chunk(part) for part in frames.split(self.chunk_size)],0)
        else:
            logits=self.network[0].weight.new_empty((0,144,22))
        return logits.split((len(ROLE_NAMES),*ATTRIBUTE_SIZES),dim=-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
