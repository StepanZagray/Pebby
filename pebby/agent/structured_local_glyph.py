"""Global glyph transition plus a shared learned board-local residual.

Only public fields/actions enter the model. The local branch has no movement,
collision or mechanic rules; global attention still handles nonlocal effects.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from .structured_global_glyph import GlobalGlyphConfig, GlobalGlyphTransition

LOCAL_GLYPH_FORMAT = 'pebby.structured-transition-local-global-glyph.v1'


@dataclass(frozen=True)
class LocalGlyphConfig:
    loops: int = 2
    heads: int = 4
    expansion: int = 2
    event_hidden: int = 128
    steps_classes: int = 44
    glyph_hidden: int = 96
    variant: str = 'local_global_glyph'

    def __post_init__(self):
        GlobalGlyphConfig(**self.global_config())
        if self.variant != 'local_global_glyph':
            raise ValueError('variant must be local_global_glyph')

    def global_config(self):
        return {**{key:getattr(self,key) for key in ('loops','heads','expansion','event_hidden','steps_classes','glyph_hidden')},'variant':'global_glyph'}


class LocalFieldBlock(nn.Module):
    """Keep the existing block and add one zero-initialized parallel residual.

    The same convolution is reused on each refinement iteration. Padding is
    ordinary zero feature padding; it does not encode walls or collision rules.
    HUD tokens receive no direct contribution from this local branch.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.local_norm = nn.LayerNorm(96)
        self.local_conv = nn.Conv2d(96,96,kernel_size=3,padding=1)
        nn.init.zeros_(self.local_conv.weight)
        nn.init.zeros_(self.local_conv.bias)

    def local_update(self,state):
        board=self.local_norm(state[:,:144]).reshape(-1,12,12,96).permute(0,3,1,2)
        return self.local_conv(board).permute(0,2,3,1).reshape(-1,144,96)

    def forward(self,state,source):
        refined=self.base(state,source)
        board=refined[:,:144]+self.local_update(state)
        return torch.cat((board,refined[:,144:]),dim=1)


class LocalGlobalGlyphTransition(GlobalGlyphTransition):
    """Global model unchanged except for wrapping its shared refinement block.

    Construct all original parameters before the local branch, preserving the
    same-seed global initialization. New normalization gradients are initially
    zero because the convolution is zero; convolution gradients are live on
    the first step, and normalization gradients become live once it moves.
    """
    checkpoint_format = LOCAL_GLYPH_FORMAT

    def __init__(self,config=None,**overrides):
        if config is None:config=LocalGlyphConfig(**overrides)
        elif isinstance(config,dict):config=LocalGlyphConfig(**(config|overrides))
        elif overrides or not isinstance(config,LocalGlyphConfig):raise ValueError('pass local config or keyword overrides')
        super().__init__(config.global_config())
        self.cfg=config
        self.block=LocalFieldBlock(self.block)

    def config(self):return asdict(self.cfg)

    def warmstart_from_global_state_dict(self,state_dict):
        """Validate all global keys/shapes before copying; preserve local params."""
        own=self.state_dict()
        local={key for key in own if key.startswith(('block.local_norm.','block.local_conv.'))}
        mapping={key:('block.'+key[len('block.base.'):] if key.startswith('block.base.') else key) for key in own if key not in local}
        if set(state_dict)!=set(mapping.values()):raise ValueError('warmstart requires exactly complete global glyph state')
        if any(own[key].shape!=state_dict[source].shape for key,source in mapping.items()):raise ValueError('global warmstart shape mismatch')
        result=self.load_state_dict({key:state_dict[source] for key,source in mapping.items()},strict=False)
        if set(result.missing_keys)!=local or result.unexpected_keys:raise RuntimeError('unexpected local warmstart mismatch')
        return sorted(local)
