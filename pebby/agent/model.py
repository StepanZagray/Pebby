"""Legacy CNN policy and architecture-aware checkpoint IO.

New training defaults to ``looped.LoopedLs20Policy``. This CNN stays available
as a baseline and to load existing weights exactly. The observations below
motivated that baseline; failures of earlier looped implementations do not
rule out shared transformer depth for LS20.

Three decisions here are inherited from measurements, not taste.

*Cell alignment.* Play happens on a fixed 12x12 lattice of 5x5 cells inset at
``names.X_ORIGIN``/``names.Y_ORIGIN``. Cropping the playfield to 60x60 and
hitting it with a 5x5 stride-5 convolution lands exactly one feature vector on
each cell, so the residual trunk reasons about cells instead of about pixels
that happen to straddle a cell boundary. The lattice is fixed by the game's own
geometry, so this is an architectural prior, not knowledge of any level.

*No auxiliary head, no memory.* An actions-to-completion head was tried and
dropped: auxiliary prediction and world-model heads have measured failures on
this machine's earlier ARC-AGI-3 work (tofy-py ``docs/COLOUR_AXIS_RESULTS.md``,
all four preregistered gates failed; ``docs/RECURRENT_WORLD_MODEL.md``, 0/16
exact on fresh dev). Recurrence was also measured to change action count
without changing win rate. So: pure action cross-entropy over the current
frame. ``to_go`` is still written into the shards for analysis, but nothing
here consumes it.

*Small.* Controllers at 309k, 2.03M and 7.81M parameters all scored 128/128 in
``docs/MODEL_SIZE_AND_EVENT_POLICY_RESULTS.md`` -- width bought nothing, not
even action efficiency -- so the default lands near 0.3M rather than filling the
card.

*The carried triple can be broadcast into the trunk.* Deciding "may I step on
this pad" means comparing the carried triple, drawn in the HUD at rows 55-60
cols 3-8, against a 3x3 goal icon drawn on the pad cell itself, rotation
included. Those two regions are far apart, and a late concat forces the network
to learn that comparison independently for every pad position. With
``broadcast_hud`` the HUD vector is projected and tiled across all 12x12 cells
before the residual stack, so "does my triple match the icon in THIS cell"
becomes one local operation applied everywhere. The 5x5 stride-5 stem sees all
25 pixels of a cell at once, so the icon detail needed for that comparison
survives the projection.

*Colour is a task attribute.* One-hot planes, one per palette index, so every
colour gets its own weights. Deliberately NOT a colour-shared or
colour-equivariant encoder: there is an executable witness that pooling over a
colour axis makes the action logits bitwise identical under a colour swap, and
in LS20 colour is one third of the goal triple, not a nuisance symmetry.
"""

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..ls20 import names

# Bumped whenever the meaning of the weights changes, so an old checkpoint fails
# loudly instead of being silently reinterpreted. v2 dropped the value head.
MODEL_FORMAT = "pebby.ls20-policy.v2"

PALETTE = 16  # Frames are colour indices 0..15; measurement finds 10 in use.
ACTION_COUNT = len(names.ACTION_IDS)

# Playfield crop: 12 cells of 5 pixels from each origin. Measured over 7,672
# rendered frames, no playfield sprite ever falls outside rows 0..59, cols 4..63,
# and cols 0..3 are a constant margin, so the crop discards only constants.
PLAY_TOP = names.Y_ORIGIN
PLAY_BOTTOM = names.Y_ORIGIN + names.CELL * names.GRID_ROWS
PLAY_LEFT = names.X_ORIGIN
PLAY_RIGHT = names.X_ORIGIN + names.CELL * names.GRID_COLS
# HUD crop. Row 52 is the topmost HUD pixel (the chrome box border) and rows
# 52..59 of cols 12..63 are still playfield, which is harmless duplication. What
# matters is inside: the carried token at rows 55-60 cols 3-8, the step bar at
# rows 61-62 cols 13-54 (one column per remaining step) and three two-pixel life
# pips at cols 56-57, 59-60, 62-63.
HUD_TOP = 52
HUD_BOTTOM = names.FRAME_SIZE
HUD_COLUMNS = 16  # Column bins kept after pooling; see Ls20Policy.hud.


def _norm(channels):
    """GroupNorm, not BatchNorm: a batch of one in a test must behave like a
    batch of 128 on the GPU. The group count adapts so small configs build."""
    return nn.GroupNorm(math.gcd(8, channels), channels)


def _channels(patch):
    """[B, H, W] colour indices -> [B, 16, H, W] one-hot planes."""
    return F.one_hot(patch.long(), num_classes=PALETTE).permute(0, 3, 1, 2).float()


class ResidualBlock(nn.Module):
    """Post-activation residual block, AlphaZero-style, at the cell resolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _norm(channels)
        # Starting every block as the identity keeps the trunk stable at step 1.
        nn.init.zeros_(self.norm2.weight)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        return F.relu(x + self.norm2(self.conv2(y)))


class Ls20Policy(nn.Module):
    """One frame in, four action logits out. Memoryless by design."""

    def __init__(self, channels=48, blocks=4, hud_channels=32, reduce_channels=8, hidden=64,
                 broadcast_hud=False, condition_channels=32):
        super().__init__()
        self.hyper = {"channels": channels, "blocks": blocks, "hud_channels": hud_channels,
                      "reduce_channels": reduce_channels, "hidden": hidden,
                      "broadcast_hud": broadcast_hud, "condition_channels": condition_channels}
        # Stride 5 with kernel 5: every output pixel sees exactly one game cell.
        self.stem = nn.Sequential(
            nn.Conv2d(PALETTE, channels, names.CELL, stride=names.CELL, bias=False),
            _norm(channels), nn.ReLU())
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        # 1x1 down-projection before flattening. Every one of the 144 positions
        # is kept -- where the player stands relative to a pad is the whole game,
        # and global pooling would erase it -- but flattening the full trunk
        # width would put more parameters in one matrix than the rest combined.
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, reduce_channels, 1, bias=False),
            _norm(reduce_channels), nn.ReLU())
        self.hud = nn.Sequential(
            nn.Conv2d(PALETTE, hud_channels // 2, 3, padding=1, bias=False),
            _norm(hud_channels // 2), nn.ReLU(),
            nn.Conv2d(hud_channels // 2, hud_channels, 3, stride=2, padding=1, bias=False),
            _norm(hud_channels), nn.ReLU(),
            # Pool the rows away, keep the columns: the budget is drawn as a bar
            # whose filled run is `55 - steps_left .. 54`, so the number IS a
            # horizontal position and averaging across columns would destroy it.
            nn.AdaptiveAvgPool2d((1, HUD_COLUMNS)))
        hud_features = hud_channels * HUD_COLUMNS
        # Project the HUD vector down to a few planes and tile them over the
        # lattice, so the trunk can compare the carried triple against each
        # cell's goal icon locally instead of memorising every pad position.
        self.condition = nn.Sequential(nn.Linear(hud_features, condition_channels), nn.ReLU()) \
            if broadcast_hud else None
        self.fuse = nn.Sequential(nn.Conv2d(channels + condition_channels, channels, 1, bias=False),
                                  _norm(channels), nn.ReLU()) if broadcast_hud else None
        play_features = reduce_channels * names.GRID_ROWS * names.GRID_COLS
        self.head = nn.Sequential(
            nn.Linear(play_features + hud_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.action = nn.Linear(hidden // 2, ACTION_COUNT)

    def forward(self, frames):
        """`frames` is [B, 64, 64] of colour indices. Returns logits [B, 4]."""
        if frames.dim() != 3 or frames.shape[-2:] != (names.FRAME_SIZE, names.FRAME_SIZE):
            raise ValueError(f"frames must be [B, {names.FRAME_SIZE}, {names.FRAME_SIZE}], got {tuple(frames.shape)}")
        # Crop first, then one-hot: expanding the whole frame to 16 planes before
        # throwing most of it away costs real memory at training batch sizes.
        cells = self.stem(_channels(frames[:, PLAY_TOP:PLAY_BOTTOM, PLAY_LEFT:PLAY_RIGHT]))
        hud = self.hud(_channels(frames[:, HUD_TOP:HUD_BOTTOM, :])).flatten(1)
        if self.condition is not None:
            plane = self.condition(hud)[:, :, None, None].expand(-1, -1, *cells.shape[-2:])
            cells = self.fuse(torch.cat([cells, plane], dim=1))
        play = self.reduce(self.trunk(cells))
        return self.action(self.head(torch.cat([play.flatten(1), hud], dim=1)))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def config(self):
        """Exactly the kwargs needed to rebuild this network from a checkpoint."""
        return dict(self.hyper)


def build_policy(config=None):
    """Legacy configs have no discriminator; never reinterpret their weights."""
    config = dict(config or {})
    architecture = config.pop("architecture", "cnn")
    if architecture == "cnn":
        return Ls20Policy(**config)
    if architecture == "looped":
        from .looped import LoopedLs20Policy
        return LoopedLs20Policy(**config)
    if architecture == "world":
        from .world_model import build_world_policy
        return build_world_policy(config)
    raise ValueError(f"unknown policy architecture: {architecture!r}")


def frames_to_tensor(frames, device=None):
    """One 64x64 frame or a batch of them -> int64 [B, 64, 64], ready for forward."""
    tensor = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
    if tensor.dim() == 2:
        tensor = tensor[None]
    return tensor.to(device=device, dtype=torch.int64)


def save_checkpoint(path, model, **metadata):
    """Write weights plus everything needed to rebuild and audit the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reserved = {"format", "config", "parameters", "weights"} & metadata.keys()
    if reserved:
        raise ValueError(f"reserved checkpoint metadata: {sorted(reserved)}")
    checkpoint = {"format": getattr(model, "checkpoint_format", MODEL_FORMAT), "config": model.config(),
                  "parameters": model.parameter_count(), **metadata,
                  "weights": model.state_dict()}
    # Fail here rather than at load time if a caller passes something exotic.
    json.dumps({key: value for key, value in checkpoint.items() if key != "weights"}, allow_nan=False)
    torch.save(checkpoint, path)
    return checkpoint


def load_checkpoint(path, device="cpu"):
    """Rebuild the saved network in eval mode. Returns (model, checkpoint)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") == "pebby.ls20-spatial-outcome-policy.v1":
        from .spatial_outcome_policy import load_checkpoint as load_spatial_checkpoint
        return load_spatial_checkpoint(path, device)
    if checkpoint.get("format") == "pebby.ls20-spatial-outcome-policy.v2":
        from .spatial_v2_policy import load_checkpoint as load_spatial_v2_checkpoint
        return load_spatial_v2_checkpoint(path, device)
    if checkpoint.get("format") == "pebby.ls20-spatial-route-outcome-policy.v1":
        from .spatial_route_outcome_policy import load_checkpoint as load_route_checkpoint
        return load_route_checkpoint(path, device)
    if checkpoint.get("format") == "pebby.ls20-spatial-semantic-outcome-policy.v1":
        from .spatial_semantic_outcome_policy import load_checkpoint as load_semantic_checkpoint
        return load_semantic_checkpoint(path, device)
    if checkpoint.get("format") == "pebby.structured-workspace-readout.v1":
        from .structured_workspace_controller import load_workspace_policy_checkpoint
        return load_workspace_policy_checkpoint(path, device)
    if checkpoint.get("format") == "pebby.structured-search-policy.v1":
        from .structured_search_policy import load_search_policy_checkpoint
        return load_search_policy_checkpoint(path, device)
    if checkpoint.get("format") == "pebby.structured-factored-field-policy.v1":
        from .structured_factored_policy import load_factored_policy_checkpoint
        return load_factored_policy_checkpoint(path, device)
    if checkpoint.get("format") == "pebby.structured-field-policy.v1":
        config = checkpoint.get("config")
        if not isinstance(config, dict) or config.get("architecture") != "structured":
            raise ValueError("checkpoint format and architecture disagree")
        from .structured_policy import load_structured_policy_checkpoint
        return load_structured_policy_checkpoint(path, device)
    from .looped import LOOPED_MODEL_FORMAT
    from .world_model import WORLD_MODEL_FORMAT
    formats = {MODEL_FORMAT: "cnn", LOOPED_MODEL_FORMAT: "looped", WORLD_MODEL_FORMAT: "world"}
    if checkpoint.get("format") not in formats:
        raise ValueError(f"checkpoint formats differ: unsupported {checkpoint.get('format')!r}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a dictionary")
    architecture = config.get("architecture", "cnn")
    if architecture != formats[checkpoint["format"]]:
        raise ValueError("checkpoint format and architecture disagree")
    model = build_policy(config)
    model.load_state_dict(checkpoint["weights"])
    return model.to(device).eval(), checkpoint
