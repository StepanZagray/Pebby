"""Deterministic HUD readout: exact remaining steps and lives from public pixels.

The learned encoder average-pools the 42-column step bar into 16 HUD tokens,
so it only knows the remaining budget to roughly four-step granularity. The
bar is public observation with one pixel column per remaining step, so the
same information can be read exactly, with no model. ``tests/test_hud_decoder``
checks these constants against ``Ls20Env.steps_left()``/``lives()`` on frames
rendered by the engine across many states.

Frame geometry, verified against the engine's HUD painter
(``third_party/ls20/ls20.py`` ``render_interface``) and ``Ls20Env.render``:

* Step bar: rows 61-62, columns 13..54 (one column per step; every shipped
  level and every Pebby generator uses a 42-step tank). A filled column is
  colour 11, an empty one colour 3, and the surrounding chrome is 5. With
  ``n >= 0`` steps left the filled run is columns ``55 - n .. 54``; the bar
  empties from the left, so the count of colour-11 columns equals ``n``.
* Life pips: three two-pixel pips at columns 56-57, 59-60 and 62-63 on the
  same rows; lit is colour 8, extinguished is colour 3, and they go out
  right to left, so the count of lit pips equals the remaining lives.
* ``steps_left`` is -1 only transiently (game over/last move); the bar is
  empty then and the decoder reads 0.
* During a death the engine emits a few frames of uniform colour 11 with no
  HUD drawn. Observation frames handed to a policy are taken after that
  flash, so they always carry the HUD; a flash frame would decode as a full
  bar with zero pips.

Output layout of ``decode_hud`` (``HUD_SCALARS`` = 4 columns, float32)::

    [steps_left / 42,  lives == 1,  lives == 2,  lives == 3]

Column 0 lies in [0, 1]; columns 1..3 are a one-hot of the surviving life
count (all zero only for a frame with no pips, i.e. after the last life).
"""
import numpy as np
import torch

HUD_SCALARS = 4
STEP_ROWS = (61, 63)          # rows 61 and 62
STEP_COLUMNS = (13, 55)       # columns 13..54 inclusive: 42 steps
MAX_STEPS = STEP_COLUMNS[1] - STEP_COLUMNS[0]
LIFE_COLUMNS = ((56, 58), (59, 61), (62, 64))
STEP_FILLED = 11
PIP_LIT = 8


def _last_frames(frames, library):
    """Return the current frame per batch row as [B, 64, 64] using ``library`` (torch or numpy)."""
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim == 4:
        frames = frames[:, -1]
    if frames.ndim != 3 or tuple(frames.shape[-2:]) != (64, 64):
        raise ValueError(f'frames must be [64,64], [B,64,64] or [B,H,64,64], got {tuple(frames.shape)}')
    if library is torch:
        if frames.is_floating_point() or frames.dtype == torch.bool:
            raise ValueError('frames must be integer palette indices')
    elif not np.issubdtype(frames.dtype, np.integer):
        raise ValueError('frames must be integer palette indices')
    return frames


def _decode(frames, library):
    frames = _last_frames(frames, library)
    top, bottom = STEP_ROWS
    left, right = STEP_COLUMNS
    bar = frames[:, top:bottom, left:right] == STEP_FILLED
    # Both bar rows carry the same run; require agreement per column and count.
    filled = (bar[:, 0] & bar[:, 1]).sum(-1)
    lives = sum((frames[:, top:bottom, a:b] == PIP_LIT).reshape(frames.shape[0], -1).all(-1)
                for a, b in LIFE_COLUMNS)
    return filled, lives


def decode_hud(frames):
    """Exact HUD scalars from integer frames; returns float32 ``[B, HUD_SCALARS]`` on the input device."""
    frames = frames if torch.is_tensor(frames) else torch.as_tensor(np.asarray(frames))
    filled, lives = _decode(frames, torch)
    steps = filled.to(torch.float32).clamp_(0, MAX_STEPS) / MAX_STEPS
    pips = torch.stack([(lives == count) for count in (1, 2, 3)], -1).to(torch.float32)
    return torch.cat((steps[:, None], pips), -1)


def decode_hud_numpy(frames):
    """``decode_hud`` for collectors holding numpy frames; returns float32 ``[B, HUD_SCALARS]``."""
    filled, lives = _decode(np.asarray(frames), np)
    steps = np.clip(filled.astype(np.float32), 0, MAX_STEPS) / MAX_STEPS
    pips = np.stack([(lives == count) for count in (1, 2, 3)], -1).astype(np.float32)
    return np.concatenate((steps[:, None], pips), -1)


def decode_counts(frames):
    """Integer ``(steps_left, lives)`` per frame, for tests and diagnostics."""
    frames = frames if torch.is_tensor(frames) else torch.as_tensor(np.asarray(frames))
    filled, lives = _decode(frames, torch)
    return filled.long(), lives.long()
