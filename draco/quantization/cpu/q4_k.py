"""
DRACO_Q4_K — blockwise 4-bit quantization (Draco-native layout).

One block = 64 consecutive elements of a weight row. Per block:

    32 bytes  : 64 packed 4-bit values (low nibble first)
    2 bytes   : float16 scale  (scale = (max - min) / 15)
    2 bytes   : float16 min

Reconstruction:  w ~= nibble * scale + min

Total payload per [R, K] weight: R * (K / 64) * 36 bytes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from draco.exceptions import CPUQuantizationError

BLOCK_SIZE = 64
_QMAX = 15.0


def _pack_nibbles(q: np.ndarray) -> np.ndarray:
    even = q[..., 0::2].astype(np.uint8)
    odd = q[..., 1::2].astype(np.uint8)
    return (even | (odd << 4)).reshape(q.shape[0], -1)


def _unpack_nibbles(packed: np.ndarray, n: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    even = (packed & 0x0F).astype(np.uint8)
    odd = ((packed >> 4) & 0x0F).astype(np.uint8)
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = even.reshape(-1)
    out[1::2] = odd.reshape(-1)
    return out.astype(np.int8)


def quantize(w: np.ndarray):
    """
    Quantize a [R, K] float32 weight; K must be divisible by 64.

    Returns (payload uint8, None, None, None). Scales/mins are interleaved
    inside the payload, one block at a time.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise CPUQuantizationError(f"Q4_K requires a 2D weight, got {w.shape}")
    r, k = w.shape
    if k % BLOCK_SIZE != 0:
        raise CPUQuantizationError(f"Q4_K: K={k} not divisible by block size {BLOCK_SIZE}")
    nb = k // BLOCK_SIZE
    blocks = w.reshape(r, nb, BLOCK_SIZE)
    bmin = blocks.min(axis=-1).astype(np.float16)
    bmax = blocks.max(axis=-1).astype(np.float16)
    span = bmax.astype(np.float32) - bmin.astype(np.float32)
    scale = np.where(span > 1e-12, span / _QMAX, 1.0).astype(np.float16)
    q = np.clip(
        np.rint((blocks - bmin.astype(np.float32)[:, :, None]) / scale.astype(np.float32)[:, :, None]),
        0,
        _QMAX,
    ).astype(np.uint8)

    # interleave per block: nibbles (32) + scale (2) + min (2)
    nibbles = _pack_nibbles(q.reshape(r, nb * BLOCK_SIZE)).reshape(r, nb, 32)
    out = np.empty((r, nb, 36), dtype=np.uint8)
    out[:, :, 0:32] = nibbles
    out[:, :, 32:34] = scale.view(np.uint8).reshape(r, nb, 2)
    out[:, :, 34:36] = bmin.view(np.uint8).reshape(r, nb, 2)
    return out.reshape(-1), None, None, None


def dequantize(payload: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reconstruct float32 weights from Q4_K storage (shape = [R, K])."""
    r, k = shape
    nb = k // BLOCK_SIZE
    data = np.asarray(payload, dtype=np.uint8).reshape(r, nb, 36)
    nibbles = _unpack_nibbles(data[:, :, 0:32].reshape(r, -1), r * nb * BLOCK_SIZE)
    nibbles = nibbles.reshape(r, nb, BLOCK_SIZE).astype(np.float32)
    scale = data[:, :, 32:34].view(np.float16).astype(np.float32)
    bmin = data[:, :, 34:36].view(np.float16).astype(np.float32)
    return (nibbles * scale[:, :, None] + bmin[:, :, None]).reshape(r, k)


def layout_sizes(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    r, k = shape[0], int(np.prod(shape[1:]))
    return r * (k // BLOCK_SIZE) * 36, 0, 0, 0