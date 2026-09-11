"""
DRACO_Q5_K — blockwise 5-bit quantization (Draco-native layout).

One block = 64 consecutive elements of a weight row. Per block:

    32 bytes  : low 4 bits of each of 64 values
    8 bytes   : high (5th) bit of each of 64 values
    2 bytes   : float16 scale  (scale = (max - min) / 31)
    2 bytes   : float16 min

Reconstruction:  w ~= value * scale + min   (value in 0..31)

Total payload per [R, K] weight: R * (K / 64) * 44 bytes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from draco.exceptions import CPUQuantizationError

BLOCK_SIZE = 64
_QMAX = 31.0


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
    return out.astype(np.uint8)


def quantize(w: np.ndarray):
    """
    Quantize a [R, K] float32 weight; K must be divisible by 64.

    Returns (payload uint8, None, None, None).
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise CPUQuantizationError(f"Q5_K requires a 2D weight, got {w.shape}")
    r, k = w.shape
    if k % BLOCK_SIZE != 0:
        raise CPUQuantizationError(f"Q5_K: K={k} not divisible by block size {BLOCK_SIZE}")
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

    low = _pack_nibbles(q.reshape(r, nb * BLOCK_SIZE)).reshape(r, nb, 32)
    high = ((q >> 4) & 0x01).reshape(r, nb, BLOCK_SIZE)
    # pack 64 high bits -> 8 bytes (bit i in byte i//8, bit i%8)
    high_packed = np.packbits(high.reshape(r, nb, 64), axis=-1)  # [r, nb, 8]
    out = np.empty((r, nb, 44), dtype=np.uint8)
    out[:, :, 0:32] = low
    out[:, :, 32:40] = high_packed
    out[:, :, 40:42] = scale.view(np.uint8).reshape(r, nb, 2)
    out[:, :, 42:44] = bmin.view(np.uint8).reshape(r, nb, 2)
    return out.reshape(-1), None, None, None


def dequantize(payload: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reconstruct float32 weights from Q5_K storage (shape = [R, K])."""
    r, k = shape
    nb = k // BLOCK_SIZE
    data = np.asarray(payload, dtype=np.uint8).reshape(r, nb, 44)
    low = _unpack_nibbles(data[:, :, 0:32].reshape(r, -1), r * nb * BLOCK_SIZE)
    low = low.reshape(r, nb, BLOCK_SIZE)
    high = np.unpackbits(data[:, :, 32:40], axis=-1).reshape(r, nb, BLOCK_SIZE)
    values = (low | (high << 4)).astype(np.float32)
    scale = data[:, :, 40:42].view(np.float16).astype(np.float32)
    bmin = data[:, :, 42:44].view(np.float16).astype(np.float32)
    return (values * scale[:, :, None] + bmin[:, :, None]).reshape(r, k)


def layout_sizes(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    r, k = shape[0], int(np.prod(shape[1:]))
    return r * (k // BLOCK_SIZE) * 44, 0, 0, 0