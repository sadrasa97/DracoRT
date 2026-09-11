"""
DRACO_Q6_K — blockwise symmetric 6-bit quantization (Draco-native layout).

One block = 64 consecutive elements of a weight row. Per block:

    48 bytes  : 64 packed 6-bit values (signed -32..31, stored +32 -> 0..63)
    2 bytes   : float16 scale  (scale = amax / 32)

Reconstruction:  w ~= (value - 32) * scale

Total payload per [R, K] weight: R * (K / 64) * 50 bytes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from draco.exceptions import CPUQuantizationError

BLOCK_SIZE = 64
_QMAX = 32.0


def _unpack_6bit(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack n 6-bit values from bytes, LSB-first."""
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        bit = i * 6
        byte = bit // 8
        shift = bit % 8
        lo = packed[byte] >> shift
        hi = packed[byte + 1] << (8 - shift) if shift > 2 else 0
        out[i] = (lo | hi) & 0x3F
    return out


def quantize(w: np.ndarray):
    """
    Quantize a [R, K] float32 weight; K must be divisible by 64.

    Returns (payload uint8, None, None, None).
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise CPUQuantizationError(f"Q6_K requires a 2D weight, got {w.shape}")
    r, k = w.shape
    if k % BLOCK_SIZE != 0:
        raise CPUQuantizationError(f"Q6_K: K={k} not divisible by block size {BLOCK_SIZE}")
    nb = k // BLOCK_SIZE
    blocks = w.reshape(r, nb, BLOCK_SIZE)
    amax = np.max(np.abs(blocks), axis=-1)
    scale = np.maximum(amax, 1e-12) / _QMAX
    q = np.clip(np.rint(blocks / scale[:, :, None]), -32, 31).astype(np.int16)
    stored = (q + 32).astype(np.uint8)

    payload = np.empty(r * nb * 50, dtype=np.uint8)
    out = payload.reshape(r, nb, 50)
    for b in range(nb):
        out[:, b, 0:48] = _pack_6bit_rows(stored[:, b, :])  # [r, 48]
    out[:, :, 48:50] = scale.astype(np.float16).view(np.uint8).reshape(r, nb, 2)
    return payload, None, None, None


def _pack_6bit_rows(rows: np.ndarray) -> np.ndarray:
    """Pack [r, 64] uint8 values into [r, 48] bytes."""
    r = rows.shape[0]
    out = np.zeros((r, 48), dtype=np.uint8)
    for i in range(BLOCK_SIZE):
        v = rows[:, i].astype(np.uint16) & 0x3F
        bit = i * 6
        byte = bit // 8
        shift = bit % 8
        out[:, byte] |= ((v << shift) & 0xFF).astype(np.uint8)
        if shift > 2:
            out[:, byte + 1] |= (v >> (8 - shift)).astype(np.uint8)
    return out


def dequantize(payload: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reconstruct float32 weights from Q6_K storage (shape = [R, K])."""
    r, k = shape
    nb = k // BLOCK_SIZE
    data = np.asarray(payload, dtype=np.uint8).reshape(r, nb, 50)
    scale = data[:, :, 48:50].view(np.float16).astype(np.float32)
    vals = np.empty((r, nb, BLOCK_SIZE), dtype=np.int16)
    for b in range(nb):
        for i in range(BLOCK_SIZE):
            bit = i * 6
            byte = bit // 8
            shift = bit % 8
            lo = data[:, b, byte].astype(np.uint16) >> shift
            hi = data[:, b, byte + 1].astype(np.uint16) << (8 - shift) if shift > 2 else 0
            vals[:, b, i] = ((lo | hi) & 0x3F).astype(np.int16)
    return ((vals.astype(np.float32) - 32.0) * scale[:, :, None]).reshape(r, k)


def layout_sizes(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    r, k = shape[0], int(np.prod(shape[1:]))
    return r * (k // BLOCK_SIZE) * 50, 0, 0, 0