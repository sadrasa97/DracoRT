"""
DRACO_I4_GROUPWISE — groupwise asymmetric INT4 weight-only quantization
(CPU-native, portable reference codec).

Two int4 values are packed per byte. Group size defaults to 32 columns,
matching common CPU-friendly block sizes (llama.cpp-style Q4 blocks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draco.exceptions import CPUQuantizationError

NAME = "int4_groupwise"
DEFAULT_GROUP_SIZE = 32


@dataclass
class Int4QuantizedWeight:
    packed: np.ndarray  # uint8, shape (rows, cols // 2), two nibbles per byte
    scale: np.ndarray  # float32, shape (rows, n_groups)
    zero_point: np.ndarray  # float32, shape (rows, n_groups)
    shape: tuple  # original (rows, cols)
    group_size: int


def quantize(weight: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE) -> Int4QuantizedWeight:
    if weight.ndim != 2:
        raise CPUQuantizationError(
            f"int4_groupwise quantization expects a 2D weight matrix, got shape {weight.shape}."
        )
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise CPUQuantizationError(
            f"int4_groupwise: cols ({cols}) must be divisible by group_size ({group_size})."
        )
    if cols % 2 != 0:
        raise CPUQuantizationError("int4_groupwise: cols must be even to pack two nibbles/byte.")

    w = weight.astype(np.float32, copy=False)
    n_groups = cols // group_size
    w_grouped = w.reshape(rows, n_groups, group_size)

    g_min = w_grouped.min(axis=2)
    g_max = w_grouped.max(axis=2)
    span = np.where((g_max - g_min) == 0, 1.0, g_max - g_min)
    scale = (span / 15.0).astype(np.float32)  # unsigned 4-bit: 0..15
    zero_point = g_min.astype(np.float32)

    q = np.round((w_grouped - zero_point[:, :, None]) / scale[:, :, None])
    q = np.clip(q, 0, 15).astype(np.uint8)
    q = q.reshape(rows, cols)

    # pack two nibbles per byte along the column axis
    low = q[:, 0::2] & 0x0F
    high = (q[:, 1::2] & 0x0F) << 4
    packed = (low | high).astype(np.uint8)

    return Int4QuantizedWeight(
        packed=packed, scale=scale, zero_point=zero_point, shape=(rows, cols), group_size=group_size
    )


def dequantize(q: Int4QuantizedWeight) -> np.ndarray:
    rows, cols = q.shape
    n_groups = cols // q.group_size

    low = q.packed & 0x0F
    high = (q.packed >> 4) & 0x0F
    unpacked = np.empty((rows, cols), dtype=np.uint8)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high

    unpacked_grouped = unpacked.reshape(rows, n_groups, q.group_size).astype(np.float32)
    w = unpacked_grouped * q.scale[:, :, None] + q.zero_point[:, :, None]
    return w.reshape(rows, cols)


def matmul(activation: np.ndarray, q: Int4QuantizedWeight) -> np.ndarray:
    """Reference (portable) matmul: dequantize then GEMM. See int8.matmul docstring."""
    w = dequantize(q)
    return activation.astype(np.float32, copy=False) @ w.T


def roundtrip_error(weight: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE) -> dict:
    q = quantize(weight, group_size=group_size)
    recon = dequantize(q)
    diff = recon - weight.astype(np.float32)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    w32 = weight.astype(np.float32)
    scale_floor = max(float(np.std(w32)) * 0.05, 1e-6)
    denom = np.maximum(np.abs(w32), scale_floor)
    rel = float(np.mean(np.abs(diff) / denom))
    return {"max_abs_error": max_abs, "mean_abs_error": mean_abs, "mean_rel_error": rel}
