"""
DRACO_I8_ASYM — per-row asymmetric INT8 weight-only quantization.

Complements int8.py (symmetric). Asymmetric quantization uses the full
[0, 255] range with a zero-point, which gives better accuracy than
symmetric INT8 on weight distributions that aren't centered near zero
(e.g. post-GELU/SiLU bias-shifted layers) at the cost of one extra
zero-point array per tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draco.exceptions import CPUQuantizationError

NAME = "int8_asym"


@dataclass
class Int8AsymQuantizedWeight:
    qweight: np.ndarray  # uint8, same shape as source
    scale: np.ndarray  # float32, one per row
    zero_point: np.ndarray  # float32, one per row (in original value units)


def quantize(weight: np.ndarray) -> Int8AsymQuantizedWeight:
    if weight.ndim != 2:
        raise CPUQuantizationError(
            f"int8_asym quantization expects a 2D weight matrix, got shape {weight.shape}."
        )
    w = weight.astype(np.float32, copy=False)
    row_min = np.min(w, axis=1)
    row_max = np.max(w, axis=1)
    span = np.where((row_max - row_min) == 0, 1.0, row_max - row_min)
    scale = (span / 255.0).astype(np.float32)
    zero_point = row_min.astype(np.float32)
    q = np.round((w - zero_point[:, None]) / scale[:, None])
    qweight = np.clip(q, 0, 255).astype(np.uint8)
    return Int8AsymQuantizedWeight(qweight=qweight, scale=scale, zero_point=zero_point)


def dequantize(q: Int8AsymQuantizedWeight) -> np.ndarray:
    return (q.qweight.astype(np.float32) * q.scale[:, None] + q.zero_point[:, None]).astype(np.float32)


def matmul(activation: np.ndarray, q: Int8AsymQuantizedWeight) -> np.ndarray:
    w = dequantize(q)
    return activation.astype(np.float32, copy=False) @ w.T


def roundtrip_error(weight: np.ndarray) -> dict:
    q = quantize(weight)
    recon = dequantize(q)
    diff = recon - weight.astype(np.float32)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    w32 = weight.astype(np.float32)
    scale_floor = max(float(np.std(w32)) * 0.05, 1e-6)
    denom = np.maximum(np.abs(w32), scale_floor)
    rel = float(np.mean(np.abs(diff) / denom))
    return {"max_abs_error": max_abs, "mean_abs_error": mean_abs, "mean_rel_error": rel}
