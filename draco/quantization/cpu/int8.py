"""
DRACO_I8_SYM — per-row symmetric INT8 weight-only quantization (CPU-native).

This is the portable reference codec: numpy in, numpy out, no ISA-specific
instructions. It is the correctness oracle that any future AVX2/AVX512/VNNI
kernel must match within tolerance (see draco.runtime.cpu.kernels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draco.exceptions import CPUQuantizationError

NAME = "int8_sym"


@dataclass
class Int8QuantizedWeight:
    qweight: np.ndarray  # int8, same shape as source
    scale: np.ndarray  # float32, one scale per row (axis 0)


def quantize(weight: np.ndarray) -> Int8QuantizedWeight:
    """Per-row symmetric INT8 quantization of a 2D weight matrix.

    scale[i] = max(|weight[i, :]|) / 127
    qweight[i, j] = round(weight[i, j] / scale[i])
    """
    if weight.ndim != 2:
        raise CPUQuantizationError(
            f"int8_sym quantization expects a 2D weight matrix, got shape {weight.shape}."
        )
    w = weight.astype(np.float32, copy=False)
    row_max = np.max(np.abs(w), axis=1)
    row_max = np.where(row_max == 0, 1.0, row_max)  # avoid div-by-zero for all-zero rows
    scale = (row_max / 127.0).astype(np.float32)
    qweight = np.round(w / scale[:, None]).astype(np.int8)
    return Int8QuantizedWeight(qweight=qweight, scale=scale)


def dequantize(q: Int8QuantizedWeight) -> np.ndarray:
    return (q.qweight.astype(np.float32) * q.scale[:, None]).astype(np.float32)


def matmul(activation: np.ndarray, q: Int8QuantizedWeight) -> np.ndarray:
    """activation @ weight.T using the quantized weight.

    Reference (portable) implementation: dequantize then matmul. This is the
    numerically-correct fallback path from section 20 of the CPU-runtime
    spec — a fused int8 GEMM that never materializes the full dequantized
    matrix needs a compiled kernel, which is out of scope for the pure-numpy
    reference tier.
    """
    w = dequantize(q)
    return activation.astype(np.float32, copy=False) @ w.T


def roundtrip_error(weight: np.ndarray) -> dict:
    """Numerical-validation helper (spec section 10 / 48)."""
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
