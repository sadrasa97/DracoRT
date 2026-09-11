"""
BF16 storage codec (not a quantization format — a storage dtype).

numpy has no native bfloat16 dtype (pre-2.0), so bf16 is stored as the
top 16 bits of each float32's bit pattern (round-to-nearest-even),
viewed/stored as uint16. This halves storage vs f32 while keeping f32's
full exponent range (unlike f16, which can overflow on LLM activations/
weights with values outside f16's narrower range) — the standard reason
bf16 is preferred over f16 for ML weights.
"""

from __future__ import annotations

import numpy as np


def encode(weight: np.ndarray) -> np.ndarray:
    """float32 array -> uint16 array of bf16 bit patterns (round-to-nearest-even)."""
    w = np.ascontiguousarray(weight, dtype=np.float32)
    bits = w.view(np.uint32)
    # round-to-nearest-even on the bits we're about to drop
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    rounded = bits + rounding_bias
    return (rounded >> 16).astype(np.uint16)


def decode(bf16_bits: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32 array."""
    bits = bf16_bits.astype(np.uint32) << 16
    return bits.view(np.float32)
