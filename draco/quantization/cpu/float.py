"""
Floating-point storage helpers for Draco CPU quantization.

bfloat16 is stored as raw uint16 bit patterns (numpy has no native bf16
dtype in general use), so conversions are explicit and vectorized.
"""

from __future__ import annotations

import numpy as np


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Convert float32 array to bfloat16 bit patterns (uint16, round-to-nearest-even)."""
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    rounding = 0x7FFF + ((u32 >> 16) & 1)
    return ((u32 + rounding) >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Convert bfloat16 bit patterns (uint16) to float32."""
    bits = np.asarray(bits, dtype=np.uint16)
    u32 = bits.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_f16(x: np.ndarray) -> np.ndarray:
    """Convert float32 to float16."""
    return np.asarray(x, dtype=np.float32).astype(np.float16)


def f16_to_f32(x: np.ndarray) -> np.ndarray:
    """Convert float16 to float32."""
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def bf16_bytes_from_f32(x: np.ndarray) -> bytes:
    """Serialize float32 array as bf16 bytes for .draco storage."""
    return f32_to_bf16(x).tobytes()


def f16_bytes_from_f32(x: np.ndarray) -> bytes:
    """Serialize float32 array as f16 bytes for .draco storage."""
    return f32_to_f16(x).tobytes()