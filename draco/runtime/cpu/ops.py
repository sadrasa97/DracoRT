"""
Portable numpy CPU operators for the native execution graph
(RMSNorm, RoPE, SwiGLU, softmax) — spec sections 13 and 37.

These are the correctness-oracle reference implementations. No ISA
intrinsics; they exist so a native .draco model can run end-to-end on
any host without a torch dependency.
"""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x32 = x.astype(np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 / np.sqrt(variance + eps)
    return (normed * weight).astype(x.dtype)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return silu(gate) * up


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: "np.ndarray | None" = None, eps: float = 1e-5) -> np.ndarray:
    x32 = x.astype(np.float32)
    mean = np.mean(x32, axis=-1, keepdims=True)
    var = np.mean((x32 - mean) ** 2, axis=-1, keepdims=True)
    normed = (x32 - mean) / np.sqrt(var + eps)
    out = normed * weight
    if bias is not None:
        out = out + bias
    return out.astype(x.dtype)


def gelu(x: np.ndarray) -> np.ndarray:
    # tanh approximation (GPT-2 "new gelu"), accurate to ~1e-3
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def build_alibi_slopes(num_heads: int) -> np.ndarray:
    def _slopes_power_of_2(n):
        start = 2.0 ** (-(2.0 ** -(np.log2(n) - 3)))
        return start * (start ** np.arange(n))

    if (num_heads & (num_heads - 1)) == 0:  # power of 2
        return _slopes_power_of_2(num_heads).astype(np.float32)
    closest_pow2 = 2 ** int(np.floor(np.log2(num_heads)))
    slopes = _slopes_power_of_2(closest_pow2)
    extra = _slopes_power_of_2(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
    return np.concatenate([slopes, extra]).astype(np.float32)


def build_alibi_bias(num_heads: int, key_positions: np.ndarray, query_positions: np.ndarray) -> np.ndarray:
    """Returns (num_heads, num_queries, num_keys) additive bias, causal-masked."""
    slopes = build_alibi_slopes(num_heads)
    rel = key_positions[None, :] - query_positions[:, None]  # (q, k), <=0 for causal
    bias = slopes[:, None, None] * rel[None, :, :]
    mask = key_positions[None, None, :] > query_positions[None, :, None]
    return np.where(mask, -np.inf, bias).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def build_rope_cache(head_dim: int, max_positions: int, theta: float = 10000.0) -> "tuple[np.ndarray, np.ndarray]":
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_positions, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # (max_positions, head_dim // 2)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """x: (seq_len, num_heads, head_dim). positions: (seq_len,) absolute positions."""
    seq_len, num_heads, head_dim = x.shape
    half = head_dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[positions][:, None, :]  # (seq_len, 1, half)
    s = sin[positions][:, None, :]
    rotated_x1 = x1 * c - x2 * s
    rotated_x2 = x2 * c + x1 * s
    return np.concatenate([rotated_x1, rotated_x2], axis=-1).astype(x.dtype)
