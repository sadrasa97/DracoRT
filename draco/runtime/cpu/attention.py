"""
CPU attention backend (spec section 22).

Deliberately separate from any CUDA PagedAttention implementation — this
operates directly on the block-table storage in draco.runtime.cpu.kv_cache
and is pure numpy. Prefill and decode are implemented as distinct paths
since their performance characteristics (and, in a future optimized
backend, their kernel choice) differ.
"""

from __future__ import annotations

import numpy as np

from draco.runtime.cpu.kv_cache import CPUKVCache
from draco.runtime.cpu.ops import softmax


class CPUAttentionBackend:
    def __init__(self, kv_cache: CPUKVCache, num_query_heads: int) -> None:
        self.kv_cache = kv_cache
        self.num_query_heads = num_query_heads
        self.num_kv_heads = kv_cache.config.num_key_value_heads
        assert num_query_heads % self.num_kv_heads == 0, (
            "num_query_heads must be a multiple of num_key_value_heads for GQA repeat."
        )
        self._group_size = num_query_heads // self.num_kv_heads

    def _repeat_kv(self, kv: np.ndarray) -> np.ndarray:
        # kv: (seq_len, num_kv_heads, head_dim) -> (seq_len, num_query_heads, head_dim)
        if self._group_size == 1:
            return kv
        return np.repeat(kv, self._group_size, axis=1)

    def prefill(
        self, seq_id: int, layer: int, q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True,
        extra_bias: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """q, k, v: (seq_len, num_heads_or_kv_heads, head_dim). Writes k/v into
        the KV cache and returns attention output (seq_len, num_query_heads, head_dim).

        extra_bias, if given, is an additive (num_query_heads, seq_len, total_k)
        bias applied to raw attention scores before softmax (e.g. ALiBi) — it
        is expected to already encode any causal masking, so ``causal`` is
        ignored when extra_bias is provided.
        """
        seq_len = q.shape[0]
        for t in range(seq_len):
            self.kv_cache.append(seq_id, layer, k[t], v[t])

        cached_k, cached_v = self.kv_cache.get(seq_id, layer)
        cached_k = self._repeat_kv(cached_k.astype(np.float32))
        cached_v = self._repeat_kv(cached_v.astype(np.float32))

        head_dim = q.shape[-1]
        scale = 1.0 / np.sqrt(head_dim)

        # (num_heads, seq_len, head_dim)
        q_t = np.transpose(q.astype(np.float32), (1, 0, 2))
        k_t = np.transpose(cached_k, (1, 0, 2))
        v_t = np.transpose(cached_v, (1, 0, 2))

        scores = np.einsum("hqd,hkd->hqk", q_t, k_t) * scale
        if extra_bias is not None:
            scores = scores + extra_bias
        elif causal:
            total_k = k_t.shape[1]
            q_positions = np.arange(total_k - seq_len, total_k)
            mask = np.arange(total_k)[None, :] > q_positions[:, None]
            scores = np.where(mask[None, :, :], -np.inf, scores)

        probs = softmax(scores, axis=-1)
        out = np.einsum("hqk,hkd->hqd", probs, v_t)
        return np.transpose(out, (1, 0, 2)).astype(q.dtype)

    def decode(
        self, seq_id: int, layer: int, q: np.ndarray, k: np.ndarray, v: np.ndarray,
        extra_bias: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """Single-token decode step. q, k, v: (1, heads_or_kv_heads, head_dim)."""
        return self.prefill(seq_id, layer, q, k, v, causal=False, extra_bias=extra_bias)
