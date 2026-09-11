"""
CPU-aware KV cache (spec section 21).

Block-allocated, lazily grown, bounded by an explicit memory budget —
never preallocates ``max_model_len * max_num_seqs`` up front. This is a
CPU-native reimplementation of the paged-KV-cache *algorithm*; it shares
no code with any CUDA PagedAttention kernel (spec section 22 explicitly
requires that separation), only the same block-table idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from draco.exceptions import KVCacheError

_NP_DTYPE = {"f32": np.float32, "f16": np.float16, "int8": np.int8}


@dataclass
class CPUKVCacheConfig:
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    block_size: int = 16
    dtype: str = "f16"
    max_blocks: int = 1024  # bounds total memory; see budget_to_max_blocks()


def budget_to_max_blocks(config: CPUKVCacheConfig, budget_bytes: int) -> int:
    bytes_per_token = (
        2 * config.num_layers * config.num_key_value_heads * config.head_dim
        * np.dtype(_NP_DTYPE.get(config.dtype, np.float16)).itemsize
    )
    bytes_per_block = bytes_per_token * config.block_size
    if bytes_per_block <= 0:
        raise KVCacheError("Invalid CPU KV cache configuration: zero bytes per block.")
    return max(1, budget_bytes // bytes_per_block)


class CPUKVCache:
    """Manages KV storage for many sequences sharing a fixed block pool.

    Each layer gets its own (max_blocks, block_size, num_kv_heads, head_dim)
    K and V arrays, allocated once. Sequences are assigned block ids from a
    free list as they grow — unused blocks cost nothing beyond the initial
    (bounded) allocation.
    """

    def __init__(self, config: CPUKVCacheConfig) -> None:
        self.config = config
        dtype = _NP_DTYPE.get(config.dtype)
        if dtype is None:
            raise KVCacheError(
                f"Unsupported CPU KV cache dtype '{config.dtype}'. Supported: {list(_NP_DTYPE)}."
            )
        shape = (config.max_blocks, config.block_size, config.num_key_value_heads, config.head_dim)
        self._k = [np.zeros(shape, dtype=dtype) for _ in range(config.num_layers)]
        self._v = [np.zeros(shape, dtype=dtype) for _ in range(config.num_layers)]
        self._free_blocks: List[int] = list(range(config.max_blocks))
        self._seq_blocks: Dict[int, List[int]] = {}
        self._seq_len: Dict[int, int] = {}
        self._layer_len: Dict["tuple[int, int]", int] = {}

    def allocate_sequence(self, seq_id: int) -> None:
        if seq_id in self._seq_blocks:
            raise KVCacheError(f"Sequence {seq_id} already has an allocation.")
        self._seq_blocks[seq_id] = []
        self._seq_len[seq_id] = 0

    def _ensure_capacity(self, seq_id: int, new_len: int) -> None:
        needed_blocks = (new_len + self.config.block_size - 1) // self.config.block_size
        have = len(self._seq_blocks[seq_id])
        while have < needed_blocks:
            if not self._free_blocks:
                raise KVCacheError(
                    f"CPU KV cache exhausted: no free blocks left (max_blocks="
                    f"{self.config.max_blocks}). Reduce max_num_seqs or context length, "
                    f"or raise the memory budget."
                )
            self._seq_blocks[seq_id].append(self._free_blocks.pop())
            have += 1

    def append(self, seq_id: int, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append one new token's K/V for the given layer.

        k, v: shape (num_key_value_heads, head_dim)
        """
        if seq_id not in self._seq_blocks:
            raise KVCacheError(f"Sequence {seq_id} has no allocation; call allocate_sequence first.")
        pos = self._layer_len.get((seq_id, layer), 0)
        self._ensure_capacity(seq_id, pos + 1)
        block_idx = pos // self.config.block_size
        offset = pos % self.config.block_size
        block_id = self._seq_blocks[seq_id][block_idx]
        self._k[layer][block_id, offset] = k
        self._v[layer][block_id, offset] = v
        self._layer_len[(seq_id, layer)] = pos + 1
        self._seq_len[seq_id] = max(self._seq_len.get(seq_id, 0), pos + 1)

    def get(self, seq_id: int, layer: int) -> "tuple[np.ndarray, np.ndarray]":
        """Return contiguous (seq_len, num_kv_heads, head_dim) K and V for a sequence."""
        length = self._layer_len.get((seq_id, layer), 0)
        block_ids = self._seq_blocks[seq_id]
        if not block_ids:
            empty_shape = (0, self.config.num_key_value_heads, self.config.head_dim)
            dtype = _NP_DTYPE[self.config.dtype]
            return np.zeros(empty_shape, dtype=dtype), np.zeros(empty_shape, dtype=dtype)

        k_blocks = self._k[layer][block_ids]  # (n_blocks, block_size, heads, dim)
        v_blocks = self._v[layer][block_ids]
        k_flat = k_blocks.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
        v_flat = v_blocks.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
        return k_flat[:length], v_flat[:length]

    def truncate_sequence(self, seq_id: int, new_length: int) -> None:
        """Roll back a sequence's committed KV cache length across all
        layers (used to discard rejected speculative-decoding draft
        tokens whose K/V were already written during the verify pass)."""
        for layer in range(self.config.num_layers):
            key = (seq_id, layer)
            if key in self._layer_len and self._layer_len[key] > new_length:
                self._layer_len[key] = new_length
        self._seq_len[seq_id] = new_length

    def free_sequence(self, seq_id: int) -> None:
        for block_id in self._seq_blocks.pop(seq_id, []):
            self._free_blocks.append(block_id)
        self._seq_len.pop(seq_id, None)
        for key in [k for k in self._layer_len if k[0] == seq_id]:
            self._layer_len.pop(key, None)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)
