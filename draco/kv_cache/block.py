"""
KV Cache Block Manager

Block-based KV cache with allocation, deallocation, and memory management.
Each block holds a fixed-size chunk of key/value pairs for a single sequence.

Block table mapping: logical block -> physical block.
Physical blocks are allocated from a global pool.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("draco.kv_cache")


class KVCacheBlock:
    """A single block of KV cache storing key and value tensors.

    Each block holds `block_size` tokens worth of key/value pairs
    for all layers and all KV heads.
    """

    def __init__(
        self,
        block_id: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.block_id = block_id
        self.block_size = block_size
        self.num_tokens = 0  # How many tokens are stored in this block

        # Allocate key and value caches
        # Shape: (num_layers, num_kv_heads, block_size, head_dim)
        self.key_cache = torch.zeros(
            num_layers, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )
        self.value_cache = torch.zeros(
            num_layers, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> int:
        """Append key/value to this block at the next available position.

        Args:
            layer_idx: Which transformer layer
            key: (num_kv_heads, num_tokens_to_append, head_dim)
            value: (num_kv_heads, num_tokens_to_append, head_dim)

        Returns:
            Number of tokens actually appended (may be less if block is full).
        """
        num_to_append = key.shape[1]
        available = self.block_size - self.num_tokens
        actual_append = min(num_to_append, available)

        if actual_append <= 0:
            return 0

        self.key_cache[layer_idx, :, self.num_tokens:self.num_tokens + actual_append, :] = key[:, :actual_append, :]
        self.value_cache[layer_idx, :, self.num_tokens:self.num_tokens + actual_append, :] = value[:, :actual_append, :]

        # Update count only on first layer append
        if layer_idx == 0:
            self.num_tokens += actual_append

        return actual_append

    def get_kv(
        self,
        layer_idx: int,
        num_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get key/value tensors from this block.

        Args:
            layer_idx: Which transformer layer
            num_tokens: Number of tokens to return (None = all stored tokens)

        Returns:
            key: (num_kv_heads, num_tokens, head_dim)
            value: (num_kv_heads, num_tokens, head_dim)
        """
        n = num_tokens or self.num_tokens
        return self.key_cache[layer_idx, :, :n, :], self.value_cache[layer_idx, :, :n, :]

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    @property
    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def clear(self) -> None:
        """Reset the block for reuse."""
        self.num_tokens = 0
        self.key_cache.zero_()
        self.value_cache.zero_()


class KVCacheBlockManager:
    """Manages a pool of KV cache blocks.

    Provides block allocation, deallocation, and sequence-to-block mapping.
    Uses a free list for O(1) allocation and deallocation.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        # Global block pool
        self.blocks: List[KVCacheBlock] = []
        for i in range(num_blocks):
            block = KVCacheBlock(
                block_id=i,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                dtype=dtype,
                device=device,
            )
            self.blocks.append(block)

        # Free block IDs (available for allocation)
        self._free_blocks: deque[int] = deque(range(num_blocks))

        # Sequence -> block table mapping
        # block_table[seq_id] = list of physical block IDs
        self._block_tables: Dict[int, List[int]] = {}

        # Statistics
        self._allocated_count = 0
        self._total_allocations = 0

        # get_kv() concatenates every block's tensor for a sequence on
        # every call, which is wasteful when the same (seq_id, layer_idx)
        # is read repeatedly with no intervening append (e.g. re-reading
        # KV across sampling steps for other layers first). Cache the
        # concatenated result per (seq_id, layer_idx) and invalidate it
        # whenever that sequence's blocks change.
        self._kv_cache_valid: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _invalidate_kv_cache(self, seq_id: int) -> None:
        stale = [k for k in self._kv_cache_valid if k[0] == seq_id]
        for k in stale:
            del self._kv_cache_valid[k]

    @classmethod
    def from_gpu_memory_utilization(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        gpu_memory_utilization: float = 0.90,
        model_weight_bytes: int = 0,
        activation_reserve_bytes: int = 0,
        min_blocks: int = 1,
    ) -> "KVCacheBlockManager":
        """Build a manager sized from actual available device memory.

        `gpu_memory_utilization` was previously accepted throughout the
        engine (`LLM`, `AsyncLLM`, `ModelRunner`, the CLI) but nothing used
        it to decide `num_blocks` — every manager was built with a fixed,
        hand-picked block count. This computes it from real free memory
        instead (see `draco.utils.memory.plan_kv_cache_blocks`).
        """
        from draco.utils.memory import plan_kv_cache_blocks

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_blocks = plan_kv_cache_blocks(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            model_weight_bytes=model_weight_bytes,
            activation_reserve_bytes=activation_reserve_bytes,
            min_blocks=min_blocks,
        )
        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=dtype,
            device=device,
        )

    def allocate_block(self) -> Optional[int]:
        """Allocate a single block from the free pool.

        Returns:
            Block ID if available, None if pool is exhausted.
        """
        if not self._free_blocks:
            return None

        block_id = self._free_blocks.popleft()
        self._allocated_count += 1
        self._total_allocations += 1
        return block_id

    def free_block(self, block_id: int) -> None:
        """Return a block to the free pool."""
        self.blocks[block_id].clear()
        self._free_blocks.append(block_id)
        self._allocated_count -= 1

    def allocate_sequence(self, seq_id: int) -> List[int]:
        """Allocate a new block table for a sequence.

        Starts with one block.
        """
        block_id = self.allocate_block()
        if block_id is None:
            raise RuntimeError("KV cache block pool exhausted")
        self._block_tables[seq_id] = [block_id]
        return [block_id]

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks belonging to a sequence."""
        if seq_id not in self._block_tables:
            return
        for block_id in self._block_tables[seq_id]:
            self.free_block(block_id)
        del self._block_tables[seq_id]
        self._invalidate_kv_cache(seq_id)

    def ensure_space(self, seq_id: int, num_new_tokens: int) -> bool:
        """Ensure there are enough blocks for num_new_tokens.

        Allocates additional blocks if needed.

        Returns:
            True if space is available, False if block pool is exhausted.
        """
        if seq_id not in self._block_tables:
            self.allocate_sequence(seq_id)

        table = self._block_tables[seq_id]
        last_block = self.blocks[table[-1]]
        remaining_in_last = last_block.block_size - last_block.num_tokens
        tokens_needing_new_blocks = max(0, num_new_tokens - remaining_in_last)
        blocks_needed = (tokens_needing_new_blocks + self.block_size - 1) // self.block_size

        for _ in range(blocks_needed):
            block_id = self.allocate_block()
            if block_id is None:
                return False
            table.append(block_id)

        return True

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get the block table for a sequence."""
        return self._block_tables.get(seq_id, [])

    def append_kv(
        self,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append key/value to a sequence's KV cache.

        Args:
            seq_id: Sequence ID
            key: (num_layers, num_kv_heads, num_tokens, head_dim)
            value: (num_layers, num_kv_heads, num_tokens, head_dim)

        Every layer's data for a given token must land at the SAME
        (block, offset) — layer 0's write at position [block, 2] and
        layer 5's write for the same token must both be [block, 2], or
        get_kv() will hand back layer 5's key paired with a completely
        different token's position. Layer/offset placement is therefore
        computed ONCE (the loop below) and then applied identically to
        every layer, using each block's num_tokens as it stood BEFORE
        this call — not updated mid-loop, which is what a previous
        version of this method did (looping layer 0..N and calling
        `KVCacheBlock.append()` — whose own bookkeeping only advances
        `num_tokens` on layer_idx==0 — independently per layer): layer 0
        would fill a block and advance its counter, and then layer 1's
        own `available = block_size - num_tokens` check would see that
        already-advanced counter and read it as "no room here", silently
        shifting layer 1+'s data into the wrong block/offset entirely.
        """
        if seq_id not in self._block_tables:
            self.allocate_sequence(seq_id)

        table = self._block_tables[seq_id]
        num_tokens = key.shape[2]

        # Pass 1: decide, once, which (block, offset) each new token goes
        # to — allocating additional blocks as needed.
        plan: List[Tuple[int, int, int, int]] = []  # (block_id, block_offset, src_offset, count)
        token_offset = 0
        block_idx = 0
        while token_offset < num_tokens:
            if block_idx >= len(table):
                new_block_id = self.allocate_block()
                if new_block_id is None:
                    break  # pool exhausted; caller is expected to have
                           # called ensure_space() beforehand, so this is
                           # a defensive stop, not the primary guard
                table.append(new_block_id)
            block = self.blocks[table[block_idx]]
            start = block.num_tokens
            available = block.block_size - start
            if available <= 0:
                block_idx += 1
                continue
            take = min(num_tokens - token_offset, available)
            plan.append((table[block_idx], start, token_offset, take))
            token_offset += take
            block_idx += 1

        # Pass 2: apply the SAME (block, offset) placement to every layer.
        for block_id, start, src_offset, take in plan:
            block = self.blocks[block_id]
            block.key_cache[:, :, start:start + take, :] = key[:, :, src_offset:src_offset + take, :]
            block.value_cache[:, :, start:start + take, :] = value[:, :, src_offset:src_offset + take, :]
            block.num_tokens = start + take

        self._invalidate_kv_cache(seq_id)

    def get_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get concatenated key/value for a sequence at a given layer.

        Returns:
            key: (num_kv_heads, total_tokens, head_dim)
            value: (num_kv_heads, total_tokens, head_dim)

        Results are cached per (seq_id, layer_idx): concatenating every
        block's tensor is wasted work if nothing changed since the last
        read of that layer (common when other layers are read in between).
        The cache is invalidated on any append/free for the sequence.
        """
        cache_key = (seq_id, layer_idx)
        cached = self._kv_cache_valid.get(cache_key)
        if cached is not None:
            return cached

        table = self._block_tables.get(seq_id, [])
        keys = []
        values = []
        for block_id in table:
            block = self.blocks[block_id]
            if block.num_tokens > 0:
                k, v = block.get_kv(layer_idx)
                keys.append(k)
                values.append(v)

        if keys:
            result = (torch.cat(keys, dim=1), torch.cat(values, dim=1))
        else:
            result = (
                torch.zeros(self.num_kv_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
                torch.zeros(self.num_kv_heads, 0, self.head_dim, dtype=self.dtype, device=self.device),
            )

        self._kv_cache_valid[cache_key] = result
        return result

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return self._allocated_count

    @property
    def utilization(self) -> float:
        """Fraction of blocks currently allocated."""
        if self.num_blocks == 0:
            return 0.0
        return self._allocated_count / self.num_blocks

    @property
    def total_tokens_capacity(self) -> int:
        """Total token capacity across all blocks."""
        return self.num_blocks * self.block_size

    @property
    def used_tokens(self) -> int:
        """Total tokens currently stored across all blocks."""
        return sum(b.num_tokens for b in self.blocks if b.num_tokens > 0)

    def __repr__(self) -> str:
        return (
            f"KVCacheBlockManager("
            f"blocks={self.num_blocks}, "
            f"block_size={self.block_size}, "
            f"allocated={self._allocated_count}, "
            f"free={self.num_free_blocks}, "
            f"utilization={self.utilization:.1%})"
        )
