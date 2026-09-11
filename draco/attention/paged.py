"""
PagedAttention

Attention computation using paged KV cache blocks. Instead of
concatenating all KV pairs, this implementation reads directly
from the block manager's block tables — avoiding memory copies
and enabling efficient memory sharing between sequences.

Usage:
    manager = KVCacheBlockManager(...)
    paged_attn = PagedAttention(num_heads=8, head_dim=64)
    output = paged_attn.forward(query, block_manager, seq_id=0, layer_idx=0)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("draco.attention.paged")


class PagedAttention:
    """
    PagedAttention kernel (PyTorch reference implementation).

    Performs attention computation reading KV from block-based cache.
    Each sequence has a block table mapping logical positions to physical
    blocks in the KVCacheBlockManager.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        num_kv_heads: Optional[int] = None,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale or (head_dim ** -0.5)
        self.num_kv_heads = num_kv_heads or num_heads

    def forward(
        self,
        query: torch.Tensor,
        block_manager: Any,
        seq_id: int,
        layer_idx: int,
        context_len: Optional[int] = None,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention using paged KV cache.

        Args:
            query: (batch=1, num_heads, seq_len, head_dim)
            block_manager: KVCacheBlockManager instance
            seq_id: Sequence ID in the block manager
            layer_idx: Transformer layer index
            context_len: Total context length (query_len + cached_kv_len)
            causal: Apply causal masking
            attention_mask: Optional additive attention mask

        Returns:
            output: (1, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Get KV from paged cache
        key, value = block_manager.get_kv(seq_id, layer_idx)
        # key, value: (num_kv_heads, total_kv_len, head_dim)

        if key.shape[1] == 0:
            # No cached KV yet — return zeros
            return torch.zeros_like(query)

        total_kv_len = key.shape[1]

        # Expand KV heads if needed (GQA)
        if self.num_kv_heads < num_heads:
            n_rep = num_heads // self.num_kv_heads
            key = key.repeat_interleave(n_rep, dim=0)  # (num_heads, total_kv_len, head_dim)
            value = value.repeat_interleave(n_rep, dim=0)

        # Compute attention scores
        # query: (1, heads, q_len, head_dim)
        # key:   (1, heads, kv_len, head_dim)
        q = query  # (1, heads, q_len, head_dim)
        k = key.unsqueeze(0).to(query.dtype)  # (1, heads, kv_len, head_dim)
        v = value.unsqueeze(0).to(query.dtype)  # (1, heads, kv_len, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_weights: (1, heads, q_len, kv_len)

        # Causal mask: each query position can only attend to KV positions <= its own
        # Query positions start at (total_kv_len - seq_len) in the full sequence
        if causal:
            q_start = total_kv_len - seq_len
            kv_len = total_kv_len
            # Vectorized: previously built row-by-row with a Python loop
            # over seq_len, which is O(seq_len) Python overhead per call
            # (significant during prefill with long prompts). A single
            # broadcasted comparison produces the identical mask.
            q_positions = torch.arange(q_start, q_start + seq_len, device=query.device).unsqueeze(1)
            kv_positions = torch.arange(kv_len, device=query.device).unsqueeze(0)
            causal_mask = torch.where(
                kv_positions <= q_positions,
                torch.zeros(1, device=query.device, dtype=query.dtype),
                torch.full((1,), float("-inf"), device=query.device, dtype=query.dtype),
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # Weighted sum
        output = torch.matmul(attn_weights, v)
        # output: (1, heads, q_len, head_dim)

        return output

    def forward_batch(
        self,
        queries: List[torch.Tensor],
        block_manager: Any,
        seq_ids: List[int],
        layer_idx: int,
        causal: bool = True,
    ) -> List[torch.Tensor]:
        """
        Batched PagedAttention across multiple sequences in one call.

        Continuous batching groups many in-flight sequences (each with its
        own KV length) into a single scheduler step; running `forward()`
        once per sequence in a Python loop turns every attention call in
        the engine into `batch_size` separate small GPU kernel launches.
        This pads each sequence's KV to the batch's max length, builds a
        combined causal + padding mask, and runs one batched matmul instead.

        Args:
            queries: one (1, num_heads, q_len_i, head_dim) tensor per
                sequence — q_len_i may differ per sequence (e.g. mixed
                prefill/decode within a batch).
            block_manager: KVCacheBlockManager instance
            seq_ids: sequence ID for each entry in `queries`
            layer_idx: transformer layer index
            causal: apply causal masking

        Returns:
            List of (1, num_heads, q_len_i, head_dim) output tensors, one
            per input sequence, in the same order as `queries`.
        """
        if len(queries) != len(seq_ids):
            raise ValueError("queries and seq_ids must have the same length")
        if not queries:
            return []

        device = queries[0].device
        dtype = queries[0].dtype
        num_heads = queries[0].shape[1]

        kvs = [block_manager.get_kv(sid, layer_idx) for sid in seq_ids]
        kv_lens = [k.shape[1] for k, _ in kvs]
        q_lens = [q.shape[2] for q in queries]
        max_kv_len = max(kv_lens) if kvs else 0
        max_q_len = max(q_lens)
        batch_size = len(queries)

        if max_kv_len == 0:
            return [torch.zeros_like(q) for q in queries]

        # Pad queries to (batch, heads, max_q_len, head_dim)
        q_batch = torch.zeros(batch_size, num_heads, max_q_len, self.head_dim, device=device, dtype=dtype)
        for i, q in enumerate(queries):
            q_batch[i, :, : q_lens[i], :] = q[0]

        # Pad KV to (batch, heads, max_kv_len, head_dim), expanding GQA heads per-sequence
        k_batch = torch.zeros(batch_size, num_heads, max_kv_len, self.head_dim, device=device, dtype=dtype)
        v_batch = torch.zeros(batch_size, num_heads, max_kv_len, self.head_dim, device=device, dtype=dtype)
        for i, (k, v) in enumerate(kvs):
            if k.shape[1] == 0:
                continue
            if self.num_kv_heads < num_heads:
                n_rep = num_heads // self.num_kv_heads
                k = k.repeat_interleave(n_rep, dim=0)
                v = v.repeat_interleave(n_rep, dim=0)
            k_batch[i, :, : kv_lens[i], :] = k.to(dtype)
            v_batch[i, :, : kv_lens[i], :] = v.to(dtype)

        attn_weights = torch.matmul(q_batch, k_batch.transpose(-2, -1)) * self.scale
        # attn_weights: (batch, heads, max_q_len, max_kv_len)

        # Per-sequence mask: combine causal alignment (queries in a given
        # sequence occupy the *last* q_lens[i] positions of that sequence's
        # real context) with padding masks for both the query and KV axes,
        # since different sequences pad to different real lengths.
        mask = torch.zeros(batch_size, max_q_len, max_kv_len, device=device, dtype=dtype)
        neg_inf = float("-inf")
        for i in range(batch_size):
            kv_len = kv_lens[i]
            q_len = q_lens[i]
            if kv_len < max_kv_len:
                mask[i, :, kv_len:] = neg_inf
            if q_len < max_q_len:
                # Padded query rows don't matter for output (sliced off
                # below) but must not produce NaNs from an all -inf softmax.
                mask[i, q_len:, :] = 0.0
            if causal:
                q_start = kv_len - q_len
                q_pos = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
                kv_pos = torch.arange(kv_len, device=device).unsqueeze(0)
                causal_block = torch.where(
                    kv_pos <= q_pos,
                    torch.zeros(1, device=device, dtype=dtype),
                    torch.full((1,), neg_inf, device=device, dtype=dtype),
                )
                mask[i, :q_len, :kv_len] = causal_block

        attn_weights = attn_weights + mask.unsqueeze(1)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        output = torch.matmul(attn_weights, v_batch)  # (batch, heads, max_q_len, head_dim)

        return [output[i : i + 1, :, : q_lens[i], :] for i in range(batch_size)]

    def forward_kv_cached(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard attention with already-assembled KV (for testing/verification).

        Args:
            query: (batch, heads, q_len, head_dim)
            key: (batch, heads, kv_len, head_dim)
            value: (batch, heads, kv_len, head_dim)

        Returns:
            output: (batch, heads, q_len, head_dim)
        """
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        q_len = query.shape[2]
        kv_len = key.shape[2]

        if q_len == kv_len:
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=query.device), diagonal=1
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)

    def __repr__(self) -> str:
        return (
            f"PagedAttention("
            f"heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim})"
        )
