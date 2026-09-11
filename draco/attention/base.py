"""
Attention Architecture Abstraction

The runtime must not assume only standard Multi-Head Attention.
Supports: MHA, MQA, GQA, sliding-window, local, global, hybrid, sparse, long-context.
The attention abstraction allows different implementations per architecture.
"""

from __future__ import annotations

import abc
import enum
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class AttentionType(enum.Enum):
    """Types of attention mechanisms."""

    MHA = "multi_head_attention"          # Multi-Head Attention
    MQA = "multi_query_attention"         # Multi-Query Attention
    GQA = "grouped_query_attention"       # Grouped-Query Attention
    SLIDING_WINDOW = "sliding_window"     # Sliding Window Attention
    LOCAL = "local"                       # Local Attention
    GLOBAL = "global"                     # Global Attention
    HYBRID = "hybrid"                     # Hybrid (different per layer)
    SPARSE = "sparse"                     # Sparse Attention
    FLASH = "flash"                       # Flash Attention (optimized)
    PAGED = "paged"                       # Paged Attention (vLLM-style)


class AttentionBackend(abc.ABC):
    """
    Abstract base class for attention backends.

    Each attention mechanism is implemented as a backend that can be
    selected based on model architecture, hardware capabilities, and
    configuration.
    """

    @property
    @abc.abstractmethod
    def attention_type(self) -> AttentionType:
        """The type of attention this backend implements."""
        ...

    @abc.abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        scale: Optional[float] = None,
        is_causal: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute attention.

        Args:
            query: (batch, heads, seq_len, head_dim)
            key: (batch, kv_heads, seq_len, head_dim)
            value: (batch, kv_heads, seq_len, head_dim)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            kv_cache: Optional KV cache for incremental decoding
            scale: Optional attention scale
            is_causal: Whether to apply causal masking
        """
        ...

    def get_supported_dtypes(self) -> Tuple[torch.dtype, ...]:
        """Return supported dtypes for this backend."""
        return (torch.float16, torch.bfloat16)

    def supports_flash_attention(self) -> bool:
        """Whether this backend supports flash attention kernels."""
        return False

    def supports_paged_attention(self) -> bool:
        """Whether this backend supports paged attention."""
        return False

    def compute_attention_score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute raw attention scores: Q @ K^T * scale.
        """
        if scale is None:
            head_dim = query.shape[-1]
            scale = head_dim ** -0.5
        return torch.matmul(query, key.transpose(-2, -1)) * scale

    def apply_causal_mask(
        self,
        scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply causal (autoregressive) mask to attention scores."""
        seq_len = scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=scores.dtype),
            diagonal=1,
        ).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask
        return scores

    def apply_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply softmax to attention scores."""
        return torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)


class MultiHeadAttention(AttentionBackend):
    """
    Standard Multi-Head Attention (MHA).

    Each head has its own Q, K, V projection.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bias = bias

    @property
    def attention_type(self) -> AttentionType:
        return AttentionType.MHA

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        scale: Optional[float] = None,
        is_causal: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Standard MHA forward pass.

        Args:
            query: (batch, seq_len, hidden_dim) or (batch, heads, seq_len, head_dim)
            key: same shape
            value: same shape
        """
        # Reshape to (batch, heads, seq_len, head_dim) if needed
        if query.dim() == 3:
            batch_size, seq_len, _ = query.shape
            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Handle GQA/MQA: repeat key/value heads
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        # Update KV cache
        if kv_cache is not None:
            key, value = self._update_kv_cache(kv_cache, key, value, position_ids)

        # Compute attention scores
        scores = self.compute_attention_score(query, key, scale)

        # Apply causal mask
        if is_causal:
            scores = self.apply_causal_mask(scores, attention_mask)

        # Softmax
        attn_weights = self.apply_softmax(scores)

        # Compute output
        output = torch.matmul(attn_weights, value)

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.shape[0], output.shape[1], -1)

        return output

    def _update_kv_cache(
        self,
        kv_cache: Any,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update key-value cache."""
        if isinstance(kv_cache, dict) and "key" in kv_cache:
            cached_key = kv_cache["key"]
            cached_value = kv_cache["value"]
            if cached_key.shape[2] > 0:
                key = torch.cat([cached_key, key], dim=2)
                value = torch.cat([cached_value, value], dim=2)
            kv_cache["key"] = key
            kv_cache["value"] = value
        return key, value


class GroupedQueryAttention(MultiHeadAttention):
    """
    Grouped-Query Attention (GQA).

    Multiple query heads share groups of KV heads.
    Used in Llama 2, Qwen2, Gemma 2, etc.
    """

    @property
    def attention_type(self) -> AttentionType:
        if self.num_kv_heads == 1:
            return AttentionType.MQA
        return AttentionType.GQA

    def __repr__(self) -> str:
        return (
            f"GroupedQueryAttention("
            f"heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}"
            f")"
        )


class SlidingWindowAttention(MultiHeadAttention):
    """
    Sliding Window Attention.

    Limits attention to a fixed window size, reducing compute for
    long sequences. Used in Mistral, Gemma 2, etc.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int = 4096,
        bias: bool = False,
    ):
        super().__init__(num_heads, num_kv_heads, head_dim, bias)
        self.window_size = window_size

    @property
    def attention_type(self) -> AttentionType:
        return AttentionType.SLIDING_WINDOW

    def apply_causal_mask(
        self,
        scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply causal + sliding window mask."""
        seq_len = scores.shape[-1]
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=scores.dtype),
            diagonal=1,
        ).bool()
        # Sliding window mask
        if self.window_size > 0 and seq_len > self.window_size:
            window_mask = torch.ones(
                seq_len, seq_len, device=scores.device, dtype=torch.bool
            )
            for i in range(seq_len):
                start = max(0, i - self.window_size + 1)
                window_mask[i, start : i + 1] = False
            causal_mask = causal_mask | window_mask
        scores = scores.masked_fill(causal_mask, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask
        return scores

    def __repr__(self) -> str:
        return (
            f"SlidingWindowAttention("
            f"heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, "
            f"window={self.window_size}"
            f")"
        )


def create_attention_backend(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    attention_type: str = "GQA",
    window_size: Optional[int] = None,
    bias: bool = False,
) -> AttentionBackend:
    """
    Factory function to create the appropriate attention backend.

    Args:
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Per-head dimension
        attention_type: One of "MHA", "MQA", "GQA", "SLIDING_WINDOW"
        window_size: Sliding window size (only for SLIDING_WINDOW)
        bias: Whether to use bias in projections
    """
    attention_type = attention_type.upper()

    if attention_type == "SLIDING_WINDOW" or window_size is not None:
        ws = window_size or 4096
        return SlidingWindowAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            window_size=ws,
            bias=bias,
        )
    elif attention_type == "GQA" or attention_type == "MQA":
        return GroupedQueryAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=bias,
        )
    elif attention_type == "MHA":
        return MultiHeadAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=bias,
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
