"""Alias module — re-export from base.py for clean imports."""

from draco.attention.base import (
    AttentionBackend,
    AttentionType,
    MultiHeadAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
    create_attention_backend,
)

__all__ = [
    "AttentionBackend",
    "AttentionType",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "SlidingWindowAttention",
    "create_attention_backend",
]
