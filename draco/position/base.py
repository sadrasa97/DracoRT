"""
Extensible Position Encoding System

Supports: RoPE, NTK-aware RoPE, YaRN, ALiBi, learned positional embeddings,
relative position mechanisms, architecture-specific position encoding.

The runtime detects the required mechanism from model configuration.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class PositionEncoding(abc.ABC):
    """
    Abstract base class for position encodings.

    Each architecture specifies its position encoding type and the runtime
    detects the required mechanism from model configuration.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for this position encoding."""
        ...

    @abc.abstractmethod
    def forward(
        self,
        position_ids: torch.Tensor,
        head_dim: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute position encoding.

        For RoPE-like encodings, returns (cos, sin) tensors.
        For ALiBi-like, returns attention bias.
        For learned, returns position embeddings.
        """
        ...

    @abc.abstractmethod
    def supports_extend(self) -> bool:
        """Whether this encoding can be efficiently extended to longer sequences."""
        ...

    def extend_position_ids(
        self,
        position_ids: torch.Tensor,
        new_len: int,
    ) -> torch.Tensor:
        """Extend position IDs for incremental decoding."""
        return torch.cat([
            position_ids,
            torch.arange(new_len, device=position_ids.device).unsqueeze(0),
        ], dim=-1)


class RoPEVariant:
    """Configuration for RoPE variants."""

    def __init__(
        self,
        rope_type: str = "default",
        scaling_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        original_max_position_embeddings: Optional[int] = None,
    ):
        self.rope_type = rope_type
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.original_max_position_embeddings = original_max_position_embeddings

    @classmethod
    def from_config(cls, rope_scaling: dict) -> RoPEVariant:
        """Create from HuggingFace rope_scaling config dict."""
        return cls(
            rope_type=rope_scaling.get("type", rope_scaling.get("rope_type", "default")),
            scaling_factor=rope_scaling.get("factor", 1.0),
            beta_fast=rope_scaling.get("beta_fast", 32.0),
            beta_slow=rope_scaling.get("beta_slow", 1.0),
            original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings"),
        )
