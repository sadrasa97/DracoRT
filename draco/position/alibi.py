"""
ALiBi (Attention with Linear Biases)

Positional encoding that adds a linear bias to attention scores.
Used in BLOOM, some Falcon models, etc.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

from draco.position.base import PositionEncoding


def _get_slopes(num_heads: int) -> list:
    """Get ALiBi slopes for attention heads."""
    def get_near_pow2(n: int) -> float:
        start = 2 ** math.floor(math.log2(n))
        if start == n:
            return start
        return start * 2

    max_power = get_near_pow2(num_heads)
    slopes = [2 ** (-8 * i / max_power) for i in range(1, num_heads + 1)]
    return slopes


class ALiBi(PositionEncoding):
    """
    Attention with Linear Biases (ALiBi).

    Instead of modifying the embeddings, ALiBi adds a distance-based
    bias to attention scores. No learned parameters.
    """

    def __init__(self, num_heads: int, max_position_embeddings: int = 8192):
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        slopes = _get_slopes(num_heads)
        self.slopes = torch.tensor(slopes, dtype=torch.float32)

        # Precompute bias
        self._bias = self._build_bias()

    def _build_bias(self) -> torch.Tensor:
        """Build ALiBi bias matrix."""
        positions = torch.arange(self.max_position_embeddings)
        # distance[i][j] = |i - j|
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
        # Apply slopes
        bias = distance.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)  # (heads, seq, seq)
        return bias

    @property
    def name(self) -> str:
        return "ALiBi"

    def forward(
        self,
        position_ids: torch.Tensor,
        head_dim: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute ALiBi bias.

        Args:
            position_ids: (batch, seq_len)
            head_dim: (unused for ALiBi)

        Returns:
            (bias, None) — bias is the ALiBi attention bias, None for sin.
        """
        seq_len = position_ids.shape[-1]
        max_pos = position_ids.max().item() + 1

        if max_pos > self._bias.shape[-1]:
            # Rebuild for longer sequences
            self.max_position_embeddings = max_pos + 1024
            self._bias = self._build_bias()

        # Extract relevant portion
        bias = self._bias[:, :seq_len, :seq_len].to(position_ids.device)
        # Expand for batch
        bias = bias.unsqueeze(0)  # (1, heads, seq, seq)

        return bias, None

    def supports_extend(self) -> bool:
        return True
