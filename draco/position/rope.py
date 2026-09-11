"""
Rotary Position Embedding (RoPE)

Supports:
- Standard RoPE
- NTK-aware RoPE (dynamic scaling)
- YaRN (Yet another RoPE extensioN)
- LLaMA-style rope scaling

Detects variant from model configuration.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from draco.position.base import PositionEncoding, RoPEVariant


def _precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Precompute complex exponentials for rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    if scaling_factor != 1.0:
        freqs = freqs / scaling_factor
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _precompute_freqs_cis_yarn(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_max_position_embeddings: Optional[int] = None,
) -> torch.Tensor:
    """Precompute YaRN-scaled rotary embeddings."""
    if original_max_position_embeddings is None:
        original_max_position_embeddings = max_seq_len

    # Compute wavelength-based interpolation
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    wavelengths = 2 * math.pi / freqs

    # YaRN interpolation factors per frequency
    low_freq_wavelen = original_max_position_embeddings / beta_slow
    high_freq_wavelen = original_max_position_embeddings / beta_fast

    smooth_factors = torch.zeros_like(freqs)
    for i, wl in enumerate(wavelengths):
        if wl < high_freq_wavelen:
            smooth_factors[i] = 1.0  # high freq: no change
        elif wl > low_freq_wavelen:
            smooth_factors[i] = 1.0 / scaling_factor  # low freq: full scaling
        else:
            # Smooth interpolation
            smooth_factors[i] = (
                1.0 - (low_freq_wavelen - wl) / (low_freq_wavelen - high_freq_wavelen)
            ) / scaling_factor + (
                (low_freq_wavelen - wl) / (low_freq_wavelen - high_freq_wavelen)
            )

    freqs = freqs * smooth_factors
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(1)  # (seq, 1, head_dim)
    sin = sin.unsqueeze(1)

    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class RotaryPositionEmbedding(PositionEncoding):
    """
    Rotary Position Embedding (RoPE).

    Supports standard, NTK-aware, and YaRN variants.
    Detects variant from model rope_scaling config.
    """

    def __init__(
        self,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        head_dim: int = 128,
        rope_scaling: Optional[dict] = None,
    ):
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim

        # Parse variant
        if rope_scaling is not None:
            self.variant = RoPEVariant.from_config(rope_scaling)
        else:
            self.variant = RoPEVariant()

        # Precompute frequencies
        self._freqs_cis = self._build_freqs()

    def _build_freqs(self) -> torch.Tensor:
        """Build frequency tensor based on variant."""
        rope_type = self.variant.rope_type

        if rope_type in ("default", "linear"):
            return _precompute_freqs_cis(
                dim=self.head_dim,
                max_seq_len=self.max_position_embeddings,
                theta=self.rope_theta,
                scaling_factor=self.variant.scaling_factor,
            )
        elif rope_type in ("yarn", "yet another rope extension"):
            return _precompute_freqs_cis_yarn(
                dim=self.head_dim,
                max_seq_len=self.max_position_embeddings,
                theta=self.rope_theta,
                scaling_factor=self.variant.scaling_factor,
                beta_fast=self.variant.beta_fast,
                beta_slow=self.variant.beta_slow,
                original_max_position_embeddings=self.variant.original_max_position_embeddings,
            )
        elif rope_type == "ntk":
            # NTK-aware: scale theta instead of position
            scaled_theta = self.rope_theta * (self.variant.scaling_factor ** (self.head_dim / (self.head_dim - 2)))
            return _precompute_freqs_cis(
                dim=self.head_dim,
                max_seq_len=self.max_position_embeddings,
                theta=scaled_theta,
            )
        else:
            # Fallback to default
            return _precompute_freqs_cis(
                dim=self.head_dim,
                max_seq_len=self.max_position_embeddings,
                theta=self.rope_theta,
            )

    @property
    def name(self) -> str:
        rope_type = self.variant.rope_type
        if rope_type == "default":
            return "RoPE"
        elif rope_type in ("yarn", "yet another rope extension"):
            return "YaRN"
        elif rope_type == "ntk":
            return "NTK-aware RoPE"
        return f"RoPE ({rope_type})"

    def forward(
        self,
        position_ids: torch.Tensor,
        head_dim: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for rotary embeddings.

        Args:
            position_ids: (batch, seq_len)
            head_dim: per-head dimension

        Returns:
            (cos, sin) each of shape (seq_len, head_dim)
        """
        # Extend frequencies if needed
        max_pos = position_ids.max().item() + 1
        if max_pos > self._freqs_cis.shape[0]:
            self._max_position_embeddings = max_pos + 1024
            self._freqs_cis = self._build_freqs()

        freqs = self._freqs_cis[position_ids]  # (batch, seq_len, head_dim/2)
        cos = freqs.real.float()
        sin = freqs.imag.float()
        return cos, sin

    def supports_extend(self) -> bool:
        return True
