"""
Fused CUDA Kernel Reference Implementations

Pure PyTorch implementations of operations that would be fused into
single CUDA kernels in production. These provide correct results
and serve as reference implementations for kernel verification.

In production, these would call custom CUDA kernels via:
- Triton kernels (triton.language)
- Custom C++/CUDA extensions
- Flash Attention

Usage:
    from draco.kernels.ops import fused_rms_norm, fused_rope, fused_swiGLU

    normalized = fused_rms_norm(hidden_states, weight, eps=1e-6)
    cos, sin = fused_rope(position_ids, dim=64, base=10000.0)
    output = fused_swiGLU(gate, up, down)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused Root Mean Square Layer Normalization.

    Reference PyTorch implementation. In production, this would be
    a single CUDA kernel that avoids materializing the full norm tensor.

    Args:
        x: Input tensor (..., hidden_dim)
        weight: Scale parameter (hidden_dim,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms * weight


def fused_rope(
    position_ids: torch.Tensor,
    dim: int,
    max_position_embeddings: int = 8192,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Rotary Position Embedding computation.

    Computes cos and sin tables for rotary position embeddings.
    In production, this would be fused with the attention kernel.

    Args:
        position_ids: (batch, seq_len) or (seq_len,) position indices
        dim: Embedding dimension per head
        max_position_embeddings: Maximum sequence length
        base: RoPE base frequency

    Returns:
        cos: (batch, seq_len, dim)
        sin: (batch, seq_len, dim)
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(position_ids.device) / dim))
    freqs = torch.einsum("i,j->ij", position_ids.float().flatten(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)

    if position_ids.dim() > 1:
        emb = emb.view(position_ids.shape[0], position_ids.shape[1], dim)

    return emb.cos(), emb.sin()


def apply_fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to a tensor.

    Args:
        x: (batch, heads, seq_len, head_dim)
        cos: (batch, seq_len, head_dim)
        sin: (batch, seq_len, head_dim)

    Returns:
        Rotated tensor with same shape
    """
    seq_len = x.shape[2]
    cos = cos[:, :seq_len, :].unsqueeze(1)
    sin = sin[:, :seq_len, :].unsqueeze(1)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def fused_swiGLU(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
) -> torch.Tensor:
    """
    Fused SwiGLU MLP activation.

    Equivalent to: down_proj(silu(gate_proj(x)) * up_proj(x))
    In production, this would be a single fused CUDA kernel.

    Args:
        x: Input hidden states (..., hidden_dim)
        gate_proj: Gate projection layer
        up_proj: Up projection layer
        down_proj: Down projection layer

    Returns:
        Output tensor (..., hidden_dim)
    """
    gate = F.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


def fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused scaled dot-product attention.

    In production, this would use Flash Attention or a custom CUDA kernel
    to fuse the Q*K, softmax, and attn*V operations into a single kernel,
    avoiding materialization of the full attention matrix.

    Args:
        query: (batch, heads, q_len, head_dim)
        key: (batch, heads, kv_len, head_dim)
        value: (batch, heads, kv_len, head_dim)
        scale: Attention scale (defaults to 1/sqrt(head_dim))
        causal: Apply causal masking
        attention_mask: Optional additive attention mask

    Returns:
        output: (batch, heads, q_len, head_dim)
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5

    # This is the standard PyTorch implementation
    # Flash Attention would replace this with:
    #   output = flash_attn_func(q, k, v, causal=causal)
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if causal:
        q_len = query.shape[2]
        kv_len = key.shape[2]
        causal_mask = torch.triu(
            torch.full((q_len, kv_len), float("-inf"), device=query.device), diagonal=1
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attn_weights, value)
