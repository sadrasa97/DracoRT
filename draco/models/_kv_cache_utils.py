"""
Shared KV-cache helpers used by every per-architecture adapter
(draco/models/<arch>/adapter.py).

Every adapter's attention module follows the same three-step pattern to
support a real KV cache:

    1. concat_kv(past_key_value, k, v) -> (k, v), present_key_value
    2. build a causal mask that covers the FULL key length (cached + new)
       instead of assuming keys start at position 0
    3. the enclosing *Model.forward() derives position_ids from the cache
       length instead of always starting at arange(0, seq_len)

These three helpers implement that pattern once so each adapter only has
to call them instead of re-deriving the (easy to get subtly wrong) causal
masking math independently.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def concat_kv(
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
    k: torch.Tensor,
    v: torch.Tensor,
    use_cache: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Concatenate new K/V with cached K/V (both shaped
    (bsz, num_kv_heads, seq_len, head_dim)). Returns the (possibly
    extended) k, v to attend over, plus the present_key_value to hand
    back to the caller for the next step (None when use_cache=False, so
    non-cached callers keep zero memory overhead)."""
    if past_key_value is not None:
        past_k, past_v = past_key_value
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    present = (k, v) if use_cache else None
    return k, v, present


def cache_aware_causal_mask(
    seq_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Causal mask of shape (seq_len, kv_len) that masks query position i
    (in absolute terms, offset by however many cached keys precede this
    chunk) against key positions > i. With seq_len == 1 (a decode step)
    this is a no-op, which is correct: a single new token may attend to
    every cached key before it. Using torch.triu(..., diagonal=1) here —
    as the original no-cache code did — is WRONG once kv_len != seq_len,
    since it assumes the query chunk starts at key position 0."""
    past_len = kv_len - seq_len
    query_pos = torch.arange(seq_len, device=device).unsqueeze(1) + past_len
    key_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    return torch.where(
        key_pos <= query_pos,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float("-inf"), device=device, dtype=dtype),
    )


def derive_position_ids(
    input_ids: torch.Tensor,
    past_key_values: Optional[list],
    position_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """If position_ids wasn't given explicitly, derive it from how many
    tokens are already in the cache — NOT always arange(0, seq_len). This
    is what makes RoPE/ALiBi positions correct for a token fed in
    isolation during cached decoding (position 47, not position 0)."""
    if position_ids is not None:
        return position_ids
    past_len = 0
    if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
        past_len = past_key_values[0][0].shape[2]
    return (
        torch.arange(input_ids.shape[1], device=input_ids.device) + past_len
    ).unsqueeze(0).expand(input_ids.shape[0], -1)
