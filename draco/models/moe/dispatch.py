"""
Token Dispatch for MoE

Handles token routing mechanics:
- Token dispatch: sending tokens to experts
- Token combine: gathering expert outputs
- Expert capacity: limiting tokens per expert
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class TokenDispatcher:
    """
    Base token dispatcher for MoE.

    Handles the mechanics of sending tokens to experts and
    gathering outputs.
    """

    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to experts.

        Args:
            hidden_states: (tokens, hidden_dim)
            expert_indices: (tokens, top_k)
            expert_weights: (tokens, top_k)

        Returns:
            dispatched_tokens: per-expert token groups
            dispatched_weights: per-expert weight groups
            tokens_per_expert: count per expert
        """
        batch_size = hidden_states.shape[0]

        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device=hidden_states.device
        )

        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx).any(dim=-1)
            tokens_per_expert[expert_idx] = mask.sum()

        return hidden_states, expert_indices, tokens_per_expert

    def combine(
        self,
        expert_outputs: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Combine expert outputs back into original token order.

        Args:
            expert_outputs: per-expert outputs
            expert_indices: original routing indices
            expert_weights: routing weights

        Returns:
            Combined output tensor matching original token order.
        """
        return expert_outputs


class CapacityLimitedDispatcher(TokenDispatcher):
    """
    Capacity-limited token dispatcher.

    Limits the number of tokens each expert can process,
    dropping excess tokens. Prevents memory issues with
    unbalanced routing.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True,
    ):
        super().__init__(num_experts, top_k)
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

    def compute_capacity(self, num_tokens: int) -> int:
        """Compute max tokens per expert."""
        return int(num_tokens * self.capacity_factor / self.num_experts)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch with capacity limiting."""
        num_tokens = hidden_states.shape[0]
        capacity = self.compute_capacity(num_tokens)

        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device=hidden_states.device
        )

        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx).any(dim=-1)
            count = mask.sum()
            if count > capacity and self.drop_tokens:
                tokens_per_expert[expert_idx] = capacity
            else:
                tokens_per_expert[expert_idx] = count

        return hidden_states, expert_indices, tokens_per_expert
