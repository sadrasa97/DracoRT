"""
MoE Expert implementations.

Abstractions for individual experts and expert groups,
supporting dense MLP experts and quantized variants.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    Single MLP expert in a Mixture-of-Experts layer.

    Typically a 2-layer MLP with gating activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

        if activation == "silu":
            self.act_fn = nn.SiLU()
        elif activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: down_proj(act_fn(gate_proj(x)) * up_proj(x))"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ExpertGroup(nn.Module):
    """
    Group of experts in a MoE layer.

    Manages all experts and dispatches tokens to selected experts.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.experts = nn.ModuleList([
            Expert(hidden_dim, intermediate_dim, activation, bias)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: (batch * seq_len, hidden_dim)
            expert_indices: (batch * seq_len, top_k) — which experts to use
            expert_weights: (batch * seq_len, top_k) — weights for each expert

        Returns:
            output: (batch * seq_len, hidden_dim)
        """
        output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (expert_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            token_indices = mask.nonzero(as_tuple=True)[0]
            expert_input = hidden_states[token_indices]

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)

            # Weight and accumulate
            # Find which top-k slot this expert was selected in
            slot_mask = (expert_indices[token_indices] == expert_idx)
            weights = (expert_weights[token_indices] * slot_mask.float()).sum(dim=-1, keepdim=True)

            output.index_add_(0, token_indices, expert_output * weights)

        return output
