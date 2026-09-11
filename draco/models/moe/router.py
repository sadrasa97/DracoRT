"""
MoE Router

Handles token routing to experts. Supports top-K routing,
token dispatch, and expert capacity management.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(abc.ABC):
    """
    Abstract base class for MoE routers.

    Responsible for deciding which tokens go to which experts.
    """

    @abc.abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens to experts.

        Args:
            hidden_states: (batch * seq_len, hidden_dim)

        Returns:
            expert_indices: indices of selected experts for each token
            routing_weights: weights for each selected expert
            aux_info: auxiliary information (load balancing loss, etc.)
        """
        ...

    @abc.abstractmethod
    def get_num_experts(self) -> int:
        """Number of experts available."""
        ...

    @abc.abstractmethod
    def get_top_k(self) -> int:
        """Number of experts selected per token."""
        ...


class TopKRouter(Router):
    """
    Top-K expert routing.

    Selects the top-K experts for each token based on router logits.
    Supports optional normalization and auxiliary load-balancing loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        norm_topk_prob: bool = True,
        use_load_balancing_loss: bool = True,
        jitter_noise: float = 0.0,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.use_load_balancing_loss = use_load_balancing_loss
        self.jitter_noise = jitter_noise

        # Router weights
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens via top-K selection.

        Args:
            hidden_states: (tokens, hidden_dim)

        Returns:
            expert_indices: (tokens, top_k)
            routing_weights: (tokens, top_k)
            aux_info: dict with 'load_balancing_loss' if enabled
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)  # (tokens, num_experts)

        # Optional jitter during training
        if self.jitter_noise > 0 and self.gate.training:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Normalize if requested
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Compute auxiliary info
        aux_info: dict = {}
        if self.use_load_balancing_loss:
            aux_info["load_balancing_loss"] = self._compute_load_balancing_loss(
                router_logits, top_k_indices
            )

        return top_k_indices, top_k_weights, aux_info

    def _compute_load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss for training stability.

        Encourages uniform distribution of tokens across experts.
        """
        num_tokens = router_logits.shape[0]
        num_experts = self.num_experts

        # Probability distribution over experts
        routing_weights = F.softmax(router_logits, dim=-1)  # (tokens, num_experts)

        # Fraction of tokens dispatched to each expert
        expert_mask = F.one_hot(expert_indices[:, 0], num_experts).float()  # use top-1
        tokens_per_expert = expert_mask.sum(dim=0) / num_tokens

        # Average routing probability per expert
        avg_routing = routing_weights.mean(dim=0)

        # Load balancing loss: encourages uniform distribution
        loss = num_experts * (tokens_per_expert * avg_routing).sum()
        return loss

    def get_num_experts(self) -> int:
        return self.num_experts

    def get_top_k(self) -> int:
        return self.top_k
