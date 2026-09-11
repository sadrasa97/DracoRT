"""
Tensor Parallelism

Utilities for splitting and merging model tensors across multiple GPUs.
Provides column-parallel and row-parallel linear layer wrappers,
and tensor split/merge helpers for weight distribution.

Usage:
    config = TensorParallelConfig(world_size=2, rank=0)
    q_weight = torch.randn(1024, 512)
    shards = split_tensor(q_weight, world_size=2, dim=0)  # column-parallel
    local_weight = shards[0]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("draco.parallel")


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    device: str = "cpu"

    @property
    def is_parallel(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def split_tensor(
    tensor: torch.Tensor,
    world_size: int,
    dim: int = 0,
) -> List[torch.Tensor]:
    """
    Split a tensor along a dimension for distribution across GPUs.

    Args:
        tensor: Tensor to split
        world_size: Number of partitions
        dim: Dimension to split along

    Returns:
        List of tensor shards, one per rank
    """
    if world_size <= 1:
        return [tensor]

    # Pad if necessary
    size = tensor.shape[dim]
    remainder = size % world_size
    if remainder != 0:
        pad_size = world_size - remainder
        padding = [0] * (2 * tensor.ndim)
        padding[2 * dim + 1] = pad_size  # pad at end of dim
        tensor = torch.nn.functional.pad(tensor, padding)
        # Note: caller should handle unpadding

    return list(tensor.chunk(world_size, dim=dim))


def merge_tensor(
    shards: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """
    Merge tensor shards back into a single tensor.

    Args:
        shards: List of tensor shards from split_tensor
        dim: Dimension along which they were split

    Returns:
        Merged tensor
    """
    if len(shards) == 1:
        return shards[0]
    return torch.cat(shards, dim=dim)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-parallel weight distribution.

    The weight matrix is split along the output dimension (dim=0).
    Each GPU holds a slice of the output features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[TensorParallelConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or TensorParallelConfig()

        if self.config.is_parallel:
            assert out_features % self.config.world_size == 0, (
                f"out_features ({out_features}) must be divisible by "
                f"world_size ({self.config.world_size})"
            )
            local_out = out_features // self.config.world_size
        else:
            local_out = out_features

        self.weight = nn.Parameter(torch.empty(local_out, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(local_out))
        else:
            self.bias = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row-parallel weight distribution.

    The weight matrix is split along the input dimension (dim=1).
    Each GPU holds a slice of the input features. Output is all-reduced.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[TensorParallelConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or TensorParallelConfig()

        if self.config.is_parallel:
            assert in_features % self.config.world_size == 0
            local_in = in_features // self.config.world_size
        else:
            local_in = in_features

        self.weight = nn.Parameter(torch.empty(out_features, local_in))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight)
        # In real implementation, all_reduce would happen here
        # For CPU simulation, we just return the local output
        if self.bias is not None:
            out = out + self.bias
        return out


def partition_parameters(
    model: nn.Module,
    config: TensorParallelConfig,
) -> Dict[str, Tuple[str, int]]:
    """
    Compute parameter partitioning plan for a model.

    Returns a dict mapping parameter name to (parallel_type, partition_dim).
    parallel_type is 'column' or 'row'.
    """
    plan = {}
    for name, param in model.named_parameters():
        if param.ndim == 2:
            # Weight matrices: decide column vs row parallel
            if "out_proj" in name or "o_proj" in name or "down_proj" in name or "dense" in name:
                plan[name] = ("row", 1)
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name or "gate_proj" in name or "up_proj" in name:
                plan[name] = ("column", 0)
            else:
                plan[name] = ("column", 0)
        elif param.ndim == 1:
            # Biases: replicate
            plan[name] = ("replicate", -1)
        else:
            plan[name] = ("replicate", -1)
    return plan
