"""
Pipeline Parallelism

Partitions model layers across multiple GPUs for pipeline parallelism.
Each GPU holds a contiguous subset of transformer layers, and activations
are passed between GPUs sequentially.

Usage:
    config = PipelineParallelConfig(num_stages=2, world_size=2)
    plan = create_pipeline_plan(model, config)
    # plan: {0: [0,1], 1: [2,3]} — GPU 0 holds layers 0-1, GPU 1 holds 2-3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("draco.parallel.pipeline")


@dataclass
class PipelineParallelConfig:
    """Configuration for pipeline parallelism."""
    num_stages: int = 1
    world_size: int = 1
    rank: int = 0
    device: str = "cpu"
    chunks: int = 1  # Number of micro-batches for pipelining

    @property
    def is_parallel(self) -> bool:
        return self.num_stages > 1

    @property
    def stage_id(self) -> int:
        return self.rank


@dataclass
class PipelineStage:
    """Represents one stage in the pipeline."""
    stage_id: int
    layer_indices: List[int]
    device: str = "cpu"

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    @property
    def first_layer(self) -> int:
        return self.layer_indices[0] if self.layer_indices else 0

    @property
    def last_layer(self) -> int:
        return self.layer_indices[-1] if self.layer_indices else 0


@dataclass
class PipelinePlan:
    """Complete pipeline partitioning plan."""
    stages: List[PipelineStage] = field(default_factory=list)
    embed_device: str = "cpu"
    head_device: str = "cpu"

    def get_stage(self, stage_id: int) -> Optional[PipelineStage]:
        for s in self.stages:
            if s.stage_id == stage_id:
                return s
        return None

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def summary(self) -> str:
        lines = ["Pipeline Parallelism Plan:"]
        for stage in self.stages:
            lines.append(f"  Stage {stage.stage_id}: layers {stage.layer_indices} ({stage.num_layers} layers)")
        return "\n".join(lines)


def create_pipeline_plan(
    model: nn.Module,
    config: PipelineParallelConfig,
) -> PipelinePlan:
    """
    Create a pipeline parallelism plan for a model.

    Distributes transformer layers evenly across stages.

    Args:
        model: The full model (must have a `layers` or `h` attribute)
        config: Pipeline parallelism configuration

    Returns:
        PipelinePlan with layer assignments per stage
    """
    # Find transformer layers
    layers = None
    for attr_name in ("layers", "h"):
        if hasattr(model, attr_name):
            layers = getattr(model, attr_name)
            break

    if layers is None:
        logger.warning("No transformer layers found; returning empty plan")
        return PipelinePlan()

    num_layers = len(layers)
    num_stages = config.num_stages

    # Divide layers evenly across stages
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages

    stages = []
    offset = 0
    for stage_id in range(num_stages):
        count = layers_per_stage + (1 if stage_id < remainder else 0)
        layer_indices = list(range(offset, offset + count))
        stages.append(PipelineStage(
            stage_id=stage_id,
            layer_indices=layer_indices,
            device=config.device,
        ))
        offset += count

    return PipelinePlan(stages=stages)


def create_layer_partition(
    model: nn.Module,
    num_stages: int,
) -> Dict[int, List[int]]:
    """
    Create a simple layer-to-stage mapping.

    Returns:
        Dict mapping stage_id -> list of layer indices
    """
    layers = None
    for attr_name in ("layers", "h"):
        if hasattr(model, attr_name):
            layers = getattr(model, attr_name)
            break

    if layers is None:
        return {}

    num_layers = len(layers)
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages

    partition = {}
    offset = 0
    for stage_id in range(num_stages):
        count = layers_per_stage + (1 if stage_id < remainder else 0)
        partition[stage_id] = list(range(offset, offset + count))
        offset += count

    return partition


class PipelineCommunicator:
    """
    Simulates inter-stage communication for pipeline parallelism.

    In production, this would use NCCL send/recv between GPUs.
    On CPU, it copies tensors between devices (or identity for same device).
    """

    def __init__(self, config: PipelineParallelConfig):
        self.config = config
        self._buffers: Dict[int, torch.Tensor] = {}

    def send(self, tensor: torch.Tensor, dst_stage: int) -> None:
        """Send activation tensor to the next stage."""
        self._buffers[dst_stage] = tensor.clone()

    def recv(self, src_stage: int) -> Optional[torch.Tensor]:
        """Receive activation tensor from the previous stage."""
        return self._buffers.pop(src_stage, None)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce across stages (identity for single device)."""
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast from src to all stages (identity for single device)."""
        return tensor

    def barrier(self) -> None:
        """Synchronization barrier."""
        pass

    def clear(self) -> None:
        """Clear buffered tensors."""
        self._buffers.clear()


class PipelineScheduler:
    """
    Manages micro-batch scheduling for pipeline parallelism.

    Implements a simple GPipe-style schedule where the full batch
    is split into micro-batches that flow through all stages.
    """

    def __init__(self, config: PipelineParallelConfig):
        self.config = config
        self.num_micro_batches = config.chunks

    def split_batch(
        self,
        hidden_states: torch.Tensor,
        num_micro_batches: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Split a batch into micro-batches."""
        n = num_micro_batches or self.num_micro_batches
        if n <= 1:
            return [hidden_states]

        batch_size = hidden_states.shape[0]
        micro_batch_size = max(1, batch_size // n)
        micro_batches = []

        for i in range(0, batch_size, micro_batch_size):
            micro_batches.append(hidden_states[i:i + micro_batch_size])

        return micro_batches

    def get_schedule(self) -> List[Tuple[str, int]]:
        """
        Get the forward/backward schedule for pipelining.

        Returns list of (operation, stage_id) tuples.
        """
        schedule = []
        for mb in range(self.num_micro_batches):
            for stage in range(self.config.num_stages):
                schedule.append(("forward", stage))
            for stage in reversed(range(self.config.num_stages)):
                schedule.append(("backward", stage))
        return schedule
