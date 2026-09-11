"""Draco Tensor Parallelism framework."""

from draco.parallel.tp import TensorParallelConfig, split_tensor, merge_tensor

__all__ = ["TensorParallelConfig", "split_tensor", "merge_tensor"]
