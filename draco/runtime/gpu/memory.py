"""
GPU memory detection + planning — the GPU-side counterpart to
draco.runtime.cpu.memory / draco.runtime.cpu.mem_util.

Mirrors draco.config.DracoConfig.default_gpu_memory_utilization (0.90):
detects total/free device memory when torch+CUDA are available, and
derives a usable budget as ``free * gpu_memory_utilization``. When no
GPU is present (as in this sandbox), detection reports zero rather than
guessing — callers should treat that as "no GPU available", not "GPU
with 0 bytes free".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

_BYTES_PER_DTYPE = {"f32": 4, "f16": 2, "bf16": 2, "int8": 1, "int4": 0.5}


@dataclass
class DeviceMemoryInfo:
    available: bool
    total_bytes: int = 0
    free_bytes: int = 0
    device_name: str = "none"


def detect_device_memory(device_index: int = 0) -> DeviceMemoryInfo:
    try:
        import torch
    except Exception:
        return DeviceMemoryInfo(available=False)

    try:
        if not torch.cuda.is_available():
            return DeviceMemoryInfo(available=False)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        name = torch.cuda.get_device_name(device_index)
        return DeviceMemoryInfo(
            available=True, total_bytes=total_bytes, free_bytes=free_bytes, device_name=name
        )
    except Exception:
        return DeviceMemoryInfo(available=False)


def auto_memory_budget(
    utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION, info: Optional[DeviceMemoryInfo] = None
) -> int:
    if not (0.0 < utilization <= 1.0):
        raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {utilization}.")
    device = info or detect_device_memory()
    if not device.available:
        raise RuntimeError(
            "No CUDA-capable GPU detected; cannot compute a GPU memory budget. "
            "Use the CPU-native runtime (draco.runtime.cpu) instead, or check your "
            "torch/CUDA install."
        )
    return int(device.free_bytes * utilization)


@dataclass
class GPUInferenceEstimate:
    weights_bytes: int
    kv_cache_bytes: int
    activation_workspace_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.activation_workspace_bytes


class GPUMemoryPlanner:
    """Same shape as draco.runtime.cpu.memory.CPUMemoryPlanner, for the GPU
    device — kept as a separate class rather than a shared base since the
    two have different budget sources (device memory vs. host RAM) even
    though the arithmetic is similar."""

    def estimate_inference(
        self,
        num_params: int,
        dtype: str,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        max_model_len: int,
        max_num_seqs: int,
        kv_cache_dtype: str = "f16",
        activation_workspace_bytes: int = 512 * 1024 * 1024,
    ) -> GPUInferenceEstimate:
        weights_bytes = int(num_params * _BYTES_PER_DTYPE.get(dtype, 2))
        kv_bytes_per_token = (
            2 * num_layers * num_key_value_heads * head_dim * _BYTES_PER_DTYPE.get(kv_cache_dtype, 2)
        )
        kv_cache_bytes = int(kv_bytes_per_token * max_model_len * max_num_seqs)
        return GPUInferenceEstimate(
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            activation_workspace_bytes=activation_workspace_bytes,
        )

    def max_num_seqs_that_fit(
        self,
        num_params: int,
        dtype: str,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        max_model_len: int,
        budget_bytes: int,
        kv_cache_dtype: str = "f16",
    ) -> int:
        """Binary-search-free direct solve: how many concurrent sequences'
        worth of KV cache fit after weights are loaded, used to keep the
        scheduler from admitting more sequences than memory allows (the
        actual OOM-avoidance mechanism — never admit past the budget,
        rather than admit-and-hope)."""
        weights_bytes = int(num_params * _BYTES_PER_DTYPE.get(dtype, 2))
        remaining = budget_bytes - weights_bytes
        if remaining <= 0:
            return 0
        kv_bytes_per_seq = (
            2 * num_layers * num_key_value_heads * head_dim
            * _BYTES_PER_DTYPE.get(kv_cache_dtype, 2) * max_model_len
        )
        if kv_bytes_per_seq <= 0:
            return 0
        return max(0, int(remaining // kv_bytes_per_seq))
