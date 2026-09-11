"""
Memory utilities: GPU memory profiling and KV-cache capacity planning.

`gpu_memory_utilization` is accepted as a constructor argument all the way
through `LLM`, `AsyncLLM`, `ModelRunner`, and the CLI — but nothing in the
codebase actually reads current GPU memory and uses that fraction to decide
how many KV-cache blocks to allocate. Every `KVCacheBlockManager` is built
with a fixed, hand-picked `num_blocks` regardless of how much memory is
actually available or how big the model's weights are. This module closes
that gap:

  - `get_device_memory_info` reads real (or CPU-fallback) memory totals.
  - `plan_kv_cache_blocks` turns "fraction of GPU memory to use" + "bytes
    already used by model weights/activations" into a concrete block count,
    the same computation vLLM calls `determine_num_available_blocks`.
  - `MemoryProfiler` is a small context manager to measure peak CUDA memory
    used by a "profile run" (e.g. one dummy forward pass at max batch
    size), so the block count accounts for activation memory too, not just
    weights.

Every sizing function here is pure (takes byte counts in, returns numbers
out) specifically so it can be tested without a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger("draco.utils.memory")


@dataclass
class DeviceMemoryInfo:
    """A snapshot of a device's memory state, in bytes."""
    total_bytes: int
    free_bytes: int
    used_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)


def get_device_memory_info(device: Optional[torch.device] = None) -> DeviceMemoryInfo:
    """Read current memory totals for a device.

    On CUDA devices this uses `torch.cuda.mem_get_info`, which reports the
    real free/total memory on the physical GPU (not just what this process
    has allocated through PyTorch's caching allocator). On CPU — or when
    CUDA isn't available/importable — this falls back to `psutil` if
    present, or a conservative placeholder otherwise so callers still get a
    usable (if approximate) answer instead of a crash.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_bytes = total_bytes - free_bytes
        return DeviceMemoryInfo(total_bytes=total_bytes, free_bytes=free_bytes, used_bytes=used_bytes)

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return DeviceMemoryInfo(total_bytes=vm.total, free_bytes=vm.available, used_bytes=vm.total - vm.available)
    except ImportError:
        logger.warning(
            "psutil not available and device is not CUDA — memory info is a rough placeholder. "
            "Install psutil for accurate CPU memory reporting."
        )
        placeholder_total = 16 * (1024 ** 3)
        return DeviceMemoryInfo(total_bytes=placeholder_total, free_bytes=placeholder_total // 2, used_bytes=placeholder_total // 2)


def dtype_size_bytes(dtype: torch.dtype) -> int:
    """Size in bytes of a single element of `dtype`."""
    return torch.tensor([], dtype=dtype).element_size()


def bytes_per_kv_block(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
) -> int:
    """Bytes needed for one KV-cache block (both key and value, all layers).

    Mirrors `KVCacheBlock`'s tensor shapes exactly: each of key_cache and
    value_cache is (num_layers, num_kv_heads, block_size, head_dim).
    """
    elem_size = dtype_size_bytes(dtype)
    per_tensor = num_layers * num_kv_heads * block_size * head_dim * elem_size
    return 2 * per_tensor  # key + value


def plan_kv_cache_blocks(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    gpu_memory_utilization: float = 0.90,
    device: Optional[torch.device] = None,
    model_weight_bytes: int = 0,
    activation_reserve_bytes: int = 0,
    min_blocks: int = 1,
) -> int:
    """Compute how many KV-cache blocks fit in the memory budget.

    budget = total_memory * gpu_memory_utilization
    available_for_kv = budget - already_used - model_weight_bytes - activation_reserve_bytes
    num_blocks = floor(available_for_kv / bytes_per_block)

    This is the same shape of calculation vLLM's memory profiler does
    (`determine_num_available_blocks`): decide a utilization ceiling, then
    subtract everything else that's already claiming memory, and whatever
    is left becomes KV cache capacity.

    Args:
        num_layers, num_kv_heads, head_dim, block_size, dtype: KV cache
            block shape/dtype, matching `KVCacheBlockManager`.
        gpu_memory_utilization: fraction (0, 1] of total device memory the
            engine is allowed to use in total (weights + activations + KV
            cache). Same knob already threaded through `LLM`/`ModelRunner`.
        device: device to query; defaults to CUDA if available else CPU.
        model_weight_bytes: bytes already consumed by loaded model weights
            (subtracted from the budget before computing KV capacity).
        activation_reserve_bytes: extra bytes to reserve for peak
            activation memory during a forward pass (e.g. measured via
            `MemoryProfiler`, or a conservative manual estimate).
        min_blocks: floor on the returned block count — raises if the
            computed capacity is below this, since a 0-or-negative block
            budget means the engine cannot serve any requests at all and
            failing loudly here is far more debuggable than an opaque
            "KV cache block pool exhausted" error on the first request.

    Returns:
        Number of KV-cache blocks that fit in the remaining budget.

    Raises:
        ValueError: if `gpu_memory_utilization` is out of (0, 1], or if the
            computed budget can't even fit `min_blocks` blocks.
    """
    if not (0.0 < gpu_memory_utilization <= 1.0):
        raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}")

    mem_info = get_device_memory_info(device)
    budget_bytes = int(mem_info.total_bytes * gpu_memory_utilization)
    reserved = mem_info.used_bytes + model_weight_bytes + activation_reserve_bytes
    available_for_kv = budget_bytes - reserved

    per_block = bytes_per_kv_block(num_layers, num_kv_heads, head_dim, block_size, dtype)
    if per_block <= 0:
        raise ValueError("Computed 0 bytes per KV cache block — check num_layers/num_kv_heads/head_dim/block_size")

    num_blocks = max(0, available_for_kv) // per_block

    if num_blocks < min_blocks:
        raise ValueError(
            f"Only {num_blocks} KV cache blocks fit in the memory budget "
            f"(total={mem_info.total_gb:.2f}GB, utilization={gpu_memory_utilization:.0%}, "
            f"already used={mem_info.used_gb:.2f}GB, model weights={model_weight_bytes / (1024**3):.2f}GB, "
            f"activation reserve={activation_reserve_bytes / (1024**3):.2f}GB) — need at least {min_blocks}. "
            f"Reduce max_model_len/batch size, use a smaller block_size, quantize the model, "
            f"or raise gpu_memory_utilization."
        )

    logger.info(
        "KV cache capacity plan: %d blocks (%d tokens) — total=%.2fGB util=%.0f%% "
        "reserved(used+weights+activation)=%.2fGB per_block=%dB",
        num_blocks, num_blocks * block_size, mem_info.total_gb, gpu_memory_utilization * 100,
        reserved / (1024 ** 3), per_block,
    )
    return int(num_blocks)


class MemoryProfiler:
    """Context manager that measures peak CUDA memory allocated in a block.

    Usage:
        with MemoryProfiler(device) as prof:
            model(dummy_input)  # a representative "profile run"
        activation_bytes = prof.peak_bytes

    On non-CUDA devices `peak_bytes` is always 0 (there's no equivalent
    lightweight peak-memory counter for CPU tensors), so callers should
    treat that as "unknown" and fall back to a manual
    `activation_reserve_bytes` estimate for `plan_kv_cache_blocks`.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.peak_bytes = 0
        self._start_bytes = 0

    def __enter__(self) -> "MemoryProfiler":
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            self._start_bytes = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            self.peak_bytes = max(0, peak - self._start_bytes)
        # Exceptions (if any) propagate normally; we only measure here.
