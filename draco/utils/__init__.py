"""Utility modules: memory profiling / KV-cache capacity planning."""

from draco.utils.memory import (
    DeviceMemoryInfo,
    MemoryProfiler,
    bytes_per_kv_block,
    dtype_size_bytes,
    get_device_memory_info,
    plan_kv_cache_blocks,
)

__all__ = [
    "DeviceMemoryInfo",
    "MemoryProfiler",
    "bytes_per_kv_block",
    "dtype_size_bytes",
    "get_device_memory_info",
    "plan_kv_cache_blocks",
]
