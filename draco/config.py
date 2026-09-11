"""Draco configuration and global settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DracoConfig:
    """Global Draco configuration."""

    # Default model settings
    default_dtype: str = "auto"
    default_max_model_len: int = 8192
    default_gpu_memory_utilization: float = 0.90
    default_tensor_parallel_size: int = 1

    # Quantization defaults
    default_quantization: Optional[str] = None

    # Scheduler settings
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    # KV cache settings
    kv_cache_block_size: int = 16
    enable_paged_attention: bool = True

    # Metrics settings
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Benchmark settings
    benchmark_output_dir: str = "benchmarks/results"

    # Debug
    verbose: bool = False
    log_level: str = os.environ.get("DRACO_LOG_LEVEL", "INFO")


# Global config singleton
_config: Optional[DracoConfig] = None


def get_config() -> DracoConfig:
    """Get global Draco configuration."""
    global _config
    if _config is None:
        _config = DracoConfig()
    return _config


def set_config(config: DracoConfig) -> None:
    """Set global Draco configuration."""
    global _config
    _config = config
