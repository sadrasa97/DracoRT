"""
Metrics Types

Data classes for all metric categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeMetrics:
    """Top-level runtime metrics snapshot."""

    uptime_seconds: float = 0.0
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_tokens_prompt: int = 0

    # Request metrics
    request: "RequestMetrics" = field(default_factory=lambda: RequestMetrics())
    # GPU metrics
    gpu: "GPUMetrics" = field(default_factory=lambda: GPUMetrics())
    # Scheduler metrics
    scheduler: "SchedulerMetrics" = field(default_factory=lambda: SchedulerMetrics())
    # KV cache metrics
    kv_cache: "KVMetrics" = field(default_factory=lambda: KVMetrics())
    # Generation metrics
    generation: "GenerationMetrics" = field(default_factory=lambda: GenerationMetrics())
    # Speculative decoding metrics
    speculative: "SpeculativeMetrics" = field(default_factory=lambda: SpeculativeMetrics())
    # Quantization metrics
    quantization: "QuantizationMetrics" = field(default_factory=lambda: QuantizationMetrics())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_prompt": self.total_tokens_prompt,
            "request": self.request.to_dict(),
            "gpu": self.gpu.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "kv_cache": self.kv_cache.to_dict(),
            "generation": self.generation.to_dict(),
            "speculative": self.speculative.to_dict(),
            "quantization": self.quantization.to_dict(),
        }


@dataclass
class RequestMetrics:
    """Per-request latency metrics."""

    # Latency metrics
    ttft_ms: float = 0.0          # Time to First Token
    itl_ms: float = 0.0           # Inter-Token Latency
    tpot_ms: float = 0.0          # Time Per Output Token
    request_latency_ms: float = 0.0
    queue_latency_ms: float = 0.0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Throughput
    prompt_tokens_per_sec: float = 0.0
    generation_tokens_per_sec: float = 0.0

    # Percentiles (computed over multiple requests)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "itl_ms": self.itl_ms,
            "tpot_ms": self.tpot_ms,
            "request_latency_ms": self.request_latency_ms,
            "queue_latency_ms": self.queue_latency_ms,
            "prefill_latency_ms": self.prefill_latency_ms,
            "decode_latency_ms": self.decode_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens_per_sec": self.prompt_tokens_per_sec,
            "generation_tokens_per_sec": self.generation_tokens_per_sec,
        }


@dataclass
class GPUMetrics:
    """GPU hardware metrics."""

    gpu_name: str = "unknown"
    compute_capability: str = "unknown"
    total_memory_gb: float = 0.0
    allocated_memory_gb: float = 0.0
    reserved_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_watts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "total_memory_gb": self.total_memory_gb,
            "allocated_memory_gb": self.allocated_memory_gb,
            "reserved_memory_gb": self.reserved_memory_gb,
            "free_memory_gb": self.free_memory_gb,
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
            "power_watts": self.power_watts,
        }


@dataclass
class SchedulerMetrics:
    """Scheduler metrics."""

    active_requests: int = 0
    queued_requests: int = 0
    batch_size: int = 0
    scheduler_utilization: float = 0.0
    total_scheduled: int = 0
    preemptions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "batch_size": self.batch_size,
            "scheduler_utilization": self.scheduler_utilization,
            "total_scheduled": self.total_scheduled,
            "preemptions": self.preemptions,
        }


@dataclass
class KVMetrics:
    """KV cache metrics."""

    total_blocks: int = 0
    used_blocks: int = 0
    free_blocks: int = 0
    utilization: float = 0.0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    allocated_memory_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "utilization": self.utilization,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "allocated_memory_gb": self.allocated_memory_gb,
        }


@dataclass
class GenerationMetrics:
    """Generation-specific metrics."""

    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    total_generated_tokens: int = 0
    total_prompt_tokens: int = 0
    avg_sequence_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "total_generated_tokens": self.total_generated_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "avg_sequence_length": self.avg_sequence_length,
        }


@dataclass
class SpeculativeMetrics:
    """Speculative decoding metrics."""

    draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    acceptance_rate: float = 0.0
    verification_latency_ms: float = 0.0
    draft_latency_ms: float = 0.0
    target_latency_ms: float = 0.0
    speedup: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "acceptance_rate": self.acceptance_rate,
            "verification_latency_ms": self.verification_latency_ms,
            "draft_latency_ms": self.draft_latency_ms,
            "target_latency_ms": self.target_latency_ms,
            "speedup": self.speedup,
        }


@dataclass
class QuantizationMetrics:
    """Quantization-related metrics."""

    model_memory_bytes: int = 0
    baseline_memory_bytes: int = 0  # FP16/BF16 baseline
    quantized_memory_bytes: int = 0
    memory_reduction: float = 0.0
    quantization_method: str = "none"
    bits: int = 16
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_memory_bytes": self.model_memory_bytes,
            "baseline_memory_bytes": self.baseline_memory_bytes,
            "quantized_memory_bytes": self.quantized_memory_bytes,
            "memory_reduction": self.memory_reduction,
            "quantization_method": self.quantization_method,
            "bits": self.bits,
            "tokens_per_second": self.tokens_per_second,
            "latency_ms": self.latency_ms,
        }
