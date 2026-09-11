"""
Metrics Collector

Collects, aggregates, and reports runtime metrics from all Draco subsystems.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

import torch

from draco.metrics.types import (
    GenerationMetrics,
    GPUMetrics,
    KVMetrics,
    QuantizationMetrics,
    RequestMetrics,
    RuntimeMetrics,
    SchedulerMetrics,
    SpeculativeMetrics,
)


class MetricsCollector:
    """
    Collects and aggregates runtime metrics.

    Provides snapshot() for current state and per-request tracking.
    """

    def __init__(self, history_size: int = 1000) -> None:
        self._start_time = time.monotonic()
        self._history_size = history_size

        # Request latencies for percentile computation
        self._request_latencies: deque = deque(maxlen=history_size)
        self._ttft_values: deque = deque(maxlen=history_size)
        self._itl_values: deque = deque(maxlen=history_size)
        self._tpot_values: deque = deque(maxlen=history_size)

        # Cumulative counters
        self._total_requests = 0
        self._total_tokens_generated = 0
        self._total_tokens_prompt = 0

        # Current state
        self._active_requests = 0
        self._queued_requests = 0
        self._batch_size = 0

        # GPU metrics cache
        self._gpu_metrics_cache: Optional[GPUMetrics] = None
        self._gpu_cache_time: float = 0
        self._gpu_cache_ttl: float = 1.0  # refresh every second

        # Speculative decoding
        self._speculative = SpeculativeMetrics()

        # Quantization
        self._quantization = QuantizationMetrics()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> RuntimeMetrics:
        """Take a snapshot of all current metrics."""
        uptime = time.monotonic() - self._start_time

        return RuntimeMetrics(
            uptime_seconds=uptime,
            total_requests=self._total_requests,
            total_tokens_generated=self._total_tokens_generated,
            total_tokens_prompt=self._total_tokens_prompt,
            request=self._compute_request_metrics(),
            gpu=self._collect_gpu_metrics(),
            scheduler=SchedulerMetrics(
                active_requests=self._active_requests,
                queued_requests=self._queued_requests,
                batch_size=self._batch_size,
                scheduler_utilization=self._compute_scheduler_utilization(),
            ),
            kv_cache=KVMetrics(),
            generation=self._compute_generation_metrics(uptime),
            speculative=self._speculative,
            quantization=self._quantization,
        )

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def record_request(
        self,
        latency_ms: float,
        ttft_ms: float,
        itl_ms: float,
        tpot_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record metrics for a completed request."""
        self._total_requests += 1
        self._total_tokens_generated += output_tokens
        self._total_tokens_prompt += input_tokens

        self._request_latencies.append(latency_ms)
        self._ttft_values.append(ttft_ms)
        self._itl_values.append(itl_ms)
        self._tpot_values.append(tpot_ms)

    def request_start(self) -> None:
        """Mark a request as starting."""
        self._active_requests += 1
        self._queued_requests = max(0, self._queued_requests - 1)

    def request_end(self) -> None:
        """Mark a request as ending."""
        self._active_requests = max(0, self._active_requests - 1)

    def enqueue_request(self) -> None:
        """Mark a request as queued."""
        self._queued_requests += 1

    def set_batch_size(self, batch_size: int) -> None:
        """Set current batch size."""
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Speculative decoding metrics
    # ------------------------------------------------------------------

    def record_speculative(
        self,
        draft_tokens: int,
        accepted_tokens: int,
        rejected_tokens: int,
        draft_latency_ms: float = 0.0,
        verify_latency_ms: float = 0.0,
    ) -> None:
        """Record speculative decoding metrics."""
        self._speculative.draft_tokens += draft_tokens
        self._speculative.accepted_tokens += accepted_tokens
        self._speculative.rejected_tokens += rejected_tokens
        total = accepted_tokens + rejected_tokens
        if total > 0:
            self._speculative.acceptance_rate = accepted_tokens / total
        self._speculative.draft_latency_ms = draft_latency_ms
        self._speculative.verification_latency_ms = verify_latency_ms

        if self._speculative.target_latency_ms > 0:
            baseline = self._speculative.target_latency_ms
            speculative = self._speculative.draft_latency_ms + self._speculative.verification_latency_ms
            if speculative > 0:
                self._speculative.speedup = baseline / speculative

    # ------------------------------------------------------------------
    # Quantization metrics
    # ------------------------------------------------------------------

    def record_quantization(
        self,
        method: str,
        bits: int,
        model_memory: int,
        baseline_memory: int,
    ) -> None:
        """Record quantization metrics."""
        self._quantization.quantization_method = method
        self._quantization.bits = bits
        self._quantization.model_memory_bytes = model_memory
        self._quantization.quantized_memory_bytes = model_memory
        self._quantization.baseline_memory_bytes = baseline_memory
        if baseline_memory > 0:
            self._quantization.memory_reduction = 1.0 - (model_memory / baseline_memory)

    # ------------------------------------------------------------------
    # Internal computations
    # ------------------------------------------------------------------

    def _compute_request_metrics(self) -> RequestMetrics:
        """Compute aggregate request metrics."""
        metrics = RequestMetrics()

        if self._request_latencies:
            lats = sorted(self._request_latencies)
            n = len(lats)
            metrics.p50_latency_ms = lats[n // 2]
            metrics.p95_latency_ms = lats[int(n * 0.95)] if n > 1 else lats[-1]
            metrics.p99_latency_ms = lats[int(n * 0.99)] if n > 1 else lats[-1]
            metrics.request_latency_ms = sum(lats) / n

        if self._ttft_values:
            metrics.ttft_ms = sum(self._ttft_values) / len(self._ttft_values)
        if self._itl_values:
            metrics.itl_ms = sum(self._itl_values) / len(self._itl_values)
        if self._tpot_values:
            metrics.tpot_ms = sum(self._tpot_values) / len(self._tpot_values)

        metrics.total_tokens = self._total_tokens_generated + self._total_tokens_prompt
        metrics.input_tokens = self._total_tokens_prompt
        metrics.output_tokens = self._total_tokens_generated

        return metrics

    def _compute_generation_metrics(self, uptime: float) -> GenerationMetrics:
        """Compute generation throughput metrics."""
        metrics = GenerationMetrics()
        metrics.total_generated_tokens = self._total_tokens_generated
        metrics.total_prompt_tokens = self._total_tokens_prompt

        if uptime > 0:
            metrics.tokens_per_second = self._total_tokens_generated / uptime
            metrics.prompt_tokens_per_second = self._total_tokens_prompt / uptime
            if self._total_requests > 0:
                metrics.avg_sequence_length = (
                    (self._total_tokens_generated + self._total_tokens_prompt) / self._total_requests
                )

        return metrics

    def _compute_scheduler_utilization(self) -> float:
        """Compute scheduler utilization."""
        total = self._active_requests + self._queued_requests
        if total == 0:
            return 0.0
        return self._active_requests / max(total, 1)

    def _collect_gpu_metrics(self) -> GPUMetrics:
        """Collect GPU metrics from CUDA."""
        now = time.monotonic()
        if self._gpu_metrics_cache and (now - self._gpu_cache_time) < self._gpu_cache_ttl:
            return self._gpu_metrics_cache

        metrics = GPUMetrics()

        if not torch.cuda.is_available():
            self._gpu_metrics_cache = metrics
            self._gpu_cache_time = now
            return metrics

        try:
            metrics.gpu_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            metrics.compute_capability = f"{cap[0]}.{cap[1]}"

            total_mem = torch.cuda.get_device_properties(0).total_mem
            metrics.total_memory_gb = total_mem / (1024 ** 3)
            metrics.allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            metrics.reserved_memory_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
            metrics.free_memory_gb = (total_mem - torch.cuda.memory_allocated(0)) / (1024 ** 3)

            # Utilization
            if hasattr(torch.cuda, "utilization"):
                metrics.utilization_percent = torch.cuda.utilization(0)

            # Temperature and power via nvidia-smi if available
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 2:
                        metrics.temperature_celsius = float(parts[0])
                        metrics.power_watts = float(parts[1])
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass

        except Exception:
            pass

        self._gpu_metrics_cache = metrics
        self._gpu_cache_time = now
        return metrics

    # ------------------------------------------------------------------
    # Prometheus export
    # ------------------------------------------------------------------

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus-compatible format.

        Returns text suitable for /metrics endpoint.
        """
        snapshot = self.snapshot()
        lines = []

        def add(name: str, value: float, help_text: str = "", metric_type: str = "gauge"):
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {value}")

        add("draco_requests_total", snapshot.total_requests, "Total requests processed", "counter")
        add("draco_tokens_generated_total", snapshot.total_tokens_generated, "Total tokens generated", "counter")
        add("draco_tokens_prompt_total", snapshot.total_tokens_prompt, "Total prompt tokens", "counter")
        add("draco_ttft_seconds", snapshot.request.ttft_ms / 1000.0, "Time to first token in seconds")
        add("draco_itl_seconds", snapshot.request.itl_ms / 1000.0, "Inter-token latency in seconds")
        add("draco_tpot_seconds", snapshot.request.tpot_ms / 1000.0, "Time per output token in seconds")
        add("draco_generation_tokens_per_second", snapshot.generation.tokens_per_second, "Generation throughput")
        add("draco_gpu_memory_bytes", snapshot.gpu.allocated_memory_gb * 1024**3, "GPU memory allocated in bytes")
        add("draco_gpu_utilization", snapshot.gpu.utilization_percent / 100.0, "GPU utilization")
        add("draco_kv_cache_utilization", snapshot.kv_cache.utilization, "KV cache utilization")
        add("draco_active_requests", snapshot.scheduler.active_requests, "Active requests")
        add("draco_queued_requests", snapshot.scheduler.queued_requests, "Queued requests")

        return "\n".join(lines)
