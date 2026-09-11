"""
Benchmark Framework

Core framework for running benchmarks and collecting results.
Records full environment metadata for reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("draco.benchmarks")


@dataclass
class EnvironmentInfo:
    """Environment information for reproducibility."""

    draco_version: str = "0.1.0"
    git_commit: str = ""
    timestamp: str = ""

    # Hardware
    gpu_model: str = ""
    gpu_memory_gb: float = 0.0
    compute_capability: str = ""

    # Software
    cuda_version: str = ""
    pytorch_version: str = ""
    python_version: str = ""
    driver_version: str = ""

    @classmethod
    def collect(cls) -> EnvironmentInfo:
        """Collect current environment information."""
        import sys
        import torch

        info = cls()
        info.timestamp = datetime.now(timezone.utc).isoformat()
        info.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        info.pytorch_version = torch.__version__

        # CUDA
        if torch.cuda.is_available():
            info.gpu_model = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.gpu_memory_gb = props.total_mem / (1024 ** 3)
            cap = torch.cuda.get_device_capability(0)
            info.compute_capability = f"{cap[0]}.{cap[1]}"
            info.cuda_version = torch.version.cuda or "unknown"

        # Git commit
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=2,
                cwd=str(Path(__file__).parent.parent.parent),
            )
            if result.returncode == 0:
                info.git_commit = result.stdout.strip()
        except Exception:
            pass

        # Driver version
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                info.driver_version = result.stdout.strip()
        except Exception:
            pass

        return info


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    model: str
    dtype: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    quantization: str = "none"

    # Timing
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    itl_ms: float = 0.0
    total_time_ms: float = 0.0

    # Throughput
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0

    # Memory
    gpu_memory_gb: float = 0.0
    peak_gpu_memory_gb: float = 0.0

    # Latency percentiles
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Additional
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def format_table(self) -> str:
        """Format as readable table."""
        lines = [
            f"{'Draco Benchmark':-^44}",
            "",
            f"  Model:      {self.model}",
            f"  dtype:      {self.dtype}",
            f"  Batch:      {self.batch_size}",
            f"  Input:      {self.input_tokens} tokens",
            f"  Output:     {self.output_tokens} tokens",
            f"  Quant:      {self.quantization}",
            "",
            f"  TTFT:       {self.ttft_ms:.1f} ms",
            f"  TPOT:       {self.tpot_ms:.1f} ms",
            f"  ITL:        {self.itl_ms:.1f} ms",
            f"  Throughput: {self.tokens_per_second:.1f} tok/s",
            f"  Prompt:     {self.prompt_tokens_per_second:.1f} tok/s",
            f"  Total Time: {self.total_time_ms:.1f} ms",
            f"  GPU Mem:    {self.gpu_memory_gb:.2f} GB",
            "",
            f"  p50:        {self.p50_latency_ms:.1f} ms",
            f"  p95:        {self.p95_latency_ms:.1f} ms",
            f"  p99:        {self.p99_latency_ms:.1f} ms",
        ]
        return "\n".join(lines)


class BenchmarkRunner:
    """
    Core benchmark runner.

    Manages benchmark execution, timing, and result collection.
    """

    def __init__(
        self,
        output_dir: str = "benchmarks/results",
        save_results: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.save_results = save_results
        self.env = EnvironmentInfo.collect()

    def run_benchmark(
        self,
        name: str,
        benchmark_fn: Any,
        model: str,
        dtype: str = "float16",
        batch_size: int = 1,
        input_tokens: int = 512,
        output_tokens: int = 128,
        quantization: str = "none",
        num_iterations: int = 1,
        warmup_iterations: int = 1,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Run a benchmark function and collect timing results.

        Args:
            name: Benchmark name
            benchmark_fn: Callable that runs the benchmark
            model: Model identifier
            dtype: Data type
            batch_size: Batch size
            input_tokens: Input token count
            output_tokens: Output token count
            quantization: Quantization method
            num_iterations: Number of measured iterations
            warmup_iterations: Warmup iterations (not measured)
            **kwargs: Additional arguments passed to benchmark_fn

        Returns:
            BenchmarkResult with timing and throughput data.
        """
        logger.info("Running benchmark: %s", name)
        logger.info(
            "Model: %s, dtype: %s, batch: %d, input: %d, output: %d",
            model, dtype, batch_size, input_tokens, output_tokens,
        )

        # Warmup
        for i in range(warmup_iterations):
            logger.info("Warmup %d/%d", i + 1, warmup_iterations)
            benchmark_fn(batch_size=batch_size, input_tokens=input_tokens,
                        output_tokens=output_tokens, **kwargs)

        # Measured runs
        latencies = []
        token_counts = []
        ttft_values = []
        tpot_values = []
        itl_values = []
        memory_values = []

        for i in range(num_iterations):
            logger.info("Iteration %d/%d", i + 1, num_iterations)

            start_time = time.perf_counter()

            # Run benchmark
            result_data = benchmark_fn(
                batch_size=batch_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                **kwargs,
            )

            end_time = time.perf_counter()
            total_ms = (end_time - start_time) * 1000

            latencies.append(total_ms)

            # Extract metrics from result_data if available
            if isinstance(result_data, dict):
                ttft_values.append(result_data.get("ttft_ms", 0.0))
                tpot_values.append(result_data.get("tpot_ms", 0.0))
                itl_values.append(result_data.get("itl_ms", 0.0))
                token_counts.append(result_data.get("tokens_generated", output_tokens))
                memory_values.append(result_data.get("gpu_memory_gb", 0.0))
            else:
                token_counts.append(output_tokens)
                memory_values.append(0.0)

        # Compute aggregate metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_tokens = sum(token_counts)
        total_time_s = sum(latencies) / 1000.0
        tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        result = BenchmarkResult(
            name=name,
            model=model,
            dtype=dtype,
            batch_size=batch_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quantization=quantization,
            ttft_ms=sum(ttft_values) / len(ttft_values) if ttft_values else 0.0,
            tpot_ms=sum(tpot_values) / len(tpot_values) if tpot_values else 0.0,
            itl_ms=sum(itl_values) / len(itl_values) if itl_values else 0.0,
            total_time_ms=avg_latency,
            tokens_per_second=tokens_per_sec,
            prompt_tokens_per_second=input_tokens / (avg_latency / 1000.0) if avg_latency > 0 else 0,
            gpu_memory_gb=max(memory_values) if memory_values else 0.0,
            p50_latency_ms=sorted_latencies[n // 2] if n > 0 else 0,
            p95_latency_ms=sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1] if n > 0 else 0,
            p99_latency_ms=sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1] if n > 0 else 0,
            metadata={
                "num_iterations": num_iterations,
                "warmup_iterations": warmup_iterations,
                "environment": asdict(self.env),
            },
        )

        # Save results
        if self.save_results:
            self._save_result(result)

        logger.info("Benchmark '%s' complete: %.1f tok/s, %.1f ms",
                     name, result.tokens_per_second, result.total_time_ms)
        return result

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        model_name = result.model.replace("/", "_").replace(" ", "_")
        filename = f"{model_name}_{result.dtype}_{result.quantization}.json"
        filepath = self.output_dir / filename

        # Load existing results and append
        results = []
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    results = data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, IOError):
                pass

        results.append(result.to_dict())

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Saved benchmark result to %s", filepath)

    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        """Load benchmark results from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    def compare(
        self, results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], label_a: str = "A", label_b: str = "B"
    ) -> str:
        """Compare two sets of benchmark results."""
        lines = [f"{'Comparison':-^44}", ""]

        for key in ("ttft_ms", "tpot_ms", "tokens_per_second", "gpu_memory_gb"):
            val_a = sum(r.get(key, 0) for r in results_a) / max(len(results_a), 1)
            val_b = sum(r.get(key, 0) for r in results_b) / max(len(results_b), 1)
            diff = ((val_b - val_a) / val_a * 100) if val_a > 0 else 0
            sign = "+" if diff > 0 else ""
            lines.append(f"  {key:30s}  {label_a}: {val_a:>10.2f}  {label_b}: {val_b:>10.2f}  ({sign}{diff:.1f}%)")

        return "\n".join(lines)
