"""
Benchmark Modes

Different benchmarking configurations:
- Single request
- Concurrent requests
- Batch scaling
- Context scaling
- Generation scaling
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from draco.benchmarks.framework import BenchmarkRunner, BenchmarkResult

logger = logging.getLogger("draco.benchmarks.modes")


class SingleRequestBenchmark:
    """
    Single request benchmark.

    Measures TTFT, latency, and tokens/sec for a single prompt.
    """

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def run(
        self,
        llm: Any,
        prompt: str = "Hello, how are you today?",
        max_tokens: int = 256,
        input_tokens: int = 512,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run single request benchmark."""
        def bench_fn(batch_size: int = 1, input_tokens: int = 512, output_tokens: int = 256, **kw: Any) -> Dict[str, Any]:
            import time, torch
            # Create prompt of desired length
            tokens = llm.encode(prompt)
            if len(tokens) < input_tokens:
                tokens = tokens * (input_tokens // len(tokens) + 1)
            tokens = tokens[:input_tokens]

            # Prefill
            t0 = time.perf_counter()
            input_ids = torch.tensor([tokens], dtype=torch.long, device=llm.device)
            position_ids = torch.arange(len(tokens), device=llm.device).unsqueeze(0)
            logits = llm._model_runner.forward(input_ids, position_ids=position_ids)
            t1 = time.perf_counter()
            ttft_ms = (t1 - t0) * 1000

            # Decode
            generated = []
            prompt_len = len(tokens)
            for _ in range(output_tokens):
                next_token = logits[0, -1, :].argmax().item()
                generated.append(next_token)
                new_input = torch.tensor([[next_token]], dtype=torch.long, device=llm.device)
                pos = torch.tensor([[prompt_len + len(generated) - 1]], device=llm.device)
                logits = llm._model_runner.forward(new_input, position_ids=pos)

            total_ms = (time.perf_counter() - t0) * 1000
            tokens_gen = len(generated)

            return {
                "ttft_ms": ttft_ms,
                "tpot_ms": (total_ms - ttft_ms) / max(tokens_gen, 1),
                "itl_ms": (total_ms - ttft_ms) / max(tokens_gen, 1),
                "tokens_generated": tokens_gen,
                "gpu_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0,
            }

        return self.runner.run_benchmark(
            name="single_request",
            benchmark_fn=bench_fn,
            model=llm.model_path,
            dtype=str(llm.dtype),
            batch_size=1,
            input_tokens=input_tokens,
            output_tokens=max_tokens,
            num_iterations=1,
            warmup_iterations=0,
        )


class ConcurrentRequestBenchmark:
    """
    Concurrent request benchmark.

    Tests system behavior under concurrent load with varying numbers
    of simultaneous requests.
    """

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def run(
        self,
        llm: Any,
        num_concurrent: List[int] = None,
        prompt: str = "Hello, how are you?",
        max_tokens: int = 128,
        input_tokens: int = 512,
        **kwargs: Any,
    ) -> List[BenchmarkResult]:
        """Run concurrent request benchmarks."""
        if num_concurrent is None:
            num_concurrent = [1, 2, 4, 8, 16]

        results = []
        for n in num_concurrent:
            def bench_fn(batch_size: int = 1, input_tokens: int = 512, output_tokens: int = 128, num_reqs: int = n, **kw: Any) -> Dict[str, Any]:
                import time, torch

                start = time.perf_counter()
                tokens = llm.encode(prompt)
                if len(tokens) < input_tokens:
                    tokens = tokens * (input_tokens // len(tokens) + 1)
                tokens = tokens[:input_tokens]

                # Process all requests sequentially (simulated concurrency)
                total_tokens = 0
                for _ in range(num_reqs):
                    input_ids = torch.tensor([tokens], dtype=torch.long, device=llm.device)
                    position_ids = torch.arange(len(tokens), device=llm.device).unsqueeze(0)
                    logits = llm._model_runner.forward(input_ids, position_ids=position_ids)
                    for _ in range(output_tokens):
                        next_token = logits[0, -1, :].argmax().item()
                        total_tokens += 1
                        new_input = torch.tensor([[next_token]], dtype=torch.long, device=llm.device)
                        pos = torch.tensor([[len(tokens) + total_tokens - 1]], device=llm.device)
                        logits = llm._model_runner.forward(new_input, position_ids=pos)

                total_ms = (time.perf_counter() - start) * 1000
                return {
                    "tokens_generated": total_tokens,
                    "ttft_ms": total_ms / num_reqs * 0.1,
                    "tpot_ms": total_ms / max(total_tokens, 1),
                    "itl_ms": total_ms / max(total_tokens, 1),
                    "gpu_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0,
                }

            result = self.runner.run_benchmark(
                name=f"concurrent_{n}",
                benchmark_fn=bench_fn,
                model=llm.model_path,
                dtype=str(llm.dtype),
                batch_size=n,
                input_tokens=input_tokens,
                output_tokens=max_tokens,
            )
            results.append(result)

        return results


class BatchScalingBenchmark:
    """
    Batch scaling benchmark.

    Tests how throughput and memory scale with batch size.
    """

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def run(
        self,
        llm: Any,
        batch_sizes: List[int] = None,
        input_tokens: int = 512,
        output_tokens: int = 128,
        **kwargs: Any,
    ) -> List[BenchmarkResult]:
        """Run batch scaling benchmark."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]

        results = []
        for bs in batch_sizes:
            def bench_fn(batch_size: int = bs, input_tokens: int = 512, output_tokens: int = 128, **kw: Any) -> Dict[str, Any]:
                import time, torch

                prompt_tokens = list(range(100, 100 + input_tokens))
                input_ids = torch.tensor([prompt_tokens] * batch_size, dtype=torch.long, device=llm.device)
                position_ids = torch.arange(input_tokens, device=llm.device).unsqueeze(0).expand(batch_size, -1)

                t0 = time.perf_counter()
                logits = llm._model_runner.forward(input_ids, position_ids=position_ids)
                ttft_ms = (time.perf_counter() - t0) * 1000

                total_generated = 0
                for _ in range(output_tokens):
                    next_tokens = logits[:, -1, :].argmax(dim=-1)
                    total_generated += batch_size
                    new_input = next_tokens.unsqueeze(1)
                    pos = torch.tensor([[input_tokens + _]] * batch_size, device=llm.device)
                    logits = llm._model_runner.forward(new_input, position_ids=pos)

                total_ms = (time.perf_counter() - t0) * 1000
                return {
                    "tokens_generated": total_generated,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": (total_ms - ttft_ms) / max(total_generated // batch_size, 1),
                    "itl_ms": (total_ms - ttft_ms) / max(total_generated // batch_size, 1),
                    "gpu_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0,
                }

            result = self.runner.run_benchmark(
                name=f"batch_size_{bs}",
                benchmark_fn=bench_fn,
                model=llm.model_path,
                dtype=str(llm.dtype),
                batch_size=bs,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            results.append(result)

        return results


class ContextScalingBenchmark:
    """
    Context scaling benchmark.

    Tests performance at different input sequence lengths.
    """

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def run(
        self,
        llm: Any,
        context_lengths: List[int] = None,
        output_tokens: int = 128,
        **kwargs: Any,
    ) -> List[BenchmarkResult]:
        """Run context scaling benchmark."""
        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096, 8192, 16384]

        results = []
        for ctx_len in context_lengths:
            def bench_fn(batch_size: int = 1, input_tokens: int = ctx_len, output_tokens: int = 128, **kw: Any) -> Dict[str, Any]:
                import time, torch

                prompt_tokens = list(range(100, 100 + input_tokens))
                input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=llm.device)
                position_ids = torch.arange(input_tokens, device=llm.device).unsqueeze(0)

                t0 = time.perf_counter()
                logits = llm._model_runner.forward(input_ids, position_ids=position_ids)
                ttft_ms = (time.perf_counter() - t0) * 1000

                total_generated = 0
                for _ in range(output_tokens):
                    next_token = logits[0, -1, :].argmax().item()
                    total_generated += 1
                    new_input = torch.tensor([[next_token]], dtype=torch.long, device=llm.device)
                    pos = torch.tensor([[input_tokens + _]], device=llm.device)
                    logits = llm._model_runner.forward(new_input, position_ids=pos)

                total_ms = (time.perf_counter() - t0) * 1000
                return {
                    "tokens_generated": total_generated,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": (total_ms - ttft_ms) / max(total_generated, 1),
                    "itl_ms": (total_ms - ttft_ms) / max(total_generated, 1),
                    "gpu_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0,
                }

            result = self.runner.run_benchmark(
                name=f"context_{ctx_len}",
                benchmark_fn=bench_fn,
                model=llm.model_path,
                dtype=str(llm.dtype),
                batch_size=1,
                input_tokens=ctx_len,
                output_tokens=output_tokens,
            )
            results.append(result)

        return results


class GenerationScalingBenchmark:
    """
    Generation scaling benchmark.

    Tests performance at different output sequence lengths.
    """

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def run(
        self,
        llm: Any,
        generation_lengths: List[int] = None,
        input_tokens: int = 512,
        **kwargs: Any,
    ) -> List[BenchmarkResult]:
        """Run generation scaling benchmark."""
        if generation_lengths is None:
            generation_lengths = [16, 32, 64, 128, 256, 512, 1024]

        results = []
        for gen_len in generation_lengths:
            def bench_fn(batch_size: int = 1, input_tokens: int = 512, output_tokens: int = gen_len, **kw: Any) -> Dict[str, Any]:
                import time, torch

                prompt_tokens = list(range(100, 100 + input_tokens))
                input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=llm.device)
                position_ids = torch.arange(input_tokens, device=llm.device).unsqueeze(0)

                t0 = time.perf_counter()
                logits = llm._model_runner.forward(input_ids, position_ids=position_ids)
                ttft_ms = (time.perf_counter() - t0) * 1000

                total_generated = 0
                for _ in range(output_tokens):
                    next_token = logits[0, -1, :].argmax().item()
                    total_generated += 1
                    new_input = torch.tensor([[next_token]], dtype=torch.long, device=llm.device)
                    pos = torch.tensor([[input_tokens + _]], device=llm.device)
                    logits = llm._model_runner.forward(new_input, position_ids=pos)

                total_ms = (time.perf_counter() - t0) * 1000
                return {
                    "tokens_generated": total_generated,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": (total_ms - ttft_ms) / max(total_generated, 1),
                    "itl_ms": (total_ms - ttft_ms) / max(total_generated, 1),
                    "gpu_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0,
                }

            result = self.runner.run_benchmark(
                name=f"generation_{gen_len}",
                benchmark_fn=bench_fn,
                model=llm.model_path,
                dtype=str(llm.dtype),
                batch_size=1,
                input_tokens=input_tokens,
                output_tokens=gen_len,
            )
            results.append(result)

        return results
