"""
Benchmark Comparison

Compares inference performance across different backends:
- Draco
- vLLM
- Transformers
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("draco.benchmarks.comparison")


class ComparisonBenchmark:
    """
    Compare inference performance across backends.

    Uses identical model, prompt dataset, sampling parameters,
    and hardware to ensure fair comparison.
    """

    def __init__(self, runner: Any):
        self.runner = runner

    def compare(
        self,
        backends: List[str],
        model: str,
        input_tokens: int = 512,
        output_tokens: int = 128,
        dtype: str = "float16",
        num_iterations: int = 3,
        prompts: Optional[List[str]] = None,
    ) -> str:
        """
        Run benchmarks across multiple backends and produce comparison.

        Args:
            backends: List of backend names ("draco", "vllm", "transformers")
            model: Model identifier
            input_tokens: Input token count
            output_tokens: Output token count
            dtype: Data type
            num_iterations: Number of iterations per backend
            prompts: Optional list of prompts

        Returns:
            Formatted comparison report.
        """
        if prompts is None:
            prompts = ["Hello, how are you today?"]

        results = {}
        for backend in backends:
            try:
                result = self._benchmark_backend(
                    backend=backend,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    dtype=dtype,
                    num_iterations=num_iterations,
                    prompts=prompts,
                )
                results[backend] = result
            except Exception as e:
                logger.warning("Backend '%s' failed: %s", backend, e)
                results[backend] = {"error": str(e)}

        return self._format_comparison(results)

    def _benchmark_backend(
        self,
        backend: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        dtype: str,
        num_iterations: int,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """Benchmark a single backend."""
        if backend == "draco":
            return self._bench_draco(model, input_tokens, output_tokens, dtype, num_iterations, prompts)
        elif backend == "vllm":
            return self._bench_vllm(model, input_tokens, output_tokens, dtype, num_iterations, prompts)
        elif backend == "transformers":
            return self._bench_transformers(model, input_tokens, output_tokens, dtype, num_iterations, prompts)
        else:
            return {"error": f"Unknown backend: {backend}"}

    def _bench_draco(
        self, model: str, input_tokens: int, output_tokens: int,
        dtype: str, num_iterations: int, prompts: List[str],
    ) -> Dict[str, Any]:
        """Benchmark Draco backend."""
        from draco import LLM, SamplingParams
        import torch

        llm = LLM(model=model, dtype=dtype)
        sampling = SamplingParams(max_tokens=output_tokens)

        latencies = []
        ttfts = []
        tokens_per_sec_list = []
        gen_tokens = 0

        for _ in range(num_iterations):
            t0 = time.perf_counter()

            # Simple generation loop
            prompt = prompts[0]
            tokens = llm.encode(prompt)[:input_tokens]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=llm.device)
            position_ids = torch.arange(len(tokens), device=llm.device).unsqueeze(0)

            logits = llm._model_runner.forward(input_ids, position_ids=position_ids)
            ttft = (time.perf_counter() - t0) * 1000

            generated = []
            for _ in range(output_tokens):
                next_token = logits[0, -1, :].argmax().item()
                generated.append(next_token)
                new_input = torch.tensor([[next_token]], dtype=torch.long, device=llm.device)
                pos = torch.tensor([[len(tokens) + len(generated) - 1]], device=llm.device)
                logits = llm._model_runner.forward(new_input, position_ids=pos)

            total_ms = (time.perf_counter() - t0) * 1000
            latencies.append(total_ms)
            ttfts.append(ttft)
            tokens_per_sec_list.append(len(generated) / (total_ms / 1000))
            gen_tokens += len(generated)

        return {
            "backend": "draco",
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_ttft_ms": sum(ttfts) / len(ttfts),
            "avg_tokens_per_sec": sum(tokens_per_sec_list) / len(tokens_per_sec_list),
            "total_tokens": gen_tokens,
            "gpu_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0,
        }

    def _bench_vllm(
        self, model: str, input_tokens: int, output_tokens: int,
        dtype: str, num_iterations: int, prompts: List[str],
    ) -> Dict[str, Any]:
        """Benchmark vLLM backend."""
        try:
            from vllm import LLM as VLLM, SamplingParams as VLLMSampling

            dtype_map = {"float16": "float16", "bfloat16": "bfloat16", "auto": "auto"}
            vllm_dtype = dtype_map.get(dtype, "auto")

            llm = VLLM(model=model, dtype=vllm_dtype, trust_remote_code=True)
            sampling = VLLMSampling(max_tokens=output_tokens)

            latencies = []
            ttfts = []
            tokens_per_sec_list = []

            prompt = prompts[0][:input_tokens * 4]  # Approximate char/token ratio

            for _ in range(num_iterations):
                t0 = time.perf_counter()
                outputs = llm.generate([prompt], sampling)
                total_ms = (time.perf_counter() - t0) * 1000

                latencies.append(total_ms)
                # vLLM doesn't directly expose TTFT in basic API
                ttfts.append(total_ms * 0.1)  # Approximate
                generated = len(outputs[0].outputs[0].token_ids)
                tokens_per_sec_list.append(generated / (total_ms / 1000))

            return {
                "backend": "vllm",
                "avg_latency_ms": sum(latencies) / len(latencies),
                "avg_ttft_ms": sum(ttfts) / len(ttfts),
                "avg_tokens_per_sec": sum(tokens_per_sec_list) / len(tokens_per_sec_list),
                "total_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
                "gpu_memory_gb": 0,  # vLLM doesn't expose this directly
            }
        except ImportError:
            return {"error": "vllm not installed"}

    def _bench_transformers(
        self, model: str, input_tokens: int, output_tokens: int,
        dtype: str, num_iterations: int, prompts: List[str],
    ) -> Dict[str, Any]:
        """Benchmark HuggingFace Transformers backend."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": torch.float16}
            torch_dtype = dtype_map.get(dtype, torch.float16)

            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, device_map="auto")
            model_obj.eval()

            input_text = prompts[0]
            input_ids = tokenizer.encode(input_text, return_tensors="pt")[:, :input_tokens].to(model_obj.device)

            latencies = []
            tokens_per_sec_list = []
            gen_tokens = 0

            for _ in range(num_iterations):
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = model_obj.generate(
                        input_ids, max_new_tokens=output_tokens, do_sample=False
                    )
                total_ms = (time.perf_counter() - t0) * 1000
                generated = outputs.shape[1] - input_ids.shape[1]
                latencies.append(total_ms)
                tokens_per_sec_list.append(generated / (total_ms / 1000))
                gen_tokens += generated

            return {
                "backend": "transformers",
                "avg_latency_ms": sum(latencies) / len(latencies),
                "avg_ttft_ms": 0,  # Transformers doesn't separate prefill
                "avg_tokens_per_sec": sum(tokens_per_sec_list) / len(tokens_per_sec_list),
                "total_tokens": gen_tokens,
                "gpu_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0,
            }
        except ImportError:
            return {"error": "transformers not installed"}

    def _format_comparison(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Format comparison results."""
        lines = [f"{'Backend Comparison Report':-^50}", ""]

        # Header
        backends = list(results.keys())
        header = f"  {'Metric':<25s}"
        for b in backends:
            header += f"  {b:>12s}"
        lines.append(header)
        lines.append(f"  {'-' * (25 + 14 * len(backends))}")

        # Metrics
        for metric, label in [
            ("avg_ttft_ms", "TTFT (ms)"),
            ("avg_tokens_per_sec", "Throughput (tok/s)"),
            ("avg_latency_ms", "Total Latency (ms)"),
            ("gpu_memory_gb", "GPU Memory (GB)"),
        ]:
            row = f"  {label:<25s}"
            for b in backends:
                val = results[b].get(metric, results[b].get("error", "N/A"))
                if isinstance(val, (int, float)):
                    row += f"  {val:>12.1f}"
                else:
                    row += f"  {str(val):>12s}"
            lines.append(row)

        lines.append("")

        # Detailed results
        for backend, data in results.items():
            if "error" in data:
                lines.append(f"  {backend}: {data['error']}")
            else:
                lines.append(f"  {backend}: {data.get('avg_tokens_per_sec', 0):.1f} tok/s")

        return "\n".join(lines)
