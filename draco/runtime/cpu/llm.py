"""
High-level native-CPU entry point (spec section 45):

    from draco.runtime.cpu.llm import CPUNativeLLM
    llm = CPUNativeLLM("model.draco")
    ids = llm.generate([1, 2, 3], max_new_tokens=16)
    print(llm.runtime_info())

This wraps DracoReader + CPUExecutionBackend + the threading/memory
planners into the single object a caller actually wants. It does not
implement tokenization, sampling strategies beyond greedy, or batching
across multiple prompts — see draco.engine.llm.LLM for the full-featured
(GPU-oriented) engine this complements.
"""

from __future__ import annotations

from typing import List, Optional

from draco.format.reader import DracoReader
from draco.runtime.cpu.capabilities import detect
from draco.runtime.cpu.dispatcher import select_kernel
from draco.runtime.cpu.executor import CPUExecutionBackend
from draco.runtime.cpu.threading import plan_threads


class CPUNativeLLM:
    def __init__(
        self,
        model: str,
        num_threads: Optional[int] = None,
        kv_cache_dtype: str = "f32",
        max_num_seqs: int = 1,
        memory_utilization: Optional[float] = None,
    ) -> None:
        self.model_path = model
        from draco.format.gguf import GGUFReader, is_gguf_file

        if is_gguf_file(model):
            self.reader = GGUFReader(model)
        else:
            self.reader = DracoReader(model)
        self.capabilities = detect()
        self.thread_config = plan_threads(self.capabilities, num_threads=num_threads)
        self.kernel_selection = select_kernel(self.capabilities)
        self.executor = CPUExecutionBackend(
            self.reader,
            max_num_seqs=max_num_seqs,
            kv_cache_dtype=kv_cache_dtype,
            memory_utilization=memory_utilization,
        )

    def generate(
        self, prompt_ids: List[int], max_new_tokens: int = 32, speculative: bool = False
    ) -> List[int]:
        if speculative:
            from draco.runtime.cpu.speculative import AdaptiveSpeculativeDecoder

            decoder = AdaptiveSpeculativeDecoder(self.executor)
            return decoder.generate(prompt_ids, max_new_tokens)
        return self.executor.generate_greedy(prompt_ids, max_new_tokens)

    def generate_batch(
        self, prompts: List[List[int]], max_new_tokens: int = 32
    ) -> List[List[int]]:
        """Continuously-batched generation over multiple prompts at once,
        using the same ContinuousBatchScheduler as the GPU engine."""
        from draco.runtime.cpu.batch_engine import CPUBatchEngine

        engine = CPUBatchEngine(self.executor, max_num_seqs=len(prompts))
        for i, prompt in enumerate(prompts):
            engine.add_request(i, prompt, max_tokens=max_new_tokens)
        results = engine.run_to_completion()
        return [results[i] for i in range(len(prompts))]

    def runtime_info(self) -> dict:
        weight_bytes = sum(t.nbytes for t in self.reader.tensors())
        return {
            "device": "cpu",
            "cpu": {
                "vendor": self.capabilities.vendor,
                "model": self.capabilities.model_name,
                "avx2": self.capabilities.avx2,
                "avx512": self.capabilities.avx512f,
                "amx": self.capabilities.amx_int8 or self.capabilities.amx_bf16,
            },
            "format": f"Draco v{self.reader.header.version}" if hasattr(self.reader, "header") else "GGUF",
            "quantization": self.kernel_selection.actual_tier,
            "kernel": self.kernel_selection.actual_tier,
            "kernel_downgraded": self.kernel_selection.downgraded,
            "threads": self.thread_config.num_threads,
            "memory": {
                "weights_bytes": weight_bytes,
                "kv_budget_bytes": self.executor.kv_budget_bytes,
                "kv_free_blocks": self.executor.kv_cache.num_free_blocks,
            },
        }

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "CPUNativeLLM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
