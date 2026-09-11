"""
Async LLM Engine

Provides async/await wrappers for the Draco LLM engine.
Enables non-blocking generation in async applications (FastAPI, aiohttp, etc.).

Usage:
    async_llm = AsyncLLM(model="meta-llama/Llama-3-8B")
    await async_llm.initialize()

    output = await async_llm.generate("Hello", max_tokens=64)

    async for delta in async_llm.stream("Tell me a joke", max_tokens=128):
        print(delta.text, end="")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

logger = logging.getLogger("draco.async_engine")


class AsyncLLM:
    """
    Async wrapper around the Draco LLM engine.

    Offloads blocking generation calls to a thread pool executor
    to keep the event loop responsive.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        dtype: Optional[str] = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        trust_remote_code: bool = False,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        self._model_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "quantization": quantization,
            "trust_remote_code": trust_remote_code,
            "seed": seed,
            **kwargs,
        }
        self._llm = None
        self._executor = None

    async def initialize(self) -> None:
        """Initialize the LLM engine asynchronously."""
        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(None, self._create_llm)

    def _create_llm(self):
        """Create LLM instance (blocking)."""
        from draco import LLM
        return LLM(**self._model_kwargs)

    async def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Generate text asynchronously.

        Args:
            prompts: Single prompt or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            stop: Stop strings
            seed: Random seed

        Returns:
            List of RequestOutput objects
        """
        from draco.engine.sampling import SamplingParams

        if self._llm is None:
            raise RuntimeError("AsyncLLM not initialized. Call await initialize() first.")

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            **kwargs,
        )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self._llm.generate(prompts, params)
        )

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Stream tokens asynchronously.

        Yields StreamOutput objects as tokens are generated.
        """
        from draco.engine.stream import StreamingGenerator

        if self._llm is None:
            raise RuntimeError("AsyncLLM not initialized. Call await initialize() first.")

        gen = StreamingGenerator(self._llm)

        loop = asyncio.get_event_loop()

        # Run the synchronous stream in a thread
        def _sync_stream():
            return list(gen.stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                seed=seed,
            ))

        # Pre-fetch all results in a thread (for simplicity)
        # In a production implementation, you'd use an async iterator pattern
        results = await loop.run_in_executor(self._executor, _sync_stream)

        for output in results:
            yield output

    async def encode(self, text: str) -> List[int]:
        """Encode text to token IDs asynchronously."""
        if self._llm is None:
            raise RuntimeError("AsyncLLM not initialized.")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._llm.encode, text)

    async def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text asynchronously."""
        if self._llm is None:
            raise RuntimeError("AsyncLLM not initialized.")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._llm.decode, token_ids)

    async def metrics(self) -> Any:
        """Get current metrics asynchronously."""
        if self._llm is None:
            raise RuntimeError("AsyncLLM not initialized.")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._llm.metrics)

    async def model_info(self) -> Dict[str, Any]:
        """Get model info asynchronously."""
        if self._llm is None:
            raise RuntimeError("AsyncLLM not initialized.")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._llm.get_model_info)

    @property
    def is_initialized(self) -> bool:
        return self._llm is not None

    def __repr__(self) -> str:
        model = self._model_kwargs.get("model", "?")
        return (
            f"AsyncLLM("
            f"model={model!r}, "
            f"initialized={self.is_initialized})"
        )
