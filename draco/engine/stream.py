"""
Streaming Generation

Token-by-token streaming output from the LLM engine.
Supports both synchronous iteration and async generators.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


@dataclass
class StreamDelta:
    """A single token delta from streaming generation."""
    token_id: int
    text: str
    index: int
    logprob: Optional[float] = None


@dataclass
class StreamOutput:
    """Accumulated streaming output for a single request."""
    request_id: int
    prompt: str
    deltas: List[StreamDelta] = field(default_factory=list)
    finished: bool = False

    @property
    def text(self) -> str:
        """Accumulated text so far."""
        return "".join(d.text for d in self.deltas)

    @property
    def token_ids(self) -> List[int]:
        """All generated token IDs so far."""
        return [d.token_id for d in self.deltas]

    @property
    def num_tokens(self) -> int:
        return len(self.deltas)


class StreamingGenerator:
    """Token-by-token streaming generator.

    Wraps the LLM engine to yield tokens as they are generated,
    rather than waiting for the full generation to complete.

    Usage:
        gen = StreamingGenerator(llm)
        for delta in gen.stream("Hello, how are you?", max_tokens=64):
            print(delta.text, end="", flush=True)
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any = None,
    ):
        self.llm = llm
        self._tokenizer = tokenizer or getattr(llm, "_tokenizer", None)

    def stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Generator[StreamOutput, None, None]:
        """Stream tokens one at a time.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop: Stop strings
            stop_token_ids: Stop token IDs
            seed: Random seed

        Yields:
            StreamOutput with accumulated text and new delta
        """
        import torch

        # Encode prompt (chat-template aware, same as LLM.generate)
        if hasattr(self.llm, "_encode_prompt_for_generation"):
            prompt_token_ids = self.llm._encode_prompt_for_generation(prompt)
        elif hasattr(self.llm, "encode"):
            prompt_token_ids = self.llm.encode(prompt)
        elif self._tokenizer is not None:
            prompt_token_ids = self._tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt.encode("utf-8"))

        # Create sampling params
        from draco.engine.sampling import SamplingParams
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=1,
            stop=stop,
            stop_token_ids=stop_token_ids,
            seed=seed,
        )

        request_id = id(prompt)  # Simple request ID
        output = StreamOutput(request_id=request_id, prompt=prompt)

        # State for autoregressive generation
        all_token_ids = list(prompt_token_ids)
        past_length = len(prompt_token_ids)

        # Get model runner
        runner = getattr(self.llm, "_model_runner", None)
        if runner is None:
            return

        # Prefill: try requesting a cache; falls back to full-refeed below
        # for any architecture adapter that doesn't return one yet.
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.llm.device)
        with torch.no_grad():
            out = runner.forward(input_ids, use_cache=True)
        using_cache = isinstance(out, tuple) and out[1] is not None and out[1][0] is not None
        if using_cache:
            logits, past_key_values = out
        else:
            position_ids = torch.arange(past_length, device=self.llm.device).unsqueeze(0)
            with torch.no_grad():
                logits = runner.forward(input_ids, position_ids=position_ids)

        # Autoregressive decode. When the loaded architecture's adapter
        # supports a real KV cache (see draco/models/<arch>/adapter.py),
        # each step below feeds only the newest token — O(n) instead of
        # O(n^2). Otherwise it falls back to re-feeding the FULL sequence
        # so far on every step (O(n^2), but unambiguously correct).
        stop_strings = stop or []
        for step in range(max_tokens):
            # Sample next token (with repetition/presence/frequency penalties
            # applied to the tokens generated so far)
            next_logits = logits[0, -1, :]
            generated_ids = all_token_ids[past_length:]
            next_token = self._sample_token(next_logits, params, generated_ids)

            # Decode token to text
            token_text = self._decode_token(next_token)
            all_token_ids.append(next_token)

            # Create delta
            delta = StreamDelta(
                token_id=next_token,
                text=token_text,
                index=step,
            )
            output.deltas.append(delta)

            # Check stop conditions
            if stop_token_ids and next_token in stop_token_ids:
                output.finished = True
                break

            # Check string stop conditions
            if stop_strings:
                accumulated = output.text
                for s in stop_strings:
                    if s in accumulated:
                        output.finished = True
                        break
                if output.finished:
                    break

            # Check EOS (tokenizer eos + generation_config ids)
            eos_ids = []
            eos_fn = getattr(self.llm, "_eos_token_ids", None)
            if callable(eos_fn):
                try:
                    eos_ids = eos_fn()
                except Exception:
                    eos_ids = []
            if not eos_ids and self._tokenizer is not None:
                eos_id = getattr(self._tokenizer, "eos_token_id", None)
                if eos_id is not None:
                    eos_ids = [eos_id]
            if next_token in eos_ids:
                output.finished = True
                break

            # Next forward pass
            if using_cache:
                step_input = torch.tensor([[next_token]], dtype=torch.long, device=self.llm.device)
                with torch.no_grad():
                    logits, past_key_values = runner.forward(
                        step_input, past_key_values=past_key_values, use_cache=True
                    )
            else:
                # Re-feed the full sequence so far (no cache available).
                new_input = torch.tensor(
                    [all_token_ids], dtype=torch.long, device=self.llm.device
                )
                position_ids = torch.arange(
                    len(all_token_ids), device=self.llm.device
                ).unsqueeze(0)
                with torch.no_grad():
                    logits = runner.forward(new_input, position_ids=position_ids)

            yield output

        # Mark as finished if not already
        output.finished = True
        yield output

    def stream_batch(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Generator[Dict[int, StreamOutput], None, None]:
        """Stream tokens for multiple prompts concurrently.

        Yields a dict mapping request_id to StreamOutput for each step.
        """
        generators = {}
        outputs = {}

        for prompt in prompts:
            req_id = id(prompt)
            gen = self.stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                stop_token_ids=stop_token_ids,
            )
            generators[req_id] = gen
            outputs[req_id] = StreamOutput(request_id=req_id, prompt=prompt)

        active = set(generators.keys())
        while active:
            step_results = {}
            for req_id in list(active):
                try:
                    result = next(generators[req_id])
                    outputs[req_id] = result
                    step_results[req_id] = result
                    if result.finished:
                        active.discard(req_id)
                except StopIteration:
                    active.discard(req_id)

            if step_results:
                yield step_results

    def _sample_token(
        self,
        logits: torch.Tensor,
        params: Any,
        generated_ids: Optional[List[int]] = None,
    ) -> int:
        """Sample a single token from logits."""
        import torch

        if params.temperature <= 0 or params.temperature < 1e-7:
            return logits.argmax().item()

        logits = logits / max(params.temperature, 1e-7)

        # Repetition / presence / frequency penalties over generated tokens
        generated_ids = generated_ids or []
        if generated_ids:
            penalty = getattr(params, "repetition_penalty", 1.0)
            presence = getattr(params, "presence_penalty", 0.0)
            frequency = getattr(params, "frequency_penalty", 0.0)
            if penalty != 1.0:
                for token_id in set(generated_ids):
                    if logits[token_id] > 0:
                        logits[token_id] = logits[token_id] / penalty
                    else:
                        logits[token_id] = logits[token_id] * penalty
            if presence != 0.0 or frequency != 0.0:
                from collections import Counter

                counts = Counter(generated_ids)
                for token_id, count in counts.items():
                    logits[token_id] -= presence + frequency * count

        if params.top_k > 0:
            top_k_vals, _ = torch.topk(logits, params.top_k)
            min_val = top_k_vals[-1]
            logits[logits < min_val] = float("-inf")

        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > params.top_p
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        if self._tokenizer is not None:
            return self._tokenizer.decode([token_id], skip_special_tokens=True)
        # Fallback
        if token_id < 128:
            return chr(token_id)
        return ""
