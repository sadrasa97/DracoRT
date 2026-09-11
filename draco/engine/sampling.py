"""
Sampling Parameters

Controls text generation behavior: temperature, top-k, top-p, repetition penalty, etc.
Compatible with vLLM-style API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class SamplingParams:
    """
    Parameters for text generation / sampling.

    Example::

        params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_tokens=256,
        )
    """

    # Nucleus / top-k sampling
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    min_p: float = 0.0

    # Generation limits
    max_tokens: int = 16
    min_tokens: int = 0

    # Stop conditions
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None

    # Repetition / frequency penalties
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Log probabilities
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None

    # Sampling seed
    seed: Optional[int] = None

    # Special flags
    best_of: int = 1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    n: int = 1  # number of completions

    # Ignore EOS
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.repetition_penalty < 0.0:
            raise ValueError(
                f"repetition_penalty must be >= 0, got {self.repetition_penalty}"
            )
        # Normalize stop to list
        if isinstance(self.stop, str):
            self.stop = [self.stop]

    @property
    def effective_temperature(self) -> float:
        """Temperature used for sampling (0 means greedy)."""
        return max(self.temperature, 1e-7)

    def __repr__(self) -> str:
        parts = [f"temperature={self.temperature}", f"top_p={self.top_p}"]
        if self.top_k > 0:
            parts.append(f"top_k={self.top_k}")
        parts.append(f"max_tokens={self.max_tokens}")
        return f"SamplingParams({', '.join(parts)})"
