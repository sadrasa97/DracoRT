"""
OOM-avoidance retry wrapper for GPU (or CPU) inference calls.

The actual OOM-*avoidance* mechanism is GPUMemoryPlanner.max_num_seqs_that_fit
(admission control: never schedule past the budget in the first place).
This module is the *recovery* mechanism for when an OOM happens anyway —
model size misestimated, fragmentation, a concurrent process — because
admission control alone can't guarantee it never happens. On a caught
out-of-memory error, it clears the CUDA cache (if torch is available),
backs off a memory-fraction-like keyword argument, and retries with a
smaller budget, up to a bounded number of attempts, instead of crashing
the whole request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_OOM_MARKERS = ("out of memory", "cuda oom", "cudnn_status_alloc_failed", "resourceexhausted")


def is_oom_error(exc: BaseException) -> bool:
    """Heuristic OOM detection that works whether or not torch is
    installed / a specific torch.cuda.OutOfMemoryError class exists in
    this version: checks the exception's type name and message text
    rather than importing a torch-specific exception class directly."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


@dataclass
class OOMRetryConfig:
    max_retries: int = 3
    backoff_factor: float = 0.7  # multiply the memory-fraction kwarg by this each retry
    min_value: float = 0.3  # never back off below this fraction
    kwarg_name: str = "gpu_memory_utilization"


@dataclass
class OOMRetryResult:
    result: Any
    attempts: int
    final_value: float
    oom_events: int


def run_with_oom_backoff(
    fn: Callable[..., T],
    initial_value: float,
    config: OOMRetryConfig = None,
    **fn_kwargs,
) -> OOMRetryResult:
    """Call ``fn(**{config.kwarg_name: current_value}, **fn_kwargs)``,
    backing off ``current_value`` and retrying on OOM-like errors.

    Also calls ``torch.cuda.empty_cache()`` between attempts when torch
    with CUDA is available, since a cleared allocator cache is often
    enough by itself to let a retry at the *same* budget succeed —  the
    backoff on top of that is for when it isn't.
    """
    cfg = config or OOMRetryConfig()
    current_value = initial_value
    oom_events = 0

    for attempt in range(1, cfg.max_retries + 2):  # +1 initial try, +max_retries retries
        try:
            result = fn(**{cfg.kwarg_name: current_value}, **fn_kwargs)
            return OOMRetryResult(
                result=result, attempts=attempt, final_value=current_value, oom_events=oom_events
            )
        except Exception as e:
            if not is_oom_error(e):
                raise
            oom_events += 1
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            if attempt > cfg.max_retries:
                raise

            current_value = max(cfg.min_value, current_value * cfg.backoff_factor)

    raise RuntimeError("unreachable")  # pragma: no cover
