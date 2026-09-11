"""
CPU thread-runtime configuration (spec section 15).

Computes sane worker-thread defaults from *physical* core count (not
logical/hyperthreaded count, which oversubscribes compute-bound GEMM
work), while always respecting explicit user overrides and the standard
BLAS/OpenMP environment variables when present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from draco.runtime.cpu.capabilities import CPUCapabilities


@dataclass
class ThreadConfig:
    num_threads: int
    interop_threads: int
    source: str  # "user_override" | "env_var" | "heuristic"


_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _env_override() -> Optional[int]:
    for var in _ENV_VARS:
        val = os.environ.get(var)
        if val:
            try:
                n = int(val)
                if n > 0:
                    return n
            except ValueError:
                continue
    return None


def plan_threads(
    capabilities: CPUCapabilities,
    num_threads: Optional[int] = None,
    interop_threads: Optional[int] = None,
) -> ThreadConfig:
    """Pick intra-op / inter-op thread counts.

    Priority: explicit ``num_threads`` argument > OMP/MKL/OPENBLAS env vars
    > heuristic default (physical cores, capped so a handful of concurrent
    requests don't oversubscribe the machine).
    """
    if num_threads is not None:
        n, source = num_threads, "user_override"
    else:
        env_n = _env_override()
        if env_n is not None:
            n, source = env_n, "env_var"
        else:
            # Physical cores, never logical/hyperthreaded count: GEMM is
            # compute-bound, so hyperthreads mostly add contention, not
            # throughput. Cap at 16 by default — matmul kernels tend to
            # stop scaling well past that without NUMA-aware tiling, which
            # is not implemented in the generic kernel tier.
            n = max(1, min(capabilities.physical_cores, 16))
            source = "heuristic"

    interop = interop_threads if interop_threads is not None else 1
    return ThreadConfig(num_threads=n, interop_threads=interop, source=source)
