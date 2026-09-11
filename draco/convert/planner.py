"""
Automatic CPU quantization + inference planner (spec section 18).

Chooses quantization/dtype/thread/KV-cache settings from actual available
RAM, detected CPU ISA, and the implemented kernel tiers — not just "use
the smallest possible format". Memory feasibility is a hard constraint;
among feasible options it prefers less aggressive quantization (better
accuracy) that still fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from draco.exceptions import InsufficientMemoryError
from draco.runtime.cpu.capabilities import CPUCapabilities
from draco.runtime.cpu.dispatcher import registered_tiers
from draco.runtime.cpu.memory import CPUMemoryPlanner
from draco.runtime.cpu.threading import plan_threads

# Ordered from least to most aggressive; planner walks from the end
# (most memory-hungry / most accurate) backwards until something fits.
_QUANT_CANDIDATES_BEST_FIRST = ["f32", "f16", "int8_sym", "int4_groupwise"]


@dataclass
class CPUInferencePlan:
    quantization: str
    dtype: str
    kernel_backend: str
    num_threads: int
    kv_cache_dtype: str
    max_num_seqs: int
    estimated_memory_bytes: int


class CPUQuantizationPlanner:
    def __init__(self) -> None:
        self._memory_planner = CPUMemoryPlanner()

    def plan(
        self,
        num_params: int,
        num_layers: int,
        hidden_size: int,
        num_key_value_heads: int,
        head_dim: int,
        cpu_capabilities: CPUCapabilities,
        available_memory_bytes: int,
        desired_context_length: int = 4096,
        desired_concurrency: int = 1,
        requested_quantization: Optional[str] = None,
    ) -> CPUInferencePlan:
        # Kernel backend: whatever is actually registered for this host's
        # detected tier (dispatcher already refuses to overclaim — see
        # draco.runtime.cpu.dispatcher.select_kernel).
        from draco.runtime.cpu.dispatcher import select_kernel

        selection = select_kernel(cpu_capabilities)

        candidates = (
            [requested_quantization]
            if requested_quantization
            else list(_QUANT_CANDIDATES_BEST_FIRST)
        )

        kv_dtype = "f16"
        budget_fraction = 0.8  # leave headroom for OS/other processes

        for quant in candidates:
            estimate = self._memory_planner.estimate_inference(
                num_params=num_params,
                quantization=quant,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_model_len=desired_context_length,
                max_num_seqs=desired_concurrency,
                kv_cache_dtype=kv_dtype,
            )
            if estimate.total_bytes <= available_memory_bytes * budget_fraction:
                threads = plan_threads(cpu_capabilities)
                return CPUInferencePlan(
                    quantization=quant,
                    dtype="f32" if quant in ("f32", "f16") else quant,
                    kernel_backend=selection.actual_tier,
                    num_threads=threads.num_threads,
                    kv_cache_dtype=kv_dtype,
                    max_num_seqs=desired_concurrency,
                    estimated_memory_bytes=estimate.total_bytes,
                )

        # Nothing fit, even the most aggressive quantization.
        smallest = self._memory_planner.estimate_inference(
            num_params=num_params,
            quantization=_QUANT_CANDIDATES_BEST_FIRST[-1],
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_model_len=desired_context_length,
            max_num_seqs=desired_concurrency,
            kv_cache_dtype=kv_dtype,
        )
        raise InsufficientMemoryError(
            f"No feasible plan: even int4_groupwise needs an estimated "
            f"{smallest.total_bytes / (1024**3):.2f} GB, but only "
            f"{available_memory_bytes * budget_fraction / (1024**3):.2f} GB is budgeted "
            f"(available={available_memory_bytes / (1024**3):.2f} GB). Reduce context "
            f"length or concurrency."
        )
