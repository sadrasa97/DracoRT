"""
CPU kernel dispatch.

Selects the best *implemented* kernel tier — never the best *detected*
ISA tier if no kernel exists for it. Capability detection and kernel
availability are deliberately kept separate: a host reporting AVX512
support with no AVX512 kernel registered still runs on 'generic', and
the runtime reports this explicitly via KernelSelection.requested_tier
vs .actual_tier instead of silently pretending the fast path was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from draco.exceptions import CPUKernelError
from draco.runtime.cpu.capabilities import CPUCapabilities, detect
from draco.runtime.cpu.kernels.base import CPUGemmKernel
from draco.runtime.cpu.kernels.generic import GenericGemmKernel

# Only tiers with an actual, tested kernel implementation are registered.
# Adding "avx2": Avx2GemmKernel() here without a real, hardware-tested
# implementation would violate the project's no-fabricated-support rule.
_REGISTERED_KERNELS: Dict[str, CPUGemmKernel] = {
    "generic": GenericGemmKernel(),
}

_TIER_PRIORITY = ["amx", "avx512_vnni", "avx512", "avx2", "neon", "generic"]


@dataclass
class KernelSelection:
    kernel: CPUGemmKernel
    requested_tier: str  # best tier the host's hardware could theoretically support
    actual_tier: str  # tier actually selected (only ever an implemented one)
    downgraded: bool


def select_kernel(capabilities: CPUCapabilities = None) -> KernelSelection:
    caps = capabilities or detect()
    requested = caps.best_kernel_tier

    for tier in _TIER_PRIORITY:
        if tier == requested or _TIER_PRIORITY.index(tier) >= _TIER_PRIORITY.index(requested):
            if tier in _REGISTERED_KERNELS:
                return KernelSelection(
                    kernel=_REGISTERED_KERNELS[tier],
                    requested_tier=requested,
                    actual_tier=tier,
                    downgraded=(tier != requested),
                )

    if "generic" not in _REGISTERED_KERNELS:
        raise CPUKernelError("No CPU kernel available, not even the generic portable tier.")
    return KernelSelection(
        kernel=_REGISTERED_KERNELS["generic"],
        requested_tier=requested,
        actual_tier="generic",
        downgraded=(requested != "generic"),
    )


def registered_tiers() -> list:
    return sorted(_REGISTERED_KERNELS.keys())
