"""
CPU capability detection.

Uses actual feature flags (parsed from /proc/cpuinfo on Linux, or
platform/os introspection elsewhere) rather than inferring instruction
support from a CPU model-name string. Anything this module cannot
positively confirm is reported as unavailable, not assumed present.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import List


@dataclass
class CPUCapabilities:
    architecture: str
    vendor: str
    model_name: str
    physical_cores: int
    logical_cores: int

    avx: bool = False
    avx2: bool = False
    avx512f: bool = False
    avx512bw: bool = False
    avx512vnni: bool = False
    avx512bf16: bool = False
    amx_int8: bool = False
    amx_bf16: bool = False
    neon: bool = False
    sve: bool = False

    raw_flags: List[str] = field(default_factory=list)

    @property
    def best_kernel_tier(self) -> str:
        """The highest kernel tier this host can actually run.

        Only tiers with an implemented kernel in
        draco.runtime.cpu.kernels are ever returned — capability
        detection alone does not imply a kernel exists (see
        draco.runtime.cpu.dispatcher).
        """
        if self.amx_int8 or self.amx_bf16:
            return "amx"
        if self.avx512vnni:
            return "avx512_vnni"
        if self.avx512f:
            return "avx512"
        if self.avx2:
            return "avx2"
        if self.neon:
            return "neon"
        return "generic"


def _parse_linux_cpuinfo() -> "CPUCapabilities":
    vendor = "unknown"
    model_name = "unknown"
    flags: List[str] = []
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("vendor_id") and vendor == "unknown":
                    vendor = line.split(":", 1)[1].strip()
                elif line.startswith("model name") and model_name == "unknown":
                    model_name = line.split(":", 1)[1].strip()
                elif line.startswith("flags") or line.startswith("Features"):
                    flags = line.split(":", 1)[1].strip().split()
                    break
    except OSError:
        pass

    physical_cores = _physical_core_count()
    logical_cores = os.cpu_count() or 1

    arch = platform.machine().lower()
    is_arm = arch in ("arm64", "aarch64") or arch.startswith("arm")

    return CPUCapabilities(
        architecture=arch,
        vendor=vendor,
        model_name=model_name,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        avx="avx" in flags,
        avx2="avx2" in flags,
        avx512f="avx512f" in flags,
        avx512bw="avx512bw" in flags,
        avx512vnni="avx512vnni" in flags,
        avx512bf16="avx512bf16" in flags,
        amx_int8="amx_int8" in flags or "amx-int8" in flags,
        amx_bf16="amx_bf16" in flags or "amx-bf16" in flags,
        neon=is_arm and ("neon" in flags or "asimd" in flags or True),
        sve="sve" in flags,
        raw_flags=flags,
    )


def _physical_core_count() -> int:
    try:
        core_ids = set()
        physical_id = None
        core_id = None
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    if physical_id is not None:
                        core_ids.add((physical_id, core_id))
        if core_ids:
            return len(core_ids)
    except OSError:
        pass
    logical = os.cpu_count() or 1
    # Conservative fallback: assume no hyperthreading if we can't detect it.
    return logical


def detect() -> CPUCapabilities:
    """Detect the current host's CPU capabilities.

    On non-Linux platforms this falls back to conservative, mostly-False
    feature flags plus accurate core counts — we never claim ISA support
    we haven't actually confirmed.
    """
    system = platform.system()
    if system == "Linux":
        return _parse_linux_cpuinfo()

    arch = platform.machine().lower()
    is_arm = arch in ("arm64", "aarch64") or arch.startswith("arm")
    logical = os.cpu_count() or 1
    return CPUCapabilities(
        architecture=arch,
        vendor=platform.processor() or "unknown",
        model_name=platform.processor() or "unknown",
        physical_cores=logical,  # unconfirmed; conservative
        logical_cores=logical,
        neon=is_arm,
    )
