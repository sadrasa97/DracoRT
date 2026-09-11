"""
CPU memory auto-detection and utilization budgeting.

Mirrors the role of ``gpu_memory_utilization`` in draco.config /
draco.engine.llm.LLM (default 0.90 of device memory), but for system RAM:
detects total/available memory from /proc/meminfo (Linux) and derives a
usable budget as ``available * cpu_memory_utilization``, leaving headroom
for the OS and other processes rather than claiming everything free.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CPU_MEMORY_UTILIZATION = 0.85


@dataclass
class HostMemoryInfo:
    total_bytes: int
    available_bytes: int  # what's actually usable without swapping, per the OS


def detect_host_memory() -> HostMemoryInfo:
    total = available = None
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                if total is not None and available is not None:
                    break
    except OSError:
        pass

    if total is None:
        # Unknown host: report conservatively rather than guessing a large
        # number we can't back up.
        total = available = 2 * 1024**3
    if available is None:
        available = total

    return HostMemoryInfo(total_bytes=total, available_bytes=available)


def auto_memory_budget(
    utilization: float = DEFAULT_CPU_MEMORY_UTILIZATION, info: "HostMemoryInfo | None" = None
) -> int:
    if not (0.0 < utilization <= 1.0):
        raise ValueError(f"cpu_memory_utilization must be in (0, 1], got {utilization}.")
    host = info or detect_host_memory()
    return int(host.available_bytes * utilization)
