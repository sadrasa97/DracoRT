"""
Device selection (spec section 27-28).

device="auto" resolution order:
  1. Inspect .draco metadata (CPU-oriented quantization strongly favors CPU)
  2. Detect hardware (CPU always available; GPU only if torch+CUDA present)
  3. Pick GPU when available and the format is GPU-compatible, else CPU

This module never fabricates GPU kernel work: if CUDA is unavailable (as
in this sandbox), it always resolves to "cpu" and says so.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceDecision:
    device: str  # "cpu" | "cuda"
    reason: str


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(requested: str, draco_metadata: "dict | None" = None) -> DeviceDecision:
    if requested in ("cpu", "cuda"):
        if requested == "cuda" and not _cuda_available():
            raise RuntimeError(
                "device='cuda' was requested but no CUDA-capable torch install was detected."
            )
        return DeviceDecision(device=requested, reason="explicit request")

    if requested != "auto":
        raise ValueError(f"Unknown device '{requested}'. Expected 'cpu', 'cuda', or 'auto'.")

    cpu_hint = False
    if draco_metadata:
        cpu_meta = draco_metadata.get("cpu")
        if cpu_meta and cpu_meta.get("preferred_quantization"):
            cpu_hint = True

    if cpu_hint:
        return DeviceDecision(
            device="cpu", reason="model.draco carries CPU-oriented quantization metadata"
        )

    if _cuda_available():
        return DeviceDecision(device="cuda", reason="CUDA-capable GPU detected")

    return DeviceDecision(device="cpu", reason="no CUDA-capable GPU detected")
