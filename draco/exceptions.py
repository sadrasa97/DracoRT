"""Draco error hierarchy."""

from __future__ import annotations


class DracoError(Exception):
    """Base error for all Draco exceptions."""


class ModelNotFoundError(DracoError):
    """Raised when a model architecture is not in the registry."""

    def __init__(self, architecture: str) -> None:
        self.architecture = architecture
        super().__init__(
            f"No registered model adapter for architecture '{architecture}' in Draco's "
            f"native backend (MODEL_REGISTRY). Either register an adapter with "
            f"MODEL_REGISTRY.register(), or use LLM(..., backend='transformers') "
            f"(the default) to load this model via the installed transformers "
            f"library instead, which supports a much wider range of architectures "
            f"including text+vision \"ForConditionalGeneration\" models."
        )


class WeightLoadingError(DracoError):
    """Raised when weight loading fails."""


class QuantizationError(DracoError):
    """Raised when quantization operations fail."""


class GPUError(DracoError):
    """Raised when GPU operations fail."""


class CheckpointError(DracoError):
    """Raised when checkpoint loading/reading fails."""


class ConfigError(DracoError):
    """Raised when model configuration is invalid or incompatible."""


class KernelError(DracoError):
    """Raised when a CUDA kernel is unavailable or fails."""


class BenchmarkError(DracoError):
    """Raised when benchmarking fails."""


class MetricsError(DracoError):
    """Raised when metrics collection fails."""


class SchedulerError(DracoError):
    """Raised when scheduler operations fail."""


class KVCacheError(DracoError):
    """Raised when KV cache operations fail."""


class DracoFormatError(DracoError):
    """Raised when a .draco file is malformed or corrupt."""


class DracoFormatVersionError(DracoFormatError):
    """Raised when a .draco file's format version is unsupported."""

    def __init__(self, found_version: int, expected_version: int) -> None:
        self.found_version = found_version
        self.expected_version = expected_version
        super().__init__(
            f"Unsupported .draco format version {found_version} "
            f"(this runtime supports version {expected_version})."
        )


class DracoCorruptModelError(DracoFormatError):
    """Raised when a .draco file fails structural/bounds validation."""


class CPUError(DracoError):
    """Raised when native CPU runtime operations fail."""


class CPUKernelError(CPUError):
    """Raised when a CPU kernel is unavailable or fails."""


class CPUCapabilityError(CPUError):
    """Raised when required CPU instruction-set support is missing."""


class CPUQuantizationError(CPUError):
    """Raised when CPU-native quantization/dequantization fails."""


class ConversionError(DracoError):
    """Raised when HuggingFace/safetensors -> .draco conversion fails."""


class InsufficientMemoryError(DracoError):
    """Raised when an inference or conversion plan does not fit in available RAM."""
