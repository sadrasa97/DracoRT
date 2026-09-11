"""Abstract CPU GEMM kernel interface."""

from __future__ import annotations

import abc

import numpy as np


class CPUGemmKernel(abc.ABC):
    """A CPU kernel that computes ``activation @ weight.T`` for one of the
    .draco CPU quantization formats.
    """

    #: kernel tier name, must match a value CPUCapabilities.best_kernel_tier
    #: can return (e.g. "generic", "avx2", "avx512").
    tier: str = "generic"

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def supports(self, quantization: str) -> bool: ...

    @abc.abstractmethod
    def matmul(self, activation: np.ndarray, tensor_info, weight_bytes: bytes) -> np.ndarray:
        """Compute activation @ weight.T for a tensor described by
        tensor_info (a DracoTensorInfo) with raw payload weight_bytes.
        """
        ...
