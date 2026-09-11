"""
Quantization Backend Base Class

Abstract interface for all quantization backends.
Each backend implements dequantization, weight loading, and kernel dispatch.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class QuantizationBackend(abc.ABC):
    """
    Abstract base class for quantization backends.

    Each quantization method (GPTQ, AWQ, FP8, etc.) implements this interface.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for this quantization method."""
        ...

    @property
    @abc.abstractmethod
    def supported_bits(self) -> List[int]:
        """List of bit widths supported by this backend."""
        ...

    @property
    @abc.abstractmethod
    def supported_schemes(self) -> List[str]:
        """List of quantization schemes (e.g., 'weight_only', 'w8a8')."""
        ...

    @abc.abstractmethod
    def load_quantized_weights(
        self,
        model_path: str,
        target_model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """
        Load quantized weights into a model.

        Args:
            model_path: Path to checkpoint/model directory
            target_model: The model to load weights into
            config: Quantization configuration
            device: Target device

        Returns:
            Model with quantized weights loaded.
        """
        ...

    @abc.abstractmethod
    def dequantize(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        """
        Dequantize weights to float format.

        Args:
            weights: Quantized weight tensor
            scales: Scale factors
            zeros: Zero points (optional)
            group_size: Group size for grouped quantization

        Returns:
            Dequantized weight tensor.
        """
        ...

    def quantized_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform quantized linear operation: input @ weight.T + bias.
        Override in backends with optimized kernels.
        """
        dequant_weight = self.dequantize(weight, weight, zeros)
        return torch.nn.functional.linear(input, dequant_weight, bias)

    def supports_gpu_kernel(self, device: torch.device) -> bool:
        """Whether this backend has GPU-optimized kernels available."""
        return device.type == "cuda"

    def get_kernel_info(self) -> Dict[str, Any]:
        """Return information about available kernels."""
        return {
            "backend": self.name,
            "gpu_kernel": False,
            "optimized": False,
        }

    def estimate_memory(
        self,
        num_params: int,
        bits: int,
        group_size: int = 128,
    ) -> int:
        """Estimate memory usage in bytes for quantized model."""
        # Each quantized weight takes bits/8 bytes
        quantized_bytes = num_params * bits // 8
        # Scale + zero point overhead
        num_groups = num_params // group_size
        scale_bytes = num_groups * 2  # float16 scales
        zero_bytes = num_groups * 2 if group_size > 0 else 0
        return quantized_bytes + scale_bytes + zero_bytes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"bits={self.supported_bits})"
        )
