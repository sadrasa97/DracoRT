"""
bitsandbytes Quantization Backend

Integration layer for bitsandbytes library providing 8-bit and 4-bit
quantization (NF4, FP4).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from draco.quantization.base import QuantizationBackend

logger = logging.getLogger("draco.quantization.bitsandbytes")


class BitsAndBytesBackend(QuantizationBackend):
    """
    bitsandbytes quantization backend.

    Provides integration with the bitsandbytes library for:
    - 8-bit quantization
    - 4-bit NF4 quantization
    - 4-bit FP4 quantization
    """

    @property
    def name(self) -> str:
        return "bitsandbytes"

    @property
    def supported_bits(self) -> List[int]:
        return [4, 8]

    @property
    def supported_schemes(self) -> List[str]:
        return ["weight_only", "w4a16", "w8a8", "nf4", "fp4"]

    def _check_available(self) -> bool:
        """Check if bitsandbytes is installed."""
        try:
            import bitsandbytes  # noqa: F401
            return True
        except ImportError:
            return False

    def load_quantized_weights(
        self,
        model_path: str,
        target_model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """Load bitsandbytes quantized weights."""
        if not self._check_available():
            logger.warning("bitsandbytes not installed. Falling back to unquantized.")
            return target_model

        quant_type = config.get("quant_type", "nf4")
        logger.info("Loading bitsandbytes %s weights", quant_type)

        from draco.weights.loader import get_checkpoint_loader
        loader = get_checkpoint_loader()
        try:
            weights = loader.load(model_path, device=device)
        except Exception as e:
            logger.warning("Failed to load bitsandbytes weights: %s", e)
            return target_model

        target_model.load_state_dict(weights, strict=False)
        return target_model

    def dequantize(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Dequantize bitsandbytes weights."""
        return weights.float() * scales

    def quantized_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use bitsandbytes optimized matmul if available."""
        if self._check_available():
            import bitsandbytes as bnb
            # Use bnb matmul for quantized inference
            return bnb.matmul_4bit(input, weight.t(), bias=bias)
        else:
            dequant = self.dequantize(weight, scales)
            return torch.nn.functional.linear(input, dequant, bias)

    def get_kernel_info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "gpu_kernel": True,
            "optimized": True,
            "description": "bitsandbytes NF4/FP4/INT8 kernels",
            "available": self._check_available(),
        }

    def estimate_memory(
        self,
        num_params: int,
        bits: int = 4,
        group_size: int = 128,
    ) -> int:
        quantized_bytes = num_params * bits // 8
        # bitsandbytes uses 32-bit quantization state per 256 values
        num_blocks = num_params // 256
        state_bytes = num_blocks * 4 * 2  # absmax + codebook
        return quantized_bytes + state_bytes
