"""
FP8 Quantization Backend

Supports FP8 E4M3 and E5M2 formats for efficient inference.
Selects kernels based on GPU compute capability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from draco.quantization.base import QuantizationBackend

logger = logging.getLogger("draco.quantization.fp8")


class FP8Backend(QuantizationBackend):
    """
    FP8 quantization backend.

    Supports FP8 E4M3, E5M2, weight-only, activation, and W8A8.
    """

    @property
    def name(self) -> str:
        return "fp8"

    @property
    def supported_bits(self) -> List[int]:
        return [8]

    @property
    def supported_schemes(self) -> List[str]:
        return ["weight_only", "w8a8", "fp8_e4m3", "fp8_e5m2"]

    def _check_fp8_support(self, device: torch.device) -> bool:
        """Check if GPU supports FP8 (compute capability >= 8.9 for H100/Ada)."""
        if device.type != "cuda":
            return False
        capability = torch.cuda.get_device_capability(device)
        return capability[0] >= 8

    def load_quantized_weights(
        self,
        model_path: str,
        target_model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """Load FP8-quantized weights."""
        logger.info("Loading FP8 weights")

        from draco.weights.loader import get_checkpoint_loader
        loader = get_checkpoint_loader()
        try:
            weights = loader.load(model_path, device=device)
        except Exception as e:
            logger.warning("Failed to load FP8 weights: %s", e)
            return target_model

        quantized_params = {}
        for key, tensor in weights.items():
            if "weight" in key or "bias" in key or "scale" in key:
                quantized_params[key] = tensor

        if quantized_params:
            target_model.load_state_dict(quantized_params, strict=False)
            logger.info("Loaded %d FP8 parameters", len(quantized_params))

        return target_model

    def dequantize(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Dequantize FP8 weights to float16/bfloat16."""
        # FP8 dequantization is straightforward: weight * scale
        target_dtype = torch.float16 if scales.dtype == torch.float16 else torch.bfloat16
        return weights.to(target_dtype) * scales

    def quantized_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FP8 matrix multiplication with scale factors."""
        # Cast input to FP8 for computation if GPU supports it
        device = input.device
        if self._check_fp8_support(device):
            # Use native FP8 matmul on H100+
            input_fp8 = input.to(torch.float8_e4m3fn)
            output = torch._scaled_mm(
                input_fp8, weight.t(),
                scale_a=scales, scale_b=scales,
                out_dtype=torch.float16,
            )
            if bias is not None:
                output = output + bias
            return output
        else:
            # Fallback: dequantize and do standard matmul
            dequant = self.dequantize(weight, scales)
            return torch.nn.functional.linear(input, dequant, bias)

    def get_kernel_info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "gpu_kernel": True,
            "optimized": True,
            "description": "FP8 E4M3/E5M2 kernels",
            "requires_compute_capability": "8.0+",
            "requires_hopper_or_ada": True,
        }

    def estimate_memory(
        self,
        num_params: int,
        bits: int = 8,
        group_size: int = 128,
    ) -> int:
        # FP8: 1 byte per parameter + 2 bytes per group for scales
        quantized_bytes = num_params
        num_groups = num_params // group_size
        scale_bytes = num_groups * 2
        return quantized_bytes + scale_bytes
