"""
AWQ (Activation-aware Weight Quantization) Backend

Supports AWQ checkpoint loading, INT4 weight-only quantization,
and efficient W4A16 inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from draco.quantization.base import QuantizationBackend

logger = logging.getLogger("draco.quantization.awq")


class AWQBackend(QuantizationBackend):
    """
    AWQ quantization backend.

    Activation-aware weight quantization for efficient INT4 inference.
    """

    @property
    def name(self) -> str:
        return "awq"

    @property
    def supported_bits(self) -> List[int]:
        return [4]

    @property
    def supported_schemes(self) -> List[str]:
        return ["weight_only", "w4a16"]

    def load_quantized_weights(
        self,
        model_path: str,
        target_model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """Load AWQ-quantized weights from a checkpoint."""
        group_size = config.get("group_size", 128)

        logger.info("Loading AWQ weights: group_size=%d", group_size)

        from draco.weights.loader import get_checkpoint_loader
        loader = get_checkpoint_loader()
        try:
            weights = loader.load(model_path, device=device)
        except Exception as e:
            logger.warning("Failed to load AWQ weights: %s", e)
            return target_model

        # AWQ uses qweight, scales, qzeros
        quantized_params = {}
        for key, tensor in weights.items():
            if any(s in key for s in ("qweight", "scales", "qzeros", "bias")):
                quantized_params[key] = tensor

        if quantized_params:
            target_model.load_state_dict(quantized_params, strict=False)
            logger.info("Loaded %d AWQ parameters", len(quantized_params))

        return target_model

    def dequantize(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        """Dequantize AWQ INT4 weights to float."""
        weights = weights.to(scales.dtype)

        if zeros is not None:
            zeros = zeros.to(scales.dtype)
            # AWQ dequant: (weight - zero) * scale
            if scales.dim() > 1 and scales.shape[-1] != weights.shape[-1]:
                # Expand grouped scales
                repeats = weights.shape[-1] // scales.shape[-1]
                scales = scales.repeat_interleave(repeats, dim=-1)
                zeros = zeros.repeat_interleave(repeats, dim=-1)
            return (weights - zeros) * scales
        else:
            return weights * scales

    def get_kernel_info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "gpu_kernel": True,
            "optimized": True,
            "description": "AWQ INT4 weight-only dequantization",
            "autoawq_available": self._check_autoawq(),
        }

    def _check_autoawq(self) -> bool:
        try:
            import autoawq  # noqa: F401
            return True
        except ImportError:
            return False

    def estimate_memory(
        self,
        num_params: int,
        bits: int = 4,
        group_size: int = 128,
    ) -> int:
        quantized_bytes = num_params * bits // 8
        num_groups = num_params // group_size
        scale_bytes = num_groups * 2
        zero_bytes = num_groups * 2
        return quantized_bytes + scale_bytes + zero_bytes
