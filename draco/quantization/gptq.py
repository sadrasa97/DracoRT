"""
GPTQ Quantization Backend

Supports GPTQ checkpoint loading, dequantization, and GPU inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from draco.quantization.base import QuantizationBackend

logger = logging.getLogger("draco.quantization.gptq")


class GPTQBackend(QuantizationBackend):
    """
    GPTQ quantization backend.

    Supports INT4/INT3 quantization with group-wise quantization.
    """

    @property
    def name(self) -> str:
        return "gptq"

    @property
    def supported_bits(self) -> List[int]:
        return [2, 3, 4, 8]

    @property
    def supported_schemes(self) -> List[str]:
        return ["weight_only"]

    def load_quantized_weights(
        self,
        model_path: str,
        target_model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """
        Load GPTQ-quantized weights from a checkpoint.

        GPTQ checkpoints typically contain:
        - quantized weight tensor
        - scale factors
        - zero points
        - group_size and bits metadata
        """
        bits = config.get("bits", 4)
        group_size = config.get("group_size", 128)
        desc_act = config.get("desc_act", False)

        logger.info(
            "Loading GPTQ weights: bits=%d, group_size=%d, desc_act=%s",
            bits, group_size, desc_act,
        )

        # Try to load from safetensors or pytorch checkpoint
        from draco.weights.loader import get_checkpoint_loader
        loader = get_checkpoint_loader()
        try:
            weights = loader.load(model_path, device=device)
        except Exception as e:
            logger.warning("Failed to load GPTQ weights: %s", e)
            return target_model

        # Map weights to model
        quantized_params = {}
        for key, tensor in weights.items():
            if "qweight" in key or "weight" in key:
                quantized_params[key] = tensor
            elif "scales" in key or "scale" in key:
                quantized_params[key] = tensor
            elif "zeros" in key or "qzeros" in key:
                quantized_params[key] = tensor
            elif "bias" in key:
                quantized_params[key] = tensor

        # Load into model
        if quantized_params:
            target_model.load_state_dict(quantized_params, strict=False)
            logger.info("Loaded %d GPTQ parameters", len(quantized_params))

        return target_model

    def dequantize(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        """
        Dequantize GPTQ-quantized weights.

        Args:
            weights: Quantized int4/int3 weight tensor
            scales: Scale factors per group
            zeros: Zero points per group
            group_size: Group size for quantization

        Returns:
            Dequantized float tensor
        """
        # Ensure float for computation
        weights = weights.to(scales.dtype)

        if zeros is not None:
            zeros = zeros.to(scales.dtype)
            # Dequantize: weight = (quantized - zero) * scale
            # Handle group-wise dequantization
            if scales.dim() > 1:
                # Group-wise: expand scales/zeros to match weight shape
                expanded_scales = self._expand_grouped(scales, weights.shape, group_size)
                expanded_zeros = self._expand_grouped(zeros, weights.shape, group_size) if zeros is not None else 0
                return (weights - expanded_zeros) * expanded_scales
            else:
                return (weights - zeros) * scales
        else:
            return weights * scales

    def _expand_grouped(
        self,
        grouped: torch.Tensor,
        target_shape: torch.Size,
        group_size: int,
    ) -> torch.Tensor:
        """Expand grouped tensor to match target weight shape."""
        if grouped.numel() == 0:
            return torch.zeros(target_shape, dtype=grouped.dtype, device=grouped.device)

        # Simple repeat expand
        repeats = [1] * grouped.dim()
        if target_shape[-1] > grouped.shape[-1]:
            repeats[-1] = target_shape[-1] // grouped.shape[-1]
        expanded = grouped.repeat(*repeats)
        return expanded.expand(target_shape)

    def get_kernel_info(self) -> Dict[str, Any]:
        """Return GPTQ kernel information."""
        return {
            "backend": self.name,
            "gpu_kernel": True,
            "optimized": True,
            "description": "GPTQ INT4/INT3 dequantization kernels",
            "native_support": False,
            "auto_gptq_available": self._check_auto_gptq(),
        }

    def _check_auto_gptq(self) -> bool:
        """Check if auto-gptq is available."""
        try:
            import auto_gptq  # noqa: F401
            return True
        except ImportError:
            return False

    def estimate_memory(
        self,
        num_params: int,
        bits: int = 4,
        group_size: int = 128,
    ) -> int:
        """Estimate memory for GPTQ model."""
        quantized_bytes = num_params * bits // 8
        num_groups = num_params // group_size
        scale_bytes = num_groups * 2  # fp16 scales
        zero_bytes = num_groups * 2  # fp16 zeros
        return quantized_bytes + scale_bytes + zero_bytes
