"""
Quantization Configuration

Configuration for quantization methods, detected from model metadata
or provided explicitly by the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class QuantizationConfig:
    """
    Quantization configuration.

    Can be auto-detected from model metadata or provided explicitly.
    """

    method: Optional[str] = None  # gptq, awq, fp8, int8, int4, bitsandbytes
    bits: int = 16
    group_size: int = 128
    zero_point: bool = True
    symmetric: bool = True
    dtype: str = "float16"  # target dtype
    scheme: str = "weight_only"  # weight_only, w8a8, w4a16, w8a16

    # Method-specific settings
    desc_act: bool = False  # GPTQ: descending activation order
    true_sequential: bool = False  # GPTQ
    damp_percent: float = 0.01  # GPTQ
    clip_threshold: float = 1.0  # AWQ: clipping threshold

    # Auto-detection metadata
    quantization_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuantizationConfig:
        """Create from a dictionary (e.g., from config.json quantization_config)."""
        method = data.get("quant_method", data.get("quantization_method"))
        return cls(
            method=method,
            bits=data.get("bits", 16),
            group_size=data.get("group_size", 128),
            zero_point=data.get("zero_point", True),
            symmetric=data.get("symmetric", True),
            dtype=data.get("dtype", "float16"),
            scheme=data.get("scheme", "weight_only"),
            desc_act=data.get("desc_act", False),
            true_sequential=data.get("true_sequential", False),
            damp_percent=data.get("damp_percent", 0.01),
            clip_threshold=data.get("clip_threshold", 1.0),
            quantization_config=data,
        )

    @classmethod
    def auto_detect(cls, config_data: Dict[str, Any]) -> QuantizationConfig:
        """
        Auto-detect quantization config from model config.json data.
        Returns QuantizationConfig with method=None if not quantized.
        """
        qc = config_data.get("quantization_config")
        if qc is None:
            return cls()  # Not quantized

        method = qc.get("quant_method", qc.get("quantization_method"))
        if method is None:
            return cls()

        return cls.from_dict(qc)

    @property
    def is_quantized(self) -> bool:
        return self.method is not None

    def __repr__(self) -> str:
        if not self.is_quantized:
            return "QuantizationConfig(quantized=False)"
        return (
            f"QuantizationConfig(method={self.method!r}, bits={self.bits}, "
            f"group_size={self.group_size}, scheme={self.scheme!r})"
        )
