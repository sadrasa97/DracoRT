"""Tensor index entry for .draco files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# dtypes with an actual numpy mapping + implemented (de)quant kernel.
# Only formats listed here may be written or read — never claim support
# for a format without a working codec (see draco.quantization.cpu.registry).
SUPPORTED_DTYPES = ("f32", "f16", "bf16")
SUPPORTED_QUANTIZATIONS = ("int8_sym", "int8_asym", "int4_groupwise")


@dataclass
class DracoTensorInfo:
    name: str

    dtype: str  # one of SUPPORTED_DTYPES — the *storage* dtype
    quantization: Optional[str] = None  # one of SUPPORTED_QUANTIZATIONS or None

    shape: Tuple[int, ...] = field(default_factory=tuple)

    offset: int = 0
    nbytes: int = 0

    alignment: int = 64

    block_size: Optional[int] = None
    group_size: Optional[int] = None

    scale_offset: Optional[int] = None
    scale_nbytes: Optional[int] = None
    zero_point_offset: Optional[int] = None
    zero_point_nbytes: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "shape": list(self.shape),
            "offset": self.offset,
            "nbytes": self.nbytes,
            "alignment": self.alignment,
            "block_size": self.block_size,
            "group_size": self.group_size,
            "scale_offset": self.scale_offset,
            "scale_nbytes": self.scale_nbytes,
            "zero_point_offset": self.zero_point_offset,
            "zero_point_nbytes": self.zero_point_nbytes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DracoTensorInfo":
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            quantization=d.get("quantization"),
            shape=tuple(d["shape"]),
            offset=d["offset"],
            nbytes=d["nbytes"],
            alignment=d.get("alignment", 64),
            block_size=d.get("block_size"),
            group_size=d.get("group_size"),
            scale_offset=d.get("scale_offset"),
            scale_nbytes=d.get("scale_nbytes"),
            zero_point_offset=d.get("zero_point_offset"),
            zero_point_nbytes=d.get("zero_point_nbytes"),
        )
