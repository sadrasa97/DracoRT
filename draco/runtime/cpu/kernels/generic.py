"""
Portable (numpy) CPU GEMM kernel.

This is the correctness oracle every future ISA-specific kernel (AVX2,
AVX512, VNNI, AMX, NEON) must match within tolerance before it can be
trusted (spec section 35). No CPU-specific intrinsics are used here —
it runs identically, if not maximally fast, on every host.
"""

from __future__ import annotations

import numpy as np

from draco.format.tensor import DracoTensorInfo
from draco.quantization.cpu import int4, int8, int8_asym
from draco.quantization.cpu.int4 import Int4QuantizedWeight
from draco.quantization.cpu.int8 import Int8QuantizedWeight
from draco.quantization.cpu.int8_asym import Int8AsymQuantizedWeight
from draco.runtime.cpu.kernels.base import CPUGemmKernel


class GenericGemmKernel(CPUGemmKernel):
    tier = "generic"

    @property
    def name(self) -> str:
        return "generic (portable numpy)"

    def supports(self, quantization: str) -> bool:
        return quantization in ("int8_sym", "int8_asym", "int4_groupwise", None)

    def matmul_f32(self, activation: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return activation.astype(np.float32, copy=False) @ weight.astype(np.float32, copy=False).T

    def matmul(
        self, activation: np.ndarray, tensor_info: DracoTensorInfo, payload_bytes: bytes, **kw
    ) -> np.ndarray:
        if tensor_info.quantization == "int8_sym":
            qweight = np.frombuffer(payload_bytes, dtype=np.int8).reshape(tensor_info.shape)
            scale = kw["scale"]
            q = Int8QuantizedWeight(qweight=qweight, scale=scale)
            return int8.matmul(activation, q)
        elif tensor_info.quantization == "int8_asym":
            qweight = np.frombuffer(payload_bytes, dtype=np.uint8).reshape(tensor_info.shape)
            q = Int8AsymQuantizedWeight(qweight=qweight, scale=kw["scale"], zero_point=kw["zero_point"])
            return int8_asym.matmul(activation, q)
        elif tensor_info.quantization == "int4_groupwise":
            rows, cols = tensor_info.shape
            packed = np.frombuffer(payload_bytes, dtype=np.uint8).reshape(rows, cols // 2)
            q = Int4QuantizedWeight(
                packed=packed,
                scale=kw["scale"],
                zero_point=kw["zero_point"],
                shape=(rows, cols),
                group_size=tensor_info.group_size,
            )
            return int4.matmul(activation, q)
        else:
            weight = np.frombuffer(payload_bytes, dtype=np.float32).reshape(tensor_info.shape)
            return self.matmul_f32(activation, weight)
