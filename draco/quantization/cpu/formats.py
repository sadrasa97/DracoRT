"""
Draco CPU quantization format registry.

Canonical format names (what `.draco` stores in the tensor index):

    DRACO_F32             float32 weights
    DRACO_F16             float16 weights
    DRACO_BF16            bfloat16 weights (stored as uint16 bit patterns)
    DRACO_I8_SYM          per-row symmetric INT8 (int8 payload + f32 scales)
    DRACO_I8_ASYM         per-row asymmetric INT8 (+ f32 mins)
    DRACO_I4_GROUPWISE    INT4 with per-group scales/zero points
    DRACO_Q4_K            64-block 4-bit (fp16 scale + min per block)
    DRACO_Q5_K            64-block 5-bit (fp16 scale + min per block)
    DRACO_Q6_K            64-block symmetric 6-bit (fp16 scale per block)

A format is only registered here when it has a real quantize/dequantize
implementation AND a GEMM kernel in ``draco.runtime.cpu.kernels``; the
dispatcher refuses formats without kernels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from draco.exceptions import CPUQuantizationError
from draco.quantization.cpu import float as _float
from draco.quantization.cpu import int4 as _int4
from draco.quantization.cpu import int8 as _int8
from draco.quantization.cpu import q4_k as _q4_k
from draco.quantization.cpu import q5_k as _q5_k
from draco.quantization.cpu import q6_k as _q6_k

F32 = "DRACO_F32"
F16 = "DRACO_F16"
BF16 = "DRACO_BF16"
I8_SYM = "DRACO_I8_SYM"
I8_ASYM = "DRACO_I8_ASYM"
I4_GROUPWISE = "DRACO_I4_GROUPWISE"
Q4_K = "DRACO_Q4_K"
Q5_K = "DRACO_Q5_K"
Q6_K = "DRACO_Q6_K"

# CLI aliases -> canonical names
ALIASES = {
    "none": F32,
    "f32": F32,
    "float32": F32,
    "f16": F16,
    "float16": F16,
    "bf16": BF16,
    "bfloat16": BF16,
    "int8": I8_SYM,
    "i8": I8_SYM,
    "i8_sym": I8_SYM,
    "i8_asym": I8_ASYM,
    "int4": I4_GROUPWISE,
    "i4": I4_GROUPWISE,
    "i4_groupwise": I4_GROUPWISE,
    "q4_k": Q4_K,
    "q4k": Q4_K,
    "q5_k": Q5_K,
    "q5k": Q5_K,
    "q6_k": Q6_K,
    "q6k": Q6_K,
}

# formats with a real quantize/dequantize + kernel implementation
IMPLEMENTED: List[str] = [
    F32,
    F16,
    BF16,
    I8_SYM,
    I8_ASYM,
    I4_GROUPWISE,
    Q4_K,
    Q5_K,
    Q6_K,
]


def canonical(name: str) -> str:
    """Normalize a format name/alias to its canonical form."""
    if name is None:
        return F32
    key = str(name).lower()
    if key in ALIASES:
        return ALIASES[key]
    upper = str(name).upper()
    if upper in IMPLEMENTED:
        return upper
    raise CPUQuantizationError(
        f"Unknown quantization format '{name}'. Supported: {IMPLEMENTED}"
    )


def supported_formats() -> List[str]:
    return list(IMPLEMENTED)


def quantize(
    weights: np.ndarray,
    fmt: str,
    group_size: int = 128,
) -> Dict[str, Any]:
    """
    Quantize a float32 weight to the given format.

    Returns a dict with byte payloads ready for ``DracoWriter.add_tensor``:

        payload, scale_bytes, zero_point_bytes, min_bytes,
        block_size, group_size, num_groups
    """
    fmt = canonical(fmt)
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise CPUQuantizationError(
            f"CPU quantization supports 2D weights, got shape {w.shape} for '{fmt}'"
        )

    if fmt == F32:
        payload = w.tobytes()
        return _result(payload, None, None, None, None, None, 0)
    if fmt == F16:
        return _result(_float.f16_bytes_from_f32(w), None, None, None, None, None, 0)
    if fmt == BF16:
        return _result(_float.bf16_bytes_from_f32(w), None, None, None, None, None, 0)
    if fmt == I8_SYM:
        q, scale = _int8.quantize_sym(w)
        return _result(q.tobytes(), scale.tobytes(), None, None, None, None, 0)
    if fmt == I8_ASYM:
        q, scale, mins = _int8.quantize_asym(w)
        return _result(q.tobytes(), scale.tobytes(), None, mins.tobytes(), None, None, 0)
    if fmt == I4_GROUPWISE:
        payload, scale, zp, ng = _int4.quantize(w, group_size)
        return _result(
            payload.tobytes(), scale.tobytes(), zp.tobytes(), None,
            None, group_size, ng,
        )
    if fmt == Q4_K:
        payload, _, _, _ = _q4_k.quantize(w)
        return _result(payload.tobytes(), None, None, None, _q4_k.BLOCK_SIZE, None, 0)
    if fmt == Q5_K:
        payload, _, _, _ = _q5_k.quantize(w)
        return _result(payload.tobytes(), None, None, None, _q5_k.BLOCK_SIZE, None, 0)
    if fmt == Q6_K:
        payload, _, _, _ = _q6_k.quantize(w)
        return _result(payload.tobytes(), None, None, None, _q6_k.BLOCK_SIZE, None, 0)
    raise CPUQuantizationError(f"Format '{fmt}' not implemented")


def _result(payload, scale, zp, mins, block_size, group_size, num_groups) -> Dict[str, Any]:
    return {
        "payload": payload,
        "scale_bytes": scale,
        "zero_point_bytes": zp,
        "min_bytes": mins,
        "block_size": block_size,
        "group_size": group_size,
        "num_groups": num_groups,
    }


def dequantize(
    payload: np.ndarray,
    fmt: str,
    shape: Tuple[int, ...],
    scale: Optional[np.ndarray] = None,
    zero_point: Optional[np.ndarray] = None,
    mins: Optional[np.ndarray] = None,
    group_size: int = 128,
) -> np.ndarray:
    """Reconstruct a float32 [R, K] weight from stored arrays (reference path)."""
    fmt = canonical(fmt)
    if fmt == F32:
        return np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()
    if fmt == F16:
        return _float.f16_to_f32(np.frombuffer(payload, dtype=np.float16).reshape(shape))
    if fmt == BF16:
        return _float.bf16_to_f32(
            np.frombuffer(payload, dtype=np.uint16).reshape(shape)
        )
    if fmt == I8_SYM:
        q = np.frombuffer(payload, dtype=np.int8).reshape(shape)
        return _int8.dequantize_sym(q, scale)
    if fmt == I8_ASYM:
        q = np.frombuffer(payload, dtype=np.int8).reshape(shape)
        return _int8.dequantize_asym(q, scale, mins)
    if fmt == I4_GROUPWISE:
        return _int4.dequantize(payload, scale, zero_point, group_size)
    if fmt == Q4_K:
        return _q4_k.dequantize(payload, shape)
    if fmt == Q5_K:
        return _q5_k.dequantize(payload, shape)
    if fmt == Q6_K:
        return _q6_k.dequantize(payload, shape)
    raise CPUQuantizationError(f"Format '{fmt}' not implemented")


def storage_dtype(fmt: str) -> str:
    """Raw storage dtype string recorded in the tensor index."""
    fmt = canonical(fmt)
    return {
        F32: "float32",
        F16: "float16",
        BF16: "bfloat16",
        I8_SYM: "int8",
        I8_ASYM: "int8",
        I4_GROUPWISE: "uint8",
        Q4_K: "uint8",
        Q5_K: "uint8",
        Q6_K: "uint8",
    }[fmt]


# ---------------------------------------------------------------------------
# mmap-backed tensor views
# ---------------------------------------------------------------------------


class DracoTensorView:
    """
    Lazy view of a tensor inside a memory-mapped `.draco` container.

    Kernels consume the raw mapped arrays directly (``raw_payload``,
    ``scales``, ...) — a full float32 weight matrix is never materialized
    unless ``dequantize()`` is explicitly called (tests / fallback paths).
    """

    def __init__(self, reader, info) -> None:
        self._reader = reader
        self.info = info
        self.name = info.name
        self.shape = info.shape
        self.quantization = info.quantization
        self.dtype = info.dtype

    def _region(self, offset: Optional[int], size: int) -> Optional[memoryview]:
        if offset is None or size <= 0:
            return None
        if not (self.info.offset <= offset < self.info.end):
            raise CPUQuantizationError(
                f"Tensor '{self.name}': auxiliary region at {offset} outside tensor"
            )
        return self._reader.tensor_view(self.name, offset - self.info.offset, size)

    def raw_payload(self) -> np.ndarray:
        dt = self._np_dtype()
        size = self._payload_size()
        view = self._reader.tensor_view(self.name, 0, size)
        return np.frombuffer(view, dtype=dt)

    def scales(self) -> Optional[np.ndarray]:
        info = self.info
        size = self.shape[0] * 4  # float32 per row
        if info.quantization == I4_GROUPWISE:
            size = self.shape[0] * info.num_groups * 2  # float16
        view = self._region(info.scale_offset, size)
        if view is None:
            return None
        dt = np.float16 if info.quantization == I4_GROUPWISE else np.float32
        return np.frombuffer(view, dtype=dt)

    def zero_points(self) -> Optional[np.ndarray]:
        info = self.info
        if info.zero_point_offset is None:
            return None
        size = self.shape[0] * info.num_groups if info.quantization == I4_GROUPWISE else 0
        view = self._region(info.zero_point_offset, size)
        if view is None:
            return None
        return np.frombuffer(view, dtype=np.uint8)

    def mins(self) -> Optional[np.ndarray]:
        info = self.info
        if info.min_offset is None:
            return None
        size = self.shape[0] * 4
        view = self._region(info.min_offset, size)
        if view is None:
            return None
        return np.frombuffer(view, dtype=np.float32)

    def _np_dtype(self):
        return {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.uint16,
            "int8": np.int8,
            "uint8": np.uint8,
        }.get(self.dtype, np.uint8)

    def _payload_size(self) -> int:
        numel = 1
        for d in self.shape:
            numel *= d
        q = self.quantization
        if q in (F32,):
            return numel * 4
        if q in (F16, BF16):
            return numel * 2
        if q in (I8_SYM, I8_ASYM):
            return numel
        if q == I4_GROUPWISE:
            return numel // 2
        if q == Q4_K:
            return numel // 64 * 36
        if q == Q5_K:
            return numel // 64 * 44
        if q == Q6_K:
            return numel // 64 * 50
        return self.info.nbytes

    def dequantize(self) -> np.ndarray:
        """Materialize the tensor as float32 (reference/test path only)."""
        return dequantize(
            self.raw_payload(),
            self.quantization,
            self.shape,
            scale=self.scales(),
            zero_point=self.zero_points(),
            mins=self.mins(),
            group_size=self.info.group_size or 128,
        )

    def __repr__(self) -> str:
        return (
            f"DracoTensorView({self.name!r}, shape={self.shape}, "
            f"quantization={self.quantization})"
        )


def create_tensor_view(reader, info) -> DracoTensorView:
    """Factory used by the reader for lazy tensor access."""
    return DracoTensorView(reader, info)