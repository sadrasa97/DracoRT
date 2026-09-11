"""
DracoWriter — builds a .draco container.

Writes are atomic: content is assembled into ``<path>.tmp-<pid>-<rand>``,
fsynced, and only renamed to the final path once fully written and
self-validated. A failed conversion can never leave a corrupt file at the
final path (spec section 9).
"""

from __future__ import annotations

import os
import struct
import tempfile
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from draco.exceptions import DracoFormatError
from draco.format.header import HEADER_SIZE, FORMAT_VERSION, DracoHeader
from draco.format.metadata import encode as encode_metadata, validate_metadata
from draco.format.tensor import SUPPORTED_DTYPES, SUPPORTED_QUANTIZATIONS, DracoTensorInfo
from draco.quantization.cpu import CODECS

_DTYPE_NP = {"f32": np.float32, "f16": np.float16}
# bf16 has no native numpy dtype pre-2.0; store as raw uint16 bit pattern.
_DEFAULT_ALIGNMENT = 64


def _align_up(offset: int, alignment: int) -> int:
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)


@dataclass
class _PendingTensor:
    name: str
    dtype: str
    quantization: Optional[str]
    shape: tuple
    payload: bytes  # primary tensor bytes (raw or quantized/packed)
    scale_bytes: Optional[bytes] = None
    zero_point_bytes: Optional[bytes] = None
    group_size: Optional[int] = None
    block_size: Optional[int] = None
    alignment: int = _DEFAULT_ALIGNMENT


class DracoWriter:
    def __init__(self, path: str, alignment: int = _DEFAULT_ALIGNMENT) -> None:
        self.path = path
        self.alignment = alignment
        self._metadata: Dict[str, Any] = {}
        self._tokenizer: Optional[Dict[str, Any]] = None
        self._tensors: List[_PendingTensor] = []
        self._names_seen = set()
        self._finalized = False

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        self._metadata.update(metadata)

    def add_tokenizer(self, tokenizer: Dict[str, Any]) -> None:
        self._tokenizer = tokenizer

    def add_tensor(
        self,
        name: str,
        array: np.ndarray,
        quantization: Optional[str] = None,
        group_size: Optional[int] = None,
        dtype: Optional[str] = None,
    ) -> None:
        """Add a tensor. If ``quantization`` is set, ``array`` is quantized
        with the matching codec from draco.quantization.cpu before storage.
        If ``dtype='bf16'`` (and quantization is None), the tensor is stored
        as bf16 (half the size of f32, wider dynamic range than f16) via
        draco.quantization.cpu.bf16.
        """
        if name in self._names_seen:
            raise DracoFormatError(f"Duplicate tensor name: '{name}'.")
        self._names_seen.add(name)

        if quantization is None:
            if dtype == "bf16":
                from draco.quantization.cpu import bf16 as bf16_codec

                packed = bf16_codec.encode(array)
                self._tensors.append(
                    _PendingTensor(
                        name=name,
                        dtype="bf16",
                        quantization=None,
                        shape=tuple(array.shape),
                        payload=np.ascontiguousarray(packed).tobytes(),
                        alignment=self.alignment,
                    )
                )
                return

            dtype_name = {np.float32: "f32", np.float16: "f16"}.get(array.dtype.type)
            if dtype_name is None:
                raise DracoFormatError(
                    f"Unsupported unquantized dtype {array.dtype} for tensor '{name}'. "
                    f"Supported: {SUPPORTED_DTYPES}, or pass dtype='bf16'."
                )
            self._tensors.append(
                _PendingTensor(
                    name=name,
                    dtype=dtype_name,
                    quantization=None,
                    shape=tuple(array.shape),
                    payload=np.ascontiguousarray(array).tobytes(),
                    alignment=self.alignment,
                )
            )
            return

        if quantization not in SUPPORTED_QUANTIZATIONS:
            raise DracoFormatError(
                f"Unsupported quantization '{quantization}' for tensor '{name}'. "
                f"Supported: {SUPPORTED_QUANTIZATIONS}. Only formats with a real, "
                f"tested codec may be written."
            )
        codec = CODECS[quantization]
        if quantization == "int8_sym":
            q = codec.quantize(array)
            self._tensors.append(
                _PendingTensor(
                    name=name,
                    dtype="i8",
                    quantization=quantization,
                    shape=tuple(array.shape),
                    payload=np.ascontiguousarray(q.qweight).tobytes(),
                    scale_bytes=np.ascontiguousarray(q.scale).tobytes(),
                    alignment=self.alignment,
                )
            )
        elif quantization == "int8_asym":
            q = codec.quantize(array)
            self._tensors.append(
                _PendingTensor(
                    name=name,
                    dtype="u8",
                    quantization=quantization,
                    shape=tuple(array.shape),
                    payload=np.ascontiguousarray(q.qweight).tobytes(),
                    scale_bytes=np.ascontiguousarray(q.scale).tobytes(),
                    zero_point_bytes=np.ascontiguousarray(q.zero_point).tobytes(),
                    alignment=self.alignment,
                )
            )
        elif quantization == "int4_groupwise":
            gs = group_size or codec.DEFAULT_GROUP_SIZE
            q = codec.quantize(array, group_size=gs)
            self._tensors.append(
                _PendingTensor(
                    name=name,
                    dtype="u4x2",
                    quantization=quantization,
                    shape=tuple(array.shape),
                    payload=np.ascontiguousarray(q.packed).tobytes(),
                    scale_bytes=np.ascontiguousarray(q.scale).tobytes(),
                    zero_point_bytes=np.ascontiguousarray(q.zero_point).tobytes(),
                    group_size=gs,
                    alignment=self.alignment,
                )
            )

    def finalize(self, checksum: bool = True) -> str:
        if self._finalized:
            raise DracoFormatError("This DracoWriter has already been finalized.")
        validate_metadata(self._metadata)

        # Lay out tensor data region first so we know offsets, then build
        # the tensor index, then metadata/tokenizer, then the header.
        data_chunks: List[bytes] = []
        cursor = 0  # relative to start of tensor-data region
        tensor_infos: List[DracoTensorInfo] = []

        for t in self._tensors:
            cursor = _align_up(cursor, t.alignment)
            pad = cursor - sum(len(c) for c in data_chunks)
            if pad > 0:
                data_chunks.append(b"\x00" * pad)
            payload_offset = cursor
            data_chunks.append(t.payload)
            cursor += len(t.payload)

            scale_offset = scale_nbytes = zp_offset = zp_nbytes = None
            if t.scale_bytes is not None:
                scale_offset = cursor
                data_chunks.append(t.scale_bytes)
                scale_nbytes = len(t.scale_bytes)
                cursor += scale_nbytes
            if t.zero_point_bytes is not None:
                zp_offset = cursor
                data_chunks.append(t.zero_point_bytes)
                zp_nbytes = len(t.zero_point_bytes)
                cursor += zp_nbytes

            tensor_infos.append(
                DracoTensorInfo(
                    name=t.name,
                    dtype=t.dtype,
                    quantization=t.quantization,
                    shape=t.shape,
                    offset=payload_offset,  # relative; rebased below
                    nbytes=len(t.payload),
                    alignment=t.alignment,
                    group_size=t.group_size,
                    block_size=t.block_size,
                    scale_offset=scale_offset,
                    scale_nbytes=scale_nbytes,
                    zero_point_offset=zp_offset,
                    zero_point_nbytes=zp_nbytes,
                )
            )

        tensor_data = b"".join(data_chunks)

        metadata_bytes = encode_metadata(self._metadata)
        tokenizer_bytes = encode_metadata(self._tokenizer) if self._tokenizer is not None else b""

        metadata_offset = HEADER_SIZE
        tokenizer_offset = metadata_offset + len(metadata_bytes)
        tensor_index_offset_provisional = tokenizer_offset + len(tokenizer_bytes)

        # Tensor index is JSON; build it once, then rebase tensor offsets to
        # be absolute from file start (index needs to know data offset first,
        # which depends on index size -> two-pass).
        def build_index(data_offset: int) -> bytes:
            import json

            entries = []
            for info in tensor_infos:
                d = info.to_dict()
                d["offset"] = info.offset + data_offset
                if info.scale_offset is not None:
                    d["scale_offset"] = info.scale_offset + data_offset
                if info.zero_point_offset is not None:
                    d["zero_point_offset"] = info.zero_point_offset + data_offset
                entries.append(d)
            return json.dumps(entries).encode("utf-8")

        # First pass with a guess to size the index, then fix point iterate
        # (index size is stable once offsets are filled in with real ints).
        index_bytes = build_index(0)
        tensor_data_offset = _align_up(
            tensor_index_offset_provisional + len(index_bytes), self.alignment
        )
        index_bytes = build_index(tensor_data_offset)  # re-encode with final absolute offsets
        # Re-encoding with larger absolute integers cannot change byte length
        # materially enough to shift alignment in practice, but guard anyway:
        tensor_data_offset2 = _align_up(
            tensor_index_offset_provisional + len(index_bytes), self.alignment
        )
        if tensor_data_offset2 != tensor_data_offset:
            index_bytes = build_index(tensor_data_offset2)
            tensor_data_offset = tensor_data_offset2

        pad_before_data = tensor_data_offset - (tensor_index_offset_provisional + len(index_bytes))

        checksum_value = zlib.crc32(tensor_data) if checksum else 0

        flags = 0
        if self._tokenizer is not None:
            flags |= DracoHeader.FLAG_HAS_TOKENIZER
        if checksum:
            flags |= DracoHeader.FLAG_HAS_CHECKSUM

        header = DracoHeader(
            version=FORMAT_VERSION,
            header_size=HEADER_SIZE,
            flags=flags,
            metadata_offset=metadata_offset,
            metadata_size=len(metadata_bytes),
            tensor_index_offset=tensor_index_offset_provisional,
            tensor_index_size=len(index_bytes),
            tensor_data_offset=tensor_data_offset,
            tensor_data_size=len(tensor_data),
            tokenizer_offset=tokenizer_offset if self._tokenizer is not None else 0,
            tokenizer_size=len(tokenizer_bytes),
            checksum=checksum_value & 0xFFFFFFFF,
        )

        out_dir = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".draco.tmp.", dir=out_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(header.pack())
                f.write(metadata_bytes)
                f.write(tokenizer_bytes)
                f.write(index_bytes)
                f.write(b"\x00" * pad_before_data)
                f.write(tensor_data)
                f.flush()
                os.fsync(f.fileno())

            # Self-validate before publishing under the real name.
            from draco.format.validator import validate_file

            validate_file(tmp_path)

            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        self._finalized = True
        return self.path
