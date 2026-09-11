"""
DracoReader — memory-maps a .draco file and provides lazy tensor access.

The file is never fully read into RAM. ``mmap`` is used so the OS page
cache controls physical residency; tensor accessors return numpy views
(or freshly-dequantized arrays) computed on demand.
"""

from __future__ import annotations

import json
import mmap
import os
from typing import Dict, Iterator, List, Optional

import numpy as np

from draco.exceptions import DracoFormatError
from draco.format.header import HEADER_SIZE, DracoHeader
from draco.format.metadata import decode as decode_metadata
from draco.format.tensor import DracoTensorInfo
from draco.quantization.cpu import CODECS
from draco.quantization.cpu import bf16 as bf16_codec
from draco.quantization.cpu.int4 import Int4QuantizedWeight
from draco.quantization.cpu.int8 import Int8QuantizedWeight
from draco.quantization.cpu.int8_asym import Int8AsymQuantizedWeight

_DTYPE_NP = {"f32": np.float32, "f16": np.float16, "i8": np.int8, "u8": np.uint8, "u4x2": np.uint8, "bf16": np.uint16}


class DracoReader:
    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, "rb")
        size = os.fstat(self._file.fileno()).st_size
        if size < HEADER_SIZE:
            self._file.close()
            raise DracoFormatError(f"'{path}' is too small to be a valid .draco file.")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        self.header = DracoHeader.unpack(bytes(self._mmap[:HEADER_SIZE]))
        self._file_size = size

        self.metadata: Dict = decode_metadata(
            bytes(
                self._mmap[
                    self.header.metadata_offset : self.header.metadata_offset
                    + self.header.metadata_size
                ]
            )
        )

        self.tokenizer: Optional[Dict] = None
        if self.header.has_tokenizer and self.header.tokenizer_size > 0:
            self.tokenizer = decode_metadata(
                bytes(
                    self._mmap[
                        self.header.tokenizer_offset : self.header.tokenizer_offset
                        + self.header.tokenizer_size
                    ]
                )
            )

        index_bytes = bytes(
            self._mmap[
                self.header.tensor_index_offset : self.header.tensor_index_offset
                + self.header.tensor_index_size
            ]
        )
        raw_entries = json.loads(index_bytes.decode("utf-8"))
        self._index: Dict[str, DracoTensorInfo] = {
            e["name"]: DracoTensorInfo.from_dict(e) for e in raw_entries
        }

    @property
    def architecture(self) -> str:
        return self.metadata.get("architecture", "")

    def tensor_names(self) -> List[str]:
        return list(self._index.keys())

    def tensors(self) -> Iterator[DracoTensorInfo]:
        return iter(self._index.values())

    def tensor_info(self, name: str) -> DracoTensorInfo:
        if name not in self._index:
            raise KeyError(f"No tensor named '{name}' in '{self.path}'.")
        return self._index[name]

    def _raw_bytes(self, offset: int, nbytes: int) -> bytes:
        if offset + nbytes > self._file_size:
            raise DracoFormatError(
                f"Tensor region [{offset}:{offset + nbytes}] extends past end of "
                f"file ({self._file_size} bytes) in '{self.path}'."
            )
        return bytes(self._mmap[offset : offset + nbytes])

    def get_tensor(self, name: str, dequantize: bool = True) -> np.ndarray:
        """Return the tensor as a numpy array.

        If the tensor is quantized and ``dequantize`` is True (default), the
        codec's dequantize() is applied. If False, the raw quantized/packed
        bytes are returned instead (for direct-consumption kernels).
        """
        info = self.tensor_info(name)
        payload = self._raw_bytes(info.offset, info.nbytes)

        if info.quantization is None:
            arr = np.frombuffer(payload, dtype=_DTYPE_NP[info.dtype]).reshape(info.shape)
            if info.dtype == "bf16":
                return bf16_codec.decode(arr) if dequantize else arr
            return arr

        codec = CODECS[info.quantization]
        if info.quantization == "int8_sym":
            qweight = np.frombuffer(payload, dtype=np.int8).reshape(info.shape)
            scale = np.frombuffer(
                self._raw_bytes(info.scale_offset, info.scale_nbytes), dtype=np.float32
            )
            q = Int8QuantizedWeight(qweight=qweight, scale=scale)
            return codec.dequantize(q) if dequantize else qweight
        elif info.quantization == "int8_asym":
            qweight = np.frombuffer(payload, dtype=np.uint8).reshape(info.shape)
            scale = np.frombuffer(
                self._raw_bytes(info.scale_offset, info.scale_nbytes), dtype=np.float32
            )
            zero_point = np.frombuffer(
                self._raw_bytes(info.zero_point_offset, info.zero_point_nbytes), dtype=np.float32
            )
            q = Int8AsymQuantizedWeight(qweight=qweight, scale=scale, zero_point=zero_point)
            return codec.dequantize(q) if dequantize else qweight
        elif info.quantization == "int4_groupwise":
            rows, cols = info.shape
            packed = np.frombuffer(payload, dtype=np.uint8).reshape(rows, cols // 2)
            n_groups = cols // info.group_size
            scale = np.frombuffer(
                self._raw_bytes(info.scale_offset, info.scale_nbytes), dtype=np.float32
            ).reshape(rows, n_groups)
            zero_point = np.frombuffer(
                self._raw_bytes(info.zero_point_offset, info.zero_point_nbytes), dtype=np.float32
            ).reshape(rows, n_groups)
            q = Int4QuantizedWeight(
                packed=packed,
                scale=scale,
                zero_point=zero_point,
                shape=(rows, cols),
                group_size=info.group_size,
            )
            return codec.dequantize(q) if dequantize else packed
        raise DracoFormatError(f"Unknown quantization '{info.quantization}' for tensor '{name}'.")

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "DracoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
