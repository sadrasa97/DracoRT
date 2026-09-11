"""
.draco binary header.

Fixed-size, versioned header. The file's internal magic/version is the
source of truth for validity — the ``.draco`` extension is advisory only.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from draco.exceptions import DracoFormatError, DracoFormatVersionError

MAGIC = b"DRACO\x00\x00\x00"  # 8 bytes, null-padded
FORMAT_VERSION = 1

# < little-endian
# 8s  magic
# I   version
# I   header_size
# I   flags
# Q Q metadata_offset, metadata_size
# Q Q tensor_index_offset, tensor_index_size
# Q Q tensor_data_offset, tensor_data_size
# Q Q tokenizer_offset, tokenizer_size   (0,0 if absent)
# I   checksum (crc32 of tensor data region, 0 if disabled)
_STRUCT_FMT = "<8sIII" + "QQ" * 4 + "I"
HEADER_SIZE = struct.calcsize(_STRUCT_FMT)  # fixed, independent of content


@dataclass
class DracoHeader:
    version: int
    header_size: int
    flags: int
    metadata_offset: int
    metadata_size: int
    tensor_index_offset: int
    tensor_index_size: int
    tensor_data_offset: int
    tensor_data_size: int
    tokenizer_offset: int
    tokenizer_size: int
    checksum: int

    FLAG_HAS_TOKENIZER = 1 << 0
    FLAG_HAS_CHECKSUM = 1 << 1

    def pack(self) -> bytes:
        return struct.pack(
            _STRUCT_FMT,
            MAGIC,
            self.version,
            self.header_size,
            self.flags,
            self.metadata_offset,
            self.metadata_size,
            self.tensor_index_offset,
            self.tensor_index_size,
            self.tensor_data_offset,
            self.tensor_data_size,
            self.tokenizer_offset,
            self.tokenizer_size,
            self.checksum,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "DracoHeader":
        if len(data) < HEADER_SIZE:
            raise DracoFormatError(
                f"File too small to contain a Draco header "
                f"({len(data)} bytes < {HEADER_SIZE} bytes required)."
            )
        fields = struct.unpack(_STRUCT_FMT, data[:HEADER_SIZE])
        magic = fields[0]
        if magic != MAGIC:
            raise DracoFormatError(
                f"Not a valid .draco file: bad magic bytes {magic!r}. "
                f"The file extension is not sufficient — the internal "
                f"magic must match."
            )
        version = fields[1]
        if version != FORMAT_VERSION:
            raise DracoFormatVersionError(version, FORMAT_VERSION)
        return cls(
            version=fields[1],
            header_size=fields[2],
            flags=fields[3],
            metadata_offset=fields[4],
            metadata_size=fields[5],
            tensor_index_offset=fields[6],
            tensor_index_size=fields[7],
            tensor_data_offset=fields[8],
            tensor_data_size=fields[9],
            tokenizer_offset=fields[10],
            tokenizer_size=fields[11],
            checksum=fields[12],
        )

    @property
    def has_tokenizer(self) -> bool:
        return bool(self.flags & self.FLAG_HAS_TOKENIZER)

    @property
    def has_checksum(self) -> bool:
        return bool(self.flags & self.FLAG_HAS_CHECKSUM)
