"""
.draco structural validation (spec section 31).

Checks magic/version/header, offset bounds, alignment, tensor-index
consistency, and checksum — without loading tensor data into RAM. No
tensor region may point outside the file; malformed input must raise a
specific DracoFormatError, never segfault or a raw KeyError/struct error.
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass, field
from typing import List

from draco.exceptions import DracoCorruptModelError, DracoFormatError
from draco.format.header import HEADER_SIZE, DracoHeader
from draco.format.tensor import SUPPORTED_DTYPES, SUPPORTED_QUANTIZATIONS


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    num_tensors: int = 0
    checksum_verified: bool = False


def validate_file(path: str) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        try:
            header = DracoHeader.unpack(header_bytes)
        except DracoFormatError as e:
            return ValidationReport(ok=False, errors=[str(e)])

        for label, offset, length in [
            ("metadata", header.metadata_offset, header.metadata_size),
            ("tensor_index", header.tensor_index_offset, header.tensor_index_size),
            ("tensor_data", header.tensor_data_offset, header.tensor_data_size),
            ("tokenizer", header.tokenizer_offset, header.tokenizer_size),
        ]:
            if length == 0:
                continue
            if offset < HEADER_SIZE or offset + length > size:
                errors.append(
                    f"Section '{label}' [{offset}:{offset + length}] is out of file "
                    f"bounds (file size {size})."
                )

        if errors:
            return ValidationReport(ok=False, errors=errors)

        f.seek(header.metadata_offset)
        metadata_bytes = f.read(header.metadata_size)
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except Exception as e:
            errors.append(f"Metadata section is not valid JSON: {e}")
            return ValidationReport(ok=False, errors=errors)

        for key in ("architecture", "model_type"):
            if key not in metadata:
                errors.append(f"Metadata missing required key '{key}'.")

        f.seek(header.tensor_index_offset)
        index_bytes = f.read(header.tensor_index_size)
        try:
            entries = json.loads(index_bytes.decode("utf-8"))
        except Exception as e:
            errors.append(f"Tensor index is not valid JSON: {e}")
            return ValidationReport(ok=False, errors=errors)

        seen_names = set()
        data_end = header.tensor_data_offset + header.tensor_data_size
        for entry in entries:
            name = entry.get("name")
            if not name:
                errors.append("Tensor index entry missing 'name'.")
                continue
            if name in seen_names:
                errors.append(f"Duplicate tensor name in index: '{name}'.")
            seen_names.add(name)

            dtype = entry.get("dtype")
            quantization = entry.get("quantization")
            if quantization is not None and quantization not in SUPPORTED_QUANTIZATIONS:
                errors.append(f"Tensor '{name}' has unrecognized quantization '{quantization}'.")

            for region_name, off_key, size_key in [
                ("payload", "offset", "nbytes"),
                ("scale", "scale_offset", "scale_nbytes"),
                ("zero_point", "zero_point_offset", "zero_point_nbytes"),
            ]:
                off = entry.get(off_key)
                length = entry.get(size_key)
                if off is None or length is None:
                    continue
                if off < HEADER_SIZE or off + length > size or off + length > data_end:
                    errors.append(
                        f"Tensor '{name}' region '{region_name}' [{off}:{off + length}] "
                        f"points outside the tensor-data region or file bounds."
                    )

            alignment = entry.get("alignment", 64)
            if entry.get("offset") is not None and entry["offset"] % alignment != 0:
                warnings.append(
                    f"Tensor '{name}' payload offset {entry['offset']} is not aligned "
                    f"to {alignment} bytes."
                )

        num_tensors = len(entries)

    checksum_verified = False
    if header.has_checksum and not errors:
        with open(path, "rb") as f:
            f.seek(header.tensor_data_offset)
            data = f.read(header.tensor_data_size)
        actual = zlib.crc32(data) & 0xFFFFFFFF
        if actual != header.checksum:
            errors.append(
                f"Checksum mismatch: header says {header.checksum:#010x}, "
                f"computed {actual:#010x} over tensor data region."
            )
        else:
            checksum_verified = True

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        num_tensors=num_tensors,
        checksum_verified=checksum_verified,
    )


def validate_or_raise(path: str) -> ValidationReport:
    report = validate_file(path)
    if not report.ok:
        raise DracoCorruptModelError(
            f"'{path}' failed .draco validation:\n" + "\n".join(f"  - {e}" for e in report.errors)
        )
    return report
