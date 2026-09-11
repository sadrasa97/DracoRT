"""The .draco native model container format.

A self-contained, mmap-able, versioned binary format for CPU-native
inference — see draco.format.reader / .writer / .validator.
"""

from __future__ import annotations

from draco.format.header import FORMAT_VERSION, HEADER_SIZE, DracoHeader
from draco.format.reader import DracoReader
from draco.format.tensor import DracoTensorInfo
from draco.format.validator import ValidationReport, validate_file, validate_or_raise
from draco.format.writer import DracoWriter

__all__ = [
    "DracoReader",
    "DracoWriter",
    "DracoHeader",
    "DracoTensorInfo",
    "ValidationReport",
    "validate_file",
    "validate_or_raise",
    "FORMAT_VERSION",
    "HEADER_SIZE",
]
