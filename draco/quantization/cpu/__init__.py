"""CPU-native weight quantization codecs for the .draco format.

Only formats with an actual, tested codec are listed here — never expose
a quantization option that lacks a working implementation.
"""

from __future__ import annotations

from . import bf16, int4, int8, int8_asym

CODECS = {
    int8.NAME: int8,
    int8_asym.NAME: int8_asym,
    int4.NAME: int4,
}

__all__ = ["int8", "int8_asym", "int4", "bf16", "CODECS"]
