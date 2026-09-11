"""
GGUF reader — loads llama.cpp ".gguf" model files directly.

GGUF is the native model format of llama.cpp. This module is a pure-numpy,
dependency-free reader (no ``gguf`` pip package, no torch) that exposes the
same duck-typed interface as :class:`draco.format.reader.DracoReader`, so the
existing CPU execution graph (``draco.runtime.cpu.executor``) can run a .gguf
model with zero changes:

    from draco.format.gguf import GGUFReader
    from draco.runtime.cpu.executor import CPUExecutionBackend

    reader = GGUFReader("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    backend = CPUExecutionBackend(reader)
    logits = backend.forward_step(seq_id, prompt_ids, 0)

What is supported:

* GGUF container parsing (v1/v2/v3, little-endian): header, metadata KV,
  tensor infos, and the tensor data section.
* Weight dequantization to float32 for the common ggml types:
  F32, F16, BF16, F64, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_K, Q2_K, Q3_K,
  Q4_K, Q5_K, Q6_K. Anything else (IQ*, MXFP4, ...) raises a clear error.
* Mapping GGUF metadata keys (``llama.*`` / ``qwen2.*`` / ...) onto the keys
  the CPU executor reads (``hidden_size``, ``num_hidden_layers``, ...).
* Mapping GGUF tensor names (``blk.0.attn_q.weight``, ``token_embd.weight``,
  ...) onto canonical HF-style names the executor's weight lookup expects
  (``model.layers.0.self_attn.q_proj.weight``, ...).

Honest limitations: quantized tensors are dequantized to float32 on first
access and cached in RAM (no block-format GEMM kernels), and only the
architectures the CPU executor can actually run are accepted — llama-family
(llama, qwen2, qwen3, mistral, phi3, yi, starcoder2) and gpt2.
"""

from __future__ import annotations

import mmap
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from draco.exceptions import DracoFormatError

# ---------------------------------------------------------------------------
# GGUF constants (see ggml/docs/gguf.md and ggml/src/ggml-quants.c)
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"

# ggml_type enum (uint32)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_F64 = 28
GGML_TYPE_BF16 = 30

# (block_size, block_bytes) for every supported quantized type.
# Layouts come from ggml-quants.h; dequant formulas from ggml-quants.c.
_BLOCK_LAYOUTS: Dict[int, Tuple[int, int]] = {
    GGML_TYPE_Q4_0: (32, 18),  # d: f16 + qs: uint8[16]
    GGML_TYPE_Q4_1: (32, 20),  # d: f16 + m: f16 + qs: uint8[16]
    GGML_TYPE_Q5_0: (32, 22),  # d: f16 + qh: uint8[4] + qs: uint8[16]
    GGML_TYPE_Q5_1: (32, 24),  # d: f16 + m: f16 + qh: uint8[4] + qs: uint8[16]
    GGML_TYPE_Q8_0: (32, 34),  # d: f16 + qs: int8[32]
    GGML_TYPE_Q8_K: (256, 258),  # d: f16 + qs: int8[256]
    GGML_TYPE_Q2_K: (256, 84),  # d: f16 + dmin: f16 + qs: uint8[64] + scales: uint8[16]
    GGML_TYPE_Q3_K: (256, 110),  # d: f16 + scales: uint8[12] + qs: uint8[64] + hmask: uint8[32]
    GGML_TYPE_Q4_K: (256, 144),  # d: f16 + dmin: f16 + scales: uint8[12] + qs: uint8[128]
    GGML_TYPE_Q5_K: (256, 176),  # d: f16 + dmin: f16 + scales: uint8[12] + qh: uint8[32] + qs: uint8[128]
    GGML_TYPE_Q6_K: (256, 210),  # d: f16 + ql: uint8[128] + qh: uint8[64] + scales: int8[16]
}

# gguf_metadata_value_type enum (uint32)
_VAL_UINT8 = 0
_VAL_INT8 = 1
_VAL_UINT16 = 2
_VAL_INT16 = 3
_VAL_UINT32 = 4
_VAL_INT32 = 5
_VAL_FLOAT32 = 6
_VAL_BOOL = 7
_VAL_STRING = 8
_VAL_ARRAY = 9
_VAL_UINT64 = 10
_VAL_INT64 = 11
_VAL_FLOAT64 = 12

# ggml architectures -> executor family. Only architectures the CPU executor
# can actually run correctly are accepted (see executor docstring).
_ARCH_FAMILY = {
    "llama": "llama",
    "qwen2": "llama",
    "qwen3": "llama",
    "mistral": "llama",
    "phi3": "llama",
    "yi": "llama",
    "starcoder2": "llama",
    "gpt2": "gpt2",
}
_LLAMA_FAMILY_NAMES = sorted(
    a for a, f in _ARCH_FAMILY.items() if f == "llama"
)

# GGUF tensor name -> canonical HF-style name used by the executor.
# Both the plain and "model."-prefixed forms of the HF names are searched by
# the executor, so we emit the "model."-prefixed canonical form.
_GPT2_TENSOR_MAP = {
    "wte.weight": "model.wte.weight",
    "wpe.weight": "model.wpe.weight",
    "ln_f.weight": "model.ln_f.weight",
    "ln_f.bias": "model.ln_f.bias",
    "output.weight": "model.lm_head.weight",
}
_GPT2_BLOCK_MAP = {
    "ln_1.weight": "ln_1.weight",
    "ln_1.bias": "ln_1.bias",
    "attn.c_attn.weight": "attn.c_attn.weight",
    "attn.c_attn.bias": "attn.c_attn.bias",
    "attn.c_proj.weight": "attn.c_proj.weight",
    "attn.c_proj.bias": "attn.c_proj.bias",
    "ln_2.weight": "ln_2.weight",
    "ln_2.bias": "ln_2.bias",
    "mlp.c_fc.weight": "mlp.c_fc.weight",
    "mlp.c_fc.bias": "mlp.c_fc.bias",
    "mlp.c_proj.weight": "mlp.c_proj.weight",
    "mlp.c_proj.bias": "mlp.c_proj.bias",
}
_LLAMA_TENSOR_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "model.lm_head.weight",
}
_LLAMA_BLOCK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_norm.bias": "input_layernorm.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_qkv.bias": "self_attn.qkv_proj.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.bias": "post_attention_layernorm.bias",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_gate.bias": "mlp.gate_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
}

_BLOCK_RE = re.compile(r"blk\.(\d+)\.(.+)")


def is_gguf_file(path: str) -> bool:
    """True if the file at ``path`` starts with the GGUF magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


# ---------------------------------------------------------------------------
# small binary readers
# ---------------------------------------------------------------------------

def _read_string(buf: bytes, pos: int) -> Tuple[str, int]:
    if pos + 8 > len(buf):
        raise DracoFormatError("GGUF string length field extends past end of file.")
    (length,) = struct.unpack_from("<Q", buf, pos)
    if pos + 8 + length > len(buf):
        raise DracoFormatError("GGUF string data extends past end of file.")
    return buf[pos + 8 : pos + 8 + length].decode("utf-8", errors="replace"), pos + 8 + length


def _read_metadata_value(buf: bytes, pos: int, vtype: int) -> Tuple[Any, int]:
    """Read one GGUF metadata value (non-array) of the given type."""
    if vtype == _VAL_UINT8:
        return buf[pos], pos + 1
    if vtype == _VAL_INT8:
        return struct.unpack_from("<b", buf, pos)[0], pos + 1
    if vtype == _VAL_UINT16:
        return struct.unpack_from("<H", buf, pos)[0], pos + 2
    if vtype == _VAL_INT16:
        return struct.unpack_from("<h", buf, pos)[0], pos + 2
    if vtype == _VAL_UINT32:
        return struct.unpack_from("<I", buf, pos)[0], pos + 4
    if vtype == _VAL_INT32:
        return struct.unpack_from("<i", buf, pos)[0], pos + 4
    if vtype == _VAL_FLOAT32:
        return struct.unpack_from("<f", buf, pos)[0], pos + 4
    if vtype == _VAL_BOOL:
        return bool(buf[pos]), pos + 1
    if vtype == _VAL_STRING:
        return _read_string(buf, pos)
    if vtype == _VAL_UINT64:
        return struct.unpack_from("<Q", buf, pos)[0], pos + 8
    if vtype == _VAL_INT64:
        return struct.unpack_from("<q", buf, pos)[0], pos + 8
    if vtype == _VAL_FLOAT64:
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    raise DracoFormatError(f"Unsupported GGUF metadata value type {vtype}.")


def _read_metadata(buf: bytes, pos: int, kv_count: int) -> Tuple[Dict[str, Any], int]:
    """Read the metadata KV section; returns (dict, position_after_last_kv)."""
    meta: Dict[str, Any] = {}
    for _ in range(kv_count):
        key, pos = _read_string(buf, pos)
        if pos + 4 > len(buf):
            raise DracoFormatError("GGUF metadata value type extends past end of file.")
        (vtype,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if vtype == _VAL_ARRAY:
            if pos + 12 > len(buf):
                raise DracoFormatError("GGUF metadata array header extends past end of file.")
            (elem_type,) = struct.unpack_from("<I", buf, pos)
            (array_len,) = struct.unpack_from("<Q", buf, pos + 4)
            pos += 12
            items: List[Any] = []
            for _ in range(array_len):
                val, pos = _read_metadata_value(buf, pos, elem_type)
                items.append(val)
            meta[key] = items
        else:
            val, pos = _read_metadata_value(buf, pos, vtype)
            meta[key] = val
    return meta, pos


# ---------------------------------------------------------------------------
# float16 / bfloat16 decoding (manual, endian-explicit, no alignment issues)
# ---------------------------------------------------------------------------

def _f16_from_u8(u8: np.ndarray) -> np.ndarray:
    """(..., 2) uint8 little-endian float16 -> (...) float32."""
    u = u8[..., 0].astype(np.uint32) | (u8[..., 1].astype(np.uint32) << 8)
    return u.astype(np.uint16).view(np.float16).astype(np.float32)


def _bf16_from_u8(u8: np.ndarray) -> np.ndarray:
    """(..., 2) uint8 little-endian bfloat16 -> (...) float32."""
    u = u8[..., 0].astype(np.uint32) | (u8[..., 1].astype(np.uint32) << 8)
    return (u << 16).view(np.float32)


# ---------------------------------------------------------------------------
# dequantization (formulas from ggml-quants.c, dequantize_row_*)
# ---------------------------------------------------------------------------

def _dequantize(qtype: int, payload: bytes, n: int) -> np.ndarray:
    if qtype == GGML_TYPE_F32:
        return np.frombuffer(payload, dtype=np.float32).astype(np.float32, copy=False)
    if qtype == GGML_TYPE_F16:
        u8 = np.frombuffer(payload, dtype=np.uint8).reshape(n, 2)
        return _f16_from_u8(u8)
    if qtype == GGML_TYPE_BF16:
        u8 = np.frombuffer(payload, dtype=np.uint8).reshape(n, 2)
        return _bf16_from_u8(u8)
    if qtype == GGML_TYPE_F64:
        return np.frombuffer(payload, dtype=np.float64).astype(np.float32)

    if qtype not in _BLOCK_LAYOUTS:
        raise DracoFormatError(
            f"GGUF quantization type {qtype} is not supported by this build "
            f"(supported: F32/F16/BF16/F64 + Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/"
            f"Q8_K/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K). Try an F16 or Q8_0/Q4_K_M file."
        )

    block_size, block_bytes = _BLOCK_LAYOUTS[qtype]
    if n % block_size != 0:
        raise DracoFormatError(
            f"GGUF tensor element count {n} is not divisible by block size "
            f"{block_size} for ggml type {qtype}."
        )
    nb = n // block_size
    b = np.frombuffer(payload, dtype=np.uint8).reshape(nb, block_bytes)

    if qtype == GGML_TYPE_Q8_0:
        d = _f16_from_u8(b[:, 0:2])
        qs = b[:, 2:34].view(np.int8).astype(np.float32)
        return (qs * d[:, None]).reshape(-1)

    if qtype == GGML_TYPE_Q8_K:
        d = _f16_from_u8(b[:, 0:2])
        qs = b[:, 2:258].view(np.int8).astype(np.float32)
        return (qs * d[:, None]).reshape(-1)

    if qtype == GGML_TYPE_Q4_0:
        d = _f16_from_u8(b[:, 0:2])
        q = b[:, 2:18]
        x0 = (q & 0x0F).astype(np.int32) - 8
        x1 = (q >> 4).astype(np.int32) - 8
        vals = np.concatenate([x0, x1], axis=1).astype(np.float32) * d[:, None]
        return vals.reshape(-1)

    if qtype == GGML_TYPE_Q4_1:
        d = _f16_from_u8(b[:, 0:2])
        m = _f16_from_u8(b[:, 2:4])
        q = b[:, 4:20]
        x0 = (q & 0x0F).astype(np.float32) * d[:, None] + m[:, None]
        x1 = (q >> 4).astype(np.float32) * d[:, None] + m[:, None]
        return np.concatenate([x0, x1], axis=1).reshape(-1)

    if qtype in (GGML_TYPE_Q5_0, GGML_TYPE_Q5_1):
        # Q5_0: d(2) + qh(4) + qs(16).  Q5_1: d(2) + m(2) + qh(4) + qs(16)
        if qtype == GGML_TYPE_Q5_0:
            d = _f16_from_u8(b[:, 0:2])
            m = None
            qh = b[:, 2:6]
            q = b[:, 6:22]
        else:
            d = _f16_from_u8(b[:, 0:2])
            m = _f16_from_u8(b[:, 2:4])
            qh = b[:, 4:8]
            q = b[:, 8:24]
        qh32 = (
            qh[:, 0].astype(np.uint32)
            | (qh[:, 1].astype(np.uint32) << 8)
            | (qh[:, 2].astype(np.uint32) << 16)
            | (qh[:, 3].astype(np.uint32) << 24)
        )
        j = np.arange(16, dtype=np.uint32)
        xh0 = (((qh32[:, None] >> j[None, :]) << 4) & 0x10).astype(np.uint8)
        xh1 = ((qh32[:, None] >> (j[None, :] + 12)) & 0x10).astype(np.uint8)
        if qtype == GGML_TYPE_Q5_0:
            x0 = ((q & 0x0F) | xh0).astype(np.int32) - 16
            x1 = ((q >> 4) | xh1).astype(np.int32) - 16
            return (np.concatenate([x0, x1], axis=1).astype(np.float32) * d[:, None]).reshape(-1)
        x0 = (((q & 0x0F) | xh0).astype(np.float32)) * d[:, None] + m[:, None]
        x1 = (((q >> 4) | xh1).astype(np.float32)) * d[:, None] + m[:, None]
        return np.concatenate([x0, x1], axis=1).reshape(-1)

    if qtype == GGML_TYPE_Q2_K:
        d = _f16_from_u8(b[:, 0:2])
        dmin = _f16_from_u8(b[:, 2:4])
        qs = b[:, 4:68]
        sc = b[:, 68:84]
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
        qs_h = qs.reshape(nb, 2, 32)
        sc_h = sc.reshape(nb, 2, 8)
        low = ((qs_h[:, :, None, 0:16] >> shifts[None, None, :, None]) & 3).astype(np.int32)
        high = ((qs_h[:, :, None, 16:32] >> shifts[None, None, :, None]) & 3).astype(np.int32)
        sc_low = sc_h[:, :, 0::2]
        sc_high = sc_h[:, :, 1::2]
        dl_low = d[:, None, None] * (sc_low & 0x0F).astype(np.float32)
        ml_low = dmin[:, None, None] * (sc_low >> 4).astype(np.float32)
        dl_high = d[:, None, None] * (sc_high & 0x0F).astype(np.float32)
        ml_high = dmin[:, None, None] * (sc_high >> 4).astype(np.float32)
        v_low = dl_low[..., None] * low - ml_low[..., None]
        v_high = dl_high[..., None] * high - ml_high[..., None]
        vals = np.concatenate([v_low, v_high], axis=-1).reshape(nb, 2, 128)
        return vals.reshape(-1)

    if qtype == GGML_TYPE_Q3_K:
        d = _f16_from_u8(b[:, 0:2])
        s12 = b[:, 2:14]
        qs = b[:, 14:78]
        hm = b[:, 78:110]
        aux = np.empty((nb, 3), dtype=np.uint32)
        for k in range(3):
            o = k * 4
            aux[:, k] = (
                s12[:, o].astype(np.uint32)
                | (s12[:, o + 1].astype(np.uint32) << 8)
                | (s12[:, o + 2].astype(np.uint32) << 16)
                | (s12[:, o + 3].astype(np.uint32) << 24)
            )
        kmask1 = np.uint32(0x03030303)
        kmask2 = np.uint32(0x0F0F0F0F)
        tmp = aux[:, 2]
        a0 = (aux[:, 0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        a1 = (aux[:, 1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        a2 = ((aux[:, 0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        a3 = ((aux[:, 1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        stacked = np.stack([a0, a1, a2, a3], axis=1)  # (nb, 4) uint32
        scales = stacked.view(np.uint8)[:, 0:16].view(np.int8).astype(np.int32)

        out = np.empty((nb, 256), dtype=np.float32)
        hm_low = hm[:, 0:16]
        hm_high = hm[:, 16:32]
        mask = np.zeros((nb, 256), dtype=np.int32)
        for n in range(2):
            for j in range(4):
                bit = n * 4 + j
                sel_low = ((hm_low >> bit) & 1).astype(np.int32)
                sel_high = ((hm_high >> bit) & 1).astype(np.int32)
                base = n * 128 + j * 32
                mask[:, base : base + 16] = sel_low
                mask[:, base + 16 : base + 32] = sel_high
        for n in range(2):
            qh_ = qs[:, n * 32 : n * 32 + 32]
            for j in range(4):
                shift = 2 * j
                dl1 = d * (scales[:, n * 8 + 2 * j] - 32).astype(np.float32)
                dl2 = d * (scales[:, n * 8 + 2 * j + 1] - 32).astype(np.float32)
                qv1 = ((qh_[:, 0:16] >> shift) & 3).astype(np.int32)
                qv2 = ((qh_[:, 16:32] >> shift) & 3).astype(np.int32)
                base = n * 128 + j * 32
                out[:, base : base + 16] = dl1[:, None] * (qv1 - 4 * (1 - mask[:, base : base + 16]))
                out[:, base + 16 : base + 32] = dl2[:, None] * (qv2 - 4 * (1 - mask[:, base + 16 : base + 32]))
        return out.reshape(-1)

    if qtype in (GGML_TYPE_Q4_K, GGML_TYPE_Q5_K):
        d = _f16_from_u8(b[:, 0:2])
        dmin = _f16_from_u8(b[:, 2:4])
        sc = b[:, 4:16]
        if qtype == GGML_TYPE_Q4_K:
            qh = None
            qs = b[:, 16:144]
        else:
            qh = b[:, 16:48]
            qs = b[:, 48:176]

        def _scale_min(jj: int) -> Tuple[np.ndarray, np.ndarray]:
            if jj < 4:
                return sc[:, jj] & 63, sc[:, jj + 4] & 63
            return (
                (sc[:, jj + 4] & 0x0F) | ((sc[:, jj - 4] >> 6) << 4),
                (sc[:, jj + 4] >> 4) | ((sc[:, jj] >> 6) << 4),
            )

        out = np.empty((nb, 256), dtype=np.float32)
        for jj in range(4):
            sc0, m0 = _scale_min(2 * jj)
            sc1, m1 = _scale_min(2 * jj + 1)
            d1 = d * sc0.astype(np.float32)
            mm1 = dmin * m0.astype(np.float32)
            d2 = d * sc1.astype(np.float32)
            mm2 = dmin * m1.astype(np.float32)
            q = qs[:, jj * 32 : jj * 32 + 32]
            base = jj * 64
            if qtype == GGML_TYPE_Q4_K:
                out[:, base : base + 32] = d1[:, None] * (q & 0x0F).astype(np.float32) - mm1[:, None]
                out[:, base + 32 : base + 64] = d2[:, None] * (q >> 4).astype(np.float32) - mm2[:, None]
            else:
                u1 = np.uint8(1 << (2 * jj))
                u2 = np.uint8(1 << (2 * jj + 1))
                v1 = d1[:, None] * (
                    (q & 0x0F).astype(np.int32) + np.where((qh & u1) != 0, 16, 0)
                ) - mm1[:, None]
                v2 = d2[:, None] * (
                    (q >> 4).astype(np.int32) + np.where((qh & u2) != 0, 16, 0)
                ) - mm2[:, None]
                out[:, base : base + 32] = v1
                out[:, base + 32 : base + 64] = v2
        return out.reshape(-1)

    if qtype == GGML_TYPE_Q6_K:
        d = _f16_from_u8(b[:, 0:2])
        ql = b[:, 2:130]
        qh = b[:, 130:194]
        sc = b[:, 194:210].view(np.int8).astype(np.int32)
        out = np.empty((nb, 256), dtype=np.float32)
        for n in range(2):
            ql_ = ql[:, n * 64 : n * 64 + 64]
            qh_ = qh[:, n * 32 : n * 32 + 32]
            sc_ = sc[:, n * 8 : n * 8 + 8]
            for l in range(32):
                is_ = l // 16
                q1 = ((ql_[:, l] & 0x0F) | (((qh_[:, l] >> 0) & 3) << 4)).astype(np.int32) - 32
                q2 = ((ql_[:, l + 32] & 0x0F) | (((qh_[:, l] >> 2) & 3) << 4)).astype(np.int32) - 32
                q3 = ((ql_[:, l] >> 4) | (((qh_[:, l] >> 4) & 3) << 4)).astype(np.int32) - 32
                q4 = ((ql_[:, l + 32] >> 4) | (((qh_[:, l] >> 6) & 3) << 4)).astype(np.int32) - 32
                dl0 = d * sc_[:, is_ + 0].astype(np.float32)
                dl1 = d * sc_[:, is_ + 2].astype(np.float32)
                dl2 = d * sc_[:, is_ + 4].astype(np.float32)
                dl3 = d * sc_[:, is_ + 6].astype(np.float32)
                base = n * 128 + l
                out[:, base] = dl0 * q1
                out[:, base + 32] = dl1 * q2
                out[:, base + 64] = dl2 * q3
                out[:, base + 96] = dl3 * q4
        return out.reshape(-1)

    raise DracoFormatError(f"GGUF quantization type {qtype} is not supported.")


# ---------------------------------------------------------------------------
# tensor info + reader
# ---------------------------------------------------------------------------

@dataclass
class GgufTensorInfo:
    """Duck-typed to be a drop-in for DracoTensorInfo in the CPU executor."""

    name: str
    ggml_type: int
    shape: Tuple[int, ...]
    offset: int  # absolute file offset
    nbytes: int
    alignment: int = 32
    dtype: str = "f32"
    quantization: Optional[str] = None  # always None: GGUF weights -> float32
    group_size: Optional[int] = None
    scale_offset: Optional[int] = None
    scale_nbytes: Optional[int] = None
    zero_point_offset: Optional[int] = None
    zero_point_nbytes: Optional[int] = None


class _GgufHeader:
    def __init__(self, version: int) -> None:
        self.version = version


class GGUFReader:
    """Memory-mapped reader for .gguf files with a DracoReader-like API."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, "rb")
        try:
            size = os.fstat(self._file.fileno()).st_size
            if size < 24:
                raise DracoFormatError(f"'{path}' is too small to be a valid GGUF file.")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._file_size = size

            head = bytes(self._mmap[:24])
            magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", head)
            if magic != GGUF_MAGIC:
                raise DracoFormatError(
                    f"Not a valid GGUF file: bad magic bytes {magic!r} in '{path}'."
                )
            if version not in (1, 2, 3):
                raise DracoFormatError(
                    f"Unsupported GGUF format version {version} in '{path}' (supported: 1-3)."
                )

            meta_bytes = bytes(self._mmap[24:])
            self.metadata_raw, meta_end = _read_metadata(meta_bytes, 0, kv_count)

            alignment = int(self.metadata_raw.get("general.alignment", 32))
            if alignment <= 0 or alignment % 8 != 0:
                raise DracoFormatError(
                    f"Invalid GGUF alignment {alignment} in '{path}' (must be a positive multiple of 8)."
                )

            # --- tensor infos (they start right after the metadata KV block) ---
            pos = meta_end
            self._tensor_infos_raw: Dict[str, Tuple[Tuple[int, ...], int, int]] = {}
            for _ in range(tensor_count):
                name, pos = _read_string(meta_bytes, pos)
                (n_dims,) = struct.unpack_from("<I", meta_bytes, pos)
                pos += 4
                dims = struct.unpack_from(f"<{n_dims}Q", meta_bytes, pos)
                pos += 8 * n_dims
                (ttype,) = struct.unpack_from("<I", meta_bytes, pos)
                pos += 4
                (toffset,) = struct.unpack_from("<Q", meta_bytes, pos)
                pos += 8
                self._tensor_infos_raw[name] = (dims, ttype, toffset)

            data_start = 24 + pos + ((alignment - (24 + pos) % alignment) % alignment)
            self._data_start = data_start
            self.alignment = alignment

            self._build_index()
            self._build_metadata()
            self._build_tokenizer()
            self._cache: Dict[str, np.ndarray] = {}
            self.header = _GgufHeader(version)
        except Exception:
            self._close_files()
            raise

    def _close_files(self) -> None:
        try:
            if hasattr(self, "_mmap"):
                self._mmap.close()
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # building blocks
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        arch = self.metadata_raw.get("general.architecture", "")
        family = _ARCH_FAMILY.get(arch)
        if family is None:
            raise DracoFormatError(
                f"GGUF architecture '{arch}' is not supported by the Draco CPU "
                f"executor. Supported: {sorted(_ARCH_FAMILY)} (llama-family models "
                f"with standard blk.* tensor names, and gpt2)."
            )
        self._family = family
        self.architecture_name = arch

        has_expert_tensors = any("_exps." in n for n in self._tensor_infos_raw)
        if has_expert_tensors:
            raise DracoFormatError(
                f"GGUF '{self.path}' contains MoE expert tensors; the Draco CPU "
                f"executor does not support MoE models yet."
            )

        index: Dict[str, GgufTensorInfo] = {}
        for name, (dims, ttype, rel_offset) in self._tensor_infos_raw.items():
            canonical = _canonical_name(name, family)
            if canonical in index:
                raise DracoFormatError(
                    f"Duplicate canonical tensor name '{canonical}' (from GGUF '{name}')."
                )
            n_elements = 1
            for d in dims:
                n_elements *= d
            if ttype in (GGML_TYPE_F32, GGML_TYPE_F64):
                elem_bytes = 4 if ttype == GGML_TYPE_F32 else 8
                nbytes = n_elements * elem_bytes
            elif ttype == GGML_TYPE_F16 or ttype == GGML_TYPE_BF16:
                nbytes = n_elements * 2
            else:
                block_size, block_bytes = _BLOCK_LAYOUTS[ttype]
                if n_elements % block_size != 0:
                    raise DracoFormatError(
                        f"GGUF tensor '{name}' has {n_elements} elements, not a "
                        f"multiple of block size {block_size} for ggml type {ttype}."
                    )
                nbytes = (n_elements // block_size) * block_bytes
            # dims are stored in reverse order relative to the tensor's natural
            # (row-major) shape; numpy shape = dims[::-1]
            shape = tuple(reversed(dims))
            offset = self._data_start + rel_offset
            if offset + nbytes > self._file_size:
                raise DracoFormatError(
                    f"GGUF tensor '{name}' extends past end of file ({offset}+{nbytes} > {self._file_size})."
                )
            index[canonical] = GgufTensorInfo(
                name=canonical,
                ggml_type=ttype,
                shape=shape,
                offset=offset,
                nbytes=nbytes,
                alignment=self.alignment,
            )
        self._index = index

    def _build_metadata(self) -> None:
        raw = self.metadata_raw
        arch = self.architecture_name
        family = self._family
        p = arch + "."

        def get_int(key: str, default: Optional[int] = None) -> Optional[int]:
            v = raw.get(p + key)
            if v is None:
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        def get_float(key: str, default: Optional[float] = None) -> Optional[float]:
            v = raw.get(p + key)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def get_bool(key: str, default: bool) -> bool:
            v = raw.get(p + key)
            if v is None:
                return default
            return bool(v)

        hidden = get_int("embedding_length")
        layers = get_int("block_count")
        heads = get_int("attention.head_count")
        kv_heads = get_int("attention.head_count_kv") or heads
        vocab = get_int("vocab_size")
        context = get_int("context_length") or 4096
        rope_theta = get_float("rope.freq_base", 10000.0)
        rms_eps = get_float("attention.layer_norm_rms_epsilon") or get_float(
            "attention.layer_norm_epsilon"
        ) or 1e-5

        key_len = get_int("attention.key_length")
        if key_len is not None and key_len != hidden // heads:
            raise DracoFormatError(
                f"GGUF '{self.path}' uses a non-standard key_length {key_len} "
                f"(head_dim={hidden // heads}); the Draco CPU executor requires "
                f"head_dim == hidden_size // head_count."
            )

        if vocab is None:
            emb = self._index.get("model.embed_tokens.weight")
            if emb is not None:
                vocab = emb.shape[0]

        if None in (hidden, layers, heads, vocab):
            raise DracoFormatError(
                f"GGUF '{self.path}' is missing required architecture metadata "
                f"(embedding_length / block_count / attention.head_count / vocab_size)."
            )

        has_output = "model.lm_head.weight" in self._index
        tie = get_bool("tie_embeddings", not has_output)

        if family == "gpt2":
            ln_eps = get_float("attention.layer_norm_epsilon", 1e-5)
            metadata = {
                "architecture": "GPT2LMHeadModel",
                "model_type": "gpt2",
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": heads,
                "vocab_size": vocab,
                "max_position_embeddings": context,
                "tie_word_embeddings": tie,
                "layer_norm_eps": ln_eps,
                "norm_type": "layernorm",
                "norm_bias": True,
                "position_encoding_type": "learned",
                "mlp_type": "standard",
                "hidden_act": "gelu",
                "qkv_fused": True,
                "attention_bias": True,
                "mlp_bias": True,
            }
        else:
            metadata = {
                "architecture": _llama_hf_architecture(arch),
                "model_type": arch,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "vocab_size": vocab,
                "max_position_embeddings": context,
                "rope_theta": rope_theta,
                "rms_norm_eps": rms_eps,
                "tie_word_embeddings": tie,
                "norm_type": "rmsnorm",
                "norm_bias": False,
                "position_encoding_type": "rope",
                "mlp_type": "gated",
                "hidden_act": "silu",
                "qkv_fused": "model.layers.0.self_attn.qkv_proj.weight" in self._index,
                "attention_bias": "model.layers.0.self_attn.q_proj.bias" in self._index,
                "mlp_bias": "model.layers.0.mlp.gate_proj.bias" in self._index,
            }
        metadata["gguf_architecture"] = arch
        self.metadata = metadata

    def _build_tokenizer(self) -> None:
        raw = self.metadata_raw
        if "tokenizer.ggml.tokens" not in raw:
            self.tokenizer = None
            return
        tokenizer: Dict[str, Any] = {
            "model": raw.get("tokenizer.ggml.model"),
            "tokens": raw.get("tokenizer.ggml.tokens"),
            "scores": raw.get("tokenizer.ggml.scores"),
            "token_type": raw.get("tokenizer.ggml.token_type"),
            "merges": raw.get("tokenizer.ggml.merges"),
            "bos_token_id": raw.get("tokenizer.ggml.bos_token_id"),
            "eos_token_id": raw.get("tokenizer.ggml.eos_token_id"),
            "unknown_token_id": raw.get("tokenizer.ggml.unknown_token_id"),
            "padding_token_id": raw.get("tokenizer.ggml.padding_token_id"),
            "chat_template": raw.get("tokenizer.chat_template"),
        }
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # DracoReader-compatible accessors
    # ------------------------------------------------------------------

    @property
    def architecture(self) -> str:
        return self.metadata.get("architecture", "")

    def tensor_names(self) -> List[str]:
        return list(self._index.keys())

    def tensors(self) -> Iterator[GgufTensorInfo]:
        return iter(self._index.values())

    def tensor_info(self, name: str) -> GgufTensorInfo:
        if name not in self._index:
            raise KeyError(f"No tensor named '{name}' in '{self.path}'.")
        return self._index[name]

    def _raw_bytes(self, offset: int, nbytes: int) -> bytes:
        if offset + nbytes > self._file_size:
            raise DracoFormatError(
                f"Tensor region [{offset}:{offset + nbytes}] extends past end of "
                f"file ({self._file_size}) in '{self.path}'."
            )
        return bytes(self._mmap[offset : offset + nbytes])

    def get_tensor(self, name: str, dequantize: bool = True) -> np.ndarray:
        """Return the tensor dequantized to float32 (cached).

        GGUF weights are always surfaced as float32 — there are no
        Draco-native block-format kernels to hand raw payloads to, so
        ``dequantize=False`` returns the same float32 array.
        """
        if name in self._cache:
            return self._cache[name]
        info = self.tensor_info(name)
        payload = self._raw_bytes(info.offset, info.nbytes)
        n_elements = int(np.prod(info.shape))
        arr = _dequantize(info.ggml_type, payload, n_elements).reshape(info.shape)
        self._cache[name] = arr
        return arr

    def close(self) -> None:
        self._cache.clear()
        self._close_files()

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# name / architecture mapping helpers
# ---------------------------------------------------------------------------

def _canonical_name(gguf_name: str, family: str) -> str:
    if family == "gpt2":
        if gguf_name in _GPT2_TENSOR_MAP:
            return _GPT2_TENSOR_MAP[gguf_name]
        m = _BLOCK_RE.match(gguf_name)
        if m:
            layer, rest = int(m.group(1)), m.group(2)
            if rest in _GPT2_BLOCK_MAP:
                return f"model.h.{layer}.{_GPT2_BLOCK_MAP[rest]}"
        raise DracoFormatError(
            f"Unsupported GPT-2 GGUF tensor name '{gguf_name}'."
        )

    if gguf_name in _LLAMA_TENSOR_MAP:
        return _LLAMA_TENSOR_MAP[gguf_name]
    m = _BLOCK_RE.match(gguf_name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        if rest in _LLAMA_BLOCK_MAP:
            return f"model.layers.{layer}.{_LLAMA_BLOCK_MAP[rest]}"
    raise DracoFormatError(
        f"Unsupported GGUF tensor name '{gguf_name}' for llama-family "
        f"architecture. Supported: {sorted(_LLAMA_TENSOR_MAP)} and blk.N.{{"
        f"{', '.join(sorted(_LLAMA_BLOCK_MAP))}}}."
    )


def _llama_hf_architecture(gguf_arch: str) -> str:
    return {
        "llama": "LlamaForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
        "mistral": "MistralForCausalLM",
        "phi3": "Phi3ForCausalLM",
        "yi": "YiForCausalLM",
        "starcoder2": "Starcoder2ForCausalLM",
    }.get(gguf_arch, "LlamaForCausalLM")