"""
GGUF reader tests.

The dequantization tests compare the vectorized numpy implementation in
draco.format.gguf against small, independently-written reference decoders
that mirror the llama.cpp (ggml-quants.c) formulas with plain loops — so a
transcription error in the production code shows up as a mismatch.
"""

import os
import struct

import numpy as np
import pytest

from draco.format.gguf import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_K,
    GGUFReader,
    _dequantize,
)
from draco.format.gguf_tokenizer import GGUFBPETokenizer, bytes_to_unicode

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


# ---------------------------------------------------------------------------
# a small GGUF writer for tests
# ---------------------------------------------------------------------------

def _w_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _w_meta(key: str, val) -> bytes:
    out = bytearray(_w_string(key))
    if isinstance(val, bool):
        out += struct.pack("<IB", _VAL_BOOL, int(val))
    elif isinstance(val, int):
        out += struct.pack("<IQ", _VAL_UINT64, val)
    elif isinstance(val, float):
        out += struct.pack("<If", _VAL_FLOAT32, val)
    elif isinstance(val, str):
        out += struct.pack("<I", _VAL_STRING) + _w_string(val)
    elif isinstance(val, (list, tuple)):
        if val and isinstance(val[0], str):
            out += struct.pack("<IIQ", _VAL_ARRAY, _VAL_STRING, len(val))
            for v in val:
                out += _w_string(v)
        elif val and isinstance(val[0], float):
            out += struct.pack("<IIQ", _VAL_ARRAY, _VAL_FLOAT32, len(val))
            for v in val:
                out += struct.pack("<f", v)
        else:
            out += struct.pack("<IIQ", _VAL_ARRAY, _VAL_INT32, len(val))
            for v in val:
                out += struct.pack("<i", v)
    else:
        raise TypeError(f"unsupported metadata value: {val!r}")
    return bytes(out)


def _normalize_spec(spec):
    """(name, np_array) or (name, payload_bytes, ggml_type, shape)."""
    if len(spec) == 4:
        name, data, gtype, shape = spec
        return name, data, gtype, tuple(int(d) for d in shape)
    name, arr = spec
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.float32:
        gtype = GGML_TYPE_F32
    elif arr.dtype == np.float16:
        gtype = GGML_TYPE_F16
    else:
        arr = arr.astype(np.float32)
        gtype = GGML_TYPE_F32
    return name, arr, gtype, tuple(arr.shape)


def write_gguf(path, metadata, tensor_specs, alignment: int = 32):
    infos = []
    meta_bytes = b"".join(_w_meta(k, v) for k, v in metadata.items())
    pos_after_infos = 24 + len(meta_bytes)
    for spec in tensor_specs:
        name, data, gtype, shape = _normalize_spec(spec)
        pos_after_infos += len(_w_string(name)) + 4 + 8 * len(shape) + 4 + 8
    data_start = pos_after_infos + ((alignment - pos_after_infos % alignment) % alignment)

    off = 0
    for spec in tensor_specs:
        name, data, gtype, shape = _normalize_spec(spec)
        payload = data if isinstance(data, bytes) else data.tobytes()
        aligned = off + ((alignment - off % alignment) % alignment)
        infos.append((name, shape, gtype, aligned, payload))
        off = aligned + len(payload)

    buf = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensor_specs), len(metadata)))
    buf += meta_bytes
    for name, shape, gtype, offset, _ in infos:
        buf += _w_string(name)
        buf += struct.pack("<I", len(shape))
        for d in reversed(shape):
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", gtype, offset)
    while len(buf) < data_start:
        buf += b"\x00"
    for name, shape, gtype, offset, payload in infos:
        while len(buf) % alignment:
            buf += b"\x00"
        buf += payload
    with open(path, "wb") as f:
        f.write(bytes(buf))
    return path


# ---------------------------------------------------------------------------
# encoders for the simple block quant formats (used only by the tests)
# ---------------------------------------------------------------------------

def _enc_q8_0(arr: np.ndarray) -> bytes:
    n = len(arr)
    nb = n // 32
    a = arr.reshape(nb, 32)
    amax = np.max(np.abs(a), axis=1)
    d = np.maximum(amax, 1e-12) / 127.0
    qs = np.clip(np.rint(a / d[:, None]), -127, 127).astype(np.int8)
    out = np.empty((nb, 34), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:34] = qs.view(np.uint8)
    return out.tobytes()


def _enc_q4_0(arr: np.ndarray) -> bytes:
    n = len(arr)
    nb = n // 32
    a = arr.reshape(nb, 32)
    amax = np.max(np.abs(a), axis=1)
    d = np.maximum(amax, 1e-12) / 8.0
    q = np.clip(np.rint(a / d[:, None]) + 8, 0, 15).astype(np.uint8)
    out = np.empty((nb, 18), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    lo, hi = q[:, 0:16], q[:, 16:32]
    out[:, 2:18] = lo | (hi << 4)
    return out.tobytes()


def _enc_q4_1(arr: np.ndarray) -> bytes:
    n = len(arr)
    nb = n // 32
    a = arr.reshape(nb, 32)
    amin = np.min(a, axis=1)
    span = np.max(a, axis=1) - amin
    d = np.where(span > 1e-12, span / 15.0, 1.0)
    q = np.clip(np.rint((a - amin[:, None]) / d[:, None]), 0, 15).astype(np.uint8)
    out = np.empty((nb, 20), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:4] = amin.astype(np.float16).view(np.uint8).reshape(nb, 2)
    lo, hi = q[:, 0:16], q[:, 16:32]
    out[:, 4:20] = lo | (hi << 4)
    return out.tobytes()


def _enc_q5_0(arr: np.ndarray) -> bytes:
    n = len(arr)
    nb = n // 32
    a = arr.reshape(nb, 32)
    amax = np.max(np.abs(a), axis=1)
    d = np.maximum(amax, 1e-12) / 16.0
    q = np.clip(np.rint(a / d[:, None]) + 16, 0, 31).astype(np.uint8)
    qh = np.zeros(nb, dtype=np.uint32)
    for j in range(16):
        qh |= ((q[:, j] >> 4).astype(np.uint32) & 1) << j
        qh |= ((q[:, 16 + j] >> 4).astype(np.uint32) & 1) << (j + 16)
    out = np.empty((nb, 22), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:6] = qh.view(np.uint8).reshape(nb, 4)
    lo, hi = q[:, 0:16], q[:, 16:32]
    out[:, 6:22] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return out.tobytes()


def _enc_q5_1(arr: np.ndarray) -> bytes:
    n = len(arr)
    nb = n // 32
    a = arr.reshape(nb, 32)
    amin = np.min(a, axis=1)
    span = np.max(a, axis=1) - amin
    d = np.where(span > 1e-12, span / 31.0, 1.0)
    q = np.clip(np.rint((a - amin[:, None]) / d[:, None]), 0, 31).astype(np.uint8)
    qh = np.zeros(nb, dtype=np.uint32)
    for j in range(16):
        qh |= ((q[:, j] >> 4).astype(np.uint32) & 1) << j
        qh |= ((q[:, 16 + j] >> 4).astype(np.uint32) & 1) << (j + 16)
    out = np.empty((nb, 24), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:4] = amin.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 4:8] = qh.view(np.uint8).reshape(nb, 4)
    lo, hi = q[:, 0:16], q[:, 16:32]
    out[:, 8:24] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return out.tobytes()


# ---------------------------------------------------------------------------
# reference decoders for the K-quants (plain loops, from ggml-quants.c)
# ---------------------------------------------------------------------------

def _ref_q2_k(d, dmin, qs, sc):
    y = []
    is_ = 0
    for half in range(2):
        shift = 0
        for j in range(4):
            s = sc[is_]
            dl, ml = d * (s & 0xF), dmin * (s >> 4)
            is_ += 1
            for l in range(16):
                y.append(dl * ((qs[half * 32 + l] >> shift) & 3) - ml)
            s = sc[is_]
            dl, ml = d * (s & 0xF), dmin * (s >> 4)
            is_ += 1
            for l in range(16):
                y.append(dl * ((qs[half * 32 + 16 + l] >> shift) & 3) - ml)
            shift += 2
    return np.asarray(y, dtype=np.float32)


def _ref_q3_k(d, scales12, qs, hm):
    aux = np.frombuffer(bytes(scales12), dtype=np.uint32)
    tmp = int(aux[2])
    a0 = (int(aux[0]) & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4)
    a1 = (int(aux[1]) & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4)
    a2 = ((int(aux[0]) >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4)
    a3 = ((int(aux[1]) >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4)
    sc = np.frombuffer(struct.pack("<4I", a0, a1, a2, a3), dtype=np.uint8)[:16].view(np.int8)
    y = []
    is_ = 0
    m = 1
    for half in range(2):
        shift = 0
        for j in range(4):
            dl = d * (int(sc[is_]) - 32)
            is_ += 1
            for l in range(16):
                qv = (qs[half * 32 + l] >> shift) & 3
                y.append(dl * (qv - (0 if (hm[l] & m) else 4)))
            dl = d * (int(sc[is_]) - 32)
            is_ += 1
            for l in range(16):
                qv = (qs[half * 32 + 16 + l] >> shift) & 3
                y.append(dl * (qv - (0 if (hm[16 + l] & m) else 4)))
            shift += 2
            m <<= 1
    return np.asarray(y, dtype=np.float32)


def _gsm(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _ref_q4_k(d, dmin, scales12, qs):
    y = []
    is_ = 0
    for g in range(4):
        sc0, m0 = _gsm(is_, scales12)
        d1, m1 = d * sc0, dmin * m0
        sc1, m1b = _gsm(is_ + 1, scales12)
        d2, m2 = d * sc1, dmin * m1b
        q = qs[g * 32 : g * 32 + 32]
        for l in range(32):
            y.append(d1 * (q[l] & 0xF) - m1)
        for l in range(32):
            y.append(d2 * (q[l] >> 4) - m2)
        is_ += 2
    return np.asarray(y, dtype=np.float32)


def _ref_q5_k(d, dmin, scales12, qh, ql):
    y = []
    is_ = 0
    u1, u2 = 1, 2
    for g in range(4):
        sc0, m0 = _gsm(is_, scales12)
        d1, m1 = d * sc0, dmin * m0
        sc1, m1b = _gsm(is_ + 1, scales12)
        d2, m2 = d * sc1, dmin * m1b
        q = ql[g * 32 : g * 32 + 32]
        for l in range(32):
            y.append(d1 * ((q[l] & 0xF) + (16 if (qh[l] & u1) else 0)) - m1)
        for l in range(32):
            y.append(d2 * ((q[l] >> 4) + (16 if (qh[l] & u2) else 0)) - m2)
        is_ += 2
        u1 <<= 2
        u2 <<= 2
    return np.asarray(y, dtype=np.float32)


def _ref_q6_k(d, ql, qh, sc):
    y = []
    for half in range(2):
        for l in range(32):
            is_ = l // 16
            q1 = ((ql[half * 64 + l] & 0xF) | (((qh[half * 32 + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[half * 64 + l + 32] & 0xF) | (((qh[half * 32 + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[half * 64 + l] >> 4) | (((qh[half * 32 + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[half * 64 + l + 32] >> 4) | (((qh[half * 32 + l] >> 6) & 3) << 4)) - 32
            y.append(d * sc[is_ + 0] * q1)
            y.append(d * sc[is_ + 2] * q2)
            y.append(d * sc[is_ + 4] * q3)
            y.append(d * sc[is_ + 6] * q4)
    return np.asarray(y, dtype=np.float32)


def _ref_q8_k(d, qs):
    return (np.asarray(qs, dtype=np.int8).astype(np.float32) * d)


# ---------------------------------------------------------------------------
# parser tests
# ---------------------------------------------------------------------------

def test_header_and_metadata_parsing(tmp_path):
    path = write_gguf(
        str(tmp_path / "meta.gguf"),
        {
            "general.architecture": "qwen2",
            "general.name": "tiny",
            "qwen2.embedding_length": 8,
            "qwen2.block_count": 1,
            "qwen2.context_length": 4096,
            "qwen2.attention.head_count": 32,
            "qwen2.vocab_size": 10,
            "general.alignment": 32,
            "general.file_type": 1,
        },
        [("token_embd.weight", np.zeros((10, 8), dtype=np.float32))],
    )
    with GGUFReader(path) as reader:
            assert reader.header.version == 3
            assert reader.metadata_raw["general.architecture"] == "qwen2"
            assert reader.metadata_raw["qwen2.context_length"] == 4096
            assert reader.metadata_raw["general.file_type"] == 1
            # raw GGUF name is preserved; tensor_names() returns canonical HF names
            assert "token_embd.weight" in reader._tensor_infos_raw
            assert "model.embed_tokens.weight" in reader.tensor_names()
            info = reader.tensor_info("model.embed_tokens.weight")
            assert info.shape == (10, 8)


def test_float_tensors_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    f32 = (rng.normal(size=64) * 3).astype(np.float32)
    f16 = (rng.normal(size=48) * 2).astype(np.float16)
    bf16_bytes = (
        ((f16.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)).tobytes()
    )
    path = write_gguf(
        str(tmp_path / "floats.gguf"),
        {
            "general.architecture": "llama",
            "llama.embedding_length": 8,
            "llama.block_count": 1,
            "llama.attention.head_count": 8,
            "llama.vocab_size": 8,
            "llama.context_length": 64,
        },
        [
            ("token_embd.weight", f32.reshape(8, 8)),
            ("blk.0.attn_q.weight", f16.reshape(6, 8)),
            ("blk.0.attn_k.weight", bf16_bytes, GGML_TYPE_BF16, (6, 8)),
        ],
    )
    with GGUFReader(path) as reader:
        got_f32 = reader.get_tensor("model.embed_tokens.weight")
        assert np.allclose(got_f32, f32.reshape(8, 8), atol=1e-6)
        got_f16 = reader.get_tensor("model.layers.0.self_attn.q_proj.weight")
        assert np.allclose(got_f16, f16.reshape(6, 8), atol=1e-3)
        got_bf16 = reader.get_tensor("model.layers.0.self_attn.k_proj.weight")
        assert np.allclose(got_bf16, f16.reshape(6, 8), atol=1e-2)


# ---------------------------------------------------------------------------
# dequantization tests (against the plain-loop references)
# ---------------------------------------------------------------------------

def test_simple_block_quants_roundtrip():
    rng = np.random.default_rng(3)
    x = (rng.normal(size=64) * 2.0).astype(np.float32)
    # tolerances account for the naive test encoders saturating at the
    # representation limits (amax maps onto the top step); a nibble/bit
    # layout bug would produce errors of several units instead
    cases = [
        (GGML_TYPE_Q8_0, _enc_q8_0, 0.05),
        (GGML_TYPE_Q4_0, _enc_q4_0, 1.0),
        (GGML_TYPE_Q4_1, _enc_q4_1, 0.6),
        (GGML_TYPE_Q5_0, _enc_q5_0, 0.6),
        (GGML_TYPE_Q5_1, _enc_q5_1, 0.4),
    ]
    for gtype, enc, tol in cases:
        payload = enc(x)
        y = _dequantize(gtype, payload, len(x))
        assert y.shape == (len(x),)
        err = np.max(np.abs(y - x))
        assert err < tol, f"ggml type {gtype}: max err {err} >= {tol}"


def test_k_quants_match_reference_decoders():
    rng = np.random.default_rng(11)
    # Q2_K block: d(2) + dmin(2) + qs(64) + scales(16)
    d = 1.0
    dmin = 0.5
    qs = rng.integers(0, 256, 64, dtype=np.uint8)
    scales = np.full(16, 0x11, dtype=np.uint8)
    payload = (
        np.float16(d).view(np.uint16).tobytes()
        + np.float16(dmin).view(np.uint16).tobytes()
        + qs.tobytes()
        + scales.tobytes()
    )
    got = _dequantize(GGML_TYPE_Q2_K, payload, 256)
    exp = _ref_q2_k(d, dmin, qs, scales)
    assert np.array_equal(got, exp)

    # Q3_K block: d(2) + scales(12) + qs(64) + hmask(32)
    d = 2.0
    qs = rng.integers(0, 256, 64, dtype=np.uint8)
    scales12 = np.full(12, 0x21, dtype=np.uint8)
    hm = np.zeros(32, dtype=np.uint8)
    hm[5] = 0x01  # one bit set so the -4 adjustment is exercised
    payload = (
        np.float16(d).view(np.uint16).tobytes()
        + scales12.tobytes()
        + qs.tobytes()
        + hm.tobytes()
    )
    got = _dequantize(GGML_TYPE_Q3_K, payload, 256)
    exp = _ref_q3_k(d, scales12, qs, hm)
    assert np.array_equal(got, exp)

    # Q4_K block: d(2) + dmin(2) + scales(12) + qs(128)
    d, dmin = 1.0, 1.0
    qs = rng.integers(0, 256, 128, dtype=np.uint8)
    scales12 = np.full(12, 0x01, dtype=np.uint8)
    payload = (
        np.float16(d).view(np.uint16).tobytes()
        + np.float16(dmin).view(np.uint16).tobytes()
        + scales12.tobytes()
        + qs.tobytes()
    )
    got = _dequantize(GGML_TYPE_Q4_K, payload, 256)
    exp = _ref_q4_k(d, dmin, scales12, qs)
    assert np.array_equal(got, exp)

    # Q5_K block: d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
    qh = rng.integers(0, 256, 32, dtype=np.uint8)
    ql = rng.integers(0, 256, 128, dtype=np.uint8)
    payload = (
        np.float16(d).view(np.uint16).tobytes()
        + np.float16(dmin).view(np.uint16).tobytes()
        + scales12.tobytes()
        + qh.tobytes()
        + ql.tobytes()
    )
    got = _dequantize(GGML_TYPE_Q5_K, payload, 256)
    exp = _ref_q5_k(d, dmin, scales12, qh, ql)
    assert np.array_equal(got, exp)

    # Q6_K block: d(2) + ql(128) + qh(64) + scales(16, int8)
    d = 1.5
    ql = rng.integers(0, 256, 128, dtype=np.uint8)
    qh = rng.integers(0, 256, 64, dtype=np.uint8)
    sc = np.full(16, 2, dtype=np.int8)
    payload = (
        np.float16(d).view(np.uint16).tobytes()
        + ql.tobytes()
        + qh.tobytes()
        + sc.tobytes()
    )
    got = _dequantize(GGML_TYPE_Q6_K, payload, 256)
    exp = _ref_q6_k(d, ql, qh, sc)
    assert np.array_equal(got, exp)

    # Q8_K block: d(2) + qs(256, int8)
    d = 0.25
    qs = rng.integers(-128, 128, 256, dtype=np.int8)
    payload = np.float16(d).view(np.uint16).tobytes() + qs.tobytes()
    got = _dequantize(GGML_TYPE_Q8_K, payload, 256)
    assert np.array_equal(got, _ref_q8_k(d, qs))


def test_unsupported_quant_raises():
    with pytest.raises(Exception):
        _dequantize(16, b"\x00" * 100, 32)  # IQ2_XXS


# ---------------------------------------------------------------------------
# metadata + tensor-name mapping
# ---------------------------------------------------------------------------

def test_metadata_and_name_mapping_llama_family(tmp_path):
    rng = np.random.default_rng(1)
    path = write_gguf(
        str(tmp_path / "qwen.gguf"),
        {
            "general.architecture": "qwen2",
            "qwen2.embedding_length": 16,
            "qwen2.block_count": 2,
            "qwen2.feed_forward_length": 32,
            "qwen2.attention.head_count": 4,
            "qwen2.attention.head_count_kv": 2,
            "qwen2.vocab_size": 10,
            "qwen2.context_length": 64,
            "qwen2.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen2.rope.freq_base": 10000.0,
            "qwen2.tie_embeddings": False,
        },
        [
            ("token_embd.weight", rng.normal(size=(10, 16)).astype(np.float32)),
            ("blk.0.attn_norm.weight", np.ones(16, dtype=np.float32)),
            ("blk.0.attn_q.weight", rng.normal(size=(16, 16)).astype(np.float32)),
            ("blk.0.attn_k.weight", rng.normal(size=(8, 16)).astype(np.float32)),
            ("blk.0.attn_v.weight", rng.normal(size=(8, 16)).astype(np.float32)),
            ("blk.0.attn_output.weight", rng.normal(size=(16, 16)).astype(np.float32)),
            ("blk.0.ffn_norm.weight", np.ones(16, dtype=np.float32)),
            ("blk.0.ffn_gate.weight", rng.normal(size=(32, 16)).astype(np.float32)),
            ("blk.0.ffn_up.weight", rng.normal(size=(32, 16)).astype(np.float32)),
            ("blk.0.ffn_down.weight", rng.normal(size=(16, 32)).astype(np.float32)),
            ("blk.1.attn_norm.weight", np.ones(16, dtype=np.float32)),
            ("blk.1.attn_q.weight", rng.normal(size=(16, 16)).astype(np.float32)),
            ("blk.1.attn_k.weight", rng.normal(size=(8, 16)).astype(np.float32)),
            ("blk.1.attn_v.weight", rng.normal(size=(8, 16)).astype(np.float32)),
            ("blk.1.attn_output.weight", rng.normal(size=(16, 16)).astype(np.float32)),
            ("blk.1.ffn_norm.weight", np.ones(16, dtype=np.float32)),
            ("blk.1.ffn_gate.weight", rng.normal(size=(32, 16)).astype(np.float32)),
            ("blk.1.ffn_up.weight", rng.normal(size=(32, 16)).astype(np.float32)),
            ("blk.1.ffn_down.weight", rng.normal(size=(16, 32)).astype(np.float32)),
            ("output_norm.weight", np.ones(16, dtype=np.float32)),
            ("output.weight", rng.normal(size=(10, 16)).astype(np.float32)),
        ],
    )
    with GGUFReader(path) as reader:
        meta = reader.metadata
        assert meta["architecture"] == "Qwen2ForCausalLM"
        assert meta["hidden_size"] == 16
        assert meta["num_hidden_layers"] == 2
        assert meta["num_key_value_heads"] == 2
        assert meta["position_encoding_type"] == "rope"
        assert meta["mlp_type"] == "gated"
        assert meta["norm_type"] == "rmsnorm"
        assert meta["tie_word_embeddings"] is False
        assert "model.layers.0.self_attn.q_proj.weight" in reader.tensor_names()
        assert "model.embed_tokens.weight" in reader.tensor_names()
        assert "model.lm_head.weight" in reader.tensor_names()
        # dims come back in HF layout: q_proj is (out=16, in=16), ffn_down (16, 32)
        assert reader.tensor_info("model.layers.0.mlp.down_proj.weight").shape == (16, 32)
        assert reader.tensor_info("model.embed_tokens.weight").shape == (10, 16)


def test_metadata_mapping_gpt2(tmp_path):
    rng = np.random.default_rng(2)
    path = write_gguf(
        str(tmp_path / "gpt2.gguf"),
        {
            "general.architecture": "gpt2",
            "gpt2.embedding_length": 8,
            "gpt2.block_count": 1,
            "gpt2.attention.head_count": 2,
            "gpt2.vocab_size": 12,
            "gpt2.context_length": 32,
            "gpt2.attention.layer_norm_epsilon": 1e-5,
        },
        [
            ("wte.weight", rng.normal(size=(12, 8)).astype(np.float32)),
            ("wpe.weight", rng.normal(size=(32, 8)).astype(np.float32)),
            ("blk.0.ln_1.weight", np.ones(8, dtype=np.float32)),
            ("blk.0.ln_1.bias", np.zeros(8, dtype=np.float32)),
            ("blk.0.attn.c_attn.weight", rng.normal(size=(6, 8)).astype(np.float32)),
            ("blk.0.attn.c_attn.bias", np.zeros(6, dtype=np.float32)),
            ("blk.0.attn.c_proj.weight", rng.normal(size=(8, 6)).astype(np.float32)),
            ("blk.0.attn.c_proj.bias", np.zeros(8, dtype=np.float32)),
            ("blk.0.ln_2.weight", np.ones(8, dtype=np.float32)),
            ("blk.0.ln_2.bias", np.zeros(8, dtype=np.float32)),
            ("blk.0.mlp.c_fc.weight", rng.normal(size=(16, 8)).astype(np.float32)),
            ("blk.0.mlp.c_fc.bias", np.zeros(16, dtype=np.float32)),
            ("blk.0.mlp.c_proj.weight", rng.normal(size=(8, 16)).astype(np.float32)),
            ("blk.0.mlp.c_proj.bias", np.zeros(8, dtype=np.float32)),
            ("ln_f.weight", np.ones(8, dtype=np.float32)),
            ("ln_f.bias", np.zeros(8, dtype=np.float32)),
        ],
    )
    with GGUFReader(path) as reader:
        meta = reader.metadata
        assert meta["architecture"] == "GPT2LMHeadModel"
        assert meta["position_encoding_type"] == "learned"
        assert meta["qkv_fused"] is True
        assert meta["mlp_type"] == "standard"
        assert meta["norm_type"] == "layernorm"
        assert "model.h.0.attn.c_attn.weight" in reader.tensor_names()
        assert "model.wte.weight" in reader.tensor_names()
        # gpt2 without output.weight -> tied embeddings
        assert meta["tie_word_embeddings"] is True


def test_unsupported_architecture_raises(tmp_path):
    path = write_gguf(
        str(tmp_path / "mamba.gguf"),
        {"general.architecture": "mamba", "mamba.embedding_length": 8},
        [("token_embd.weight", np.zeros((8, 8), dtype=np.float32))],
    )
    with pytest.raises(Exception, match="mamba"):
        GGUFReader(path)


# ---------------------------------------------------------------------------
# end-to-end tiny model (independent numpy reference forward pass)
# ---------------------------------------------------------------------------

HIDDEN = 16
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 32
VOCAB = 32


def _tiny_gguf_model(path):
    from draco.runtime.cpu.ops import apply_rope, build_rope_cache, rms_norm, softmax, swiglu

    rng = np.random.default_rng(0)
    weights = {}

    def add_gguf(name, hf_shape, quant=False):
        arr = (rng.normal(size=hf_shape) * 0.05).astype(np.float32)
        weights[name] = arr
        if quant:
            return name, _enc_q8_0(arr.reshape(-1)), GGML_TYPE_Q8_0, hf_shape
        return name, arr

    def add_plain(name, hf_shape):
        arr = (rng.normal(size=hf_shape) * 0.05).astype(np.float32)
        weights[name] = arr
        return name, arr

    specs = []
    specs.append(add_plain("token_embd.weight", (VOCAB, HIDDEN)))
    for i in range(LAYERS):
        p = f"blk.{i}"
        specs.append(add_plain(f"{p}.attn_norm.weight", (HIDDEN,)))
        specs.append(add_plain(f"{p}.attn_q.weight", (HEADS * HEAD_DIM, HIDDEN)))
        specs.append(add_plain(f"{p}.attn_k.weight", (KV_HEADS * HEAD_DIM, HIDDEN)))
        specs.append(add_plain(f"{p}.attn_v.weight", (KV_HEADS * HEAD_DIM, HIDDEN)))
        specs.append(add_plain(f"{p}.attn_output.weight", (HIDDEN, HEADS * HEAD_DIM)))
        specs.append(add_plain(f"{p}.ffn_norm.weight", (HIDDEN,)))
        specs.append(add_gguf(f"{p}.ffn_gate.weight", (INTERMEDIATE, HIDDEN), quant=True))
        specs.append(add_gguf(f"{p}.ffn_up.weight", (INTERMEDIATE, HIDDEN), quant=True))
        specs.append(add_plain(f"{p}.ffn_down.weight", (HIDDEN, INTERMEDIATE)))
    specs.append(add_plain("output_norm.weight", (HIDDEN,)))
    specs.append(add_plain("output.weight", (VOCAB, HIDDEN)))

    write_gguf(
        path,
        {
            "general.architecture": "qwen2",
            "qwen2.embedding_length": HIDDEN,
            "qwen2.block_count": LAYERS,
            "qwen2.feed_forward_length": INTERMEDIATE,
            "qwen2.attention.head_count": HEADS,
            "qwen2.attention.head_count_kv": KV_HEADS,
            "qwen2.vocab_size": VOCAB,
            "qwen2.context_length": 64,
            "qwen2.attention.layer_norm_rms_epsilon": 1e-6,
        },
        specs,
    )

    # independent reference forward (same math as test_execution_graph_e2e)
    def reference(token_ids):
        hidden = weights["token_embd.weight"][np.array(token_ids)].astype(np.float32)
        seq_len = hidden.shape[0]
        cos, sin = build_rope_cache(HEAD_DIM, 64, 10000.0)
        positions = np.arange(seq_len)
        for i in range(LAYERS):
            p = f"blk.{i}"
            residual = hidden
            normed = rms_norm(hidden, weights[f"{p}.attn_norm.weight"])
            q = normed @ weights[f"{p}.attn_q.weight"].T
            k = normed @ weights[f"{p}.attn_k.weight"].T
            v = normed @ weights[f"{p}.attn_v.weight"].T
            q = q.reshape(seq_len, HEADS, HEAD_DIM)
            k = k.reshape(seq_len, KV_HEADS, HEAD_DIM)
            v = v.reshape(seq_len, KV_HEADS, HEAD_DIM)
            q = apply_rope(q, cos, sin, positions)
            k = apply_rope(k, cos, sin, positions)
            group = HEADS // KV_HEADS
            k_rep = np.repeat(k, group, axis=1)
            v_rep = np.repeat(v, group, axis=1)
            scores = (
                np.einsum("hqd,hkd->hqk", q.transpose(1, 0, 2), k_rep.transpose(1, 0, 2))
                / np.sqrt(HEAD_DIM)
            )
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores = np.where(mask[None, :, :], -np.inf, scores)
            probs = softmax(scores, axis=-1)
            attn = np.einsum("hqk,hkd->hqd", probs, v_rep.transpose(1, 0, 2)).transpose(1, 0, 2)
            attn = attn.reshape(seq_len, HEADS * HEAD_DIM)
            attn_out = attn @ weights[f"{p}.attn_output.weight"].T
            hidden = residual + attn_out
            residual = hidden
            normed = rms_norm(hidden, weights[f"{p}.ffn_norm.weight"])
            gate = normed @ weights[f"{p}.ffn_gate.weight"].T
            up = normed @ weights[f"{p}.ffn_up.weight"].T
            mlp = swiglu(gate, up) @ weights[f"{p}.ffn_down.weight"].T
            hidden = residual + mlp
        hidden = rms_norm(hidden, weights["output_norm.weight"])
        return (hidden[-1:] @ weights["output.weight"].T)[0]

    return reference


def test_end_to_end_gguf_matches_independent_reference(tmp_path):
    from draco.runtime.cpu.executor import CPUExecutionBackend

    path = str(tmp_path / "tiny.gguf")
    reference = _tiny_gguf_model(path)

    with GGUFReader(path) as reader:
        backend = CPUExecutionBackend(reader)
        seq_id = backend.new_sequence()
        prompt = [1, 2, 3, 4]
        logits = backend.forward_step(seq_id, np.array(prompt, dtype=np.int64), start_position=0)
        backend.end_sequence(seq_id)

        expected = reference(prompt)
        assert logits.shape == (VOCAB,)
        assert np.allclose(logits, expected, atol=1e-2, rtol=1e-2)


def test_cpu_native_llm_accepts_gguf(tmp_path):
    from draco.runtime.cpu.llm import CPUNativeLLM

    path = str(tmp_path / "tiny2.gguf")
    _tiny_gguf_model(path)
    with CPUNativeLLM(path) as llm:
        out = llm.generate([1, 2, 3], max_new_tokens=5)
        assert len(out) == 3 + 5
        assert all(0 <= t < VOCAB for t in out)
        info = llm.runtime_info()
        assert info["device"] == "cpu"


# ---------------------------------------------------------------------------
# byte-level BPE tokenizer
# ---------------------------------------------------------------------------

def test_bpe_tokenizer_roundtrip_and_specials():
    b2u = bytes_to_unicode()
    # vocab: all 256 byte tokens + a few merged tokens + specials
    tokens = []
    for b in range(256):
        tokens.append(b2u[b])
    merged = ["Ġ", "Ġh", "Ġhe", "Ġhel", "Ġhello", "t", "th", "the"]
    for t in merged:
        if t not in tokens:
            tokens.append(t)
    specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tokens.extend(specials)
    special_ids = {len(tokens) - 3 + i: s for i, s in enumerate(specials)}

    merges = ["Ġ h", "Ġh e", "Ġhe l", "Ġhel lo", "t h", "th e"]
    tok = GGUFBPETokenizer(
        tokens=tokens,
        merges=merges,
        token_type=[1] * len(tokens),
        chat_template=(
            "{{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
            "{{- '<|im_start|>user\n' }}{{- content }}{{ '<|im_end|>\n' }}"
            "{{- '<|im_start|>assistant\n' }}"
        ),
    )
    # mark the specials as special via the heuristic (they match <|...|>)
    assert "<|im_start|>" in tok.all_special_tokens

    text = "hello world"
    ids = tok.encode(text)
    assert ids, "expected non-empty encoding"
    # special tokens are matched as single ids
    chat_ids = tok.encode("<|im_start|>hello<|im_end|>")
    assert tok._vocab["<|im_start|>"] in chat_ids
    assert tok._vocab["<|im_end|>"] in chat_ids
    # round trip
    assert tok.decode(ids) == text
    assert tok.decode(chat_ids) == "hello"

    # chat template rendering
    rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}])
    assert rendered.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
    assert rendered.endswith("<|im_start|>assistant\n")
    # encoding the rendered template round-trips through decode
    # (skip_special_tokens=False so the <|im_start|>/<|im_end|> markers stay)
    rids = tok.encode(rendered)
    assert tok.decode(rids, skip_special_tokens=False) == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.mark.skipif(
    not os.path.isdir(r"E:\models\Qwen2.5-0.5B-Instruct"),
    reason="local Qwen2.5 HF model not present",
)
@pytest.mark.skipif(
    not os.environ.get("DRACO_GGUF_TOKENIZER_VERIFY"),
    reason="set DRACO_GGUF_TOKENIZER_VERIFY=1 to run the real-Qwen tokenizer comparison",
)
def test_bpe_matches_real_qwen_tokenizer():
    """Compare the GGUF-style BPE against the actual Qwen2.5 tokenizer."""
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(r"E:\models\Qwen2.5-0.5B-Instruct")
    vocab = hf.get_vocab()
    tokens = [""] * len(vocab)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    merges = list(getattr(hf, "merges", []) or [])
    tok = GGUFBPETokenizer(
        tokens=tokens,
        merges=merges,
        token_type=[1] * len(tokens),
        chat_template=hf.chat_template,
    )
    samples = [
        "Hello, how are you?",
        "The capital of France is Paris.",
        "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    ]
    for text in samples:
        assert tok.encode(text) == hf(text, add_special_tokens=False)["input_ids"], text