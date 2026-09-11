import numpy as np
import pytest

from draco.exceptions import CPUQuantizationError
from draco.quantization.cpu import int4, int8


def test_int8_roundtrip_error_bounded():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 128)).astype(np.float32)
    stats = int8.roundtrip_error(w)
    assert stats["mean_rel_error"] < 0.05
    assert stats["max_abs_error"] < 0.1


def test_int8_matmul_matches_dense_within_tolerance():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(32, 64)).astype(np.float32)
    x = rng.normal(size=(4, 64)).astype(np.float32)
    q = int8.quantize(w)
    out_q = int8.matmul(x, q)
    out_dense = x @ w.T
    rel_err = np.abs(out_q - out_dense) / (np.abs(out_dense) + 1e-6)
    assert np.mean(rel_err) < 0.05


def test_int8_rejects_non_2d():
    with pytest.raises(CPUQuantizationError):
        int8.quantize(np.zeros((4, 4, 4), dtype=np.float32))


def test_int8_all_zero_row_does_not_nan():
    w = np.zeros((2, 8), dtype=np.float32)
    q = int8.quantize(w)
    recon = int8.dequantize(q)
    assert not np.isnan(recon).any()
    assert np.allclose(recon, 0)


def test_int4_roundtrip_error_bounded():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(16, 64)).astype(np.float32)
    stats = int4.roundtrip_error(w, group_size=32)
    assert stats["mean_rel_error"] < 0.30


def test_int4_pack_unpack_shapes():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(8, 32)).astype(np.float32)
    q = int4.quantize(w, group_size=16)
    assert q.packed.shape == (8, 16)
    recon = int4.dequantize(q)
    assert recon.shape == w.shape


def test_int4_rejects_bad_group_size():
    w = np.zeros((4, 30), dtype=np.float32)
    with pytest.raises(CPUQuantizationError):
        int4.quantize(w, group_size=32)


def test_int4_matmul_matches_dense_within_tolerance():
    rng = np.random.default_rng(4)
    w = rng.normal(size=(16, 64)).astype(np.float32)
    x = rng.normal(size=(3, 64)).astype(np.float32)
    q = int4.quantize(w, group_size=32)
    out_q = int4.matmul(x, q)
    out_dense = x @ w.T
    denom = np.maximum(np.abs(out_dense), 0.1 * np.std(out_dense))
    rel_err = np.abs(out_q - out_dense) / denom
    assert np.mean(rel_err) < 0.30


def test_int8_asym_roundtrip_error_bounded():
    from draco.quantization.cpu import int8_asym

    rng = np.random.default_rng(20)
    w = (rng.normal(size=(64, 128)) * 3 + 5).astype(np.float32)  # shifted, not centered on 0
    stats = int8_asym.roundtrip_error(w)
    assert stats["mean_rel_error"] < 0.05


def test_int8_asym_matmul_matches_dense():
    from draco.quantization.cpu import int8_asym

    rng = np.random.default_rng(21)
    w = (rng.normal(size=(32, 64)) * 2 + 4).astype(np.float32)
    x = rng.normal(size=(4, 64)).astype(np.float32)
    q = int8_asym.quantize(w)
    out_q = int8_asym.matmul(x, q)
    out_dense = x @ w.T
    denom = np.maximum(np.abs(out_dense), 0.1 * np.std(out_dense))
    rel_err = np.abs(out_q - out_dense) / denom
    assert np.mean(rel_err) < 0.05


def test_int8_asym_beats_symmetric_on_shifted_distribution():
    """The whole point of the asymmetric format: better accuracy when the
    weight distribution isn't centered near zero."""
    from draco.quantization.cpu import int8, int8_asym

    rng = np.random.default_rng(22)
    w = (rng.normal(size=(32, 64)) * 0.5 + 10).astype(np.float32)  # far from zero
    sym_stats = int8.roundtrip_error(w)
    asym_stats = int8_asym.roundtrip_error(w)
    assert asym_stats["mean_abs_error"] < sym_stats["mean_abs_error"]


def test_bf16_roundtrip_within_bf16_precision():
    from draco.quantization.cpu import bf16

    rng = np.random.default_rng(23)
    w = rng.normal(size=(100,)).astype(np.float32) * 100
    encoded = bf16.encode(w)
    assert encoded.dtype == np.uint16
    decoded = bf16.decode(encoded)
    # bf16 has ~3 significant decimal digits; relative error should be small
    rel = np.abs(decoded - w) / (np.abs(w) + 1e-6)
    assert np.mean(rel) < 0.01


def test_bf16_preserves_wide_dynamic_range():
    """bf16's whole reason to exist vs f16: it shares f32's exponent range."""
    from draco.quantization.cpu import bf16

    big = np.array([1e30, -1e30, 1e-30], dtype=np.float32)
    encoded = bf16.encode(big)
    decoded = bf16.decode(encoded)
    assert np.all(np.isfinite(decoded))
    # order of magnitude should survive even though precision is coarse
    assert np.allclose(np.sign(decoded), np.sign(big))
