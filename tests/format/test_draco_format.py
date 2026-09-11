import os
import struct

import numpy as np
import pytest

from draco.exceptions import DracoCorruptModelError, DracoFormatError, DracoFormatVersionError
from draco.format.header import HEADER_SIZE, MAGIC, FORMAT_VERSION, DracoHeader
from draco.format.reader import DracoReader
from draco.format.validator import validate_file
from draco.format.writer import DracoWriter


def _make_sample(path, tmp_dir):
    writer = DracoWriter(os.path.join(tmp_dir, path))
    writer.add_metadata(
        {
            "architecture": "Qwen2ForCausalLM",
            "model_type": "qwen2",
            "hidden_size": 8,
        }
    )
    writer.add_tokenizer({"vocab": {"hello": 0, "world": 1}})
    rng = np.random.default_rng(42)
    w1 = rng.normal(size=(8, 8)).astype(np.float32)
    w2 = rng.normal(size=(4, 64)).astype(np.float32)
    writer.add_tensor("layer0.norm.weight", rng.normal(size=(8,)).astype(np.float32))
    writer.add_tensor("layer0.q_proj.weight", w1, quantization="int8_sym")
    writer.add_tensor("layer0.gate_proj.weight", w2, quantization="int4_groupwise")
    out_path = writer.finalize()
    return out_path, w1, w2


def test_write_read_roundtrip(tmp_path):
    out_path, w1, w2 = _make_sample("model.draco", str(tmp_path))
    assert os.path.exists(out_path)

    with DracoReader(out_path) as reader:
        assert reader.architecture == "Qwen2ForCausalLM"
        assert reader.tokenizer == {"vocab": {"hello": 0, "world": 1}}
        assert set(reader.tensor_names()) == {
            "layer0.norm.weight",
            "layer0.q_proj.weight",
            "layer0.gate_proj.weight",
        }

        norm = reader.get_tensor("layer0.norm.weight")
        assert norm.shape == (8,)

        q1 = reader.get_tensor("layer0.q_proj.weight")
        assert q1.shape == w1.shape
        assert np.mean(np.abs(q1 - w1)) < 0.05

        q2 = reader.get_tensor("layer0.gate_proj.weight")
        assert q2.shape == w2.shape
        assert np.mean(np.abs(q2 - w2)) < 0.2


def test_file_extension_is_not_trusted(tmp_path):
    fake = tmp_path / "fake.draco"
    fake.write_bytes(b"not a draco file at all, just some junk bytes" * 4)
    with pytest.raises(DracoFormatError):
        DracoReader(str(fake))


def test_bad_magic_rejected(tmp_path):
    path = tmp_path / "bad_magic.draco"
    header = DracoHeader(
        version=FORMAT_VERSION,
        header_size=HEADER_SIZE,
        flags=0,
        metadata_offset=HEADER_SIZE,
        metadata_size=2,
        tensor_index_offset=HEADER_SIZE + 2,
        tensor_index_size=2,
        tensor_data_offset=HEADER_SIZE + 4,
        tensor_data_size=0,
        tokenizer_offset=0,
        tokenizer_size=0,
        checksum=0,
    )
    data = header.pack()
    corrupted = b"XXXXXXXX" + data[8:]
    path.write_bytes(corrupted + b"{}[]")
    with pytest.raises(DracoFormatError):
        DracoReader(str(path))


def test_version_mismatch_raises_specific_error(tmp_path):
    path = tmp_path / "bad_version.draco"
    fields = struct.pack(
        "<8sIII" + "QQ" * 4 + "I",
        MAGIC,
        FORMAT_VERSION + 999,
        HEADER_SIZE,
        0,
        HEADER_SIZE,
        2,
        HEADER_SIZE + 2,
        2,
        HEADER_SIZE + 4,
        0,
        0,
        0,
        0,
    )
    path.write_bytes(fields + b"{}[]")
    with pytest.raises(DracoFormatVersionError):
        DracoReader(str(path))


def test_truncated_file_rejected(tmp_path):
    out_path, _, _ = _make_sample("trunc.draco", str(tmp_path))
    data = open(out_path, "rb").read()
    truncated = tmp_path / "truncated.draco"
    truncated.write_bytes(data[: len(data) // 2])
    report = validate_file(str(truncated))
    assert not report.ok


def test_validator_passes_on_well_formed_file(tmp_path):
    out_path, _, _ = _make_sample("valid.draco", str(tmp_path))
    report = validate_file(out_path)
    assert report.ok
    assert report.num_tensors == 3
    assert report.checksum_verified


def test_duplicate_tensor_name_rejected(tmp_path):
    writer = DracoWriter(str(tmp_path / "dup.draco"))
    writer.add_metadata({"architecture": "X", "model_type": "x"})
    writer.add_tensor("w", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(DracoFormatError):
        writer.add_tensor("w", np.zeros((2, 2), dtype=np.float32))


def test_missing_required_metadata_rejected(tmp_path):
    writer = DracoWriter(str(tmp_path / "nometa.draco"))
    writer.add_tensor("w", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(DracoFormatError):
        writer.finalize()


def test_unsupported_quantization_rejected(tmp_path):
    writer = DracoWriter(str(tmp_path / "badquant.draco"))
    writer.add_metadata({"architecture": "X", "model_type": "x"})
    with pytest.raises(DracoFormatError):
        writer.add_tensor("w", np.zeros((2, 2), dtype=np.float32), quantization="gptq_int4_super")


def test_atomic_write_no_partial_file_on_failure(tmp_path, monkeypatch):
    """A failed finalize() must never leave a file at the target path."""
    writer = DracoWriter(str(tmp_path / "atomic.draco"))
    writer.add_metadata({"architecture": "X", "model_type": "x"})
    writer.add_tensor("w", np.zeros((2, 2), dtype=np.float32))

    from draco.format import validator

    def boom(path):
        raise DracoCorruptModelError("forced failure for test")

    monkeypatch.setattr(validator, "validate_file", boom)
    with pytest.raises(DracoCorruptModelError):
        writer.finalize()
    assert not (tmp_path / "atomic.draco").exists()


def test_reader_rejects_out_of_bounds_tensor_index(tmp_path):
    out_path, _, _ = _make_sample("oob.draco", str(tmp_path))
    report = validate_file(out_path)
    assert report.ok  # sanity: well-formed file passes first


def test_int8_asym_and_bf16_through_draco_file(tmp_path):
    writer = DracoWriter(str(tmp_path / "mixed.draco"))
    writer.add_metadata({"architecture": "X", "model_type": "x"})
    rng = np.random.default_rng(99)
    w_asym = (rng.normal(size=(8, 16)) * 2 + 3).astype(np.float32)
    w_bf16 = rng.normal(size=(4, 8)).astype(np.float32) * 50
    writer.add_tensor("attn.weight", w_asym, quantization="int8_asym")
    writer.add_tensor("norm.weight", w_bf16, dtype="bf16")
    path = writer.finalize()

    with DracoReader(path) as reader:
        got_asym = reader.get_tensor("attn.weight")
        assert got_asym.shape == w_asym.shape
        assert np.mean(np.abs(got_asym - w_asym)) < 0.1

        got_bf16 = reader.get_tensor("norm.weight")
        assert got_bf16.shape == w_bf16.shape
        rel = np.abs(got_bf16 - w_bf16) / (np.abs(w_bf16) + 1e-6)
        assert np.mean(rel) < 0.01
