import pytest

from draco.runtime.device_manager import resolve_device


def test_auto_resolves_to_cpu_without_cuda():
    decision = resolve_device("auto", draco_metadata=None)
    # In this sandbox there is no CUDA-capable torch install.
    assert decision.device == "cpu"


def test_explicit_cpu_always_works():
    decision = resolve_device("cpu")
    assert decision.device == "cpu"


def test_cpu_metadata_hint_favors_cpu():
    decision = resolve_device(
        "auto", draco_metadata={"cpu": {"preferred_quantization": "q4_k_m"}}
    )
    assert decision.device == "cpu"
    assert "CPU-oriented" in decision.reason


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_explicit_cuda_without_torch_raises():
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
