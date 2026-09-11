import pytest

from draco.runtime.gpu.memory import DeviceMemoryInfo, GPUMemoryPlanner, auto_memory_budget, detect_device_memory


def test_detect_device_memory_reports_unavailable_without_gpu():
    info = detect_device_memory()
    # This sandbox has no CUDA-capable torch install.
    assert info.available is False
    assert info.total_bytes == 0


def test_auto_memory_budget_raises_without_gpu():
    with pytest.raises(RuntimeError):
        auto_memory_budget(0.9)


def test_auto_memory_budget_computes_fraction_of_free():
    info = DeviceMemoryInfo(available=True, total_bytes=24 * 1024**3, free_bytes=20 * 1024**3, device_name="fake-gpu")
    budget = auto_memory_budget(0.5, info=info)
    assert budget == 10 * 1024**3


def test_auto_memory_budget_rejects_bad_utilization():
    with pytest.raises(ValueError):
        auto_memory_budget(0.0)
    with pytest.raises(ValueError):
        auto_memory_budget(2.0)


def test_estimate_inference_scales_with_dtype():
    planner = GPUMemoryPlanner()
    est_f32 = planner.estimate_inference(
        num_params=7_000_000_000, dtype="f32", num_layers=32,
        num_key_value_heads=8, head_dim=128, max_model_len=4096, max_num_seqs=8,
    )
    est_int4 = planner.estimate_inference(
        num_params=7_000_000_000, dtype="int4", num_layers=32,
        num_key_value_heads=8, head_dim=128, max_model_len=4096, max_num_seqs=8,
    )
    assert est_int4.weights_bytes < est_f32.weights_bytes


def test_max_num_seqs_that_fit_never_exceeds_budget():
    planner = GPUMemoryPlanner()
    n = planner.max_num_seqs_that_fit(
        num_params=7_000_000_000, dtype="f16", num_layers=32,
        num_key_value_heads=8, head_dim=128, max_model_len=4096,
        budget_bytes=24 * 1024**3,
    )
    assert n >= 1
    est = planner.estimate_inference(
        num_params=7_000_000_000, dtype="f16", num_layers=32,
        num_key_value_heads=8, head_dim=128, max_model_len=4096, max_num_seqs=n,
    )
    assert est.weights_bytes + est.kv_cache_bytes <= 24 * 1024**3


def test_max_num_seqs_that_fit_zero_when_weights_alone_exceed_budget():
    planner = GPUMemoryPlanner()
    n = planner.max_num_seqs_that_fit(
        num_params=70_000_000_000, dtype="f32", num_layers=80,
        num_key_value_heads=64, head_dim=128, max_model_len=4096,
        budget_bytes=8 * 1024**3,
    )
    assert n == 0
