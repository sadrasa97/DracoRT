import pytest

from draco.exceptions import InsufficientMemoryError
from draco.convert.planner import CPUQuantizationPlanner
from draco.runtime.cpu.capabilities import CPUCapabilities
from draco.runtime.cpu.memory import CPUMemoryPlanner
from draco.runtime.cpu.threading import plan_threads


def _caps(physical=8, logical=16):
    return CPUCapabilities(
        architecture="x86_64", vendor="v", model_name="m",
        physical_cores=physical, logical_cores=logical, avx2=True,
    )


def test_memory_planner_inference_scales_with_quantization():
    planner = CPUMemoryPlanner()
    est_f32 = planner.estimate_inference(
        num_params=1_000_000_000, quantization="f32", num_layers=32,
        hidden_size=4096, num_key_value_heads=8, head_dim=128,
        max_model_len=2048, max_num_seqs=1,
    )
    est_int4 = planner.estimate_inference(
        num_params=1_000_000_000, quantization="int4_groupwise", num_layers=32,
        hidden_size=4096, num_key_value_heads=8, head_dim=128,
        max_model_len=2048, max_num_seqs=1,
    )
    assert est_int4.weights_bytes < est_f32.weights_bytes
    assert est_int4.total_bytes < est_f32.total_bytes


def test_memory_planner_conversion_vs_inference_differ():
    planner = CPUMemoryPlanner()
    conv = planner.estimate_conversion(num_params=1_000_000_000, source_dtype="f32")
    inf = planner.estimate_inference(
        num_params=1_000_000_000, quantization="int8_sym", num_layers=32,
        hidden_size=4096, num_key_value_heads=8, head_dim=128,
        max_model_len=2048, max_num_seqs=1,
    )
    assert conv.peak_bytes != inf.total_bytes


def test_threading_respects_explicit_override():
    cfg = plan_threads(_caps(), num_threads=3)
    assert cfg.num_threads == 3
    assert cfg.source == "user_override"


def test_threading_uses_physical_not_logical_cores(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    cfg = plan_threads(_caps(physical=8, logical=32))
    assert cfg.num_threads == 8
    assert cfg.source == "heuristic"


def test_quantization_planner_picks_less_aggressive_when_it_fits():
    planner = CPUQuantizationPlanner()
    plan = planner.plan(
        num_params=100_000_000, num_layers=12, hidden_size=768,
        num_key_value_heads=12, head_dim=64,
        cpu_capabilities=_caps(), available_memory_bytes=64 * 1024**3,
        desired_context_length=2048, desired_concurrency=1,
    )
    assert plan.quantization == "f32"


def test_quantization_planner_downgrades_under_memory_pressure():
    planner = CPUQuantizationPlanner()
    plan = planner.plan(
        num_params=13_000_000_000, num_layers=40, hidden_size=5120,
        num_key_value_heads=40, head_dim=128,
        cpu_capabilities=_caps(), available_memory_bytes=16 * 1024**3,
        desired_context_length=4096, desired_concurrency=1,
    )
    assert plan.quantization in ("int8_sym", "int4_groupwise")


def test_quantization_planner_raises_when_nothing_fits():
    planner = CPUQuantizationPlanner()
    with pytest.raises(InsufficientMemoryError):
        planner.plan(
            num_params=70_000_000_000, num_layers=80, hidden_size=8192,
            num_key_value_heads=64, head_dim=128,
            cpu_capabilities=_caps(), available_memory_bytes=1 * 1024**3,
            desired_context_length=4096, desired_concurrency=1,
        )
