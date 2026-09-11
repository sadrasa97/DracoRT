import numpy as np

from draco.exceptions import CPUKernelError
from draco.runtime.cpu.capabilities import CPUCapabilities, detect
from draco.runtime.cpu.dispatcher import registered_tiers, select_kernel
from draco.quantization.cpu import int8
from draco.format.tensor import DracoTensorInfo


def test_detect_returns_positive_core_counts():
    caps = detect()
    assert caps.physical_cores >= 1
    assert caps.logical_cores >= 1


def test_never_claims_unimplemented_kernel_tier():
    caps = CPUCapabilities(
        architecture="x86_64",
        vendor="GenuineIntel",
        model_name="fake cpu",
        physical_cores=8,
        logical_cores=16,
        avx512f=True,
        avx512vnni=True,
    )
    selection = select_kernel(caps)
    # avx512_vnni is detected but has no registered kernel in this build ->
    # must fall back to an implemented tier, and say so honestly.
    assert selection.actual_tier in registered_tiers()
    assert selection.requested_tier == "avx512_vnni"
    assert selection.downgraded is True


def test_generic_tier_always_available():
    caps = CPUCapabilities(
        architecture="x86_64", vendor="v", model_name="m", physical_cores=1, logical_cores=1
    )
    selection = select_kernel(caps)
    assert selection.actual_tier == "generic"
    assert selection.downgraded is False


def test_generic_kernel_int8_matmul_matches_reference():
    rng = np.random.default_rng(7)
    w = rng.normal(size=(16, 32)).astype(np.float32)
    x = rng.normal(size=(2, 32)).astype(np.float32)
    q = int8.quantize(w)

    info = DracoTensorInfo(name="w", dtype="i8", quantization="int8_sym", shape=w.shape)
    selection = select_kernel()
    out = selection.kernel.matmul(x, info, q.qweight.tobytes(), scale=q.scale)
    expected = int8.matmul(x, q)
    assert np.allclose(out, expected)
