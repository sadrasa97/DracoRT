import pytest

from draco.utils.oom_guard import OOMRetryConfig, is_oom_error, run_with_oom_backoff


class FakeCudaOOM(RuntimeError):
    pass


def test_is_oom_error_detects_common_messages():
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_oom_error(FakeCudaOOM("CUDA OOM"))
    assert not is_oom_error(ValueError("bad argument"))
    assert not is_oom_error(FileNotFoundError("model.draco not found"))


def test_run_with_oom_backoff_succeeds_after_retries():
    calls = []

    def flaky(gpu_memory_utilization):
        calls.append(gpu_memory_utilization)
        if len(calls) < 3:
            raise FakeCudaOOM("CUDA out of memory")
        return f"ok at {gpu_memory_utilization}"

    result = run_with_oom_backoff(
        flaky, initial_value=0.9, config=OOMRetryConfig(max_retries=5, backoff_factor=0.5, min_value=0.1)
    )
    assert result.oom_events == 2
    assert result.attempts == 3
    assert calls == [0.9, 0.45, 0.225]
    assert "ok at 0.225" in result.result


def test_run_with_oom_backoff_raises_after_exhausting_retries():
    def always_oom(gpu_memory_utilization):
        raise FakeCudaOOM("CUDA out of memory")

    with pytest.raises(FakeCudaOOM):
        run_with_oom_backoff(always_oom, initial_value=0.9, config=OOMRetryConfig(max_retries=2))


def test_run_with_oom_backoff_never_goes_below_min_value():
    calls = []

    def flaky(gpu_memory_utilization):
        calls.append(gpu_memory_utilization)
        if len(calls) < 6:
            raise FakeCudaOOM("CUDA out of memory")
        return "ok"

    with pytest.raises(FakeCudaOOM):
        run_with_oom_backoff(
            flaky, initial_value=0.4, config=OOMRetryConfig(max_retries=4, backoff_factor=0.5, min_value=0.3)
        )
    assert all(v >= 0.3 for v in calls)


def test_non_oom_errors_propagate_immediately():
    calls = []

    def raises_value_error(gpu_memory_utilization):
        calls.append(gpu_memory_utilization)
        raise ValueError("not an OOM at all")

    with pytest.raises(ValueError):
        run_with_oom_backoff(raises_value_error, initial_value=0.9)
    assert len(calls) == 1  # no retries for non-OOM errors


def test_success_on_first_try_reports_zero_oom_events():
    result = run_with_oom_backoff(lambda gpu_memory_utilization: "immediate", initial_value=0.9)
    assert result.oom_events == 0
    assert result.attempts == 1
