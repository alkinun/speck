import pytest

from speck.training.benchmark import arguments, resolve_activation_checkpointing


def test_benchmark_activation_checkpointing_uses_config_by_default():
    assert resolve_activation_checkpointing({}, None) is False
    assert resolve_activation_checkpointing({"activation_checkpointing": True}, None) is True


def test_benchmark_activation_checkpointing_has_explicit_runtime_override():
    assert arguments(["experiment", "--activation-checkpointing"]).activation_checkpointing is True
    assert (
        arguments(["experiment", "--no-activation-checkpointing"]).activation_checkpointing is False
    )
    assert resolve_activation_checkpointing({"activation_checkpointing": False}, True) is True
    assert resolve_activation_checkpointing({"activation_checkpointing": True}, False) is False


def test_benchmark_rejects_invalid_checkpointing_config():
    with pytest.raises(ValueError, match="must be boolean"):
        resolve_activation_checkpointing({"activation_checkpointing": "yes"}, None)


def test_kernel_summary_excludes_cpu_custom_autograd_and_merges_device_rows():
    from types import SimpleNamespace

    from torch.autograd import DeviceType

    from speck.training.benchmark import kernel_summary

    def event(name, micros, device=DeviceType.CUDA):
        return SimpleNamespace(key=name, self_device_time_total=micros, count=1, device_type=device)

    events = [
        event("ChunkKDAFunctionBackward", 900, DeviceType.CPU),
        event("Torch-Compiled Region", 500, DeviceType.CPU),
        event("chunk_kda_bwd", 30),
        event("ampere_bf16_gemm", 20),
        event("ampere_bf16_gemm", 30),
        event("vectorized_elementwise", 20),
        event("Command Buffer Full", 40),
    ]
    summary = kernel_summary(SimpleNamespace(key_averages=lambda: events))
    assert summary["self_device_time_total_us"] == 100
    assert summary["launch_stall_us"] == 40
    assert summary["useful_percent"] == 80
    assert summary["category_percent"] == {"mixer": 30, "gemm": 50, "pointwise": 20}
    assert summary["top_kernels"][0]["count"] == 2
    assert len(summary["top_kernels"]) == 3


def test_kernel_summary_without_cuda_events_has_no_utilization_claim():
    from types import SimpleNamespace

    from torch.autograd import DeviceType

    from speck.training.benchmark import kernel_summary

    profiler = SimpleNamespace(
        key_averages=lambda: [
            SimpleNamespace(device_type=DeviceType.CPU, self_device_time_total=100)
        ]
    )
    summary = kernel_summary(profiler)
    assert summary["useful_percent"] is None
    assert summary["top_kernels"] == []


def test_selected_benchmark_mode_compiles_the_reproducible_production_options():
    from speck.operations.runtime import COMPILE_OPTIONS
    from speck.training.benchmark import _COMPILE_MODE_OPTIONS

    # Coordinate descent re-times reductions per process and breaks restart parity.
    assert "coordinate_descent_tuning" not in COMPILE_OPTIONS
    selected = {**_COMPILE_MODE_OPTIONS["max-autotune-no-cudagraphs"], "aggressive_fusion": True}
    assert selected == COMPILE_OPTIONS


def test_hopper_cublas_kernels_count_as_gemm():
    from speck.training.benchmark import classify_kernel

    assert classify_kernel("nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT") == "gemm"
    assert classify_kernel("chunk_kda_bwd_kernel_intra") == "mixer"
