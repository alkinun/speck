import json

import pytest

from scripts.throughput_summary import summarize


def _receipt(path, rate, *, batch_size=1):
    value = {
        "format": "speck_throughput_probe",
        "format_version": 2,
        "label": path.stem,
        "quality": {"stable": True},
        "environment": {
            "device": "cuda",
            "device_name": "test",
            "device_capability": [9, 0],
            "torch": "pinned",
            "cuda": "12.8",
            "packages": {},
            "cublas_workspace_config": ":4096:8",
            "git_revision": "abc",
            "git_dirty": False,
        },
        "benchmark": {
            "mode": "compute",
            "steps": 30,
            "warmup_steps": 10,
            "seed": 42,
            "peak_tflops": 1.0,
            "aggressive_fusion": False,
            "compiled": False,
            "compile_mode": None,
            "loss_backend": "liger",
            "activation_checkpointing": True,
            "deterministic": True,
            "training_output": False,
            "optimizer": "muon",
            "optimizer_step_compiled": False,
        },
        "geometry": {
            "batch_size": batch_size,
            "sequence_length": 4096,
            "accumulation": 32 // batch_size,
            "tokens_per_step": 131072,
        },
        "performance": {
            "tokens_per_second": rate,
            "step_seconds_median": 131072 / rate,
            "model_flops_utilization": rate * 1e9 / 1e12,
        },
        "model": {"flops_per_token": 1e9, "flops_convention": "test"},
        "experiment": {"fingerprint": "same", "manifest": None},
    }
    path.write_text(json.dumps(value))


def test_summary_reports_noise_band(tmp_path):
    paths = [tmp_path / f"run-{index}.json" for index in range(3)]
    for path, rate in zip(paths, (100.0, 110.0, 90.0)):
        _receipt(path, rate)

    result = summarize(paths)

    assert result["runs"] == 3
    assert result["summary"]["tokens_per_second"]["median"] == 100.0
    assert result["summary"]["tokens_per_second"]["maximum_absolute_relative_to_median"] == 0.1


def test_summary_rejects_mixed_geometry(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    _receipt(first, 100.0)
    _receipt(second, 100.0, batch_size=2)

    with pytest.raises(ValueError, match="does not match"):
        summarize([first, second])


def test_summary_requires_repeated_receipts(tmp_path):
    path = tmp_path / "only.json"
    _receipt(path, 100.0)

    with pytest.raises(ValueError, match="at least two"):
        summarize([path])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("environment", "device_name", "different GPU"),
        ("environment", "torch", "different torch"),
        ("environment", "git_revision", "different source"),
        ("environment", "git_dirty", True),
        ("environment", "kernel_caches", {"triton": "/different-cache"}),
        ("benchmark", "steps", 5),
        ("benchmark", "seed", 123),
        ("experiment", "manifest", "different data"),
        ("quality", "stable", False),
        ("performance", "tokens_per_second", float("nan")),
        ("performance", "step_seconds_median", 0),
        ("performance", "model_flops_utilization", 0.9),
    ],
)
def test_rejects_incomparable_or_invalid_receipts(tmp_path, section, key, value):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        _receipt(path, 100.0)
    record = json.loads(paths[1].read_text())
    record[section][key] = value
    paths[1].write_text(json.dumps(record))
    with pytest.raises(ValueError):
        summarize(paths)


def test_copied_receipt_is_not_a_repeat(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    _receipt(first, 100.0)
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="duplicate receipt"):
        summarize([first, second])


def test_summary_accepts_no_mfu_without_peak(tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        _receipt(path, 100.0)
        record = json.loads(path.read_text())
        record["benchmark"]["peak_tflops"] = None
        record["performance"]["model_flops_utilization"] = None
        path.write_text(json.dumps(record))
    assert summarize(paths)["summary"]["model_flops_utilization"] is None
