import json

import pytest

from scripts.throughput_summary import summarize


def _receipt(path, rate, *, batch_size=1):
    value = {
        "format": "speck_throughput_probe",
        "benchmark": {
            "mode": "compute",
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
            "model_flops_utilization": 0.1,
        },
        "model": {},
        "experiment": {"fingerprint": "same"},
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
