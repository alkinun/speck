from pathlib import Path

import pytest
import torch

from scripts import paper_baseline_preflight as preflight


def test_baseline_preflight_arguments_require_an_output():
    args = preflight.arguments(["matrix.json", "--output", "report.json"])

    assert args.matrix == Path("matrix.json")
    assert args.output == Path("report.json")
    assert args.device == "cuda"


def test_baseline_preflight_waits_for_the_frozen_start_temperature(monkeypatch):
    temperatures = iter([53, 52, 50])
    monotonic = iter([0.0, 1.0, 2.0, 3.0])
    sleeps = []
    monkeypatch.setattr(preflight, "_temperature_c", lambda: next(temperatures))
    monkeypatch.setattr(preflight.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(preflight.time, "sleep", lambda seconds: sleeps.append(seconds))

    temperature, waited = preflight._wait_for_temperature(poll_seconds=5)

    assert temperature == 50
    assert waited == 3.0
    assert sleeps == [5, 5]


def test_baseline_preflight_temperature_timeout_is_explicit(monkeypatch):
    temperatures = iter([53, 52])
    monotonic = iter([0.0, 6.0])
    monkeypatch.setattr(preflight, "_temperature_c", lambda: next(temperatures))
    monkeypatch.setattr(preflight.time, "monotonic", lambda: next(monotonic))

    with pytest.raises(RuntimeError, match="did not cool"):
        preflight._wait_for_temperature(timeout_seconds=5)


def test_baseline_preflight_retains_export_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "prepare_current_release_code", lambda directory: None)
    monkeypatch.setattr(
        preflight,
        "release_state",
        lambda state: {"native.weight": torch.zeros(1, dtype=torch.bfloat16)},
    )
    monkeypatch.setattr(preflight, "release_config", lambda metadata: {"model_type": "speck"})
    monkeypatch.setattr(
        preflight,
        "validate_export",
        lambda directory, metadata: (_ for _ in ()).throw(ValueError("parity failed")),
    )

    result = preflight._export_case({}, {}, tmp_path)

    assert result["passed"] is False
    assert result["failure_type"] == "ValueError"
    assert result["failure"] == "parity failed"
    assert "model.safetensors" in result["files"]


def test_baseline_preflight_parity_metrics_retain_extent_and_argmax():
    expected = torch.tensor([[[1.0, 0.0], [0.5, 1.0]]])
    actual = torch.tensor([[[1.01, 0.0], [1.1, 1.0]]])

    result = preflight._parity_metrics(actual, expected)

    assert result["passed"] is False
    assert result["mismatched_logits"] == 1
    assert result["total_logits"] == 4
    assert result["argmax_agreement"] == 0.5
    assert result["maximum_absolute_logit_error"] == pytest.approx(0.6)
