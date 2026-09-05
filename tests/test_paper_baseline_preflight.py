from pathlib import Path

import pytest

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
