from types import SimpleNamespace

import torch

from speck.diagnostics import command_output, maximum_error, nearest_percentile, synchronize


def test_command_output_returns_stdout_or_none():
    assert command_output(["git", "rev-parse", "--show-toplevel"])
    assert command_output(["false"]) is None


def test_maximum_error_compares_in_float32():
    actual = torch.tensor([1.0, 2.5], dtype=torch.bfloat16)
    expected = torch.tensor([1.0, 2.0])
    assert maximum_error(actual, expected) == 0.5


def test_nearest_percentile_sorts_values():
    values = [4.0, 1.0, 3.0, 2.0]
    assert nearest_percentile(values, 0.25) == 2.0
    assert nearest_percentile(values, 0.75) == 3.0


def test_synchronize_only_calls_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    cpu = SimpleNamespace(type="cpu")
    cuda = SimpleNamespace(type="cuda")

    synchronize(cpu)
    synchronize(cuda)

    assert calls == [cuda]
