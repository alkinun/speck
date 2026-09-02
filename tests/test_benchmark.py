import pytest

from scripts.benchmark import arguments, resolve_activation_checkpointing


def test_benchmark_activation_checkpointing_uses_config_by_default():
    assert resolve_activation_checkpointing({}, None) is False
    assert resolve_activation_checkpointing({"activation_checkpointing": True}, None) is True


def test_benchmark_activation_checkpointing_has_explicit_runtime_override():
    assert arguments(["--activation-checkpointing"]).activation_checkpointing is True
    assert arguments(["--no-activation-checkpointing"]).activation_checkpointing is False
    assert resolve_activation_checkpointing({"activation_checkpointing": False}, True) is True
    assert resolve_activation_checkpointing({"activation_checkpointing": True}, False) is False


def test_benchmark_rejects_invalid_checkpointing_config():
    with pytest.raises(ValueError, match="must be boolean"):
        resolve_activation_checkpointing({"activation_checkpointing": "yes"}, None)
