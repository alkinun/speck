import pytest
import torch

from speck import common


@pytest.mark.parametrize(
    "environment",
    (
        {"RANK": "0"},
        {"RANK": "four", "LOCAL_RANK": "0", "WORLD_SIZE": "4"},
        {"RANK": "4", "LOCAL_RANK": "0", "WORLD_SIZE": "4"},
        {"RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "4"},
        {"RANK": "0", "LOCAL_RANK": "1", "WORLD_SIZE": "1"},
    ),
)
def test_distributed_environment_tuple_fails_closed(monkeypatch, environment):
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "SPECK_EXPECTED_LOCAL_WORLD_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="distributed|single-process"):
        common.dist_info()


def test_cuda_local_rank_is_checked_before_process_group_initialization(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    initialized = []
    monkeypatch.setattr(
        torch.distributed, "init_process_group", lambda *args, **kwargs: initialized.append(1)
    )

    with pytest.raises(ValueError, match="visible local CUDA device"):
        common.init_runtime("cuda")
    assert initialized == []


def test_single_node_four_gpu_contract_rejects_missing_or_wrong_world(monkeypatch):
    monkeypatch.setenv("SPECK_EXPECTED_LOCAL_WORLD_SIZE", "4")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="single-node launch contract"):
        common.dist_info()

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "3")
    with pytest.raises(ValueError, match="single-node launch contract"):
        common.dist_info()


def test_distributed_manifest_identity_is_all_gathered_and_must_match(monkeypatch):
    def mismatch(output, identity):
        output[:] = [identity, {**identity, "manifest": "other"}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", mismatch)
    with pytest.raises(ValueError, match="differs between ranks"):
        common.verify_distributed_identity({"manifest": "one"}, 2, "training manifest")

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, identity: output.__setitem__(slice(None), [identity, identity]),
    )
    identity = {"manifest": "one"}
    assert common.verify_distributed_identity(identity, 2, "training manifest") is identity
