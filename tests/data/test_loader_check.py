import json

import pytest

from speck.data.loader_check import check_loader
from speck.training.smoke import run_smoke


def test_loader_replay_uses_identical_real_batches_and_detects_changed_expectations(
    tmp_path, monkeypatch
):
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "SPECK_EXPECTED_LOCAL_WORLD_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "smoke"
    run_smoke(root)
    output = tmp_path / "loader"
    scan = check_loader(root / "experiment", output, batches=8)
    assert scan["rank_training_tokens"] == 256
    replay = check_loader(root / "experiment", output, batches=8, mode="replay")
    assert replay["status"] == "exact_replay"
    assert replay["replayed_batches"] == 4
    (output / "rank-0.replay.json").unlink()
    scan["replay_batches"][0]["inputs_sha256"] = "0" * 64
    (output / "rank-0.scan.json").write_text(json.dumps(scan))
    with pytest.raises(AssertionError, match="replay differs"):
        check_loader(root / "experiment", output, batches=8, mode="replay")
