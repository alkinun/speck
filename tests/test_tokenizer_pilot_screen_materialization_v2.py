import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/data/tokenizer-pilot-screen-materialization-v2-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_corrected_materialization_is_bound_and_execution_blocked():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "corrected_three_screen_run_manifests_complete_execution_blocked"
    for key in ("plan", "correction", "rejected_predecessor"):
        assert sha256(ROOT / result[key]["path"]) == result[key]["sha256"]
    assert result["gates"]["corrected_flops_independently_recomputed"] == "pass_all"
    assert result["gates"]["optimizer_boundaries_unchanged"] is True
    assert result["gates"]["model_outputs_created"] == 0
    assert result["authority"]["screen_execution"] is False
    assert result["authority"]["D5_opening"] is False


def test_corrected_runtime_records_match_when_available():
    result = json.loads(RESULT.read_text())
    manifest = Path(result["runtime"]["manifest"]["path"])
    if not manifest.is_file():
        pytest.skip("requires maintainer-local corrected tokenizer pilot records")

    assert sha256(manifest) == result["runtime"]["manifest"]["sha256"]
    for run in result["runtime"]["runs"]:
        path = Path(result["runtime"]["directory"]) / run["path"]
        assert sha256(path) == run["sha256"]
        value = json.loads(path.read_text())
        assert value["run_fingerprint"] == run["run_fingerprint"]
        assert value["status"] == "corrected_materialized_not_started_execution_blocked"
        assert value["authority"]["screen_execution"] is False
