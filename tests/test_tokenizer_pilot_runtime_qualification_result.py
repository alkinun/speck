import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/data/tokenizer-pilot-runtime-qualification-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_runtime_qualification_binds_implementation_and_remains_no_model():
    result = json.loads(RESULT.read_text())

    assert (
        result["status"]
        == "real_v2_streams_evaluation_and_resume_inputs_pass_model_execution_blocked"
    )
    for path, digest in result["implementation"].values():
        assert sha256(ROOT / path) == digest
    assert result["coverage"]["runs"] == 3
    assert result["coverage"]["packed_shards"] == 63
    assert result["coverage"]["evaluation_documents"] == 12_470
    assert result["coverage"]["evaluation_utf8_bytes"] == 60_035_301
    assert result["gates"]["packed_shard_sizes_and_sha256"] == "pass_all"
    assert result["gates"]["model_outputs_created"] == 0
    assert result["authority"]["screen_execution"] is False


def test_runtime_qualification_bytes_match_when_available():
    result = json.loads(RESULT.read_text())
    path = Path(result["runtime"]["path"])
    if not path.is_file():
        pytest.skip("requires maintainer-local tokenizer pilot runtime qualification")

    assert path.stat().st_size == result["runtime"]["bytes"]
    assert sha256(path) == result["runtime"]["sha256"]
    runtime = json.loads(path.read_text())
    assert runtime["unique_files_verified"] == result["coverage"]["unique_files_verified"]
    assert runtime["gates"]["initial_resume_and_final_batch_cursors"] == "pass_all"
