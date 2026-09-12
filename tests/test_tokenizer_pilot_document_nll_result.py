import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/systems/tokenizer-pilot-document-nll-qualification-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_document_nll_result_is_bound_and_below_frozen_limit():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "six_category_cuda_chunk_parity_pass_execution_records_pending"
    assert sha256(ROOT / result["policy"]["path"]) == result["policy"]["sha256"]
    for path, digest in result["implementation"].values():
        assert sha256(ROOT / path) == digest
    assert result["coverage"]["categories"] == 6
    assert result["coverage"]["maximum_prediction_tokens"] == 19_629
    assert (
        result["parity"]["maximum_observed_mean_nll_delta_nats_per_token"]
        <= result["parity"]["maximum_allowed_mean_nll_delta_nats_per_token"]
    )
    assert result["parity"]["all_values_finite"] is True
    assert result["gates"]["scientific_model_outputs_created"] == 0
    assert result["gates"]["screen_execution_authority"] is False


def test_document_nll_runtime_result_matches_when_available():
    result = json.loads(RESULT.read_text())
    path = Path(result["runtime"]["path"])
    if not path.is_file():
        pytest.skip("requires maintainer-local document NLL qualification")
    assert path.stat().st_size == result["runtime"]["bytes"]
    assert sha256(path) == result["runtime"]["sha256"]
    runtime = json.loads(path.read_text())
    assert runtime["status"] == "six_category_cuda_chunk_parity_pass_no_screen_authority"
    assert len(runtime["results"]) == 6
    assert runtime["screen_execution_authority"] is False
