import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/systems/tokenizer-pilot-cuda-resume-exact-failure-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_exact_cuda_resume_failure_is_bound_and_fail_closed():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "exact_resume_equivalence_failed_tolerance_successor_required"
    for path, digest in result["implementation"].values():
        assert sha256(ROOT / path) == digest
    assert result["equivalence"]["second_step_loss_exact"] is True
    assert result["equivalence"]["data_cursor_exact"] is True
    assert result["equivalence"]["model_exact"] is False
    assert result["equivalence"]["model_max_absolute_error"] == 2**-14
    assert result["equivalence"]["optimizer_max_absolute_error"] == 2**-14
    assert result["gates"]["screen_execution_authority"] is False
    assert result["gates"]["D5_audit_opening_authority"] is False


def test_failure_bytes_match_when_available():
    result = json.loads(RESULT.read_text())
    path = Path(result["runtime"]["failure"]["path"])
    if not path.is_file():
        pytest.skip("requires maintainer-local CUDA resume failure")
    assert sha256(path) == result["runtime"]["failure"]["sha256"]
    failure = json.loads(path.read_text())
    assert failure["status"] == "failed_no_screen_authority"
    assert failure["scientific_run"] is False
