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
    assert result["implementation"] == {
        "module": [
            "speck/tokenizer_pilot_train.py",
            "0ddfd644414e641646d403c07e2aa19db5e7809098ccffbb5e4d07b28722b7ec",
        ],
        "cli": [
            "scripts/tokenizer_pilot_train_qualify.py",
            "76992d85e314f73963b02dbb5020ce14a2f6399b35af8052039d2e55ee929f82",
        ],
        "tests": [
            "tests/test_tokenizer_pilot_train.py",
            "a7b9dc76a0df3e32b2ce316a94785bd0bce5235c09fab05a3eb7b4f6025b26da",
        ],
    }
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
