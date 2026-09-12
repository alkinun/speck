import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULT = ROOT / "results/systems/tokenizer-pilot-cuda-resume-successor-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_cuda_resume_successor_binds_policy_and_passes_without_authority():
    result = json.loads(RESULT.read_text())

    assert result["status"] == "exact_shape_bf16_checkpoint_resume_within_frozen_policy"
    assert sha256(ROOT / result["policy"]["path"]) == result["policy"]["sha256"]
    for path, digest in result["implementation"].values():
        assert sha256(ROOT / path) == digest
    equivalence = result["equivalence"]
    assert equivalence["loss_exact"] is True
    assert equivalence["data_cursor_exact"] is True
    assert equivalence["all_values_finite"] is True
    assert equivalence["model_max_absolute_error"] <= equivalence["model_limit"]
    assert equivalence["optimizer_max_absolute_error"] <= equivalence["optimizer_limit"]
    assert result["gates"]["scientific_model_outputs_created"] == 0
    assert result["gates"]["screen_execution_authority"] is False


def test_cuda_resume_qualification_bytes_match_when_available():
    result = json.loads(RESULT.read_text())
    path = Path(result["runtime"]["qualification"]["path"])
    if not path.is_file():
        pytest.skip("requires maintainer-local CUDA resume qualification")
    assert sha256(path) == result["runtime"]["qualification"]["sha256"]
    qualification = json.loads(path.read_text())
    assert qualification["status"] == (
        "two_step_checkpoint_resume_within_frozen_policy_no_screen_authority"
    )
    assert qualification["screen_execution_authority"] is False
