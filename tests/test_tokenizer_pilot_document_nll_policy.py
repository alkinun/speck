import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
POLICY = ROOT / "research/flagship/tokenizer_pilot_document_nll_policy_v1.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_document_nll_policy_is_frozen_before_cuda_output():
    policy = json.loads(POLICY.read_text())

    assert policy["status"] == "pre_output_cuda_chunk_parity_policy_frozen"
    assert policy["sample"]["documents"] == 6
    assert policy["sample"]["selection_uses_model_output"] is False
    assert policy["requirements"] == {
        "identical_document_and_token_identity": True,
        "all_nll_values_finite": True,
        "maximum_mean_nll_delta_nats_per_token": 0.0001,
        "maximum_context_tokens": 131_072,
        "categories_required": 6,
    }
    for identity in policy["qualified_implementation_inputs"].values():
        assert sha256(ROOT / identity["path"]) == identity["sha256"]
    resume = policy["inputs"]["resume_qualification"]
    assert sha256(ROOT / resume["path"]) == resume["sha256"]
    assert policy["authority"]["screen_execution"] is False
    assert policy["authority"]["D5_opening"] is False


def test_local_policy_inputs_match_when_available():
    policy = json.loads(POLICY.read_text())
    for name in ("run", "checkpoint"):
        identity = policy["inputs"][name]
        path = Path(identity["path"])
        if not path.is_file():
            pytest.skip(f"requires maintainer-local document NLL input: {path}")
        assert sha256(path) == identity["sha256"]
