import json
from pathlib import Path

import pytest

from speck.tokenizer_pilot_cuda_evaluation import load_document_nll_policy

ROOT = Path(__file__).parents[1]
POLICY = ROOT / "research/flagship/tokenizer_pilot_document_nll_policy_v1.json"


def test_cuda_evaluation_loads_exact_frozen_policy():
    policy = load_document_nll_policy(POLICY)

    assert policy["value"]["requirements"]["maximum_mean_nll_delta_nats_per_token"] == 0.0001
    assert policy["value"]["sample"]["documents"] == 6
    assert policy["value"]["authority"]["screen_execution"] is False


def test_cuda_evaluation_rejects_policy_drift(tmp_path):
    value = json.loads(POLICY.read_text())
    value["requirements"]["maximum_mean_nll_delta_nats_per_token"] = 1.0
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="unsupported"):
        load_document_nll_policy(path)
