import hashlib
import json
from pathlib import Path

from speck.tokenizer_pilot_train import load_resume_policy

ROOT = Path(__file__).parents[1]
POLICY = ROOT / "research/flagship/tokenizer_pilot_resume_policy_v1.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_resume_policy_is_frozen_before_retry_and_binds_successor_code():
    value = json.loads(POLICY.read_text())
    policy = load_resume_policy(POLICY)

    assert policy["requirements"] == {
        "loss_exact": True,
        "data_cursor_exact": True,
        "model_max_absolute_error": 2**-13,
        "optimizer_max_absolute_error": 2**-13,
        "all_values_finite": True,
    }
    assert (
        sha256(ROOT / value["failed_predecessor"]["path"]) == value["failed_predecessor"]["sha256"]
    )
    for identity in value["implementation"].values():
        assert sha256(ROOT / identity["path"]) == identity["sha256"]
    assert value["retry"]["fresh_initialization_required"] is True
    assert value["retry"]["failed_checkpoint_parent_forbidden"] is True
    assert value["authority"]["screen_execution"] is False
    assert value["authority"]["D5_opening"] is False
