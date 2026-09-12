import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEFECT = ROOT / "results/data/tokenizer-pilot-fixed-flop-drift-20260912.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_fixed_flop_drift_is_fail_closed_before_model_outputs():
    result = json.loads(DEFECT.read_text())
    pilot = json.loads((ROOT / result["pilot_plan"]["path"]).read_text())
    preflight = json.loads((ROOT / result["preflight_result"]["path"]).read_text())
    mismatch = result["mismatch"]

    assert sha256(ROOT / result["pilot_plan"]["path"]) == result["pilot_plan"]["sha256"]
    assert sha256(ROOT / result["preflight_result"]["path"]) == result["preflight_result"]["sha256"]
    assert (
        mismatch["v10_reference_flops_per_token"]
        == pilot["fixed_flop_materialization"]["reference"]["flops_per_token"]
    )
    measured = {item["id"]: item for item in preflight["tokenizers"]}
    corrected_reference = measured["mistral-32k"]["analytic_training_flops_per_token"]
    assert mismatch["preflight_and_current_reference_flops_per_token"] == corrected_reference
    assert mismatch["v10_reference_flops_per_token"] != corrected_reference

    reference_tokens = pilot["stopping"]["fixed_document_mistral_tokens"]
    corrected_target = corrected_reference * reference_tokens
    candidate_flops = measured["speck-bpe-40960-whitespace"]["analytic_training_flops_per_token"]
    corrected_stop = math.ceil(corrected_target / candidate_flops)
    candidate = mismatch["speck_bpe_40960_whitespace"]
    assert mismatch["corrected_analytic_flop_target"] == corrected_target
    assert candidate["corrected_fixed_flop_token_stop"] == corrected_stop
    assert candidate["token_delta"] == candidate["v10_fixed_flop_token_stop"] - corrected_stop
    assert candidate["aligned_stop_unchanged"] is True
    assert result["detected_before_model_outputs"] is True
    assert result["gates"]["screen_execution_authority"] is False
