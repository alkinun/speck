import json
from pathlib import Path

from scripts.quantile_balancing_tie_validate import file_sha256, load_object, validate_protocol


def test_QB_tie_v2_qualification_is_bound_and_training_policy_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "quantile-balancing-tie-v2-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    protocol = artifact["protocol"]

    assert file_sha256(root / protocol["path"]) == protocol["sha256"]
    validate_protocol(load_object(root / protocol["path"]), root)
    assert file_sha256(root / "scripts" / "quantile_balancing_tie_validate.py") == artifact[
        "runner_sha256"
    ]
    assert artifact["status"] == "fixed_score_CPU_midpoint_qualified_training_policy_blocked"
    assert artifact["candidate_gate"]["passed"] is True

    midpoint = artifact["unseen_validation"]["interval_midpoint"]
    assert midpoint["cases"] == 256
    assert midpoint["perfect_balance_cases"] == 256
    assert midpoint["cycle_cases"] == 0
    assert midpoint["unresolved_cases"] == 0
    assert midpoint["coordinate_subgradient_failures"] == 0
    assert midpoint["coordinate_interval_violations"] == 0
    assert midpoint["nonfinite_values"] == 0
    assert midpoint["maximum_updates_among_passes"] == 33

    upper = artifact["unseen_validation"]["source_upper_endpoint"]
    histogram = artifact["unseen_validation"]["source_histogram_1000"]
    assert (upper["perfect_balance_cases"], upper["cycle_cases"]) == (28, 228)
    assert (histogram["perfect_balance_cases"], histogram["cycle_cases"]) == (241, 15)
    assert artifact["decision"]["fixed_score_CPU_midpoint_reference_qualified"] is True
    assert artifact["decision"]["training_tie_policy_selected"] is False
    assert artifact["decision"]["model_integration_authorized"] is False
    assert artifact["decision"]["training_authorized"] is False
