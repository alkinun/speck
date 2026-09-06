import json
from pathlib import Path

from scripts.stable_latentmoe_cpu_reference_qualify import file_sha256


def test_stable_latentmoe_CPU_qualification_is_bound_and_integration_blocked():
    root = Path(__file__).parents[1]
    path = (
        root
        / "results"
        / "Speck-Paper1"
        / "stable-latentmoe-cpu-reference-qualified-v1.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for reference in (
        artifact["protocol"],
        artifact["implementation"],
        artifact["unit_tests"],
    ):
        assert file_sha256(root / reference["path"]) == reference["sha256"]
    assert file_sha256(root / "scripts" / "stable_latentmoe_cpu_reference_qualify.py") == artifact[
        "runner_sha256"
    ]
    assert artifact["status"] == (
        "four_isolated_CPU_primitives_qualified_composition_training_blocked"
    )
    assert artifact["results"]["normalized_LatentMoE"] == {
        "finite_gradient_tensors": 26112,
        "maximum_manual_forward_error": 0.0,
        "shape_seed_cases": 3072,
        "valid_shape_combinations": 96,
    }
    quantile = artifact["results"]["Quantile_Balancing"]
    assert quantile["exact_shape_seed_cases"] == 128
    assert quantile["boundary_tie_experts"] == 204
    assert quantile["target_rank_bracket_failures"] == 0
    assert quantile["histogram_cases"] == 384
    assert quantile["source_1000_bin_cases"] == 128
    assert quantile["partition_cases"] == 768
    assert quantile["partition_count_mismatches"] == 0
    assert quantile["maximum_interval_error_in_bin_widths"] <= 1
    assert artifact["decision"]["primitive_composition_authorized"] is False
    assert artifact["decision"]["model_integration_authorized"] is False
    assert artifact["decision"]["training_authorized"] is False
