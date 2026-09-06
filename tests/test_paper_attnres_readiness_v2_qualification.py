import json
from pathlib import Path

from scripts.paper_attnres_readiness_v2_validate import file_sha256


def test_attnres_v2_qualification_is_bound_and_training_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "attnres-readiness-v2-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    references = (
        artifact["inputs"]["predecessor"],
        artifact["inputs"]["primary_source_audit"],
        artifact["inputs"]["design"],
        artifact["inputs"]["active_experiment_program"],
        artifact["implementation"]["source_validator"],
        artifact["implementation"]["source_tests"],
        artifact["implementation"]["readiness_validator"],
        artifact["implementation"]["readiness_tests"],
    )
    for reference in references:
        assert file_sha256(root / reference["path"]) == reference["sha256"]
    assert artifact["status"] == (
        "qualified_logical_boundary_and_zero_initialization_correction_training_blocked"
    )
    assert artifact["correction"]["v1_eight_by_five_partition_valid"] is False
    assert artifact["correction"]["N8_module_sizes"] == [4, 6, 4, 6, 4, 6, 4, 6]
    assert artifact["correction"]["pseudo_query_initialization"] == "exact zero"
    assert artifact["decision"]["reference_implementation_authorized"] is False
    assert artifact["decision"]["training_authorized"] is False
