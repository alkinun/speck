import json
from pathlib import Path

from speck.paper_finalist_systems_analysis import file_sha256


def test_finalist_systems_analysis_qualification_is_bound_and_execution_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-analysis-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    protocol = artifact["inputs"]["protocol"]
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == "qualified_offline_analysis_execution_still_blocked"
    assert file_sha256(root / protocol["path"]) == protocol["sha256"]
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["statistical_implementation"][
        "gross_energy_primary_uses_candidate_upper_over_control_lower"
    ]
    assert artifact["failure_semantics"]["imputation"] is False
    assert artifact["decision"]["execution_authorized"] is False
