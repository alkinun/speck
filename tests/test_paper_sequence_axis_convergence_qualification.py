import json
from pathlib import Path

from scripts.paper_sequence_axis_convergence_validate import file_sha256


def test_sequence_axis_convergence_qualification_is_bound_and_F0_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "sequence-axis-convergence-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for reference in (
        *artifact["inputs"].values(),
        *artifact["implementation"].values(),
    ):
        assert file_sha256(root / reference["path"]) == reference["sha256"]
    assert artifact["status"] == "qualified_pre_results_dependency_DAG_active_finalist_blocked"
    assert artifact["DAG"]["nodes"] == 11
    assert artifact["DAG"]["acyclic"] is True
    assert artifact["DAG"]["root"] == "F0_active_finalist"
    assert artifact["critical_corrections"][
        "base_exact_attention_ratio_selection_is_not_compressed_schedule_revalidation"
    ] is True
    assert artifact["decision"]["dependency_DAG_qualified"] is True
    assert artifact["decision"]["descendant_configs_materialized"] is False
    assert artifact["decision"]["implementation_authorized"] is False
    assert artifact["decision"]["training_authorized"] is False
