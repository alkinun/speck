import json
from pathlib import Path

from speck.paper_finalist_systems_workload import file_sha256


def test_systems_workload_plan_qualification_is_bound_and_execution_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-workload-plan-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    protocol = artifact["inputs"]["protocol"]
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_static_plan_and_mutation_detection_engine_and_kernel_isolation_blocked"
    )
    assert file_sha256(root / protocol["path"]) == protocol["sha256"]
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["static_plan"]["checkpoint_paths_statted_or_opened_during_planning"] is False
    assert artifact["mutation_detection"]["kernel_read_only_enforced"] is False
    assert artifact["decision"]["benchmark_engine_qualified"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
