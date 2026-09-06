import json
from pathlib import Path

from speck.paper_finalist_systems_workload import file_sha256


def test_systems_sandbox_qualification_is_bound_and_actual_checkpoint_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "finalist-systems-sandbox-qualified-v1.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    workload = artifact["inputs"]["workload_plan_qualification"]
    module = artifact["implementation"]["module"]
    cli = artifact["implementation"]["cli"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == (
        "qualified_disposable_kernel_read_only_fixture_checkpoint_engine_execution_blocked"
    )
    assert file_sha256(root / workload["path"]) == workload["sha256"]
    assert file_sha256(root / module["path"]) == module["sha256"]
    assert file_sha256(root / cli["path"]) == cli["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["disposable_live_fixture"]["write_blocked"] is True
    assert artifact["disposable_live_fixture"]["errno_name"] == "EROFS"
    assert artifact["disposable_live_fixture"]["post_collect_load_state"] == "not-found"
    assert artifact["decision"]["actual_finalist_experiment_or_checkpoint_path_tested"] is False
    assert artifact["decision"]["systems_execution_authorized"] is False
