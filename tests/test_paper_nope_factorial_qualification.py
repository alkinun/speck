import json
from pathlib import Path

from scripts.paper_nope_factorial_validate import file_sha256


def test_nope_factorial_qualification_is_bound_and_training_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "nope-factorial-v1-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    design = artifact["inputs"]["design"]
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == "qualified_interaction_aware_factorial_training_blocked"
    assert file_sha256(root / design["path"]) == design["sha256"]
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["historical_boundary"]["three_seed_NoPE_replication"] is False
    assert artifact["factorial"]["mixer_by_position_interaction"] is True
    assert artifact["efficient_reuse"]["outcomes_inspected"] is False
    assert artifact["decision"]["training_authorized"] is False
