import json
from pathlib import Path

from scripts.paper_ratio_placement_v2_validate import file_sha256


def test_ratio_placement_v2_qualification_is_bound_and_training_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "ratio-placement-v2-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    design = artifact["inputs"]["design"]
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == "qualified_deconfounded_raw_ratio_design_training_blocked"
    assert file_sha256(root / design["path"]) == design["sha256"]
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["reproduced_v1_confound"]["pure_ratio_attribution_from_matched_arms"] is False
    assert artifact["qualified_design"]["matched_sensitivity_can_choose_placement_count"] is False
    assert artifact["decision"]["v2_registered_in_active_program"] is False
    assert artifact["decision"]["v2_training_authorized"] is False
