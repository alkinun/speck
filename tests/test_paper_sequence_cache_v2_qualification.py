import json
from pathlib import Path

from scripts.paper_sequence_cache_v2_validate import file_sha256


def test_cache_representation_v2_qualification_is_bound_and_training_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "sequence-cache-representation-v2-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    design = artifact["inputs"]["design"]
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == "qualified_deconfounded_four_arm_design_training_blocked"
    assert file_sha256(root / design["path"]) == design["sha256"]
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["reproduced_v1_confound"]["pure_MQA_attribution_from_v1_matched_arm"] is False
    assert artifact["code_derived_raw_MQA"]["parameters"] == 152975898
    assert artifact["decision"]["v2_registered_in_active_program"] is False
    assert artifact["decision"]["v2_training_authorized"] is False
