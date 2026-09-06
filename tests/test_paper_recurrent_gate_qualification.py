import json
from pathlib import Path

from scripts.paper_recurrent_gate_validate import file_sha256


def test_recurrent_gate_qualification_is_bound_and_KDA_SiLU_blocked():
    root = Path(__file__).parents[1]
    path = root / "results" / "Speck-Paper1" / "recurrent-gate-v1-qualified.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    design = artifact["inputs"]["design"]
    validator = artifact["implementation"]["validator"]
    tests = artifact["implementation"]["tests"]

    assert artifact["status"] == ("qualified_conditional_design_KDA_SiLU_implementation_blocked")
    assert file_sha256(root / design["path"]) == design["sha256"]
    assert file_sha256(root / validator["path"]) == validator["sha256"]
    assert file_sha256(root / tests["path"]) == tests["sha256"]
    assert artifact["KDA_implementation_boundary"]["KDA_SiLU_config_currently_expressible"] is False
    assert artifact["decision"]["general_sigmoid_claim_authorized"] is False
    assert artifact["decision"]["training_authorized"] is False
