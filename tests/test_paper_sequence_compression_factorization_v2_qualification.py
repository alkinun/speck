import json
from pathlib import Path

from scripts.paper_sequence_compression_factorization_v2_validate import file_sha256


def test_sequence_factorization_v2_qualification_is_bound_and_training_blocked():
    root = Path(__file__).parents[1]
    path = (
        root
        / "results"
        / "Speck-Paper1"
        / "sequence-compression-factorization-v2-qualified.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    references = (
        *artifact["inputs"].values(),
        *artifact["implementation"].values(),
    )
    for reference in references:
        assert file_sha256(root / reference["path"]) == reference["sha256"]
    assert artifact["status"] == (
        "qualified_joint_HCA_CSA_local_factorization_parents_implementation_training_blocked"
    )
    assert artifact["correction"]["same_layer_HCA_plus_CSA"] is False
    assert artifact["correction"]["HCA_compressor_arms"] == 4
    assert artifact["correction"]["CSA_selected_unit"] == "overlapping compressed entry"
    assert artifact["correction"]["raw_compressed_cross_family_deduplication"] is False
    assert artifact["decision"]["parent_selected"] is False
    assert artifact["decision"]["reference_implementation_authorized"] is False
    assert artifact["decision"]["training_authorized"] is False
