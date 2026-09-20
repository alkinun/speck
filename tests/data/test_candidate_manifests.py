import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[2] / "experiments/main-data/check_candidate_manifests.py"
_SPEC = importlib.util.spec_from_file_location("check_candidate_manifests", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_candidate_manifests_are_receipt_bound_and_non_admitting():
    result = _MODULE.validate()

    assert result["manifests"] == 4
    assert result["candidates"] == 9
    assert result["receipts_checked"] == 23
    assert result["admitted_candidates"] == 0
    assert result["eligible_unique_tokens_established"] == 0
