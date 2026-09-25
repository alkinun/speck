import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "experiments/main-data/check_candidate_manifests.py"
_SPEC = importlib.util.spec_from_file_location("check_candidate_manifests", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_candidate_manifests_name_readiness_sources_and_resolve_receipts():
    result = _MODULE.validate()

    assert result == {"manifests": 3, "candidates": 9, "receipts_checked": 20}


@pytest.mark.parametrize(
    "mutation", ["hash", "copied_gates", "unknown", "duplicate", "coverage", "stage"]
)
def test_manifest_rejects_drift(tmp_path, mutation):
    manifest = json.loads((_MODULE.ROOT / _MODULE.MANIFESTS[0]).read_text())
    if mutation == "hash":
        manifest["source_of_truth"]["data_design_contract"]["sha256"] = "0" * 64
    elif mutation == "copied_gates":
        manifest["candidates"][0]["gate_status"] = {"source_use": "open"}
    elif mutation == "unknown":
        manifest["candidates"][0]["id"] = "unlisted_source"
    elif mutation == "duplicate":
        manifest["candidates"].append(manifest["candidates"][0])
    elif mutation == "coverage":
        manifest["common_contract"]["coverage_strata"] = None
    else:
        manifest["common_contract"]["stage"] = ""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _MODULE._validate_manifest(path, {"fineweb_edu", "ultrafineweb_hq"})
