import importlib.util
import json
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("value", [1, -1, True, "0"])
def test_preflight_cannot_claim_eligible_tokens(tmp_path, monkeypatch, value):
    relative = "experiments/main-data/candidate-manifest-preflight.json"
    preflight = json.loads((_MODULE.ROOT / relative).read_text())
    preflight["summary"]["eligible_unique_tokens_established"] = value
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(preflight))
    resolve = _MODULE._path
    monkeypatch.setattr(_MODULE, "_path", lambda name: path if name == relative else resolve(name))
    with pytest.raises(ValueError, match="cannot claim eligible tokens"):
        _MODULE.validate()


@pytest.mark.parametrize("mutation", ["hash", "admitted", "duplicate", "coverage", "stage"])
def test_manifest_rejects_drift(tmp_path, mutation):
    manifest = json.loads((_MODULE.ROOT / _MODULE.MANIFESTS[0]).read_text())
    if mutation == "hash":
        manifest["source_of_truth"]["data_design_contract"]["sha256"] = "0" * 64
    elif mutation == "admitted":
        manifest["candidates"][0]["admitted"] = True
    elif mutation == "duplicate":
        manifest["candidates"].append(manifest["candidates"][0])
    elif mutation == "coverage":
        manifest["common_contract"]["coverage_strata"] = None
    else:
        manifest["common_contract"]["stage"] = ""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _MODULE._validate_manifest(path)
