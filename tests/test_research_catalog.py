import json
from pathlib import Path

import pytest

from scripts.research_catalog import arguments
from speck.research_catalog import validate_claim_registry, validate_research_catalog

ROOT = Path(__file__).parents[1]


def test_checked_research_catalog_resolves_collections_notebook_and_claims():
    result = validate_research_catalog(ROOT / "research/catalog.json")

    assert result["format"] == "speck_research_catalog_validation"
    assert result["active_program"] == "flagship"
    assert set(result["collections"]) == {
        "flagship",
        "promotion_protocol",
        "paper1_archive",
        "notebook",
        "literature",
        "findings",
        "experiments",
        "results",
        "paper_workspace",
        "software",
    }
    assert result["notebook_entries"] == 3
    assert result["lifecycle_stages"] == 9
    assert result["paper"]["claims"] == 4
    assert result["paper"]["claim_statuses"] == {
        "planned": 2,
        "prior_evidence_only": 2,
    }
    assert result["paper"]["headline_ready"] == []
    claims = json.loads((ROOT / "paper/claims.json").read_text())["claims"]
    assert [claim["id"] for claim in claims] == [
        "C-DATA",
        "C-MEMORY",
        "C-TRANSFER",
        "C-SYSTEM",
    ]


def test_research_catalog_cli_defaults_to_checked_catalog():
    assert arguments([]).catalog == Path("research/catalog.json")


def test_claim_registry_rejects_missing_evidence(tmp_path):
    registry = json.loads((ROOT / "paper/claims.json").read_text())
    registry["claims"][0]["evidence"] = ["results/does-not-exist.json"]
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(registry))

    with pytest.raises(ValueError, match="evidence.*does not exist"):
        validate_claim_registry(path, repository_root=ROOT)


def test_claim_registry_rejects_duplicate_claim_ids(tmp_path):
    registry = json.loads((ROOT / "paper/claims.json").read_text())
    registry["claims"][1]["id"] = registry["claims"][0]["id"]
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(registry))

    with pytest.raises(ValueError, match="duplicate paper claim ids"):
        validate_claim_registry(path, repository_root=ROOT)


def test_supported_claim_requires_finding_and_result_evidence(tmp_path):
    registry = json.loads((ROOT / "paper/claims.json").read_text())
    claim = registry["claims"][2]
    claim["status"] = "supported"
    claim["evidence"] = ["findings/105_paper_1_paired_proxy_analysis.md"]
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(registry))

    with pytest.raises(ValueError, match="requires both a finding and result"):
        validate_claim_registry(path, repository_root=ROOT)


def test_catalog_requires_active_program_to_be_a_collection(tmp_path):
    catalog = json.loads((ROOT / "research/catalog.json").read_text())
    catalog["active_program"]["id"] = "missing_program"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="active program must be a catalog collection"):
        validate_research_catalog(path, repository_root=ROOT)
