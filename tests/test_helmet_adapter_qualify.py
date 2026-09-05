import json

from scripts.helmet_adapter_qualify import adapter_contract_identity, directory_identity


def test_directory_identity_changes_with_contents(tmp_path):
    (tmp_path / "a").write_text("first", encoding="utf-8")
    first = directory_identity(tmp_path)
    (tmp_path / "a").write_text("second", encoding="utf-8")
    second = directory_identity(tmp_path)

    assert first["sha256"] != second["sha256"]


def test_adapter_contract_identity_ignores_evidence_registration():
    contract = {
        "suite_version": "version",
        "upstream": {"revision": "a" * 40},
        "benchmark": {"lengths": [8192]},
        "model_adapter": {"mode": "native", "status": "blocked"},
    }
    before = adapter_contract_identity(contract)
    copied = json.loads(json.dumps(contract))
    copied["model_adapter"].update(
        {"status": "qualified_data_blocked", "qualification": "report.json"}
    )

    assert adapter_contract_identity(copied) == before
