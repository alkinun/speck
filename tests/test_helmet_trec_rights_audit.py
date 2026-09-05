import json

from scripts.helmet_trec_rights_audit import loader_analysis, protocol_identity


def test_loader_analysis_detects_urls_without_license(tmp_path):
    path = tmp_path / "trec.py"
    path.write_text(
        '_URLs = {"train": "https://example/train", "test": "https://example/test"}\n',
        encoding="utf-8",
    )
    assert loader_analysis(path) == {
        "source_urls": {
            "train": "https://example/train",
            "test": "https://example/test",
        },
        "license_assignment_present": False,
    }


def test_trec_protocol_identity_excludes_registration_fields():
    frozen = {"format": "protocol", "status": "frozen", "source": {"id": "trec"}}
    executed = json.loads(json.dumps(frozen))
    executed.update({"status": "executed", "result": {"sha256": "abc"}})
    assert protocol_identity(frozen) == protocol_identity(executed)
