import json

from scripts.helmet_trec_rights_audit import (
    canonical_visible_text,
    loader_analysis,
    protocol_identity,
)


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


def test_visible_text_ignores_transport_scripts_and_link_attributes():
    first = '<a href="volatile-a">Contact</a><script>token-a</script><p>Terms text</p>'
    second = '<a href="volatile-b">Contact</a><script>token-b</script><p>Terms text</p>'
    assert canonical_visible_text(first) == canonical_visible_text(second)
