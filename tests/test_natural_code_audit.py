"""Lineage must fail closed when otherwise well-formed evidence changes."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlencode

import pytest

SPEC = importlib.util.spec_from_file_location(
    "natural_code_audit",
    Path(__file__).resolve().parents[1] / "experiments/corpus-audit/audit_natural_code.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def evidence(tmp_path):
    source = b"def increment(value):\n    return value + 1\n"
    license_text = b"Example license evidence; no legal classification in this test.\n"
    revision = "1" * 40
    repo, path = "owner/library", "library/example.py"
    base = "https://api.github.com/repos/" + repo
    prefix = "https://raw.githubusercontent.com/" + repo + "/" + revision + "/"
    candidate = {
        "repo": repo,
        "path": path,
        "expected_sha256": audit.digest(source),
        "content_id": hashlib.sha1(source).hexdigest(),
    }
    record = {"repo_path": repo, "file_path": "/" + path, "text": source.decode()}
    requests = []

    def artifact(name, payload, url):
        target = tmp_path / name
        target.write_bytes(payload)
        result = {
            "path": str(target),
            "sha256": audit.digest(payload),
            "bytes": len(payload),
            "status": 200,
            "url": url,
        }
        requests.append(result)
        return result

    meta = artifact("meta", json.dumps({"full_name": repo}).encode(), base)
    commits = artifact(
        "commits",
        json.dumps([{"sha": revision}]).encode(),
        base + "/commits?" + urlencode({"path": path, "per_page": 5}),
    )
    upstream = artifact("source", source, prefix + path)
    tree = artifact(
        "tree",
        json.dumps(
            {
                "truncated": False,
                "tree": [
                    {"path": path, "type": "blob", "mode": "100644", "sha": audit.blob_sha(source)},
                    {
                        "path": "LICENSE",
                        "type": "blob",
                        "mode": "100644",
                        "sha": audit.blob_sha(license_text),
                    },
                ],
            }
        ).encode(),
        base + "/git/trees/" + revision + "?recursive=1",
    )
    notice = artifact("license", license_text, prefix + "LICENSE")
    notice.update(repository_path="LICENSE", git_blob_sha1=audit.blob_sha(license_text))
    acquired = {
        "candidate": copy.deepcopy(candidate),
        "requests": requests,
        "repository_metadata": meta,
        "commits": commits,
        "source": upstream,
        "matched_revision": revision,
        "tree": tree,
        "notices": [notice],
    }
    return candidate, record, acquired


def test_complete_evidence_is_ready_for_review(evidence):
    result = audit.check_lineage(*evidence)
    assert result["lineage_evidence_ready"]
    assert result["repository_family"] == "owner/library"
    assert "admitted" not in result


def test_missing_notice_is_held(evidence):
    evidence[2]["notices"] = []
    result = audit.check_lineage(*evidence)
    assert not result["lineage_evidence_ready"]
    assert "incomplete_tree_or_notice_evidence" in result["lineage_hold_reasons"]


def test_changed_payload_fails_hash(evidence):
    Path(evidence[2]["source"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit.check_lineage(*evidence)


def test_wrong_revision_url_rejected_even_with_valid_hash(evidence):
    source = evidence[2]["source"]
    source["url"] = source["url"].replace("1" * 40, "2" * 40)
    with pytest.raises(ValueError, match="URL mismatch"):
        audit.check_lineage(*evidence)


def test_wrong_archive_origin_rejected(evidence):
    evidence[1]["repo_path"] = "other/library"
    with pytest.raises(ValueError, match="archive origin mismatch"):
        audit.check_lineage(*evidence)


@pytest.mark.parametrize("change", ["blob", "truncated"])
def test_tree_binding_and_completeness(evidence, change):
    tree = evidence[2]["tree"]
    content = json.loads(Path(tree["path"]).read_text())
    if change == "blob":
        content["tree"][0]["sha"] = "0" * 40
    else:
        content["truncated"] = True
    payload = json.dumps(content).encode()
    Path(tree["path"]).write_bytes(payload)
    tree.update(sha256=audit.digest(payload), bytes=len(payload))
    if change == "blob":
        with pytest.raises(ValueError, match="source/tree blob mismatch"):
            audit.check_lineage(*evidence)
    else:
        assert not audit.check_lineage(*evidence)["lineage_evidence_ready"]
