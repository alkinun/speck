"""Supply accounting must not count duplicate text or miss transitive benchmark holds."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "code_supply_audit",
    Path(__file__).resolve().parents[1] / "experiments/corpus-audit/audit_code_supply.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(tmp_path):
    rows = []
    for index, (repo, sha, role, tokens) in enumerate(
        [
            ("held/project", "shared", "other", 10),
            ("fork/project", "shared", "test", 10),
            ("renamed/project", "separate", "docs_example", 20),
            ("unrelated/project", "unrelated", "other", 30),
        ]
    ):
        rows.append(
            {
                "id": str(index),
                "language": "Python",
                "repository": repo,
                "path": "file.py",
                "sha256": sha,
                "tokens": tokens,
                "utf8_bytes": 100,
                "role_hint": role,
                "commit_id": None,
                "license_labels": ["MIT"],
                "blob_sha1_matches_text": True,
                "generated_header_hint": False,
                "source_revision": "dataset-pin",
            }
        )
    acquisition = tmp_path / "acquisition.json"
    audit.save(
        acquisition,
        {
            "progress": {
                "by_language_before_full_exclusion": {"Python": {"documents": 4, "tokens": 70}}
            }
        },
    )
    qualification = tmp_path / "qualification.json"
    audit.save(
        qualification,
        {"held_repositories": ["held/project"], "aliases": [["fork/project", "renamed/project"]]},
    )
    plan = {
        "acquisition": audit.identity(acquisition),
        "qualification_inputs": audit.identity(qualification),
    }
    documents = tmp_path / "documents.jsonl"
    documents.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit.save(
        tmp_path / "scan.json",
        {
            "acquisition": plan["acquisition"],
            "script": audit.identity(audit.__file__),
            "documents": audit.identity(documents),
        },
    )
    return plan


def test_exact_copy_and_alias_holds_propagate_without_admission(tmp_path):
    plan = fixture(tmp_path)
    audit.summarize(plan, tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["tokens_with_bos_eos"] == 70
    assert result["exact_text"]["distinct_text_tokens"] == 60
    assert result["known_benchmark_repository_holds"] == {
        "direct_name_documents": 1,
        "direct_name_tokens": 10,
        "name_alias_exact_component_documents": 3,
        "name_alias_exact_component_tokens": 40,
    }
    assert result["eligible_unique_tokens_established"] == 0
    assert result["training_admitted"] is False


def test_changed_document_index_cannot_publish_metrics(tmp_path):
    plan = fixture(tmp_path)
    with (tmp_path / "documents.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="input identity mismatch"):
        audit.summarize(plan, tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_recount_must_reconcile_with_acquisition(tmp_path):
    plan = fixture(tmp_path)
    acquisition = tmp_path / "acquisition.json"
    value = json.loads(acquisition.read_text())
    value["progress"]["by_language_before_full_exclusion"]["Python"]["tokens"] = 71
    audit.save(acquisition, value)
    plan["acquisition"] = audit.identity(acquisition)
    scan = json.loads((tmp_path / "scan.json").read_text())
    scan["acquisition"] = plan["acquisition"]
    audit.save(tmp_path / "scan.json", scan)
    with pytest.raises(ValueError, match="language census differs"):
        audit.summarize(plan, tmp_path)
    assert not (tmp_path / "summary.json").exists()
