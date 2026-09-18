import hashlib
import json

import pytest

from speck.data.corpus_audit import audit_source, build_audit, diagnostic_flags
from speck.provenance.io import file_sha256


def source(tmp_path, indexed=True):
    texts = ["short " + str(i) for i in range(10)] + ["λ" * 2000, "x" * 10000, "y" * 40000]
    rows = [
        {
            "content_id": str(i),
            "text": text,
            "language": "Python",
            "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        for i, text in enumerate(texts)
    ]
    raw = tmp_path / "text.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec = {
        "id": "fixture",
        "kind": "jsonl",
        "path": str(raw),
        "sha256": file_sha256(raw),
        "scope": "fixture",
    }
    if indexed:
        offset, spans = 0, []
        for i, row in enumerate(rows):
            size = len(row["text"].encode())
            spans.append(
                {
                    "ordinal": i,
                    "content_id": str(i),
                    "token_start": offset,
                    "token_count": size + 2,
                    "utf8_bytes": size,
                    "released_content_sha256": row["released_content_sha256"],
                }
            )
            offset += size + 2
        index = tmp_path / "index.jsonl"
        index.write_text("".join(json.dumps(row) + "\n" for row in spans))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete_document_token_cache_not_training_view",
                    "plan": {"input": {"path": str(raw), "sha256": file_sha256(raw)}},
                    "documents": {"path": index.name, "sha256": file_sha256(index)},
                    "document_count": len(rows),
                    "token_count": offset,
                }
            )
        )
        spec.update(kind="token_stock", path=str(manifest), sha256=file_sha256(manifest))
    return spec


def test_samples_cover_length_strata_and_match_hash_ranking(tmp_path):
    spec = source(tmp_path)
    report, samples = audit_source(spec, seed=42, per_stratum=2)
    expected = sorted(range(10), key=lambda n: hashlib.sha256(f"42:fixture:{n}".encode()).digest())[
        :2
    ]
    assert {s["ordinal"] for s in samples} == {*expected, 10, 11, 12}
    assert report["documents"] == 13
    assert len(report["strata"]) == 4
    assert report["strata"]["all/bytes_le_8192"]["utf8_bytes"] == 4000
    assert report["strata"]["all/bytes_gt_32768"]["documents_over_4096_tokens"] == 1
    assert audit_source(spec, 42, 2) == (report, samples)


@pytest.mark.parametrize("filename", ["text.jsonl", "index.jsonl", "manifest.json"])
def test_mutated_inputs_fail_closed(tmp_path, filename):
    spec = source(tmp_path)
    with (tmp_path / filename).open("a") as handle:
        handle.write(" ")
    with pytest.raises((ValueError, KeyError)):
        audit_source(spec, 42, 2)


def test_unindexed_code_keeps_language_and_does_not_invent_tokens(tmp_path):
    spec = source(tmp_path, indexed=False)
    report, samples = audit_source(spec, 42, 2)
    assert all(key.startswith("Python/") for key in report["strata"])
    assert not report["token_counts_available"]
    assert all(s["token_count"] is None for s in samples)
    assert all("tokens" not in row for row in report["strata"].values())


def test_report_is_not_quality_approval_and_preserves_existing_output(tmp_path):
    spec = source(tmp_path)
    plan = {"seed": 42, "per_stratum": 2, "sources": [spec]}
    output = tmp_path / "audit"
    result = build_audit(plan, output)
    assert result["status"] == "sampled_not_quality_approved"
    assert result["sample_documents"] == 5
    assert result["samples"]["sha256"] == file_sha256(output / "samples.jsonl")
    with pytest.raises(FileExistsError):
        build_audit(plan, output)


def test_review_flags_are_narrow_hints():
    assert diagnostic_flags("A plain mathematical explanation.") == []
    assert "replacement_character" in diagnostic_flags("bad \ufffd extraction")
    assert "repeated_long_lines" in diagnostic_flags(
        ("a repeated line longer than thirty characters\n") * 5
    )
