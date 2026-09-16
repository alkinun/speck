import json
from pathlib import Path

import pytest

from speck.data.document_lengths import census
from speck.provenance.io import durable_json, file_sha256


@pytest.fixture
def stock(tmp_path):
    rows = []
    position = 0
    for i, count in enumerate((4096, 4097, 32769, 65537, 131073)):
        rows.append(
            dict(
                ordinal=i,
                token_start=position,
                token_count=count,
                utf8_bytes=count * 3,
                content_id=f"fixture/{i}",
                released_content_sha256="a" * 64,
            )
        )
        position += count
    index = tmp_path / "documents.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = dict(
        format="speck_document_token_stock",
        status="complete_document_token_cache_not_training_view",
        training_authority=False,
        documents=dict(path=index.name, sha256=file_sha256(index)),
        document_count=len(rows),
        token_count=position,
        plan=dict(source_id="fixture", category="reference", tokenizer={"sha256": "fixture"}),
    )
    path = tmp_path / "manifest.json"
    durable_json(path, manifest)
    return {"path": str(path), "sha256": file_sha256(path)}, manifest, rows


def test_whole_document_capacity_needs_lookahead_and_never_concatenates(stock):
    binding, _, _ = stock
    result = census(binding)
    assert result["context_capacity"]["4096"]["documents_at_least_context_plus_one"] == 4
    assert result["context_capacity"]["131072"]["documents_at_least_context_plus_one"] == 1
    assert result["context_capacity"]["65536"]["nonoverlapping_input_windows_with_lookahead"] == 3
    assert result["maximum_document_tokens"] == 131073
    assert result["qualified_coherent_family_tokens"] is None


@pytest.mark.parametrize("change", ["ordinal", "start", "count", "hash", "total"])
def test_changed_or_inconsistent_index_rejected(stock, change):
    binding, manifest, rows = stock
    path = Path(binding["path"])
    index = path.parent / "documents.jsonl"
    if change == "ordinal":
        rows[1]["ordinal"] = 0
    elif change == "start":
        rows[1]["token_start"] += 1
    elif change == "count":
        rows[0]["token_count"] = -1
    elif change == "total":
        manifest["token_count"] += 1
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest["documents"]["sha256"] = "changed" if change == "hash" else file_sha256(index)
    durable_json(path, manifest)
    binding["sha256"] = file_sha256(path)
    with pytest.raises(ValueError):
        census(binding)
