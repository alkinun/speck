import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from speck.data.joint_graph import build
from speck.data.production_data import preprocess_sources, validate_preprocess_config

SHARED = "A globally exact duplicate shared by two retained stocks with enough distinct tokens."
BASE = (
    "This technical reference document explains deterministic joint family graph "
    "construction with many distinct lexical tokens and a stable conclusion."
)
POLICY = {
    "normalization": "NFKC+lower+lexical-code-tokens",
    "token_pattern": "[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\\s]",
    "shingle_tokens": 3,
    "minimum_document_tokens": 6,
    "maximum_document_tokens": 1000,
    "num_perm": 32,
    "minhash_seed": 42,
    "bands": 32,
    "verified_jaccard_threshold": 0.8,
    "domain_match": "exact_or_subdomain",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _entry(path):
    return {"path": str(path), "sha256": _sha256(path)}


def _stock(tmp_path, name, texts, policy=POLICY):
    """Run one real single-source preprocess pass and index its output as a token stock."""

    directory = tmp_path / name
    directory.mkdir()
    source = directory / "input.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "text": text,
                    "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "url": None,
                    "host": None,
                    "content_id": None,
                }
            )
            + "\n"
            for text in texts
        )
    )
    ledger = directory / "deny.json"
    ledger.write_text(
        json.dumps(
            {
                "format": "speck_removal_deny_ledger",
                "format_version": 1,
                "status": "human_reviewed_deny_entries",
                "entries": [],
            }
        )
    )
    config = validate_preprocess_config(
        {
            "format": "speck_production_text_preprocess",
            "format_version": 1,
            "status": "fixture_or_rehearsal_authorized_not_training_authority",
            "sources": [
                {
                    "id": "acquired",
                    "precedence": 1,
                    "path": str(source),
                    "sha256": _sha256(source),
                    "text_field": "text",
                    "content_sha256_field": "released_content_sha256",
                    "url_field": "url",
                    "domain_field": "host",
                    "blob_field": "content_id",
                }
            ],
            "deny_ledger": _entry(ledger),
            "policy": policy,
            "checkpoint_records": 2,
            "cleanup_files": [],
            "output_directory": str(directory / "excluded"),
        }
    )
    manifest = preprocess_sources(config)["manifest"]
    # Completed production passes are checkpointed; the graph reads them immutably.
    with sqlite3.connect(directory / "excluded" / "near_duplicates.sqlite3") as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    output = directory / "excluded" / manifest["outputs"]["acquired"]["path"]
    stock = directory / "stock"
    stock.mkdir()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    (stock / "documents.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "released_content_sha256": row["released_content_sha256"],
                    "token_count": len(row["text"].split()),
                }
            )
            + "\n"
            for row in rows
        )
    )
    (stock / "manifest.json").write_text(
        json.dumps(
            {
                "documents": {"path": "documents.jsonl"},
                "plan": {"input": _entry(output)},
            }
        )
    )
    return {
        "id": name,
        "preprocess_manifest": _entry(directory / "excluded" / "manifest.json"),
        "preprocess_source": "acquired",
        "stock_manifest": _entry(stock / "manifest.json"),
    }


def _plan(*sources):
    return {
        "format": "speck_joint_family_graph_plan",
        "format_version": 1,
        "sources": list(sources),
    }


def test_cross_source_exact_and_near_duplicates_form_families(tmp_path):
    first = _stock(
        tmp_path,
        "first",
        [SHARED, BASE, "An unrelated first-stock document about harbour tides and ropes."],
    )
    second = _stock(
        tmp_path,
        "second",
        [
            SHARED,
            BASE.replace("conclusion", "result"),
            "A separate second-stock text concerning orchard soil and pruning.",
        ],
    )

    report = build(_plan(first, second), tmp_path / "graph")

    assert report["edges"]["exact"] == 1
    assert report["edges"]["near"] == 1
    assert report["linked_documents"] == {"first": 2, "second": 2}
    assert report["families"] == {"count": 2, "largest": 2, "size_histogram": {"2": 2}}
    assert report["gates"]["training_authority"] == "blocked"
    edges = [
        json.loads(line) for line in (tmp_path / "graph" / "edges.jsonl").read_text().splitlines()
    ]
    assert {edge["kind"] for edge in edges} == {"exact", "near"}
    assert report["edges"]["sha256"] == _sha256(tmp_path / "graph" / "edges.jsonl")


def test_sources_under_different_policies_are_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED, BASE])
    second = _stock(tmp_path, "second", [SHARED, BASE], policy={**POLICY, "shingle_tokens": 4})

    with pytest.raises(ValueError, match="policy differs"):
        build(_plan(first, second), tmp_path / "graph")


def test_a_stock_not_built_from_the_preprocess_output_is_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED, BASE])
    second = _stock(tmp_path, "second", [SHARED, BASE])
    second["stock_manifest"] = first["stock_manifest"]

    with pytest.raises(ValueError, match="not the preprocess output"):
        build(_plan(first, second), tmp_path / "graph")


def test_an_uncheckpointed_pass_database_is_rejected(tmp_path):
    first = _stock(tmp_path, "first", [SHARED, BASE])
    second = _stock(tmp_path, "second", [SHARED, BASE])
    wal = tmp_path / "second" / "excluded" / "near_duplicates.sqlite3-wal"
    wal.write_bytes(b"unmerged pages")

    with pytest.raises(ValueError, match="uncheckpointed log"):
        build(_plan(first, second), tmp_path / "graph")
