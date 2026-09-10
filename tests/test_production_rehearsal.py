import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import speck.production_rehearsal as production_rehearsal
from speck.code_near_duplicates import _signature
from speck.production_rehearsal import (
    _logical_sqlite_identity,
    run_production_rehearsal_stage,
    validate_production_rehearsal_plan,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"sources": [{"id": "source"}]}))
    rights = tmp_path / "rights.json"
    rights.write_text(
        json.dumps(
            {
                "format": "speck_human_source_rights_acceptance",
                "status": "all_sources_human_approved",
                "approved_source_ids": ["source"],
                "automated_approval_made": False,
            }
        )
    )
    deny = tmp_path / "deny.json"
    deny.write_text(
        json.dumps(
            {
                "format": "speck_removal_deny_ledger",
                "format_version": 1,
                "status": "human_reviewed_deny_entries",
                "entries": [],
            }
        )
    )
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fixture tokenizer identity")
    contamination = tmp_path / "contamination.json"
    contamination.write_text("{}")
    gitleaks = tmp_path / "gitleaks"
    gitleaks.write_bytes(b"fixture scanner identity")
    return {
        "format": "speck_production_rehearsal_plan",
        "format_version": 1,
        "status": "frozen_20B_rehearsal_not_training_authority",
        "target_tokens": 20_000_000_000,
        "seed": 42,
        "source_registry": {"path": str(registry), "sha256": _sha256(registry)},
        "rights_record": {"path": str(rights), "sha256": _sha256(rights)},
        "deny_ledger": {"path": str(deny), "sha256": _sha256(deny)},
        "tokenizer": {"path": str(tokenizer), "sha256": _sha256(tokenizer)},
        "contamination_plan": {
            "path": str(contamination),
            "sha256": _sha256(contamination),
        },
        "security": {
            "gitleaks_binary": {"path": str(gitleaks), "sha256": _sha256(gitleaks)},
            "gitleaks_version": "fixture",
            "maximum_duplicate_line_ratio": 0.5,
            "adult_host_terms": ["adult"],
            "allowed_emails": ["email@example.com"],
            "allowed_ipv4": ["127.0.0.1"],
        },
        "filtering": {"min_chars": 1, "max_chars": 1000},
        "partition": {
            "holdout_modulus": 20,
            "holdout_remainders": [0],
            "normalization": "NFKC+lower+whitespace",
        },
        "deduplication": {
            "normalization": "NFKC+lower+lexical-code-tokens",
            "token_pattern": "[A-Za-z]+|[^\\s]",
            "shingle_tokens": 3,
            "minimum_document_tokens": 3,
            "maximum_document_tokens": 1000,
            "num_perm": 32,
            "minhash_seed": 42,
            "bands": 8,
            "verified_jaccard_threshold": 0.8,
            "domain_match": "exact_or_subdomain",
            "checkpoint_records": 100,
        },
        "packing": {"shard_tokens": 1000, "acquisition_margin_percent": 1},
        "sources": [
            {
                "id": "source",
                "category": "web",
                "target_tokens": 20_000_000_000,
                "reader": {
                    "id": "source",
                    "repo": "fixture/source",
                    "revision": "a" * 40,
                    "tree_path": "data",
                    "content_column": "text",
                    "metadata_columns": {"url": "url"},
                    "filters": {},
                },
            }
        ],
        "output_directory": str(tmp_path / "output"),
    }


def test_plan_requires_exact_20b_approved_quotas(tmp_path):
    plan = validate_production_rehearsal_plan(_fixture(tmp_path))

    assert plan["target_tokens"] == 20_000_000_000
    assert sum(source["target_tokens"] for source in plan["sources"]) == plan["target_tokens"]
    assert len(plan["plan_fingerprint"]) == 64

    invalid = _fixture(tmp_path)
    invalid["sources"][0]["target_tokens"] -= 1
    with pytest.raises(ValueError, match="do not sum to 20B"):
        validate_production_rehearsal_plan(invalid)


def test_source_identity_and_acquisition_are_hash_bound_and_resumable(tmp_path, monkeypatch):
    plan = validate_production_rehearsal_plan(_fixture(tmp_path))
    monkeypatch.setattr(
        production_rehearsal,
        "discover_source_files",
        lambda source, seed: {
            "revision": source["revision"],
            "files": ["data/part.parquet"],
            "file_list_sha256": "b" * 64,
        },
    )

    class HugeTokens:
        def __len__(self):
            return 21_000_000_000

    class FakeTokenizer:
        def __init__(self, path):
            self.path = path

        def encode_batch(self, values, **kwargs):
            return [HugeTokens() for _ in values]

    monkeypatch.setattr(production_rehearsal, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(
        production_rehearsal,
        "_contamination_indexes",
        lambda plan: {"tasks": 1, "benchmarks": []},
    )
    monkeypatch.setattr(production_rehearsal, "_document_rejection", lambda *args: None)
    monkeypatch.setattr(
        production_rehearsal,
        "_gitleaks_filter",
        lambda path, binary, reports: (
            0,
            {"path": str(reports / "report.json"), "sha256": "c" * 64, "findings": 0},
        ),
    )
    monkeypatch.setattr(
        production_rehearsal,
        "iter_source_file_documents",
        lambda **kwargs: iter(
            [
                {
                    "content": "A real production-shaped fixture document.",
                    "metadata": {"url": "https://example.test/document"},
                }
            ]
        ),
    )
    identity_result = tmp_path / "identity.json"
    acquisition_result = tmp_path / "acquisition.json"
    run_production_rehearsal_stage(plan, "source_identity", identity_result)
    first = run_production_rehearsal_stage(plan, "acquisition", acquisition_result)
    second = run_production_rehearsal_stage(plan, "acquisition", acquisition_result)

    assert first == second
    assert first["stage_id"] == "acquisition"
    assert first["metrics"]["filtered_bytes"] > 0
    manifest = json.loads((Path(plan["output_directory"]) / "acquired/manifest.json").read_text())
    assert manifest["sources"]["source"]["counts"]["train_tokens"] >= 20_000_000_000


def test_unknown_stage_is_rejected_before_execution(tmp_path):
    plan = validate_production_rehearsal_plan(_fixture(tmp_path))
    with pytest.raises(ValueError, match="unknown production rehearsal stage"):
        run_production_rehearsal_stage(plan, "unknown", tmp_path / "result.json")


def test_batched_minhash_is_identical_to_frozen_scalar_update():
    from datasketch import MinHash

    shingles = {f"shingle-{index}".encode() for index in range(1_000)}
    scalar = _signature(shingles, 128, 42)
    batched = MinHash(num_perm=128, seed=42)
    batched.update_batch(shingles)

    assert (scalar.hashvalues == batched.hashvalues).all()


def test_logical_sqlite_identity_ignores_physical_insertion_order(tmp_path):
    paths = [tmp_path / "forward.sqlite3", tmp_path / "reverse.sqlite3"]
    rows = [
        (1, 0, 0, "source", 1, 0, "a" * 64, "b" * 64),
        (2, 1, 0, "source", 2, 100, "c" * 64, "d" * 64),
    ]
    for path, order in zip(paths, (rows, list(reversed(rows))), strict=True):
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE docs (doc_seq INTEGER PRIMARY KEY, processed_index INTEGER, source_index INTEGER, source_id TEXT, line_number INTEGER, byte_offset INTEGER, content_sha256 TEXT, dedup_sha256 TEXT)"
        )
        connection.execute("CREATE TABLE bands (band INTEGER, band_hash BLOB, doc_seq INTEGER)")
        connection.execute(
            "CREATE TABLE checkpoints (checkpoint_id INTEGER PRIMARY KEY, processed_records INTEGER, next_doc_seq INTEGER, index_chain TEXT)"
        )
        connection.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", order)
        connection.executemany("INSERT INTO bands VALUES (?, ?, ?)", [(0, b"x", 1), (0, b"y", 2)])
        connection.execute("INSERT INTO checkpoints VALUES (1, 2, 2, ?)", ("e" * 64,))
        connection.commit()
        connection.close()

    assert _logical_sqlite_identity(paths[0]) == _logical_sqlite_identity(paths[1])
