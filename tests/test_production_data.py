import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import speck.production_data as production_data
from scripts.production_data_preprocess_batched import _batched_signature
from speck.code_near_duplicates import _signature
from speck.production_data import _candidate_text, preprocess_sources, validate_preprocess_config

ROOT = Path(__file__).parents[1]


def test_batched_cli_signature_is_exactly_equivalent_to_scalar_updates():
    shingles = {f"token-{index}".encode() for index in range(100)}

    scalar = _signature(shingles, 128, 42)
    batched = _batched_signature(shingles, 128, 42)

    assert scalar.hashvalues.tolist() == batched.hashvalues.tolist()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(text, *, url=None, host=None, content_id=None):
    return {
        "text": text,
        "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "url": url,
        "host": host,
        "content_id": content_id,
    }


def _config(tmp_path, output="production"):
    shared = "A globally exact duplicate with enough distinct tokens for production processing."
    base = "This technical reference document explains deterministic global near duplicate processing with many distinct lexical tokens and a stable conclusion."
    near = base.replace("conclusion", "result")
    rows = [
        [
            _record(shared),
            _record(base),
            _record(
                "Denied domain document with enough distinct words.",
                url="https://blocked.example/page",
                host="blocked.example",
            ),
        ],
        [
            _record(shared),
            _record(near),
            _record(
                "An independent retained document with unrelated vocabulary and careful provenance."
            ),
        ],
    ]
    sources = []
    for index, records in enumerate(rows):
        path = tmp_path / f"source-{index}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        sources.append(
            {
                "id": f"source_{index}",
                "precedence": index + 1,
                "path": str(path),
                "sha256": _sha256(path),
                "text_field": "text",
                "content_sha256_field": "released_content_sha256",
                "url_field": "url",
                "domain_field": "host",
                "blob_field": "content_id",
            }
        )
    ledger = tmp_path / "deny.json"
    ledger.write_text(
        json.dumps(
            {
                "format": "speck_removal_deny_ledger",
                "format_version": 1,
                "status": "human_reviewed_deny_entries",
                "entries": [
                    {
                        "kind": "domain",
                        "value": "blocked.example",
                        "reason": "fixture removal request",
                        "authority": "fixture human",
                        "recorded_at": "2026-09-07T00:00:00Z",
                    }
                ],
            }
        )
    )
    transient = tmp_path / f"transient-{output}.bin"
    transient.write_bytes(b"temporary download")
    return {
        "format": "speck_production_text_preprocess",
        "format_version": 1,
        "status": "fixture_or_rehearsal_authorized_not_training_authority",
        "sources": sources,
        "deny_ledger": {"path": str(ledger), "sha256": _sha256(ledger)},
        "policy": {
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
        },
        "checkpoint_records": 2,
        "cleanup_files": [{"path": str(transient), "sha256": _sha256(transient)}],
        "output_directory": str(tmp_path / output),
    }


def test_global_exact_near_deny_and_cleanup_are_recorded(tmp_path):
    config = validate_preprocess_config(_config(tmp_path))
    result = preprocess_sources(config)
    manifest = result["manifest"]
    removals = [
        json.loads(line)
        for line in (Path(config["output_directory"]) / manifest["removals"]["path"])
        .read_text()
        .splitlines()
    ]

    assert manifest["counts"] == {
        "records_removed_deny_ledger": 1,
        "records_removed_exact": 1,
        "records_removed_near": 1,
        "records_retained": 3,
        "records_seen": 6,
    }
    assert {record["reason"] for record in removals} == {
        "domain",
        "exact_duplicate",
        "near_duplicate",
    }
    assert all("text" not in record for record in removals)
    assert result["cleanup"]["status"].endswith("after_published_manifest")
    assert not Path(config["cleanup_files"][0]["path"]).exists()
    assert manifest["index"]["documents"] == manifest["counts"]["records_retained"]
    assert manifest["gates"]["training_authority"] == "blocked"


def test_record_checkpoint_resume_matches_uninterrupted_outputs(tmp_path):
    interrupted = validate_preprocess_config(_config(tmp_path, "interrupted"))
    with pytest.raises(RuntimeError, match="injected production preprocess crash"):
        preprocess_sources(interrupted, crash_after_records=3)
    resumed = preprocess_sources(interrupted)
    clean = validate_preprocess_config(_config(tmp_path, "clean"))
    uninterrupted = preprocess_sources(clean)

    for source_id in resumed["manifest"]["outputs"]:
        left = (
            Path(interrupted["output_directory"])
            / resumed["manifest"]["outputs"][source_id]["path"]
        )
        right = (
            Path(clean["output_directory"])
            / uninterrupted["manifest"]["outputs"][source_id]["path"]
        )
        assert _sha256(left) == _sha256(right)
    assert resumed["manifest"]["index"]["sha256"] == uninterrupted["manifest"]["index"]["sha256"]
    left = Path(interrupted["output_directory"]) / "removals.jsonl"
    right = Path(clean["output_directory"]) / "removals.jsonl"
    assert _sha256(left) == _sha256(right)


def test_tampered_deny_ledger_and_protected_cleanup_fail_closed(tmp_path):
    raw = _config(tmp_path)
    raw["cleanup_files"] = [
        {"path": raw["sources"][0]["path"], "sha256": raw["sources"][0]["sha256"]}
    ]
    with pytest.raises(ValueError, match="cannot include source"):
        validate_preprocess_config(raw)

    raw = _config(tmp_path, "tampered")
    config = validate_preprocess_config(raw)
    Path(config["deny_ledger"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="deny ledger identity"):
        preprocess_sources(config)


def test_missing_or_output_local_cleanup_file_is_rejected(tmp_path):
    raw = _config(tmp_path, "cleanup")
    Path(raw["cleanup_files"][0]["path"]).unlink()
    config = validate_preprocess_config(raw)
    with pytest.raises(ValueError, match="identity mismatch before build"):
        preprocess_sources(config)

    raw = _config(tmp_path, "output-local")
    output = Path(raw["output_directory"])
    local = output / "raw.tmp"
    local.parent.mkdir()
    local.write_bytes(b"raw")
    raw["cleanup_files"] = [{"path": str(local), "sha256": _sha256(local)}]
    with pytest.raises(ValueError, match="inside output"):
        validate_preprocess_config(raw)


def test_resume_detects_corruption_inside_committed_output_slice(tmp_path):
    config = validate_preprocess_config(_config(tmp_path, "corrupt-resume"))
    with pytest.raises(RuntimeError, match="injected production preprocess crash"):
        preprocess_sources(config, crash_after_records=3)
    path = tmp_path / "corrupt-resume.building/source_0.jsonl"
    with path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))

    with pytest.raises(ValueError, match="checkpoint slice checksum mismatch"):
        preprocess_sources(config)


def test_reopening_published_output_detects_output_and_index_corruption(tmp_path):
    output_config = validate_preprocess_config(_config(tmp_path, "published-output"))
    result = preprocess_sources(output_config)
    output = Path(output_config["output_directory"])
    source = next(iter(result["manifest"]["outputs"].values()))
    path = output / source["path"]
    with path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="output identity mismatch"):
        preprocess_sources(output_config)

    index_config = validate_preprocess_config(_config(tmp_path, "published-index"))
    result = preprocess_sources(index_config)
    index = Path(index_config["output_directory"]) / result["manifest"]["index"]["path"]
    with index.open("r+b") as handle:
        handle.seek(100)
        value = handle.read(1)
        handle.seek(100)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="index identity mismatch"):
        preprocess_sources(index_config)

    receipt_config = validate_preprocess_config(_config(tmp_path, "published-receipt"))
    preprocess_sources(receipt_config)
    receipt = Path(receipt_config["output_directory"]) / "cleanup_receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="cleanup receipt is invalid"):
        preprocess_sources(receipt_config)


def test_candidate_text_reuses_one_source_handle(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    first = json.dumps({"text": "first candidate"}) + "\n"
    source.write_text(first + json.dumps({"text": "second candidate"}) + "\n")
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE docs (doc_seq INTEGER PRIMARY KEY, source_index INTEGER, byte_offset INTEGER)"
    )
    connection.execute("INSERT INTO docs VALUES (1, 0, 0)")
    connection.execute("INSERT INTO docs VALUES (2, 0, ?)", (len(first.encode()),))
    handles = {}
    opened = []
    original_open = Path.open

    def track_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == source and args == ("rb",):
            opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", track_open)
    try:
        sources = [{"path": str(source), "text_field": "text"}]
        assert _candidate_text(connection, sources, 1, handles) == "first candidate"
        assert _candidate_text(connection, sources, 2, handles) == "second candidate"
        assert len(opened) == 1
        assert handles[0] is opened[0]
    finally:
        connection.close()
        for handle in handles.values():
            handle.close()


def test_candidate_source_handles_close_when_preprocessing_fails(tmp_path, monkeypatch):
    config = validate_preprocess_config(_config(tmp_path, "candidate-handle-failure"))
    original_candidate_text = production_data._candidate_text
    captured = []

    def fail_after_open(connection, sources, doc_seq, handles):
        original_candidate_text(connection, sources, doc_seq, handles)
        captured.extend(handles.values())
        raise RuntimeError("injected candidate comparison failure")

    monkeypatch.setattr(production_data, "_candidate_text", fail_after_open)
    with pytest.raises(RuntimeError, match="injected candidate comparison failure"):
        preprocess_sources(config)

    assert captured
    assert all(handle.closed for handle in captured)


def test_flagship_production_plan_keeps_rehearsal_and_authority_pending():
    plan = json.loads((ROOT / "research/flagship/production_data_plan.json").read_text())

    assert plan["status"] == "fixture_tooling_ready_20B_rehearsal_pending_not_training_authority"
    assert plan["preprocessor"]["candidate_policy_for_rehearsal"]["num_perm"] == 128
    assert (
        plan["preprocessor"]["candidate_policy_for_rehearsal"]["verified_jaccard_threshold"] == 0.8
    )
    assert plan["rehearsal_20B"]["status"] == "pending"
    assert plan["production_authority_record"]["status"] == (
        "must not be issued from fixture evidence"
    )
    assert plan["training_authority"] == "blocked"
