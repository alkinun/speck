import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.stack_edu import sample_stack_edu, validate_stack_edu_config

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metadata(path, language, contents):
    rows = []
    for index, content in enumerate(contents):
        raw = content.encode()
        rows.append(
            {
                "blob_id": hashlib.sha1(raw).hexdigest(),
                "language": language,
                "repo_name": f"example/{language.lower()}-{index}",
                "path": f"src/file-{index}",
                "src_encoding": "UTF-8",
                "length_bytes": len(raw),
                "score": 4.5,
                "int_score": 4,
                "detected_licenses": ["MIT"],
                "license_type": "permissive",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)
    return rows


def _config(tmp_path):
    source_rows = {}
    files = []
    texts = {
        "Rust": [
            "// This English comment explains safe ownership and borrowing behavior.\nfn main() {}"
        ],
        "Go": [
            "// This English comment explains a deterministic server implementation.\npackage main"
        ],
        "SQL": [
            "-- This English comment explains the customer query and expected rows.\nSELECT 1;"
        ],
    }
    for language, contents in texts.items():
        path = tmp_path / f"{language}.parquet"
        rows = _metadata(path, language, contents)
        source_rows.update(
            {row["blob_id"]: contents[index].encode() for index, row in enumerate(rows)}
        )
        files.append(
            {
                "language": language,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": len(rows),
            }
        )
    return {
        "format": "speck_stack_edu_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "repo": "HuggingFaceTB/stack-edu",
            "revision": "a" * 40,
            "official_url": "https://huggingface.co/datasets/HuggingFaceTB/stack-edu",
            "metadata_files": files,
        },
        "filters": {
            "minimum_integer_score": 4,
            "license_type": "permissive",
            "accepted_detected_licenses": ["MIT"],
            "accepted_encodings": ["UTF-8"],
            "min_file_bytes": 1,
            "max_file_bytes": 1_000,
            "max_bytes_per_language_per_repository": 1_000,
            "language_sample_bytes": {language: 1 for language in texts},
            "English_prose": {
                "detector": "py3langid==0.3.0",
                "minimum_alphabetic_characters": 10,
                "minimum_probability": 0.5,
            },
        },
        "fetch": {
            "base_url": "https://softwareheritage.s3.amazonaws.com/content/",
            "workers": 2,
            "timeout_seconds": 1,
            "attempts": 1,
        },
        "output_directory": str(tmp_path / "output"),
    }, source_rows


def test_stack_edu_sample_fetches_verified_blobs_and_preserves_attribution(tmp_path):
    raw, contents = _config(tmp_path)
    metadata_path = Path(raw["source"]["metadata_files"][0]["path"])
    table = pq.read_table(metadata_path)
    values = table.to_pylist()
    values[0]["length_bytes"] += 1
    pq.write_table(pa.Table.from_pylist(values), metadata_path)
    raw["source"]["metadata_files"][0]["sha256"] = _sha256(metadata_path)
    config = validate_stack_edu_config(raw)
    report = sample_stack_edu(
        config,
        fetch_blob=lambda blob_id: ("ok", contents[blob_id]),
    )
    output = Path(config["output_directory"])

    assert report["status"] == "bounded_sample_complete_not_training_authority"
    assert report["counts"]["blob_fetch_ok"] == 3
    assert report["counts"]["records_sampled"] == 3
    assert report["counts"]["content_length_metadata_mismatch"] == 1
    assert report["repositories"] == 3
    assert report["gates"]["training_authority"] == "blocked"
    records = [
        json.loads(line) for line in (output / "tokenizer-input.jsonl").read_text().splitlines()
    ]
    attribution = [
        json.loads(line) for line in (output / "attribution.jsonl").read_text().splitlines()
    ]
    assert {record["language"] for record in records} == {"Rust", "Go", "SQL"}
    assert all("text" not in record for record in attribution)
    assert all(record["source"] == "stack_edu" for record in records)


def test_real_stack_edu_plan_is_bounded_and_pins_three_metadata_files():
    path = ROOT / "research" / "flagship" / "stack_edu_sample_v1.json"
    config = validate_stack_edu_config(json.loads(path.read_text()), config_dir=path.parent)

    assert config["source"]["revision"] == "eeec5caac5cc3758a18f1d3ba4416837a9ba814c"
    assert len(config["source"]["metadata_files"]) == 3
    assert sum(config["filters"]["language_sample_bytes"].values()) == 25_000_000
    assert config["filters"]["minimum_integer_score"] == 4
    assert config["fetch"]["workers"] == 32


def test_recorded_stack_edu_sample_binds_code_and_preserves_open_gates():
    result = json.loads(
        (ROOT / "results" / "data" / "stack-edu-bounded-sample-20260907.json").read_text()
    )
    implementation = result["implementation"]
    sample = result["sample"]

    assert (
        result["status"]
        == "bounded_sample_pass_gitleaks_exclusion_pending_training_authority_blocked"
    )
    assert implementation["config_sha256"] == _sha256(ROOT / implementation["config"])
    assert implementation["module_sha256"] == _sha256(ROOT / implementation["module"])
    assert implementation["cli_sha256"] == _sha256(ROOT / implementation["cli"])
    assert sample["records_sampled"] == 9_572
    assert sample["blob_fetch_missing"] == sample["blob_fetch_error"] == 0
    assert sample["content_hash_rejected"] == 0
    assert result["dedicated_secret_scan"]["affected_records"] == 2
    assert result["dedicated_secret_scan"]["exclusion_status"] == "pending_successor_filter"
    assert "cross-source exact and near-duplicate analysis" in result["blocked_gates"]
