import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.common_pile_peps import sample_common_pile_peps, validate_peps_config
from speck.python_edu import sample_python_edu, validate_python_edu_config
from speck.stack_v3_refine import _sample_partition

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _partitioned_texts(prefix):
    found = {}
    settings = {
        "seed": 42,
        "category": "code",
        "modulus": 2,
        "evaluation_remainders": [0],
    }
    for index in range(100):
        text = f"{prefix} {index} " + "clear technical prose and example code " * 20
        split = _sample_partition({"text": text}, settings)
        found.setdefault(split, text)
        if set(found) == {"train", "eval"}:
            return found
    raise AssertionError("failed to construct partition fixtures")


def test_pep_sample_preserves_license_and_both_partitions(tmp_path):
    texts = _partitioned_texts("Python enhancement proposal")
    source = tmp_path / "peps.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        for index, text in enumerate(texts.values(), 1):
            handle.write(
                json.dumps(
                    {
                        "id": str(index),
                        "source": "python-peps",
                        "text": text,
                        "metadata": {
                            "license": "Public Domain",
                            "pep_number": str(index),
                            "url": f"https://peps.python.org/pep-{index:04d}/",
                            "authors": ["Example"],
                            "provenance": f"fixture:{index}",
                        },
                    }
                )
                + "\n"
            )
    config = validate_peps_config(
        {
            "format": "speck_common_pile_peps_sample",
            "format_version": 1,
            "status": "bounded_sample_authorized_not_training_authority",
            "source": {
                "repo": "example/peps",
                "revision": "a" * 40,
                "official_url": "https://example.com/peps",
                "path": str(source),
                "sha256": _sha256(source),
                "rows": 2,
                "source_value": "python-peps",
            },
            "filters": {
                "accepted_metadata_license": "Public Domain",
                "min_document_bytes": 10,
                "max_document_bytes": 10_000,
                "English_prose": {
                    "detector": "py3langid==0.3.0",
                    "minimum_alphabetic_characters": 20,
                    "minimum_probability": 0.5,
                },
                "downstream_partition": {
                    "seed": 42,
                    "category": "code",
                    "modulus": 2,
                    "evaluation_remainders": [0],
                    "training_bytes": 1,
                    "evaluation_bytes": 1,
                },
            },
            "output_directory": str(tmp_path / "pep-output"),
        }
    )
    report = sample_common_pile_peps(config)

    assert report["status"] == "bounded_sample_complete_not_training_authority"
    assert report["counts"]["records_sampled"] == 2
    assert report["downstream_partition"]["observed"]["train_records"] == 1
    assert report["downstream_partition"]["observed"]["eval_records"] == 1
    assert report["gates"]["training_authority"] == "blocked"


def test_python_edu_fetches_hash_bound_blobs_and_keeps_rights_blocked(tmp_path):
    texts = _partitioned_texts("def educational_example")
    raw_values = [text.encode() for text in texts.values()]
    rows = [
        {
            "blob_id": hashlib.sha1(raw).hexdigest(),
            "repo_name": f"example/repo-{index}",
            "path": f"example_{index}.py",
            "length_bytes": len(raw),
            "score": 4.0,
            "int_score": 4,
        }
        for index, raw in enumerate(raw_values)
    ]
    source = tmp_path / "python.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    blobs = {row["blob_id"]: raw for row, raw in zip(rows, raw_values)}
    config = validate_python_edu_config(
        {
            "format": "speck_python_edu_sample",
            "format_version": 1,
            "status": "bounded_sample_authorized_not_training_authority",
            "source": {
                "repo": "example/python-edu",
                "revision": "b" * 40,
                "official_url": "https://example.com/python-edu",
                "path": str(source),
                "sha256": _sha256(source),
                "rows": 2,
            },
            "rights": {
                "dataset_license": "ODC-By-1.0",
                "source_lineage": "example",
                "file_license_metadata": "absent",
                "authority": "manual_review_required",
            },
            "filters": {
                "minimum_integer_score": 4,
                "min_file_bytes": 10,
                "max_file_bytes": 10_000,
                "max_bytes_per_repository": 10_000,
                "sample_bytes": sum(map(len, raw_values)),
                "English_prose": {
                    "detector": "py3langid==0.3.0",
                    "minimum_alphabetic_characters": 20,
                    "minimum_probability": 0.5,
                },
            },
            "fetch": {
                "base_url": "https://softwareheritage.s3.amazonaws.com/content/",
                "workers": 2,
                "timeout_seconds": 1,
                "attempts": 1,
            },
            "downstream_partition": {
                "seed": 42,
                "category": "code",
                "modulus": 2,
                "evaluation_remainders": [0],
                "training_bytes": 1,
                "evaluation_bytes": 1,
            },
            "output_directory": str(tmp_path / "python-output"),
        }
    )
    report = sample_python_edu(
        config, fetch_blob=lambda blob_id: ("ok", blobs[blob_id])
    )

    assert report["status"] == "bounded_sample_complete_not_training_authority"
    assert report["counts"]["blob_fetch_ok"] == 2
    assert report["counts"]["records_sampled"] == 2
    assert report["gates"]["file_level_license_evidence"] == "blocked_metadata_absent"
    assert report["gates"]["training_authority"] == "blocked"


def test_real_supplement_plans_keep_the_frozen_code_allocation():
    peps_path = ROOT / "research/flagship/common_pile_peps_sample_v1.json"
    python_path = ROOT / "research/flagship/python_edu_sample_v1.json"
    peps = validate_peps_config(json.loads(peps_path.read_text()), config_dir=peps_path.parent)
    python = validate_python_edu_config(
        json.loads(python_path.read_text()), config_dir=python_path.parent
    )

    assert peps["filters"]["downstream_partition"]["training_bytes"] == 5_000_000
    assert peps["filters"]["downstream_partition"]["evaluation_bytes"] == 500_000
    assert python["downstream_partition"]["training_bytes"] == 10_000_000
    assert python["downstream_partition"]["evaluation_bytes"] == 1_000_000
    assert python["rights"]["file_license_metadata"] == "absent"


def test_recorded_code_supplements_bind_code_configs_and_technical_scope():
    result = json.loads(
        (ROOT / "results/data/code-tokenizer-supplements-20260907.json").read_text()
    )
    assert result["status"] == (
        "bounded_five_source_code_slice_technical_pass_training_authority_blocked"
    )
    for key, value in result["implementation"].items():
        if key.endswith("_sha256"):
            continue
        assert result["implementation"][f"{key}_sha256"] == _sha256(ROOT / value)
    for path, digest in result["configs"].values():
        assert digest == _sha256(ROOT / path)
    python = result["sources"]["python_edu"]
    peps = result["sources"]["common_pile_python_peps"]
    assert python["final_training_bytes"] >= python["training_target_bytes"]
    assert python["final_evaluation_bytes"] >= python["evaluation_target_bytes"]
    assert peps["final_training_bytes"] >= peps["training_target_bytes"]
    assert peps["final_evaluation_bytes"] >= peps["evaluation_target_bytes"]
    assert "blocked" in python["rights_gate"]
    overlap = result["five_source_overlap"]
    assert overlap["retained_documents"] == 40_869
    assert overlap["exact_cross_source_matches"] == 0
    assert overlap["verified_near_cross_source_matches"] == 0
