import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.megamath_code import sample_megamath_code, validate_megamath_code_config
from speck.stack_v3_refine import _sample_partition

ROOT = Path(__file__).parents[1]


def _row(raw, name):
    return {
        "file_info": {
            "blob_id": hashlib.sha1(raw).hexdigest(),
            "content_id": hashlib.sha1(b"content" + raw).hexdigest(),
            "detected_licenses": ["MIT"],
            "language": "Python",
            "length_bytes": len(raw),
            "license_type": "permissive",
            "path": f"/{name}.py",
            "src_encoding": "UTF-8",
        },
        "repo_info": {
            "repo_name": f"example/{name}",
            "repo_url": f"https://github.com/example/{name}",
        },
    }


def _raw_for(split):
    settings = {
        "seed": 42,
        "category": "math",
        "modulus": 2,
        "evaluation_remainders": [0],
    }
    for index in range(100):
        raw = f"# English mathematical implementation {index}\nx = sum(range(20))\n".encode()
        record = {"text": raw.decode()}
        if _sample_partition(record, settings) == split:
            return raw
    raise AssertionError("could not build partition fixture")


def _config(path, output):
    return {
        "format": "speck_megamath_code_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "id": "megamath_code",
            "repo": "LLM360/MegaMath",
            "revision": "a" * 40,
            "official_url": "https://example.com",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 2,
        },
        "rights": {
            "dataset_license": "ODC-By-1.0",
            "upstream_terms": "original file licenses apply",
            "authority": "manual_review_required",
            "redistribution": "undecided",
        },
        "filters": {
            "license_type": "permissive",
            "accepted_detected_licenses": ["MIT"],
            "accepted_encodings": ["UTF-8"],
            "min_file_bytes": 1,
            "max_file_bytes": 10000,
            "maximum_bytes_per_repository": 10000,
            "allowed_email_placeholders": [],
            "allowed_ipv4_placeholders": [],
            "English_prose": {
                "detector": "py3langid==0.3.0",
                "minimum_alphabetic_characters": 10,
                "minimum_probability": 0.8,
            },
        },
        "fetch": {
            "base_url": "https://softwareheritage.s3.amazonaws.com/content/",
            "workers": 1,
            "timeout_seconds": 1,
            "attempts": 1,
        },
        "downstream_partition": {
            "seed": 42,
            "category": "math",
            "modulus": 2,
            "evaluation_remainders": [0],
            "training_bytes": 1,
            "evaluation_bytes": 1,
        },
        "output_directory": str(output),
    }


def test_megamath_code_fetches_hash_verified_permissive_files(tmp_path):
    raws = [_raw_for("train"), _raw_for("eval")]
    rows = [_row(raw, str(index)) for index, raw in enumerate(raws)]
    path = tmp_path / "metadata.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    by_hash = {hashlib.sha1(raw).hexdigest(): raw for raw in raws}

    report = sample_megamath_code(
        validate_megamath_code_config(_config(path, tmp_path / "output")),
        fetch_blob=lambda blob: ("ok", by_hash[blob]),
    )

    assert report["counts"]["records_sampled"] == 2
    assert report["downstream_partition"]["observed"]["train_bytes"] >= 1
    assert report["downstream_partition"]["observed"]["eval_bytes"] >= 1
    assert report["accepted_licenses"] == {"MIT": 2}
    assert report["gates"]["training_authority"] == "blocked"


def test_flagship_megamath_code_plan_is_non_authoritative_and_margin_bound():
    path = ROOT / "research/flagship/math_megamath_code_v1.json"
    plan = validate_megamath_code_config(json.loads(path.read_text()), config_dir=path.parent)

    assert plan["source"]["rows"] == 847_441
    assert plan["filters"]["license_type"] == "permissive"
    assert plan["downstream_partition"]["training_bytes"] == 6_000_000
    assert plan["downstream_partition"]["evaluation_bytes"] == 600_000
    assert plan["rights"]["authority"] == "manual_review_required"
