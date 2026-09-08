import gzip
import hashlib
import json
from pathlib import Path

from speck.common_pile_code import (
    sample_common_pile_code,
    validate_common_pile_code_config,
)

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row(text, index, **metadata_overrides):
    raw = text.encode()
    metadata = {
        "blob_id": hashlib.sha1(raw).hexdigest(),
        "content_id": "f" * 40,
        "language": "TypeScript",
        "license_type": "permissive",
        "detected_licenses": ["MIT"],
        "is_vendor": False,
        "is_generated": False,
        "src_encoding": "UTF-8",
        "length_bytes": len(raw),
        "repo_name": f"example/repo-{index}",
        "path": f"src/file-{index}.ts",
        "github_id": index,
        "revision_id": "a" * 40,
        **metadata_overrides,
    }
    return {
        "source": "stackv2",
        "text": text,
        "score": 4.5,
        "int_score": 4,
        "metadata": metadata,
    }


def _config(tmp_path):
    rows = [
        _row(
            "// This English comment describes a deterministic TypeScript function.\nconst value = 1;",
            1,
        ),
        _row(
            "// This record is rejected by its detected license.\nconst other = 2;",
            2,
            detected_licenses=["CC0-1.0"],
        ),
    ]
    path = tmp_path / "source.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {
        "format": "speck_common_pile_code_sample",
        "format_version": 1,
        "status": "bounded_sample_authorized_not_training_authority",
        "source": {
            "repo": "common-pile/stackv2_edu_filtered",
            "revision": "a" * 40,
            "official_url": "https://huggingface.co/datasets/common-pile/stackv2_edu_filtered",
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(rows),
            "source_value": "stackv2",
        },
        "filters": {
            "language": "TypeScript",
            "minimum_integer_score": 4,
            "license_type": "permissive",
            "accepted_detected_licenses": ["MIT"],
            "encoding": "UTF-8",
            "exclude_vendor": True,
            "exclude_generated": True,
            "min_file_bytes": 1,
            "max_file_bytes": 1_000,
            "max_bytes_per_repository": 1_000,
            "sample_bytes": 1,
            "English_prose": {
                "detector": "py3langid==0.3.0",
                "minimum_alphabetic_characters": 10,
                "minimum_probability": 0.5,
            },
        },
        "output_directory": str(tmp_path / "output"),
    }


def test_common_pile_code_sample_filters_and_preserves_inline_content(tmp_path):
    config = validate_common_pile_code_config(_config(tmp_path))
    report = sample_common_pile_code(config)
    output = Path(config["output_directory"])

    assert report["status"] == "bounded_sample_complete_not_training_authority"
    assert report["counts"]["source_rows"] == 2
    assert report["counts"]["records_sampled"] == 1
    assert report["gates"]["training_authority"] == "blocked"
    record = json.loads((output / "tokenizer-input.jsonl").read_text())
    attribution = json.loads((output / "attribution.jsonl").read_text())
    assert record["source"] == "common_pile_stackv2_edu"
    assert record["content_id"] == hashlib.sha1(record["text"].encode()).hexdigest()
    assert "text" not in attribution


def test_real_common_pile_code_plan_is_pinned_and_bounded():
    path = ROOT / "research" / "flagship" / "common_pile_code_sample_v1.json"
    config = validate_common_pile_code_config(json.loads(path.read_text()), config_dir=path.parent)

    assert config["source"]["revision"] == "c354dbe88469a1153e97c6a63ac50591849654de"
    assert config["source"]["rows"] == 494_753
    assert config["filters"]["language"] == "TypeScript"
    assert config["filters"]["sample_bytes"] == 12_000_000
