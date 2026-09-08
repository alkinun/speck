import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.stack_v3 import qualify_stack_v3, validate_stack_v3_config

ROOT = Path(__file__).parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file(content, **overrides):
    encoded = content.encode()
    return {
        "content_id": hashlib.sha1(encoded).hexdigest(),
        "content": content,
        "size_bytes": len(encoded),
        "file_path": "src/main.py",
        "language": "Python",
        "is_vendor": False,
        "license_type": "permissive",
        "detected_licenses": ["MIT"],
        **overrides,
    }


def _parquet(path):
    secret = "ghp_" + "A" * 30
    files = [
        _file("print('clean')  # <EMAIL>\n", file_path="src/clean.py"),
        _file(f"print('hello')  # <EMAIL> {secret}\n"),
        _file("vendor code\n", is_vendor=True, file_path="vendor/a.py"),
        _file("unlicensed code\n", license_type="no_license", detected_licenses=[]),
        _file("missing license detection\n", detected_licenses=[]),
        _file("unknown license\n", detected_licenses=["LicenseRef-unknown"]),
        _file("public class A {}\n", language="Java", file_path="A.java"),
        _file("upstream identity differs\n", content_id="d" * 40),
    ]
    rows = [
        {
            "repo_path": "example/good",
            "repo_id": 1,
            "commit_id": "a" * 40,
            "github_metadata": {"is_fork": False, "stars": 10},
            "num_files": len(files),
            "files": files,
        },
        {
            "repo_path": "example/fork",
            "repo_id": 2,
            "commit_id": "b" * 40,
            "github_metadata": {"is_fork": True, "stars": 20},
            "num_files": 1,
            "files": [_file("print('fork')\n")],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def _config(tmp_path):
    path = tmp_path / "stack-v3.parquet"
    _parquet(path)
    return {
        "format": "speck_stack_v3_qualification",
        "format_version": 1,
        "status": "bounded_profile_authorized_not_training_authority",
        "source": {
            "repo": "HuggingFaceCode/stack-v3-train",
            "revision": "c" * 40,
            "release": "v3.1-test",
            "files": [
                {
                    "path": "data/fixture.parquet",
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            ],
        },
        "filters": {
            "license_type": "permissive",
            "require_detected_licenses": True,
            "rejected_license_prefixes": ["LicenseRef-"],
            "exclude_vendor": True,
            "exclude_forks": True,
            "min_file_bytes": 1,
            "max_file_bytes": 1_000,
            "max_sample_bytes_per_language_per_repository": 1_000,
            "language_sample_bytes": {"Python": 10},
        },
        "output": {
            "directory": str(tmp_path / "qualified"),
            "download_directory": str(tmp_path / "downloads"),
            "keep_downloads": True,
        },
    }, path


def test_bounded_stack_v3_profile_filters_and_preserves_provenance(tmp_path):
    raw, path = _config(tmp_path)
    config = validate_stack_v3_config(raw)
    report = qualify_stack_v3(config, local_files=[path])
    output = Path(config["output"]["directory"])

    assert report["status"] == "bounded_profile_complete_not_training_authority"
    assert report["counts"]["repositories_seen"] == 2
    assert report["counts"]["repositories_rejected_fork"] == 1
    assert report["counts"]["files_seen"] == 8
    assert report["counts"]["files_sampled"] == 1
    assert report["counts"]["files_rejected_high_confidence_secret"] == 1
    assert report["counts"]["files_rejected_vendor"] == 1
    assert report["counts"]["files_rejected_license_type"] == 1
    assert report["counts"]["files_rejected_missing_detected_license"] == 1
    assert report["counts"]["files_rejected_license_reference"] == 1
    assert report["counts"]["files_rejected_language"] == 1
    assert report["counts"]["files_upstream_content_id_not_plain_sha1"] == 1
    assert report["upstream_pii_placeholders"]["<EMAIL>"] == 1
    assert report["high_confidence_secret_pattern_hits"]["github_token"] == 1
    assert report["gates"]["training_authority"] == "blocked"

    tokenizer_record = json.loads((output / "tokenizer-input.jsonl").read_text())
    attribution_record = json.loads((output / "attribution.jsonl").read_text())
    repository_record = json.loads((output / "repository-sample.jsonl").read_text())
    assert tokenizer_record["repo_path"] == attribution_record["repo_path"] == "example/good"
    assert tokenizer_record["text"].startswith("print")
    assert "text" not in attribution_record
    assert repository_record["files"][0]["content_id"] == tokenizer_record["content_id"]
    assert (
        tokenizer_record["released_content_sha256"]
        == hashlib.sha256(tokenizer_record["text"].encode()).hexdigest()
    )
    with pytest.raises(FileExistsError, match="already exists"):
        qualify_stack_v3(config, local_files=[path])


def test_stack_v3_config_and_real_plan_are_bounded_and_non_authoritative(tmp_path):
    raw, _ = _config(tmp_path)
    raw["filters"]["license_type"] = "no_license"
    with pytest.raises(ValueError, match="only permissive"):
        validate_stack_v3_config(raw)

    plan = json.loads((ROOT / "research" / "flagship" / "stack_v3_qualification.json").read_text())
    config = validate_stack_v3_config(plan, config_dir=ROOT / "research" / "flagship")
    assert config["source"]["revision"] == "8f3f25d86e44fd691428131efd17af75d4716499"
    assert len(config["source"]["files"]) == 12
    assert sum(config["filters"]["language_sample_bytes"].values()) == 70_000_000
    assert config["output"]["keep_downloads"] is True


def test_incomplete_language_yield_is_preserved_as_a_failed_report(tmp_path):
    raw, path = _config(tmp_path)
    raw["filters"]["language_sample_bytes"]["Python"] = 1_000_000
    config = validate_stack_v3_config(raw)

    with pytest.raises(RuntimeError, match="did not fill language quotas"):
        qualify_stack_v3(config, local_files=[path])

    report_path = Path(str(config["output"]["directory"]) + ".building") / "report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "bounded_profile_incomplete_not_training_authority"
    assert report["gates"]["restricted_filter_and_language_yield"] == "fail"
    assert report["missing_language_quotas"]["Python"]["target_bytes"] == 1_000_000


def test_recorded_bounded_profile_binds_implementation_and_remains_blocked():
    result = json.loads(
        (ROOT / "results" / "data" / "stack-v3.1-bounded-qualification-20260907.json").read_text()
    )
    implementation = result["implementation"]
    profile = result["twelve_shard_profile"]

    assert result["status"] == "bounded_profile_pass_training_authority_blocked"
    assert implementation["config_sha256"] == _sha256(ROOT / implementation["config"])
    assert implementation["module_sha256"] == _sha256(ROOT / implementation["module"])
    assert implementation["cli_sha256"] == _sha256(ROOT / implementation["cli"])
    assert profile["shards"] == 12
    assert profile["repositories_seen"] == 252_860
    assert profile["files_seen"] == 2_836_149
    assert profile["sampled_content_bytes"] >= 70_000_000
    assert sum(profile["high_confidence_secret_pattern_hits_rejected"].values()) == 3
    assert "benchmark contamination analysis" in result["blocked_gates"]
