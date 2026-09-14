import json
from pathlib import Path

import pytest

from speck.data.finemath_stock import finemath_rejection, load_finemath_preparation
from speck.data.source_stock import domain_concentration
from speck.provenance.io import file_sha256


def filters():
    root = Path(__file__).resolve().parents[2]
    return json.loads(
        (
            root / "archive/pregrant-history/research/flagship/math_finemath_4plus_v1.json"
        ).read_text()
    )["filters"]


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf"), 0.79, 1.1])
def test_finemath_requires_valid_released_language_confidence(score):
    document = {
        "content": "Valid English scientific prose. " * 20,
        "metadata": {"language_score": score},
    }
    assert finemath_rejection(document, filters()) == "finemath_metadata_language_score"


def test_domain_report_keeps_missing_hosts_separate_without_reweighting(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"text": "aaaa", "host": "one.example"},
                {"text": "bb", "host": "two.example"},
                {"text": "cc", "host": "two.example"},
                {"text": "λ", "host": None},
            ]
        )
    )
    report = domain_concentration(path)
    assert report["known_host_utf8_bytes"] == 8
    assert report["unknown_host_utf8_bytes"] == 2
    assert report["known_host_byte_hhi"] == 0.5
    assert report["known_host_documents"] == 3
    assert report["unknown_host_documents"] == 1


def test_finemath_plan_keeps_grade_and_explicit_corpus_policy():
    root = Path(__file__).resolve().parents[2]
    plan = load_finemath_preparation(root / "research/flagship/finemath_stock_preparation_v1.json")
    assert len(plan["units"]) == 8
    assert sum(unit["stop_row"] for unit in plan["units"]) == 837440
    assert all(
        unit["reader"]["filters"] == {"language": "en", "min_score": 4} for unit in plan["units"]
    )
    assert plan["base"]["finemath_filters"]["maximum_bytes_per_host"] is None
    assert plan["report_domain_concentration"] is True


def test_successor_preserves_acquisition_configuration_and_adds_complete_units():
    from speck.data.acquisition_units import _unit_config

    root = Path(__file__).resolve().parents[2] / "research/flagship"
    first = load_finemath_preparation(root / "finemath_stock_preparation_v1.json")
    expanded = load_finemath_preparation(root / "finemath_stock_preparation_v2.json")
    assert len(expanded["units"]) == 11
    assert sum(unit["stop_row"] for unit in expanded["units"]) == 1151480
    for old, new in zip(first["units"], expanded["units"][:8], strict=True):
        assert old == new
        assert _unit_config(first, old) == _unit_config(expanded, new)
    assert expanded["target_reference_tokens"] == 960000000


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("original_shard", "original shard prefix"),
        ("extra_shard", "pinned complete shards"),
        ("old_output", "separate output"),
        ("missing_reuse", "all eight"),
        ("policy", "predecessor policy"),
        ("target", "unsupported FineMath"),
        ("result", "completed shortfall"),
    ],
)
def test_successor_rejects_changed_inputs_policy_or_overwrite(tmp_path, change, error):
    root = Path(__file__).resolve().parents[2] / "research/flagship"
    value = json.loads((root / "finemath_stock_preparation_v2.json").read_text())
    for entry in value.values():
        if isinstance(entry, dict) and set(entry) == {"path", "sha256"}:
            entry["path"] = str((root / entry["path"]).resolve())
    if change in ("original_shard", "extra_shard"):
        shards = json.loads(Path(value["shard_manifest"]["path"]).read_text())
        if change == "original_shard":
            shards["files"][0]["sha256"] = "a" * 64
        else:
            shards["files"].append(shards["files"][-1])
        shard_path = tmp_path / "shards.json"
        shard_path.write_text(json.dumps(shards))
        value["shard_manifest"] = {"path": str(shard_path), "sha256": file_sha256(shard_path)}
    elif change == "old_output":
        previous = json.loads(Path(value["predecessor_plan"]["path"]).read_text())
        value["output_directory"] = previous["output_directory"]
    elif change == "missing_reuse":
        value["reuse_acquisition_units"].pop("finemath_4plus__file_0")
    elif change == "policy":
        value["raw_directory"] += "-changed"
    elif change == "target":
        value["target_reference_tokens"] = 800000000
    else:
        result = json.loads(Path(value["predecessor_result"]["path"]).read_text())
        result["capacity_target_pass"] = True
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps(result))
        value["predecessor_result"] = {
            "path": str(result_path),
            "sha256": file_sha256(result_path),
        }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=error):
        load_finemath_preparation(path)
