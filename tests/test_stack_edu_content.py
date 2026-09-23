"""The restored Stack-Edu screen must reject in the retained acquisition's order."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = importlib.util.spec_from_file_location(
    "stack_edu_content",
    Path(__file__).resolve().parents[1] / "experiments/corpus-audit/stack_edu_content.py",
)
content = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(content)

PROSE = {
    "detector": "py3langid==0.3.0",
    "minimum_alphabetic_characters": 80,
    "minimum_probability": 0.8,
}
CODE = "\n".join(f"value_{index} = compute({index}, offset={index * 3})" for index in range(12))


class Benchmarks:
    def matches(self, text):
        return ["bench"] if "benchmark_marker" in text else []


def row(raw, **change):
    return {
        "blob_id": hashlib.sha1(raw).hexdigest(),
        "length_bytes": len(raw),
        "src_encoding": "UTF-8",
        "language": "Python",
    } | change


@pytest.mark.parametrize(
    ("text", "change", "reason"),
    [
        (CODE, {"blob_id": "0" * 40}, "content_hash_mismatch"),
        (CODE, {"length_bytes": 1}, "content_length_metadata_mismatch"),
        (CODE + "\nkey = 'AKIA" + "A" * 16 + "'", {}, "code_high_confidence_secret"),
        ("x = 1\n", {}, "code_character_envelope"),
        (CODE + "\n# contact someone@company.org", {}, "raw_email_or_ipv4"),
        ("\n".join(["same_line_of_code = 1"] * 20), {}, "duplicate_lines"),
        (CODE + "\nbenchmark_marker = 1", {}, "benchmark_contamination"),
        (CODE, {}, None),
    ],
)
def test_screen_reasons(text, change, reason):
    raw = text.encode()
    assert content.screen(row(raw, **change), raw, PROSE, Benchmarks())[0] == reason


def test_missing_blob_is_counted_not_raised():
    assert content.screen(row(b""), None, PROSE, Benchmarks()) == ("blob_missing_404", None)


def test_acquire_skips_retained_rows_and_resumes(tmp_path, monkeypatch):
    rows = [
        {
            "blob_id": f"{index:040x}",
            "language": "Python",
            "repo_name": "owner/repo",
            "path": f"/m{index}.py",
            "src_encoding": "UTF-8",
            "length_bytes": 500,
            "int_score": score,
            "detected_licenses": ["MIT"],
            "license_type": "permissive",
        }
        for index, score in enumerate([4, 3, 5, 4, 4])
    ]
    metadata = tmp_path / "python.parquet"
    pq.write_table(pa.Table.from_pylist(rows), metadata)
    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps({"files": [{"path": "Python/0.parquet", "local": str(metadata)}]})
    )
    census = tmp_path / "census.json"
    history = {"historical_acquisition": {"consumed_eligible_rows": 1}}
    census.write_text(json.dumps({"by_language": {"Python": history}}))
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "filters": {
                    "accepted_detected_licenses": ["MIT"],
                    "accepted_encodings": ["UTF-8"],
                    "min_file_bytes": 100,
                    "max_file_bytes": 1000,
                    "English_prose": PROSE,
                },
                "excluded_path_components": ["vendor"],
            }
        )
    )
    calls = []

    class FakeScreen:
        def __init__(self, workers):
            pass

        def run(self, unit, workdir):
            calls.append([row["blob_id"] for row in unit])
            return [(None, "text", 7) for _ in unit]

    monkeypatch.setattr(content, "POLICY", policy)
    monkeypatch.setattr(content, "UNIT_ROWS", 2)
    monkeypatch.setattr(content, "Screen", FakeScreen)
    content.acquire(listing, census, "Python", "4+", tmp_path / "out")
    assert calls == [[f"{2:040x}", f"{3:040x}"], [f"{4:040x}"]]
    tranche = json.loads((tmp_path / "out/Python-4plus/tranche.json").read_text())
    assert (tranche["units"], tranche["totals"]["rows"], tranche["totals"]["tokens"]) == (2, 3, 21)
    content.acquire(listing, census, "Python", "4+", tmp_path / "out")
    assert len(calls) == 2
