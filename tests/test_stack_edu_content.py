"""The restored Stack-Edu screen must reject in the retained acquisition's order."""

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
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

        def run(self, unit):
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
    (tmp_path / "out/Python-3").mkdir()
    content.summarize(tmp_path / "out", tmp_path / "acquisition.json")
    summary = json.loads((tmp_path / "acquisition.json").read_text())
    assert [t["tier"] for t in summary["tranches"]] == ["4+"]
    assert summary["tokens_before_full_exclusion"] == 21


def test_convert_joins_retained_and_acquired_records_and_rejects_changes(tmp_path):
    retained_row = {
        "text": "print(1)\n",
        "content_id": "a" * 40,
        "repo_path": "owner/one",
        "file_path": "/a.py",
        "language": "Python",
        "source_file": "Python/0.parquet",
        "source_row": "3",
    }
    data = (json.dumps(retained_row) + "\n").encode()
    tar_path = tmp_path / "unit.tar"
    with tarfile.open(tar_path, "w") as archive:
        member = tarfile.TarInfo("unit/attempt-00000/records.jsonl")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    retained = tmp_path / "acquisition.json"
    unit = {
        "archive": {"tar": {"path": str(tar_path), "sha256": content.file_sha256(tar_path)}},
        "manifest": {
            "output": {
                "path": "attempt-00000/records.jsonl",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        },
    }
    retained.write_text(json.dumps({"units": [unit]}))
    tranche = tmp_path / "out/Python-3"
    tranche.mkdir(parents=True)
    records = tranche / "unit-00000.jsonl.gz"
    acquired_row = {
        "text": "print(2)\n",
        "blob_id": "b" * 40,
        "repo_name": "owner/two",
        "path": "/b.py",
        "language": "Python",
        "file": "Python/1.parquet",
        "source_row": 7,
    }
    with gzip.open(records, "wt") as handle:
        handle.write(json.dumps(acquired_row) + "\n")
    manifest = {"records": {"path": str(records), "sha256": content.file_sha256(records)}}
    (tranche / "unit-00000.json").write_text(json.dumps(manifest))
    (tranche / "tranche.json").write_text(json.dumps({"units": 1}))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"sources": [{"id": "firewall_reference__code_unseen"}]}))
    content.convert(retained, tmp_path / "out", base, tmp_path / "converted")
    rows = [
        json.loads(line) for line in (tmp_path / "converted/input.jsonl").read_text().splitlines()
    ]
    assert [(r["origin"], r["repository"], r["source_row"]) for r in rows] == [
        ("retained", "owner/one", 3),
        ("acquired", "owner/two", 7),
    ]
    assert rows[1]["released_content_sha256"] == hashlib.sha256(b"print(2)\n").hexdigest()
    plan = json.loads((tmp_path / "converted/preprocess-plan.json").read_text())
    assert [s["id"] for s in plan["sources"]] == [
        "firewall_reference__code_unseen",
        "acquired_train__code",
    ]
    records.write_bytes(gzip.compress(b"{}\n"))
    with pytest.raises(ValueError, match="acquired records changed"):
        content.convert(retained, tmp_path / "out", base, tmp_path / "again")


def _unit(directory, texts):
    directory.mkdir(parents=True, exist_ok=True)
    records = directory / "unit-00000.jsonl.gz"
    with gzip.open(records, "wt") as handle:
        for text in texts:
            raw = text.encode()
            record = {"blob_id": hashlib.sha1(raw).hexdigest(), "length_bytes": len(raw)}
            handle.write(json.dumps(record | {"text": text, "tokens": 3}) + "\n")
    manifest = directory / "unit-00000.json"
    manifest.write_text(
        json.dumps(
            {
                "kept_rows": len(texts),
                "tokens": 3 * len(texts),
                "records": content.identity(records),
            }
        )
    )
    return records


def test_verify_rederives_units_and_repair_removes_failures(tmp_path):
    _unit(tmp_path / "Python-3", ["print(1)", "print(2)"])
    records = _unit(tmp_path / "Go-3", ["package main"])
    assert content.verify(tmp_path)["failed_units"] == 0

    # A corrupted text whose file hash was recomputed after corruption: only content checks see it.
    with gzip.open(records, "wt") as handle:
        row = {"blob_id": hashlib.sha1(b"package main").hexdigest(), "length_bytes": 12}
        handle.write(json.dumps(row | {"text": "package mail", "tokens": 3}) + "\n")
    manifest = tmp_path / "Go-3/unit-00000.json"
    manifest.write_text(
        json.dumps({"kept_rows": 1, "tokens": 3, "records": content.identity(records)})
    )
    report = content.verify(tmp_path, repair=True)
    assert (report["units"], report["failed_units"]) == (2, 1)
    assert not manifest.exists() and not records.exists()
    assert (tmp_path / "Python-3/unit-00000.json").exists()
