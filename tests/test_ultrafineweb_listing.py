"""A crawl download is published only when its size and digest match the listing."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = importlib.util.spec_from_file_location(
    "ultrafineweb_listing",
    Path(__file__).resolve().parents[1] / "experiments/corpus-audit/audit_ultrafineweb_listing.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("payload", [b"", b"wrong", b"expected plus excess"])
def test_corrupt_download_is_not_published(tmp_path, monkeypatch, payload):
    expected = b"valid"
    monkeypatch.setattr(audit, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    path = tmp_path / "shard.parquet"
    with pytest.raises(ValueError, match="identity mismatch|exceeded pinned"):
        audit.fetch("url", path, len(expected), hashlib.sha256(expected).hexdigest())
    assert not path.exists()


def test_verified_file_is_kept_without_refetching(tmp_path, monkeypatch):
    expected = b"valid"
    path = tmp_path / "shard.parquet"
    monkeypatch.setattr(audit, "urlopen", lambda *args, **kwargs: io.BytesIO(expected))
    assert audit.fetch("url", path, len(expected), hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(audit, "urlopen", None)
    assert not audit.fetch("url", path, len(expected), hashlib.sha256(expected).hexdigest())
    assert path.read_bytes() == expected


def test_convert_writes_input_and_plan_and_rejects_changed_shards(tmp_path):
    crawl = tmp_path / "crawl"
    crawl.mkdir()
    shard = crawl / "part-0001.parquet"
    rows = [
        {
            "uid": f"u{i}",
            "content": f"text {i}",
            "meta": json.dumps({"url": f"https://a.example/{i}"}),
        }
        for i in range(2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), shard)
    item = {"path": "data/x/part-0001.parquet", "sha256": audit.file_sha256(shard)}
    (crawl / "acquisition.json").write_text(json.dumps({"files": [item]}))
    base = tmp_path / "base.json"
    firewall = {"id": "firewall_reference__web_unseen", "path": "ref.jsonl"}
    base.write_text(json.dumps({"sources": [firewall, {"id": "acquired_train__web"}]}))
    audit.convert(crawl, base, tmp_path / "out")
    records = [json.loads(line) for line in (tmp_path / "out/input.jsonl").read_text().splitlines()]
    assert [(r["content_id"], r["host"], r["source_ordinal"]) for r in records] == [
        ("u0", "a.example", 0),
        ("u1", "a.example", 1),
    ]
    plan = json.loads((tmp_path / "out/preprocess-plan.json").read_text())
    assert [s["id"] for s in plan["sources"]] == [firewall["id"], "acquired_train__web"]
    assert plan["output_directory"] == str(tmp_path / "out/excluded")
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from its acquisition receipt"):
        audit.convert(crawl, base, tmp_path / "again")
