"""A crawl download is published only when its size and digest match the listing."""

import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import speck.data.ultrafineweb as audit
from speck.data import document_index


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


class WordTokenizer:
    def __init__(self, path):
        pass

    def encode_batch(self, texts, bos, eos):
        return [[0] * (len(text.split()) + bos + eos) for text in texts]


def test_census_counts_retained_tokens_and_fineweb_overlap(tmp_path, monkeypatch):
    texts = ["one two", "three four five", "six"]
    rows = [
        {
            "text": text,
            "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_ordinal": index * 10,
        }
        for index, text in enumerate(texts)
    ]
    preprocessed = tmp_path / "pre"
    preprocessed.mkdir()
    retained = preprocessed / "acquired_train__web.jsonl"
    retained.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "outputs": {
            "acquired_train__web": {"path": retained.name, "sha256": audit.file_sha256(retained)}
        }
    }
    (preprocessed / "manifest.json").write_text(json.dumps(manifest))
    fineweb = tmp_path / "fineweb.jsonl"
    fineweb.write_text(
        json.dumps({"released_content_sha256": rows[1]["released_content_sha256"]}) + "\n"
    )
    monkeypatch.setattr(audit, "FINEWEB_EDU", fineweb)
    monkeypatch.setattr(audit, "TOKENIZER", fineweb)
    monkeypatch.setattr(document_index, "Tokenizer", WordTokenizer)
    audit.census(preprocessed, tmp_path / "out", tmp_path / "census.json", workers=2)
    receipt = json.loads((tmp_path / "census.json").read_text())
    assert (receipt["document_count"], receipt["tokens"]) == (3, 4 + 5 + 3)
    assert receipt["shared_with_fineweb_edu"] == {"documents": 1, "tokens": 5}
    assert receipt["distinct_from_fineweb_edu_tokens"] == 7
    index = [
        json.loads(line) for line in (tmp_path / "out/documents.jsonl").read_text().splitlines()
    ]
    assert [(row["ordinal"], row["source_ordinal"]) for row in index] == [(0, 0), (1, 10), (2, 20)]
    stock = json.loads((tmp_path / "out/manifest.json").read_text())
    assert stock["plan"]["input"] == manifest["outputs"]["acquired_train__web"] | {
        "path": str(retained)
    }
    assert receipt["document_index"]["path"] == str(tmp_path / "out/manifest.json")
