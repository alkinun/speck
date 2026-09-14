import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.data import stack_edu_stock as stock
from speck.data import stack_edu_units as units
from speck.data.source_stock import capacity_gates
from speck.provenance.io import durable_json, file_sha256

ROOT = Path(__file__).resolve().parents[2]


def policy():
    original = json.loads(
        (ROOT / "archive/pregrant-history/research/flagship/stack_edu_sample_v1.json").read_text()
    )
    return {
        "filters": original["filters"],
        "fetch": {},
        "excluded_path_components": ["vendor", "third_party", "third-party", "node_modules"],
    }


def row(raw, index=0):
    return dict(
        blob_id=hashlib.sha1(raw).hexdigest(),
        language="Rust",
        repo_name="example/repo",
        path=f"src/{index}.rs",
        src_encoding="UTF-8",
        length_bytes=len(raw),
        score=4.5,
        int_score=4,
        detected_licenses=["MIT"],
        license_type="permissive",
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"int_score": True}, "metadata_score"),
        ({"detected_licenses": ["MIT", "GPL-3.0"]}, "metadata_license"),
        ({"repo_name": ""}, "metadata_attribution"),
        ({"path": "src/vendor/foo.rs"}, "declared_vendor_path"),
    ],
)
def test_metadata_policy(changes, reason):
    assert stock.metadata_rejection({**row(b"x" * 200), **changes}, "Rust", policy()) == reason


def test_decode_requires_declared_length_and_preserves_syntax_exemption(monkeypatch):
    raw = b"fn main() { return; }\n" * 10
    item = row(raw)
    assert (
        stock.decode_code({**item, "length_bytes": len(raw) + 1}, raw, policy())[0]
        == "content_length_metadata_mismatch"
    )
    monkeypatch.setattr(stock, "_english_prose_result", lambda *a: ("insufficient_prose", None))
    assert stock.decode_code(item, raw, policy())[1:] == (
        raw.decode(),
        {"prose_status": "insufficient_prose", "detected_English_probability": None},
    )
    monkeypatch.setattr(stock, "_english_prose_result", lambda *a: ("non_English", 0.1))
    assert stock.decode_code(item, raw, policy())[0] == "code_non_English_prose"


def test_language_capacity_cannot_borrow_surplus():
    plan = dict(target_reference_tokens=100, source_language_targets={"Rust": 50, "Go": 50})
    count = dict(
        tokens=150,
        documents=3,
        by_language={"Rust": dict(tokens=110, documents=2), "Go": dict(tokens=40, documents=1)},
    )
    assert capacity_gates(plan, count) == (False, {"Rust": True, "Go": False})
    count["tokens"] = 151
    with pytest.raises(ValueError, match="conserve"):
        capacity_gates(plan, count)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(10))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    texts = [f"fn f{i}() {{}}\n".encode() * 30 for i in range(7)]
    rows = [row(raw, i) for i, raw in enumerate(texts)]
    path = tmp_path / "metadata.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    unit = dict(
        id="Rust",
        category="code",
        language="Rust",
        metadata_path=str(path),
        raw=dict(sha256=file_sha256(path), bytes=path.stat().st_size, filename="Rust/file.parquet"),
        reader=dict(repo="fixture/stack", revision="a" * 40),
        expected_file_rows=7,
        start_row=0,
        stop_row=7,
        candidate_target_tokens=1000,
    )
    plan = dict(
        checkpoint_rows=2,
        blob_cache=str(tmp_path / "cache"),
        base=dict(
            stack_edu_policy=policy(),
            filtering=dict(min_chars=200, max_chars=100000),
            security={"gitleaks_binary": {"path": "fixture"}},
            contamination_plan={},
        ),
    )
    payloads = {item["blob_id"]: raw for item, raw in zip(rows, texts, strict=True)}
    monkeypatch.setattr(
        units,
        "fetch_cached_blob",
        lambda blob, *args: (
            payloads[blob],
            {"path": "/fixture/" + blob, "sha256": hashlib.sha256(payloads[blob]).hexdigest()},
        ),
    )
    monkeypatch.setattr(stock, "_english_prose_result", lambda *args: ("insufficient_prose", None))
    monkeypatch.setattr(units, "_document_rejection", lambda *args: None)

    def security(path, binary, directory):
        report = directory / "report.json"
        durable_json(report, {"findings": []})
        return 0, {"path": str(report), "sha256": file_sha256(report)}

    monkeypatch.setattr(units, "_gitleaks_filter", security)
    return plan, unit


def run(spec, unit, output, **kwargs):
    with ThreadPoolExecutor(max_workers=2) as executor:
        return units.acquire_stack_edu_unit(spec, unit, output, {}, Tokenizer(), executor, **kwargs)


def test_interrupted_replay_matches_clean_bytes_and_preserves_tail(tmp_path, fixture):
    plan, unit = fixture
    clean = run(plan, unit, tmp_path / "clean")
    with pytest.raises(RuntimeError, match="injected"):
        run(plan, unit, tmp_path / "resume", interrupt_after_rows=2)
    scratch = tmp_path / "resume/Rust/unscanned.jsonl"
    with scratch.open("ab") as handle:
        handle.write(b"uncommitted\n")
    resumed = run(plan, unit, tmp_path / "resume")
    assert clean["manifest"]["output"] == resumed["manifest"]["output"]
    assert resumed["manifest"]["retained_records"] == 7
    assert (
        next((tmp_path / "resume/Rust").glob("uncommitted-tail-*.jsonl")).read_bytes()
        == b"uncommitted\n"
    )
    assert run(plan, unit, tmp_path / "resume")["reused"]
    altered = copy.deepcopy(plan)
    altered["base"]["stack_edu_policy"]["excluded_path_components"].append("generated")
    with pytest.raises(ValueError, match="owner/config"):
        run(altered, unit, tmp_path / "resume")


def test_whole_document_quota_and_prefetch_receipt(tmp_path, fixture):
    plan, unit = fixture
    unit["candidate_target_tokens"] = 5
    result = run(plan, unit, tmp_path / "quota")["manifest"]
    assert result["row_window"] == [0, 1]
    assert result["pre_gitleaks_candidate_tokens"] == 10
    receipt = json.loads((tmp_path / "quota/Rust" / result["fetch_batches"][0]["path"]).read_text())
    assert len(receipt["blobs"]) == 2


def test_transient_failure_does_not_advance_cursor(tmp_path, fixture, monkeypatch):
    plan, unit = fixture
    original = units.fetch_cached_blob
    monkeypatch.setattr(
        units, "fetch_cached_blob", lambda *args: (_ for _ in ()).throw(RuntimeError("transient"))
    )
    with pytest.raises(RuntimeError, match="transient"):
        run(plan, unit, tmp_path / "failure")
    assert not (tmp_path / "failure/Rust/state.json").exists()
    monkeypatch.setattr(units, "fetch_cached_blob", original)
    assert run(plan, unit, tmp_path / "failure")["manifest"]["retained_records"] == 7


def test_checkpoint_corruption_blocks_resume(tmp_path, fixture):
    plan, unit = fixture
    with pytest.raises(RuntimeError, match="injected"):
        run(plan, unit, tmp_path / "corrupt", interrupt_after_rows=2)
    scratch = tmp_path / "corrupt/Rust/unscanned.jsonl"
    data = scratch.read_bytes()
    scratch.write_bytes(b"!" + data[1:])
    with pytest.raises(ValueError, match="checksum"):
        run(plan, unit, tmp_path / "corrupt")


def test_counter_conserves_per_language_documents_and_tokens(tmp_path, monkeypatch):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.write_bytes(b"fixture")
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(dict(text=text, language=language, repo_path="example/repo")) + "\n"
            for text, language in [("abc", "Rust"), ("defgh", "Go"), ("ijk", "Rust")]
        )
    )

    class BatchTokenizer:
        def __init__(self, path):
            pass

        def encode_batch(self, texts, bos, eos):
            assert bos and eos
            return [list(range(len(text) + 2)) for text in texts]

    monkeypatch.setattr(stock, "Tokenizer", BatchTokenizer)
    count = stock.count_code_tokens(
        source, {"path": str(tokenizer), "sha256": file_sha256(tokenizer)}
    )
    assert count["tokens"] == 17
    assert count["by_language"] == {
        "Rust": {"tokens": 10, "documents": 2},
        "Go": {"tokens": 7, "documents": 1},
    }
    assert count["repository_byte_hhi"] == 1


def test_registered_plan_binds_metadata_and_all_language_headroom_targets():
    spec = stock.load_stack_edu_preparation(
        ROOT / "research/flagship/stack_edu_stock_preparation_v1.json"
    )
    assert len(spec["units"]) == 11
    assert sum(spec["source_language_targets"].values()) == 1440000000
    assert sum(unit["candidate_target_tokens"] for unit in spec["units"]) == 2400000000
    assert (
        "max_bytes_per_language_per_repository" not in spec["base"]["stack_edu_policy"]["filters"]
    )
