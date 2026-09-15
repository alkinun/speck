import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts import acquire_stack_edu_e1s as driver
from speck.data import stack_edu_ordered_units as units
from speck.data import stack_edu_stock, swh_cache
from speck.data.acquisition_units import _digest
from speck.data.firewall_integration import group_acquisition_units
from speck.data.stack_edu_ordered_stock import index_batches
from speck.data.stock_blob_store import StockBlobStore, directory_bytes
from speck.provenance.io import durable_json, file_sha256


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(10))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    policy = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "archive/pregrant-history/research/flagship/stack_edu_sample_v1.json"
        ).read_text()
    )
    settings = dict(
        base_url="https://fixture/",
        attempts=1,
        timeout_seconds=1,
        maximum_compressed_bytes=1000,
        maximum_blob_bytes=1000,
    )
    policy = {
        "filters": policy["filters"],
        "fetch": settings,
        "excluded_path_components": ["vendor"],
    }
    targets, payloads = [], {}
    for i in range(5):
        raw = f"fn f{i}() {{}}\n".encode() * 30
        blob = hashlib.sha1(raw).hexdigest()
        payloads[blob] = gzip.compress(raw)
        targets.append(
            {
                "blob_id": blob,
                "eligible_ordinal": i,
                "source_row": i * 3,
                "metadata": dict(
                    blob_id=blob,
                    language="Rust",
                    repo_name="fixture/repo",
                    path=f"src/{i}.rs",
                    src_encoding="UTF-8",
                    length_bytes=len(raw),
                    int_score=4,
                    detected_licenses=["MIT"],
                    license_type="permissive",
                ),
            }
        )

    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            yield self.data

    monkeypatch.setattr(
        swh_cache, "_get", lambda url, timeout: Response(payloads[url.split("/")[-1]])
    )
    monkeypatch.setattr(
        stack_edu_stock, "_english_prose_result", lambda *a: ("insufficient_prose", None)
    )
    monkeypatch.setattr(units, "_document_rejection", lambda *a: None)

    def security(path, binary, directory):
        report = directory / "report.json"
        durable_json(report, [])
        return 0, {"path": str(report), "sha256": file_sha256(report)}

    def count(path, identity):
        n = len(Path(path).read_text().splitlines())
        return {"documents": n, "tokens": n * 10}

    monkeypatch.setattr(units, "_gitleaks_filter", security)
    monkeypatch.setattr(units, "count_code_tokens", count)
    plan = dict(
        checkpoint_rows=2,
        workers=2,
        window=4,
        reference_tokenizer={"sha256": "fixture"},
        base=dict(
            stack_edu_policy=policy,
            filtering=dict(min_chars=200, max_chars=100000),
            security={"gitleaks_binary": {"path": "fixture"}},
            contamination_plan={},
        ),
    )
    unit = dict(
        id="Rust",
        category="code",
        language="Rust",
        candidate_tokens_remaining=25,
        targets_sha256=_digest(targets),
        reader=dict(repo="fixture/stack", revision="a" * 40),
        raw=dict(filename="Rust/file.parquet", sha256="a" * 64),
        expected_file_rows=15,
    )
    store = StockBlobStore(
        tmp_path / "work/cache", [], settings, maximum_bytes=10000000, minimum_free_bytes=0
    )
    return plan, unit, targets, store


def acquire(setup, output, **kwargs):
    plan, unit, targets, store = setup
    return units.acquire_batch(plan, unit, targets, output, store, {}, Tokenizer(), **kwargs)


def test_prefix_tail_interruption_replay_and_existing_group_handoff(tmp_path, setup):
    plan, unit, targets, store = setup
    with pytest.raises(RuntimeError, match="injected"):
        acquire(setup, tmp_path / "resumed", interrupt_after=2)
    attempt = tmp_path / "resumed/Rust/attempt-00000"
    with (attempt / "fetches.jsonl").open("ab") as handle:
        handle.write(b'{"interrupted_tail":')
    preserved = {str(p): p.read_bytes() for p in attempt.iterdir() if p.is_file()}
    resumed = acquire(setup, tmp_path / "resumed")
    clean = acquire(setup, tmp_path / "clean")
    assert resumed["output"]["sha256"] == clean["output"]["sha256"]
    assert resumed["fetch_journal"]["sha256"] == clean["fetch_journal"]["sha256"]
    assert resumed["consumed_eligible_rows"] == 3
    assert resumed["prefetched_rows_after_prefix"] == 2
    assert resumed["pre_gitleaks_candidate_tokens"] == 30
    assert resumed["row_window"] == [0, 7]
    assert all(Path(p).read_bytes() == value for p, value in preserved.items())
    assert acquire(setup, tmp_path / "resumed") == resumed
    group = group_acquisition_units(
        {**plan, "units": [unit]}, tmp_path / "resumed", tmp_path / "groups"
    )
    assert group["outputs"]["code"]["records"] == 3
    assert store.used == directory_bytes(store.cache)
    assert store.reserved == 0
    archived = units.archive_batch(tmp_path / "resumed/Rust", resumed, tmp_path / "archive")
    assert archived["all_archival_payload_hashes_reopened"]


def test_archive_complete_reopen_includes_prefetched_tail_and_detects_corruption(tmp_path, setup):
    manifest = acquire(setup, tmp_path / "units")
    directory = tmp_path / "units/Rust"
    archive = tmp_path / "archive"
    receipt = units.archive_batch(directory, manifest, archive)
    inventory = json.loads(Path(receipt["inventory"]["path"]).read_text())
    blobs = {r["path"].split("/")[1] for r in inventory if r["path"].startswith("blobs/")}
    assert blobs == {t["blob_id"] for t in setup[2]}
    assert units.archive_batch(directory, manifest, archive) == receipt
    with Path(receipt["tar"]["path"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="archival unit changed"):
        units.archive_batch(directory, manifest, archive)


def test_changed_quota_targets_or_payload_rejected(tmp_path, setup):
    manifest = acquire(setup, tmp_path / "units")
    changed = copy.deepcopy(setup[:3]) + (setup[3],)
    changed[1]["candidate_tokens_remaining"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        acquire(changed, tmp_path / "units")
    changed[2].reverse()
    with pytest.raises(ValueError, match="target identities changed"):
        acquire(changed, tmp_path / "other")
    Path(tmp_path / "units/Rust" / manifest["output"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="payload changed"):
        acquire(setup, tmp_path / "units")


def test_fallback_read_only_and_bounds_include_failed_attempts(tmp_path, setup, monkeypatch):
    _, _, targets, store = setup
    store.fetch(targets[0])
    original = {str(p): p.read_bytes() for p in store.cache.rglob("*") if p.is_file()}
    fallback = StockBlobStore(
        tmp_path / "new",
        [store.cache],
        store.settings,
        maximum_bytes=store.maximum_per_blob - 1,
        minimum_free_bytes=0,
    )
    monkeypatch.setattr(swh_cache, "_get", lambda *a: pytest.fail("unexpected network"))
    fallback.fetch(targets[0])
    assert fallback.used == 0
    assert all(Path(p).read_bytes() == value for p, value in original.items())
    with pytest.raises(ValueError, match="reservation"):
        fallback.fetch(targets[1])
    with pytest.raises(ValueError, match="invalid stock blob"):
        store.fetch({"blob_id": "../escape"})
    import speck.data.stock_blob_store as cache_module

    def failure(blob, cache, settings):
        path = cache / blob[:2] / blob
        durable_json(path / "failed.json", {"failure": True})
        raise RuntimeError("transient")

    monkeypatch.setattr(cache_module, "fetch_cached_blob", failure)
    with pytest.raises(RuntimeError, match="transient"):
        store.fetch(targets[1])
    assert store.used == directory_bytes(store.cache)
    assert store.reserved == 0 and not store.live


def test_missing_404_is_counted_not_transient_failure(tmp_path, setup, monkeypatch):
    original = swh_cache._get
    missing = setup[2][0]["blob_id"]

    def get(url, timeout):
        response = original(url, timeout)
        if url.endswith(missing):
            response.status_code = 404
        return response

    monkeypatch.setattr(swh_cache, "_get", get)
    manifest = acquire(setup, tmp_path / "units")
    assert manifest["rejections"]["blob_missing_404"] == 1
    assert manifest["consumed_eligible_rows"] == 4
    assert manifest["retained_records"] == 3


def test_original_index_order_and_policy_validation(tmp_path, setup):
    plan, unit, targets, _ = setup
    path = tmp_path / "index.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in targets))
    entry = {"unit": unit, "index": {"eligible_rows": 5, "output": {"path": str(path)}}}
    assert [len(b) for b in index_batches(entry, plan["base"]["stack_edu_policy"], 2)] == [2, 2, 1]
    targets[1]["source_row"] = 0
    path.write_text("".join(json.dumps(t) + "\n" for t in targets))
    with pytest.raises(ValueError, match="index order"):
        list(index_batches(entry, plan["base"]["stack_edu_policy"], 2))


def test_driver_pauses_resumes_archives_and_stops_dependent_shortfall(tmp_path, setup):
    plan, unit, targets, store = setup
    path = tmp_path / "index.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in targets))
    entry = {
        "unit": unit,
        "index_identity": {"path": str(path), "sha256": file_sha256(path)},
        "index": {"eligible_rows": 5, "output": {"path": str(path)}},
    }
    plan = {
        **plan,
        "language_order": ["Rust", "Go"],
        "eligible_rows_per_unit": 2,
        "candidate_nominal_multiplier": 2,
        "maximum_cache_bytes": 10000000,
        "maximum_working_bytes": 100000000,
        "minimum_free_bytes": 0,
        "targets": {
            name: {"nominal_tokens": 50, "preparation_target_tokens": 60} for name in ("Rust", "Go")
        },
        "files": [entry],
    }
    output, archive = tmp_path / "work", tmp_path / "archive"
    assert (
        driver.run_units(plan, output, archive, store, {}, Tokenizer(), pause_after_units=1) is None
    )
    assert json.loads((output / "progress.json").read_text())["complete_units"] == 1
    result = driver.run_units(plan, output, archive, store, {}, Tokenizer())
    assert result["progress"]["state"] == "pre_exclusion_language_shortfall"
    assert result["progress"]["unprocessed_languages"] == ["Go"]
    assert result["progress"]["by_language_before_full_exclusion"]["Rust"]["tokens"] == 50
    assert len(result["units"]) == 3
    assert len(list((output / "acquired").glob("*/attempt-*"))) == 3
    assert all(row["archive"]["all_archival_payload_hashes_reopened"] for row in result["units"])
