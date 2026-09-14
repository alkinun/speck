import gzip
import hashlib
import json
import threading
from pathlib import Path

import pytest

from speck.data import ordered_blob_fetch as ordered
from speck.data import swh_cache


def test_prefetch_runs_across_inputs_and_preserves_order_and_duplicate_identity():
    gate = threading.Event()
    lock = threading.Lock()
    called = []

    def work(item):
        with lock:
            called.append(item)
            if len(called) == 4:
                gate.set()
        assert gate.wait(2)
        return item * 2

    result = list(
        ordered.ordered_prefetch([0, 1, 2, 3, 0, 4], work, workers=4, window=6, key=lambda x: x)
    )
    assert [value for _, value in result] == [0, 2, 4, 6, 0, 8]
    assert called.count(0) == 1


@pytest.fixture
def setup(tmp_path, monkeypatch):
    payloads = {}
    targets = []
    for i in range(9):
        raw = f"example file {i}".encode()
        blob = hashlib.sha1(raw).hexdigest()
        payloads[blob] = gzip.compress(raw)
        targets.append({"blob_id": blob})

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
    kwargs = dict(
        cache=tmp_path / "cache",
        fallback=tmp_path / "old",
        settings=dict(
            base_url="https://fixture/",
            attempts=1,
            timeout_seconds=1,
            maximum_compressed_bytes=1000,
            maximum_blob_bytes=1000,
        ),
        workers=2,
        window=4,
        maximum_working_bytes=100000000,
        minimum_free_bytes=0,
    )
    return targets, kwargs


def test_interruption_resume_matches_clean_journal_and_completed_reopen(tmp_path, setup):
    targets, kwargs = setup
    output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="injected"):
        ordered.fetch_targets(targets, output, **kwargs, interrupt_after=2)
    journal = output / "fetches.jsonl"
    with journal.open("ab") as handle:
        handle.write(b"interrupted-tail\n")
    resumed = ordered.fetch_targets(targets, output, **kwargs, resume=True)
    clean = ordered.fetch_targets(targets, tmp_path / "clean", **kwargs)
    assert resumed["journal"]["sha256"] == clean["journal"]["sha256"]
    assert next(output.glob("uncommitted-tail-*.jsonl")).read_bytes() == b"interrupted-tail\n"
    assert ordered.fetch_targets(targets, output, **kwargs, resume=True) == resumed
    with pytest.raises(ValueError, match="original configuration"):
        ordered.fetch_targets(targets[::-1], output, **kwargs, resume=True)


def test_space_gate_and_corruption_are_explicit_failures(tmp_path, setup):
    targets, kwargs = setup
    with pytest.raises(ValueError, match="envelope"):
        ordered.fetch_targets(
            targets, tmp_path / "oversized", **{**kwargs, "maximum_working_bytes": 1}
        )
    with pytest.raises(RuntimeError, match="injected"):
        ordered.fetch_targets(targets, tmp_path / "corrupt", **kwargs, interrupt_after=2)
    journal = tmp_path / "corrupt/fetches.jsonl"
    data = journal.read_bytes()
    journal.write_bytes(b"!" + data[1:])
    with pytest.raises(ValueError, match="checksum"):
        ordered.fetch_targets(targets, tmp_path / "corrupt", **kwargs, resume=True)


def test_preserved_fallback_is_verified_and_not_refetched(tmp_path, setup, monkeypatch):
    targets, kwargs = setup
    blob = targets[0]["blob_id"]
    _, identity = swh_cache.fetch_cached_blob(blob, kwargs["fallback"], kwargs["settings"])
    monkeypatch.setattr(swh_cache, "_get", lambda *a: pytest.fail("fallback fetched again"))
    report = ordered.fetch_targets(targets[:1], tmp_path / "fallback-run", **kwargs)
    row = json.loads(Path(report["journal"]["path"]).read_text())
    assert row["manifest"] == identity


def test_transient_failure_remains_failure_and_resume_keeps_order(tmp_path, setup, monkeypatch):
    targets, kwargs = setup
    original = ordered.fetch_cached_blob

    def failing(blob, *args):
        if blob == targets[2]["blob_id"]:
            raise RuntimeError("transient")
        return original(blob, *args)

    monkeypatch.setattr(ordered, "fetch_cached_blob", failing)
    with pytest.raises(RuntimeError, match="transient"):
        ordered.fetch_targets(targets, tmp_path / "failed", **kwargs)
    monkeypatch.setattr(ordered, "fetch_cached_blob", original)
    report = ordered.fetch_targets(targets, tmp_path / "failed", **kwargs, resume=True)
    rows = [json.loads(line) for line in Path(report["journal"]["path"]).read_text().splitlines()]
    assert [row["blob_id"] for row in rows] == [t["blob_id"] for t in targets]
