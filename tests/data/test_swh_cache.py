import gzip
import hashlib
import json

import pytest

from speck.data import swh_cache as cache


class Response:
    def __init__(self, status, data=b""):
        self.status_code, self.data = status, data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.data


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(cache.time, "sleep", lambda _: None)
    return dict(
        base_url="https://fixture/",
        attempts=2,
        timeout_seconds=1,
        maximum_blob_bytes=1000,
        maximum_compressed_bytes=2000,
    )


def test_verified_cache_reopens_without_network_and_binds_attempt(tmp_path, monkeypatch, settings):
    raw = b"example source"
    blob = hashlib.sha1(raw).hexdigest()
    monkeypatch.setattr(cache, "_get", lambda *args: Response(200, gzip.compress(raw)))
    first = cache.fetch_cached_blob(blob, tmp_path, settings)
    monkeypatch.setattr(cache, "_get", lambda *args: pytest.fail("cached blob fetched again"))
    assert cache.fetch_cached_blob(blob, tmp_path, settings) == first
    assert first[0] == raw
    directory = tmp_path / blob[:2] / blob
    attempt = next(directory.glob("attempt-*.json"))
    attempt.write_text("{}")
    with pytest.raises(ValueError, match="attempt identity"):
        cache.fetch_cached_blob(blob, tmp_path, settings)


def test_missing_is_cached_but_transient_failures_are_not_omissions(
    tmp_path, monkeypatch, settings
):
    blob = "a" * 40
    monkeypatch.setattr(cache, "_get", lambda *args: Response(503))
    with pytest.raises(RuntimeError, match="recorded attempts"):
        cache.fetch_cached_blob(blob, tmp_path, settings)
    directory = tmp_path / blob[:2] / blob
    assert len(list(directory.glob("attempt-*.json"))) == 2
    assert not (directory / "manifest.json").exists()
    monkeypatch.setattr(cache, "_get", lambda *args: Response(404))
    assert cache.fetch_cached_blob(blob, tmp_path, settings)[0] is None
    monkeypatch.setattr(cache, "_get", lambda *args: pytest.fail("cached missing blob fetched"))
    assert cache.fetch_cached_blob(blob, tmp_path, settings)[0] is None
    assert len(list(directory.glob("attempt-*.json"))) == 3


@pytest.mark.parametrize(
    "payload",
    [b"invalid gzip", gzip.compress(b"wrong identity"), gzip.compress(b"x" * 1001), b"x" * 2001],
)
def test_corrupt_or_oversized_inputs_preserve_failure_attempts(
    tmp_path, monkeypatch, settings, payload
):
    monkeypatch.setattr(cache, "_get", lambda *args: Response(200, payload))
    with pytest.raises(RuntimeError):
        cache.fetch_cached_blob("b" * 40, tmp_path, settings)
    attempts = list(tmp_path.rglob("attempt-*.json"))
    assert len(attempts) == 2
    assert all("error_type" in json.loads(path.read_text()) for path in attempts)
    assert not list(tmp_path.rglob("manifest.json"))
