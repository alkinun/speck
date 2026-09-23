"""A crawl download is published only when its size and digest match the listing."""

import hashlib
import importlib.util
import io
from pathlib import Path

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
