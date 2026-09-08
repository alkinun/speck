import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from speck.io import atomic_json, file_sha256


@pytest.mark.parametrize("size", (0, 8 * 1024 * 1024 + 17))
def test_file_sha256_matches_full_file_digest(tmp_path, size):
    path = tmp_path / "data.bin"
    data = b"x" * size
    path.write_bytes(data)
    assert file_sha256(str(path)) == hashlib.sha256(data).hexdigest()


def test_atomic_json_preserves_report_bytes_and_previous_report_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "reports" / "result.json"
    value = {"z": "caf\u00e9", "a": [1, True]}
    atomic_json(path, value)
    original = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    assert path.read_bytes() == original
    with pytest.raises(TypeError):
        atomic_json(path, {"invalid": object()})

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("speck.io.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_json(path, {"replacement": True})
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_atomic_json_concurrent_writers_have_independent_staging_files(tmp_path, monkeypatch):
    import os

    replace = os.replace
    barrier = Barrier(2)

    def simultaneous_replace(source, destination):
        barrier.wait(timeout=5)
        replace(source, destination)

    monkeypatch.setattr("speck.io.os.replace", simultaneous_replace)
    path = tmp_path / "result.json"
    values = [{"writer": index, "payload": str(index) * 1000} for index in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda value: atomic_json(path, value), values))
    assert json.loads(path.read_text()) in values
    assert list(tmp_path.iterdir()) == [path]
