import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from speck.provenance.io import atomic_json, check_reference, file_sha256


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

    monkeypatch.setattr("speck.provenance.io.os.replace", fail_replace)
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

    monkeypatch.setattr("speck.provenance.io.os.replace", simultaneous_replace)
    path = tmp_path / "result.json"
    values = [{"writer": index, "payload": str(index) * 1000} for index in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda value: atomic_json(path, value), values))
    assert json.loads(path.read_text()) in values
    assert list(tmp_path.iterdir()) == [path]


def test_repository_references_are_path_only(tmp_path):
    (tmp_path / "record.json").write_text("{}")
    assert check_reference({"path": "record.json"}, root=tmp_path) == tmp_path / "record.json"
    for copied in ({"sha256": "0" * 64}, {"status": "draft"}, {"format": "x"}):
        with pytest.raises(ValueError, match="path-only"):
            check_reference({"path": "record.json", **copied}, root=tmp_path)
    with pytest.raises(ValueError, match="missing referenced file"):
        check_reference({"path": "absent.json"}, root=tmp_path)


def test_external_references_carry_a_verified_digest(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"data")
    digest = hashlib.sha256(b"data").hexdigest()
    assert check_reference({"path": str(artifact), "sha256": digest}, root=root) == artifact
    for entry in ({"path": str(artifact)}, {"path": str(artifact), "sha256": "0" * 64}):
        with pytest.raises(ValueError, match="checksum mismatch"):
            check_reference(entry, root=root)
