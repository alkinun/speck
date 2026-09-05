import io
import tarfile

import pytest

from scripts.helmet_scorer_prepare import _safe_extract_tar, directory_identity


def _archive(path, name):
    with tarfile.open(path, "w:gz") as handle:
        payload = b"value"
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    _archive(archive, "root/../../escape")

    with pytest.raises(ValueError, match="escapes"):
        _safe_extract_tar(archive, tmp_path / "output", "root")


def test_directory_identity_is_stable(tmp_path):
    (tmp_path / "b").write_text("second", encoding="utf-8")
    (tmp_path / "a").write_text("first", encoding="utf-8")

    first = directory_identity(tmp_path)
    second = directory_identity(tmp_path)

    assert first == second
    assert [entry["path"] for entry in first["files"]] == ["a", "b"]
