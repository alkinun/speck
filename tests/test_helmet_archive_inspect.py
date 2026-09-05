import io
import tarfile

import pytest

from scripts.helmet_archive_inspect import validate_member


def _member(name, kind="file"):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if kind == "directory" else tarfile.REGTYPE
    if kind == "file":
        info.size = len(b"value")
    return info


def test_validate_member_accepts_regular_relative_path():
    assert validate_member(_member("data/example.jsonl")).as_posix() == "data/example.jsonl"


@pytest.mark.parametrize("name", ("/absolute", "data/../../escape"))
def test_validate_member_rejects_path_escape(name):
    with pytest.raises(ValueError, match="unsafe"):
        validate_member(_member(name))


def test_validate_member_rejects_link():
    info = tarfile.TarInfo("data/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "target"

    with pytest.raises(ValueError, match="unsafe"):
        validate_member(info)


def test_fixture_payload_shape_is_valid():
    info = _member("data/value")
    stream = io.BytesIO(b"value")
    assert info.size == len(stream.getvalue())
