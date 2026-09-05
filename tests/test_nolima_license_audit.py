import json

from scripts.nolima_license_audit import RESTRICTED_PATHS, _lfs_pointer


def test_lfs_pointer_is_parsed_without_payload_access():
    value = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 123\n"
    ).encode()

    assert _lfs_pointer(value) == {"oid_sha256": "a" * 64, "bytes": 123}


def test_declared_restricted_paths_are_unique_and_complete():
    assert len(RESTRICTED_PATHS) == 16
    assert len(set(RESTRICTED_PATHS)) == len(RESTRICTED_PATHS)
    assert sum(path.startswith("haystack/") for path in RESTRICTED_PATHS) == 10
    assert sum(path.startswith("needlesets/") for path in RESTRICTED_PATHS) == 6


def test_lfs_pointer_rejects_extra_fields():
    value = json.dumps({"oid": "a" * 64, "size": 123}).encode()

    try:
        _lfs_pointer(value)
    except ValueError as error:
        assert "canonical" in str(error)
    else:
        raise AssertionError("non-pointer payload was accepted")
