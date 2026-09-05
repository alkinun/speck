import pytest

from scripts.helmet_data_metadata_audit import CONFIGS, _lfs_pointer


def test_lfs_pointer_is_parsed_without_payload_access():
    value = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 123\n"
    ).encode()

    assert _lfs_pointer(value) == {"oid_sha256": "a" * 64, "payload_bytes": 123}


def test_lfs_pointer_rejects_non_pointer():
    with pytest.raises(ValueError, match="canonical"):
        _lfs_pointer(b"archive bytes")


def test_active_config_matrix_contains_short_and_128k_per_category():
    assert len(CONFIGS) == 14
    for category in ("recall", "rag", "rerank", "icl", "longqa", "summ", "cite"):
        assert f"configs/{category}_short.yaml" in CONFIGS
        assert f"configs/{category}.yaml" in CONFIGS
