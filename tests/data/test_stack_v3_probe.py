from copy import deepcopy

import pytest

from speck.data.stack_v3_probe import content_rejection, metadata_rejection


@pytest.fixture
def policy():
    return {
        "accepted_detected_licenses": ["MIT"],
        "languages": ["Python"],
        "min_file_bytes": 1,
        "max_file_bytes": 1000000,
        "excluded_path_components": ["vendor"],
        "English_prose": {"minimum_alphabetic_characters": 80, "minimum_probability": 0.8},
    }


@pytest.fixture
def source():
    return (
        {"repo_path": "owner/repo", "commit_id": "a" * 40, "github_metadata": {"is_fork": False}},
        {
            "content_id": "b" * 40,
            "content": "x = 1\n",
            "size_bytes": 6,
            "language": "Python",
            "file_path": "src/example.py",
            "is_vendor": False,
            "license_type": "permissive",
            "detected_licenses": ["MIT"],
        },
    )


def test_source_gates_preserve_license_and_both_vendor_exclusions(policy, source):
    repo, file = source
    assert metadata_rejection(repo, file, policy) is None
    for key, value, reason in [
        ("detected_licenses", ["MIT", "GPL-3.0"], "detected_license_allowlist"),
        ("detected_licenses", [], "detected_license_allowlist"),
        ("is_vendor", True, "released_vendor"),
        ("file_path", "src\\vendor\\example.py", "declared_vendor_path"),
    ]:
        assert metadata_rejection(repo, {**file, key: value}, policy) == reason
    fork = deepcopy(repo)
    fork["github_metadata"]["is_fork"] = True
    assert metadata_rejection(fork, file, policy) == "repository_fork"


def test_released_length_is_strict_but_upstream_id_is_not_assumed_plain_sha1(policy, source):
    _, file = source
    assert content_rejection(file, policy) is None  # Syntax exemption, intentionally different ID.
    assert (
        content_rejection({**file, "size_bytes": 7}, policy) == "content_length_metadata_mismatch"
    )
    text = "-----BEGIN PRIVATE KEY-----"
    assert (
        content_rejection({**file, "content": text, "size_bytes": len(text)}, policy)
        == "high_confidence_secret"
    )
