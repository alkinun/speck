"""Restricted Stack v3 metadata and content gates shared by bounded supply probes."""

import re

from speck.data.sources.stack_v3 import _secret_counts
from speck.data.sources.stack_v3_refine import _english_prose_result


def metadata_rejection(repository, file, policy):
    if (repository.get("github_metadata") or {}).get("is_fork"):
        return "repository_fork"
    if not isinstance(repository.get("repo_path"), str) or not isinstance(
        repository.get("commit_id"), str
    ):
        return "repository_identity"
    blob = file.get("content_id")
    detected = file.get("detected_licenses")
    size = file.get("size_bytes")
    path = file.get("file_path")
    if not isinstance(blob, str) or not re.fullmatch("[0-9a-f]{40}", blob):
        return "content_id_format"
    if file.get("is_vendor"):
        return "released_vendor"
    if file.get("license_type") != "permissive":
        return "license_type"
    if (
        not isinstance(detected, list)
        or not detected
        or any(value not in policy["accepted_detected_licenses"] for value in detected)
    ):
        return "detected_license_allowlist"
    if file.get("language") not in policy["languages"]:
        return "unselected_language"
    if type(size) is not int or not policy["min_file_bytes"] <= size <= policy["max_file_bytes"]:
        return "declared_size"
    if not isinstance(path, str) or not path.strip():
        return "file_path_identity"
    if any(
        part in policy["excluded_path_components"]
        for part in path.replace("\\", "/").lower().split("/")
    ):
        return "declared_vendor_path"
    return None


def content_rejection(file, policy):
    text = file.get("content")
    if not isinstance(text, str) or not text:
        return "invalid_content"
    if len(text.encode("utf-8")) != file["size_bytes"]:
        return "content_length_metadata_mismatch"
    if any(_secret_counts(text).values()):
        return "high_confidence_secret"
    if _english_prose_result(text, file["language"], policy["English_prose"])[0] == "non_English":
        return "code_non_English_prose"
    # Upstream IDs precede released-text transformations. Compute a released SHA-256
    # downstream; a plain SHA-1 mismatch alone is not a content rejection.
    return None
