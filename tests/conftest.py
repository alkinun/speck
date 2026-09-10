"""Keep maintainer-local evidence checks out of portable test environments."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TOKENIZER_FREEZE = ROOT / "research/flagship/tokenizer_inputs_blocked_v1.json"
MISTRAL_BASELINE = ROOT / "research/flagship/mistral_tokenizer_baseline.json"
FULLSIZE_EVALUATION = Path(
    "/mnt/speck-data/speck/tokenizer-fixtures/fullsize-v2/run/evaluation.json"
)


def _tokenizer_evidence_paths():
    value = json.loads(TOKENIZER_FREEZE.read_text())
    paths = []
    for category in value["categories"]:
        paths.append(Path(category["parent_report"]))
        paths.extend(Path(item["path"]) for item in category["inputs"])
    return paths


LOCAL_EVIDENCE = {
    "tests/test_tokenizer_inputs.py::test_real_tokenizer_inputs_are_complete_hash_bound_and_blocked": (
        _tokenizer_evidence_paths
    ),
    "tests/test_tokenizer_inputs.py::test_mistral_baseline_payload_matches_pinned_identity_and_special_ids": lambda: [
        Path(json.loads(MISTRAL_BASELINE.read_text())["path"])
    ],
    "tests/test_tokenizer_nomination.py::test_fixture_nomination_has_no_advancement_authority": lambda: [
        FULLSIZE_EVALUATION
    ],
}


def pytest_collection_modifyitems(items):
    """Skip only tests whose immutable inputs live outside the repository."""

    for item in items:
        requirements = LOCAL_EVIDENCE.get(item.nodeid)
        if requirements is None:
            continue
        missing = [path for path in requirements() if not path.is_file()]
        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"requires maintainer-local evidence: {missing[0]}")
            )
