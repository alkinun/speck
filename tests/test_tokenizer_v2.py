import hashlib
import json
from pathlib import Path

from speck.tokenizer_nomination import validate_nomination_policy

ROOT = Path(__file__).parents[1]
FLAGSHIP = ROOT / "research/flagship"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_v2_plan_preserves_v1_and_freezes_corrected_candidates_before_formal_outputs():
    plan = json.loads((FLAGSHIP / "tokenizer_plan_v2.json").read_text())
    predecessor = ROOT / plan["supersedes"]["path"]
    evidence = ROOT / plan["evidence"]["path"]

    assert plan["format"] == "speck_flagship_tokenizer_plan"
    assert plan["format_version"] == 2
    assert plan["status"] == "corrected_contract_ready_formal_inputs_and_runs_blocked"
    assert _sha256(predecessor) == plan["supersedes"]["sha256"]
    assert _sha256(evidence) == plan["evidence"]["sha256"]
    assert plan["trainer"]["allow_whitespace_only_pieces"] is True
    assert [candidate["vocab_size"] for candidate in plan["candidates"]] == [
        32000,
        32768,
        40960,
    ]
    assert plan["retired_v1_candidate"]["id"] == "speck-bpe-49152"
    assert plan["sample"]["training_bytes_per_category"] * len(plan["categories"]) == 600_000_000
    assert plan["sample"]["evaluation_bytes_per_category"] * len(plan["categories"]) == 60_000_000
    assert plan["formal_execution"]["status"] == "blocked"
    assert plan["selection_authority"] is False
    assert plan["training_authority"] == "blocked"


def test_v2_nomination_policy_is_lineage_bound_and_non_selecting():
    policy = validate_nomination_policy(
        json.loads((FLAGSHIP / "tokenizer_static_nomination_policy_v2.json").read_text())
    )

    assert _sha256(ROOT / policy["supersedes"]["path"]) == policy["supersedes"]["sha256"]
    assert _sha256(ROOT / policy["evidence"]["path"]) == policy["evidence"]["sha256"]
    assert policy["custom_candidates"] == [
        "speck-bpe-32000-whitespace",
        "speck-bpe-32768-whitespace",
        "speck-bpe-40960-whitespace",
    ]
    assert policy["nomination_count"] == 2
    assert policy["selection_authority"] is False
    assert policy["local_evidence_nomination_authority"] is False
