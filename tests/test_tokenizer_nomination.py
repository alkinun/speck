import json
from pathlib import Path

import pytest

from speck.tokenizer_nomination import nominate_static_candidates

ROOT = Path(__file__).parents[1]
POLICY = json.loads(
    (ROOT / "research/flagship/tokenizer_static_nomination_policy.json").read_text()
)


def _evaluation(values):
    tokenizers = []
    for tokenizer_id, tokens_per_kib, parameters in values:
        tokenizers.append(
            {
                "id": tokenizer_id,
                "vocab_size": (
                    int(tokenizer_id.rsplit("-", 1)[-1]) if tokenizer_id != "mistral-32k" else 32000
                ),
                "macro": {"tokens_per_kib": tokens_per_kib},
                "embedding_and_head_parameters": parameters,
                "static_hard_gates": {
                    "uint16_with_chat_tokens": True,
                    "zero_unknown_tokens": True,
                    "probe_roundtrip_exact": True,
                },
            }
        )
    return {
        "format": "speck_tokenizer_evaluation",
        "format_version": 1,
        "plan_fingerprint": "plan",
        "sample_manifest_sha256": "sample",
        "primary_baseline": "mistral-32k",
        "selection_authority": False,
        "tokenizers": tokenizers,
    }


def test_policy_nominates_distinct_compression_and_compact_pareto_endpoints():
    evaluation = _evaluation(
        [
            ("speck-bpe-32768", 350, 134),
            ("speck-bpe-40960", 340, 168),
            ("speck-bpe-49152", 330, 201),
            ("mistral-32k", 360, 131),
        ]
    )
    result = nominate_static_candidates(evaluation, POLICY)

    assert result["pareto_frontier"] == [
        "speck-bpe-32768",
        "speck-bpe-40960",
        "speck-bpe-49152",
    ]
    assert result["nominations"] == [
        {"role": "compression_endpoint", "id": "speck-bpe-49152"},
        {"role": "compact_endpoint", "id": "speck-bpe-32768"},
    ]
    assert result["selection_authority"] is False
    assert result["final_decision_authority"] is False


def test_dominated_candidate_is_reported_but_not_nominated():
    evaluation = _evaluation(
        [
            ("speck-bpe-32768", 340, 134),
            ("speck-bpe-40960", 350, 168),
            ("speck-bpe-49152", 330, 201),
            ("mistral-32k", 360, 131),
        ]
    )
    result = nominate_static_candidates(evaluation, POLICY)

    assert "speck-bpe-40960" not in result["pareto_frontier"]
    assert "speck-bpe-40960" not in {value["id"] for value in result["nominations"]}
    assert "speck-bpe-40960" in result["all_tokenizer_metrics"]


def test_fewer_than_two_valid_pareto_candidates_fails_without_improvising():
    evaluation = _evaluation(
        [
            ("speck-bpe-32768", 340, 134),
            ("speck-bpe-40960", 350, 168),
            ("speck-bpe-49152", 360, 201),
            ("mistral-32k", 330, 131),
        ]
    )
    with pytest.raises(RuntimeError, match="fewer than two"):
        nominate_static_candidates(evaluation, POLICY)


def test_fixture_nomination_has_no_advancement_authority():
    evaluation = json.loads(
        Path("/mnt/speck-data/speck/tokenizer-fixtures/fullsize-v2/run/evaluation.json").read_text()
    )
    result = nominate_static_candidates(evaluation, POLICY, fixture=True)

    assert result["status"] == "fixture_endpoints_reported_no_advancement_authority"
    assert result["advancement_authority"] is False
    assert result["selection_authority"] is False
