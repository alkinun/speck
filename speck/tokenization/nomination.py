"""Apply the frozen non-selecting tokenizer static nomination policy."""

import json
from pathlib import Path

POLICY_FORMAT = "speck_tokenizer_static_nomination_policy"
RESULT_FORMAT = "speck_tokenizer_static_nomination"
FORMAT_VERSION = 1


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def validate_nomination_policy(policy):
    """Validate the exact endpoint policy frozen before real tokenizer outputs."""

    version = policy.get("format_version") if isinstance(policy, dict) else None
    common = {
        "format",
        "format_version",
        "status",
        "primary_baseline",
        "custom_candidates",
        "required_hard_gates",
        "pareto_objectives",
        "nomination_count",
        "nomination_roles",
        "failure_rule",
        "selection_authority",
        "final_decision",
    }
    if version == 1:
        _exact_keys(
            policy,
            common | {"fixture_nomination_authority"},
            "tokenizer nomination policy",
        )
        expected_status = "frozen_before_real_tokenizer_outputs"
        expected_candidates = ["speck-bpe-32768", "speck-bpe-40960", "speck-bpe-49152"]
        no_fixture_authority = policy["fixture_nomination_authority"] is False
    elif version == 2:
        _exact_keys(
            policy,
            common | {"supersedes", "evidence", "local_evidence_nomination_authority"},
            "tokenizer nomination policy",
        )
        expected_status = "v2_frozen_before_formal_tokenizer_outputs"
        expected_candidates = [
            "speck-bpe-32000-whitespace",
            "speck-bpe-32768-whitespace",
            "speck-bpe-40960-whitespace",
        ]
        no_fixture_authority = policy["local_evidence_nomination_authority"] is False
        if policy["supersedes"] != {
            "path": "research/flagship/tokenizer_static_nomination_policy.json",
            "sha256": "5d3cb5c8b6ec74ec31566dbf07fe28340d4f1998e1169ac92981e89d47986499",
            "repository_revision": "9089c01",
        } or policy["evidence"] != {
            "path": "results/data/tokenizer-local-study-20260907.json",
            "sha256": "6ffdaad34318383b4d08f4f6ecd8940b69366daebd3512e61be1308523b02bb1",
            "authority": "candidate_set_design_only",
        }:
            raise ValueError("tokenizer nomination v2 lineage differs from the frozen contract")
    else:
        raise ValueError("tokenizer nomination policy differs from the frozen contract")
    if (
        policy["format"] != POLICY_FORMAT
        or policy["status"] != expected_status
        or policy["primary_baseline"] != "mistral-32k"
        or policy["custom_candidates"] != expected_candidates
        or policy["required_hard_gates"]
        != ["uint16_with_chat_tokens", "zero_unknown_tokens", "probe_roundtrip_exact"]
        or policy["pareto_objectives"]
        != {
            "macro_tokens_per_kib": "minimize",
            "embedding_and_head_parameters": "minimize",
        }
        or policy["nomination_count"] != 2
        or [role.get("id") for role in policy["nomination_roles"]]
        != ["compression_endpoint", "compact_endpoint"]
        or policy["selection_authority"] is not False
        or not no_fixture_authority
    ):
        raise ValueError("tokenizer nomination policy differs from the frozen contract")
    return policy


def _dominates(left, right):
    left_values = (
        left["macro"]["tokens_per_kib"],
        left["embedding_and_head_parameters"],
    )
    right_values = (
        right["macro"]["tokens_per_kib"],
        right["embedding_and_head_parameters"],
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def nominate_static_candidates(evaluation, policy, *, fixture=False):
    """Nominate Pareto endpoints without granting selection or advancement authority."""

    policy = validate_nomination_policy(policy)
    if (
        evaluation.get("format") != "speck_tokenizer_evaluation"
        or evaluation.get("format_version") != FORMAT_VERSION
        or evaluation.get("selection_authority") is not False
        or evaluation.get("primary_baseline") != policy["primary_baseline"]
        or not isinstance(evaluation.get("tokenizers"), list)
    ):
        raise ValueError("tokenizer evaluation is not a valid non-selecting static report")
    by_id = {tokenizer.get("id"): tokenizer for tokenizer in evaluation["tokenizers"]}
    expected = {policy["primary_baseline"], *policy["custom_candidates"]}
    if set(by_id) != expected:
        raise ValueError("tokenizer evaluation does not contain the exact frozen candidate set")
    required = policy["required_hard_gates"]
    for tokenizer_id, tokenizer in by_id.items():
        gates = tokenizer.get("static_hard_gates")
        if not isinstance(gates, dict) or any(gate not in gates for gate in required):
            raise ValueError(f"tokenizer {tokenizer_id} lacks required static hard gates")
    baseline = by_id[policy["primary_baseline"]]
    if any(baseline["static_hard_gates"][gate] is not True for gate in required):
        raise ValueError("primary Mistral baseline fails a static hard gate")
    valid = [
        by_id[tokenizer_id]
        for tokenizer_id in policy["custom_candidates"]
        if all(by_id[tokenizer_id]["static_hard_gates"][gate] is True for gate in required)
    ]
    frontier = [
        candidate
        for candidate in valid
        if not any(_dominates(other, candidate) for other in valid if other is not candidate)
    ]
    if len(frontier) < policy["nomination_count"]:
        raise RuntimeError(policy["failure_rule"])
    compression = min(
        frontier,
        key=lambda value: (
            value["macro"]["tokens_per_kib"],
            value["embedding_and_head_parameters"],
            value["vocab_size"],
            value["id"],
        ),
    )
    remaining = [candidate for candidate in frontier if candidate["id"] != compression["id"]]
    compact = min(
        remaining,
        key=lambda value: (
            value["embedding_and_head_parameters"],
            value["macro"]["tokens_per_kib"],
            value["vocab_size"],
            value["id"],
        ),
    )
    nominations = [
        {"role": "compression_endpoint", "id": compression["id"]},
        {"role": "compact_endpoint", "id": compact["id"]},
    ]
    result = {
        "format": RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": (
            "fixture_endpoints_reported_no_advancement_authority"
            if fixture
            else "real_static_endpoints_nominated_for_lm_pilot_no_selection_authority"
        ),
        "plan_fingerprint": evaluation["plan_fingerprint"],
        "sample_manifest_sha256": evaluation["sample_manifest_sha256"],
        "primary_baseline": policy["primary_baseline"],
        "eligible_custom_candidates": [candidate["id"] for candidate in valid],
        "pareto_frontier": [
            candidate["id"]
            for candidate in sorted(
                frontier, key=lambda value: policy["custom_candidates"].index(value["id"])
            )
        ],
        "nominations": nominations,
        "all_tokenizer_metrics": {
            tokenizer_id: {
                "vocab_size": tokenizer["vocab_size"],
                "macro_tokens_per_kib": tokenizer["macro"]["tokens_per_kib"],
                "embedding_and_head_parameters": tokenizer["embedding_and_head_parameters"],
                "static_hard_gates": tokenizer["static_hard_gates"],
            }
            for tokenizer_id, tokenizer in by_id.items()
        },
        "selection_authority": False,
        "advancement_authority": not fixture,
        "final_decision_authority": False,
    }
    if policy["format_version"] == 2:
        result["policy_format_version"] = 2
    return result


def nominate_from_files(evaluation_path, policy_path, output_path, *, fixture=False):
    evaluation_path = Path(evaluation_path).resolve()
    policy_path = Path(policy_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"tokenizer nomination already exists: {output_path}")
    result = nominate_static_candidates(
        json.loads(evaluation_path.read_text()),
        json.loads(policy_path.read_text()),
        fixture=fixture,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
