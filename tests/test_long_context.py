import torch

from speck.long_context import (
    add_counterfactual_metrics,
    aggregate_results,
    binomial_tail_probability,
    build_multi_key_case,
    build_passkey_case,
    build_two_hop_case,
    candidate_shift_score,
    effective_length,
    evaluate_case,
    parse_lengths,
    validate_eval_settings,
)


class FakeTokenizer:
    bos_id = 1

    def encode(self, text, bos=False):
        values = [3 + byte % 61 for byte in text.encode()]
        return ([self.bos_id] if bos else []) + values


class FakeState:
    def memory_report(self):
        return {"total_bytes": 64, "by_kind": {"gated_deltanet": 64}}


class OracleModel(torch.nn.Module):
    def __init__(self, answers, vocabulary=128):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.answers = list(answers)
        self.vocabulary = vocabulary
        self.calls = 0

    def state(self, **kwargs):
        return FakeState()

    def forward(self, tokens, state=None, last_token_only=False):
        token = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        logits = torch.full((1, 1, self.vocabulary), -10.0)
        logits[0, 0, token] = 10.0
        return logits


def test_passkey_case_has_exact_requested_length_and_depth():
    tokenizer = FakeTokenizer()
    case = build_passkey_case(tokenizer, 512, seed=7, depth=0.5)
    assert len(case["prompt_tokens"]) + len(case["answer_tokens"]) == 512
    assert case["prompt_length"] == len(case["prompt_tokens"])
    assert case["answer_tokens"]
    assert case["depth"] == 0.5
    counterfactual = build_passkey_case(tokenizer, 512, seed=7, depth=0.5, answer_offset=1)
    assert counterfactual["label"] == case["label"]
    assert counterfactual["answer_index"] == (case["answer_index"] + 1) % 10


def test_multi_key_case_has_exact_length_and_stable_counterfactual_structure():
    tokenizer = FakeTokenizer()
    case = build_multi_key_case(tokenizer, 1_024, seed=9, depth=0.4, records=8)
    counterfactual = build_multi_key_case(
        tokenizer, 1_024, seed=9, depth=0.4, records=8, answer_offset=1
    )
    assert len(case["prompt_tokens"]) + len(case["answer_tokens"]) == 1_024
    assert case["records"] == 8
    assert case["label"] == counterfactual["label"]
    assert case["query_index"] == counterfactual["query_index"]
    assert counterfactual["answer_index"] == (case["answer_index"] + 1) % 10
    distractor_index = (case["query_index"] + 1) % case["records"]
    distractor = build_multi_key_case(
        tokenizer,
        1_024,
        seed=9,
        depth=0.4,
        records=8,
        answer_offset=1,
        mutation_index=distractor_index,
    )
    assert distractor["answer_index"] == case["answer_index"]
    assert distractor["mutation_index"] == distractor_index


def test_two_hop_case_has_two_ordered_facts_and_exact_length():
    tokenizer = FakeTokenizer()
    case = build_two_hop_case(tokenizer, 1_024, seed=4, first_depth=0.2, second_depth=0.8)
    counterfactual = build_two_hop_case(
        tokenizer,
        1_024,
        seed=4,
        first_depth=0.2,
        second_depth=0.8,
        answer_offset=1,
    )
    assert len(case["prompt_tokens"]) + len(case["answer_tokens"]) == 1_024
    assert case["fact_positions"][0] < case["fact_positions"][1]
    assert case["label"] == counterfactual["label"]
    assert counterfactual["answer_index"] == (case["answer_index"] + 1) % 10


def test_counterfactual_metrics_require_both_prompt_directions():
    factual = {
        "candidate_log_probabilities": [-2.0, -1.0],
        "prefill_seconds": 1.0,
    }
    counterfactual = {
        "candidate_log_probabilities": [-3.0, -0.5],
        "prefill_seconds": 2.0,
    }
    result = add_counterfactual_metrics(
        factual,
        counterfactual,
        {"answer_index": 0},
        {"answer_index": 1, "answer": "B"},
    )
    assert result["contrastive_retrieval_score"] == 0.75
    assert result["contrastive_direction_accuracy"] == 1.0
    assert result["contrastive_pair_accuracy"] == 0.0


def test_candidate_shift_score_is_symmetric_between_prompts():
    reference = {"candidate_log_probabilities": [-1.0, -2.0]}
    changed = {"candidate_log_probabilities": [-3.0, -0.5]}
    assert candidate_shift_score(reference, changed, 0, 1) == 1.75


def test_evaluate_case_streams_answer_without_full_logits():
    tokenizer = FakeTokenizer()
    case = build_passkey_case(tokenizer, 256, seed=3, depth=0.1)
    model = OracleModel(case["answer_tokens"])
    result = evaluate_case(model, case, device="cpu")
    assert result["exact_match"] == 1
    assert result["token_accuracy"] == 1
    assert result["state_memory"]["total_bytes"] == 64


def test_effective_length_and_curve_aggregation():
    curve = [
        {"length": 1_000, "exact_match": 1.0},
        {"length": 2_000, "exact_match": 0.9},
        {"length": 4_000, "exact_match": 0.8},
    ]
    assert effective_length(curve, 0.85) == 2_000
    assert effective_length([{**curve[0], "exact_match": 0.0}], 0.85) is None
    sample = {
        "task": "passkey",
        "depth": 0.5,
        "seed": 1,
        "answer_tokens": 1,
        "exact_match": 1.0,
        "token_accuracy": 1.0,
        "candidate_accuracy": 1.0,
        "candidate_count": 10,
        "candidate_probability": 0.9,
        "candidate_rank": 1,
        "candidate_margin": 2.0,
        "mean_log_probability": -0.1,
        "prefill_seconds": 1.0,
        "prefill_tokens_per_second": 1_000.0,
        "decode_tokens_per_second": 10.0,
        "state_memory": {"total_bytes": 64, "by_kind": {"attention_kv": 64}},
        "peak_allocated_bytes": None,
    }
    summary = aggregate_results([{**sample, "length": 1_000}, {**sample, "length": 2_000}])
    assert summary["effective_length"] == 2_000
    assert summary["effective_length_by_candidate_accuracy"] is None
    assert summary["curve"][1]["state_bytes"] == 64


def test_candidate_effective_length_requires_signal_above_chance():
    assert binomial_tail_probability(1, 9, 0.1) > 0.05
    assert binomial_tail_probability(4, 9, 0.1) < 0.05

    results = []
    for length in (1_000, 2_000):
        for index in range(9):
            results.append(
                {
                    **sample_result(),
                    "length": length,
                    "candidate_accuracy": float(index < 4),
                }
            )
    summary = aggregate_results(results)
    assert summary["effective_length_by_candidate_accuracy"] == 2_000


def test_contrastive_effective_length_requires_paired_direction_signal():
    results = []
    for length in (1_000, 2_000):
        for index in range(30):
            results.append(
                {
                    **sample_result(),
                    "length": length,
                    "contrastive_retrieval_score": 1.0 if index < 20 else -1.0,
                    "contrastive_direction_accuracy": float(index < 20),
                    "contrastive_pair_accuracy": float(index < 15),
                    "counterfactual_prefill_seconds": 1.0,
                }
            )
    summary = aggregate_results(results)
    assert summary["short_context_contrastive_p_value"] < 0.05
    assert summary["effective_length_by_contrastive_retrieval"] == 2_000


def test_specificity_effective_length_requires_target_over_distractor_signal():
    results = []
    for length in (1_000, 2_000):
        for index in range(30):
            results.append(
                {
                    **sample_result(),
                    "length": length,
                    "contrastive_retrieval_score": 1.0,
                    "contrastive_direction_accuracy": 1.0,
                    "contrastive_pair_accuracy": 1.0,
                    "counterfactual_prefill_seconds": 1.0,
                    "distractor_change_score": 0.0,
                    "association_specificity_score": 1.0 if index < 25 else -1.0,
                    "association_specificity_accuracy": float(index < 25),
                    "distractor_prefill_seconds": 1.0,
                }
            )
    summary = aggregate_results(results)
    assert summary["short_context_association_specificity_p_value"] < 0.05
    assert summary["effective_length_by_association_specificity"] == 2_000


def sample_result():
    return {
        "exact_match": 0.0,
        "token_accuracy": 0.0,
        "candidate_accuracy": 0.0,
        "candidate_count": 10,
        "candidate_probability": 0.1,
        "candidate_rank": 5,
        "candidate_margin": -1.0,
        "mean_log_probability": -1.0,
        "prefill_seconds": 1.0,
        "prefill_tokens_per_second": 1.0,
        "decode_tokens_per_second": 1.0,
        "state_memory": {"total_bytes": 64, "by_kind": {}},
        "peak_allocated_bytes": None,
    }


def test_long_context_settings_are_strict():
    settings = {
        "lengths": [4_096, 32_768],
        "depths": [0.1, 0.5, 0.9],
        "samples_per_depth": 2,
        "effective_threshold": 0.85,
    }
    assert validate_eval_settings(settings)["lengths"] == (4_096, 32_768)
    assert parse_lengths("4096,32768") == (4_096, 32_768)
