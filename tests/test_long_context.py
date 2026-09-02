import torch

from speck.long_context import (
    aggregate_results,
    binomial_tail_probability,
    build_passkey_case,
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
