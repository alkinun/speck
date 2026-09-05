import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts import cache_equivalence_v2 as v2

root = Path(__file__).parents[1]
contract_path = root / "research" / "paper-1" / "cache_equivalence_v2.json"
v3_contract_path = root / "research" / "paper-1" / "cache_equivalence_v3.json"


def test_checked_cache_equivalence_v2_is_control_first_and_pinned():
    _, contract, cases, checkpoints = v2.load_contract(contract_path)

    assert contract["checkpoint_source"]["control"] == "dense_global_seed42"
    assert contract["checkpoint_source"]["candidates"] == [
        "kda_seed42",
        "kda_seed43",
        "kda_seed44",
    ]
    assert contract["case_stream"]["short_cases"] == 33
    assert contract["case_stream"]["proxy_4k_cases"] == 11
    assert len(cases["source_ids"]) == 11
    assert len(checkpoints["checkpoints"]) == 4


def test_checked_cache_equivalence_v3_is_powered_disjoint_and_preserves_v2():
    _, contract, cases, _ = v2.load_contract(v3_contract_path)

    assert contract["power_analysis"]["selected_cases_per_length"] == 88
    assert contract["case_stream"]["disjointness"]
    assert cases["format_version"] == 2
    assert contract["endpoints"]["early_free_running_divergence"]["authority"] == (
        "descriptive_risk"
    )
    assert contract["evidence_basis"]["non_reinterpretation"]


def test_cache_equivalence_v2_rejects_margin_change_after_freeze(tmp_path):
    copied = tmp_path / "research" / "paper-1"
    copied.mkdir(parents=True)
    value = deepcopy(json.loads(contract_path.read_text(encoding="utf-8")))
    for group in ("case_stream", "checkpoint_source"):
        value[group]["path"] = str(root / value[group]["path"])
    for key in ("random_weight_result", "trained_sentinel_result"):
        value["evidence_basis"][key] = str(root / value["evidence_basis"][key])
    value["endpoints"]["common_history_argmax_disagreement"]["non_inferiority_margin_absolute"] = (
        0.5
    )
    changed = copied / contract_path.name
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="margins"):
        v2.load_contract(changed)


def test_distribution_rows_measure_probability_and_decision_changes():
    expected = torch.tensor([[3.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    actual = torch.tensor([[2.0, 3.0, 1.0], [0.0, 1.1, 2.0]])

    rows = v2._distribution_rows(actual, expected, top_k=2, margin_thresholds=[0.5])

    assert rows[0]["argmax_disagreement"]
    assert rows[0]["high_margin_disagreement"]["0.5"]
    assert rows[0]["top10_overlap"] == 1
    assert rows[0]["jensen_shannon"] > 0
    assert not rows[1]["argmax_disagreement"]


def test_paired_bootstrap_preserves_case_as_the_resampling_unit():
    def case(identifier, disagreements):
        return {
            "id": identifier,
            "steps": [
                {
                    "argmax_disagreement": value,
                    "full_margin": 1.0,
                    "jensen_shannon": float(value),
                    "top10_overlap": 1.0 - float(value),
                    "relative_rms": 1.0 + float(value),
                }
                for value in disagreements
            ],
            "free_running_first_divergence_step": 0 if any(disagreements) else None,
        }

    _, contract, _, _ = v2.load_contract(contract_path)
    control = [case("a", [False, False]), case("b", [False, False])]
    candidate = [case("a", [True, True]), case("b", [False, False])]

    result = v2._paired_bootstrap(control, candidate, "argmax", contract, 0)

    assert result["control_mean"] == 0
    assert result["candidate_mean"] == 0.5
    assert result["candidate_minus_control"] == 0.5


class ToyState:
    def __init__(self):
        self.tokens = None


class ToyModel:
    def state(self, **kwargs):
        return ToyState()

    def __call__(self, tokens, state=None, last_token_only=False):
        if state is not None:
            state.tokens = tokens if state.tokens is None else torch.cat((state.tokens, tokens), 1)
            tokens = state.tokens
        selected = tokens.sum(dim=1) % 5
        logits = torch.zeros(tokens.size(0), 1, 5)
        logits[torch.arange(tokens.size(0)), 0, selected] = 2
        return logits


def test_batched_common_and_free_paths_keep_per_case_histories():
    prompts = torch.tensor([[1, 2, 3], [3, 2, 1]])

    steps, first_divergence = v2.compare_batch(
        ToyModel(), prompts, steps=4, top_k=3, margin_thresholds=[0.1, 0.5]
    )

    assert len(steps) == 2
    assert all(len(case) == 4 for case in steps)
    assert all(not row["argmax_disagreement"] for case in steps for row in case)
    assert first_divergence == [None, None]
