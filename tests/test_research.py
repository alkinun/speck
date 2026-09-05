import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from speck.research import (
    load_promotion_protocol,
    resolve_adaptation_protocol,
    resolve_evaluation_protocol,
    validate_research_contract,
)

root = Path(__file__).parents[1]
contract = root / "research" / "architecture-promotion-v1"


def test_checked_architecture_promotion_contract_is_valid():
    report = validate_research_contract(contract)
    assert report == {
        "policy_id": "architecture-promotion-v1",
        "contract_files": [
            "policy.json",
            "cost_envelopes.json",
            "evaluation_manifest.json",
            "evidence_matrix.json",
        ],
        "replication_stages": 6,
        "training_profiles": 2,
        "serving_profiles": 3,
        "internal_evaluation_suites": 4,
        "external_evaluation_suites": 3,
        "external_source_qualified": 3,
        "external_execution_ready": 0,
        "promotion_protocols": 2,
        "symbolic_route_values": 100,
        "tokenizer_qualified": False,
        "evidence_components": 10,
        "status": "valid",
    }


def test_contract_rejects_equivalence_ratio_drift(tmp_path):
    destination = tmp_path / "contract"
    shutil.copytree(contract, destination)
    policy_path = destination / "policy.json"
    value = deepcopy(json.loads(policy_path.read_text(encoding="utf-8")))
    value["statistical_contract"]["language_loss"]["perplexity_ratio_at_margin"] = 1.0
    policy_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="perplexity ratio"):
        validate_research_contract(destination)


def test_contract_qualifies_declared_route_tokens():
    vocabulary = json.loads(
        (contract / "internal" / "route_values_v1.json").read_text(encoding="utf-8")
    )

    class DeclaredTokenizer:
        def __init__(self):
            self.values = {entry["text"]: entry["token_id"] for entry in vocabulary["values"]}
            self.other = {}

        def fingerprint(self):
            return vocabulary["tokenizer"]["fingerprint"]

        def encode(self, text):
            if text in self.values:
                return [self.values[text]]
            tokens = []
            for word in text.split():
                if word not in self.other:
                    self.other[word] = 10000 + len(self.other)
                tokens.append(self.other[word])
            return tokens

    report = validate_research_contract(contract, tokenizer=DeclaredTokenizer())
    assert report["tokenizer_qualified"] is True
    assert report["symbolic_route_values"] == 100


def test_structured_protocol_resolves_exact_runner_settings():
    loaded = load_promotion_protocol(contract / "internal" / "structured_retrieval_v2.json")
    adaptation = resolve_adaptation_protocol(loaded, seed=43)
    assert adaptation["train_seed_offset"] == 110000000
    assert adaptation["validation_seed_offset"] == 200000000
    assert adaptation["train_answer_sets"] == ("letters", "phrases_train_v2")
    assert adaptation["validation_answer_sets"] == ("phrases_validation_v2",)
    assert adaptation["train_record_counts"] == (2, 8)
    assert adaptation["validation_record_counts"] == (2, 8)
    assert adaptation["validation_samples"] == 200
    evaluation = resolve_evaluation_protocol(loaded)
    assert evaluation["lengths"] == (4096, 32768, 131072)
    assert evaluation["samples"] == 200
    assert [condition["records"] for condition in evaluation["conditions"]] == [2, 8]
    staged = resolve_evaluation_protocol(loaded, selected_length=4096)
    assert staged["lengths"] == (4096,)


def test_symbolic_protocol_resolves_three_views_and_large_route_vocabulary():
    loaded = load_promotion_protocol(contract / "internal" / "symbolic_composition_v2.json")
    adaptation = resolve_adaptation_protocol(loaded, seed=42)
    assert adaptation["tasks"] == (
        "two_hop_route",
        "two_hop_payload",
        "two_hop_symbolic",
    )
    assert len(adaptation["route_values"]) == 100
    evaluation = resolve_evaluation_protocol(loaded)
    assert [condition["task"] for condition in evaluation["conditions"]] == [
        "two_hop_route",
        "two_hop_payload",
        "two_hop_symbolic",
    ]
    assert evaluation["effective_threshold"] == 0.9
    assert len(evaluation["route_values"]) == 100


def test_protocol_rejects_an_undeclared_base_seed():
    loaded = load_promotion_protocol(contract / "internal" / "structured_retrieval_v2.json")
    with pytest.raises(ValueError, match="not declared"):
        resolve_adaptation_protocol(loaded, seed=45)


def test_protocol_rejects_an_undeclared_evaluation_length():
    loaded = load_promotion_protocol(contract / "internal" / "structured_retrieval_v2.json")
    with pytest.raises(ValueError, match="not declared"):
        resolve_evaluation_protocol(loaded, selected_length=65536)


def test_runner_rejects_an_unpinned_protocol_copy(tmp_path):
    source = contract / "internal" / "structured_retrieval_v2.json"
    copied = tmp_path / source.name
    shutil.copyfile(source, copied)
    with pytest.raises(ValueError, match="not pinned"):
        load_promotion_protocol(copied, repository_root=root)
