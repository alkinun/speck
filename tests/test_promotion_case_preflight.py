import json
from pathlib import Path

from scripts.promotion_case_preflight import preflight_protocol

root = Path(__file__).parents[1]
contract = root / "research" / "architecture-promotion-v1"
vocabulary = json.loads(
    (contract / "internal" / "route_values_v1.json").read_text(encoding="utf-8")
)


class ProtocolTokenizer:
    bos_id = 1

    def __init__(self):
        self.routes = {entry["text"]: entry["token_id"] for entry in vocabulary["values"]}
        self.words = {}

    def fingerprint(self):
        return vocabulary["tokenizer"]["fingerprint"]

    def encode(self, text, bos=False):
        if text in self.routes:
            tokens = [self.routes[text]]
        else:
            tokens = []
            for word in text.replace("\n", " \n ").split():
                if word not in self.words:
                    self.words[word] = 10000 + len(self.words)
                tokens.append(self.words[word])
        return ([self.bos_id] if bos else []) + tokens


def test_structured_protocol_case_preflight_is_deterministic():
    path = contract / "internal" / "structured_retrieval_v2.json"
    first = preflight_protocol(path, ProtocolTokenizer(), samples_override=2)
    second = preflight_protocol(path, ProtocolTokenizer(), samples_override=2)
    assert first == second
    assert first["status"] == "test_subset"
    assert first["total_generated_cases"] == 12
    assert [condition["candidate_count"] for condition in first["conditions"]] == [16, 16]
    assert [condition["records"] for condition in first["conditions"]] == [2, 8]


def test_symbolic_protocol_case_preflight_covers_all_views():
    path = contract / "internal" / "symbolic_composition_v2.json"
    report = preflight_protocol(path, ProtocolTokenizer(), samples_override=2)
    assert report["total_generated_cases"] == 18
    assert [condition["task"] for condition in report["conditions"]] == [
        "two_hop_route",
        "two_hop_payload",
        "two_hop_symbolic",
    ]
    assert [condition["candidate_count"] for condition in report["conditions"]] == [100, 16, 16]
