import pytest

from scripts.structured_retrieval_eval import build_case, parse_tasks, positive_integer
from tests.test_long_context import FakeTokenizer


def test_parse_tasks_is_strict():
    assert parse_tasks("multi_key,two_hop") == ("multi_key", "two_hop")
    with pytest.raises(ValueError, match="tasks"):
        parse_tasks("multi_key,multi_key")
    with pytest.raises(ValueError, match="tasks"):
        parse_tasks("unknown")


def test_build_case_is_deterministic_and_counterfactual():
    tokenizer = FakeTokenizer()
    for task in ("multi_key", "two_hop"):
        case = build_case(task, tokenizer, 1_024, 7, records=8, chains=6)
        repeated = build_case(task, tokenizer, 1_024, 7, records=8, chains=6)
        counterfactual = build_case(task, tokenizer, 1_024, 7, records=8, chains=6, answer_offset=1)
        assert case == repeated
        assert case["label"] == counterfactual["label"]
        assert counterfactual["answer_index"] == (case["answer_index"] + 1) % 10


@pytest.mark.parametrize("value", (0, -1, True, 1.2))
def test_positive_integer_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        positive_integer(value, "example")
