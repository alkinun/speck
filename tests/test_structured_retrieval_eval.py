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
        distractor_index = (case["query_index"] + 1) % (case.get("records") or case["chains"])
        distractor = build_case(
            task,
            tokenizer,
            1_024,
            7,
            records=8,
            chains=6,
            answer_offset=1,
            mutation_index=distractor_index,
        )
        assert distractor["answer_index"] == case["answer_index"]
        assert distractor["mutation_index"] == distractor_index


@pytest.mark.parametrize("template", ("registry", "ledger", "manifest", "directory"))
def test_build_case_propagates_template_and_answer_set(template):
    class WordTokenizer:
        bos_id = 1

        def __init__(self):
            self.ids = {}

        def encode(self, text, bos=False):
            tokens = []
            for word in text.replace("\n", " \n ").split():
                if word not in self.ids:
                    self.ids[word] = len(self.ids) + 3
                tokens.append(self.ids[word])
            return ([self.bos_id] if bos else []) + tokens

    case = build_case(
        "multi_key",
        WordTokenizer(),
        1_024,
        7,
        records=8,
        chains=6,
        template=template,
        answer_set="phrases",
        response_cue="answer",
    )
    assert case["template"] == template
    assert case["answer_set"] == "phrases"
    assert case["response_cue"] == "answer"
    assert len(case["answer_tokens"]) == 2


@pytest.mark.parametrize(
    "task", ("two_hop_route", "two_hop_payload", "two_hop_symbolic", "two_hop_chain")
)
def test_build_case_supports_symbolic_two_hop_auxiliaries(task):
    case = build_case(task, FakeTokenizer(), 1_024, 7, records=8, chains=4)
    assert case["task"] == task
    assert case["chains"] == 4


@pytest.mark.parametrize("value", (0, -1, True, 1.2))
def test_positive_integer_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        positive_integer(value, "example")
