from scripts import speckchat2_prepare as prepare


class CharacterTokenCounter:
    def measure(self, messages):
        assistant = sum(
            len(message["content"]) + 1 for message in messages if message["role"] == "assistant"
        )
        return 1 + sum(len(message["content"]) + 3 for message in messages), assistant


def test_source_quotas_are_pinned_and_sum_to_500k():
    assert sum(source.quota for source in prepare.SOURCES) == 500_000
    assert {source.key: source.quota for source in prepare.SOURCES} == {
        "lmsys": 200_000,
        "magpie_mt": 130_000,
        "hermes": 85_000,
        "ultrachat": 65_000,
        "magpie_reasoning": 10_000,
        "no_robots": 8_000,
        "everyday": 2_000,
    }
    assert all(len(source.revision) == 40 for source in prepare.SOURCES)


def test_lmsys_uses_deepseek_response_and_rejects_grounded_disagreement():
    example = {
        "id": "row-1",
        "conversations": [
            {"from": "human", "value": "Question"},
            {"from": "gpt", "value": "Stale copied answer"},
        ],
        "deepseek_response": {"value": "DeepSeek answer", "reward": 5.5},
        "category": "question answering",
        "flaw": "normal",
        "grounded": False,
        "agreement": None,
    }

    candidate = prepare.adapt_lmsys(example, 7)

    assert candidate["messages"][-1]["content"] == "DeepSeek answer"
    assert candidate["quality_score"] == 5.5
    example.update(grounded=True, agreement=False)
    assert prepare.adapt_lmsys(example, 7) is None


def test_everyday_removes_repeated_greeting_exchange():
    example = {
        "full_topic": "budgeting",
        "topic": "personal finance",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "How should I make a budget?"},
            {"role": "assistant", "content": "Start by listing income and expenses."},
        ],
    }

    candidate = prepare.adapt_everyday(example, 0)

    assert candidate["messages"] == example["messages"][2:]


def test_prepare_candidate_validates_and_measures_canonical_messages():
    spec = prepare.SOURCE_BY_KEY["ultrachat"]
    candidate = prepare._candidate(
        [
            {"role": "user", "content": "  Hello\r\nthere  "},
            {"role": "assistant", "content": " Hi! "},
        ],
        "source-row",
    )

    row, digest, error = prepare.prepare_candidate(spec, candidate, 3, CharacterTokenCounter())

    assert error is None
    assert row["messages"][0]["content"] == "Hello\nthere"
    assert row["turns"] == 1
    assert row["context_tokens"] > row["assistant_tokens"]
    assert len(digest) == 16


def test_prompt_dedup_ignores_assistant_text_but_keeps_followups():
    first = [
        {"role": "user", "content": "Explain gravity."},
        {"role": "assistant", "content": "First answer"},
    ]
    second = [
        {"role": "user", "content": "  explain   GRAVITY. "},
        {"role": "assistant", "content": "Different answer"},
    ]
    followup = second + [
        {"role": "user", "content": "Give an example."},
        {"role": "assistant", "content": "An apple falling."},
    ]

    assert prepare.prompt_digest(first) == prepare.prompt_digest(second)
    assert prepare.prompt_digest(second) != prepare.prompt_digest(followup)
